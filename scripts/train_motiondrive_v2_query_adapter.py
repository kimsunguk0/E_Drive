#!/usr/bin/env python3
"""P3 고정 screening: 동일 C1/T1 초기값, planner만 학습, 외부 감독 종료 판정.

기존 runtime을 수정하지 않는다. frozen auxiliary loss는 기록되는 상수이며
planner gradient를 만들지 않는다. microbatch는 full-label 분모를 사용하지만
부동소수점 연산 순서까지 full batch와 동일하다고 주장하지 않는다.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.motiondrive_v2_training import (LossWeights, build_loss_normalizers,
    compute_loss, model_inputs, tensor_state_sha256, time_input_policy, to_device)
from scripts.train_motiondrive_v2 import (atomic_checkpoint, atomic_json, autocast,
    check_cuda_headroom, configure_cuda_memory, cuda_memory_snapshot, evaluate,
    seed_all, sha256, slice_batch, source_sha, worker_seed)

ARCHITECTURE = "motiondrive_v2_image_state_query_v1"
INIT_SHA = "5cd98d28157abf169f0f2378c3d0ab29d43f4d6382e9c03ee611e7fc5d864c20"
SPLIT_SHA = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
SUPERVISION_SHA = "ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93"
CALIBRATION_SHA = "8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961"
STEPS, BATCH, MICRO, EVAL_BATCH, WORKERS = 1000, 16, 2, 4, 4
SEED, ADAPTER_SEED, WARMUP = 0, 0, 50
LR, WEIGHT_DECAY, GRAD_CLIP = 5e-5, .01, 5.
CAP_MIB, RESERVE_MIB = 12000, 8192
# Frozen requires_grad changes the bf16 runtime path. The full-tune legacy/
# control/state byte-parity probe establishes the initializer for THIS context.
INITIAL_D3 = 0.4454619773599255
UNFROZEN_INITIAL_D3 = 0.4454620049779748
INITIAL_REFERENCE_SHA = "f32680b3f13f5f85045e27120c63db7afa1e24d3c566dc756ce8ac4c7e3158e1"


def validate_initial_reference():
    path = ROOT / "reports/p3_query_frozen_full_tune_gpu5.json"
    if sha256(path) != INITIAL_REFERENCE_SHA:
        raise ValueError("동결 초기 함수의 실제 full-tune 검증 SHA가 다릅니다")
    proof = json.loads(path.read_text())
    if (proof.get("status") != "frozen_legacy_control_state_full_tune_bitwise_pass" or
            proof.get("official_d3") != INITIAL_D3 or proof.get("n") != 1998 or
            proof.get("n_sessions") != 11 or proof.get("full_forward_count") != 1500 or
            proof.get("optimizer_steps") != 0 or proof.get("final_val_accessed") is not False or
            proof["source"]["file_sha256"].get("models/motiondrive_v2_query_adapter.py") !=
            sha256(ROOT / "models/motiondrive_v2_query_adapter.py")):
        raise ValueError("동결 초기 함수 검증의 모델/평가 계약이 다릅니다")
    return {"path": str(path), "sha256": INITIAL_REFERENCE_SHA,
            "official_d3": INITIAL_D3, "scope": "frozen legacy/control/state all-output byte parity on tune1998"}


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--data-root", default=str(ROOT))
    p.add_argument("--split-manifest", default=str(ROOT / "data/etri/motiondrive_v2/grouped_split_rawtime.json"))
    p.add_argument("--supervision-root", default=str(ROOT / "data/etri/motiondrive_v2/train_tune_geometry_v2"))
    p.add_argument("--init", required=True)
    p.add_argument("--expected-init-sha256", required=True, choices=[INIT_SHA])
    p.add_argument("--adapter-on", required=True, type=int, choices=[0, 1])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", default="cuda:0", choices=["cuda:0"])
    return p.parse_args(argv)


def validate_initial_payload(payload):
    if not isinstance(payload, dict) or type(payload.get("step")) is not int or payload.get("step") != 3000:
        raise ValueError("공통 초기값은 원 C1/T1 LAST3000이어야 합니다")
    m = payload.get("manifest", {})
    args, config = m.get("arguments", {}), m.get("model_config", {})
    if (m.get("split_sha256") != SPLIT_SHA or
            m.get("supervision_manifest_sha256") != SUPERVISION_SHA or
            args.get("time_input") != "nominal" or m.get("time_input") != "nominal" or
            m.get("time_input_policy") != time_input_policy("nominal")):
        raise ValueError("원 초기값의 C1/T1 split/감독/time 계보 불일치")
    if (config.get("goal_on") is not True or config.get("state_on") is not True or
            config.get("motion_input_mode") != "low_feature" or
            list(config.get("plan_output_scale", [])) != [10., 5.]):
        raise ValueError("원 초기값의 G1/S1/low_feature/scale 계약 불일치")


def configure_planner_training(model):
    """모든 frozen 모듈/BN은 eval, planner 전체만 train 및 gradient 허용."""
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("planner."))
        if not param.requires_grad:
            param.grad = None
    model.planner.train()
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not named or not any(name.startswith("planner.query_adapter.") for name, _ in named):
        raise ValueError("새 query adapter가 있는 planner 전체가 필요합니다")
    return named


def frozen_state_sha(model):
    state = {name: value for name, value in model.state_dict().items()
             if not name.startswith("planner.")}
    if not state:
        raise ValueError("검증할 frozen encoder 상태가 없습니다")
    return tensor_state_sha256(state)


def assert_frozen(model, expected):
    if frozen_state_sha(model) != expected:
        raise RuntimeError("동결 encoder 가중치/BN buffer가 변경됐습니다")
    if any(p.requires_grad or p.grad is not None for n, p in model.named_parameters()
           if not n.startswith("planner.")):
        raise RuntimeError("동결 encoder에 gradient가 존재합니다")


def training_forward(model, batch, device, precision="bf16"):
    # inference_mode를 쓰면 planner가 backward용으로 feature를 저장할 수 없다.
    with torch.no_grad(), autocast(device, precision):
        parts = model.forward_parts(**model_inputs(batch, time_input="nominal"))
    if any(value.requires_grad for value in parts.values() if isinstance(value, torch.Tensor)):
        raise RuntimeError("frozen 특징에 autograd graph가 남아 있습니다")
    plan = model.plan_from_features(parts["scene_features"], parts["motion_features"],
                                    parts["state_hat"], parts["history_hat"])
    return {**parts, "plan_abs": plan}


def backward_logical_batch(model, raw, device, weights, *, microbatch=MICRO,
                           precision="bf16", reserve_mib=RESERVE_MIB):
    """외부에서 zero_grad/clip/step을 각 logical batch당 한 번 수행한다."""
    if type(microbatch) is not int or microbatch <= 0:
        raise ValueError("microbatch는 양의 정수여야 합니다")
    check_cuda_headroom(device, reserve_mib)
    norms = to_device(build_loss_normalizers(raw), device)
    parts_sum = {}
    for start in range(0, len(raw["images"]), microbatch):
        check_cuda_headroom(device, reserve_mib)
        batch = to_device(slice_batch(raw, start, start + microbatch), device)
        output = training_forward(model, batch, device, precision)
        loss, parts = compute_loss(output, batch, weights, normalizers=norms)
        if not torch.isfinite(loss):
            raise FloatingPointError("비유한 microbatch loss; skip하지 않고 중단합니다")
        loss.backward()
        for key, value in parts.items():
            parts_sum[key] = parts_sum.get(key, 0.) + value.detach()
        del batch, output, loss, parts
    return parts_sum


def training_loader(dataset, *, batch_size=BATCH, workers=WORKERS, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                        pin_memory=True, generator=generator, worker_init_fn=worker_seed,
                        drop_last=False, persistent_workers=False)
    return loader, generator


def learning_rate(step):
    warm = min(1., (step + 1) / WARMUP)
    progress = max(0., (step - WARMUP) / (STEPS - WARMUP))
    return LR * warm * .5 * (1. + math.cos(math.pi * min(1., progress)))


def configure_numerics():
    """원 P2와 같은 플래그; 추가 forward 이전에 적용한다."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    return {"cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "precision": "bf16", "planner_precision": "float32"}


def initial_metric_gate(report):
    if (report.get("n") != 1998 or report.get("n_sessions") != 11 or
            report.get("time_input") != "nominal" or report.get("official_d3") != INITIAL_D3):
        raise ValueError(f"초기 full-tune 재현 실패: 기대 D3={INITIAL_D3!r}, 실제={report!r}")
    return {"passed": True, "expected_official_d3": INITIAL_D3,
            "observed_official_d3": report["official_d3"], "absolute_tolerance": 0.,
            "reference_sha256": INITIAL_REFERENCE_SHA, "unfrozen_legacy_d3_not_comparison_target": UNFROZEN_INITIAL_D3,
            "scope": "동일 tune1998/11세션/batch4/bf16/nominal full-forward 집계 정확 일치; 표본별 출력 일치 검증은 별도"}


def dataset_inventory(dataset, expected_rows_sha, expected_counts):
    rows = np.asarray(dataset.rows, dtype="<i8")
    row_sha = hashlib.sha256(rows.tobytes()).hexdigest()
    scenes = set(map(str, dataset.scene_names[rows]))
    sessions = {dataset.manifest["scene_to_session"][scene] for scene in scenes}
    observed = (len(rows), len(scenes), len(sessions))
    if (row_sha != expected_rows_sha or observed != expected_counts or
            len(np.unique(rows)) != len(rows)):
        raise ValueError(f"원 초기값과 데이터 행/시나리오/세션 불일치: {observed}")
    return {"n": len(rows), "scenes": len(scenes), "sessions": len(sessions), "rows_sha256": row_sha}


def validate_eval_records(report, records, expected_ids):
    if len(records) != 1998 or len(expected_ids) != 1998 or report.get("n") != 1998:
        raise ValueError("평가 records는 고정 1998개여야 합니다")
    ids = []
    for row in records:
        if (set(row) != {"scenario", "session", "frame", "d3", "proxy"} or
                type(row["frame"]) is not int or type(row["d3"]) not in (int,float) or
                not math.isfinite(row["d3"]) or row["d3"] < 0):
            raise ValueError("기존 평가 record 형식/유한성 불일치")
        ids.append((row["scenario"], row["session"], row["frame"]))
    if (ids != expected_ids or len(set(ids)) != 1998 or len({s for _,s,_ in ids}) != 11 or
            len({scene for scene,_,_ in ids}) != 37):
        raise ValueError("평가 시나리오/세션/프레임 정체성 또는 순서 불일치")
    if float(np.mean([row["d3"] for row in records])) != report.get("official_d3"):
        raise ValueError("평가 records와 D3 집계가 정확히 일치하지 않습니다")


def write_new_evaluation(path, payload):
    """새 run의 동일 step 결과 재평가/덮어쓰기는 허용하지 않는다."""
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with Path(path).open("x") as stream:
        stream.write(encoded)


def source_snapshot():
    from scripts.evaluate_motiondrive_v2_planning import source_manifest
    result = source_manifest()
    files = result["file_sha256"]
    for name in ("scripts/train_motiondrive_v2_query_adapter.py", "models/motiondrive_v2_query_adapter.py",
                 "scripts/run_motiondrive_v2_query_trial.py"):
        files[name] = sha256(ROOT / name)
    return result


def validate_data_paths(args):
    paths = {"split": Path(args.split_manifest).resolve(),
             "supervision": Path(args.supervision_root).resolve() / "supervision_manifest.json",
             "calibration": Path(args.supervision_root).resolve() / "calibration.npz",
             "init": Path(args.init).resolve()}
    expected = {"split": SPLIT_SHA, "supervision": SUPERVISION_SHA,
                "calibration": CALIBRATION_SHA, "init": INIT_SHA}
    if args.expected_init_sha256 != INIT_SHA:
        raise ValueError("P3 사전 고정 초기값 SHA가 아닙니다")
    for key, path in paths.items():
        if sha256(path) != expected[key]:
            raise ValueError(f"{key} 원본 SHA 불일치")
    return {key: {"path": str(path), "sha256": expected[key]} for key, path in paths.items()}


def verify_inputs_unchanged(data, sources):
    for name, item in data.items():
        if sha256(item["path"]) != item["sha256"]:
            raise RuntimeError(f"실행 중 원본이 바뀌었습니다: {name}")
    if source_snapshot() != sources:
        raise RuntimeError("실행 중 소스가 변경됐습니다")


def validate_cuda_namespace(device):
    namespace = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    expected = os.environ.get("MOTIONDRIVE_EXPECTED_GPU_UUID", "")
    physical = os.environ.get("MOTIONDRIVE_EXPECTED_PHYSICAL_GPU", "")
    if (str(device) != "cuda:0" or not namespace.startswith("GPU-") or "," in namespace or
            expected != namespace or physical not in ("4", "5") or torch.cuda.device_count() != 1):
        raise ValueError("외부 supervisor가 선택한 단일 GPU UUID namespace가 필요합니다")
    torch.cuda.set_device(device)
    raw = str(torch.cuda.get_device_properties(device).uuid)
    actual = raw if raw.startswith("GPU-") else "GPU-" + raw
    if actual != namespace:
        raise ValueError("CUDA 장치와 supervisor UUID가 다릅니다")
    return {"cuda_visible_devices": namespace, "logical_device": str(device),
            "physical_gpu": int(physical), "observed_uuid_raw": raw, "observed_uuid": actual}


def main(argv=None):
    args = arguments(argv)
    data = validate_data_paths(args)
    initial_reference = validate_initial_reference()
    sources = source_snapshot()
    run_dir = Path(args.run_dir).resolve()
    # 공통 초기값/감독 원본 또는 기존 run을 덮어쓰지 않는다.
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "preparing", "pid": os.getpid(), "run_dir": str(run_dir),
                "arguments": vars(args), "git_sha": source_sha(), "source": sources,
                "data": data, "initial_reference": initial_reference, "step": 0, "nonfinite_count": 0}
    atomic_json(run_dir / "manifest.json", manifest)
    old_handlers, stop = {}, {"signal": None}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, lambda signum, _frame: stop.update(signal=signum))
        from models.motiondrive_v2_query_adapter import migrate_legacy_checkpoint
        from scripts.motiondrive_v2_data import MotionDriveDataset
        payload = torch.load(data["init"]["path"], map_location="cpu", weights_only=False)
        validate_initial_payload(payload)
        source_data_rows = {split: payload["manifest"].get(split + "_rows_sha256") for split in ("train", "eval")}
        del payload
        device = torch.device(args.device)
        namespace = validate_cuda_namespace(device)
        memory = configure_cuda_memory(device, CAP_MIB, RESERVE_MIB)
        numerics = configure_numerics()
        seed_all(SEED)
        model, migration = migrate_legacy_checkpoint(data["init"]["path"], INIT_SHA,
            adapter_seed=ADAPTER_SEED, adapter_on=bool(args.adapter_on), device="cpu")
        if model.architecture != ARCHITECTURE or model.query_adapter_on != bool(args.adapter_on):
            raise ValueError("새 모델 architecture/adapter 설정 불일치")
        configure_planner_training(model)
        frozen_sha, initial_sha = frozen_state_sha(model), tensor_state_sha256(model.state_dict())
        if (migration.get("source_checkpoint_sha256") != INIT_SHA or
                migration.get("source_checkpoint_step") != 3000 or
                migration.get("initial_model_state_sha256") != initial_sha or
                migration.get("legacy_tensors_bitwise_preserved") is not True or
                migration.get("adapter_seed") != ADAPTER_SEED):
            raise ValueError("migration 실증 계보와 실제 초기 모델이 다릅니다")
        model.to(device)
        named = configure_planner_training(model)
        optimizer = torch.optim.AdamW([p for _, p in named], lr=LR, weight_decay=WEIGHT_DECAY)
        # 초기 adapter 생성에 사용한 RNG와 데이터/후속 연산 RNG를 분리한다.
        seed_all(SEED)
        dataset_args = dict(data_root=args.data_root, split_manifest=args.split_manifest,
                            supervision_root=args.supervision_root, min_frame=30, seed=SEED)
        training = MotionDriveDataset(**dataset_args, split="train", frame_stride=1, augment=True)
        tuning = MotionDriveDataset(**dataset_args, split="tune", frame_stride=5, augment=False)
        train_inventory = dataset_inventory(training, source_data_rows["train"], (54810,203,72))
        eval_inventory = dataset_inventory(tuning, source_data_rows["eval"], (1998,37,11))
        expected_eval_ids = [(str(tuning.scene_names[row]),
                              tuning.manifest["scene_to_session"][str(tuning.scene_names[row])],
                              int(tuning.arr["frame"][row])) for row in tuning.rows]
        loader, generator = training_loader(training)
        eval_loader = DataLoader(tuning, batch_size=EVAL_BATCH, shuffle=False, num_workers=WORKERS,
                                 pin_memory=True, worker_init_fn=worker_seed)
        weights = LossWeights()
        manifest.update(architecture=ARCHITECTURE, query_adapter_on=bool(args.adapter_on),
            model_config=model.config.to_dict(), migration=migration, device_mapping=namespace,
            cuda_memory_policy=memory, numerics=numerics, initial_model_sha=initial_sha, frozen_state_sha=frozen_sha,
            init_checkpoint_sha256=INIT_SHA, frozen_initial_sha256=frozen_sha,
            split_sha256=SPLIT_SHA, supervision_manifest_sha256=SUPERVISION_SHA,
            canonical_calibration_sha256=CALIBRATION_SHA, time_input="nominal",
            time_input_policy=time_input_policy("nominal"),
            data_counts={"train": len(training), "eval": len(tuning)},
            dataset_inventory={"train": train_inventory, "eval": eval_inventory},
            train_rows_sha256=hashlib.sha256(np.asarray(training.rows, dtype="<i8").tobytes()).hexdigest(),
            eval_rows_sha256=hashlib.sha256(np.asarray(tuning.rows, dtype="<i8").tobytes()).hexdigest(),
            schedule={"steps": STEPS, "batch": BATCH, "microbatch": MICRO, "eval_batch": EVAL_BATCH,
                      "workers": WORKERS, "seed": SEED, "adapter_seed": ADAPTER_SEED,
                      "lr": LR, "warmup": WARMUP, "decay": "cosine", "weight_decay": WEIGHT_DECAY,
                      "grad_clip": GRAD_CLIP, "precision": "bf16", "eval_steps": [0,250,500,750,1000]},
            trainable_names=[n for n, _ in named], trainable_parameters=sum(p.numel() for _, p in named),
            loss_weights=dataclasses.asdict(weights), auxiliary_loss_contract="동결 영상 encoder의 상수 손실; planner gradient 기여 없음",
            forward_contract="매 microbatch 새 forward_parts(no_grad), planner autograd; eval은 full forward; feature cache 없음",
            sample_order_policy="독립 generator seed0, row/epoch 기반 동일 photometric 증강, rolling row SHA",
            selection="LAST1000 주판정; BEST는 step250/500/750/1000 tune official D3 최소, secondary",
            process_exit_contract="completed는 Python 작업 완료만 의미; 실제 OS exit는 외부 supervisor 별도 검증",
            status="running")
        step, epoch, best, best_step = 0, 0, None, None
        digest, started = hashlib.sha256(), time.monotonic()

        def checkpoint():
            rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device),
                   "numpy": np.random.get_state(), "python": random.getstate(),
                   "dataloader_generator": generator.get_state()}
            return {"architecture": ARCHITECTURE, "query_adapter_on": bool(args.adapter_on),
                    "model_config": model.config.to_dict(), "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch,
                    "manifest": dict(manifest), "rng": rng, "best_metric": best,
                    "initial_model_sha": initial_sha, "frozen_state_sha": frozen_sha,
                    "sample_order_sha256": digest.hexdigest()}

        atomic_json(run_dir / "manifest.json", manifest)
        atomic_checkpoint(run_dir / "initial.pth", checkpoint())
        with (run_dir / "metrics.jsonl").open("x", buffering=1) as log:
            def record(row):
                text = json.dumps(row, ensure_ascii=False, allow_nan=False)
                log.write(text + "\n")
                print(text, flush=True)

            def run_eval():
                assert_frozen(model, frozen_sha)
                report, records = evaluate(model, eval_loader, device, "bf16", time_input="nominal",
                                           min_free_mib=RESERVE_MIB)
                assert_frozen(model, frozen_sha)
                validate_eval_records(report, records, expected_eval_ids)
                write_new_evaluation(run_dir / f"eval_step{step:04d}.json", {
                    "report": report, "records": records, "step": step, "time_input": "nominal",
                    "architecture": ARCHITECTURE, "query_adapter_on": bool(args.adapter_on),
                    "source": sources})
                row = {"kind": "initial_eval" if step == 0 else "eval", "step": step,
                       "selection_metric": report["official_d3"], **report}
                record(row)
                atomic_json(run_dir / ("initial_eval.json" if step == 0 else "latest_eval.json"), row)
                if step == 0:
                    manifest["initial_metric_gate"] = initial_metric_gate(report)
                    atomic_json(run_dir / "manifest.json", manifest)
                return report["official_d3"]

            run_eval()
            while step < STEPS and stop["signal"] is None:
                training.set_epoch(epoch)
                for raw in loader:
                    if step >= STEPS or stop["signal"] is not None:
                        break
                    configure_planner_training(model)
                    digest.update(np.asarray(raw["row"], dtype="<i8").tobytes())
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate(step)
                    optimizer.zero_grad(set_to_none=True)
                    parts = backward_logical_batch(model, raw, device, weights)
                    try:
                        grad = torch.nn.utils.clip_grad_norm_([p for _, p in named], GRAD_CLIP, error_if_nonfinite=True)
                    except RuntimeError as exc:
                        raise FloatingPointError("planner gradient 유한성 검사 실패") from exc
                    optimizer.step()
                    step += 1
                    manifest.update(step=step, sample_order_sha256=digest.hexdigest())
                    if step == 1 or step % 10 == 0:
                        record({"kind": "train", "step": step, "epoch": epoch,
                                "grad_norm": float(grad), "lr": optimizer.param_groups[0]["lr"],
                                "sample_order_sha256": digest.hexdigest(),
                                "cuda_memory": cuda_memory_snapshot(device),
                                **{key: float(value) for key, value in parts.items()}})
                    if step % 250 == 0:
                        score = run_eval()
                        if best is None or score < best:
                            best, best_step = score, step
                            atomic_checkpoint(run_dir / "best.pth", checkpoint())
                        atomic_checkpoint(run_dir / "last.pth", checkpoint())
                        atomic_json(run_dir / "manifest.json", manifest)
                epoch += 1
        assert_frozen(model, frozen_sha)
        verify_inputs_unchanged(data, sources)
        if validate_initial_reference() != initial_reference:
            raise RuntimeError("학습 중 초기 함수 증거가 바뀌었습니다")
        manifest.update(status="completed" if step == STEPS and stop["signal"] is None else "stopped",
                        step=step, best_metric=best, best_step=best_step, received_signal=stop["signal"],
                        elapsed_seconds=time.monotonic()-started, source_unchanged=True,
                        frozen_state_sha_final=frozen_state_sha(model), frozen_final_sha256=frozen_state_sha(model),
                        cuda_memory=cuda_memory_snapshot(device))
        atomic_checkpoint(run_dir / "last.pth", checkpoint())
        atomic_json(run_dir / "manifest.json", manifest)
        return 0 if manifest["status"] == "completed" else 128 + int(stop["signal"] or 1)
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        if isinstance(exc, FloatingPointError):
            manifest["nonfinite_count"] += 1
        atomic_json(run_dir / "manifest.json", manifest)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
