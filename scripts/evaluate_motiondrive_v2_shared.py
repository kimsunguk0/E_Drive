#!/usr/bin/env python3
"""P2 LAST3000 전용 공유 GPU 평가 보호 wrapper.

기존 학습·모델·평가 소스를 변경하지 않는다. parent는 자신이 생성한 평가
child만 감시/중단하며, child는 allocator cap 설정 후 기존 평가 main을 호출한다.
기본 평가는 원래 GPU0–3 매핑을 유지한다. --physical-gpu 0..5를 명시하면
그 GPU UUID 하나만 노출하고 logical cuda:0으로 평가한다. 학습 계보는 바꾸지 않는다.
이 CLI의 실행에는 별도 승인이 필요하다. 단위 테스트는 실제 CUDA를 사용하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.supervise_motiondrive_v2_job import OwnedRecord, now, query_gpu_free_memory
from scripts.launch_motiondrive_v2_trials import inspect_repository

SCRIPT = "scripts/evaluate_motiondrive_v2_shared.py"
EVALUATOR = "scripts/evaluate_motiondrive_v2_planning.py"
TRAINER = "scripts/train_motiondrive_v2.py"
SUPERVISOR = "scripts/supervise_motiondrive_v2_job.py"
TRAINING_GIT_SHA = "c6845fb33e548462f0fecead8baa1ec259d10d4a"
ARMS = {"c0t0": (0, "train_tune_rawtime", "raw"),
        "c1t0": (1, "train_tune_geometry_v2", "raw"),
        "c0t1": (2, "train_tune_rawtime", "nominal"),
        "c1t1": (3, "train_tune_geometry_v2", "nominal")}
CONDITIONS = ["normal", "image_shuffle", "repeat_current", "reverse_history"]
CAP_MIB, RESERVE_MIB, ADMISSION_MIB = 12000, 8192, 20192
PRESSURE_SECONDS, GRACE_SECONDS = 5., 30.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    require(isinstance(value, dict), "JSON object가 필요합니다")
    return value, hashlib.sha256(raw).hexdigest()


def pinned_sha(value, length=64):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None,
            f"명시적 소문자 {length}자리 SHA가 필요합니다")
    return value


def inside(path, root):
    path = Path(path)
    resolved = (path if path.is_absolute() else Path(root) / path).resolve()
    require(resolved.is_relative_to(Path(root).resolve()), "작업 루트 밖의 경로는 허용하지 않습니다")
    return resolved


def new_artifact(path, root, boundary, suffix):
    lexical = Path(path)
    lexical = lexical if lexical.is_absolute() else Path(root) / lexical
    require(not os.path.lexists(lexical), f"기존 파일을 덮어쓰지 않습니다: {lexical}")
    resolved = inside(lexical, root)
    require(resolved.is_relative_to(Path(root) / boundary) and resolved.suffix == suffix,
            "산출물 위치 또는 확장자가 허용 범위와 다릅니다")
    require(resolved.parent.is_dir(), "산출물 부모 디렉터리가 먼저 존재해야 합니다")
    return resolved


GPU_SELECTION_KEYS = ("training_gpu", "physical_gpu", "logical_device",
                      "physical_gpu_explicit", "cuda_namespace")


def gpu_selection(arm, physical_gpu=None):
    require(arm in ARMS, "P2의 네 팔만 허용합니다")
    explicit = physical_gpu is not None
    require(not explicit or type(physical_gpu) is int and physical_gpu in range(6),
            "평가 physical GPU는 명시적인 정수0–5만 허용합니다; GPU6/7 금지")
    training = ARMS[arm][0]
    return {"training_gpu": training, "physical_gpu": physical_gpu if explicit else training,
            "logical_device": "cuda:0" if explicit else f"cuda:{training}",
            "physical_gpu_explicit": explicit,
            "cuda_namespace": "single_uuid" if explicit else "legacy_four_uuid"}


def request_gpu_selection(request):
    """Old requests retain their exact device behavior; partial new metadata fails."""
    present = set(GPU_SELECTION_KEYS) & set(request)
    if not present:
        return gpu_selection(request["arm"])
    require(present == set(GPU_SELECTION_KEYS), "평가 GPU 선택 metadata 전체가 필요합니다")
    require(type(request["physical_gpu_explicit"]) is bool
            and type(request["physical_gpu"]) is int and type(request["training_gpu"]) is int,
            "GPU 선택의 명시 여부와 물리/학습 번호 타입이 잘못되었습니다")
    expected = gpu_selection(request["arm"], request["physical_gpu"] if request["physical_gpu_explicit"] else None)
    require(all(request[key] == expected[key] for key in GPU_SELECTION_KEYS),
            "학습/물리/논리 GPU 또는 opt-in namespace 선언이 모순됩니다")
    return expected


def namespace_indices(selection):
    return [selection["physical_gpu"]] if selection["physical_gpu_explicit"] else list(range(4))


def evaluation_argv(root, arm, output, physical_gpu=None):
    require(arm in ARMS, "P2의 네 팔만 허용합니다")
    root = Path(root)
    gpu, edition, timing = ARMS[arm]
    device = gpu_selection(arm, physical_gpu)["logical_device"]
    return ["--checkpoint", str(root / f"work_dirs/motiondrive_v2/p2_{arm}_s0/last.pth"),
            "--data-root", str(root), "--split-manifest", str(root / "data/etri/motiondrive_v2/grouped_split_rawtime.json"),
            "--supervision-root", str(root / "data/etri/motiondrive_v2" / edition),
            "--split", "tune", "--out", str(output), "--conditions", *CONDITIONS,
            "--donor-seed", "20260907", "--seed", "0", "--frame-stride", "5", "--max-samples", "0",
            "--batch", "4", "--workers", "4", "--device", device, "--precision", "bf16",
            "--time-input", timing, "--include-motion-predictions"]


def verify_training(root, arm, launch_path, expected_checkpoint_sha256):
    """선택 팔의 실제 종료와 LAST 계보만 검증. 네 팔 전체 종료 gate는 호출자 책임."""
    from scripts import analyze_motiondrive_v2_p2 as p2
    import torch
    root = Path(root)
    gpu, edition, timing = ARMS[arm]
    run = root / f"work_dirs/motiondrive_v2/p2_{arm}_s0"
    manifest, manifest_sha = read_json(run / "manifest.json")
    args = manifest.get("arguments", {})
    require(manifest.get("status") == "completed" and manifest.get("step") == 3000
            and manifest.get("nonfinite_count") == 0, "학습이 LAST3000에서 정상 완료되어야 합니다")
    require(manifest.get("git_sha") == TRAINING_GIT_SHA, "P2 학습 소스 commit 불일치")
    require(all(args.get(k) == v for k, v in p2.EXPECTED_ARGS.items()), "고정 P2 학습 설정 불일치")
    require(args.get("microbatch") == 2 and args.get("cuda_memory_limit_mib") == CAP_MIB
            and args.get("cuda_min_free_mib") == RESERVE_MIB, "P2 공유 학습 메모리 설정 불일치")
    require(args.get("gpu") == gpu and inside(args.get("run_dir", ""), root) == run
            and Path(args.get("data_root", "")).resolve() == root, "학습 root/run/GPU 불일치")
    require(args.get("time_input") == timing and manifest.get("time_input") == timing,
            "팔의 학습 시간 정책 불일치")
    require(inside(args.get("supervision_root", ""), root) == root / "data/etri/motiondrive_v2" / edition,
            "팔의 감독 데이터 edition 불일치")
    c = int(arm[1])
    require(manifest.get("split_sha256") == p2.SPLIT_SHA
            and manifest.get("supervision_manifest_sha256") == p2.MANIFEST_SHAS[c], "학습 split/geometry SHA 불일치")
    require(manifest.get("data_counts") == {"train": 54810, "eval": 1998}, "full train/tune 표본 수 불일치")
    for k in ("train_rows_sha256", "eval_rows_sha256", "initial_model_state_sha256"):
        pinned_sha(manifest.get(k))
    launch_path = inside(launch_path, root)
    launch, launch_sha = read_json(launch_path)
    require(launch.get("status") == "launched" and launch.get("supervise") is True
            and launch.get("tracked_dirty") == "" and launch.get("actual_git_sha") == TRAINING_GIT_SHA,
            "정상 supervised P2 launch 계보가 필요합니다")
    plan_path = root / "configs/motiondrive_v2/p2_geometry_time_shared_r1_s0.json"
    plan, plan_sha = read_json(plan_path)
    require(plan.get("supervise") is True and plan.get("gpu_sharing") ==
            {"mode": "shared", "allocator_limit_mib": CAP_MIB, "reserve_mib": RESERVE_MIB}, "공유 P2 plan 정책 불일치")
    name = f"p2_{arm}_s0"
    matches = [job for job in launch.get("jobs", []) if job.get("name") == name]
    planned = [job for job in plan.get("jobs", []) if job.get("name") == name]
    processes = [p for p in launch.get("processes", []) if p.get("name") == name]
    require(len(matches) == len(planned) == len(processes) == 1, "선택 팔의 고유 launch/plan/process가 필요합니다")
    job = matches[0]
    require(job.get("explicit_argv") == planned[0].get("argv") and job.get("gpu") == planned[0].get("gpu") == gpu,
            "실제 launch가 고정 P2 plan과 다릅니다")
    require(job.get("inputs", {}).get("checkpoint_init", {}).get("sha256") == p2.COMMON_SHA
            and job.get("inputs", {}).get("split", {}).get("sha256") == p2.SPLIT_SHA
            and job.get("inputs", {}).get("supervision", {}).get("sha256") == p2.MANIFEST_SHAS[c],
            "launch 데이터/초기 checkpoint SHA 불일치")
    supervisor_path = inside(job.get("supervisor_record", ""), root)
    supervisor, supervisor_sha = read_json(supervisor_path)
    exit_check = p2.validate_exit(supervisor, manifest, run, job, processes[0])
    require(not Path(f"/proc/{supervisor['supervisor_pid']}").exists(), "학습 supervisor의 실제 종료를 기다려야 합니다")
    require(supervisor.get("pressure_event") is None, "메모리 pressure로 중단된 학습은 평가할 수 없습니다")
    for relative, field in ((TRAINER, "trainer_sha256"), (SUPERVISOR, "supervisor_sha256")):
        require(sha256(root / relative) == pinned_sha(launch.get(field)), "현재 guard 소스와 실제 학습 소스 SHA 불일치")
    split = root / "data/etri/motiondrive_v2/grouped_split_rawtime.json"
    sup = root / "data/etri/motiondrive_v2" / edition
    initializer = root / "work_dirs/motiondrive_v2/p1_g1s1_s0/last.pth"
    require(inside(args.get("init", ""), root) == initializer and not args.get("resume")
            and manifest.get("load_report", {}).get("common_checkpoint_sha256") == p2.COMMON_SHA,
            "원 P1 LAST weights-only 초기화 계보가 필요합니다")
    files = {run / "manifest.json": manifest_sha, launch_path: launch_sha, plan_path: plan_sha,
             supervisor_path: supervisor_sha, split: p2.SPLIT_SHA,
             sup / "supervision_manifest.json": p2.MANIFEST_SHAS[c],
             sup / "calibration.npz": p2.CALIBRATION_SHAS[c], initializer: p2.COMMON_SHA,
             root / TRAINER: launch["trainer_sha256"], root / SUPERVISOR: launch["supervisor_sha256"]}
    selected = run / "last.pth"
    pinned_sha(expected_checkpoint_sha256)
    files[selected] = expected_checkpoint_sha256
    for path, expected in files.items():
        require(sha256(path) == expected, f"원본 계보 파일 SHA 불일치: {path.name}")
    saved = torch.load(selected, map_location="cpu", weights_only=False)
    require(saved.get("step") == 3000, "선택 checkpoint는 LAST3000이어야 합니다")
    require(all(p2.json_form(saved.get("manifest", {}).get(k)) == p2.json_form(manifest.get(k))
                for k in p2.EMBEDDED_KEYS), "LAST 내장 manifest와 종료 manifest 불일치")
    signature, tensor_sha = p2.state_signature(saved.get("model"))
    require(bool(signature), "빈 model state는 허용하지 않습니다")
    del saved
    for path, expected in files.items():
        require(sha256(path) == expected, "학습 계보 검사 중 원본이 변경되었습니다")
    return {"os_exit": exit_check, "training_git_sha": TRAINING_GIT_SHA,
            "selected_checkpoint_sha256": expected_checkpoint_sha256, "last_tensor_sha256": tensor_sha,
            "eval_rows_sha256": manifest["eval_rows_sha256"],
            "input_file_sha256": {str(p): sha for p, sha in files.items()},
            "all_four_training_exits_checked": False}


def source_snapshot(root, expected_git_sha):
    pinned_sha(expected_git_sha, 40)
    repository = inspect_repository(root)
    require(repository["actual_git_sha"] == expected_git_sha and repository["tracked_dirty"] == "",
            "평가 소스는 명시한 clean commit이어야 합니다")
    from scripts.evaluate_motiondrive_v2_planning import source_manifest
    source = source_manifest()
    require(source.get("git_sha") == expected_git_sha and source.get("tracked_changes") == [],
            "기존 evaluator source 선언이 평가 commit과 다릅니다")
    return {"evaluation": source, "wrapper_sha256": sha256(Path(root) / SCRIPT),
            "supervisor_helper_sha256": sha256(Path(root) / SUPERVISOR),
            "launcher_helper_sha256": sha256(Path(root) / "scripts/launch_motiondrive_v2_trials.py")}


def prepare(root, arm, output, record, launch_record, expected_checkpoint_sha256, expected_evaluation_git_sha,
            physical_gpu=None):
    root = Path(root).resolve()
    require(root == ROOT and root.is_dir() and root != Path(root.anchor), "이 wrapper의 실제 저장소 root만 허용합니다")
    require(arm in ARMS, "P2의 네 팔만 허용합니다")
    selection = gpu_selection(arm, physical_gpu)
    pinned_sha(expected_checkpoint_sha256)
    pinned_sha(expected_evaluation_git_sha, 40)
    output = new_artifact(output, root, "reports", ".json")
    protocol = new_artifact(output.with_suffix(".protocol.json"), root, "reports", ".json")
    record = new_artifact(record, root, "logs/motiondrive_v2", ".json")
    receipt = new_artifact(record.with_suffix(".child.json"), root, "logs/motiondrive_v2", ".json")
    require(len({output, protocol, record, receipt}) == 4, "평가 산출물 경로가 서로 달라야 합니다")
    lineage = verify_training(root, arm, launch_record, expected_checkpoint_sha256)
    sources = source_snapshot(root, expected_evaluation_git_sha)
    return {"schema_version": 1, "root": str(root), "arm": arm, "gpu": ARMS[arm][0], **selection,
            "output": str(output), "protocol": str(protocol), "record": str(record), "child_receipt": str(receipt),
            "launch_record": str(inside(launch_record, root)), "checkpoint_sha256": expected_checkpoint_sha256,
            "expected_evaluation_git_sha": expected_evaluation_git_sha, "training": lineage, "sources": sources,
            "evaluation_argv": evaluation_argv(root, arm, output, physical_gpu),
            "memory_policy": {"allocator_limit_mib": CAP_MIB, "reserve_mib": RESERVE_MIB,
                              "preflight_free_mib": ADMISSION_MIB, "pressure_poll_seconds": PRESSURE_SECONDS,
                              "safety_grace_seconds": GRACE_SECONDS}}


def query_evaluation_memory(gpu):
    """Read only an allowed physical GPU. Do not broaden the old training helper."""
    require(type(gpu) is int and gpu in range(6), "메모리 조회는 physical GPU0–5만 허용합니다")
    if gpu < 4:
        return query_gpu_free_memory(gpu)
    command = ["nvidia-smi", "-i", str(gpu), "--query-gpu=index,uuid,memory.free",
               "--format=csv,noheader,nounits"]
    output = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5).stdout
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    require(len(lines) == 1, "메모리 조회는 정확히 한 물리 GPU를 반환해야 합니다")
    fields = [field.strip() for field in lines[0].split(",")]
    require(len(fields) == 3 and fields[0] == str(gpu) and fields[1].startswith("GPU-"),
            "메모리 조회 물리 GPU/UUID 형식 불일치")
    free = float(fields[2])
    require(math.isfinite(free) and free >= 0, "유효한 free MiB 조회가 필요합니다")
    return {"physical_gpu": gpu, "gpu_uuid": fields[1], "free_mib": free, "command": command}


def checked_memory(query, gpu, expected_uuid=None, admission=False):
    require(type(gpu) is int and gpu in range(6), "메모리 조회는 physical GPU0–5만 허용합니다")
    result = query(gpu)
    require(isinstance(result, dict) and result.get("physical_gpu") == gpu
            and isinstance(result.get("gpu_uuid"), str) and result["gpu_uuid"].startswith("GPU-"),
            "GPU 물리 번호/UUID 조회 불일치")
    free = result.get("free_mib")
    require(type(free) in (int, float) and math.isfinite(free) and free >= 0, "유효한 free MiB 조회가 필요합니다")
    if expected_uuid is not None:
        require(result["gpu_uuid"] == expected_uuid, "GPU UUID가 사전 고정과 다릅니다")
    if admission:
        require(free >= ADMISSION_MIB, f"평가 전 free >= {ADMISSION_MIB} MiB가 필요합니다")
    return result


def recheck_files(request):
    for path, expected in request["training"]["input_file_sha256"].items():
        require(sha256(path) == expected, "고정 학습 계보 파일이 평가 전후 변경되었습니다")
    require(source_snapshot(request["root"], request["expected_evaluation_git_sha"]) == request["sources"],
            "평가 소스가 사전 고정 이후 변경되었습니다")


def validate_request(request):
    require(isinstance(request, dict) and request.get("schema_version") == 1, "지원하는 평가 요청이 필요합니다")
    root, arm = Path(request.get("root", "")).resolve(), request.get("arm")
    require(root == ROOT and arm in ARMS and request.get("gpu") == ARMS[arm][0], "고정 root/arm/GPU 요청 불일치")
    selection = request_gpu_selection(request)
    physical = selection["physical_gpu"] if selection["physical_gpu_explicit"] else None
    require(request.get("evaluation_argv") == evaluation_argv(root, arm, request.get("output"), physical),
            "고정 평가 argv 변경은 허용하지 않습니다")
    require(request.get("memory_policy") == {"allocator_limit_mib": CAP_MIB, "reserve_mib": RESERVE_MIB,
            "preflight_free_mib": ADMISSION_MIB, "pressure_poll_seconds": PRESSURE_SECONDS,
            "safety_grace_seconds": GRACE_SECONDS}, "평가 메모리 정책 변경은 허용하지 않습니다")
    pinned_sha(request.get("checkpoint_sha256"))
    require(request["training"].get("selected_checkpoint_sha256") == request["checkpoint_sha256"], "요청 checkpoint SHA 불일치")
    output = inside(request["output"], root)
    record = inside(request["record"], root)
    require(output.is_relative_to(root / "reports") and output.suffix == ".json"
            and record.is_relative_to(root / "logs/motiondrive_v2") and record.suffix == ".json",
            "요청 산출물 경로가 허용 범위 밖입니다")
    require(request["protocol"] == str(output.with_suffix(".protocol.json"))
            and request["child_receipt"] == str(record.with_suffix(".child.json")), "요청의 파생 산출물 경로 불일치")
    mapping = request.get("gpu_uuid_mapping")
    if mapping is not None:
        indices = namespace_indices(selection)
        require(isinstance(mapping, dict) and set(mapping) == {str(i) for i in indices}
                and all(isinstance(v, str) and v.startswith("GPU-") for v in mapping.values())
                and len(set(mapping.values())) == len(indices), "선택 정책에 맞는 고유 GPU UUID 매핑이 필요합니다")


def canonical_cuda_uuid(value):
    """PyTorch _CUuuid may omit exactly the GPU- prefix used by nvidia-smi."""
    require(isinstance(value, str) and bool(value), "CUDA UUID 원문이 유효하지 않습니다")
    return value if value.startswith("GPU-") else "GPU-" + value


def verify_child_device(request, device):
    import torch
    selection = request_gpu_selection(request)
    indices = namespace_indices(selection)
    mapping = request.get("gpu_uuid_mapping", {})
    require(set(mapping) == {str(i) for i in indices}, "child 요청에 고정 GPU UUID 매핑이 필요합니다")
    require(str(device) == selection["logical_device"], "child logical CUDA 장치가 요청과 다릅니다")
    expected = ",".join(mapping[str(i)] for i in indices)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == expected
            and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "child CUDA 환경의 UUID 순서가 요청과 다릅니다")
    require(torch.cuda.device_count() == len(indices), "child CUDA namespace 크기가 선택 정책과 다릅니다")
    torch.cuda.set_device(device)
    observed_raw = str(torch.cuda.get_device_properties(device).uuid)
    observed = canonical_cuda_uuid(observed_raw)
    require(observed == mapping[str(selection["physical_gpu"])], "child CUDA 장치 UUID와 parent 물리 GPU가 다릅니다")
    return {**selection, "observed_uuid": observed, "observed_uuid_raw": observed_raw,
            "cuda_visible_devices": expected, "cuda_device_order": "PCI_BUS_ID"}


def execute_evaluation(request, *, configure=None, evaluate=None, snapshot=None, device_verifier=None):
    """child에서만 호출. 주입 인자는 CPU 테스트용이며 CLI로 노출하지 않는다."""
    validate_request(request)
    import torch
    from scripts.train_motiondrive_v2 import configure_cuda_memory, cuda_memory_snapshot
    from scripts.evaluate_motiondrive_v2_planning import main as evaluation_main
    device = torch.device(request_gpu_selection(request)["logical_device"])
    configure = configure or configure_cuda_memory
    snapshot = snapshot or cuda_memory_snapshot
    device_record = (device_verifier or verify_child_device)(request, device)
    # main/모델 GPU 할당보다 반드시 먼저 설정한다. 기존 main/forward는 변경하지 않는다.
    policy = configure(device, CAP_MIB, RESERVE_MIB)
    before = snapshot(device)
    rc = (evaluate or evaluation_main)(request["evaluation_argv"])
    after = snapshot(device)
    require(rc == 0, "기존 평가 main이 정상 반환하지 않았습니다")
    return {"main_returncode": rc, "memory_policy": policy, "device_mapping": device_record, "before_evaluation": before,
            "after_evaluation": after, "allocator_configured_before_main": True}


def run_child(record_path, expected_request_sha256):
    record, _ = read_json(record_path)
    request = record.get("request")
    require(json_digest(request) == pinned_sha(expected_request_sha256), "child 요청 SHA 불일치")
    validate_request(request)
    require(Path(record_path).resolve() == Path(request["record"]), "child execution record 경로 불일치")
    require(record.get("status") in ("reserved", "running") and record.get("actual_returncode") is None,
            "이미 종료된 execution record를 재사용할 수 없습니다")
    for key in ("output", "protocol", "child_receipt"):
        require(not os.path.lexists(request[key]), "기존 평가 산출물 재사용은 허용하지 않습니다")
    recheck_files(request)
    receipt = {"schema_version": 1, "child_pid": os.getpid(), "request_sha256": expected_request_sha256,
               "started_at": now(), "status": "failed", "memory_policy_requested": request["memory_policy"]}
    try:
        receipt.update(execute_evaluation(request))
        recheck_files(request)
        receipt.update(status="completed", ended_at=now())
    except BaseException as exc:
        receipt.update(error=f"{type(exc).__name__}: {exc}", ended_at=now())
        raise
    finally:
        with Path(request["child_receipt"]).open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return 0


def inspect_result(request, child_pid, request_sha):
    receipt, receipt_sha = read_json(request["child_receipt"])
    require(receipt.get("status") == "completed" and receipt.get("child_pid") == child_pid
            and receipt.get("request_sha256") == request_sha and receipt.get("main_returncode") == 0
            and receipt.get("allocator_configured_before_main") is True, "child 완료/메모리 적용 증거 불일치")
    policy = receipt.get("memory_policy", {})
    require(policy.get("enabled") is True and policy.get("allocator_limit_mib") == CAP_MIB
            and policy.get("min_free_mib") == RESERVE_MIB, "child allocator cap 적용값 불일치")
    selection = request_gpu_selection(request)
    device_mapping = receipt.get("device_mapping", {})
    observed_uuid = device_mapping.get("observed_uuid")
    require(observed_uuid == request["gpu_uuid_mapping"][str(selection["physical_gpu"])],
            "child의 실제 CUDA UUID 증거 불일치")
    # Old receipts have only observed_uuid; new receipts also preserve the exact
    # runtime representation and must agree after prefix-only normalization.
    if "observed_uuid_raw" in device_mapping:
        require(canonical_cuda_uuid(device_mapping["observed_uuid_raw"]) == observed_uuid,
                "child의 CUDA UUID 원문과 표준형 증거 불일치")
    if selection["physical_gpu_explicit"]:
        require(all(receipt["device_mapping"].get(key) == value for key, value in selection.items()),
                "child의 opt-in 물리/논리 GPU 적용 증거 불일치")
    result, result_sha = read_json(request["output"])
    protocol, protocol_sha = read_json(request["protocol"])
    require(result.get("protocol_sha256") == protocol_sha and result.get("protocol") == protocol
            and Path(result.get("protocol_path", "")).resolve() == Path(request["protocol"]), "평가 protocol SHA/내용 불일치")
    from scripts.evaluate_motiondrive_v2_planning import arguments, motion_prediction_contract
    expected_args = vars(arguments(request["evaluation_argv"]))
    require(protocol.get("arguments") == expected_args, "실제 평가 argv가 고정 요청과 다릅니다")
    require(result.get("status") == "completed" and result.get("selection_performed") is False
            and result.get("final_val_accessed") is False and list(result.get("conditions", {})) == CONDITIONS,
            "정상 full tune 네 조건 평가가 필요합니다")
    require(protocol.get("checkpoint_sha256") == request["checkpoint_sha256"] and protocol.get("checkpoint_step") == 3000
            and protocol.get("source") == request["sources"]["evaluation"], "결과의 LAST/source 계보 불일치")
    require(protocol.get("motion_prediction_contract") == motion_prediction_contract()
            and protocol.get("data", {}).get("receiver_count") == 1998
            and protocol["data"].get("receiver_rows_sha256") == request["training"]["eval_rows_sha256"],
            "신경 상태 예측 계약 또는 고정 tune 행 불일치")
    from scripts import analyze_motiondrive_v2_p2 as p2
    c = int(request["arm"][1])
    data = protocol["data"]
    require(data.get("split_sha256") == p2.SPLIT_SHA
            and data.get("supervision_sha256", {}).get("supervision_manifest.json") == p2.MANIFEST_SHAS[c]
            and data["supervision_sha256"].get("calibration.npz") == p2.CALIBRATION_SHAS[c],
            "평가 결과 split/geometry edition SHA 불일치")
    for item in result["conditions"].values():
        records = item.get("records", [])
        require(len(records) == 1998 and all("pred_state" in r and "pred_history" in r for r in records),
                "조건별 신경 예측 1998행이 필요합니다")
    recheck_files(request)
    return {"output_sha256": result_sha, "protocol_sha256": protocol_sha, "child_receipt_sha256": receipt_sha,
            "normal_summary": result["conditions"]["normal"]["summary"]}


def exit_outcome(returncode, pressure, cancellation, result_valid):
    signal_number = -returncode if returncode < 0 else None
    if pressure is not None:
        outcome, code = "memory_pressure_failed", 1
    elif cancellation:
        outcome, code = "cancelled", 1
    elif returncode < 0:
        outcome, code = "child_signaled", 128 - returncode
    elif returncode:
        outcome, code = "child_nonzero_exit", returncode
    elif not result_valid:
        outcome, code = "zero_exit_without_verified_result", 1
    else:
        outcome, code = "completed_cleanly", 0
    return {"actual_returncode": returncode, "termination_signal": signal_number,
            "outcome": outcome, "wrapper_exit_code": code}


def supervise(request, *, process_factory=None, gpu_query=None, sleeper=None, monotonic=None,
              result_inspector=None, install_signal_handlers=True):
    """정확한 Popen child만 대상으로 한 5초 pressure 감시. 외부 PID 목록을 받지 않는다."""
    validate_request(request)
    query = gpu_query or query_evaluation_memory
    sleep, clock = sleeper or time.sleep, monotonic or time.monotonic
    popen = process_factory or subprocess.Popen
    selection = request_gpu_selection(request)
    gpu = selection["physical_gpu"]
    indices = namespace_indices(selection)
    # 기본 4-UUID 동작 보존. 명시적 opt-in은 선택 UUID 하나만 cuda:0으로 노출.
    devices = {i: checked_memory(query, i) for i in indices}
    admitted = checked_memory(query, gpu, devices[gpu]["gpu_uuid"], admission=True)
    request = {**request, **selection, "gpu_uuid_mapping": {str(i): d["gpu_uuid"] for i, d in devices.items()}}
    validate_request(request)
    request_sha = json_digest(request)
    record = {"schema_version": 1, "supervisor_record_id": uuid.uuid4().hex,
              "request": request, "request_sha256": request_sha, "parent_pid": os.getpid(),
              "child_pid": None, "created_at": now(), "status": "reserved", "actual_returncode": None,
              "preflight": admitted, "pressure_event": None, "received_signals": [],
              "pressure_checks": {"count": 0, "minimum_free_mib": None, "last": None},
              "signal_scope": "이 wrapper가 생성한 정확한 Popen child만. 다른 PID/process group은 제외",
              "gpu_uuid_mapping": {str(i): d["gpu_uuid"] for i, d in devices.items()},
              **selection,
              "memory_scope": "PyTorch allocator 제한과 시점별 여유 검사. GPU 독점 예약이나 OOM 방지 보장 아님"}
    owned = OwnedRecord(request["record"], record)
    pending, handlers = [], {}

    def receive(signum, _frame):
        pending.append(signum)

    if install_signal_handlers:
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, receive)
    child = None
    try:
        environment = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                       "CUDA_VISIBLE_DEVICES": ",".join(devices[i]["gpu_uuid"] for i in indices),
                       "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        command = [sys.executable, str(Path(request["root"]) / SCRIPT), "--child-record", request["record"],
                   "--expected-request-sha256", request_sha]
        try:
            child = popen(command, cwd=request["root"], env=environment, stdin=subprocess.DEVNULL,
                          stdout=None, stderr=None, start_new_session=False)
        except (OSError, subprocess.SubprocessError) as exc:
            record.update(status="spawn_failed", error=f"{type(exc).__name__}: {exc}", ended_at=now(), wrapper_exit_code=125)
            owned.write(record)
            return record
        record.update(child_pid=child.pid, command=command, status="running", started_at=now())
        owned.write(record)
        next_query, stop_since = -math.inf, None
        killed = False
        while True:
            rc = child.poll()
            for sig in pending[:]:
                pending.pop(0)
                event = {"signal": int(sig), "child_pid": child.pid, "at": now(), "sent": False}
                if rc is None:
                    try:
                        child.send_signal(sig)
                        event["sent"] = True
                    except ProcessLookupError:
                        event["already_exited"] = True
                    stop_since = clock() if stop_since is None else stop_since
                record["received_signals"].append(event)
            if rc is not None:
                valid = False
                if rc == 0:
                    try:
                        record["result_validation"] = (result_inspector or inspect_result)(request, child.pid, request_sha)
                        valid = True
                    except Exception as exc:
                        record["result_validation_error"] = f"{type(exc).__name__}: {exc}"
                record.update(status="process_exited", ended_at=now(),
                              **exit_outcome(rc, record["pressure_event"], bool(record["received_signals"]), valid))
                owned.write(record)
                return record
            if stop_since is None and clock() >= next_query:
                sample, failure = {}, None
                try:
                    sample = checked_memory(query, gpu, devices[gpu]["gpu_uuid"])
                    if sample["free_mib"] < RESERVE_MIB:
                        failure = "최소 여유 메모리 부족"
                except Exception as exc:
                    failure = "GPU 메모리 조회 실패"
                    sample = {"error": f"{type(exc).__name__}: {exc}"}
                sample["observed_at"] = now()
                checks = record["pressure_checks"]
                checks["count"] += 1
                checks["last"] = sample
                if "free_mib" in sample:
                    old = checks["minimum_free_mib"]
                    checks["minimum_free_mib"] = sample["free_mib"] if old is None else min(old, sample["free_mib"])
                next_query = clock() + PRESSURE_SECONDS
                if failure is not None:
                    record["pressure_event"] = {"reason": failure, "query": sample, "child_pid": child.pid,
                                                "sigterm_sent": False, "kill_sent": False, "observed_at": now()}
                    record["status"] = "pressure_terminating"
                    owned.write(record)  # 자신의 실패 증거를 먼저 보존한 뒤 신호를 보낸다.
                    try:
                        child.send_signal(signal.SIGTERM)
                        record["pressure_event"]["sigterm_sent"] = True
                    except ProcessLookupError:
                        record["pressure_event"]["already_exited"] = True
                    stop_since = clock()
            if stop_since is not None and not killed and clock() - stop_since >= GRACE_SECONDS:
                killed = True
                record["kill_attempted_at"] = now()
                owned.write(record)
                try:
                    child.kill()
                    record["kill_sent"] = True
                    if record["pressure_event"] is not None:
                        record["pressure_event"]["kill_sent"] = True
                except ProcessLookupError:
                    record["already_exited_before_kill"] = True
            owned.write(record)
            sleep(1.)
    except BaseException as exc:
        # 보호 parent 자체 오류가 생겨도 자신의 GPU child를 방치하지 않는다.
        failure = {"error": f"{type(exc).__name__}: {exc}", "observed_at": now(),
                   "child_pid": child.pid if child is not None else None,
                   "sigterm_sent": False, "kill_sent": False, "actual_returncode": None, "cleanup_errors": []}

        def cleanup_pause():
            try:
                sleep(1.)
            except BaseException as cleanup_error:
                failure["cleanup_errors"].append({"stage": "sleep", "error": f"{type(cleanup_error).__name__}: {cleanup_error}"})
                # 사용자 정의/중단된 sleeper의 실패가 owned child 회수를 건너뛰지 않도록 한다.
                time.sleep(1.)

        if child is not None:
            rc = child.poll()
            if rc is None:
                try:
                    child.send_signal(signal.SIGTERM)
                    failure["sigterm_sent"] = True
                except ProcessLookupError:
                    pass
                except BaseException as cleanup_error:
                    failure["cleanup_errors"].append({"stage": "sigterm", "error": f"{type(cleanup_error).__name__}: {cleanup_error}"})
                deadline = clock() + GRACE_SECONDS
                rc = child.poll()
                while rc is None and clock() < deadline:
                    cleanup_pause()
                    rc = child.poll()
                if rc is None:
                    try:
                        child.kill()
                        failure["kill_sent"] = True
                    except ProcessLookupError:
                        pass
                    except BaseException as cleanup_error:
                        failure["cleanup_errors"].append({"stage": "kill", "error": f"{type(cleanup_error).__name__}: {cleanup_error}"})
                # kill 전달은 종료 증거가 아니다. 이 Popen을 poll/reap할 때까지 소유권 유지.
                rc = child.poll()
                while rc is None:
                    cleanup_pause()
                    rc = child.poll()
            failure["actual_returncode"] = rc
        record.update(status="parent_failed", parent_failure=failure, ended_at=now(),
                      actual_returncode=failure["actual_returncode"], wrapper_exit_code=1)
        try:
            owned.write(record)
        except Exception as record_error:
            # 소유권이 변조된 record는 절대 덮어쓰지 않는다. 부모 stderr에 최소 증거 보존.
            print(json.dumps({"parent_failure": failure,
                              "record_not_overwritten": True,
                              "record_error": f"{type(record_error).__name__}: {record_error}"}, ensure_ascii=False),
                  file=sys.stderr, flush=True)
        raise
    finally:
        for sig, old in handlers.items():
            signal.signal(sig, old)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--launch-record")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-evaluation-git-sha")
    parser.add_argument("--out")
    parser.add_argument("--record")
    parser.add_argument("--physical-gpu", type=int, choices=range(6), default=None,
                        help="Explicit opt-in: expose this physical GPU alone as cuda:0; never GPU6/7")
    parser.add_argument("--child-record", help=argparse.SUPPRESS)
    parser.add_argument("--expected-request-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    public = ("arm", "launch_record", "expected_checkpoint_sha256", "expected_evaluation_git_sha", "out", "record")
    if args.child_record is not None:
        require(args.expected_request_sha256 is not None and args.physical_gpu is None
                and all(getattr(args, k) is None for k in public),
                "child 모드와 parent 옵션을 섞을 수 없습니다")
    else:
        require(args.expected_request_sha256 is None and all(getattr(args, k) is not None for k in public),
                "고정 평가에 필요한 모든 parent 인자를 명시해야 합니다")
    return args


def main(argv=None):
    args = arguments(argv)
    if args.child_record is not None:
        return run_child(args.child_record, args.expected_request_sha256)
    request = prepare(args.root, args.arm, args.out, args.record, args.launch_record,
                      args.expected_checkpoint_sha256, args.expected_evaluation_git_sha,
                      physical_gpu=args.physical_gpu)
    result = supervise(request)
    print(json.dumps({"status": result["status"], "outcome": result.get("outcome"),
                      "actual_returncode": result.get("actual_returncode"), "record": args.record}, ensure_ascii=False))
    return result["wrapper_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
