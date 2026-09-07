#!/usr/bin/env python3
"""Train/evaluate MotionDrive V2. Labels never enter the model forward API.

New run directory is required unless --resume is explicit. GPU ids 0--3 only.
Pretrain phase learns shared perception/motion with zero planning-loss weight;
joint phase loads that exact common checkpoint for the paired G×S experiment.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from motiondrive_v2_training import (LossWeights, compute_loss, model_inputs,
                                     raster_counts, set_training_mode, tensor_state_sha256,
                                     to_device, weighted_d3)
ACTIVE_RUN_DIR = None


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def atomic_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_checkpoint(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temp)
    os.replace(temp, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def autocast(device, precision):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and precision == "bf16" else contextlib.nullcontext())


def restore_run_configuration(args, saved, explicit_options):
    """Python G/S flags are NOT in state_dict: checkpoints own eval/resume config."""
    config = saved.get("model_config")
    saved_args = saved.get("arguments", {})
    if not config or "phase" not in saved_args:
        raise ValueError("Checkpoint lacks model/phase configuration")
    settings = {"goal_on": int(config["goal_on"]), "state_on": int(config["state_on"]),
                "arch": config["backbone_arch"], "phase": saved_args["phase"]}
    if hasattr(args, "motion_input_mode"):
        settings["motion_input_mode"] = config.get("motion_input_mode", "legacy")
    if hasattr(args, "plan_output_scale"):
        settings["plan_output_scale"] = list(config.get("plan_output_scale", (1., 1.)))
    if hasattr(args, "bn_policy"):
        settings["bn_policy"] = saved_args.get("bn_policy", "adaptive")
    if args.resume:
        for key in ("alpha_occ", "alpha_lane", "alpha_motion", "uncertainty", "lr",
                    "backbone_lr", "weight_decay", "warmup", "precision", "seed"):
            if key not in saved_args:
                raise ValueError(f"Resume configuration missing {key}")
            settings[key] = saved_args[key]
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if flag in explicit_options and getattr(args, key) != value:
            raise ValueError(f"Checkpoint {key}={value}, incompatible explicit {flag}; use --init for a new experiment")
        setattr(args, key, value)
    return config


def initialization_configuration(saved, *, goal_on, state_on, explicit_arch=None):
    """A fresh paired run inherits every architecture field, changing G/S only."""
    config = dict(saved.get("model_config", {}))
    if not config or "backbone_arch" not in config:
        raise ValueError("Initialization lacks model configuration; do not guess defaults")
    if explicit_arch is not None and explicit_arch != config["backbone_arch"]:
        raise ValueError("--init cannot silently change backbone architecture")
    config.update(goal_on=bool(goal_on), state_on=bool(state_on))
    return config


@torch.inference_mode()
def evaluate(model, loader, device, precision):
    model.eval()
    records, state_errors, history_errors = [], [], []
    iou_counts = {"occ": [0, 0], "lane": [0, 0]}
    for raw in loader:
        batch = to_device(raw, device)
        with autocast(device, precision):
            out = model(**model_inputs(batch))
        if not torch.isfinite(out["plan_abs"]).all():
            raise FloatingPointError("Nonfinite validation prediction")
        d3 = weighted_d3(out["plan_abs"], batch["gt_plan"]).cpu().numpy()
        if not batch["plan_valid"].bool().all():
            raise ValueError("Validation contains invalid three-second ground truth")
        scenarios = raw["scenario"]
        sessions = raw["session_id"]
        frames = raw["frame"].tolist()
        proxy = raw.get("proxy_weight", torch.ones(len(d3))).tolist()
        for i, value in enumerate(d3):
            records.append({"scenario": scenarios[i], "session": sessions[i],
                            "frame": int(frames[i]), "d3": float(value),
                            "proxy": float(proxy[i])})
        err = (out["state_hat"][:, :5] - batch["state_target"][:, :5]).abs()
        mask = batch["state_valid"][:, :5].bool()
        state_errors.extend(torch.where(mask, err, float("nan")).cpu().tolist())
        he = torch.linalg.vector_norm(out["history_hat"][..., :2] - batch["history_target"][..., :2], dim=-1)
        hv = batch["history_valid"][..., :2].all(-1)
        history_errors.extend(torch.where(hv, he, float("nan")).cpu().tolist())
        for task in iou_counts:
            a, b = raster_counts(out[f"{task}_logits"], batch[f"{task}_target"], batch[f"{task}_valid"])
            iou_counts[task][0] += a
            iou_counts[task][1] += b
    if not records:
        raise ValueError("Empty evaluation split")
    d3 = np.array([r["d3"] for r in records])
    proxy = np.array([r["proxy"] for r in records])
    by_session = {}
    for row in records:
        by_session.setdefault(row["session"], []).append(row["d3"])
    def finite_mean(values, axis=0):
        values = np.asarray(values, dtype=float)
        count = np.isfinite(values).sum(axis=axis)
        result = np.nansum(values, axis=axis) / np.maximum(count, 1)
        return [float(v) if n else None for v, n in zip(np.ravel(result), np.ravel(count))]
    report = {
        "n": len(records), "official_d3": float(d3.mean()),
        "session_mean_d3": float(np.mean([np.mean(v) for v in by_session.values()])),
        "n_sessions": len(by_session),
        "proxy_d3": float(np.average(d3, weights=proxy)) if proxy.sum() > 0 else None,
        "proxy_is_not_official_sample_weight": True,
        "state_mae_vx_vy_ax_ay_yawrate": finite_mean(state_errors),
        "history_position_mae_by_offset": finite_mean(history_errors),
        "session_d3": {k: float(np.mean(v)) for k, v in by_session.items()},
    }
    for name, (inter, union) in iou_counts.items():
        report[f"{name}_iou"] = inter / union if union else None
    return report, records


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(ROOT))
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--supervision-root", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--phase", choices=["pretrain", "joint"], default="joint")
    p.add_argument("--goal-on", type=int, choices=[0, 1], default=1)
    p.add_argument("--state-on", type=int, choices=[0, 1], default=1)
    p.add_argument("--gpu", type=int, choices=[0, 1, 2, 3], default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--eval-batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=5.)
    p.add_argument("--alpha-occ", type=float, default=.2)
    p.add_argument("--alpha-lane", type=float, default=.2)
    p.add_argument("--alpha-motion", type=float, default=.2)
    p.add_argument("--uncertainty", type=int, choices=[0, 1], default=1)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--bn-policy", choices=["adaptive", "fixed"], default="adaptive",
                   help="Training only: adaptive batch statistics, or fixed running statistics; all weights still train")
    p.add_argument("--pretrained", help="Public backbone only; never a holdout-trained checkpoint")
    p.add_argument("--init", help="Common fold-trained initialization; weights only")
    p.add_argument("--resume", help="Explicit full resume checkpoint")
    p.add_argument("--arch", choices=["resnet34", "resnet50"], default="resnet50")
    p.add_argument("--motion-input-mode", choices=["legacy", "high_feature", "low_feature"])
    p.add_argument("--plan-output-scale", type=float, nargs=2, metavar=("X", "Y"),
                   help="Internal neural output units; inverse-rescale last Linear to preserve initial predictions")
    p.add_argument("--train-stride", type=int, default=1)
    p.add_argument("--eval-stride", type=int, default=5)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument("--train-scenes", nargs="+")
    p.add_argument("--eval-scenes", nargs="+")
    p.add_argument("--eval-split", choices=["train", "tune", "val", "historical_val"], default="tune",
                   help="train is diagnostic eval-only; only tune may select checkpoints")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--allow-unpretrained", action="store_true", help="Explicit diagnostic only")
    return p.parse_args()


def main():
    global ACTIVE_RUN_DIR
    args = arguments()
    explicit = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
    from motiondrive_v2_data import MotionDriveDataset
    from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
    if args.init and args.resume:
        raise ValueError("Choose --init or --resume, not both")
    if not (args.pretrained or args.init or args.resume or args.allow_unpretrained):
        raise ValueError("Refusing accidental random-backbone training")
    if args.eval_split != "tune" and not args.eval_only:
        raise ValueError("Checkpoint selection must use tune, never final val")
    if args.steps < 1 or args.batch < 1 or args.eval_every < 1:
        raise ValueError("steps/batch/eval-every must be positive")
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cpu" if args.cpu else f"cuda:{args.gpu}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
    load_report = {}
    common = None
    if args.init or args.resume:
        common = torch.load(args.resume or args.init, map_location="cpu", weights_only=False)
        expected_split_sha = common.get("manifest", {}).get("split_sha256")
        if expected_split_sha != sha256(args.split_manifest):
            raise ValueError("Initialization/resume split lineage mismatch or missing")
        load_report["common_checkpoint_sha256"] = sha256(args.resume or args.init)
    if common is not None and (args.resume or args.eval_only):
        config = MotionDriveV2Config(**restore_run_configuration(args, common["manifest"], explicit))
    elif common is not None:
        config = MotionDriveV2Config(**initialization_configuration(
            common["manifest"], goal_on=args.goal_on, state_on=args.state_on,
            explicit_arch=args.arch if "--arch" in explicit else None))
        args.arch = config.backbone_arch
    else:
        config = MotionDriveV2Config(backbone_arch=args.arch, goal_on=bool(args.goal_on),
                                    state_on=bool(args.state_on))
    if args.phase == "pretrain":
        config.goal_on = False
        config.state_on = False
    if args.motion_input_mode is not None:
        previous_mode = config.motion_input_mode
        config.motion_input_mode = args.motion_input_mode
        if previous_mode != config.motion_input_mode:
            load_report["motion_input_mode_override"] = {"from": previous_mode, "to": config.motion_input_mode}
    model = MotionDriveV2(config)
    if args.pretrained:
        load_report["backbone"] = model.load_pretrained_backbone(args.pretrained)
        if (load_report["backbone"]["nonhead_missing"] or load_report["backbone"]["unexpected"]):
            raise ValueError(f"Incomplete public backbone load: {load_report['backbone']}")
    if common is not None:
        model.load_state_dict(common["model"], strict=True)
    if args.plan_output_scale is not None:
        previous_scale = list(config.plan_output_scale)
        model.reparameterize_plan_output_scale(args.plan_output_scale, preserve_function=True)
        load_report["plan_output_unit_reparameterization"] = {
            "from": previous_scale, "to": list(config.plan_output_scale),
            "initial_function_preserved": True,
            "note": "Inverse scale of final Linear rows; Adam parameterization changes, not target coordinates"}
    initial_state_sha = tensor_state_sha256(model.state_dict())
    model.to(device)
    bn_module_count = set_training_mode(model, args.bn_policy)
    backbone, other = [], []
    for name, param in model.named_parameters():
        is_backbone = name.startswith("backbone_fpn.") and not name.startswith(
            ("backbone_fpn.lat", "backbone_fpn.out"))
        (backbone if is_backbone else other).append(param)
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr, "base_lr": args.backbone_lr},
        {"params": other, "lr": args.lr, "base_lr": args.lr},
    ], weight_decay=args.weight_decay)
    weights = LossWeights(plan=0. if args.phase == "pretrain" else 1.,
                          occupancy=args.alpha_occ, lane=args.alpha_lane,
                          motion=args.alpha_motion, uncertainty=bool(args.uncertainty))
    manifest = {
        "schema_version": 1, "git_sha": source_sha(), "arguments": vars(args),
        "model_config": dataclasses.asdict(config), "loss_weights": dataclasses.asdict(weights),
        "split_sha256": sha256(args.split_manifest), "load_report": load_report,
        "torch": torch.__version__, "numpy": np.__version__,
        "device": str(device), "metric": "mean of cumulative ADE@1/2/3s; no sample proxy",
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "pretrained_sha256": sha256(args.pretrained) if args.pretrained else None,
        "supervision_manifest_sha256": sha256(Path(args.supervision_root) / "supervision_manifest.json"),
        "initial_model_state_sha256": initial_state_sha,
        "initial_parameter_count": sum(p.numel() for p in model.parameters()),
        "bn_training": {"policy": args.bn_policy, "modules": bn_module_count,
                        "affine_and_backbone_weights_trainable": True,
                        "inference_policy": "eval running statistics for both arms"},
        "status": "starting", "pid": os.getpid(),
    }
    atomic_json(run_dir / "manifest.json", manifest)
    ACTIVE_RUN_DIR = run_dir
    def dataset(split, stride, maximum, scenes):
        kwargs = dict(data_root=args.data_root, split_manifest=args.split_manifest,
                      split=split, supervision_root=args.supervision_root,
                      min_frame=30, frame_stride=stride, max_samples=maximum,
                      augment=(split == "train" and not args.eval_only), seed=args.seed)
        if scenes:
            kwargs["scenes"] = scenes
        return MotionDriveDataset(**kwargs)
    evaluation = dataset(args.eval_split, args.eval_stride, args.max_eval_samples, args.eval_scenes)
    eval_loader = DataLoader(evaluation, batch_size=args.eval_batch, shuffle=False,
                             num_workers=args.workers, pin_memory=device.type == "cuda")
    if args.eval_only:
        report, records = evaluate(model, eval_loader, device, args.precision)
        atomic_json(run_dir / "evaluation.json", {"report": report, "records": records})
        print(json.dumps(report), flush=True)
        manifest.update(status="completed", evaluation=report)
        atomic_json(run_dir / "manifest.json", manifest)
        return
    training = dataset("train", args.train_stride, args.max_train_samples, args.train_scenes)
    manifest["data_counts"] = {"train": len(training), "eval": len(evaluation)}
    manifest["train_rows_sha256"] = hashlib.sha256(
        np.asarray(training.rows, dtype="<i8").tobytes()).hexdigest()
    manifest["eval_rows_sha256"] = hashlib.sha256(
        np.asarray(evaluation.rows, dtype="<i8").tobytes()).hexdigest()
    manifest["sample_order_policy"] = "dedicated torch generator(seed); row+epoch deterministic photometric jitter; rolling row SHA per log"
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(training, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda",
                              generator=generator, worker_init_fn=worker_seed,
                              drop_last=False, persistent_workers=False)
    if len(train_loader) == 0:
        raise ValueError("Empty training split")
    step, epoch, best, nonfinite_count = 0, 0, math.inf, 0
    if args.resume:
        optimizer.load_state_dict(common["optimizer"])
        step, epoch, best = common["step"], common.get("epoch", 0), common.get("best_metric", math.inf)
        if "rng" in common:
            torch.set_rng_state(common["rng"]["torch"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(common["rng"]["cuda"], device)
            np.random.set_state(common["rng"]["numpy"])
            random.setstate(common["rng"]["python"])
        # Exact in-epoch sample replay is not promised for resumes.
        manifest["resume_sample_order_exact"] = False
    requested_stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: requested_stop.__setitem__(0, True))
    def checkpoint():
        rng = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
               "python": random.getstate()}
        if device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(device)
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "epoch": epoch, "best_metric": best,
                "manifest": manifest, "rng": rng}
    if not args.resume:
        atomic_checkpoint(run_dir / "initial.pth", checkpoint())
    started = time.monotonic()
    sample_order_digest = hashlib.sha256()
    manifest["status"] = "running"
    atomic_json(run_dir / "manifest.json", manifest)
    with (run_dir / "metrics.jsonl").open("a", buffering=1) as log:
        if not args.resume:
            initial_report, _ = evaluate(model, eval_loader, device, args.precision)
            atomic_json(run_dir / "initial_eval.json", initial_report)
            log.write(json.dumps({"kind": "initial_eval", "step": 0, **initial_report}, allow_nan=False) + "\n")
            print(json.dumps({"kind": "initial_eval", "step": 0, **initial_report}), flush=True)
        while step < args.steps and not requested_stop[0]:
            if hasattr(training, "set_epoch"):
                training.set_epoch(epoch)
            for raw in train_loader:
                if step >= args.steps or requested_stop[0]:
                    break
                set_training_mode(model, args.bn_policy)
                sample_order_digest.update(np.asarray(raw["row"], dtype="<i8").tobytes())
                batch = to_device(raw, device)
                warm = min(1., (step + 1) / max(1, args.warmup))
                progress = max(0., (step - args.warmup) / max(1, args.steps - args.warmup))
                factor = warm * .5 * (1. + math.cos(math.pi * min(1., progress)))
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * factor
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, args.precision):
                    output = model(**model_inputs(batch))
                loss, parts = compute_loss(output, batch, weights)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite loss at step {step}; no silent NaN skip")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip,
                                                       error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if step % args.log_every == 0 or step == 1:
                    row = {"kind": "train", "step": step, "epoch": epoch,
                           "elapsed_seconds": time.monotonic() - started,
                           "grad_norm": float(norm), "lr": optimizer.param_groups[-1]["lr"],
                           "sample_order_sha256": sample_order_digest.hexdigest(),
                           **{k: float(v.detach()) for k, v in parts.items()}}
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    print(json.dumps(row, allow_nan=False), flush=True)
                if step % args.eval_every == 0 or step == args.steps:
                    report, _ = evaluate(model, eval_loader, device, args.precision)
                    # Pretrain selection is task quality, not an untrained planner's D3.
                    if args.phase == "pretrain":
                        motion_value = report["history_position_mae_by_offset"]
                        valid_motion = [x for x in motion_value if x is not None]
                        if not valid_motion:
                            raise ValueError("No valid motion labels in pretrain evaluation")
                        score = float(np.mean(valid_motion))
                    else:
                        score = report["official_d3"]
                    row = {"kind": "eval", "step": step, "selection_metric": score,
                           "selection_definition": "history_position_mae" if args.phase == "pretrain" else "official_d3",
                           **report}
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    print(json.dumps(row, allow_nan=False), flush=True)
                    atomic_json(run_dir / "latest_eval.json", row)
                    if score < best:
                        best = score
                        atomic_checkpoint(run_dir / "best.pth", checkpoint())
                if step % args.save_every == 0:
                    atomic_checkpoint(run_dir / "last.pth", checkpoint())
            epoch += 1
        atomic_checkpoint(run_dir / "last.pth", checkpoint())
    manifest.update(status="stopped" if requested_stop[0] else "completed", step=step,
                    best_metric=best if math.isfinite(best) else None,
                    elapsed_seconds=time.monotonic() - started, nonfinite_count=nonfinite_count)
    atomic_json(run_dir / "manifest.json", manifest)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if ACTIVE_RUN_DIR is not None:
            manifest_path = ACTIVE_RUN_DIR / "manifest.json"
            with manifest_path.open() as stream:
                failed = json.load(stream)
            if failed.get("pid") == os.getpid():
                failed.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                atomic_json(manifest_path, failed)
        raise
