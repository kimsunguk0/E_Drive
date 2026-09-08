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
from motiondrive_v2_training import (LossWeights, compute_loss, global_binary_class_weights, model_inputs,
                                     raster_counts, set_training_mode, tensor_state_sha256,
                                     time_input_policy, TIME_INPUT_MODES, to_device, weighted_d3)
ACTIVE_RUN_DIR = None
MIB = 1024 ** 2


def configure_cuda_memory(device, limit_mib=0, min_free_mib=0):
    """Opt-in allocator cap before model allocation; no whole-GPU reservation.

    The PyTorch cap does not cover all CUDA-library allocations or prevent a
    different process from growing. Keep external headroom and runtime guards.
    """
    if any(type(v) is not int or v < 0 for v in (limit_mib, min_free_mib)):
        raise ValueError("CUDA memory values must be nonnegative integer MiB")
    if bool(limit_mib) != bool(min_free_mib):
        raise ValueError("CUDA limit and minimum free memory must be enabled together")
    if not limit_mib:
        return {"enabled": False}
    if device.type != "cuda":
        raise ValueError("CUDA memory sharing requires a CUDA device")
    free, total = torch.cuda.mem_get_info(device)
    requested = (limit_mib + min_free_mib) * MIB
    if not (0 < free <= total) or requested > free:
        raise RuntimeError(f"Insufficient CUDA headroom before allocation: free={free / MIB:.1f} MiB, "
                           f"required={limit_mib}+{min_free_mib} MiB")
    physical_total = torch.cuda.get_device_properties(device).total_memory
    if limit_mib * MIB >= physical_total:
        raise ValueError("CUDA allocator limit must be below device capacity")
    fraction = float(limit_mib * MIB / physical_total)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    return {"enabled": True, "allocator_limit_mib": limit_mib,
            "min_free_mib": min_free_mib, "allocator_fraction": fraction,
            "initial_free_bytes": free, "device_total_bytes": physical_total,
            "scope": "PyTorch caching allocator only; other CUDA allocations and other processes are not capped"}


def check_cuda_headroom(device, min_free_mib=0):
    if not min_free_mib:
        return
    free, _ = torch.cuda.mem_get_info(device)
    if free < min_free_mib * MIB:
        raise RuntimeError(f"CUDA shared-memory pressure: free={free / MIB:.1f} MiB "
                           f"below reserve={min_free_mib} MiB; stopping only this trainer")


def cuda_memory_snapshot(device):
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device_free_bytes": free, "device_total_bytes": total}


def slice_batch(batch, start, end):
    """Slice collated samples before GPU transfer, retaining metadata order."""
    size = len(batch["images"])
    result = {}
    for key, value in batch.items():
        if isinstance(value, (torch.Tensor, list, tuple)):
            if len(value) != size:
                raise ValueError(f"Non-sample batch field cannot be microbatched: {key}")
            result[key] = value[start:end]
        else:
            raise TypeError(f"Unsupported collated batch field: {key}")
    return result


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
    if hasattr(args, "cross_cell_goal_mode"):
        settings["cross_cell_goal_mode"] = config.get("cross_cell_goal_mode", "disabled")
    if hasattr(args, "history_contract"):
        settings["history_contract"] = config.get("history_contract", "control")
    if hasattr(args, "bn_policy"):
        settings["bn_policy"] = saved_args.get("bn_policy", "adaptive")
    if hasattr(args, "time_input"):
        settings["time_input"] = saved_args.get("time_input", "raw")
        time_input_policy(settings["time_input"],
                          config.get("nominal_history_seconds", (.1, .2, .5, 1.)))
        saved_time_policy = saved.get("time_input_policy", {})
        if not isinstance(saved_time_policy, dict):
            raise ValueError("Checkpoint time_input_policy must be a mapping")
        for declaration in (saved.get("time_input", settings["time_input"]),
                            saved_time_policy.get("mode", settings["time_input"])):
            if declaration != settings["time_input"]:
                raise ValueError("Checkpoint time_input declarations conflict")
    if args.resume:
        for key in ("alpha_occ", "alpha_lane", "alpha_motion", "uncertainty", "lr",
                    "backbone_lr", "weight_decay", "warmup", "precision", "seed"):
            if key not in saved_args:
                raise ValueError(f"Resume configuration missing {key}")
            settings[key] = saved_args[key]
        for key in ("microbatch", "cuda_memory_limit_mib", "cuda_min_free_mib"):
            if hasattr(args, key):
                settings[key] = saved_args.get(key, 0)
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if flag in explicit_options and getattr(args, key) != value:
            raise ValueError(f"Checkpoint {key}={value}, incompatible explicit {flag}; use --init for a new experiment")
        setattr(args, key, value)
    return config


def initialization_configuration(saved, *, goal_on, state_on, explicit_arch=None,
                                 cross_cell_goal_mode=None, history_contract=None,
                                 allow_legacy_history_override=False):
    """A fresh paired run inherits every architecture field, changing G/S only."""
    config = dict(saved.get("model_config", {}))
    if not config or "backbone_arch" not in config:
        raise ValueError("Initialization lacks model configuration; do not guess defaults")
    if explicit_arch is not None and explicit_arch != config["backbone_arch"]:
        raise ValueError("--init cannot silently change backbone architecture")
    config.update(goal_on=bool(goal_on), state_on=bool(state_on))
    if cross_cell_goal_mode is not None:
        config["cross_cell_goal_mode"] = cross_cell_goal_mode
    if history_contract is not None:
        from models.motiondrive_v2_temporal_contract import temporal_contract
        temporal = temporal_contract(history_contract)
        temporal_keys = {"history_contract", "history_frame_offsets", "nominal_history_seconds"}
        present = temporal_keys & set(config)
        if present and present != temporal_keys:
            raise ValueError("Checkpoint has a partial temporal contract")
        if present:
            saved_temporal = temporal_contract(config["history_contract"])
            if (tuple(config["history_frame_offsets"]) != saved_temporal.frame_offsets
                    or tuple(config["nominal_history_seconds"]) != saved_temporal.nominal_seconds):
                raise ValueError("Checkpoint temporal contract is internally inconsistent")
            if saved_temporal.name != temporal.name:
                raise ValueError("A trained checkpoint temporal contract cannot be overridden")
        elif temporal.name != "control" and not allow_legacy_history_override:
            raise ValueError("Only the exact neutral public initializer may acquire wide history metadata")
        config.update(history_contract=temporal.name,
                      history_frame_offsets=temporal.frame_offsets,
                      nominal_history_seconds=temporal.nominal_seconds)
    return config


@torch.inference_mode()
def evaluate(model, loader, device, precision, time_input="raw", min_free_mib=0,
             detailed_records=False, nominal_history_seconds=(.1, .2, .5, 1.)):
    time_input_policy(time_input, nominal_history_seconds)
    model.eval()
    records, state_errors, history_errors = [], [], []
    iou_counts = {"occ": [0, 0], "lane": [0, 0]}
    for raw in loader:
        check_cuda_headroom(device, min_free_mib)
        batch = to_device(raw, device)
        with autocast(device, precision):
            out = model(**model_inputs(batch, time_input=time_input,
                                       nominal_history_seconds=nominal_history_seconds))
        if not torch.isfinite(out["plan_abs"]).all():
            raise FloatingPointError("Nonfinite validation prediction")
        d3 = weighted_d3(out["plan_abs"], batch["gt_plan"]).cpu().numpy()
        if not batch["plan_valid"].bool().all():
            raise ValueError("Validation contains invalid three-second ground truth")
        scenarios = raw["scenario"]
        sessions = raw["session_id"]
        frames = raw["frame"].tolist()
        proxy = raw.get("proxy_weight", torch.ones(len(d3))).tolist()
        if detailed_records:
            detailed_pred = out["plan_abs"].float().cpu()
            detailed_gt = batch["gt_plan"].float().cpu()
            detailed_pred_state = out["state_hat"].float().cpu()
            detailed_gt_state = batch["state_target"].float().cpu()
            detailed_gt_state_valid = batch["state_valid"].bool().cpu()
            detailed_stop_valid = batch["state_valid"][:, 5].bool().cpu()
            detailed_stop_target = batch["state_target"][:, 5].float().cpu()
            detailed_max_displacement = torch.linalg.vector_norm(detailed_gt, dim=-1).max(-1).values
        for i, value in enumerate(d3):
            record = {"scenario": scenarios[i], "session": sessions[i],
                      "frame": int(frames[i]), "d3": float(value),
                      "proxy": float(proxy[i])}
            if detailed_records:
                valid_stop = bool(detailed_stop_valid[i])
                stop_target = float(detailed_stop_target[i]) if valid_stop else None
                gt = detailed_gt[i]
                max_displacement = float(detailed_max_displacement[i])
                bucket = ("invalid_stop" if not valid_stop else
                          "nonstop" if stop_target < .5 else
                          "steady" if max_displacement <= .2 else "depart")
                record.update(row=int(raw["row"][i]),
                              pred_abs_xy=detailed_pred[i].tolist(),
                              gt_abs_xy=gt.tolist(), stop_valid=valid_stop,
                              stop_target=stop_target, bucket=bucket,
                              stop_bucket=bucket,
                              max_gt_displacement_m=max_displacement,
                              pred_state=detailed_pred_state[i].tolist(),
                              gt_state=detailed_gt_state[i].tolist(),
                              gt_state_valid=detailed_gt_state_valid[i].tolist())
            records.append(record)
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
        "time_input": time_input,
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


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--data-root", default=str(ROOT))
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--supervision-root", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--phase", choices=["pretrain", "joint"], default="joint")
    p.add_argument("--goal-on", type=int, choices=[0, 1], default=1)
    p.add_argument("--state-on", type=int, choices=[0, 1], default=1)
    p.add_argument("--gpu", type=int, choices=[0, 1, 2, 3], default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--cuda-memory-limit-mib", type=int, default=0,
                   help="Opt-in PyTorch allocator limit; requires --cuda-min-free-mib")
    p.add_argument("--cuda-min-free-mib", type=int, default=0,
                   help="Shared GPU safety reserve; stop our trainer if free memory falls below this")
    p.add_argument("--microbatch", type=int, default=0,
                   help="Split each logical --batch for gradient accumulation; 0 preserves the original path; fixed BN only")
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
    p.add_argument("--time-input", choices=TIME_INPUT_MODES, default="raw",
                   help="Only model time_offsets: raw preserves P1, nominal fixes [.1,.2,.5,1.] seconds; GT is unchanged")
    p.add_argument("--bn-policy", choices=["adaptive", "fixed"], default="adaptive",
                   help="Training only: adaptive batch statistics, or fixed running statistics; all weights still train")
    p.add_argument("--pretrained", help="Public backbone only; never a holdout-trained checkpoint")
    p.add_argument("--init", help="Common fold-trained initialization; weights only")
    p.add_argument("--resume", help="Explicit full resume checkpoint")
    p.add_argument("--arch", choices=["resnet34", "resnet50"], default="resnet50")
    p.add_argument("--motion-input-mode", choices=["legacy", "high_feature", "low_feature"])
    p.add_argument("--cross-cell-goal-mode", choices=["disabled", "zero", "real"])
    p.add_argument("--history-contract", choices=["control", "wide"])
    p.add_argument("--history-overlay-root")
    p.add_argument("--expected-history-overlay-sha256")
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
    return p.parse_args(argv)


def _validate_experimental_protocol(experiment):
    if experiment is None:
        return None
    if isinstance(experiment, dict) and experiment.get("name") == "p8_wide_history":
        required = {"schema_version", "name", "stage", "arm", "temporal_contract",
                    "history_overlay_manifest_sha256", "expected_initial_checkpoint_sha256",
                    "expected_p0_model_state_sha256", "expected_initial_model_state_sha256",
                    "expected_optimizer_groups", "expected_branch_state_sha256", "branch",
                    "expected_missing_state_keys", "train_data", "tune_data", "source",
                    "fresh_optimizer_step_zero", "all_model_parameters_trainable",
                    "p0_branch_disabled", "joint_branch_zero", "final_validation_accessed"}
        if set(experiment) != required:
            raise ValueError("P8 experimental protocol must be complete")
        from models.motiondrive_v2_temporal_contract import temporal_contract
        temporal = temporal_contract(experiment["arm"])
        expected_temporal = {"name": temporal.name,
                             "frame_offsets": list(temporal.frame_offsets),
                             "nominal_seconds": list(temporal.nominal_seconds)}
        stage = experiment["stage"]
        expected_branch = {
            "mode": "disabled" if stage == "pretrain" else "zero",
            "sigma_m": [10., 32. / 3.], "pool_size": 4, "source_cells": 192,
            "destination_cells": 3072, "attention_dim": 32, "cosine_scale": 8.,
            "attention_precision": "fp32_autocast_disabled",
            "goal_enters_distance_score_only": True,
            "new_value_projection_adds_goal_or_position": False,
            "output_bias": False, "output_weight_zero_initialized_at_joint": True}
        if (experiment["schema_version"] != 1 or stage not in ("pretrain", "joint")
                or experiment["temporal_contract"] != expected_temporal
                or experiment["fresh_optimizer_step_zero"] is not True
                or experiment["all_model_parameters_trainable"] is not True
                or experiment["p0_branch_disabled"] is not True
                or experiment["joint_branch_zero"] is not True
                or experiment["final_validation_accessed"] is not False
                or experiment["branch"] != expected_branch):
            raise ValueError("Unsupported P8 experimental protocol")
        for key in ("history_overlay_manifest_sha256", "expected_initial_checkpoint_sha256",
                    "expected_p0_model_state_sha256", "expected_initial_model_state_sha256"):
            if not isinstance(experiment[key], str) or len(experiment[key]) != 64:
                raise ValueError(f"P8 {key} must be a full SHA256")
        if experiment["expected_optimizer_groups"] != [
                {"name": "backbone", "base_lr": 1e-5},
                {"name": "head", "base_lr": 1e-4}]:
            raise ValueError("P8 optimizer group contract mismatch")
        missing = experiment["expected_missing_state_keys"]
        branch_sha = experiment["expected_branch_state_sha256"]
        if stage == "pretrain" and (missing != [] or branch_sha is not None):
            raise ValueError("P8 P0 must use the branch-disabled legacy graph")
        if stage == "joint" and (not isinstance(missing, list) or not missing
                or len(missing) != len(set(missing))
                or not all(key.startswith("scene_encoder.cross_cell_goal_residual.") for key in missing)
                or not isinstance(branch_sha, str) or len(branch_sha) != 64):
            raise ValueError("P8 joint must add exactly the new residual state")
        if experiment["train_data"] != {"rows": 54810,
                "rows_sha256": "75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854"}:
            raise ValueError("P8 fixed train rows contract mismatch")
        if experiment["tune_data"] != {"rows": 1998,
                "rows_sha256": "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88"}:
            raise ValueError("P8 fixed tune rows contract mismatch")
        if not isinstance(experiment["source"], dict) or not experiment["source"]:
            raise ValueError("P8 source provenance is required")
        return None
    if isinstance(experiment, dict) and experiment.get("name") == "p7_cross_cell_goal_routing":
        required = {"schema_version", "name", "arm", "last_only_final_eval",
                    "expected_initial_model_state_sha256", "expected_p0_model_state_sha256",
                    "expected_optimizer_groups", "expected_missing_state_keys",
                    "train_data", "tune_data", "source", "fresh_optimizer_step_zero",
                    "all_model_parameters_joint_trainable", "existing_goal_path_on",
                    "planner_signature_unchanged", "branch"}
        if set(experiment) != required:
            raise ValueError("P7 experimental protocol must be complete")
        mode = {"control_zero_slot": "zero", "goal_real_slot": "real"}.get(experiment["arm"])
        branch = experiment["branch"]
        expected_branch = {
            "mode": mode, "sigma_m": [10., 32. / 3.], "sigma_selection": "fixed_geometry_not_tuned",
            "pool_size": 4, "source_cells": 192, "destination_cells": 3072,
            "attention_dim": 32, "cosine_scale": 8.,
            "attention_precision": "fp32_autocast_disabled",
            "goal_enters_distance_score_only": True,
            "new_value_projection_adds_goal_or_position": False, "output_bias": False,
            "output_weight_zero_initialized": True,
        }
        if (experiment["schema_version"] != 1 or mode is None
                or experiment["last_only_final_eval"] is not True
                or experiment["fresh_optimizer_step_zero"] is not True
                or experiment["all_model_parameters_joint_trainable"] is not True
                or experiment["existing_goal_path_on"] is not True
                or experiment["planner_signature_unchanged"] is not True
                or branch != expected_branch):
            raise ValueError("Unsupported P7 experimental protocol")
        for key in ("expected_initial_model_state_sha256", "expected_p0_model_state_sha256"):
            value = experiment[key]
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"P7 {key} is required")
        if experiment["expected_optimizer_groups"] != [
                {"name": "backbone", "base_lr": 1e-5}, {"name": "head", "base_lr": 1e-4}]:
            raise ValueError("P7 optimizer group contract mismatch")
        missing = experiment["expected_missing_state_keys"]
        if (not isinstance(missing, list) or not missing
                or len(missing) != len(set(missing))
                or not all(isinstance(key, str)
                           and key.startswith("scene_encoder.cross_cell_goal_residual.")
                           for key in missing)):
            raise ValueError("P7 old-checkpoint missing-key contract is invalid")
        if experiment["train_data"] != {
                "rows": 54810,
                "rows_sha256": "75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854"}:
            raise ValueError("P7 fixed train rows contract mismatch")
        if experiment["tune_data"] != {
                "rows": 1998,
                "rows_sha256": "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88"}:
            raise ValueError("P7 fixed tune rows contract mismatch")
        if not isinstance(experiment["source"], dict) or not experiment["source"]:
            raise ValueError("P7 source provenance is required")
        return None
    required = {"schema_version", "name", "arm", "last_only_final_eval", "stop_class_weights",
                "expected_initial_model_state_sha256", "expected_optimizer_groups",
                "train_label_counts", "source", "fresh_optimizer_step_zero",
                "all_model_parameters_joint_trainable", "stop_target_definition",
                "balanced_output_is_calibrated_posterior", "raw_stop_logit_is_planner_input"}
    if not isinstance(experiment, dict) or set(experiment) != required:
        raise ValueError("Experimental protocol must be a complete changed-configuration manifest")
    if (experiment["schema_version"] != 1 or experiment["name"] != "p6_global_stop_class_balance"
            or experiment["arm"] not in ("control_unweighted", "balanced_global_train")
            or experiment["last_only_final_eval"] is not True
            or experiment["fresh_optimizer_step_zero"] is not True
            or experiment["all_model_parameters_joint_trainable"] is not True
            or experiment["balanced_output_is_calibrated_posterior"] is not False
            or experiment["raw_stop_logit_is_planner_input"] is not True):
        raise ValueError("Unsupported experimental protocol")
    expected_weights = None if experiment["arm"] == "control_unweighted" else experiment["stop_class_weights"]
    if experiment["arm"] == "control_unweighted" and experiment["stop_class_weights"] is not None:
        raise ValueError("Control arm must use the exact unweighted stop BCE")
    if expected_weights is not None:
        if (not isinstance(expected_weights, list) or len(expected_weights) != 2
                or any(type(x) is not float or not math.isfinite(x) or x <= 0 for x in expected_weights)):
            raise ValueError("Treatment stop class weights must be finite positive [negative, positive]")
        expected_weights = tuple(expected_weights)
    if (not isinstance(experiment["expected_initial_model_state_sha256"], str)
            or len(experiment["expected_initial_model_state_sha256"]) != 64):
        raise ValueError("Expected initial state SHA is required")
    groups = experiment["expected_optimizer_groups"]
    if groups != [{"name": "backbone", "base_lr": 5e-6}, {"name": "head", "base_lr": 5e-5}]:
        raise ValueError("P6 optimizer group contract mismatch")
    counts = experiment["train_label_counts"]
    if (not isinstance(counts, dict)
            or set(counts) != {"rows", "rows_sha256", "labels_sha256", "valid", "negative", "positive"}
            or any(type(counts[k]) is not int or counts[k] <= 0
                   for k in ("rows", "valid", "negative", "positive"))
            or any(not isinstance(counts[k], str) or len(counts[k]) != 64
                   for k in ("rows_sha256", "labels_sha256"))
            or counts["valid"] != counts["negative"] + counts["positive"]):
        raise ValueError("P6 train stop-label counts are incomplete")
    formula = global_binary_class_weights(counts["negative"], counts["positive"])
    if expected_weights is not None and expected_weights != tuple(formula):
        raise ValueError("Treatment stop class weights do not equal pinned global train formula")
    if not isinstance(experiment["source"], dict) or not experiment["source"]:
        raise ValueError("P6 source/count provenance is required")
    return expected_weights


def _validate_experimental_runtime_args(args):
    """Backward-compatible P6 unit-test/driver validation entrypoint."""
    return _validate_experimental_runtime(args, {"name": "p6_global_stop_class_balance"})


def _validate_experimental_runtime(args, experiment):
    if experiment.get("name") == "p8_wide_history":
        stage, arm = experiment["stage"], experiment["arm"]
        steps = 2000 if stage == "pretrain" else 6000
        expected = {"phase": stage if stage == "pretrain" else "joint",
                    "goal_on": 0 if stage == "pretrain" else 1,
                    "state_on": 0 if stage == "pretrain" else 1,
                    "steps": steps, "batch": 16, "microbatch": 2, "eval_batch": 4,
                    "eval_every": 250 if stage == "pretrain" else 6000,
                    "save_every": 250 if stage == "pretrain" else 6000,
                    "lr": 1e-4, "backbone_lr": 1e-5, "weight_decay": .01,
                    "warmup": 200, "grad_clip": 5., "alpha_occ": .2,
                    "alpha_lane": .2, "alpha_motion": .2, "uncertainty": 1,
                    "precision": "bf16", "time_input": "nominal", "bn_policy": "fixed",
                    "arch": "resnet50", "motion_input_mode": "low_feature",
                    "cross_cell_goal_mode": "disabled" if stage == "pretrain" else "zero",
                    "history_contract": arm, "train_stride": 1, "eval_stride": 5,
                    "max_train_samples": 0, "max_eval_samples": 0, "eval_split": "tune"}
        if any(getattr(args, key) != value for key, value in expected.items()):
            raise ValueError("P8 fixed stage recipe mismatch")
        if (not args.init or args.resume or args.pretrained or args.eval_only
                or args.train_scenes or args.eval_scenes
                or not args.history_overlay_root
                or args.expected_history_overlay_sha256 != experiment["history_overlay_manifest_sha256"]):
            raise ValueError("P8 requires pinned overlay, weights-only init, full train+tune, fresh optimizer")
        return
    if experiment.get("name") == "p7_cross_cell_goal_routing":
        mode = {"control_zero_slot": "zero", "goal_real_slot": "real"}[experiment["arm"]]
        expected = {"phase": "joint", "goal_on": 1, "state_on": 1, "steps": 6000,
                    "batch": 16, "microbatch": 2, "eval_batch": 4, "eval_every": 6000,
                    "save_every": 6000, "lr": 1e-4, "backbone_lr": 1e-5,
                    "weight_decay": .01, "warmup": 200, "grad_clip": 5.,
                    "alpha_occ": .2, "alpha_lane": .2, "alpha_motion": .2,
                    "uncertainty": 1, "precision": "bf16", "time_input": "nominal",
                    "bn_policy": "fixed", "arch": "resnet50", "motion_input_mode": "low_feature",
                    "cross_cell_goal_mode": mode, "train_stride": 1, "eval_stride": 5,
                    "max_train_samples": 0, "max_eval_samples": 0, "eval_split": "tune"}
        if any(getattr(args, key) != value for key, value in expected.items()):
            raise ValueError("P7 fixed joint recipe mismatch")
        if not args.init or args.resume or args.pretrained or args.eval_only or args.train_scenes or args.eval_scenes:
            raise ValueError("P7 requires full train+tune, weights-only P0 init, and fresh optimizer")
        return
    expected = {"phase": "joint", "goal_on": 1, "state_on": 1, "steps": 1000,
                "batch": 16, "microbatch": 2, "eval_batch": 4, "eval_every": 1000,
                "save_every": 1000, "lr": 5e-5, "backbone_lr": 5e-6,
                "weight_decay": .01, "warmup": 100, "grad_clip": 5.,
                "alpha_occ": .2, "alpha_lane": .2, "alpha_motion": .2,
                "uncertainty": 1, "precision": "bf16", "time_input": "nominal",
                "bn_policy": "fixed", "arch": "resnet50", "motion_input_mode": "low_feature",
                "train_stride": 1, "eval_stride": 5, "max_train_samples": 0,
                "max_eval_samples": 0, "eval_split": "tune"}
    if any(getattr(args, key) != value for key, value in expected.items()):
        raise ValueError("P6 fixed continuation recipe mismatch")
    if not args.init or args.resume or args.pretrained or args.eval_only or args.train_scenes or args.eval_scenes:
        raise ValueError("P6 requires full train+tune, weights-only init, and fresh optimizer")


def _training_schedule_actions(step, args, experiment):
    """Return evaluation/periodic-save decisions without changing default policy."""
    periodic_protocol = (experiment is None or
                         (experiment.get("name") == "p8_wide_history"
                          and experiment.get("stage") == "pretrain"))
    evaluate_now = step == args.steps or (periodic_protocol and step % args.eval_every == 0)
    periodic_save = periodic_protocol and step % args.save_every == 0
    return evaluate_now, periodic_save


def _load_initial_model_state(model, common, experiment=None):
    """Load a weights-only initializer, narrowly permitting P7's new keys."""
    if experiment is not None and (experiment.get("name") == "p7_cross_cell_goal_routing"
                                   or (experiment.get("name") == "p8_wide_history"
                                       and experiment.get("stage") == "joint")):
        incompatible = model.load_state_dict(common["model"], strict=False)
        expected_missing = experiment["expected_missing_state_keys"]
        if list(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise ValueError("P7 P0 load must miss exactly the new residual state")
        expected_p0 = experiment["expected_p0_model_state_sha256"]
        if tensor_state_sha256(common["model"]) != expected_p0:
            raise ValueError("P7/P8 P0 model tensor SHA mismatch")
        branch = model.scene_encoder.cross_cell_goal_residual
        if branch.output.bias is not None:
            raise ValueError("P7 output projection must be bias-free")
        if bool(branch.output.weight.count_nonzero()):
            raise ValueError("P7 initial output projection must be exactly zero")
        load_name = ("p7_old_checkpoint_load" if experiment.get("name") == "p7_cross_cell_goal_routing"
                     else "p8_p0_checkpoint_load")
        return {load_name: {
            "strict_existing_keys": True, "missing_keys": expected_missing,
            "unexpected_keys": [], "weights_only": True}}
    model.load_state_dict(common["model"], strict=True)
    if (experiment is not None and experiment.get("name") == "p8_wide_history"
            and tensor_state_sha256(common["model"]) != experiment[
                "expected_p0_model_state_sha256"]):
        raise ValueError("P8 P0 public-I0 model tensor SHA mismatch")
    return {}


def main():
    return run_training()


def run_training(argv=None, *, experiment=None):
    global ACTIVE_RUN_DIR
    args = arguments(argv)
    command = list(sys.argv[1:] if argv is None else argv)
    explicit = {token.split("=", 1)[0] for token in command if token.startswith("--")}
    stop_class_weights = _validate_experimental_protocol(experiment)
    if experiment is not None:
        _validate_experimental_runtime(args, experiment)
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
    if args.microbatch < 0 or args.microbatch > args.batch:
        raise ValueError("microbatch must be zero or no larger than logical batch")
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
        if (experiment is not None and experiment.get("name") == "p8_wide_history"
                and load_report["common_checkpoint_sha256"]
                != experiment["expected_initial_checkpoint_sha256"]):
            raise ValueError("P8 initializer checkpoint SHA mismatch")
    if common is not None and (args.resume or args.eval_only):
        config = MotionDriveV2Config(**restore_run_configuration(args, common["manifest"], explicit))
    elif common is not None:
        config = MotionDriveV2Config(**initialization_configuration(
            common["manifest"], goal_on=args.goal_on, state_on=args.state_on,
            explicit_arch=args.arch if "--arch" in explicit else None,
            cross_cell_goal_mode=args.cross_cell_goal_mode
            if "--cross-cell-goal-mode" in explicit else None,
            history_contract=args.history_contract
            if "--history-contract" in explicit else None,
            allow_legacy_history_override=(experiment is not None
                and experiment.get("name") == "p8_wide_history"
                and experiment.get("stage") == "pretrain")))
        args.arch = config.backbone_arch
    else:
        config = MotionDriveV2Config(backbone_arch=args.arch, goal_on=bool(args.goal_on),
                                    state_on=bool(args.state_on),
                                    cross_cell_goal_mode=args.cross_cell_goal_mode or "disabled",
                                    history_contract=args.history_contract or "control",
                                    history_frame_offsets=(2, 5, 10, 20)
                                    if args.history_contract == "wide" else (1, 2, 5, 10),
                                    nominal_history_seconds=(.2, .5, 1., 2.)
                                    if args.history_contract == "wide" else (.1, .2, .5, 1.))
    if args.phase == "pretrain":
        config.goal_on = False
        config.state_on = False
    if args.motion_input_mode is not None:
        previous_mode = config.motion_input_mode
        config.motion_input_mode = args.motion_input_mode
        if previous_mode != config.motion_input_mode:
            load_report["motion_input_mode_override"] = {"from": previous_mode, "to": config.motion_input_mode}
    if args.microbatch < 0 or args.microbatch > args.batch:
        raise ValueError("Restored microbatch must be zero or no larger than logical batch")
    if args.microbatch and args.bn_policy != "fixed":
        raise ValueError("Microbatch accumulation requires fixed BN running statistics")
    # Resume owns this execution policy too; apply it only AFTER restoration,
    # still before any model/optimizer tensors move to CUDA.
    memory_policy = configure_cuda_memory(device, args.cuda_memory_limit_mib, args.cuda_min_free_mib)
    model = MotionDriveV2(config)
    if args.pretrained:
        load_report["backbone"] = model.load_pretrained_backbone(args.pretrained)
        if (load_report["backbone"]["nonhead_missing"] or load_report["backbone"]["unexpected"]):
            raise ValueError(f"Incomplete public backbone load: {load_report['backbone']}")
    if common is not None:
        load_report.update(_load_initial_model_state(model, common, experiment))
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
    if experiment is not None:
        if initial_state_sha != experiment["expected_initial_model_state_sha256"]:
            raise ValueError("Experimental initial model state SHA mismatch")
        actual_groups = [{"name": "backbone", "base_lr": optimizer.param_groups[0]["base_lr"]},
                         {"name": "head", "base_lr": optimizer.param_groups[1]["base_lr"]}]
        if actual_groups != experiment["expected_optimizer_groups"]:
            raise ValueError("Experimental optimizer group LR mismatch")
        if args.resume:
            raise ValueError("Experimental run uses weights-only --init and a fresh optimizer")
        if optimizer.state or not all(param.requires_grad for param in model.parameters()):
            raise ValueError("Experimental run must begin at optimizer step0 with every model parameter trainable")
    weights = LossWeights(plan=0. if args.phase == "pretrain" else 1.,
                          occupancy=args.alpha_occ, lane=args.alpha_lane,
                          motion=args.alpha_motion, uncertainty=bool(args.uncertainty))
    manifest = {
        "schema_version": 1, "git_sha": source_sha(), "arguments": vars(args),
        "model_config": dataclasses.asdict(config), "loss_weights": dataclasses.asdict(weights),
        "time_input": args.time_input, "time_input_policy": time_input_policy(
            args.time_input, config.nominal_history_seconds),
        "split_sha256": sha256(args.split_manifest), "load_report": load_report,
        "torch": torch.__version__, "numpy": np.__version__,
        "device": str(device), "metric": "mean of cumulative ADE@1/2/3s; no sample proxy",
        "cuda_memory_policy": memory_policy,
        "microbatch_policy": {"enabled": bool(args.microbatch), "microbatch": args.microbatch,
                              "logical_batch": args.batch, "optimizer_updates_per_logical_batch": 1,
                              "loss_normalization": "full logical batch label/mask denominators",
                              "bitwise_equivalence_to_unsplit_training_promised": False},
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
    if args.history_overlay_root:
        manifest["history_overlay_manifest_sha256"] = sha256(
            Path(args.history_overlay_root) / "overlay_manifest.json")
    if experiment is not None:
        manifest["experimental_protocol"] = experiment
    atomic_json(run_dir / "manifest.json", manifest)
    ACTIVE_RUN_DIR = run_dir
    def dataset(split, stride, maximum, scenes):
        kwargs = dict(data_root=args.data_root, split_manifest=args.split_manifest,
                      split=split, supervision_root=args.supervision_root,
                      min_frame=30, frame_stride=stride, max_samples=maximum,
                      augment=(split == "train" and not args.eval_only), seed=args.seed,
                      history_contract=config.history_contract,
                      history_overlay_root=args.history_overlay_root,
                      expected_history_overlay_sha256=args.expected_history_overlay_sha256)
        if scenes:
            kwargs["scenes"] = scenes
        return MotionDriveDataset(**kwargs)
    evaluation = dataset(args.eval_split, args.eval_stride, args.max_eval_samples, args.eval_scenes)
    eval_loader = DataLoader(evaluation, batch_size=args.eval_batch, shuffle=False,
                             num_workers=args.workers, pin_memory=device.type == "cuda")
    if args.eval_only:
        report, records = evaluate(model, eval_loader, device, args.precision, time_input=args.time_input,
                                   min_free_mib=args.cuda_min_free_mib,
                                   nominal_history_seconds=config.nominal_history_seconds)
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
    if experiment is not None:
        declared = (experiment["train_label_counts"] if experiment.get("name") == "p6_global_stop_class_balance"
                    else experiment["train_data"])
        if len(training) != declared["rows"] or manifest["train_rows_sha256"] != declared["rows_sha256"]:
            raise ValueError("Experimental train rows differ from the pinned protocol")
        if experiment.get("name") in ("p7_cross_cell_goal_routing", "p8_wide_history"):
            declared_tune = experiment["tune_data"]
            if (len(evaluation) != declared_tune["rows"]
                    or manifest["eval_rows_sha256"] != declared_tune["rows_sha256"]):
                raise ValueError("P7 tune rows differ from the pinned protocol")
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
    periodic_protocol = (experiment is None or
                         (experiment.get("name") == "p8_wide_history"
                          and experiment.get("stage") == "pretrain"))
    if not args.resume and periodic_protocol:
        atomic_checkpoint(run_dir / "initial.pth", checkpoint())
    started = time.monotonic()
    sample_order_digest = hashlib.sha256()
    manifest["status"] = "running"
    atomic_json(run_dir / "manifest.json", manifest)
    with (run_dir / "metrics.jsonl").open("a", buffering=1) as log:
        if not args.resume and periodic_protocol:
            initial_report, _ = evaluate(model, eval_loader, device, args.precision, time_input=args.time_input,
                                          min_free_mib=args.cuda_min_free_mib,
                                          nominal_history_seconds=config.nominal_history_seconds)
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
                warm = min(1., (step + 1) / max(1, args.warmup))
                progress = max(0., (step - args.warmup) / max(1, args.steps - args.warmup))
                factor = warm * .5 * (1. + math.cos(math.pi * min(1., progress)))
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * factor
                optimizer.zero_grad(set_to_none=True)
                if args.microbatch:
                    # Normalizers are computed once from the WHOLE logical batch.
                    # Taking a mean of microbatch means would change raster/mask loss.
                    from motiondrive_v2_training import build_loss_normalizers
                    check_cuda_headroom(device, args.cuda_min_free_mib)
                    normalizers = to_device(build_loss_normalizers(raw), device)
                    parts = {}
                    for start in range(0, len(raw["images"]), args.microbatch):
                        check_cuda_headroom(device, args.cuda_min_free_mib)
                        batch = to_device(slice_batch(raw, start, start + args.microbatch), device)
                        with autocast(device, args.precision):
                            output = model(**model_inputs(
                                batch, time_input=args.time_input,
                                nominal_history_seconds=config.nominal_history_seconds))
                        loss, micro_parts = compute_loss(output, batch, weights, normalizers=normalizers,
                                                         stop_class_weights=stop_class_weights)
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(f"Nonfinite microbatch loss at step {step}; no silent NaN skip")
                        loss.backward()
                        for name, value in micro_parts.items():
                            parts[name] = parts.get(name, 0.) + value.detach()
                        del batch, output, loss, micro_parts
                else:
                    check_cuda_headroom(device, args.cuda_min_free_mib)
                    batch = to_device(raw, device)
                    with autocast(device, args.precision):
                        output = model(**model_inputs(
                            batch, time_input=args.time_input,
                            nominal_history_seconds=config.nominal_history_seconds))
                    loss, parts = compute_loss(output, batch, weights,
                                               stop_class_weights=stop_class_weights)
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
                           "cuda_memory": cuda_memory_snapshot(device),
                           **{k: float(v.detach()) for k, v in parts.items()}}
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    print(json.dumps(row, allow_nan=False), flush=True)
                evaluate_now, periodic_save = _training_schedule_actions(step, args, experiment)
                if evaluate_now:
                    report, eval_records = evaluate(model, eval_loader, device, args.precision,
                                                    time_input=args.time_input,
                                                    min_free_mib=args.cuda_min_free_mib,
                                                    detailed_records=experiment is not None and not (
                                                        experiment.get("name") == "p8_wide_history"
                                                        and experiment.get("stage") == "pretrain"),
                                                    nominal_history_seconds=config.nominal_history_seconds)
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
                    terminal_experiment = (experiment is not None and not (
                        experiment.get("name") == "p8_wide_history"
                        and experiment.get("stage") == "pretrain"))
                    if terminal_experiment:
                        best = score
                        atomic_json(run_dir / "final_eval.json", {"report": row, "records": eval_records})
                    elif score < best:
                        best = score
                        atomic_checkpoint(run_dir / "best.pth", checkpoint())
                if periodic_save:
                    atomic_checkpoint(run_dir / "last.pth", checkpoint())
            epoch += 1
        atomic_checkpoint(run_dir / "last.pth", checkpoint())
    manifest.update(status="stopped" if requested_stop[0] else "completed", step=step,
                    best_metric=best if math.isfinite(best) else None,
                    cuda_memory=cuda_memory_snapshot(device),
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
