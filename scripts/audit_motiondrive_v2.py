#!/usr/bin/env python3
"""Executable V2 information-flow audit; recomputes the actual image model.

This tests the goal-conditioned direct planner, not the legacy row-selector.
Passing these checks is an implementation diagnostic, not organizer approval.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INPUT_KEYS = ("images", "history_images", "lidar2img", "history_transforms",
              "time_offsets", "goal_xy")
OUTPUT_KEYS = ("plan_abs", "scene_features", "motion_features", "history_hat",
               "state_hat", "occ_logits", "lane_logits")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def synthetic_batch(batch_size=1, image_hw=(432, 768), history_hw=(216, 384), seed=7):
    """Camera-valid synthetic inputs. Never label their timing as real-batch timing."""
    gen = torch.Generator().manual_seed(seed)
    h, w = image_hw
    projection = torch.tensor([[w / 2, -0.7 * w, 0, 0],
                               [h / 2, 0, -0.7 * w, 1.5 * 0.7 * w],
                               [1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.float32)
    cameras = []
    for yaw in (0., -0.7, 0.7, math.pi, -2.2, 2.2):
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = torch.tensor([[c, s, 0, 0], [-s, c, 0, 0],
                                 [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
        cameras.append(projection @ rotation)
    dt = torch.tensor([0.1, 0.2, 0.5, 1.0]).repeat(batch_size, 1)
    transforms = torch.eye(4).repeat(batch_size, 4, 1, 1)
    transforms[:, :, 0, 3] = 8.0 * dt
    return {"images": torch.randn(batch_size, 6, 3, h, w, generator=gen),
            "history_images": torch.randn(batch_size, 4, 3, *history_hw, generator=gen),
            "lidar2img": torch.stack(cameras).repeat(batch_size, 1, 1, 1),
            "history_transforms": transforms, "time_offsets": dt,
            "goal_xy": torch.tensor([20., 2.]).repeat(batch_size, 1)}


def validate_batch(batch):
    missing = set(INPUT_KEYS) - set(batch)
    if missing:
        raise ValueError(f"missing model inputs: {sorted(missing)}")
    images, history = batch["images"], batch["history_images"]
    if images.ndim != 5 or images.shape[1:3] != (6, 3):
        raise ValueError("images must have shape [B,6,3,H,W]")
    b = images.shape[0]
    if history.ndim != 5 or history.shape[:3] != (b, 4, 3):
        raise ValueError("history_images must have shape [B,4,3,H,W]")
    for key, shape in {"lidar2img": (b, 6, 4, 4), "history_transforms": (b, 4, 4, 4),
                       "time_offsets": (b, 4), "goal_xy": (b, 2)}.items():
        if tuple(batch[key].shape) != shape:
            raise ValueError(f"{key} must have shape {shape}")
    for key in INPUT_KEYS:
        if not batch[key].is_floating_point() or not torch.isfinite(batch[key]).all():
            raise ValueError(f"{key} must be finite floating point")
    if not torch.all(batch["time_offsets"] > 0):
        raise ValueError("time_offsets are positive past-time distances in seconds")


def load_batch(path=None, *, device="cpu", seed=7):
    if path is None:
        batch = synthetic_batch(seed=seed)
        source = {"kind": "synthetic", "seed": seed,
                  "notice": "Synthetic image timing only; not real-data accuracy or final latency gate."}
    else:
        path = Path(path)
        identity = {}
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as saved:
                batch = {k: torch.from_numpy(saved[k].copy()) for k in INPUT_KEYS}
                if "gt_plan_abs" in saved:
                    batch["gt_plan_abs"] = torch.from_numpy(saved["gt_plan_abs"].copy())
                elif "gt_plan" in saved:
                    batch["gt_plan_abs"] = torch.from_numpy(saved["gt_plan"].copy())
        else:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            batch = dict(saved.get("batch", saved))
            for key in ("scenario", "session_id", "frame", "row", "scen_idx"):
                if key in batch:
                    identity[key] = batch[key].tolist() if torch.is_tensor(batch[key]) else batch[key]
            if "gt_plan_abs" not in batch and "gt_plan" in batch:
                batch["gt_plan_abs"] = batch["gt_plan"]
            batch = {k: torch.as_tensor(batch[k]) for k in (*INPUT_KEYS, "gt_plan_abs") if k in batch}
        source = {"kind": "file", "path": str(path.resolve()), "sha256": sha256(path),
                  "sample_identity": identity,
                  "notice": "File provenance must establish real imagery; loading a file is not proof."}
    batch = {k: v.to(device=device, dtype=torch.float32) for k, v in batch.items()}
    validate_batch(batch)
    return batch, source


def model_inputs(batch):
    return {k: batch[k] for k in INPUT_KEYS}


def construct_model(args):
    """Restore a trusted user-owned LOCAL training checkpoint and its run config.

    Trainer payloads contain NumPy/Python RNG states, so weights_only=False is
    intentional here. Never point this option at an untrusted downloaded pickle.
    Input-batch loading remains weights_only=True.
    """
    from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
    checkpoint = None
    config = {}
    info = {"checkpoint": None, "config_source": "model_defaults",
            "checkpoint_load_policy": "not_applicable", "explicit_overrides": {},
            "explicit_config_sources": [], "evaluation_kind": "random_initialization"}
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise ValueError("--checkpoint must be an existing trusted local file")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must contain a state dictionary and model configuration")
        manifest = checkpoint.get("manifest", {})
        if isinstance(manifest, dict) and "model_config" in manifest:
            saved_config, source = manifest["model_config"], "manifest.model_config"
        elif "model_config" in checkpoint:
            saved_config, source = checkpoint["model_config"], "model_config"
        elif "config" in checkpoint:
            saved_config, source = checkpoint["config"], "config"
        elif args.config_json:
            saved_config, source = {}, "explicit_config_json_required"
        else:
            raise ValueError("checkpoint has no model configuration; provide --config-json explicitly")
        if not isinstance(saved_config, dict):
            raise ValueError(f"checkpoint {source} must be a configuration dictionary")
        config.update(saved_config)
        info.update(checkpoint=str(checkpoint_path.resolve()),
                    checkpoint_sha256=sha256(checkpoint_path), config_source=source,
                    checkpoint_model_config=dict(saved_config),
                    checkpoint_load_policy="trusted_user_local_pickle_weights_only_false",
                    evaluation_kind="checkpoint_configuration")

    def apply_explicit_config(values, source):
        defaults = dataclasses.asdict(MotionDriveV2Config(**config))
        for key, value in values.items():
            old = info["explicit_overrides"].get(key, {}).get("from", defaults.get(key))
            # JSON arrays and dataclass tuples describe the same configuration.
            if json.dumps(old) != json.dumps(value):
                info["explicit_overrides"][key] = {"from": old, "to": value, "source": source}
            else:
                info["explicit_overrides"].pop(key, None)
            config[key] = value
        info["explicit_config_sources"].append(source)

    if args.config_json:
        apply_explicit_config(json.loads(Path(args.config_json).read_text()), "--config-json")
    for key in ("goal_on", "state_on"):
        value = getattr(args, key, None)
        if value is not None:
            apply_explicit_config({key: bool(value)}, "--" + key.replace("_", "-"))
    if checkpoint is not None and any(key not in config for key in ("goal_on", "state_on")):
        raise ValueError("checkpoint goal_on/state_on are unspecified; provide them explicitly via --config-json or --goal-on/--state-on")
    model = MotionDriveV2(MotionDriveV2Config(**config))
    if checkpoint is not None:
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state, strict=True)
        if info["explicit_overrides"]:
            info["evaluation_kind"] = "explicit_checkpoint_ablation"
    info["effective_model_config"] = dataclasses.asdict(model.config)
    model.audit_load_metadata = info
    model.to(args.device).eval()
    return model


def delta(a, b):
    return {"bitwise_equal": bool(torch.equal(a, b)),
            "max_abs": float((a.float() - b.float()).abs().max().detach().cpu())}


def output_checks(output, batch_size):
    checks = {}
    for key in OUTPUT_KEYS:
        checks[f"output_{key}_present"] = key in output
        if key in output:
            checks[f"output_{key}_finite"] = bool(torch.isfinite(output[key]).all())
    for key, shape in {"plan_abs": (batch_size, 6, 2),
                       "history_hat": (batch_size, 4, 4), "state_hat": (batch_size, 6)}.items():
        checks[f"{key}_shape"] = key in output and tuple(output[key].shape) == shape
        checks[f"{key}_fp32"] = key in output and output[key].dtype == torch.float32
    return checks


def final_head_precision(model, batch, device, precision):
    """Inspect inputs inside final projection modules, before any output cast."""
    observed = {}
    handles = []
    modules = dict(model.named_modules())
    for name in ("planner.xy_head", "motion_encoder.history_head", "motion_encoder.state_head",
                 "motion_encoder.history_uncertainty_head", "motion_encoder.state_uncertainty_head"):
        if name not in modules:
            continue
        def capture(module, inputs, name=name):
            observed[name] = [str(t.dtype) for t in inputs if torch.is_tensor(t)]
        handles.append(modules[name].register_forward_pre_hook(capture))
    try:
        with torch.inference_mode(), autocast_context(device, precision):
            model(**model_inputs(batch))
    finally:
        for handle in handles:
            handle.remove()
    return observed


def gradient_evidence(model, batch, device, precision):
    """Differentiate real task outputs w.r.t. the returned shared visual features."""
    model.zero_grad(set_to_none=True)
    captured = {}
    refine = getattr(getattr(model, "scene_encoder", None), "refine", None)
    handle = None
    if refine is not None:
        handle = refine.register_forward_hook(
            lambda module, args, output: captured.update(shared_raster=output))
    with torch.enable_grad(), autocast_context(device, precision):
        try:
            out = model(**model_inputs(batch))
        finally:
            if handle is not None:
                handle.remove()
        findings = {}
        for task, feature in (("plan_abs", "scene_features"),
                              ("plan_abs", "motion_features"),
                              ("occ_logits", "scene_features"),
                              ("lane_logits", "scene_features")):
            value = out[feature]
            target_kind = "returned_feature"
            if feature == "scene_features" and "shared_raster" in captured:
                raster = captured["shared_raster"]
                # The exported token tensor and perception raster may be sibling
                # views: differentiate their actual shared ancestor, not a view
                # that is downstream of neither perception head.
                if raster.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
                    value = raster
                    target_kind = "shared_raster_verified_storage_alias"
            if not value.requires_grad:
                grad = None
            else:
                grad = torch.autograd.grad(out[task].float().square().mean(), value,
                                           retain_graph=True, allow_unused=True)[0]
            norm = None if grad is None else float(grad.float().norm().detach().cpu())
            findings[f"{task}_uses_{feature}"] = {
                "connected": grad is not None,
                "finite": grad is not None and bool(torch.isfinite(grad).all()),
                "gradient_l2": norm,
                "nonzero": norm is not None and math.isfinite(norm) and norm > 0.,
                "gradient_target": target_kind,
            }
    return findings


def run_audit(model, batch, *, precision="bf16", goal_on=True, with_gradients=True):
    device = batch["images"].device
    validate_batch(batch)
    checks, measurements = {}, {}
    planner_params = set(inspect.signature(model.plan_from_features).parameters)
    checks["planner_signature_no_direct_goal_or_supplied_state"] = not bool(
        planner_params & {"goal", "goal_xy", "goal_embedding", "status", "history",
                          "history_transforms", "provided_state", "provided_history"})

    def execute(inputs):
        with torch.inference_mode(), autocast_context(device, precision):
            return model(**model_inputs(inputs))

    normal = execute(batch)
    checks.update(output_checks(normal, batch["images"].shape[0]))
    if "gt_plan_abs" in batch:
        weights = normal["plan_abs"].new_tensor([11, 11, 5, 5, 2, 2]) / 36
        measurements["normal_official_temporal_d3_frame_mean"] = float(
            ((normal["plan_abs"] - batch["gt_plan_abs"]).norm(dim=-1) * weights).sum(-1).mean())
    if "scene_visible" in normal:
        visibility = normal["scene_visible"].float()
        measurements["calibration_scene_visibility"] = {
            "fraction": float(visibility.mean().cpu()),
            "visible_cells": int((visibility > 0).sum().cpu()),
            "total_cells": visibility.numel(),
        }
        checks["calibration_has_visible_scene_cells"] = bool((visibility > 0).any())
    observed_precision = final_head_precision(model, batch, device, precision)
    measurements["final_head_input_dtypes"] = observed_precision
    for name, dtypes in observed_precision.items():
        checks[f"{name}_input_fp32"] = bool(dtypes) and all(d == "torch.float32" for d in dtypes)
    repeated = execute(batch)
    checks["same_input_repeat_bitwise"] = all(torch.equal(normal[k], repeated[k]) for k in OUTPUT_KEYS)
    changed_goal = dict(batch, goal_xy=-batch["goal_xy"] + batch["goal_xy"].new_tensor([7., 11.]))
    counter = execute(changed_goal)
    for key in ("motion_features", "history_hat", "state_hat"):
        checks[f"goal_counterfactual_{key}_bitwise"] = torch.equal(normal[key], counter[key])
    measurements["goal_changes"] = {k: delta(normal[k], counter[k]) for k in OUTPUT_KEYS}
    if not goal_on:
        checks["goal_off_whole_forward_bitwise"] = all(torch.equal(normal[k], counter[k]) for k in OUTPUT_KEYS)

    changed_alignment = dict(batch, history_transforms=batch["history_transforms"].clone())
    changed_alignment["history_transforms"][:, :, 0, 3] += 3.
    aligned = execute(changed_alignment)
    for key in ("motion_features", "history_hat", "state_hat"):
        checks[f"alignment_counterfactual_{key}_bitwise"] = torch.equal(normal[key], aligned[key])

    with torch.inference_mode(), autocast_context(device, precision):
        fixed = {"scene_features": normal["scene_features"], "motion_features": normal["motion_features"],
                 "predicted_state": normal["state_hat"], "predicted_history": normal["history_hat"]}
        fixed_before = model.plan_from_features(**fixed)
        model(**model_inputs(changed_goal))  # Exercises mutable caches/closures, if any.
        fixed_after = model.plan_from_features(**fixed)
    checks["fixed_features_planner_goal_cache_invariant"] = torch.equal(fixed_before, fixed_after)
    checks["planner_replay_matches_forward"] = torch.equal(fixed_before, normal["plan_abs"])

    perturbations = {
        "image_zero": dict(batch, images=torch.zeros_like(batch["images"]),
                           history_images=torch.zeros_like(batch["history_images"])),
        "image_constant": dict(batch, images=torch.full_like(batch["images"], 0.5),
                               history_images=torch.full_like(batch["history_images"], 0.5)),
        "temporal_shuffle": dict(batch, history_images=batch["history_images"].flip(1)),
        "image_mismatch": dict(batch, images=batch["images"].roll(1, 1),
                               history_images=batch["history_images"].flip(-1)),
    }
    measurements["image_perturbations"] = {}
    for name, modified in perturbations.items():
        out = execute(modified)
        changes = {k: delta(normal[k], out[k]) for k in OUTPUT_KEYS}
        if "gt_plan_abs" in batch:
            w = out["plan_abs"].new_tensor([11, 11, 5, 5, 2, 2]) / 36
            changes["official_temporal_d3_frame_mean"] = float(
                ((out["plan_abs"] - batch["gt_plan_abs"]).norm(dim=-1) * w).sum(-1).mean())
        measurements["image_perturbations"][name] = changes
    checks["actual_image_perturbation_changes_scene"] = not measurements["image_perturbations"]["image_zero"]["scene_features"]["bitwise_equal"]
    checks["actual_temporal_perturbation_changes_motion"] = not measurements["image_perturbations"]["temporal_shuffle"]["motion_features"]["bitwise_equal"]
    if with_gradients:
        gradients = gradient_evidence(model, batch, device, precision)
        measurements["feature_gradients"] = gradients
        for name, result in gradients.items():
            checks[name] = result["connected"] and result["finite"] and result["nonzero"]
    else:
        measurements["feature_gradients"] = "NOT RUN"
    return {"checks": checks, "measurements": measurements,
            "all_executed_checks_pass": all(checks.values()),
            "gradient_checks_executed": with_gradients,
            "limitations": ["Architecture/dataflow diagnostic, not organizer approval.",
                            "Image perturbation sensitivity does not establish heldout prediction quality.",
                            "Only the V2 direct planner is audited; legacy candidate-row audit is separate."]}


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", help="Trusted user-owned local training checkpoint; uses Python pickle (weights_only=False) to read saved RNG state/config")
    parser.add_argument("--config-json")
    parser.add_argument("--batch", help="NPZ or weights-only Torch tensor dict, model input keys")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--goal-on", type=int, choices=(0, 1), help="Explicit goal-condition ablation; saved checkpoint value is preserved when omitted")
    parser.add_argument("--state-on", type=int, choices=(0, 1), help="Explicit inferred-state ablation; saved checkpoint value is preserved when omitted")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    return parser


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--skip-gradients", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = construct_model(args)
    batch, source = load_batch(args.batch, device=args.device, seed=args.seed)
    cfg = getattr(model, "config", getattr(model, "cfg", None))
    goal_on = getattr(cfg, "goal_on", bool(args.goal_on) if args.goal_on is not None else True)
    report = run_audit(model, batch, precision=args.precision, goal_on=goal_on,
                       with_gradients=not args.skip_gradients)
    report.update({"input_source": source, "checkpoint": args.checkpoint,
                   "model_load": model.audit_load_metadata,
                   "precision": args.precision, "torch_version": torch.__version__})
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps({"pass": report["all_executed_checks_pass"],
                      "failed": [k for k, v in report["checks"].items() if not v],
                      "model_load": model.audit_load_metadata,
                      "input_source": source, "report": str(target)}, indent=2))
    return 0 if report["all_executed_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
