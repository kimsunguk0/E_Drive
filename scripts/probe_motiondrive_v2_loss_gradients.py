#!/usr/bin/env python3
"""Small labeled-batch gradient diagnosis; no optimizer or checkpoint changes.

Uses a trusted local training checkpoint (Python pickle) with its COMPLETE saved
model configuration and loss weights, without overrides. A fixture is a CPU-only
torch.save({"batch": collated_dataset_dict, "metadata": json_metadata}) file.
Each small batch gets ONE eval-mode bf16 forward and separate autograd.grad
queries. No training-BN pass or synthetic/GT substitution is performed.

Example: python scripts/probe_motiondrive_v2_loss_gradients.py \
  --checkpoint run/best.pth --fixture train8.pt --output reports/new.json \
  --device cuda:0 --batch-size 4 --max-samples 8
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from motiondrive_v2_training import (LossWeights, MODEL_INPUTS, compute_loss,
                                     model_inputs, tensor_state_sha256, to_device)

TERMS = {"plan": "plan_d3", "occ": "occ_bce", "lane": "lane_bce",
         "history": "history", "state": "state", "stop": "stop_bce",
         "motion": "motion", "total": "total"}
REQUIRED_LABELS = ("gt_plan", "plan_valid", "occ_target", "occ_valid", "lane_target", "lane_valid",
                   "history_target", "history_valid", "state_target", "state_valid")
HEAD_PATHS = ("planner.xy_head", "motion_encoder.history_head", "motion_encoder.state_head",
              "motion_encoder.history_uncertainty_head", "motion_encoder.state_uncertainty_head")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def module_group(name):
    if name.startswith("backbone_fpn."):
        return "fpn" if name.startswith(("backbone_fpn.lat", "backbone_fpn.out")) else "backbone"
    if name.startswith("planner."):
        child = name.split(".")[1]
        return "planner." + (child if child in ("xy_head", "decoder", "state_projection", "scene_position") else "other")
    for prefix, heads in (("motion_encoder", ("history_head", "state_head", "history_uncertainty_head",
                                                "state_uncertainty_head")),
                           ("scene_encoder", ("occ_head", "lane_head"))):
        if name.startswith(prefix + "."):
            child = name.split(".")[1]
            return prefix + "." + (child if child in heads else "shared")
    return name.split(".")[0]


def fingerprints(model):
    parameters = dict(model.named_parameters())
    gradients = {name: p.grad for name, p in parameters.items() if p.grad is not None}
    return {"parameters_sha256": tensor_state_sha256(parameters),
            "buffers_sha256": tensor_state_sha256(dict(model.named_buffers())),
            "parameter_grads_sha256": tensor_state_sha256(gradients),
            "parameters_with_grad": sorted(gradients)}


def gradient_summary(named_parameters, gradients, coefficient):
    result = {}
    for name, parameter in named_parameters:
        group = module_group(name)
        result.setdefault(group, {"squared_norm": 0., "connected_tensors": 0,
                                  "connected_numel": 0, "group_parameter_numel": 0})
        result[group]["group_parameter_numel"] += parameter.numel()
    for (name, _), gradient in zip(named_parameters, gradients):
        if gradient is not None:
            item = result[module_group(name)]
            item["squared_norm"] += float(gradient.double().square().sum())
            item["connected_tensors"] += 1
            item["connected_numel"] += gradient.numel()
    result["all_parameters"] = {key: sum(item[key] for item in result.values())
                                 for key in ("squared_norm", "connected_tensors", "connected_numel",
                                             "group_parameter_numel")}
    for item in result.values():
        item["raw_l2_norm"] = math.sqrt(item.pop("squared_norm"))
        if not math.isfinite(item["raw_l2_norm"]):
            raise FloatingPointError("Nonfinite loss-component gradient")
        item["weighted_contribution_l2_norm"] = abs(coefficient) * item["raw_l2_norm"]
    return result


def shared_gradient_statistics(named_parameters, gradients_a, gradients_b, coefficient_a=1., coefficient_b=1.):
    """Restrict to parameter tensors connected to BOTH losses; never concatenate."""
    accumulators = {}
    for (name, _), a, b in zip(named_parameters, gradients_a, gradients_b):
        if a is None or b is None:
            continue
        item = accumulators.setdefault(module_group(name), dict(aa=0., bb=0., dot=0., numel=0, tensors=0))
        a, b = a.double(), b.double()
        item["aa"] += float(a.square().sum())
        item["bb"] += float(b.square().sum())
        item["dot"] += float((a * b).sum())
        item["numel"] += a.numel()
        item["tensors"] += 1
    accumulators["all_shared_parameters"] = {key: sum(item[key] for item in accumulators.values())
                                               for key in ("aa", "bb", "dot", "numel", "tensors")}
    result = {}
    for group, item in accumulators.items():
        a, b = math.sqrt(item["aa"]), math.sqrt(item["bb"])
        cosine = item["dot"] / (a * b) if a > 0 and b > 0 else None
        wa, wb = abs(coefficient_a) * a, abs(coefficient_b) * b
        result[group] = {"connected_parameter_tensors": item["tensors"],
                         "connected_parameter_numel": item["numel"],
                         "raw_a_norm": a, "raw_b_norm": b, "raw_dot": item["dot"], "raw_cosine": cosine,
                         "weighted_a_norm": wa, "weighted_b_norm": wb,
                         "weighted_dot": coefficient_a * coefficient_b * item["dot"],
                         "weighted_cosine": cosine if wa > 0 and wb > 0 else None}
    return result


def logvar_statistics(value, valid=None):
    value = value.detach().float().cpu()
    if valid is None:
        mask = torch.ones_like(value, dtype=torch.bool)
    else:
        mask = valid.detach().bool().cpu()
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(value)
    def summarize(values):
        values = values.flatten()
        if not values.numel():
            return {"n_valid": 0}
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite valid log variance")
        return {"n_valid": values.numel(), "min": float(values.min()), "mean": float(values.mean()),
                "max": float(values.max()), "p05_p50_p95": torch.quantile(values, values.new_tensor([.05, .5, .95])).tolist(),
                "loss_clamp_below_minus6_fraction": float((values < -6).float().mean()),
                "loss_clamp_above_plus6_fraction": float((values > 6).float().mean()),
                "at_or_beyond_loss_lower_fraction": float((values <= -6).float().mean()),
                "at_or_beyond_loss_upper_fraction": float((values >= 6).float().mean()),
                "at_or_beyond_model_minus8_fraction": float((values <= -8).float().mean()),
                "at_or_beyond_model_plus8_fraction": float((values >= 8).float().mean()),
                "loss_precision_min_max": [float((-values.clamp(-6, 6)).exp().min()),
                                             float((-values.clamp(-6, 6)).exp().max())]}
    return {"all_valid": summarize(value[mask]),
            "by_component": [summarize(value[..., i][mask[..., i]]) for i in range(value.shape[-1])],
            "definition": "Model output logvar before loss clamp; normalized physical units. Strictly outside +/-6 has zero clamp derivative; equality is reported separately."}


def loss_coefficients(weights):
    return {"plan": weights.plan, "occ": weights.occupancy, "lane": weights.lane,
            "history": weights.motion, "state": weights.motion, "stop": .2 * weights.motion,
            "motion": weights.motion, "total": 1.}


def probe_batch(model, raw_batch, weights, device):
    """One forward; no optimizer, backward(), .grad clearing, or BN adaptation."""
    device = torch.device(device)
    previous_modes = [(module, module.training) for module in model.modules()]
    before = fingerprints(model)
    handles, head_inputs = [], {}
    model.eval()
    try:
        modules = dict(model.named_modules())
        for name in HEAD_PATHS:
            if name in modules:
                handles.append(modules[name].register_forward_pre_hook(
                    lambda _module, args, name=name: head_inputs.setdefault(name, []).append(str(args[0].dtype))))
        batch = to_device(raw_batch, device)
        with torch.enable_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(**model_inputs(batch))
            if weights.uncertainty and any(key not in outputs for key in ("history_logvar", "state_logvar")):
                raise ValueError("Checkpoint enables uncertainty; missing logvar would silently change the loss")
            for key in ("plan_abs", "occ_logits", "lane_logits", "history_hat", "state_hat",
                        "history_logvar", "state_logvar"):
                if key in outputs and outputs[key].dtype != torch.float32:
                    raise ValueError(f"Expected existing FP32 output {key}; no precision override is permitted")
            if any(dtype != "torch.float32" for dtypes in head_inputs.values() for dtype in dtypes):
                raise ValueError("Coordinate/state/uncertainty head did not receive FP32 input")
            with torch.autocast(device_type=device.type, enabled=False):
                _, parts = compute_loss(outputs, batch, weights)
            if any(value.dtype != torch.float32 or not bool(torch.isfinite(value)) for value in parts.values()):
                raise FloatingPointError("Expected finite FP32 losses")
            named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            if not named:
                raise ValueError("No trainable parameters to probe")
            parameters = [p for _, p in named]
            gradients = {}
            for index, (term, part) in enumerate(TERMS.items()):
                value = parts[part]
                gradient = (torch.autograd.grad(value, parameters, retain_graph=index < len(TERMS) - 1,
                                                allow_unused=True) if value.requires_grad else (None,) * len(parameters))
                gradients[term] = tuple(g.detach().to(device="cpu", dtype=torch.float32) if g is not None else None
                                        for g in gradient)
                del gradient
        coefficients = loss_coefficients(weights)
        pairs = [("plan", term) for term in TERMS if term != "plan"] + [("history", "state")]
        report = {"batch_size": len(raw_batch["images"]), "forward_count": 1,
                  "model_mode": "eval; all BN running statistics fixed",
                  "bn_training_modules_during_probe": sum(m.training for m in model.modules()
                      if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)),
                  "backbone_precision": "bf16 autocast", "loss_precision": "float32",
                  "perception_head_precision": "Existing graph preserved: raster conv autocast, FP32 returned logits; no head precision override",
                  "coordinate_state_head_input_dtypes": head_inputs,
                  "loss_values": {term: float(parts[part].detach()) for term, part in TERMS.items()},
                  "coefficients_in_total": coefficients,
                  "gradient_norms": {term: gradient_summary(named, gs, coefficients[term]) for term, gs in gradients.items()},
                  "shared_gradient_pairs": {f"{a}__vs__{b}": shared_gradient_statistics(named, gradients[a], gradients[b],
                        coefficients[a], coefficients[b]) for a, b in pairs},
                  "logvar": {},
                  "interpretation_limit": "One small eval-mode batch, not training-BN gradients or a generalization/causal result. Norms are not additive; Adam updates cannot be inferred by multiplying LR by clip5/norm. Motion duplicates history+state+0.2*stop and must not be added again to those terms. Separately backpropagated bf16 component gradients may differ slightly from a summed-loss backward due to rounding."}
        if "history_logvar" in outputs:
            report["logvar"]["history"] = logvar_statistics(outputs["history_logvar"], batch.get("history_valid"))
        if "state_logvar" in outputs:
            valid = batch.get("state_valid")
            report["logvar"]["state"] = logvar_statistics(outputs["state_logvar"], valid[..., :5] if valid is not None else None)
    finally:
        for handle in handles:
            handle.remove()
        for module, mode in previous_modes:
            module.training = mode
        after = fingerprints(model)
        if before != after:
            raise RuntimeError("Probe modified weights, buffers, or existing parameter .grad values")
    report["immutability"] = {"before": before, "after": after, "bitwise_unchanged": True}
    return report


def load_checkpoint(path, device, model_factory=None):
    from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("manifest"), dict):
        raise ValueError("Trusted checkpoint must contain model and manifest")
    manifest = checkpoint["manifest"]
    saved = manifest.get("model_config")
    required = {field.name for field in dataclasses.fields(MotionDriveV2Config)}
    if not isinstance(saved, dict) or set(saved) != required:
        raise ValueError("A complete, exact saved model_config is required; no defaults or overrides")
    loss_saved = manifest.get("loss_weights")
    if not isinstance(loss_saved, dict) or set(loss_saved) != {f.name for f in dataclasses.fields(LossWeights)}:
        raise ValueError("Complete checkpoint loss_weights are required")
    weights = LossWeights(**loss_saved)
    if any(not math.isfinite(x) or x < 0 for x in (weights.plan, weights.occupancy, weights.lane, weights.motion)):
        raise ValueError("Loss coefficients must be finite and nonnegative")
    config = MotionDriveV2Config(**saved)
    model = (model_factory or MotionDriveV2)(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, weights, {"checkpoint_step": checkpoint.get("step"), "manifest": manifest,
                           "effective_model_config": dataclasses.asdict(config), "load_state_dict_strict": True,
                           "actual_parameter_count": sum(p.numel() for p in model.parameters()),
                           "actual_trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                           "configuration_overrides": {}, "checkpoint_trust": "trusted local pickle weights_only=False"}


def load_fixture(path):
    fixture = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(fixture, dict):
        raise ValueError("Fixture must be a dictionary")
    batch = fixture.get("batch", fixture)
    metadata = fixture.get("metadata", {}) if "batch" in fixture else {}
    if not isinstance(batch, dict):
        raise ValueError("Fixture batch must be a dictionary")
    json.dumps(metadata, allow_nan=False)
    for key in (*MODEL_INPUTS, *REQUIRED_LABELS):
        if key not in batch or not isinstance(batch[key], torch.Tensor):
            raise ValueError(f"Missing tensor fixture field: {key}")
    size = len(batch["images"])
    if size < 1:
        raise ValueError("Fixture is empty")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.requires_grad:
                raise ValueError(f"Fixture tensors must be detached CPU tensors: {key}")
            if value.ndim < 1 or len(value) != size:
                raise ValueError(f"Fixture tensor leading batch dimension mismatch: {key}")
    return batch, metadata


def slice_batch(batch, begin, end):
    n = len(batch["images"])
    return {key: value[begin:end] if isinstance(value, torch.Tensor) or
            (isinstance(value, (list, tuple)) and len(value) == n) else value for key, value in batch.items()}


def source_provenance():
    paths = [Path(__file__).resolve(), ROOT / "scripts/motiondrive_v2_training.py", ROOT / "scripts/sparse_scoredrive.py",
             *sorted((ROOT / "models/motiondrive_v2").glob("*.py"))]
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                          stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        git_sha = None
    return {"git_sha": git_sha, "files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths}}


def run_probe(checkpoint_path, fixture_path, *, device="cpu", batch_size=4, max_samples=8, model_factory=None):
    if not 1 <= batch_size <= 8 or not 1 <= max_samples <= 8:
        raise ValueError("Diagnostic scope is batch_size/max_samples in [1,8]")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and device.index not in (0, 1, 2, 3)):
        raise ValueError("Use cpu or an explicitly authorized cuda:0--3; device choice does not grant authorization")
    hashes = {"checkpoint_sha256": sha256(checkpoint_path), "fixture_sha256": sha256(fixture_path)}
    source = source_provenance()
    batch, metadata = load_fixture(fixture_path)
    model, weights, load_report = load_checkpoint(checkpoint_path, device, model_factory)
    reports = []
    limit = min(max_samples, len(batch["images"]))
    for begin in range(0, limit, batch_size):
        end = min(begin + batch_size, limit)
        small = slice_batch(batch, begin, end)
        result = probe_batch(model, small, weights, device)
        result["fixture_slice"] = [begin, end]
        result["sample_identity"] = {key: value.tolist() if isinstance(value, torch.Tensor) else value
                                     for key, value in small.items() if key in ("row", "frame", "scenario", "session_id")}
        reports.append(result)
    after = {"checkpoint_sha256": sha256(checkpoint_path), "fixture_sha256": sha256(fixture_path)}
    if hashes != after or source != source_provenance():
        raise RuntimeError("Checkpoint, fixture, or source changed during read-only probe")
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(Path(checkpoint_path).resolve()), "fixture": str(Path(fixture_path).resolve()),
            **hashes, "input_files_unchanged": True, "source": source, "load_report": load_report,
            "fixture_metadata": metadata, "device": str(device), "batch_size": batch_size,
            "fixture_samples": len(batch["images"]), "probed_samples": limit, "batches": reports,
            "aggregation_policy": "Batches remain separate; their norms/cosines are not a full-fixture gradient.",
            "optimizer_created": False, "optimizer_steps": 0, "bn_adaptive_passes": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True, help="New JSON report; existing files are never overwritten")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing report: {output}")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    report = run_probe(args.checkpoint, args.fixture, device=args.device,
                       batch_size=args.batch_size, max_samples=args.max_samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(output), "probed_samples": report["probed_samples"],
                      "batch_size": report["batch_size"], "optimizer_steps": 0}))


if __name__ == "__main__":
    main()
