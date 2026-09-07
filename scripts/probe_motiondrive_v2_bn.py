#!/usr/bin/env python3
"""Fixed-weight, train-images-only shared-BN recalibration diagnostic.

No optimizer, loss, goal or GT is used during recalibration. Updated BN buffers
exist only in memory; the source checkpoint is never written. Train/tune labels
are read only by the separate post-calibration evaluation. A tiny train set is
not generalization evidence, and this is not a deployment change.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from probe_motiondrive_v2_fit import make_dataset, predict, sha

CAMERAS = ("camera_front", "camera_front_right", "camera_front_left",
           "camera_rear_wide", "camera_rear_left", "camera_rear_right")
OFFSETS = (1, 2, 5, 10)
HOOK_BNS = ("stem.1", "layer1.0.bn1", "layer2.0.bn1", "layer3.0.bn1", "layer4.0.bn1")


class ImagesOnlyDataset(Dataset):
    """Do not call the full dataset __getitem__, or load scene target NPZs."""
    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source.rows)

    def __getitem__(self, index):
        row = int(self.source.rows[index])
        frame = int(self.source.arr["frame"][row])
        scene = str(self.source.scene_names[row])
        return {
            "images": torch.stack([self.source._image(scene, c, frame, (768, 432), None) for c in CAMERAS]),
            "history_images": torch.stack([self.source._image(scene, CAMERAS[0], frame - dt, (384, 216), None)
                                             for dt in OFFSETS]),
            "scenario": scene, "frame": frame, "row": row,
        }


def merge_moments(parts):
    """Combine per-channel population moments, accounting for between-batch mean."""
    count = sum(p["count"] for p in parts)
    mean = sum(p["count"] * np.asarray(p["mean"], float) for p in parts) / count
    second = sum(p["count"] * (np.asarray(p["variance"], float) + np.asarray(p["mean"], float) ** 2)
                 for p in parts) / count
    return {"count": count, "mean": mean, "variance": np.maximum(second - mean ** 2, 0)}


def channel_summary(values):
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "max": float(values.max())}


class BNMomentHooks:
    def __init__(self, backbone, names=HOOK_BNS):
        self.parts, self.tag, self.handles = {}, "unset", []
        modules = dict(backbone.named_modules())
        self.selected = [name for name in names if name in modules]
        for name in self.selected:
            module = modules[name]
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                raise TypeError(f"{name} is not BatchNorm")
            def capture(_module, inputs, name=name):
                value = inputs[0].detach().float()
                variance, mean = torch.var_mean(value, dim=(0, 2, 3), correction=0)
                self.parts.setdefault((self.tag, name), []).append({
                    "count": value.shape[0] * value.shape[2] * value.shape[3],
                    "mean": mean.cpu().numpy(), "variance": variance.cpu().numpy(),
                    "input_shape": list(value.shape)})
            self.handles.append(module.register_forward_pre_hook(capture))

    def close(self):
        for handle in self.handles:
            handle.remove()

    def report(self):
        result = {}
        modes = sorted({tag.split("/")[0] for tag, _ in self.parts})
        for mode in modes:
            layers = {}
            for name in self.selected:
                current = merge_moments(self.parts[(f"{mode}/current", name)])
                history = merge_moments(self.parts[(f"{mode}/history", name)])
                pooled_std = np.sqrt(.5 * (current["variance"] + history["variance"]) + 1e-8)
                normalized_gap = np.abs(current["mean"] - history["mean"]) / pooled_std
                log_var_ratio = np.abs(np.log((current["variance"] + 1e-8) / (history["variance"] + 1e-8)))
                layers[name] = {
                    "current": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in current.items()},
                    "history": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in history.items()},
                    "abs_mean_gap_in_pooled_std": channel_summary(normalized_gap),
                    "abs_log_variance_ratio": channel_summary(log_var_ratio),
                    "current_batch_input_shapes": [p["input_shape"] for p in self.parts[(f"{mode}/current", name)]],
                    "history_batch_input_shapes": [p["input_shape"] for p in self.parts[(f"{mode}/history", name)]],
                }
            result[mode] = layers
        return result


@torch.no_grad()
def image_passes(model, batch, device, hooks=None, mode="eval"):
    """Only shared visual encoder calls; no planner/scene/goal/label calls."""
    if set(batch) - {"images", "history_images", "scenario", "frame", "row"}:
        raise ValueError("Recalibration batch contains information outside image-only whitelist")
    current = batch["images"].to(device)
    history = batch["history_images"].to(device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    with autocast:
        if hooks is not None:
            hooks.tag = f"{mode}/current"
        model.backbone_fpn(current.flatten(0, 1))
        if model.config.motion_input_mode != "legacy":
            small = F.interpolate(current[:, 0], size=history.shape[-2:], mode="bilinear",
                                  align_corners=False, antialias=True)
            history = torch.cat([small[:, None], history], 1)
        if hooks is not None:
            hooks.tag = f"{mode}/history"
        model.backbone_fpn(history.flatten(0, 1))


@torch.no_grad()
def recalibrate_shared_bn(model, loader, device, hooks=None):
    """Standard reset + cumulative average over the fixed unlabeled train pass."""
    bns = {name: m for name, m in model.backbone_fpn.named_modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)}
    original_momentum = {name: m.momentum for name, m in bns.items()}
    model.eval()
    for m in bns.values():
        if not m.track_running_stats:
            raise ValueError("Recalibration requires tracked BN statistics")
        m.reset_running_stats()
        m.momentum = None
        m.train()
    records = []
    try:
        for batch in loader:
            image_passes(model, batch, device, hooks, mode="recalibration_train_bn")
            records.extend({"scenario": s, "frame": int(f)} for s, f in zip(batch["scenario"], batch["frame"]))
    finally:
        for name, m in bns.items():
            m.momentum = original_momentum[name]
            m.eval()
        model.eval()
    return {"n_imagesets": len(records), "records": records, "n_bn_modules": len(bns),
            "num_batches_tracked_by_module": {name: int(m.num_batches_tracked) for name, m in bns.items()},
            "method": "reset_running_stats; momentum=None; equal-weight cumulative mean of batch statistics; current pass then history pass",
            "calibration_inputs": ["images", "history_images"],
            "goal_or_labels_used": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, choices=[1], default=1)
    args = p.parse_args()
    from models.motiondrive_v2 import MotionDriveV2
    from motiondrive_v2_training import tensor_state_sha256
    torch.set_num_threads(4)
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f"cuda:{args.gpu}")
    run = Path(args.run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    checkpoint_path = run / "last.pth"
    checkpoint_sha = sha(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ckpt["step"] != 500 or tuple(ckpt["manifest"]["model_config"]["plan_output_scale"]) != (10., 5.):
        raise ValueError("This predeclared diagnostic expects the unit10x5 LAST500 checkpoint")
    model = MotionDriveV2(ckpt["manifest"]["model_config"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.requires_grad_(False).to(device).eval()
    original_buffers = {name: value.clone() for name, value in model.named_buffers()}
    parameter_hash_before = tensor_state_sha256(dict(model.named_parameters()))
    datasets = {split: make_dataset(manifest, split) for split in ("train", "tune")}
    if len(datasets["train"]) != 16 or len(datasets["tune"]) != 12:
        raise ValueError("Expected the predeclared train16/tune12 sample set")
    report = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
              "step": ckpt["step"], "original": {}, "recalibrated": {},
              "interpretation_limits": [
                  "No model parameters changed; only a temporary copy's BN running buffers were recalibrated.",
                  "Recalibration used train16 images only. Tune12 was evaluated afterwards and was not used to select statistics.",
                  "Current6 and history4 differ in camera composition, time and resolution; their moment gap does not isolate resolution as the cause.",
                  "Batch-stat mode is a diagnostic, not the deployment metric. This does not alter the fixed LAST500 pair decision.",
                  "A tiny fitting set and one tune scene cannot establish generalization or an optimal normalization strategy."]}
    for split, data in datasets.items():
        report["original"][split] = predict(model, data, device, 4)
    report["original"]["train_batchstats_diagnostic"] = predict(model, datasets["train"], device, 8, train_bn=True)
    images_only = DataLoader(ImagesOnlyDataset(datasets["train"]), batch_size=8,
                             shuffle=False, num_workers=0)
    hooks = BNMomentHooks(model.backbone_fpn)
    for batch in images_only:
        image_passes(model, batch, device, hooks, mode="original_eval")
    report["recalibration"] = recalibrate_shared_bn(model, images_only, device, hooks)
    for batch in images_only:
        image_passes(model, batch, device, hooks, mode="recalibrated_eval")
    hooks.close()
    report["bn_input_moments"] = hooks.report()
    for split, data in datasets.items():
        report["recalibrated"][split] = predict(model, data, device, 4)
    parameter_hash_after = tensor_state_sha256(dict(model.named_parameters()))
    if parameter_hash_before != parameter_hash_after:
        raise AssertionError("Learned parameters changed during a no-training diagnostic")
    changed = [name for name, buf in model.named_buffers() if not torch.equal(buf, original_buffers[name])]
    allowed_suffixes = ("running_mean", "running_var", "num_batches_tracked")
    if any(not name.startswith("backbone_fpn.") or not name.endswith(allowed_suffixes) for name in changed):
        raise AssertionError(f"Unexpected non-BN buffer changed: {changed}")
    report["integrity"] = {"parameter_sha256_before": parameter_hash_before,
                           "parameter_sha256_after": parameter_hash_after,
                           "parameters_bit_identical": True, "changed_buffers": changed,
                           "only_bn_running_buffers_changed": True}
    for name, buf in model.named_buffers():
        buf.copy_(original_buffers[name])
    report["integrity"]["original_buffers_restored_in_memory"] = all(
        torch.equal(buf, original_buffers[name]) for name, buf in model.named_buffers())
    report["integrity"]["checkpoint_file_unchanged"] = sha(checkpoint_path) == checkpoint_sha
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"original_train_d3": report["original"]["train"]["d3"],
                      "original_batchstats_d3": report["original"]["train_batchstats_diagnostic"]["d3"],
                      "recalibrated_train_d3": report["recalibrated"]["train"]["d3"],
                      "original_tune_d3": report["original"]["tune"]["d3"],
                      "recalibrated_tune_d3": report["recalibrated"]["tune"]["d3"],
                      "parameters_unchanged": True, "report": str(path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
