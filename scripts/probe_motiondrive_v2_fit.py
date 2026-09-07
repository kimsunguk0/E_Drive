#!/usr/bin/env python3
"""Read-only checkpoint fitting audit. No optimizer steps or weight-file edits.

Metrics use deterministic, unaugmented samples selected exactly as the run.
Training-BN mode is a diagnostic on an in-memory model with buffers restored.
Gradient cosine is local evidence, not a causal ablation of the training run.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from motiondrive_v2_training import LossWeights, compute_loss, model_inputs, to_device

WEIGHTS = np.asarray([11, 11, 5, 5, 2, 2], float) / 36


def summary(pred, target):
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    delta = pred - target
    errors = np.linalg.norm(delta, axis=-1)
    pd = np.linalg.norm(np.diff(pred, axis=1), axis=-1)
    td = np.linalg.norm(np.diff(target, axis=1), axis=-1)
    progression = np.linalg.norm(pred[:, -1] - pred[:, 0], axis=-1)
    reference = np.linalg.norm(target[:, -1] - target[:, 0], axis=-1)
    return {
        "n": len(pred), "d3": float(np.mean(errors @ WEIGHTS)),
        "d3_per_frame": (errors @ WEIGHTS).tolist(),
        "l2_by_time": errors.mean(0).tolist(),
        "longitudinal_mae_by_time": np.abs(delta[..., 0]).mean(0).tolist(),
        "lateral_mae_by_time": np.abs(delta[..., 1]).mean(0).tolist(),
        "bias_xy_by_time": delta.mean(0).tolist(),
        "pred_xy_mean_by_time": pred.mean(0).tolist(),
        "gt_xy_mean_by_time": target.mean(0).tolist(),
        "pred_xy_std_by_time": pred.std(0).tolist(),
        "gt_xy_std_by_time": target.std(0).tolist(),
        "pred_adjacent_distance_mean": pd.mean(0).tolist(),
        "gt_adjacent_distance_mean": td.mean(0).tolist(),
        "first_to_last_pred_p50": float(np.median(progression)),
        "first_to_last_gt_p50": float(np.median(reference)),
        "collapsed_below_0p1m_on_gt_progress_over_1m": int(((progression < .1) & (reference > 1)).sum()),
        "pred_range_xy": [pred.min((0, 1)).tolist(), pred.max((0, 1)).tolist()],
        "gt_range_xy": [target.min((0, 1)).tolist(), target.max((0, 1)).tolist()],
    }


def parameter_group(name):
    if name.startswith("backbone_fpn."):
        return "backbone_fpn"
    return name.split(".")[0]


def vector_statistics(a, b):
    aa, bb, dot = 0., 0., 0.
    for av, bv in zip(a, b):
        if av is not None:
            aa += float(av.double().square().sum())
        if bv is not None:
            bb += float(bv.double().square().sum())
        if av is not None and bv is not None:
            dot += float((av.double() * bv.double()).sum())
    return {"plan_grad_norm": math.sqrt(aa), "aux_grad_norm": math.sqrt(bb),
            "cosine": dot / math.sqrt(aa * bb) if aa * bb > 0 else None,
            "aux_to_plan_norm_ratio": math.sqrt(bb / aa) if aa > 0 else None}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def make_dataset(manifest, split):
    from motiondrive_v2_data import MotionDriveDataset
    a = manifest["arguments"]
    train = split == "train"
    return MotionDriveDataset(data_root=a["data_root"], split_manifest=a["split_manifest"],
        supervision_root=a["supervision_root"], split=split, min_frame=30,
        frame_stride=a["train_stride"] if train else a["eval_stride"],
        max_samples=a["max_train_samples"] if train else a["max_eval_samples"],
        scenes=a["train_scenes"] if train else a["eval_scenes"], augment=False, seed=a["seed"])


@torch.inference_mode()
def predict(model, data, device, batch_size, train_bn=False):
    saved_buffers = {name: buf.clone() for name, buf in model.named_buffers()} if train_bn else None
    model.train(train_bn)
    predictions, targets, state_predictions, state_targets, records = [], [], [], [], []
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    for raw in loader:
        batch = to_device(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**model_inputs(batch))
        predictions.append(out["plan_abs"].cpu().numpy())
        targets.append(batch["gt_plan"].cpu().numpy())
        state_predictions.append(out["state_hat"][:, :5].cpu().numpy())
        state_targets.append(batch["state_target"][:, :5].cpu().numpy())
        records.extend({"scenario": s, "frame": int(f)} for s, f in zip(raw["scenario"], raw["frame"]))
    pred, target = np.concatenate(predictions), np.concatenate(targets)
    result = summary(pred, target)
    result["state_mae_vx_vy_ax_ay_yawrate"] = np.abs(np.concatenate(state_predictions) - np.concatenate(state_targets)).mean(0).tolist()
    result["records"] = records
    result["predictions"] = pred.tolist()
    result["targets"] = target.tolist()
    if saved_buffers is not None:
        for name, buf in model.named_buffers():
            buf.copy_(saved_buffers[name])
    model.eval()
    return result


def gradient_probe(model, dataset, device, batch_size=2):
    model.eval()
    raw = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)))
    batch = to_device(raw, device)
    shared = {}
    hook = model.scene_encoder.refine.register_forward_hook(lambda _m, _i, o: shared.__setitem__("raster", o))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**model_inputs(batch))
    hook.remove()
    _, parts = compute_loss(out, batch, LossWeights())
    _, plain = compute_loss(out, batch, LossWeights(uncertainty=False))
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    tensors = [p for _, p in parameters]
    # Two complete gradient vectors, immediately detached; no optimizer update.
    gp = torch.autograd.grad(parts["plan_d3"], tensors, retain_graph=True, allow_unused=True)
    aux = .2 * (parts["occ_bce"] + parts["lane_bce"] + parts["motion"])
    ga = torch.autograd.grad(aux, tensors, retain_graph=True, allow_unused=True)
    stats = {"all": vector_statistics(gp, ga)}
    for group in sorted(set(parameter_group(n) for n, _ in parameters)):
        ids = [i for i, (n, _) in enumerate(parameters) if parameter_group(n) == group]
        stats[group] = vector_statistics([gp[i] for i in ids], [ga[i] for i in ids])
    grads = {}
    for task in ("plan_d3", "occ_bce", "lane_bce", "motion"):
        value = torch.autograd.grad(parts[task], shared["raster"], retain_graph=True, allow_unused=True)[0]
        grads[task] = float(value.float().norm()) if value is not None else None
    queried = {"scene_features": out["scene_features"], "motion_features": out["motion_features"],
               "state_hat": out["state_hat"], "history_hat": out["history_hat"],
               "waypoint_queries": model.planner.waypoint_queries,
               "xy_output_weight": model.planner.xy_head[-1].weight}
    path = {}
    for name, tensor in queried.items():
        g = torch.autograd.grad(parts["plan_d3"], tensor, retain_graph=True, allow_unused=True)[0]
        path[name] = {"value_norm": float(tensor.float().norm()),
                      "gradient_norm": float(g.float().norm()) if g is not None else None}
    motion_params = [p for n, p in parameters if parameter_group(n) in ("motion_encoder", "backbone_fpn")]
    gm = torch.autograd.grad(.2 * parts["motion"], motion_params, retain_graph=True, allow_unused=True)
    gplain = torch.autograd.grad(.2 * plain["motion"], motion_params, retain_graph=False, allow_unused=True)
    scale_comparison = vector_statistics(gplain, gm)
    return {"batch_records": [{"scenario": s, "frame": int(f)} for s, f in zip(raw["scenario"], raw["frame"])],
            "loss_parts": {k: float(v.detach()) for k, v in parts.items()},
            "shared_parameter_plan_vs_weighted_aux": stats, "shared_raster_gradient_norms": grads,
            "planning_path_gradients": path, "motion_smoothl1_vs_nll_gradient": scale_comparison,
            "history_logvar_range": [float(out["history_logvar"].min()), float(out["history_logvar"].max())],
            "state_logvar_range": [float(out["state_logvar"].min()), float(out["state_logvar"].max())],
            "interpretation_limit": "One small eval-mode batch; local gradient conflict is not causal evidence of training limitation."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, choices=[1], default=1)
    args = parser.parse_args()
    from models.motiondrive_v2 import MotionDriveV2
    run = Path(args.run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f"cuda:{args.gpu}")
    datasets = {split: make_dataset(manifest, split) for split in ("train", "tune")}
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    trains = [r for r in rows if r["kind"] == "train"]
    report = {"run_manifest": manifest,
              "protocol": "unaugmented fixed rows, eval-mode bf16 forward, FP32 direct output/loss; diagnostic only",
              "logged_evaluation_curve": [r for r in rows if "eval" in r["kind"]],
              "logged_train_last_20": trains[-20:],
              "clipped_logged_step_fraction": float(np.mean([r["grad_norm"] > manifest["arguments"]["grad_clip"] for r in trains])),
              "checkpoints": {}}
    for name in ("best", "last"):
        ckpt_path = run / f"{name}.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = MotionDriveV2(ckpt["manifest"]["model_config"])
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device).eval()
        result = {"step": ckpt["step"], "sha256": sha(ckpt_path)}
        for split, data in datasets.items():
            result[split] = predict(model, data, device, 4)
        result["train_bn_batch8_diagnostic"] = predict(model, datasets["train"], device, 8, train_bn=True)
        result["gradient_probe"] = gradient_probe(model, datasets["train"], device)
        report["checkpoints"][name] = result
        print(json.dumps({"checkpoint": name, "step": ckpt["step"],
                          "train_d3": result["train"]["d3"], "tune_d3": result["tune"]["d3"],
                          "train_bn_d3": result["train_bn_batch8_diagnostic"]["d3"]}), flush=True)
        del model, ckpt
        torch.cuda.empty_cache()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report: {path}", flush=True)


if __name__ == "__main__":
    main()
