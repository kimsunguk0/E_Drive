#!/usr/bin/env python3
"""Audit whether the legacy VO checkpoint relies on dynamic pose alignment.

This is an evaluation-only diagnostic.  Ground-truth ``hist_T`` is retained for
metrics while the transform and/or past images passed to the model are changed.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import sparse_common as C  # noqa: E402
from train_sparse_scoredrive import TrainableSparseScoreDrive  # noqa: E402


def weighted(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(x, weights=w))


def summarize(pred: np.ndarray, gt: np.ndarray, weight: np.ndarray) -> dict:
    out = {}
    for k, dt in enumerate((0.5, 1.0, 1.5)):
        pnorm = np.linalg.norm(pred[:, k], axis=-1)
        gnorm = np.linalg.norm(gt[:, k], axis=-1)
        out[str(dt)] = {
            "vector_mae_m": weighted(np.linalg.norm(pred[:, k] - gt[:, k], axis=-1), weight),
            "speed_mae_mps": weighted(np.abs(pnorm - gnorm) / dt, weight),
            "speed_corr": float(np.corrcoef(pnorm / dt, gnorm / dt)[0, 1]),
            "pred_speed_mean_mps": weighted(pnorm / dt, weight),
            "gt_speed_mean_mps": weighted(gnorm / dt, weight),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="work_dirs/sc_vo_w10/best.pth")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--limit", type=int, default=600)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    ckpt = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu")
    saved = ckpt["args"]
    n_hist = int(saved.get("n_hist", 3))
    model = TrainableSparseScoreDrive(
        C.BANK_A0,
        arch=str(saved.get("arch", "resnet34")),
        logit_norm=bool(saved.get("logit_norm", 1)),
        n_hist=n_hist,
        speed_head=bool(saved.get("w_speed", 0) > 0),
        speed_gamma=0.0,
        vo_head=bool(saved.get("w_vo", 0) > 0),
        vo_bins=int(saved.get("vo_bins", 0)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.fuse_mul = bool(saved.get("fuse_mul", 0))
    model.merge_encode = bool(saved.get("merge_encode", 0))
    model.eval()

    arrays = C.load_arrays()
    rows_all = arrays["val_idx"]
    keep = (arrays["frame"][rows_all] >= 30) & C.history_available(arrays, rows_all, n_hist)
    rows = rows_all[keep][: args.limit]
    weight = arrays["val_weight"][keep][: args.limit].astype(np.float64)
    loader = DataLoader(
        C.SparseFrameDataset(arrays, rows, n_hist=n_hist),
        batch_size=args.batch,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )
    lidar2img = torch.from_numpy(C.build_global_lidar2img()).to(device)
    modes = ("actual", "identity_pose", "current_as_past", "shuffled_pose", "shuffled_past")
    predictions = {mode: [] for mode in modes}
    labels = []

    with torch.inference_mode():
        for batch in loader:
            image = batch["img"].to(device, non_blocking=True)
            past = batch["img_hist"].to(device, non_blocking=True)
            true_transform = batch["hist_T"].to(device, non_blocking=True)
            batch_lidar2img = lidar2img.unsqueeze(0).expand(image.shape[0], -1, -1, -1)
            identity = torch.eye(4, device=device, dtype=true_transform.dtype).view(1, 1, 4, 4)
            identity = identity.expand_as(true_transform)
            current_past = image[:, None].expand(-1, n_hist, -1, -1, -1, -1).contiguous()
            order = torch.arange(image.shape[0] - 1, -1, -1, device=device)
            inputs = {
                "actual": (past, true_transform),
                "identity_pose": (past, identity),
                "current_as_past": (current_past, true_transform),
                "shuffled_pose": (past, true_transform[order]),
                "shuffled_past": (past[order], true_transform),
            }
            for mode, (mode_past, mode_transform) in inputs.items():
                with torch.autocast("cuda", dtype=torch.float16):
                    model.temporal_logits(image, mode_past, batch_lidar2img, mode_transform)
                predictions[mode].append(model._vo_pred.float().cpu().numpy())
            labels.append(true_transform[:, :, :2, 3].float().cpu().numpy())

    gt = np.concatenate(labels).astype(np.float64)
    actual = np.concatenate(predictions["actual"]).astype(np.float64)
    print(f"checkpoint={args.ckpt} rows={len(rows)}")
    for mode in modes:
        pred = np.concatenate(predictions[mode]).astype(np.float64)
        report = summarize(pred, gt, weight)
        delta = weighted(np.linalg.norm(pred - actual, axis=-1).mean(axis=1), weight)
        print(f"\n[{mode}] mean vector change vs actual={delta:.6f} m")
        for dt, metrics in report.items():
            fields = " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
            print(f"  dt={dt}s {fields}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
