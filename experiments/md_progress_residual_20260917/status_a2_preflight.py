#!/usr/bin/env python3
"""One-batch zero-init parity check for the STATUS-A2-S experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
BASE_DIR = ROOT / "experiments/md_r0_reset_20260914"
EXPERIMENT_DIR = ROOT / "experiments/md_progress_residual_20260917"
for path in (ROOT, ROOT / "scripts", BASE_DIR, EXPERIMENT_DIR):
    sys.path.insert(0, str(path))

import matching_resolution as mr
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from motiondrive_v2_data import MotionDriveDataset
from progress_residual import (CausalStatusDataset, PROVIDED_STATUS_KEY,
                               ResidualMotionDriveV2)


CHECKPOINT = ROOT / "work_dirs/md_r0_reset_20260914/MR-NATIVE-s1/last.pth"
SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json"
SUPERVISION = ROOT / "data/etri/motiondrive_v2/r0reset_tplus_geometry_v2"


def batch_tensor(item, key, device):
    return item[key][None].to(device, non_blocking=True)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--scene-refiner", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full-resolution parity preflight")
    device = torch.device("cuda:0")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config_dict = payload["manifest"]["model_config"]

    base = MotionDriveV2(MotionDriveV2Config(**config_dict))
    base.load_state_dict(payload["model"], strict=True)
    mr.install(base, "native")
    residual = ResidualMotionDriveV2(
        MotionDriveV2Config(**config_dict), coefficient_count=1,
        side_enabled=False, side_auxiliary=False, progress_denoise=False,
        shared_status_query=True, scene_refiner=args.scene_refiner, cap=1.)
    incompatible = residual.load_state_dict(payload["model"], strict=False)
    allowed = ("progress_refiner.", "shared_status_query_fusion.")
    if incompatible.unexpected_keys or not incompatible.missing_keys \
            or any(not key.startswith(allowed) for key in incompatible.missing_keys):
        raise RuntimeError(f"unexpected initialization delta: {incompatible}")

    raw = MotionDriveDataset(
        data_root="/tmp/pm97", split_manifest=str(SPLIT), split="tune",
        supervision_root=str(SUPERVISION), min_frame=30, frame_stride=5,
        max_samples=1, augment=False, seed=1, history_contract="control")
    dataset = CausalStatusDataset(mr.MotionCanvasDataset(raw, "native"))
    item = dataset[0]
    common_keys = ("images", "history_images", "lidar2img", "history_transforms",
                   "time_offsets", "goal_xy")
    common = {key: batch_tensor(item, key, device) for key in common_keys}
    common["motion_current"] = batch_tensor(item, mr.MOTION_CURRENT_KEY, device)
    common["motion_history"] = batch_tensor(item, mr.MOTION_HISTORY_KEY, device)
    status = batch_tensor(item, PROVIDED_STATUS_KEY, device)

    base.eval().to(device)
    residual.eval().to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        base_output = base(**common)
        residual_output = residual(**common, provided_status5=status)
    comparisons = {
        "base_vs_residual_base": float((base_output["plan_abs"]
                                         - residual_output["plan_base_abs"]).abs().max()),
        "base_vs_residual_final": float((base_output["plan_abs"]
                                          - residual_output["plan_abs"]).abs().max()),
        "coefficient_abs_max": float(residual_output["progress_coeff"].abs().max()),
    }
    if any(value != 0. for value in comparisons.values()):
        raise RuntimeError(f"zero-init parity failed: {comparisons}")
    if residual._shared_status_context is not None:
        raise RuntimeError("shared-status context leaked beyond forward")
    print({"status": "passed", **comparisons,
           "scene_refiner": args.scene_refiner,
           "provided_status5": status.float().cpu().tolist()[0],
           "missing_new_keys": len(incompatible.missing_keys)})


if __name__ == "__main__":
    main()
