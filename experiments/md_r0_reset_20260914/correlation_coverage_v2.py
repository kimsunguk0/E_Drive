#!/usr/bin/env python3
"""Geometric support of the local correlation window, computed correctly.

Fixes three things the review caught in the first pass:
  * the window is a square, so exceedance is |du| > r*s_x OR |dv| > r*s_y,
    not sqrt(du^2+dv^2) > r*s;
  * fine and coarse levels have different strides, so they are reported
    separately together with the "outside both" case;
  * a point must be valid (positive depth, inside the frame) in BOTH the current
    and the past view before its displacement counts.

This is geometric support, not matching success: texture, occlusion, rotation and
the ground-plane assumption are all outside it, and the exceedance fraction is not
a share of D3.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models import motiondrive_v2_inputs as adapter

OUT = ROOT / "reports/md_exp_diagnosis_20260915/correlation_coverage_v2.json"
TRACE = ROOT / "reports/md_exp_diagnosis_20260915/motion_shape_trace.json"
# Fixed diagnostic ROI, declared here rather than tuned afterwards.
FORWARD = [5.0, 8.0, 12.0, 20.0, 30.0, 45.0, 60.0]
LATERAL = [-4.0, -2.0, 0.0, 2.0, 4.0]
HEIGHTS = {"ground_z0": 0.0, "above_ground_z1": 1.0}


def project(points_ego, lidar2img, width, height):
    homogeneous = np.concatenate([points_ego, np.ones((len(points_ego), 1))], axis=1)
    camera = homogeneous @ np.asarray(lidar2img, dtype=np.float64).T
    depth = camera[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        pixels = camera[:, :2] / depth[:, None]
    valid = (depth > 0.1) & np.isfinite(pixels).all(1) \
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) \
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    return pixels, valid


def clips_from_train(split_manifest, supervision_root, limit):
    """Use dataset rows, so the summary is on train/V0 data rather than test."""
    from motiondrive_v2_data import MotionDriveDataset
    dataset = MotionDriveDataset(
        data_root="/tmp/pm97", split_manifest=str(split_manifest), split="tune",
        supervision_root=str(supervision_root), min_frame=30, frame_stride=5,
        augment=False, seed=0, history_contract="control")
    step = max(1, len(dataset) // limit)
    for index in range(0, len(dataset), step):
        item = dataset[index]
        yield (item["lidar2img"].numpy()[0], item["history_transforms"].numpy(),
               item["time_offsets"].numpy())


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--rows", type=int, default=120)
    args = parser.parse_args()

    trace = json.loads(TRACE.read_text())
    radius = trace["config"]["correlation_radius"]
    current_h, current_w = trace["derived"]["current_image_hw"]
    history_h, history_w = trace["derived"]["history_image_hw"]
    levels = {}
    for name, call in zip(("fine", "coarse"), trace["local_correlation_calls"]):
        fh, fw = call["current_in"][-2:]
        levels[name] = {
            "feature_hw": [fh, fw],
            "nominal_stride_halfres_px": [history_h / fh, history_w / fw],
            "search_halfwidth_halfres_px": [radius * history_h / fh, radius * history_w / fw],
            "nominal_stride_current_px": [current_h / fh, current_w / fw],
            "search_halfwidth_current_px": [radius * current_h / fh, radius * current_w / fw],
            "bins": call["bins"],
            "window_shape": "square, per-axis test",
            "stride_caveat": ("nominal stride from the image size over the feature size; the "
                              "true mapping follows the network's sampling and resize contract "
                              "and is affected by padding and rounding"),
        }

    seconds = trace["config"]["nominal_history_seconds"]
    records = {name: {str(s): {"du": [], "dv": []} for s in seconds} for name in HEIGHTS}
    speeds, used = [], 0
    for lidar2img, transforms, offsets in clips_from_train(
            ROOT / "data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json",
            ROOT / "data/etri/motiondrive_v2/r0reset_tplus_geometry_v2", args.rows):
        used += 1
        speeds.append(float(np.linalg.norm(transforms[-1][:3, 3]) / offsets[-1]))
        for name, z in HEIGHTS.items():
            grid = np.array([[x, y, z] for x in FORWARD for y in LATERAL], dtype=np.float64)
            now, now_ok = project(grid, lidar2img, current_w, current_h)
            for k, second in enumerate(seconds):
                past_points = (np.concatenate([grid, np.ones((len(grid), 1))], axis=1)
                               @ transforms[k].T)[:, :3]
                past, past_ok = project(past_points, lidar2img, current_w, current_h)
                both = now_ok & past_ok          # valid in BOTH views
                delta = np.abs(now - past) / 2.0  # half-res pixels, per axis
                records[name][str(second)]["du"].extend(delta[both, 0].tolist())
                records[name][str(second)]["dv"].extend(delta[both, 1].tolist())

    summary = {}
    for name, per_second in records.items():
        summary[name] = {}
        for second, axes in per_second.items():
            du, dv = np.asarray(axes["du"]), np.asarray(axes["dv"])
            if not len(du):
                continue
            entry = {"samples_valid_in_both_views": int(len(du)),
                     "median_du_halfres_px": float(np.median(du)),
                     "median_dv_halfres_px": float(np.median(dv)),
                     "p90_du_halfres_px": float(np.percentile(du, 90)),
                     "p90_dv_halfres_px": float(np.percentile(dv, 90))}
            outside = {}
            for level, spec in levels.items():
                sy, sx = spec["search_halfwidth_halfres_px"]
                beyond = (du > sx) | (dv > sy)
                outside[level] = float(beyond.mean())
                entry[f"beyond_{level}_window"] = float(beyond.mean())
            sy_c, sx_c = levels["coarse"]["search_halfwidth_halfres_px"]
            sy_f, sx_f = levels["fine"]["search_halfwidth_halfres_px"]
            entry["beyond_both_windows"] = float((((du > sx_f) | (dv > sy_f))
                                                  & ((du > sx_c) | (dv > sy_c))).mean())
            summary[name][second] = entry

    payload = {
        "schema_version": 2,
        "supersedes": "correlation_coverage.json, which used a Euclidean test and one stride",
        "levels": levels,
        "measured_on": {
            "source": "tune (V0) dataset rows, stride 5",
            "rows_used": used, "camera": "front",
            "roi": {"forward_m": FORWARD, "lateral_m": LATERAL, "heights_m": HEIGHTS},
            "median_speed_ms": float(np.median(speeds)) if speeds else None,
            "validity": "a point counts only if it has positive depth and lies inside the frame in BOTH views",
        },
        "per_axis_displacement_halfres_px": summary,
        "what_this_is": ("geometric support of the search window for static ground points; it "
                         "says how often the window is asked for an offset it cannot represent"),
        "what_this_is_not": [
            "matching success: texture, occlusion, rotation and the ground-plane assumption are outside it",
            "a share of D3 or a cause attribution",
            "evidence that the cost volume responds monotonically beyond its range",
            "a distribution to align the training sampler or model selection to",
        ],
        "reference_only": ("the earlier 40-clip figure came from test inputs and is kept as a "
                           "geometric reference; nothing is tuned to it, and H is not used"),
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "rows": used,
        "median_speed_ms": payload["measured_on"]["median_speed_ms"],
        "fine_search_halfwidth_halfres_px": levels["fine"]["search_halfwidth_halfres_px"],
        "coarse_search_halfwidth_halfres_px": levels["coarse"]["search_halfwidth_halfres_px"],
        "ground_z0": summary["ground_z0"],
    }, indent=1))


if __name__ == "__main__":
    main()
