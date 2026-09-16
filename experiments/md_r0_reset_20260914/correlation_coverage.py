#!/usr/bin/env python3
"""How far does a static point move in the image between t0 and each past frame,
compared with how far the local correlation can search?

The trace fixes the search: radius 2 cells at stride 8 on the half-resolution
image, i.e. +/-16 half-res pixels, equivalently +/-32 current-resolution pixels.
This measures the displacement the ego's own motion produces for static ground
points, using the clip's real calibration and its real pose alignment.

Geometry only. No model, no training, no claim that a displacement beyond the
window makes the branch useless: features are pooled and fused, and a cost volume
that saturates still carries information.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models import motiondrive_v2_inputs as adapter

OUT = ROOT / "reports/md_exp_diagnosis_20260915/correlation_coverage.json"
TEST_ROOT = ROOT / "test"
FORWARD = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0]
LATERAL = [-3.0, 0.0, 3.0]
HEIGHTS = {"ground_z0": 0.0, "above_ground_z1": 1.0}


def project(points_ego, lidar2img, width, height):
    """Project ego-frame points with the clip's own matrix; returns pixels or NaN."""
    homogeneous = np.concatenate([points_ego, np.ones((len(points_ego), 1))], axis=1)
    camera = homogeneous @ np.asarray(lidar2img, dtype=np.float64).T
    depth = camera[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        pixels = camera[:, :2] / depth[:, None]
    inside = (depth > 0.1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) \
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    pixels[~inside] = np.nan
    return pixels


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--clips", type=int, default=40)
    args = parser.parse_args()

    trace = json.loads((ROOT / "reports/md_exp_diagnosis_20260915/motion_shape_trace.json").read_text())
    radius = trace["config"]["correlation_radius"]
    current_h, current_w = trace["derived"]["current_image_hw"]
    history_h, history_w = trace["derived"]["history_image_hw"]
    fine_h, fine_w = trace["derived"]["correlation_feature_hw"]
    stride_half = history_w / fine_w                 # pixels of the half-res image per cell
    search_half_px = radius * stride_half            # search halfwidth, half-res pixels
    seconds = trace["config"]["nominal_history_seconds"]

    rows = {name: {str(s): [] for s in seconds} for name in HEIGHTS}
    speeds = []
    used = 0
    for tar in sorted(TEST_ROOT.glob("*.tar"))[:args.clips]:
        try:
            import tarfile, io
            import pyarrow.parquet as pq
            with tarfile.open(tar) as archive:
                members = {m.name: m for m in archive.getmembers() if m.isfile()}
                token = sorted({n.split("/", 1)[0] for n in members})[0]
                read = lambda n: archive.extractfile(members[f"{token}/{n}"]).read()
                calibration = pq.read_table(io.BytesIO(read("calibration.parquet")),
                                            columns=list(adapter.CALIBRATION_COLUMNS)).to_pylist()
                poses = pq.read_table(io.BytesIO(read("ego_pose.parquet")),
                                      columns=list(adapter.POSE_COLUMNS)).to_pylist()
                prepared = adapter.prepare_clip_from_records(
                    calibration, poses, lambda camera, frame: read(f"{camera}/frame_{frame}.jpg"))
        except Exception:
            continue
        used += 1
        lidar2img = prepared.inputs["lidar2img"].numpy()[0][0]      # front camera
        transforms = prepared.inputs["history_transforms"].numpy()[0]
        # travelled distance to each past frame, from the alignment translation
        travel = np.linalg.norm(transforms[:, :3, 3], axis=-1)
        speeds.append(float(travel[-1] / seconds[-1]))

        for name, z in HEIGHTS.items():
            grid = np.array([[x, y, z] for x in FORWARD for y in LATERAL], dtype=np.float64)
            now = project(grid, lidar2img, current_w, current_h)
            for k, second in enumerate(seconds):
                past_points = (np.concatenate([grid, np.ones((len(grid), 1))], axis=1)
                               @ transforms[k].T)[:, :3]
                past = project(past_points, lidar2img, current_w, current_h)
                delta = np.linalg.norm(now - past, axis=-1) / 2.0    # half-res pixels
                rows[name][str(second)].extend(delta[np.isfinite(delta)].tolist())

    summary = {}
    for name, per_second in rows.items():
        summary[name] = {}
        for second, values in per_second.items():
            values = np.asarray(values)
            if not len(values):
                continue
            summary[name][second] = {
                "samples": int(len(values)),
                "median_halfres_px": float(np.median(values)),
                "p90_halfres_px": float(np.percentile(values, 90)),
                "max_halfres_px": float(values.max()),
                "fraction_beyond_search_window": float((values > search_half_px).mean()),
            }

    payload = {
        "schema_version": 1,
        "search_window": {
            "correlation_radius_cells": radius,
            "cell_stride_halfres_px": stride_half,
            "search_halfwidth_halfres_px": search_half_px,
            "search_halfwidth_current_px": search_half_px * (current_w / history_w),
            "source": "reports/md_exp_diagnosis_20260915/motion_shape_trace.json",
        },
        "measured_on": {"clips": used, "camera": "front",
                        "points": "static ego-frame grid, forward "
                                  f"{FORWARD} m by lateral {LATERAL} m",
                        "median_speed_ms": float(np.median(speeds)) if speeds else None},
        "displacement_halfres_px": summary,
        "reading": (
            "displacement grows with the time gap and shrinks with distance ahead; the fraction "
            "beyond the search window says how often the local cost volume is asked for a match "
            "outside the offsets it can represent"),
        "not_claimed": [
            "a displacement beyond the window does not make the branch useless: the coarse level "
            "has a wider reach in pixels, features are pooled and fused, and a saturated cost "
            "volume still carries a monotone cue",
            "this is front-camera ground-plane geometry, not a measurement of what the network uses",
            "no conclusion here about whether adding near or long observations helps",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"search_halfwidth_halfres_px": search_half_px,
                      "clips": used,
                      "median_speed_ms": payload["measured_on"]["median_speed_ms"],
                      "ground_z0": summary["ground_z0"]}, indent=1))


if __name__ == "__main__":
    main()
