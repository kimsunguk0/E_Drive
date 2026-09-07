#!/usr/bin/env python3
"""Build current annotation rasters/full-SE3 motion targets, never input GT status.

Only user-owned TRAIN metadata is read. Existing outputs are not overwritten.
Occupancy is ANNOTATED-VALID-OBJECT FOOTPRINT, not certified physical free space.
Negatives assume annotation completeness only within a conservative support mask;
all invisible cells, invalid boxes, and outside-support cells remain unknown.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from build_grouped_split_v2 import sha256, validate_manifest
from motiondrive_v2_data import (CAMERA_ORDER, GRID_EXTENT, GRID_SHAPE,
                                HISTORY_OFFSETS, full_pose_matrices, grid_centers,
                                load_calibration, motion_targets)

ANNOTATION_CONTRACT = {
    "occupancy_target": "current annotated non-ego object footprint with num_points > 0",
    "negative_assumption": "annotations exhaustive only inside visible conservative current valid-object support; this assumption is not independently certified",
    "support": "NEGATIVES: valid-object center bounding rectangle padded by 3m, intersect camera-visible grid and radius<=40m; POSITIVES: any visible valid annotated footprint inside grid",
    "unknown": "outside support, camera-invisible, invalid-object footprints, frames with no valid objects",
    "not_claimed": "physical free-space, obstacle-free ray tracing, complete occupancy, drivable area",
    "lane_target": "actual map.parquet polylines; lane-valid support <=approximately 8m from known lines, camera-visible; not hd_map coordinate mixing",
    "object_dimensions": "raw width[m] is longitudinal (heading-axis) extent; raw length[m] is lateral extent, matching source converter swap",
}


def transform_points(points, world_to_ego):
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected Nx3 global coordinates")
    return points @ world_to_ego[:3, :3].T + world_to_ego[:3, 3]


def camera_visible(lidar2img, centers=None):
    xy = grid_centers() if centers is None else centers
    # Ground and object midpoint: visibility only, not evidence of free space.
    visible = np.zeros(xy.shape[:-1], dtype=bool)
    for height in (0., 1.):
        points = np.concatenate((xy, np.full((*xy.shape[:-1], 1), height), np.ones((*xy.shape[:-1], 1))), -1)
        projected = np.einsum("cij,xyj->cxyi", lidar2img.astype(np.float64), points)
        z = projected[..., 2]
        safe_z = np.where(z > .05, z, 1.)
        u, v = projected[..., 0] / safe_z, projected[..., 1] / safe_z
        visible |= ((z > .05) & (u >= 0) & (u < 768) & (v >= 0) & (v < 432)).any(0)
    return visible


def pixel_coordinates(xy, shape=GRID_SHAPE, extent=GRID_EXTENT):
    """PIL axis0=column=y, axis1=row=x; grid cells represent their centers."""
    x0, x1, y0, y1 = extent
    nx, ny = shape
    xy = np.asarray(xy)
    return np.column_stack(((xy[:, 1] - y0) * ny / (y1 - y0) - .5,
                            (xy[:, 0] - x0) * nx / (x1 - x0) - .5))


def polygon_mask(xy):
    canvas = Image.new("L", (GRID_SHAPE[1], GRID_SHAPE[0]), 0)
    coords = pixel_coordinates(xy)
    if np.isfinite(coords).all():
        ImageDraw.Draw(canvas).polygon([tuple(p) for p in coords], fill=1)
    return np.asarray(canvas, dtype=bool)


def object_corners_world(record):
    x, y, z = (float(record[k]) for k in ("x[m]", "y[m]", "z[m]"))
    yaw = float(record["heading[rad]"])
    longitudinal, lateral = float(record["width[m]"]), float(record["length[m]"])
    if not np.isfinite([x, y, z, yaw, longitudinal, lateral]).all() or min(longitudinal, lateral) <= 0:
        return None
    local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
    local *= np.asarray([longitudinal, lateral]) / 2
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xy = local @ rotation.T + [x, y]
    return np.column_stack((xy, np.full(4, z)))


def rasterize_objects(records, world_to_ego, visible):
    target = np.zeros(GRID_SHAPE, bool)
    invalid = np.zeros_like(target)
    centers, malformed = [], False
    for record in records:
        if str(record["class"]).lower() == "ego":
            continue
        corners = object_corners_world(record)
        if corners is None:
            malformed = True
            continue
        local = transform_points(corners, world_to_ego)
        mask = polygon_mask(local[:, :2])
        num_points = float(record.get("num_points", 0))
        if np.isfinite(num_points) and num_points > 0:
            target |= mask
            if (mask & visible).any():
                centers.append(local[:, :2].mean(0))
        else:
            invalid |= mask
    support = np.zeros_like(target)
    xy = grid_centers()
    if centers and not malformed:
        centers = np.asarray(centers)
        lower, upper = centers.min(0) - 3., centers.max(0) + 3.
        support = ((xy >= lower) & (xy <= upper)).all(-1)
        support &= (np.linalg.norm(xy, axis=-1) <= 40.) & visible & ~invalid
    # A visible, valid annotated footprint itself is valid even outside negative support.
    support |= target & visible & ~invalid
    if malformed:
        # No negative can be certified if a malformed object could occupy it.
        support = target & visible & ~invalid
    return target[None].astype(np.float32), support[None]


def normalize_map(table):
    if "points" not in table.columns:
        raise ValueError(f"Unsupported map schema: {list(table.columns)}")
    polylines = []
    for points in table["points"]:
        points = np.asarray(list(points), np.float64)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("Invalid map polyline")
        if points.shape[1] == 2:
            # map.parquet without z is allowed only if explicitly documented in report.
            points = np.column_stack((points, np.zeros(len(points))))
        if len(points) >= 2 and np.isfinite(points).all():
            polylines.append(points)
    return polylines


def rasterize_lanes(polylines, world_to_ego, visible):
    lane = Image.new("L", (GRID_SHAPE[1], GRID_SHAPE[0]), 0)
    support = Image.new("L", lane.size, 0)
    dl, ds = ImageDraw.Draw(lane), ImageDraw.Draw(support)
    for points in polylines:
        local = transform_points(points, world_to_ego)
        xy = pixel_coordinates(local[:, :2])
        # PIL clips line segments. Keep continuous polylines (no artificial stitching).
        coords = [tuple(p) for p in xy]
        dl.line(coords, fill=1, width=1)
        ds.line(coords, fill=1, width=13)
    target = np.asarray(lane, dtype=np.float32)
    valid = np.asarray(support, dtype=bool) & visible
    return target[None], valid[None]


def read_scene_metadata(path):
    import pandas as pd
    path = Path(path)
    ts = pd.read_parquet(path / "meta/timestamps.parquet")
    ep = pd.read_parquet(path / "annotation/ego_pose.parquet")
    joined = ts[["timestamp", "frame_id"]].merge(ep, on="timestamp", how="left", validate="one_to_one")
    joined = joined.sort_values("frame_id")
    if joined[["x", "y", "z", "roll", "pitch", "yaw"]].isna().any().any():
        raise ValueError(f"Unmatched or invalid ego pose timestamps: {path}")
    frames = joined["frame_id"].to_numpy(np.int64)
    timestamps = joined["timestamp"].to_numpy(np.float64)
    if len(set(frames)) != len(frames) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Nonmonotonic or repeated timestamps")
    poses = full_pose_matrices(joined[["x", "y", "z"]].to_numpy(), joined[["roll", "pitch", "yaw"]].to_numpy())
    objects = pd.read_parquet(path / "annotation/object.parquet")
    required = {"timestamp", "class", "x[m]", "y[m]", "z[m]", "heading[rad]", "width[m]", "length[m]", "num_points"}
    if not required <= set(objects.columns):
        raise ValueError(f"Unsupported object schema: {set(objects.columns)}")
    objects = objects.merge(ts[["timestamp", "frame_id"]], on="timestamp", how="left", validate="many_to_one")
    if objects["frame_id"].isna().any():
        raise ValueError("Object timestamps not exactly synchronized")
    grouped = {int(k): group.to_dict("records") for k, group in objects.groupby("frame_id")}
    table = pd.read_parquet(path / "annotation/map.parquet")
    lines = normalize_map(table)
    return frames, timestamps, poses, grouped, lines, list(table.columns)


def build_scene(scene, source, rows, cache, lidar2img, min_frame=30, stride=1,
                max_frames=0, proxy_weights=None):
    frames, timestamps, poses, grouped, lines, map_schema = read_scene_metadata(source)
    lookup = {int(frame): i for i, frame in enumerate(frames)}
    selected = [int(r) for r in rows if cache["frame"][r] >= min_frame and cache["frame"][r] % stride == 0]
    if max_frames:
        selected = selected[:max_frames]
    if not selected:
        raise ValueError(f"No selected frames for {scene}")
    visible = camera_visible(lidar2img)
    arrays = {}
    errors, goal_errors = [], []
    for row in selected:
        frame = int(cache["frame"][row])
        now = lookup[frame]
        ids = [lookup[frame - int(offset)] for offset in HISTORY_OFFSETS]
        targets = motion_targets(poses, (timestamps - timestamps[0]) / 1000., now, ids)
        w2e = np.linalg.inv(poses[now])
        # Exact same full-pose convention as ego_cache.fut. Fail rather than switch GT.
        future_ids = [lookup[frame + int(step)] for step in (5, 10, 15, 20, 25, 30)]
        gt_check = transform_points(poses[future_ids, :3, 3], w2e)[:, :2].astype(np.float32)
        err = float(np.max(np.abs(gt_check - cache["fut"][row])))
        goal_check = transform_points(poses[[lookup[frame + 50]], :3, 3], w2e)[0, :2]
        gerr = float(np.max(np.abs(goal_check - cache["goal"][row])))
        if err > 2e-5 or gerr > 2e-5:
            raise ValueError(f"GT coordinate mismatch {scene}/{frame}: fut={err}, goal={gerr}")
        errors.append(err)
        goal_errors.append(gerr)
        targets["occ_target"], targets["occ_valid"] = rasterize_objects(grouped.get(frame, []), w2e, visible)
        targets["lane_target"], targets["lane_valid"] = rasterize_lanes(lines, w2e, visible)
        targets["frame"] = frame
        targets["row"] = row
        targets["timestamp_seconds"] = timestamps[now] / 1000.
        targets["proxy_weight"] = float((proxy_weights or {}).get(row, 1.))
        for k, v in targets.items():
            arrays.setdefault(k, []).append(v)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    report = {"scene": scene, "frames": len(selected), "map_schema": map_schema,
              "coordinate_fut_max_abs": max(errors), "coordinate_goal_max_abs": max(goal_errors),
              "state_valid_fraction": float(arrays["state_valid"].mean()),
              "occ_positive_valid": int((arrays["occ_target"].astype(bool) & arrays["occ_valid"]).sum()),
              "occ_negative_valid": int((~arrays["occ_target"].astype(bool) & arrays["occ_valid"]).sum()),
              "lane_positive_valid": int((arrays["lane_target"].astype(bool) & arrays["lane_valid"]).sum()),
              "lane_negative_valid": int((~arrays["lane_target"].astype(bool) & arrays["lane_valid"]).sum()),
              "time_offsets_min": arrays["time_offsets"].min(0).tolist(),
              "time_offsets_max": arrays["time_offsets"].max(0).tolist(),
              "annotation_contract": ANNOTATION_CONTRACT,
              "sources": {str(Path(source) / p): sha256(Path(source) / p) for p in
                          ("meta/timestamps.parquet", "annotation/ego_pose.parquet", "annotation/object.parquet", "annotation/map.parquet")}}
    return arrays, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="/NHNHOME/data/sukim/adcl")
    p.add_argument("--meta-root", default="/tmp/pm97/data/etri/meta_train")
    p.add_argument("--ego-cache", default="/tmp/pm97/data/etri/ego_cache.npz")
    p.add_argument("--historical-split", default="/tmp/pm97/data/etri/val_clips.npz")
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--split", default="train,tune,val")
    p.add_argument("--output-root", required=True)
    p.add_argument("--calibration", default="/tmp/pm97/data/etri/pkl/etri_train330_goal.pkl")
    p.add_argument("--scenes", default="", help="comma-separated explicit scenes, restricted to selected splits")
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--min-frame", type=int, default=30)
    p.add_argument("--frame-stride", type=int, default=1)
    a = p.parse_args()
    if a.min_frame < 10 or a.frame_stride < 1:
        raise ValueError("min_frame>=10 and frame_stride>=1 required")
    with open(a.split_manifest) as f:
        manifest = json.load(f)
    validate_manifest(manifest)
    scenes = sorted(set(s for key in a.split.split(",") for s in manifest["splits"][key]))
    if a.scenes:
        requested = set(a.scenes.split(","))
        if not requested <= set(scenes):
            raise ValueError("Explicit scenes outside declared split")
        scenes = sorted(requested)
    if a.max_scenes:
        scenes = scenes[:a.max_scenes]
    output = Path(a.output_root)
    output.mkdir(parents=True, exist_ok=True)
    l2i = load_calibration(a.calibration)
    cp = output / "calibration.npz"
    if cp.exists():
        if not np.array_equal(load_calibration(cp), l2i):
            raise ValueError("Existing output calibration differs")
    else:
        np.savez_compressed(cp, lidar2img=l2i)
    with np.load(a.ego_cache, allow_pickle=False) as z:
        cache = {k: z[k] for k in ("scenarios", "scen_idx", "frame", "fut", "goal")}
    with np.load(a.historical_split, allow_pickle=False) as z:
        weights = dict(zip(z["val_idx"].tolist(), z["val_weight"].tolist()))
    names = cache["scenarios"].astype(str)[cache["scen_idx"]]
    split_sha = sha256(a.split_manifest)
    cache_sha = sha256(a.ego_cache)
    contract = {
        "schema_version": 1, "split_manifest_sha256": split_sha,
        "ego_cache_sha256": cache_sha, "calibration_sha256": sha256(a.calibration),
        "grid_shape": list(GRID_SHAPE), "grid_extent": list(GRID_EXTENT),
        "grid_axis_order": "axis0 forward x, axis1 left y; cell centers",
        "history_frame_offsets": HISTORY_OFFSETS.tolist(),
        "state_target_order": ["vx", "vy", "ax", "ay", "yaw_rate", "stop_binary"],
        "state_fit": "causal quadratic fit over <=1.001s, >=6 samples, stop speed<0.2m/s",
        "annotation_contract": ANNOTATION_CONTRACT,
    }
    contract_path = output / "supervision_manifest.json"
    if contract_path.exists():
        with contract_path.open() as f:
            if json.load(f) != contract:
                raise ValueError("Existing supervision provenance/contract differs")
    else:
        with contract_path.open("x") as f:
            json.dump(contract, f, indent=2)
            f.write("\n")
    for scene in scenes:
        dst, report_path = output / f"{scene}.npz", output / f"{scene}.json"
        if dst.exists() or report_path.exists():
            raise FileExistsError(f"Do not overwrite supervision: {dst}")
        rows = np.flatnonzero(names == scene)
        arrays, report = build_scene(scene, Path(a.meta_root) / scene, rows, cache, l2i,
                                     a.min_frame, a.frame_stride, a.max_frames, weights)
        report["split_manifest_sha256"] = split_sha
        report["ego_cache_sha256"] = cache_sha
        np.savez_compressed(dst, **arrays)
        with report_path.open("x") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(json.dumps({k: v for k, v in report.items() if k not in ("sources", "annotation_contract")}), flush=True)


if __name__ == "__main__":
    main()
