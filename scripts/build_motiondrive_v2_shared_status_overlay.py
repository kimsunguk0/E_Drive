#!/usr/bin/env python3
"""Build the fixed train/tune nominal-causal 5D status overlay for A1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_grouped_split_v2 import sha256, validate_manifest
from motiondrive_v2_data import full_pose_matrices
from motiondrive_v2_shared_status_data import (
    NOMINAL_FRAMES, NOMINAL_TIMES, STATUS_FIELDS, array_sha,
    causal_status5_from_pose_matrices, require,
)

EXPECTED_SPLIT_SHA256 = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
EXPECTED_EGO_SHA256 = "d35a69bb229b2d4736aff7dddefd3522ddbd346e112283e2ce58ce3986a023bd"
EXPECTED_ROWS = {"train": (54810, "75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854"),
                 "tune": (1998, "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88")}


def selected_rows(ego, manifest, split):
    names = ego["scenarios"].astype(str)[ego["scen_idx"]]
    stride = 1 if split == "train" else 5
    mask = np.isin(names, manifest["splits"][split])
    mask &= (ego["frame"] >= 30) & (ego["frame"] % stride == 0)
    return np.flatnonzero(mask).astype(np.int64), names


def scene_poses(source_root: Path, scene: str):
    scene_root = source_root / scene
    pose_path = scene_root / "annotation/ego_pose.parquet"
    time_path = scene_root / "meta/timestamps.parquet"
    pose = pd.read_parquet(pose_path)
    times = pd.read_parquet(time_path)
    require({"timestamp", "x", "y", "z", "roll", "pitch", "yaw"} <= set(pose),
            f"ego pose schema mismatch: {scene}")
    require({"timestamp", "frame_id"} <= set(times), f"timestamp schema mismatch: {scene}")
    joined = times[["timestamp", "frame_id"]].merge(
        pose[["timestamp", "x", "y", "z", "roll", "pitch", "yaw"]],
        on="timestamp", how="left", validate="one_to_one").sort_values("frame_id")
    require(not joined.isna().any().any() and joined["frame_id"].is_unique,
            f"pose/timestamp identity join mismatch: {scene}")
    frames = joined["frame_id"].to_numpy(np.int64)
    matrices = full_pose_matrices(joined[["x", "y", "z"]].to_numpy(),
                                  joined[["roll", "pitch", "yaw"]].to_numpy())
    return {int(frame): matrices[i] for i, frame in enumerate(frames)}, {
        "ego_pose_sha256": sha256(pose_path), "timestamps_sha256": sha256(time_path)}


def build(args):
    split_path, ego_path = Path(args.split_manifest).resolve(), Path(args.ego_cache).resolve()
    source_root, output = Path(args.source_root).resolve(), Path(args.output_root).resolve()
    require(sha256(split_path) == EXPECTED_SPLIT_SHA256, "grouped split SHA mismatch")
    require(sha256(ego_path) == EXPECTED_EGO_SHA256, "ego cache SHA mismatch")
    require(not output.exists(), "refusing to overwrite shared-status overlay")
    manifest = json.loads(split_path.read_text())
    validate_manifest(manifest)
    require(set(manifest["splits"]) >= {"train", "tune"}, "train/tune splits required")
    with np.load(ego_path, allow_pickle=False) as src:
        ego = {key: src[key] for key in ("scenarios", "scen_idx", "frame")}
    artifacts, arrays, source_files = {}, {}, {}
    for split in ("train", "tune"):
        rows, names = selected_rows(ego, manifest, split)
        expected_n, expected_sha = EXPECTED_ROWS[split]
        require((len(rows), array_sha(rows)) == (expected_n, expected_sha),
                f"{split} row population mismatch")
        status = np.empty((len(rows), 5), np.float32)
        frames = np.asarray(ego["frame"][rows], np.int64)
        for scene in sorted(set(names[rows])):
            poses, identities = scene_poses(source_root, scene)
            source_files[scene] = identities
            indices = np.flatnonzero(names[rows] == scene)
            for out_index in indices:
                frame = int(frames[out_index])
                required_frames = [frame + offset for offset in NOMINAL_FRAMES]
                require(all(value in poses for value in required_frames),
                        f"missing causal pose: {scene}/{frame}")
                status[out_index] = causal_status5_from_pose_matrices(
                    np.stack([poses[value] for value in required_frames]))
        arrays[split] = {"row": rows, "frame": frames, "status5": status}
    output.mkdir(parents=True, exist_ok=False)
    for split, data in arrays.items():
        path = output / f"{split}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **data)
        artifacts[split] = {"file": path.name, "sha256": sha256(path), "rows": len(data["row"]),
                            "rows_sha256": array_sha(data["row"]),
                            "status_sha256": array_sha(data["status5"]), "valid_rows": len(data["row"])}
    source_files = {scene: source_files[scene] for scene in sorted(source_files)}
    source_files_sha256 = hashlib.sha256(json.dumps(
        source_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload = {
        "schema_version": 1, "status": "frozen", "name": "shared_status_a1",
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256, "ego_cache_sha256": EXPECTED_EGO_SHA256,
        "source_root": str(source_root),
        "source_files": source_files, "source_files_sha256": source_files_sha256,
        "causal_contract": {"fields": list(STATUS_FIELDS), "frames": list(NOMINAL_FRAMES),
                            "seconds": NOMINAL_TIMES.tolist(),
                            "method": "motion_targets causal quadratic fit; nominal 10Hz",
                            "future_values_used": False,
                            "ego_pose_file_all_rows_read_for_identity_join": True,
                            "goal_command_hd_files_read": False},
        "splits": {key: {"rows": value[0], "rows_sha256": value[1]}
                   for key, value in EXPECTED_ROWS.items()},
        "artifacts": artifacts,
    }
    (output / "overlay_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    # Source bytes and values must remain stable through publication.
    for scene, ids in source_files.items():
        root = source_root / scene
        require(sha256(root / "annotation/ego_pose.parquet") == ids["ego_pose_sha256"]
                and sha256(root / "meta/timestamps.parquet") == ids["timestamps_sha256"],
                f"source pose metadata changed: {scene}")
    print(json.dumps({"status": "completed", "output_root": str(output),
                      "overlay_manifest_sha256": sha256(output / "overlay_manifest.json"),
                      "rows": {k: len(v["row"]) for k, v in arrays.items()},
                      "source_scene_count": len(source_files),
                      "source_files_sha256": source_files_sha256}, sort_keys=True))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--ego-cache", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(arguments())
