#!/usr/bin/env python3
"""Build a train/tune-only, history-only MotionDrive V2 supervision overlay.

The immutable C1 supervision remains the source of plan/state/raster labels.
This tool emits only pose/timestamp-derived history alignment targets.  The
control contract must reproduce C1 bitwise before a wide overlay is allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_grouped_split_v2 import sha256, validate_manifest
from models.motiondrive_v2_temporal_contract import temporal_contract
from motiondrive_v2_data import full_pose_matrices, history_targets

TEMPORAL_KEYS = ("history_transforms", "history_target", "history_valid", "time_offsets")
EXPECTED_SPLITS = {"train": (1, 54810), "tune": (5, 1998)}
EXPECTED_SPLIT_SHA256 = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
EXPECTED_EGO_SHA256 = "d35a69bb229b2d4736aff7dddefd3522ddbd346e112283e2ce58ce3986a023bd"
EXPECTED_C1_SHA256 = "ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_sha(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def rows_sha(rows) -> str:
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def read_pose_timeline(scene_root: Path):
    import pandas as pd
    timestamp_path = scene_root / "meta/timestamps.parquet"
    pose_path = scene_root / "annotation/ego_pose.parquet"
    timestamps = pd.read_parquet(timestamp_path)[["timestamp", "frame_id"]]
    poses = pd.read_parquet(pose_path)
    joined = timestamps.merge(poses, on="timestamp", how="left", validate="one_to_one").sort_values("frame_id")
    columns = ["x", "y", "z", "roll", "pitch", "yaw"]
    require(not joined[columns].isna().any().any(), f"invalid pose join: {scene_root.name}")
    frames = joined["frame_id"].to_numpy(np.int64)
    times = joined["timestamp"].to_numpy(np.float64)
    require(len(np.unique(frames)) == len(frames) and np.all(np.diff(times) > 0),
            f"nonmonotonic pose timeline: {scene_root.name}")
    matrices = full_pose_matrices(joined[["x", "y", "z"]].to_numpy(),
                                  joined[["roll", "pitch", "yaw"]].to_numpy())
    return frames, (times - times[0]) / 1000., matrices, {
        "timestamps": sha256(timestamp_path), "ego_pose": sha256(pose_path)}


def temporal_arrays(cache_frames, selected_rows, timeline_frames, times, poses, offsets):
    lookup = {int(frame): index for index, frame in enumerate(timeline_frames)}
    result = {"row": np.asarray(selected_rows, dtype=np.int64),
              "frame": np.asarray(cache_frames[selected_rows], dtype=np.int64)}
    collected = {key: [] for key in TEMPORAL_KEYS}
    for frame in result["frame"]:
        require(int(frame) in lookup, f"current frame absent from raw timeline: {frame}")
        try:
            past = [lookup[int(frame) - offset] for offset in offsets]
        except KeyError as exc:
            raise ValueError(f"past frame absent from raw timeline: {frame}/{exc.args[0]}") from exc
        values = history_targets(poses, times, lookup[int(frame)], past)
        for key in TEMPORAL_KEYS:
            collected[key].append(values[key])
    result.update({key: np.asarray(value) for key, value in collected.items()})
    return result


def assert_control_parity(arrays, base_path: Path, scene: str):
    with np.load(base_path, allow_pickle=False) as source:
        require("row" in source.files and "frame" in source.files,
                f"base supervision identity missing: {scene}")
        lookup = {int(row): index for index, row in enumerate(source["row"])}
        try:
            indices = np.asarray([lookup[int(row)] for row in arrays["row"]], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"base supervision row missing: {scene}/{exc.args[0]}") from exc
        require(np.array_equal(source["frame"][indices], arrays["frame"]),
                f"base supervision frame mismatch: {scene}")
        for key in TEMPORAL_KEYS:
            require(key in source.files and np.array_equal(source[key][indices], arrays[key]),
                    f"control overlay is not bitwise C1: {scene}/{key}")


def validate_control_barrier(path: Path, expected_sha: str):
    require(len(expected_sha) == 64 and sha256(path) == expected_sha,
            "wide generation requires exact control-overlay manifest SHA")
    manifest = json.loads(path.read_text())
    require(manifest.get("status") == "frozen"
            and manifest.get("temporal_contract", {}).get("name") == "control"
            and manifest.get("boundaries", {}).get("base_temporal_bitwise_equal_all_rows") is True,
            "wide generation requires completed all-row control parity")
    return {"path": str(path.resolve()), "sha256": expected_sha}


def build(args):
    contract = temporal_contract(args.history_contract)
    split_path, ego_path = Path(args.split_manifest).resolve(), Path(args.ego_cache).resolve()
    base_root, meta_root = Path(args.base_supervision_root).resolve(), Path(args.meta_root).resolve()
    output = Path(args.output_root).resolve()
    require(not output.exists(), "history-overlay output must be new")
    split = json.loads(split_path.read_text())
    validate_manifest(split)
    with np.load(ego_path, allow_pickle=False) as source:
        cache = {key: source[key] for key in ("scenarios", "scen_idx", "frame")}
    names = cache["scenarios"].astype(str)[cache["scen_idx"]]
    base_manifest = base_root / "supervision_manifest.json"
    require(base_manifest.exists(), "base supervision manifest missing")
    require(sha256(split_path) == EXPECTED_SPLIT_SHA256
            and sha256(ego_path) == EXPECTED_EGO_SHA256
            and sha256(base_manifest) == EXPECTED_C1_SHA256,
            "P8 history overlay requires the immutable current split/ego/C1 inputs")
    control_barrier = None
    if contract.name == "wide":
        require(args.control_overlay_manifest is not None
                and args.expected_control_overlay_manifest_sha256 is not None,
                "wide generation requires the control parity barrier")
        control_barrier = validate_control_barrier(
            Path(args.control_overlay_manifest).resolve(),
            args.expected_control_overlay_manifest_sha256)
    else:
        require(args.control_overlay_manifest is None
                and args.expected_control_overlay_manifest_sha256 is None,
                "control generation must not consume a control barrier")

    prepared, artifacts, split_receipts = [], {}, {}
    for split_name, (stride, expected_count) in EXPECTED_SPLITS.items():
        allowed = set(split["splits"][split_name])
        selected = np.flatnonzero(np.isin(names, list(allowed))
                                  & (cache["frame"] >= 30)
                                  & (cache["frame"] % stride == 0))
        require(len(selected) == expected_count,
                f"fixed {split_name} selected-row count mismatch")
        split_receipts[split_name] = {"rows": len(selected), "rows_sha256": rows_sha(selected)}
        for scene in sorted(allowed):
            scene_rows = selected[names[selected] == scene]
            require(len(scene_rows) > 0, f"empty selected scene: {scene}")
            timeline = read_pose_timeline(meta_root / scene)
            arrays = temporal_arrays(cache["frame"], scene_rows, *timeline[:3], contract.frame_offsets)
            require(arrays["row"].dtype == np.int64 and arrays["frame"].dtype == np.int64
                    and arrays["history_transforms"].dtype == np.float32
                    and arrays["history_target"].dtype == np.float32
                    and arrays["history_valid"].dtype == np.bool_
                    and arrays["time_offsets"].dtype == np.float32
                    and all(np.isfinite(arrays[key]).all() for key in TEMPORAL_KEYS),
                    f"history-overlay dtype/finite contract mismatch: {scene}")
            if contract.name == "control":
                assert_control_parity(arrays, base_root / f"{scene}.npz", scene)
            source_sha = timeline[3]
            artifact = {"split": split_name, "file": f"{scene}.npz", "sha256": None,
                        "rows": len(scene_rows), "rows_sha256": rows_sha(scene_rows),
                        "arrays_sha256": {key: array_sha(arrays[key]) for key in arrays},
                        "source_sha256": source_sha}
            prepared.append((scene, arrays, artifact))
            artifacts[scene] = artifact

    output.mkdir(parents=True, exist_ok=False)
    for scene, arrays, artifact in prepared:
        path = output / artifact["file"]
        np.savez_compressed(path, **arrays)
        artifact["sha256"] = sha256(path)
    manifest = {
        "schema_version": 1, "status": "frozen",
        "producer": {"script": "scripts/build_motiondrive_v2_history_overlay.py",
                     "script_sha256": sha256(Path(__file__).resolve()),
                     "numpy": np.__version__},
        "temporal_contract": {"name": contract.name,
                              "frame_offsets": list(contract.frame_offsets),
                              "nominal_seconds": list(contract.nominal_seconds)},
        "split_manifest_sha256": sha256(split_path), "ego_cache_sha256": sha256(ego_path),
        "base_supervision_manifest_sha256": sha256(base_manifest),
        "allowed_splits": ["train", "tune"], "splits": split_receipts,
        "artifacts": artifacts,
        "boundaries": {"history_only": True, "final_rows_accessed": False,
                       "state_stop_occ_lane_plan_recomputed": False,
                       "base_temporal_bitwise_equal_all_rows": contract.name == "control",
                       "control_barrier": control_barrier,
                       "shared_source_bytes_may_include_other_splits_but_only_train_tune_rows_indexed": True}}
    manifest_path = output / "overlay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for _, _, artifact in prepared:
        require(sha256(output / artifact["file"]) == artifact["sha256"],
                "history-overlay artifact changed while publishing")
    print(json.dumps({"status": "frozen", "manifest": str(manifest_path),
                      "manifest_sha256": sha256(manifest_path), "rows": split_receipts,
                      "final_rows_accessed": False}, sort_keys=True))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--history-contract", choices=("control", "wide"), required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--ego-cache", required=True)
    parser.add_argument("--base-supervision-root", required=True)
    parser.add_argument("--meta-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--control-overlay-manifest")
    parser.add_argument("--expected-control-overlay-manifest-sha256")
    return parser.parse_args(argv)


def main(argv=None):
    build(arguments(argv))


if __name__ == "__main__":
    main()
