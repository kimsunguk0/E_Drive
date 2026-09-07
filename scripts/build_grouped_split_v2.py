#!/usr/bin/env python3
"""Freeze timestamp-contiguous session splits without inspecting any test data.

The gap is between a main clip's raw frame299+median-dt END and the next
clip's raw frame0 START. Filename times are never parsed in production. Groups
touching the repeatedly used historical val38 are quarantined in ``val``;
``historical_val`` is its overlapping reporting-only subset, never a train split.
This is temporal grouping, NOT a claim of route-disjoint or untouched validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def group_scenes(scenes, gap_seconds=60.0, clip_seconds=30.0):
    """Legacy/synthetic helper only; production grouping uses raw timestamps."""
    if gap_seconds < 0 or clip_seconds <= 0:
        raise ValueError("Invalid gap or clip duration")
    groups, previous = [], None
    for scene in sorted(set(map(str, scenes))):
        now = datetime.strptime(scene, "%Y%m%d-%H%M%S")
        if previous is None or (now - previous).total_seconds() > clip_seconds + gap_seconds:
            groups.append([])
        groups[-1].append(scene)
        previous = now
    return groups


def read_raw_scene_bounds(scenes, meta_root):
    """Use actual frame0 start and frame299 + median-dt end, never scene names."""
    import pyarrow.parquet as pq
    result = {}
    for scene in scenes:
        source = Path(meta_root) / str(scene) / "meta/timestamps.parquet"
        table = pq.read_table(source, columns=["frame_id", "timestamp"])
        frames = np.asarray(table["frame_id"].to_pylist(), np.int64)
        times = np.asarray(table["timestamp"].to_pylist(), np.float64) / 1000.
        order = np.argsort(frames)
        frames, times = frames[order], times[order]
        if len(np.unique(frames)) != len(frames) or np.any(np.diff(times) <= 0):
            raise ValueError(f"Invalid raw scene timestamps: {scene}")
        if np.count_nonzero(frames == 0) != 1 or np.count_nonzero(frames == 299) != 1:
            raise ValueError(f"Main clip boundaries missing: {scene}")
        dt = float(np.median(np.diff(times)))
        result[str(scene)] = {
            "start_seconds": float(times[frames == 0][0]),
            "end_seconds": float(times[frames == 299][0] + dt),
            "median_dt_seconds": dt, "source": str(source.resolve()),
            "source_sha256": sha256(source),
        }
    return result


def group_raw_scene_bounds(bounds, gap_seconds=60.):
    if gap_seconds < 0:
        raise ValueError("Invalid gap")
    groups, previous_end = [], None
    for scene, interval in sorted(bounds.items(), key=lambda item: (item[1]["start_seconds"], item[0])):
        start, end = interval["start_seconds"], interval["end_seconds"]
        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError(f"Invalid interval: {scene}")
        if previous_end is None or start - previous_end > gap_seconds:
            groups.append([])
            previous_end = end
        groups[-1].append(scene)
        previous_end = max(previous_end, end)
    return groups


def rebind_raw_sessions(previous, bounds, gap_seconds=60.):
    """Preserve preregistered membership; fail if a true raw session crosses it."""
    validate_manifest(previous)
    if set(bounds) != set(previous["scene_to_session"]):
        raise ValueError("Raw bounds and previous manifest scenes differ")
    groups = group_raw_scene_bounds(bounds, gap_seconds)
    membership = {s: split for split in ("train", "tune", "val") for s in previous["splits"][split]}
    for group in groups:
        splits = {membership[s] for s in group}
        if len(splits) != 1:
            raise ValueError(f"RAW_TIMESTAMP_SESSION_CROSSES_PRIMARY_SPLITS: {group}: {sorted(splits)}")
    manifest = json.loads(json.dumps(previous))
    manifest.update(schema_version=2, gap_seconds=gap_seconds,
                    gap_definition="next raw frame0 timestamp - maximum preceding raw(frame299 timestamp + median frame dt)",
                    grouping_source="raw metadata timestamps, sorted numerically; never parsed filenames",
                    membership_policy="prior primary split scene membership preserved exactly; no performance-based reshuffle",
                    scene_time_bounds=bounds)
    manifest["scene_to_session"] = {s: f"session_{i:03d}_{g[0]}" for i, g in enumerate(groups) for s in g}
    manifest["sessions"] = {f"session_{i:03d}_{g[0]}": g for i, g in enumerate(groups)}
    manifest["counts"] = {k: {"scenes": len(v), "sessions": len({manifest["scene_to_session"][s] for s in v})}
                          for k, v in manifest["splits"].items()}
    validate_manifest(manifest)
    return manifest


def make_manifest(scenes, historical_val=(), gap_seconds=60., seed=20260907,
                  tune_fraction=.15):
    if not 0 < tune_fraction < .5:
        raise ValueError("tune_fraction must be between 0 and .5")
    groups = group_scenes(scenes, gap_seconds)
    old = set(map(str, historical_val))
    unknown = old - set(map(str, scenes))
    if unknown:
        raise ValueError(f"Historical scenes absent from cache: {unknown}")
    val_groups = [i for i, group in enumerate(groups) if old.intersection(group)]
    pool = [i for i in range(len(groups)) if i not in val_groups]
    if len(pool) < 3:
        raise ValueError("Too few non-quarantined sessions")
    rng = np.random.default_rng(seed)
    order = rng.permutation(pool).tolist()
    if not val_groups:
        val_groups = order[:max(1, round(len(order) * .15))]
        order = order[len(val_groups):]
    target = sum(len(groups[i]) for i in order) * tune_fraction
    tune_groups, n = [], 0
    for i in order:
        if n < target and len(tune_groups) < len(order) - 1:
            tune_groups.append(i)
            n += len(groups[i])
    train_groups = [i for i in order if i not in tune_groups]
    selected = {"train": train_groups, "tune": tune_groups, "val": val_groups}
    splits = {k: sorted(s for i in ids for s in groups[i]) for k, ids in selected.items()}
    splits["historical_val"] = sorted(old)
    return {
        "schema_version": 1, "seed": seed, "gap_seconds": gap_seconds,
        "gap_definition": "next_start - (previous_start + 30 seconds)",
        "clip_seconds": 30, "tune_fraction": tune_fraction,
        "protocol": "quarantine entire sessions touching historical val38; random session holdout of remaining train-only pool",
        "caveats": ["Temporal groups do not prove spatial-route disjointness.",
                    "Historical labels were previously inspected; no split is advertised as untouched.",
                    "All ETRI weights must be retrained without tune/val groups; public pretrained weights only."],
        "splits": splits,
        "scene_to_session": {s: f"session_{i:03d}_{g[0]}" for i, g in enumerate(groups) for s in g},
        "sessions": {f"session_{i:03d}_{g[0]}": g for i, g in enumerate(groups)},
        "counts": {k: {"scenes": len(v), "sessions": len({i for i, g in enumerate(groups) if set(g).intersection(v)})} for k, v in splits.items()},
    }


def validate_manifest(manifest):
    splits = manifest["splits"]
    primary = [set(splits[k]) for k in ("train", "tune", "val")]
    for i in range(3):
        for j in range(i):
            if primary[i] & primary[j]:
                raise ValueError("Overlapping primary scene splits")
    mapping = manifest["scene_to_session"]
    sessions = [{mapping[s] for s in group} for group in primary]
    for i in range(3):
        for j in range(i):
            if sessions[i] & sessions[j]:
                raise ValueError("Session leakage between primary splits")
    if not set(splits.get("historical_val", [])) <= primary[2]:
        raise ValueError("Historical validation is not quarantined")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ego-cache", default="/tmp/pm97/data/etri/ego_cache.npz")
    p.add_argument("--historical-split", default="/tmp/pm97/data/etri/val_clips.npz")
    p.add_argument("--output", required=True)
    p.add_argument("--gap-seconds", type=float, default=60.)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--tune-fraction", type=float, default=.15)
    p.add_argument("--meta-root", default="/tmp/pm97/data/etri/meta_train")
    p.add_argument("--preserve-membership", help="Existing manifest; update only raw-time session grouping, fail on leakage")
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(f"Immutable split already exists: {out}")
    with np.load(a.ego_cache, allow_pickle=False) as z:
        scenes = z["scenarios"].astype(str)
    with np.load(a.historical_split, allow_pickle=False) as z:
        historical = z["holdout"].astype(str)
    bounds = read_raw_scene_bounds(scenes, a.meta_root)
    if a.preserve_membership:
        with open(a.preserve_membership) as f:
            previous = json.load(f)
        manifest = rebind_raw_sessions(previous, bounds, a.gap_seconds)
        manifest["prior_manifest"] = {"path": str(Path(a.preserve_membership).resolve()),
                                      "sha256": sha256(a.preserve_membership)}
    else:
        # Fresh sampling operates directly on actual raw-time groups.
        groups = group_raw_scene_bounds(bounds, a.gap_seconds)
        old = set(historical)
        val_groups = [i for i, g in enumerate(groups) if old.intersection(g)]
        pool = [i for i in range(len(groups)) if i not in val_groups]
        if len(pool) < 3:
            raise ValueError("Too few train-only raw sessions")
        order = np.random.default_rng(a.seed).permutation(pool).tolist()
        target = sum(len(groups[i]) for i in order) * a.tune_fraction
        tune_groups, n = [], 0
        for i in order:
            if n < target and len(tune_groups) < len(order) - 1:
                tune_groups.append(i)
                n += len(groups[i])
        selected = {"train": [i for i in order if i not in tune_groups], "tune": tune_groups, "val": val_groups}
        manifest = {"schema_version": 2, "seed": a.seed, "gap_seconds": a.gap_seconds,
                    "tune_fraction": a.tune_fraction, "splits": {k: sorted(s for i in ids for s in groups[i]) for k, ids in selected.items()},
                    "scene_to_session": {s: f"session_{i:03d}_{g[0]}" for i, g in enumerate(groups) for s in g},
                    "sessions": {f"session_{i:03d}_{g[0]}": g for i, g in enumerate(groups)},
                    "caveats": ["Temporal grouping does not prove route disjointness or untouched validation."]}
        manifest["splits"]["historical_val"] = sorted(old)
        manifest = rebind_raw_sessions(manifest, bounds, a.gap_seconds)
        manifest["membership_policy"] = "deterministic seed-based actual-session sampling before training"
    manifest["sources"] = {str(Path(x).resolve()): sha256(x) for x in (a.ego_cache, a.historical_split)}
    validate_manifest(manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
