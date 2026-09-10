#!/usr/bin/env python3
"""Metadata-only, CPU split audit and immutable public-init experiment manifests.

No target trajectory, goal, status, image content, or model prediction is read.
The original 203/37/136 split is never changed. Additional session crossfit and
12-session confirmation manifests are optional independent experiment contracts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pyarrow.parquet as pq


EXPECTED_SPLIT = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
EXPECTED_EGO = "d35a69bb229b2d4736aff7dddefd3522ddbd346e112283e2ce58ce3986a023bd"
EXPECTED_TRAIN_ROWS = "75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854"
EXPECTED_TUNE_ROWS = "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest()


def publish(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == content, f"Refusing to replace different frozen file: {path}")
    else:
        with path.open("xb") as f:
            f.write(content)
    return {"path": str(path.resolve()), "sha256": file_sha(path), "bytes": len(content)}


def publish_json(path, value):
    return publish(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def publish_rows(path, rows):
    stream = io.BytesIO()
    np.save(stream, np.asarray(rows, dtype="<i8"), allow_pickle=False)
    return {**publish(path, stream.getvalue()), "row_array_sha256": array_sha(rows), "rows": len(rows)}


def row_identity_sha(rows, names, frames):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(names[row].encode())
        digest.update(b"\0")
        digest.update(np.asarray([row, frames[row]], dtype="<i8").tobytes())
    return digest.hexdigest()


def validate(manifest):
    primary = [set(manifest["splits"][name]) for name in ("train", "tune", "val")]
    mapping = manifest["scene_to_session"]
    for i, first in enumerate(primary):
        require(len(first) == len(manifest["splits"][("train", "tune", "val")[i]]), "Repeated scene")
        for second in primary[:i]:
            require(not first.intersection(second), "Primary scene leakage")
            require(not {mapping[s] for s in first}.intersection({mapping[s] for s in second}), "Session leakage")
    require(set(manifest["splits"].get("historical_val", [])) <= primary[2], "Historical val not quarantined")
    require(set.union(*primary) == set(mapping), "Primary split coverage mismatch")


def derive(primary, splits, purpose, parent_path):
    result = json.loads(json.dumps(primary))
    result["splits"] = {key: sorted(value) for key, value in splits.items()}
    result["counts"] = {
        key: {"scenes": len(value), "sessions": len({primary["scene_to_session"][s] for s in value})}
        for key, value in result["splits"].items()
    }
    result["protocol"] = purpose
    result["membership_policy"] = "Original primary preserved separately; this is an explicitly named derived experiment"
    result["parent_manifest"] = {"path": str(parent_path.resolve()), "sha256": file_sha(parent_path)}
    result["initialization_contract"] = {
        "public_pretrained_only_or_current_fit_split_only": True,
        "every_ancestor_fit_scene_must_be_subset_of_train": True,
        "bank_scalers_normalizers_teachers_fit_split": "train only",
        "external_test_data_used": False,
        "evaluation_labels_must_not_be_fit_inputs": True,
        "reusing_fulltrain_or_primary_train203_checkpoint_for_crossfit": False,
    }
    validate(result)
    return result


def interval_audit(manifest, bounds, start_key, end_key):
    result = {}
    for left, right in (("train", "tune"), ("train", "val"), ("tune", "val")):
        overlaps, minimum, nearest = [], float("inf"), None
        for first in manifest["splits"][left]:
            for second in manifest["splits"][right]:
                a, b = bounds[first], bounds[second]
                gap = max(a[start_key], b[start_key]) - min(a[end_key], b[end_key])
                if gap < minimum:
                    minimum, nearest = gap, [first, second]
                if gap <= 0:
                    overlaps.append([first, second, gap])
        require(not overlaps, f"{left}/{right} temporal-support overlap: {overlaps[:3]}")
        result[f"{left}_vs_{right}"] = {
            "overlap_pairs": len(overlaps), "minimum_gap_seconds": minimum,
            "nearest_pair": nearest,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/NHNHOME/data/sukim/adcl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    split_path = base / "data/etri/motiondrive_v2/grouped_split_rawtime.json"
    ego_path = Path("/tmp/pm97/data/etri/ego_cache.npz").resolve()
    holdout_path = base / "reports/motiondrive_v2_longrun_holdout_split_20260909_ops.json"
    require(file_sha(split_path) == EXPECTED_SPLIT, "Canonical split changed")
    require(file_sha(ego_path) == EXPECTED_EGO, "Canonical ego identity source changed")
    primary = json.loads(split_path.read_text())
    validate(primary)
    with np.load(ego_path, allow_pickle=False) as z:
        # Select only identifiers. Do not load fut, goal, history, or status arrays.
        scenes = z["scenarios"].astype(str)
        scene_indices = np.asarray(z["scen_idx"], dtype=np.int64)
        frames = np.asarray(z["frame"], dtype=np.int64)
    names = scenes[scene_indices]
    require(len(scenes) == 376 and len(names) == 112800, "Unexpected raw cache inventory")
    require(set(scenes) == set(primary["scene_to_session"]), "Split does not cover cache scenes")
    require(np.array_equal(np.unique(frames), np.arange(300)), "Unexpected anchor frame IDs")
    require(all(np.array_equal(np.sort(frames[names == s]), np.arange(300)) for s in scenes), "Per-scene row mismatch")

    bounds, dates, metadata_receipts = {}, Counter(), {}
    unusual_dt = []
    for scene in sorted(scenes):
        path = base / "data/etri/meta_train" / scene / "meta/timestamps.parquet"
        table = pq.read_table(path, columns=["frame_id", "timestamp"], use_threads=False)
        ids = np.asarray(table["frame_id"].to_pylist(), dtype=np.int64)
        ts = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64) / 1000.
        order = np.argsort(ids)
        ids, ts = ids[order], ts[order]
        require(len(ids) == len(np.unique(ids)) and (np.diff(ts) > 0).all(), f"Invalid timestamps: {scene}")
        lookup = dict(zip(ids.tolist(), ts.tolist()))
        require(set(range(350)) <= set(lookup), f"Missing full history/5s target metadata: {scene}")
        dt = float(np.median(np.diff(ts)))
        legacy = primary["scene_time_bounds"][scene]
        require(file_sha(path) == legacy["source_sha256"], f"Raw timestamps changed: {scene}")
        require(abs(lookup[0] - legacy["start_seconds"]) < 1e-6, "Raw start mismatch")
        require(abs(lookup[299] + dt - legacy["end_seconds"]) < 1e-6, "Raw end mismatch")
        # 30..299 anchors; 30-frame max history stays in 0..299; goal +50 reaches349.
        max_h3 = max(lookup[f] - lookup[f - 30] for f in range(30, 300))
        min_h3 = min(lookup[f] - lookup[f - 30] for f in range(30, 300))
        bounds[scene] = {
            "main_start": lookup[0], "main_end": lookup[299] + dt,
            "used_support_start": lookup[0], "used_support_end": lookup[349],
            "raw_available_start": float(ts[0]), "raw_available_end": float(ts[-1]),
            "available_min_frame": int(ids[0]), "available_max_frame": int(ids[-1]),
            "median_dt_seconds": dt, "history30_actual_min_seconds": min_h3,
            "history30_actual_max_seconds": max_h3,
            "five_second_label_max_actual_seconds": max(lookup[f + 50] - lookup[f] for f in range(30, 300)),
            "temporal_inputs_boundary_note": "Use nominal provided frame offsets; actual >3.001s at -30 must be separately masked if enforcing wall-clock 3s",
        }
        dates[datetime.fromtimestamp(lookup[0], timezone.utc).strftime("%Y-%m-%d")] += 1
        metadata_receipts[scene] = {"path": str(path), "sha256": file_sha(path)}
        if abs(dt - .1) > .001:
            unusual_dt.append({"scene": scene, "split": next(k for k in ("train", "tune", "val") if scene in primary["splits"][k]), "median_dt_seconds": dt})

    group_crossings = []
    memberships = {s: key for key in ("train", "tune", "val") for s in primary["splits"][key]}
    recomputed, previous_end = [], None
    for scene in sorted(bounds, key=lambda s: (bounds[s]["main_start"], s)):
        current = bounds[scene]
        if previous_end is None or current["main_start"] - previous_end > 60.:
            recomputed.append([])
        recomputed[-1].append(scene)
        previous_end = max(previous_end or current["main_end"], current["main_end"])
    require({frozenset(g) for g in recomputed} == {frozenset(g) for g in primary["sessions"].values()}, "Raw session reconstruction mismatch")
    for group in recomputed:
        if len({memberships[s] for s in group}) != 1:
            group_crossings.append(group)
    require(not group_crossings, "Raw session crosses primary split")
    primary_time = interval_audit(primary, bounds, "used_support_start", "used_support_end")
    conservative_time = interval_audit(primary, bounds, "raw_available_start", "raw_available_end")

    # Canonical byte copy preserves old supervision SHA contracts exactly.
    manifest_records = {"primary": publish(output / "primary_manifest.json", split_path.read_bytes())}
    manifests = {"primary": primary}
    train_scenes = set(primary["splits"]["train"])
    train_sessions = sorted({primary["scene_to_session"][s] for s in train_scenes})
    # Deterministic balancing only on scene counts/session names; never outcome labels.
    buckets, scene_load = [[] for _ in range(3)], [0, 0, 0]
    for session in sorted(train_sessions, key=lambda session: (-len(primary["sessions"][session]), session)):
        fold = min(range(3), key=lambda i: (scene_load[i], len(buckets[i]), i))
        buckets[fold].append(session)
        scene_load[fold] += len(primary["sessions"][session])
    require(set().union(*(set(x) for x in buckets)) == set(train_sessions), "Crossfit assignment incomplete")
    for fold, held_sessions in enumerate(buckets):
        held = {s for session in held_sessions for s in primary["sessions"][session]}
        splits = {
            "train": train_scenes - held, "tune": held,
            "val": set(primary["splits"]["tune"]) | set(primary["splits"]["val"]),
            "historical_val": primary["splits"]["historical_val"],
            "screen_tune37": primary["splits"]["tune"],
            "reserve136": primary["splits"]["val"],
        }
        name = f"crossfit_fold{fold}"
        manifests[name] = derive(primary, splits, "Inner 3-fold session crossfit within original train203; tune=out-of-fold rows; original tune37/reserve136 never fit", split_path)
        manifests[name]["crossfit_assignment"] = {"fold": fold, "folds": 3, "rule": "largest-session scene count first; lowest total scene count, session count, fold index tie-break", "held_sessions": sorted(held_sessions)}
        manifest_records[name] = publish_json(output / f"{name}_manifest.json", manifests[name])

    # Preserve the previously proposed 12-session holdout, but start public-only.
    holdout = json.loads(holdout_path.read_text())
    held12 = set(holdout["holdout_scenes"])
    require(held12 <= train_scenes and len(held12) == 32, "12-session confirmation membership changed")
    require({primary["scene_to_session"][s] for s in held12} == set(holdout["holdout_sessions"]), "Confirmation session IDs mismatch")
    require(all(set(primary["sessions"][sid]) <= held12 for sid in holdout["holdout_sessions"]), "Confirmation cuts a raw session")
    confirmation = derive(primary, {
        "train": train_scenes - held12, "tune": primary["splits"]["tune"],
        "val": set(primary["splits"]["val"]) | held12,
        "confirmation12": held12, "reserve136": primary["splits"]["val"],
        "historical_val": primary["splits"]["historical_val"],
    }, "Independent public-init confirmation: train171; unchanged tune37; 32 scenes/12 sessions excluded from every learned ancestor and bank", split_path)
    confirmation["prior_holdout_membership_source"] = {"path": str(holdout_path), "sha256": file_sha(holdout_path)}
    manifests["confirmation12"] = confirmation
    manifest_records["confirmation12"] = publish_json(output / "confirmation12_manifest.json", confirmation)

    row_contracts, temporal_contracts = {}, {}
    for experiment, manifest in manifests.items():
        contracts = {}
        temporal_contracts[experiment] = interval_audit(manifest, bounds, "used_support_start", "used_support_end")
        for split, chosen in manifest["splits"].items():
            stride = 1 if split == "train" else 5
            rows = np.flatnonzero(np.isin(names, chosen) & (frames >= 30) & (frames % stride == 0))
            contracts[split] = {
                **publish_rows(output / "rows" / f"{experiment}_{split}.npy", rows),
                "frame_stride": stride, "min_frame": 30, "max_frame": 299,
                "scenes": len(chosen), "sessions": len({manifest["scene_to_session"][s] for s in chosen}),
                "metadata_identity_sha256": row_identity_sha(rows, names, frames),
                "metadata_identity_encoding": "repeated scene_utf8,NUL,row_<i8,frame_<i8; no labels",
                "row_order": "ascending original ego_cache row index", "row_index_source": str(ego_path),
            }
        row_contracts[experiment] = contracts
    require(row_contracts["primary"]["train"]["row_array_sha256"] == EXPECTED_TRAIN_ROWS, "Primary train row SHA changed")
    require(row_contracts["primary"]["tune"]["row_array_sha256"] == EXPECTED_TUNE_ROWS, "Primary tune row SHA changed")
    report = {
        "schema_version": 1, "status": "PASS_KEEP_PRIMARY_MEMBERSHIP",
        "inputs": {"canonical_split": {"path": str(split_path), "sha256": file_sha(split_path)}, "ego_identity_cache": {"path": str(ego_path), "sha256": file_sha(ego_path)}, "script": {"path": str(Path(__file__).resolve()), "sha256": file_sha(__file__)}},
        "boundaries": {"gpu_used": False, "test_data_opened": False, "trajectory_goal_status_labels_opened": False, "prediction_metrics_opened": False, "original_files_changed": False},
        "inventory": {"scenes": len(scenes), "raw_sessions": len(recomputed), "cache_rows": len(names), "utc_dates": dict(sorted(dates.items())), "primary_counts": primary["counts"]},
        "support_contract": {"main_anchor_frames": [30, 299], "maximum_past_frame_offset": 30, "official_future_frame_offsets": [5, 10, 15, 20, 25, 30], "goal_future_frame_offset": 50, "combined_required_metadata_frames": [0, 349], "timestamps_unit_in_parquet": "milliseconds since epoch; audit converts to seconds", "available_raw_support_audited_separately": True},
        "primary_used_temporal_support": primary_time,
        "primary_conservative_entire_raw_available_support": conservative_time,
        "all_experiment_temporal_support": temporal_contracts,
        "raw_session_crossings": group_crossings, "unusual_frame_dt": unusual_dt,
        "scene_bounds": bounds, "timestamp_source_receipts": metadata_receipts,
        "manifests": manifest_records, "row_contracts": row_contracts,
        "recommendation": {
            "primary_screen": "Use unchanged train203/72 sessions and tune37/11 sessions from public initialization; row SHA remains directly comparable.",
            "reserve": "Keep all original val136/31 sessions outside screen fitting and metric peeks. It is a fixed confirmation reserve, not historically unseen data or proven route-disjoint data.",
            "crossfit": "Optional 3 folds within train203 only. Fit each encoder/bank/normalizer/teacher adaptation on that fold train. tune is OOF generation; original tune37 and val136 stay excluded. For genuine OOF predictions use fixed training budget or separate inner stopping, not OOF labels for early stopping.",
            "confirmation12": "For the already named 12-session/32-scene confirmation, use train171 fresh public-only ancestry. Original train203 checkpoint or its banks cannot initialize it. Existing tune37 remains model-selection split.",
            "supervision_loader": "Primary byte-copy keeps canonical split SHA. Derived manifests have new SHA and need their own supervision provenance envelope/loader whitelist; never overwrite old supervision or weaken SHA checks.",
            "temporal_caveat": "Some datasets have .10624s actual cadence. Do not silently resample official future offsets. For max-history -30, enforce chosen nominal-vs-actual input policy explicitly.",
        },
    }
    receipt = publish_json(output / "audit.json", report)
    print(json.dumps({"status": report["status"], "audit": receipt, "primary_counts": primary["counts"], "primary_temporal_support": primary_time, "manifest_paths": manifest_records, "primary_row_sha": {key: row_contracts["primary"][key]["row_array_sha256"] for key in ("train", "tune", "val")}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
