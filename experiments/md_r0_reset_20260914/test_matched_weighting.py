#!/usr/bin/env python3
"""Rebuild the test-matched importance weighting and apply it to V0.

The first submission (2026-09-01) established a val number within -0.3% of the
leaderboard. That number was NOT a plain held-out mean: it was an importance-
weighted average, matching the held-out clips to the joint distribution of the
TEST SET'S PUBLIC INPUTS (vad_cmd x current speed bin x goal distance bin). The
split script calls it "the leaderboard predictor" in so many words.

So the transferable asset is the WEIGHTING, not the particular 38 scenes that
carried it -- 29 of those 38 are in our training set now. This rebuilds the
weighting from the original constants and applies it to the V0 rows we already
score, which costs no training data at all.

Only the test set's PUBLIC INPUTS are used: the provided command one-hot and the
ego poses shipped with each test clip. No test label or ground-truth future is
read, and none exists on our side.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT / "experiments/md_r0_reset_20260914"))
from analyze_mr import W as D3_W, load, seg

OUT = ROOT / "reports/md_exp_diagnosis_20260915"
TEST_ROOT = Path("/tmp/etri_test")
TRAIN_META = Path("/tmp/pm97/data/etri/meta_train")
WORK = ROOT / "work_dirs/md_r0_reset_20260914"

# Constants copied from the original scripts/etri_ego_cache.py and etri_split.py
GOAL_FRAME, DT, WEIGHT_CAP = 50, 0.1, 20.0
SPEED_BINS = [0.0, 0.5, 2.0, 5.0, 10.0, 15.0, np.inf]
GOAL_BINS = [0.0, 1.0, 10.0, 30.0, 50.0, 70.0, np.inf]
LATERAL_THRESHOLD = 2.0          # converter rule: |final lateral| >= 2 m turns

RUNS = {"LEN-s0": "E1-EXP-LEN-s0", "MR-LOWDETAIL-s0": "MR-LOWDETAIL-s0",
        "MR-NATIVE-s0": "MR-NATIVE-s0", "MR-NATIVE-s1": "MR-NATIVE-s1",
        "MR-W64-s0": "MR-W64-s0"}


def rot(rpy):
    return Rotation.from_euler("xyz", rpy).as_matrix()


def binof(x, edges):
    return int(np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2))


def col(table, name):
    return np.asarray(table.column(name).to_pylist())


def pose_arrays(table, frame_column):
    frames = col(table, frame_column).astype(int)
    order = np.argsort(frames)
    xyz = np.stack([col(table, k)[order] for k in "xyz"], 1).astype(np.float64)
    rpy = np.stack([col(table, k)[order] for k in ("roll", "pitch", "yaw")],
                   1).astype(np.float64)
    return {f: i for i, f in enumerate(frames[order])}, xyz, rpy


def ego_quantities(index, xyz, rpy, current_frame):
    """Goal offset and current speed, exactly as the original cache computed them."""
    i0 = index.get(current_frame)
    previous = index.get(current_frame - 1)
    goal_i = index.get(current_frame + GOAL_FRAME)
    if i0 is None or previous is None or goal_i is None:
        return None
    R = rot(rpy[i0])
    goal = ((xyz[goal_i] - xyz[i0]) @ R)[:2]
    velocity = ((xyz[i0] - xyz[previous]) / DT @ R)[:2]
    return float(np.linalg.norm(goal)), float(np.linalg.norm(velocity))


def test_cells():
    cells, skipped = [], 0
    for clip in sorted(p for p in TEST_ROOT.iterdir() if p.is_dir()):
        index, xyz, rpy = pose_arrays(pq.read_table(clip / "ego_pose.parquet"), "frame")
        quantities = ego_quantities(index, xyz, rpy, 0)
        if quantities is None:
            skipped += 1
            continue
        goal_distance, speed = quantities
        command = pq.read_table(clip / "command.parquet")
        # The provided one-hot is trusted as shipped; there is no future to
        # recompute it from, and the organisers built it from their own hidden
        # future with the same rule the converter uses on train.
        vad_cmd = int(np.argmax(list(command.column("vad_cmd").to_pylist()[0])))
        cells.append((vad_cmd, binof(speed, SPEED_BINS), binof(goal_distance, GOAL_BINS)))
    return cells, skipped


def val_cells(keys, gt):
    """Same triple for the V0 rows. vad_cmd comes from OUR future, as on train."""
    scenes = {}
    cells, valid = [], []
    for (session, scenario, frame), future in zip(keys, gt):
        scene = scenario
        if scene not in scenes:
            path = TRAIN_META / scene / "annotation/ego_pose.parquet"
            stamps = TRAIN_META / scene / "meta/timestamps.parquet"
            if not (path.exists() and stamps.exists()):
                scenes[scene] = None
            else:
                # Train poses carry a timestamp, not a frame; the frame id comes
                # from meta/timestamps.parquet, exactly as the original cache did.
                clock = pq.read_table(stamps)
                to_frame = dict(zip(col(clock, "timestamp"),
                                    col(clock, "frame_id").astype(int)))
                table = pq.read_table(path)
                frames = np.asarray([to_frame[t] for t in col(table, "timestamp")])
                order = np.argsort(frames)
                xyz = np.stack([col(table, k)[order] for k in "xyz"], 1).astype(np.float64)
                rpy = np.stack([col(table, k)[order] for k in
                                ("roll", "pitch", "yaw")], 1).astype(np.float64)
                scenes[scene] = ({f: i for i, f in enumerate(frames[order])}, xyz, rpy)
        entry = scenes[scene]
        if entry is None:
            cells.append(None); valid.append(False); continue
        quantities = ego_quantities(*entry, frame)
        if quantities is None:
            cells.append(None); valid.append(False); continue
        goal_distance, speed = quantities
        lateral = float(future[-1, 1])
        vad_cmd = 0 if lateral <= -LATERAL_THRESHOLD else (
            1 if lateral >= LATERAL_THRESHOLD else 2)
        cells.append((vad_cmd, binof(speed, SPEED_BINS), binof(goal_distance, GOAL_BINS)))
        valid.append(True)
    return cells, np.asarray(valid)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--weight-cap", type=float, default=WEIGHT_CAP)
    args = parser.parse_args()

    cells_t, skipped = test_cells()
    count_t = Counter(cells_t)
    n_t = sum(count_t.values())

    reference, gt, d3s = None, None, {}
    for label, run in RUNS.items():
        path = WORK / run / "final_eval.json"
        if not path.exists():
            continue
        keys, pred, gt_r, session, _, _ = load(path)
        if reference is None:
            reference, gt = keys, gt_r
        elif keys != reference or not np.array_equal(gt_r, gt):
            raise SystemExit(f"{label} is not scored on the same rows")
        d3s[label] = np.linalg.norm(pred - gt, axis=-1) @ D3_W

    cells_v, valid = val_cells(reference, gt)
    usable = [c for c, ok in zip(cells_v, valid) if ok]
    count_v = Counter(usable)
    n_v = len(usable)

    weights = np.zeros(len(cells_v))
    for i, (cell, ok) in enumerate(zip(cells_v, valid)):
        if not ok or count_t.get(cell, 0) == 0:
            continue
        weights[i] = (count_t[cell] / n_t) / (count_v[cell] / n_v)
    weights = np.minimum(weights, args.weight_cap)
    if weights.sum() > 0:
        weights = weights * (len(weights) / weights.sum())

    covered = sum(v for c, v in count_t.items() if count_v.get(c, 0) > 0)
    effective_n = float(weights.sum() ** 2 / np.square(weights).sum())

    payload = {
        "schema_version": 1,
        "purpose": ("reproduce the importance weighting that made the 2026-09-01 val "
                    "number land within -0.3% of the leaderboard, and apply it to the "
                    "V0 rows we already score"),
        "provenance": {
            "recipe": "scripts/etri_split.py on the H200 (the original project)",
            "constants": {"GOAL_FRAME": GOAL_FRAME, "DT": DT,
                          "SPEED_BINS": [float(x) for x in SPEED_BINS],
                          "GOAL_BINS": [float(x) for x in GOAL_BINS],
                          "weight_cap": args.weight_cap,
                          "lateral_threshold_m": LATERAL_THRESHOLD},
            "calibrated_on": {"model": "v2a_goal epoch_2 (ruled DISALLOWED, goal in the "
                                       "planner's attention query)",
                              "val_weighted": 0.23696, "leaderboard_test_L2_avg": 0.236238922,
                              "gap_percent": -0.3,
                              "note": ("the calibration was measured on a different graph "
                                       "at a different error level; the weighting is "
                                       "reused, the agreement is NOT assumed to transfer")},
        },
        "inputs_used": ("test side: the provided vad_cmd one-hot and the ego poses shipped "
                        "with each test clip. No test label or future trajectory is read."),
        "test": {"clips": n_t, "skipped_for_missing_poses": skipped,
                 "cells": len(count_t)},
        "val": {"rows": len(cells_v), "usable": n_v, "cells": len(count_v),
                "rows_with_zero_weight": int((weights == 0).sum())},
        "weighting": {
            "test_mass_covered_by_val": round(covered / n_t, 5),
            "effective_n": round(effective_n, 1),
            "effective_fraction": round(effective_n / max(n_v, 1), 4),
            "w_min": float(weights[weights > 0].min()) if (weights > 0).any() else 0.0,
            "w_max": float(weights.max()),
        },
        "scores": {},
    }
    for label, d3 in d3s.items():
        plain = float(d3.mean())
        weighted = float((d3 * weights).sum() / weights.sum())
        payload["scores"][label] = {
            "v0_plain_d3": plain, "v0_test_matched_d3": weighted,
            "delta": weighted - plain,
        }

    # Paired session bootstrap under the SAME weights, so the comparison and the
    # headline number are the same estimator.
    sessions = np.asarray([k[0] for k in reference])
    unique = sorted(set(sessions.tolist()))
    index = {x: np.flatnonzero(sessions == x) for x in unique}
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(unique), size=(20000, len(unique)))
    picks = [np.concatenate([index[unique[j]] for j in row]) for row in draws]

    def compare(base, arm):
        diff = d3s[arm] - d3s[base]
        means = np.asarray([
            (diff[p_] * weights[p_]).sum() / max(weights[p_].sum(), 1e-12) for p_ in picks])
        point = float((diff * weights).sum() / weights.sum())
        return {"delta": point,
                "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                "ci_includes_zero": bool(np.percentile(means, 2.5) <= 0
                                         <= np.percentile(means, 97.5)),
                "sessions_improved": sum(
                    1 for x in unique
                    if (diff[index[x]] * weights[index[x]]).sum() < 0),
                "sessions": len(unique)}

    pairs = [("LEN-s0", "MR-NATIVE-s0"), ("MR-LOWDETAIL-s0", "MR-NATIVE-s0"),
             ("MR-NATIVE-s0", "MR-NATIVE-s1"), ("MR-NATIVE-s0", "MR-W64-s0"),
             ("MR-NATIVE-s1", "MR-W64-s0")]
    payload["test_matched_comparisons"] = {
        f"{a} -> {b}": compare(a, b) for a, b in pairs if a in d3s and b in d3s}

    reference_run = "MR-NATIVE-s0"
    if reference_run in payload["scores"]:
        payload["reading"] = {
            "candidate": reference_run,
            "plain": payload["scores"][reference_run]["v0_plain_d3"],
            "test_matched": payload["scores"][reference_run]["v0_test_matched_d3"],
            "leaderboard_first_place_2026_09_01": 0.130536583,
            "our_only_submission_2026_09_01": 0.236238922,
            "caveats": [
                "This is a prediction from a reweighted development set, not a server score.",
                "The -0.3% agreement was established on a different, disallowed graph.",
                "V0 has been read many times; the weighting does not undo that.",
            ],
        }
    (OUT / "test_matched_weighting.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n")
    for label, v in payload["scores"].items():
        print(f"{label:18s} plain {v['v0_plain_d3']:.6f}  "
              f"test-matched {v['v0_test_matched_d3']:.6f}  ({v['delta']:+.6f})")
    print()
    for name, c in payload["test_matched_comparisons"].items():
        print(f"{name:34s} {c['delta']:+.6f}  CI [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}]  "
              f"{c['sessions_improved']}/{c['sessions']}")
    print()
    print(json.dumps(payload["weighting"], indent=1))


if __name__ == "__main__":
    main()
