#!/usr/bin/env python3
"""Read-only fixed aggregation for the terminal P8 C/W reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


FP32_D3_ATOL = float(2 * np.finfo(np.float32).eps)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path_text: str, expected_sha: str) -> dict:
    path = Path(path_text)
    actual = sha256(path)
    if actual != expected_sha:
        raise SystemExit(f"SHA mismatch for {path}: {actual} != {expected_sha}")
    obj = json.loads(path.read_text())
    report = obj["report"]
    if (
        len(obj["records"]) != 1998
        or report["n"] != 1998
        or report["kind"] != "eval"
        or report["step"] != 6000
    ):
        raise SystemExit(f"unexpected row count in {path}")
    pred = np.asarray([r["pred_abs_xy"] for r in obj["records"]], dtype=np.float32)
    gt = np.asarray([r["gt_abs_xy"] for r in obj["records"]], dtype=np.float32)
    point_l2 = np.sqrt(np.sum((pred - gt) * (pred - gt), axis=-1, dtype=np.float32), dtype=np.float32)
    weights = np.asarray([11, 11, 5, 5, 2, 2], dtype=np.float32) / np.float32(36)
    recomputed = np.sum(point_l2 * weights, axis=1, dtype=np.float32).astype(np.float64)
    stored = np.asarray([r["d3"] for r in obj["records"]], dtype=np.float64)
    max_abs = float(np.max(np.abs(recomputed - stored)))
    if max_abs > FP32_D3_ATOL:
        raise SystemExit(f"row D3 FP32 mismatch in {path}: {max_abs} > {FP32_D3_ATOL}")
    if float(stored.mean()) != float(report["official_d3"]):
        raise SystemExit(f"report D3 does not equal stored-row mean in {path}")
    obj["_audit"] = {"fp32_row_d3_max_abs": max_abs, "fp32_row_d3_atol": FP32_D3_ATOL}
    return obj


def identity(record: dict) -> tuple:
    return record["scenario"], record["session"], int(record["frame"]), int(record["row"])


def summarize(obj: dict) -> dict:
    rec = obj["records"]
    pred = np.asarray([r["pred_abs_xy"] for r in rec], dtype=np.float64)
    gt = np.asarray([r["gt_abs_xy"] for r in rec], dtype=np.float64)
    point_l2 = np.linalg.norm(pred - gt, axis=-1)
    ade = [float(point_l2[:, :n].mean()) for n in (2, 4, 6)]
    state = obj["report"]["state_mae_vx_vy_ax_ay_yawrate"]
    return {
        "official_d3": float(obj["report"]["official_d3"]),
        "ade1_first2": ade[0],
        "ade2_first4": ade[1],
        "ade3_first6": ade[2],
        "ade_arithmetic": "descriptive NumPy float64 norm over saved prediction/GT rows",
        "fp32_row_d3_audit": obj["_audit"],
        "vx_mae": float(state[0]),
        "ax_mae": float(state[2]),
        "history_position_mae": [float(x) for x in obj["report"]["history_position_mae_by_offset"]],
        "occ_iou": float(obj["report"]["occ_iou"]),
        "lane_iou": float(obj["report"]["lane_iou"]),
    }


def paired_seed(control: dict, wide: dict) -> dict:
    cr = control["records"]
    wr = wide["records"]
    if [identity(r) for r in cr] != [identity(r) for r in wr]:
        raise SystemExit("C/W row identity mismatch")
    for c, w in zip(cr, wr):
        for key in ("gt_abs_xy", "gt_state", "gt_state_valid", "stop_valid", "stop_target"):
            if c[key] != w[key]:
                raise SystemExit(f"C/W GT mismatch at {identity(c)} key={key}")
    delta = np.asarray([float(w["d3"]) - float(c["d3"]) for c, w in zip(cr, wr)], dtype=np.float64)
    steady = np.asarray([
        bool(r["stop_valid"]) and bool(r["stop_target"]) and float(r["max_gt_displacement_m"]) <= 0.2
        for r in cr
    ])
    depart = np.asarray([
        bool(r["stop_valid"]) and bool(r["stop_target"]) and float(r["max_gt_displacement_m"]) > 0.2
        for r in cr
    ])
    nonstop = np.asarray([bool(r["stop_valid"]) and not bool(r["stop_target"]) for r in cr])
    if not np.all(steady | depart | nonstop):
        raise SystemExit("unexpected invalid stop row")
    cs = summarize(control)
    ws = summarize(wide)
    return {
        "control": cs,
        "wide": ws,
        "wide_minus_control_official_d3": ws["official_d3"] - cs["official_d3"],
        "row_delta": delta,
        "sessions": np.asarray([r["session"] for r in cr], dtype=object),
        "bucket": {
            "steady_stop": {"n": int(steady.sum()), "wide_minus_control": float(delta[steady].mean())},
            "stop_depart": {"n": int(depart.sum()), "wide_minus_control": float(delta[depart].mean())},
            "nonstop": {"n": int(nonstop.sum()), "wide_minus_control": float(delta[nonstop].mean())},
        },
        "shared_history_wide_minus_control": {
            "0.2s": ws["history_position_mae"][0] - cs["history_position_mae"][1],
            "0.5s": ws["history_position_mae"][1] - cs["history_position_mae"][2],
            "1.0s": ws["history_position_mae"][2] - cs["history_position_mae"][3],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    for base in (0, 1):
        for arm in ("control", "wide"):
            p.add_argument(f"--b{base}-{arm}", nargs=2, metavar=("REPORT", "SHA256"), required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    loaded = {}
    for base in (0, 1):
        for arm in ("control", "wide"):
            path, digest = getattr(args, f"b{base}_{arm}")
            loaded[(base, arm)] = load(path, digest)
    reference_ids = [identity(r) for r in loaded[(0, "control")]["records"]]
    reference_gt = [r["gt_abs_xy"] for r in loaded[(0, "control")]["records"]]
    for key, obj in loaded.items():
        if [identity(r) for r in obj["records"]] != reference_ids:
            raise SystemExit(f"cross-base/arm row identity mismatch: {key}")
        if [r["gt_abs_xy"] for r in obj["records"]] != reference_gt:
            raise SystemExit(f"cross-base/arm GT mismatch: {key}")
    paired = {base: paired_seed(loaded[(base, "control")], loaded[(base, "wide")]) for base in (0, 1)}
    sessions = sorted(set(paired[0]["sessions"].tolist()))
    if len(sessions) != 11 or sessions != sorted(set(paired[1]["sessions"].tolist())):
        raise SystemExit("unexpected session set")
    sums = np.zeros((2, 11), dtype=np.float64)
    counts = np.zeros((2, 11), dtype=np.int64)
    for base in (0, 1):
        for j, session in enumerate(sessions):
            mask = paired[base]["sessions"] == session
            sums[base, j] = paired[base]["row_delta"][mask].sum(dtype=np.float64)
            counts[base, j] = int(mask.sum())
    if not np.array_equal(counts[0], counts[1]):
        raise SystemExit("cross-base session count mismatch")
    rng = np.random.default_rng(20260908)
    draws = rng.integers(0, 11, size=(10000, 11))
    boot = np.empty(10000, dtype=np.float64)
    for i, draw in enumerate(draws):
        boot[i] = sums[:, draw].sum(dtype=np.float64) / counts[:, draw].sum(dtype=np.int64)
    serializable = {}
    for base, value in paired.items():
        serializable[f"base{base}"] = {k: v for k, v in value.items() if k not in ("row_delta", "sessions")}
    result = {
        "schema_version": 1,
        "status": "completed_read_only_existing_terminal_artifacts",
        "rows": 1998,
        "sessions": sessions,
        "bootstrap": {
            "repeats": 10000,
            "seed": 20260908,
            "same_session_draw_applied_to_both_seeds": True,
            "frame_weighted": True,
            "observed_wide_minus_control": float(sums.sum(dtype=np.float64) / counts.sum(dtype=np.int64)),
            "ci95_percentile": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        },
        "mean": {
            "control_d3": float(np.mean([paired[b]["control"]["official_d3"] for b in (0, 1)])),
            "wide_d3": float(np.mean([paired[b]["wide"]["official_d3"] for b in (0, 1)])),
            "wide_minus_control": float(np.mean([paired[b]["wide_minus_control_official_d3"] for b in (0, 1)])),
        },
        "seeds": serializable,
        "input_sha256": {
            f"base{base}_{arm}": sha256(Path(getattr(args, f"b{base}_{arm}")[0]))
            for base in (0, 1) for arm in ("control", "wide")
        },
        "boundaries": {
            "model_forward": False,
            "new_evaluation": False,
            "final_rows_accessed": False,
            "selection_or_retuning": False,
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
