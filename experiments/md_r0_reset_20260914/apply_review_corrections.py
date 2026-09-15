#!/usr/bin/env python3
"""Apply the reporting corrections raised in the 2026-09-16 review.

These are naming and scope fixes.  No measured value changes; the point is that
some columns were labelled in a way that invited the wrong reading.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("/NHNHOME/data/sukim/adcl")
OUT = ROOT / "reports/md_exp_diagnosis_20260915"
EXP_EVAL = ROOT / "work_dirs/md_r0_reset_20260914/E1-EXP/final_eval.json"
PRODUCER = ROOT / "experiments/md_r0_reset_20260914/diagnose_exp.py"

RENAMES = {
    "chord_abs_error_mean_m": "mean_sum_abs_interval_chord_error_m",
    "chord_error_mean_m": "mean_sum_signed_interval_chord_error_m",
}
# The thresholds the regime labeller actually used, recorded so the grouping can
# be reproduced without reading the script.
REGIME_THRESHOLDS = {
    "stop_hold": "total GT chord over 3 s <= 1.0 m",
    "departing": "total > 1.0 m and first-interval vbar < 1.0 m/s",
    "decelerating": "vbar[5] - vbar[0] <= -1.0 m/s and no interval step > +0.5 m/s",
    "accelerating": "vbar[5] - vbar[0] >= +1.0 m/s and no interval step < -0.5 m/s",
    "transition": "both an interval step < -0.5 and one > +0.5 m/s",
    "constant": "none of the above",
    "precedence": "stop_hold, then departing, then transition, then accel/decel, then constant",
    "labels_use_future_gt": True,
    "labels_never_reach_a_model_input": True,
}


def main() -> None:
    # 1. regime CSV column names
    path = OUT / "regime_error_budget.csv"
    rows = list(csv.DictReader(path.open()))
    fixed = [{RENAMES.get(k, k): v for k, v in row.items()} for row in rows]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fixed[0]))
        writer.writeheader()
        writer.writerows(fixed)

    diagnosis = json.loads((OUT / "diagnosis.json").read_text())
    for row in diagnosis["regime_error_budget"]["rows"]:
        for old, new in RENAMES.items():
            if old in row:
                row[new] = row.pop(old)
    diagnosis["regime_error_budget"]["column_definitions"] = {
        "mean_sum_abs_interval_chord_error_m": (
            "per row, the SUM of |interval chord error| over the six intervals, then averaged; "
            "this is not the per-interval mean, which is about 0.0903 m"),
        "mean_sum_signed_interval_chord_error_m": (
            "per row, the SUM of signed interval chord errors, then averaged; equals the "
            "cumulative chord error"),
    }
    diagnosis["regime_error_budget"]["thresholds"] = REGIME_THRESHOLDS

    # 2. the two different agreement numbers must not be conflated
    records = json.loads(EXP_EVAL.read_text())["records"]
    checks = diagnosis["point_error_budget"]["checks"]
    diagnosis["point_error_budget"]["agreement"] = {
        "internal_algebra_max_diff": checks["max_abs_diff"],
        "internal_algebra_meaning": (
            "difference between the sum of the six weighted contributions and the recomputed "
            "mean D3; this is an algebraic identity check inside this analysis"),
        "vs_stored_row_d3_max_diff": checks["stored_d3_max_row_diff"],
        "vs_stored_row_d3_meaning": (
            "largest per-row difference between the D3 recomputed here and the D3 the "
            "evaluation stored; this is the reproduction check against the original run"),
        "do_not_conflate": True,
    }

    # 3. relative error definition
    diagnosis["readout_vs_plan_relative"]["definition"] = (
        "relative error = mean(absolute error) / mean(ground-truth magnitude), an aggregate "
        "ratio. It is NOT a per-row MAPE, and stopped or slow rows can be hidden inside the "
        "mean denominator.")

    # 4. the first-interval speed metric is a chord scalar, not a velocity vector
    diagnosis["head_vs_plan"]["definitions"]["first_segment_abs_speed_metric"] = (
        "first_segment_mean_abs_speed_error_ms equals 2 * mean(|interval chord error|): it is a "
        "scalar progress-speed magnitude, not a velocity vector component, and it must not be "
        "described as comparing the same x-velocity the state head predicts")

    # 5. provenance for independent reproduction
    diagnosis["reproduction"] = {
        "producer": "experiments/md_r0_reset_20260914/diagnose_exp.py",
        "producer_sha256": hashlib.sha256(PRODUCER.read_bytes()).hexdigest(),
        "source_predictions": str(EXP_EVAL),
        "source_predictions_sha256": hashlib.sha256(EXP_EVAL.read_bytes()).hexdigest(),
        "rows": len(records),
        "note": ("the source pred/GT file is under work_dirs and is gitignored; its hash is "
                 "recorded here so the reported numbers can be checked against it"),
    }
    (OUT / "diagnosis.json").write_text(json.dumps(diagnosis, indent=1, sort_keys=True,
                                                   default=float) + "\n")

    # 6. relative CSV gets the definition in-band
    path = OUT / "readout_vs_plan_relative.csv"
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        row["relative_error_definition"] = "mean(abs error) / mean(gt magnitude), not per-row MAPE"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "regime_columns_renamed": RENAMES,
        "internal_algebra_max_diff": checks["max_abs_diff"],
        "vs_stored_row_d3_max_diff": checks["stored_d3_max_row_diff"],
        "producer_sha256": diagnosis["reproduction"]["producer_sha256"][:16],
    }, indent=1))


if __name__ == "__main__":
    main()
