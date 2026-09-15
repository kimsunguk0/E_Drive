#!/usr/bin/env python3
"""Per-segment length sensitivity: which interval's progress error costs the most.

Same framing as section 6: replace ONE segment's predicted chord length with the
GT length, keep every predicted direction, and re-integrate.  Because the points
are cumulative, fixing an early segment also moves every later point, which is
exactly the mechanical question that separates the two candidate next steps.

Not additive, not a reachable score, not a percentage attribution of cause.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
OUT = ROOT / "reports/md_exp_diagnosis_20260915"
EXP_EVAL = ROOT / "work_dirs/md_r0_reset_20260914/E1-EXP/final_eval.json"
W = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0
DIRECTION_MIN_M = 0.01


def main() -> None:
    records = json.loads(EXP_EVAL.read_text())["records"]
    pred = np.asarray([r["pred_abs_xy"] for r in records], dtype=np.float64)
    gt = np.asarray([r["gt_abs_xy"] for r in records], dtype=np.float64)

    def seg(p):
        start = np.concatenate([np.zeros((len(p), 1, 2)), p[:, :-1]], axis=1)
        dp = p - start
        return dp, np.linalg.norm(dp, axis=-1)

    dp_p, ell_p = seg(pred)
    dp_g, ell_g = seg(gt)
    subset = (ell_g > DIRECTION_MIN_M).all(1) & (ell_p > DIRECTION_MIN_M).all(1)
    u_p = dp_p / np.maximum(ell_p, 1e-12)[..., None]

    def d3(points):
        return float((np.linalg.norm(points[subset] - gt[subset], axis=-1) @ W).mean())

    baseline = d3(pred)
    rows = []
    for k in range(6):
        lengths = ell_p.copy()
        lengths[:, k] = ell_g[:, k]
        fixed = np.cumsum(lengths[..., None] * u_p, axis=1)
        value = d3(fixed)
        rows.append({"segment": k, "t_start_s": 0.5 * k, "t_end_s": 0.5 * (k + 1),
                     "d3_with_this_segment_length_replaced": value,
                     "reduction_from_baseline": baseline - value,
                     "reduction_share_of_baseline": (baseline - value) / baseline})
    lengths = ell_g.copy()
    all_fixed = d3(np.cumsum(lengths[..., None] * u_p, axis=1))
    first_two = ell_p.copy()
    first_two[:, :2] = ell_g[:, :2]
    first_two_fixed = d3(np.cumsum(first_two[..., None] * u_p, axis=1))

    payload = {
        "schema_version": 1,
        "subset_rows": int(subset.sum()), "subset_share": float(subset.mean()),
        "baseline_d3_on_subset": baseline,
        "all_segment_lengths_replaced_d3": all_fixed,
        "first_two_segment_lengths_replaced_d3": first_two_fixed,
        "per_segment": rows,
        "sum_of_single_segment_reductions": float(sum(r["reduction_from_baseline"] for r in rows)),
        "caveats": [
            "single-segment reductions do NOT sum to the all-segment reduction; the points are cumulative and the effects interact",
            "this is a sensitivity probe with GT substituted, not a reachable score and not a cause attribution",
            "rows with any segment under the 1 cm direction threshold are excluded from the subset",
        ],
    }
    (OUT / "segment_sensitivity.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"baseline": baseline, "all_replaced": all_fixed,
                      "first_two_replaced": first_two_fixed,
                      "per_segment_reduction": [round(r["reduction_from_baseline"], 5) for r in rows],
                      "per_segment_share": [round(r["reduction_share_of_baseline"], 4) for r in rows],
                      "sum_of_singles": payload["sum_of_single_segment_reductions"]}, indent=1))


if __name__ == "__main__":
    main()
