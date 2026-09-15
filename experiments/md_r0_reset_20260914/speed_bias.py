#!/usr/bin/env python3
"""Is the travel-distance bias a property of the regime label, or of speed?

The clip review found the same over-prediction in slow congested rows that the
threshold labelling files under "constant". This bins by GT initial speed
instead, so the pattern can be read without the label.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
OUT = ROOT / "reports/md_exp_diagnosis_20260915"
EXP_EVAL = ROOT / "work_dirs/md_r0_reset_20260914/E1-EXP/final_eval.json"
W = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0
DT = 0.5
EDGES = [0.0, 0.5, 2.0, 5.0, 8.0, 11.0, 14.0, np.inf]


def main() -> None:
    records = json.loads(EXP_EVAL.read_text())["records"]
    pred = np.asarray([r["pred_abs_xy"] for r in records], dtype=np.float64)
    gt = np.asarray([r["gt_abs_xy"] for r in records], dtype=np.float64)
    d3 = np.linalg.norm(pred - gt, axis=-1) @ W

    def seg(p):
        start = np.concatenate([np.zeros((len(p), 1, 2)), p[:, :-1]], axis=1)
        return np.linalg.norm(p - start, axis=-1)

    ell_p, ell_g = seg(pred), seg(gt)
    b = ((ell_p - ell_g) / DT).mean(axis=1)
    v0 = ell_g[:, 0] / DT                      # GT speed over the first interval

    rows = []
    for low, high in zip(EDGES[:-1], EDGES[1:]):
        mask = (v0 >= low) & (v0 < high)
        if not mask.any():
            continue
        rows.append({
            "v0_low_ms": low, "v0_high_ms": (None if np.isinf(high) else high),
            "rows": int(mask.sum()), "row_share": float(mask.mean()),
            "gt_total_chord_mean_m": float(ell_g[mask].sum(1).mean()),
            "signed_b_ms": float(b[mask].mean()),
            "mean_abs_b_ms": float(np.abs(b[mask]).mean()),
            "mean_total_chord_error_m": float(3.0 * b[mask].mean()),
            "relative_total_chord_error_pct": float(
                100 * 3.0 * b[mask].mean() / max(ell_g[mask].sum(1).mean(), 1e-9)),
            "mean_d3_m": float(d3[mask].mean()),
            "share_of_total_d3": float(d3[mask].sum() / d3.sum()),
        })

    slow = v0 < 2.0
    payload = {
        "schema_version": 1,
        "question": "does the travel-distance bias follow the regime label or the speed?",
        "binning": "GT first-interval chord speed, an offline diagnostic quantity",
        "bins": rows,
        "slow_rows_under_2ms": {
            "rows": int(slow.sum()), "row_share": float(slow.mean()),
            "signed_b_ms": float(b[slow].mean()),
            "mean_total_chord_error_m": float(3.0 * b[slow].mean()),
            "share_of_total_d3": float(d3[slow].sum() / d3.sum()),
            "mean_d3_m": float(d3[slow].mean()),
        },
        "reading": ("the over-prediction of travel distance is concentrated at low GT speed and "
                    "is not confined to the rows the threshold labelling calls stop_hold or "
                    "departing; congested crawling rows that fall under 'constant' show it too"),
        "not_claimed": [
            "v0 here is a GT first-interval chord speed computed offline, not a sensor reading "
            "and not a model input",
            "a chord length has no direction, so this is excessive predicted travel distance, "
            "not necessarily forward drift",
            "binning by speed does not establish that speed causes the bias",
        ],
    }
    (OUT / "speed_bias.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    with (OUT / "speed_bias.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("%-16s %6s %8s %10s %12s %9s %8s" % ("v0 bin (m/s)", "rows", "GT 3s(m)",
                                               "signed b", "chord err(m)", "rel %", "D3"))
    for r in rows:
        high = "inf" if r["v0_high_ms"] is None else f"{r['v0_high_ms']:g}"
        print("%-16s %6d %8.2f %+10.4f %+12.3f %+9.1f %8.4f" % (
            f"{r['v0_low_ms']:g}-{high}", r["rows"], r["gt_total_chord_mean_m"],
            r["signed_b_ms"], r["mean_total_chord_error_m"],
            r["relative_total_chord_error_pct"], r["mean_d3_m"]))
    print("slow (<2 m/s):", json.dumps(payload["slow_rows_under_2ms"], indent=1))


if __name__ == "__main__":
    main()
