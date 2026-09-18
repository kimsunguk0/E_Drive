#!/usr/bin/env python3
"""Paired session bootstrap for the A2 NOM arms on the same V0 rows.

BASE and MH4 share data, budget, initialisation and sample order and differ
only in the scene evidence aggregation, so their difference is the quantity of
interest. FULL is reported for completeness but is NOT comparable: its training
set contains these very rows, so its number here is in-fit by construction and
is labelled as such rather than ranked against the others.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
W = ROOT / "work_dirs/md_a2_nominal_mh4_20260918"
OUT = ROOT / "reports/md_a2_nominal_mh4_20260918"
WEIGHTS = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0
RESAMPLES = 20000
RUNS = ("A2-BASE-NOM-s1", "A2-MH4-NOM-s1", "A2-FULL-NOM-s1")
IN_FIT = {"A2-FULL-NOM-s1"}


def load(path):
    records = json.loads(Path(path).read_text())["records"]
    order = sorted(range(len(records)),
                   key=lambda i: (records[i]["session"], records[i]["scenario"],
                                  int(records[i]["frame"])))
    records = [records[i] for i in order]
    return ([(r["session"], r["scenario"], int(r["frame"])) for r in records],
            np.asarray([r["pred_abs_xy"] for r in records], dtype=np.float64),
            np.asarray([r["gt_abs_xy"] for r in records], dtype=np.float64),
            np.asarray([r["session"] for r in records]),
            np.asarray([r.get("bucket", r.get("stop_bucket")) for r in records]))


def main():
    reference = gt = sessions = bucket = None
    d3s = {}
    for name in RUNS:
        path = W / name / "final_eval.json"
        if not path.exists():
            print(f"{name}: not finished, skipped")
            continue
        keys, pred, gt_r, sess, buck = load(path)
        if reference is None:
            reference, gt, sessions, bucket = keys, gt_r, sess, buck
        elif keys != reference or not np.array_equal(gt_r, gt):
            raise SystemExit(f"{name} is not scored on the same rows or ground truth")
        d3s[name] = np.linalg.norm(pred - gt, axis=-1) @ WEIGHTS

    unique = sorted(set(sessions.tolist()))
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(unique), size=(RESAMPLES, len(unique)))
    picks = [np.concatenate([index[unique[j]] for j in row]) for row in draws]

    def compare(base, arm):
        diff = d3s[arm] - d3s[base]
        means = np.asarray([diff[p].mean() for p in picks])
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        return {"delta": float(diff.mean()), "ci95": [lo, hi],
                "ci_includes_zero": bool(lo <= 0 <= hi),
                "p_worse": float((means > 0).mean()),
                "sessions_improved": sum(1 for s in unique if diff[index[s]].mean() < 0),
                "sessions": len(unique)}

    primary = compare("A2-BASE-NOM-s1", "A2-MH4-NOM-s1")
    groups = {}
    for name in sorted(set(bucket.tolist())):
        mask = bucket == name
        entry = {"rows": int(mask.sum()), "share": float(mask.mean())}
        for run, d3 in d3s.items():
            entry[run] = {"mean_d3": float(d3[mask].mean()),
                          "contribution": float(d3[mask].sum() / len(d3))}
        if "A2-MH4-NOM-s1" in d3s:
            diff = d3s["A2-MH4-NOM-s1"] - d3s["A2-BASE-NOM-s1"]
            entry["mh4_minus_base_within_group"] = float(diff[mask].mean())
            entry["mh4_minus_base_contribution"] = float(diff[mask].sum() / len(diff))
        groups[name] = entry

    # The seed spread measured on the MR lineage is the only empirical scale we
    # have for "how much do two identical recipes differ"; there is no A2 seed
    # pair yet, so it is quoted as a reference, not as this experiment's own.
    reference_seed_spread = 0.000890

    payload = {
        "schema_version": 1,
        "rows": int(len(next(iter(d3s.values())))),
        "sessions": len(unique),
        "resamples": RESAMPLES,
        "terminals": {k: float(v.mean()) for k, v in d3s.items()},
        "in_fit_and_not_comparable": sorted(IN_FIT & set(d3s)),
        "PRIMARY A2-BASE-NOM -> A2-MH4-NOM": primary,
        "reference_seed_spread_from_mr_lineage": reference_seed_spread,
        "delta_vs_reference_seed_spread": {
            "delta": primary["delta"],
            "reference_spread": reference_seed_spread,
            "exceeds": bool(abs(primary["delta"]) > reference_seed_spread),
        },
        "groups": groups,
        "caveats": [
            "BASE and MH4 are not parameter-matched: 26,557,208 vs 26,581,784.",
            "One seed each; no A2 seed pair exists, so the spread quoted is the MR lineage's.",
            "A2-FULL trained on these rows; its number here is in-fit, not a held-out score.",
        ],
    }
    (OUT / "a2_dev_bootstrap.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    print("terminals:")
    for k, v in payload["terminals"].items():
        tag = "   <-- IN-FIT, not comparable" if k in IN_FIT else ""
        print(f"  {k:18s} {v:.6f}{tag}")
    print(f"\nPRIMARY  BASE -> MH4: {primary['delta']:+.6f}  "
          f"CI [{primary['ci95'][0]:+.6f}, {primary['ci95'][1]:+.6f}]  "
          f"{primary['sessions_improved']}/{primary['sessions']} sessions  "
          f"p_worse {primary['p_worse']:.3f}")
    print(f"  CI includes zero: {primary['ci_includes_zero']}")
    print(f"  |delta| {abs(primary['delta']):.6f} vs MR seed spread {reference_seed_spread:.6f} "
          f"-> exceeds: {payload['delta_vs_reference_seed_spread']['exceeds']}")
    print("\nby bucket (rows, BASE, MH4, MH4-BASE contribution):")
    for name, g in groups.items():
        if "A2-MH4-NOM-s1" not in g:
            continue
        print(f"  {name:10s} n={g['rows']:5d}  {g['A2-BASE-NOM-s1']['mean_d3']:.6f}  "
              f"{g['A2-MH4-NOM-s1']['mean_d3']:.6f}  "
              f"{g['mh4_minus_base_contribution']:+.6f}")
    print("\nwrote a2_dev_bootstrap.json")


if __name__ == "__main__":
    main()
