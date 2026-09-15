#!/usr/bin/env python3
"""Does the length auxiliary's effect replicate on a second seed?

Each seed is paired against its OWN control, so the shuffle and augmentation
stream is held fixed within each comparison.  The pooled line applies the same
session resample indices to both seeds and averages the two differences; two
seeds do not turn 11 sessions into 22.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
WORK = ROOT / "work_dirs/md_r0_reset_20260914"
OUT = ROOT / "reports/md_exp_diagnosis_20260915"
W = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0
DT = 0.5
RESAMPLES = 20000
REGIMES = ["stop_hold", "departing", "constant", "decelerating", "accelerating"]
PAIRS = {0: (WORK / "E1-EXP", WORK / "E1-EXP-LEN-s0"),
         1: (WORK / "E1-EXP_s1", WORK / "E1-EXP-LEN-s1")}


def load(path):
    records = json.loads(Path(path).read_text())["records"]
    order = sorted(range(len(records)), key=lambda i: (
        records[i]["session"], records[i]["scenario"], int(records[i]["frame"])))
    records = [records[i] for i in order]
    return ([(r["session"], r["scenario"], int(r["frame"])) for r in records],
            np.asarray([r["pred_abs_xy"] for r in records], dtype=np.float64),
            np.asarray([r["gt_abs_xy"] for r in records], dtype=np.float64),
            np.asarray([r["session"] for r in records]))


def seg(p):
    start = np.concatenate([np.zeros((len(p), 1, 2)), p[:, :-1]], axis=1)
    return np.linalg.norm(p - start, axis=-1)


def regimes(ell_g):
    total, vbar = ell_g.sum(1), ell_g / DT
    change, dv = vbar[:, -1] - vbar[:, 0], np.diff(vbar, axis=1)
    has_d, has_a = (dv < -0.5).any(1), (dv > 0.5).any(1)
    label = np.full(len(total), "constant", dtype=object)
    label[(change <= -1.0) & ~has_a] = "decelerating"
    label[(change >= 1.0) & ~has_d] = "accelerating"
    label[has_d & has_a] = "transition"
    label[(total > 1.0) & (vbar[:, 0] < 1.0)] = "departing"
    label[total <= 1.0] = "stop_hold"
    return label


def curve(run_dir):
    rows = [json.loads(line) for line in
            (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    return {int(r["step"]): r for r in rows if r["kind"] == "eval"}


def main() -> None:
    reference, gt, sessions, label = None, None, None, None
    diffs, table = {}, []
    for seed, (control_dir, arm_dir) in PAIRS.items():
        if json.loads((arm_dir / "manifest.json").read_text())["status"] != "completed":
            raise SystemExit(f"seed {seed} arm is not completed")
        key_c, pred_c, gt_c, session = load(control_dir / "final_eval.json")
        key_a, pred_a, gt_a, _ = load(arm_dir / "final_eval.json")
        if key_c != key_a or not np.array_equal(gt_c, gt_a):
            raise SystemExit(f"seed {seed}: rows or ground truth differ")
        if reference is None:
            reference, gt, sessions = key_c, gt_c, session
            label = regimes(seg(gt))
        elif key_c != reference:
            raise SystemExit("the two seeds are not scored on the same rows")
        d3_c = np.linalg.norm(pred_c - gt, axis=-1) @ W
        d3_a = np.linalg.norm(pred_a - gt, axis=-1) @ W
        diffs[seed] = d3_a - d3_c
        b_c = ((seg(pred_c) - seg(gt)) / DT).mean(1)
        b_a = ((seg(pred_a) - seg(gt)) / DT).mean(1)
        control_curve, arm_curve = curve(control_dir), curve(arm_dir)
        best = min(sorted(arm_curve), key=lambda s: (arm_curve[s]["official_d3"], s))
        row = {"seed": seed,
               "control_terminal": control_curve[20554]["official_d3"],
               "arm_terminal": arm_curve[20554]["official_d3"],
               "arm_best": arm_curve[best]["official_d3"], "arm_best_step": best,
               "delta_terminal": arm_curve[20554]["official_d3"] - control_curve[20554]["official_d3"],
               "mean_abs_b_control": float(np.abs(b_c).mean()),
               "mean_abs_b_arm": float(np.abs(b_a).mean())}
        for name in REGIMES:
            mask = label == name
            row[f"{name}_signed_b_control"] = float(b_c[mask].mean())
            row[f"{name}_signed_b_arm"] = float(b_a[mask].mean())
            row[f"{name}_d3_delta"] = float(d3_a[mask].mean() - d3_c[mask].mean())
        table.append(row)

    unique = sorted(set(sessions.tolist()))
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(unique), size=(RESAMPLES, len(unique)))
    picks = [np.concatenate([index[unique[j]] for j in row]) for row in draws]

    per_seed, means_by_seed = {}, {}
    for seed, diff in diffs.items():
        means = np.asarray([diff[p].mean() for p in picks])
        means_by_seed[seed] = means
        per_seed[f"seed{seed}"] = {
            "delta": float(diff.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "p_worse": float((means > 0).mean()),
            "sessions_improved": sum(1 for s in unique if diff[index[s]].mean() < 0),
            "sessions": len(unique)}
    pooled_means = (means_by_seed[0] + means_by_seed[1]) / 2.0
    pooled = {
        "delta_mean_of_two_seeds": float(np.mean([d.mean() for d in diffs.values()])),
        "ci95": [float(np.percentile(pooled_means, 2.5)), float(np.percentile(pooled_means, 97.5))],
        "p_worse": float((pooled_means > 0).mean()),
        "method": "same session resample indices applied to both seeds, then averaged",
        "caveat": "two seeds do not make 11 sessions into 22 independent ones"}

    both_negative = all(r["delta_terminal"] < 0 for r in table)
    both_consistent = all(per_seed[f"seed{s}"]["sessions_improved"] >= 7 for s in PAIRS)
    regimes_consistent = all(
        np.sign(r[f"{name}_d3_delta"]) <= 0 for r in table for name in REGIMES)
    if both_negative and both_consistent:
        verdict = "REPLICATED"
        action = ("the effect is real and consistent, but at about -0.0017 it stays below this "
                  "project's -0.005 candidate line; adopt only as a small, cheap improvement "
                  "on top of the expanded-data model, and do not raise lambda in search of more")
    elif both_negative:
        verdict = "REPLICATED_WEAKLY"
        action = "same direction in both seeds but session-inconsistent; treat as marginal"
    else:
        verdict = "NOT_REPLICATED"
        action = "close the lambda 0.25 setting as no confirmed effect and move to observation"

    payload = {
        "schema_version": 1,
        "design": ("each seed's arm is paired against its own control, so the shuffle and "
                   "augmentation stream is identical inside each comparison"),
        "table": table, "per_seed": per_seed, "pooled": pooled,
        "all_regime_deltas_non_positive_in_both_seeds": bool(regimes_consistent),
        "verdict": verdict, "action": action,
        "project_thresholds": {"candidate": -0.005, "priority": -0.010,
                               "note": "the operating lines used in the E1 round"},
        "caveats": ["V0 is a repeatedly used development set",
                    "two seeds bound seed sensitivity loosely, they do not estimate it",
                    "H remains unopened"],
    }
    (OUT / "length_aux_replication.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n")
    with (OUT / "length_aux_replication.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({"verdict": verdict, "action": action,
                      "table": [{k: r[k] for k in ("seed", "control_terminal", "arm_terminal",
                                                   "arm_best", "arm_best_step", "delta_terminal",
                                                   "mean_abs_b_control", "mean_abs_b_arm")}
                                for r in table],
                      "per_seed": per_seed, "pooled": pooled,
                      "all_regime_deltas_non_positive": regimes_consistent},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
