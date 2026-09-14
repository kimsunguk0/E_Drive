#!/usr/bin/env python3
"""E1 analysis: paired session bootstrap between R0, E1-T203 and E1-EXP.

Uses the existing project tool: resample the 11 tune sessions 20,000 times and
average the ROW-LEVEL differences, so every comparison is paired on the same
1,998 rows.  Absolute session bootstraps are reported too, because the paired
interval is the narrow one and reading the absolute one instead is how an
earlier round misread a real effect as noise.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
REPORTS = ROOT / "reports/md_r0_reset_20260914"
RESAMPLES = 20000
SEED = 0

MODELS = {
    "R0": ROOT / "work_dirs/motiondrive_v2/q10_q10_flip50_s0/final_eval.json",
    "E1-T203": ROOT / "work_dirs/md_r0_reset_20260914/E1-T203/final_eval.json",
    "E1-EXP": ROOT / "work_dirs/md_r0_reset_20260914/E1-EXP/final_eval.json",
}
COMPARISONS = [("R0", "E1-T203"), ("R0", "E1-EXP"), ("E1-T203", "E1-EXP")]
# Roadmap 10.1 development promotion thresholds; not an official pass mark.
CANDIDATE = -0.005
PRIORITY = -0.010


def load(path):
    records = json.loads(Path(path).read_text())["records"]
    keys = [(r["session"], r["scenario"], int(r["frame"])) for r in records]
    order = np.argsort([f"{s}|{c}|{f:06d}" for s, c, f in keys])
    keys = [keys[i] for i in order]
    d3 = np.asarray([records[i]["d3"] for i in order], dtype=np.float64)
    sessions = np.asarray([k[0] for k in keys])
    return keys, d3, sessions


def bootstrap_paired(diff, sessions, rng):
    unique = np.unique(sessions)
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    draws = rng.integers(0, len(unique), size=(RESAMPLES, len(unique)))
    means = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        picked = np.concatenate([index[unique[j]] for j in draws[i]])
        means[i] = diff[picked].mean()
    return means


def bootstrap_absolute(values, sessions, rng):
    unique = np.unique(sessions)
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    draws = rng.integers(0, len(unique), size=(RESAMPLES, len(unique)))
    means = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        picked = np.concatenate([index[unique[j]] for j in draws[i]])
        means[i] = values[picked].mean()
    return means


def main() -> None:
    loaded = {}
    reference_keys = None
    for name, path in MODELS.items():
        keys, d3, sessions = load(path)
        if reference_keys is None:
            reference_keys, reference_sessions = keys, sessions
        elif keys != reference_keys:
            raise SystemExit(f"{name} is not scored on the same rows")
        loaded[name] = d3
    sessions = reference_sessions
    unique = sorted(set(sessions.tolist()))

    rng = np.random.default_rng(SEED)
    absolute = {}
    for name, d3 in loaded.items():
        means = bootstrap_absolute(d3, sessions, rng)
        absolute[name] = {
            "official_d3": float(d3.mean()),
            "session_mean_d3": float(np.mean([d3[sessions == s].mean() for s in unique])),
            "bootstrap_sd": float(means.std(ddof=1)),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "session_d3": {s: float(d3[sessions == s].mean()) for s in unique},
        }

    comparisons = {}
    for base, arm in COMPARISONS:
        diff = loaded[arm] - loaded[base]
        means = bootstrap_paired(diff, sessions, rng)
        per_session = {s: float(diff[sessions == s].mean()) for s in unique}
        delta = float(diff.mean())
        upper = float(np.percentile(means, 97.5))
        improved = sum(1 for v in per_session.values() if v < 0)
        if delta <= PRIORITY and upper < 0:
            verdict = "PRIORITY_CANDIDATE"
        elif delta <= CANDIDATE and upper < 0:
            verdict = "CANDIDATE"
        elif upper >= 0:
            verdict = "UNCERTAIN_CI_INCLUDES_ZERO"
        else:
            verdict = "BELOW_PROMOTION_THRESHOLD"
        comparisons[f"{base} -> {arm}"] = {
            "delta": delta,
            "bootstrap_sd": float(means.std(ddof=1)),
            "ci95": [float(np.percentile(means, 2.5)), upper],
            "p_worse": float((means > 0).mean()),
            "sessions_improved": improved, "sessions": len(unique),
            "per_session_delta": per_session,
            "verdict": verdict,
        }

    payload = {
        "schema_version": 1,
        "evaluation": {"split": "tune (V0)", "rows": int(len(sessions)),
                       "sessions": len(unique), "eval_batch": 4, "precision": "bf16",
                       "mode": "same rows, same eval mode for all three models"},
        "bootstrap": {"resamples": RESAMPLES, "unit": "session", "seed": SEED,
                      "statistic": "mean of row-level differences"},
        "promotion_thresholds": {"candidate": CANDIDATE, "priority": PRIORITY,
                                 "rule": "delta below threshold AND paired CI upper bound below 0"},
        "absolute": absolute,
        "comparisons": comparisons,
        "caveats": [
            "V0 is a repeatedly used development set; this is not an independent confirmation.",
            "Single training seed; the paired interval covers evaluation sampling only.",
            "The terminal checkpoint is also the best-on-V0 checkpoint for both arms, so no selection optimism is added here, but V0 still chose the budget.",
        ],
    }
    (REPORTS / "e1_paired_analysis.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "absolute": {k: round(v["official_d3"], 6) for k, v in absolute.items()},
        "comparisons": {k: {"delta": round(v["delta"], 6), "ci95": [round(x, 6) for x in v["ci95"]],
                            "p_worse": v["p_worse"], "sessions_improved": v["sessions_improved"],
                            "verdict": v["verdict"]}
                        for k, v in comparisons.items()}}, indent=1))


if __name__ == "__main__":
    main()
