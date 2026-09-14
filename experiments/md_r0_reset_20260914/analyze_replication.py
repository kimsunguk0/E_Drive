#!/usr/bin/env python3
"""Replication: does seed 1 reproduce the seed 0 data-expansion effect?

Terminal-to-terminal is the primary comparison, as the work order specifies.
Best-to-best is reported separately because the checkpoint rule is a
development choice, not an independent result.

The pooled line applies the SAME session resample indices to both seeds and
averages the two seed differences.  Two seeds do not turn 11 tune sessions into
22 independent ones, and this is not an estimate of the training-seed
population.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
WORK = ROOT / "work_dirs/md_r0_reset_20260914"
REPORTS = ROOT / "reports/md_r0_reset_20260914"
EVAL = REPORTS / "eval"
RESAMPLES = 20000
SEED = 0
BEST_STEP = {"E1-T203": 13704, "E1-EXP": 20554, "E1-T203_s1": 13704, "E1-EXP_s1": 13704}

RUNS = {
    0: {"T203": WORK / "E1-T203", "EXP": WORK / "E1-EXP"},
    1: {"T203": WORK / "E1-T203_s1", "EXP": WORK / "E1-EXP_s1"},
}
R0_EVAL = ROOT / "work_dirs/motiondrive_v2/q10_q10_flip50_s0/final_eval.json"


def load_records(path):
    records = json.loads(Path(path).read_text())["records"]
    order = sorted(range(len(records)),
                   key=lambda i: (records[i]["session"], records[i]["scenario"],
                                  int(records[i]["frame"])))
    keys = [(records[i]["session"], records[i]["scenario"], int(records[i]["frame"]))
            for i in order]
    d3 = np.asarray([records[i]["d3"] for i in order], dtype=np.float64)
    sessions = np.asarray([k[0] for k in keys])
    return keys, d3, sessions


def curve(run_dir):
    rows = [json.loads(line) for line in
            (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    return {int(r["step"]): r for r in rows if r["kind"] == "eval"}


def probe(tag):
    path = EVAL / f"{tag}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["metrics"]["official_d3_weighted"]


def tune_b1(tag):
    path = EVAL / f"{tag}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["metrics"]["official_d3_weighted"]


def main() -> None:
    reference_keys = None
    d3, sessions = {}, None
    for seed, arms in RUNS.items():
        for arm, run_dir in arms.items():
            keys, values, session_ids = load_records(run_dir / "final_eval.json")
            if reference_keys is None:
                reference_keys, sessions = keys, session_ids
            elif keys != reference_keys:
                raise SystemExit(f"seed {seed} {arm} is not scored on the same rows")
            d3[(seed, arm)] = values
    r0_keys, r0_d3, _ = load_records(R0_EVAL)
    if r0_keys != reference_keys:
        raise SystemExit("R0 is not scored on the same rows")

    unique = sorted(set(sessions.tolist()))
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(unique), size=(RESAMPLES, len(unique)))
    picks = [np.concatenate([index[unique[j]] for j in row]) for row in draws]

    def interval(diff):
        means = np.asarray([diff[p].mean() for p in picks])
        return {"delta": float(diff.mean()),
                "bootstrap_sd": float(means.std(ddof=1)),
                "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                "p_worse": float((means > 0).mean())}, means

    comparisons, per_seed_means = {}, {}
    for seed in RUNS:
        diff = d3[(seed, "EXP")] - d3[(seed, "T203")]
        summary, means = interval(diff)
        per_seed_means[seed] = means
        per_session = {s: float(diff[index[s]].mean()) for s in unique}
        summary.update(sessions_improved=sum(1 for v in per_session.values() if v < 0),
                       sessions=len(unique), per_session_delta=per_session)
        comparisons[f"seed{seed}: T203 -> EXP"] = summary
        vs_r0, _ = interval(d3[(seed, "EXP")] - r0_d3)
        vs_r0["sessions_improved"] = sum(
            1 for s in unique if (d3[(seed, "EXP")] - r0_d3)[index[s]].mean() < 0)
        comparisons[f"seed{seed}: R0 -> EXP"] = vs_r0

    pooled_means = (per_seed_means[0] + per_seed_means[1]) / 2.0
    pooled_delta = float(np.mean([(d3[(s, "EXP")] - d3[(s, "T203")]).mean() for s in RUNS]))
    pooled = {
        "delta_mean_of_two_seeds": pooled_delta,
        "bootstrap_sd": float(pooled_means.std(ddof=1)),
        "ci95": [float(np.percentile(pooled_means, 2.5)),
                 float(np.percentile(pooled_means, 97.5))],
        "p_worse": float((pooled_means > 0).mean()),
        "method": ("the same session resample indices are applied to both seeds and the two "
                   "seed differences are averaged"),
        "caveat": ("two seeds do not make 11 tune sessions into 22 independent ones, and this "
                   "is not an estimate of the training-seed population"),
    }

    table = []
    for seed, arms in RUNS.items():
        names = {"T203": "E1-T203" + ("_s1" if seed else ""),
                 "EXP": "E1-EXP" + ("_s1" if seed else "")}
        row = {"seed": seed}
        for arm, name in names.items():
            evals = curve(arms[arm])
            best_step = min(sorted(evals), key=lambda s: (evals[s]["official_d3"], s))
            row[f"{arm}_terminal"] = evals[20554]["official_d3"]
            row[f"{arm}_best"] = evals[best_step]["official_d3"]
            row[f"{arm}_best_step"] = best_step
            row[f"{arm}_probe_terminal"] = probe(f"{name}_probe_step20554")
            row[f"{arm}_tune_b1_terminal"] = tune_b1(f"{name}_tuneB1_step20554")
            row[f"{arm}_vx_mae"] = evals[20554]["state_mae_vx_vy_ax_ay_yawrate"][0]
        row["EXP_minus_T203_terminal"] = row["EXP_terminal"] - row["T203_terminal"]
        row["EXP_minus_T203_best"] = row["EXP_best"] - row["T203_best"]
        table.append(row)

    replicated = all(r["EXP_minus_T203_terminal"] < 0 for r in table) and all(
        comparisons[f"seed{s}: T203 -> EXP"]["sessions_improved"] >= 2 for s in RUNS)
    summary = {
        "schema_version": 1,
        "question": "does a second continuation seed reproduce the data-expansion improvement?",
        "design": ("continuation replication from the same R0 weights with a different shuffle "
                   "and augmentation seed; not two independently pretrained models"),
        "evaluation": {"split": "tune (V0)", "rows": int(len(sessions)),
                       "sessions": len(unique), "eval_batch": 4, "precision": "bf16",
                       "row_identity_verified_across_all_runs": True},
        "bootstrap": {"resamples": RESAMPLES, "unit": "session", "seed": SEED,
                      "shared_indices_across_seeds": True},
        "table": table,
        "comparisons": comparisons,
        "pooled_seed_mean": pooled,
        "verdict": "REPLICATED" if replicated else "NOT_REPLICATED",
        "reading": ("terminal-to-terminal is the primary comparison; best-to-best is a "
                    "development result of the pre-registered checkpoint rule"),
        "caveats": [
            "V0 is a repeatedly used development set; a good interval is not an independent confirmation.",
            "Two seeds bound seed sensitivity loosely; they do not estimate the seed population.",
            "H remains unopened.",
        ],
    }
    (REPORTS / "replication_summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n")
    with (REPORTS / "replication_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        for row in table:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    print(json.dumps({"verdict": summary["verdict"], "table": table,
                      "per_seed": {k: {"delta": round(v["delta"], 6),
                                       "ci95": [round(x, 6) for x in v["ci95"]],
                                       "sessions_improved": v.get("sessions_improved")}
                                   for k, v in comparisons.items()},
                      "pooled": {"delta": round(pooled["delta_mean_of_two_seeds"], 6),
                                 "ci95": [round(x, 6) for x in pooled["ci95"]]}}, indent=1))


if __name__ == "__main__":
    main()
