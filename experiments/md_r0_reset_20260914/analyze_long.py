#!/usr/bin/env python3
"""Work item C judgement: EXP-LONG against the short expanded run on V0.

Applies the table in section 4.5 of the work order.  The comparison is a
schedule-and-budget contrast, not the causal effect of exposure count alone,
because the LONG run's cosine horizon is its own total from step one.
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
SHORT = WORK / "E1-EXP"
LONG = WORK / "E1-EXP-LONG-s0"
R0_EVAL = ROOT / "work_dirs/motiondrive_v2/q10_q10_flip50_s0/final_eval.json"
RESAMPLES = 20000
PROMOTION = -0.005          # section 4.5 operating line, a development threshold


def curve(run_dir):
    rows = [json.loads(line) for line in
            (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    return {int(r["step"]): r for r in rows if r["kind"] == "eval"}


def load_records(path):
    records = json.loads(Path(path).read_text())["records"]
    order = sorted(range(len(records)), key=lambda i: (
        records[i]["session"], records[i]["scenario"], int(records[i]["frame"])))
    keys = [(records[i]["session"], records[i]["scenario"], int(records[i]["frame"]))
            for i in order]
    return (keys,
            np.asarray([records[i]["d3"] for i in order], dtype=np.float64),
            np.asarray([records[i]["session"] for i in order]))


def probe(tag):
    path = EVAL / f"{tag}.json"
    return json.loads(path.read_text())["metrics"]["official_d3_weighted"] if path.exists() else None


def main() -> None:
    short_curve, long_curve = curve(SHORT), curve(LONG)
    long_manifest = json.loads((LONG / "manifest.json").read_text())
    if long_manifest["status"] != "completed":
        raise SystemExit(f"EXP-LONG is {long_manifest['status']}; judge it when it completes")

    short_best = min(sorted(short_curve), key=lambda s: (short_curve[s]["official_d3"], s))
    long_best = min(sorted(long_curve), key=lambda s: (long_curve[s]["official_d3"], s))
    long_terminal = max(long_curve)

    keys_s, d3_short, sessions = load_records(SHORT / "final_eval.json")
    keys_l, d3_long, _ = load_records(LONG / "final_eval.json")
    keys_r, d3_r0, _ = load_records(R0_EVAL)
    if not (keys_s == keys_l == keys_r):
        raise SystemExit("the three models are not scored on the same rows")

    unique = sorted(set(sessions.tolist()))
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(unique), size=(RESAMPLES, len(unique)))
    picks = [np.concatenate([index[unique[j]] for j in row]) for row in draws]

    def compare(base, arm):
        diff = arm - base
        means = np.asarray([diff[p].mean() for p in picks])
        per_session = {s: float(diff[index[s]].mean()) for s in unique}
        return {"delta": float(diff.mean()),
                "bootstrap_sd": float(means.std(ddof=1)),
                "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                "p_worse": float((means > 0).mean()),
                "sessions_improved": sum(1 for v in per_session.values() if v < 0),
                "sessions": len(unique), "per_session_delta": per_session}

    terminal_vs_short = compare(d3_short, d3_long)
    terminal_vs_r0 = compare(d3_r0, d3_long)

    long_terminal_d3 = long_curve[long_terminal]["official_d3"]
    short_terminal_d3 = short_curve[20554]["official_d3"]
    short_best_d3 = short_curve[short_best]["official_d3"]
    long_best_d3 = long_curve[long_best]["official_d3"]
    best_delta = long_best_d3 - short_best_d3
    consistent = terminal_vs_short["sessions_improved"] >= 7

    if best_delta <= PROMOTION and terminal_vs_short["delta"] <= PROMOTION and consistent:
        verdict = "PRIORITY_CANDIDATE_FOR_CONFIRMATION"
        action = "add the LONG checkpoint to the H confirmation alongside R0 and the short run"
    elif best_delta < 0 and terminal_vs_short["delta"] < 0:
        verdict = "SMALLER_IMPROVEMENT"
        action = ("weigh cost, per-session consistency and the need for independent "
                  "confirmation; do not discard it merely for being small")
    elif (probe(f"E1-EXP-LONG-s0_probe_step{long_terminal}") or 1) < (probe("E1-EXP_probe_step20554") or 0):
        verdict = "TRAIN_BETTER_V0_NOT_BETTER"
        action = "do not extend LONG further; keep the short expanded run"
    else:
        verdict = "NO_GAIN"
        action = "keep the short expanded run as the candidate"

    table = {
        "R0": {"v0_terminal": json.loads(R0_EVAL.read_text())["report"]["official_d3"]},
        "E1-EXP (short)": {
            "updates": 20554, "exposures_of_own_train": 3.929,
            "v0_terminal": short_terminal_d3, "v0_best": short_best_d3,
            "best_step": short_best,
            "train_probe_terminal": probe("E1-EXP_probe_step20554"),
            "vx_mae_terminal": short_curve[20554]["state_mae_vx_vy_ax_ay_yawrate"][0],
        },
        "E1-EXP-LONG-s0": {
            "updates": long_terminal, "exposures_of_own_train": 6.0001,
            "v0_terminal": long_terminal_d3, "v0_best": long_best_d3,
            "best_step": long_best,
            "train_probe_terminal": probe(f"E1-EXP-LONG-s0_probe_step{long_terminal}"),
            "vx_mae_terminal": long_curve[long_terminal]["state_mae_vx_vy_ax_ay_yawrate"][0],
        },
    }
    payload = {
        "schema_version": 1,
        "question": "does giving the expanded split six exposures beat the fixed-compute run?",
        "design_note": ("LONG sets its cosine horizon to its own total from step one, so this "
                        "is a schedule-and-budget comparison, not the causal effect of "
                        "exposure count alone"),
        "promotion_threshold": PROMOTION,
        "table": table,
        "v0_curves": {
            "E1-EXP (short)": {str(s): short_curve[s]["official_d3"] for s in sorted(short_curve)},
            "E1-EXP-LONG-s0": {str(s): long_curve[s]["official_d3"] for s in sorted(long_curve)},
        },
        "terminal_comparisons": {
            "E1-EXP -> E1-EXP-LONG": terminal_vs_short,
            "R0 -> E1-EXP-LONG": terminal_vs_r0,
        },
        "best_to_best_delta": best_delta,
        "verdict": verdict, "action": action,
        "caveats": [
            "V0 is a repeatedly used development set; this is not an independent confirmation.",
            "Single seed for the LONG arm.",
            "H remains unopened and the candidate for submission is still chosen on V0.",
        ],
    }
    (REPORTS / "long_judgement.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    with (REPORTS / "long_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "step", "v0_official_d3", "vx_mae"])
        for name, source in (("E1-EXP", short_curve), ("E1-EXP-LONG-s0", long_curve)):
            for step in sorted(source):
                writer.writerow([name, step, source[step]["official_d3"],
                                 source[step]["state_mae_vx_vy_ax_ay_yawrate"][0]])
    print(json.dumps({"verdict": verdict, "action": action, "table": table,
                      "terminal_vs_short": {k: terminal_vs_short[k] for k in
                                            ("delta", "ci95", "sessions_improved")},
                      "best_to_best_delta": best_delta}, indent=1))


if __name__ == "__main__":
    main()
