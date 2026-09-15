#!/usr/bin/env python3
"""Judge E1-EXP-LEN against E1-EXP using the criteria fixed before the result.

The five regimes stay separate because accelerating and decelerating carry
opposite signed biases that cancel when merged.  The verdict is driven by the
actual XY D3, never by the auxiliary value or by the common-component share,
whose denominator moves.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
WORK = ROOT / "work_dirs/md_r0_reset_20260914"
OUT = ROOT / "reports/md_exp_diagnosis_20260915"
CONTROL, ARM = WORK / "E1-EXP", WORK / "E1-EXP-LEN-s0"
W = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0
DT = 0.5
RESAMPLES = 20000
REGIMES = ["stop_hold", "departing", "constant", "decelerating", "accelerating"]
SPEED_EDGES = [0.0, 0.5, 2.0, 5.0, 8.0, 11.0, 14.0, np.inf]


def load(path):
    records = json.loads(Path(path).read_text())["records"]
    order = sorted(range(len(records)), key=lambda i: (
        records[i]["session"], records[i]["scenario"], int(records[i]["frame"])))
    records = [records[i] for i in order]
    return {
        "key": [(r["session"], r["scenario"], int(r["frame"])) for r in records],
        "pred": np.asarray([r["pred_abs_xy"] for r in records], dtype=np.float64),
        "gt": np.asarray([r["gt_abs_xy"] for r in records], dtype=np.float64),
        "session": np.asarray([r["session"] for r in records]),
        "state": np.asarray([r["pred_state"] for r in records], dtype=np.float64),
        "gt_state": np.asarray([r["gt_state"] for r in records], dtype=np.float64),
    }


def seg(p):
    start = np.concatenate([np.zeros((len(p), 1, 2)), p[:, :-1]], axis=1)
    dp = p - start
    return dp, np.linalg.norm(dp, axis=-1)


def regimes(ell_g):
    total, vbar = ell_g.sum(1), ell_g / DT
    change, dv = vbar[:, -1] - vbar[:, 0], np.diff(vbar, axis=1)
    has_decel, has_accel = (dv < -0.5).any(1), (dv > 0.5).any(1)
    label = np.full(len(total), "constant", dtype=object)
    label[(change <= -1.0) & ~has_accel] = "decelerating"
    label[(change >= 1.0) & ~has_decel] = "accelerating"
    label[has_decel & has_accel] = "transition"
    label[(total > 1.0) & (vbar[:, 0] < 1.0)] = "departing"
    label[total <= 1.0] = "stop_hold"
    return label


def curve(run_dir):
    rows = [json.loads(line) for line in
            (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    return {int(r["step"]): r for r in rows if r["kind"] == "eval"}


def summarise(model, gt, ell_g, label, v0):
    dp_p, ell_p = seg(model)
    dp_g, _ = seg(gt)
    d3 = np.linalg.norm(model - gt, axis=-1) @ W
    b = ((ell_p - ell_g) / DT).mean(axis=1)
    defined = ell_g > 0.01
    with np.errstate(invalid="ignore", divide="ignore"):
        u_g = np.where(defined[..., None], dp_g / np.maximum(ell_g, 1e-12)[..., None], 0.0)
        u_p = np.where((ell_p > 0.01)[..., None],
                       dp_p / np.maximum(ell_p, 1e-12)[..., None], 0.0)
    both = defined & (ell_p > 0.01)
    dot = np.clip((u_p * u_g).sum(-1), -1, 1)
    heading = np.where(both, np.degrees(np.arccos(dot)), np.nan)
    out = {
        "official_d3": float(d3.mean()),
        "point_l2": [float(np.linalg.norm(model[:, k] - gt[:, k], axis=-1).mean())
                     for k in range(6)],
        "interval_chord_mae_m": float(np.abs(ell_p - ell_g).mean()),
        "mean_abs_b_ms": float(np.abs(b).mean()),
        "signed_b_ms": float(b.mean()),
        "heading_error_mean_deg": float(np.nanmean(heading)),
        "regimes": {}, "speed_bands": {},
    }
    for name in REGIMES:
        mask = label == name
        if not mask.any():
            continue
        out["regimes"][name] = {
            "rows": int(mask.sum()), "mean_d3_m": float(d3[mask].mean()),
            "share_of_total_d3": float(d3[mask].sum() / d3.sum()),
            "signed_b_ms": float(b[mask].mean()),
            "mean_abs_b_ms": float(np.abs(b[mask]).mean()),
            "interval_chord_mae_m": float(np.abs(ell_p[mask] - ell_g[mask]).mean()),
        }
    for low, high in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:]):
        mask = (v0 >= low) & (v0 < high)
        if not mask.any():
            continue
        key = f"{low:g}-{'inf' if np.isinf(high) else f'{high:g}'}"
        out["speed_bands"][key] = {
            "rows": int(mask.sum()), "mean_d3_m": float(d3[mask].mean()),
            "signed_b_ms": float(b[mask].mean()),
            "mean_total_chord_error_m": float(3.0 * b[mask].mean()),
        }
    return out, d3


def main() -> None:
    manifest = json.loads((ARM / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise SystemExit(f"E1-EXP-LEN is {manifest['status']}; judge it when it completes")

    control, arm = load(CONTROL / "final_eval.json"), load(ARM / "final_eval.json")
    if control["key"] != arm["key"]:
        raise SystemExit("the two runs are not scored on the same rows")
    gt = control["gt"]
    if not np.array_equal(gt, arm["gt"]):
        raise SystemExit("ground truth differs between the two runs")
    _, ell_g = seg(gt)
    label, v0 = regimes(ell_g), ell_g[:, 0] / DT

    control_summary, d3_control = summarise(control["pred"], gt, ell_g, label, v0)
    arm_summary, d3_arm = summarise(arm["pred"], gt, ell_g, label, v0)

    sessions = control["session"]
    unique = sorted(set(sessions.tolist()))
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(unique), size=(RESAMPLES, len(unique)))
    picks = [np.concatenate([index[unique[j]] for j in row]) for row in draws]
    diff = d3_arm - d3_control
    means = np.asarray([diff[p].mean() for p in picks])
    paired = {
        "delta": float(diff.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "p_worse": float((means > 0).mean()),
        "sessions_improved": sum(1 for s in unique if diff[index[s]].mean() < 0),
        "sessions": len(unique),
    }

    arm_curve, control_curve = curve(ARM), curve(CONTROL)
    best_step = min(sorted(arm_curve), key=lambda s: (arm_curve[s]["official_d3"], s))
    control_best = min(sorted(control_curve), key=lambda s: (control_curve[s]["official_d3"], s))

    improved_overall = paired["delta"] < 0 and paired["ci95"][1] < 0
    constant_better = (arm_summary["regimes"]["constant"]["mean_d3_m"]
                       < control_summary["regimes"]["constant"]["mean_d3_m"])
    stop_depart_better = all(
        abs(arm_summary["regimes"][name]["signed_b_ms"])
        < abs(control_summary["regimes"][name]["signed_b_ms"])
        for name in ("stop_hold", "departing") if name in arm_summary["regimes"])
    heading_worse = (arm_summary["heading_error_mean_deg"]
                     > control_summary["heading_error_mean_deg"] * 1.05)

    if improved_overall and constant_better:
        verdict, action = "IMPROVES_INCLUDING_NORMAL_DRIVING", \
            "confirm with a second seed before considering adoption"
    elif improved_overall:
        verdict, action = "IMPROVES_BUT_NOT_IN_CONSTANT", \
            "credit the gain, keep the normal-driving motion question open"
    elif heading_worse:
        verdict, action = "DIRECTION_REGRESSED", \
            "record as a side effect of the length term and keep the existing EXP"
    else:
        verdict, action = "NO_EFFECT_UNDER_THIS_SETTING", \
            ("close this loss setting without raising lambda; move to the recent-observation "
             "and matching-resolution contrast, one variable at a time")

    payload = {
        "schema_version": 1,
        "criteria": "reports/md_exp_diagnosis_20260915/LENGTH_AUX_CRITERIA.md",
        "control": {"run": "E1-EXP", "terminal_d3": control_curve[20554]["official_d3"],
                    "best_step": control_best,
                    "best_d3": control_curve[control_best]["official_d3"],
                    "summary": control_summary},
        "arm": {"run": "E1-EXP-LEN-s0", "lambda": 0.25,
                "terminal_d3": arm_curve[20554]["official_d3"],
                "best_step": best_step, "best_d3": arm_curve[best_step]["official_d3"],
                "summary": arm_summary},
        "v0_curves": {"control": {str(s): control_curve[s]["official_d3"] for s in sorted(control_curve)},
                      "arm": {str(s): arm_curve[s]["official_d3"] for s in sorted(arm_curve)}},
        "paired_terminal": paired,
        "regime_deltas": {
            name: {
                "d3_delta": arm_summary["regimes"][name]["mean_d3_m"]
                            - control_summary["regimes"][name]["mean_d3_m"],
                "signed_b_control": control_summary["regimes"][name]["signed_b_ms"],
                "signed_b_arm": arm_summary["regimes"][name]["signed_b_ms"],
                "abs_signed_b_reduced": abs(arm_summary["regimes"][name]["signed_b_ms"])
                                        < abs(control_summary["regimes"][name]["signed_b_ms"]),
            } for name in REGIMES if name in arm_summary["regimes"]},
        "verdict": verdict, "action": action,
        "reading_rules": [
            "the verdict is driven by the actual XY D3, not by the auxiliary value",
            "the common-component share is not used as a success signal because its "
            "denominator moves; absolute magnitudes are compared instead",
            "accelerating and decelerating are never merged here",
        ],
        "caveats": [
            "V0 is a repeatedly used development set and this is a single seed",
            "H remains unopened",
        ],
    }
    (OUT / "length_aux_judgement.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n")
    with (OUT / "length_aux_regimes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["regime", "rows", "control_d3", "arm_d3", "d3_delta",
                         "control_signed_b", "arm_signed_b", "control_abs_b", "arm_abs_b"])
        for name in REGIMES:
            if name not in arm_summary["regimes"]:
                continue
            c, a = control_summary["regimes"][name], arm_summary["regimes"][name]
            writer.writerow([name, c["rows"], c["mean_d3_m"], a["mean_d3_m"],
                             a["mean_d3_m"] - c["mean_d3_m"], c["signed_b_ms"],
                             a["signed_b_ms"], c["mean_abs_b_ms"], a["mean_abs_b_ms"]])
    print(json.dumps({
        "verdict": verdict, "action": action,
        "control_terminal": payload["control"]["terminal_d3"],
        "arm_terminal": payload["arm"]["terminal_d3"],
        "arm_best": payload["arm"]["best_d3"], "arm_best_step": best_step,
        "paired": paired,
        "regime_deltas": payload["regime_deltas"],
        "mean_abs_b": {"control": control_summary["mean_abs_b_ms"],
                       "arm": arm_summary["mean_abs_b_ms"]},
        "heading_deg": {"control": control_summary["heading_error_mean_deg"],
                        "arm": arm_summary["heading_error_mean_deg"]},
    }, indent=1, default=float))


if __name__ == "__main__":
    main()
