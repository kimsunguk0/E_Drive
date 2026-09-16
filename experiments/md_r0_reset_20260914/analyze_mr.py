#!/usr/bin/env python3
"""Judge the MR pair on the criteria fixed before the result.

Primary comparison: MR-NATIVE minus MR-LOWDETAIL, which isolates native detail.
Practical comparison: each MR against LEN, which includes the matching-graph
change. If both MR arms improve, that is evidence about the graph, not about
native detail.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
WORK = ROOT / "work_dirs/md_r0_reset_20260914"
EVAL = ROOT / "reports/md_r0_reset_20260914/eval"
OUT = ROOT / "reports/md_exp_diagnosis_20260915"
W = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0
DT = 0.5
RESAMPLES = 20000
REGIMES = ["stop_hold", "departing", "constant", "decelerating", "accelerating"]
BANDS = [0.0, 0.5, 2.0, 5.0, 8.0, 11.0, 14.0, np.inf]
RUNS = {"LEN": WORK / "E1-EXP-LEN-s0", "MR-LOWDETAIL": WORK / "MR-LOWDETAIL-s0",
        "MR-NATIVE": WORK / "MR-NATIVE-s0"}
PROBE = {"LEN": "E1-EXP-LEN-s0", "MR-LOWDETAIL": "MR-LOWDETAIL-s0", "MR-NATIVE": "MR-NATIVE-s0"}


def load(path):
    records = json.loads(Path(path).read_text())["records"]
    order = sorted(range(len(records)), key=lambda i: (
        records[i]["session"], records[i]["scenario"], int(records[i]["frame"])))
    records = [records[i] for i in order]
    return ([(r["session"], r["scenario"], int(r["frame"])) for r in records],
            np.asarray([r["pred_abs_xy"] for r in records], dtype=np.float64),
            np.asarray([r["gt_abs_xy"] for r in records], dtype=np.float64),
            np.asarray([r["session"] for r in records]),
            np.asarray([r["pred_state"] for r in records], dtype=np.float64),
            np.asarray([r["gt_state"] for r in records], dtype=np.float64))


def seg(p):
    start = np.concatenate([np.zeros((len(p), 1, 2)), p[:, :-1]], axis=1)
    dp = p - start
    return dp, np.linalg.norm(dp, axis=-1)


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


def read_metric(tag, field="official_d3_weighted"):
    path = EVAL / f"{tag}.json"
    return json.loads(path.read_text())["metrics"][field] if path.exists() else None


def summarise(pred, gt, ell_g, label, v0, state, gt_state):
    dp_p, ell_p = seg(pred)
    dp_g, _ = seg(gt)
    d3 = np.linalg.norm(pred - gt, axis=-1) @ W
    b = ((ell_p - ell_g) / DT).mean(1)
    defined = ell_g > 0.01
    both = defined & (ell_p > 0.01)
    u_g = np.where(defined[..., None], dp_g / np.maximum(ell_g, 1e-12)[..., None], 0.0)
    u_p = np.where((ell_p > 0.01)[..., None], dp_p / np.maximum(ell_p, 1e-12)[..., None], 0.0)
    heading = np.where(both, np.degrees(np.arccos(np.clip((u_p * u_g).sum(-1), -1, 1))), np.nan)
    out = {
        "official_d3": float(d3.mean()),
        "point_l2": [float(np.linalg.norm(pred[:, k] - gt[:, k], axis=-1).mean()) for k in range(6)],
        "interval_chord_mae_m": float(np.abs(ell_p - ell_g).mean()),
        "mean_abs_b_ms": float(np.abs(b).mean()), "signed_b_ms": float(b.mean()),
        "heading_error_mean_deg": float(np.nanmean(heading)),
        "heading_common_mask_coverage": float(both.mean()),
        "state_vx_mae_ms": float(np.abs(state[:, 0] - gt_state[:, 0]).mean()),
        "regimes": {}, "speed_bands": {},
    }
    for name in REGIMES:
        mask = label == name
        if mask.any():
            out["regimes"][name] = {
                "rows": int(mask.sum()), "mean_d3_m": float(d3[mask].mean()),
                "contribution_to_overall_mean": float(d3[mask].sum() / len(d3)),
                "signed_b_ms": float(b[mask].mean()),
                "mean_abs_b_ms": float(np.abs(b[mask]).mean())}
    for low, high in zip(BANDS[:-1], BANDS[1:]):
        mask = (v0 >= low) & (v0 < high)
        if mask.any():
            out["speed_bands"][f"{low:g}-{'inf' if np.isinf(high) else f'{high:g}'}"] = {
                "rows": int(mask.sum()), "mean_d3_m": float(d3[mask].mean()),
                "signed_b_ms": float(b[mask].mean())}
    return out, d3


def main() -> None:
    for name, run in RUNS.items():
        status = json.loads((run / "manifest.json").read_text())["status"]
        if status != "completed":
            raise SystemExit(f"{name} is {status}; judge when every arm completes")

    reference, gt, sessions, label, v0 = None, None, None, None, None
    summaries, d3s = {}, {}
    for name, run in RUNS.items():
        key, pred, gt_r, session, state, gt_state = load(run / "final_eval.json")
        if reference is None:
            reference, gt, sessions = key, gt_r, session
            _, ell_g = seg(gt)
            label, v0 = regimes(ell_g), ell_g[:, 0] / DT
        elif key != reference or not np.array_equal(gt_r, gt):
            raise SystemExit(f"{name} is not scored on the same rows or ground truth")
        _, ell_g = seg(gt)
        summaries[name], d3s[name] = summarise(pred, gt, ell_g, label, v0, state, gt_state)

    unique = sorted(set(sessions.tolist()))
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(unique), size=(RESAMPLES, len(unique)))
    picks = [np.concatenate([index[unique[j]] for j in row]) for row in draws]

    def compare(base, arm):
        diff = d3s[arm] - d3s[base]
        means = np.asarray([diff[p].mean() for p in picks])
        return {"delta": float(diff.mean()),
                "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                "p_worse": float((means > 0).mean()),
                "sessions_improved": sum(1 for s in unique if diff[index[s]].mean() < 0),
                "sessions": len(unique)}

    comparisons = {
        "PRIMARY MR-LOWDETAIL -> MR-NATIVE": compare("MR-LOWDETAIL", "MR-NATIVE"),
        "LEN -> MR-NATIVE": compare("LEN", "MR-NATIVE"),
        "LEN -> MR-LOWDETAIL": compare("LEN", "MR-LOWDETAIL"),
    }
    probes = {name: read_metric(f"{tag}_probe_step20554") for name, tag in PROBE.items()}
    b1 = {name: read_metric(f"{tag}_tuneB1_step20554") for name, tag in PROBE.items()}
    curves = {name: {str(s): c["official_d3"] for s, c in sorted(curve(run).items())}
              for name, run in RUNS.items()}
    best = {name: min(sorted(curve(run)), key=lambda s: (curve(run)[s]["official_d3"], s))
            for name, run in RUNS.items()}

    primary = comparisons["PRIMARY MR-LOWDETAIL -> MR-NATIVE"]
    native_vs_len = comparisons["LEN -> MR-NATIVE"]
    lowdetail_vs_len = comparisons["LEN -> MR-LOWDETAIL"]
    native_beats_both = primary["delta"] < 0 and primary["ci95"][1] < 0 and native_vs_len["delta"] < 0
    both_improve = native_vs_len["delta"] < 0 and lowdetail_vs_len["delta"] < 0
    train_only = (probes["MR-NATIVE"] is not None and probes["LEN"] is not None
                  and probes["MR-NATIVE"] < probes["LEN"] and native_vs_len["delta"] >= 0)

    if native_beats_both:
        verdict = "NATIVE_DETAIL_HELPS"
        action = "native detail earns its own credit; confirm with a second seed before adoption"
    elif both_improve:
        verdict = "MATCHING_GRAPH_HELPS_DETAIL_UNPROVEN"
        action = ("both arms beat LEN while native does not beat lowdetail, so the gain belongs "
                  "to the finer matching graph and not to native detail; keep the graph change "
                  "as the candidate direction and do not claim a detail effect")
    elif train_only:
        verdict = "TRAIN_ONLY"
        action = "not adopted as a generalization gain"
    else:
        verdict = "NO_EFFECT_UNDER_THIS_SETTING"
        action = ("record as a failure of this matching graph and training setting; do not "
                  "extend it to a limit of video speed estimation")

    payload = {
        "schema_version": 1,
        "criteria": "reports/md_exp_diagnosis_20260915/MATCHING_REVIEW_RESPONSE.md section 4",
        "terminal_d3": {name: curves[name]["20554"] for name in RUNS},
        "best_d3": {name: {"step": best[name], "d3": curve(RUNS[name])[best[name]]["official_d3"]}
                    for name in RUNS},
        "v0_curves": curves,
        "t0_probe_d3": probes, "tune_b1_d3": b1,
        "comparisons": comparisons,
        "summaries": summaries,
        "verdict": verdict, "action": action,
        "reading_rules": [
            "the primary comparison is native minus lowdetail; LEN comparisons include the graph change",
            "if both MR arms beat LEN the credit goes to the matching graph, not to native detail",
            "constant is the residual label, not a strict constant-speed group",
            "speed bands are gt_first_interval_progress_speed, a GT future quantity",
        ],
        "caveats": ["V0 is a repeatedly used development set", "single seed per arm",
                    "H remains unopened"],
    }
    (OUT / "mr_judgement.json").write_text(json.dumps(payload, indent=1, sort_keys=True,
                                                      default=float) + "\n")
    with (OUT / "mr_regimes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["regime", "rows"] + [f"{n}_{f}" for n in RUNS
                                              for f in ("d3", "contribution", "signed_b")])
        for name in REGIMES:
            if name not in summaries["LEN"]["regimes"]:
                continue
            row = [name, summaries["LEN"]["regimes"][name]["rows"]]
            for run in RUNS:
                entry = summaries[run]["regimes"][name]
                row += [entry["mean_d3_m"], entry["contribution_to_overall_mean"],
                        entry["signed_b_ms"]]
            writer.writerow(row)
    print(json.dumps({"verdict": verdict, "action": action,
                      "terminal_d3": payload["terminal_d3"], "best_d3": payload["best_d3"],
                      "t0_probe_d3": probes, "tune_b1_d3": b1,
                      "comparisons": comparisons,
                      "heading_deg": {n: summaries[n]["heading_error_mean_deg"] for n in RUNS},
                      "mean_abs_b": {n: summaries[n]["mean_abs_b_ms"] for n in RUNS},
                      "state_vx_mae": {n: summaries[n]["state_vx_mae_ms"] for n in RUNS}},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
