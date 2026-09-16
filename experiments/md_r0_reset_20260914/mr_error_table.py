#!/usr/bin/env python3
"""The §4.1 error table: LEN / MR-LOWDETAIL / MR-NATIVE on the same V0 rows.

Reuses the existing producers and group definitions. No new grouping is
invented here, and no share from an earlier round is carried over as if it were
this round's residual composition.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT / "experiments/md_r0_reset_20260914"))
from analyze_mr import (W, DT, REGIMES, RUNS, PROBE, EVAL, load, seg, regimes,
                        read_metric)

OUT = ROOT / "reports/md_exp_diagnosis_20260915"
# Every label the classifier can emit, so a non-empty transition group can never
# be dropped from the table by being absent from REGIMES.
ALL_LABELS = REGIMES + ["transition"]


def per_run(pred, gt, ell_g, label, state, gt_state, gt_hist, pred_hist):
    dp_p, ell_p = seg(pred)
    dp_g, _ = seg(gt)
    point_l2 = np.linalg.norm(pred - gt, axis=-1)          # [N, 6]
    d3 = point_l2 @ W
    b = ((ell_p - ell_g) / DT).mean(1)
    defined = ell_g > 0.01
    both = defined & (ell_p > 0.01)
    u_g = np.where(defined[..., None], dp_g / np.maximum(ell_g, 1e-12)[..., None], 0.0)
    u_p = np.where((ell_p > 0.01)[..., None], dp_p / np.maximum(ell_p, 1e-12)[..., None], 0.0)
    heading = np.where(both, np.degrees(np.arccos(np.clip((u_p * u_g).sum(-1), -1, 1))), np.nan)
    chord_err = ell_p - ell_g

    out = {
        "official_d3": float(d3.mean()),
        "timesteps": [
            {"k": k + 1, "t_s": round(0.5 * (k + 1), 1), "weight": float(W[k]),
             "mean_l2_m": float(point_l2[:, k].mean()),
             "weighted_contribution_m": float(W[k] * point_l2[:, k].mean())}
            for k in range(6)],
        "progress": {
            "interval_chord_mae_m": float(np.abs(chord_err).mean()),
            "interval_chord_rms_m": float(np.sqrt((chord_err ** 2).mean())),
            "mean_abs_b_ms": float(np.abs(b).mean()),
            "signed_b_ms": float(b.mean()),
            "b_rms_ms": float(np.sqrt((b ** 2).mean())),
        },
        "direction": {
            "heading_error_mean_deg": float(np.nanmean(heading)),
            "heading_error_rms_deg": float(np.sqrt(np.nanmean(heading ** 2))),
            "valid_coverage_intervals": float(both.mean()),
            "gt_defined_coverage_intervals": float(defined.mean()),
            "note": ("coverage is the share of the 6 intervals per row where both the "
                     "GT and the prediction move more than 1 cm, the only intervals "
                     "where a direction is defined for both"),
        },
        "residual": {
            "point_l2_rms_m": float(np.sqrt((point_l2 ** 2).mean())),
            "d3_rms_m": float(np.sqrt((d3 ** 2).mean())),
            "d3_p50_m": float(np.percentile(d3, 50)),
            "d3_p90_m": float(np.percentile(d3, 90)),
            "d3_p99_m": float(np.percentile(d3, 99)),
        },
        "readout_heads": {
            "target": "gt_first_interval_progress_speed",
            "target_definition": ("the GT FUTURE first-interval progress speed, "
                                  "ell_gt[:,0]/0.5 -- this is not the current vx and is "
                                  "not called one"),
            "state_slot0_mae_ms": float(np.abs(state[:, 0] - gt_state[:, 0]).mean()),
            "state_slot0_rms_ms": float(np.sqrt(((state[:, 0] - gt_state[:, 0]) ** 2).mean())),
            "state_slot0_signed_ms": float((state[:, 0] - gt_state[:, 0]).mean()),
        },
        "regimes": {},
    }
    if gt_hist is not None and pred_hist is not None:
        err = np.abs(pred_hist - gt_hist)
        out["readout_heads"]["history_mae_per_slot"] = [float(x) for x in err.mean((0, 1))]

    n = len(d3)
    for name in ALL_LABELS:
        mask = label == name
        rows = int(mask.sum())
        entry = {"rows": rows, "share_of_rows": rows / n}
        if rows:
            entry.update({
                "mean_d3_m": float(d3[mask].mean()),
                "contribution_to_overall_mean_m": float(d3[mask].sum() / n),
                "signed_b_ms": float(b[mask].mean()),
                "mean_abs_b_ms": float(np.abs(b[mask]).mean()),
                "interval_chord_mae_m": float(np.abs(chord_err[mask]).mean()),
                "heading_error_mean_deg": float(np.nanmean(heading[mask])),
                "state_slot0_mae_ms": float(np.abs(state[mask, 0] - gt_state[mask, 0]).mean()),
            })
        out["regimes"][name] = entry
    return out, d3, label


def main() -> None:
    reference = gt = label = None
    runs, d3s = {}, {}
    for name, run in RUNS.items():
        key, pred, gt_r, session, state, gt_state = load(run / "final_eval.json")
        if reference is None:
            reference, gt = key, gt_r
            _, ell_g = seg(gt)
            label = regimes(ell_g)
        elif key != reference or not np.array_equal(gt_r, gt):
            raise SystemExit(f"{name} is not scored on the same rows or ground truth")
        _, ell_g = seg(gt)
        runs[name], d3s[name], _ = per_run(pred, gt, ell_g, label, state, gt_state, None, None)

    n = len(d3s["LEN"])
    payload = {
        "schema_version": 1,
        "rows": n,
        "same_rows_and_gt_for_every_run": True,
        "group_definition": "unchanged from the existing producer (analyze_mr.regimes)",
        "runs": runs,
        "deltas_vs_LEN": {},
        "timestep_deltas_vs_LEN": {},
        "train_probe": {
            "scope": ("the fixed T0 probe only (3,456 rows of the EXISTING train split). "
                      "It is not the whole expanded train pool, so a change here is "
                      "evidence about T0 and is not called a train-wide improvement."),
            "official_d3": {name: read_metric(f"{tag}_probe_step20554")
                            for name, tag in PROBE.items()},
        },
        "tune_b1": {name: read_metric(f"{tag}_tuneB1_step20554") for name, tag in PROBE.items()},
    }

    for name in ("MR-LOWDETAIL", "MR-NATIVE"):
        groups = {}
        for g in ALL_LABELS:
            a, c = runs["LEN"]["regimes"][g], runs[name]["regimes"][g]
            if not a["rows"]:
                groups[g] = {"rows": 0}
                continue
            groups[g] = {
                "rows": a["rows"],
                "within_group_delta_d3_m": c["mean_d3_m"] - a["mean_d3_m"],
                "contribution_delta_m": c["contribution_to_overall_mean_m"]
                                        - a["contribution_to_overall_mean_m"],
                "note": ("within-group delta is the change inside the group; "
                         "contribution delta is n_g/N times it, which is what moves "
                         "the overall mean. They are not the same number."),
                "signed_b_delta_ms": c["signed_b_ms"] - a["signed_b_ms"],
            }
        payload["deltas_vs_LEN"][name] = {
            "overall_d3_delta_m": runs[name]["official_d3"] - runs["LEN"]["official_d3"],
            "groups": groups,
            "contribution_delta_sums_to_overall": sum(
                v.get("contribution_delta_m", 0.0) for v in groups.values()),
        }
        payload["timestep_deltas_vs_LEN"][name] = [
            {"t_s": runs[name]["timesteps"][k]["t_s"],
             "mean_l2_delta_m": (runs[name]["timesteps"][k]["mean_l2_m"]
                                 - runs["LEN"]["timesteps"][k]["mean_l2_m"]),
             "weighted_contribution_delta_m": (runs[name]["timesteps"][k]["weighted_contribution_m"]
                                               - runs["LEN"]["timesteps"][k]["weighted_contribution_m"])}
            for k in range(6)]

    (OUT / "mr_error_table.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    names = list(RUNS)
    with (OUT / "mr_error_table_regimes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "rows", "share_of_rows"]
                   + [f"{n_}_{f}" for n_ in names for f in
                      ("mean_d3_m", "contribution_m", "signed_b_ms", "mean_abs_b_ms",
                       "chord_mae_m", "heading_deg", "state_slot0_mae_ms")])
        for g in ALL_LABELS:
            base = runs["LEN"]["regimes"][g]
            row = [g, base["rows"], round(base["share_of_rows"], 6)]
            for n_ in names:
                e = runs[n_]["regimes"][g]
                row += ([""] * 7 if not e["rows"] else
                        [round(e["mean_d3_m"], 6), round(e["contribution_to_overall_mean_m"], 6),
                         round(e["signed_b_ms"], 6), round(e["mean_abs_b_ms"], 6),
                         round(e["interval_chord_mae_m"], 6), round(e["heading_error_mean_deg"], 6),
                         round(e["state_slot0_mae_ms"], 6)])
            w.writerow(row)

    with (OUT / "mr_error_table_timesteps.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "weight"] + [f"{n_}_{f}" for n_ in names
                                        for f in ("mean_l2_m", "weighted_contribution_m")])
        for k in range(6):
            row = [runs["LEN"]["timesteps"][k]["t_s"], round(float(W[k]), 6)]
            for n_ in names:
                e = runs[n_]["timesteps"][k]
                row += [round(e["mean_l2_m"], 6), round(e["weighted_contribution_m"], 6)]
            w.writerow(row)

    for n_ in names:
        r = runs[n_]
        print(f"{n_:14s} D3={r['official_d3']:.6f}  chordMAE={r['progress']['interval_chord_mae_m']:.5f}  "
              f"|b|={r['progress']['mean_abs_b_ms']:.5f}  head={r['direction']['heading_error_mean_deg']:.4f}deg  "
              f"cov={r['direction']['valid_coverage_intervals']:.4f}  "
              f"state0MAE={r['readout_heads']['state_slot0_mae_ms']:.4f}")
    print("\ngroup rows:", {g: runs["LEN"]["regimes"][g]["rows"] for g in ALL_LABELS},
          "sum:", sum(runs["LEN"]["regimes"][g]["rows"] for g in ALL_LABELS), "of", n)
    for name in ("MR-LOWDETAIL", "MR-NATIVE"):
        d = payload["deltas_vs_LEN"][name]
        print(f"\n{name}: overall {d['overall_d3_delta_m']:+.6f}  "
              f"contribution sum {d['contribution_delta_sums_to_overall']:+.6f}")
    print("\nwrote mr_error_table.json, mr_error_table_regimes.csv, mr_error_table_timesteps.csv")


if __name__ == "__main__":
    main()
