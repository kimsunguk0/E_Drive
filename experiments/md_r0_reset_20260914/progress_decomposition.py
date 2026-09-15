#!/usr/bin/env python3
"""Review section 5.1: split the interval progress-speed error into a per-sample
common component and a time-varying residual.

    delta_v[n,k] = 2 * (ell_pred[n,k] - ell_gt[n,k])     # 0.5 s intervals
    b[n]         = mean_k delta_v[n,k]
    r[n,k]       = delta_v[n,k] - b[n]
    sum delta_v^2 = 6 * sum b^2 + sum r^2                # exact for equal intervals

This is a decomposition of the error signal.  It is not a causal attribution and
not a sensor v0 diagnosis.  b[n] needs future GT to compute and is never a model
input or an inference-time correction.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
OUT = ROOT / "reports/md_exp_diagnosis_20260915"
EXP_EVAL = ROOT / "work_dirs/md_r0_reset_20260914/E1-EXP/final_eval.json"
DT = 0.5


def segments(p):
    start = np.concatenate([np.zeros((len(p), 1, 2)), p[:, :-1]], axis=1)
    return np.linalg.norm(p - start, axis=-1)


def regimes(ell_g):
    total = ell_g.sum(1)
    vbar = ell_g / DT
    change = vbar[:, -1] - vbar[:, 0]
    dv = np.diff(vbar, axis=1)
    has_decel, has_accel = (dv < -0.5).any(1), (dv > 0.5).any(1)
    label = np.full(len(total), "constant", dtype=object)
    label[(change <= -1.0) & ~has_accel] = "decelerating"
    label[(change >= 1.0) & ~has_decel] = "accelerating"
    label[has_decel & has_accel] = "transition"
    label[(total > 1.0) & (vbar[:, 0] < 1.0)] = "departing"
    label[total <= 1.0] = "stop_hold"
    return label


def decompose(delta_v, sessions):
    b = delta_v.mean(axis=1)
    r = delta_v - b[:, None]
    total_energy = float((delta_v ** 2).sum())
    common_energy = float(6 * (b ** 2).sum())
    residual_energy = float((r ** 2).sum())
    large_common = np.abs(b) >= np.percentile(np.abs(b), 90)
    large_residual = np.sqrt((r ** 2).mean(1)) >= np.percentile(np.sqrt((r ** 2).mean(1)), 90)
    return {
        "rows": int(len(delta_v)),
        "sessions": int(len(set(sessions.tolist()))),
        "identity_residual": abs(total_energy - common_energy - residual_energy),
        "total_energy": total_energy,
        "common_component_energy_share": common_energy / total_energy,
        "residual_component_energy_share": residual_energy / total_energy,
        "mean_abs_b_ms": float(np.abs(b).mean()),
        "signed_mean_b_ms": float(b.mean()),
        "mean_total_chord_error_m": float(3.0 * b.mean()),
        "mean_abs_total_chord_error_m": float(3.0 * np.abs(b).mean()),
        "rms_residual_ms": float(np.sqrt((r ** 2).mean())),
        "large_common_rows": int(large_common.sum()),
        "large_common_sessions": int(len(set(sessions[large_common].tolist()))),
        "large_residual_rows": int(large_residual.sum()),
        "large_residual_sessions": int(len(set(sessions[large_residual].tolist()))),
        "rows_in_both_tails": int((large_common & large_residual).sum()),
    }


def main() -> None:
    records = json.loads(EXP_EVAL.read_text())["records"]
    pred = np.asarray([r["pred_abs_xy"] for r in records], dtype=np.float64)
    gt = np.asarray([r["gt_abs_xy"] for r in records], dtype=np.float64)
    sessions = np.asarray([r["session"] for r in records])

    ell_p, ell_g = segments(pred), segments(gt)
    delta_v = (ell_p - ell_g) / DT
    label = regimes(ell_g)

    # The five regimes stay separate: merging accelerating with decelerating
    # cancels two opposite signed biases and hides both.
    groups = {
        "stop_hold": label == "stop_hold",
        "departing": label == "departing",
        "constant": label == "constant",
        "decelerating": label == "decelerating",
        "accelerating": label == "accelerating",
        "all": np.ones(len(delta_v), dtype=bool),
        "SUMMARY_ONLY_stop_or_depart": np.isin(label, ["stop_hold", "departing"]),
        "SUMMARY_ONLY_accel_or_decel": np.isin(label, ["accelerating", "decelerating"]),
    }
    results = {name: decompose(delta_v[mask], sessions[mask])
               for name, mask in groups.items() if mask.any()}

    correlation = np.corrcoef(delta_v.T)
    payload = {
        "schema_version": 1,
        "definition": {
            "delta_v": "2 * (predicted interval chord length - GT interval chord length), m/s",
            "b": "per-sample mean of delta_v over the six intervals",
            "r": "delta_v minus b, the time-varying residual",
            "identity": "sum delta_v^2 = 6 * sum b^2 + sum r^2, exact for equal intervals",
        },
        "groups": results,
        "signed_delta_v_correlation_6x6": correlation.tolist(),
        "mean_correlation_offdiagonal": float(
            (correlation.sum() - np.trace(correlation)) / (36 - 6)),
        "merged_groups_are_summary_only": (
            "accelerating and decelerating carry opposite signed biases that cancel when "
            "merged, and stop_hold and departing differ in magnitude; the five regimes are "
            "the reporting unit and the SUMMARY_ONLY rows exist only for continuity"),
        "reading": ("a high common-component share means the interval errors of one sample "
                    "mostly move together, which is consistent with a persistent progress "
                    "offset within that sample; it is not evidence about which input caused it"),
        "not_claimed": [
            "this is not a causal attribution and not a sensor v0 diagnosis",
            "b needs future GT and is never a model input or an inference-time correction",
            "b is an error component of a 3 s prediction against future GT, not a measurement "
            "of the current v0",
            "the common-component SHARE has a moving denominator: it can rise even when the "
            "absolute common component falls, so judge on absolute D3, not on the ratio",
            "a chord length carries no direction, so an over-large predicted length is an "
            "excessive predicted travel distance, not necessarily forward drift",
            "a common component does not imply a single global speed correction would help, "
            "because the signed mean of b across samples is near zero",
        ],
    }
    (OUT / "progress_decomposition.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n")
    with (OUT / "progress_decomposition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group"] + list(results["all"]))
        writer.writeheader()
        for name, values in results.items():
            writer.writerow({"group": name, **values})
    print(json.dumps({name: {k: (round(v, 6) if isinstance(v, float) else v)
                             for k, v in values.items()}
                      for name, values in results.items()}, indent=1))
    print("mean off-diagonal correlation of signed delta_v:",
          round(payload["mean_correlation_offdiagonal"], 4))


if __name__ == "__main__":
    main()
