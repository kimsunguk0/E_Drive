#!/usr/bin/env python3
"""Apply the 2026-09-16 integrated review: definition fixes, verified claims,
and the development-best / deployment-fallback split.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
OUT = ROOT / "reports/md_exp_diagnosis_20260915"
WORK = ROOT / "work_dirs/md_r0_reset_20260914"
W = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0
DT = 0.5

DEFINITIONS = {
    "speed_band_field": {
        "old_name": "v0",
        "correct_name": "gt_first_interval_progress_speed",
        "definition": "GT chord length of the FUTURE [0, 0.5] s interval divided by 0.5",
        "is_not": ("the causal-quadratic-fit current vx the state head predicts, and not a "
                   "sensor reading; the earlier phrase 'v0 < 0.5 m/s, 131 rows' means "
                   "'GT future first-interval progress speed below 0.5 m/s'"),
        "unaffected": ("the separate observation that the history readout carries a flat ~8.3 "
                       "percent relative error stands on its own and is not changed by this"),
    },
    "stop_hold": {
        "rule": "total GT chord over the 3 s horizon <= 1.0 m",
        "is_not": "an exact three-second standstill label; it admits creep",
    },
    "constant": {
        "rule": "the default label for rows no other rule claimed",
        "is_not": "a zero-acceleration label",
    },
    "signed_b_vs_scatter": {
        "observed": ("in seed 1 the decelerating group's |mean b| fell further than in seed 0 "
                     "while its D3 rose"),
        "not_established": ("that the variance increased. b is a 3 s total-chord error over 3 "
                            "and D3 is a weighted distance over six positions; a redistribution "
                            "in time, a direction change or cancellation between samples could "
                            "each produce this. Scatter growth is a hypothesis, not a measurement"),
    },
    "best_equals_terminal": {
        "note": ("best coinciding with terminal removes the extra optimism of picking an "
                 "interior checkpoint; it does not remove the development-set bias of having "
                 "chosen the budget and the recipe on V0"),
    },
}


def load(path):
    records = json.loads(Path(path).read_text())["records"]
    order = sorted(range(len(records)), key=lambda i: (
        records[i]["session"], records[i]["scenario"], int(records[i]["frame"])))
    records = [records[i] for i in order]
    return (np.asarray([r["pred_abs_xy"] for r in records], dtype=np.float64),
            np.asarray([r["gt_abs_xy"] for r in records], dtype=np.float64))


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


def main() -> None:
    pred_c, gt = load(WORK / "E1-EXP/final_eval.json")
    pred_a, _ = load(WORK / "E1-EXP-LEN-s0/final_eval.json")
    ell_g = seg(gt)
    label = regimes(ell_g)
    v0 = ell_g[:, 0] / DT
    d3_c = np.linalg.norm(pred_c - gt, axis=-1) @ W
    d3_a = np.linalg.norm(pred_a - gt, axis=-1) @ W
    b_c = ((seg(pred_c) - ell_g) / DT).mean(1)
    b_a = ((seg(pred_a) - ell_g) / DT).mean(1)

    # 1. group-internal delta versus contribution to the overall mean
    contributions = []
    for name in ["stop_hold", "departing", "constant", "decelerating", "accelerating"]:
        mask = label == name
        contributions.append({
            "regime": name, "rows": int(mask.sum()),
            "group_internal_d3_delta": float(d3_a[mask].mean() - d3_c[mask].mean()),
            "contribution_to_overall_mean_delta": float((d3_a[mask] - d3_c[mask]).sum() / len(d3_c)),
            "abs_mean_b_control": float(np.abs(b_c[mask]).mean()),
            "abs_mean_b_arm": float(np.abs(b_a[mask]).mean()),
            "abs_mean_b_rose": bool(np.abs(b_a[mask]).mean() > np.abs(b_c[mask]).mean()),
            "signed_mean_b_control": float(b_c[mask].mean()),
            "signed_mean_b_arm": float(b_a[mask].mean()),
        })
    total_check = sum(r["contribution_to_overall_mean_delta"] for r in contributions)

    # 2. speed bands, including the fast band the review says worsened
    bands = []
    for low, high in zip([0.0, 0.5, 2.0, 5.0, 8.0, 11.0, 14.0],
                         [0.5, 2.0, 5.0, 8.0, 11.0, 14.0, np.inf]):
        mask = (v0 >= low) & (v0 < high)
        if not mask.any():
            continue
        bands.append({
            "band": f"{low:g}-{'inf' if np.isinf(high) else f'{high:g}'}",
            "rows": int(mask.sum()),
            "control_d3": float(d3_c[mask].mean()), "arm_d3": float(d3_a[mask].mean()),
            "d3_delta": float(d3_a[mask].mean() - d3_c[mask].mean()),
        })

    payload = {
        "schema_version": 1,
        "purpose": "definition corrections and independently verified review claims",
        "definitions": DEFINITIONS,
        "regime_contributions_seed0": {
            "rows": contributions,
            "sum_of_contributions": total_check,
            "note": ("group-internal delta and contribution to the overall mean are different "
                     "quantities; only the contributions sum to the total"),
        },
        "speed_band_deltas_seed0": {
            "rows": bands,
            "field": "gt_first_interval_progress_speed",
            "note": ("the five regimes all improved, but not every condition did; the fast band "
                     "is the counterexample"),
        },
        "verified_review_claims": {
            "decelerating_abs_mean_b_rose_seed0": next(
                r["abs_mean_b_rose"] for r in contributions if r["regime"] == "decelerating"),
            "fast_band_worsened_seed0": next(
                r["d3_delta"] > 0 for r in bands if r["band"] == "14-inf"),
            "contribution_sum_matches_total": abs(total_check - float(d3_a.mean() - d3_c.mean())) < 1e-12,
        },
    }
    (OUT / "len_review_corrections.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n")
    with (OUT / "len_regime_contributions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contributions[0]))
        writer.writeheader()
        writer.writerows(contributions)

    # 3. role split, without touching the existing candidate entry
    candidate = json.loads((ROOT / "reports/md_r0_reset_20260914/candidate_registry.json").read_text())
    roles = {
        "schema_version": 1,
        "DEPLOY_FALLBACK": {
            "run": "E1-EXP terminal (seed 0)",
            "checkpoint": candidate["checkpoint_path"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "v0_d3": candidate["results"]["v0_b4_official_d3"],
            "status": ("fully prepared: submission file for all 1,125 clips, RTX 4090 timing, "
                       "raw B1 parity, negative controls, reproduction manifest"),
        },
        "DEV_BEST": {
            "run": "E1-EXP-LEN-s0 terminal",
            "checkpoint": str(WORK / "E1-EXP-LEN-s0/ckpt_step20554.pth"),
            "v0_d3": float(d3_a.mean()),
            "replicated_at_seed1": True,
            "pooled_delta_vs_fallback": -0.001715,
            "status": "lowest V0 seen; deployment artifacts NOT yet built for it",
            "to_promote_it": [
                "build the 1,125-clip submission with the same writer and record its SHA",
                "re-run raw B1 input and output parity on the new weights",
                "re-run the RTX 4090 forward measurement",
                "re-link all of the above in the candidate registry",
                "log the candidate change before H is opened",
            ],
            "not_required_again": ("the inference graph and the six model inputs are unchanged, "
                                   "so the trainer contract tests and the architecture FLOPs "
                                   "count do not have to be redone from scratch"),
        },
        "decision": ("roles are separated so a good checkpoint is preserved without an automatic "
                     "submission change; promoting DEV_BEST is a user decision"),
        "not_claimed": ("that the -0.0017 will survive on the server distribution, and equally "
                        "not that it will vanish"),
    }
    (OUT / "candidate_roles.json").write_text(json.dumps(roles, indent=1, sort_keys=True) + "\n")

    print(json.dumps({
        "contribution_sum_matches_total": payload["verified_review_claims"]["contribution_sum_matches_total"],
        "decelerating_abs_mean_b_rose": payload["verified_review_claims"]["decelerating_abs_mean_b_rose_seed0"],
        "fast_band_worsened": payload["verified_review_claims"]["fast_band_worsened_seed0"],
        "contributions": [{k: r[k] for k in ("regime", "group_internal_d3_delta",
                                             "contribution_to_overall_mean_delta")}
                          for r in contributions],
        "speed_bands": bands,
    }, indent=1))


if __name__ == "__main__":
    main()
