#!/usr/bin/env python3
"""Measure whether predicted or causal pose status explains progress error.

The fit is leave-one-session-out.  It cannot memorize a held-out session and is
used only as a diagnostic; it is not a submission model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT / "experiments/md_progress_residual_20260917"))
from progress_residual import apply_progress_correction

PREFIX_WEIGHTS = np.asarray([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.


def correlation(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def prefix_per_row(prediction, target):
    distance = np.linalg.norm(prediction - target, axis=-1)
    return (distance * PREFIX_WEIGHTS[None]).sum(1)


def ridge_loso(features, target, sessions, alpha=1.):
    output = np.zeros(len(target), dtype=np.float64)
    for session in np.unique(sessions):
        train = sessions != session
        held = ~train
        mean = features[train].mean(0)
        std = features[train].std(0)
        std[std < 1e-6] = 1.
        x_train = np.column_stack(((features[train] - mean) / std,
                                   np.ones(train.sum())))
        x_held = np.column_stack(((features[held] - mean) / std,
                                  np.ones(held.sum())))
        regularizer = np.eye(x_train.shape[1]) * alpha
        regularizer[-1, -1] = 0.
        weights = np.linalg.solve(x_train.T @ x_train + regularizer,
                                  x_train.T @ target[train])
        output[held] = x_held @ weights
    return np.clip(output, -1., 1.)


def result_for(coefficient, oracle, base, gt, buckets):
    final, _ = apply_progress_correction(
        torch.as_tensor(base, dtype=torch.float32),
        torch.as_tensor(coefficient[:, None], dtype=torch.float32))
    per_row = prefix_per_row(final.numpy(), gt)

    def subset(mask):
        strong = mask & (np.abs(oracle) >= .1)
        return {
            "n": int(mask.sum()),
            "coefficient_mae_mps": float(np.abs(coefficient[mask] - oracle[mask]).mean()),
            "coefficient_correlation": correlation(coefficient[mask], oracle[mask]),
            "prefix": float(per_row[mask].mean()),
            "strong_sign_n_abs_oracle_ge_0p1": int(strong.sum()),
            "strong_sign_accuracy": (float((np.sign(coefficient[strong])
                                             == np.sign(oracle[strong])).mean())
                                     if strong.any() else None),
        }

    output = {"all": subset(np.ones(len(oracle), dtype=bool)), "buckets": {}}
    for bucket in sorted(set(buckets)):
        output["buckets"][str(bucket)] = subset(buckets == bucket)
    return output


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    evaluation_path = Path(args.evaluation).resolve()
    oracle_path = Path(args.oracle).resolve()
    evaluation = json.loads(evaluation_path.read_text())
    oracle_payload = json.loads(oracle_path.read_text())
    records = evaluation["records"]
    coefficient = np.asarray(
        oracle_payload["arms"]["scalar_dv"]["record_coefficients"],
        dtype=np.float64)[:, 0]
    if len(records) != len(coefficient) or int(oracle_payload["n"]) != len(records):
        raise ValueError("evaluation and oracle population differ")

    base = np.asarray([row["base_abs_xy"] for row in records], dtype=np.float64)
    gt = np.asarray([row["gt_abs_xy"] for row in records], dtype=np.float64)
    pred_state = np.asarray([row["pred_state"][:5] for row in records], dtype=np.float64)
    causal_status = np.asarray([row["gt_state"][:5] for row in records], dtype=np.float64)
    sessions = np.asarray([row["session"] for row in records])
    buckets = np.asarray([row["bucket"] for row in records])
    delta = np.diff(np.concatenate([np.zeros((len(base), 1, 2)), base], axis=1), axis=1)
    lengths = np.linalg.norm(delta, axis=-1)
    plan_features = np.concatenate([base.reshape(len(base), -1), lengths], axis=1)

    arms = {
        "base_plan_only": plan_features,
        "base_plan_plus_image_predicted_state": np.concatenate(
            [plan_features, pred_state], axis=1),
        "base_plan_plus_causal_pose_status": np.concatenate(
            [plan_features, causal_status], axis=1),
    }
    fitted = {name: ridge_loso(features, coefficient, sessions)
              for name, features in arms.items()}
    output = {
        "schema_version": 1,
        "purpose": "diagnostic only; no fit sees rows from its held-out session",
        "evaluation": str(evaluation_path),
        "oracle": str(oracle_path),
        "n": len(records),
        "sessions": int(len(np.unique(sessions))),
        "official_prefix_weights": PREFIX_WEIGHTS.tolist(),
        "fit": {"kind": "standardized leave-one-session-out ridge",
                "alpha": 1., "coefficient_clip_mps": [-1., 1.]},
        "base_prefix": float(prefix_per_row(base, gt).mean()),
        "scalar_oracle_prefix": float(oracle_payload["arms"]["scalar_dv"]
                                      ["metrics"]["PREFIX"]),
        "direct_signal": {
            "image_predicted_vx_minus_base_first_interval_speed_correlation": correlation(
                pred_state[:, 0] - lengths[:, 0] / .5, coefficient),
            "causal_pose_vx_minus_base_first_interval_speed_correlation": correlation(
                causal_status[:, 0] - lengths[:, 0] / .5, coefficient),
        },
        "arms": {name: result_for(value, coefficient, base, gt, buckets)
                 for name, value in fitted.items()},
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
