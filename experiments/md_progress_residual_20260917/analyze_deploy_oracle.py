#!/usr/bin/env python3
"""Optimize coefficients through the exact deployed nonnegative geometry."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar


WEIGHTS = np.asarray([11, 11, 5, 5, 2, 2], np.float64) / 36
DT = .5
T_MID = np.asarray([.25, .75, 1.25, 1.75, 2.25, 2.75], np.float64)


def stable_geometry(plan, eps=1e-3):
    delta = np.diff(np.concatenate([np.zeros((1, 2)), plan], 0), axis=0)
    length = np.linalg.norm(delta, axis=1)
    good = np.flatnonzero(length >= eps)
    direction = np.zeros_like(delta)
    if not len(good):
        direction[:, 0] = 1.
    else:
        direction[good] = delta[good] / length[good, None]
        for index in np.flatnonzero(length < eps):
            nearest = good[np.argmin(np.abs(good - index))]
            direction[index] = direction[nearest]
    return length, direction, int(not len(good)), int((length < eps).sum())


def deploy(plan, coefficient):
    length, direction, _, _ = stable_geometry(plan)
    speed = coefficient[0]
    if len(coefficient) == 2:
        speed = speed + coefficient[1] * T_MID
    corrected = np.maximum(length + DT * speed, 0.)
    correction = np.cumsum(direction * (corrected - length)[:, None], axis=0)
    return plan + correction


def prefix(prediction, target):
    return float(np.sum(WEIGHTS * np.linalg.norm(prediction - target, axis=1)))


def solve(payload):
    plan, target, count, cap = payload
    raw = lambda coefficient: prefix(deploy(plan, np.asarray(coefficient)), target)
    # Tiny deterministic tie-break for stop plateaus.  Reported scores below
    # always use the unregularized PREFIX objective.
    regularized = lambda coefficient: raw(coefficient) + 1e-8 * np.dot(coefficient, coefficient)
    if count == 1:
        result = minimize_scalar(lambda value: regularized([value]), bounds=(-cap, cap),
                                 method="bounded", options={"xatol": 1e-10})
        coefficient = np.asarray([result.x])
    else:
        grid = np.linspace(-cap, cap, 31)
        best = min(((raw([cv, ca]), cv, ca) for cv in grid for ca in grid), key=lambda x: x[0])
        candidates = []
        for start in ([best[1], best[2]], [0., 0.]):
            result = minimize(regularized, start, method="Powell",
                              bounds=[(-cap, cap), (-cap, cap)],
                              options={"xtol": 1e-8, "ftol": 1e-11, "maxiter": 240})
            candidates.append(np.asarray(result.x))
        coefficient = min(candidates, key=raw)
    prediction = deploy(plan, coefficient)
    length, _, all_zero, short = stable_geometry(plan)
    return {"score": raw(coefficient), "coefficient": coefficient,
            "prediction": prediction, "all_zero": all_zero,
            "short": short, "base_lengths": length}


def metrics(prediction, target):
    distance = np.linalg.norm(prediction - target, axis=-1)
    return {"L2_1s": float(distance[:, :2].mean()),
            "L2_2s": float(distance[:, :4].mean()),
            "L2_3s": float(distance.mean()),
            "PREFIX": float(np.mean(distance @ WEIGHTS))}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-key", default="pred_abs_xy",
                        help="record field to refine (use base_abs_xy for residual evaluations)")
    parser.add_argument("--save-record-coefficients", action="store_true",
                        help="store coefficients in evaluation-record order for error analysis")
    parser.add_argument("--cap", type=float, default=1.)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    source = json.loads(Path(args.evaluation).read_text())
    records = source["records"]
    missing = [index for index, record in enumerate(records) if args.plan_key not in record]
    if missing:
        raise KeyError(f"{args.plan_key!r} absent from records, first index {missing[0]}")
    plan = np.asarray([record[args.plan_key] for record in records], np.float64)
    target = np.asarray([record["gt_abs_xy"] for record in records], np.float64)
    result = {
        "schema_version": 2, "evaluation": str(Path(args.evaluation).resolve()),
        "plan_key": args.plan_key,
        "n": len(records), "cap_abs": args.cap,
        "geometry": {"dt": DT, "t_mid": T_MID.tolist(), "short_eps": 1e-3,
                     "short_fallback": "closest valid predicted segment",
                     "all_zero_fallback": "ego +x", "negative_interval": "clamp to zero",
                     "objective": "official PREFIX",
                     "coefficient_tie_break": "1e-8 times squared coefficient norm"},
        "base": metrics(plan, target), "arms": {},
    }
    context = mp.get_context("fork")
    with context.Pool(args.workers) as pool:
        for count, name in ((1, "scalar_dv"), (2, "two_coefficient_dv_da")):
            solved = pool.map(solve, [(p, y, count, args.cap) for p, y in zip(plan, target)],
                              chunksize=8)
            prediction = np.stack([item["prediction"] for item in solved])
            coefficient = np.stack([item["coefficient"] for item in solved])
            saturation = np.isclose(np.abs(coefficient), args.cap, atol=1e-5)
            result["arms"][name] = {
                "metrics": metrics(prediction, target),
                "coefficient_mean": coefficient.mean(0).tolist(),
                "coefficient_abs_mean": np.abs(coefficient).mean(0).tolist(),
                "coefficient_abs_quantiles": {
                    str(q): np.quantile(np.abs(coefficient), q, axis=0).tolist()
                    for q in (.5, .9, .95, .99, 1.)},
                "saturation_fraction": saturation.mean(0).tolist(),
                "all_zero_base_rows": int(sum(item["all_zero"] for item in solved)),
                "short_base_segments": int(sum(item["short"] for item in solved)),
            }
            if args.save_record_coefficients:
                result["arms"][name]["record_coefficients"] = coefficient.tolist()
            print(name, json.dumps(result["arms"][name], sort_keys=True), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
