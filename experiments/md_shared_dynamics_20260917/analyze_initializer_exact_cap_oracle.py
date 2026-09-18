#!/usr/bin/env python3
import json
import multiprocessing as mp
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "/NHNHOME/data/sukim/adcl/experiments/md_progress_residual_20260917")
from analyze_deploy_oracle import deploy, metrics, prefix


def solve(payload):
    plan, target = payload
    raw = lambda coefficient: prefix(deploy(plan, np.asarray(coefficient)), target)
    regularized = lambda coefficient: raw(coefficient) + 1e-8 * np.dot(coefficient, coefficient)
    starts = []
    best = (float("inf"), 0.0, 0.0)
    for dv in np.linspace(-1.0, 1.0, 31):
        for da in np.linspace(-0.5, 0.5, 31):
            candidate = (raw([dv, da]), dv, da)
            if candidate[0] < best[0]:
                best = candidate
    starts.extend(([best[1], best[2]], [0.0, 0.0]))
    candidates = []
    for start in starts:
        result = minimize(regularized, start, method="Powell",
                          bounds=[(-1.0, 1.0), (-0.5, 0.5)],
                          options={"xtol": 1e-8, "ftol": 1e-11, "maxiter": 240})
        candidates.append(np.asarray(result.x))
    coefficient = min(candidates, key=raw)
    return deploy(plan, coefficient), coefficient


def main():
    source = json.load(open(sys.argv[1]))
    records = source["records"]
    plan = np.asarray([record["pred_abs_xy"] for record in records], np.float64)
    target = np.asarray([record["gt_abs_xy"] for record in records], np.float64)
    with mp.get_context("fork").Pool(32) as pool:
        solved = pool.map(solve, zip(plan, target), chunksize=8)
    prediction = np.stack([value[0] for value in solved])
    coefficient = np.stack([value[1] for value in solved])
    result = {
        "base": metrics(plan, target),
        "two_coefficient_dv_da": metrics(prediction, target),
        "caps": {"delta_v": 1.0, "delta_a": 0.5},
        "coefficient_abs_mean": np.abs(coefficient).mean(0).tolist(),
        "saturation_fraction": np.isclose(
            np.abs(coefficient), np.asarray([1.0, 0.5]), atol=1e-5).mean(0).tolist(),
        "source": sys.argv[1],
    }
    json.dump(result, open(sys.argv[2], "w"), indent=1, sort_keys=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
