"""CPU-only, post-evaluation diagnostics; future labels never enter a model.

Restricts analysis to the audited tune rows. Longitudinal/lateral absolute
errors and squared-energy fractions are diagnostics, not additive D3 terms.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WEIGHTS = np.asarray([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36
TUNE_ROWS_SHA = "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88"


def summarize(pred, gt, mask):
    pred, gt = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
    if not len(pred):
        return {"n": 0}
    delta = pred - gt
    point = np.linalg.norm(delta, axis=-1)
    energy = ((delta ** 2) * WEIGHTS[None, :, None]).sum(1).mean(0)
    return {"n": len(pred), "d3": float((point * WEIGHTS).sum(1).mean()),
            "point_l2": point.mean(0).tolist(),
            "weighted_signed_xy": (delta * WEIGHTS[None, :, None]).sum(1).mean(0).tolist(),
            "weighted_abs_xy": (np.abs(delta) * WEIGHTS[None, :, None]).sum(1).mean(0).tolist(),
            "weighted_squared_xy": energy.tolist(),
            "longitudinal_squared_energy_fraction": float(energy[0] / max(energy.sum(), 1e-30)),
            "initial_progress_speed_bias_mps": float(((np.linalg.norm(pred[:, 0], axis=-1)
                - np.linalg.norm(gt[:, 0], axis=-1)) / .5).mean())}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base", required=True)
    parser.add_argument("--eval", action="append", required=True, help="NAME=eval_STEP.npz")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    with np.load("/tmp/pm97/data/etri/ego_cache.npz", allow_pickle=False) as z:
        all_gt = z["fut"]
        names = z["scenarios"].astype(str)[z["scen_idx"]]
    base = Path(args.base)
    split = json.loads((base / "data/etri/motiondrive_v2/grouped_split_rawtime.json").read_text())
    with np.load(base / "data/etri/motiondrive_v2_shared_status_a1_20260908_ops/tune.npz", allow_pickle=False) as z:
        status_lookup = {int(r): x for r, x in zip(z["row"], z["status5"])}
    results, errors, expected_rows = {}, {}, None
    for argument in args.eval:
        name, path = argument.split("=", 1)
        with np.load(path, allow_pickle=False) as z:
            rows, pred, recorded = z["rows"].astype(np.int64), z["pred"], z["d3"]
        if hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest() != TUNE_ROWS_SHA:
            raise ValueError("Only exact audited tune1998 row order is permitted")
        if expected_rows is None:
            expected_rows = rows
        if not np.array_equal(rows, expected_rows):
            raise ValueError("Evaluations use different rows")
        gt = all_gt[rows].astype(np.float64)
        sessions = np.asarray([split["scene_to_session"][s] for s in names[rows]])
        status = np.asarray([status_lookup[int(r)] for r in rows])
        cost = (np.linalg.norm(pred.astype(np.float64) - gt, axis=-1) * WEIGHTS).sum(-1)
        mismatch = float(np.max(np.abs(cost - recorded)))
        if mismatch > 1e-5:
            raise ValueError("Saved metric differs from independent D3")
        speeds = np.linalg.norm(np.diff(gt, axis=1, prepend=np.zeros_like(gt[:, :1])), axis=-1) / .5
        trend = speeds[:, -1] - speeds[:, 0]
        groups = {"all": np.ones(len(rows), dtype=bool),
                  "gt_decelerating_gt0p5mps": trend < -.5,
                  "gt_near_constant_le0p5mps": np.abs(trend) <= .5,
                  "gt_accelerating_gt0p5mps": trend > .5,
                  "gt_endpoint_abs_y_gt1m": np.abs(gt[:, -1, 1]) > 1,
                  "gt_endpoint_abs_y_le1m": np.abs(gt[:, -1, 1]) <= 1}
        bounds = [0, .5, 5, 10, 15, 20, float("inf")]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            groups[f"causal_vx_{lo}_to_{hi}"] = (status[:, 0] >= lo) & (status[:, 0] < hi)
        results[name] = {"source": str(Path(path).resolve()), "metric_recompute_max_abs_error": mismatch,
                         "groups": {k: summarize(pred, gt, mask) for k, mask in groups.items()},
                         "session_d3": {s: float(cost[sessions == s].mean()) for s in sorted(set(sessions))}}
        errors[name] = cost
    pairs = {}
    rng = np.random.default_rng(0)
    unique = sorted(set(sessions))
    indices = rng.integers(len(unique), size=(20000, len(unique)))
    counts = np.asarray([(sessions == s).sum() for s in unique])
    keys = list(errors)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            delta = errors[b] - errors[a]
            totals = np.asarray([delta[sessions == s].sum() for s in unique])
            samples = totals[indices].sum(-1) / counts[indices].sum(-1)
            pairs[f"{b} minus {a}"] = {"difference_d3": float(delta.mean()),
                "session_bootstrap_95ci": np.quantile(samples, [.025, .975]).tolist(),
                "improved_rows": int((delta < 0).sum()), "sessions": len(unique),
                "bootstrap_seed": 0, "bootstrap_repetitions": 20000}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"population": "original tune37 / 1998 rows", "results": results,
        "pairs": pairs, "notes": "Future-speed/curvature groups are retrospective labels only; XY diagnostics do not add to D3."},
        indent=2, allow_nan=False) + "\n")
    print(json.dumps({"results": {k: v["groups"]["all"] for k,v in results.items()}, "pairs":pairs}))


if __name__ == "__main__":
    main()
