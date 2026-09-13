"""Corrected re-aggregation of an existing MotionDriveV2 evaluation.

No model is run and no prediction is changed. This module recomputes the error
decomposition from the stored per-row predictions with the axis, tangent and
oracle definitions fixed, and keeps the earlier definitions alongside under
`legacy_` names so the two can be compared rather than silently replaced.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

T = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float64)
W = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0
TANGENT_EPS = 1e-6           # metres; below this a GT segment has no direction
NEAR_STATIONARY_M = 1.0      # total GT progress at or below this is not a bearing


def row_errors(pred, gt):
    """Per-waypoint L2, the project metric, and the cumulative ADEs it averages."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[1:] != (6, 2):
        raise ValueError("Expected matching [N,6,2] arrays")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("Non-finite trajectory")
    point_l2 = np.linalg.norm(pred - gt, axis=-1)
    d3 = point_l2 @ W
    ade123 = np.stack([point_l2[:, :2].mean(-1),
                       point_l2[:, :4].mean(-1),
                       point_l2.mean(-1)], axis=-1)
    np.testing.assert_allclose(d3, ade123.mean(-1), rtol=1e-12, atol=1e-12)
    return point_l2, d3, ade123


def along_cross(pred, gt):
    """Signed along/cross error on GT segment tangents, with validity kept.

    A GT segment shorter than TANGENT_EPS has no direction. Those entries are
    returned as zero and flagged invalid rather than being folded into the
    decomposition, so a stopped row's error cannot disappear from the totals.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    error = pred - gt
    segment = np.diff(gt, axis=1, prepend=np.zeros_like(gt[:, :1]))
    length = np.linalg.norm(segment, axis=-1)
    valid = length > TANGENT_EPS
    tangent = np.zeros_like(segment)
    np.divide(segment, length[..., None], out=tangent, where=valid[..., None])
    normal = np.stack([-tangent[..., 1], tangent[..., 0]], axis=-1)
    along = np.where(valid, (error * tangent).sum(-1), 0.0)
    cross = np.where(valid, (error * normal).sum(-1), 0.0)
    residual = np.abs(along ** 2 + cross ** 2
                      - np.linalg.norm(error, axis=-1) ** 2)[valid]
    if residual.size and residual.max() > 1e-9:
        raise AssertionError("along/cross decomposition is not orthonormal")
    return along, cross, valid


def arclength(points):
    """Cumulative arclength from the ego origin through each waypoint."""
    points = np.asarray(points, dtype=np.float64)
    step = np.linalg.norm(np.diff(points, axis=1,
                                  prepend=np.zeros_like(points[:, :1])), axis=-1)
    return step.cumsum(-1)


def progress_scale_oracle(pred, gt, scales):
    """Resample the predicted polyline by a per-row uniform progress scale.

    This reproduces the earlier computation exactly, including np.interp's
    endpoint clamping. It selects the scale with the GT, so the result is an
    oracle over one particular operation - not a reachable score, not a pure
    shape error, and not something to attach to the model as a corrector.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if not np.any(np.isclose(scales, 1.0)):
        raise ValueError("The scale grid must contain 1.0")
    n = len(pred)
    s = arclength(pred)
    knots = np.concatenate([np.zeros((n, 1)), s], axis=1)
    values = np.concatenate([np.zeros((n, 1, 2)), pred], axis=1)
    monotone = np.all(np.diff(knots, axis=1) >= 0, axis=1)
    degenerate = s[:, -1] <= TANGENT_EPS
    best = np.full(n, np.inf)
    best_scale = np.ones(n)
    best_clamped = np.zeros(n)
    for scale in scales:
        target = s * scale
        resampled = np.empty_like(pred)
        for i in range(n):
            resampled[i, :, 0] = np.interp(target[i], knots[i], values[i, :, 0])
            resampled[i, :, 1] = np.interp(target[i], knots[i], values[i, :, 1])
        value = np.linalg.norm(resampled - gt, axis=-1) @ W
        clamped = (target > s[:, -1:]).mean(-1)
        better = value < best
        best_scale = np.where(better, scale, best_scale)
        best_clamped = np.where(better, clamped, best_clamped)
        best = np.minimum(best, value)
    return {"d3": best, "scale": best_scale, "clamped_fraction": best_clamped,
            "monotone_arclength": monotone, "degenerate_polyline": degenerate,
            "scale_count": int(len(scales)),
            "scale_range": [float(scales.min()), float(scales.max())]}


def bucket_table(d3, groups):
    """Mean D3 per group, plus what the mean becomes if a group were solved."""
    rows = {}
    for label, mask in groups.items():
        mask = np.asarray(mask, dtype=bool)
        rows[label] = {
            "rows": int(mask.sum()),
            "mean_d3": float(d3[mask].mean()) if mask.any() else None,
            "share_of_total_d3": float(d3[mask].sum() / d3.sum()) if mask.any() else 0.0,
            # Counterfactual for this group alone; overlapping groups do not add.
            "d3_if_group_were_exact": float(np.where(mask, 0.0, d3).mean()),
        }
    return rows


def concentration(d3, fractions=(0.05, 0.10, 0.25)):
    order = np.argsort(-d3)
    out = {}
    for fraction in fractions:
        k = int(len(d3) * fraction)
        idx = order[:k]
        out[f"worst_{int(fraction * 100)}pct"] = {
            "rows": int(k),
            "share_of_total_d3": float(d3[idx].sum() / d3.sum()),
            "mean_d3": float(d3[idx].mean()),
            "d3_if_group_were_exact": float(np.where(
                np.isin(np.arange(len(d3)), idx), 0.0, d3).mean()),
        }
    return out


def load_records(path):
    payload = json.loads(Path(path).read_text())
    records = payload.get("records")
    if records is None:
        raise ValueError("evaluation artifact carries no per-row records")
    pred = np.array([r["pred_abs_xy"] for r in records], dtype=np.float64)
    gt = np.array([r["gt_abs_xy"] for r in records], dtype=np.float64)
    meta = {
        "session": np.array([r["session"] for r in records]),
        "scenario": np.array([r["scenario"] for r in records]),
        "frame": np.array([r["frame"] for r in records], dtype=np.int64),
        "row": np.array([r["row"] for r in records], dtype=np.int64),
        "reported_d3": np.array([r["d3"] for r in records], dtype=np.float64),
        "gt_state": np.array([r["gt_state"] for r in records], dtype=np.float64),
        "gt_state_valid": np.array([r["gt_state_valid"] for r in records]),
    }
    report = payload.get("report", {})
    return pred, gt, meta, report
