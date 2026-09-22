#!/usr/bin/env python3
"""Interval-length auxiliary loss, added beside the existing D3 loss.

    ell(p)[k]  = ||p[k] - p[k-1]||,  p[-1] = the ego origin
    L_length   = sum_n complete[n] * sum_k |ell_pred[n,k] - ell_gt[n,k]| / (6 * C_full)
    L_total    = L_existing + lambda * L_length

`C_full` is the SAME full-effective-batch complete-sample count the D3 loss
normalises by, taken from the trainer's own normalizer dict, so every microbatch
divides by the same denominator and the contributions add.  With lambda = 0 the
total is the original loss unchanged.

Only interval magnitudes are compared here.  A norm carries no direction, so this
can never replace the D3 loss; it is added to it.
"""
from __future__ import annotations

import torch

ORIGIN_PAD = 1


def interval_lengths(points: torch.Tensor) -> torch.Tensor:
    """Chord length of each 0.5 s interval, starting from the ego origin."""
    if points.ndim != 3 or points.shape[1:] != (6, 2):
        raise ValueError(f"expected [B,6,2], got {tuple(points.shape)}")
    start = torch.cat([torch.zeros_like(points[:, :ORIGIN_PAD]), points[:, :-1]], dim=1)
    return torch.linalg.vector_norm(points - start, dim=-1)


def length_loss(outputs, batch, normalizers=None):
    """Mean absolute interval-length error, normalised like the plan loss."""
    pred = outputs["plan_abs"].float()
    plan_valid = batch.get("plan_valid", torch.ones_like(pred[..., 0], dtype=torch.bool))
    complete = plan_valid.bool().all(-1)
    safe_gt = torch.where(complete[:, None, None], batch["gt_plan"].float(),
                          torch.zeros_like(pred))
    error = (interval_lengths(pred) - interval_lengths(safe_gt)).abs().sum(-1)
    error = torch.where(complete, error, torch.zeros_like(error))
    if normalizers is None:
        denominator = complete.sum().clamp_min(1).to(error.dtype)
    else:
        denominator = normalizers["plan_complete"].clamp_min(1).to(error.dtype)
    return error.sum() / (denominator * pred.shape[1])


def wrap_compute_loss(original, lam: float):
    """Return a compute_loss that adds lambda * L_length to the existing total."""
    if lam < 0:
        raise ValueError("lambda must be nonnegative")

    def compute_loss(outputs, batch, weights, *, normalizers=None, stop_class_weights=None):
        total, parts = original(outputs, batch, weights, normalizers=normalizers,
                                stop_class_weights=stop_class_weights)
        if lam == 0:
            return total, parts
        extra = length_loss(outputs, batch, normalizers)
        total = total + lam * extra
        parts = dict(parts)
        parts["plan_interval_length"] = extra
        parts["total"] = total
        return total, parts

    compute_loss.length_lambda = lam
    return compute_loss

# --- metric-aligned interval weighting -------------------------------------
#
# The plan is emitted as a cumulative sum, so an error in interval k moves every
# position from k to the end. Interval k's real weight in the official metric is
# therefore the TAIL SUM of the position weights, not 1/6:
#
#   positions  [11, 11, 5, 5, 2, 2] / 36
#   interval 1 = (11+11+5+5+2+2)/36 = 1.0000
#   interval 2 =    (11+5+5+2+2)/36 = 0.6944
#   interval 3 =       (5+5+2+2)/36 = 0.3889
#   interval 4 =         (5+2+2)/36 = 0.2500
#   interval 5 =           (2+2)/36 = 0.1111
#   interval 6 =             (2)/36 = 0.0556
#
# Interval 1 matters 18x more than interval 6, while length_loss above weights
# them equally. The weights are rescaled to mean 1 so that the same lambda keeps
# the same overall magnitude and only the distribution across intervals changes.
_POSITION_WEIGHTS = torch.tensor([11., 11., 5., 5., 2., 2.]) / 36.
METRIC_INTERVAL_WEIGHTS = torch.stack(
    [_POSITION_WEIGHTS[k:].sum() for k in range(6)])
METRIC_INTERVAL_WEIGHTS = (METRIC_INTERVAL_WEIGHTS
                           / METRIC_INTERVAL_WEIGHTS.mean())


def length_loss_metric_weighted(outputs, batch, normalizers=None):
    """length_loss with intervals weighted by their true share of the metric."""
    pred = outputs["plan_abs"].float()
    plan_valid = batch.get("plan_valid", torch.ones_like(pred[..., 0], dtype=torch.bool))
    complete = plan_valid.bool().all(-1)
    safe_gt = torch.where(complete[:, None, None], batch["gt_plan"].float(),
                          torch.zeros_like(pred))
    weights = METRIC_INTERVAL_WEIGHTS.to(pred.device, pred.dtype)
    per_interval = (interval_lengths(pred) - interval_lengths(safe_gt)).abs()
    error = (per_interval * weights).sum(-1)
    error = torch.where(complete, error, torch.zeros_like(error))
    if normalizers is None:
        denominator = complete.sum().clamp_min(1).to(error.dtype)
    else:
        denominator = normalizers["plan_complete"].clamp_min(1).to(error.dtype)
    return error.sum() / (denominator * pred.shape[1])


def wrap_compute_loss_metric_weighted(original, lam: float):
    """wrap_compute_loss, but the auxiliary uses the metric-aligned weights."""
    if lam < 0:
        raise ValueError("lambda must be nonnegative")

    def compute_loss(outputs, batch, weights, *, normalizers=None, stop_class_weights=None):
        total, parts = original(outputs, batch, weights, normalizers=normalizers,
                                stop_class_weights=stop_class_weights)
        if lam == 0:
            return total, parts
        extra = length_loss_metric_weighted(outputs, batch, normalizers)
        total = total + lam * extra
        # The trainer accumulates these and calls .detach(), so they stay tensors.
        parts = dict(parts)
        parts["plan_interval_length"] = extra
        parts["total"] = total
        return total, parts

    compute_loss.length_lambda = lam
    compute_loss.interval_weighting = "metric_aligned"
    return compute_loss
