"""Reference implementation: replace the existing length-only auxiliary.

The final model remains one H4-PROGRESS network. This changes training only.
IMPORTANT: wrap the ORIGINAL common compute_loss, not its length-aux wrapper.
The per-microbatch numerator uses the full-effective-batch plan_complete count.
This file has not been applied to the user's repository or GPU training.
"""
from __future__ import annotations
import math
from collections.abc import Mapping
from typing import Callable
import torch
from torch import Tensor


def interval_displacements(points: Tensor) -> Tensor:
    if points.ndim != 3 or tuple(points.shape[1:]) != (6, 2):
        raise ValueError(f'Expected [B,6,2], got {tuple(points.shape)}')
    previous = torch.cat((torch.zeros_like(points[:, :1]), points[:, :-1]), dim=1)
    return points - previous


def vector_loss(outputs: Mapping[str, Tensor], batch: Mapping[str, Tensor],
                normalizers: Mapping[str, Tensor] | None = None) -> Tensor:
    pred = outputs['plan_abs']
    if pred.ndim != 3 or tuple(pred.shape[1:]) != (6, 2):
        raise ValueError('plan_abs must be [B,6,2]')
    target = batch['gt_plan']
    if target.shape != pred.shape:
        raise ValueError('gt_plan and plan_abs must have identical shapes')
    mask = batch.get('plan_valid')
    if mask is None:
        mask = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
    if mask.shape != pred.shape[:2]:
        raise ValueError('plan_valid must be [B,6]')
    complete = mask.bool().all(-1)
    with torch.autocast(device_type=pred.device.type, enabled=False):
        # Mask BEFORE norms, so all-invalid rows with NaN labels are harmless.
        p = torch.where(complete[:, None, None], pred.float(), torch.zeros_like(pred, dtype=torch.float32))
        g = torch.where(complete[:, None, None], target.float(), torch.zeros_like(pred, dtype=torch.float32))
        err = torch.linalg.vector_norm(interval_displacements(p) - interval_displacements(g), dim=-1)
        if normalizers is None:
            count = complete.sum()
        else:
            count = normalizers['plan_complete']
        denominator = torch.as_tensor(count, device=pred.device, dtype=torch.float32).clamp_min(1)
        return err.sum() / (6 * denominator)


def wrap_compute_loss(original_compute_loss: Callable, coefficient: float = .25) -> Callable:
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('coefficient must be finite and nonnegative')
    def compute_loss(outputs, batch, weights, *, normalizers=None, stop_class_weights=None):
        total, parts = original_compute_loss(outputs, batch, weights,
            normalizers=normalizers, stop_class_weights=stop_class_weights)
        if 'plan_interval_length' in parts or 'plan_interval_vector' in parts:
            raise ValueError('Call on original compute_loss; do NOT stack length and vector wrappers')
        if coefficient == 0:
            return total, parts
        extra = vector_loss(outputs, batch, normalizers)
        total = total + coefficient * extra
        parts = dict(parts)
        parts['plan_interval_vector'] = extra
        parts['total'] = total
        return total, parts
    compute_loss.vector_lambda = coefficient
    return compute_loss
