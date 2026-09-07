"""V2 losses, exact metric and deploy-input boundary (no dataset I/O at import)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

MODEL_INPUTS = (
    "images", "history_images", "lidar2img", "history_transforms",
    "time_offsets", "goal_xy",
)
TIME_WEIGHTS = (11 / 36, 11 / 36, 5 / 36, 5 / 36, 2 / 36, 2 / 36)
HISTORY_SCALE = (10., 5., 1., 1.)
STATE_SCALE = (10., 5., 3., 3., .5)


@dataclass
class LossWeights:
    plan: float = 1.
    occupancy: float = .2
    lane: float = .2
    motion: float = .2
    uncertainty: bool = True


def model_inputs(batch: Mapping[str, object]) -> dict[str, Tensor]:
    """Whitelist only inference inputs; never forward labels or supplied status."""
    missing = set(MODEL_INPUTS).difference(batch)
    if missing:
        raise KeyError(f"Missing model inputs: {sorted(missing)}")
    return {key: batch[key] for key in MODEL_INPUTS}


def to_device(batch: Mapping, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, Tensor) else v
            for k, v in batch.items()}


def weighted_d3(pred: Tensor, target: Tensor) -> Tensor:
    """Per-frame official mean of cumulative 1/2/3-second ADE, in metres."""
    if pred.shape != target.shape or pred.shape[-2:] != (6, 2):
        raise ValueError(f"Expected matching [B,6,2], got {pred.shape}/{target.shape}")
    distance = torch.linalg.vector_norm(pred.float() - target.float(), dim=-1)
    return (distance * distance.new_tensor(TIME_WEIGHTS)).sum(-1)


def _mask_like(mask: Tensor | None, value: Tensor) -> Tensor:
    if mask is None:
        return torch.ones_like(value, dtype=torch.bool)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return mask.bool().expand_as(value)


def masked_mean(value: Tensor, mask: Tensor | None) -> Tensor:
    valid = _mask_like(mask, value)
    clean = torch.where(valid, value, torch.zeros_like(value))
    return clean.sum() / valid.sum().clamp_min(1)


def balanced_raster_bce(logit: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Balance annotated foreground/background, ignoring unsupported cells."""
    if logit.shape != target.shape or target.shape != valid.shape:
        raise ValueError(f"Raster mismatch: {logit.shape}/{target.shape}/{valid.shape}")
    mask = valid.bool()
    safe_target = torch.where(mask, target.float(), torch.zeros_like(target.float()))
    bce = F.binary_cross_entropy_with_logits(logit.float(), safe_target, reduction="none")
    positive, negative = mask & (safe_target >= .5), mask & (safe_target < .5)
    pos_count, neg_count = positive.sum(), negative.sum()
    # No Python GPU synchronization in loss construction.
    pos_present, neg_present = (pos_count > 0).float(), (neg_count > 0).float()
    return (masked_mean(bce, positive) * pos_present + masked_mean(bce, negative) *
            neg_present) / (pos_present + neg_present).clamp_min(1)


def regression_loss(pred: Tensor, target: Tensor, valid: Tensor | None,
                    scale: tuple[float, ...], logvar: Tensor | None = None) -> Tensor:
    pred = pred.float()
    mask = _mask_like(valid, pred)
    target = torch.where(mask, target.float(), torch.zeros_like(pred))
    error = (pred - target) / pred.new_tensor(scale)
    if logvar is None:
        loss = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none")
    else:
        if logvar.shape != pred.shape:
            raise ValueError("Uncertainty tensor must match regression tensor")
        # Log-variance is in the normalized physical units defined above.
        lv = logvar.float().clamp(-6, 6)
        loss = .5 * (error.square() * (-lv).exp() + lv)
    return masked_mean(loss, mask)


def compute_loss(outputs: Mapping[str, Tensor], batch: Mapping[str, Tensor],
                 weights: LossWeights) -> tuple[Tensor, dict[str, Tensor]]:
    pred = outputs["plan_abs"].float()
    plan_valid = batch.get("plan_valid", torch.ones_like(pred[..., 0], dtype=torch.bool))
    # Official metric requires all six samples. Do not quietly redefine metric.
    complete = plan_valid.bool().all(-1)
    safe_gt = torch.where(complete[:, None, None], batch["gt_plan"].float(),
                          torch.zeros_like(pred))
    d3 = weighted_d3(pred, safe_gt)
    plan_loss = masked_mean(d3, complete)
    occ = balanced_raster_bce(outputs["occ_logits"], batch["occ_target"], batch["occ_valid"])
    lane = balanced_raster_bce(outputs["lane_logits"], batch["lane_target"], batch["lane_valid"])
    history = regression_loss(outputs["history_hat"], batch["history_target"],
                              batch.get("history_valid"), HISTORY_SCALE,
                              outputs.get("history_logvar") if weights.uncertainty else None)
    state_valid = batch.get("state_valid", torch.ones_like(batch["state_target"], dtype=torch.bool))
    state = regression_loss(outputs["state_hat"][..., :5], batch["state_target"][..., :5],
                            state_valid[..., :5], STATE_SCALE,
                            outputs.get("state_logvar") if weights.uncertainty else None)
    stop_target = torch.where(state_valid[..., 5].bool(), batch["state_target"][..., 5].float(),
                              torch.zeros_like(batch["state_target"][..., 5].float()))
    stop = masked_mean(F.binary_cross_entropy_with_logits(
        outputs["state_hat"][..., 5].float(), stop_target, reduction="none"), state_valid[..., 5])
    motion = history + state + .2 * stop
    loss = (weights.plan * plan_loss + weights.occupancy * occ +
            weights.lane * lane + weights.motion * motion)
    return loss, {"total": loss, "plan_d3": plan_loss, "occ_bce": occ,
                  "lane_bce": lane, "history": history, "state": state,
                  "stop_bce": stop, "motion": motion}


def raster_counts(logit: Tensor, target: Tensor, valid: Tensor) -> tuple[int, int]:
    positive = target >= .5
    predicted = logit >= 0
    mask = valid.bool()
    intersection = (predicted & positive & mask).sum()
    union = ((predicted | positive) & mask).sum()
    return int(intersection), int(union)
