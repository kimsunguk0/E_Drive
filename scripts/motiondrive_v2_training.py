"""V2 losses, exact metric and deploy-input boundary (no dataset I/O at import)."""
from __future__ import annotations

import hashlib
import math
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
TIME_INPUT_MODES = ("raw", "nominal")
NOMINAL_HISTORY_SECONDS = (.1, .2, .5, 1.)


def set_training_mode(model: torch.nn.Module, bn_policy: str = "adaptive") -> int:
    """Train all weights; optionally keep only BN statistics in inference mode.

    This is NOT a backbone/affine freeze. GroupNorm, decoder and every trainable
    parameter retain normal training behavior. Inference always uses eval mode.
    """
    if bn_policy not in ("adaptive", "fixed"):
        raise ValueError(f"Unknown BN training policy: {bn_policy}")
    modules = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if bn_policy == "fixed" and any(not m.track_running_stats for m in modules):
        raise ValueError("Fixed BN policy requires existing running statistics")
    model.train()
    if bn_policy == "fixed":
        for module in modules:
            module.eval()
    return len(modules)


def tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    """Order-independent bitwise fingerprint; includes buffers and tensor metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Non-tensor state entry: {name}")
        value = tensor.detach().cpu().contiguous()
        digest.update((name + "\0" + str(value.dtype) + "\0" + str(tuple(value.shape)) + "\0").encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass
class LossWeights:
    plan: float = 1.
    occupancy: float = .2
    lane: float = .2
    motion: float = .2
    uncertainty: bool = True


def time_input_policy(mode: str, nominal_history_seconds=NOMINAL_HISTORY_SECONDS) -> dict:
    if mode not in TIME_INPUT_MODES:
        raise ValueError("time_input must be raw or nominal")
    nominal = tuple(float(value) for value in nominal_history_seconds)
    if nominal not in (NOMINAL_HISTORY_SECONDS, (.2, .5, 1., 2.)):
        raise ValueError("Only the fixed control/wide nominal history times are supported")
    result = {"mode": mode, "scope": "model input time_offsets only; same for training and every evaluation",
            "source": "dataset-provided raw offsets, unchanged" if mode == "raw" else "fixed nominal frame-offset seconds",
            "nominal_history_seconds": list(nominal), "nominal_dtype": "float32",
            "supervision_and_dataset_modified": False}
    if nominal == (.2, .5, 1., 2.):
        result["supervision_and_dataset_modified"] = True
        result["temporal_contract"] = "wide; aligned pose-derived history supervision supplied by pinned overlay"
    return result


def model_inputs(batch: Mapping[str, object], time_input: str = "raw",
                 nominal_history_seconds=NOMINAL_HISTORY_SECONDS) -> dict[str, Tensor]:
    """Whitelist inputs; raw preserves tensor identity/RNG and never edits GT."""
    if time_input not in TIME_INPUT_MODES:
        raise ValueError("time_input must be raw or nominal")
    missing = set(MODEL_INPUTS).difference(batch)
    if missing:
        raise KeyError(f"Missing model inputs: {sorted(missing)}")
    inputs = {key: batch[key] for key in MODEL_INPUTS}
    if time_input == "nominal":
        policy = time_input_policy(time_input, nominal_history_seconds)
        raw = inputs["time_offsets"]
        if not isinstance(raw, Tensor) or raw.ndim != 2 or raw.shape != (len(inputs["images"]), 4):
            raise ValueError("Nominal timing requires exactly four historical offsets [B,4]")
        inputs["time_offsets"] = torch.tensor(policy["nominal_history_seconds"], dtype=torch.float32,
                                               device=raw.device).expand(raw.shape[0], -1)
    return inputs


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


def masked_mean(value: Tensor, mask: Tensor | None, *, normalizer: Tensor | None = None) -> Tensor:
    valid = _mask_like(mask, value)
    clean = torch.where(valid, value, torch.zeros_like(value))
    if normalizer is not None:
        return clean.sum() / normalizer.clamp_min(1)
    return clean.sum() / valid.sum().clamp_min(1)


def balanced_raster_bce(logit: Tensor, target: Tensor, valid: Tensor,
                        *, normalizers: tuple[Tensor, Tensor, Tensor] | None = None) -> Tensor:
    """Balance annotated foreground/background, ignoring unsupported cells."""
    if logit.shape != target.shape or target.shape != valid.shape:
        raise ValueError(f"Raster mismatch: {logit.shape}/{target.shape}/{valid.shape}")
    mask = valid.bool()
    safe_target = torch.where(mask, target.float(), torch.zeros_like(target.float()))
    bce = F.binary_cross_entropy_with_logits(logit.float(), safe_target, reduction="none")
    positive, negative = mask & (safe_target >= .5), mask & (safe_target < .5)
    if normalizers is not None:
        pos_count, neg_count, class_count = normalizers
        # Class presence AND denominators belong to the FULL effective batch,
        # even if this chunk contains only one class or no valid raster cells.
        pos_present, neg_present = (pos_count > 0).float(), (neg_count > 0).float()
        return (masked_mean(bce, positive, normalizer=pos_count) * pos_present +
                masked_mean(bce, negative, normalizer=neg_count) * neg_present) / class_count.clamp_min(1)
    pos_count, neg_count = positive.sum(), negative.sum()
    # No Python GPU synchronization in loss construction.
    pos_present, neg_present = (pos_count > 0).float(), (neg_count > 0).float()
    return (masked_mean(bce, positive) * pos_present + masked_mean(bce, negative) *
            neg_present) / (pos_present + neg_present).clamp_min(1)


def regression_loss(pred: Tensor, target: Tensor, valid: Tensor | None,
                    scale: tuple[float, ...], logvar: Tensor | None = None,
                    *, normalizer: Tensor | None = None) -> Tensor:
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
    return masked_mean(loss, mask, normalizer=normalizer)


LOSS_NORMALIZER_KEYS = ("plan_complete", "occ_positive", "occ_negative", "occ_classes",
                        "lane_positive", "lane_negative", "lane_classes", "history_valid",
                        "state_valid", "stop_valid")


def build_loss_normalizers(full_cpu_batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Count full-effective-batch labels BEFORE moving sliced batches to CUDA.

    Return detached scalar CPU int64 tensors. Move this dict to the loss device
    once with to_device(), then pass it unchanged to EVERY microbatch. Each
    compute_loss(..., normalizers=...) is an additive contribution: sum losses,
    parts and gradients; do not average again by chunk count or batch size.

    This preserves reduction semantics, not dropout RNG, floating-point order,
    adaptive BatchNorm statistics, or other batch-coupled model computation.
    """
    required = ("gt_plan", "history_target", "state_target", "occ_target", "occ_valid",
                "lane_target", "lane_valid")
    optional = ("plan_valid", "history_valid", "state_valid")
    for name in (*required, *[k for k in optional if k in full_cpu_batch]):
        value = full_cpu_batch.get(name)
        if not isinstance(value, Tensor) or value.device.type != "cpu" or value.requires_grad:
            raise ValueError(f"Full-batch label {name} must be a detached CPU tensor")
    gt, history, state = (full_cpu_batch[k] for k in ("gt_plan", "history_target", "state_target"))
    if gt.ndim != 3 or gt.shape[1:] != (6, 2) or len(gt) < 1:
        raise ValueError("Full-batch gt_plan must be nonempty [B,6,2]")
    b = len(gt)
    if state.shape != (b, 6) or history.ndim != 3 or history.shape[0] != b or history.shape[-1] != 4:
        raise ValueError("Full-batch state/history target shape mismatch")
    plan_mask = full_cpu_batch.get("plan_valid", torch.ones((b, 6), dtype=torch.bool))
    if plan_mask.shape != (b, 6):
        raise ValueError("Full-batch plan_valid must be [B,6]")
    state_mask = full_cpu_batch.get("state_valid", torch.ones_like(state, dtype=torch.bool))
    if state_mask.shape != state.shape:
        raise ValueError("Full-batch state_valid must match [B,6]")
    result = {"plan_complete": plan_mask.bool().all(-1).sum(),
              "history_valid": _mask_like(full_cpu_batch.get("history_valid"), history).sum(),
              "state_valid": _mask_like(state_mask[..., :5], state[..., :5]).sum(),
              "stop_valid": state_mask[..., 5].bool().sum()}
    for prefix in ("occ", "lane"):
        target, valid = full_cpu_batch[prefix + "_target"], full_cpu_batch[prefix + "_valid"]
        if target.shape != valid.shape or target.ndim < 2 or target.shape[0] != b:
            raise ValueError(f"Full-batch {prefix} raster shape mismatch")
        mask = valid.bool()
        safe = torch.where(mask, target.float(), torch.zeros_like(target.float()))
        positive, negative = (mask & (safe >= .5)).sum(), (mask & (safe < .5)).sum()
        result[prefix + "_positive"], result[prefix + "_negative"] = positive, negative
        result[prefix + "_classes"] = (positive > 0).long() + (negative > 0).long()
    return {name: result[name].detach().to(dtype=torch.int64) for name in LOSS_NORMALIZER_KEYS}


def _validate_loss_normalizers(normalizers: Mapping[str, Tensor], device: torch.device) -> None:
    if not isinstance(normalizers, Mapping) or set(normalizers) != set(LOSS_NORMALIZER_KEYS):
        raise ValueError("Use the complete build_loss_normalizers result without changing keys")
    for name, value in normalizers.items():
        if (not isinstance(value, Tensor) or value.ndim != 0 or value.dtype != torch.int64
                or value.device != device or value.requires_grad):
            raise ValueError(f"Loss normalizer {name} must be detached scalar int64 on {device}")
        # Builder guarantees nonnegative counts; metadata checks above never
        # synchronize GPU execution. Extra value validation is CPU-only.
        if device.type == "cpu" and value.item() < 0:
            raise ValueError(f"Loss normalizer {name} must be nonnegative")


def global_binary_class_weights(negative_count: int, positive_count: int) -> tuple[float, float]:
    """Return fixed inverse-frequency weights as (negative, positive).

    Counts must cover the complete train split.  They are deliberately not
    inferred from a minibatch, where either class can be absent.
    """
    if (type(negative_count) is not int or type(positive_count) is not int
            or negative_count <= 0 or positive_count <= 0):
        raise ValueError("Global binary class counts must be positive integers")
    total = negative_count + positive_count
    return total / (2. * negative_count), total / (2. * positive_count)


def _stop_sample_weights(target: Tensor, class_weights: tuple[float, float]) -> Tensor:
    if (not isinstance(class_weights, tuple) or len(class_weights) != 2
            or any(type(x) is not float or not math.isfinite(x) or x <= 0
                   for x in class_weights)):
        raise ValueError("Stop class weights must be finite and positive")
    value = target.new_tensor(class_weights)
    return torch.where(target >= .5, value[1], value[0])


def compute_loss(outputs: Mapping[str, Tensor], batch: Mapping[str, Tensor],
                 weights: LossWeights, *, normalizers: Mapping[str, Tensor] | None = None,
                 stop_class_weights: tuple[float, float] | None = None
                 ) -> tuple[Tensor, dict[str, Tensor]]:
    pred = outputs["plan_abs"].float()
    if normalizers is not None:
        _validate_loss_normalizers(normalizers, pred.device)
    plan_valid = batch.get("plan_valid", torch.ones_like(pred[..., 0], dtype=torch.bool))
    # Official metric requires all six samples. Do not quietly redefine metric.
    complete = plan_valid.bool().all(-1)
    safe_gt = torch.where(complete[:, None, None], batch["gt_plan"].float(),
                          torch.zeros_like(pred))
    d3 = weighted_d3(pred, safe_gt)
    plan_loss = masked_mean(d3, complete, normalizer=None if normalizers is None else normalizers["plan_complete"])
    raster_norms = lambda prefix: None if normalizers is None else tuple(
        normalizers[prefix + "_" + suffix] for suffix in ("positive", "negative", "classes"))
    occ = balanced_raster_bce(outputs["occ_logits"], batch["occ_target"], batch["occ_valid"], normalizers=raster_norms("occ"))
    lane = balanced_raster_bce(outputs["lane_logits"], batch["lane_target"], batch["lane_valid"], normalizers=raster_norms("lane"))
    history = regression_loss(outputs["history_hat"], batch["history_target"],
                              batch.get("history_valid"), HISTORY_SCALE,
                              outputs.get("history_logvar") if weights.uncertainty else None,
                              normalizer=None if normalizers is None else normalizers["history_valid"])
    state_valid = batch.get("state_valid", torch.ones_like(batch["state_target"], dtype=torch.bool))
    state = regression_loss(outputs["state_hat"][..., :5], batch["state_target"][..., :5],
                            state_valid[..., :5], STATE_SCALE,
                            outputs.get("state_logvar") if weights.uncertainty else None,
                            normalizer=None if normalizers is None else normalizers["state_valid"])
    stop_target = torch.where(state_valid[..., 5].bool(), batch["state_target"][..., 5].float(),
                              torch.zeros_like(batch["state_target"][..., 5].float()))
    stop_values = F.binary_cross_entropy_with_logits(
        outputs["state_hat"][..., 5].float(), stop_target, reduction="none")
    if stop_class_weights is not None:
        stop_values = stop_values * _stop_sample_weights(stop_target, stop_class_weights)
    stop = masked_mean(stop_values, state_valid[..., 5],
        normalizer=None if normalizers is None else normalizers["stop_valid"])
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
