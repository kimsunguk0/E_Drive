"""Status-conditioned shared perception with an explicit path/progress planner.

Provided causal status is used only while forming image-derived features shared
by occupancy, lane, motion and planning.  The progress head receives decoded
planner features and the model's own detached proposal; it has no raw status,
pose or goal argument.
"""
from __future__ import annotations

import contextlib

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.motiondrive_v2.model import MotionDriveV2
from models.motiondrive_v2.shared_status_query import SharedCausalStatusQuery


STATUS_SCALE = (10.0, 5.0, 3.0, 3.0, 0.5)
DT = 0.5
T_MID = (0.25, 0.75, 1.25, 1.75, 2.25, 2.75)


class SharedStatusFeatureConditioner(nn.Module):
    """Multiplicatively condition common image features without adding values."""

    def __init__(self, channels: int = 128, gain: float = 0.5):
        super().__init__()
        if channels <= 0 or gain <= 0:
            raise ValueError("channels and gain must be positive")
        self.channels = int(channels)
        self.gain = float(gain)
        self.register_buffer("status_scale", torch.tensor(STATUS_SCALE, dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(5, channels), nn.GELU(),
            nn.Linear(channels, 3 * channels),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, status5: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if status5.ndim != 2 or status5.shape[1] != 5:
            raise ValueError("provided causal status must be [B,5]")
        if not bool(torch.isfinite(status5).all()):
            raise ValueError("provided causal status must be finite")
        with torch.autocast(device_type=status5.device.type, enabled=False):
            raw = self.network(status5.float() / self.status_scale.float())
            gates = 1.0 + self.gain * raw.tanh()
        return tuple(gates.chunk(3, dim=-1))


def _apply_gate(feature: Tensor, gate: Tensor) -> Tensor:
    """Apply [B,C] multiplicative conditioning to [B,...,C,H,W]."""
    if feature.shape[0] != gate.shape[0] or feature.shape[-3] != gate.shape[1]:
        raise ValueError(f"feature/gate mismatch: {tuple(feature.shape)} / {tuple(gate.shape)}")
    shape = [len(gate)] + [1] * (feature.ndim - 4) + [gate.shape[1], 1, 1]
    return feature * gate.to(feature.dtype).reshape(shape)


def differentiable_directions(plan: Tensor, eps: float = 1e-3) -> tuple[Tensor, Tensor]:
    """Return segment lengths and stable directions while retaining shape gradients."""
    if plan.ndim != 3 or plan.shape[-2:] != (6, 2):
        raise ValueError(f"plan must be [B,6,2], got {tuple(plan.shape)}")
    source = plan.float()
    segments = torch.diff(torch.cat([torch.zeros_like(source[:, :1]), source], dim=1), dim=1)
    lengths = torch.linalg.vector_norm(segments, dim=-1)
    directions = segments / lengths.clamp_min(eps)[..., None]
    # Short segments borrow the nearest valid predicted direction.  If the
    # whole proposal is stationary, ego-forward +x is the deterministic basis.
    rows = []
    for batch in range(len(source)):
        good = torch.nonzero(lengths[batch].detach() >= eps, as_tuple=False).flatten()
        if not len(good):
            fallback = torch.zeros_like(directions[batch])
            fallback[:, 0] = 1.0
            rows.append(fallback)
            continue
        row = []
        for index in range(6):
            if bool(lengths[batch, index].detach() >= eps):
                row.append(directions[batch, index])
            else:
                nearest = good[torch.argmin((good - index).abs())]
                row.append(directions[batch, nearest])
        rows.append(torch.stack(row))
    return lengths, torch.stack(rows)


def compose_factorized_plan(base_plan: Tensor, coefficients: Tensor,
                            eps: float = 1e-3) -> tuple[Tensor, dict[str, Tensor]]:
    """Compose direction and progress, blocking the proposal's length gradient.

    Values at zero correction reproduce the proposal.  Gradients reaching the
    proposal are carried by unit directions only; longitudinal progress must be
    learned by the dedicated coefficient head.
    """
    if coefficients.ndim != 2 or coefficients.shape[0] != len(base_plan) \
            or coefficients.shape[1] not in (1, 2):
        raise ValueError("coefficients must be [B,1] or [B,2]")
    lengths, directions = differentiable_directions(base_plan, eps=eps)
    detached_lengths = lengths.detach()
    correction_speed = coefficients.float()[:, :1]
    if coefficients.shape[1] == 2:
        correction_speed = correction_speed + coefficients.float()[:, 1:2] \
            * coefficients.new_tensor(T_MID)[None]
    adjusted_lengths = (detached_lengths + DT * correction_speed).clamp_min(0.0)
    reconstructed_base = torch.cumsum(directions * detached_lengths[..., None], dim=1)
    # This detached residual makes the forward value exactly follow the proposal
    # without restoring its direct longitudinal gradient path.
    parity_residual = (base_plan.float() - reconstructed_base).detach()
    final = torch.cumsum(directions * adjusted_lengths[..., None], dim=1) + parity_residual
    return final, {
        "base_segment_length": detached_lengths,
        "progress_segment_length": adjusted_lengths,
        "progress_direction": directions,
    }


class ContinuousProgressHead(nn.Module):
    """Predict a low-dimensional continuous correction from planner evidence."""

    def __init__(self, channels: int, coefficient_count: int):
        super().__init__()
        if coefficient_count not in (1, 2):
            raise ValueError("coefficient_count must be one or two")
        self.coefficient_count = int(coefficient_count)
        self.decoded = nn.Sequential(
            nn.LayerNorm(6 * channels), nn.Linear(6 * channels, 2 * channels),
            nn.GELU(), nn.Linear(2 * channels, channels),
        )
        self.plan = nn.Sequential(nn.Linear(18, channels), nn.GELU(), nn.Linear(channels, channels))
        self.output = nn.Sequential(
            nn.LayerNorm(2 * channels), nn.Linear(2 * channels, channels),
            nn.GELU(), nn.Linear(channels, coefficient_count),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        caps = [1.0] if coefficient_count == 1 else [1.0, 0.5]
        self.register_buffer("coefficient_cap", torch.tensor(caps, dtype=torch.float32))

    @staticmethod
    def encode_plan(base_plan: Tensor) -> Tensor:
        plan = base_plan.detach().float()
        segments = torch.diff(torch.cat([torch.zeros_like(plan[:, :1]), plan], 1), dim=1)
        lengths = torch.linalg.vector_norm(segments, dim=-1)
        return torch.cat([(plan / plan.new_tensor([10.0, 5.0])).flatten(1), lengths / 10.0], -1)

    def forward(self, decoded: Tensor, base_plan: Tensor) -> Tensor:
        if decoded.ndim != 3 or decoded.shape[1:] != (6, self.decoded[1].in_features // 6):
            raise ValueError("decoded planner tensor must be [B,6,C]")
        with torch.autocast(device_type=decoded.device.type, enabled=False):
            hidden = torch.cat([
                self.decoded(decoded.float().flatten(1)),
                self.plan(self.encode_plan(base_plan)),
            ], dim=-1)
            return self.coefficient_cap * self.output(hidden).tanh()


class SharedDynamicsMotionDriveV2(MotionDriveV2):
    """MR graph with shared status perception and optional factorized progress."""

    VALID_ARMS = {
        "A2-DIRECT": 0,
        "A3-DIRECT": 0,
        "A3-FP-S": 1,
        "A3-FP-VA": 2,
    }

    def __init__(self, config, *, arm: str):
        if arm not in self.VALID_ARMS:
            raise ValueError(f"unknown shared-dynamics arm: {arm}")
        super().__init__(config)
        self.arm = arm
        self.coefficient_count = self.VALID_ARMS[arm]
        self._provided_status_context: Tensor | None = None

        # New-module construction must not perturb the RNG stream used by the
        # registered MR correlation-fuse rebuild.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(2026091703)
            self.shared_status_query_fusion = SharedCausalStatusQuery(
                self.config.scene_attention_channels)
            self.shared_status_feature_conditioner = (
                SharedStatusFeatureConditioner(self.config.channels)
                if arm.startswith("A3-") else None)
            self.progress_head = (
                ContinuousProgressHead(self.config.channels, self.coefficient_count)
                if self.coefficient_count else None)

        def condition_query(_module, _inputs, output):
            status = self._provided_status_context
            if status is None:
                raise RuntimeError("shared causal status context is missing")
            return self.shared_status_query_fusion(output, status)

        query_handle = self.scene_encoder.query_context.register_forward_hook(condition_query)
        object.__setattr__(self, "_shared_status_query_hook_handle", query_handle)

        self._decoded_capture: list[Tensor] = []
        if self.progress_head is not None:
            def capture_decoded(_module, inputs):
                if self._decoded_capture or len(inputs) != 1:
                    raise RuntimeError("planner decoder capture is stale or malformed")
                self._decoded_capture.append(inputs[0])
            handle = self.planner.xy_head.register_forward_pre_hook(capture_decoded)
            object.__setattr__(self, "_decoded_hook_handle", handle)

    def _condition_features(self, status5: Tensor, current_levels, current_p4,
                            history_levels, motion_levels):
        conditioner = self.shared_status_feature_conditioner
        if conditioner is None:
            return current_levels, current_p4, history_levels, motion_levels
        scene_gate, motion_gate, global_gate = conditioner(status5)
        current_levels = tuple(_apply_gate(value, scene_gate) for value in current_levels)
        history_levels = tuple(_apply_gate(value, scene_gate) for value in history_levels)
        current_p4 = _apply_gate(current_p4, global_gate)
        motion_levels = tuple(_apply_gate(value, motion_gate) for value in motion_levels)
        return current_levels, current_p4, history_levels, motion_levels

    def forward_parts(self, images, history_images, lidar2img, history_transforms,
                      time_offsets, goal_xy, motion_current=None, motion_history=None):
        if motion_current is None or motion_history is None:
            raise ValueError("shared-dynamics MR requires native motion canvas tensors")
        status5 = self._provided_status_context
        if status5 is None:
            raise RuntimeError("provided status context must be set before feature extraction")
        b = len(images)
        current_levels, current_p4 = self.backbone_fpn(images.flatten(0, 1))
        current_levels = tuple(value.reshape(b, 6, *value.shape[1:]) for value in current_levels)
        current_p4 = current_p4.reshape(b, 6, *current_p4.shape[1:])

        scene_small = F.interpolate(images[:, 0], size=history_images.shape[-2:],
                                    mode="bilinear", align_corners=False, antialias=True)
        scene_stack = torch.cat([scene_small[:, None], history_images], dim=1)
        scene_levels, _ = self.backbone_fpn(scene_stack.flatten(0, 1))
        scene_levels = tuple(value.reshape(b, self.config.n_history + 1, *value.shape[1:])
                             for value in scene_levels)
        history_levels = tuple(value[:, 1:] for value in scene_levels)

        motion_stack = torch.cat([motion_current[:, None], motion_history], dim=1)
        motion_levels, _ = self.backbone_fpn(motion_stack.flatten(0, 1))
        motion_levels = tuple(value.reshape(b, self.config.n_history + 1, *value.shape[1:])
                              for value in motion_levels)
        current_levels, current_p4, history_levels, motion_levels = self._condition_features(
            status5, current_levels, current_p4, history_levels, motion_levels)

        motion = self.motion_encoder(tuple(value[:, 0] for value in motion_levels),
                                     tuple(value[:, 1:] for value in motion_levels), time_offsets)
        scene = self.scene_encoder(current_levels, history_levels, current_p4, lidar2img,
                                   history_transforms, time_offsets, goal_xy, images.shape[-2:])
        return {**scene, **motion}

    def forward(self, images, history_images, lidar2img, history_transforms,
                time_offsets, goal_xy, motion_current=None, motion_history=None,
                provided_status5=None):
        if provided_status5 is None:
            raise ValueError("shared-dynamics model requires provided_status5")
        if self._provided_status_context is not None or self._decoded_capture:
            raise RuntimeError("shared-dynamics forward is reentrant or stale")
        self._provided_status_context = provided_status5
        try:
            parts = self.forward_parts(images, history_images, lidar2img,
                                       history_transforms, time_offsets, goal_xy,
                                       motion_current, motion_history)
            base_plan = self.plan_from_features(parts["scene_features"], parts["motion_features"],
                                                parts["state_hat"], parts["history_hat"])
            if self.progress_head is None:
                return {**parts, "plan_abs": base_plan}
            if len(self._decoded_capture) != 1:
                raise RuntimeError("factorized planner did not capture exactly one decoded tensor")
            coefficient = self.progress_head(self._decoded_capture[0], base_plan.detach())
            final_plan, geometry = compose_factorized_plan(base_plan, coefficient)
            return {**parts, **geometry, "plan_base_abs": base_plan,
                    "progress_coeff": coefficient, "plan_abs": final_plan}
        finally:
            self._provided_status_context = None
            self._decoded_capture.clear()


__all__ = [
    "SharedStatusFeatureConditioner", "ContinuousProgressHead",
    "SharedDynamicsMotionDriveV2", "compose_factorized_plan",
    "differentiable_directions",
]
