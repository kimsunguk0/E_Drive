"""Opt-in causal-status conditioning of shared scene-attention queries.

The provided status changes only the query used to select image-derived
values.  It is not forwarded to the planner or converted into a scene value.
This experimental helper does not modify the production model class.
"""
from __future__ import annotations

import types
from typing import Mapping

import torch
from torch import Tensor, nn


STATUS_SCALE = (10.0, 5.0, 3.0, 3.0, 0.5)


class SharedCausalStatusQuery(nn.Module):
    """Add a causal-status delta to pre-attention query context."""

    def __init__(self, query_channels: int = 32):
        super().__init__()
        if query_channels != 32:
            raise ValueError("shared-status query experiment is fixed at 32 channels")
        self.query_channels = query_channels
        self.status_mlp = nn.Sequential(
            nn.Linear(5, 32, bias=True),
            nn.GELU(),
            nn.Linear(32, query_channels, bias=True),
        )
        self.register_buffer("status_scale", torch.tensor(STATUS_SCALE, dtype=torch.float32))
        nn.init.zeros_(self.status_mlp[-1].weight)
        nn.init.zeros_(self.status_mlp[-1].bias)

    def forward(self, query_context: Tensor, status5: Tensor) -> Tensor:
        if query_context.ndim != 3 or query_context.shape[-1] != self.query_channels:
            raise ValueError("query context must be [B,Q,32]")
        if status5.shape != (len(query_context), 5) or not status5.is_floating_point():
            raise ValueError("provided causal status must be floating [B,5]")
        if status5.device != query_context.device:
            raise ValueError("query context and status must share one device")
        if (not bool(torch.isfinite(query_context).all())
                or not bool(torch.isfinite(status5).all())):
            raise ValueError("query context and status must be finite")
        with torch.autocast(device_type=query_context.device.type, enabled=False):
            delta = self.status_mlp(status5.float() / self.status_scale.float())
            result = query_context.float() + delta[:, None, :]
        return result.to(query_context.dtype)


def install_shared_status_query(model: nn.Module) -> nn.Module:
    """Install exception-safe status conditioning before scene attention."""
    if hasattr(model, "shared_status_query_fusion"):
        raise ValueError("shared-status query fusion is already installed")
    if hasattr(model, "shared_status_fusion"):
        raise ValueError("shared-status query fusion cannot stack with A1 fusion")
    query_context = getattr(getattr(model, "scene_encoder", None), "query_context", None)
    if not isinstance(query_context, nn.Module):
        raise ValueError("MotionDrive scene query-context module is missing")
    fusion = SharedCausalStatusQuery()
    model.add_module("shared_status_query_fusion", fusion)
    context: dict[str, Tensor | None] = {"status": None}

    def after_query_context(_module, _args, output):
        status = context["status"]
        if status is None:
            raise RuntimeError("shared-status query context is missing or stale")
        return fusion(output, status)

    handle = query_context.register_forward_hook(after_query_context)
    original_forward = model.forward

    def experimental_forward(self, images, history_images, lidar2img, history_transforms,
                             time_offsets, goal_xy, provided_status5):
        if context["status"] is not None:
            raise RuntimeError("shared-status query model does not permit reentrant forward")
        context["status"] = provided_status5
        try:
            return original_forward(images, history_images, lidar2img, history_transforms,
                                    time_offsets, goal_xy)
        finally:
            context["status"] = None

    model.forward = types.MethodType(experimental_forward, model)
    object.__setattr__(model, "_shared_status_query_hook_handle", handle)
    object.__setattr__(model, "_shared_status_query_context", context)
    return model


def shared_status_query_state(model: nn.Module) -> Mapping[str, Tensor]:
    prefix = "shared_status_query_fusion."
    state = {name: value for name, value in model.state_dict().items()
             if name.startswith(prefix)}
    if not state:
        raise ValueError("shared-status query state is absent")
    return state
