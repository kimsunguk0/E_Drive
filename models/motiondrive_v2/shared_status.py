"""Opt-in shared-scene conditioning by a causal provided ego status.

This module is deliberately separate from the MotionDrive V2 production tree.
It modulates continuous image-derived scene features before the existing shared
SpatialMix refinement.  Status never becomes an additive scene value or a
planner argument.
"""
from __future__ import annotations

import types
from typing import Mapping

import torch
from torch import Tensor, nn


STATUS_SCALE = (10.0, 5.0, 3.0, 3.0, 0.5)


class SharedCausalStatusFiLM(nn.Module):
    """Beta-free channel FiLM: F * (1 + tanh(gamma(status)))."""

    def __init__(self, channels: int = 128):
        super().__init__()
        if channels != 128:
            raise ValueError("shared_status_a1 is fixed at 128 scene channels")
        self.channels = channels
        self.status_mlp = nn.Sequential(
            nn.Linear(5, 32, bias=True), nn.GELU(),
            nn.Linear(32, channels, bias=True),
        )
        self.register_buffer("status_scale", torch.tensor(STATUS_SCALE, dtype=torch.float32))
        nn.init.zeros_(self.status_mlp[-1].weight)
        nn.init.zeros_(self.status_mlp[-1].bias)

    def forward(self, scene_raster: Tensor, status5: Tensor) -> Tensor:
        if scene_raster.ndim != 4 or scene_raster.shape[1] != self.channels:
            raise ValueError("scene raster must be [B,128,H,W]")
        if status5.shape != (len(scene_raster), 5) or not status5.is_floating_point():
            raise ValueError("provided causal status must be floating [B,5]")
        if status5.device != scene_raster.device:
            raise ValueError("scene and status must share one device")
        if not bool(torch.isfinite(scene_raster).all()) or not bool(torch.isfinite(status5).all()):
            raise ValueError("scene and status must be finite")
        with torch.autocast(device_type=scene_raster.device.type, enabled=False):
            gamma = self.status_mlp(status5.float() / self.status_scale.float())
            result = scene_raster.float() * (1.0 + gamma.tanh())[:, :, None, None]
        return result.to(scene_raster.dtype)


def install_shared_status(model: nn.Module) -> nn.Module:
    """Install the opt-in FiLM and an exception-safe per-call status context.

    The returned object is the original MotionDriveV2 instance, preserving its
    parameter names and planner API.  Only this experimental instance is
    modified; the class and production source are untouched.
    """
    if hasattr(model, "shared_status_fusion"):
        raise ValueError("shared-status fusion is already installed")
    refine = getattr(getattr(model, "scene_encoder", None), "refine", None)
    if not isinstance(refine, nn.Module):
        raise ValueError("MotionDrive scene refinement module is missing")
    fusion = SharedCausalStatusFiLM()
    model.add_module("shared_status_fusion", fusion)
    context: dict[str, Tensor | None] = {"status": None}

    def before_refine(_module, args):
        if len(args) != 1:
            raise RuntimeError("shared-status refine hook expected one raster")
        status = context["status"]
        if status is None:
            raise RuntimeError("shared-status context is missing or stale")
        return (fusion(args[0], status),)

    handle = refine.register_forward_pre_hook(before_refine)
    original_forward = model.forward

    def experimental_forward(self, images, history_images, lidar2img, history_transforms,
                             time_offsets, goal_xy, provided_status5):
        if context["status"] is not None:
            raise RuntimeError("shared-status model does not permit reentrant forward")
        context["status"] = provided_status5
        try:
            return original_forward(images, history_images, lidar2img, history_transforms,
                                    time_offsets, goal_xy)
        finally:
            context["status"] = None

    model.forward = types.MethodType(experimental_forward, model)
    # Non-module metadata is intentionally excluded from state_dict.
    object.__setattr__(model, "_shared_status_hook_handle", handle)
    object.__setattr__(model, "_shared_status_context", context)
    return model


def shared_status_state(model: nn.Module) -> Mapping[str, Tensor]:
    prefix = "shared_status_fusion."
    state = {name: value for name, value in model.state_dict().items() if name.startswith(prefix)}
    if not state:
        raise ValueError("shared-status state is absent")
    return state
