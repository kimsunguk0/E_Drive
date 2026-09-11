"""Temporal common perception around an unchanged public SparseDriveV2 model.

The wrapped planner always receives status=zeros. Optional perception_status is
used only to condition candidate-independent image temporal attention queries.
No goal, candidate bank, planner token, GT, or state auxiliary prediction is an
input to temporal fusion. The original planner forward is reused without edits
through one scoped backbone-output hook; the wrapper owns its base exclusively.

Current camera order is [front-left, front, front-right]. Historical front images
are at nominal .1/.5 seconds. Public DFA remains a three-current-camera operator.
"""
from __future__ import annotations

import threading
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ImageTemporalFusion(nn.Module):
    """Image attention, an image-only state auxiliary branch, and FPN residual."""

    def __init__(self, channels=256, heads=8):
        super().__init__()
        self.channels = int(channels)
        self.position = nn.Linear(2, channels)
        self.time = nn.Sequential(nn.Linear(1, channels), nn.GELU(), nn.Linear(channels, channels))
        self.query_norm = nn.LayerNorm(channels)
        self.key_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=0., batch_first=True)
        self.status_condition = nn.Sequential(nn.Linear(4, channels), nn.GELU(), nn.Linear(channels, channels))
        nn.init.zeros_(self.status_condition[-1].weight)
        nn.init.zeros_(self.status_condition[-1].bias)
        self.register_buffer("status_scale", torch.tensor([20., 5., 3., 3.]))
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.state_head = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 128),
                                        nn.GELU(), nn.Linear(128, 4))

    def forward(self, current: torch.Tensor, history: torch.Tensor,
                time_offsets: torch.Tensor, perception_status: torch.Tensor):
        # current B,C,H,W; history B,2,C,H,W. Neither contains candidate features.
        b, c, h, w = current.shape
        if c != self.channels or history.shape != (b, 2, c, h, w):
            raise ValueError("Temporal FPN shape mismatch")
        yy, xx = torch.meshgrid(torch.linspace(-1., 1., h, device=current.device),
                                torch.linspace(-1., 1., w, device=current.device), indexing="ij")
        spatial = self.position(torch.stack((xx, yy), -1).reshape(h*w, 2)).to(current.dtype)
        now = current.flatten(2).transpose(1, 2)
        past = history.flatten(3).transpose(2, 3)  # B,T,N,C
        temporal = self.time((-time_offsets.float())[..., None]).to(current.dtype)
        query = self.query_norm(now + spatial[None])
        keys = self.key_norm(past + spatial[None, None] + temporal[:, :, None]).flatten(1, 2)
        values = past.flatten(1, 2)  # image values only, no status/time/position residual
        # Unconditioned pass is mandatory in all arms. State aux cannot copy raw status.
        unconditioned = self.attention(query, keys, values, need_weights=False)[0]
        state = self.state_head((now + unconditioned).mean(1)).float()
        with torch.autocast(device_type=current.device.type, enabled=False):
            condition = self.status_condition(perception_status.float() / self.status_scale.float())
        # This query belongs to image feature fusion, upstream of every planner query.
        conditioned = self.attention(query + condition[:, None].to(query.dtype),
                                     keys, values, need_weights=False)[0]
        delta = self.output(conditioned.transpose(1, 2).reshape(b, c, h, w))
        return delta, state


class SpatialPerceptionAux(nn.Module):
    """Current annotated-object/lane prediction; not complete free space.

    Raster axes: row=x forward [-10,70], column=y left [-32,32], 64x48
    cell centers. Read projected points at z=0 OR z=1m from the same fused
    three-camera FPN used by planning. Labels and label validity are loss-only.
    """

    def __init__(self, channels=256, hidden=32, grid_shape=(64, 48)):
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        nx, ny = self.grid_shape
        x = -10. + (torch.arange(nx).float()+.5) * (80./nx)
        y = -32. + (torch.arange(ny).float()+.5) * (64./ny)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        xy = torch.stack((xx, yy), -1).reshape(-1, 1, 2).expand(-1, 2, -1)
        z = torch.tensor([0., 1.]).reshape(1, 2, 1).expand(nx*ny, -1, -1)
        points = torch.cat((xy, z, torch.ones_like(z)), -1)
        self.register_buffer("points", points)
        self.projections = nn.ModuleList([nn.Conv2d(channels, hidden, 1) for _ in range(4)])
        self.shared = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.occupancy = nn.Conv2d(hidden, 1, 1)
        self.lane = nn.Conv2d(hidden, 1, 1)

    def projection_grid(self, lidar2img, image_hw):
        b = len(lidar2img)
        with torch.autocast(device_type=lidar2img.device.type, enabled=False):
            hw = torch.as_tensor(image_hw, device=lidar2img.device, dtype=torch.float32)
            if hw.shape == (2,):
                hw = hw[None].expand(b, -1)
            if hw.shape == (b, 2):
                hw = hw[:, None].expand(-1, 3, -1)
            if hw.shape != (b, 3, 2) or bool((hw <= 0).any()) or not bool(torch.isfinite(hw).all()):
                raise ValueError("image_hw must be positive [H,W], [B,2], or [B,3,2]")
            if lidar2img.shape != (b, 3, 4, 4) or not bool(torch.isfinite(lidar2img).all()):
                raise ValueError("Invalid camera projection")
            projected = torch.einsum("bcij,nhj->bcnhi", lidar2img.float(), self.points.float())
            depth = projected[..., 2]
            pixel = projected[..., :2] / depth.clamp_min(1e-6)[..., None]
            wh = hw.flip(-1)[:, :, None, None]
            uv = pixel / wh
            valid = (depth > .05) & (uv[..., 0] >= 0) & (uv[..., 0] < 1) & (uv[..., 1] >= 0) & (uv[..., 1] < 1)
            # Invalid grid points never contribute and finite sentinels avoid NaN backward.
            grid = (uv*2.-1.).masked_fill(~valid[..., None], 2.)
        return grid, valid

    def forward(self, levels: Sequence[torch.Tensor], lidar2img, image_hw):
        grid, valid = self.projection_grid(lidar2img, image_hw)
        b, cams, n, heights, _ = grid.shape
        total = None
        for projection, feature in zip(self.projections, levels):
            projected = projection(feature.flatten(0, 1))
            with torch.autocast(device_type=feature.device.type, enabled=False):
                sampled = F.grid_sample(projected.float(), grid.reshape(b*cams, n, heights, 2),
                                        mode="bilinear", padding_mode="zeros", align_corners=False)
                sampled = sampled.reshape(b, cams, projected.shape[1], n, heights)
                pooled = (sampled * valid[:, :, None].float()).sum((1, 4))
                total = pooled if total is None else total + pooled
        denominator = (valid.float().sum((1, 3)) * len(levels)).clamp_min(1.)
        raster = (total / denominator[:, None]).reshape(b, -1, *self.grid_shape)
        visible = valid.any(1).any(-1).reshape(b, 1, *self.grid_shape)
        shared = self.shared(raster)
        return self.occupancy(shared).float(), self.lane(shared).float(), visible


class TemporalPerceptionModel(nn.Module):
    """Matched repeat/history/common-status wrapper preserving base tensors.

    Construct with an already verified PublicSparseDriveV2 from the pinned public
    checkpoint and chosen train-only bank. No weight downloads or state remapping
    happen here. All base parameter names/values are retained below ``base.``.
    """

    def __init__(self, base: nn.Module, *, freeze_batch_norm=True, grid_shape=(64, 48)):
        super().__init__()
        if not isinstance(getattr(base, "_backbone", None), nn.Module):
            raise TypeError("Expected public model with _backbone")
        if not isinstance(getattr(base, "_status_encoding", None), nn.Linear):
            raise TypeError("Expected original public status encoding for constant-zero input")
        if base._status_encoding.in_features != 8:
            raise ValueError("Base status input must be exactly eight-dimensional")
        self.base = base
        self.temporal = ImageTemporalFusion()
        self.perception = SpatialPerceptionAux(grid_shape=grid_shape)
        self.freeze_batch_norm = bool(freeze_batch_norm)
        self.register_buffer("nominal_time_offsets", torch.tensor([.1, .5]))
        self._forward_lock = threading.Lock()
        self.train(self.training)

    @property
    def _trajectory_head(self):
        return self.base._trajectory_head

    @property
    def _backbone(self):
        return self.base._backbone

    def __getstate__(self):
        state = super().__getstate__().copy()
        state.pop("_forward_lock", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._forward_lock = threading.Lock()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_batch_norm:
            for module in self.base._backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def _validate(self, images, lidar2img, history_images, time_offsets, perception_status):
        if images.ndim != 5 or images.shape[1:3] != (3, 3) or not images.is_floating_point():
            raise ValueError("images must be floating [B,3,3,H,W]")
        b = len(images)
        if history_images.shape != (b, 2, 3, *images.shape[-2:]) or not history_images.is_floating_point():
            raise ValueError("history_images must be [B,2,3,H,W] at current image resolution")
        if history_images.device != images.device or lidar2img.device != images.device:
            raise ValueError("All input tensors must share device")
        if time_offsets.shape != (b, 2) or time_offsets.device != images.device:
            raise ValueError("time_offsets must be [B,2] on image device")
        if not torch.allclose(time_offsets.float(), self.nominal_time_offsets[None].expand(b, -1), atol=1e-6, rtol=0.):
            raise ValueError("Only pinned nominal .1/.5-second history is supported")
        if perception_status is None:
            perception_status = images.new_zeros(b, 4, dtype=torch.float32)
        if (perception_status.shape != (b, 4) or perception_status.device != images.device
                or not perception_status.is_floating_point() or not bool(torch.isfinite(perception_status).all())):
            raise ValueError("perception_status must be finite floating [B,4] on image device")
        return perception_status

    def forward(self, images, lidar2img, image_hw, history_images, time_offsets,
                perception_status=None):
        if not self._forward_lock.acquire(blocking=False):
            raise RuntimeError("Temporal wrapper forbids concurrent/reentrant forward")
        handle = None
        try:
            status = self._validate(images, lidar2img, history_images, time_offsets, perception_status)
            # History executes before the hook exists, so the original backbone is used.
            histories = self.base._backbone(history_images)
            if len(histories) != 4:
                raise ValueError("Expected four public FPN levels")
            auxiliary = {}
            calls = 0

            def fuse_backbone(_module, _args, levels):
                nonlocal calls
                calls += 1
                if calls != 1:
                    raise RuntimeError("Original base invoked backbone more than once")
                if len(levels) != 4 or any(f.shape[1:3] != (3, 256) for f in levels):
                    raise ValueError("Current public FPN must contain 3 cameras and 256 channels")
                # Public FPN strides 4/8/16/32: index2 is P4, front index1.
                delta, state = self.temporal(levels[2][:, 1], histories[2], time_offsets, status)
                fused = []
                for feature in levels:
                    update = F.interpolate(delta, size=feature.shape[-2:], mode="bilinear", align_corners=False)
                    front = feature[:, 1] + update.to(feature.dtype)
                    fused.append(torch.stack((feature[:, 0], front, feature[:, 2]), 1))
                occ, lane, visible = self.perception(fused, lidar2img, image_hw)
                auxiliary.update(aux_occ=occ, aux_lane=lane, aux_state=state, aux_visible=visible)
                return fused

            handle = self.base._backbone.register_forward_hook(fuse_backbone)
            # Explicitly constructed zeros: raw/aux states cannot enter original planner.
            result = self.base(images=images, lidar2img=lidar2img, image_hw=image_hw,
                               status=images.new_zeros(len(images), 8))
            if calls != 1:
                raise RuntimeError("Original base did not invoke expected backbone exactly once")
            if any(k in result for k in auxiliary):
                raise ValueError("Base output conflicts with auxiliary namespace")
            return {**result, **auxiliary}
        finally:
            if handle is not None:
                handle.remove()
            self._forward_lock.release()


TemporalSparseDriveV2 = TemporalPerceptionModel
