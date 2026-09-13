"""Ego-aligned temporal scene adapter and a future non-ego footprint head.

The parent's scene raster already fuses six current cameras. This adds, beside
it, the same front view resolved separately at each of the five encoded times,
projected onto the current BEV grid with the existing calibration and the
existing past-pose alignment, so the shared scene can learn where annotated
non-ego objects are going. The ego-motion branch is untouched: this is a
scene-side path, not a replacement for it.

Nothing in the pinned model files is modified. The five per-time front feature
maps are taken from the backbone call the parent already makes for its
low_feature motion path, so no image is encoded twice.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.motiondrive_v2.model import MotionDriveV2
from models.motiondrive_v2.scene_encoder import project_scene_points

TIME_CHANNELS = 32


class TemporalSceneAdapter(nn.Module):
    """Per-time front features on the current grid -> a residual scene delta."""

    def __init__(self, config, times=5, channels=TIME_CHANNELS):
        super().__init__()
        self.times, self.channels = int(times), int(channels)
        self.project = nn.Conv2d(config.channels, channels, 1)
        width = self.times * (channels + 2) + 2
        self.mix = nn.Sequential(
            nn.Conv2d(width, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, config.channels, 1))
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)
        self.input_width = width

    def forward(self, samples, valid, offsets, grid_xy):
        """samples [B,T,C,H,W], valid [B,T,1,H,W], offsets [B,T], grid_xy [1,2,H,W]."""
        b, t, c, h, w = samples.shape
        if t != self.times or valid.shape != (b, t, 1, h, w) or offsets.shape != (b, t):
            raise ValueError("temporal adapter shape contract violated")
        with torch.autocast(device_type=samples.device.type, enabled=False):
            projected = self.project(samples.reshape(b * t, c, h, w).float())
            projected = projected.reshape(b, t, self.channels, h, w)
            # Re-apply the geometric mask after a projection that carries a bias.
            projected = projected * valid.float()
            time = offsets.float()[:, :, None, None, None].expand(b, t, 1, h, w) * valid.float()
            stacked = torch.cat([projected, valid.float(), time], 2).reshape(b, -1, h, w)
            features = torch.cat([stacked, grid_xy.expand(b, -1, h, w).float()], 1)
            if features.shape[1] != self.input_width:
                raise ValueError("temporal adapter channel count drifted")
            return self.mix(features)


class DynamicSceneMotionDrive(MotionDriveV2):
    """Parent graph plus the temporal scene adapter and a future footprint head."""

    def __init__(self, config, times=5):
        super().__init__(config)
        self.scene_adapter = TemporalSceneAdapter(config, times=times)
        self.future_head = nn.Sequential(
            nn.Conv2d(config.channels, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 2, 1))
        self._front_levels = None
        self.backbone_fpn.register_forward_hook(self._capture)

    def _capture(self, _module, _args, output):
        """Keep the five-image temporal stack the low_feature path already builds."""
        levels = output[0] if isinstance(output, tuple) else output
        first = levels[0]
        if first.shape[0] % (self.config.n_history + 1) == 0:
            self._front_levels = first
        return output

    def _temporal_samples(self, lidar2img, history_transforms, time_offsets, image_hw):
        times = self.config.n_history + 1
        stack = self._front_levels
        if stack is None:
            raise RuntimeError("the temporal backbone pass was not captured")
        b = lidar2img.shape[0]
        if stack.shape[0] != b * times:
            raise RuntimeError("captured stack does not hold one front view per time")
        feature = stack.reshape(b, times, *stack.shape[1:])
        with torch.autocast(device_type=lidar2img.device.type, enabled=False):
            front = lidar2img[:, :1].float()
            matrices = torch.cat([front, front @ history_transforms.float()], 1)
            grid, ok = project_scene_points(self.scene_encoder.points, matrices, image_hw)
        # grid [B,T,Q,Hh,2]; sample every height, then average the visible ones.
        q, heights = grid.shape[2], grid.shape[3]
        flat = feature.reshape(b * times, *feature.shape[2:])
        sampled = F.grid_sample(flat, grid.reshape(b * times, q, heights, 2).to(flat.dtype),
                                mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.reshape(b, times, -1, q, heights)
        weight = ok.reshape(b, times, 1, q, heights).float()
        pooled = (sampled.float() * weight).sum(-1) / weight.sum(-1).clamp_min(1.0)
        visible = weight.sum(-1) > 0
        nx, ny = self.config.grid_size
        return (pooled.reshape(b, times, -1, nx, ny),
                visible.reshape(b, times, 1, nx, ny),
                torch.cat([time_offsets.new_zeros(b, 1), -time_offsets.float()], 1))

    def forward(self, images, history_images, lidar2img, history_transforms,
                time_offsets, goal_xy):
        parts = self.forward_parts(images, history_images, lidar2img,
                                   history_transforms, time_offsets, goal_xy)
        b = images.shape[0]
        nx, ny = self.config.grid_size
        raster = parts["scene_features"].transpose(1, 2).reshape(b, -1, nx, ny)
        samples, visible, offsets = self._temporal_samples(
            lidar2img, history_transforms, time_offsets, images.shape[-2:])
        grid_xy = self.scene_encoder.cell_xy.float().reshape(nx, ny, 2).permute(2, 0, 1)[None]
        grid_xy = grid_xy / grid_xy.new_tensor([80.0, 64.0])[None, :, None, None]
        delta = self.scene_adapter(samples, visible, offsets, grid_xy)
        # Only where the current front and at least one past front both landed.
        gate = (visible[:, 0] & visible[:, 1:].any(1)).float()
        raster = raster + delta * gate
        parts["scene_features"] = raster.flatten(2).transpose(1, 2)
        parts["future_logits"] = self.future_head(raster).float()
        parts["scene_delta_gate"] = gate
        plan = self.plan_from_features(parts["scene_features"], parts["motion_features"],
                                       parts["state_hat"], parts["history_hat"])
        return {**parts, "plan_abs": plan}


def future_footprint_loss(logits, target, valid, pos_weight):
    """Masked BCE averaged inside a row/horizon, then over rows, then horizons.

    A frame with many valid cells must not outweigh a frame with few, and a
    row or horizon with no valid cell drops out instead of contributing zero.
    """
    if logits.shape != target.shape or logits.shape != valid.shape:
        raise ValueError("future footprint shapes must match")
    weight = torch.where(target > .5, logits.new_tensor(pos_weight)[None, :, None, None],
                         logits.new_tensor([1.0]))
    element = F.binary_cross_entropy_with_logits(logits.float(), target.float(),
                                                 reduction="none") * weight
    mask = valid.float()
    counts = mask.flatten(2).sum(-1)
    per_horizon = (element * mask).flatten(2).sum(-1) / counts.clamp_min(1.0)
    present = counts > 0
    totals = (per_horizon * present).sum(0) / present.sum(0).clamp_min(1.0)
    used = present.any(0)
    if not bool(used.any()):
        return logits.sum() * 0.0
    return (totals * used).sum() / used.sum()
