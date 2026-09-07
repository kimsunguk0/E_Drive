"""Unaligned, image-derived temporal correspondence and physical state heads.

This module deliberately has NO pose, goal or provided-status argument. Pair
correlation keeps displacement bins and spatial position, unlike averages of
features after ego-pose warping. Calibration is fixed by the dataset; changing
camera/FOV requires retraining or supplying a reviewed calibration encoding.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def local_correlation(current, history, radius):
    """Normalized current-to-history cost volume, channels ordered dy then dx."""
    b, c, h, w = current.shape
    width = 2 * radius + 1
    current = F.normalize(current.float(), dim=1, eps=1e-6)
    history = F.normalize(history.float(), dim=1, eps=1e-6)
    patches = F.unfold(history, kernel_size=width, padding=radius)
    patches = patches.reshape(b, c, width * width, h, w)
    return (current[:, :, None] * patches).sum(1)


class MotionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c, cm = config.channels, config.correlation_channels
        bins = (2 * config.correlation_radius + 1) ** 2
        self.projections = nn.ModuleList([nn.Conv2d(c, cm, 1, bias=False) for _ in range(2)])
        self.correlation_fuse = nn.Sequential(
            nn.Conv2d(2 * (bins + 2 * cm), c, 3, padding=1),
            nn.GroupNorm(1, c), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(1, c), nn.GELU())
        # Explicit XY locations and true temporal intervals survive aggregation.
        self.position = nn.Linear(2, c, bias=False)
        self.time_embed = nn.Sequential(nn.Linear(2, c), nn.GELU(), nn.Linear(c, c))
        self.time_attention = nn.Linear(c, 1)
        self.token_refine = nn.Sequential(nn.Linear(c, c * 2), nn.GELU(), nn.Linear(c * 2, c))
        self.history_head = nn.Sequential(nn.Linear(c * 12, c), nn.GELU(), nn.Linear(c, 4))
        self.history_uncertainty_head = nn.Sequential(nn.Linear(c * 12, c), nn.GELU(), nn.Linear(c, 4))
        self.state_head = nn.Sequential(nn.Linear(c * 12, c), nn.GELU(), nn.Linear(c, 6))
        self.state_uncertainty_head = nn.Sequential(nn.Linear(c * 12, c), nn.GELU(), nn.Linear(c, 5))
        self.register_buffer("history_scale", torch.tensor([10.0, 5.0, 1.0, 1.0]))
        self.register_buffer("state_scale", torch.tensor([10.0, 5.0, 3.0, 3.0, 0.5, 1.0]))
        # Heads emit physical units; these fixed output scales are not GT inputs.
        with torch.no_grad():
            self.history_head[-1].bias[3] = 1.0
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, config.motion_grid[0]),
                                torch.linspace(-1, 1, config.motion_grid[1]), indexing="ij")
        self.register_buffer("positions", torch.stack([gx, gy], -1).reshape(-1, 2))

    def forward(self, current_front_levels, history_front_levels, time_offsets):
        b, t = time_offsets.shape
        features = []
        ref_hw = history_front_levels[0].shape[-2:]
        for i, (current, history) in enumerate(zip(current_front_levels, history_front_levels)):
            _, _, c, h, w = history.shape
            cur = self.projections[i](current)
            cur = F.interpolate(cur, (h, w), mode="bilinear", align_corners=False)
            cur = cur[:, None].expand(-1, t, -1, -1, -1).reshape(b * t, -1, h, w)
            past = self.projections[i](history.reshape(b * t, c, h, w))
            with torch.autocast(device_type=cur.device.type, enabled=False):
                corr = local_correlation(cur, past, self.config.correlation_radius)
            # Cost bins remain separate. Feature context disambiguates dynamic
            # objects, repetitive markings, occlusions and motion confidence.
            pair = torch.cat([corr.to(cur.dtype), cur, past], 1)
            features.append(F.interpolate(pair, ref_hw, mode="bilinear", align_corners=False))
        pair_maps = self.correlation_fuse(torch.cat(features, 1))
        pair_maps = F.adaptive_avg_pool2d(pair_maps, self.config.motion_grid)
        pair_tokens = pair_maps.flatten(2).transpose(1, 2).reshape(b, t, -1, self.config.channels)
        dt = time_offsets.clamp_min(1e-3)
        time_feature = self.time_embed(torch.stack([dt, dt.log()], -1))
        pair_tokens = pair_tokens + self.position(self.positions)[None, None] + time_feature[:, :, None]
        with torch.autocast(device_type=pair_tokens.device.type, enabled=False):
            weights = self.time_attention(pair_tokens.float()).softmax(1)
        motion = (pair_tokens * weights.to(pair_tokens.dtype)).sum(1)
        motion = motion + self.token_refine(motion)
        # Preserve coarse spatial bins in state/history heads. A global mean of
        # already aligned differences is intentionally NOT used here.
        history_map = pair_tokens.reshape(b * t, *self.config.motion_grid, self.config.channels)
        history_vec = F.adaptive_avg_pool2d(history_map.permute(0, 3, 1, 2), (3, 4)).flatten(1)
        motion_map = motion.reshape(b, *self.config.motion_grid, self.config.channels).permute(0, 3, 1, 2)
        state_vec = F.adaptive_avg_pool2d(motion_map, (3, 4)).flatten(1)
        # Compute projection FROM float32 inputs. Casting a rounded bf16 output
        # afterwards would not recover sub-metre displacement precision.
        with torch.autocast(device_type=motion.device.type, enabled=False):
            history_hat = self.history_head(history_vec.float()).reshape(b, t, 4) * self.history_scale
            state_hat = self.state_head(state_vec.float()) * self.state_scale
            history_logvar = self.history_uncertainty_head(history_vec.float()).reshape(b, t, 4).clamp(-8, 8)
            state_logvar = self.state_uncertainty_head(state_vec.float()).clamp(-8, 8)
        return {"motion_features": motion, "history_hat": history_hat, "state_hat": state_hat,
                "history_logvar": history_logvar, "state_logvar": state_logvar}
