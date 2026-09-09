"""Opt-in ordered temporal-motion residual for the early-precision screen.

The production :mod:`motion_encoder` is deliberately unchanged.  This module
copies its forward exactly and inserts one zero-initialized, image-derived
residual between temporal weighting and the existing token refiner.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .motion_encoder import MotionEncoder, local_correlation


N_HISTORY = 4
CHANNELS = 128
BOTTLENECK = 64


class OrderedTemporalResidual(nn.Module):
    """Mix the four ordered, already time-encoded tokens at each spatial site."""

    def __init__(self, channels: int = CHANNELS, n_history: int = N_HISTORY,
                 bottleneck: int = BOTTLENECK):
        super().__init__()
        if (channels, n_history, bottleneck) != (CHANNELS, N_HISTORY, BOTTLENECK):
            raise ValueError("early-precision ordered mixer is fixed at 4x128 -> 64 -> 128")
        width = channels * n_history
        self.channels = channels
        self.n_history = n_history
        self.norm = nn.LayerNorm(width)
        self.input_projection = nn.Linear(width, bottleneck)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(bottleneck, channels)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, pair_tokens: Tensor) -> Tensor:
        if (pair_tokens.ndim != 4 or pair_tokens.shape[1] != self.n_history
                or pair_tokens.shape[-1] != self.channels):
            raise ValueError("ordered temporal tokens must be [B,4,N,128]")
        # Order is the dataset/config order [.1, .2, .5, 1.0] seconds.  The
        # token values already include MotionEncoder's [dt, log(dt)] embedding.
        ordered = pair_tokens.permute(0, 2, 1, 3).flatten(2)
        with torch.autocast(device_type=pair_tokens.device.type, enabled=False):
            residual = self.output_projection(
                self.activation(self.input_projection(self.norm(ordered.float()))))
        return residual.to(pair_tokens.dtype)


class OrderedTemporalMotionEncoder(MotionEncoder):
    """MotionEncoder with one exact-zero ordered residual before token_refine."""

    def __init__(self, config):
        super().__init__(config)
        if (config.n_history != N_HISTORY or config.channels != CHANNELS
                or tuple(config.motion_grid) != (12, 16)
                or tuple(config.nominal_history_seconds) != (.1, .2, .5, 1.)):
            raise ValueError(
                "ordered temporal screen requires nominal [.1,.2,.5,1], grid12x16, channels128")
        # The extra branch must not perturb the parent/control RNG stream used
        # later by augmentation or stochastic modules.
        rng = torch.get_rng_state()
        try:
            self.ordered_temporal_residual = OrderedTemporalResidual()
        finally:
            torch.set_rng_state(rng)

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
            pair = torch.cat([corr.to(cur.dtype), cur, past], 1)
            features.append(F.interpolate(pair, ref_hw, mode="bilinear", align_corners=False))
        pair_maps = self.correlation_fuse(torch.cat(features, 1))
        pair_maps = F.adaptive_avg_pool2d(pair_maps, self.config.motion_grid)
        pair_tokens = pair_maps.flatten(2).transpose(1, 2).reshape(
            b, t, -1, self.config.channels)
        dt = time_offsets.clamp_min(1e-3)
        time_feature = self.time_embed(torch.stack([dt, dt.log()], -1))
        pair_tokens = (pair_tokens + self.position(self.positions)[None, None]
                       + time_feature[:, :, None])
        with torch.autocast(device_type=pair_tokens.device.type, enabled=False):
            weights = self.time_attention(pair_tokens.float()).softmax(1)
        weighted_motion = (pair_tokens * weights.to(pair_tokens.dtype)).sum(1)
        motion = weighted_motion + self.ordered_temporal_residual(pair_tokens)
        motion = motion + self.token_refine(motion)
        history_map = pair_tokens.reshape(
            b * t, *self.config.motion_grid, self.config.channels)
        history_vec = F.adaptive_avg_pool2d(
            history_map.permute(0, 3, 1, 2), (3, 4)).flatten(1)
        motion_map = motion.reshape(
            b, *self.config.motion_grid, self.config.channels).permute(0, 3, 1, 2)
        state_vec = F.adaptive_avg_pool2d(motion_map, (3, 4)).flatten(1)
        with torch.autocast(device_type=motion.device.type, enabled=False):
            history_hat = self.history_head(history_vec.float()).reshape(b, t, 4) * self.history_scale
            state_hat = self.state_head(state_vec.float()) * self.state_scale
            history_logvar = self.history_uncertainty_head(history_vec.float()).reshape(b, t, 4).clamp(-8, 8)
            state_logvar = self.state_uncertainty_head(state_vec.float()).clamp(-8, 8)
        return {"motion_features": motion, "history_hat": history_hat, "state_hat": state_hat,
                "history_logvar": history_logvar, "state_logvar": state_logvar}


def ordered_residual_state(model: nn.Module) -> dict[str, Tensor]:
    prefix = "motion_encoder.ordered_temporal_residual."
    state = {name: value for name, value in model.state_dict().items()
             if name.startswith(prefix)}
    if not state:
        raise ValueError("ordered temporal residual state is absent")
    return state


def install_ordered_temporal_encoder(model: nn.Module) -> nn.Module:
    """Replace only the encoder implementation while preserving parent tensors/RNG."""
    original = getattr(model, "motion_encoder", None)
    if not isinstance(original, MotionEncoder) or isinstance(original, OrderedTemporalMotionEncoder):
        raise ValueError("expected one unmodified MotionEncoder")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("install ordered temporal encoder on CPU before moving the model")
    rng = torch.get_rng_state()
    try:
        replacement = OrderedTemporalMotionEncoder(model.config)
    finally:
        torch.set_rng_state(rng)
    incompatible = replacement.load_state_dict(original.state_dict(), strict=False)
    expected = [name.removeprefix("motion_encoder.") for name in ordered_residual_state(
        nn.ModuleDict({"motion_encoder": replacement})).keys()]
    if sorted(incompatible.missing_keys) != sorted(expected) or incompatible.unexpected_keys:
        raise ValueError("base MotionEncoder did not map exactly into ordered encoder")
    model.motion_encoder = replacement
    return model
