#!/usr/bin/env python3
"""MR-ADJ: adjacent-past comparisons added to the existing star of t0 comparisons.

The MR motion branch compares every past frame to t0 and to nothing else. This
adds the three comparisons BETWEEN neighbouring past frames, over the same five
images. No new sensor observation enters; only the number of relations computed
from images that were already read goes up.

    existing (star)  (0,-0.1) (0,-0.2) (0,-0.5) (0,-1.0)
    added (adjacent) (-0.1,-0.2) (-0.2,-0.5) (-0.5,-1.0)

In every pair the first entry is the REFERENCE and the second is the older
SOURCE, matching local_correlation(current, history, r): its first argument is
the frame whose pixel grid the cost volume is indexed by.

Two arms differ ONLY in the edge mask:

    MR-ADJ0   the same new module, but the three adjacent edges are masked out
    MR-ADJ1   the three adjacent edges are readable

so the extra module's capacity is present in both and the contrast isolates the
information in the new edges rather than the module that reads them.

The star path is preserved bit for bit: h_star and the history readout are
computed exactly as MR computes them, and the new module contributes through a
zero-initialised output projection, so at step zero the model IS MR.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.motiondrive_v2 import motion_encoder as motion_module

# Indices into the five-frame stack [t0, -0.1, -0.2, -0.5, -1.0].
STAR_EDGES = ((0, 1), (0, 2), (0, 3), (0, 4))
ADJACENT_EDGES = ((1, 2), (2, 3), (3, 4))
ALL_EDGES = STAR_EDGES + ADJACENT_EDGES


def edge_ages(time_offsets):
    """Age in seconds of each of the five frames; frame 0 is the current one."""
    b = time_offsets.shape[0]
    zero = time_offsets.new_zeros(b, 1)
    return torch.cat([zero, time_offsets], 1)


class AdjacentEdgeAttention(nn.Module):
    """Reads a pair-indexed memory and returns a residual for the motion feature.

    Memory entries keep what they are: which pair produced them, the age of both
    endpoints, and the spatial position IN THEIR OWN REFERENCE FRAME. The query
    carries its own position separately. Nothing here asserts that position n of
    the -0.1 s frame is the same physical point as position n of t0; the model
    is given the coordinates and has to learn any correspondence itself.
    """

    def __init__(self, channels, n_edges, n_new, positions):
        super().__init__()
        self.channels = channels
        self.n_edges = n_edges
        # [age_ref, age_src, dt, log(dt / 1 s)] -- dt alone is ambiguous once
        # adjacent edges exist: (0,-0.1) and (-0.1,-0.2) share dt = 0.1, and
        # (0,-0.5) and (-0.5,-1.0) share dt = 0.5.
        self.pair_time = nn.Sequential(nn.Linear(4, channels), nn.GELU(),
                                       nn.Linear(channels, channels))
        self.edge_kind = nn.Embedding(2, channels)          # 0 star, 1 adjacent
        self.edge_id = nn.Embedding(n_edges, channels)
        self.reference_position = nn.Linear(2, channels, bias=False)
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        # Zero output projection: at step zero the residual is exactly zero and
        # the model reproduces MR. The gradient of everything BEHIND this layer
        # is therefore zero on the first backward by construction -- that is the
        # intent, not a defect, so the check is whether the path opens over the
        # first updates, not whether every new tensor moves immediately.
        self.to_out = nn.Linear(channels, channels)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)
        self.register_buffer("positions", positions)
        kind = torch.zeros(n_edges, dtype=torch.long)
        kind[n_edges - n_new:] = 1
        self.register_buffer("edge_kind_index", kind)
        self.register_buffer("edge_index", torch.arange(n_edges))

    def forward(self, h_star, pair_tokens, ages, use_adjacent):
        """h_star [B,N,C]; pair_tokens [B,E,N,C]; ages [B,5]."""
        b, e, n, c = pair_tokens.shape
        device = pair_tokens.device
        ref = torch.stack([ages[:, i] for i, _ in ALL_EDGES], 1)     # [B,E]
        src = torch.stack([ages[:, j] for _, j in ALL_EDGES], 1)
        dt = (src - ref).clamp_min(1e-3)
        feature = torch.stack([ref, src, dt, dt.log()], -1)          # [B,E,4]
        memory = pair_tokens + self.pair_time(feature)[:, :, None]
        memory = memory + self.edge_kind(self.edge_kind_index)[None, :, None]
        memory = memory + self.edge_id(self.edge_index)[None, :, None]
        memory = memory + self.reference_position(self.positions)[None, None]
        memory = memory.reshape(b, e * n, c)

        q = self.to_q(self.query_norm(h_star))
        k = self.to_k(self.memory_norm(memory))
        v = self.to_v(self.memory_norm(memory))
        mask = None
        if not use_adjacent:
            allowed = (self.edge_kind_index == 0)
            # Star edges are always readable, so no query ever sees a fully
            # masked memory and the softmax cannot produce NaN.
            mask = allowed[:, None].expand(e, n).reshape(1, 1, e * n)
        with torch.autocast(device_type=device.type, enabled=False):
            scores = (q.float() @ k.float().transpose(1, 2)) / (c ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            weights = scores.softmax(-1)
        attended = weights.to(v.dtype) @ v
        return self.to_out(attended)


def install(model, use_adjacent: bool):
    """Replace MotionEncoder.forward with the seven-edge version."""
    encoder = model.motion_encoder
    config = encoder.config
    if not hasattr(encoder, "adjacent_attention"):
        encoder.adjacent_attention = AdjacentEdgeAttention(
            config.channels, len(ALL_EDGES), len(ADJACENT_EDGES),
            encoder.positions.clone())

    def forward(self, current_front_levels, history_front_levels, time_offsets):
        b, t = time_offsets.shape
        if t != len(history_front_levels[0][0]):
            pass
        ref_hw = history_front_levels[0].shape[-2:]
        # Rebuild the five-frame stack the MR forward_parts split apart.
        stacks = [torch.cat([cur[:, None], hist], 1)
                  for cur, hist in zip(current_front_levels, history_front_levels)]
        n_edges = len(ALL_EDGES)
        features = []
        for i, stack in enumerate(stacks):
            _, frames, c, h, w = stack.shape
            projected = self.projections[i](stack.reshape(b * frames, c, h, w))
            projected = projected.reshape(b, frames, -1, h, w)
            ref = torch.stack([projected[:, a] for a, _ in ALL_EDGES], 1)
            src = torch.stack([projected[:, s] for _, s in ALL_EDGES], 1)
            ref = ref.reshape(b * n_edges, -1, h, w)
            src = src.reshape(b * n_edges, -1, h, w)
            # The correlation runs at THIS level's own resolution and the result is
            # resampled afterwards, exactly as MR does it. Resampling the
            # descriptors first would make the coarse level search the same
            # original-image distance as the fine one and silently halve its reach.
            with torch.autocast(device_type=ref.device.type, enabled=False):
                corr = motion_module.local_correlation(ref, src, self.config.correlation_radius)
            pair = torch.cat([corr.to(ref.dtype), ref, src], 1)
            features.append(F.interpolate(pair, ref_hw, mode="bilinear", align_corners=False))
        pair_maps = self.correlation_fuse(torch.cat(features, 1))
        pair_maps = F.adaptive_avg_pool2d(pair_maps, self.config.motion_grid)
        all_tokens = pair_maps.flatten(2).transpose(1, 2).reshape(
            b, n_edges, -1, self.config.channels)

        # --- the original star path, unchanged ---
        star_tokens = all_tokens[:, :len(STAR_EDGES)]
        dt = time_offsets.clamp_min(1e-3)
        time_feature = self.time_embed(torch.stack([dt, dt.log()], -1))
        star_tokens = (star_tokens + self.position(self.positions)[None, None]
                       + time_feature[:, :, None])
        with torch.autocast(device_type=star_tokens.device.type, enabled=False):
            weights = self.time_attention(star_tokens.float()).softmax(1)
        motion = (star_tokens * weights.to(star_tokens.dtype)).sum(1)
        motion = motion + self.token_refine(motion)

        # --- the new residual, zero at step zero ---
        motion = motion + self.adjacent_attention(
            motion, all_tokens, edge_ages(time_offsets), use_adjacent)

        # history keeps its four existing targets and reads the star edges only
        history_map = star_tokens.reshape(b * len(STAR_EDGES), *self.config.motion_grid,
                                          self.config.channels)
        history_vec = F.adaptive_avg_pool2d(history_map.permute(0, 3, 1, 2), (3, 4)).flatten(1)
        motion_map = motion.reshape(b, *self.config.motion_grid,
                                    self.config.channels).permute(0, 3, 1, 2)
        state_vec = F.adaptive_avg_pool2d(motion_map, (3, 4)).flatten(1)
        with torch.autocast(device_type=motion.device.type, enabled=False):
            history_hat = self.history_head(history_vec.float()).reshape(b, t, 4) * self.history_scale
            state_hat = self.state_head(state_vec.float()) * self.state_scale
            history_logvar = self.history_uncertainty_head(
                history_vec.float()).reshape(b, t, 4).clamp(-8, 8)
            state_logvar = self.state_uncertainty_head(state_vec.float()).clamp(-8, 8)
        return {"motion_features": motion, "history_hat": history_hat, "state_hat": state_hat,
                "history_logvar": history_logvar, "state_logvar": state_logvar}

    type(encoder).forward = forward
    return {"edges": [list(e) for e in ALL_EDGES],
            "star_edges": [list(e) for e in STAR_EDGES],
            "adjacent_edges": [list(e) for e in ADJACENT_EDGES],
            "use_adjacent": use_adjacent,
            "pair_convention": "(reference, source); source is the older frame",
            "new_parameters": int(sum(p.numel() for p in encoder.adjacent_attention.parameters())),
            "output_projection_init": "zeros, so the step-zero model reproduces MR exactly"}
