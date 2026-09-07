"""Goal-scored cross-cell residual over continuous image-derived scene values.

Goal coordinates enter only the squared-distance attention score.  They are
never embedded in values or appended to the planner input.  The existing
per-cell goal conditioning in :mod:`scene_encoder` remains a common input to
both P7 arms.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .scene_encoder import grid_centers, masked_softmax


class GoalScoredImageValueResidual(nn.Module):
    """Fixed 3072x192 score-only goal routing with a zero-init residual."""

    COSINE_SCALE = 8.0
    SIGMA_M = (10.0, 32.0 / 3.0)
    POOL_SIZE = 4

    def __init__(self, *, channels: int = 128, attention_dim: int = 32,
                 grid_size: tuple[int, int] = (64, 48),
                 x_range: tuple[float, float] = (-10., 70.),
                 y_range: tuple[float, float] = (-32., 32.)):
        super().__init__()
        if (channels != 128 or attention_dim != 32 or tuple(grid_size) != (64, 48)):
            raise ValueError("P7 is fixed at 3072 cells, 192 pooled sources, and 128->32")
        self.channels = channels
        self.attention_dim = attention_dim
        self.grid_size = tuple(grid_size)
        self.norm = nn.LayerNorm(channels)
        self.query_image = nn.Linear(channels, attention_dim, bias=False)
        self.key_image = nn.Linear(channels, attention_dim, bias=False)
        self.value_image = nn.Linear(channels, attention_dim, bias=False)
        self.position = nn.Sequential(nn.Linear(4, attention_dim), nn.GELU(),
                                      nn.Linear(attention_dim, attention_dim, bias=False))
        self.output = nn.Linear(attention_dim, channels, bias=False)
        nn.init.zeros_(self.output.weight)

        xy = grid_centers(self.grid_size, x_range, y_range).float()
        source_xy = F.avg_pool2d(xy.permute(2, 0, 1)[None], self.POOL_SIZE,
                                 stride=self.POOL_SIZE).squeeze(0).permute(1, 2, 0)
        self.register_buffer("destination_xy", xy.reshape(-1, 2))
        self.register_buffer("source_xy", source_xy.reshape(-1, 2))
        self.register_buffer("position_scale", torch.tensor(
            [x_range[1] - x_range[0], y_range[1] - y_range[0]], dtype=torch.float32))
        self.register_buffer("sigma_m", torch.tensor(self.SIGMA_M, dtype=torch.float32))
        if self.destination_xy.shape != (3072, 2) or self.source_xy.shape != (192, 2):
            raise AssertionError("Fixed grid/pooling geometry did not produce 3072x192")

    def _position_features(self, xy: Tensor) -> Tensor:
        normalized = xy.float() / self.position_scale.float()
        return torch.cat([normalized, normalized.square()], -1)

    def _pool_image_values(self, scene: Tensor, source_valid: Tensor) -> tuple[Tensor, Tensor]:
        b = len(scene)
        raster = scene.float().transpose(1, 2).reshape(b, self.channels, *self.grid_size)
        mask = source_valid.float().reshape(b, 1, *self.grid_size)
        mean_mask = F.avg_pool2d(mask, self.POOL_SIZE, stride=self.POOL_SIZE)
        pooled = F.avg_pool2d(raster * mask, self.POOL_SIZE, stride=self.POOL_SIZE)
        pooled = pooled / mean_mask.clamp_min(1. / (self.POOL_SIZE ** 2))
        pooled = torch.where(mean_mask > 0, pooled, torch.zeros_like(pooled))
        return pooled.flatten(2).transpose(1, 2), mean_mask.flatten(1) > 0

    def forward(self, scene_features: Tensor, goal_xy_m: Tensor,
                source_valid: Tensor | None = None) -> Tensor:
        if (scene_features.ndim != 3 or scene_features.shape[1:] != (3072, self.channels)
                or not scene_features.is_floating_point()):
            raise ValueError("scene_features must be floating [B,3072,128]")
        b = len(scene_features)
        if goal_xy_m.shape != (b, 2) or not goal_xy_m.is_floating_point():
            raise ValueError("goal_xy_m must be floating metric coordinates [B,2]")
        if not bool(torch.isfinite(scene_features).all()) or not bool(torch.isfinite(goal_xy_m).all()):
            raise ValueError("Inputs must be finite")
        if source_valid is None:
            source_valid = torch.ones((b, 3072), dtype=torch.bool, device=scene_features.device)
        if source_valid.shape != (b, 3072) or source_valid.dtype != torch.bool:
            raise ValueError("source_valid must be bool [B,3072]")
        if source_valid.device != scene_features.device or goal_xy_m.device != scene_features.device:
            raise ValueError("Inputs must share one device")

        # The surrounding model uses BF16 autocast for its encoders.  P7's
        # complete routing branch is deliberately FP32: disabling autocast is
        # required because merely casting Linear inputs to float does not stop
        # autocast from producing BF16 Linear/einsum outputs.
        with torch.autocast(device_type=scene_features.device.type, enabled=False):
            normalized = self.norm(scene_features.float())
            sources, pooled_valid = self._pool_image_values(scene_features, source_valid)
            source_normalized = self.norm(sources)
            # Q/K retain affine normalization. V cannot acquire a learned constant
            # from LayerNorm bias when its continuous scene value is zero.
            source_value_normalized = F.layer_norm(
                sources, (self.channels,), self.norm.weight, None, self.norm.eps)
            destination_position = self.position(self._position_features(self.destination_xy))
            source_position = self.position(self._position_features(self.source_xy))
            query = F.normalize(self.query_image(normalized) + destination_position[None], dim=-1)
            key = F.normalize(self.key_image(source_normalized) + source_position[None], dim=-1)
            value = self.value_image(source_value_normalized)

            content_score = torch.einsum("bqd,bkd->bqk", query, key) * self.COSINE_SCALE
            # Both operands are raw metres. Position-feature normalization above is
            # deliberately not reused by this physical distance term.
            distance = ((self.source_xy.float()[None] - goal_xy_m.float()[:, None])
                        / self.sigma_m.float()).square().sum(-1)
            score = content_score - .5 * distance[:, None]
            attention = masked_softmax(score, pooled_valid[:, None].expand_as(score), dim=-1)
            context = torch.einsum("bqk,bkd->bqd", attention, value)
            residual = self.output(context).to(scene_features.dtype)
        return scene_features + residual

    def parameter_accounting(self) -> dict[str, int]:
        expected = {
            "layer_norm": 2 * self.channels,
            "qkv_image": 3 * self.channels * self.attention_dim,
            "goal_free_position": 4 * self.attention_dim + self.attention_dim
                                  + self.attention_dim * self.attention_dim,
            "bias_free_output": self.attention_dim * self.channels,
        }
        expected["total"] = sum(expected.values())
        expected["actual"] = sum(parameter.numel() for parameter in self.parameters())
        return expected

    def mac_accounting_per_sample(self) -> dict[str, int]:
        q, k, c, d = 3072, 192, self.channels, self.attention_dim
        result = {
            "query_image": q * c * d,
            "key_value_image": 2 * k * c * d,
            "goal_free_position": (q + k) * (4 * d + d * d),
            "cosine_logits": q * k * d,
            "attention_value": q * k * d,
            "output_projection": q * d * c,
        }
        result["total"] = sum(result.values())
        return result
