"""Spatial, image-valued scene attention: goal changes evidence weights only.

There are no trajectory queries in this module. The fixed XY raster is shared
by real occupancy/lane heads and the planner. History poses occur only in
geometric projection; they are never embedded as feature values.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def grid_centers(grid_size, x_range, y_range):
    nx, ny = grid_size
    x = x_range[0] + (torch.arange(nx) + 0.5) * (x_range[1] - x_range[0]) / nx
    y = y_range[0] + (torch.arange(ny) + 0.5) * (y_range[1] - y_range[0]) / ny
    return torch.stack(torch.meshgrid(x, y, indexing="ij"), -1)


def pixel_to_normalized_grid(uv: Tensor, image_hw: tuple[int, int]) -> Tensor:
    """Pixel centers to align_corners=False coordinates (not corner coordinates)."""
    h, w = image_hw
    return (uv + 0.5) * uv.new_tensor([2.0 / w, 2.0 / h]) - 1.0


def project_scene_points(points: Tensor, matrices: Tensor, image_hw):
    """Full homogeneous projection, including z/roll/pitch in every transform."""
    with torch.autocast(device_type=matrices.device.type, enabled=False):
        projected = torch.einsum("bvij,qhj->bvqhi", matrices.float(), points.float())
        depth = projected[..., 2]
        uv = projected[..., :2] / depth.clamp_min(1e-5).unsqueeze(-1)
        h, w = image_hw
        valid = ((depth > 1e-5) & (uv[..., 0] >= 0) & (uv[..., 0] < w)
                 & (uv[..., 1] >= 0) & (uv[..., 1] < h)
                 & torch.isfinite(uv).all(-1))
        grid = pixel_to_normalized_grid(uv, image_hw)
        # Out-of-frustum values need not carry extreme/NaN grids through CUDA.
        grid = torch.where(valid[..., None], grid, torch.full_like(grid, 2.0))
    return grid, valid


def masked_softmax(scores: Tensor, valid: Tensor, dim: int = -1) -> Tensor:
    """All-invisible rows return exact zero, including in backward."""
    scores = scores.float().masked_fill(~valid, -1e9)
    result = scores.softmax(dim=dim) * valid.float()
    return result / result.sum(dim=dim, keepdim=True).clamp_min(1e-12)


class SpatialMix(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(1, channels), nn.GELU(),
            nn.Conv2d(channels, channels * 2, 1), nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1))

    def forward(self, x):
        return x + self.mix(x)


class SharedSceneEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c, d = config.channels, config.scene_attention_channels
        self.register_buffer("cell_xy", grid_centers(config.grid_size, config.x_range,
                                                    config.y_range).reshape(-1, 2))
        xy = self.cell_xy[:, None, :].expand(-1, len(config.heights), -1)
        z = torch.tensor(config.heights)[None, :, None].expand(len(xy), -1, -1)
        self.register_buffer("points", torch.cat([xy, z, torch.ones_like(z)], -1))
        self.key_proj = nn.ModuleList([nn.Conv2d(c, d, 1, bias=False) for _ in range(2)])
        self.value_proj = nn.ModuleList([nn.Conv2d(c, c, 1, bias=False) for _ in range(2)])
        self.query_image = nn.Linear(c, d, bias=False)
        self.query_context = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        # Metadata appears only in attention keys, never as a goal-bearing value.
        self.camera_keys = nn.Parameter(torch.randn(6, d) * 0.02)
        self.height_keys = nn.Parameter(torch.randn(len(config.heights), d) * 0.02)
        self.scale_keys = nn.Parameter(torch.randn(2, d) * 0.02)
        self.time_key = nn.Sequential(nn.Linear(1, d), nn.Tanh(), nn.Linear(d, d))
        self.global_proj = nn.Linear(c, c, bias=False)
        if config.cross_cell_goal_mode == "disabled":
            self.cross_cell_goal_residual = None
        else:
            # Local import avoids a module cycle: the residual reuses the
            # geometry and all-invalid softmax helpers defined above.
            from .cross_cell_goal_residual import GoalScoredImageValueResidual
            self.cross_cell_goal_residual = GoalScoredImageValueResidual(
                channels=c, attention_dim=32, grid_size=config.grid_size,
                x_range=config.x_range, y_range=config.y_range)
        self.refine = nn.Sequential(SpatialMix(c), SpatialMix(c))
        self.occ_head = nn.Sequential(nn.Conv2d(c, c // 2, 3, padding=1), nn.GELU(),
                                      nn.Conv2d(c // 2, 1, 1))
        self.lane_head = nn.Sequential(nn.Conv2d(c, c // 2, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(c // 2, 1, 1))

    def _aggregate_evidence(self, query, key, value, valid):
        # Optional image-value aggregation; the query is never a scene value.
        multihead = getattr(self, "evidence_attention", None)
        if multihead is not None:
            return multihead(query, key, value, valid)
        with torch.autocast(device_type=query.device.type, enabled=False):
            score = (query.float()[:, :, None] * key.float()).sum(-1) / math.sqrt(key.shape[-1])
            attention = masked_softmax(score, valid)
            return (attention[..., None] * value.float()).sum(2)

    def _sample(self, feature, grid):
        b, v, c, h, w = feature.shape
        q, nh = grid.shape[2:4]
        feat = feature.flatten(0, 1)
        # Keep FP32 geometry until grid_sample; CUDA requires a matching grid dtype.
        sampled = F.grid_sample(feat, grid.reshape(b * v, q, nh, 2).to(feat.dtype),
                                mode="bilinear", padding_mode="zeros", align_corners=False)
        return sampled.reshape(b, v, -1, q, nh).permute(0, 3, 1, 4, 2)

    def _apply_cross_cell_goal_residual(self, scene: Tensor, visible: Tensor,
                                        goal_xy_m: Tensor) -> Tensor:
        if self.cross_cell_goal_residual is None:
            return scene
        branch_goal_m = (goal_xy_m if self.config.cross_cell_goal_mode == "real"
                         else torch.zeros_like(goal_xy_m))
        return self.cross_cell_goal_residual(scene, branch_goal_m, visible)

    def forward(self, current_levels, history_levels, current_global, lidar2img,
                history_transforms, time_offsets, goal_xy, image_hw):
        b = lidar2img.shape[0]
        c = self.config.channels
        # Each historical front camera has its own exact current->past SE(3).
        with torch.autocast(device_type=lidar2img.device.type, enabled=False):
            historical_matrices = (lidar2img[:, :1].float()
                                   @ history_transforms.float())
            matrices = torch.cat([lidar2img.float(), historical_matrices], 1)
        camera_ids = torch.tensor([0, 1, 2, 3, 4, 5] + [0] * self.config.n_history,
                                  device=lidar2img.device)
        view_times = torch.cat([time_offsets.new_zeros(b, 6), -time_offsets], 1)
        metadata = (self.camera_keys[camera_ids][None, :, None, :]
                    + self.height_keys[None, None]
                    + self.time_key(view_times[..., None])[:, :, None])
        used_goal = goal_xy if self.config.goal_on else torch.zeros_like(goal_xy)
        scale = self.cell_xy.new_tensor([80.0, 64.0])
        context = torch.cat([
            self.cell_xy[None].expand(b, -1, -1) / scale,
            used_goal[:, None].expand(-1, len(self.cell_xy), -1) / scale,
            (self.cell_xy[None] - used_goal[:, None]) / scale], -1)
        q_context = self.query_context(context)
        # Project feature maps ONCE, not once per spatial attention chunk.
        projected_levels = []
        for i, (cur, past) in enumerate(zip(current_levels, history_levels)):
            def projected(feat, layer):
                out = layer(feat.flatten(0, 1))
                return out.reshape(feat.shape[0], feat.shape[1], *out.shape[1:])
            projected_levels.append((projected(cur, self.value_proj[i]),
                                     projected(past, self.value_proj[i]),
                                     projected(cur, self.key_proj[i]),
                                     projected(past, self.key_proj[i])))
        chunks, visible_chunks = [], []
        for start in range(0, len(self.points), self.config.scene_chunk_size):
            stop = start + self.config.scene_chunk_size
            all_grid, all_valid = project_scene_points(self.points[start:stop], matrices, image_hw)
            values, keys, masks = [], [], []
            for level_index, (cur_value, past_value, cur_key, past_key) in enumerate(projected_levels):
                # Historical features use a half-resolution image of the SAME FOV;
                # normalized grid coordinates are shared, not raw pixel offsets.
                level_values = torch.cat([
                    self._sample(cur_value, all_grid[:, :6]),
                    self._sample(past_value, all_grid[:, 6:])], 2)
                level_keys = torch.cat([
                    self._sample(cur_key, all_grid[:, :6]),
                    self._sample(past_key, all_grid[:, 6:])], 2)
                level_keys = level_keys + metadata[:, None] + self.scale_keys[level_index]
                values.append(level_values.flatten(2, 3))
                keys.append(level_keys.flatten(2, 3))
                masks.append(all_valid.permute(0, 2, 1, 3).flatten(2, 3))
            value, key, valid = torch.cat(values, 2), torch.cat(keys, 2), torch.cat(masks, 2)
            # This base is only a query initializer; distinct camera/height/scale
            # samples remain available to the actual learned attention.
            base = (value * valid[..., None]).sum(2) / valid.sum(2).clamp_min(1)[..., None]
            query = self.query_image(base) + q_context[:, start:stop]
            evidence = self._aggregate_evidence(query, key, value, valid)
            chunks.append(base + evidence.to(base.dtype))
            visible_chunks.append(valid.any(-1))
        scene = torch.cat(chunks, 1)
        visible = torch.cat(visible_chunks, 1)
        # Broad image context for traffic lights / actors beyond ground samples.
        context_global = self.global_proj(current_global.mean(dim=(1, 3, 4)))
        scene = scene + context_global[:, None]
        scene = self._apply_cross_cell_goal_residual(scene, visible, goal_xy)
        raster = self.refine(scene.transpose(1, 2).reshape(b, c, *self.config.grid_size))
        return {"scene_features": raster.flatten(2).transpose(1, 2),
                "occ_logits": self.occ_head(raster).float(),
                "lane_logits": self.lane_head(raster).float(),
                "scene_visible": visible.reshape(b, 1, *self.config.grid_size)}
