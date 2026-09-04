"""Latency-first, current-only sparse visual scorer for ETRI ScoreDrive.

The model consumes exactly the six current camera images and their calibrated
ego/lidar-to-image matrices.  It deliberately contains no dense BEV, temporal
state, object/map decoder, goal, command, status, or past-trajectory input.

Candidate geometry is a fixed, train-only A0 bank.  It is used only as image
sampling addresses and as fixed kinematic descriptors.  Only the image-selected
12 complete candidates leave ``forward``; full-K scores stay private.

This is a Phase-5B feasibility skeleton.  Its weights are random and it makes
no accuracy claim.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import resnet34


CAMERA_ORDER = (
    "camera_front",
    "camera_front_right",
    "camera_front_left",
    "camera_rear_wide",
    "camera_rear_left",
    "camera_rear_right",
)
CROP_KEEP_TOP = frozenset(
    ("camera_front_left", "camera_front_right", "camera_rear_left",
     "camera_rear_right", "camera_rear_wide")
)


def _lidar2img_from_cam(cam: dict) -> np.ndarray:
    """Mirror VADCustomNuScenesDataset.get_data_info exactly."""
    lidar2cam_r = np.linalg.inv(np.asarray(cam["sensor2lidar_rotation"], np.float64))
    lidar2cam_t = np.asarray(cam["sensor2lidar_translation"], np.float64) @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float64)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[:3, :3] = np.asarray(cam["cam_intrinsic"], np.float64)
    return viewpad @ lidar2cam_rt.T


def build_cached_lidar2img(
    info: dict,
    scale: float = 0.4,
    crop_size: Tuple[int, int] = (1920, 1080),
    camera_order: Sequence[str] = CAMERA_ORDER,
) -> np.ndarray:
    """Build calibration for the existing 768x432 undistort/crop cache.

    This mirrors ``CachedImageGeometry(scale=.4)``.  The stored intrinsic is
    already the undistorted intrinsic; crop and scale are the only remaining
    matrix transforms.
    """
    matrices = []
    crop_w, crop_h = crop_size
    for name in camera_order:
        cam = info["cams"][name]
        ox = (int(cam["image_width"]) - crop_w) // 2
        oy = 0 if name in CROP_KEEP_TOP else int(cam["image_height"]) - crop_h
        shift = np.eye(4, dtype=np.float64)
        shift[0, 2] = -ox
        shift[1, 2] = -oy
        resize = np.eye(4, dtype=np.float64)
        resize[0, 0] = scale
        resize[1, 1] = scale
        matrices.append(resize @ shift @ _lidar2img_from_cam(cam))
    return np.asarray(matrices, dtype=np.float32)


class ResNet34FPN128(nn.Module):
    """Shared six-camera ResNet-34 with a compact 128-channel FPN."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        net = resnet34(weights=None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.lat2 = nn.Conv2d(128, channels, 1, bias=False)
        self.lat3 = nn.Conv2d(256, channels, 1, bias=False)
        self.lat4 = nn.Conv2d(512, channels, 1, bias=False)
        self.out2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, images: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        x = self.stem(images)
        x = self.layer1(x)
        c2 = self.layer2(x)  # stride 8
        c3 = self.layer3(c2)  # stride 16
        c4 = self.layer4(c3)  # stride 32
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        return (self.out2(p2), self.out3(p3)), p4


def project_candidate_points(
    candidate_abs: Tensor,
    lidar2img: Tensor,
    image_hw: Tuple[int, int],
    heights: Iterable[float] = (0.0, 1.0),
) -> Tuple[Tensor, Tensor]:
    """Project fixed candidate waypoints using the real camera matrices.

    Args:
        candidate_abs: [K,T,2] ego/lidar-frame absolute XY.
        lidar2img: [B,6,4,4], already transformed to the input image geometry.
    Returns:
        grid: [B,6,K,T,H,2] in grid_sample coordinates.
        visible: [B,6,K,T,H].
    """
    image_h, image_w = image_hw
    z_values = candidate_abs.new_tensor(tuple(heights))
    k, t, _ = candidate_abs.shape
    nh = int(z_values.numel())
    xy = candidate_abs[:, :, None, :].expand(k, t, nh, 2)
    z = z_values.view(1, 1, nh, 1).expand(k, t, nh, 1)
    ones = torch.ones_like(z)
    points = torch.cat((xy, z, ones), dim=-1)
    # B,C,K,T,H,4.  Projection stays fp32 even under autocast.
    with torch.autocast(device_type=candidate_abs.device.type, enabled=False):
        projected = torch.einsum(
            "bcij,kthj->bckthi", lidar2img.float(), points.float())
        depth = projected[..., 2]
        safe_depth = depth.clamp_min(1.0e-5)
        uv = projected[..., :2] / safe_depth[..., None]
        visible = (
            (depth > 1.0e-5)
            & (uv[..., 0] >= 0.0)
            & (uv[..., 0] < float(image_w))
            & (uv[..., 1] >= 0.0)
            & (uv[..., 1] < float(image_h))
        )
        grid_x = 2.0 * uv[..., 0] / float(image_w) - 1.0
        grid_y = 2.0 * uv[..., 1] / float(image_h) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
    return grid, visible


class SparseScoreDrive(nn.Module):
    """Current-only six-camera sparse candidate scorer and candidate API."""

    def __init__(
        self,
        bank_path: str | Path,
        channels: int = 128,
        heights: Tuple[float, ...] = (0.0, 1.0),
        score_top: int = 3,
        n_out: int = 12,
        nms_pool: int = 64,
    ) -> None:
        super().__init__()
        bank = np.load(str(bank_path), allow_pickle=False)
        abs5 = np.asarray(bank["candidate_xy_abs_5s"], np.float32)
        inc5 = np.asarray(bank["candidate_xy_inc_5s"], np.float32)
        ids = np.asarray(bank["candidate_ids"], np.int64)
        anchor_dist = np.asarray(bank["anchor_dist"], np.float32)
        if abs5.shape != (1024, 10, 2) or inc5.shape != abs5.shape:
            raise ValueError(f"unexpected A0 shapes: abs={abs5.shape}, inc={inc5.shape}")
        self.register_buffer("candidate_abs", torch.from_numpy(abs5), persistent=True)
        self.register_buffer("candidate_inc", torch.from_numpy(inc5), persistent=True)
        self.register_buffer("candidate_ids", torch.from_numpy(ids), persistent=True)
        self.register_buffer("anchor_dist", torch.from_numpy(anchor_dist), persistent=True)
        self.register_buffer("nms_tau", torch.tensor(float(bank["nms_tau"])), persistent=True)
        self.backbone_fpn = ResNet34FPN128(channels)
        self.visual_proj = nn.Linear(channels, channels, bias=False)
        self.kinematic_proj = nn.Sequential(
            nn.Linear(6, channels, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channels, channels, bias=False),
        )
        self.local_score = nn.Linear(channels, 1, bias=False)
        self.scene_proj = nn.Linear(channels, channels, bias=False)
        self.heights = tuple(float(x) for x in heights)
        self.score_top = int(score_top)
        self.n_out = int(n_out)
        self.nms_pool = int(nms_pool)
        if (self.score_top, self.n_out, self.nms_pool) != (3, 12, 64):
            raise ValueError("Phase-5B canonical shortlist must be score3+nms9 from top64")

        desc = self._kinematic_descriptors(torch.from_numpy(abs5))
        self.register_buffer("kinematic_desc", desc, persistent=True)

    @staticmethod
    def _kinematic_descriptors(path: Tensor) -> Tensor:
        delta = torch.diff(path, dim=1, prepend=torch.zeros_like(path[:, :1]))
        speed = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        progress = torch.cumsum(speed, dim=1)
        total = progress[:, -1:].clamp_min(1.0e-3)
        progress = progress / total
        return torch.cat(
            (path[..., 0:1] / 30.0, path[..., 1:2] / 15.0,
             delta[..., 0:1] / 5.0, delta[..., 1:2] / 5.0,
             speed / 5.0, progress[..., None] if progress.ndim == 2 else progress),
            dim=-1,
        )

    def encode_images(self, images: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        if images.ndim != 5 or images.shape[1:3] != (6, 3):
            raise ValueError(f"images must be [B,6,3,H,W], got {tuple(images.shape)}")
        b, cams, channels, h, w = images.shape
        flat = images.reshape(b * cams, channels, h, w)
        levels, p4 = self.backbone_fpn(flat)
        levels = tuple(x.reshape(b, cams, *x.shape[1:]) for x in levels)
        p4 = p4.reshape(b, cams, *p4.shape[1:])
        return levels, p4

    @staticmethod
    def _sample_level(feature: Tensor, grid: Tensor, visible: Tensor) -> Tuple[Tensor, Tensor]:
        b, cams, channels, fh, fw = feature.shape
        _, _, k, t, nh, _ = grid.shape
        flat_feature = feature.reshape(b * cams, channels, fh, fw)
        flat_grid = grid.reshape(b * cams, k * t * nh, 1, 2).to(feature.dtype)
        sampled = F.grid_sample(
            flat_feature, flat_grid, mode="bilinear", padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(b, cams, channels, k, t, nh)
        sampled = sampled.permute(0, 1, 3, 4, 5, 2)
        weight = visible[..., None].to(sampled.dtype)
        count = weight.sum(dim=(1, 4)).clamp_min(1.0)
        aggregate = (sampled * weight).sum(dim=(1, 4)) / count
        point_visible = visible.any(dim=(1, 4))
        return aggregate, point_visible

    def score_features(
        self,
        levels: Tuple[Tensor, Tensor],
        p4: Tensor,
        lidar2img: Tensor,
        image_hw: Tuple[int, int],
        evidence: Tensor,
    ) -> Tensor:
        grid, visible = project_candidate_points(
            self.candidate_abs, lidar2img, image_hw, self.heights)
        sampled_levels = []
        valid = None
        for level in levels:
            sampled, level_valid = self._sample_level(level, grid, visible)
            sampled_levels.append(sampled)
            valid = level_valid if valid is None else (valid | level_valid)
        sampled = torch.stack(sampled_levels, dim=0).mean(dim=0)
        kin = self.kinematic_proj(self.kinematic_desc)
        visual = self.visual_proj(sampled)
        compatibility = (visual * kin[None]).sum(dim=-1) / math.sqrt(visual.shape[-1])
        compatibility = compatibility + self.local_score(sampled).squeeze(-1)
        valid_f = valid.to(compatibility.dtype)
        path_score = (compatibility * valid_f).sum(dim=-1) / valid_f.sum(dim=-1).clamp_min(1.0)

        scene = p4.mean(dim=(1, 3, 4))
        scene = self.scene_proj(scene)
        candidate_profile = kin.mean(dim=1)
        global_score = torch.einsum("bd,kd->bk", scene, candidate_profile)
        logits = path_score + 0.25 * global_score / math.sqrt(scene.shape[-1])
        # Exact C4 invariant: fixed geometry cannot produce a nonzero score when
        # the visual tensor contains no evidence, even after future training.
        return logits * evidence[:, None].to(logits.dtype)

    def _stable_score3_nms9(self, logits: Tensor) -> Tensor:
        b, k = logits.shape
        order = torch.argsort(logits, dim=-1, descending=True, stable=True)
        selected = order[:, :self.score_top]
        selected_mask = torch.zeros((b, k), dtype=torch.bool, device=logits.device)
        selected_mask.scatter_(1, selected, True)
        pool = order[:, :self.nms_pool]
        min_distance = torch.full(
            (b, self.nms_pool), torch.inf, dtype=self.anchor_dist.dtype,
            device=logits.device,
        )
        for j in range(self.score_top):
            min_distance = torch.minimum(
                min_distance, self.anchor_dist[pool, selected[:, j:j + 1]])
        pieces = [selected]
        for _ in range(self.n_out - self.score_top):
            eligible = (min_distance > self.nms_tau) & (~selected_mask.gather(1, pool))
            has_eligible = eligible.any(dim=1)
            pool_rank = eligible.to(torch.int8).argmax(dim=1)
            nms_choice = pool.gather(1, pool_rank[:, None]).squeeze(1)
            remaining = ~selected_mask.gather(1, order)
            fallback_rank = remaining.to(torch.int8).argmax(dim=1)
            fallback = order.gather(1, fallback_rank[:, None]).squeeze(1)
            choice = torch.where(has_eligible, nms_choice, fallback)
            pieces.append(choice[:, None])
            selected_mask.scatter_(1, choice[:, None], True)
            min_distance = torch.minimum(
                min_distance, self.anchor_dist[pool, choice[:, None]])
        return torch.cat(pieces, dim=1)

    def build_candidates(self, logits: Tensor) -> Dict[str, Tensor]:
        shortlist = self._stable_score3_nms9(logits)
        return {
            "candidate_xy_abs_5s": self.candidate_abs[shortlist],
            "candidate_xy_inc_5s": self.candidate_inc[shortlist],
            "visual_logits": logits.gather(1, shortlist),
            "candidate_ids": self.candidate_ids[shortlist],
        }

    def forward(self, images: Tensor, lidar2img: Tensor) -> Dict[str, Tensor]:
        """One complete measured forward.  There is intentionally no goal input."""
        image_hw = (int(images.shape[-2]), int(images.shape[-1]))
        evidence = images.abs().amax(dim=(1, 2, 3, 4)) > 0
        levels, p4 = self.encode_images(images)
        logits = self.score_features(levels, p4, lidar2img, image_hw, evidence)
        return self.build_candidates(logits)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
