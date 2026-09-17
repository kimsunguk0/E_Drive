"""Deployable progress refinement for the registered MR-NATIVE graph.

The coefficient head receives image-derived visual/motion features and a
stop-gradient encoding of the model's own base trajectory.  It never receives
raw goal, dynamic pose, or provided ego status.  The final trajectory remains
inside the neural forward.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path("/NHNHOME/data/sukim/adcl")
BASE_DIR = ROOT / "experiments/md_r0_reset_20260914"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(BASE_DIR))

from models.motiondrive_v2.model import MotionDriveV2
from models.motiondrive_v2.motion_encoder import MotionEncoder
from motiondrive_v2_data import CAMERA_ORDER, MEAN, STD


SIDE_HISTORY_KEY = "side_history_images"
SIDE_CAMERA_INDICES = (1, 2)  # front-right, front-left in the canonical order
SIDE_OFFSETS = (1, 2, 5)
SIDE_TIMES = (.1, .2, .5)
DT = .5
T_MID = (.25, .75, 1.25, 1.75, 2.25, 2.75)


def stable_predicted_directions(plan: torch.Tensor, eps: float = 1e-3):
    """Return detached segment lengths/directions with a deterministic fallback.

    A short segment uses the closest non-short segment of the same predicted
    path.  An entirely stationary prediction uses ego-forward +x.  No GT value
    participates in this function.
    """
    if plan.ndim != 3 or plan.shape[-2:] != (6, 2):
        raise ValueError(f"plan must be [B,6,2], got {tuple(plan.shape)}")
    source = plan.detach().float()
    origin = torch.zeros_like(source[:, :1])
    delta = torch.diff(torch.cat([origin, source], 1), dim=1)
    lengths = torch.linalg.vector_norm(delta, dim=-1)
    directions = torch.zeros_like(delta)
    for batch in range(len(source)):
        good = torch.nonzero(lengths[batch] >= eps, as_tuple=False).flatten()
        if not len(good):
            directions[batch, :, 0] = 1.
            continue
        directions[batch, good] = delta[batch, good] / lengths[batch, good, None]
        for index in torch.nonzero(lengths[batch] < eps, as_tuple=False).flatten():
            nearest = good[torch.argmin((good - index).abs())]
            directions[batch, index] = directions[batch, nearest]
    return lengths.detach(), directions.detach()


def apply_progress_correction(base_plan: torch.Tensor, coefficients: torch.Tensor,
                              *, eps: float = 1e-3):
    """Apply scalar dv or (dv,da) without permitting interval reversal.

    The correction path is detached, while ``base_plan`` retains its direct
    gradient path.  At exactly zero coefficients the output is bitwise the
    base plan (the explicit zero shortcut also avoids signed-zero surprises).
    """
    if coefficients.ndim != 2 or coefficients.shape[0] != len(base_plan) \
            or coefficients.shape[1] not in (1, 2):
        raise ValueError("coefficients must be [B,1] or [B,2]")
    lengths, directions = stable_predicted_directions(base_plan, eps=eps)
    coeff = coefficients.float()
    delta_speed = coeff[:, :1]
    if coeff.shape[1] == 2:
        midpoint = coeff.new_tensor(T_MID)[None]
        delta_speed = delta_speed + coeff[:, 1:2] * midpoint
    adjusted = (lengths + DT * delta_speed).clamp_min(0.)
    delta_length = adjusted - lengths
    correction = torch.cumsum(directions * delta_length[..., None], dim=1)
    return base_plan.float() + correction, {
        "base_segment_length": lengths,
        "progress_delta_length": delta_length,
        "progress_direction": directions,
    }


class ProgressRefiner(nn.Module):
    """One learned query over raw visual/motion tokens and the detached plan."""

    def __init__(self, channels=128, coefficient_count=1, cap=1.0,
                 side_enabled=False, heads=4):
        super().__init__()
        if coefficient_count not in (1, 2) or cap <= 0:
            raise ValueError("coefficient_count must be 1/2 and cap positive")
        self.coefficient_count = coefficient_count
        self.cap = float(cap)
        self.side_enabled = bool(side_enabled)
        self.plan_encoder = nn.Sequential(
            nn.Linear(18, channels), nn.GELU(), nn.Linear(channels, channels))
        self.visual_camera = nn.Parameter(torch.randn(6, channels) * .02)
        self.visual_type = nn.Parameter(torch.randn(1, 1, channels) * .02)
        self.front_type = nn.Parameter(torch.randn(1, 1, channels) * .02)
        self.side_type = nn.Parameter(torch.randn(1, 1, channels) * .02)
        self.query = nn.Parameter(torch.randn(1, 1, channels) * .02)
        self.memory_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=0., batch_first=True)
        self.refine = nn.Sequential(nn.LayerNorm(channels * 2),
                                    nn.Linear(channels * 2, channels * 2),
                                    nn.GELU(), nn.Linear(channels * 2, channels))
        self.output = nn.Linear(channels, coefficient_count)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def encode_plan(base_plan):
        plan = base_plan.detach().float()
        delta = torch.diff(torch.cat([torch.zeros_like(plan[:, :1]), plan], 1), dim=1)
        length = torch.linalg.vector_norm(delta, dim=-1)
        scale = plan.new_tensor([10., 5.])
        return torch.cat([(plan / scale).flatten(1), length / 10.], -1)

    def forward(self, base_plan, visual_tokens, front_motion, side_motion=None):
        if visual_tokens.ndim != 4 or visual_tokens.shape[1] != 6:
            raise ValueError("visual_tokens must be [B,6,N,C]")
        b, _, n, c = visual_tokens.shape
        visual = visual_tokens + self.visual_camera[None, :, None]
        visual = visual.reshape(b, 6 * n, c) + self.visual_type
        memory = [visual, front_motion.float() + self.front_type]
        if self.side_enabled:
            if side_motion is None or side_motion.ndim != 4 or side_motion.shape[:2] != (b, 2):
                raise ValueError("SIDE refiner requires [B,2,N,C] side motion")
            memory.append(side_motion.float().flatten(1, 2) + self.side_type)
        elif side_motion is not None:
            raise ValueError("FRONT refiner cannot consume side motion")
        plan_token = self.plan_encoder(self.encode_plan(base_plan))[:, None]
        query = plan_token + self.query
        memory = self.memory_norm(torch.cat(memory, 1))
        attended, _ = self.attention(query, memory, memory, need_weights=False)
        hidden = attended + self.refine(torch.cat([attended, plan_token], -1))
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.cap * torch.tanh(self.output(hidden[:, 0].float()))


class ResidualMotionDriveV2(MotionDriveV2):
    """MR-NATIVE plus an optional two-camera raw temporal branch."""

    def __init__(self, config, *, coefficient_count=1, side_enabled=False, cap=1.0):
        super().__init__(config)
        self.side_enabled = bool(side_enabled)
        self.coefficient_count = int(coefficient_count)
        if self.config.correlation_radius != 4:
            raise ValueError("Residual lineage requires the registered MR radius-4 config")
        if self.side_enabled:
            self.side_motion_encoder = MotionEncoder(self.config)
            self.side_calibration = nn.Sequential(nn.Linear(12, self.config.channels),
                                                  nn.GELU(), nn.Linear(self.config.channels,
                                                                     self.config.channels))
            self.side_state_delta = nn.Linear(self.config.channels, 6)
            self.side_history_delta = nn.Linear(4, 4)
            for module in (self.side_state_delta, self.side_history_delta):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        self.progress_refiner = ProgressRefiner(
            self.config.channels, coefficient_count, cap, self.side_enabled,
            self.config.planner_heads)

    @staticmethod
    def _visual_tokens(current_p4):
        b, cameras, channels = current_p4.shape[:3]
        pooled = F.adaptive_avg_pool2d(current_p4.flatten(0, 1), (2, 3))
        return pooled.flatten(2).transpose(1, 2).reshape(b, cameras, 6, channels).float()

    def _side_features(self, images, side_history, lidar2img, time_offsets):
        if side_history is None:
            raise ValueError("SIDE model requires side_history_images")
        b = len(images)
        if side_history.shape[:4] != (b, 2, len(SIDE_OFFSETS), 3):
            raise ValueError(f"side history shape mismatch: {tuple(side_history.shape)}")
        current = images[:, list(SIDE_CAMERA_INDICES)]
        stack = torch.cat([current[:, :, None], side_history], 2)
        levels, _ = self.backbone_fpn(stack.flatten(0, 2))
        levels = tuple(value.reshape(b, 2, len(SIDE_OFFSETS) + 1, *value.shape[1:])
                       for value in levels)
        current_levels = tuple(value[:, :, 0].flatten(0, 1) for value in levels)
        history_levels = tuple(value[:, :, 1:].flatten(0, 1) for value in levels)
        side_times = time_offsets[:, :len(SIDE_OFFSETS), None].expand(-1, -1, 2)
        side_times = side_times.transpose(1, 2).reshape(b * 2, len(SIDE_OFFSETS))
        output = self.side_motion_encoder(current_levels, history_levels, side_times)
        tokens = output["motion_features"].reshape(b, 2, -1, self.config.channels)
        calibration = lidar2img[:, list(SIDE_CAMERA_INDICES), :3].float().reshape(b, 2, 12)
        calibration = calibration / calibration.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        tokens = tokens + self.side_calibration(calibration)[:, :, None]
        state_feature = tokens.mean((1, 2))
        state_delta = self.side_state_delta(state_feature) * self.side_motion_encoder.state_scale
        raw_history = output["history_hat"].reshape(b, 2, len(SIDE_OFFSETS), 4).mean(1)
        history_delta = self.side_history_delta(
            raw_history / self.side_motion_encoder.history_scale) * self.side_motion_encoder.history_scale
        return tokens, state_delta, history_delta

    def forward_parts(self, images, history_images, lidar2img, history_transforms,
                      time_offsets, goal_xy, motion_current=None, motion_history=None,
                      side_history_images=None):
        if motion_current is None or motion_history is None:
            raise ValueError("Residual MR arms require native motion canvas tensors")
        b = images.shape[0]
        current_levels, current_p4 = self.backbone_fpn(images.flatten(0, 1))
        current_levels = tuple(f.reshape(b, 6, *f.shape[1:]) for f in current_levels)
        current_p4 = current_p4.reshape(b, 6, *current_p4.shape[1:])

        scene_small = F.interpolate(images[:, 0], size=history_images.shape[-2:],
                                    mode="bilinear", align_corners=False, antialias=True)
        scene_stack = torch.cat([scene_small[:, None], history_images], 1)
        scene_levels, _ = self.backbone_fpn(scene_stack.flatten(0, 1))
        scene_levels = tuple(f.reshape(b, self.config.n_history + 1, *f.shape[1:])
                             for f in scene_levels)
        history_levels = tuple(f[:, 1:] for f in scene_levels)

        motion_stack = torch.cat([motion_current[:, None], motion_history], 1)
        motion_levels, _ = self.backbone_fpn(motion_stack.flatten(0, 1))
        motion_levels = tuple(f.reshape(b, self.config.n_history + 1, *f.shape[1:])
                              for f in motion_levels)
        motion = self.motion_encoder(tuple(f[:, 0] for f in motion_levels),
                                     tuple(f[:, 1:] for f in motion_levels), time_offsets)
        scene = self.scene_encoder(current_levels, history_levels, current_p4, lidar2img,
                                   history_transforms, time_offsets, goal_xy,
                                   images.shape[-2:])
        parts = {**scene, **motion, "raw_visual_tokens": self._visual_tokens(current_p4)}
        if self.side_enabled:
            side, state_delta, history_delta = self._side_features(
                images, side_history_images, lidar2img, time_offsets)
            parts["side_motion_features"] = side
            parts["state_hat"] = parts["state_hat"] + state_delta
            parts["history_hat"] = torch.cat([
                parts["history_hat"][:, :len(SIDE_OFFSETS)] + history_delta,
                parts["history_hat"][:, len(SIDE_OFFSETS):],
            ], 1)
        return parts

    def forward(self, images, history_images, lidar2img, history_transforms,
                time_offsets, goal_xy, motion_current=None, motion_history=None,
                side_history_images=None):
        parts = self.forward_parts(images, history_images, lidar2img,
                                   history_transforms, time_offsets, goal_xy,
                                   motion_current, motion_history, side_history_images)
        base_plan = self.plan_from_features(parts["scene_features"], parts["motion_features"],
                                            parts["state_hat"], parts["history_hat"])
        coefficient = self.progress_refiner(
            base_plan.detach(), parts["raw_visual_tokens"], parts["motion_features"],
            parts.get("side_motion_features"))
        final_plan, geometry = apply_progress_correction(base_plan, coefficient)
        return {**parts, **geometry, "plan_base_abs": base_plan,
                "progress_coeff": coefficient, "plan_abs": final_plan}


class SideHistoryDataset(torch.utils.data.Dataset):
    """Add native-resolution short histories for the canonical side pair."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def _jitter(self, row, camera_index):
        if not getattr(self.base, "augment", False):
            return None
        rng = np.random.default_rng(self.base.seed + self.base.epoch * 1000003 + row)
        return rng.uniform(.9, 1.1, (6, 3))[camera_index]

    def _image(self, scene, camera, frame, jitter):
        path = Path(self.base.image_root) / scene / camera / f"{frame:08d}.jpg"
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        if image.size != (768, 432):
            raise ValueError(f"side cache geometry mismatch: {path}: {image.size}")
        if jitter is not None:
            for cls, factor in zip((ImageEnhance.Brightness, ImageEnhance.Contrast,
                                    ImageEnhance.Color), jitter):
                image = cls(image).enhance(float(factor))
        array = np.asarray(image, np.float32) / 255.
        return torch.from_numpy(((array - MEAN) / STD).transpose(2, 0, 1).copy())

    def __getitem__(self, index):
        item = self.base[index]
        row = int(self.base.rows[index])
        scene = str(self.base.scene_names[row])
        frame = int(self.base.arr["frame"][row])
        cameras = []
        for camera_index in SIDE_CAMERA_INDICES:
            camera = CAMERA_ORDER[camera_index]
            jitter = self._jitter(row, camera_index)
            cameras.append(torch.stack([
                self._image(scene, camera, frame - offset, jitter)
                for offset in SIDE_OFFSETS]))
        item[SIDE_HISTORY_KEY] = torch.stack(cameras)
        return item


def wrap_flip_item(original):
    def flip_item(item, width_full, width_hist):
        output = original(item, width_full, width_hist)
        if SIDE_HISTORY_KEY in item:
            # Canonical side order is right,left; a mirror swaps the two views.
            output[SIDE_HISTORY_KEY] = torch.flip(item[SIDE_HISTORY_KEY], [-1])[[1, 0]]
        return output
    return flip_item


def add_residual_inputs(original, batch, **kwargs):
    inputs = original(batch, **kwargs)
    if SIDE_HISTORY_KEY in batch:
        inputs[SIDE_HISTORY_KEY] = batch[SIDE_HISTORY_KEY]
    return inputs
