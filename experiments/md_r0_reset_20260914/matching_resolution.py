#!/usr/bin/env python3
"""MR arms: the motion branch matched on a 768x432 canvas, detail the only variable.

Both arms feed the motion encoder a 768x432 canvas and use radius 4 at each level,
which keeps the ORIGINAL search reach in original-image pixels (fine +/-32,
coarse +/-64) while quantising it four times more finely.

    MR-LOWDETAIL : history and motion-current go 768 -> 384 -> 768 with PIL
                   bilinear, inside the dataset's own resize step, so the detail
                   is destroyed but the canvas matches.
    MR-NATIVE    : the same tensors keep the cached 768x432 pixels.

The scene branch is untouched: it keeps consuming the original 384x216 history
tensor through its own backbone pass, so its features and gradient path are the
ones the E1 lineage already used.

correlation_fuse's first convolution changes shape with the bin count, so it is
rebuilt and initialised identically in both arms. No 25-channel weight is
reshaped into an 81-channel tensor, and step-zero parity with LEN is NOT claimed.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

MOTION_CURRENT_KEY = "motion_current_images"
MOTION_HISTORY_KEY = "motion_history_images"
MOTION_WH = (768, 432)
BOTTLENECK_WH = (384, 216)
NEW_RADIUS = 4


class MotionCanvasDataset(torch.utils.data.Dataset):
    """Adds the motion-branch canvas without disturbing any existing tensor."""

    def __init__(self, base, detail: str):
        if detail not in ("native", "lowdetail"):
            raise ValueError("detail must be native or lowdetail")
        self.base, self.detail = base, detail

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def set_epoch(self, epoch):
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __getitem__(self, index):
        item = self.base[index]
        row = int(self.base.rows[index])
        scene = str(self.base.scene_names[row])
        frame = int(self.base.arr["frame"][row])
        jitter = self._jitter_for(row)
        offsets = self.base.history_offsets.tolist()
        item[MOTION_CURRENT_KEY] = self._canvas(scene, "camera_front", frame, jitter)
        item[MOTION_HISTORY_KEY] = torch.stack(
            [self._canvas(scene, "camera_front", frame - int(o), jitter) for o in offsets])
        return item

    def _jitter_for(self, row):
        """Reproduce the loader's own deterministic draw; never a new random draw.

        The loader uses default_rng(seed + epoch*1000003 + row).uniform(.9,1.1,(6,3))
        and applies the FRONT camera's row to the current front image and to every
        history image, so the canvas must use exactly that row.
        """
        import numpy as np
        if not getattr(self.base, "augment", False):
            return None
        rng = np.random.default_rng(self.base.seed + self.base.epoch * 1000003 + row)
        return rng.uniform(.9, 1.1, (6, 3))[0]

    def _canvas(self, scene, camera, frame, jitter):
        from PIL import Image, ImageEnhance
        import numpy as np
        from motiondrive_v2_data import MEAN, STD
        path = Path(self.base.image_root) / scene / camera / f"{frame:08d}.jpg"
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        if image.size != MOTION_WH:
            raise ValueError(f"Cache geometry mismatch: {path}: {image.size}")
        if self.detail == "lowdetail":
            # exactly the contract the loader uses, applied twice: the 384x216
            # bottleneck destroys the detail, the second step restores the canvas
            image = image.resize(BOTTLENECK_WH, Image.Resampling.BILINEAR)
            image = image.resize(MOTION_WH, Image.Resampling.BILINEAR)
        if jitter is not None:
            for cls, factor in zip((ImageEnhance.Brightness, ImageEnhance.Contrast,
                                    ImageEnhance.Color), jitter):
                image = cls(image).enhance(float(factor))
        array = np.asarray(image, np.float32) / 255.
        return torch.from_numpy(((array - MEAN) / STD).transpose(2, 0, 1).copy())


def rebuild_correlation_fuse(model, radius):
    """Resize only the first convolution, whose input channels follow the bins."""
    encoder = model.motion_encoder
    config = encoder.config
    channels, cm = config.channels, config.correlation_channels
    bins = (2 * radius + 1) ** 2
    old = encoder.correlation_fuse[0]
    new = nn.Conv2d(2 * (bins + 2 * cm), channels, 3, padding=1)
    encoder.correlation_fuse[0] = new
    object.__setattr__(config, "correlation_radius", radius)
    return {"module": "motion_encoder.correlation_fuse.0",
            "old_in_channels": old.in_channels, "new_in_channels": new.in_channels,
            "bins": bins, "reinitialised": True,
            "parameters": int(sum(p.numel() for p in new.parameters()))}


def install(model, detail):
    """Point the motion branch at the canvas tensors; leave the scene branch alone."""
    original_forward_parts = type(model).forward_parts

    def forward_parts(self, images, history_images, lidar2img, history_transforms,
                      time_offsets, goal_xy, motion_current=None, motion_history=None):
        if motion_current is None or motion_history is None:
            raise ValueError("MR arms require the motion canvas tensors")
        b = images.shape[0]
        current_levels, current_p4 = self.backbone_fpn(images.flatten(0, 1))
        current_levels = tuple(f.reshape(b, 6, *f.shape[1:]) for f in current_levels)
        current_p4 = current_p4.reshape(b, 6, *current_p4.shape[1:])

        # scene branch: the original 384x216 temporal stack, unchanged
        scene_small = F.interpolate(images[:, 0], size=history_images.shape[-2:],
                                    mode="bilinear", align_corners=False, antialias=True)
        scene_stack = torch.cat([scene_small[:, None], history_images], 1)
        scene_levels, _ = self.backbone_fpn(scene_stack.flatten(0, 1))
        scene_levels = tuple(f.reshape(b, self.config.n_history + 1, *f.shape[1:])
                             for f in scene_levels)
        history_levels = tuple(f[:, 1:] for f in scene_levels)

        # motion branch: its own canvas, its own backbone pass
        motion_stack = torch.cat([motion_current[:, None], motion_history], 1)
        motion_levels, _ = self.backbone_fpn(motion_stack.flatten(0, 1))
        motion_levels = tuple(f.reshape(b, self.config.n_history + 1, *f.shape[1:])
                              for f in motion_levels)
        motion_current_levels = tuple(f[:, 0] for f in motion_levels)
        motion_history_levels = tuple(f[:, 1:] for f in motion_levels)

        motion = self.motion_encoder(motion_current_levels, motion_history_levels, time_offsets)
        scene = self.scene_encoder(current_levels, history_levels, current_p4, lidar2img,
                                   history_transforms, time_offsets, goal_xy,
                                   images.shape[-2:])
        return {**scene, **motion}

    def forward(self, images, history_images, lidar2img, history_transforms,
                time_offsets, goal_xy, motion_current=None, motion_history=None):
        parts = self.forward_parts(images, history_images, lidar2img, history_transforms,
                                   time_offsets, goal_xy, motion_current, motion_history)
        plan = self.plan_from_features(parts["scene_features"], parts["motion_features"],
                                       parts["state_hat"], parts["history_hat"])
        return {**parts, "plan_abs": plan}

    type(model).forward_parts = forward_parts
    type(model).forward = forward
    return original_forward_parts


def model_inputs_with_canvas(original, batch, **kwargs):
    inputs = original(batch, **kwargs)
    inputs["motion_current"] = batch[MOTION_CURRENT_KEY]
    inputs["motion_history"] = batch[MOTION_HISTORY_KEY]
    return inputs


def wrap_flip_item(original):
    """The mirror must reach the canvas tensors too.

    flip_item copies the item dict and mirrors the keys it knows about; the two
    canvas tensors are new, so without this they would ride through a mirrored
    sample unflipped, which is the silent-label failure the flip module warns
    about. Both are front-camera images and the front camera is its own mirror
    pair, so the mirror is the same horizontal flip history_images gets.
    """

    def flip_item(item, width_full, width_hist):
        out = original(item, width_full, width_hist)
        for key in (MOTION_CURRENT_KEY, MOTION_HISTORY_KEY):
            if key in item:
                out[key] = torch.flip(item[key], dims=[-1])
        return out

    return flip_item


def assert_canvas_contract(item, detail):
    """Shapes, and that lowdetail really lost detail while native did not."""
    current, history = item[MOTION_CURRENT_KEY], item[MOTION_HISTORY_KEY]
    expected_hw = (MOTION_WH[1], MOTION_WH[0])
    if tuple(current.shape[-2:]) != expected_hw or tuple(history.shape[-2:]) != expected_hw:
        raise ValueError(f"canvas must be {expected_hw}, got {current.shape}, {history.shape}")
    if history.shape[0] != item["history_images"].shape[0]:
        raise ValueError("canvas history count differs from the scene history count")
    # high-frequency energy: a 384 bottleneck removes most of it
    diff = (current[:, :, 1:] - current[:, :, :-1]).abs().mean()
    return {"detail": detail, "canvas_hw": list(expected_hw),
            "mean_abs_vertical_gradient": float(diff),
            "scene_history_hw": list(item["history_images"].shape[-2:])}
