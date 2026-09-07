"""Deployable V2 graph: raw images -> shared perception/motion -> direct plan.

All visual forwards, auxiliary heads, and the planner run in ``forward``. No
precomputed feature cache or GT status is accepted by the deployment API.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from scripts.sparse_scoredrive import ResNet34FPN128

from .config import MotionDriveV2Config, validated_plan_output_scale
from .motion_encoder import MotionEncoder
from .planner import DirectTrajectoryPlanner
from .scene_encoder import SharedSceneEncoder


class MotionDriveV2(nn.Module):
    def __init__(self, config: MotionDriveV2Config | dict | None = None):
        super().__init__()
        if config is None:
            config = MotionDriveV2Config()
        elif isinstance(config, dict):
            config = MotionDriveV2Config(**config)
        self.config = config
        self.backbone_fpn = ResNet34FPN128(config.channels, arch=config.backbone_arch)
        self.scene_encoder = SharedSceneEncoder(config)
        self.motion_encoder = MotionEncoder(config)
        self.planner = DirectTrajectoryPlanner(config)

    @torch.no_grad()
    def reparameterize_plan_output_scale(self, new_scale, preserve_function=True):
        """Change XY head units, optionally preserving the current physical plan.

        With preservation, W'/b' = (old_scale/new_scale) * W/b row-wise.
        This is a coordinate-unit reparameterization INSIDE the neural head:
        there is no goal, trajectory correction, or kinematic extrapolation.
        It introduces no parameter/buffer keys, so legacy strict loads work.

        Call after checkpoint loading and BEFORE creating the new optimizer.
        Existing optimizer moments are not transformed by this method; using
        them after a unit change would not be an equivalent optimizer restart.
        """
        scale = validated_plan_output_scale(new_scale)
        old = validated_plan_output_scale(self.config.plan_output_scale)
        head = self.planner.xy_head[-1]
        if not isinstance(head, nn.Linear) or head.out_features != 2:
            raise TypeError("Expected the final XY projection to be Linear(...,2)")
        if preserve_function:
            ratio = head.weight.new_tensor(old) / head.weight.new_tensor(scale)
            head.weight.mul_(ratio[:, None])
            if head.bias is not None:
                head.bias.mul_(ratio)
        self.config.plan_output_scale = scale
        self.planner.config.plan_output_scale = scale
        return {"old_scale": list(old), "new_scale": list(scale),
                "preserve_function": bool(preserve_function),
                "optimizer_state_transformed": False}

    def load_pretrained_backbone(self, path: str | Path):
        """Load public ResNet trunk only; random compact FPN stays explicit.

        Supports torchvision, MMDetection ``backbone.`` and VAD
        ``img_backbone.`` key layouts. This is a trusted local checkpoint loader;
        callers must document provenance and must not load holdout-trained ETRI
        weights as public pretraining.
        """
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        raw = raw.get("state_dict", raw)
        target = self.backbone_fpn.state_dict()
        remap = {}
        for name, tensor in raw.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            key = name.removeprefix("module.")
            for prefix in ("img_backbone.", "backbone.", "backbone_fpn."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            if key.startswith("conv1."):
                key = "stem.0." + key[len("conv1."):]
            elif key.startswith("bn1."):
                key = "stem.1." + key[len("bn1."):]
            if key.startswith(("stem.", "layer1.", "layer2.", "layer3.", "layer4.")):
                if key in target and target[key].shape == tensor.shape:
                    remap[key] = tensor
        if not remap:
            raise ValueError(f"No compatible public ResNet tensors in {path}")
        missing, unexpected = self.backbone_fpn.load_state_dict(remap, strict=False)
        nonhead_missing = [key for key in missing if not key.startswith(("lat", "out"))]
        return {"injected": len(remap), "unexpected": list(unexpected),
                "nonhead_missing": nonhead_missing,
                "fpn_initialized_from_checkpoint": False}

    def forward_parts(self, images, history_images, lidar2img, history_transforms,
                      time_offsets, goal_xy):
        """Pure feature extraction shared by training, audit and deployment.

        Image tensors are normalized RGB floats. ``lidar2img`` is expressed in
        current input-image pixels; histories have the same cropped field of
        view at lower resolution. ``history_transforms`` maps CURRENT ego to
        each PAST ego using full SE(3). Time offsets are positive elapsed seconds.
        """
        if images.ndim != 5 or images.shape[1:3] != (6, 3):
            raise ValueError("images must have shape [B,6,3,H,W]")
        b = images.shape[0]
        if history_images.ndim != 5 or history_images.shape[:3] != (b, self.config.n_history, 3):
            raise ValueError("history_images must have shape [B,n_history,3,Hh,Wh]")
        if lidar2img.shape != (b, 6, 4, 4):
            raise ValueError("lidar2img must be [B,6,4,4]")
        if history_transforms.shape != (b, self.config.n_history, 4, 4):
            raise ValueError("history_transforms must be [B,n_history,4,4]")
        if time_offsets.shape != (b, self.config.n_history) or goal_xy.shape != (b, 2):
            raise ValueError("time_offsets or goal_xy shape mismatch")
        current_levels, current_p4 = self.backbone_fpn(images.flatten(0, 1))
        current_levels = tuple(f.reshape(b, 6, *f.shape[1:]) for f in current_levels)
        current_p4 = current_p4.reshape(b, 6, *current_p4.shape[1:])
        if self.config.motion_input_mode == "legacy":
            # Keep the original four-history-image BN and compute path intact.
            history_levels, _ = self.backbone_fpn(history_images.flatten(0, 1))
            history_levels = tuple(f.reshape(b, self.config.n_history, *f.shape[1:]) for f in history_levels)
            motion_current = tuple(f[:, 0] for f in current_levels)
        else:
            # BOTH experimental arms execute exactly the same five-image pass.
            # Thus the raw current-feature source is the only contrast, without
            # confounding it with history BatchNorm statistics or compute cost.
            current_front_small = F.interpolate(images[:, 0], size=history_images.shape[-2:],
                                                mode="bilinear", align_corners=False,
                                                antialias=True)
            temporal_images = torch.cat([current_front_small[:, None], history_images], 1)
            temporal_levels, _ = self.backbone_fpn(temporal_images.flatten(0, 1))
            temporal_levels = tuple(f.reshape(b, self.config.n_history + 1, *f.shape[1:])
                                    for f in temporal_levels)
            history_levels = tuple(f[:, 1:] for f in temporal_levels)
            motion_current = (tuple(f[:, 0] for f in current_levels)
                              if self.config.motion_input_mode == "high_feature"
                              else tuple(f[:, 0] for f in temporal_levels))
        # Raw motion branches BEFORE any use of goal or pose alignment.
        motion = self.motion_encoder(motion_current, history_levels, time_offsets)
        scene = self.scene_encoder(current_levels, history_levels, current_p4, lidar2img,
                                   history_transforms, time_offsets, goal_xy,
                                   images.shape[-2:])
        return {**scene, **motion}

    def plan_from_features(self, scene_features, motion_features, predicted_state,
                           predicted_history):
        return self.planner(scene_features, motion_features, predicted_state, predicted_history)

    def forward(self, images, history_images, lidar2img, history_transforms,
                time_offsets, goal_xy):
        parts = self.forward_parts(images, history_images, lidar2img,
                                   history_transforms, time_offsets, goal_xy)
        plan = self.plan_from_features(parts["scene_features"], parts["motion_features"],
                                       parts["state_hat"], parts["history_hat"])
        return {**parts, "plan_abs": plan}
