"""Independent SIDE-SCENE and sequential image-query refinement experiments."""
from pathlib import Path
import sys
import math
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/NHNHOME/data/sukim/adcl')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/md_shared_dynamics_20260917'))
from factorized_model import SharedDynamicsMotionDriveV2
from models.motiondrive_v2.scene_encoder import masked_softmax

SIDE = 'A2-SIDE-SCENE-NOM'
QREFINE = 'A2-QREFINE-NOM'
ARMS = (SIDE, QREFINE)
SIDE_CAMERA_IDS = (2, 2, 1, 1)  # left -.1/-.5, right -.1/-.5
SIDE_POSE_INDICES = (0, 2, 0, 2)  # CONTROL offsets are (1, 2, 5, 10)
SIDE_FRAME_OFFSETS = (1, 5, 1, 5)
EXTRA_PREFIX = 'scene_encoder.evidence_attention.'


class ReadRefineEvidence(nn.Module):
    """Read image values, update query using that read, then read once more."""
    def __init__(self):
        super().__init__()
        self.query_update = nn.Sequential(nn.Linear(32 + 128, 64), nn.GELU(), nn.Linear(64, 32))
        self.output = nn.Linear(128, 128, bias=False)
        nn.init.zeros_(self.output.weight)
        self.refinement_enabled = True

    def forward(self, query, key, value, valid):
        with torch.autocast(device_type=query.device.type, enabled=False):
            q, k, v = query.float(), key.float(), value.float()
            first_weight = masked_softmax((q[:, :, None] * k).sum(-1) / math.sqrt(32), valid)
            first_read = (first_weight[..., None] * v).sum(2)
            if not self.refinement_enabled:
                return first_read
            # q already contains cell-position and the A2 goal/status condition.
            # The first image read changes the second query. Neither query is a value.
            second_query = q + self.query_update(torch.cat([q, first_read], -1))
            second_weight = masked_softmax((second_query[:, :, None] * k).sum(-1) / math.sqrt(32), valid)
            second_read = (second_weight[..., None] * v).sum(2)
            return first_read + self.output(second_read)

    def assert_zero_output(self):
        if bool(self.output.weight.detach().count_nonzero()):
            raise ValueError('QREFINE output projection must begin at zero')


class SceneExtensionModel(SharedDynamicsMotionDriveV2):
    VALID_ARMS = {**SharedDynamicsMotionDriveV2.VALID_ARMS, SIDE: 0, QREFINE: 0}

    def __init__(self, config, *, arm):
        if arm not in ARMS:
            raise ValueError('Unknown scene extension arm')
        super().__init__(config, arm=arm)
        self.side_scene_enabled = arm == SIDE
        self._side_scene_context = None
        if config.n_history != 4 or config.scene_attention_channels != 32 or config.channels != 128:
            raise ValueError('Scene extension requires the frozen A2 configuration')
        if arm == QREFINE:
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(2026091806)
                self.scene_encoder.add_module('evidence_attention', ReadRefineEvidence())
            self.scene_encoder.evidence_attention.assert_zero_output()
        if self.shared_status_feature_conditioner is not None or self.progress_head is not None:
            raise ValueError('Scene experiments must preserve query-only A2 direct planning')

    def forward(self, *args, side_scene_images=None, **kwargs):
        if self.arm == SIDE:
            if side_scene_images is None or tuple(side_scene_images.shape[1:]) != (4, 3, 216, 384):
                raise ValueError('SIDE-SCENE requires four 384x216 historical images')
        elif side_scene_images is not None:
            raise ValueError('QREFINE must retain the BASE observations')
        if self._side_scene_context is not None:
            raise RuntimeError('Stale or reentrant side scene input')
        self._side_scene_context = side_scene_images
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._side_scene_context = None

    def forward_parts(self, images, history_images, lidar2img, history_transforms,
                      time_offsets, goal_xy, motion_current=None, motion_history=None):
        if not self.side_scene_enabled:
            return super().forward_parts(images, history_images, lidar2img, history_transforms,
                                         time_offsets, goal_xy, motion_current, motion_history)
        if motion_current is None or motion_history is None or self._provided_status_context is None:
            raise ValueError('Missing native motion canvas or provided-status context')
        b = len(images)
        current_levels, current_p4 = self.backbone_fpn(images.flatten(0, 1))
        current_levels = tuple(x.reshape(b, 6, *x.shape[1:]) for x in current_levels)
        current_p4 = current_p4.reshape(b, 6, *current_p4.shape[1:])
        scene_small = F.interpolate(images[:, 0], size=history_images.shape[-2:],
                                    mode='bilinear', align_corners=False, antialias=True)
        scene_stack = torch.cat([scene_small[:, None], history_images], 1)
        front_levels, _ = self.backbone_fpn(scene_stack.flatten(0, 1))
        history_levels = tuple(x.reshape(b, 5, *x.shape[1:])[:, 1:] for x in front_levels)
        motion_stack = torch.cat([motion_current[:, None], motion_history], 1)
        motion_levels, _ = self.backbone_fpn(motion_stack.flatten(0, 1))
        motion_levels = tuple(x.reshape(b, 5, *x.shape[1:]) for x in motion_levels)
        motion = self.motion_encoder(tuple(x[:, 0] for x in motion_levels),
                                     tuple(x[:, 1:] for x in motion_levels), time_offsets)
        side_levels, _ = self.backbone_fpn(self._side_scene_context.flatten(0, 1))
        side_levels = tuple(x.reshape(b, 4, *x.shape[1:]) for x in side_levels)
        history_levels = tuple(torch.cat([front, side], 1) for front, side in zip(history_levels, side_levels))
        scene = self.scene_encoder(current_levels, history_levels, current_p4, lidar2img,
            history_transforms, time_offsets, goal_xy, images.shape[-2:],
            history_camera_ids=(0, 0, 0, 0) + SIDE_CAMERA_IDS,
            history_pose_indices=(0, 1, 2, 3) + SIDE_POSE_INDICES)
        return {**scene, **motion}
