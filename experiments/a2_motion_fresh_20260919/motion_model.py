"""Independent A2 experiments: image-only coarse-to-fine motion or fresh init.

The scene remains the registered QREFINE A2. External status only conditions its
shared scene query. Fine matching receives unaligned RGB features exclusively;
neither pose alignment, goal nor provided status participates in this branch.
"""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/NHNHOME/data/sukim/adcl')
sys.path.insert(0, str(ROOT / 'experiments/md_a2_scene_extensions_20260918'))
from scene_extensions import SceneExtensionModel, QREFINE
from models.motiondrive_v2.motion_encoder import local_correlation

FINE = 'A2-C2F-MOTION'
FRESH = 'A2-FRESH-NUIM'
ARMS = (FINE, FRESH)
FINE_PREFIX = 'motion_encoder.fine_match.'


def pixel_grid(height, width, device):
    y, x = torch.meshgrid(torch.arange(height, device=device),
                          torch.arange(width, device=device), indexing='ij')
    return torch.stack((x, y), -1).float()


def coarse_correspondence(current, history, radius=4, temperature=.07):
    """Current pixel x matches history x+(dx,dy), in COARSE feature cells."""
    h, w = current.shape[-2:]
    with torch.autocast(device_type=current.device.type, enabled=False):
        costs = local_correlation(current.float(), history.float(), radius)
        offsets = current.new_tensor([(x, y) for y in range(-radius, radius + 1)
                                     for x in range(-radius, radius + 1)], dtype=torch.float32)
        destination = pixel_grid(h, w, current.device)[None] + offsets[:, None, None]
        valid = ((destination[..., 0] >= 0) & (destination[..., 0] <= w-1)
                 & (destination[..., 1] >= 0) & (destination[..., 1] <= h-1))
        probability = (costs / temperature).masked_fill(~valid[None], -torch.inf).softmax(1)
        flow = torch.einsum('bkhw,kd->bdhw', probability, offsets)
        confidence = probability.amax(1, keepdim=True)
    return flow, confidence


def sample_neighborhood(history, flow, radius=2):
    """Sample history at x+flow(x)+offset, NOT unfold a spatially varying warp.

    Coordinates and grid_sample are FP32 with align_corners=False. The radius is
    expressed in fine FEATURE CELLS. Outside-image evidence is explicitly zero.
    """
    b, c, h, w = history.shape
    with torch.autocast(device_type=history.device.type, enabled=False):
        offsets = history.new_tensor([(x, y) for y in range(-radius, radius+1)
                                     for x in range(-radius, radius+1)], dtype=torch.float32)
        destinations = (pixel_grid(h, w, history.device)[None, None]
                        + flow.float().permute(0, 2, 3, 1)[:, None]
                        + offsets[None, :, None, None])
        valid = ((destinations[..., 0] >= 0) & (destinations[..., 0] <= w-1)
                 & (destinations[..., 1] >= 0) & (destinations[..., 1] <= h-1))
        scale = history.new_tensor((w, h), dtype=torch.float32)
        grid = 2 * (destinations + .5) / scale - 1
        sampled = F.grid_sample(history.float(), grid.reshape(b, -1, w, 2),
                                mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled = sampled.reshape(b, c, len(offsets), h, w)
        sampled = sampled * valid[:, None]
    return sampled, valid


class FineMatching(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.descriptor = nn.Conv2d(256, 32, 1, bias=False)
        # 25 fine costs + two 32D visual descriptors + image-derived dx/dy and
        # confidence + 25 visibility masks. No supplied numerical state.
        self.refine = nn.Sequential(
            nn.Conv2d(117, channels, 3, stride=2, padding=1),
            nn.GroupNorm(1, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels), nn.GELU())
        self.output = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.collect_stats = False
        self.last_stats = None

    def forward(self, current_fine, history_fine, current_coarse, history_coarse,
                coarse_projection):
        b, t, c, h, w = history_fine.shape
        coarse_h, coarse_w = history_coarse.shape[-2:]
        cur_coarse = coarse_projection(current_coarse)[:, None].expand(-1, t, -1, -1, -1)
        cur_coarse = cur_coarse.reshape(b*t, -1, coarse_h, coarse_w)
        past_coarse = coarse_projection(history_coarse.flatten(0, 1))
        flow, confidence = coarse_correspondence(cur_coarse, past_coarse)
        # Native canvas dimensions are divisible by 16. ResNet C1 and the coarse
        # feature are stride4/16, so displacement is multiplied by exactly four.
        if (h, w) != (4*coarse_h, 4*coarse_w):
            raise ValueError('Frozen native geometry requires exact stride16 -> stride4 ratio')
        flow = F.interpolate(flow, (h, w), mode='bilinear', align_corners=False) * 4
        confidence = F.interpolate(confidence, (h, w), mode='bilinear', align_corners=False)
        cur = self.descriptor(current_fine)[:, None].expand(-1, t, -1, -1, -1)
        cur = cur.reshape(b*t, 32, h, w)
        past = self.descriptor(history_fine.reshape(b*t, c, h, w))
        with torch.autocast(device_type=cur.device.type, enabled=False):
            cur_unit = F.normalize(cur.float(), dim=1, eps=1e-6)
            past_unit = F.normalize(past.float(), dim=1, eps=1e-6)
            sampled, valid = sample_neighborhood(past_unit, flow)
            sampled = F.normalize(sampled, dim=1, eps=1e-6)
            costs = (cur_unit[:, :, None] * sampled).sum(1) * valid
            evidence = torch.cat((costs, cur_unit, sampled[:, :, 12],
                                  flow / 16., confidence, valid.float()), 1)
        correction = self.output(self.refine(evidence.to(cur.dtype)))
        if self.collect_stats:
            self.last_stats = {
                'fine_hw': [h, w], 'coarse_hw': [coarse_h, coarse_w],
                'flow_mean_abs_fine_cells': float(flow.detach().abs().mean()),
                'invalid_fraction': float((~valid).float().mean()),
                'correction_rms': float(correction.detach().float().square().mean().sqrt())}
        return correction


class ExperimentModel(SceneExtensionModel):
    VALID_ARMS = {**SceneExtensionModel.VALID_ARMS, FINE: 0, FRESH: 0}

    def __init__(self, config, *, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(config, arm=QREFINE)
        self.experiment_arm = arm
        if arm == FINE:
            if config.backbone_arch != 'resnet50':
                raise ValueError('C1 descriptor contract is ResNet50, 256 channels')
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(2026091907)
                self.motion_encoder.add_module('fine_match', FineMatching(config.channels))

    def forward_parts(self, images, history_images, lidar2img, history_transforms,
                      time_offsets, goal_xy, motion_current=None, motion_history=None):
        if self.experiment_arm == FRESH:
            return super().forward_parts(images, history_images, lidar2img, history_transforms,
                                          time_offsets, goal_xy, motion_current, motion_history)
        if motion_current is None or motion_history is None or self._provided_status_context is None:
            raise ValueError('Missing native visual canvas or A2 status context')
        b, t = time_offsets.shape
        current_levels, current_p4 = self.backbone_fpn(images.flatten(0, 1))
        current_levels = tuple(v.reshape(b, 6, *v.shape[1:]) for v in current_levels)
        current_p4 = current_p4.reshape(b, 6, *current_p4.shape[1:])
        small = F.interpolate(images[:, 0], history_images.shape[-2:], mode='bilinear',
                               align_corners=False, antialias=True)
        history_levels, _ = self.backbone_fpn(torch.cat((small[:, None], history_images), 1).flatten(0, 1))
        history_levels = tuple(v.reshape(b, t+1, *v.shape[1:])[:, 1:] for v in history_levels)
        # Capture C1 from the EXISTING third, native temporal backbone pass.
        captured = []
        handle = self.backbone_fpn.layer1.register_forward_hook(lambda m, args, out: captured.append(out))
        try:
            levels, _ = self.backbone_fpn(torch.cat((motion_current[:, None], motion_history), 1).flatten(0, 1))
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('Native temporal C1 must be captured exactly once')
        c1 = captured.pop()
        fine = c1.reshape(b, t+1, *c1.shape[1:])
        levels = tuple(v.reshape(b, t+1, *v.shape[1:]) for v in levels)
        correction = self.motion_encoder.fine_match(
            fine[:, 0], fine[:, 1:], levels[1][:, 0], levels[1][:, 1:],
            self.motion_encoder.projections[1])
        def add_fine(_module, _args, output):
            if correction.shape != output.shape:
                raise ValueError('Fine refinement and original pair-map geometry differ')
            return output + correction.to(output.dtype)
        handle = self.motion_encoder.correlation_fuse.register_forward_hook(add_fine)
        try:
            motion = self.motion_encoder(tuple(v[:, 0] for v in levels),
                                         tuple(v[:, 1:] for v in levels), time_offsets)
        finally:
            handle.remove()
        scene = self.scene_encoder(current_levels, history_levels, current_p4, lidar2img,
            history_transforms, time_offsets, goal_xy, images.shape[-2:])
        return {**scene, **motion}
