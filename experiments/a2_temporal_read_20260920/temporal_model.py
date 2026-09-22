"""A2: each decoded waypoint reads unpooled, image-derived temporal memory."""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/NHNHOME/data/sukim/adcl')
sys.path.insert(0, str(ROOT / 'experiments/a2_motion_fresh_20260919'))
from motion_model import ExperimentModel, FRESH
from models.motiondrive_v2.motion_encoder import MotionEncoder, local_correlation
from models.motiondrive_v2.planner import DirectTrajectoryPlanner

ARM = 'A2-TEMPORAL-READ'
NEW_PREFIX = 'planner.temporal_read.'


class TemporalReadMotionEncoder(MotionEncoder):
    """Original motion computation; additionally return its pre-pooling tokens.

    The original forward is kept algebraically and in operation order. No new
    parameters, conditioning, detach, image pass, or normalization enters it.
    """

    def forward(self, current_front_levels, history_front_levels, time_offsets):
        b, t = time_offsets.shape
        features = []
        ref_hw = history_front_levels[0].shape[-2:]
        for i, (current, history) in enumerate(zip(current_front_levels, history_front_levels)):
            _, _, c, h, w = history.shape
            cur = self.projections[i](current)
            cur = F.interpolate(cur, (h, w), mode='bilinear', align_corners=False)
            cur = cur[:, None].expand(-1, t, -1, -1, -1).reshape(b * t, -1, h, w)
            past = self.projections[i](history.reshape(b * t, c, h, w))
            with torch.autocast(device_type=cur.device.type, enabled=False):
                corr = local_correlation(cur, past, self.config.correlation_radius)
            pair = torch.cat([corr.to(cur.dtype), cur, past], 1)
            features.append(F.interpolate(pair, ref_hw, mode='bilinear', align_corners=False))
        pair_maps = self.correlation_fuse(torch.cat(features, 1))
        pair_maps = F.adaptive_avg_pool2d(pair_maps, self.config.motion_grid)
        pair_tokens = pair_maps.flatten(2).transpose(1, 2).reshape(b, t, -1, self.config.channels)
        dt = time_offsets.clamp_min(1e-3)
        time_feature = self.time_embed(torch.stack([dt, dt.log()], -1))
        pair_tokens = pair_tokens + self.position(self.positions)[None, None] + time_feature[:, :, None]
        with torch.autocast(device_type=pair_tokens.device.type, enabled=False):
            weights = self.time_attention(pair_tokens.float()).softmax(1)
        motion = (pair_tokens * weights.to(pair_tokens.dtype)).sum(1)
        motion = motion + self.token_refine(motion)
        history_map = pair_tokens.reshape(b * t, *self.config.motion_grid, self.config.channels)
        history_vec = F.adaptive_avg_pool2d(history_map.permute(0, 3, 1, 2), (3, 4)).flatten(1)
        motion_map = motion.reshape(b, *self.config.motion_grid, self.config.channels).permute(0, 3, 1, 2)
        state_vec = F.adaptive_avg_pool2d(motion_map, (3, 4)).flatten(1)
        with torch.autocast(device_type=motion.device.type, enabled=False):
            history_hat = self.history_head(history_vec.float()).reshape(b, t, 4) * self.history_scale
            state_hat = self.state_head(state_vec.float()) * self.state_scale
            history_logvar = self.history_uncertainty_head(history_vec.float()).reshape(b, t, 4).clamp(-8, 8)
            state_logvar = self.state_uncertainty_head(state_vec.float()).clamp(-8, 8)
        return {'motion_features': motion, 'history_hat': history_hat, 'state_hat': state_hat,
                'history_logvar': history_logvar, 'state_logvar': state_logvar,
                'motion_pair_features': pair_tokens}


class WaypointTemporalRead(nn.Module):
    """Read [time, spatial-site] visual tokens separately for six output queries."""

    def __init__(self, channels=128, heads=4):
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=0.0, batch_first=True)
        # Exactly ONE zero output projection. Q/K/V remain nonzero initialized.
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, decoded, pair_features):
        if decoded.ndim != 3 or decoded.shape[1:] != (6, 128):
            raise ValueError('Temporal read requires six 128D decoded waypoint features')
        # The attention runs over the flattened [time x site] memory, so the time
        # count need not be four; H6 feeds six. The spatial grid and channel width
        # stay strict because those would change the read's meaning, not its length.
        if (pair_features.ndim != 4 or pair_features.shape[0] != decoded.shape[0]
                or pair_features.shape[2:] != (192, 128) or pair_features.shape[1] < 4):
            raise ValueError('Temporal memory must retain time-labelled 12x16x128 grids, '
                             'at least four of them')
        with torch.autocast(device_type=decoded.device.type, enabled=False):
            q = self.query_norm(decoded.float())
            memory = self.memory_norm(pair_features.float().flatten(1, 2))
            correction, _ = self.attention(q, memory, memory, need_weights=False)
        return correction


class TemporalReadPlanner(DirectTrajectoryPlanner):
    def __init__(self, config):
        super().__init__(config)
        self.temporal_read = WaypointTemporalRead(config.channels, config.planner_heads)

    def forward(self, scene_features, motion_features, predicted_state, predicted_history,
                motion_pair_features):
        b = scene_features.shape[0]
        with torch.autocast(device_type=scene_features.device.type, enabled=False):
            scene = scene_features.float() + self.scene_position(self.scene_xy.float())[None]
            motion = motion_features.float() + self.motion_type.float()
            state = predicted_state.float() / self.state_scale
            history = (predicted_history.float() / self.history_scale).flatten(1)
            status = torch.cat([state, history], -1)
            if not self.config.state_on:
                status = torch.zeros_like(status)
            state_token = self.state_projection(status)[:, None]
            memory = torch.cat([scene, motion, state_token], 1)
            queries = self.waypoint_queries.float()[None].expand(b, -1, -1)
            decoded = self.decoder(queries, memory)
            decoded = decoded + self.temporal_read(decoded, motion_pair_features)
            normalized_xy = self.xy_head(decoded.float())
            return normalized_xy * normalized_xy.new_tensor(self.config.plan_output_scale)


class TemporalReadModel(ExperimentModel):
    VALID_ARMS = {**ExperimentModel.VALID_ARMS, ARM: 0}

    def __init__(self, config, *, arm=ARM):
        if arm != ARM:
            raise ValueError(arm)
        super().__init__(config, arm=FRESH)
        if (config.n_history != 4 or tuple(config.motion_grid) != (12, 16)
                or config.channels != 128 or config.planner_heads != 4):
            raise ValueError('Frozen FRESH geometry required')
        # Preserve the old constructor RNG so the later correlation rebuild and
        # data stream start from exactly the FRESH control state.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(2026092001)
            motion = TemporalReadMotionEncoder(config)
            planner = TemporalReadPlanner(config)
        motion.load_state_dict(self.motion_encoder.state_dict(), strict=True)
        merged = planner.state_dict()
        old = self.planner.state_dict()
        if set(merged).difference(old) != {
                'temporal_read.' + k for k in planner.temporal_read.state_dict()}:
            raise ValueError('Unexpected planner replacement state')
        merged.update(old)
        planner.load_state_dict(merged, strict=True)
        self.motion_encoder, self.planner = motion, planner
        self.temporal_read_arm = ARM
        if self.shared_status_feature_conditioner is not None or self.progress_head is not None:
            raise ValueError('Only the A2 query-only direct graph is supported')

    def plan_from_features(self, scene_features, motion_features, predicted_state,
                           predicted_history, motion_pair_features):
        return self.planner(scene_features, motion_features, predicted_state,
                            predicted_history, motion_pair_features)

    def forward(self, images, history_images, lidar2img, history_transforms,
                time_offsets, goal_xy, motion_current=None, motion_history=None,
                provided_status5=None):
        if provided_status5 is None:
            raise ValueError('A2 requires its existing provided_status5 scene query condition')
        if self._provided_status_context is not None or self._decoded_capture:
            raise RuntimeError('Reentrant or stale A2 forward')
        self._provided_status_context = provided_status5
        try:
            parts = self.forward_parts(images, history_images, lidar2img,
                                       history_transforms, time_offsets, goal_xy,
                                       motion_current, motion_history)
            plan = self.plan_from_features(parts['scene_features'], parts['motion_features'],
                parts['state_hat'], parts['history_hat'], parts['motion_pair_features'])
            return {**parts, 'plan_abs': plan}
        finally:
            self._provided_status_context = None
            self._decoded_capture.clear()


def load_fresh_parent(model, state):
    """Load every control tensor, keeping only the explicitly named new module."""
    merged = model.state_dict()
    extras = sorted(k for k in merged if k.startswith(NEW_PREFIX))
    if not extras or set(merged).difference(extras) != set(state):
        raise ValueError('Parent/child keys differ beyond the declared temporal read')
    for name, value in state.items():
        if merged[name].shape != value.shape:
            raise ValueError('Parent tensor shape changed: ' + name)
    merged.update(state)
    model.load_state_dict(merged, strict=True)
    return extras
