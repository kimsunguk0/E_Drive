"""Existing motion operations plus explicit same-forward pre-pool map return."""
import torch
from torch.nn import functional as F
from models.motiondrive_v2.motion_encoder import MotionEncoder, local_correlation

class FineMotionEncoder(MotionEncoder):
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
        pre_pool_map = pair_maps
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
                'motion_pair_features': pair_tokens, 'motion_pair_map': pre_pool_map}

