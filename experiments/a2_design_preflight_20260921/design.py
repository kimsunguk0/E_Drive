"""Isolated design checks only. No training launcher or optimizer step."""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SplitReadPlanner(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.config = copy.deepcopy(parent.config)
        for name, module in parent.named_children():
            if name not in ('temporal_read', 'xy_head'):
                self.add_module(name, copy.deepcopy(module))
        for name, parameter in parent.named_parameters(recurse=False):
            self.register_parameter(name, nn.Parameter(parameter.detach().clone(), requires_grad=parameter.requires_grad))
        for name, buffer in parent.named_buffers(recurse=False):
            self.register_buffer(name, buffer.detach().clone(), persistent=name not in parent._non_persistent_buffers_set)
        self.length_read = copy.deepcopy(parent.temporal_read)
        self.heading_read = copy.deepcopy(parent.temporal_read)
        # Constructing the two new Linear objects must not consume the training RNG.
        with torch.random.fork_rng(devices=[]):
            self.length_head = self._copy_head(parent.xy_head, 0)
            self.heading_head = self._copy_head(parent.xy_head, 1)

    @staticmethod
    def _copy_head(head, channel):
        assert len(head) == 4 and isinstance(head[-1], nn.Linear) and head[-1].out_features == 2
        final = nn.Linear(head[-1].in_features, 1, bias=True,
                          device=head[-1].weight.device, dtype=head[-1].weight.dtype)
        with torch.no_grad():
            final.weight.copy_(head[-1].weight[channel:channel+1])
            final.bias.copy_(head[-1].bias[channel:channel+1])
        return nn.Sequential(*(copy.deepcopy(m) for m in list(head.children())[:-1]), final)

    def raw_logits(self, scene_features, motion_features, predicted_state,
                   predicted_history, motion_pair_features):
        b = scene_features.shape[0]
        with torch.autocast(device_type=scene_features.device.type, enabled=False):
            scene = scene_features.float() + self.scene_position(self.scene_xy.float())[None]
            motion = motion_features.float() + self.motion_type.float()
            numeric = torch.cat([predicted_state.float()/self.state_scale,
                                 (predicted_history.float()/self.history_scale).flatten(1)], -1)
            if not self.config.state_on:
                numeric = torch.zeros_like(numeric)
            memory = torch.cat([scene, motion, self.state_projection(numeric)[:, None]], 1)
            decoded = self.decoder(self.waypoint_queries.float()[None].expand(b, -1, -1), memory)
            # TemporalRead returns a residual, NOT the complete decoded feature.
            length = self.length_head(decoded + self.length_read(decoded, motion_pair_features))
            heading = self.heading_head(decoded + self.heading_read(decoded, motion_pair_features))
            return torch.cat([length, heading], -1)

    def forward(self, *args):
        from progress_model import compose_progress
        return compose_progress(self.raw_logits(*args), self.progress_units)


GEOMETRY_POLICY = {
    'schema_version': 3,
    'radius_m': 4.0,
    'near_distance_m': 0.05,
    'point_agreement_m': 1e-4,
    'axis_disagreement_degrees': 15.0,
    'exact_distance_tie_m': 1e-9,
    'minimum_segment_m': 1e-6,
    'axis': 'cos(2theta),sin(2theta); exact-distance ties average agreeing axes',
    'masks': 'offset and axis ambiguity handled independently',
    'candidate_grouping': 'nearest point per original polyline; adjacent segments are not separate competing lines',
    'agreement': 'all candidate pairs, independent of the first/argmin candidate',
}


def _pairwise_disagreement(values, active, *, distance_limit=None, dot_floor=None):
    """A first-candidate comparison is order-dependent with three or more ties."""
    bad = np.zeros(len(values), bool)
    for row in np.flatnonzero(active.sum(-1) > 1):
        selected = values[row, active[row]]
        if distance_limit is not None:
            bad[row] = (np.linalg.norm(selected[:, None]-selected[None], axis=-1) > distance_limit).any()
        else:
            bad[row] = (selected @ selected.T < dot_floor).any()
    return bad


def geometry_targets(polylines, centers, original_valid, policy=GEOMETRY_POLICY):
    """Continuous nearest-segment targets; annotations only, never ego future."""
    centers = np.asarray(centers, np.float64)
    shape = centers.shape[:-1]
    x = centers.reshape(-1, 2)
    original_valid = np.asarray(original_valid, bool).reshape(-1)
    groups = []
    radius = float(policy['radius_m'])
    lo, hi = x.min(0)-radius, x.max(0)+radius
    for points in polylines:
        points = np.asarray(points, np.float64)[..., :2]
        segments = []
        for a, b in zip(points[:-1], points[1:]):
            if not np.isfinite([a, b]).all() or np.linalg.norm(b-a) <= policy['minimum_segment_m']:
                continue
            if np.any(np.maximum(a, b) < lo) or np.any(np.minimum(a, b) > hi):
                continue
            segments.append((a, b))
        if segments:
            groups.append(np.asarray(segments))
    offset = np.zeros((len(x), 2), np.float64)
    axis = np.zeros_like(offset)
    om = np.zeros(len(x), bool)
    am = np.zeros_like(om)
    distance = np.full(len(x), np.inf)
    if not groups:
        return dict(offset=offset.reshape(*shape, 2).astype(np.float32),
                    axis=axis.reshape(*shape, 2).astype(np.float32),
                    offset_valid=om.reshape(shape), axis_valid=am.reshape(shape),
                    distance=distance.reshape(shape), segments=0)
    angle_cos = math.cos(math.radians(2*policy['axis_disagreement_degrees']))
    for start in range(0, len(x), 128):
        stop = min(start+128, len(x))
        xx = x[start:stop]
        line_q, line_d, line_axis, line_point_ambiguous, line_axis_ambiguous = [], [], [], [], []
        for seg in groups:
            a, v = seg[:, 0], seg[:, 1]-seg[:, 0]
            ll = (v*v).sum(-1)
            unit = v/np.sqrt(ll)[:, None]
            segment_axes = np.stack([unit[:, 0]**2-unit[:, 1]**2, 2*unit[:, 0]*unit[:, 1]], -1)
            t = np.clip(((xx[:, None]-a)*v).sum(-1)/ll, 0, 1)
            qs = a[None]+t[..., None]*v[None]
            ds = np.linalg.norm(qs-xx[:, None], axis=-1)
            ii = ds.argmin(-1)
            dl = ds[np.arange(len(xx)), ii]
            exact_segment = np.abs(ds-dl[:, None]) <= policy['exact_distance_tie_m']
            ql = (qs*exact_segment[..., None]).sum(1)/exact_segment.sum(1)[:, None]
            al = (segment_axes[None]*exact_segment[..., None]).sum(1)/exact_segment.sum(1)[:, None]
            point_bad = _pairwise_disagreement(qs, exact_segment, distance_limit=policy['point_agreement_m'])
            axis_bad = _pairwise_disagreement(np.broadcast_to(segment_axes, qs.shape), exact_segment, dot_floor=angle_cos)
            al /= np.maximum(np.linalg.norm(al,axis=-1,keepdims=True),1e-8)
            line_q.append(ql); line_d.append(dl); line_axis.append(al)
            line_point_ambiguous.append(point_bad); line_axis_ambiguous.append(axis_bad)
        q = np.stack(line_q, 1); dd = np.stack(line_d, 1); axes = np.stack(line_axis, 1)
        internal_point_bad = np.stack(line_point_ambiguous, 1)
        internal_axis_bad = np.stack(line_axis_ambiguous, 1)
        index = dd.argmin(-1)
        d = dd[np.arange(len(xx)), index]
        near = dd <= d[:, None]+policy['near_distance_m']
        exact = np.abs(dd-d[:, None]) <= policy['exact_distance_tie_m']
        point_ambiguous = _pairwise_disagreement(q, near, distance_limit=policy['point_agreement_m']) | (near & internal_point_bad).any(-1)
        axis_ambiguous = _pairwise_disagreement(axes, near, dot_floor=angle_cos) | (near & internal_axis_bad).any(-1)
        # Stable under polyline order/reversal at duplicate points or corners.
        q_target = (q*exact[..., None]).sum(1)/exact.sum(1)[:, None]
        axis_target = (axes*exact[..., None]).sum(1)/exact.sum(1)[:, None]
        axis_norm = np.linalg.norm(axis_target, axis=-1)
        support = original_valid[start:stop] & (d <= radius)
        om[start:stop] = support & ~point_ambiguous
        am[start:stop] = support & ~axis_ambiguous & (axis_norm > 1e-8)
        offset[start:stop] = q_target-xx
        axis[start:stop] = axis_target/np.maximum(axis_norm[:, None], 1e-8)
        distance[start:stop] = d
    offset[~om] = 0
    axis[~am] = 0
    return dict(offset=offset.reshape(*shape, 2).astype(np.float32),
                axis=axis.reshape(*shape, 2).astype(np.float32),
                offset_valid=om.reshape(shape), axis_valid=am.reshape(shape),
                distance=distance.reshape(shape), segments=sum(len(g) for g in groups))


def geometry_normalizers(batch):
    return {key: batch[key+'_valid'].bool().sum().detach()*2 for key in ('offset', 'axis')}


def geometry_loss(prediction, batch, normalizers):
    """Prediction Bx4xXxY; offsets normalized by 4m, axes unitless."""
    values = []
    for key, channel, scale in [('offset', slice(0, 2), 4.), ('axis', slice(2, 4), 1.)]:
        mask = batch[key+'_valid'].bool()[:, None]
        target = torch.where(mask, batch[key].float()/scale, torch.zeros_like(batch[key].float()))
        error = F.smooth_l1_loss(prediction[:, channel].float(), target, reduction='none', beta=1.)
        values.append(torch.where(mask, error, torch.zeros_like(error)).sum()/normalizers[key].clamp_min(1))
    return sum(values)


class LaneGeometryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 4, 1))

    def forward(self, raster):
        with torch.autocast(device_type=raster.device.type, enabled=False):
            return self.net(raster.float())
