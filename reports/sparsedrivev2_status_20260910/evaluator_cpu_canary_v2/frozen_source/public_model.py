"""SparseDriveV2 NAVSIM-v1 weights adapted to immutable ETRI candidate selection.

Architecture/source: swc-17/SparseDriveV2 @ 696ef77924eb9e0a4b4047d013a50e9854bfa026.
No NAVSIM installation is needed. Names/shapes match all public checkpoint tensors.
Inputs are current RGB images normalized by ImageNet mean/std, projected in pixel
coordinates. Status only conditions candidate scores; candidate coordinates are
frozen bank tensors. No ground truth is accepted by forward().
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PUBLIC_SHA256 = "330072b133981f77b0b18b669216ad50b211552a82ea45ac52d507ec9021a735"
PUBLIC_PREFIX = "agent._sparsedrive_model."
METRICS_V1 = ("no_at_fault_collisions", "drivable_area_compliance",
              "driving_direction_compliance", "time_to_collision_within_bound",
              "comfort", "ego_progress")
_NATIVE = None


def load_native_extension(verbose=False):
    """Compile official CUDA op into a task-local directory, never site-packages.

    The only source change specifies PyTorch's current stream for both launches.
    Original pinned sources remain intact; emitted copies can be audited.
    """
    global _NATIVE
    if _NATIVE is not None:
        return _NATIVE
    from torch.utils.cpp_extension import load
    base = Path(__file__).resolve().parents[2]
    vendor = base / "third_party/SparseDriveV2"
    src = vendor / "navsim/agents/sparsedrive/ops/src"
    build = vendor / ".etri_extension" / ("torch_" + torch.__version__.replace("+", "_"))
    build.mkdir(parents=True, exist_ok=True)
    sources = []
    for name in ("deformable_aggregation.cpp", "deformable_aggregation_cuda.cu"):
        content = (src / name).read_text()
        if name.endswith(".cu"):
            content = content.replace(")), 512>>>(", ")), 512, 0, at::cuda::getCurrentCUDAStream()>>>(")
        dest = build / name
        if not dest.exists() or dest.read_text() != content:
            dest.write_text(content)
        sources.append(str(dest))
    os.environ.setdefault("MAX_JOBS", "2")
    _NATIVE = load(name="etri_sparsedrive_dfa", sources=sources,
                   build_directory=str(build), verbose=verbose,
                   extra_cuda_cflags=["-O3"], extra_cflags=["-O3"])
    return _NATIVE


class _NativeAggregation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, shapes, starts, locations, weights):
        ext = load_native_extension()
        args = (features.contiguous().float(), shapes.contiguous().int(),
                starts.contiguous().int(), locations.contiguous().float(),
                weights.contiguous().float())
        ctx.save_for_backward(*args)
        return ext.deformable_aggregation_forward(*args)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        f, shapes, starts, xy, weights = ctx.saved_tensors
        gf, gxy, gw = torch.zeros_like(f), torch.zeros_like(xy), torch.zeros_like(weights)
        load_native_extension().deformable_aggregation_backward(
            f, shapes, starts, xy, weights, grad.contiguous(), gf, gxy, gw)
        return gf, None, None, gxy, gw


def _format_features(feature_maps):
    shapes = torch.tensor([f.shape[-2:] for f in feature_maps], device=feature_maps[0].device)
    starts = torch.cat([shapes.new_zeros(1), shapes.prod(-1).cumsum(0)[:-1]])
    flat = torch.cat([f.flatten(-2).transpose(-1, -2) for f in feature_maps], -2)
    return flat, shapes, starts


def _grid_aggregate(feature_maps, locations, weights):
    """Reference for native DFA; align_corners=False and strict (0,1) bounds.

    locations [B,A,P,Cam,2], weights [B,A,Cam,L,P,G]. Chunk anchors to
    avoid materializing every camera/scale sampled feature simultaneously.
    """
    bs, anchors, points, cams, _ = locations.shape
    dim, groups = feature_maps[0].shape[2], weights.shape[-1]
    outputs = []
    for first in range(0, anchors, 8):
        xy = locations[:, first:first + 8]
        count = xy.shape[1]
        valid = ((xy > 0) & (xy < 1)).all(-1)
        grid = (xy.permute(0, 3, 1, 2, 4).reshape(bs*cams, count, points, 2) * 2 - 1)
        value = feature_maps[0].new_zeros(bs, count, dim)
        for level, feature in enumerate(feature_maps):
            sampled = F.grid_sample(feature.flatten(0, 1), grid, align_corners=False)
            sampled = sampled.reshape(bs, cams, groups, dim//groups, count, points)
            sampled = sampled.permute(0, 4, 1, 5, 2, 3)
            w = weights[:, first:first+8, :, level] * valid.permute(0, 1, 3, 2)[..., None]
            value = value + (sampled * w[..., None]).sum((2, 3)).flatten(-2)
        outputs.append(value)
    return torch.cat(outputs, 1)


class SparsePoint3DKeyPointsGenerator(nn.Module):
    def __init__(self, num_sample):
        super().__init__()
        self.num_sample, self.num_learnable_pts = num_sample, 2
        self.fix_height = (0., -.25, -.5, .25, .5)
        self.num_pts = num_sample * len(self.fix_height) * 2
        self.learnable_fc = nn.Linear(256, self.num_pts * 2)

    def forward(self, anchor, feature):
        b, a = feature.shape[:2]
        offset = self.learnable_fc(feature).reshape(b, a, self.num_sample, 5, 2, 2)
        xy = anchor.reshape(b, a, self.num_sample, 2)[..., None, None, :] + offset
        heights = xy.new_tensor(self.fix_height).reshape(1, 1, 1, 5, 1, 1)
        z = heights.expand(*xy.shape[:-1], 1)
        return torch.cat([xy, z], -1).flatten(2, 4)


class DeformableFeatureAggregation(nn.Module):
    def __init__(self, num_sample, backend="native"):
        super().__init__()
        self.backend = backend
        self.kps_generator = SparsePoint3DKeyPointsGenerator(num_sample)
        self.num_pts = self.kps_generator.num_pts
        self.output_proj = nn.Linear(256, 256)
        self.camera_encoder = nn.Sequential(nn.Linear(12, 256), nn.ReLU(inplace=True),
            nn.LayerNorm(256), nn.Linear(256, 256), nn.ReLU(inplace=True), nn.LayerNorm(256))
        self.weights_fc = nn.Linear(256, 8 * 4 * self.num_pts)

    def forward(self, feature, anchor_xy, feature_maps, projection_mat, image_wh, formatted=None):
        # The original native operator and camera projection are float32.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            feature, anchor_xy = feature.float(), anchor_xy.float()
            b, a = feature.shape[:2]
            points = self.kps_generator(anchor_xy, feature)
            homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], -1)
            projected = (projection_mat.float()[:, :, None, None] @ homogeneous[:, None, ..., None]).squeeze(-1)
            depth = projected[..., 2]
            xy = projected[..., :2] / depth[..., None].clamp(min=1e-5)
            xy = xy / image_wh.float()[:, :, None, None]
            visible = (depth > 1e-5) & (xy[..., 0] > 0) & (xy[..., 1] > 0) & (xy[..., 0] < 1) & (xy[..., 1] < 1)
            cam = self.camera_encoder(projection_mat.float()[:, :, :3].reshape(b, 3, 12))
            weights = self.weights_fc(feature[:, :, None] + cam[:, None]).reshape(b, a, 3, 4, self.num_pts, 8)
            mask = visible.permute(0, 2, 1, 3)[..., None, :, None]
            weights = weights.masked_fill((~mask) & (mask.sum(2, keepdim=True) != 0), -torch.inf)
            weights = weights.reshape(b, a, -1, 8).softmax(-2).reshape(b, a, 3, 4, self.num_pts, 8)
            loc = xy.permute(0, 2, 3, 1, 4)
            if self.backend == "native" and feature.is_cuda:
                fmt = formatted if formatted is not None else _format_features(feature_maps)
                w = weights.permute(0, 1, 4, 2, 3, 5).reshape(b, a*self.num_pts, 3, 4, 8)
                values = _NativeAggregation.apply(*fmt, loc.reshape(b, a*self.num_pts, 3, 2), w)
                values = values.reshape(b, a, self.num_pts, 256).sum(2)
            else:
                values = _grid_aggregate([f.float() for f in feature_maps], loc, weights)
            return feature + self.output_proj(values)


class SparseBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        from torchvision.ops import FeaturePyramidNetwork
        self.img_backbone = timm.create_model("resnet34", pretrained=False, features_only=True, out_indices=(1, 2, 3, 4))
        self.img_neck = FeaturePyramidNetwork([64, 128, 256, 512], 256)

    def forward(self, images):
        b, cams = images.shape[:2]
        feats = self.img_backbone(images.flatten(0, 1))
        feats = self.img_neck(OrderedDict((str(i), f) for i, f in enumerate(feats)))
        return [f.reshape(b, cams, *f.shape[1:]) for f in feats.values()]


def _mlp(out_dim):
    return nn.Sequential(nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, out_dim))


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, final, backend):
        super().__init__()
        self.p_deform_model = DeformableFeatureAggregation(50, backend)
        self.v_img_attention = nn.MultiheadAttention(256, 8, batch_first=True)
        for prefix in ("p", "v") + (("t",) if final else ()):
            setattr(self, prefix + "_attention", nn.MultiheadAttention(256, 8, batch_first=True))
            setattr(self, prefix + "_ffn", _mlp(256))
            for i in (1, 2):
                setattr(self, prefix + f"_norm{i}", nn.LayerNorm(256))
                setattr(self, prefix + f"_dropout{i}", nn.Dropout(.1))
        self.path_mlp, self.vel_mlp = _mlp(1), _mlp(1)
        if final:
            self.t_deform_model = DeformableFeatureAggregation(8, backend)
            self.traj_mlp = _mlp(1)
            self.metric_heads = nn.ModuleDict({name: _mlp(1) for name in METRICS_V1})

    def attend(self, value, prefix):
        attended = getattr(self, prefix + "_attention")(value, value, value, need_weights=False)[0]
        value = getattr(self, prefix + "_norm1")(value + getattr(self, prefix + "_dropout1")(attended))
        return getattr(self, prefix + "_norm2")(value + getattr(self, prefix + "_dropout2")(getattr(self, prefix + "_ffn")(value)))


class TrajectoryHead(nn.Module):
    def __init__(self, bank, backend):
        super().__init__()
        for key, value in bank.items():
            self.register_buffer(key, value.detach().float().contiguous())
        self.path_pos_embed = nn.Sequential(nn.Linear(150, 1024), nn.ReLU(), nn.Linear(1024, 256))
        self.vel_pos_embed = nn.Sequential(nn.Linear(8, 1024), nn.ReLU(), nn.Linear(1024, 256))
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([CustomTransformerDecoderLayer(False, backend), CustomTransformerDecoderLayer(True, backend)])


class PublicSparseDriveV2(nn.Module):
    """Two-stage path/velocity filtering + candidate deformable reconditioning.

    Default scores are the pretrained imitation head for ETRI L2 fine-tuning.
    ``navsim_v1`` reproduces the public checkpoint's final metric combination.
    Public checkpoint evaluation on ETRI has not established competitive accuracy.
    """
    def __init__(self, bank, backend="native", score_mode="imitation",
                 path_filter=(128, 20), velocity_filter=(64, 10), mask_invalid_candidates=False):
        super().__init__()
        if score_mode not in ("imitation", "navsim_v1"):
            raise ValueError(score_mode)
        self.score_mode = score_mode
        self.mask_invalid_candidates = mask_invalid_candidates
        self.path_filter, self.velocity_filter = tuple(path_filter), tuple(velocity_filter)
        self._backbone = SparseBackbone()
        self._status_encoding = nn.Linear(8, 256)
        self._trajectory_head = TrajectoryHead(bank, backend)
        self._validate_bank()

    def _validate_bank(self):
        h = self._trajectory_head
        p, v = h.path_vocab.shape[0], h.vel_vocab.shape[0]
        assert h.path_vocab.shape == (p, 50, 3)
        assert h.vel_vocab.shape == (v, 8)
        assert h.traj_vocab.shape == (p, v, 8, 3)
        assert h.traj_mask.shape == (p, v, 8)
        assert all(torch.isfinite(x).all() for x in (h.path_vocab, h.vel_vocab, h.traj_vocab, h.traj_mask))

    @classmethod
    def from_public_checkpoint(cls, checkpoint, bank_path=None, **kwargs):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sd = {k.removeprefix(PUBLIC_PREFIX): v for k, v in payload["state_dict"].items() if k.startswith(PUBLIC_PREFIX)}
        bank_keys = ("path_vocab", "vel_vocab", "traj_vocab", "traj_mask")
        bank = {k: sd["_trajectory_head." + k] for k in bank_keys}
        if bank_path is not None:
            bank = load_etri_bank(bank_path)
        kwargs.setdefault("mask_invalid_candidates", bank_path is not None)
        model = cls(bank=bank, **kwargs)
        destination = model.state_dict()
        reused, excluded, mismatched = {}, [], {}
        for k, value in sd.items():
            if bank_path is not None and k in {"_trajectory_head." + x for x in bank_keys}:
                excluded.append(k)
            elif k not in destination or destination[k].shape != value.shape:
                mismatched[k] = {"public": list(value.shape), "destination": list(destination[k].shape) if k in destination else None}
            else:
                reused[k] = value
        incompatible = model.load_state_dict(reused, strict=False)
        learned_names = set(dict(model.named_parameters()))
        report = {"public_tensor_count": len(sd), "reused_tensor_count": len(reused),
                  "public_numel": sum(x.numel() for x in sd.values()),
                  "reused_numel": sum(x.numel() for x in reused.values()),
                  "learned_parameter_numel": sum(p.numel() for p in model.parameters()),
                  "reused_learned_parameter_numel": sum(v.numel() for k, v in reused.items() if k in learned_names),
                  "intentional_bank_replacements": excluded, "shape_mismatches": mismatched,
                  "missing_keys": list(incompatible.missing_keys), "unexpected_keys": list(incompatible.unexpected_keys),
                  "reused_keys": list(reused), "score_mode": model.score_mode,
                  "bank_shape": {k: list(v.shape) for k, v in bank.items()}}
        if mismatched or set(incompatible.missing_keys) != set(excluded) or incompatible.unexpected_keys:
            raise RuntimeError("Unexpected public checkpoint coverage gap: " + str(report))
        return model, report

    def forward(self, images, lidar2img, image_hw=None, status=None):
        if images.ndim != 5 or images.shape[1:3] != (3, 3):
            raise ValueError("images must be [B,3 cameras,3 channels,H,W]")
        b = images.shape[0]
        if lidar2img.shape != (b, 3, 4, 4):
            raise ValueError("lidar2img must be [B,3,4,4] in resized pixel coordinates")
        if image_hw is None:
            image_hw = images.new_tensor(images.shape[-2:]).expand(b, 2)
        image_hw = torch.as_tensor(image_hw, device=images.device, dtype=torch.float32)
        if image_hw.ndim == 1:
            image_hw = image_hw.expand(b, 2)
        if image_hw.shape == (b, 2):
            image_hw = image_hw[:, None].expand(b, 3, 2)
        if image_hw.shape != (b, 3, 2):
            raise ValueError("image_hw must be [H,W], [B,2], or [B,3,2]")
        image_wh = image_hw.flip(-1)
        if status is None:
            status = images.new_zeros(b, 8)
        if status.shape != (b, 8):
            raise ValueError("status must have 8 entries per batch row")
        encoded_status = self._status_encoding(status.to(images))
        features = self._backbone(images)
        formatted = _format_features(features)
        img_value = features[-1].permute(0, 1, 3, 4, 2).flatten(1, 3)
        h = self._trajectory_head
        path_ids = torch.arange(len(h.path_vocab), device=images.device).expand(b, -1)
        velocity_ids = torch.arange(len(h.vel_vocab), device=images.device).expand(b, -1)
        path = h.path_pos_embed(h.path_vocab.flatten(-2)).expand(b, -1, -1)
        velocity = h.vel_pos_embed(h.vel_vocab).expand(b, -1, -1)
        coarse = []
        for i, layer in enumerate(h.decoder.layers):
            path, velocity = path + encoded_status[:, None], velocity + encoded_status[:, None]
            path = layer.p_deform_model(path, h.path_vocab[path_ids, :, :2], features, lidar2img, image_wh, formatted)
            path = layer.attend(path, "p")
            velocity = velocity + layer.v_img_attention(velocity, img_value, img_value, need_weights=False)[0]
            velocity = layer.attend(velocity, "v")
            p_logits, v_logits = layer.path_mlp(path).squeeze(-1), layer.vel_mlp(velocity).squeeze(-1)
            coarse.append({"path_scores": p_logits, "velocity_scores": v_logits,
                           "path_ids": path_ids, "velocity_ids": velocity_ids})
            def select(scores, values, ids, count):
                if values.shape[1] <= count:
                    return values, ids
                idx = scores.topk(count, 1).indices
                return values.gather(1, idx[..., None].expand(-1, -1, values.shape[-1])), ids.gather(1, idx)
            path, path_ids = select(p_logits, path, path_ids, self.path_filter[i])
            v_select_logits = v_logits
            if self.mask_invalid_candidates:
                eligible = h.traj_mask[path_ids[:, :, None], velocity_ids[:, None, :], :6].bool().all(-1).any(1)
                v_select_logits = v_logits.masked_fill(~eligible, -1e4)
            velocity, velocity_ids = select(v_select_logits, velocity, velocity_ids, self.velocity_filter[i])
        # Coordinates are direct lookups in a fixed, precomposed bank. Status only
        # influenced the indices/scores above; no interpolation/residual is used.
        rows = h.traj_vocab[path_ids[:, :, None], velocity_ids[:, None, :]].flatten(1, 2)
        row_ids = (path_ids[:, :, None] * len(h.vel_vocab) + velocity_ids[:, None, :]).flatten(1, 2)
        traj = (path[:, :, None] + velocity[:, None]).flatten(1, 2)
        traj = layer.t_deform_model(traj, rows[..., :2], features, lidar2img, image_wh, formatted)
        traj = layer.attend(traj, "t")
        imitation = layer.traj_mlp(traj).squeeze(-1)
        metrics = {name: head(traj).squeeze(-1) for name, head in layer.metric_heads.items()}
        navsim = (metrics["no_at_fault_collisions"].sigmoid() * metrics["drivable_area_compliance"].sigmoid()) * (
            5*metrics["time_to_collision_within_bound"].sigmoid() + 5*metrics["ego_progress"].sigmoid() + 2*metrics["comfort"].sigmoid())
        scores = imitation if self.score_mode == "imitation" else navsim
        candidate_valid = h.traj_mask[path_ids[:, :, None], velocity_ids[:, None, :], :6].bool().all(-1).flatten(1, 2)
        if self.mask_invalid_candidates:
            if not candidate_valid.any(1).all():
                raise RuntimeError("Filtering yielded no valid frozen bank candidate")
            scores = scores.masked_fill(~candidate_valid, -1e4)
        else:
            candidate_valid = torch.ones_like(candidate_valid)
        selected = scores.argmax(1)
        candidates = rows[..., :6, :2]
        batch_idx = torch.arange(b, device=images.device)
        return {"trajectory": candidates[batch_idx, selected], "candidate_xy": candidates,
                "candidate_ids": row_ids, "selected_candidate_id": row_ids[batch_idx, selected],
                "candidate_valid": candidate_valid,
                "scores": scores, "imitation_scores": imitation, "navsim_scores": navsim,
                "metric_logits": metrics, "coarse": coarse,
                "path_ids": path_ids, "velocity_ids": velocity_ids}


def load_etri_bank(path):
    """Accept the bank agent's NPZ without online candidate coordinate changes."""
    with np.load(path, allow_pickle=False) as data:
        def field(*names):
            for name in names:
                if name in data:
                    return np.asarray(data[name], dtype=np.float32)
            raise KeyError(f"Expected one of {names}; got {list(data)}")
        p = field("path_xyz", "path_vocab", "path_xy")
        v = field("velocity8", "vel_vocab")
        t = field("traj_xyz8", "traj_xy8", "traj_vocab")
        def heading(x):
            origin = np.zeros_like(x[..., :1, :])
            delta = np.diff(np.concatenate([origin, x], axis=-2), axis=-2)
            return np.concatenate([x, np.arctan2(delta[..., 1:2], delta[..., :1])], -1)
        if p.shape[-1] == 2:
            p = heading(p)
        if t.shape[-1] == 2:
            t = heading(t)
        mask = next((np.asarray(data[k], dtype=np.float32) for k in ("traj_mask8", "mask8", "traj_mask") if k in data), np.ones(t.shape[:-1], dtype=np.float32))
        return {k: torch.from_numpy(np.ascontiguousarray(value)) for k, value in
                dict(path_vocab=p, vel_vocab=v, traj_vocab=t, traj_mask=mask).items()}


SparseDriveV2Public = PublicSparseDriveV2
