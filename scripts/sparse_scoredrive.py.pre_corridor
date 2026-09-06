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
from torchvision.models import resnet34, resnet50


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
    """Shared six-camera ResNet trunk with a compact 128-channel FPN.

    arch='resnet34' (⑤-B 지연 기준) 또는 'resnet50' (dense VAD img_backbone 이식용).
    """

    def __init__(self, channels: int = 128, arch: str = "resnet34",
                 use_p1: bool = False) -> None:
        super().__init__()
        if arch == "resnet50":
            net = resnet50(weights=None)
            c1, c2, c3, c4 = 256, 512, 1024, 2048
        else:
            net = resnet34(weights=None)
            c1, c2, c3, c4 = 64, 128, 256, 512
        self.use_p1 = bool(use_p1)
        self.arch = arch
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.lat2 = nn.Conv2d(c2, channels, 1, bias=False)
        self.lat3 = nn.Conv2d(c3, channels, 1, bias=False)
        self.lat4 = nn.Conv2d(c4, channels, 1, bias=False)
        self.out2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        if self.use_p1:   # stride 4 (후보 판별력 69%->89% 측정치)
            self.lat1 = nn.Conv2d(c1, channels, 1, bias=False)
            self.out1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, images: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        x = self.stem(images)
        x = self.layer1(x)   # stride 4
        c2 = self.layer2(x)  # stride 8
        c3 = self.layer3(c2)  # stride 16
        c4 = self.layer4(c3)  # stride 32
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        levels = [self.out2(p2), self.out3(p3)]
        if self.use_p1:
            p1 = self.lat1(x) + F.interpolate(p2, size=x.shape[-2:], mode="nearest")
            levels.insert(0, self.out1(p1))
        return tuple(levels), p4


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
        feature_norm: bool = False,
        logit_norm: bool = False,
        arch: str = "resnet34",
        occ_aux: bool = False,
        offset_sample: bool = False,
        use_p1: bool = False,
        n_hist: int = 0,
        speed_head: bool = False,
        speed_gamma: float = 0.0,
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
        self.backbone_fpn = ResNet34FPN128(channels, arch=arch, use_p1=use_p1)
        off = 0.75   # feature cell 단위
        self._offsets = (((0.0, 0.0),) if not offset_sample else
                         ((0.0, 0.0), (off, 0.0), (-off, 0.0),
                          (0.0, off), (0.0, -off)))
        self.n_off = len(self._offsets)
        if self.n_off > 1:
            self.offset_proj = nn.Linear(channels * self.n_off, channels,
                                         bias=False)
        # 보조 perception supervision: 이미 샘플된 waypoint feature 로
        # '이 지면 위치가 t 시점에 점유되는가'를 예측한다. 학습 전용 head 이고
        # 배포 forward 의 반환에는 절대 들어가지 않는다(추론 시 제거 가능).
        self.occ_aux = bool(occ_aux)
        if self.occ_aux:
            self.occ_head = nn.Sequential(
                nn.Linear(channels, channels // 2), nn.ReLU(inplace=True),
                nn.Linear(channels // 2, 1))
        self._occ_logits = None      # train-only stash
        # ⑤-D temporal: 과거 n_hist 시점의 같은 후보 위치 표본을 설계 §6.4 대로
        # [f_now, f_hist, f_now-f_hist, f_now*f_hist] 로 묶어 bias-free MLP 에 넣는다.
        # 상대 pose 는 좌표 정렬 행렬로만 쓰이고 여기 feature 로 들어가지 않는다.
        self.n_hist = int(n_hist)
        if self.n_hist:
            self.temporal_mlp = nn.Sequential(
                nn.Linear(channels * (1 + 3 * self.n_hist), channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels, channels, bias=False))
        self.visual_proj = nn.Linear(channels, channels, bias=False)
        self.kinematic_proj = nn.Sequential(
            nn.Linear(6, channels, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channels, channels, bias=False),
        )
        self.local_score = nn.Linear(channels, 1, bias=False)
        self.scene_proj = nn.Linear(channels, channels, bias=False)
        # Phase-5C: sampled/scene feature 정규화. dense decoder 의 value_norm 과 같은 역할.
        # elementwise_affine=False 라 학습 파라미터가 0개 → state_dict/param count 불변.
        # 없으면 frozen-BN backbone 에서 logit scale 이 발산한다(실측 gn>3000).
        # Phase-5C: logit scale 을 코사인+학습가능 온도로 묶는다.
        # 없으면 |logits| 가 무한정 커져 1024-way softmax 가 포화되고(실측 std 11~18)
        # 학습이 step~3k 에서 플래토 후 fp16 오버플로우로 NaN 이 된다.
        self.logit_norm = bool(logit_norm)
        if self.logit_norm:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        self.feature_norm = bool(feature_norm)
        if self.feature_norm:
            self.value_norm = nn.LayerNorm(channels, elementwise_affine=False)
            self.scene_norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.heights = tuple(float(x) for x in heights)
        self.score_top = int(score_top)
        self.n_out = int(n_out)
        self.nms_pool = int(nms_pool)
        if (self.score_top, self.n_out, self.nms_pool) != (3, 12, 64):
            raise ValueError("Phase-5B canonical shortlist must be score3+nms9 from top64")

        # ⑤-E3: 후보별 3초/5초 누적 이동거리(고정 bank 속성).
        _a = torch.from_numpy(abs5)
        _inc = torch.diff(_a, dim=1, prepend=torch.zeros_like(_a[:, :1]))
        _step = torch.linalg.vector_norm(_inc, dim=-1)
        self.register_buffer("cand_s3", _step[:, :6].sum(-1), persistent=True)
        self.register_buffer("cand_s5", _step.sum(-1), persistent=True)
        # 영상에서 자차 진행량을 회귀한다(설계 §4.5 권장 auxiliary label).
        # 입력은 영상 융합 feature 뿐이고 ego status 는 들어가지 않는다.
        self.speed_head_on = bool(speed_head)
        # speed_gamma <= 0 이면 logit 감점을 아예 끄고 보조 회귀만 한다.
        # 무작위 초기화된 head 의 난수 예측으로 감점하면 path_score([-2,2])가
        # 통째로 덮여 학습이 붕괴한다(실측 slO 10.8). 2단계로 학습해야 한다:
        #   A) gamma=0 으로 head 만 학습  B) 그 체크포인트에서 gamma 켜기
        self.speed_pen_on = bool(speed_head) and float(speed_gamma) > 0.0
        if self.speed_head_on:
            self.speed_head = nn.Sequential(
                nn.Linear(channels, channels), nn.ReLU(inplace=True),
                nn.Linear(channels, 8))          # [S3, S5, r1..r6]
            g0 = max(float(speed_gamma), 1e-3)
            inv = math.log(math.expm1(g0))       # softplus^-1
            self.gamma_raw = nn.Parameter(torch.tensor(inv, dtype=torch.float32))
        self._speed_pred = None      # [B,8] stash (학습/평가 진단용)

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

    def _sample_level(self, feature: Tensor, grid: Tensor, visible: Tensor) -> Tuple[Tensor, Tensor]:
        """설계 §6.3: projection 지점 + 주변 4 offset 표본을 채널로 concat.

        측정치(measure_aliasing): offset 없이 stride8 에서 후보 구분 가능률 69.3%,
        4 offset 을 더하면 85.3%. 평균이 아니라 concat 이라 이웃 정보가 뭉개지지 않는다.
        """
        b, cams, channels, fh, fw = feature.shape
        _, _, k, t, nh, _ = grid.shape
        flat_feature = feature.reshape(b * cams, channels, fh, fw)
        outs = []
        for ox, oy in self._offsets:
            g = grid
            if ox or oy:
                g = grid + grid.new_tensor([2.0 * ox / fw, 2.0 * oy / fh])
            flat_grid = g.reshape(b * cams, k * t * nh, 1, 2).to(feature.dtype)
            sm = F.grid_sample(
                flat_feature, flat_grid, mode="bilinear", padding_mode="zeros",
                align_corners=False,
            )
            outs.append(sm.reshape(b, cams, channels, k, t, nh)
                        .permute(0, 1, 3, 4, 5, 2))
        sampled = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
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
        """최종 logits. 성분 분해는 score_components 가 담당한다(수식 단일 출처)."""
        return self.combine_logits(*self.score_components(
            levels, p4, lidar2img, image_hw, evidence))

    def combine_logits(self, path_score, g_term, ev):
        """성분 -> 최종 logits. current-only 와 temporal 경로가 공유한다.

        speed head 가 켜져 있으면 예측 진행량과 후보 진행량의 불일치를 감점한다.
        예측은 영상에서만 나오고 goal/status 와 무관하므로 goal counterfactual
        불변이 유지된다. 외부에 노출되는 것은 여전히 후보 12개와 logits 뿐이다.
        """
        if self.speed_pen_on and self._speed_pred is not None:
            s3 = self._speed_pred[:, 0] * 30.0                  # [B] m
            pen = (self.cand_s3.unsqueeze(0) - s3.unsqueeze(1)).abs() / 10.0
            path_score = path_score - F.softplus(self.gamma_raw) * pen
        if self.logit_norm:
            logits = self.logit_scale.exp().clamp(max=100.0) * (path_score + g_term)
        else:
            logits = path_score + g_term
        return logits * ev

    def sample_candidate_features(self, levels, lidar2img, image_hw):
        """후보 waypoint 위치의 이미지 표본 [B,K,T,C] 와 가시성 [B,K,T].
        temporal 확장에서 시점별로 각각 호출한다(과거는 정렬된 lidar2img 를 넘긴다)."""
        grid, visible = project_candidate_points(
            self.candidate_abs, lidar2img, image_hw, self.heights)
        sampled_levels = []
        valid = None
        for level in levels:
            sm, lv = self._sample_level(level, grid, visible)
            sampled_levels.append(sm)
            valid = lv if valid is None else (valid | lv)
        sampled = torch.stack(sampled_levels, dim=0).mean(dim=0)
        if self.n_off > 1:
            sampled = self.offset_proj(sampled)
        return sampled, valid

    def fuse_temporal(self, f_now, f_hist):
        """설계 §6.4 visual motion branch. f_hist = [n_hist] 개 [B,K,T,C]."""
        parts = [f_now]
        mul = getattr(self, "fuse_mul", True)
        for fh in f_hist:
            parts += [fh, f_now - fh]
            parts.append(f_now * fh if mul else torch.zeros_like(fh))
        return self.temporal_mlp(torch.cat(parts, dim=-1))

    def score_components(
        self,
        levels: Tuple[Tensor, Tensor],
        p4: Tensor,
        lidar2img: Tensor,
        image_hw: Tuple[int, int],
        evidence: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """(path_score, g_term, evidence_mask). g_term 은 이미 0.25 및 스케일이 적용된
        global 기여분이라 호출부에서 단순 덧셈만 하면 된다. ablation 진단용."""
        sampled, valid = self.sample_candidate_features(levels, lidar2img, image_hw)
        return self.score_from_sampled(sampled, valid, p4, evidence)

    def score_from_sampled(self, sampled, valid, p4, evidence):
        """표본이 주어진 뒤의 점수화. temporal 경로는 융합된 표본을 여기로 넘긴다."""
        if self.speed_head_on:
            # 후보·시점 평균 = 장면 수준 서술자. temporal 경로에서는 과거 정보가
            # 이미 융합돼 있어 진행량 추정에 필요한 신호를 담는다.
            self._speed_pred = self.speed_head(sampled.mean(dim=(1, 2)))
        if self.feature_norm:
            sampled = self.value_norm(sampled)
        if self.occ_aux and self.training:
            # [B,K,T] 점유 logit. 학습 손실에서만 읽고 forward 반환에는 없다.
            self._occ_logits = self.occ_head(sampled).squeeze(-1)
        kin = self.kinematic_proj(self.kinematic_desc)
        visual = self.visual_proj(sampled)
        if self.logit_norm:
            # 코사인 유사도(유계) + tanh 국소항 -> path_score in [-2,2]
            vn = F.normalize(visual, dim=-1)
            kn = F.normalize(kin, dim=-1)
            compatibility = (vn * kn[None]).sum(dim=-1)
            compatibility = compatibility + torch.tanh(
                self.local_score(F.normalize(sampled, dim=-1)).squeeze(-1))
        else:
            compatibility = (visual * kin[None]).sum(dim=-1) / math.sqrt(visual.shape[-1])
            compatibility = compatibility + self.local_score(sampled).squeeze(-1)
        valid_f = valid.to(compatibility.dtype)
        path_score = (compatibility * valid_f).sum(dim=-1) / valid_f.sum(dim=-1).clamp_min(1.0)

        scene = p4.mean(dim=(1, 3, 4))
        if self.feature_norm:
            scene = self.scene_norm(scene)
        scene = self.scene_proj(scene)
        candidate_profile = kin.mean(dim=1)
        if self.logit_norm:
            global_score = torch.einsum(
                "bd,kd->bk", F.normalize(scene, dim=-1),
                F.normalize(candidate_profile, dim=-1))
            g_term = 0.25 * global_score
        else:
            global_score = torch.einsum("bd,kd->bk", scene, candidate_profile)
            g_term = 0.25 * global_score / math.sqrt(scene.shape[-1])
        # Exact C4 invariant: fixed geometry cannot produce a nonzero score when
        # the visual tensor contains no evidence, even after future training.
        return path_score, g_term, evidence[:, None].to(path_score.dtype)

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

    def temporal_logits(self, images, img_hist, lidar2img, hist_T):
        """⑤-D: 현재 + 과거 n_hist 시점을 shared backbone 으로 처리.

        images [B,6,3,H,W], img_hist [B,n_hist,6,3,Hh,Wh],
        lidar2img [B,6,4,4] (현재), hist_T [B,n_hist,4,4] (현재->과거 ego 강체변환).

        과거 카메라 투영 = lidar2img @ hist_T[k] 로, 현재 좌표계의 후보를 과거 시점
        카메라에 정렬한다(설계 §3.3: 고정 행렬 곱, feature/token 입력 아님).
        해상도가 같으면 현재+과거를 한 번의 batch 로 encode 한다(7-frame 순차 아님).
        """
        b, nh = img_hist.shape[0], img_hist.shape[1]
        hw_now = (int(images.shape[-2]), int(images.shape[-1]))
        hw_his = (int(img_hist.shape[-2]), int(img_hist.shape[-1]))
        evidence = images.abs().amax(dim=(1, 2, 3, 4)) > 0
        flat_h = img_hist.reshape(b * nh, *img_hist.shape[2:])
        if hw_now == hw_his and getattr(self, "merge_encode", False):
            merged = torch.cat([images, flat_h], dim=0)      # [(B + B*nh),6,3,H,W]
            lv, p4a = self.encode_images(merged)
            levels_now = tuple(x[:b] for x in lv)
            levels_his = tuple(x[b:] for x in lv)
            p4 = p4a[:b]
        else:
            levels_now, p4 = self.encode_images(images)
            levels_his, _ = self.encode_images(flat_h)
        # 기하 기준은 항상 lidar2img 가 만들어진 768x432 픽셀 공간(hw_now)이다.
        # grid_sample 은 [-1,1] 을 feature map 전체 범위로 매핑하므로 과거 프레임을
        # 저해상도로 넣어도 정규화 기준을 바꾸면 안 된다(바꾸면 좌표가 배율만큼 틀린다).
        f_now, valid = self.sample_candidate_features(levels_now, lidar2img, hw_now)
        l2i_h = torch.einsum("bcij,bkjm->bkcim",
                             lidar2img, hist_T)               # [B,nh,6,4,4]
        f_hist = []
        for k in range(nh):
            lk = tuple(x.reshape(b, nh, *x.shape[1:])[:, k] for x in levels_his)
            fk, vk = self.sample_candidate_features(lk, l2i_h[:, k], hw_now)
            f_hist.append(fk)
            valid = valid | vk
        fused = self.fuse_temporal(f_now, f_hist)
        return self.combine_logits(
            *self.score_from_sampled(fused, valid, p4, evidence))

    def forward_temporal(self, images, img_hist, lidar2img, hist_T):
        """⑤-D 배포 API. 완성 후보 12개 + visual logits 만 반환하고 full-K 는 숨긴다.
        goal/command/status 는 어떤 인자에도 없다."""
        return self.build_candidates(
            self.temporal_logits(images, img_hist, lidar2img, hist_T))

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
