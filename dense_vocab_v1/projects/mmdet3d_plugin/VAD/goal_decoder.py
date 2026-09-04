"""goal-conditioned waypoint decoder — C0-T의 핵심.

컴플라이언스 설계 (8/18 공지 3-1항)
---------------------------------
    "goal은 참조 지시 정보이며, 궤적 도출 계산의 근거가 될 수 없다."

그래서 이 모듈은 **goal이 값을 만들지 못하게 구조로 막는다.**

    goal + command  ->  waypoint query        (어느 시각 증거를 읽을지 결정)
    temporal BEV    ->  key / value           (궤적 값의 유일한 출처)
    cross-attention ->  waypoint 6개

핵심 불변식: 출력은 `Linear(attn_out)` 이고 `attn_out`은 **value의 볼록결합**이다.
value는 BEV visual token만으로 만든다. query는 attention weight만 바꾼다.
즉 `goal`이 궤적 값에 들어가는 경로가 **없다**. query residual을 출력에 더하지 않는
것이 이 불변식의 전부이므로, 표준 transformer decoder layer를 쓰지 않고 직접 쓴다.

금지한 것 (형님 §10)
    goal -> MLP -> trajectory
    goal + visual feature 단순 concat -> 큰 MLP
    goal trajectory prior + image residual
    goal/command query residual -> output

`scripts/etri_goal_pathcheck.py`가 이 불변식을 실증한다: BEV를 상수로 만들면
goal을 어떻게 흔들어도 출력이 **정확히** 같아야 한다.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def goal_features(goal_xy, x_scale=60.0, y_scale=30.0, dist_scale=60.0):
    """[B, 2] -> [B, 5]  = (x/xs, y/ys, dist/ds, sin(bearing), cos(bearing)).

    미래 goal yaw는 쓰지 않는다 (형님 §10). bearing은 goal 방향각이다.
    scale은 pc_range(60 x 30 m)에서 왔다 -- 정규화 상수일 뿐 학습 대상이 아니다.
    """
    x = goal_xy[..., 0]
    y = goal_xy[..., 1]
    dist = torch.sqrt(x * x + y * y + 1e-6)
    bearing = torch.atan2(y, x)
    return torch.stack([x / x_scale, y / y_scale, dist / dist_scale,
                        torch.sin(bearing), torch.cos(bearing)], dim=-1)


class _CrossAttnBlock(nn.Module):
    """query는 attention weight만 만든다. 출력은 value의 결합뿐이다."""

    def __init__(self, dims, n_head, drop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dims, n_head, dropout=drop)
        self.norm1 = nn.LayerNorm(dims)
        self.ffn = nn.Sequential(nn.Linear(dims, dims * 2), nn.ReLU(inplace=True),
                                 nn.Linear(dims * 2, dims))
        self.norm2 = nn.LayerNorm(dims)

    def forward(self, q, kv, h):
        # q: [T, B, C] (goal/command 조건), kv: [HW, B, C] (BEV), h: [T, B, C] or None
        query = q if h is None else q + h
        a, _ = self.attn(query=query, key=kv, value=kv)
        # ★ a 에만 residual을 준다. q 를 더하면 goal이 출력 경로로 새어 들어간다.
        h = a if h is None else self.norm1(h + a)
        return self.norm2(h + self.ffn(h))


class GoalWaypointDecoder(nn.Module):
    """goal+command로 waypoint query를 만들고 BEV에서 값을 읽는다.

    Args:
        embed_dims: BEV 채널 (VAD tiny 256)
        fut_ts: waypoint 수 (6 = 3초 @ 0.5초)
        n_layers: cross-attention 층 수
        cmd_dim: command one-hot 차원 (3)
        pos_scale: 출력 스케일 (m). tanh 없이 선형이지만 초기 크기를 잡아준다.
    """

    def __init__(self, embed_dims=256, fut_ts=6, n_head=8, n_layers=2,
                 cmd_dim=3, goal_dim=5, x_scale=60.0, y_scale=30.0):
        super().__init__()
        self.fut_ts = fut_ts
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.ts_embed = nn.Embedding(fut_ts, embed_dims)
        self.cond_mlp = nn.Sequential(
            nn.Linear(goal_dim + cmd_dim, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims))
        self.blocks = nn.ModuleList(
            [_CrossAttnBlock(embed_dims, n_head) for _ in range(n_layers)])
        self.out = nn.Linear(embed_dims, 2)
        self.bev_pos = None          # 필요하면 외부에서 넣는다
        self._init()

    def _init(self):
        nn.init.normal_(self.ts_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 출력은 작게 시작한다 (증분 단위 m). 초기에 큰 궤적을 내면 loss가 폭발한다.
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, bev, goal_xy, cmd_onehot, bev_pos=None):
        """
        bev:        [B, HW, C] 또는 [HW, B, C]  -- temporal BEV feature
        goal_xy:    [B, 2]
        cmd_onehot: [B, cmd_dim]
        returns:    [B, fut_ts, 2]  **증분** (cumsum 전)
        """
        if bev.dim() != 3:
            raise ValueError(f"bev must be 3D, got {tuple(bev.shape)}")
        B = goal_xy.shape[0]
        # [B, HW, C] -> [HW, B, C]
        kv = bev.permute(1, 0, 2) if bev.shape[0] == B else bev
        if bev_pos is not None:
            kv = kv + bev_pos
        g = goal_features(goal_xy, self.x_scale, self.y_scale, self.x_scale)
        cond = self.cond_mlp(torch.cat([g, cmd_onehot], dim=-1))     # [B, C]
        q = self.ts_embed.weight.unsqueeze(1) + cond.unsqueeze(0)    # [T, B, C]
        h = None
        for blk in self.blocks:
            h = blk(q, kv, h)
        d = self.out(h).permute(1, 0, 2)                             # [B, T, 2]
        return d


class ImageEvidenceGoalWaypointDecoder(nn.Module):
    """C0-T v2a: goal-addressed reads from current camera content only.

    ``bev_key`` may contain geometry, position, ego shift, temporal state, and
    other addressing information.  None of it is a value.  The only value is
    ``image_value``, tapped before the BEV encoder residual/projection/norm.
    There is deliberately no query/key residual or FFN after attention.
    """

    requires_image_evidence = True

    def __init__(self, embed_dims=256, fut_ts=6, n_head=8, cmd_dim=3,
                 goal_dim=5, x_scale=60.0, y_scale=30.0, disable_goal=False,
                 query_norm=False):
        super().__init__()
        self.fut_ts = fut_ts
        self.x_scale = x_scale
        self.y_scale = y_scale
        # A/B 대조군용. goal feature를 **0으로 대체**한다 (입력 차원을 줄이지 않는다).
        # 차원을 줄이면 `cond_mlp.0`의 shape가 달라져 같은 seed로도 init이 갈리고,
        # "goal만 다르다"가 성립하지 않는다. 여기서는 파라미터 shape와 초기값이
        # goal 판과 완전히 동일하고 goal이 나르는 **정보량만** 0이다.
        self.disable_goal = bool(disable_goal)
        self.ts_embed = nn.Embedding(fut_ts, embed_dims)
        self.cond_mlp = nn.Sequential(
            nn.Linear(goal_dim + cmd_dim, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims))
        # Q에만 걸리는 정규화. attention logit이 포화하지 않게 query 크기를 묶는다.
        # `V=0 => Z=0` 불변식과 무관하다 -- 출력은 여전히 value의 결합뿐이고
        # Q는 attention weight만 만든다. affine을 허용해도 zero-evidence 응답은 0이다.
        self.query_norm = (nn.LayerNorm(embed_dims) if query_norm else None)
        self.value_norm = nn.LayerNorm(
            embed_dims, elementwise_affine=False)
        # bias=False covers Q/K/V projections and attention output projection.
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, n_head, dropout=0.0, bias=False)
        self.out = nn.Linear(embed_dims, 2)
        self._init()

    def _init(self):
        nn.init.normal_(self.ts_embed.weight, std=0.02)
        for module in self.cond_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _sequence_first(tensor, batch_size, name):
        if tensor.dim() != 3:
            raise ValueError(f'{name} must be 3D, got {tuple(tensor.shape)}')
        return tensor.permute(1, 0, 2) if tensor.shape[0] == batch_size else tensor

    def forward(self, bev_key, image_value, goal_xy, cmd_onehot,
                bev_pos=None):
        """Return six per-step increments, before the loss-side cumsum."""
        batch_size = goal_xy.shape[0]
        key = self._sequence_first(bev_key, batch_size, 'bev_key')
        value = self._sequence_first(image_value, batch_size, 'image_value')
        if key.shape != value.shape:
            raise ValueError(
                f'key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}')
        if bev_pos is not None:
            key = key + bev_pos

        features = goal_features(
            goal_xy, self.x_scale, self.y_scale, self.x_scale)
        if self.disable_goal:
            features = torch.zeros_like(features)
        condition = self.cond_mlp(
            torch.cat([features, cmd_onehot], dim=-1))
        query = self.ts_embed.weight.unsqueeze(1) + condition.unsqueeze(0)
        if self.query_norm is not None:
            query = self.query_norm(query)

        # LN has no affine term and every projection in this path has no bias,
        # so image_value == 0 implies attention_output == 0 exactly.
        value = self.value_norm(value)
        attended, _ = self.cross_attn(
            query=query, key=key, value=value, need_weights=False)

        # Final algebraic safety: even if a future edit gives `out` a bias,
        # the zero-evidence response is cancelled rather than becoming a path.
        trajectory = self.out(attended) - self.out(torch.zeros_like(attended))
        return trajectory.permute(1, 0, 2)


class DenseVocabularyDecoder(nn.Module):
    """Select one fixed train-only trajectory using visual scene evidence.

    Candidate coordinates never depend on goal, command, or ego status. Goal and
    command only address camera-derived values and therefore only change scores.
    Vocabulary index zero must be the exact stopped trajectory; all-zero image
    evidence then produces tied logits and deterministically selects index zero.
    """

    requires_image_evidence = True

    def __init__(self, anchors_path, embed_dims=256, fut_ts=6, n_head=8,
                 cmd_dim=3, goal_dim=5, x_scale=60.0, y_scale=30.0,
                 loss_weight=1.0, target_temperature=0.25):
        super().__init__()
        if not os.path.isfile(anchors_path):
            raise FileNotFoundError(anchors_path)
        anchors_abs = np.load(anchors_path).astype(np.float32)
        if anchors_abs.ndim != 3 or anchors_abs.shape[1:] != (fut_ts, 2):
            raise ValueError(
                f'anchors must have shape [K,{fut_ts},2], got {anchors_abs.shape}')
        if not np.array_equal(anchors_abs[0], np.zeros((fut_ts, 2), np.float32)):
            raise ValueError('anchor index zero must be the exact stopped trajectory')

        anchors_inc = np.diff(
            np.concatenate([np.zeros_like(anchors_abs[:, :1]), anchors_abs], axis=1),
            axis=1)
        self.register_buffer('anchors_abs', torch.from_numpy(anchors_abs))
        self.register_buffer('anchors_inc', torch.from_numpy(anchors_inc))
        self.fut_ts = fut_ts
        self.loss_weight = float(loss_weight)
        if target_temperature <= 0:
            raise ValueError('target_temperature must be positive')
        self.target_temperature = float(target_temperature)
        self.x_scale = x_scale
        self.y_scale = y_scale

        self.ts_embed = nn.Embedding(fut_ts, embed_dims)
        self.cond_mlp = nn.Sequential(
            nn.Linear(goal_dim + cmd_dim, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims))
        self.value_norm = nn.LayerNorm(embed_dims, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, n_head, dropout=0.0, bias=False)
        self.scene_proj = nn.Linear(fut_ts * embed_dims, embed_dims, bias=False)
        self.anchor_encoder = nn.Sequential(
            nn.Linear(fut_ts * 2, embed_dims, bias=False),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims, bias=False))
        self.logit_scale = embed_dims ** -0.5
        self._init()

    def _init(self):
        nn.init.normal_(self.ts_embed.weight, std=0.02)
        for module in self.cond_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.scene_proj.weight)
        for module in self.anchor_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    @staticmethod
    def _sequence_first(tensor, batch_size, name):
        if tensor.dim() != 3:
            raise ValueError(f'{name} must be 3D, got {tuple(tensor.shape)}')
        return tensor.permute(1, 0, 2) if tensor.shape[0] == batch_size else tensor

    def forward(self, bev_key, image_value, goal_xy, cmd_onehot,
                bev_pos=None):
        batch_size = goal_xy.shape[0]
        key = self._sequence_first(bev_key, batch_size, 'bev_key')
        value = self._sequence_first(image_value, batch_size, 'image_value')
        if key.shape != value.shape:
            raise ValueError(
                f'key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}')
        if bev_pos is not None:
            key = key + bev_pos

        features = goal_features(
            goal_xy, self.x_scale, self.y_scale, self.x_scale)
        condition = self.cond_mlp(torch.cat([features, cmd_onehot], dim=-1))
        query = self.ts_embed.weight.unsqueeze(1) + condition.unsqueeze(0)
        attended, _ = self.cross_attn(
            query=query, key=key, value=self.value_norm(value),
            need_weights=False)
        scene = self.scene_proj(
            attended.permute(1, 0, 2).reshape(batch_size, -1))
        candidate = self.anchor_encoder(
            self.anchors_abs.reshape(self.anchors_abs.shape[0], -1))
        logits = torch.matmul(scene, candidate.t()) * self.logit_scale
        selected_index = logits.argmax(dim=-1)
        selected = self.anchors_inc.index_select(0, selected_index)
        return selected, logits, selected_index

    def vocabulary_loss(self, logits, ego_fut_gt, ego_fut_masks):
        """Metric-aligned soft ranking supervision over the fixed vocabulary."""
        gt_abs = ego_fut_gt.cumsum(dim=-2)
        distance = torch.linalg.vector_norm(
            self.anchors_abs.unsqueeze(0) - gt_abs.unsqueeze(1), dim=-1)
        weights = distance.new_tensor([11, 11, 5, 5, 2, 2]) / 36.0
        valid = ego_fut_masks.to(distance.dtype)
        weighted = (distance * weights.view(1, 1, -1) * valid.unsqueeze(1)).sum(-1)
        target = weighted.argmin(dim=-1)
        # Near-equivalent candidates should not be treated as equally wrong as
        # distant trajectories merely because their K-means indices differ.
        target_prob = F.softmax(-weighted / self.target_temperature, dim=-1)
        loss = -(target_prob * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        loss = loss * self.loss_weight
        selected = logits.argmax(dim=-1)
        return {
            'loss_plan_vocab': loss,
            'plan_vocab_oracle_l2': weighted.gather(1, target[:, None]).mean().detach(),
            'plan_vocab_selected_l2': weighted.gather(1, selected[:, None]).mean().detach(),
            'plan_vocab_top1': selected.eq(target).float().mean().detach(),
        }
