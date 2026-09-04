"""각 후보 궤적이 **자기 웨이포인트 위치의 BEV를 직접 조회**하는 스코어러.

왜 바꾸는가 (실측 근거)
-----------------------
`vocab_decoder.ImageOnlyVocabularyDecoder` 는 조건 없는 시간 쿼리 6개가 BEV 전체를
cross-attend 해 **벡터 하나**로 요약하고, 그것을 K개 anchor 임베딩과 내적한다.
장면의 어느 부분이 어느 후보를 지지하는지 연결할 통로가 없다.

4 epoch 학습 실측 (K=1024, val 사전 oracle 0.095):

    pc_range ±30 : top6_l2 0.963 -> 0.793 -> 0.799
    pc_range ±60 : top6_l2 1.127 -> 0.771 -> 0.687 -> 0.655
    top-1        : 3.75 (oracle 대비 40배)

pc_range 를 2배로 늘려도 사실상 차이가 없었다. 병목은 커버리지가 아니라
**장면과 후보를 잇는 구조**다.

이 디코더는 anchor k 의 T개 웨이포인트를 BEV 격자 좌표로 변환해 `grid_sample` 로
그 지점의 영상 증거를 직접 읽는다. "이 경로로 가면 거기에 무엇이 있나"를 후보마다
따로 본다. VADv2 / Hydra-MDP / DiffusionDrive 계열이 쓰는 방식이다.

규정 준수 (변함 없음)
---------------------
* anchor 좌표는 **train 으로 만든 고정 상수**다. goal 과 무관하고 학습으로 변하지 않는다.
* `forward` 는 goal/command/ego status 를 인자로 받지 않는다.
* 모든 투영이 bias 없음 -> image_value == 0 이면 표본이 전부 0, anchor 별 특징이
  동일해져 logits 가 정확히 동률이 되고 `argmax` 가 index 0(정지 궤적)을 반환한다.
* BEV 밖 웨이포인트는 `padding_mode='zeros'` 로 0 을 읽는다 -- 증거가 없는 곳에서
  증거를 지어내지 않는다.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHALLENGE_W = (11.0, 11.0, 5.0, 5.0, 2.0, 2.0)


class AnchorGroundedVocabularyDecoder(nn.Module):
    """후보별 BEV 조회 + 전역 장면 요약으로 고정 사전에 점수를 매긴다.

    Args:
        anchors_path: `[K, fut_ts, 2]` 누적 좌표. index 0 = 정지.
        pc_range: BEV 가 덮는 범위 `[x0, y0, z0, x1, y1, z1]`.
        bev_h, bev_w: BEV 격자. **bev_w <-> x, bev_h <-> y** 다
            (`modules/encoder.get_reference_points` 실측: xs 가 W 에서 나온다).
        use_global: 전역 장면 항을 더한다. 속도처럼 국소 조회로 안 잡히는 단서를 준다.
    """

    requires_image_evidence = True
    uses_condition = False          # goal/cmd 를 받지 않는다. head 가 이걸 본다.

    def __init__(self, anchors_path, pc_range, bev_h, bev_w,
                 embed_dims=256, fut_ts=6, n_head=8, loss_weight=1.0,
                 target_temperature=0.1, top_m=6, use_global=True,
                 loss_mode='softce',
                 w_pos=1.0, w_rank=0.5, w_exp=0.25,
                 eps_abs=0.05, tau_score=0.1,
                 m0=0.2, beta=0.5, dmax=1.0, neg_topk=32,
                 tau_exp=0.1, gt_near=16, rank_neg_margin=0.10,
                 residual=False, res_beta=1.0, kin_hidden=128,
                 use_timing=False, use_stop=False,
                 timing_tau=0.1, timing_k=16, timing_delta=2.0,
                 stop_thresh=0.5, stop_weight=1.0):
        super().__init__()
        if not os.path.isfile(anchors_path):
            raise FileNotFoundError(anchors_path)
        a = np.load(anchors_path).astype(np.float32)
        if a.ndim != 3 or a.shape[1:] != (fut_ts, 2):
            raise ValueError(f'anchors must be [K,{fut_ts},2], got {a.shape}')
        if not np.array_equal(a[0], np.zeros((fut_ts, 2), np.float32)):
            raise ValueError('anchor index 0 must be the exact stopped trajectory')

        inc = np.diff(np.concatenate([np.zeros_like(a[:, :1]), a], 1), axis=1)
        self.register_buffer('anchors_abs', torch.from_numpy(a))
        self.register_buffer('anchors_inc', torch.from_numpy(inc))
        self.register_buffer(
            'challenge_w',
            torch.tensor(CHALLENGE_W, dtype=torch.float32) / sum(CHALLENGE_W))

        # 웨이포인트 -> grid_sample 정규좌표 [-1, 1]. x 는 W 축, y 는 H 축.
        x0, y0, _, x1, y1, _ = [float(v) for v in pc_range]
        gx = (a[..., 0] - x0) / (x1 - x0) * 2.0 - 1.0
        gy = (a[..., 1] - y0) / (y1 - y0) * 2.0 - 1.0
        # grid_sample 의 grid 는 (..., 2) = (x_norm, y_norm) 순서다.
        self.register_buffer('grid', torch.from_numpy(
            np.stack([gx, gy], -1).astype(np.float32)).reshape(1, -1, 1, 2))
        inside = ((np.abs(gx) <= 1) & (np.abs(gy) <= 1)).mean()
        self._inside_ratio = float(inside)

        self.bev_h, self.bev_w = int(bev_h), int(bev_w)
        self.fut_ts, self.num_anchors = fut_ts, a.shape[0]
        self.embed_dims = embed_dims
        self.top_m = int(top_m)
        self.loss_weight = float(loss_weight)
        if target_temperature <= 0:
            raise ValueError('target_temperature must be positive')
        self.target_temperature = float(target_temperature)
        self.use_global = bool(use_global)

        # --- Phase A ranking-loss tournament (ETRI_SCOREDRIVE_DESIGN.md §8) ---
        # loss_mode 로만 학습 목적을 바꾼다. 진단 지표(top-1/oracle@M/gap)는
        # 어느 mode 에서도 동일하게 찍어 판끼리 사과-대-사과로 비교한다.
        valid_modes = ('softce', 'pos', 'pos_rank', 'pos_rank_exp',
                       'pos_exp', 'softce_exp')
        if loss_mode not in valid_modes:
            raise ValueError(f'loss_mode must be one of {valid_modes}, got {loss_mode}')
        self.loss_mode = loss_mode
        self.w_pos = float(w_pos)
        self.w_rank = float(w_rank)
        self.w_exp = float(w_exp)
        self.eps_abs = float(eps_abs)              # positive set 반경 (m)
        self.tau_score = float(tau_score)          # L_pos/rank score softmax 온도
        self.m0 = float(m0)                        # ranking base margin
        self.beta = float(beta)                    # margin 의 D 비례 계수
        self.dmax = float(dmax)                    # margin clamp 상한 (m)
        self.neg_topk = int(neg_topk)              # visual hard-negative pool
        self.tau_exp = float(tau_exp)              # expected-L2 softmax 온도
        self.gt_near = int(gt_near)               # L_exp 후보에 넣을 GT 최근접 수
        self.rank_neg_margin = float(rank_neg_margin)  # negative D 하한 (Dmin+이 값)

        self.value_norm = nn.LayerNorm(embed_dims, elementwise_affine=False)
        # 국소 경로: 후보의 T개 표본 -> 후보 특징 -> 점수. 전부 bias 없음.
        self.local_mlp = nn.Sequential(
            nn.Linear(fut_ts * embed_dims, embed_dims, bias=False),
            nn.LayerNorm(embed_dims, elementwise_affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims, bias=False))
        self.local_score = nn.Linear(embed_dims, 1, bias=False)
        # 전역 경로: 기존 구조와 동일한 조건 없는 요약.
        if self.use_global:
            self.ts_embed = nn.Embedding(fut_ts, embed_dims)
            self.cross_attn = nn.MultiheadAttention(
                embed_dims, n_head, dropout=0.0, bias=False)
            self.scene_proj = nn.Linear(fut_ts * embed_dims, embed_dims, bias=False)
        self.logit_scale = embed_dims ** -0.5

        # --- Phase C: image-only velocity/stop residual (§Phase C) ---
        # base scorer(champion) 는 freeze, residual 만 학습. zero-init 라 시작 시
        # final_logit == base_logit (champion 재현). goal/status 미입력 유지.
        self.residual = bool(residual)
        self.res_beta = float(res_beta)
        self.use_timing = bool(use_timing)
        self.use_stop = bool(use_stop)
        self.timing_tau = float(timing_tau)
        self.timing_k = int(timing_k)
        self.timing_delta = float(timing_delta)
        self.stop_thresh = float(stop_thresh)
        self.stop_weight = float(stop_weight)
        self._cache_scene_stop = None
        if self.residual:
            kin = self._candidate_kinematics(a)              # [K, kin_dim] numpy
            km, ks = kin.mean(0), kin.std(0) + 1e-6
            self.register_buffer('vel_kin', torch.from_numpy(
                ((kin - km) / ks).astype(np.float32)))         # 표준화 [K,kin_dim]
            kd = kin.shape[1]
            # 후보 kinematics -> motion 임베딩 (bias 허용: 영상 아님, 상수 후보특성)
            self.vel_kin_mlp = nn.Sequential(
                nn.Linear(kd, kin_hidden), nn.ReLU(inplace=True),
                nn.Linear(kin_hidden, embed_dims))
            # 장면 motion 투영 (bias 없음 -> image_value=0 이면 0 -> 규정 준수 유지)
            self.vel_scene_proj = nn.Linear(embed_dims, embed_dims, bias=False)
            if self.use_stop:
                self.stop_head = nn.Linear(embed_dims, 1)      # 장면 stop/move (aux only)

        self._init()
        if self.residual:
            nn.init.zeros_(self.vel_scene_proj.weight)        # zero-init -> residual=0 시작

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        if self.use_global:
            nn.init.normal_(self.ts_embed.weight, std=0.02)

    @staticmethod
    def _candidate_kinematics(a):
        """후보 궤적 자체에서 고정 kinematic descriptor 계산 (raw ego status 아님).
        a: [K, T, 2] 누적 좌표. 반환 [K, kin_dim]."""
        T = a.shape[1]
        steps = np.diff(np.concatenate([np.zeros_like(a[:, :1]), a], 1), axis=1)  # [K,T,2]
        ds = np.linalg.norm(steps, axis=-1)                  # [K,T] 0.5초 이동거리
        dds = np.diff(ds, axis=1)                            # [K,T-1] 속도변화(가감속)
        cums = np.cumsum(ds, axis=1)                         # 누적 진행거리
        prog = cums[:, [min(1, T-1), min(3, T-1), T-1]]      # 1s,2s,3s 누적
        last_speed = ds[:, -1:]                              # 마지막 구간 속도
        head = np.arctan2(steps[..., 1], steps[..., 0])      # [K,T]
        dhead = np.diff(head, axis=1)
        dhead = (dhead + np.pi) % (2 * np.pi) - np.pi
        curv = np.abs(dhead).sum(1, keepdims=True)           # 총 heading 변화
        return np.concatenate([ds, dds, prog, last_speed, curv], 1).astype(np.float32)

    @staticmethod
    def _sequence_first(t, batch_size, name):
        if t.dim() != 3:
            raise ValueError(f'{name} must be 3D, got {tuple(t.shape)}')
        return t.permute(1, 0, 2) if t.shape[0] == batch_size else t

    def forward(self, bev_key, image_value, batch_size, bev_pos=None):
        key = self._sequence_first(bev_key, batch_size, 'bev_key')
        value = self._sequence_first(image_value, batch_size, 'image_value')
        if key.shape[1] != batch_size:
            raise ValueError(f'batch 축이 어긋났다: {tuple(key.shape)}, B={batch_size}')
        v = self.value_norm(value)                       # [N, B, D]
        n_bev = v.shape[0]
        if n_bev != self.bev_h * self.bev_w:
            raise ValueError(f'BEV 크기 불일치: {n_bev} != '
                             f'{self.bev_h}x{self.bev_w}')

        # --- 국소: 후보 웨이포인트에서 영상 증거를 직접 조회 ---
        vm = v.permute(1, 2, 0).reshape(
            batch_size, self.embed_dims, self.bev_h, self.bev_w)
        grid = self.grid.expand(batch_size, -1, -1, -1)   # [B, K*T, 1, 2]
        s = F.grid_sample(vm, grid, mode='bilinear',
                          padding_mode='zeros', align_corners=False)
        s = s.reshape(batch_size, self.embed_dims,
                      self.num_anchors, self.fut_ts)
        s = s.permute(0, 2, 3, 1).reshape(
            batch_size, self.num_anchors, self.fut_ts * self.embed_dims)
        local = self.local_mlp(s)                         # [B, K, D]
        logits = self.local_score(local).squeeze(-1)      # [B, K]

        # --- 전역: 조건 없는 장면 요약을 후보 특징에 내적 ---
        if self.use_global:
            q = self.ts_embed.weight.unsqueeze(1).expand(-1, batch_size, -1)
            att, _ = self.cross_attn(query=q, key=key, value=v,
                                     need_weights=False)
            scene = self.scene_proj(
                att.permute(1, 0, 2).reshape(batch_size, -1))     # [B, D]
            logits = logits + torch.einsum(
                'bkd,bd->bk', local, scene) * self.logit_scale

        # --- Phase C: velocity residual (zero-init -> 시작 시 0) ---
        if self.residual:
            scene_motion = v.mean(0)                          # [B,D] 시간융합 BEV pool
            cand_motion = self.vel_kin_mlp(self.vel_kin)      # [K,D]
            sp = self.vel_scene_proj(scene_motion)            # [B,D] (image=0 -> 0)
            vel_logit = torch.einsum('bd,kd->bk', sp, cand_motion) * self.logit_scale
            logits = logits + self.res_beta * vel_logit
            self._cache_scene_stop = (
                self.stop_head(scene_motion).squeeze(-1) if self.use_stop else None)  # [B]

        index = logits.argmax(dim=-1)
        return self.anchors_inc.index_select(0, index), logits, index

    # ---------------------------------------------- ③ 완성-후보 API (opt-in)
    def attach_deploy_bank(self, path):
        """A0 배포 bank(abs/inc/ids/anchor_dist/nms_tau) 를 buffer 로 붙인다.
        학습 경로를 건드리지 않는 opt-in. first-6 불변식(abs/inc)을 로드 시 검증."""
        import numpy as _np
        z = _np.load(path, allow_pickle=True)
        self.register_buffer('candidate_abs_5s',
                             torch.from_numpy(z['candidate_xy_abs_5s'].astype('float32')))
        self.register_buffer('candidate_inc_5s',
                             torch.from_numpy(z['candidate_xy_inc_5s'].astype('float32')))
        self.register_buffer('candidate_ids',
                             torch.from_numpy(z['candidate_ids'].astype('int64')))
        self.register_buffer('anchor_dist',
                             torch.from_numpy(z['anchor_dist'].astype('float32')))
        self.nms_tau = float(z['nms_tau'])
        assert torch.equal(self.candidate_abs_5s[:, :6], self.anchors_abs), \
            'candidate_abs_5s first-6 != anchors_abs'
        assert torch.equal(self.candidate_inc_5s[:, :6], self.anchors_inc), \
            'candidate_inc_5s first-6 != anchors_inc'
        return self

    def build_candidates(self, logits):
        """규정 준수 완성-후보 API. logits [B,K] -> dict(N=12).
        내부에서 image-only score3+nms9 shortlist 만 수행. full-K/goal/cmd 미노출.
        반환: candidate_xy_abs_5s/inc_5s [B,12,10,2], visual_logits [B,12], candidate_ids [B,12]."""
        assert hasattr(self, 'candidate_abs_5s'), 'attach_deploy_bank() 를 먼저 호출하라'
        B = logits.shape[0]
        tau = self.nms_tau
        ad = self.anchor_dist
        order = torch.argsort(logits, dim=-1, descending=True, stable=True)  # 동률->최저 idx
        sel = logits.new_zeros(B, 12, dtype=torch.long)
        for b in range(B):
            ob = order[b]
            s = ob[:3].tolist(); ss = set(s)
            for c in ob[:64].tolist():
                if len(s) >= 12:
                    break
                if c in ss:
                    continue
                if float(ad[c, s].min()) > tau:
                    s.append(c); ss.add(c)
            for c in ob.tolist():
                if len(s) >= 12:
                    break
                if c not in ss:
                    s.append(c); ss.add(c)
            sel[b] = torch.tensor(s[:12], device=logits.device)
        return {
            'candidate_ids': self.candidate_ids[sel],
            'candidate_xy_abs_5s': self.candidate_abs_5s[sel],
            'candidate_xy_inc_5s': self.candidate_inc_5s[sel],
            'visual_logits': torch.gather(logits, 1, sel),
        }

    # ------------------------------------------------------------------ loss
    def _metric_distance(self, ego_fut_gt, ego_fut_masks):
        gt_abs = ego_fut_gt.cumsum(dim=-2)
        d = torch.linalg.vector_norm(
            self.anchors_abs.unsqueeze(0) - gt_abs.unsqueeze(1), dim=-1)
        valid = ego_fut_masks.to(d.dtype)
        return (d * self.challenge_w.view(1, 1, -1) * valid.unsqueeze(1)).sum(-1)

    # ---- 개별 loss 항 (ETRI_SCOREDRIVE_DESIGN.md §8.3) -------------------
    def _loss_softce(self, weighted, logits):
        """기존 full-K soft cross entropy (baseline 대조군)."""
        target_prob = F.softmax(-weighted / self.target_temperature, dim=-1)
        return -(target_prob * F.log_softmax(logits, dim=-1)).sum(-1).mean()

    def _positive_mask(self, weighted):
        """metric 상 정답과 사실상 동급인 후보 집합 P (Dmin+eps 이내)."""
        dmin = weighted.min(dim=-1, keepdim=True).values          # [B,1]
        return (weighted <= dmin + self.eps_abs), dmin

    def _loss_pos(self, weighted, logits, pmask):
        """§8.3.A positive-set probability: P 전체의 확률질량을 올린다."""
        logp = F.log_softmax(logits / self.tau_score, dim=-1)     # [B,K]
        masked = logp.masked_fill(~pmask, float('-inf'))
        return -(torch.logsumexp(masked, dim=-1)).mean()

    def _loss_rank(self, weighted, logits, pmask, dmin):
        """§8.3.B hard-negative pairwise ranking. pos=P 내 최고점,
        neg=visual top-K 중 P 밖이고 D>=Dmin+margin."""
        neg_pos = logits.masked_fill(~pmask, float('-inf'))
        pos_idx = neg_pos.argmax(dim=-1)                          # [B]
        pos_logit = logits.gather(1, pos_idx[:, None]).squeeze(1)  # [B]
        pos_D = weighted.gather(1, pos_idx[:, None]).squeeze(1)    # [B]
        k = min(self.neg_topk, logits.shape[-1])
        top_idx = logits.topk(k, dim=-1).indices                 # [B,k]
        neg_logit = logits.gather(1, top_idx)                    # [B,k]
        neg_D = weighted.gather(1, top_idx)                      # [B,k]
        valid = (~pmask.gather(1, top_idx)) & (neg_D >= dmin + self.rank_neg_margin)
        margin = self.m0 + self.beta * (neg_D - pos_D[:, None]).clamp(0, self.dmax)
        raw = F.softplus(neg_logit - pos_logit[:, None] + margin)
        denom = valid.float().sum().clamp(min=1.0)
        return (raw * valid.float()).sum() / denom

    def _loss_exp(self, weighted, logits, pmask):
        """§8.3.C expected official L2 over S = P ∪ GT최근접16 ∪ visual top32."""
        k = min(self.neg_topk, logits.shape[-1])
        gk = min(self.gt_near, logits.shape[-1])
        top_idx = logits.topk(k, dim=-1).indices
        near_idx = (-weighted).topk(gk, dim=-1).indices           # 가장 가까운 D
        smask = pmask.clone()
        smask.scatter_(1, top_idx, True)
        smask.scatter_(1, near_idx, True)
        score = (logits / self.tau_exp).masked_fill(~smask, float('-inf'))
        q = F.softmax(score, dim=-1)                              # S 밖은 0
        return (q * weighted).sum(dim=-1).mean()

    def _loss_timing(self, weighted, logits, ego_fut_gt):
        """§Phase C endpoint-near timing loss. 3초 endpoint 가 GT 와 같은(=목적지 동일)
        후보만 모아 official 3s weighted L2 로 local soft-CE. 전체 K 순서화 아님."""
        gt_end = ego_fut_gt.cumsum(dim=-2)[:, -1]             # [B,2]
        anc_end = self.anchors_abs[:, -1]                     # [K,2]
        ep = torch.linalg.vector_norm(
            anc_end.unsqueeze(0) - gt_end.unsqueeze(1), dim=-1)   # [B,K]
        k = min(self.timing_k, ep.shape[-1])
        H = ep.topk(k, dim=-1, largest=False).indices        # [B,k] endpoint 최근접
        wH = weighted.gather(1, H); lH = logits.gather(1, H)
        target = F.softmax(-wH / self.timing_tau, dim=-1)
        return -(target * F.log_softmax(lH, dim=-1)).sum(-1).mean()

    def _loss_stop(self, ego_fut_gt, scene_stop):
        """§Phase C stop/move auxiliary. GT 3초 arclen < stop_thresh = stop.
        정지 8.5%/regret 21% 이므로 class-balanced pos_weight."""
        arclen = torch.linalg.vector_norm(ego_fut_gt, dim=-1).sum(-1)   # [B]
        label = (arclen < self.stop_thresh).float()
        pos = label.sum(); neg = label.numel() - pos
        pw = (neg / (pos + 1e-6)).clamp(1.0, 20.0)
        return F.binary_cross_entropy_with_logits(
            scene_stop, label, pos_weight=pw)

    def vocabulary_loss(self, logits, ego_fut_gt, ego_fut_masks):
        weighted = self._metric_distance(ego_fut_gt, ego_fut_masks)   # D[B,K]
        pmask, dmin = self._positive_mask(weighted)

        comp = {}   # 성분별 값(진단) — 재현성 감사용
        if self.loss_mode == 'softce':
            t = self._loss_softce(weighted, logits)
            loss = t
            comp['plan_vocab_t_softce'] = t.detach()
        elif self.loss_mode == 'softce_exp':
            # positive-set(L_pos)의 실제 기여 분리: base term 을 softCE 로 두고 L_exp 만 더한다.
            ts = self._loss_softce(weighted, logits)
            te = self._loss_exp(weighted, logits, pmask)
            loss = ts + self.w_exp * te
            comp['plan_vocab_t_softce'] = ts.detach()
            comp['plan_vocab_t_exp'] = te.detach()
        else:
            tp = self._loss_pos(weighted, logits, pmask)
            loss = self.w_pos * tp
            comp['plan_vocab_t_pos'] = tp.detach()
            if 'rank' in self.loss_mode:                 # pos_rank, pos_rank_exp
                tr = self._loss_rank(weighted, logits, pmask, dmin)
                loss = loss + self.w_rank * tr
                comp['plan_vocab_t_rank'] = tr.detach()
            if 'exp' in self.loss_mode:                   # pos_exp, pos_rank_exp
                te = self._loss_exp(weighted, logits, pmask)
                loss = loss + self.w_exp * te
                comp['plan_vocab_t_exp'] = te.detach()

        # --- Phase C residual auxiliary (logits=final=base+residual) ---
        if self.residual and self.use_timing:
            lt = self._loss_timing(weighted, logits, ego_fut_gt)
            loss = loss + lt
            comp['plan_vocab_t_timing'] = lt.detach()
        if self.residual and self.use_stop and self._cache_scene_stop is not None:
            ls = self._loss_stop(ego_fut_gt, self._cache_scene_stop)
            loss = loss + self.stop_weight * ls
            comp['plan_vocab_t_stop'] = ls.detach()

        # ---- 진단 지표: mode 와 무관하게 항상 동일하게 찍는다 ----
        with torch.no_grad():
            K = weighted.shape[-1]
            best = weighted.argmin(dim=-1)                        # metric 최근접
            sel = logits.argmax(dim=-1)                           # 모델 top-1
            sel_l2 = weighted.gather(1, sel[:, None]).squeeze(1)   # = oracle@1

            def oracle_at(mm):
                idx = logits.topk(min(mm, K), dim=-1).indices
                return weighted.gather(1, idx).min(-1).values

            o3, o6, o20 = oracle_at(3), oracle_at(6), oracle_at(20)
            # 분포 진단: 붕괴/발산 조기 신호 (entropy↓ maxprob↑ = 과확신)
            logp_full = F.log_softmax(logits, dim=-1)
            p_full = logp_full.exp()
            entropy = -(p_full * logp_full).sum(-1).mean()
            maxprob = p_full.max(-1).values.mean()
            diag = {
                'plan_vocab_entropy': entropy,
                'plan_vocab_maxprob': maxprob,
                'plan_vocab_selected_l2': sel_l2.mean(),          # top-1 L2
                'plan_vocab_oracle3_l2': o3.mean(),
                'plan_vocab_oracle6_l2': o6.mean(),
                'plan_vocab_oracle20_l2': o20.mean(),
                'plan_vocab_cover_l2': weighted.min(-1).values.mean(),  # full-K
                'plan_vocab_gap3': (sel_l2 - o3).mean(),          # top1-oracle@3
                'plan_vocab_top1hit': sel.eq(best).float().mean(),
                'plan_vocab_psize': pmask.float().sum(-1).mean(),  # |P| 평균
            }
        out = {'loss_plan_vocab': loss * self.loss_weight}
        out.update(diag)
        out.update(comp)
        return out
