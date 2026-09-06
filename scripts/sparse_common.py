"""⑤-C current-only sparse ScoreDrive: shared data / loss / eval utilities.

이 모듈은 학습(train_sparse_scoredrive.py)과 평가(eval_sparse_scoredrive.py) 공용이다.

핵심 사실 (모두 노드에서 검증됨, 2026-09-04):
  - ego_cache.fut [N,6,2] = 절대(누적) 3초 waypoint, anchors_abs 와 동일 좌표계.
  - ego_cache_5s.fut5 [N,10,2] = 절대 5초 waypoint (fut5[:,:6]==fut, fut5[:,9]==goal).
  - 카메라 캘리브레이션은 전 scene/frame 에서 완전 동일(diff 0.0) → lidar2img 1회 구축.
  - 이미지: cache/etri_768/<scen>/<cam>/<frame:08d>.jpg, 768x432.
  - split(val_clips.npz): train 330 / val(holdout) 38 / minival 8 scene, 서로소.
    train_idx 19800 = 330 scene x 60 clip-frame. frame>=30 이 배포충실.

compliance: 이미지 6장 + 고정 A0 bank 좌표만 사용. goal/cmd/status/history 미입력.
  goal 은 외부 selector 에서만(제출용 index 선택) 쓰고 forward 에는 넣지 않는다.
  5초 GT 는 학습 정답일 뿐 모델 입력이 아니다.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

A = "/NHNHOME/data/sukim/adcl"
SD = os.path.join(A, "scripts")
sys.path.insert(0, SD)
from sparse_scoredrive import build_cached_lidar2img, CAMERA_ORDER  # noqa: E402

EGO = "/tmp/pm97/data/etri/ego_cache.npz"
EGO5 = "/tmp/pm97/data/etri/ego_cache_5s.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"
TRAIN_PKL = "/tmp/pm97/data/etri/pkl/etri_train330_goal.pkl"
VAL_PKL = "/tmp/pm97/data/etri/pkl/etri_val38_goal.pkl"
CACHE = os.path.join(A, "cache/etri_768")
BANK_A0 = os.path.join(A, "data/etri/bank_A0_deploy.npz")
SPLIT_JSON = os.path.join(A, "data/etri/sparse_split.json")

CHALLENGE_W = np.array([11, 11, 5, 5, 2, 2], np.float64)
CW3 = CHALLENGE_W / CHALLENGE_W.sum()
# 5초 aux(설계 §8.3.E L_tail): 3초 score 와 분리 — 3.5~5.0s tail(step 6..9)만 본다.
# 첫 6점 가중 0 → aux 는 오로지 tail 궤적/endpoint 신호만 더한다(3초 저하 방지, §5.1).
# L_exp(D_tail) soft-expected 형태(user spec "L_exp(D5 trajectory/endpoint)")로 쓴다.
CHALLENGE_W5 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], np.float64)
CW5 = CHALLENGE_W5 / CHALLENGE_W5.sum()

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

N_TUNE_SCENES = 30          # train330 에서 떼어내는 고정 튜닝 holdout (val38 은 최종 보고 전용)
STOP_ARCLEN = 0.5           # GT 3초 이동거리 < 0.5m = 정지 버킷
HELD_OUT_OFFSET = 4         # ⑤-C2: frame%5==4 는 학습 제외(암기 probe)
HIS_NOW = 30                # ego_cache.his 는 오래된 순, index 30 = 현재
HIS_STRIDE = 5              # 0.1s 간격이므로 5 = 0.5s (ego2global 로 교차검증)


# --------------------------------------------------------------------------- #
#  calibration (global constant → 1회 구축)
# --------------------------------------------------------------------------- #
def build_global_lidar2img(pkl_path: str = TRAIN_PKL) -> np.ndarray:
    import pickle
    info = pickle.load(open(pkl_path, "rb"))["infos"][0]
    return build_cached_lidar2img(info).astype(np.float32)   # [6,4,4]


# --------------------------------------------------------------------------- #
#  split (train330 → train 300 / tuneval 30) — 결정적, 파일로 고정
# --------------------------------------------------------------------------- #
def load_arrays() -> Dict:
    ec = np.load(EGO, allow_pickle=True)
    e5 = np.load(EGO5, allow_pickle=True)
    sp = np.load(SPLIT, allow_pickle=True)
    assert np.array_equal(ec["scen_idx"], e5["scen_idx"])
    assert np.array_equal(ec["frame"], e5["frame"])
    return dict(
        scenarios=ec["scenarios"].astype(str),
        scen_idx=ec["scen_idx"].astype(np.int64),
        frame=ec["frame"].astype(np.int64),
        his=ec["his"].astype(np.float32),          # [N,31,2] 오래된순 0.1s 간격
        his_yaw=ec["his_yaw"].astype(np.float32),  # [N,31]
        fut=ec["fut"].astype(np.float32),          # [N,6,2] abs
        goal=ec["goal"].astype(np.float32),        # [N,2] = 5s endpoint
        vad_cmd=ec["vad_cmd"].astype(np.int64),
        speed=ec["speed"].astype(np.float32),
        acc=ec["acc"].astype(np.float32),
        fut5=e5["fut5"].astype(np.float32),        # [N,10,2] abs
        mask5=e5["mask5"].astype(np.float32),      # [N,10]
        train_idx=sp["train_idx"].astype(np.int64),
        val_idx=sp["val_idx"].astype(np.int64),
        val_weight=sp["val_weight"].astype(np.float64),
        minival_idx=sp["minival_idx"].astype(np.int64),
    )


def make_split(arr: Dict, persist: bool = True, all_frames: bool = False,
               holdout_offset: bool = False) -> Dict:
    """train330 을 train 300 scene / tuneval 30 scene 으로 결정적 분할.

    all_frames=True 면 학습 scene 의 10Hz 전체 프레임(scene당 300장)을 쓴다.
    비-main frame(frame%5!=0) GT 유효성 검증됨(2026-09-04): nearest-anchor D3
    0.1018 vs main 0.1007, mask5=1.0, zero-fut 0% → 학습 라벨로 안전.
    tuneval 은 비교 가능성 유지를 위해 항상 train_idx∩frame>=30 만 쓴다.
    """
    if os.path.isfile(SPLIT_JSON):
        d = json.load(open(SPLIT_JSON))
        tune_scenes = set(d["tuneval_scenes"])
    else:
        tr_scen = np.unique(arr["scen_idx"][arr["train_idx"]])
        names = sorted(arr["scenarios"][s] for s in tr_scen)
        # 균등 커버리지: 정렬된 이름에서 offset 5, stride 11 로 30개 추출
        tune_scenes = set(names[5::11][:N_TUNE_SCENES])
        assert len(tune_scenes) == N_TUNE_SCENES
        if persist:
            json.dump({"tuneval_scenes": sorted(tune_scenes),
                       "n_tune": N_TUNE_SCENES, "stride": 11, "offset": 5},
                      open(SPLIT_JSON, "w"), indent=1)
    scen_name = arr["scenarios"][arr["scen_idx"]]
    is_tune = np.array([n in tune_scenes for n in scen_name])
    tr = arr["train_idx"]
    tr_is_tune = is_tune[tr]
    tr_scen_idx = np.unique(arr["scen_idx"][tr])
    in_train_scen = np.isin(arr["scen_idx"], tr_scen_idx)
    frame = arr["frame"]
    off = frame % 5
    if holdout_offset:
        # ⑤-C2: 0.5초 구간의 5프레임 중 %5==4 를 학습에서 완전히 제외해
        # "같은 시나리오 · 미학습 프레임" probe(평가 A)를 무료로 만든다.
        train_pool = in_train_scen & ~is_tune & (off != HELD_OUT_OFFSET)
    elif all_frames:
        train_pool = in_train_scen & ~is_tune
    else:
        train_pool = np.zeros(len(frame), bool)
        train_pool[tr[~tr_is_tune]] = True
    train_rows = np.where(train_pool)[0]
    # probe A: 학습 시나리오, %5==4(미학습 offset), 배포충실 frame>=30
    probe_rows = np.where(in_train_scen & ~is_tune &
                          (off == HELD_OUT_OFFSET) & (frame >= 30))[0]
    # probe B(=tuneval): 미학습 시나리오, 배포충실 frame>=30
    tune_rows = tr[tr_is_tune & (arr["frame"][tr] >= 30)]
    return dict(train_rows=train_rows, tune_rows=tune_rows,
                probe_rows=probe_rows, tune_scenes=sorted(tune_scenes))


def make_epoch_rows(arr, train_rows, seed):
    """⑤-C2 sampling: 각 (시나리오, 0.5초 구간)에서 프레임 1개만 뽑고,
    한 batch 가 서로 다른 시나리오로 채워지도록 시나리오 round-robin 순서로 낸다.

    - epoch 당 약 18K (300 scene x 60 bucket) → 인접 10Hz 프레임 반복 감소
    - batch 16 이 16개 서로 다른 시나리오가 되어 scenario memorization 완화
    """
    rng = np.random.default_rng(seed)
    scen = arr["scen_idx"][train_rows]
    bucket = arr["frame"][train_rows] // 5
    key = scen.astype(np.int64) * 100000 + bucket.astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks, starts = np.unique(key[order], return_index=True)
    ends = np.r_[starts[1:], len(order)]
    picked = np.array([order[s + rng.integers(e - s)] for s, e in zip(starts, ends)])
    # 시나리오별로 모아 round-robin
    ps = scen[picked]
    by = {}
    for i, sc in enumerate(ps):
        by.setdefault(int(sc), []).append(int(picked[i]))
    for v in by.values():
        rng.shuffle(v)
    keys = list(by)
    rng.shuffle(keys)
    out, depth = [], max(len(by[k]) for k in keys)
    for d in range(depth):
        layer = [by[k][d] for k in keys if d < len(by[k])]
        rng.shuffle(layer)
        out.extend(layer)
    return train_rows[np.array(out, dtype=np.int64)]


def history_transforms(arr, row, n_hist=3, stride=HIS_STRIDE):
    """현재 ego 좌표계 -> 과거 시점 ego 좌표계 강체변환 [n_hist,4,4].

    설계 §3.3: 상대 pose 는 과거 image feature 를 현재 후보 좌표에 정렬하는
    고정 행렬 곱에만 쓴다. 이 행렬은 scorer 에 feature/token 으로 들어가지 않는다.

    ego_cache.his 는 오래된 순 [31,2] (0.1s 간격, index 30 = 현재),
    his_yaw 는 같은 색인의 heading. lidar2ego 는 항등(검증됨)이라 lidar==ego.

    과거 pose 가 현재 프레임에서 (p, θ) 이면 X_cur = R(θ)X_past + p 이므로
    X_past = R(θ)^T (X_cur - p).  k=0 은 항등이 되어 current-only 와 정확히 일치한다.
    """
    his = arr["his"][row]
    yaw = arr["his_yaw"][row]
    out = np.zeros((n_hist, 4, 4), np.float32)
    for k in range(n_hist):
        idx = HIS_NOW - stride * (k + 1)
        p = his[idx].astype(np.float64)
        th = float(yaw[idx])
        c, sn = np.cos(th), np.sin(th)
        R = np.array([[c, sn, 0.0], [-sn, c, 0.0], [0.0, 0.0, 1.0]])  # R(θ)^T
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = -R @ np.array([p[0], p[1], 0.0])
        out[k] = M
    return out


def history_available(arr, rows, n_hist=3, stride=HIS_STRIDE):
    """과거 n_hist 장이 같은 시나리오 안에 존재하는 row 만 True.
    프레임 인덱스가 시나리오 내부 0..299 이므로 frame >= n_hist*stride 면 충분하다."""
    return arr["frame"][rows] >= n_hist * stride


# --------------------------------------------------------------------------- #
#  이미지 로딩
# --------------------------------------------------------------------------- #
def _photometric(im, rng):
    """색/노출/감마/블러. spatial crop·flip 은 calibration 이 깨지므로 금지."""
    im = im * rng.uniform(0.75, 1.30)                       # brightness
    m = im.mean()
    im = (im - m) * rng.uniform(0.75, 1.30) + m             # contrast
    g = im.mean(axis=2, keepdims=True)
    im = g + (im - g) * rng.uniform(0.70, 1.30)             # saturation
    im = np.clip(im, 1e-4, 1.0) ** rng.uniform(0.80, 1.25)  # gamma
    if rng.random() < 0.20:                                  # 3x3 box blur
        k = np.ones((3, 3), np.float32) / 9.0
        pad = np.pad(im, ((1, 1), (1, 1), (0, 0)), mode="edge")
        acc = np.zeros_like(im)
        for dy in range(3):
            for dx in range(3):
                acc += pad[dy:dy + im.shape[0], dx:dx + im.shape[1]] * k[dy, dx]
        im = acc
    return np.clip(im, 0.0, 1.0)


def load_six_images(scenario: str, frame: int, aug: bool = False,
                    rng=None, cam_dropout: float = 0.0) -> torch.Tensor:
    """[6,3,432,768] ImageNet 정규화 float32 (RGB).

    aug=True 면 카메라별 독립 photometric jitter + camera dropout 을 적용한다.
    공간 변환(crop/flip)은 후보 waypoint projection 이 깨지므로 절대 하지 않는다.
    """
    out = np.empty((6, 3, 432, 768), np.float32)
    drop = -1
    if aug and cam_dropout > 0.0 and rng.random() < cam_dropout:
        drop = int(rng.integers(6))
    for ci, cam in enumerate(CAMERA_ORDER):
        if ci == drop:
            out[ci] = 0.0
            continue
        p = os.path.join(CACHE, scenario, cam, f"{frame:08d}.jpg")
        im = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0  # H,W,3
        if aug:
            im = _photometric(im, rng)
        im = (im - IMAGENET_MEAN) / IMAGENET_STD
        out[ci] = im.transpose(2, 0, 1)
    return torch.from_numpy(out)


class SparseFrameDataset(Dataset):
    """단일 프레임: 6 이미지 + GT(3s/5s). lidar2img 는 전역이라 미포함."""

    def __init__(self, arr: Dict, rows: np.ndarray, aug: bool = False,
                 cam_dropout: float = 0.0, epoch: int = 0,
                 n_hist: int = 0, hist_scale: float = 1.0):
        self.arr = arr
        self.rows = np.asarray(rows, np.int64)
        self.aug = bool(aug)
        self.cam_dropout = float(cam_dropout)
        self.epoch = int(epoch)
        self.n_hist = int(n_hist)          # ⑤-D: 과거 프레임 수 (0 = current-only)
        self.hist_scale = float(hist_scale)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        a = self.arr
        scen = a["scenarios"][a["scen_idx"][r]]
        fr = int(a["frame"][r])
        rng = np.random.default_rng((self.epoch << 32) ^ (r * 2654435761 + i)) \
            if self.aug else None
        img = load_six_images(scen, fr, aug=self.aug, rng=rng,
                              cam_dropout=self.cam_dropout)
        extra = {}
        if self.n_hist:
            hs = []
            for k in range(self.n_hist):
                pf = fr - HIS_STRIDE * (k + 1)
                hi = load_six_images(scen, pf, aug=self.aug, rng=rng,
                                     cam_dropout=self.cam_dropout)
                if self.hist_scale != 1.0:
                    hi = torch.nn.functional.interpolate(
                        hi, scale_factor=self.hist_scale, mode="bilinear",
                        align_corners=False, recompute_scale_factor=False)
                hs.append(hi)
            extra["img_hist"] = torch.stack(hs)                     # [n_hist,6,3,H,W]
            extra["hist_T"] = torch.from_numpy(
                history_transforms(self.arr, r, self.n_hist))       # [n_hist,4,4]
        return dict(
            img=img, **extra,
            gt3=torch.from_numpy(a["fut"][r]),        # [6,2] abs
            gt5=torch.from_numpy(a["fut5"][r]),       # [10,2] abs
            mask5=torch.from_numpy(a["mask5"][r]),    # [10]
            row=r,
        )


# --------------------------------------------------------------------------- #
#  D3 / D5 metric (배치, GPU)
# --------------------------------------------------------------------------- #
def metric_d3(gt3_abs: torch.Tensor, anchors_abs: torch.Tensor,
              cw3: torch.Tensor) -> torch.Tensor:
    """gt3_abs [B,6,2], anchors_abs [K,6,2] -> D3 [B,K] (공식 가중)."""
    d = torch.linalg.vector_norm(
        anchors_abs.unsqueeze(0) - gt3_abs.unsqueeze(1), dim=-1)   # [B,K,6]
    return (d * cw3.view(1, 1, -1)).sum(-1)


def metric_d5(gt5_abs: torch.Tensor, mask5: torch.Tensor,
              cand_abs5: torch.Tensor, cw5: torch.Tensor) -> torch.Tensor:
    """gt5_abs [B,10,2], mask5 [B,10], cand_abs5 [K,10,2] -> D5 [B,K]."""
    d = torch.linalg.vector_norm(
        cand_abs5.unsqueeze(0) - gt5_abs.unsqueeze(1), dim=-1)     # [B,K,10]
    w = (cw5.view(1, 1, -1) * mask5.unsqueeze(1))
    return (d * w).sum(-1)


def progress_labels(gt3_abs, gt5_abs):
    """설계 §4.5 권장 auxiliary label: 자차 진행량.
    반환 [B,8] = [S3/30, S5/30, r1..r6]  (r_i = i 스텝까지 누적거리 / S3).
    train pose 에서 계산한 라벨이며 모델 입력이 아니다."""
    def _arc(x):
        z = torch.zeros_like(x[:, :1])
        inc = torch.diff(torch.cat([z, x], 1), dim=1)
        return torch.linalg.vector_norm(inc, dim=-1)            # [B,T]
    st3 = _arc(gt3_abs)
    s3 = st3.sum(-1, keepdim=True)
    s5 = _arc(gt5_abs).sum(-1, keepdim=True)
    r = torch.cumsum(st3, dim=1) / s3.clamp(min=1e-3)           # [B,6]
    return torch.cat([s3 / 30.0, s5 / 30.0, r], dim=-1)


def occupancy_labels(cand6, apos, arad, amask):
    """후보 waypoint 점유 라벨 [B,K,6].

    cand6 [K,6,2] 고정 후보(3초 6점), apos [B,A,6,2] 에이전트 시각별 중심,
    arad [B,A] 외접반경+여유, amask [B,A,6] 유효.
    goal/cmd 와 무관한 순수 기하 라벨이라 compliance 안전(학습 라벨 전용).
    """
    d = torch.linalg.vector_norm(
        cand6[None, :, None, :, :] - apos[:, None, :, :, :], dim=-1)  # [B,K,A,6]
    hit = (d < arad[:, None, :, None]) & amask[:, None, :, :]
    return hit.any(dim=2).float()                                     # [B,K,6]


# --------------------------------------------------------------------------- #
#  loss (Phase A softce + L_exp, 그리고 5초 aux L_exp(D5))
# --------------------------------------------------------------------------- #
class ScoreDriveLoss:
    """Phase A 확정 loss(softce_exp) 이식 + 선택적 5초 aux.

    hyperparam 은 dense decoder 기본값과 동일:
      target_temperature=0.1, tau_exp=0.1, w_exp=0.25, neg_topk=32,
      gt_near=16, eps_abs=0.05.
    """

    def __init__(self, target_temp=0.1, tau_exp=0.1, w_exp=0.25,
                 neg_topk=32, gt_near=16, eps_abs=0.05,
                 aux5=False, beta5=0.1,
                 objective="softce_exp", set_k=16, w_exp_set=0.1,
                 alpha_kd=0.0, tau_kd=1.0, w_occ=0.0, w_rank=1.0,
                 kd_mode="full", w_speed=0.0, w_vo=0.0,
                 w_hardneg=0.0, hn_k=16, hn_margin=0.5):
        self.objective = objective      # softce_exp | set
        self.set_k = int(set_k)         # L_set 의 GT-근접 후보 수
        self.w_exp_set = float(w_exp_set)   # set 목적일 때의 작은 L_exp 가중
        self.alpha_kd = float(alpha_kd)     # teacher KL 가중
        self.tau_kd = float(tau_kd)         # teacher/student 분포 온도
        self.w_occ = float(w_occ)           # 보조 perception(점유) 가중
        self.w_rank = float(w_rank)         # 0 = occ-only (점유 학습 상한 진단)
        self.kd_mode = kd_mode              # full | top64 | shortlist
        self.w_speed = float(w_speed)       # 진행량 회귀 보조 가중
        self.w_vo = float(w_vo)             # 과거 변위(VO) 회귀 보조 가중
        self.w_hardneg = float(w_hardneg)   # 초기 종방향 hard negative 가중
        self.hn_k = int(hn_k)               # D5 상위 몇 개를 혼동 후보 풀로 볼지
        self.hn_margin = float(hn_margin)
        self.target_temp = target_temp
        self.tau_exp = tau_exp
        self.w_exp = w_exp
        self.neg_topk = neg_topk
        self.gt_near = gt_near
        self.eps_abs = eps_abs
        self.aux5 = aux5
        self.beta5 = beta5

    @staticmethod
    def _softce(D, logits, temp):
        target = torch.softmax(-D / temp, dim=-1)
        return -(target * torch.log_softmax(logits, dim=-1)).sum(-1).mean()

    def _exp(self, D, logits):
        K = logits.shape[-1]
        k = min(self.neg_topk, K)
        gk = min(self.gt_near, K)
        dmin = D.min(dim=-1, keepdim=True).values
        pmask = D <= dmin + self.eps_abs
        top_idx = logits.topk(k, dim=-1).indices
        near_idx = (-D).topk(gk, dim=-1).indices
        smask = pmask.clone()
        smask.scatter_(1, top_idx, True)
        smask.scatter_(1, near_idx, True)
        score = (logits / self.tau_exp).masked_fill(~smask, float("-inf"))
        q = torch.softmax(score, dim=-1)
        return (q * D).sum(dim=-1).mean()

    def _set(self, D3, logits):
        """§목표 불일치 교정: exact top-1 대신 GT-근접 상위 set_k 후보 집합에
        확률 질량을 올린다.  L_set = -log Σ_{k∈P} softmax(logit)[k].
        goal 을 못 보는 모델에 '실제 주행 경로 하나'를 강요하지 않는다."""
        k = min(self.set_k, logits.shape[-1])
        P = (-D3).topk(k, dim=-1).indices                  # GT D3 가장 가까운 k개
        logp = torch.log_softmax(logits, dim=-1)
        return -(torch.logsumexp(logp.gather(1, P), dim=-1)).mean()

    def _kd_subset(self, logits, teacher, idx, mask=None):
        """teacher 상위 subset 안에서만 listwise KL. full-K 분포를 통째로 모방하는 대신
        '어느 후보들이 상위권인가'의 순서만 옮긴다(⑤-E1 top64 rank distillation)."""
        t = torch.log_softmax(teacher.float().gather(1, idx) / self.tau_kd, dim=-1)
        q = torch.log_softmax(logits.gather(1, idx) / self.tau_kd, dim=-1)
        per = (t.exp() * (t - q)).sum(-1)
        if mask is None:
            return per.mean()
        w = mask.to(per.dtype)
        return (per * w).sum() / w.sum().clamp(min=1.0)

    def _kd_set(self, logits, set_idx, mask=None):
        """teacher 의 score3+nms9 shortlist 를 positive set 으로 하는 집합 loss.
        '좋은 후보군을 shortlist 에 남긴다'만 옮기고 그 안의 순서는 강요하지 않는다."""
        logp = torch.log_softmax(logits, dim=-1)
        per = -torch.logsumexp(logp.gather(1, set_idx), dim=-1)
        if mask is None:
            return per.mean()
        w = mask.to(per.dtype)
        return (per * w).sum() / w.sum().clamp(min=1.0)

    def _kd(self, logits, teacher, mask=None):
        """KL(teacher || student), per-sample. teacher 도 goal/cmd 미입력이라
        compliance 무관하고 추론 시 사용하지 않는다.

        mask: [B] 0/1. teacher logits 가 신뢰 가능한 샘플만 쓴다.
        (frame<30 은 dense streaming 이력이 짧아 teacher real 0.43 vs 0.26 로 나쁘다)
        """
        t = torch.log_softmax(teacher.float() / self.tau_kd, dim=-1)
        q = torch.log_softmax(logits / self.tau_kd, dim=-1)
        per = (t.exp() * (t - q)).sum(-1)                  # [B]
        if mask is None:
            return per.mean()
        w = mask.to(per.dtype)
        return (per * w).sum() / w.sum().clamp(min=1.0)

    def _occ(self, occ_logits, occ_lab):
        """보조 perception: 샘플된 waypoint feature 로 점유 예측 (class-balanced BCE).
        dense 가 가진 object supervision 을 sparse 경로에 직접 주는 항."""
        z = occ_logits[..., :occ_lab.shape[-1]].float()
        pos = occ_lab.sum()
        neg = occ_lab.numel() - pos
        pw = (neg / (pos + 1e-6)).clamp(1.0, 50.0)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            z, occ_lab, pos_weight=pw)

    def _hardneg(self, logits, D3, D5):
        """⑤-V: "5초 목적지는 맞는데 초기 타이밍이 틀린" 후보를 눌러준다.

        ⑤-H 실측: regret 의 77% 가 endpoint 2m 이내 프레임에서 나오고,
        5초 endpoint 를 완벽히 알아도 이득이 0 인 반면 1.5~2초 종방향 위치는
        83% 를 회수한다. 즉 진짜 혼동은 **같은 목적지, 다른 초기 진행**이다.
        그 후보는 정확히 'D5 는 낮고 D3 는 높은' 후보다.

        D5 최저 hn_k 개를 혼동 풀로 잡고, 그 안에서 D3 가 가장 나쁜 것을 negative,
        전체 D3 최저를 positive 로 두어 hinge margin 을 건다.
        """
        with torch.no_grad():
            pos = D3.argmin(dim=1)                                  # [B]
            pool = D5.topk(self.hn_k, dim=1, largest=False).indices  # [B,k] 목적지 근접
            d3p = torch.gather(D3, 1, pool)                          # [B,k]
            neg = torch.gather(pool, 1, d3p.argmax(dim=1, keepdim=True)).squeeze(1)
            # positive 가 풀에 뽑혀 negative 와 같아지면 그 표본은 뺀다
            valid = (neg != pos).float()
        lp = torch.gather(logits, 1, pos[:, None]).squeeze(1)
        ln = torch.gather(logits, 1, neg[:, None]).squeeze(1)
        gap = torch.gather(D3, 1, neg[:, None]).squeeze(1) - \
            torch.gather(D3, 1, pos[:, None]).squeeze(1)
        # margin 은 실제 D3 격차에 비례시킨다(격차가 작으면 강하게 밀 이유가 없다)
        m = self.hn_margin * torch.clamp(gap / 0.5, max=2.0)
        loss = F.relu(m - (lp - ln)) * valid
        return loss.sum() / valid.sum().clamp_min(1.0)

    def _speed(self, pred, lab):
        """진행량 회귀 (smooth L1). S3/S5 는 /30 정규화, r 은 [0,1]."""
        return torch.nn.functional.smooth_l1_loss(pred.float(), lab, beta=0.05)

    def __call__(self, logits, D3, D5=None, teacher=None, kd_mask=None,
                 occ_logits=None, occ_lab=None, kd_idx=None,
                 speed_pred=None, speed_lab=None, vo_pred=None, vo_lab=None):
        logits = logits.float()
        comp = {}
        if self.objective == "set":
            tset = self._set(D3, logits)
            te = self._exp(D3, logits)
            loss = self.w_rank * (tset + self.w_exp_set * te)
            comp["set"] = tset.detach(); comp["exp3"] = te.detach()
        else:
            ts = self._softce(D3, logits, self.target_temp)
            te = self._exp(D3, logits)
            loss = self.w_rank * (ts + self.w_exp * te)
            comp["softce"] = ts.detach(); comp["exp3"] = te.detach()
        if self.alpha_kd > 0.0 and teacher is not None:
            if self.kd_mode == "top64" and kd_idx is not None:
                tkd = self._kd_subset(logits, teacher, kd_idx, kd_mask)
            elif self.kd_mode == "shortlist" and kd_idx is not None:
                tkd = self._kd_set(logits, kd_idx, kd_mask)
            else:
                tkd = self._kd(logits, teacher, kd_mask)
            loss = loss + self.alpha_kd * tkd
            comp["kd"] = tkd.detach()
        if self.w_occ > 0.0 and occ_logits is not None and occ_lab is not None:
            tocc = self._occ(occ_logits, occ_lab)
            loss = loss + self.w_occ * tocc
            comp["occ"] = tocc.detach()
        if self.w_speed > 0.0 and speed_pred is not None and speed_lab is not None:
            tsp = self._speed(speed_pred, speed_lab)
            loss = loss + self.w_speed * tsp
            comp["speed"] = tsp.detach()
        if self.w_vo > 0.0 and vo_pred is not None and vo_lab is not None:
            # 과거 변위는 미터 단위라 10m 로 정규화. 미래 예측이 아니라 관측 복원이다.
            tvo = torch.nn.functional.smooth_l1_loss(
                vo_pred.float() / 10.0, vo_lab / 10.0, beta=0.02)
            loss = loss + self.w_vo * tvo
            comp["vo"] = tvo.detach()
        if self.w_hardneg > 0.0 and D5 is not None:
            thn = self._hardneg(logits, D3, D5)
            loss = loss + self.w_hardneg * thn
            comp["hn"] = thn.detach()
        if self.aux5 and D5 is not None:
            te5 = self._exp(D5, logits)
            loss = loss + self.beta5 * te5
            comp["exp5"] = te5.detach()
        return loss, comp


# --------------------------------------------------------------------------- #
#  shortlist / selector / 지표 (offline, numpy) — scoredrive_api 와 동일 규칙
# --------------------------------------------------------------------------- #
def _shortlist(logit_row, anchor_dist, nms_tau, n_out=12, score_top=3, nms_pool=64):
    order = np.argsort(-logit_row, kind="stable")
    sel = list(order[:score_top])
    sset = set(int(x) for x in sel)
    for c in order[:nms_pool]:
        if len(sel) >= n_out:
            break
        c = int(c)
        if c in sset:
            continue
        if anchor_dist[c, sel].min() > nms_tau:
            sel.append(c); sset.add(c)
    for c in order:
        if len(sel) >= n_out:
            break
        c = int(c)
        if c not in sset:
            sel.append(c); sset.add(c)
    return np.array(sel[:n_out], dtype=np.int64)


def eval_logits(logits, D3gt, goal_xy, cand_end5, anchor_dist, nms_tau,
                weight, lambdas=(0.0, 0.05, 0.1, 0.25, 0.5), buckets=None):
    """logits [N,K], D3gt [N,K] (각 후보의 GT 까지 공식 D3),
    goal_xy [N,2], cand_end5 [K,2] (각 후보 5초 endpoint, selector goal 거리용).
    반환 dict: top1, oracle@3/6/12, shortlist_oracle@12, realized(λ별), 버킷.
    weight [N] 가중(공식 proxy). 모든 지표 weighted mean.
    """
    N, K = logits.shape
    w = np.asarray(weight, np.float64)
    wm = lambda a: float(np.average(a, weights=w))

    top1 = np.empty(N); o3 = np.empty(N); o6 = np.empty(N); o12 = np.empty(N)
    sl_or = np.empty(N)
    real = {lm: np.empty(N) for lm in lambdas}
    for n in range(N):
        lg = logits[n]; Dg = D3gt[n]
        order = np.argsort(-lg, kind="stable")
        top1[n] = Dg[order[0]]
        o3[n] = Dg[order[:3]].min()
        o6[n] = Dg[order[:6]].min()
        o12[n] = Dg[order[:12]].min()
        S = _shortlist(lg, anchor_dist, nms_tau)
        sl_or[n] = Dg[S].min()
        gc = np.linalg.norm(cand_end5[S] - goal_xy[n], axis=1)
        vc = lg.max() - lg[S]
        gcn = (gc - gc.min()) / (gc.max() - gc.min() + 1e-9)
        vcn = (vc - vc.min()) / (vc.max() - vc.min() + 1e-9)
        for lm in lambdas:
            if not np.isfinite(lm):
                j = int(np.argmin(vc))
            else:
                j = int(np.argmin(gcn + lm * vcn))
            real[lm][n] = Dg[S[j]]

    out = dict(
        n=int(N), top1=wm(top1),
        oracle3=wm(o3), oracle6=wm(o6), oracle12=wm(o12),
        shortlist_oracle12=wm(sl_or),
        realized={f"{lm:g}": wm(real[lm]) for lm in lambdas},
    )
    if buckets is not None:
        bkt = {}
        best_lm = 0.1
        rr = real[best_lm]
        for name, mask in buckets.items():
            mask = np.asarray(mask, bool)
            if mask.sum() == 0:
                bkt[name] = None
                continue
            ww = w[mask]
            bkt[name] = dict(
                n=int(mask.sum()),
                realized=float(np.average(rr[mask], weights=ww)),
                shortlist_oracle12=float(np.average(sl_or[mask], weights=ww)))
        out["buckets_lambda0.1"] = bkt
    return out


def run_logits(model, arr, rows, lidar2img_t, device, batch=16,
               amp=True, num_workers=6, n_hist=0, hist_scale=1.0):
    """model.forward_logits 로 rows 의 full-K logits [len(rows),K] (numpy) 계산."""
    from torch.utils.data import DataLoader
    ds = SparseFrameDataset(arr, rows, n_hist=n_hist, hist_scale=hist_scale)
    dl = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    model.eval()
    out = []
    l2i = lidar2img_t.to(device)
    with torch.no_grad():
        for b in dl:
            img = b["img"].to(device, non_blocking=True)
            l2ib = l2i.unsqueeze(0).expand(img.shape[0], -1, -1, -1)
            with torch.autocast(device_type="cuda", enabled=amp,
                                dtype=torch.float16):
                if n_hist:
                    lg = model.forward_logits_temporal(
                        img, b["img_hist"].to(device, non_blocking=True), l2ib,
                        b["hist_T"].to(device, non_blocking=True))
                else:
                    lg = model.forward_logits(img, l2ib)
            out.append(lg.float().cpu().numpy())
    return np.concatenate(out, 0)


def precompute_targets(arr, rows, bank, weight=None):
    """평가용 GT 타깃 사전계산.
    반환: D3gt [N,K], goal_xy [N,2], cand_end5 [K,2], anchor_dist [K,K],
          nms_tau, weight [N], buckets.
    D3gt[n,k] = 각 후보(=anchor first6)의 GT 3초까지 공식 D3.
    """
    rows = np.asarray(rows, np.int64)
    anchors_abs = bank["anchors_abs"].astype(np.float64)          # [K,6,2]
    cand_end5 = bank["candidate_xy_abs_5s"][:, 9].astype(np.float64)  # [K,2]
    anchor_dist = bank["anchor_dist"].astype(np.float64)
    nms_tau = float(bank["nms_tau"])
    gt3 = arr["fut"][rows].astype(np.float64)                     # [N,6,2] abs
    N, K = len(rows), anchors_abs.shape[0]
    D3gt = np.empty((N, K), np.float64)
    for s in range(0, K, 256):
        a = anchors_abs[s:s + 256]
        D3gt[:, s:s + 256] = (
            np.linalg.norm(gt3[:, None] - a[None], axis=-1) * CW3).sum(-1)
    goal_xy = arr["goal"][rows].astype(np.float64)                # 5s endpoint
    if weight is None:
        weight = np.ones(N, np.float64)
    buckets = make_buckets(gt3, arr["vad_cmd"][rows])
    return dict(D3gt=D3gt, goal_xy=goal_xy, cand_end5=cand_end5,
                anchor_dist=anchor_dist, nms_tau=nms_tau,
                weight=np.asarray(weight, np.float64), buckets=buckets)


def make_buckets(fut3_abs, vad_cmd, mask5=None):
    """정지/가속/좌/우 버킷. fut3_abs [N,6,2] abs, vad_cmd [N]."""
    inc = np.diff(np.concatenate(
        [np.zeros_like(fut3_abs[:, :1]), fut3_abs], axis=1), axis=1)  # [N,6,2]
    step = np.linalg.norm(inc, axis=-1)                               # [N,6]
    arclen = step.sum(-1)
    stop = arclen < STOP_ARCLEN
    # 가속: 후반 3스텝 평균속도 - 전반 3스텝 평균속도 > 0.15m/step
    accel = (step[:, 3:].mean(1) - step[:, :3].mean(1)) > 0.15
    right = (vad_cmd == 0) & ~stop
    left = (vad_cmd == 1) & ~stop
    return dict(stop=stop, accel=accel & ~stop, left=left, right=right)
