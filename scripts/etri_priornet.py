#!/usr/bin/env python
"""Phase 1 — PriorNet v1. ego 관측 + goal 만으로 3초 궤적.

설계 (4탄 Phase 1 + 부록3/4)
  입력   31프레임 상대궤적 (x, y, yaw) + 파생(속도·가속·yaw rate·곡률)
         + goal (x, y, 거리, 방위각, yaw)   ← goal yaw는 0-5에서 실재 확정
         + vad_cmd(3) + meta command(6) + 미러 플래그
  구조   프레임별 토큰화 -> {MLP | GRU 2층 | Transformer 2층 4헤드} -> PLAN -> 6×2 증분
  변형   A = 증분 직접 회귀
         B = 등가속 goal-보간 + 학습 잔차 (잔차 헤드 0-초기화 -> 시작점 = 등가속 prior)
  loss   **cumsum 후** 누적 waypoint에 가중 ADE [11,11,5,5,2,2]/36  (규칙 6)
         Huber 기본 (부록3 항목3: 글리치 꼬리 존재)
  증강   미러 — 궤적·goal·yaw y반전 + cmd L/R 스왑 + 플래그. **U_TURN 프레임 제외**
  게이트 deep-stop v=0.5 / goal=1.0 을 forward에 buffer로 내장 (동결값)
  앙상블 K-시드를 **단일 nn.Module**로 래핑, 출력 평균, 단일 ckpt

    python scripts/etri_priornet.py --arch gru --param B --mirror 1
    python scripts/etri_priornet.py --sweep          # 표 전체
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from challenge_metrics import waypoint_weights            # noqa: E402
from etri_table import COLUMNS, ValSet, fmt, render, residual  # noqa: E402

CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"
GLITCH = "/tmp/pm97/data/etri/glitch_flags.npz"
OUTDIR = "/home/pm97/workspace/sukim/adcl/logs"
CKPT = "/tmp/pm97/ckpt/etri_priornet"

NWP, TS_WP, T_GOAL, DT = 6, 0.5, 5.0, 0.1
TS = np.arange(1, NWP + 1) * TS_WP
STOP_V, STOP_GOAL = 0.5, 1.0          # 동결 (부록4 항목4)
HIS = 31
META_L = ["LANE_KEEP", "TURN_LEFT", "TURN_RIGHT",
          "LANE_CHANGE_L", "LANE_CHANGE_R", "U_TURN"]
# 미러 시 좌우 스왑되는 meta 쌍. U_TURN·LANE_KEEP은 그대로.
META_SWAP = {1: 2, 2: 1, 3: 4, 4: 3}
UTURN = META_L.index("U_TURN")


# --------------------------------------------------------------------- 데이터
def build_feats(d, idx, use_goal_yaw=True, ego_only=False, hist_1s=False,
                frames=None):
    """(N, HIS, F_seq), (N, F_ctx) 두 덩어리. 시퀀스와 문맥을 나눠야 GRU/TF가 쓴다."""
    his = d["his"][idx].astype(np.float32)                 # (N,31,2)
    hy = d["his_yaw"][idx].astype(np.float32)[..., None]   # (N,31,1)
    # 프레임간 증분 = 속도의 대용. 절대 위치만 주면 원점 근처 정보가 압축된다.
    dxy = np.diff(his, axis=1, prepend=his[:, :1])
    dyaw = np.diff(hy, axis=1, prepend=hy[:, :1])
    spd = np.linalg.norm(dxy, axis=2, keepdims=True) / DT
    # 곡률 = yawrate / speed. 저속에서 발산하므로 clip.
    curv = np.clip(dyaw / DT / np.maximum(spd, 0.5), -1.0, 1.0)
    t = (np.arange(HIS, dtype=np.float32) - (HIS - 1))[None, :, None] * DT
    t = np.repeat(t, len(idx), 0)
    seq = np.concatenate([his, hy, dxy, dyaw, spd, curv, t], 2)   # (N,31,9)
    if hist_1s:
        seq = seq[:, -11:, :]
    if frames is not None:
        # 공지 4조: "과거 정보를 활용하는 모든 프레임에 대해 해당 프레임의 영상이 함께
        # 입력되어야". ego 프레임 수 = 영상 프레임 수이므로 여기를 줄이면 카메라 비용이
        # 그대로 줄어든다. his 인덱스 0 = -30, 30 = 0.
        seq = seq[:, [f + 30 for f in frames], :]

    goal = d["goal"][idx].astype(np.float32)
    gd = np.linalg.norm(goal, axis=1, keepdims=True)
    gb = np.arctan2(goal[:, 1:2], goal[:, 0:1])
    gy = d["goal_yaw"][idx].astype(np.float32)[:, None]
    if not use_goal_yaw:
        gy = np.zeros_like(gy)
    if ego_only:
        # 제출 #1 위장 후보 (a2): goal 계열을 전부 0으로. 열 위치는 유지해야
        # 정규화·미러 인덱스가 어긋나지 않는다.
        goal = np.zeros_like(goal)
        gd = np.zeros_like(gd)
        gb = np.zeros_like(gb)
        gy = np.zeros_like(gy)
    vc = np.eye(3, dtype=np.float32)[np.clip(d["vad_cmd"][idx].astype(int), 0, 2)]
    mt = d["meta"][idx].astype(int)
    om = np.zeros((len(idx), 6), np.float32)
    ok = mt >= 0
    om[np.arange(len(idx))[ok], mt[ok]] = 1.0
    ctx = np.concatenate([
        goal, gd, gb, gy,
        d["vel"][idx], d["acc"][idx], d["yawrate"][idx][:, None],
        d["speed"][idx][:, None], vc, om,
        np.zeros((len(idx), 1), np.float32)], 1)            # 마지막 = 미러 플래그
    return seq.astype(np.float32), ctx.astype(np.float32)


def mirror(seq, ctx, gt, meta, mode="all", goal_yaw=None):
    """y 부호 반전 + cmd 좌우 스왑 + 플래그. U_TURN 샘플은 반전하지 않는다.

    빠뜨리면 안 되는 것: 궤적 y, yaw, goal y, goal 방위각, goal yaw, 속도/가속 y,
    yawrate, 곡률, vad_cmd, meta. 하나라도 남으면 오염 데이터다.
    """
    m = meta != UTURN
    if mode == "norot":
        # 부록5 항목6 옵션: 회전 프레임을 미러 대상에서 제외. 회전 구간은 좌우가
        # 기하적으로 대칭이 아닐 수 있다(교차로 형상·차선 수). |goal yaw|로 가른다.
        gy = ctx[:, 4] if goal_yaw is None else goal_yaw
        m = m & (np.abs(gy) < 0.1)
    s, c, g = seq.copy(), ctx.copy(), gt.copy()
    # seq: [x, y, yaw, dx, dy, dyaw, spd, curv, t]
    for j in (1, 2, 4, 5, 7):
        s[m, :, j] *= -1
    # ctx: [gx, gy, gd, gb, gyaw, vx, vy, ax, ay, yawrate, speed, vc(3), meta(6), flag]
    for j in (1, 3, 4, 6, 8, 9):
        c[m, j] *= -1
    vc = c[:, 11:14].copy()
    c[m, 11], c[m, 12] = vc[m, 1], vc[m, 0]          # right <-> left
    om = c[:, 14:20].copy()
    for a, b in META_SWAP.items():
        c[m, 14 + a] = om[m, b]
    c[m, -1] = 1.0
    g[m, :, 1] *= -1
    return s[m], c[m], g[m]


# ---------------------------------------------------------------------- 모델
class Backbone(nn.Module):
    def __init__(self, fs, fc, arch, d=192):
        super().__init__()
        self.arch = arch
        self.tok = nn.Sequential(nn.Linear(fs, d), nn.GELU(), nn.Linear(d, d))
        self.ctx = nn.Sequential(nn.Linear(fc, d), nn.GELU(), nn.Linear(d, d))
        if arch == "gru":
            self.enc = nn.GRU(d, d, num_layers=2, batch_first=True)
        elif arch == "transformer":
            layer = nn.TransformerEncoderLayer(d, 4, d * 4, dropout=0.0,
                                               batch_first=True, norm_first=True,
                                               activation="gelu")
            self.enc = nn.TransformerEncoder(layer, 2)
            self.plan = nn.Parameter(torch.zeros(1, 1, d))
            # 히스토리 길이가 가변(a3는 11프레임)이므로 여유를 둔다
            self.pos = nn.Parameter(torch.zeros(1, HIS + 4, d))
            nn.init.normal_(self.plan, std=0.02)
            nn.init.normal_(self.pos, std=0.02)
        elif arch == "mlp":
            self.enc = None
        else:
            raise ValueError(arch)
        self.head = nn.Sequential(nn.LayerNorm(d * 2), nn.Linear(d * 2, d),
                                  nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.dim = d

    def forward(self, seq, ctx):
        c = self.ctx(ctx)
        if self.arch == "mlp":
            h = self.tok(seq).mean(1)
        elif self.arch == "gru":
            h, _ = self.enc(self.tok(seq))
            h = h[:, -1]
        else:
            t = self.tok(seq)
            x = torch.cat([t, c[:, None], self.plan.expand(len(t), -1, -1)], 1)
            x = x + self.pos[:, : x.shape[1]]
            h = self.enc(x)[:, -1]
        return self.head(torch.cat([h, c], -1))


class PriorNet(nn.Module):
    """K개 시드를 **하나의 Module**로. 개별 ckpt 금지(4탄 함정 목록)."""

    def __init__(self, fs, fc, arch="gru", param="B", K=1, d=192, gate=True):
        super().__init__()
        self.param, self.K, self.gate = param, K, gate
        self.nets = nn.ModuleList([Backbone(fs, fc, arch, d) for _ in range(K)])
        self.heads = nn.ModuleList([nn.Linear(d, NWP * 2) for _ in range(K)])
        for h in self.heads:
            if param == "B":
                # 잔차 헤드 0-초기화 -> 학습 시작 시점 출력 = 등가속 prior 그대로.
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)
        self.register_buffer("ts", torch.tensor(TS, dtype=torch.float32))
        self.register_buffer("stop_v", torch.tensor(STOP_V))
        self.register_buffer("stop_goal", torch.tensor(STOP_GOAL))

    def goal_accel(self, ctx):
        """등가속 prior. ctx의 [gx,gy]와 [vx,vy]로 forward 안에서 계산."""
        goal = ctx[:, 0:2]
        vel = ctx[:, 5:7]
        a = 2.0 * (goal - vel * T_GOAL) / T_GOAL ** 2
        t = self.ts[None, :, None]
        return vel[:, None] * t + 0.5 * a[:, None] * t ** 2

    def forward(self, seq, ctx, seq_raw, ctx_raw):
        """seq/ctx는 정규화 입력, *_raw는 물리 단위.

        goal_accel(변형 B)과 deep-stop 게이트는 미터·m/s를 봐야 하므로 원시값을
        따로 받는다. 정규화값으로 임계를 재면 조용히 틀린 게이트가 된다.
        """
        out = 0.0
        for net, head in zip(self.nets, self.heads):
            inc = head(net(seq, ctx)).view(-1, NWP, 2)
            out = out + inc.cumsum(1)          # 규칙 6: 증분 -> 누적
        out = out / self.K
        if self.param == "B":
            out = out + self.goal_accel(ctx_raw)
        if not self.gate:
            # 규칙 기반 게이트를 끄면 정지 판단도 네트워크가 배워야 한다.
            return out
        spd = seq_raw[:, :, 6]
        gd = ctx_raw[:, 2]
        g = ((spd < self.stop_v).all(1) & (gd < self.stop_goal))
        return torch.where(g[:, None, None], torch.zeros_like(out), out)


# ---------------------------------------------------------------------- 학습
def wl2_torch(pred, gt, w, huber=True, delta=0.5):
    """cumsum 이후 누적 waypoint에 적용 (규칙 6). Huber는 거리에 건다."""
    dist = torch.sqrt(((pred - gt) ** 2).sum(-1) + 1e-12)
    if huber:
        dist = torch.where(dist < delta, 0.5 * dist ** 2 / delta,
                           dist - 0.5 * delta)
    return (dist * w).sum(-1).mean()


def run(cfg, v, d, sp, dev):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    tr = sp["train_idx"]
    if cfg["drop_glitch"]:
        g = np.load(GLITCH)["glitch"]
        tr = tr[~g[tr]]
    eo = cfg.get("ego_only", 0)
    h1 = cfg.get("hist_1s", 0)
    fr = cfg.get("frames", None)
    seq_t, ctx_t = build_feats(d, tr, cfg["goal_yaw"], eo, h1, fr)
    gt_t = d["fut"][tr].astype(np.float32)
    meta_t = d["meta"][tr].astype(int)
    if cfg["mirror"]:
        ms, mc, mg = mirror(seq_t, ctx_t, gt_t, meta_t,
                            cfg.get("mirror_mode", "all"),
                            d["goal_yaw"][tr].astype(np.float32))
        seq_t = np.concatenate([seq_t, ms])
        ctx_t = np.concatenate([ctx_t, mc])
        gt_t = np.concatenate([gt_t, mg])

    seq_v, ctx_v = build_feats(d, v.vi, cfg["goal_yaw"], eo, h1, fr)
    seq_m, ctx_m = build_feats(d, sp["minival_idx"], cfg["goal_yaw"], eo, h1, fr)
    gt_m = d["fut"][sp["minival_idx"]].astype(np.float64)

    mu_s, sd_s = seq_t.reshape(-1, seq_t.shape[-1]).mean(0), \
        seq_t.reshape(-1, seq_t.shape[-1]).std(0) + 1e-6
    mu_c, sd_c = ctx_t.mean(0), ctx_t.std(0) + 1e-6

    def T(a, mu, sd):
        return torch.tensor((a - mu) / sd, device=dev)

    xs, xc = T(seq_t, mu_s, sd_s), T(ctx_t, mu_c, sd_c)
    ys = torch.tensor(gt_t, device=dev)
    vs, vc_ = T(seq_v, mu_s, sd_s), T(ctx_v, mu_c, sd_c)
    ms_, mc_ = T(seq_m, mu_s, sd_s), T(ctx_m, mu_c, sd_c)
    # goal_accel과 deep-stop 게이트는 **원시 스케일**을 봐야 한다 -> 별도로 전달
    raw_c = torch.tensor(ctx_t, device=dev)
    raw_s = torch.tensor(seq_t, device=dev)
    raw_vc, raw_vs = torch.tensor(ctx_v, device=dev), torch.tensor(seq_v, device=dev)
    raw_mc, raw_ms = torch.tensor(ctx_m, device=dev), torch.tensor(seq_m, device=dev)

    net = PriorNet(xs.shape[-1], xc.shape[-1], cfg["arch"], cfg["param"],
                   cfg["K"], gate=bool(cfg.get("gate", 1))).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, cfg["lr"], total_steps=cfg["epochs"] * max(1, len(xs) // cfg["bs"]),
        pct_start=0.2)
    w = torch.tensor(waypoint_weights().astype(np.float32), device=dev)
    yv = torch.tensor(v.gt.astype(np.float32), device=dev)

    best, best_state = 1e9, None
    for ep in range(cfg["epochs"]):
        net.train()
        perm = torch.randperm(len(xs), device=dev)
        for i in range(0, len(xs) - cfg["bs"] + 1, cfg["bs"]):
            b = perm[i:i + cfg["bs"]]
            p = net(xs[b], xc[b], raw_s[b], raw_c[b])
            loss = wl2_torch(p, ys[b], w, cfg["huber"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            pv = net(vs, vc_, raw_vs, raw_vc)
            vl = wl2_torch(pv, yv, w, False).item()
        if vl < best:
            best = vl
            best_state = copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(vs, vc_, raw_vs, raw_vc).cpu().numpy().astype(np.float64)
        pm = net(ms_, mc_, raw_ms, raw_mc).cpu().numpy().astype(np.float64)
    return net, pred, pm, gt_m, (mu_s, sd_s, mu_c, sd_c)


def affine_fit(pred_m, gt_m):
    """시점별·축별 affine (6×2 각각 scale·bias). mini-val에서만 적합 (규칙 3)."""
    A = np.zeros((NWP, 2, 2))
    for t in range(NWP):
        for a in range(2):
            x, y = pred_m[:, t, a], gt_m[:, t, a]
            X = np.stack([x, np.ones_like(x)], 1)
            A[t, a] = np.linalg.lstsq(X, y, rcond=None)[0]
    return A


def affine_apply(pred, A):
    out = pred.copy()
    for t in range(NWP):
        for a in range(2):
            out[:, t, a] = A[t, a, 0] * pred[:, t, a] + A[t, a, 1]
    return out


# ---------------------------------------------------------------- 스윕/보고
# 부록5 항목2: Huber 기본 철회 -> L2 기본. 실측이 부록3 지시를 뒤집었다.
DEFAULT = dict(arch="gru", param="B", mirror=1, mirror_mode="all", goal_yaw=1,
               drop_glitch=0, huber=0, ego_only=0, hist_1s=0, frames=None,
               gate=1, K=1, seed=0,
               epochs=40, bs=256, lr=2e-3)


def cfg_of(**kw):
    c = dict(DEFAULT)
    c.update(kw)
    return c


def label(c):
    parts = [f"{c['arch']}/{c['param']}"]
    parts.append("+mirror" if c["mirror"] else "-mirror")
    if not c["goal_yaw"]:
        parts.append("-goalyaw")
    if c["drop_glitch"]:
        parts.append("-glitch")
    if c["huber"]:
        parts.append("Huber")
    if c.get("mirror_mode", "all") == "norot":
        parts.append("norot")
    if c.get("ego_only"):
        parts.append("EGO-ONLY")
    if c.get("hist_1s"):
        parts.append("HIST1S")
    if c.get("frames") is not None:
        parts.append(f"F{len(c['frames'])}")
    if not c.get("gate", 1):
        parts.append("-gate")
    if c["K"] > 1:
        parts.append(f"K={c['K']}")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--kcurve", action="store_true")
    ap.add_argument("--extras", action="store_true")
    ap.add_argument("--compliance", action="store_true")
    ap.add_argument("--tag", default="priornet")
    ap.add_argument("--arch", default="gru")
    ap.add_argument("--param", default="B")
    ap.add_argument("--mirror", type=int, default=1)
    ap.add_argument("--goal-yaw", type=int, default=1)
    ap.add_argument("--drop-glitch", type=int, default=0)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--huber", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(CACHE, allow_pickle=True)
    sp = np.load(SPLIT, allow_pickle=True)
    v = ValSet()
    os.makedirs(CKPT, exist_ok=True)

    if args.grid:
        # 부록5 항목3: {mlp,gru,tf} x {A,B} x {Huber,L2} x {glitch 유/무} = 24셀
        grid = [cfg_of(arch=a, param=p_, huber=h, drop_glitch=g,
                       epochs=args.epochs)
                for a in ("mlp", "gru", "transformer")
                for p_ in ("A", "B")
                for h in (0, 1)
                for g in (0, 1)]
    elif args.kcurve:
        # 부록5 항목4: 승자에 K 곡선. 단일 Module·단일 ckpt 유지
        base = json.load(open(f"{OUTDIR}/phase1_grid.json")) if \
            os.path.exists(f"{OUTDIR}/phase1_grid.json") else None
        if base:
            w = min(base, key=lambda e: e["가중"])["cfg"]
        else:
            w = cfg_of(arch="transformer", param="B", huber=0)
        grid = [cfg_of(**{**w, "K": k, "epochs": args.epochs})
                for k in (1, 4, 8, 16)]
    elif args.compliance:
        # 2026-08-18 공지 대응 측정
        #   gate 제거 비용 / ego 프레임 축소 손실 / 변형 B 은퇴 비용
        F7 = [-30, -25, -20, -15, -10, -5, 0]     # 베이스라인 STREAM_FRAMES
        F3 = [-10, -5, 0]
        base = dict(arch="transformer", mirror=1, goal_yaw=1, drop_glitch=1,
                    huber=0, epochs=args.epochs)
        grid = [
            cfg_of(**base, param="B", gate=1),                 # 현행 기준
            cfg_of(**base, param="B", gate=0),                 # 게이트 제거
            cfg_of(**base, param="A", gate=0),                 # 변형 B 은퇴
            cfg_of(**base, param="A", gate=0, frames=F7),      # 7프레임
            cfg_of(**base, param="A", gate=0, frames=F3),      # 3프레임
            cfg_of(**base, param="B", gate=0, frames=F3),      # 참고: B+3프레임
        ]
    elif args.extras:
        # 항목6 미러 변형 + 항목8 위장 후보 (a2)
        base = dict(arch="transformer", param="B", huber=0)
        grid = [cfg_of(**base, mirror_mode="norot", epochs=args.epochs),
                cfg_of(**base, mirror=0, epochs=args.epochs),
                cfg_of(arch="transformer", param="A", huber=0, ego_only=1,
                       goal_yaw=0, epochs=args.epochs),
                cfg_of(arch="gru", param="A", huber=0, ego_only=1,
                       goal_yaw=0, epochs=args.epochs),
                cfg_of(arch="mlp", param="A", huber=0, ego_only=1, goal_yaw=0,
                       hist_1s=1, mirror=1, epochs=args.epochs),
                cfg_of(arch="gru", param="A", huber=0, ego_only=1, goal_yaw=0,
                       hist_1s=1, mirror=1, epochs=args.epochs)]
    elif args.sweep:

        grid = []
        # 1) 구조 × 파라미터화 (미러 on, goal yaw on)
        for a in ("mlp", "gru", "transformer"):
            for p in ("A", "B"):
                grid.append(cfg_of(arch=a, param=p, epochs=args.epochs))
        # 2) 승자 축 절제는 스윕 후 결정 -> 여기서는 gru/B 기준 절제를 함께 돌린다
        grid += [cfg_of(mirror=0, epochs=args.epochs),
                 cfg_of(goal_yaw=0, epochs=args.epochs),
                 cfg_of(drop_glitch=1, epochs=args.epochs),
                 cfg_of(huber=0, epochs=args.epochs)]
    else:
        grid = [cfg_of(arch=args.arch, param=args.param, mirror=args.mirror,
                       goal_yaw=args.goal_yaw, drop_glitch=args.drop_glitch,
                       huber=args.huber, K=args.K, epochs=args.epochs)]

    ledger, rows = [], [("**라벨 모호성 프록시**", v.label_proxy_row())]
    for c in grid:
        t0 = time.time()
        net, pred, pm, gt_m, nrm = run(c, v, d, sp, dev)
        A = affine_fit(pm, gt_m)
        pred_cal = affine_apply(pred, A)
        r, rc = v.row(pred), v.row(pred_cal)
        dt = time.time() - t0
        nm = label(c)
        rows.append((nm, r))
        rows.append((nm + " +affine", rc))
        ledger.append(dict(cfg=c, name=nm, sec=round(dt, 1),
                           params=int(sum(p.numel() for p in net.parameters())),
                           **{k: r[k] for k in COLUMNS},
                           **{"cal_" + k: rc[k] for k in COLUMNS}))
        print(f"{nm:<26} {dt:5.0f}s  " +
              " ".join(f"{c2}={fmt(r[c2])}" for c2 in COLUMNS), flush=True)
        # mu/sd를 빼면 ckpt만으로 추론이 불가능하다 (부록5 1b에서 발견).
        torch.save({"state": net.state_dict(), "cfg": c, "affine": A,
                    "norm": {k: np.asarray(x) for k, x in
                             zip(("mu_s", "sd_s", "mu_c", "sd_c"), nrm)}},
                   f"{CKPT}/{nm.replace('/', '_').replace(' ', '')}.pt")

    md = ["# Phase 1 — PriorNet v1\n\n",
          render(rows, note="`+affine`은 mini-val에서 적합한 시점별·축별 affine을 적용한 값이다 "
                            "(규칙 3: val에서 적합 금지).\n"),
          "\n## 실험 대장\n\n| 설정 | params | 초 | " + " | ".join(COLUMNS) + " |\n",
          "|" + "---|" * (len(COLUMNS) + 3) + "\n"]
    for e in ledger:
        md.append(f"| {e['name']} | {e['params']:,} | {e['sec']:.0f} | "
                  + " | ".join(fmt(e[c]) for c in COLUMNS) + " |\n")
    open(f"{OUTDIR}/phase1_{args.tag}.md", "w").write("".join(md))
    json.dump(ledger, open(f"{OUTDIR}/phase1_{args.tag}.json", "w"),
              indent=1, ensure_ascii=False)
    print(f"\n저장 {OUTDIR}/phase1_{args.tag}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
