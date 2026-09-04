#!/usr/bin/env python
"""Phase D — prior 바닥 측정. 무학습 5종 + prior-MLP.

모든 L2는 재작성된 `src/challenge_metrics.py`(단위테스트 4건 통과본)로만 계산한다.
즉 "1·2·3초 누적 ADE 3개의 평균" 하나뿐이고, 이는 SparseDrive/VAD 리포트값과 같은 단위다.

분해 축 (부록2 항목 2)
    기본: vad_cmd(right/left/straight) × 이동상태(정지 |v|<0.5 m/s vs 이동)
    부표: meta command 6종
`vad_cmd right` 부분집합 L2는 **항상 병기**한다 -- 앞으로 모든 모델 비교의 1차 지표다.

지표 2종 (부록2 항목 1)
    (a) 무가중  (b) test 정합 가중  ← (b)가 리더보드 예측기

prior 정의
    CV        p(t) = v t
    CTRV      등속·등yaw. 초기 헤딩 오프셋 atan2(vy,vx)까지 반영한다 -- ego 프레임에서
              v가 정확히 +x는 아니고, 무시하면 저속에서 횡오차가 생긴다
    goal-등가속   p(T)=goal 을 만족하는 등가속:  a = 2(goal - vT)/T²
    goal-에르미트 3차 에르미트. 끝단 접선은 |v|·unit(goal) -- +50 시점 헤딩을 모르므로
              목표 방향으로 가정한다. 이 가정이 이 prior의 한계다
    블렌드     종방향(x)=CTRV, 횡방향(y)=에르미트

    python scripts/etri_priors.py                 # 무학습만
    python scripts/etri_priors.py --mlp           # prior-MLP 학습 포함
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from challenge_metrics import (l2_challenge, l2_from_per_step,  # noqa: E402
                              waypoint_weights)

CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
CACHE_TEST = "/tmp/pm97/data/etri/ego_cache_test.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"
PERSIST = "/home/pm97/workspace/sukim/adcl"
OUT_MD = os.path.join(PERSIST, "logs", "baselines_etri.md")
OUT_JSON = os.path.join(PERSIST, "logs", "baselines_etri.json")

DT_WP = 0.5
NWP = 6
T_GOAL = 5.0
TS = np.arange(1, NWP + 1) * DT_WP
STOP_SPEED = 0.5
DEEP_STOP_GOAL = 1.0
VCMD = ["right", "left", "straight"]


def cv(vel, *_):
    return vel[:, None, :] * TS[None, :, None]


def ctrv(vel, yawrate, *_):
    s = np.linalg.norm(vel, axis=1)
    th0 = np.arctan2(vel[:, 1], vel[:, 0])
    w = yawrate
    small = np.abs(w) < 1e-4
    t = TS[None, :]
    wsafe = np.where(small, 1.0, w)[:, None]
    xs = np.where(small[:, None], s[:, None] * t,
                  s[:, None] / wsafe * np.sin(wsafe * t))
    ys = np.where(small[:, None], 0.0,
                  s[:, None] / wsafe * (1 - np.cos(wsafe * t)))
    c, sn = np.cos(th0)[:, None], np.sin(th0)[:, None]
    out = np.zeros((len(vel), NWP, 2))
    out[..., 0] = c * xs - sn * ys
    out[..., 1] = sn * xs + c * ys
    return out


def goal_accel(vel, yawrate, goal):
    a = 2.0 * (goal - vel * T_GOAL) / T_GOAL ** 2
    t = TS[None, :, None]
    return vel[:, None, :] * t + 0.5 * a[:, None, :] * t ** 2


def goal_hermite(vel, yawrate, goal):
    s = (TS / T_GOAL)[None, :, None]
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    gn = np.linalg.norm(goal, axis=1, keepdims=True)
    unit = goal / np.where(gn < 1e-6, 1.0, gn)
    e = unit * np.linalg.norm(vel, axis=1, keepdims=True)
    return (h10 * (vel * T_GOAL)[:, None, :] + h01 * goal[:, None, :]
            + h11 * (e * T_GOAL)[:, None, :])


def blend(vel, yawrate, goal):
    out = ctrv(vel, yawrate, goal).copy()
    out[..., 1] = goal_hermite(vel, yawrate, goal)[..., 1]
    return out


def zeros_pred(vel, *_):
    return np.zeros((len(vel), NWP, 2))


PRIORS = [("CV (등속)", cv),
          ("CTRV", ctrv),
          ("goal-보간 등가속", goal_accel),
          ("goal-보간 에르미트", goal_hermite),
          ("블렌드 (종=CTRV, 횡=에르미트)", blend),
          ("정지 출력 [0,0]x6", zeros_pred)]


def wl2(pred, gt, w=None):
    """챌린지 L2. w가 있으면 clip 가중 평균을 시점별 평균에 반영한다."""
    if w is None:
        return l2_challenge(pred, gt)["L2_avg"]
    d = np.sqrt(((pred[..., :2] - gt[..., :2]) ** 2).sum(-1))
    if w.sum() <= 0:
        return float("nan")
    per = (d * (w / w.sum())[:, None]).sum(0)
    return l2_from_per_step(per)["L2_avg"]


def subset_row(pred, gt, w, mask):
    if mask.sum() == 0:
        return None, 0
    return (wl2(pred[mask], gt[mask], None if w is None else w[mask]),
            int(mask.sum()))


def breakdown(p, gt, w, vcm, mtm, moving, meta_labels):
    row = {"overall_unw": round(wl2(p, gt), 4), "overall_w": round(wl2(p, gt, w), 4)}
    for ci, cn in enumerate(VCMD):
        for mv, mn in ((True, "이동"), (False, "정지")):
            v, n = subset_row(p, gt, w, (vcm == ci) & (moving == mv))
            row[f"{cn}/{mn}"] = None if v is None else round(v, 4)
            row[f"n_{cn}/{mn}"] = n
        v, n = subset_row(p, gt, w, vcm == ci)
        row[f"{cn}/전체"] = None if v is None else round(v, 4)
        row[f"n_{cn}"] = n
    for mi, ml in enumerate(meta_labels):
        v, n = subset_row(p, gt, w, mtm == mi)
        row[f"meta_{ml}"] = None if v is None else round(v, 4)
        row[f"n_meta_{ml}"] = n
    return row


def past_speeds(his):
    """his는 -30..0의 누적 상대위치. 연속 차분/0.1 = 프레임별 속도 크기."""
    return np.linalg.norm(np.diff(his.astype(np.float64), axis=1), axis=2) / 0.1


def deep_stop_mask(his, goal):
    gd = np.linalg.norm(goal, axis=1)
    return (past_speeds(his) < STOP_SPEED).all(axis=1) & (gd < DEEP_STOP_GOAL)


def make_feats(d, idx, hist):
    his = d["his"][idx].astype(np.float32)
    h = (his[:, [20, 25, 30], :] if hist == "sparse" else his).reshape(len(idx), -1)
    vcm = d["vad_cmd"][idx].astype(int)
    mtm = d["meta"][idx].astype(int)
    oh_v = np.eye(3, dtype=np.float32)[np.clip(vcm, 0, 2)]
    oh_m = np.zeros((len(idx), 6), np.float32)
    ok = mtm >= 0
    oh_m[np.arange(len(idx))[ok], mtm[ok]] = 1.0
    return np.concatenate([h, d["vel"][idx], d["acc"][idx],
                           d["yawrate"][idx][:, None], d["goal"][idx],
                           oh_v, oh_m], axis=1).astype(np.float32)


def train_mlp(d, ti_, vi, gt, w, vcm, mtm, moving, epochs, hist, meta_labels):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    Xtr = make_feats(d, ti_, hist)
    Ytr = d["fut"][ti_].astype(np.float32)
    Xva = make_feats(d, vi, hist)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    xt = torch.tensor((Xtr - mu) / sd, device=dev)
    xv = torch.tensor((Xva - mu) / sd, device=dev)
    ytr = torch.tensor(Ytr, device=dev)
    yva = torch.tensor(gt.astype(np.float32), device=dev)
    wgt = torch.tensor(waypoint_weights().astype(np.float32), device=dev)

    # 증분으로 회귀한 뒤 cumsum. 누적을 직접 회귀하면 후반 시점 스케일이 커서 초반
    # waypoint 오차가 손실에 거의 반영되지 않는다.
    net = nn.Sequential(nn.Linear(xt.shape[1], 256), nn.ReLU(),
                        nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, NWP * 2)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs, best, best_state = 512, 1e9, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]
            cum = net(xt[b]).view(-1, NWP, 2).cumsum(1)
            loss = (torch.sqrt(((cum - ytr[b]) ** 2).sum(-1) + 1e-12)
                    * wgt).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            pv = net(xv).view(-1, NWP, 2).cumsum(1)
            vl = (torch.sqrt(((pv - yva) ** 2).sum(-1) + 1e-12)
                  * wgt).sum(-1).mean().item()
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"  MLP ep{ep+1:3d}  train {loss.item():.4f}  val {vl:.4f}",
                  flush=True)
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p = net(xv).view(-1, NWP, 2).cumsum(1).cpu().numpy().astype(np.float64)
    row = breakdown(p, gt, w, vcm, mtm, moving, meta_labels)
    row.update(hist=hist, epochs=epochs, n_features=int(xt.shape[1]),
               device=dev, best_val_weighted_loss=round(best, 5))
    print(f"\nprior-MLP  무가중 {row['overall_unw']}  가중 {row['overall_w']}"
          f"  right {row['right/전체']}")
    return row


def write_md(out, meta_labels):
    P = dict(out["priors"])
    names = list(out["priors"])
    if "prior_mlp" in out:
        P["prior-MLP (학습)"] = out["prior_mlp"]
        names.append("prior-MLP (학습)")

    def c(v):
        return "—" if v is None else f"{v:.4f}"

    L = ["# ETRI 챌린지 — prior 바닥 (로컬 val)\n\n"]
    L.append(f"- val clip **{out['n_val_clips']:,}** (홀드아웃 38 시나리오, 2Hz 앵커), "
             f"train 앵커 {out['n_train_anchors']:,}\n")
    L.append(f"- test 정합 가중의 유효표본 **{out['weights_effective_n']}**\n")
    L.append(f"- 모든 값은 `src/challenge_metrics.py`의 챌린지 L2 (1·2·3초 누적 ADE 평균, "
             f"waypoint 가중 {out['waypoint_weights']}/36). 단위테스트 4건 통과본.\n")
    L.append("- **가중 열이 리더보드 예측기다.** 무가중은 val 자체의 난이도.\n")
    L.append("- **`right` 열이 1차 지표다** (부록2 항목 3).\n")
    L.append("- 참고: VAD tiny 리더보드 0.561, 우리 SparseDrive stage2(nuScenes) 0.5939 "
             "— 같은 단위다.\n")

    L.append("\n## 주표 — vad_cmd × 이동상태\n\n")
    L.append("| prior | 전체(무가중) | **전체(가중)** | **right 전체** | right/이동 | right/정지 "
             "| left 전체 | left/이동 | left/정지 | straight 전체 | straight/이동 | straight/정지 |\n")
    L.append("|" + "---|" * 12 + "\n")
    for n in names:
        r = P[n]
        L.append(f"| {n} | {c(r['overall_unw'])} | **{c(r['overall_w'])}** | "
                 f"**{c(r['right/전체'])}** | {c(r['right/이동'])} | {c(r['right/정지'])} | "
                 f"{c(r['left/전체'])} | {c(r['left/이동'])} | {c(r['left/정지'])} | "
                 f"{c(r['straight/전체'])} | {c(r['straight/이동'])} | {c(r['straight/정지'])} |\n")
    b = out["priors"][names[0]]
    L.append("\n표본 수 — " + ", ".join(
        f"{cn} {b[f'n_{cn}']} (이동 {b[f'n_{cn}/이동']} / 정지 {b[f'n_{cn}/정지']})"
        for cn in VCMD) + "\n")

    L.append("\n## 부표 — meta command 6종 (무가중)\n\n")
    L.append("| prior | " + " | ".join(meta_labels) + " |\n")
    L.append("|" + "---|" * (len(meta_labels) + 1) + "\n")
    for n in names:
        L.append(f"| {n} | " + " | ".join(c(P[n].get(f"meta_{m}"))
                                          for m in meta_labels) + " |\n")
    L.append("\n표본 수 — " + ", ".join(
        f"{m} {b[f'n_meta_{m}']}" for m in meta_labels) + "\n")

    ds = out["deep_stop"]
    L.append("\n## deep-stop (관측 기준)\n\n")
    L.append(f"기준: {ds['criterion']}\n\n")
    L.append(f"- **test {ds['test_n']} / {ds['test_total']} clip = "
             f"{ds['test_ratio']*100:.2f}%**\n")
    L.append(f"- val {ds['val_n']} / {ds['val_total']} = {ds['val_ratio']*100:.2f}%\n")
    if "val_zero_output_L2" in ds:
        L.append(f"- val deep-stop 부분집합에서 `[0,0]x6` 고정 출력 L2 = "
                 f"**{ds['val_zero_output_L2']}** (같은 집합 CV = {ds['val_CV_L2']})\n")
        L.append(f"- 같은 집합 GT 3초 변위 p95 = {ds['val_gt_disp_p95']} m\n")
    L.append(f"- {ds['note']}\n")
    L.append(f"- test deep-stop의 vad_cmd 분포: {ds['test_vad_cmd_dist']}\n")
    open(OUT_MD, "w").write("".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hist", choices=["sparse", "full"], default="full")
    args = ap.parse_args()

    d = np.load(CACHE, allow_pickle=True)
    t = np.load(CACHE_TEST, allow_pickle=True)
    sp = np.load(SPLIT, allow_pickle=True)
    vi, ti_ = sp["val_idx"], sp["train_idx"]
    w = sp["val_weight"].astype(np.float64)
    meta_labels = list(d["meta_labels"])

    gt = d["fut"][vi].astype(np.float64)
    vel = d["vel"][vi].astype(np.float64)
    yr = d["yawrate"][vi].astype(np.float64)
    goal = d["goal"][vi].astype(np.float64)
    spd = d["speed"][vi].astype(np.float64)
    vcm = d["vad_cmd"][vi].astype(int)
    mtm = d["meta"][vi].astype(int)
    moving = spd >= STOP_SPEED

    results, preds = {}, {}
    for name, fn in PRIORS:
        p = fn(vel, yr, goal)
        preds[name] = p
        results[name] = breakdown(p, gt, w, vcm, mtm, moving, meta_labels)
        r = results[name]
        print(f"{name:30s} 무가중 {r['overall_unw']:.4f}  가중 {r['overall_w']:.4f}"
              f"  right {r['right/전체']}", flush=True)

    ds_test = deep_stop_mask(t["his"], t["goal"].astype(np.float64))
    ds_val = deep_stop_mask(d["his"][vi], goal)
    deep = {
        "criterion": f"과거 31프레임 전부 |v|<{STOP_SPEED} m/s AND |goal|<{DEEP_STOP_GOAL} m",
        "test_n": int(ds_test.sum()), "test_total": int(len(ds_test)),
        "test_ratio": round(float(ds_test.mean()), 5),
        "val_n": int(ds_val.sum()), "val_total": int(len(ds_val)),
        "val_ratio": round(float(ds_val.mean()), 5),
        "note": "L2는 GT가 있는 val 부분집합에서만 계산 가능 (test GT 비공개)",
        "test_vad_cmd_dist": {VCMD[i]: int((t["vad_cmd"][ds_test] == i).sum())
                              for i in range(3)},
    }
    if ds_val.sum():
        deep["val_zero_output_L2"] = round(wl2(zeros_pred(vel)[ds_val], gt[ds_val]), 5)
        deep["val_CV_L2"] = round(wl2(preds["CV (등속)"][ds_val], gt[ds_val]), 5)
        deep["val_gt_disp_p95"] = round(float(np.percentile(
            np.linalg.norm(gt[ds_val][:, -1, :], axis=1), 95)), 4)
    print("\ndeep-stop:", json.dumps(deep, ensure_ascii=False))

    out = {"n_val_clips": int(len(vi)), "n_train_anchors": int(len(ti_)),
           "weights_effective_n": round(float(w.sum() ** 2 / (w ** 2).sum()), 1),
           "waypoint_weights": (waypoint_weights() * 36).round().astype(int).tolist(),
           "priors": results, "deep_stop": deep}
    if args.mlp:
        out["prior_mlp"] = train_mlp(d, ti_, vi, gt, w, vcm, mtm, moving,
                                     args.epochs, args.hist, meta_labels)
    json.dump(out, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    write_md(out, meta_labels)
    print(f"\n저장: {OUT_JSON}\n      {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
