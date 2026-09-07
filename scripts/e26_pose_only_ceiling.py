#!/usr/bin/env python
"""과거 pose 만으로 학습한 모델의 상한 — 리더보드 0.0924 의 출처 진단.

**금지 구성이다.** 제출 계획이 아니라 경쟁 환경을 이해하기 위한 진단이다.
공지1 3-2/3-3 은 과거 상태만으로 궤적을 산출하는 것을 금지한다(신경망 형태 포함).

train330 의 과거 3초 pose(31점) + goal 로 미래 6 waypoint 를 회귀 학습하고
val38 로 평가한다. 이 값이 0.09~0.13 대면 상위권은 코드 심사 탈락 대상이고,
0.20 을 못 내려가면 상위권은 진짜 영상 모델을 가진 것이다.
"""
import os
import sys

import numpy as np

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402


def official_l2(pred, gt, w):
    e = np.linalg.norm(pred - gt, axis=-1)
    l1 = e[:, :2].mean(1); l2 = e[:, :4].mean(1); l3 = e[:, :6].mean(1)
    return float(np.average((l1 + l2 + l3) / 3.0, weights=w))


def make_X(his, yaw, now, g5=None):
    """과거 3초 pose 를 현재 기준 상대좌표로. goal 포함 여부 선택."""
    p = his[:, :now + 1] - his[:, now:now + 1]                 # [N,31,2]
    y = yaw[:, :now + 1] - yaw[:, now:now + 1]
    v = np.diff(his[:, ::2], axis=1) / 0.2                     # 0.2s 간격 속도
    f = [p.reshape(len(p), -1), y, v.reshape(len(v), -1)]
    if g5 is not None:
        d = np.linalg.norm(g5, axis=1, keepdims=True)
        f += [g5, d, g5 / np.clip(d, 1e-6, None)]
    return np.concatenate(f, 1)


def fit_ridge(X, Y, lam=1e-3):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    G = Xb.T @ Xb + lam * len(X) * np.eye(Xb.shape[1])
    return np.linalg.solve(G, Xb.T @ Y)


def pred_ridge(Wc, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ Wc


def main():
    arr = C.load_arrays()
    now = C.HIS_NOW
    s = C.make_split(arr, all_frames=True)
    scen = arr["scen_idx"]
    ts = np.unique(scen[s["train_rows"]])
    tr = np.where(np.isin(scen, ts) & (arr["frame"] >= 30))[0]
    vr = arr["val_idx"]
    vsub = arr["frame"][vr] >= 30
    vr = vr[vsub]
    vw = arr["val_weight"][vsub].astype(np.float64)
    print(f"train {len(tr)}행 / val38 {len(vr)}행\n")

    def get(rows):
        return (arr["his"][rows].astype(np.float64), arr["his_yaw"][rows].astype(np.float64),
                arr["fut"][rows].astype(np.float64), arr["fut5"][rows][:, 9].astype(np.float64))
    ht, yt, Gt, gt5 = get(tr)
    hv, yv, Gv, gv5 = get(vr)

    print(f"{'모델':>38} {'L2 avg':>9}")
    print(f"{'등가속 외삽 (규칙, 대조)':>38} {0.2442:>9.4f}")

    for tag, use_goal in (("과거 pose only", False), ("과거 pose + goal", True)):
        Xt = make_X(ht, yt, now, gt5 if use_goal else None)
        Xv = make_X(hv, yv, now, gv5 if use_goal else None)
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        Xt = (Xt - mu) / sd; Xv = (Xv - mu) / sd
        Yt = Gt.reshape(len(Gt), -1)
        for lam in (1e-4, 1e-3, 1e-2):
            Wc = fit_ridge(Xt, Yt, lam)
            P = pred_ridge(Wc, Xv).reshape(-1, 6, 2)
            print(f"{f'ridge {tag} (lam={lam})':>38} {official_l2(P, Gv, vw):>9.4f}")

    # MLP (비선형)
    import torch
    import torch.nn as nn
    dev = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    for tag, use_goal in (("과거 pose only", False), ("과거 pose + goal", True)):
        Xt = make_X(ht, yt, now, gt5 if use_goal else None)
        Xv = make_X(hv, yv, now, gv5 if use_goal else None)
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        Xt = ((Xt - mu) / sd).astype(np.float32); Xv = ((Xv - mu) / sd).astype(np.float32)
        Yt = Gt.reshape(len(Gt), -1).astype(np.float32)
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(Xt.shape[1], 512), nn.GELU(), nn.LayerNorm(512),
                            nn.Linear(512, 512), nn.GELU(), nn.LayerNorm(512),
                            nn.Linear(512, 12)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        Xtt = torch.from_numpy(Xt); Ytt = torch.from_numpy(Yt)
        EP = 60
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP * (len(Xtt) // 1024 + 1))
        for ep in range(EP):
            net.train()
            pm = torch.randperm(len(Xtt))
            for i in range(0, len(pm), 1024):
                b = pm[i:i + 1024]
                loss = nn.functional.smooth_l1_loss(net(Xtt[b].to(dev)), Ytt[b].to(dev), beta=0.1)
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        net.eval()
        with torch.no_grad():
            P = net(torch.from_numpy(Xv).to(dev)).cpu().numpy().reshape(-1, 6, 2)
        print(f"{f'MLP {tag}':>38} {official_l2(P.astype(np.float64), Gv, vw):>9.4f}")

    print(f"\n리더보드 1위 0.0924106 / 3위 0.1375893")
    print(f"우리 image-only (A0+learned selector) 0.2522")
    return 0


if __name__ == "__main__":
    sys.exit(main())
