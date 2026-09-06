#!/usr/bin/env python
"""⑥-G: 현재 scorer 의 순위상관은 얼마인가 — fine-scorer 요구선 0.55 가 현실적인가.

⑥-F 잡음 스윕: child fine-scorer 순위상관 0.55 -> realized 0.2426 (기준 0.2522 상회),
0.77 -> 0.2096. 그렇다면 0.55 가 달성 가능한 수준인지가 관건이다.
비교 기준으로 현재 모델이 A0 12행 shortlist 안에서 내는 상관을 잰다.
"""
import argparse
import os
import sys

import numpy as np
import torch

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402
from train_sparse_scoredrive import TrainableSparseScoreDrive  # noqa: E402
from e22_bank_realized import d3_pair  # noqa: E402


def spearman(a, b):
    ra = np.argsort(np.argsort(a, 1), 1).astype(np.float64)
    rb = np.argsort(np.argsort(b, 1), 1).astype(np.float64)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / (np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1)) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/sc_r5_offic/best.pth")
    ap.add_argument("--gpu", type=int, default=7)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}"); torch.cuda.set_device(dev)
    arr = C.load_arrays()
    a0 = np.load(C.BANK_A0, allow_pickle=False)
    ck = torch.load(os.path.join(A, args.ckpt), map_location="cpu")
    ar = ck["args"]; nh = int(ar.get("n_hist", 3))
    m = TrainableSparseScoreDrive(
        C.BANK_A0, logit_norm=bool(ar.get("logit_norm", 1)), n_hist=nh,
        score_weight=ar.get("score_weight", "uniform")).to(dev)
    m.load_state_dict(ck["model"]); m.fuse_mul = bool(ar.get("fuse_mul", 0)); m.eval()

    rows = arr["val_idx"]
    sub = (arr["frame"][rows] >= 30) & C.history_available(arr, rows, nh)
    rows = rows[sub]
    w = arr["val_weight"][sub].astype(np.float64)
    lg = C.run_logits(m, arr, rows, torch.from_numpy(C.build_global_lidar2img()),
                      dev, batch=24, amp=True, n_hist=nh).astype(np.float64)
    V3 = arr["fut"][rows].astype(np.float64)
    D = d3_pair(V3, a0["anchors_abs"].astype(np.float64)).astype(np.float64)
    ad, tau = a0["anchor_dist"].astype(np.float64), float(a0["nms_tau"])
    n = len(rows)
    S = np.stack([C._shortlist(lg[i], ad, tau) for i in range(n)])
    wm = lambda x: float(np.average(x, weights=w))                  # noqa: E731

    print(f"ckpt={os.path.basename(os.path.dirname(args.ckpt))}  val38 n={n}\n")
    print("현재 scorer 의 logit vs 참 D3 순위상관 (높을수록 좋음)")
    sl = spearman(np.take_along_axis(lg, S, 1), -np.take_along_axis(D, S, 1))
    print(f"  A0 shortlist 12행 안         {wm(sl):>7.3f}   (중앙값 {np.median(sl):.3f})")
    for k in (32, 64, 128):
        top = np.argsort(-lg, 1)[:, :k]
        sk = spearman(np.take_along_axis(lg, top, 1), -np.take_along_axis(D, top, 1))
        print(f"  모델 상위 {k:<4} 안            {wm(sk):>7.3f}")
    sa = spearman(lg, -D)
    print(f"  전체 1024 안                 {wm(sa):>7.3f}")

    # parent 간 vs parent 내부: child fine-scorer 가 풀어야 할 문제의 난이도
    print("\n같은 D3 대역 안에서의 상관 (child fine-scorer 가 실제로 푸는 문제와 유사)")
    for lo, hi in ((0.0, 0.3), (0.0, 0.5), (0.0, 1.0)):
        sel = []
        for i in range(n):
            idx = np.where((D[i] >= lo) & (D[i] < hi))[0]
            if len(idx) >= 8:
                sel.append(spearman(lg[i][idx][None], -D[i][idx][None])[0])
            else:
                sel.append(np.nan)
        sel = np.array(sel)
        ok = ~np.isnan(sel)
        print(f"  D3 in [{lo},{hi})  n={int(ok.sum()):5d}  상관 {np.average(sel[ok], weights=w[ok]):>7.3f}")
    print("\n⑥-F 요구선: 순위상관 0.55 -> realized 0.2426 (기준 0.2522 상회)")
    print("            순위상관 0.77 -> realized 0.2096")
    return 0


if __name__ == "__main__":
    sys.exit(main())
