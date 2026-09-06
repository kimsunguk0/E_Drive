#!/usr/bin/env python
"""⑥-D: coarse-to-fine 을 **모델 parent 순위**로 평가한다.

build_hier_bank 의 top-t 표는 GT 로 parent 를 정렬한 상한이다.
배포에서는 모델 logits 가 parent 를 정렬하므로 그것으로 다시 재야 한다.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402
from train_sparse_scoredrive import TrainableSparseScoreDrive  # noqa: E402

W6 = np.array([11., 11., 5., 5., 2., 2.]); W6 /= W6.sum()


def d3_pair(a, b, chunk=256):
    out = np.empty((len(a), len(b)), np.float32)
    for s in range(0, len(a), chunk):
        d = np.linalg.norm(a[s:s + chunk, None] - b[None], axis=-1)
        out[s:s + chunk] = (d * W6).sum(-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--ckpt", default="work_dirs/sc_r3ctrl_s0/best.pth")
    ap.add_argument("--gpu", type=int, default=6)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    arr = C.load_arrays()
    hb = np.load(os.path.join(A, args.bank))
    kids = hb["candidate_xy_abs_5s"].astype(np.float64)
    kpar = hb["parent_id"]
    K = len(kids)

    ck = torch.load(os.path.join(A, args.ckpt), map_location="cpu")
    a = ck["args"]; nh = int(a.get("n_hist", 3))
    m = TrainableSparseScoreDrive(
        C.BANK_A0, logit_norm=bool(a.get("logit_norm", 1)), n_hist=nh,
        speed_head=bool(a.get("w_speed", 0) > 0), speed_gamma=0.0).to(dev)
    m.load_state_dict(ck["model"]); m.fuse_mul = bool(a.get("fuse_mul", 0)); m.eval()

    rows = arr["val_idx"]
    sub = (arr["frame"][rows] >= 30) & C.history_available(arr, rows, nh)
    rows = rows[sub]
    w = arr["val_weight"][sub].astype(np.float64)
    lg = C.run_logits(m, arr, rows, torch.from_numpy(C.build_global_lidar2img()),
                      dev, batch=24, amp=True, n_hist=nh).astype(np.float64)
    V3 = arr["fut"][rows].astype(np.float64)
    Dc = d3_pair(V3, kids[:, :6])
    Dp_gt = d3_pair(V3, np.load(C.BANK_A0)["anchors_abs"].astype(np.float64))
    wm = lambda x: float(np.average(x, weights=w))                 # noqa: E731

    best_child = Dc.argmin(1)
    need_parent = kpar[best_child]
    rank_model = np.empty(len(rows), np.int32)
    order = np.argsort(-lg, axis=1)
    pos = np.argsort(order, axis=1)                                # parent -> 순위
    for i in range(len(rows)):
        rank_model[i] = pos[i, need_parent[i]]

    print(f"bank={os.path.basename(args.bank)} K={K}  val38 n={len(rows)}")
    print(f"full-bank oracle {wm(Dc.min(1)):.6f}\n")
    print(f"정답 child 의 parent 가 모델 순위 몇 위인가: "
          f"중앙값 {int(np.median(rank_model))}, "
          f"top16 {100*wm((rank_model<16).astype(float)):.1f}%, "
          f"top32 {100*wm((rank_model<32).astype(float)):.1f}%, "
          f"top64 {100*wm((rank_model<64).astype(float)):.1f}%\n")
    print(f"{'parent 순위원':>12} {'GT 정렬(상한)':>14} {'모델 정렬(실제)':>16} {'child 수':>9}")
    res = {"K": int(K), "full": wm(Dc.min(1))}
    for t in (16, 32, 64, 128, 256):
        topm = order[:, :t]
        topg = np.argpartition(Dp_gt, t, axis=1)[:, :t]
        bm = np.empty(len(rows)); bg = np.empty(len(rows)); ncs = []
        for i in range(len(rows)):
            mm = np.isin(kpar, topm[i]); gg = np.isin(kpar, topg[i])
            bm[i] = Dc[i, mm].min() if mm.any() else 1e9
            bg[i] = Dc[i, gg].min() if gg.any() else 1e9
            if i % 20 == 0:
                ncs.append(int(mm.sum()))
        res[f"model_top{t}"] = wm(bm); res[f"gt_top{t}"] = wm(bg)
        print(f"{t:>12} {wm(bg):>14.6f} {wm(bm):>16.6f} {int(np.mean(ncs)):>9}")
    json.dump(res, open(os.path.join(
        A, f"logs/c2f_{os.path.basename(args.bank).replace('.npz','')}.json"), "w"),
        indent=1, default=float)
    print("\n주의: 이 표는 coverage(oracle) 만 본다. shortlist/selector 오차는 별도다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
