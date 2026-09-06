#!/usr/bin/env python
"""⑤-E7-a: learned row selector 학습용 shortlist 덤프.

모델을 train/tune/val 행에 돌려 shortlist 12행과 그 후보의 D3gt, goal, cmd 를 저장한다.
selector 는 이 12행 중 index 하나만 고르므로 compliance 는 현재 규칙과 동일하다.
ego status(speed/acc)는 담지 않는다(운영국 확인 전까지 사용 불가).
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


def load(ckpt, dev):
    ck = torch.load(os.path.join(A, ckpt), map_location="cpu")
    a = ck["args"]
    m = TrainableSparseScoreDrive(
        C.BANK_A0, logit_norm=bool(a.get("logit_norm", 1)),
        n_hist=int(a.get("n_hist", 3)),
        speed_head=bool(a.get("w_speed", 0) > 0), speed_gamma=0.0).to(dev)
    m.load_state_dict(ck["model"])
    m.fuse_mul = bool(a.get("fuse_mul", 0))
    m.merge_encode = bool(a.get("merge_encode", 0))
    m.eval()
    return m, int(a.get("n_hist", 3))


def pick_rows(arr, split, nh):
    s = C.make_split(arr, all_frames=True)
    if split == "val":
        r = arr["val_idx"]
        sub = arr["frame"][r] >= 30
        r = r[sub]
        w = arr["val_weight"][sub].astype(np.float64)
    elif split == "tune":
        r = s["tune_rows"]
        w = np.ones(len(r))
    else:
        scen = arr["scen_idx"]
        tr_scen = np.unique(scen[s["train_rows"]])
        r = np.where(np.isin(scen, tr_scen) & (arr["frame"] >= 30))[0]
        w = np.ones(len(r))
    keep = C.history_available(arr, r, nh)
    return r[keep], w[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/sc_sp_ctrl/best.pth")
    ap.add_argument("--split", required=True, choices=["train", "tune", "val"])
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=1, help="train 행 서브샘플")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)
    m, nh = load(args.ckpt, dev)
    rows, w = pick_rows(arr, args.split, nh)
    if args.stride > 1:
        rows, w = rows[::args.stride], w[::args.stride]
    print(f"{args.split}: {len(rows)}행  n_hist={nh}", flush=True)

    lg = C.run_logits(m, arr, rows, torch.from_numpy(C.build_global_lidar2img()),
                      dev, batch=args.batch, amp=True, n_hist=nh)
    T = C.precompute_targets(arr, rows, bank, weight=w)
    ad, tau = T["anchor_dist"], T["nms_tau"]
    N = len(rows)
    S = np.empty((N, 12), np.int32)
    for n in range(N):
        S[n] = C._shortlist(lg[n], ad, tau)
    take = lambda M: np.take_along_axis(M, S, 1)  # noqa: E731
    out = args.out or os.path.join(A, f"data/etri/selector_{args.split}.npz")
    np.savez_compressed(
        out, rows=rows, shortlist=S,
        logit=take(lg).astype(np.float32),
        logit_max=lg.max(1).astype(np.float32),
        D3=take(T["D3gt"]).astype(np.float32),
        goal=T["goal_xy"].astype(np.float32),
        cmd=arr["vad_cmd"][rows].astype(np.int8),
        weight=T["weight"].astype(np.float32),
        frame=arr["frame"][rows].astype(np.int32),
        scen=arr["scen_idx"][rows].astype(np.int32))
    d = take(T["D3gt"])
    print(f"저장 {out}\n  shortlist oracle {np.average(d.min(1), weights=T['weight']):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
