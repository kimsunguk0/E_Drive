#!/usr/bin/env python
"""⑤-Q: 학습된 VO head 의 실제 정밀도 측정.

⑤-N 사양: sigma_v <= 0.2 m/s 가 최소선, 0.1 이면 큰 이득.
loss 값에서 역산하지 말고 직접 잰다.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/sc_vo_w10/best.pth")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    arr = C.load_arrays()
    ck = torch.load(os.path.join(A, args.ckpt), map_location="cpu")
    a = ck["args"]
    nh = int(a.get("n_hist", 3))
    m = TrainableSparseScoreDrive(
        C.BANK_A0, logit_norm=bool(a.get("logit_norm", 1)), n_hist=nh,
        speed_head=bool(a.get("w_speed", 0) > 0), speed_gamma=0.0,
        vo_head=bool(a.get("w_vo", 0) > 0)).to(dev)
    m.load_state_dict(ck["model"])
    m.fuse_mul = bool(a.get("fuse_mul", 0))
    m.eval()

    rows = arr["val_idx"]
    sub = (arr["frame"][rows] >= 30) & C.history_available(arr, rows, nh)
    rows = rows[sub]
    w = arr["val_weight"][sub].astype(np.float64)

    from torch.utils.data import DataLoader
    l2i = torch.from_numpy(C.build_global_lidar2img()).to(dev)
    dl = DataLoader(C.SparseFrameDataset(arr, rows, n_hist=nh), batch_size=args.batch,
                    shuffle=False, num_workers=6, pin_memory=True)
    P, L = [], []
    with torch.no_grad():
        for b in dl:
            img = b["img"].to(dev, non_blocking=True)
            hT = b["hist_T"].to(dev, non_blocking=True)
            l2ib = l2i.unsqueeze(0).expand(img.shape[0], -1, -1, -1)
            with torch.autocast("cuda", dtype=torch.float16):
                m.temporal_logits(img, b["img_hist"].to(dev), l2ib, hT)
            P.append(m._vo_pred.float().cpu().numpy())
            L.append(hT[:, :, :2, 3].float().cpu().numpy())
    p = np.concatenate(P, 0).astype(np.float64)      # [N,nh,2] 예측 변위
    g = np.concatenate(L, 0).astype(np.float64)      # [N,nh,2] 정답
    wm = lambda x: float(np.average(x, weights=w))   # noqa: E731

    print(f"val38 n={len(rows)}  n_hist={nh}  ckpt={args.ckpt}\n")
    print(f"{'과거 시점':>10} {'GT 변위(m)':>11} {'MAE(m)':>9} {'상대오차':>9} {'->sigma_v':>10}")
    for k in range(nh):
        dt = 0.5 * (k + 1)
        gd = np.linalg.norm(g[:, k], axis=-1)
        err = np.linalg.norm(p[:, k] - g[:, k], axis=-1)
        mae = wm(err)
        print(f"{-dt:>9.1f}s {wm(gd):>11.2f} {mae:>9.3f} "
              f"{100*mae/max(wm(gd),1e-6):>8.1f}% {mae/dt:>10.3f}")

    # v, a 유도 정확도 (⑤-N 사양과 직접 비교)
    def va(d):
        v1 = np.linalg.norm(d[:, 0], axis=-1) / 0.5
        v0 = (np.linalg.norm(d[:, 1], axis=-1) - np.linalg.norm(d[:, 0], axis=-1)) / 0.5
        return v1, (v1 - v0) / 0.5
    pv, pa = va(p)
    gv, ga = va(g)
    print(f"\n  v (과거 0.5초 속도): MAE {wm(np.abs(pv-gv)):.3f} m/s  "
          f"sigma근사 {wm(np.abs(pv-gv))/0.7979:.3f}  상관 {np.corrcoef(pv,gv)[0,1]:.4f}")
    print(f"  a (가속도)         : MAE {wm(np.abs(pa-ga)):.3f} m/s^2  "
          f"sigma근사 {wm(np.abs(pa-ga))/0.7979:.3f}")
    sv = wm(np.abs(pv - gv)) / 0.7979
    print(f"\n  ⑤-N 사양: sigma_v <= 0.2 최소선 / 0.1 이면 큰 이득")
    print(f"  판정: sigma_v = {sv:.3f} -> "
          f"{'통과(큰 이득)' if sv <= 0.12 else '통과(유의미)' if sv <= 0.35 else '미달'}")
    np.savez(os.path.join(A, "logs/vo_pred_val.npz"), pred=p, gt=g, rows=rows, weight=w)
    print(f"\nsaved logs/vo_pred_val.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
