#!/usr/bin/env python
"""⑤-E4: 예측 진행량을 '선택'이 아니라 'factorized bank 의 velocity 축 좁히기'에 쓴다.

selection 은 12개 중 argmin 이라 0.5m 정밀도를 요구해 실패했다(⑤-E3).
velocity 축 좁히기는 '구간'만 맞히면 되므로 요구 정밀도가 훨씬 낮을 것이라는 가설.

측정: coverage(oracle D3) 를
  - legacy K=1024
  - factorized 전체 65,025
  - velocity 상위 k 개로 좁힌 부분집합 (예측 S3 / GT S3 각각)
로 비교하고, 노이즈를 주입해 요구 정밀도를 잰다.
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

FB = os.path.join(A, "data/etri/factorized_phase5a/factorized_P512_V128.npz")


def arclen(x):
    z = np.zeros_like(x[..., :1, :])
    inc = np.diff(np.concatenate([z, x], axis=-2), axis=-2)
    return np.linalg.norm(inc, axis=-1).sum(-1)


def d3_min_grouped(bank6, gt6, cw, groups, chunk=24):
    """프레임별로 (전체 최소, 그룹별 최소) 반환.
    bank6 [M,6,2], gt6 [N,6,2], groups [M] int (velocity_id)."""
    M = bank6.shape[0]
    N = gt6.shape[0]
    G = int(groups.max()) + 1
    full = np.empty(N)
    per_g = np.empty((N, G))
    for s in range(0, N, chunk):
        g = gt6[s:s + chunk]                                   # [c,6,2]
        d = np.linalg.norm(bank6[None] - g[:, None], axis=-1)  # [c,M,6]
        d = (d * cw).sum(-1)                                   # [c,M]
        full[s:s + chunk] = d.min(1)
        for gi in range(G):
            m = groups == gi
            per_g[s:s + chunk, gi] = d[:, m].min(1)
    return full, per_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/sc_sp_a_w10/best.pth")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    arr = C.load_arrays()
    bank0 = np.load(C.BANK_A0, allow_pickle=False)
    fb = np.load(FB, allow_pickle=True)
    bank = fb["bank"].astype(np.float64)
    vid = fb["velocity_id"].astype(np.int64)
    V = int(vid.max()) + 1
    cw = C.CW3

    # 평가 행 = val38 frame>=30 & history 가능 (E3 와 동일)
    rows = arr["val_idx"]
    sub = (arr["frame"][rows] >= 30) & C.history_available(arr, rows, 3)
    rows = rows[sub]
    w = arr["val_weight"][sub].astype(np.float64)
    gt6 = arr["fut"][rows].astype(np.float64)
    gS3 = arclen(gt6)
    wm = lambda a: float(np.average(a, weights=w))  # noqa: E731

    # 예측 S3 (학습된 speed head)
    dev = torch.device(f"cuda:{args.gpu}")
    ck = torch.load(os.path.join(A, args.ckpt), map_location="cpu")
    a = ck["args"]
    m = TrainableSparseScoreDrive(
        C.BANK_A0, logit_norm=True, n_hist=int(a.get("n_hist", 3)),
        speed_head=True, speed_gamma=0.0).to(dev)
    m.load_state_dict(ck["model"])
    m.fuse_mul = False
    m.eval()
    from torch.utils.data import DataLoader
    dl = DataLoader(C.SparseFrameDataset(arr, rows, n_hist=3), batch_size=16,
                    shuffle=False, num_workers=6, pin_memory=True)
    l2i = torch.from_numpy(C.build_global_lidar2img()).to(dev)
    SP = []
    with torch.no_grad():
        for b in dl:
            img = b["img"].to(dev, non_blocking=True)
            l2ib = l2i.unsqueeze(0).expand(img.shape[0], -1, -1, -1)
            with torch.autocast("cuda", dtype=torch.float16):
                m.temporal_logits(img, b["img_hist"].to(dev), l2ib,
                                  b["hist_T"].to(dev))
            SP.append(m._speed_pred.float().cpu().numpy())
    pS3 = np.concatenate(SP, 0)[:, 0].astype(np.float64) * 30.0
    print(f"예측 S3: MAE {np.abs(pS3-gS3).mean():.3f}m  상관 "
          f"{np.corrcoef(pS3,gS3)[0,1]:.4f}", flush=True)

    # velocity 축의 3초 진행량
    v_s3 = np.array([arclen(bank[vid == v][:1, :6])[0] for v in range(V)])
    print(f"velocity 축 {V}개의 3초 진행량: {v_s3.min():.1f}~{v_s3.max():.1f}m, "
          f"인접 간격 중앙값 {np.median(np.diff(np.sort(v_s3))):.2f}m", flush=True)

    print("\nD3 계산 중 (65,025 후보 x %d 프레임)..." % len(rows), flush=True)
    full, per_v = d3_min_grouped(bank[:, :6], gt6, cw, vid)
    leg = np.load(C.BANK_A0, allow_pickle=False)["anchors_abs"].astype(np.float64)
    dleg = np.empty(len(rows))
    for s in range(0, len(rows), 64):
        g = gt6[s:s + 64]
        dleg[s:s + 64] = ((np.linalg.norm(leg[None] - g[:, None], axis=-1)
                           * cw).sum(-1)).min(1)

    print("\n=== coverage (oracle D3, val38 frame>=30) ===")
    print(f"  legacy K=1024                    {wm(dleg):.4f}")
    print(f"  factorized 전체 65,025           {wm(full):.4f}")

    def topk_cover(src, k):
        idx = np.argsort(np.abs(v_s3[None, :] - src[:, None]), axis=1)[:, :k]
        return wm(np.take_along_axis(per_v, idx, axis=1).min(1))

    print("\n=== velocity 축을 상위 k 개로 좁혔을 때 ===")
    print(f"{'k':>4} {'후보수':>8} {'GT S3':>10} {'예측 S3':>10}")
    for k in (1, 2, 4, 8, 16, 32):
        print(f"{k:>4} {k*512:>8} {topk_cover(gS3,k):>10.4f} {topk_cover(pS3,k):>10.4f}")

    print("\n=== 요구 정밀도 (GT + 노이즈, k=8) ===")
    rng = np.random.default_rng(0)
    for sig in (0.0, 0.5, 1.0, 2.0, 2.45, 4.0, 8.0):
        src = gS3 + (rng.normal(0, sig, len(gS3)) if sig > 0 else 0)
        tag = "  <-- 우리 예측 MAE" if abs(sig - 2.45) < 1e-9 else ""
        print(f"  sigma={sig:4.2f}m -> {topk_cover(src,8):.4f}{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
