#!/usr/bin/env python
"""⑤-I: full-K logits 안에 3초 진행량 정보가 이미 있는가.

⑤-H: 3초 종방향 위치(=S3)가 유일한 큰 레버인데 5초 goal 로는 유도되지 않는다.
⑤-E3: 전용 speed head 의 S3 MAE 는 2.45m (요구 ~0.5m) 로 부족했다.
그런데 모델은 1024개 후보 전체를 점수화한다. 그 분포의 S3 기댓값이
전용 회귀 head 보다 정확하다면, 정보는 이미 있고 selector 가 버리는 것이다.
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


def steps(xy):
    z = np.zeros_like(xy[..., :1, :])
    return np.linalg.norm(np.diff(np.concatenate([z, xy], -2), axis=-2), axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/sc_sp_ctrl/best.pth")
    ap.add_argument("--gpu", type=int, default=7)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)
    c5 = bank["candidate_xy_abs_5s"].astype(np.float64)
    cS3 = steps(c5[:, :6]).sum(-1)

    ck = torch.load(os.path.join(A, args.ckpt), map_location="cpu")
    a = ck["args"]
    nh = int(a.get("n_hist", 3))
    m = TrainableSparseScoreDrive(C.BANK_A0, logit_norm=bool(a.get("logit_norm", 1)),
                                  n_hist=nh, speed_head=bool(a.get("w_speed", 0) > 0),
                                  speed_gamma=0.0).to(dev)
    m.load_state_dict(ck["model"]); m.fuse_mul = bool(a.get("fuse_mul", 0)); m.eval()

    rows = arr["val_idx"]
    sub = (arr["frame"][rows] >= 30) & C.history_available(arr, rows, nh)
    rows = rows[sub]
    w = arr["val_weight"][sub].astype(np.float64)
    lg = C.run_logits(m, arr, rows, torch.from_numpy(C.build_global_lidar2img()),
                      dev, batch=24, amp=True, n_hist=nh).astype(np.float64)
    G = arr["fut"][rows].astype(np.float64)
    gS3 = steps(G).sum(-1)
    wm = lambda x: float(np.average(x, weights=w))                      # noqa: E731

    print(f"val38 n={len(rows)}  GT S3 평균 {gS3.mean():.2f}m std {gS3.std():.2f}\n")
    print(f"{'추정자':>34} {'MAE(m)':>8} {'p50':>7} {'상관':>7}")

    def rep(nm, p):
        e = np.abs(p - gS3)
        print(f"{nm:>34} {wm(e):>8.3f} {np.percentile(e,50):>7.3f} "
              f"{np.corrcoef(p,gS3)[0,1]:>7.4f}")
        return wm(e)

    rep("전역 상수(데이터셋 평균)", np.full_like(gS3, gS3.mean()))
    rep("top-1 후보의 S3", cS3[lg.argmax(1)])
    for T in (0.05, 0.1, 0.25, 0.5, 1.0):
        p = (np.exp((lg - lg.max(1, keepdims=True)) / T) @ cS3) / \
            np.exp((lg - lg.max(1, keepdims=True)) / T).sum(1)
        rep(f"softmax(logit/{T}) 가중 S3 기댓값", p)
    # 상위 k 평균
    for k in (3, 12, 64):
        idx = np.argsort(-lg, 1)[:, :k]
        rep(f"상위 {k}개 평균 S3", cS3[idx].mean(1))
    # 참고: 전용 speed head
    if bool(a.get("w_speed", 0) > 0):
        print("  (이 ckpt 은 speed head 없음)")
    print("\n  ⑤-E3 전용 speed head(sc_sp_a_w10) 실측 MAE = 2.446m")
    print("  ⑤-H 요구 정밀도 ~= 0.5~0.75m (shortlist 내 oracle-선택 S3 차이 0.743m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
