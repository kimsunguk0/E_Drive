#!/usr/bin/env python
"""⑤-E3: 학습된 speed head 의 예측 정확도 + 오라클 실현율.

  python e3_speed_realize.py --ckpt work_dirs/sc_sp_a_w10/best.pth

측정:
  1) 예측 S3 vs GT S3 (MAE / 상관 / 버킷별)
  2) selector 를 예측 S3 로 돌렸을 때 realized  <-- 실제로 얻는 값
  3) GT S3 오라클 대비 실현율
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


def arclen(xy):
    z = np.zeros_like(xy[..., :1, :])
    inc = np.diff(np.concatenate([z, xy], axis=-2), axis=-2)
    return np.linalg.norm(inc, axis=-1).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/sc_sp_a_w10/best.pth")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)

    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)
    ck = torch.load(os.path.join(A, args.ckpt), map_location="cpu")
    a = ck["args"]
    nh = int(a.get("n_hist", 3))
    m = TrainableSparseScoreDrive(
        C.BANK_A0, logit_norm=bool(a.get("logit_norm", 1)), n_hist=nh,
        speed_head=(a.get("w_speed", 0) > 0),
        speed_gamma=float(a.get("speed_gamma", 0))).to(dev)
    m.load_state_dict(ck["model"])
    m.fuse_mul = bool(a.get("fuse_mul", 0))
    m.merge_encode = bool(a.get("merge_encode", 0))
    m.eval()

    rows = arr["val_idx"]
    sub = (arr["frame"][rows] >= 30) & C.history_available(arr, rows, nh)
    rows = rows[sub]
    w = arr["val_weight"][sub]
    T = C.precompute_targets(arr, rows, bank, weight=w)
    l2i = torch.from_numpy(C.build_global_lidar2img()).to(dev)

    from torch.utils.data import DataLoader
    dl = DataLoader(C.SparseFrameDataset(arr, rows, n_hist=nh), batch_size=args.batch,
                    shuffle=False, num_workers=6, pin_memory=True)
    LG, SP = [], []
    with torch.no_grad():
        for b in dl:
            img = b["img"].to(dev, non_blocking=True)
            l2ib = l2i.unsqueeze(0).expand(img.shape[0], -1, -1, -1)
            with torch.autocast("cuda", dtype=torch.float16):
                lg = m.temporal_logits(img, b["img_hist"].to(dev),
                                       l2ib, b["hist_T"].to(dev))
            LG.append(lg.float().cpu().numpy())
            SP.append(m._speed_pred.float().cpu().numpy())
    lg = np.concatenate(LG, 0).astype(np.float64)
    sp = np.concatenate(SP, 0).astype(np.float64)

    pS3 = sp[:, 0] * 30.0
    gS3 = arclen(arr["fut"][rows].astype(np.float64))
    err = pS3 - gS3
    print(f"=== 1) 진행량 예측 정확도 (val38, n={len(rows)}) ===")
    print(f"  GT S3  평균 {gS3.mean():.2f}m  std {gS3.std():.2f}")
    print(f"  MAE {np.abs(err).mean():.3f}m   p50 {np.percentile(np.abs(err),50):.3f}  "
          f"p90 {np.percentile(np.abs(err),90):.3f}")
    print(f"  상관 {np.corrcoef(pS3,gS3)[0,1]:.4f}   "
          f"상대오차 {np.abs(err).mean()/max(gS3.mean(),1e-6)*100:.1f}%")
    for k, mk in T["buckets"].items():
        mk = np.asarray(mk, bool)
        if mk.sum():
            print(f"    {k:8s} n={mk.sum():4d}  MAE {np.abs(err[mk]).mean():.3f}m  "
                  f"GT평균 {gS3[mk].mean():.2f}m")

    cand5 = bank["candidate_xy_abs_5s"].astype(np.float64)
    cS3 = arclen(cand5[:, :6])
    Dg, ce, gx = T["D3gt"], T["cand_end5"], T["goal_xy"]
    ad, tau = T["anchor_dist"], T["nms_tau"]
    wm = lambda x: float(np.average(x, weights=w))  # noqa: E731
    nz = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)  # noqa: E731

    modes = ["cur", "pred1", "pred2", "gt2", "orc"]
    R = {k: np.empty(len(rows)) for k in modes}
    for n in range(len(rows)):
        S = C._shortlist(lg[n], ad, tau)
        d = Dg[n][S]
        R["orc"][n] = d.min()
        gcn = nz(np.linalg.norm(ce[S] - gx[n], axis=1))
        vcn = nz(lg[n].max() - lg[n][S])
        pn = nz(np.abs(cS3[S] - pS3[n]))
        gn_ = nz(np.abs(cS3[S] - gS3[n]))
        R["cur"][n] = Dg[n][S[int(np.argmin(gcn + 0.2 * vcn))]]
        R["pred1"][n] = Dg[n][S[int(np.argmin(gcn + 1.0 * pn))]]
        R["pred2"][n] = Dg[n][S[int(np.argmin(gcn + 2.0 * pn))]]
        R["gt2"][n] = Dg[n][S[int(np.argmin(gcn + 2.0 * gn_))]]
    print("\n=== 2) selector realized ===")
    for k, nm in (("orc", "shortlist oracle (상한)"), ("cur", "현재 (goal+0.2*visual)"),
                  ("pred1", "goal + 1.0*예측진행량"), ("pred2", "goal + 2.0*예측진행량"),
                  ("gt2", "goal + 2.0*GT진행량 (오라클)")):
        print(f"  {nm:32s} {wm(R[k]):.4f}")
    best = min(("pred1", "pred2"), key=lambda k: wm(R[k]))
    gain_p = wm(R["cur"]) - wm(R[best])
    gain_o = wm(R["cur"]) - wm(R["gt2"])
    print(f"\n  예측 이득 {gain_p:+.4f} / 오라클 이득 {gain_o:+.4f}  "
          f"-> 실현율 {100*gain_p/max(gain_o,1e-9):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
