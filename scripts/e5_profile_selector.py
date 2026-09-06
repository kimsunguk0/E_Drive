#!/usr/bin/env python
"""⑤-E5: normalized timing profile(r1..r6) 을 selector 에 쓴다.

⑤-E3 은 speed head 8차원 중 index0(S3, 절대 진행량) 만 썼고 실패했다.
r1..r6 = 누적거리/S3 는 **scale-free** 라 절대 정밀도(σ<=0.5m) 벽에 걸리지 않는다.
"어느 속도로" 가 아니라 "가속인가 감속인가 등속인가" 만 묻는 양이다.

방법론: mu 를 tune 에서 고정한 뒤 val38 에 1회 적용한다(val 에서 고르지 않는다).
compliance: 후보는 이미 완성된 12행이고 selector 는 행 index 만 고른다.
speed_pred 는 visual_logits 와 같은 지위의 모델 출력이다(goal 무관, 생성 그래프 밖).
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


def steplen(xy):
    """[...,T,2] 절대좌표 -> 스텝별 이동거리 [...,T]"""
    z = np.zeros_like(xy[..., :1, :])
    return np.linalg.norm(np.diff(np.concatenate([z, xy], -2), axis=-2), axis=-1)


def profile_of(xy6):
    """[...,6,2] -> (S3 [...], r1..r6 [...,6])  r_i = 누적거리_i / S3"""
    st = steplen(xy6)
    s3 = st.sum(-1)
    r = np.cumsum(st, -1) / np.clip(s3, 1e-3, None)[..., None]
    return s3, r


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
    return m, a


def forward(m, arr, rows, dev, batch=16):
    from torch.utils.data import DataLoader
    l2i = torch.from_numpy(C.build_global_lidar2img()).to(dev)
    dl = DataLoader(C.SparseFrameDataset(arr, rows, n_hist=3), batch_size=batch,
                    shuffle=False, num_workers=6, pin_memory=True)
    LG, SP = [], []
    with torch.no_grad():
        for b in dl:
            img = b["img"].to(dev, non_blocking=True)
            l2ib = l2i.unsqueeze(0).expand(img.shape[0], -1, -1, -1)
            with torch.autocast("cuda", dtype=torch.float16):
                lg = m.temporal_logits(img, b["img_hist"].to(dev), l2ib,
                                       b["hist_T"].to(dev))
            LG.append(lg.float().cpu().numpy())
            SP.append(m._speed_pred.float().cpu().numpy()
                      if m._speed_pred is not None else np.zeros((img.shape[0], 8), np.float32))
    return (np.concatenate(LG, 0).astype(np.float64),
            np.concatenate(SP, 0).astype(np.float64))


def rows_of(arr, split):
    if split == "val":
        rows = arr["val_idx"]
        sub = (arr["frame"][rows] >= 30) & C.history_available(arr, rows, 3)
        return rows[sub], arr["val_weight"][sub].astype(np.float64)
    s = C.make_split(arr)
    rows = s["tune_rows"]
    sub = C.history_available(arr, rows, 3)
    return rows[sub], None


def run_split(m, arr, bank, dev, split, mus, lams, batch):
    rows, w = rows_of(arr, split)
    lg, sp = forward(m, arr, rows, dev, batch)
    T = C.precompute_targets(arr, rows, bank, weight=w)
    w = T["weight"]
    Dg, ce, gx = T["D3gt"], T["cand_end5"], T["goal_xy"]
    ad, tau = T["anchor_dist"], T["nms_tau"]

    cand5 = bank["candidate_xy_abs_5s"].astype(np.float64)
    cS3, cR = profile_of(cand5[:, :6])                   # [K], [K,6]
    pR = sp[:, 2:8]                                      # [N,6] 예측 profile
    pS3 = sp[:, 0] * 30.0

    gS3, gR = profile_of(arr["fut"][rows].astype(np.float64))
    prof_mae = float(np.abs(pR - gR).mean())
    print(f"  [{split}] n={len(rows)}  profile MAE {prof_mae:.4f} "
          f"(r 무작위추측 상수 프로파일 MAE {np.abs(gR - gR.mean(0)).mean():.4f})  "
          f"S3 MAE {np.abs(pS3 - gS3).mean():.3f}m", flush=True)

    nz = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)  # noqa: E731
    wm = lambda x: float(np.average(x, weights=w))              # noqa: E731

    # 세 소스를 같은 selector 로 비교한다.
    #   pred  = 학습된 head 의 r1..r6
    #   const = 데이터셋 평균 프로파일 (프레임 무관 상수) -- 신호 없음의 귀무가설
    #   gt    = 정답 r1..r6                              -- 이 축의 상한
    SRC = {"pred": pR, "const": np.repeat(gR.mean(0, keepdims=True), len(rows), 0),
           "gt": gR}
    out = {"mae": {k: float(np.abs(v - gR).mean()) for k, v in SRC.items()}}
    R_orc = np.empty(len(rows))
    grid = {k: {} for k in SRC}
    for n in range(len(rows)):
        S = C._shortlist(lg[n], ad, tau)
        R_orc[n] = Dg[n][S].min()
        gcn = nz(np.linalg.norm(ce[S] - gx[n], axis=1))
        vcn = nz(lg[n].max() - lg[n][S])
        for src, Rsrc in SRC.items():
            pd = np.abs(cR[S] - Rsrc[n][None]).mean(1)
            pdn = nz(pd) if (pd.max() - pd.min()) > 1e-9 else np.zeros_like(pd)
            for lam in lams:
                base = gcn + lam * vcn
                for mu in mus:
                    grid[src].setdefault((lam, mu), np.empty(len(rows)))[n] = \
                        Dg[n][S[int(np.argmin(base + mu * pdn))]]
    out["oracle"] = wm(R_orc)
    out["grid"] = {src: {f"{lam}|{mu}": wm(v) for (lam, mu), v in g.items()}
                   for src, g in grid.items()}
    out["profile_mae"] = prof_mae
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/sc_sp_a_w10/best.pth")
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(A, "logs/e5_profile.json"))
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)
    m, a = load(args.ckpt, dev)
    print(f"ckpt={args.ckpt}  speed_head={bool(a.get('w_speed',0)>0)}", flush=True)

    lams = (0.0, 0.1, 0.2, 0.3)
    mus = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)

    print("=== tune (source 별 mu, lambda 고정용) ===", flush=True)
    tu = run_split(m, arr, bank, dev, "tune", mus, lams, args.batch)
    print(f"  oracle {tu['oracle']:.4f}   profile MAE " +
          "  ".join(f"{k}={v:.4f}" for k, v in tu["mae"].items()), flush=True)
    fixed = {}
    for src in ("pred", "const", "gt"):
        g = tu["grid"][src]
        bk = min(g, key=lambda k: g[k])
        fixed[src] = bk
        print(f"   [{src:5s}] 고정 lam|mu = {bk}  tune {g[bk]:.4f}", flush=True)

    print("\n=== val38 (tune 고정값 1회 적용) ===", flush=True)
    va = run_split(m, arr, bank, dev, "val", mus, lams, args.batch)
    print(f"  shortlist oracle {va['oracle']:.4f}")
    print(f"  profile MAE  " + "  ".join(f"{k}={v:.4f}" for k, v in va["mae"].items()))
    print(f"\n  {'source':7s} {'lam|mu':10s} {'mu=0 기준':>10s} {'적용후':>10s} {'이득':>9s}")
    for src in ("pred", "const", "gt"):
        bk = fixed[src]
        lam0 = bk.split("|")[0]
        base = va["grid"][src][f"{lam0}|0.0"]
        got = va["grid"][src][bk]
        print(f"  {src:7s} {bk:10s} {base:>10.4f} {got:>10.4f} {base-got:>+9.4f}")
    print("\n  [참고] val 전체 격자 (선택에 쓰지 않음)")
    for src in ("pred", "const", "gt"):
        print(f"   --- {src} ---")
        for lam in lams:
            row = "  ".join(f"mu{mu}={va['grid'][src][f'{lam}|{mu}']:.4f}" for mu in mus)
            print(f"    lam{lam}: {row}")
    json.dump({"ckpt": args.ckpt, "tune": tu, "val": va, "fixed": fixed},
              open(args.out, "w"), indent=1, default=float)
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
