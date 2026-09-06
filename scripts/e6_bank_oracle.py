#!/usr/bin/env python
"""⑥-B: 고정 bank 의 D3 oracle 한계를 잰다 (CPU).

배경: 공식 리더보드 1위 0.0924106 인데 우리 A0 K=1024 의 완벽선택 oracle 이
0.099634 다. 즉 ranker/selector 가 완벽해도 현재 후보 좌표로는 1위가 불가능하다.
1등용 gate 는 bank oracle <= 0.055~0.060.

핵심 단순화: **D3 = w·||C_i - G_i||, w=[11,11,5,5,2,2]/36 은 첫 6점(3초)만 본다.**
따라서 bank oracle 은 3초 기하 12차원 공간의 양자화 오차일 뿐이고,
5초 꼬리/속도 축은 oracle 에 전혀 기여하지 않는다(선택에만 쓰인다).

측정:
  1) A0 K=1024                      (기준 0.099634)
  2) train GT 전량을 후보로          -> 데이터 기반 bank 의 내재 바닥
  3) D3-가중 k-means, K 스윕         -> 실제로 만들 수 있는 곡선
  4) 필요 K 외삽                     -> 0.060 에 필요한 후보 수
"""
import argparse
import json
import os
import sys

import numpy as np

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402

W6 = np.array([11, 11, 5, 5, 2, 2], np.float64)
W6 = W6 / W6.sum()


def d3_oracle(cands, gt, w, chunk=64):
    """cands [M,6,2], gt [N,6,2] -> 프레임별 min D3 [N]."""
    out = np.empty(len(gt))
    for s in range(0, len(gt), chunk):
        g = gt[s:s + chunk]
        d = np.linalg.norm(cands[None] - g[:, None], axis=-1)   # [c,M,6]
        out[s:s + chunk] = (d * W6).sum(-1).min(1)
    return out


def wkmeans(X, K, iters=25, seed=0, batch=None):
    """D3 가중 k-means. X [N,6,2] -> 좌표축에 sqrt(w) 를 걸어 12차원 유클리드로 근사.
    (D3 는 점별 L2 의 가중합이라 제곱합과 완전 일치하진 않지만 군집 유도에는 충분)"""
    rng = np.random.default_rng(seed)
    sw = np.sqrt(W6)[None, :, None]
    Z = (X * sw).reshape(len(X), -1)                             # [N,12]
    # k-means++ 초기화 (부분표본)
    sub = Z[rng.choice(len(Z), min(len(Z), 50000), replace=False)]
    cen = np.empty((K, Z.shape[1]))
    cen[0] = sub[rng.integers(len(sub))]
    d2 = ((sub - cen[0]) ** 2).sum(1)
    for k in range(1, K):
        p = d2 / max(d2.sum(), 1e-12)
        cen[k] = sub[rng.choice(len(sub), p=p)]
        d2 = np.minimum(d2, ((sub - cen[k]) ** 2).sum(1))
    for _ in range(iters):
        lab = np.empty(len(Z), np.int32)
        for s in range(0, len(Z), 4096):
            zz = Z[s:s + 4096]
            lab[s:s + 4096] = np.argmin(
                (zz * zz).sum(1)[:, None] - 2 * zz @ cen.T + (cen * cen).sum(1)[None],
                axis=1)
        newc = np.zeros_like(cen)
        cnt = np.bincount(lab, minlength=K).astype(np.float64)
        np.add.at(newc, lab, Z)
        empty = cnt == 0
        newc[~empty] /= cnt[~empty, None]
        newc[empty] = cen[empty]
        shift = np.linalg.norm(newc - cen, axis=1).max()
        cen = newc
        if shift < 1e-4:
            break
    return (cen.reshape(K, 6, 2) / sw[0]).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--all-frames", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(A, "logs/e6_bank_oracle.json"))
    args = ap.parse_args()
    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)
    s = C.make_split(arr)

    vr = arr["val_idx"]
    vsub = arr["frame"][vr] >= 30
    vr = vr[vsub]
    vw = arr["val_weight"][vsub].astype(np.float64)
    Gv = arr["fut"][vr].astype(np.float64)                   # [Nv,6,2] 절대 3초
    wm = lambda x: float(np.average(x, weights=vw))          # noqa: E731

    tr = s["train_rows"]
    if args.all_frames:
        # 비-main frame 도 GT 유효(⑤-C 확인). 서로 다른 궤적 표본이 4배 늘어난다.
        scen = arr["scen_idx"]
        train_scen = np.unique(scen[tr])
        tr = np.where(np.isin(scen, train_scen) & (arr["frame"] >= 30))[0]
    Gt = arr["fut"][tr].astype(np.float64)
    print(f"train {len(Gt)}행 / val38 {len(Gv)}행 (frame>=30)", flush=True)

    res = {}
    a0 = bank["anchors_abs"].astype(np.float64)
    res["A0_K1024"] = wm(d3_oracle(a0, Gv, vw))
    print(f"  A0 K=1024                {res['A0_K1024']:.6f}   (기대 0.099634)", flush=True)

    res["train_GT_all"] = wm(d3_oracle(Gt, Gv, vw))
    print(f"  train GT 전량 K={len(Gt):<8} {res['train_GT_all']:.6f}   "
          f"<- 데이터 기반 bank 의 내재 바닥", flush=True)

    res["kmeans"] = {}
    for K in [int(x) for x in args.ks.split(",")]:
        cen = wkmeans(Gt, K, seed=0)
        v = wm(d3_oracle(cen, Gv, vw))
        res["kmeans"][K] = v
        print(f"  D3-가중 k-means K={K:<6}  {v:.6f}", flush=True)

    # 필요 K 외삽: oracle ~ c * K^(-alpha)
    ks = np.array(sorted(res["kmeans"]), float)
    vs = np.array([res["kmeans"][int(k)] for k in ks])
    al, lc = np.polyfit(np.log(ks), np.log(vs), 1)
    res["fit"] = {"alpha": float(-al), "c": float(np.exp(lc))}
    print(f"\n  적합: oracle ≈ {np.exp(lc):.4f}·K^({al:.3f})", flush=True)
    for tgt in (0.060, 0.055, 0.0500):
        need = float(np.exp((np.log(tgt) - lc) / al))
        res.setdefault("need_K", {})[str(tgt)] = need
        print(f"   oracle {tgt:.3f} 에 필요한 K ≈ {need:,.0f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
