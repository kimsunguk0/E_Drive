#!/usr/bin/env python
"""⑤-K: 중간구간 종방향 위치를 '얼마나 정확히' 알아야 이득이 나는가.

⑤-H: GT 1.5~2.0초 위치를 알면 불일치 프레임 realized 0.4022 -> 0.2154 (83% 회수).
head 를 만들기 전에 요구 정밀도를 먼저 잰다(⑤-E3 에서 어긴 순서).
GT + 가우시안 노이즈 sigma 를 쓸어 이득이 사라지는 지점을 찾는다.
"""
import os
import sys

import numpy as np

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402


def main():
    bank = np.load(C.BANK_A0, allow_pickle=False)
    c5 = bank["candidate_xy_abs_5s"].astype(np.float64)
    z = np.load(os.path.join(A, "data/etri/selector_val.npz"))
    S = z["shortlist"].astype(np.int64)
    D3, lg, goal, w = (z["D3"].astype(np.float64), z["logit"].astype(np.float64),
                       z["goal"].astype(np.float64), z["weight"].astype(np.float64))
    arr = C.load_arrays()
    G = arr["fut"][z["rows"]].astype(np.float64)
    nz1 = lambda x: (x - x.min(1, keepdims=True)) / (
        x.max(1, keepdims=True) - x.min(1, keepdims=True) + 1e-9)     # noqa: E731
    gd = np.linalg.norm(c5[S, 9] - goal[:, None], axis=-1)
    Jbase = nz1(gd) + 0.1 * nz1(lg.max(1, keepdims=True) - lg)
    wm = lambda x: float(np.average(x, weights=w))                    # noqa: E731
    take = lambda i: np.take_along_axis(D3, i[:, None], 1)[:, 0]      # noqa: E731

    base = wm(take(Jbase.argmin(1)))
    orc = wm(D3.min(1))
    print(f"val38 n={len(S)}  현재 selector {base:.4f}   shortlist oracle {orc:.4f}\n")

    rng = np.random.default_rng(0)
    for ti, tname in ((1, "1.0초"), (3, "2.0초"), (5, "3.0초")):
        cv = c5[:, ti, 0][S]                    # 후보의 해당 시각 종방향 x
        gv = G[:, ti, 0]
        spread = wm(cv.max(1) - cv.min(1))
        print(f"=== {tname} 종방향 위치 (shortlist 내 폭 {spread:.2f}m) ===")
        print(f"  {'sigma(m)':>9} {'mu=0.3':>9} {'mu=0.6':>9} {'mu=1.0':>9} {'mu=2.0':>9} {'최선이득':>9}")
        for sig in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            est = gv + (rng.normal(0, sig, len(gv)) if sig > 0 else 0.0)
            d = np.abs(cv - est[:, None])
            dn = nz1(d)
            vals = {}
            for mu in (0.3, 0.6, 1.0, 2.0):
                vals[mu] = wm(take((Jbase + mu * dn).argmin(1)))
            best = base - min(vals.values())
            print(f"  {sig:>9.2f} " + " ".join(f"{vals[m]:>9.4f}" for m in (0.3, 0.6, 1.0, 2.0))
                  + f" {best:>+9.4f}")
        print()
    print("참고: full-K logit 유래 S3 추정 MAE 1.30m / 전용 speed head 2.45m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
