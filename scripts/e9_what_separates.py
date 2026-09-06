#!/usr/bin/env python
"""⑤-H: shortlist 안에서 oracle 후보와 선택 후보를 실제로 가르는 것은 무엇인가.

⑤-G 에서 차선 기하(centerline 거리/solid 교차)는 둘을 전혀 구분하지 못했다.
그러면 남은 축을 하나씩 재서 '어떤 양이 다른가'를 직접 본다.
"""
import os
import sys

import numpy as np

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402


def steps(xy):
    z = np.zeros_like(xy[..., :1, :])
    return np.linalg.norm(np.diff(np.concatenate([z, xy], -2), axis=-2), axis=-1)


def main():
    bank = np.load(C.BANK_A0, allow_pickle=False)
    c5 = bank["candidate_xy_abs_5s"].astype(np.float64)
    st = steps(c5)
    S3, S5 = st[:, :6].sum(-1), st.sum(-1)
    d = np.diff(np.concatenate([np.zeros((len(c5), 1, 2)), c5], 1), axis=1)
    hd = np.unwrap(np.arctan2(d[..., 1], d[..., 0]), axis=1)
    ATTR = {
        "0.5초 위치 x(m)": c5[:, 0, 0], "1.0초 위치 x(m)": c5[:, 1, 0],
        "1.5초 위치 x(m)": c5[:, 2, 0], "2.0초 위치 x(m)": c5[:, 3, 0],
        "S3 진행량(m)": S3, "S5 진행량(m)": S5,
        "3초 종방향 x(m)": c5[:, 5, 0], "3초 횡방향 y(m)": c5[:, 5, 1],
        "5초 종방향 x(m)": c5[:, 9, 0], "5초 횡방향 y(m)": c5[:, 9, 1],
        "3초 heading(deg)": np.degrees(hd[:, 5]),
        "총 곡률(deg)": np.degrees(hd[:, 9] - hd[:, 0]),
        "|곡률| 합(deg)": np.degrees(np.abs(np.diff(hd, axis=1)).sum(1)),
    }
    z = np.load(os.path.join(A, "data/etri/selector_val.npz"))
    S = z["shortlist"].astype(np.int64)
    D3, lg, goal, w = (z["D3"].astype(np.float64), z["logit"].astype(np.float64),
                       z["goal"].astype(np.float64), z["weight"].astype(np.float64))
    nz = lambda x: (x - x.min(1, keepdims=True)) / (
        x.max(1, keepdims=True) - x.min(1, keepdims=True) + 1e-9)      # noqa: E731
    gd = np.linalg.norm(c5[S, 9] - goal[:, None], axis=-1)
    sel = (nz(gd) + 0.1 * nz(lg.max(1, keepdims=True) - lg)).argmin(1)
    orc = D3.argmin(1)
    miss = sel != orc
    wm = lambda x, m: float(np.average(x[m], weights=w[m]))            # noqa: E731
    print(f"val38 n={len(S)}  선택==oracle {100*np.average((~miss),weights=w):.1f}%  "
          f"불일치 {miss.sum()}프레임")
    print(f"regret(가중) {wm(np.take_along_axis(D3,sel[:,None],1)[:,0]-D3.min(1), np.ones(len(S),bool)):.4f}\n")
    print(f"{'속성':>18} {'oracle':>9} {'선택':>9} {'차이':>9} {'|차이|평균':>10} {'shortlist폭':>11}")
    for nm, v in ATTR.items():
        V = v[S]
        a = np.take_along_axis(V, orc[:, None], 1)[:, 0]
        b = np.take_along_axis(V, sel[:, None], 1)[:, 0]
        spread = (V.max(1) - V.min(1))
        print(f"{nm:>18} {wm(a,miss):>9.3f} {wm(b,miss):>9.3f} {wm(a-b,miss):>9.3f} "
              f"{wm(np.abs(a-b),miss):>10.3f} {wm(spread,miss):>11.3f}")

    print("\n=== 단일 속성만으로 oracle 을 고르면 (불일치 프레임 realized) ===")
    cur = wm(np.take_along_axis(D3, sel[:, None], 1)[:, 0], miss)
    orcv = wm(D3.min(1), miss)
    print(f"  {'현재 selector':>22} {cur:.4f}")
    print(f"  {'shortlist oracle':>22} {orcv:.4f}")
    gtA = {"S3 진행량(m)": steps(np.zeros((1, 6, 2)))}  # placeholder
    # GT 값과 가장 가까운 후보를 고르는 오라클 (속성별)
    arr = C.load_arrays()
    rows = z["rows"]
    G = arr["fut"][rows].astype(np.float64)
    g5 = arr["fut5"][rows].astype(np.float64)
    gst = steps(G)
    gd_ = np.diff(np.concatenate([np.zeros((len(G), 1, 2)), G], 1), axis=1)
    ghd = np.unwrap(np.arctan2(gd_[..., 1], gd_[..., 0]), axis=1)
    GT = {"0.5초 위치 x(m)": G[:, 0, 0], "1.0초 위치 x(m)": G[:, 1, 0],
          "1.5초 위치 x(m)": G[:, 2, 0], "2.0초 위치 x(m)": G[:, 3, 0],
          "S3 진행량(m)": gst.sum(-1), "3초 종방향 x(m)": G[:, 5, 0],
          "3초 횡방향 y(m)": G[:, 5, 1], "5초 종방향 x(m)": g5[:, 9, 0],
          "5초 횡방향 y(m)": g5[:, 9, 1], "3초 heading(deg)": np.degrees(ghd[:, 5]),
          "총 곡률(deg)": np.degrees(ghd[:, 5] - ghd[:, 0])}
    for nm, gv in GT.items():
        V = ATTR[nm][S]
        pick = np.abs(V - gv[:, None]).argmin(1)
        r = wm(np.take_along_axis(D3, pick[:, None], 1)[:, 0], miss)
        print(f"  {nm:>22} {r:.4f}   (GT 그 값을 알 때)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
