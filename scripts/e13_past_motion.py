#!/usr/bin/env python
"""⑤-L: 과거 자차 운동으로 2초 뒤 종방향 위치를 얼마나 맞히는가.

⑤-K: 요구 정밀도 sigma <= 0.5m. 영상 기반 추정자는 1.6~3.1m 로 미달.
그런데 temporal 모델은 history_alignment(정렬 행렬)를 **입력으로 받는다**.
거기서 과거 1.5초 변위 -> 현재 속도를 유도할 수 있다.

**주의**: 이건 compliance 판단이 필요한 경계다. 정렬 행렬은 과거 프레임을 맞추는
용도로 허용됐고, 거기서 자차 속도를 유도하는 것은 ego status 사용에 해당할 수 있다.
여기서는 '가치가 있는가'만 재고, 사용 여부는 운영국 서면 확인 뒤에 정한다.
"""
import os
import sys

import numpy as np

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402


def main():
    arr = C.load_arrays()
    z = np.load(os.path.join(A, "data/etri/selector_val.npz"))
    rows = z["rows"]
    w = z["weight"].astype(np.float64)
    his = arr["his"][rows].astype(np.float64)        # [N,31,2] 오래된순 0.1s
    G = arr["fut"][rows].astype(np.float64)          # [N,6,2] 앞으로 0.5s 간격
    now = C.HIS_NOW                                  # index 30 = 현재
    wm = lambda x: float(np.average(x, weights=w))   # noqa: E731

    print(f"val38 n={len(rows)}\n")
    print(f"{'추정자':>42} {'MAE(m)':>8} {'sigma근사':>9} {'상관':>7}")

    def rep(nm, p, gt):
        e = np.abs(p - gt)
        print(f"{nm:>42} {wm(e):>8.3f} {wm(e)/0.7979:>9.3f} {np.corrcoef(p,gt)[0,1]:>7.4f}")

    for ti, tn in ((1, "1.0초"), (3, "2.0초"), (5, "3.0초")):
        gt = G[:, ti, 0]
        t = 0.5 * (ti + 1)
        print(f"--- 목표: {tn} 뒤 종방향 위치 (GT 평균 {gt.mean():.2f}m) ---")
        # 등속 외삽: 최근 0.5초 변위로 속도 추정
        for back, bn in ((5, "0.5초"), (10, "1.0초"), (15, "1.5초")):
            v = (his[:, now] - his[:, now - back]) / (0.1 * back)   # [N,2] m/s
            rep(f"등속 외삽 (과거 {bn} 속도)", np.linalg.norm(v, axis=1) * t, gt)
        # 등가속: 최근 1.5초에서 속도+가속도
        v1 = np.linalg.norm(his[:, now] - his[:, now - 5], axis=1) / 0.5
        v0 = np.linalg.norm(his[:, now - 5] - his[:, now - 10], axis=1) / 0.5
        acc = (v1 - v0) / 0.5
        rep("등가속 외삽 (과거 1.0초)", v1 * t + 0.5 * acc * t * t, gt)
        # 가속도 클립
        rep("등가속 (가속도 +-2 m/s^2 클립)",
            v1 * t + 0.5 * np.clip(acc, -2, 2) * t * t, gt)
        print()

    print("요구(⑤-K): sigma <= 0.5m 에서 +0.024,  sigma <= 0.25m 에서 +0.085")
    print("영상 기반 최선: logit 유래 sigma~1.63m / speed head sigma~3.07m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
