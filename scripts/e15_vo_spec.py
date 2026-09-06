#!/usr/bin/env python
"""⑤-N: 영상 기반 VO head 의 사양 — 과거 v,a 를 얼마나 정확히 맞혀야 하는가.

⑤-M: GT 과거운동학 외삽으로 val38 0.3281 -> 0.2422 (+0.0858).
그 운동학은 ego status 라 그대로는 못 쓴다. 그러나 **과거** 자차 변위는
연속 프레임에서 복원 가능한 순수 영상량이다(visual odometry).

여기서는 v(과거 0.5초 속도)와 a(가속도)에 노이즈를 넣어
'VO head 가 어느 정밀도까지 가야 이득이 남는가'를 정한다.
speed head 가 실패한 이유와의 차이: 그것은 **미래** 진행량을 예측하려 했다.
"""
import os
import sys

import numpy as np

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402


def main():
    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)
    c5 = bank["candidate_xy_abs_5s"].astype(np.float64)
    now = C.HIS_NOW
    nz1 = lambda x: (x - x.min(1, keepdims=True)) / (
        x.max(1, keepdims=True) - x.min(1, keepdims=True) + 1e-9)      # noqa: E731
    rng = np.random.default_rng(0)

    out = {}
    for split in ("tune", "val"):
        z = np.load(os.path.join(A, f"data/etri/selector_{split}.npz"))
        S = z["shortlist"].astype(np.int64)
        D3 = z["D3"].astype(np.float64)
        lg = z["logit"].astype(np.float64)
        goal = z["goal"].astype(np.float64)
        w = z["weight"].astype(np.float64)
        his = arr["his"][z["rows"]].astype(np.float64)
        v1 = np.linalg.norm(his[:, now] - his[:, now - 5], axis=1) / 0.5
        v0 = np.linalg.norm(his[:, now - 5] - his[:, now - 10], axis=1) / 0.5
        acc = (v1 - v0) / 0.5
        gd = np.linalg.norm(c5[S, 9] - goal[:, None], axis=-1)
        Jb = nz1(gd) + 0.1 * nz1(lg.max(1, keepdims=True) - lg)
        # 기본인자로 묶어 late-binding 을 막는다(각 split 의 배열을 고정)
        wm = lambda x, _w=w: float(np.average(x, weights=_w))              # noqa: E731
        take = lambda i, _d=D3: np.take_along_axis(_d, i[:, None], 1)[:, 0]  # noqa: E731
        cv = c5[:, 1, 0][S]                       # 1.0초 목표(⑤-M 최적)
        out[split] = dict(v1=v1, acc=acc, cv=cv, Jb=Jb, D3=D3, w=w,
                          base=wm(take(Jb.argmin(1))), oracle=wm(D3.min(1)),
                          wm=wm, take=take)

    def run(d, sv, sa, mu, seed=0):
        r = np.random.default_rng(seed)
        v = d["v1"] + (r.normal(0, sv, len(d["v1"])) if sv > 0 else 0)
        a = d["acc"] + (r.normal(0, sa, len(d["acc"])) if sa > 0 else 0)
        p = np.maximum(v, 0) * 1.0 + 0.5 * a * 1.0
        dn = nz1(np.abs(d["cv"] - p[:, None]))
        return d["wm"](d["take"]((d["Jb"] + mu * dn).argmin(1)))

    print(f"기준 selector: tune {out['tune']['base']:.4f}  val {out['val']['base']:.4f}")
    print(f"shortlist oracle: tune {out['tune']['oracle']:.4f}  val {out['val']['oracle']:.4f}\n")
    print("목표 = 1.0초 뒤 종방향 위치 = v*1.0 + 0.5*a*1.0  (v=과거 0.5초 속도, a=가속도)")
    print(f"실제 분포: v 평균 {out['val']['v1'].mean():.2f} m/s, "
          f"a 표준편차 {out['val']['acc'].std():.2f} m/s^2\n")

    MUS = (0.4, 0.6, 1.0, 1.5, 2.0)
    print(f"{'v오차σ(m/s)':>12} {'a오차σ(m/s²)':>13} {'tune최선':>9} {'val적용':>9} {'val이득':>9}")
    for sv, sa in ((0.0, 0.0), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.3, 0.6),
                   (0.5, 1.0), (0.75, 1.5), (1.0, 2.0), (1.5, 3.0)):
        tv = {mu: run(out["tune"], sv, sa, mu, 1) for mu in MUS}
        mu0 = min(tv, key=lambda k: tv[k])
        vv = run(out["val"], sv, sa, mu0, 2)
        print(f"{sv:>12.2f} {sa:>13.2f} {tv[mu0]:>9.4f} {vv:>9.4f} "
              f"{out['val']['base']-vv:>+9.4f}")
    print("\n  참고: 순수 영상 VO 의 현실적 정밀도 목표는 v ~0.1-0.3 m/s 수준.")
    print("  (0.5초 간격 프레임, 알려진 intrinsic, 30km/h 에서 변위 ~4.2m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
