#!/usr/bin/env python
"""⑤-M: 자차 과거 운동학 외삽을 selector 에 넣으면 실제로 얼마를 얻는가.

**채택 결정이 아니다.** 운영국 서면 확인 전까지 사용 불가.
확인 요청의 근거가 될 '얻는 양'을 정확히 재는 것이 목적이다.

구조: 후보 12행은 이미 완성돼 있고 selector 는 행 index 만 고른다.
운동학은 생성 그래프에 들어가지 않으며 goal counterfactual 도 불변이다.
그럼에도 ego status 사용이므로 현재 규정 해석으로는 금지 가능성이 높다.
"""
import argparse
import json
import os
import sys

import numpy as np

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402


def kin_pred(his, now, ts):
    """등가속 외삽. 과거 1.0초(속도 2구간)로 v,a 추정 -> 시각 ts 의 이동거리."""
    v1 = np.linalg.norm(his[:, now] - his[:, now - 5], axis=1) / 0.5
    v0 = np.linalg.norm(his[:, now - 5] - his[:, now - 10], axis=1) / 0.5
    a = (v1 - v0) / 0.5
    return {t: v1 * t + 0.5 * a * t * t for t in ts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(A, "logs/e14_kinematic_sel.json"))
    args = ap.parse_args()
    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)
    c5 = bank["candidate_xy_abs_5s"].astype(np.float64)
    now = C.HIS_NOW
    TS = (1.0, 2.0, 3.0)
    nz1 = lambda x: (x - x.min(1, keepdims=True)) / (
        x.max(1, keepdims=True) - x.min(1, keepdims=True) + 1e-9)      # noqa: E731

    store = {}
    for split in ("tune", "val"):
        z = np.load(os.path.join(A, f"data/etri/selector_{split}.npz"))
        S = z["shortlist"].astype(np.int64)
        D3, lg, goal = (z["D3"].astype(np.float64), z["logit"].astype(np.float64),
                        z["goal"].astype(np.float64))
        w = z["weight"].astype(np.float64)
        his = arr["his"][z["rows"]].astype(np.float64)
        pk = kin_pred(his, now, TS)
        # 후보의 해당 시각 종방향 위치 (index: 1.0s->1, 2.0s->3, 3.0s->5)
        TI = {1.0: 1, 2.0: 3, 3.0: 5}
        gd = np.linalg.norm(c5[S, 9] - goal[:, None], axis=-1)
        Jb = nz1(gd) + 0.1 * nz1(lg.max(1, keepdims=True) - lg)
        wm = lambda x: float(np.average(x, weights=w))                 # noqa: E731
        take = lambda i: np.take_along_axis(D3, i[:, None], 1)[:, 0]   # noqa: E731
        grid = {}
        for t in TS:
            cv = c5[:, TI[t], 0][S]
            dn = nz1(np.abs(cv - pk[t][:, None]))
            for mu in (0.0, 0.2, 0.4, 0.6, 1.0, 1.5, 2.0):
                grid[(t, mu)] = wm(take((Jb + mu * dn).argmin(1)))
        # 세 시각 동시 사용
        DN = sum(nz1(np.abs(c5[:, TI[t], 0][S] - pk[t][:, None])) for t in TS) / 3.0
        for mu in (0.0, 0.2, 0.4, 0.6, 1.0, 1.5, 2.0):
            grid[("all", mu)] = wm(take((Jb + mu * DN).argmin(1)))
        store[split] = dict(grid=grid, base=wm(take(Jb.argmin(1))),
                            oracle=wm(D3.min(1)), n=len(S), w=w)
        print(f"[{split}] n={len(S)}  기준 {store[split]['base']:.4f}  "
              f"oracle {store[split]['oracle']:.4f}", flush=True)
        for t in list(TS) + ["all"]:
            print(f"   {str(t):>4}: " + "  ".join(
                f"mu{mu}={grid[(t,mu)]:.4f}" for mu in (0.0, 0.2, 0.4, 0.6, 1.0, 1.5, 2.0)),
                flush=True)

    tg = store["tune"]["grid"]
    key = min(tg, key=lambda k: tg[k])
    vg = store["val"]["grid"]
    base = store["val"]["base"]
    got = vg[key]
    print(f"\n=== val38 (tune 고정 {key}) ===")
    print(f"  shortlist oracle       {store['val']['oracle']:.4f}")
    print(f"  현재 selector          {base:.4f}")
    print(f"  운동학 추가            {got:.4f}   이득 {base-got:+.4f}")
    print(f"  regret {base-store['val']['oracle']:.4f} -> {got-store['val']['oracle']:.4f}")
    print("\n  ** 채택 아님. 운영국 서면 확인 전까지 사용 금지. **")
    json.dump({"fixed": [str(key[0]), key[1]], "val_base": base, "val_kin": got,
               "val_oracle": store["val"]["oracle"],
               "tune_grid": {f"{a}|{b}": v for (a, b), v in tg.items()},
               "val_grid": {f"{a}|{b}": v for (a, b), v in vg.items()}},
              open(args.out, "w"), indent=1, default=float)
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
