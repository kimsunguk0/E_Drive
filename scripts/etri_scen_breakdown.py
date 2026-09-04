#!/usr/bin/env python
"""저장된 pred npy로 **시나리오별** 챌린지 L2를 낸다 (게이트 11 판정용).

`etri_vad_eval.py`는 집계 행만 찍는다. 형님 §7 기준은 시나리오 단위다.
    최소 통과: 전체 ≤ 1.0 / 8개 중 6개 이상 ≤ 1.0 / 최악 ≤ 2.5
    강한 통과: 전체 ≤ 0.5 / 8개 모두 ≤ 1.0

    python scripts/etri_scen_breakdown.py /tmp/pm97/eval_fixed/ep08_pred.npy ...
"""
import argparse
import os
import pickle
import sys

import numpy as np

CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SUB = "/tmp/pm97/data/etri/pkl/overfit8.pkl"
W = np.array([11.0, 11.0, 5.0, 5.0, 2.0, 2.0]) / 36.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    args = ap.parse_args()

    d = np.load(CACHE, allow_pickle=True)
    scen_all = np.array([str(x) for x in d["scenarios"]])[d["scen_idx"]]
    frame_all = d["frame"].astype(int)
    want = sorted({i["scene_token"] for i in
                   pickle.load(open(SUB, "rb"))["infos"]})
    idx = np.where(np.isin(scen_all, want) & (frame_all % 5 == 0))[0]
    idx = idx[np.lexsort((frame_all[idx], scen_all[idx]))]
    gt = d["fut"][idx].astype(np.float64)
    scen = scen_all[idx]
    speed = d["speed"][idx].astype(np.float64)

    for p in args.preds:
        P = np.load(p)
        if P.shape != gt.shape:
            print(f"{p}: shape {P.shape} != {gt.shape} -- 건너뜀")
            continue
        per = (np.sqrt(((P - gt) ** 2).sum(-1)) * W).sum(-1)
        print(f"\n=== {os.path.basename(p)} ===")
        print(f"전체 {per.mean():.4f}   p50 {np.median(per):.4f}   "
              f"p90 {np.percentile(per,90):.4f}   최악앵커 {per.max():.4f}")
        print(f"{'시나리오':<20}{'앵커':>5}{'L2':>9}{'최악':>9}{'속도p50':>9}"
              f"{'>1.0':>7}")
        means = []
        for s in want:
            m = scen == s
            means.append(per[m].mean())
            print(f"{s:<20}{int(m.sum()):>5}{per[m].mean():>9.4f}"
                  f"{per[m].max():>9.4f}{np.median(speed[m]):>9.2f}"
                  f"{int((per[m]>1.0).sum()):>7}")
        means = np.array(means)
        ok_min = (per.mean() <= 1.0 and (means <= 1.0).sum() >= 6
                  and means.max() <= 2.5)
        ok_str = per.mean() <= 0.5 and (means <= 1.0).all()
        print(f"\n  ≤1.0 시나리오 {int((means<=1.0).sum())}/8   최악 시나리오 "
              f"{means.max():.4f}")
        print(f"  최소 통과(전체≤1.0 · 6개이상≤1.0 · 최악≤2.5): "
              f"{'PASS' if ok_min else 'FAIL'}")
        print(f"  강한 통과(전체≤0.5 · 모두≤1.0):              "
              f"{'PASS' if ok_str else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
