#!/usr/bin/env python
"""⑤-E2: selector regret 원인분해 (CPU, dump 기반).

  python e2_selector_analysis.py --tag sc_e1_gt

측정 (user spec):
  - oracle 후보의 goal-distance 순위 분포
  - endpoint <2m 안에서의 regret (= 도착지는 맞는데 타이밍이 틀린 비중)
  - 종/횡 endpoint 오차
  - stop/accel regret
  - visual rank 와 goal rank 의 상관
  - λ 를 tune scenario CV 에서 고정 -> val 에 적용
"""
import argparse
import os
import sys

import numpy as np

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402

DUMP = os.path.join(A, "logs/dump_sparse_c")
LAMS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8)


def load(tag, split, arr, bank):
    z = np.load(os.path.join(DUMP, f"logits_{tag}_{split}.npz"), allow_pickle=True)
    rows = z["rows"]
    T = C.precompute_targets(arr, rows, bank, weight=z["weight"])
    return z["logits"].astype(np.float64), rows, T


def per_frame(lg, T, lam):
    """프레임별 shortlist / 선택 / regret 성분."""
    K = lg.shape[1]
    ad, tau = T["anchor_dist"], T["nms_tau"]
    ce, gx, Dg = T["cand_end5"], T["goal_xy"], T["D3gt"]
    out = dict(sl_or=[], real=[], sel=[], orc=[], grank=[], vrank=[], nsl=[])
    for n in range(len(lg)):
        S = C._shortlist(lg[n], ad, tau)
        d = Dg[n][S]
        orc = int(S[int(np.argmin(d))])                 # shortlist 내 최적 후보
        gc = np.linalg.norm(ce[S] - gx[n], axis=1)
        vc = lg[n].max() - lg[n][S]
        gcn = (gc - gc.min()) / (gc.max() - gc.min() + 1e-9)
        vcn = (vc - vc.min()) / (vc.max() - vc.min() + 1e-9)
        j = int(np.argmin(gcn + lam * vcn))
        out["sl_or"].append(d.min())
        out["real"].append(Dg[n][S[j]])
        out["sel"].append(int(S[j]))
        out["orc"].append(orc)
        # oracle 후보의 goal 거리 순위 / visual 순위 (shortlist 내)
        out["grank"].append(int(np.where(np.argsort(gc, kind="stable") ==
                                         int(np.argmin(d)))[0][0]) + 1)
        out["vrank"].append(int(np.where(np.argsort(vc, kind="stable") ==
                                         int(np.argmin(d)))[0][0]) + 1)
        out["nsl"].append(len(S))
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sc_e1_gt")
    args = ap.parse_args()
    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)

    # ---- 1) tune scenario CV 로 λ 고정 ----
    lg_t, rows_t, T_t = load(args.tag, "tune", arr, bank)
    scen_t = arr["scen_idx"][rows_t]
    uniq = np.unique(scen_t)
    folds = [uniq[0::2], uniq[1::2]]
    print("=== 1) tune scenario 2-fold CV 로 λ 고정 ===")
    fold_best = []
    for fi, f in enumerate(folds):
        m = np.isin(scen_t, f)
        best, bl = 1e9, None
        for lam in LAMS:
            r = per_frame(lg_t[m], {k: (v[m] if isinstance(v, np.ndarray) and
                                        v.shape[:1] == (len(lg_t),) else v)
                                    for k, v in T_t.items()}, lam)
            val = float(np.average(r["real"], weights=T_t["weight"][m]))
            if val < best:
                best, bl = val, lam
        fold_best.append(bl)
        print(f"  fold{fi} (scene {len(f)}): 최적 λ={bl}  realized={best:.4f}")
    lam_fix = fold_best[0] if fold_best[0] == fold_best[1] else float(
        np.mean(fold_best))
    print(f"  --> 고정 λ = {lam_fix}  (두 fold 일치={fold_best[0]==fold_best[1]})")

    # ---- 2) val 에 적용 + regret 분해 ----
    lg_v, rows_v, T_v = load(args.tag, "val", arr, bank)
    R = per_frame(lg_v, T_v, lam_fix)
    w = T_v["weight"]
    wm = lambda a: float(np.average(a, weights=w))  # noqa: E731
    regret = R["real"] - R["sl_or"]
    print("\n=== 2) val38, tune 고정 λ 적용 ===")
    print(f"  shortlist oracle = {wm(R['sl_or']):.4f}")
    print(f"  realized         = {wm(R['real']):.4f}")
    print(f"  selector regret  = {wm(regret):.4f}")
    print(f"  선택==oracle 비율 = {wm((R['sel']==R['orc']).astype(float))*100:.1f}%")

    # ---- 3) oracle 후보의 goal / visual 순위 ----
    print("\n=== 3) shortlist 내 oracle 후보의 순위 ===")
    for nm, key in (("goal 거리 순위", "grank"), ("visual 순위", "vrank")):
        r = R[key]
        print(f"  {nm}: 중앙값 {np.median(r):.0f}  1위 {100*(r==1).mean():.1f}%  "
              f"top3 {100*(r<=3).mean():.1f}%  평균 {r.mean():.2f}")
    print(f"  goal rank 와 visual rank 상관(Spearman 근사) = "
          f"{np.corrcoef(R['grank'], R['vrank'])[0,1]:.3f}")

    # ---- 4) endpoint 오차 분해 ----
    ce, gx = T_v["cand_end5"], T_v["goal_xy"]
    sel_end = ce[R["sel"]]
    err = sel_end - gx
    dist = np.linalg.norm(err, axis=1)
    near = dist < 2.0
    print("\n=== 4) 선택 후보의 5초 endpoint 오차 ===")
    print(f"  |err| p50={np.percentile(dist,50):.2f} p90={np.percentile(dist,90):.2f}m")
    print(f"  종방향 |x| p50={np.percentile(np.abs(err[:,0]),50):.2f} "
          f"p90={np.percentile(np.abs(err[:,0]),90):.2f}m")
    print(f"  횡방향 |y| p50={np.percentile(np.abs(err[:,1]),50):.2f} "
          f"p90={np.percentile(np.abs(err[:,1]),90):.2f}m")
    print(f"  endpoint <2m 인 프레임: {100*near.mean():.1f}%  "
          f"그 안에서의 regret={float(np.average(regret[near],weights=w[near])):.4f}  "
          f"(전체 regret 의 {100*np.average(regret[near],weights=w[near])*w[near].sum()/(np.average(regret,weights=w)*w.sum()):.0f}%)")

    # ---- 5) 버킷별 regret ----
    print("\n=== 5) 버킷별 regret 기여 ===")
    tot = np.average(regret, weights=w) * w.sum()
    for k, m in T_v["buckets"].items():
        m = np.asarray(m, bool)
        if m.sum() == 0:
            continue
        share = np.average(regret[m], weights=w[m]) * w[m].sum() / tot
        print(f"  {k:8s} n={m.sum():4d} ({100*w[m].sum()/w.sum():4.1f}% 가중) "
              f"regret={float(np.average(regret[m],weights=w[m])):.4f}  "
              f"전체 regret 기여 {100*share:4.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
