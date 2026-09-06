#!/usr/bin/env python
"""⑥-C: 완성형 parent-child bank 생성 (CPU).

설계 원칙
  1) parent = 기존 A0 1024 anchor 를 그대로 쓴다(호환/coarse score 재사용).
  2) child = **실제 train 궤적의 medoid** 를 쓴다. 합성 center 가 아니므로
     10-waypoint 5초 경로가 실제 주행이고 3->3.5초 연결 불연속이 원천적으로 0 이다.
  3) child 수는 parent 지지도에 비례하되 min/max 로 자른다.
     정지 parent 는 child 1개(정확히 0 벡터), 희소 좌/우회전은 최소 보장.

산출 지표
  full-bank D3 oracle / parent-top32,64 안의 child oracle /
  goal-addressability(같은 parent 내 5초 endpoint 퍼짐) / 지지도 / 연속성
"""
import argparse
import json
import os
import sys

import numpy as np

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402

W6 = np.array([11., 11., 5., 5., 2., 2.]); W6 /= W6.sum()


def d3_pair(a, b, chunk=256):
    """a [n,6,2], b [m,6,2] -> [n,m] 공식가중 D3."""
    out = np.empty((len(a), len(b)), np.float32)
    for s in range(0, len(a), chunk):
        d = np.linalg.norm(a[s:s + chunk, None] - b[None], axis=-1)   # [c,m,6]
        out[s:s + chunk] = (d * W6).sum(-1)
    return out


def kmedoid(X, k, seed=0, iters=12):
    """D3 기준 k-medoid. X [n,6,2] -> 선택된 행 index [k]."""
    n = len(X)
    if n <= k:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    D = d3_pair(X, X)
    # k-medoid++ 초기화
    med = [int(rng.integers(n))]
    d = D[med[0]].copy()
    for _ in range(1, k):
        p = d / max(d.sum(), 1e-12)
        med.append(int(rng.choice(n, p=p)))
        d = np.minimum(d, D[med[-1]])
    med = np.array(med)
    for _ in range(iters):
        lab = D[med].argmin(0)
        new = med.copy()
        for j in range(k):
            m = lab == j
            if not m.any():
                continue
            idx = np.where(m)[0]
            new[j] = idx[D[np.ix_(idx, idx)].sum(1).argmin()]
        if np.array_equal(new, med):
            break
        med = new
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-k", type=int, default=8192)
    ap.add_argument("--min-child", type=int, default=1)
    ap.add_argument("--max-child", type=int, default=16)
    ap.add_argument("--rare-min", type=int, default=2, help="좌/우회전 parent 최소 child")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)
    parents = bank["anchors_abs"].astype(np.float64)          # [1024,6,2]
    P = len(parents)

    s = C.make_split(arr)
    scen = arr["scen_idx"]
    tr_scen = np.unique(scen[s["train_rows"]])
    rows = np.where(np.isin(scen, tr_scen) & (arr["frame"] >= 30))[0]
    G3 = arr["fut"][rows].astype(np.float64)                  # [N,6,2]
    G5 = arr["fut5"][rows].astype(np.float64)                 # [N,10,2]
    M5 = arr["mask5"][rows]
    ok = M5.min(1) > 0.5
    rows, G3, G5 = rows[ok], G3[ok], G5[ok]
    N = len(G3)
    print(f"train 궤적 {N}행 (5초 전량 유효), parent {P}", flush=True)

    # 1) parent 배정
    assign = np.empty(N, np.int32)
    for st in range(0, N, 2048):
        assign[st:st + 2048] = d3_pair(G3[st:st + 2048], parents).argmin(1)
    sup = np.bincount(assign, minlength=P)
    print(f"parent 지지도: 0인 parent {int((sup==0).sum())}, "
          f"중앙값 {int(np.median(sup))}, 최대 {int(sup.max())}", flush=True)

    # 2) child 배분 (지지도 제곱근 비례 — 큰 군집 독식 방지)
    stop_p = int(np.linalg.norm(parents.reshape(P, -1), axis=1).argmin())
    w = np.sqrt(sup.astype(np.float64))
    w[stop_p] = 0.0
    nch = np.zeros(P, np.int32)
    live = sup > 0
    budget = args.target_k - 1                                # 정지 child 1개 예약
    if w.sum() > 0:
        nch = np.floor(budget * w / w.sum()).astype(np.int32)
    nch = np.clip(nch, 0, args.max_child)
    nch[live] = np.maximum(nch[live], args.min_child)
    # 희소 좌/우회전 보장: 3초 횡변위 |y|>2m 인 parent
    rare = live & (np.abs(parents[:, 5, 1]) > 2.0)
    nch[rare] = np.maximum(nch[rare], args.rare_min)
    nch[~live] = 0
    nch[stop_p] = 1
    nch = np.minimum(nch, np.maximum(sup, 1))                 # 지지도 초과 금지
    print(f"child 배분: 총 {int(nch.sum())} (목표 {args.target_k}), "
          f"활성 parent {int(live.sum())}, 희소(좌우) {int(rare.sum())}", flush=True)

    # 3) parent 별 medoid 추출
    kids_abs5, kids_parent, kids_support = [], [], []
    for p in range(P):
        if nch[p] == 0:
            continue
        idx = np.where(assign == p)[0]
        if p == stop_p:
            kids_abs5.append(np.zeros((10, 2)))
            kids_parent.append(p); kids_support.append(len(idx))
            continue
        sel = kmedoid(G3[idx], int(nch[p]), seed=p)
        for j in sel:
            kids_abs5.append(G5[idx[j]])
            kids_parent.append(p)
        # 지지도는 child 배정 기준으로 다시 센다
        lab = d3_pair(G3[idx], G3[idx][sel]).argmin(1)
        kids_support.extend(np.bincount(lab, minlength=len(sel)).tolist())
        if p % 200 == 0:
            print(f"  parent {p}/{P} 처리", flush=True)
    kids_abs5 = np.asarray(kids_abs5)                          # [K,10,2]
    kids_parent = np.asarray(kids_parent, np.int32)
    kids_support = np.asarray(kids_support, np.int32)
    K = len(kids_abs5)
    print(f"\nchild 생성 완료: K={K}", flush=True)

    # 4) 평가 (val38)
    vr = arr["val_idx"]
    vs = arr["frame"][vr] >= 30
    vr = vr[vs]
    vw = arr["val_weight"][vs].astype(np.float64)
    V3 = arr["fut"][vr].astype(np.float64)
    wm = lambda x: float(np.average(x, weights=vw))            # noqa: E731

    Dv_p = d3_pair(V3, parents)                                # [n,1024]
    Dv_c = d3_pair(V3, kids_abs5[:, :6])                       # [n,K]
    res = {"K": K, "target_k": args.target_k}
    res["parent_oracle"] = wm(Dv_p.min(1))
    res["full_oracle"] = wm(Dv_c.min(1))
    print(f"\n=== oracle (val38 n={len(vr)}) ===")
    print(f"  parent 1024 (=A0)          {res['parent_oracle']:.6f}")
    print(f"  full child bank K={K:<6}    {res['full_oracle']:.6f}")

    print("\n=== coarse-to-fine: parent 상위 t 개의 child 만 ===")
    for t in (16, 32, 64, 128):
        top = np.argpartition(Dv_p, t, axis=1)[:, :t]
        best = np.empty(len(vr))
        for i in range(len(vr)):
            m = np.isin(kids_parent, top[i])
            best[i] = Dv_c[i, m].min()
        res[f"top{t}_child_oracle"] = wm(best)
        nc = float(np.mean([np.isin(kids_parent, top[i]).sum() for i in range(0, len(vr), 20)]))
        print(f"  parent top-{t:<4} (child 평균 {nc:6.0f}개)  {wm(best):.6f}")

    print("\n=== 지지도 / 연속성 / goal 분별력 ===")
    print(f"  child 지지도: 최소 {int(kids_support.min())}, "
          f"중앙값 {int(np.median(kids_support))}, 1인 child {int((kids_support<=1).sum())}")
    step = np.linalg.norm(kids_abs5[:, 6] - kids_abs5[:, 5], axis=-1)
    prev = np.linalg.norm(kids_abs5[:, 5] - kids_abs5[:, 4], axis=-1)
    jump = np.abs(step - prev)
    print(f"  3->3.5초 연결 불연속: p50 {np.percentile(jump,50):.4f}m  "
          f"p99 {np.percentile(jump,99):.4f}m  (실제 궤적이라 0 이어야 정상)")
    ends = kids_abs5[:, 9]
    spread = []
    for p in np.unique(kids_parent):
        e = ends[kids_parent == p]
        if len(e) > 1:
            spread.append(float(np.linalg.norm(e - e.mean(0), axis=1).mean()))
    print(f"  같은 parent 내 5초 endpoint 퍼짐(goal 분별력): "
          f"평균 {np.mean(spread):.2f}m  중앙값 {np.median(spread):.2f}m")
    res["support_min"] = int(kids_support.min())
    res["jump_p99"] = float(np.percentile(jump, 99))
    res["goal_spread_mean"] = float(np.mean(spread))

    out = args.out or os.path.join(A, f"data/etri/bank_hier_K{K}.npz")
    np.savez_compressed(out, candidate_xy_abs_5s=kids_abs5.astype(np.float32),
                        parent_id=kids_parent, support=kids_support,
                        parents_abs=parents.astype(np.float32))
    json.dump(res, open(out.replace(".npz", "_report.json"), "w"), indent=1, default=float)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
