#!/usr/bin/env python
"""저장된 val logits 로 image-only shortlist 실험 (학습 불필요, CPU).

병목 분해: L_exp 가 회수한 것은 top-3 내부 selection regret 인가, 아니면 좋은 후보를
top-M 에 넣는 coverage 도 개선했나? 여러 shortlist 구성법의 weighted oracle@M 을 비교.

방법:
  score        : logits 내림차순 top-M
  nms          : trajectory NMS (score 순, 이미 뽑은 것과 D_anchor<tau 면 억제)
  bucket       : geometry-bucket(진행거리×lateral×curvature) balanced round-robin
  diversity    : score+diversity greedy (top-64 안에서 중복 억제 + 새 bucket 우선)

후보 간 거리 D(i,j)=Σ_t w_t ||a_i,t-a_j,t|| (공식 가중). oracle@M = min_{c∈S} Dgt[n,c].
goal-selection = shortlist 중 attached 5s-endpoint 가 provided 5s goal 에 가장 가까운 c.
  (5s-endpoint 는 train-only median: 각 train 앵커를 최근접 3s anchor 에 배정 후 goal median)

    python shortlist_experiment.py --champ champ.npz --base base.npz
"""
import argparse
import numpy as np

ECACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"


def d3_matrix(A, G, cw):
    """A[K,6,2] anchors, G[M,6,2] gts -> D[M,K] weighted."""
    # chunk over K to bound memory
    M = G.shape[0]; K = A.shape[0]
    out = np.empty((M, K), np.float64)
    step = 128
    for s in range(0, K, step):
        a = A[s:s+step]                                  # [c,6,2]
        diff = G[:, None] - a[None]                      # [M,c,6,2]
        dist = np.linalg.norm(diff, axis=-1)             # [M,c,6]
        out[:, s:s+step] = (dist * cw[None, None]).sum(-1)
    return out


def build_5s_bank(anchors, cw, K):
    d = np.load(ECACHE, allow_pickle=True)
    tr = np.load(SPLIT, allow_pickle=True)["train_idx"]
    gt_tr = d["fut"][tr].astype(np.float64)              # [T,6,2] 3s cumulative
    goal_tr = d["goal"][tr].astype(np.float64)           # [T,2] 5s endpoint
    D = d3_matrix(anchors, gt_tr, cw)                    # [T,K]
    assign = D.argmin(1)                                 # [T]
    bank = np.full((K, 2), np.nan)
    for k in range(K):
        m = assign == k
        if m.any():
            bank[k] = np.median(goal_tr[m], 0)
    gm = np.nanmedian(bank, 0)
    bank[np.isnan(bank[:, 0])] = gm                      # 빈 anchor fallback
    return bank


def anchor_geometry(anchors, cw):
    """진행거리(arclen), lateral endpoint(y), curvature(heading 변화량)."""
    steps = np.diff(np.concatenate(
        [np.zeros_like(anchors[:, :1]), anchors], 1), axis=1)   # [K,6,2]
    arclen = np.linalg.norm(steps, axis=-1).sum(1)              # [K]
    lat_end = anchors[:, -1, 1]                                 # y endpoint
    head = np.arctan2(steps[..., 1], steps[..., 0])            # [K,6]
    curv = np.abs(np.diff(head, axis=1)).sum(1)                 # heading 총변화
    return arclen, lat_end, curv


def bucket_ids(anchors, cw, n_prog=4, n_curv=2):
    arclen, lat_end, curv = anchor_geometry(anchors, cw)
    pb = np.clip((np.searchsorted(
        np.quantile(arclen, np.linspace(0, 1, n_prog + 1)[1:-1]), arclen)), 0, n_prog - 1)
    lb = np.where(lat_end < -1.5, 0, np.where(lat_end > 1.5, 2, 1))   # right/straight/left
    cb = (curv > np.median(curv)).astype(int)
    return pb * 6 + lb * 2 + cb                                  # 합성 bucket id


def sl_score(logits, M, **kw):
    return np.argsort(-logits)[:M]


def sl_nms(logits, M, Danch=None, tau=None, pool=64, **kw):
    order = np.argsort(-logits)[:pool]
    sel = []
    for c in order:
        if not sel or Danch[c, sel].min() > tau:
            sel.append(c)
        if len(sel) == M:
            break
    for c in order:                      # 부족하면 score 순으로 채움
        if len(sel) >= M:
            break
        if c not in sel:
            sel.append(c)
    return np.array(sel[:M])


def sl_bucket(logits, M, buckets=None, **kw):
    order = np.argsort(-logits)
    seen = {}; sel = []
    # round-robin: 각 bucket 최고점부터
    by_b = {}
    for c in order:
        by_b.setdefault(buckets[c], []).append(c)
    bl = sorted(by_b, key=lambda b: -logits[by_b[b][0]])
    ptr = {b: 0 for b in bl}
    while len(sel) < M:
        added = False
        for b in bl:
            if ptr[b] < len(by_b[b]):
                sel.append(by_b[b][ptr[b]]); ptr[b] += 1; added = True
                if len(sel) == M:
                    break
        if not added:
            break
    return np.array(sel[:M])


def sl_diversity(logits, M, Danch=None, tau=None, buckets=None, pool=64, **kw):
    order = np.argsort(-logits)[:pool]
    sel = [order[0]]                     # 첫 후보 = visual top-1 보존
    bset = {buckets[order[0]]}
    for c in order[1:]:
        if len(sel) >= M:
            break
        if Danch[c, sel].min() <= tau:   # 중복 억제
            continue
        sel.append(c); bset.add(buckets[c])
    for c in order:                      # 부족분 채움
        if len(sel) >= M:
            break
        if c not in sel:
            sel.append(c)
    return np.array(sel[:M])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champ", required=True)
    ap.add_argument("--base", default="")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="0=자동(최근접이웃 D_anchor p50)")
    args = ap.parse_args()
    z = np.load(args.champ, allow_pickle=True)
    logits = z["logits"].astype(np.float64)              # [N,K]
    Dgt = z["Dgt"].astype(np.float64)                    # [N,K]
    wrow = z["wrow"].astype(np.float64); frame = z["frame"]
    keep = z["keep"]; anchors = z["anchors"].astype(np.float64); cw = z["cw"].astype(np.float64)
    N, K = logits.shape
    valid = keep & (frame >= 30) & np.isfinite(Dgt[:, 0])
    idx = np.where(valid)[0]
    w = wrow[idx]
    print(f"anchors={K} val(frame>=30)={len(idx)}")

    # 후보 간 거리 D_anchor[K,K]
    Danch = d3_matrix(anchors, anchors, cw)             # [K,K]
    np.fill_diagonal(Danch, 1e9)
    nn = Danch.min(1)
    tau = args.tau or float(np.percentile(nn, 50))
    print(f"D_anchor 최근접이웃 p50={np.percentile(nn,50):.3f}  tau={tau:.3f}")
    buckets = bucket_ids(anchors, cw)
    print(f"buckets={len(np.unique(buckets))}")

    # 5s bank + provided goal (val)
    bank5 = build_5s_bank(anchors, cw, K)               # [K,2]
    val_idx = np.load(SPLIT, allow_pickle=True)["val_idx"]
    goal_all = np.load(ECACHE, allow_pickle=True)["goal"].astype(np.float64)
    goal_val = goal_all[val_idx]                         # [N,2] dump 순서와 동일

    methods = {"score": sl_score, "nms": sl_nms, "bucket": sl_bucket,
               "diversity": sl_diversity}
    Ms = [3, 6, 12, 20]

    def wmean(a):
        return float(np.average(a, weights=w))

    print("\n== weighted oracle@M (min Dgt over shortlist) ==")
    print(f"{'method':10s}" + "".join(f"  M={m:<2d}" for m in Ms))
    oracle_tab = {}
    for name, fn in methods.items():
        row = []
        for M in Ms:
            vals = np.empty(len(idx))
            for j, n in enumerate(idx):
                S = fn(logits[n], M, Danch=Danch, tau=tau, buckets=buckets)
                vals[j] = Dgt[n, S].min()
            row.append(wmean(vals))
        oracle_tab[name] = row
        print(f"{name:10s}" + "".join(f"  {v:.4f}" for v in row))

    print("\n== goal-selection L2 (shortlist 중 5s-endpoint가 goal 최근접) ==")
    print(f"{'method':10s}" + "".join(f"  M={m:<2d}" for m in Ms))
    for name, fn in methods.items():
        row = []
        for M in Ms:
            vals = np.empty(len(idx))
            for j, n in enumerate(idx):
                S = fn(logits[n], M, Danch=Danch, tau=tau, buckets=buckets)
                c = S[np.argmin(np.linalg.norm(bank5[S] - goal_val[n], axis=1))]
                vals[j] = Dgt[n, c]
            row.append(wmean(vals))
        print(f"{name:10s}" + "".join(f"  {v:.4f}" for v in row))

    # top-1 (score) 참고
    top1 = np.array([Dgt[n, logits[n].argmax()] for n in idx])
    print(f"\ntop-1 (score) weighted = {wmean(top1):.4f}")

    # set-overlap: champion top-3 vs baseline top-3
    if args.base:
        b = np.load(args.base, allow_pickle=True)["logits"].astype(np.float64)
        jac = []
        for n in idx:
            c3 = set(np.argsort(-logits[n])[:3]); b3 = set(np.argsort(-b[n])[:3])
            jac.append(len(c3 & b3) / len(c3 | b3))
        print(f"\ntop-3 set overlap (champ vs baseline) Jaccard mean = {np.mean(jac):.3f}  "
              f"(1.0=동일집합, 순서만 바뀜)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
