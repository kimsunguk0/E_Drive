#!/usr/bin/env python
"""Phase B selector 확정 (오프라인, 덤프 logits). tie-break 를 deployment(argmax=최소
인덱스)로 통일하고 λ=∞ 불변식을 강제한다.

- canonical top-1 = np.argmax (torch.argmax 와 동일 = 동점 시 최소 인덱스 = stop 우선)
- 모든 shortlist 정렬 = argsort(kind='stable') → 동점 시 최소 인덱스 우선 → S[0]==argmax
- selector J = norm_goal + λ·norm_visual ; λ=inf 는 순수 visual(S[0]) 특수 처리
- 불변식: S[:,0]==argmax, λ=inf 선택==argmax
- hybrid oracle + selector regret, goal-distance rank 분석
"""
import numpy as np
A = "/NHNHOME/data/sukim/adcl"
ECACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"


def d3_matrix(A_, G, cw):
    M, K = G.shape[0], A_.shape[0]
    out = np.empty((M, K), np.float64)
    for s in range(0, K, 128):
        a = A_[s:s+128]
        out[:, s:s+128] = (np.linalg.norm(G[:, None] - a[None], axis=-1) * cw).sum(-1)
    return out


def build_5s_bank(anchors, cw):
    d = np.load(ECACHE, allow_pickle=True)
    tr = np.load(SPLIT, allow_pickle=True)["train_idx"]
    D = d3_matrix(anchors, d["fut"][tr].astype(np.float64), cw)
    assign = D.argmin(1)
    goal_tr = d["goal"][tr].astype(np.float64)
    bank = np.full((anchors.shape[0], 2), np.nan)
    for k in range(anchors.shape[0]):
        m = assign == k
        if m.any():
            bank[k] = np.median(goal_tr[m], 0)
    bank[np.isnan(bank[:, 0])] = np.nanmedian(bank, 0)
    return bank


def stable_order(logvec):
    # 동점 시 최소 인덱스 우선 (argmax 와 일치)
    return np.argsort(-logvec, kind="stable")


def main():
    z = np.load(A + "/logs/dump/champ_det.npz", allow_pickle=True)
    logits = z["logits"].astype(np.float64); Dgt = z["Dgt"].astype(np.float64)
    wrow = z["wrow"].astype(np.float64); frame = z["frame"]; keep = z["keep"]
    scen = z["scen"].astype(str); anchors = z["anchors"].astype(np.float64)
    cw = z["cw"].astype(np.float64)
    b = np.load(A + "/logs/dump/base.npz", allow_pickle=True)
    blog = b["logits"].astype(np.float64); bDgt = b["Dgt"].astype(np.float64)
    N, K = logits.shape
    v = keep & (frame >= 30) & np.isfinite(Dgt[:, 0]); idx = np.where(v)[0]; w = wrow[idx]
    wm = lambda a: float(np.average(a, weights=w))  # noqa: E731

    argmax = logits.argmax(1)
    champ_top1 = wm(Dgt[idx, argmax[idx]])
    base_top1 = wm(bDgt[idx, blog.argmax(1)[idx]])
    print(f"canonical top-1 (argmax tie-break): champion={champ_top1:.4f}  baseline={base_top1:.4f}")

    Danch = d3_matrix(anchors, anchors, cw); np.fill_diagonal(Danch, 1e9)
    tau = float(np.percentile(Danch.min(1), 50))
    bank5 = build_5s_bank(anchors, cw)
    goal_val = np.load(ECACHE, allow_pickle=True)["goal"].astype(np.float64)[
        np.load(SPLIT, allow_pickle=True)["val_idx"]]

    def shortlist(n, method):
        order = stable_order(logits[n])

        def nms_fill(base_k, total, pool=64):
            sel = list(order[:base_k])
            for c in order[:pool]:
                if len(sel) >= total:
                    break
                if c in sel:
                    continue
                if Danch[c, sel].min() > tau:
                    sel.append(c)
            for c in order:
                if len(sel) >= total:
                    break
                if c not in sel:
                    sel.append(c)
            return np.array(sel[:total])
        if method == "score12":
            return order[:12]
        if method == "score6+nms6":
            return nms_fill(6, 12)
        if method == "score3+nms9":
            return nms_fill(3, 12)
        raise ValueError(method)

    methods = ["score12", "score6+nms6", "score3+nms9"]

    # ---- 불변식 검증 ----
    ok_s0 = True; ok_inf = True
    for n in idx:
        for m in methods:
            S = shortlist(n, m)
            if S[0] != argmax[n]:
                ok_s0 = False
    print(f"\n[불변식] 모든 shortlist S[0]==argmax : {'PASS' if ok_s0 else 'FAIL'}")

    def select(n, S, lam):
        gc = np.linalg.norm(bank5[S] - goal_val[n], axis=1)
        vc = logits[n].max() - logits[n][S]
        if np.isinf(lam):
            return S[0]                        # 순수 visual top-1 (deployment)
        gcn = (gc - gc.min()) / (gc.max() - gc.min() + 1e-9)
        vcn = (vc - vc.min()) / (vc.max() - vc.min() + 1e-9)
        return S[int(np.argmin(gcn + lam * vcn))]

    # λ=inf 불변식
    for n in idx[:500]:
        if select(n, shortlist(n, "score12"), np.inf) != argmax[n]:
            ok_inf = False
    print(f"[불변식] λ=inf 선택==argmax : {'PASS' if ok_inf else 'FAIL'}  "
          f"(λ=inf realized should == canonical {champ_top1:.4f})")

    lam_grid = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, np.inf]

    def realized(method, lam, rows):
        ww = wrow[rows]; vals = np.array([Dgt[n, select(n, shortlist(n, method), lam)] for n in rows])
        return float(np.average(vals, weights=ww))

    print("\n== goal-selection L2 vs λ (full val) ==")
    print(f"{'method':13s}" + "".join(f"  λ={l}" for l in lam_grid))
    for m in methods:
        print(f"{m:13s}" + "".join(f"  {realized(m, l, idx):.4f}" for l in lam_grid))

    # 2-fold CV
    scen_u = sorted(np.unique(scen[idx]))
    fold = {s: i % 2 for i, s in enumerate(scen_u)}
    f0 = np.array([n for n in idx if fold[scen[n]] == 0])
    f1 = np.array([n for n in idx if fold[scen[n]] == 1])
    grid = [l for l in lam_grid if not np.isinf(l)]
    print("\n== 2-fold CV (λ*는 반대 fold) ==")
    print(f"{'method':13s}{'CV L2':>9}{'λ*(f0/f1)':>12}{'λ=0':>8}")
    for m in methods:
        l0 = min(grid, key=lambda l: realized(m, l, f1)); e0 = realized(m, l0, f0)
        l1 = min(grid, key=lambda l: realized(m, l, f0)); e1 = realized(m, l1, f1)
        cv = (e0 * len(f0) + e1 * len(f1)) / (len(f0) + len(f1))
        print(f"{m:13s}{cv:>9.4f}{f'{l0}/{l1}':>12}{realized(m,0.0,idx):>8.4f}")

    # ---- hybrid oracle + regret + goal-distance rank ----
    print("\n== hybrid(score6+nms6) oracle / regret / goal-rank ==")
    print(f"{'M':>3}{'oracle':>9}{'realized(λ=0.1)':>16}{'regret':>9}{'goalrank(oracle후보)':>22}")
    for M in [3, 6, 12]:
        orv = np.empty(len(idx)); rzv = np.empty(len(idx)); grk = np.empty(len(idx))
        for j, n in enumerate(idx):
            S = shortlist(n, "score6+nms6")[:M]
            dd = Dgt[n, S]
            orv[j] = dd.min()
            rzv[j] = Dgt[n, select(n, S, 0.1)]
            # oracle 후보가 goal-distance 순위에서 몇 번째인가
            gc = np.linalg.norm(bank5[S] - goal_val[n], axis=1)
            grk[j] = int(np.where(np.argsort(gc, kind="stable") == dd.argmin())[0][0]) + 1
        oc = wm(orv); rz = wm(rzv)
        print(f"{M:>3}{oc:>9.4f}{rz:>16.4f}{rz-oc:>9.4f}{np.median(grk):>22.0f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
