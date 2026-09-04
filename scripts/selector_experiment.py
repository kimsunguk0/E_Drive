#!/usr/bin/env python
"""visual-score-aware goal selection 실험 (학습 불필요, CPU).

병목: endpoint-only goal selector 가 다양한 후보 중 5s-endpoint 만 그럴듯한 goal
false-positive 를 고른다. 해법: 완성 후보 index 선택 cost 에 visual score 도 넣는다.
모델엔 goal 이 안 들어가고 완성 후보 index 만 고르므로 규정 준수.

    J_i = norm_goal_cost_i + λ * norm_visual_cost_i     (shortlist 내 min-max 정규화)
    visual_cost = max_logit - logit_i ;  goal_cost = ||5s_endpoint_i - provided_goal||
    λ=0 → endpoint-only(현재) ;  λ→∞ → visual top-1

shortlist: score12 / score6+nms6 / score3+nms9 / softnms12.
λ 선택은 시나리오 2-fold CV (final-val 과적합 방지). λ∈{0,0.1,0.25,0.5,1,2,5,inf}.

    python selector_experiment.py --dump champ_det.npz
"""
import argparse
import numpy as np

ECACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"


def d3_matrix(A, G, cw):
    M, K = G.shape[0], A.shape[0]
    out = np.empty((M, K), np.float64)
    for s in range(0, K, 128):
        a = A[s:s+128]
        out[:, s:s+128] = (np.linalg.norm(G[:, None] - a[None], axis=-1) * cw).sum(-1)
    return out


def build_5s_bank(anchors, cw, K):
    d = np.load(ECACHE, allow_pickle=True)
    tr = np.load(SPLIT, allow_pickle=True)["train_idx"]
    D = d3_matrix(anchors, d["fut"][tr].astype(np.float64), cw)
    assign = D.argmin(1)
    goal_tr = d["goal"][tr].astype(np.float64)
    bank = np.full((K, 2), np.nan)
    for k in range(K):
        m = assign == k
        if m.any():
            bank[k] = np.median(goal_tr[m], 0)
    bank[np.isnan(bank[:, 0])] = np.nanmedian(bank, 0)
    return bank


def make_shortlists(logits, Danch, tau):
    """4개 shortlist 를 후보 index 리스트로 반환 (각 앵커 아님, 벡터 밖에서 호출)."""
    order = np.argsort(-logits)

    def nms_fill(base_k, total, pool=64):
        sel = list(order[:base_k])
        for c in order[:pool]:
            if len(sel) >= total:
                break
            if c in sel:
                continue
            if not sel or Danch[c, sel].min() > tau:
                sel.append(c)
        for c in order:                      # 부족분 score 순 채움
            if len(sel) >= total:
                break
            if c not in sel:
                sel.append(c)
        return np.array(sel[:total])

    def soft_nms(total, sigma, pool=64):
        cand = list(order[:pool])
        sc = logits[cand].astype(np.float64).copy()
        chosen = []
        cand = np.array(cand)
        while len(chosen) < total and len(cand):
            j = int(np.argmax(sc))
            c = cand[j]; chosen.append(c)
            d = Danch[c, cand]
            # 표준 soft-nms: 남은 후보 점수를 근접도(gaussian)로 감쇠
            sc = sc * (1 - np.exp(-(d**2) / (2 * sigma**2)))
            sc[j] = -1e18
        return np.array(chosen[:total])

    return {
        "score12": order[:12],
        "score6+nms6": nms_fill(6, 12),
        "score3+nms9": nms_fill(3, 12),
        "softnms12": soft_nms(12, sigma=max(tau, 1e-3) * 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tau", type=float, default=0.0)
    args = ap.parse_args()
    z = np.load(args.dump, allow_pickle=True)
    logits = z["logits"].astype(np.float64); Dgt = z["Dgt"].astype(np.float64)
    wrow = z["wrow"].astype(np.float64); frame = z["frame"]; keep = z["keep"]
    scen = z["scen"].astype(str)
    anchors = z["anchors"].astype(np.float64); cw = z["cw"].astype(np.float64)
    N, K = logits.shape
    valid = keep & (frame >= 30) & np.isfinite(Dgt[:, 0])
    idx = np.where(valid)[0]
    Danch = d3_matrix(anchors, anchors, cw); np.fill_diagonal(Danch, 1e9)
    tau = args.tau or float(np.percentile(Danch.min(1), 50))
    bank5 = build_5s_bank(anchors, cw, K)
    goal_val = np.load(ECACHE, allow_pickle=True)["goal"].astype(np.float64)[
        np.load(SPLIT, allow_pickle=True)["val_idx"]]
    print(f"val(frame>=30)={len(idx)}  K={K}  tau={tau:.3f}")

    # 앵커별 shortlist 사전계산 + cost
    lam_grid = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, np.inf]
    SL = {}
    goalc = {}; visc = {}
    for n in idx:
        SL[n] = make_shortlists(logits[n], Danch, tau)
    methods = list(SL[idx[0]].keys())

    def realized(method, lam, rows):
        w = wrow[rows]; vals = np.empty(len(rows))
        for j, n in enumerate(rows):
            S = SL[n][method]
            gc = np.linalg.norm(bank5[S] - goal_val[n], axis=1)
            vc = logits[n].max() - logits[n][S]
            gc_n = (gc - gc.min()) / (gc.max() - gc.min() + 1e-9)
            vc_n = (vc - vc.min()) / (vc.max() - vc.min() + 1e-9)
            J = gc_n + (1e18 if np.isinf(lam) else lam) * vc_n
            c = S[int(np.argmin(J))]
            vals[j] = Dgt[n, c]
        return float(np.average(vals, weights=w))

    # 전체 λ-curve (참고)
    print("\n== goal-selection L2 vs λ (full val, 참고용) ==")
    print(f"{'method':13s}" + "".join(f"  λ={l}" for l in lam_grid))
    for m in methods:
        print(f"{m:13s}" + "".join(f"  {realized(m, l, idx):.4f}" for l in lam_grid))

    # 2-fold 시나리오 CV: fold 로 λ 고르고 반대 fold 에서 평가
    scen_u = np.unique(scen[idx])
    fold = {s: (i % 2) for i, s in enumerate(sorted(scen_u))}
    f0 = np.array([n for n in idx if fold[scen[n]] == 0])
    f1 = np.array([n for n in idx if fold[scen[n]] == 1])
    print(f"\n== 2-fold 시나리오 CV (λ는 반대 fold 에서 선택) ==  |f0|={len(f0)} |f1|={len(f1)}")
    print(f"{'method':13s}{'CV L2':>9}{'λ*(f0/f1)':>12}{'λ=0':>9}{'visual-top1':>12}")
    for m in methods:
        # fit on f1 -> eval f0
        l_for_f0 = min(lam_grid, key=lambda l: realized(m, l, f1))
        e0 = realized(m, l_for_f0, f0)
        l_for_f1 = min(lam_grid, key=lambda l: realized(m, l, f0))
        e1 = realized(m, l_for_f1, f1)
        cv = (e0 * len(f0) + e1 * len(f1)) / (len(f0) + len(f1))
        base0 = realized(m, 0.0, idx)
        vtop = realized(m, np.inf, idx)
        print(f"{m:13s}{cv:>9.4f}{str(l_for_f0)+'/'+str(l_for_f1):>12}"
              f"{base0:>9.4f}{vtop:>12.4f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
