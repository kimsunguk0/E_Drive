#!/usr/bin/env python
"""Phase B selector v2 (오프라인, champ_det.npz). winner=score3+nms9.

A. winner oracle/regret/goal-rank (M=3/6/12)
B. 순위기반 selector 규칙 vs linear fusion (2-fold CV, M=12)
C. regret 분해 (정지/가감속/command/종·횡/timing)

tie-break: stable argmax/argsort (동점→최소 인덱스=deployment).
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
    assign = D.argmin(1); goal_tr = d["goal"][tr].astype(np.float64)
    bank = np.full((anchors.shape[0], 2), np.nan)
    for k in range(anchors.shape[0]):
        m = assign == k
        if m.any():
            bank[k] = np.median(goal_tr[m], 0)
    bank[np.isnan(bank[:, 0])] = np.nanmedian(bank, 0)
    return bank


def main():
    z = np.load(A + "/logs/dump/champ_det.npz", allow_pickle=True)
    logits = z["logits"].astype(np.float64); Dgt = z["Dgt"].astype(np.float64)
    wrow = z["wrow"].astype(np.float64); frame = z["frame"]; keep = z["keep"]
    scen = z["scen"].astype(str); anchors = z["anchors"].astype(np.float64)
    cw = z["cw"].astype(np.float64)
    N, K = logits.shape
    v = keep & (frame >= 30) & np.isfinite(Dgt[:, 0]); idx = np.where(v)[0]; w = wrow[idx]
    wm = lambda a, ww=w: float(np.average(a, weights=ww))  # noqa: E731

    Danch = d3_matrix(anchors, anchors, cw); np.fill_diagonal(Danch, 1e9)
    tau = float(np.percentile(Danch.min(1), 50))
    bank5 = build_5s_bank(anchors, cw)
    d = np.load(ECACHE, allow_pickle=True); vi = np.load(SPLIT, allow_pickle=True)["val_idx"]
    goal_val = d["goal"][vi].astype(np.float64)
    cmd = d["vad_cmd"][vi].astype(int)          # 0=right,1=left,2=straight
    speed = d["speed"][vi].astype(np.float64); acc = d["acc"][vi].astype(np.float64)

    # anchor lateral direction: y<-1.5 right(0), y>1.5 left(1), else straight(2)
    lat = anchors[:, -1, 1]
    adir = np.where(lat < -1.5, 0, np.where(lat > 1.5, 1, 2))

    def shortlist(n):                            # score3 + nms9 (stable)
        order = np.argsort(-logits[n], kind="stable")
        sel = list(order[:3])
        for c in order[:64]:
            if len(sel) >= 12:
                break
            if c in sel:
                continue
            if Danch[c, sel].min() > tau:
                sel.append(c)
        for c in order:
            if len(sel) >= 12:
                break
            if c not in sel:
                sel.append(c)
        return np.array(sel[:12])

    SL = {n: shortlist(n) for n in idx}

    # ---- selector 규칙들 (S, n) -> 선택 index ----
    def gc_of(n, S):
        return np.linalg.norm(bank5[S] - goal_val[n], axis=1)

    def linear(n, S, lam=0.1):
        gc = gc_of(n, S); vc = logits[n].max() - logits[n][S]
        gcn = (gc - gc.min()) / (gc.max() - gc.min() + 1e-9)
        vcn = (vc - vc.min()) / (vc.max() - vc.min() + 1e-9)
        return S[int(np.argmin(gcn + lam * vcn))]

    def goal_topk_visual(n, S, k=2):
        gc = gc_of(n, S)
        cand = S[np.argsort(gc, kind="stable")[:k]]
        return cand[int(np.argmax(logits[n][cand]))]

    def visual_topk_goal(n, S, k=3):
        cand = S[np.argsort(-logits[n][S], kind="stable")[:k]]
        gc = np.linalg.norm(bank5[cand] - goal_val[n], axis=1)
        return cand[int(np.argmin(gc))]

    def rank_fusion(n, S, alpha=1.0):
        gc = gc_of(n, S)
        grank = np.argsort(np.argsort(gc, kind="stable"))          # 0=best goal
        vrank = np.argsort(np.argsort(-logits[n][S], kind="stable"))
        return S[int(np.argmin(grank + alpha * vrank))]

    def cmd_compatible(n, S):
        want = cmd[n]
        comp = S[adir[S] == want]
        return comp if len(comp) else S

    def goal2_visual_cmd(n, S):                  # rule5: command 필터 후 goal-top2+visual
        Sc = cmd_compatible(n, S)
        gc = np.linalg.norm(bank5[Sc] - goal_val[n], axis=1)
        cand = Sc[np.argsort(gc, kind="stable")[:2]]
        return cand[int(np.argmax(logits[n][cand]))]

    def realized(fn, rows, **kw):
        vals = np.array([Dgt[n, fn(n, SL[n], **kw)] for n in rows])
        return wm(vals, wrow[rows])

    # ---- A. winner oracle/regret/goal-rank ----
    print("== A. winner score3+nms9 oracle/regret/goal-rank ==")
    print(f"{'M':>3}{'oracle':>9}{'realized(lin.1)':>16}{'regret':>9}{'oracle후보 goal순위med':>20}")
    for M in [3, 6, 12]:
        orv = np.array([Dgt[n, SL[n][:M]].min() for n in idx])
        rzv = np.array([Dgt[n, linear(n, SL[n][:M])] for n in idx])
        grk = []
        for n in idx:
            S = SL[n][:M]; dd = Dgt[n, S]; gc = gc_of(n, S)
            grk.append(np.where(np.argsort(gc, kind="stable") == dd.argmin())[0][0] + 1)
        print(f"{M:>3}{wm(orv):>9.4f}{wm(rzv):>16.4f}{wm(rzv)-wm(orv):>9.4f}{np.median(grk):>20.0f}")

    # ---- B. selector 규칙 2-fold CV (M=12) ----
    scen_u = sorted(np.unique(scen[idx])); fold = {s: i % 2 for i, s in enumerate(scen_u)}
    f0 = np.array([n for n in idx if fold[scen[n]] == 0]); f1 = np.array([n for n in idx if fold[scen[n]] == 1])

    def cv(fn, **kw):
        e0 = realized(fn, f0, **kw); e1 = realized(fn, f1, **kw)
        return (e0 * len(f0) + e1 * len(f1)) / (len(f0) + len(f1))

    print("\n== B. selector 규칙 (M=12, full-val / 2-fold CV) ==")
    rules = [
        ("linear λ=0.1", lambda: (realized(linear, idx), cv(linear))),
        ("goal-top2→visual", lambda: (realized(goal_topk_visual, idx, k=2), cv(goal_topk_visual, k=2))),
        ("goal-top3→visual", lambda: (realized(goal_topk_visual, idx, k=3), cv(goal_topk_visual, k=3))),
        ("visual-top3→goal", lambda: (realized(visual_topk_goal, idx, k=3), cv(visual_topk_goal, k=3))),
        ("rank α=1", lambda: (realized(rank_fusion, idx, alpha=1.0), cv(rank_fusion, alpha=1.0))),
        ("rank α=0.5", lambda: (realized(rank_fusion, idx, alpha=0.5), cv(rank_fusion, alpha=0.5))),
        ("goal2+visual+cmd", lambda: (realized(goal2_visual_cmd, idx), cv(goal2_visual_cmd))),
    ]
    print(f"{'rule':20s}{'full':>9}{'CV':>9}")
    for name, f in rules:
        full, c = f()
        print(f"{name:20s}{full:>9.4f}{c:>9.4f}")
    print(f"{'endpoint-only(λ0)':20s}{realized(linear, idx, lam=0.0):>9.4f}")

    # ---- C. regret 분해 (best = goal-top2→visual) ----
    best = goal_topk_visual
    sel = np.array([best(n, SL[n], k=2) for n in idx])
    ora = np.array([SL[n][Dgt[n, SL[n]].argmin()] for n in idx])
    regret = Dgt[idx, sel] - Dgt[idx, ora]
    tot = wm(regret)
    print(f"\n== C. regret 분해 (selector=goal-top2→visual, mean regret={tot:.4f}) ==")
    acc_x = acc[:, 0]
    grp = {
        "정지(spd<0.5)": speed[idx] < 0.5, "이동": speed[idx] >= 0.5,
        "감속(ax<-0.5)": acc_x[idx] < -0.5, "등속": np.abs(acc_x[idx]) <= 0.5, "가속(ax>0.5)": acc_x[idx] > 0.5,
        "cmd우": cmd[idx] == 0, "cmd좌": cmd[idx] == 1, "cmd직": cmd[idx] == 2,
    }
    print(f"{'group':16s}{'n':>6}{'mean regret':>13}{'regret share':>14}")
    for name, m in grp.items():
        if m.sum() == 0:
            continue
        share = wm(regret * m) / (tot + 1e-9)
        print(f"{name:16s}{int(m.sum()):>6}{wm(regret[m], wrow[idx][m]):>13.4f}{share:>13.1%}")
    # 종/횡 오차 & timing
    end_sel = anchors[sel, -1]; end_gt = d["fut"][vi][idx][:, -1]
    long_err = np.abs(end_sel[:, 0] - end_gt[:, 0]); lat_err = np.abs(end_sel[:, 1] - end_gt[:, 1])
    print(f"\n선택 후보 endpoint 오차: 종(x) p50 {np.percentile(long_err,50):.2f} p90 {np.percentile(long_err,90):.2f}"
          f"  |  횡(y) p50 {np.percentile(lat_err,50):.2f} p90 {np.percentile(lat_err,90):.2f}")
    # endpoint 비슷(<2m)인데 regret 큰 = timing 문제
    endclose = long_err + lat_err < 2.0
    print(f"endpoint 근접(<2m) 비율 {endclose.mean():.1%}, 그중 regret mean {wm(regret[endclose], wrow[idx][endclose]):.4f} "
          f"(종방향 timing 문제 지표)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
