#!/usr/bin/env python
"""Phase C 체크포인트 선택: 각 덤프에 대해 확정 selector(score3+nms9 + linear λ0.1)의
realized L2 + top-1(argmax) + oracle@12 를 계산. champion 과 비교. (오프라인, CPU)

    python eval_pc_selector.py dump1.npz dump2.npz ...
"""
import sys
import numpy as np
A = "/NHNHOME/data/sukim/adcl"
ECACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"


def d3_matrix(An, G, cw):
    M, K = G.shape[0], An.shape[0]
    out = np.empty((M, K), np.float64)
    for s in range(0, K, 128):
        a = An[s:s+128]
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


def evaluate(path, anchors, cw, Danch, tau, bank5, goal_val):
    z = np.load(path, allow_pickle=True)
    logits = z["logits"].astype(np.float64); Dgt = z["Dgt"].astype(np.float64)
    wrow = z["wrow"].astype(np.float64); frame = z["frame"]; keep = z["keep"]
    v = keep & (frame >= 30) & np.isfinite(Dgt[:, 0]); idx = np.where(v)[0]; w = wrow[idx]
    wm = lambda a: float(np.average(a, weights=w))  # noqa: E731
    top1 = wm(Dgt[idx, logits.argmax(1)[idx]])

    def shortlist(n):
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

    real = np.empty(len(idx)); orac = np.empty(len(idx))
    for j, n in enumerate(idx):
        S = shortlist(n)
        orac[j] = Dgt[n, S].min()
        gc = np.linalg.norm(bank5[S] - goal_val[n], axis=1)
        vc = logits[n].max() - logits[n][S]
        gcn = (gc - gc.min()) / (gc.max() - gc.min() + 1e-9)
        vcn = (vc - vc.min()) / (vc.max() - vc.min() + 1e-9)
        real[j] = Dgt[n, S[int(np.argmin(gcn + 0.1 * vcn))]]
    return top1, wm(orac), wm(real)


def main():
    paths = sys.argv[1:]
    z0 = np.load(paths[0], allow_pickle=True)
    anchors = z0["anchors"].astype(np.float64); cw = z0["cw"].astype(np.float64)
    Danch = d3_matrix(anchors, anchors, cw); np.fill_diagonal(Danch, 1e9)
    tau = float(np.percentile(Danch.min(1), 50))
    bank5 = build_5s_bank(anchors, cw)
    goal_val = np.load(ECACHE, allow_pickle=True)["goal"].astype(np.float64)[
        np.load(SPLIT, allow_pickle=True)["val_idx"]]
    ref = A + "/logs/dump/champ_det.npz"
    print(f"{'ckpt':26s}{'top-1':>9}{'oracle@12':>11}{'realized':>10}{'Δ vs champ':>12}")
    t1, o, r = evaluate(ref, anchors, cw, Danch, tau, bank5, goal_val)
    champ_real = r
    print(f"{'champion(softce_exp)':26s}{t1:>9.4f}{o:>11.4f}{r:>10.4f}{'—':>12}")
    for p in paths:
        name = p.split("/")[-1].replace(".npz", "")
        try:
            t1, o, r = evaluate(p, anchors, cw, Danch, tau, bank5, goal_val)
            print(f"{name:26s}{t1:>9.4f}{o:>11.4f}{r:>10.4f}{r-champ_real:>+12.4f}")
        except Exception as e:
            print(f"{name:26s}  ERROR {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
