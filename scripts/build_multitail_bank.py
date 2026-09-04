#!/usr/bin/env python
"""② anchor-preserving multi-tail bank (설계서 §5.2 Milestone A).

고정 조건 (user spec):
  - 데이터: train_idx 중 frame>=30 (17,820)
  - assignment: 공식 D3 [11,11,5,5,2,2]/36 (3초 prefix)
  - tail 표현: fut5[:,6:10] - fut5[:,5:6]
  - 완성: anchor[:,5] + relative_tail
  - 첫 6점: 원 anchor buffer 그대로 복사 (재계산 금지)
  - mode 수: min(3, n//12), 최소 1
  - 전체 후보 <= 2048 (초과 시 저빈도 mode 병합)
  - 희소 fallback: train-only anchor 이웃만
  - candidate 0: 10점 exact-zero 보존

산출:
  A0.npz  bank[1024,10,2]           (대표 median tail 하나)
  A1.npz  bank[K',10,2] (<=2048)    + parent_anchor_id/tail_mode_id/support_count/fallback_source
검증표 전부 출력 + report 저장.
"""
import numpy as np, hashlib, json

A = "/NHNHOME/data/sukim/adcl"
ANCH_P = A + "/h200_latest/artifacts/dense_vocab/etri_train330_vocab_k1024.npy"
E5_P = "/tmp/pm97/data/etri/ego_cache_5s.npz"
SPLIT_P = "/tmp/pm97/data/etri/val_clips.npz"
CHAMP_P = A + "/logs/dump/champ_det.npz"
OUT_A0 = A + "/data/etri/bank_A0_onetail.npz"
OUT_A1 = A + "/data/etri/bank_A1_multitail.npz"
REPORT = A + "/logs/bank_A_report.json"
MIN_COUNT = 12
MAX_MODE = 3
MAX_CAND = 2048
STOP_TAIL_THRESH = 0.5   # tail endpoint |.|<thr = continued-stop, else delayed-start
CW = np.array([11, 11, 5, 5, 2, 2], np.float64) / 36.0


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def d3_3s(An, G):
    """D3(3초) 거리행렬 [M,K].  An[K,6,2], G[M,6,2]."""
    M, K = G.shape[0], An.shape[0]
    out = np.empty((M, K), np.float64)
    for s in range(0, K, 256):
        a = An[s:s + 256]
        out[:, s:s + 256] = (np.linalg.norm(G[:, None] - a[None], axis=-1) * CW).sum(-1)
    return out


def kmeans(X, k, iters=25):
    """작은 deterministic k-means (k<=3, 8D). 반환: 라벨, 각 cluster 좌표별 median tail."""
    n = X.shape[0]
    if k <= 1 or n <= k:
        lab = np.zeros(n, int)
        return lab, np.array([np.median(X, 0)])
    # deterministic init: 1D projection 분위수 지점
    proj = X @ (X.std(0) + 1e-9)
    q = np.quantile(proj, np.linspace(0.1, 0.9, k))
    cent = np.array([X[np.argmin(np.abs(proj - v))] for v in q], float)
    lab = np.zeros(n, int)
    for _ in range(iters):
        d = ((X[:, None] - cent[None]) ** 2).sum(-1)
        nl = d.argmin(1)
        if np.array_equal(nl, lab):
            break
        lab = nl
        for c in range(k):
            m = lab == c
            if m.any():
                cent[c] = X[m].mean(0)
    reps = np.array([np.median(X[lab == c], 0) if (lab == c).any()
                     else np.median(X, 0) for c in range(k)])
    return lab, reps


def main():
    An = np.load(ANCH_P).astype(np.float64)            # [1024,6,2] cumulative
    K = An.shape[0]
    e5 = np.load(E5_P, allow_pickle=True)
    fut5 = e5["fut5"].astype(np.float64)               # [N,10,2] cumulative
    frame_all = e5["frame"]
    sp = np.load(SPLIT_P, allow_pickle=True)
    tr = sp["train_idx"]
    tr = tr[frame_all[tr] >= 30]                        # 17,820
    assert tr.shape[0] == 17820, tr.shape
    G = fut5[tr]                                        # [17820,10,2]
    G3 = G[:, :6]
    tail_rel = G[:, 6:10] - G[:, 5:6]                   # [17820,4,2]  endpoint 상대

    # ---- assignment (official D3, 3s) ----
    D = d3_3s(An, G3)
    assign = D.argmin(1)
    counts = np.bincount(assign, minlength=K)

    # ---- anchor-anchor D3 for fallback neighbors ----
    Da = d3_3s(An, An); np.fill_diagonal(Da, 1e18)
    sufficient = counts >= MIN_COUNT

    # ---- per-anchor A0 median tail + A1 modes ----
    A0 = np.zeros((K, 10, 2))
    A0[:, :6] = An                                      # 첫6점 원본 복사 (bitwise)
    A0_fallback = np.full(K, -1, int)
    a1_rows, a1_parent, a1_mode, a1_supp, a1_fb = [], [], [], [], []

    def anchor_tails(k):
        """(tail_rel_of_k, fallback_source). 희소면 이웃 train-only anchor."""
        if sufficient[k]:
            return tail_rel[assign == k], -1
        # nearest sufficient anchor
        order = np.argsort(Da[k])
        for nb in order:
            if sufficient[nb]:
                return tail_rel[assign == nb], int(nb)
        return tail_rel[assign == counts.argmax()], int(counts.argmax())

    for k in range(K):
        T, fb = anchor_tails(k)
        A0_fallback[k] = fb
        n_eff = T.shape[0]
        A0[k, 6:10] = An[k, 5] + np.median(T, 0)
        nmode = max(1, min(MAX_MODE, n_eff // MIN_COUNT))
        X = T.reshape(n_eff, -1)
        lab, reps = kmeans(X, nmode)
        for c in range(reps.shape[0]):
            supp = int((lab == c).sum())
            if supp == 0:
                continue
            row = np.zeros((10, 2))
            row[:6] = An[k]
            row[6:10] = An[k, 5] + reps[c].reshape(4, 2)
            a1_rows.append(row); a1_parent.append(k); a1_mode.append(c)
            a1_supp.append(supp); a1_fb.append(fb)

    # ---- stop candidate 0: exact-zero 보존 ----
    A0[0] = 0.0
    # A1: parent0 의 continued-stop(=zero) 모드를 정확히 zero 로, row0 에 배치
    a1_rows = np.array(a1_rows); a1_parent = np.array(a1_parent)
    a1_mode = np.array(a1_mode); a1_supp = np.array(a1_supp); a1_fb = np.array(a1_fb)
    p0 = np.where(a1_parent == 0)[0]
    # parent0 의 각 mode: tail endpoint |.| < thr 이면 continued-stop → exact zero
    for i in p0:
        if np.linalg.norm(a1_rows[i, 9]) < STOP_TAIL_THRESH:
            a1_rows[i] = 0.0
    # 하나 이상 exact-zero 보장 + 맨 앞으로
    zmask = (np.abs(a1_rows).reshape(len(a1_rows), -1).sum(1) == 0)
    if not zmask.any():
        z = np.zeros((1, 10, 2))
        a1_rows = np.concatenate([z, a1_rows]); a1_parent = np.concatenate([[0], a1_parent])
        a1_mode = np.concatenate([[0], a1_mode]); a1_supp = np.concatenate([[int(counts[0])], a1_supp])
        a1_fb = np.concatenate([[-1], a1_fb]); zmask = np.r_[True, np.zeros(len(a1_rows) - 1, bool)]
    z0 = np.where(zmask)[0][0]
    ordv = np.r_[z0, [i for i in range(len(a1_rows)) if i != z0]]
    a1_rows, a1_parent, a1_mode, a1_supp, a1_fb = (a1_rows[ordv], a1_parent[ordv],
                                                   a1_mode[ordv], a1_supp[ordv], a1_fb[ordv])

    # ---- prune to <=2048 (저빈도 mode 병합: 각 parent 최소1 유지) ----
    Ktot0 = len(a1_rows)
    pruned = 0
    if Ktot0 > MAX_CAND:
        # parent별 최소1 유지, 나머지 중 support 낮은 것부터 제거
        keep = np.ones(len(a1_rows), bool)
        # 각 parent 의 대표(최대 support, 또는 zero row) 1개는 보호
        prot = np.zeros(len(a1_rows), bool)
        prot[0] = True  # stop zero row
        for k in np.unique(a1_parent):
            idxk = np.where(a1_parent == k)[0]
            prot[idxk[np.argmax(a1_supp[idxk])]] = True
        cand = np.where(~prot)[0]
        cand = cand[np.argsort(a1_supp[cand])]          # 낮은 support 먼저
        ndrop = Ktot0 - MAX_CAND
        keep[cand[:ndrop]] = False
        pruned = int((~keep).sum())
        a1_rows, a1_parent, a1_mode, a1_supp, a1_fb = (a1_rows[keep], a1_parent[keep],
                                                       a1_mode[keep], a1_supp[keep], a1_fb[keep])
    Ktot = len(a1_rows)

    # ================= 검증 =================
    R = {}
    R["source_sha"] = {"anchor": sha16(ANCH_P), "ego_cache_5s": sha16(E5_P),
                       "split": sha16(SPLIT_P), "champ_det": sha16(CHAMP_P)}
    R["n_train_used"] = int(tr.shape[0])
    # 1) first-6 bitwise
    R["A0_first6_bitwise_eq"] = bool(np.array_equal(A0[:, :6], An))
    R["A1_first6_bitwise_eq"] = bool(np.array_equal(a1_rows[:, :6], An[a1_parent]))
    # candidate 0 exact zero
    R["A0_cand0_exact_zero"] = bool(np.all(A0[0] == 0))
    R["A1_row0_exact_zero"] = bool(np.all(a1_rows[0] == 0))

    # 2/3) D3 oracle (val, frame>=30, keep) — Dgt 재사용으로 배포 일치
    ch = np.load(CHAMP_P, allow_pickle=True)
    Dgt = ch["Dgt"].astype(np.float64); wrow = ch["wrow"].astype(np.float64)
    fr = ch["frame"]; keep = ch["keep"]
    v = keep & (fr >= 30) & np.isfinite(Dgt[:, 0]); w = wrow[v]
    wm = lambda a: float(np.average(a, weights=w))
    legacy_oracle = wm(Dgt[v].min(1))                                   # 1024 anchors
    A0_oracle = wm(Dgt[v].min(1))                                       # A0 first6==anchors
    A1_oracle = wm(Dgt[v][:, a1_parent].min(1))                         # parent 중복
    R["D3oracle_legacyK1024"] = legacy_oracle
    R["D3oracle_A0"] = A0_oracle
    R["D3oracle_A1_expanded"] = A1_oracle
    R["D3oracle_A0_eq_legacy"] = bool(abs(A0_oracle - legacy_oracle) < 1e-12)
    R["D3oracle_A1_eq_legacy"] = bool(abs(A1_oracle - legacy_oracle) < 1e-12)

    # 4) 3->3.5s step jump (A1 candidates): prev step (wp4->5) vs next (wp5->6)
    def jumps(bank):
        prev = bank[:, 5] - bank[:, 4]
        nxt = bank[:, 6] - bank[:, 5]
        dmag = np.abs(np.linalg.norm(nxt, axis=1) - np.linalg.norm(prev, axis=1))
        # heading jump (deg), only where both steps nonzero
        pn = np.linalg.norm(prev, 1) if False else np.linalg.norm(prev, axis=1)
        nn = np.linalg.norm(nxt, axis=1)
        m = (pn > 1e-3) & (nn > 1e-3)
        cos = np.clip((prev[m] * nxt[m]).sum(1) / (pn[m] * nn[m]), -1, 1)
        hd = np.degrees(np.arccos(cos))
        return dmag, hd
    dmag, hd = jumps(a1_rows)
    R["stepjump_mag_p50_p90_p99"] = [float(np.percentile(dmag, q)) for q in (50, 90, 99)]
    R["stepjump_heading_deg_p50_p90_p99"] = [float(np.percentile(hd, q)) for q in (50, 90, 99)]
    dmag0, hd0 = jumps(A0)
    R["A0_stepjump_mag_p50_p90_p99"] = [float(np.percentile(dmag0, q)) for q in (50, 90, 99)]

    # 5) support count 분포
    R["A1_support_min_p50_max"] = [int(a1_supp.min()), int(np.median(a1_supp)), int(a1_supp.max())]
    R["A1_modecount_hist"] = {int(m): int(c) for m, c in
                              zip(*np.unique(np.bincount(a1_parent, minlength=K), return_counts=True))}
    # 6) fallback 비율
    R["fallback_anchor_frac"] = float((A0_fallback >= 0).mean())
    R["fallback_anchor_count"] = int((A0_fallback >= 0).sum())
    R["anchors_with_zero_support"] = int((counts == 0).sum())
    # 7) 총 후보/pruning
    R["A1_total_before_prune"] = int(Ktot0)
    R["A1_total_after_prune"] = int(Ktot)
    R["A1_pruned"] = int(pruned)
    # 8) stop parent modes
    p0m = a1_parent == 0
    cont = int(np.sum(p0m & (np.abs(a1_rows).reshape(len(a1_rows), -1).sum(1) == 0)))
    delayed = int(np.sum(p0m) - cont)
    R["stop_parent_continued_stop_modes"] = cont
    R["stop_parent_delayed_start_modes"] = delayed
    R["stop_parent_total_support"] = int(counts[0])

    # 9) A0 5s endpoint selector realized L2 (score3+nms9 + λ0.1)
    logits = ch["logits"].astype(np.float64)
    goal_val = fut5[sp["val_idx"]][:, 9]                # 5s endpoint GT (val order)
    Danch = d3_3s(An, An); np.fill_diagonal(Danch, 1e18)
    tau = float(np.percentile(Danch.min(1), 50))

    def shortlist(n):
        order = np.argsort(-logits[n], kind="stable")
        sel = list(order[:3])
        for c in order[:64]:
            if len(sel) >= 12: break
            if c in sel: continue
            if Danch[c, sel].min() > tau: sel.append(c)
        for c in order:
            if len(sel) >= 12: break
            if c not in sel: sel.append(c)
        return np.array(sel[:12])

    def realized(bank5_end):
        idxv = np.where(v)[0]; out = np.empty(len(idxv))
        for j, n in enumerate(idxv):
            S = shortlist(n)
            gc = np.linalg.norm(bank5_end[S] - goal_val[n], axis=1)
            vc = logits[n].max() - logits[n][S]
            gcn = (gc - gc.min()) / (gc.max() - gc.min() + 1e-9)
            vcn = (vc - vc.min()) / (vc.max() - vc.min() + 1e-9)
            out[j] = Dgt[n, S[int(np.argmin(gcn + 0.1 * vcn))]]
        return wm(out)
    # 기존 proxy endpoint = train-median goal per anchor
    goal_tr = fut5[tr][:, 9]
    proxy_end = np.full((K, 2), np.nan)
    for k in range(K):
        m = assign == k
        if m.any(): proxy_end[k] = np.median(goal_tr[m], 0)
    proxy_end[np.isnan(proxy_end[:, 0])] = np.nanmedian(proxy_end, 0)
    R["realized_L2_proxy_endpoint"] = realized(proxy_end)
    R["realized_L2_A0_endpoint"] = realized(A0[:, 9])

    # 10) A1 ideal endpoint & 5s oracle (weighted, val frame>=30)
    gt5 = fut5[sp["val_idx"]]                           # [2280,10,2]
    def endpoint_oracle(bank):
        e = bank[:, 9]
        idxv = np.where(v)[0]
        best = np.array([np.linalg.norm(e - goal_val[n], axis=1).min() for n in idxv])
        return wm(best)
    def traj5_oracle(bank):
        idxv = np.where(v)[0]
        best = np.empty(len(idxv))
        for j, n in enumerate(idxv):
            best[j] = np.linalg.norm(bank - gt5[n][None], axis=2).mean(1).min()
        return wm(best)
    R["endpoint5s_oracle_A0"] = endpoint_oracle(A0)
    R["endpoint5s_oracle_A1"] = endpoint_oracle(a1_rows)
    R["traj5s_meanL2_oracle_A0"] = traj5_oracle(A0)
    R["traj5s_meanL2_oracle_A1"] = traj5_oracle(a1_rows)

    # ---- save artifacts ----
    np.savez_compressed(OUT_A0, bank=A0.astype(np.float32), fallback_source=A0_fallback,
                        support_count=counts, cw=CW,
                        src_anchor_sha=R["source_sha"]["anchor"],
                        src_ego5_sha=R["source_sha"]["ego_cache_5s"],
                        src_split_sha=R["source_sha"]["split"])
    np.savez_compressed(OUT_A1, bank=a1_rows.astype(np.float32),
                        parent_anchor_id=a1_parent.astype(np.int32),
                        tail_mode_id=a1_mode.astype(np.int32),
                        support_count=a1_supp.astype(np.int32),
                        fallback_source=a1_fb.astype(np.int32), cw=CW,
                        src_anchor_sha=R["source_sha"]["anchor"],
                        src_ego5_sha=R["source_sha"]["ego_cache_5s"],
                        src_split_sha=R["source_sha"]["split"])
    with open(REPORT, "w") as f:
        json.dump(R, f, indent=2)

    print("===== ② multi-tail bank report =====")
    for k, val in R.items():
        print(f"{k}: {val}")
    print(f"\nsaved A0 {OUT_A0}  ({A0.shape})")
    print(f"saved A1 {OUT_A1}  ({a1_rows.shape})")
    # success gate summary
    gates = [R["A0_first6_bitwise_eq"], R["A1_first6_bitwise_eq"],
             R["D3oracle_A0_eq_legacy"], R["D3oracle_A1_eq_legacy"],
             R["A0_cand0_exact_zero"], R["A1_row0_exact_zero"],
             Ktot <= MAX_CAND]
    print("\nSUCCESS_GATES_PASS" if all(gates) else "SUCCESS_GATES_FAIL",
          "prefix/oracle/zero/cand<=2048:", gates)
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
