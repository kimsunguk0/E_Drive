#!/usr/bin/env python
"""③ A0 배포 artifact(abs/inc) 생성 + Gate-③ 전체 검증 (오프라인, 덤프 logits).

산출 bank_A0_deploy.npz:
    candidate_xy_abs_5s [1024,10,2]  (selector endpoint)
    candidate_xy_inc_5s [1024,10,2]  (제출 출력)
    candidate_ids       [1024]
    anchor_dist         [1024,1024]  (D3 3s, diag +inf) train-only
    nms_tau             scalar       train-only
    manifest(SHA)
검증: 반환 shape/키, prefix bitwise(abs&inc), in-model IDs==offline, torch==numpy,
counterfactual, λ=inf, zero-feature→cand0, realized L2 3-way(<1e-6 재현), latency.
"""
import sys, json, hashlib, time
import numpy as np
sys.path.insert(0, "/NHNHOME/data/sukim/adcl/scripts")
import scoredrive_api as api

A = "/NHNHOME/data/sukim/adcl"
ANCH_P = A + "/h200_latest/artifacts/dense_vocab/etri_train330_vocab_k1024.npy"
A0_P = A + "/data/etri/bank_A0_onetail.npz"
E5_P = "/tmp/pm97/data/etri/ego_cache_5s.npz"
SPLIT_P = "/tmp/pm97/data/etri/val_clips.npz"
DUMP = A + "/logs/dump"
OUT = A + "/data/etri/bank_A0_deploy.npz"
REPORT = A + "/logs/gate3_report.json"
CW = np.array([11, 11, 5, 5, 2, 2], np.float64) / 36.0


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def d3(An, G):
    M, K = G.shape[0], An.shape[0]
    out = np.empty((M, K), np.float64)
    for s in range(0, K, 256):
        a = An[s:s + 256]
        out[:, s:s + 256] = (np.linalg.norm(G[:, None] - a[None], axis=-1) * CW).sum(-1)
    return out


# ---- offline score3+nms9 (eval_pc_selector 원본과 동일, 독립 재구현) ----
def offline_shortlist(logit_row, Danch, tau):
    order = np.argsort(-logit_row, kind="stable")
    sel = list(order[:3])
    for c in order[:64]:
        if len(sel) >= 12: break
        if c in sel: continue
        if Danch[c, sel].min() > tau: sel.append(c)
    for c in order:
        if len(sel) >= 12: break
        if c not in sel: sel.append(c)
    return np.array(sel[:12])


def main():
    R = {}
    An = np.load(ANCH_P).astype(np.float32)                       # [1024,6,2]
    anchors_abs = An.copy()
    anchors_inc = np.diff(np.concatenate([np.zeros_like(An[:, :1]), An], 1), axis=1)  # 모델과 동일
    A0 = np.load(A0_P, allow_pickle=True)["bank"].astype(np.float32)             # [1024,10,2]
    K = A0.shape[0]

    # ---- abs/inc 이중 buffer ----
    abs5 = A0.copy()                                              # abs, first6==anchors_abs (복사됨)
    inc5 = np.zeros_like(abs5)
    inc5[:, :6] = anchors_inc                                     # prefix 복사 (runtime diff 금지)
    inc5[:, 6] = abs5[:, 6] - abs5[:, 5]
    inc5[:, 7] = abs5[:, 7] - abs5[:, 6]
    inc5[:, 8] = abs5[:, 8] - abs5[:, 7]
    inc5[:, 9] = abs5[:, 9] - abs5[:, 8]
    ids = np.arange(K, dtype=np.int64)

    Danch = d3(An.astype(np.float64), An.astype(np.float64)); np.fill_diagonal(Danch, np.inf)
    nms_tau = float(np.percentile(Danch.min(1), 50))
    anchor_dist = Danch.astype(np.float32); np.fill_diagonal(anchor_dist, np.inf)

    # invariant checks
    R["abs_first6_bitwise"] = bool(np.array_equal(abs5[:, :6], anchors_abs))
    R["inc_first6_bitwise"] = bool(np.array_equal(inc5[:, :6], anchors_inc))
    R["cumsum_inc_vs_abs_maxerr"] = float(np.abs(np.cumsum(inc5, 1) - abs5).max())
    R["cand0_abs_zero"] = bool(np.all(abs5[0] == 0))
    R["cand0_inc_zero"] = bool(np.all(inc5[0] == 0))
    R["nms_tau"] = nms_tau
    R["manifest_sha"] = {"anchor": sha16(ANCH_P), "A0": sha16(A0_P),
                         "anchor_dist": arr_sha(anchor_dist), "abs5": arr_sha(abs5),
                         "inc5": arr_sha(inc5)}

    np.savez_compressed(OUT, candidate_xy_abs_5s=abs5, candidate_xy_inc_5s=inc5,
                        candidate_ids=ids, anchor_dist=anchor_dist, nms_tau=nms_tau,
                        anchors_abs=anchors_abs, anchors_inc=anchors_inc,
                        manifest=json.dumps(R["manifest_sha"]))

    # ---- validation dumps ----
    e5 = np.load(E5_P, allow_pickle=True); sp = np.load(SPLIT_P, allow_pickle=True)
    goal_val = e5["fut5"][sp["val_idx"]][:, 9].astype(np.float64)  # 5s endpoint GT (val order)

    def load(name):
        z = np.load(f"{DUMP}/{name}.npz", allow_pickle=True)
        return (z["logits"].astype(np.float64), z["Dgt"].astype(np.float64),
                z["wrow"].astype(np.float64), z["frame"], z["keep"])

    logC, Dgt, wrow, frame, keep = load("champ_det")
    v = keep & (frame >= 30) & np.isfinite(Dgt[:, 0]); idxv = np.where(v)[0]; w = wrow[idxv]
    wm = lambda a: float(np.average(a, weights=w))

    # build candidates (numpy in-model path)
    cand = api.build_candidates_np(logC, abs5, inc5, ids, anchor_dist, nms_tau)
    R["return_shape_abs"] = list(cand["candidate_xy_abs_5s"].shape)
    R["return_shape_inc"] = list(cand["candidate_xy_inc_5s"].shape)
    R["return_shape_vlog"] = list(cand["visual_logits"].shape)
    R["return_keys"] = sorted([k for k in cand if not k.startswith("_")])
    R["no_fullK_in_return"] = bool(cand["visual_logits"].shape[1] == 12
                                   and all(cand[k].shape[1] == 12 for k in
                                           ["candidate_ids", "candidate_xy_abs_5s",
                                            "candidate_xy_inc_5s"]))
    # prefix bitwise on returned candidates
    sl = cand["_shortlist_idx"]
    R["ret_abs_first6_eq_anchors"] = bool(np.array_equal(
        cand["candidate_xy_abs_5s"][:, :, :6], anchors_abs[sl]))
    R["ret_inc_first6_eq_anchors"] = bool(np.array_equal(
        cand["candidate_xy_inc_5s"][:, :, :6], anchors_inc[sl]))

    # in-model IDs == offline score3+nms9 (독립 재구현), 전 sample
    off = np.stack([offline_shortlist(logC[b], Danch, nms_tau) for b in range(logC.shape[0])])
    R["ids_match_offline_frac"] = float(np.mean([np.array_equal(sl[b], off[b])
                                                 for b in range(len(sl))]))
    # torch build == numpy build
    try:
        import torch
        slt = api.shortlist_torch(torch.from_numpy(logC),
                                  torch.from_numpy(anchor_dist.astype(np.float64)), nms_tau).numpy()
        R["torch_eq_numpy_frac"] = float(np.mean([np.array_equal(slt[b], sl[b])
                                                  for b in range(len(sl))]))
    except Exception as e:
        R["torch_eq_numpy_frac"] = f"skip:{e}"

    # counterfactual: 100 random goals -> candidates/logits/IDs 불변, 선택 path는 항상 후보 row
    rng = np.random.RandomState(0)
    n0 = int(idxv[0])
    cb = api.slice_batch(cand, n0)
    sel_idx_set, path_ok = set(), True
    base_ids = cb["candidate_ids"].copy(); base_vlog = cb["visual_logits"].copy()
    for _ in range(100):
        g = rng.randn(2) * 8
        p6, j, cid = api.select_path_np(cb, g, 0.1)
        sel_idx_set.add(j)
        if not np.array_equal(p6, cb["candidate_xy_inc_5s"][j, :6]):
            path_ok = False
    R["cf_candidates_invariant"] = bool(np.array_equal(base_ids, cb["candidate_ids"])
                                        and np.array_equal(base_vlog, cb["visual_logits"]))
    R["cf_selected_path_is_row"] = bool(path_ok)
    R["cf_selected_index_varies"] = int(len(sel_idx_set))   # >1 이면 goal이 index만 바꿈

    # λ=inf == visual top-1 (shortlist[0])
    inf_ok = True
    for n in idxv[:500]:
        cbn = api.slice_batch(cand, int(n))
        j = api.select_index_np(cbn, goal_val[n], np.inf)
        if j != 0 or cbn["candidate_ids"][0] != int(np.argmax(logC[n])):
            inf_ok = False; break
    R["lambda_inf_eq_visual_top1"] = bool(inf_ok)

    # zero-feature -> candidate 0 (stop)
    logZ = np.zeros((1, K))
    candZ = api.build_candidates_np(logZ, abs5, inc5, ids, anchor_dist, nms_tau)
    cbz = api.slice_batch(candZ, 0)
    jz = api.select_index_np(cbz, np.array([5.0, 3.0]), 0.1)  # 임의 goal
    R["zerofeat_selected_id"] = int(cbz["candidate_ids"][jz])
    R["zerofeat_path_is_stop"] = bool(np.all(cbz["candidate_xy_inc_5s"][jz] == 0))

    # ---- realized L2 3-way (선택기) ----
    def realized(logits_model, lam):
        candm = api.build_candidates_np(logits_model, abs5, inc5, ids, anchor_dist, nms_tau)
        out = np.empty(len(idxv))
        for j2, n in enumerate(idxv):
            cbn = api.slice_batch(candm, int(n))
            k = api.select_index_np(cbn, goal_val[n], lam)
            out[j2] = Dgt[n, cbn["candidate_ids"][k]]
        return wm(out)

    R["realized_champion_lam0.1"] = realized(logC, 0.1)
    R["realized_endpoint_lam0"] = realized(logC, 0.0)
    logP, DgtP, wrowP, frameP, keepP = load("pc_res_ts_ep2")
    assert np.array_equal(frameP, frame) and np.array_equal(keepP, keep), "val order mismatch"
    R["realized_pcres_lam0.25"] = realized(logP, 0.25)
    R["offline_A0_champ_lam0.1_ref"] = 0.23923057745390167   # ②에서 계산
    R["realized_reprodiff_vs_offline"] = abs(R["realized_champion_lam0.1"]
                                             - R["offline_A0_champ_lam0.1_ref"])

    # ---- latency (shortlist+selector, per sample, CPU) ----
    t0 = time.time()
    for n in idxv[:500]:
        cbn = api.slice_batch(cand, int(n))
        api.select_index_np(cbn, goal_val[n], 0.1)
    R["selector_ms_per_sample_cpu"] = (time.time() - t0) / 500 * 1e3
    t0 = time.time()
    api.build_candidates_np(logC[:500], abs5, inc5, ids, anchor_dist, nms_tau)
    R["shortlist_ms_per_sample_cpu"] = (time.time() - t0) / 500 * 1e3

    with open(REPORT, "w") as f:
        json.dump(R, f, indent=2)
    print("===== Gate-③ report =====")
    for k, val in R.items():
        print(f"{k}: {val}")

    gates = {
        "abs_first6_bitwise": R["abs_first6_bitwise"],
        "inc_first6_bitwise": R["inc_first6_bitwise"],
        "return_shape[B,12,10,2]": R["return_shape_abs"][1:] == [12, 10, 2],
        "no_fullK_in_return": R["no_fullK_in_return"],
        "ret_abs_first6_eq": R["ret_abs_first6_eq_anchors"],
        "ret_inc_first6_eq": R["ret_inc_first6_eq_anchors"],
        "ids_match_offline=1.0": R["ids_match_offline_frac"] == 1.0,
        "torch_eq_numpy=1.0": R["torch_eq_numpy_frac"] == 1.0,
        "cf_candidates_invariant": R["cf_candidates_invariant"],
        "cf_path_is_row": R["cf_selected_path_is_row"],
        "lambda_inf_eq_top1": R["lambda_inf_eq_visual_top1"],
        "zerofeat_id==0": R["zerofeat_selected_id"] == 0,
        "zerofeat_is_stop": R["zerofeat_path_is_stop"],
        "realized_reprodiff<1e-6": R["realized_reprodiff_vs_offline"] < 1e-6,
    }
    print("\nGATE SUMMARY:", gates)
    print("ALL_GATE3_PASS" if all(gates.values()) else "GATE3_FAIL")
    print(f"saved {OUT}\nsaved {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
