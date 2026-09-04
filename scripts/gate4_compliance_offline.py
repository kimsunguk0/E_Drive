#!/usr/bin/env python
"""④ Compliance suite — offline/static 파트 (§12).
  C3  selector row identity (최종 path == 후보 한 row bitwise, 좌표연산 없음) + tie-break unit test
  C5  goal-only negative control (visual 파괴 시 붕괴 + full-K goal-only L2 보고)
  C6  pose-path audit (정적 grep: ego pose/his/lcf/status 가 scorer graph 에 없음)
결과를 compliance_manifest.json(부분)에 기록.
"""
import os, sys, re, json, hashlib
import numpy as np
sys.path.insert(0, "/NHNHOME/data/sukim/adcl/scripts")
import scoredrive_api as api

A = "/NHNHOME/data/sukim/adcl"
VADP = A + "/dense_vocab_v1/projects/mmdet3d_plugin/VAD"
DEP = A + "/data/etri/bank_A0_deploy.npz"
CHAMP = A + "/logs/dump/champ_det.npz"
E5 = "/tmp/pm97/data/etri/ego_cache_5s.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"
MAN = A + "/logs/compliance_manifest.json"


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def main():
    dep = np.load(DEP, allow_pickle=True)
    abs5 = dep["candidate_xy_abs_5s"]; inc5 = dep["candidate_xy_inc_5s"]
    ids = dep["candidate_ids"]; adist = dep["anchor_dist"]; tau = float(dep["nms_tau"])
    z = np.load(CHAMP, allow_pickle=True)
    logC = z["logits"].astype(np.float64); Dgt = z["Dgt"].astype(np.float64)
    wrow = z["wrow"].astype(np.float64); frame = z["frame"]; keep = z["keep"]
    v = keep & (frame >= 30) & np.isfinite(Dgt[:, 0]); idxv = np.where(v)[0]; w = wrow[idxv]
    wm = lambda a: float(np.average(a, weights=w))
    e5 = np.load(E5, allow_pickle=True); sp = np.load(SPLIT, allow_pickle=True)
    goal_val = e5["fut5"][sp["val_idx"]][:, 9].astype(np.float64)

    R = {}

    # ---------------- C3 selector row identity ----------------
    cand = api.build_candidates_np(logC, abs5, inc5, ids, adist, tau)
    row_ok = True; slice_ok = True
    for n in idxv:
        cb = api.slice_batch(cand, int(n))
        p6, j, cid = api.select_path_np(cb, goal_val[n], 0.1)
        # 최종 제출 path(inc 첫6) == 배포 bank 의 그 id row inc 첫6 bitwise
        if not np.array_equal(p6, inc5[cid, :6]):
            row_ok = False
        # 좌표연산 없음: p6 는 정확히 후보 한 row 의 slice (평균/보간 아님)
        if not np.array_equal(p6, cb["candidate_xy_inc_5s"][j, :6]):
            slice_ok = False
    R["C3_final_path_eq_bank_row_bitwise"] = bool(row_ok)
    R["C3_no_coord_ops_pure_slice"] = bool(slice_ok)

    # tie-break unit test
    K = abs5.shape[0]
    # (a) 전부 동률 -> stable order [0,1,2..], fallback -> id0 stop
    lz = np.zeros((1, K))
    cz = api.build_candidates_np(lz, abs5, inc5, ids, adist, tau)
    cbz = api.slice_batch(cz, 0)
    _, jz, cidz = api.select_path_np(cbz, np.array([9.0, 4.0]), 0.1)
    R["C3_tie_all_equal_selects_stop0"] = bool(cidz == 0 and np.all(inc5[cidz] == 0))
    # (b) 두 index 동률(최고) -> 더 낮은 index 가 top-1
    lt = np.full((1, K), -5.0); lt[0, 7] = 3.0; lt[0, 3] = 3.0    # 3,7 동률 최고
    ct = api.build_candidates_np(lt, abs5, inc5, ids, adist, tau)
    R["C3_tie_two_max_picks_lower_index"] = bool(ct["candidate_ids"][0, 0] == 3)

    # ---------------- C5 goal-only negative control ----------------
    def realized_from_logits(L, lam=0.1):
        cm = api.build_candidates_np(L, abs5, inc5, ids, adist, tau)
        out = np.empty(len(idxv))
        for j2, n in enumerate(idxv):
            cb = api.slice_batch(cm, int(n))
            k = api.select_index_np(cb, goal_val[n], lam)
            out[j2] = Dgt[n, cb["candidate_ids"][k]]
        return wm(out)

    R["C5_realized_normal"] = realized_from_logits(logC)
    # (a) full-K within-sample shuffle: 각 sample 의 1024 logits 를 순열 -> ranking 이전에
    #     영상 순위 자체를 파괴(shortlist 무너짐). N=12 안에서가 아니라 full-K 대상.
    rng = np.random.RandomState(0)
    logSh = np.empty_like(logC)
    for i in range(logC.shape[0]):
        logSh[i] = logC[i][rng.permutation(K)]
    R["C5_realized_fullK_shuffled"] = realized_from_logits(logSh)
    # (b) scenario 간 shuffle: 다른 sample 의 full-K logits 를 사용(영상-프레임 불일치)
    perm = rng.permutation(logC.shape[0])
    R["C5_realized_cross_scenario"] = realized_from_logits(logC[perm])
    R["C5_realized_visual_shuffled"] = R["C5_realized_fullK_shuffled"]  # 호환 alias
    # visual zero: 전부 동률 -> fallback stop
    R["C5_realized_visual_zero"] = realized_from_logits(np.zeros_like(logC))
    # full-K goal-only (production 경로 아님): 1024 전체에서 goal-endpoint 최근접
    endp = abs5[:, 9].astype(np.float64)
    go = np.empty(len(idxv))
    for j2, n in enumerate(idxv):
        k = int(np.argmin(np.linalg.norm(endp - goal_val[n], axis=1)))
        go[j2] = Dgt[n, k]
    R["C5_fullK_goal_only_L2_NOT_in_production"] = wm(go)
    R["C5_visual_shuffle_collapses"] = bool(
        R["C5_realized_visual_shuffled"] > R["C5_realized_normal"] + 0.02)

    # ---------------- C6 pose-path audit (static grep) ----------------
    dec = open(f"{VADP}/anchor_vocab_decoder.py").read()
    head = open(f"{VADP}/VAD_head.py").read()
    # decoder forward 안에서 ego pose/status 계열이 tensor 로 흐르는지
    forward_body = dec[dec.find("def forward(self, bev_key"):dec.find("def attach_deploy_bank")]
    banned_tokens = ["ego_his", "ego_lcf", "his_traj", "lcf_feat", "target_point",
                     "yaw_rate", "ego_fut_goal", "ego_fut_cmd"]
    hits_dec = [t for t in banned_tokens if t in forward_body]
    # 배포 분기(uses_condition=False)에서 goal/cmd 사용 여부
    br = head[head.find("if not getattr(self.goal_decoder, 'uses_condition'"):]
    br = br[:br.find("else:")]
    hits_head = [t for t in ["ego_fut_goal", "ego_fut_cmd", " g,", " c)", "target_point"]
                 if t in br]
    R["C6_decoder_forward_pose_tokens"] = hits_dec
    R["C6_deploy_branch_goal_tokens"] = hits_head
    R["C6_pose_path_clean"] = bool(len(hits_dec) == 0 and len(hits_head) == 0)
    # 주: residual 의 vel_kin 은 anchor(후보) kinematics 이지 ego status 아님 -> 허용
    R["C6_note"] = "residual vel_kin = candidate(anchor) kinematics, not ego status (allowed)"

    # ---------------- manifest merge ----------------
    R["_artifact_sha"] = {"deploy_bank": sha16(DEP), "champ_dump": sha16(CHAMP),
                          "decoder_py": sha16(f"{VADP}/anchor_vocab_decoder.py"),
                          "vad_py": sha16(f"{VADP}/VAD.py"),
                          "scoredrive_api": sha16(A + "/scripts/scoredrive_api.py")}
    man = {}
    if os.path.exists(MAN):
        man = json.load(open(MAN))
    man["offline"] = R
    json.dump(man, open(MAN, "w"), indent=2, ensure_ascii=False)

    print(json.dumps(R, indent=2, ensure_ascii=False))
    gates = [R["C3_final_path_eq_bank_row_bitwise"], R["C3_no_coord_ops_pure_slice"],
             R["C3_tie_all_equal_selects_stop0"], R["C3_tie_two_max_picks_lower_index"],
             R["C5_visual_shuffle_collapses"], R["C6_pose_path_clean"]]
    print("\nOFFLINE_COMPLIANCE_PASS" if all(gates) else "OFFLINE_COMPLIANCE_FAIL", gates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
