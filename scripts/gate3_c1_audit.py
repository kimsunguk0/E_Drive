#!/usr/bin/env python
"""Gate-③ C1 signature audit + 실제 in-model decoder 메서드 검증 (CPU).

1) 정적 audit: predict_candidates / build_candidates / 배포 decoder 호출부에
   goal/command/status/ego_his/ego_lcf 가 없는지.
2) 실제 decoder 인스턴스로 attach_deploy_bank 불변식(candidate_inc_5s[:,:6]==anchors_inc,
   모델 자신의 buffer 기준) + build_candidates(챔피언 덤프 logits) == offline shortlist.
"""
import os, sys, re, inspect
import numpy as np

A = "/NHNHOME/data/sukim/adcl"
REPO = A + "/dense_vocab_v1"
sys.path.insert(0, REPO); sys.path.insert(0, A + "/scripts")
os.chdir(REPO)
VADP = REPO + "/projects/mmdet3d_plugin/VAD"

BAN = ["goal", "command", "cmd", "status", "ego_his", "ego_lcf", "target_point"]


def static_audit():
    dec = open(f"{VADP}/anchor_vocab_decoder.py").read()
    vad = open(f"{VADP}/VAD.py").read()
    out = {}
    # predict_candidates signature
    m = re.search(r"def predict_candidates\(self,(.*?)\):", vad, re.S)
    sig = m.group(1) if m else ""
    out["predict_candidates_sig"] = " ".join(sig.split())
    out["predict_candidates_no_banned"] = not any(re.search(rf"\b{b}\w*\s*=", sig) or
                                                  re.search(rf"\b{b}\b", sig) for b in BAN)
    # build_candidates signature
    m2 = re.search(r"def build_candidates\(self,(.*?)\):", dec, re.S)
    out["build_candidates_sig"] = " ".join((m2.group(1) if m2 else "").split())
    out["build_candidates_only_logits"] = (out["build_candidates_sig"].strip() == "logits")
    # 배포 decoder 호출부 (uses_condition=False 분기)
    head = open(f"{VADP}/VAD_head.py").read()
    m3 = re.search(r"if not getattr\(self\.goal_decoder, 'uses_condition'.*?decoded = self\.goal_decoder\((.*?)\)",
                   head, re.S)
    call = " ".join((m3.group(1) if m3 else "").split())
    out["deploy_decoder_call_args"] = call
    out["deploy_call_no_goal"] = not any(b in call for b in ["goal", "cmd", "command", "g,", "c)"])
    return out


def model_method_test():
    import torch
    import importlib
    importlib.import_module("projects.mmdet3d_plugin")  # register plugin
    from projects.mmdet3d_plugin.VAD.anchor_vocab_decoder import AnchorGroundedVocabularyDecoder
    dec = AnchorGroundedVocabularyDecoder(
        anchors_path=A + "/h200_latest/artifacts/dense_vocab/etri_train330_vocab_k1024.npy",
        pc_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0], bev_h=100, bev_w=200,
        use_global=True, n_head=8, loss_weight=1.0, target_temperature=0.1, top_m=6).eval()
    dep = A + "/data/etri/bank_A0_deploy.npz"
    dec.attach_deploy_bank(dep)            # 불변식 assert 내부 수행 (실패 시 예외)
    out = {"attach_ok": True}
    out["inc_first6_eq_model_anchors_inc"] = bool(
        torch.equal(dec.candidate_inc_5s[:, :6], dec.anchors_inc))
    out["abs_first6_eq_model_anchors_abs"] = bool(
        torch.equal(dec.candidate_abs_5s[:, :6], dec.anchors_abs))

    # 챔피언 덤프 logits 로 build_candidates
    z = np.load(A + "/logs/dump/champ_det.npz", allow_pickle=True)
    logits = torch.from_numpy(z["logits"].astype(np.float32))
    with torch.no_grad():
        cand = dec.build_candidates(logits)
    out["ret_keys"] = sorted(cand.keys())
    out["no_fullK"] = all(cand[k].shape[1] == 12 for k in cand)
    out["shape_abs"] = list(cand["candidate_xy_abs_5s"].shape)
    # offline shortlist 비교
    import scoredrive_api as api
    dd = np.load(dep, allow_pickle=True)
    adist = dd["anchor_dist"]; tau = float(dd["nms_tau"])
    off = np.stack([api.shortlist_np(z["logits"].astype(np.float64)[b], adist, tau)
                    for b in range(logits.shape[0])])
    ids_model = cand["candidate_ids"].numpy()
    out["ids_eq_offline_frac"] = float(np.mean([np.array_equal(ids_model[b], off[b])
                                                for b in range(len(off))]))
    # prefix bitwise on returned candidates vs model anchors
    sel = ids_model
    ai = dec.anchors_inc.numpy()
    out["ret_inc_first6_bitwise"] = bool(np.array_equal(
        cand["candidate_xy_inc_5s"].numpy()[:, :, :6], ai[sel]))
    return out


def main():
    R = {"static": static_audit()}
    try:
        R["model"] = model_method_test()
    except Exception as e:
        import traceback
        R["model"] = {"ERROR": str(e), "tb": traceback.format_exc()[-800:]}
    import json
    print(json.dumps(R, indent=2, ensure_ascii=False))
    s, m = R["static"], R.get("model", {})
    gates = [s["predict_candidates_no_banned"], s["build_candidates_only_logits"],
             s["deploy_call_no_goal"], m.get("attach_ok", False),
             m.get("inc_first6_eq_model_anchors_inc", False),
             m.get("abs_first6_eq_model_anchors_abs", False),
             m.get("no_fullK", False), m.get("ids_eq_offline_frac", 0) == 1.0,
             m.get("ret_inc_first6_bitwise", False)]
    print("\nALL_C1_PASS" if all(gates) else "C1_FAIL", gates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
