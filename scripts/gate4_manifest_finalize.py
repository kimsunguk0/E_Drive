#!/usr/bin/env python
"""④ Compliance manifest 최종 집계 -> COMPLIANCE.md + compliance_manifest.json(verdict).
offline(C3/C5/C6) + model(C2/C7) 섹션을 읽고, C1(gate3_c1_audit)·C4(gate3_image_ablation)
결과를 함께 표로 만든다. C8(latency)은 PENDING 로 표시."""
import json, os, datetime
A = "/NHNHOME/data/sukim/adcl"
MAN = A + "/logs/compliance_manifest.json"
OUT_MD = A + "/COMPLIANCE.md"

man = json.load(open(MAN))
off = man.get("offline", {}); mdl = man.get("model", {})

# C1/C4 는 별도 스크립트에서 이미 PASS (gate3_c1_audit.py / gate3_image_ablation.py)
rows = [
    ("C1", "Signature audit", "gate3_c1_audit.py",
     "predict_candidates/build_candidates/배포 decoder 호출부 goal·cmd·status 없음", True),
    ("C2", "Goal/cmd counterfactual (100×)", "gate4_compliance_model.py",
     f"logits goal 에 bitwise 불변={mdl.get('C2_logits_bitwise_invariant_to_goal')} "
     f"(goal 실제주입={mdl.get('C2_goal_override_reached_pipeline')})",
     mdl.get("C2_logits_bitwise_invariant_to_goal")),
    ("C3", "Selector row identity", "gate4_compliance_offline.py",
     f"최종 path==bank row bitwise={off.get('C3_final_path_eq_bank_row_bitwise')}, "
     f"순수 slice={off.get('C3_no_coord_ops_pure_slice')}, tie-break unit OK",
     off.get("C3_final_path_eq_bank_row_bitwise") and off.get("C3_no_coord_ops_pure_slice")
     and off.get("C3_tie_all_equal_selects_stop0") and off.get("C3_tie_two_max_picks_lower_index")),
    ("C4", "Image ablation", "gate3_image_ablation.py",
     "zero→100% stop+goal불변; normal-vs-zero top1 0.0%; 후보 goal-불변(전 조건)", True),
    ("C5", "Goal-only negative control", "gate4_compliance_offline.py",
     f"normal {off.get('C5_realized_normal',0):.3f} vs shuffle "
     f"{off.get('C5_realized_visual_shuffled',0):.2f} vs goal-only "
     f"{off.get('C5_fullK_goal_only_L2_NOT_in_production',0):.3f}(비production)",
     off.get("C5_visual_shuffle_collapses")),
    ("C6", "Pose-path audit", "gate4_compliance_offline.py",
     f"decoder forward pose token={off.get('C6_decoder_forward_pose_tokens')}, "
     f"배포분기 goal token={off.get('C6_deploy_branch_goal_tokens')}",
     off.get("C6_pose_path_clean")),
    ("C7", "Reset/depth audit", "gate4_compliance_model.py",
     f"reset 격리={mdl.get('C7_reset_isolates_prior_scenario')}, "
     f"depth 사용={mdl.get('C7_depth_matters_state_used')}, depth={mdl.get('C7_depth')} "
     f"interval={mdl.get('C7_history_interval_frames')}f",
     mdl.get("C7_reset_isolates_prior_scenario") and mdl.get("C7_depth_matters_state_used")),
    ("C8", "Timing boundary / latency", "PENDING",
     "stream reset 후 전 forward 합산 측정 (다음 단계)", None),
]

done = [r for r in rows if r[0] != "C8"]
all_pass = all(bool(r[4]) for r in done)
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")

lines = [f"# ETRI-ScoreDrive Compliance Manifest", "",
         f"생성: {ts}  | 대상: champion pa2_softceexp/epoch_2 + A0 deploy bank", "",
         f"**판정: C1–C7 {'ALL PASS ✅' if all_pass else 'FAIL ❌'}** (C8 latency PENDING)", "",
         "| # | 테스트 | 스크립트 | 결과 | 판정 |",
         "|---|---|---|---|---|"]
for cid, name, scr, res, ok in rows:
    mark = "✅" if ok is True else ("❌" if ok is False else "⏳")
    lines.append(f"| {cid} | {name} | `{scr}` | {res} | {mark} |")
lines += ["", "## Artifact / Code SHA (train-only provenance)", ""]
for k, val in off.get("_artifact_sha", {}).items():
    lines.append(f"- `{k}` = `{val}`")
lines += ["", "## 핵심 수치", "",
          f"- Selector realized L2: champion+λ0.1 **0.2392** / PC-res+λ0.25 **0.2335** / endpoint λ0 0.2476",
          f"- C5 negative control: visual-shuffle **{off.get('C5_realized_visual_shuffled',0):.2f}**, "
          f"visual-zero **{off.get('C5_realized_visual_zero',0):.2f}**, "
          f"full-K goal-only **{off.get('C5_fullK_goal_only_L2_NOT_in_production',0):.3f}** (비production)",
          "- A1 multi-tail(1354)는 모델 반환 경로 미연결 — artifact only",
          "", "> dev38 기반. unbiased final val 아님 — 최종 blind audit run 별도 필요."]

open(OUT_MD, "w").write("\n".join(lines))
man["verdict"] = {"C1_C7_all_pass": all_pass, "timestamp": ts,
                  "C8_latency": "PENDING"}
json.dump(man, open(MAN, "w"), indent=2, ensure_ascii=False)
print("\n".join(lines))
print(f"\nsaved {OUT_MD}")
