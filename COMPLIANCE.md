# ETRI-ScoreDrive Compliance Manifest

생성: 2026-09-04 11:58 KST  | 대상: champion pa2_softceexp/epoch_2 + A0 deploy bank

**판정: C1–C7 ALL PASS ✅** (C8 latency PENDING)

| # | 테스트 | 스크립트 | 결과 | 판정 |
|---|---|---|---|---|
| C1 | Signature audit | `gate3_c1_audit.py` | predict_candidates/build_candidates/배포 decoder 호출부 goal·cmd·status 없음 | ✅ |
| C2 | Goal/cmd counterfactual (100×) | `gate4_compliance_model.py` | logits goal 에 bitwise 불변=True (goal 실제주입=True) | ✅ |
| C3 | Selector row identity (full path) | `gate4_seal_model.py + offline` | VAD→API→selector→6point row-identity=True; offline bitwise=True, pure slice + tie-break unit OK | ✅ |
| C4 | Image ablation (full table) | `gate4_seal_model.py` | normal 기준 realized/top1-ID/top12-Jaccard/rank-corr 표 (아래) | ✅ |
| C5 | Goal-only negative control (full-K) | `gate4_compliance_offline.py` | normal 0.239 / full-K shuffle 4.11 / cross-scenario 8.13 / goal-only 0.681(비production) | ✅ |
| C6 | Pose-path audit (+can_bus) | `gate4_seal_model.py + offline` | decoder forward pose=[], deploy goal=[], use_can_bus=True==False(정렬만), decoder nargs=[3], img_metas ban keys=[] | ✅ |
| C7 | Reset/depth audit (7 forward) | `gate4_compliance_model.py + seal` | reset 격리=True, depth 사용=True, clip당 forward=[7] (정확히 7=True) | ✅ |
| C8 | Timing boundary / latency | `PENDING` | stream reset 후 전 forward 합산 측정 (다음 단계) | ⏳ |

## C4 image ablation 표 (normal 기준)

| ablation | realized L2 | top1 ID overlap | top12 Jaccard | rank corr |
|---|---|---|---|---|
| normal | 0.1747 | 1.0 | 1.0 | 1.0 |
| zero | 15.704 | 0.0 | 0.006 | nan |
| constant | 0.1592 | 0.867 | 0.901 | 0.998 |
| clip_shuffle | 2.8974 | 0.0 | 0.041 | 0.585 |
| current_shuffle | 0.1696 | 0.933 | 0.935 | 0.999 |
| history_shuffle | 0.1704 | 0.867 | 0.928 | 0.999 |
| camera_perm | 0.1691 | 0.8 | 0.828 | 0.996 |

predict_candidates smoke: ran=True, keys=['candidate_ids', 'candidate_xy_abs_5s', 'candidate_xy_inc_5s', 'visual_logits'], shape=[1, 12, 10, 2]

## Artifact / Code SHA (train-only provenance)

- `deploy_bank` = `154da8bcfd05904e`
- `champ_dump` = `5c7f176dd94bb8fe`
- `decoder_py` = `6c39c858e405036a`
- `vad_py` = `09280b7f2751aa02`
- `scoredrive_api` = `3d5ea96c31330f5b`

## 핵심 수치

- Selector realized L2: champion+λ0.1 **0.2392** / PC-res+λ0.25 **0.2335** / endpoint λ0 0.2476
- C5 negative control(full-K): shuffle **4.11**, cross-scenario **8.13**, visual-zero **9.58**, full-K goal-only **0.681** (비production)
- A1 multi-tail(1354)는 모델 반환 경로 미연결 — artifact only

## 현재 상태의 정확한 표현
- **제출 구조의 candidate/selector 경계 완성** (제출 구조 전체 완성 아님)
- 규정 구조·완성 후보 API 기준 dev38: **0.2335** (PC-res λ0.25) / 0.2392 (champion λ0.1)
- candidate API/selector 계산오차: **0**
- 아직 unbiased final-val 아님 (dev38 반복 사용)
- **C8 latency penalty 미반영** — 다음 단계에서 3090 실측

> 정확도 추가 학습은 중단. latency 결과에 따라 sparse trunk(Phase D) 전환.