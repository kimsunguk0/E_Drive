# ETRI-ScoreDrive Compliance Manifest

생성: 2026-09-04 11:38 KST  | 대상: champion pa2_softceexp/epoch_2 + A0 deploy bank

**판정: C1–C7 ALL PASS ✅** (C8 latency PENDING)

| # | 테스트 | 스크립트 | 결과 | 판정 |
|---|---|---|---|---|
| C1 | Signature audit | `gate3_c1_audit.py` | predict_candidates/build_candidates/배포 decoder 호출부 goal·cmd·status 없음 | ✅ |
| C2 | Goal/cmd counterfactual (100×) | `gate4_compliance_model.py` | logits goal 에 bitwise 불변=True (goal 실제주입=True) | ✅ |
| C3 | Selector row identity | `gate4_compliance_offline.py` | 최종 path==bank row bitwise=True, 순수 slice=True, tie-break unit OK | ✅ |
| C4 | Image ablation | `gate3_image_ablation.py` | zero→100% stop+goal불변; normal-vs-zero top1 0.0%; 후보 goal-불변(전 조건) | ✅ |
| C5 | Goal-only negative control | `gate4_compliance_offline.py` | normal 0.239 vs shuffle 4.11 vs goal-only 0.681(비production) | ✅ |
| C6 | Pose-path audit | `gate4_compliance_offline.py` | decoder forward pose token=[], 배포분기 goal token=[] | ✅ |
| C7 | Reset/depth audit | `gate4_compliance_model.py` | reset 격리=True, depth 사용=True, depth=6 interval=5f | ✅ |
| C8 | Timing boundary / latency | `PENDING` | stream reset 후 전 forward 합산 측정 (다음 단계) | ⏳ |

## Artifact / Code SHA (train-only provenance)

- `deploy_bank` = `154da8bcfd05904e`
- `champ_dump` = `5c7f176dd94bb8fe`
- `decoder_py` = `9bed92a8296a82ca`
- `vad_py` = `3b01382e8bf54b0f`
- `scoredrive_api` = `3d5ea96c31330f5b`

## 핵심 수치

- Selector realized L2: champion+λ0.1 **0.2392** / PC-res+λ0.25 **0.2335** / endpoint λ0 0.2476
- C5 negative control: visual-shuffle **4.11**, visual-zero **9.58**, full-K goal-only **0.681** (비production)
- A1 multi-tail(1354)는 모델 반환 경로 미연결 — artifact only

> dev38 기반. unbiased final val 아님 — 최종 blind audit run 별도 필요.