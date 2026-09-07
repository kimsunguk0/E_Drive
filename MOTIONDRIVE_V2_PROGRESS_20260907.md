# MotionDrive V2 진행 기록 — 2026-09-07

## 15:48 KST: P0 보완 대조 시작

사용자 요청에 따라 장기 목표를 활성화하고 데이터/모델/독립감사 에이전트를 운영한다.
B200 `/NHNHOME/data/sukim/adcl`, GPU0–3만 사용한다. GPU6 기존 작업은 건드리지 않는다.
제출이나 외부 유료 자원 사용은 별도 승인 없이 하지 않는다.

코드·사전 판정 기준 커밋: `92fbdc6` (B200 테스트 86개 통과).
계획: `configs/motiondrive_v2/p0_repair_pairs_r1_s0.json`.
원본 실행 명세: `logs/motiondrive_v2/launch_p0_repair_pairs_r1_s0.json`.

| GPU | run | 단일 비교 변수 | 단계 |
|---|---|---|---|
| 0 | p0_tail_unit1_s0 | XY 출력 단위 (1,1), 대조 | joint 500 step, train16 fitting |
| 1 | p0_tail_unit10x5_s0 | XY 출력 단위 (10,5), 초기 함수 보존 | 동일 |
| 2 | p0_motion_highfeature_s0 | 현재 고해상도 feature 축소 | pretrain 1000 step, 전체 train/tune |
| 3 | p0_motion_lowfeature_s0 | 현재 영상부터 저해상도로 맞춰 추출 | 동일 |

학습 중간값을 최종 채택 결과로 읽지 않는다. 비교 기준은 별도 사전 프로토콜에 고정했다.
이 네 판은 P1의 G×S 실험이 아니다. P0 전제조건 보완을 먼저 판정한다.

## 이전 두 판의 종료 결과

- `p0_common_rawtime_s0`: 2000 step 완료, 사전 선택 지표인 전체 tune history MAE로 best step750.
- `p0_joint_fit_rawtime_s0`: 1500 step 완료, 미니 train16 평가 D3 0.5222.
  첫 2초에 비해 마지막 3초 waypoint 오차가 크다. query collapse나 gradient 단절은 확인되지 않았다.
- 공통 사전학습은 lane/객체 raster와 영상 조건부 속도를 학습했지만,
  정상 history가 현재 영상 반복/반전보다 실질적으로 우수하다는 증거는 아직 약하다.
- 같은 영상에 ±8px 수평 이동을 준 correspondence 진단에서는 feature 해상도 정합이
  올바른 이동 검출률 약13%→약96%를 만들었다. 실제 주행 속도·planning 이득의 증거는 아니다.

전체 감사 원본:
`reports/motiondrive_v2_p0_aux_audit.json`,
`reports/motiondrive_v2_p0_fit_audit.json`,
`reports/motiondrive_v2_correspondence_probe.json`.

## 실제 3090 측정의 범위

학습된 common best750, G0S0/legacy 원설정, 실제 영상 batch1:
CUDA median 32.1296ms, p95 32.1691ms, p99 32.1926ms, wall median 32.1484ms.
warmup20/repeats50, 전체 encoder+motion+scene+heads+planner 포함,
CUDA event와 synchronize 사용. 전처리/H2D 제외. 감사 43/43 통과.

이 숫자는 새 high_feature/low_feature 추가 current encoder 모드나 최종 G1S1의 측정값이 아니다.
채택 구조와 학습 체크포인트에서 다시 전체 forward를 측정한다. 4090 실측값이라고 환산하지 않는다.
원본: `reports/latency_3090_r50_p0_common.json`, `reports/audit_3090_r50_p0_common.json`.

## 아직 달성하지 않은 것

V2의 grouped holdout planning 경쟁력, P1 G×S 순효과, 복수 seed 재현성,
채택 모델의 전체 이미지 교란/출처 감사와 latency, 기존 제출 후보 대비 우위는 미확인이다.
P0 인지 점수나 작은 fitting 결과로 1위 가능성을 확정하지 않는다.
