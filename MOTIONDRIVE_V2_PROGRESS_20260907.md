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

## 16:09 KST: 보완 결과 및 BN 정책 대조 시작

앞선 네 판은 모두 종료했다. 결과/후속 사전 기준 커밋은 `38ee977`이다.

| LAST500 정상 train16 | 단위(1,1) | 단위(10,5) |
|---|---:|---:|
| D3 | .28081 | .21418 |
| 3초 L2 | 1.96714 | .80705 |
| 첫2초 정규화 D3 | .16313 | .16296 |

출력 단위 변경은 초기 실제 좌표를1.526e-5m 이내로 보존했고, 앞부분 악화 없이 tail을 개선했다.
대조군 자체의 .52216→.28081은 추가 일정의 효과이며, scale의 순효과와 구분한다.
정상 D3<=.15는 두 판 모두 미달이다. train-BN .12012로 기준을 바꾸지 않는다.

Motion LAST1000 고정 tune370에서 low-high vx MAE 차이-.1480m/s,
11세션 paired CI[-.3082,+.0184], history 차이-.0316m, CI[-.1045,+.0414]다.
full tune1998 vx는1.3682→1.2702, history는.6182→.6025다.
시간반전 민감도의 일부 증거는 있지만 현재영상 반복 대조는 CI가0을 포함한다.
확정적 정상 개선이나 정확한 속도 관측 해결로 선언하지 않는다.

train16 영상만으로 shared BN을 표준 재보정했을 때 .21418→.21892로 개선되지 않았다.
가중치/원 checkpoint는 그대로다. 통계를 재평균하면 해결된다는 가설은 지지되지 않는다.
이후에는 **학습 중 BN 정책만** adaptive/fixed로 대조한다. 모든 가중치는 계속 학습한다.

| GPU | 현재 run | 시작점 | 일정 |
|---|---|---|---|
| 0 | p0_tail_bn_adaptive_s0 | unit10x5 LAST500 | 같은 train16, 500 step |
| 1 | p0_tail_bn_fixed_s0 | 동일 | 동일 |
| 2 | p0_motion_bn_adaptive_s0 | low_feature LAST1000 | 전체 train/tune, 1000 step |
| 3 | p0_motion_bn_fixed_s0 | 동일 | 동일 |

프로토콜 `MOTIONDRIVE_V2_P0_BN_PROTOCOL.md`에 후속 판정 기준을 사전 고정했다.
실제 초기 state SHA와 데이터 순서는 각 쌍 안에서 일치했다. P1 G×S는 아직 시작하지 않았다.

## P0 low_feature 실제 3090 재측정

새로운 low_feature LAST1000 원설정(G0S0/scale1,1)에서 추가 current front encoder까지 포함:
CUDA median32.256ms/p95 32.292ms/p99 32.308ms, wall median32.282ms,
VRAM alloc403.0/reserved500MiB, warmup20/repeats50, 감사43/43 통과.
이는 최종 G1S1 정확도·심사 승인 결과가 아니다.

배포 bundle은 optimizer/RNG를 제외하고 모든 config/출처 SHA를 보존하여
317,745,437→106,193,135bytes로 줄였다. 기본 config로 복원해 scale이나 mode를 잃지 않는다.
원본 보고서는 `reports/*p0_motion_lowfeature_last1000.json`에 보존했다.
