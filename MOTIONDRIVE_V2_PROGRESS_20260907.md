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

## 16:38 KST: P0 BN 결과 및 P1 실행 준비

BN 대조 네 판의 계산이 끝났다. 작은 train16의 정상 eval D3는 adaptive .19161,
fixed .04360이며 fixed만 사전 fitting gate를 통과했다. 3초 L2는.31013m다.
train-BN 모드로 평가 기준을 바꾸지 않았다. 이 결과는 작은 train fitting이며 일반화 주장이 아니다.

전체 tune1998에서 fixed는 vx MAE1.21150→1.03759m/s,
history MAE.56234→.48646m를 보였다. footprint IoU 약2%p 하락/lane IoU 약1.3%p 상승도 기록한다.
고정 tune370의 11세션 paired CI는 vx/history 개선에서0을 제외했다.
저해상도 feature 정합 + fixed BN statistics를 P1 공통 개발 설정으로 선택했다.

adaptive motion run은 checkpoint 저장·평가 완료 후 native SIGSEGV 문구와 종료 지연이 있었다.
원인은 확정하지 않았다. 프로세스는 개별 중지 시도 전 이미 사라졌으므로 실제 signal을 보내지 않았다.
두 LAST checkpoint 무결성/모든 tensor 유한성을 검사하고 별도 전체 tune replay에서
보고된 모든 지표를 정확히 재현했다. 학습 결과 유효성과 정상 종료 여부는 구분한다.
후속 P1에는 실제 OS 종료를 별도 기록하는 supervisor를 사용한다. native 문제를 해결했다고 주장하지 않는다.

원본은 `reports/p0_tail_bn_pair_comparison.json`,
`reports/p0_bn_motion_pair_comparison.json`,
`reports/p0_bn_motion_checkpoint_integrity.json`,
`reports/p0_bn_motion_independent_replay.json`이다.

P1은 planning 미학습 공통 BN-fixed best1000에서 G×S 네 판을 같은6000step으로 시작한다.
LAST6000 2×2 비교를 주표, 팔별 tune-BEST를 보조표로 사전 고정했다.
정확한 계보·손익·실행/평가 조건은 `MOTIONDRIVE_V2_P1_PROTOCOL.md`에 고정했다.
이 문단 작성 시 P1은 아직 미실행이며 최종val136과 과거val38을 열지 않았다.

## 16:45 KST: P1 네 팔 발사

소스/계획 커밋 `fdd309d8ef23a6ea35131636551c51e51aa12b89`, B200194 tests 통과.
실행 manifest: `logs/motiondrive_v2/launch_p1_gs_r1_s0.json`.
실제 시작은16:45:17 KST, GPU0–3에 G0S0/G1S0/G0S1/G1S1 순서다.
supervisor PID1921374–1921377, 실제 trainer PID1921378–1921381.

공통 초기 tensor state SHA `30324570835ad3895322c596c1666ee983b46dd8f331be7b8d9e972dc9de86c0`,
parameter26,409,112개, train/eval row SHA가 네 팔 모두 같다.
step1–210의 공통22개 로그에서 누적 샘플 순서 SHA가 모두 일치했다.
모델 설정 차이는 G/S뿐이다. 영상에서 추론한 motion/history/state auxiliary는 모두 유지한다.
초기 정상 tune D3는 약13.1m이며 planning 미학습 초기값이므로 성공/실패 판정 값이 아니다.

GPU6 기존 작업은 보존했다. 시작 이후 GPU1/2에 짧게 보인 별도 프로세스는
추가 신원 조회 전에 종료되어 소유/작업을 확인하지 못했다. 어떤 signal도 보내지 않았다.
16:47경에는 GPU0–3 각각 우리 trainer만 남았고 VRAM은 각약43,844MiB(42.8GiB)였다.
이 일시적 관찰은 실패 원인으로 단정하지 않으며 최종 latency는 별도3090 단독 측정한다.

실험 진행 중이므로 P1 효과·경쟁력·규정 최종 적합성을 아직 선언하지 않는다.
