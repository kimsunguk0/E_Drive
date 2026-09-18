# A2 통합 개선안 검토 및 P0 실측 — 2026-09-18

사용자 계획 `A2_Integrated_Improvement_Plan_20260918.md`를 현행 source와 대조했다.
이번에 실행한 것은 기존 A2 checkpoint의 입력 교체 평가와 producer/attention 초기화 원리 검사다.
새 FULL/BASE-NOM/MH4 학습 및 MH4 실제 모델 통합은 아직 실행하지 않았다.

## 판단

제안 순서를 채택한다: 배포 status 정합성 → A2 FULL-NOM 및 BASE-NOM/MH4-NOM 비교 → SIDE-SCENE.
QREFINE은 독립 후속 가설, scene/global/motion status value gate는 기본안에서 제외한다.
4-head의 성능 개선은 아직 미측정이며, 일반 주행 오류가 단일 attention 때문에 생겼다고 입증된 것은 아니다.

P0의 실측 결과는 작은 입력 불일치가 있음을 확인하면서도, 그것이 현재 V0의 주요 점수 병목은 아님을 보여준다.
향후 입력 producer는 통일하되, 이 문제를 큰 성능 회복이 예상되는 수정으로 과장하거나
추가 진단으로 구조 실험을 오래 지연시키지 않는다.

## P0: 고정 A2 checkpoint에서 status만 교체

- checkpoint: `/NHNHOME/data/sukim/adcl/work_dirs/md_shared_dynamics_20260917/A2-DIRECT-s1/ckpt_step20554.pth`
- SHA256: `ffeac3447eb777e744ab4f3501988b2279f895fa0077fe5111a7516157aa0469`
- GPU 0, V0 1,998행. 새로운 학습 0 update.
- 이미지·calibration·goal·pose alignment·nominal time_offsets·GT·가중치 모두 고정.
- S-REAL: 원본 pose와 실제 timestamps로 계산한 status. cached state target와 전체 행 정확히 일치.
- S-NOM: 공식 clip과 같은 pose record만 받아 frame 차이 × 0.1초로 계산한 status.
- state/history supervision은 원래 정의를 유지했다.

| 지표 | S-REAL | S-NOM |
|---|---:|---:|
| PREFIX | 0.164280978527 | 0.164455314083 |
| L2_1s | 0.096108015024 | 0.096191632473 |
| L2_2s | 0.163552517263 | 0.163700054968 |
| L2_3s | 0.233182403296 | 0.233474254809 |

- PREFIX 차이(S-NOM−S-REAL): **+0.000174335556**.
- 두 예측 궤적 간 PREFIX 거리: **0.005607703121m**.
- 샘플별 점수 차이 범위: -0.02741968 ~ 0.03100815.
  평균 차이가 작아도 모든 샘플의 출력이 동일하다는 뜻은 아니다.
- 원래 저장된 A2 예측과 S-REAL의 좌표 차이: 0.
- status 교체에 따른 motion_features/state_hat/history_hat 차이: 모두 0.
- 샘플별 삼각부등식 `|D(S-NOM,GT)-D(S-REAL,GT)| <= D(S-NOM,S-REAL)` 확인.
- 양쪽 status fit에 쓰인 pose 수는 모든 V0 행에서 11개로 동일했다.

### MAE와 signed mean

| 채널 | S-NOM−S-REAL signed mean | MAE |
|---|---:|---:|
| vx | -0.000085574240 | 0.0088496435 |
| vy | -0.0000083224478 | 0.00023852294 |
| ax | -0.00016388122 | 0.016266957 |
| ay | -0.000010329498 | 0.00044907116 |
| yaw_rate | 0.0000029875676 | 0.000014452066 |

이전에 보고한 vx 0.00885m/s / ax 0.01627m/s²는 **MAE**다.

### 실제 test 입력

1,125개 clip 모두 최상위 파일은 calibration.parquet, command.parquet, ego_pose.parquet였다.
pose column은 frame/x/y/z/roll/pitch/yaw로 timestamp가 없다.
동일한 frame-based producer가 모든 clip에서 유효한 status를 생성했다.
모든 clip에서 미래 +50 pose의 XYZ/RPY를 NaN으로 교체해도 status가 동일했다.
이는 status producer 검사이며 전체 raw-image adapter/제출 parity 완료를 뜻하지 않는다.

## 4-head 초기화에 대한 검토

제안한 Q/K head당 32, value head당 32, concatenate 128 구성을 유지할 수 있다.
기존 conditioned query 및 metadata가 포함된 key 뒤에 head별 identity 변환을 넣고,
value의 128채널을 같은 순서로 네 묶음으로 나누면 초기 weighted pooling 함수를 보존할 수 있다.

현행 production masked_softmax를 사용한 합성 FP32/BF16 입력에서:
- 초기 pooled evidence 차이 0.
- 모든 source가 invalid인 cell의 evidence가 정확히 0.
- head별 query transform gradient가 서로 달랐고 유한했다.

이 검사는 실제 encoder 전체·checkpoint의 FP32/BF16 parity가 아니다.
실제 통합 시 base mean, global image context, cross-cell branch, SpatialMix, metadata,
mask와 source 순서를 보존하고 최종 plan까지 검사해야 한다.

초기함수 보존과 학습 출발점을 혼동하지 않는다.
본 비교는 성공한 upstream 초기값에서 BASE-NOM/MH4-NOM을 같은 예산으로 학습하는 계획이다.
terminal A2 변환 parity를 검사하더라도 그 자체가 terminal continuation을 뜻하지 않는다.

## 확정할 비교 설계

| run | 데이터 | 입력 status | scene | update |
|---|---|---|---|---:|
| A2-FULL-NOM | train376, 유효 101,520행 확인 후 | 공통 nominal producer | 기존 single-head | 24,931 |
| A2-BASE-NOM | train310 | 같은 producer | 기존 single-head | 20,554 |
| A2-MH4-NOM | train310 | 같은 producer | 4-head | 20,554 |

DEV seed/초기값/샘플 순서/유효 batch16/LR/loss/flip을 맞춘다. FULL 가중치·teacher·통계는 DEV로 돌려주지 않는다.
기존 실제 timestamp로 학습한 A2의 배포 입력 점수 0.164455는 이전 후보 기준으로 보존한다.
MH4의 구조 효과는 같은 nominal 입력으로 학습한 BASE-NOM과 비교한다.

- 기존 state/history supervision을 nominal 값으로 덮어쓰지 않는다.
- 4-head와 QREFINE을 첫 run에서 합치지 않는다.
- SIDE-SCENE는 현재6cam+frontH4를 유지하고 좌우 전방의 짧은 과거 4장을 shared scene에 추가하는 별도 관측 실험이다.
- 상태의 value gate 및 motion/state/history conditioning 경로는 추가하지 않는다.
- 규정에 대한 개별 승인으로 표현하지 않는다.

## 오차 진단을 판정에 반영

별도 저장 예측 분석에서 A2 점수의 첫 2초 기여는 74.81%다.
일반 주행의 첫 2초 구간 속도 제곱오차 중 공통 성분 비중은 75.20%다.
이것은 D3의 가산 분해 또는 v0/가감속 타이밍의 인과 기여도가 아니다.

최종 PREFIX와 함께 첫 2초 PREFIX 기여, 일반 주행 그룹, 공통 진행량 오차 및 시간변화 성분,
같은 GT tangent mask의 종/횡 절대오차를 기록한다. 보조 지표가 좋아져도 PREFIX가 나빠지면 채택 근거로 쓰지 않는다.

## 산출물

- `paired_status_eval.json`: checkpoint/source hash, 입력 차이, paired metrics, 불변성 검사.
- `paired_status_eval.npz`: 원본 row 순서의 status/예측/GT/점수 차이.
- `producer_and_mh4_principle_checks.json`: producer 경계조건과 attention 초기화 원리 검사.
- `experiments/md_a2_deploy_status_20260918/nominal_status.py`: supervision과 분리된 배포형 입력 producer.
- 같은 폴더의 `paired_status_eval.py`: 재현 평가.
