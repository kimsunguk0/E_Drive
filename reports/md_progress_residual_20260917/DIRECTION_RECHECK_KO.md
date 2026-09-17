# MotionDrive 방향 재점검

작성일: 2026-09-17

## 판정

`MR FULL + base-plan-conditioned scalar progress residual`은 유지한다. 다만 현재 병목은
residual의 표현력이 아니라 새 세션에서 보정 부호와 크기를 식별하는 능력이다. 따라서
두 번째 계수 `delta_a`보다 SIDE의 직접 영상 운동 감독과 plan-denoising을 먼저 검증한다.

## 판정 근거

| 항목 | 실측값 |
|---|---:|
| 등록 MR V0 PREFIX | 0.191002 |
| 배포형 scalar oracle | 0.080906 |
| 배포형 `delta_v+delta_a` oracle | 0.063959 |
| FRONT scalar warmup final | 0.190236 |
| warmup 실제 `abs(c)` 평균 | 0.002009 m/s |
| V0 scalar oracle `abs(c)` 평균 | 0.141596 m/s |
| 일반 주행 실제-oracle 계수 상관 | 0.0603 |
| 일반 주행 강한 oracle 표본 부호 정확도 | 0.5499 |

Scalar 하나로도 목표 점수보다 충분히 낮은 oracle이 나오므로 자유도 증가는 우선 문제가
아니다. Warmup이 회수한 개선은 대부분 `steady/depart`에서 나왔고, 1,875개 `nonstop`
표본에서는 FRONT가 거의 변화가 없고 SIDE는 소폭 악화했다. 반면 scalar oracle은
`nonstop`을 0.19674에서 0.08524까지 낮춘다. 일반 주행의 식별 실패가 주된 간극이다.

## 현재 GPU 배치

| GPU | 실행 |
|---|---|
| 0 | 등록 MR 레시피의 376-scene FULL fit |
| 1 | FRONT scalar joint control |
| 2 | SIDE scalar + image-only absolute motion auxiliary |
| 3 | GPU2와 같은 scalar 모델 + synthetic plan-progress denoising |

SIDE auxiliary는 좌우 과거 영상과 고정 calibration에서만 state/history를 예측해 직접
감독한다. Denoising은 학습 중 base plan에 알려진 progress 오차를 넣고 원래 plan을
복원하게 한다. 두 경로 모두 raw goal, dynamic pose, provided status를 입력하지 않는다.
Base plan이 전달하는 goal 영향은 간접 의존성으로 문서화한다.

## 평가 게이트

각 공통 endpoint에서 다음을 함께 본다.

1. 최종 PREFIX와 새 모델 base PREFIX
2. `final - base` residual 순기여
3. `nonstop`, `steady`, `depart`별 PREFIX 변화
4. 같은 새 base의 배포형 oracle
5. 실제 계수와 oracle 계수의 MAE, 상관, 강한 표본 부호 정확도
6. 1초, 2초, 3초 PREFIX 성분

첫 joint endpoint에서도 일반 주행 개선이 없고 residual 순기여가 0.005보다 작으면 해당
계보를 terminal까지 관성적으로 연장하지 않는다. Scalar가 실제 일반 주행 개선을 보인
뒤에만 `delta_a`를 다시 검토한다.

## 실패 시 다음 구조

SIDE와 denoising도 일반 주행 부호를 알아내지 못하면 MLP 크기나 계수 수의 문제가 아니다.
다음 공격안은 고정 calibration과 road-plane warp/cost volume을 이용해 metric ego-motion을
명시적으로 추정하는 visual-odometry 분기다. 이 경우에도 정렬 전 motion 관측을 유지하고,
추정 운동과 base plan의 progress 차이를 최종 PREFIX로 학습한다. 준비되지 않은 recurrent
BEV나 6-camera memory를 먼저 만들지는 않는다.
