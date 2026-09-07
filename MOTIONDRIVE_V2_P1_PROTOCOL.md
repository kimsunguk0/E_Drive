# P1 G×S 실험 프로토콜 — 첫 planning holdout 결과 전 고정

2026-09-07. 이 문서 작성 시 P1은 미실행이다. P0 BN 최종 감사와 공통 초기값의
선정을 완료한 후 정확한 실행 JSON/원본 checkpoint SHA를 Git에 남기고 시작한다.
이 문서는 P0 미통과 상태를 통과로 바꾸는 문서가 아니다.

## 질문

실제 shared visual perception에만 goal을 반영하는 효과 G와,
영상이 예측한 상태/이력을 planner에 명시적으로 전달하는 효과 S를 분리한다.
현재6camera + 과거front4frame + low-resolution current front의 **원본 영상 계산**,
raw motion M, 실제 occupancy/lane/state supervision, direct3s decoder는 네 팔 모두 같다.

| GPU | G | S | run |
|---|---|---|---|
| 0 | OFF | OFF | p1_g0s0_s0 |
| 1 | ON | OFF | p1_g1s0_s0 |
| 2 | OFF | ON | p1_g0s1_s0 |
| 3 | ON | ON | p1_g1s1_s0 |

S OFF도 raw motion feature와 동일한 motion/state auxiliary를 유지한다.
G OFF는 goal을 neutral 처리하고 command는 네 팔 모두 입력하지 않는다.
planner goal token/GT status/GT history 직접 입력, 외부 경로 보정은 없다.

## 공통 초기값 및 training policy

- train203 scene/72 rawtime session으로 학습한 **planning loss=0 공통 P0**에서만 초기화한다.
- 작은16표본 G1S1 fitting checkpoint나 기존 full330 teacher는 초기값으로 사용하지 않는다.
- P0 BN paired 판정 후 policy와 common checkpoint를 고정한다. 해당 pretrain run의
  best는 기존 full-tune mean history MAE로 선택한 파일이며 planning 결과로 재선택하지 않는다.
- output_scale=(10,5)는 마지막 Linear 역재매개변수화로 초기 물리 출력을 보존한다.
  네 팔에 동일하게 적용하고 initial state SHA까지 같음을 검사한다.
- 네 팔은 같은 backbone/모든 trainable weights, loss, optimizer, precision, 데이터 순서를 쓴다.
- low_feature와 BN policy 채택은 P0 screening에 근거한 개발 선택이지 복수seed 우월성 증명이 아니다.

## 첫 일정: 6,000 step, 단일 seed screening

Train54,810 frame / tune1,998 frame, frame>=30, train stride1 / tune stride5.
batch16, eval batch4, workers4, seed0, bf16 backbone/FP32 final heads.
새 AdamW(head/FPN/새module LR1e-4, backbone LR1e-5), weight decay.01,
warmup200 후6,000step cosine, grad clip5. 약96,000표본 노출(약1.75 epoch)이며
epoch 끝의 작은 batch도 버리지 않으므로 정확한 노출 수는 row trace로 검증한다.

L=official D3 +.2 occupancy+.2 lane+.2 motion, 기존 uncertainty/stop 항 유지.
5초 auxiliary=0, smoothness=0, logit KD=0, 앙상블=0.
250step마다 같은 full tune를 평가하고 official temporal weighted D3가 최소인 best를 저장한다.
loss/raster/motion/정확한 step·학습 row SHA와 예외를 기록한다. NaN을 조용히 건너뛰지 않는다.
LAST6,000도 보존하고, best는 팔별 동일 선택 절차를 사용한다.
**동일 LAST6,000이 학습 노출을 고정한 2×2 효과의 주표**, 팔별 BEST는 실용적
checkpoint 선택 절차를 포함한 보조표다. 결과를 본 뒤 두 표 중 좋은 쪽으로 대표값을 바꾸지 않는다.

이 일정 자체가 수렴 상한이나 목표 성능 보장이 아니다. 명백한 실행 실패/비유한 gradient이면
해당 프로세스만 안전하게 중지·원인 분석하며, 단일 팔만 불공정한 sample 재시작을 하지 않는다.
중간에 좋은 팔만 longer schedule로 바꿔 네 팔 비교표에 섞지 않는다.

## 평가 및 해석

1차 metric은 frame mean of sum([11,11,5,5,2,2]/36 × 6점 L2)다.
기존 proxy 가중치는 별도 참고치이고 선택 기준이 아니다.
정상 전체 tune1,998, per-time/종횡/stop·accel·decel별 표본수와 session 평균을 보고한다.
11 rawtime session 단위 paired bootstrap, 같은 frame끼리 G/S 차이를 계산한다.
하나의 session 재표집을 네 팔에 공동 적용하고 각 session의 frame 수를 보존해
frame-mean을 재계산한다. session별 평균을 단순평균하는 다른 추정량의 CI는 보조표로 분리한다.

- G|S0 = D3(G1S0)-D3(G0S0), G|S1 = D3(G1S1)-D3(G0S1).
- S|G0 = D3(G0S1)-D3(G0S0), S|G1 = D3(G1S1)-D3(G1S0).
- 상호작용 = D3(G1S1)-D3(G1S0)-D3(G0S1)+D3(G0S0).
- 낮을수록 좋다. .01m 절대 개선은 개발 screening이며 신뢰구간/seed 확인 없이 확정하지 않는다.

같은 tune에서 checkpoint를 선택하므로 이 CI는 untouched test 성능 인증이 아니다.
과거 val38은 역사적 비교 집합이고 이번 checkpoint 선택/튜닝에 사용하지 않는다.
공통 초기값의 partial pretraining만으로 미학습 시나리오 planning 성공을 주장하지 않는다.

유망한 G1 모델은 3seed 확대 **전에** 전체 영상 교란, motion의 goal/pose 불변성,
고정 feature의 planner 독립성 및 shared perception 학습 품질을 확인한다.
통과한 구성과 가까운 대조군만 별도3seed 복제와 전체 forward 재측정을 한다.
P0 G0S0의32ms를 최종 G1S1 timing으로 재사용하지 않는다.
공식 제출·GPU4–7·외부 유료 자원·추가 데이터 취득은 이 실험의 권한에 포함하지 않는다.
