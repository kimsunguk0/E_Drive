# P0 BatchNorm 학습 정책 대조 — 결과 전 고정

2026-09-07. 이전 P0 두 쌍은 완료했으나 전체 fitting 게이트가 미통과다.
계획: `configs/motiondrive_v2/p0_bn_pairs_r2_s0.json`.

## 근거와 반증

단위변경 LAST500은 3초 L2를1.967→.807m로 낮췄지만 정상 train16 D3=.2142,
train batch-stat 진단=.1201로 차이가 남았다. 첫2초 정상 D3는.1630이다.
train16 영상만으로 단일 shared BN을 표준 재보정한 결과는.2189로 개선되지 않았다.
따라서 "running mean을 다시 계산하면 해결"이라는 주장은 기각한다.
이 결과로 해상도나 BN이 유일한 원인이라고 확정하지 않는다.

별도 시간 특징 LAST1000 비교에서 low_feature는 고정370 vx MAE를.148m/s 줄였으나
11세션 CI[-.308,+.018]은0을 포함했다. 시간반전 악화는 일부 있지만 현재영상 반복
대조의 CI는0을 포함한다. 정확한 motion 관측 문제가 해결됐다는 뜻이 아니다.

## 조작 변수

매 iteration `model.train()` 이후 BN만 다음 중 하나로 설정한다.

- adaptive: 기존 batch-stat 정규화 및 running-stat 갱신.
- fixed: 시작 checkpoint의 running-stat으로 정규화하며 running-stat을 갱신하지 않음.

양쪽 모두 backbone, BN affine, FPN, motion, scene, planner의 모든 기존 trainable
가중치를 계속 학습한다. stage freeze가 아니다. GroupNorm/decoder와 보조손실은 바꾸지 않는다.
추론은 양쪽 모두 정상 eval 모드이며 per-test batch-stat, online 적응, 후처리를 쓰지 않는다.
두 팔의 시작 가중치·BN buffer 해시는 같아야 하고, 전체 데이터·누적 row 순서는 같아야 한다.

## 쌍 C: train16 fitting

GPU0 adaptive, GPU1 fixed. 같은 `p0_tail_unit10x5_s0/last.pth`, scale(10,5),
legacy motion, seed0, batch8, 새 AdamW, warmup20/cosine500step.
headLR1e-4/backboneLR1e-5, 보조가중치각.2/uncertainty on 유지.
이전과 같은16 train/12 tune. 같은 마지막500step의 정상 train16 평가가 판정용이다.

기존 gate D3<=.15 및3초L2<=1m를 그대로 유지한다. 첫2초 정규화D3가 대조보다.02m
넘게 나빠지면 자동 채택하지 않는다. 둘 다 통과하면 단순한 기존 정책을 우선하고,
fixed만 통과하면 큰 데이터 학습으로 전이되는지 별도 확인한다. 둘 다 실패하면 여전히 P0 미통과.
tune12의 우열은 일반화나 채택 근거로 쓰지 않는다.

## 쌍 D: 공통 perception/motion 사전학습

GPU2 adaptive, GPU3 fixed. 같은 `p0_motion_lowfeature_s0/last.pth`, low_feature,
G0S0, scale(1,1), train54,810/tune1,998, seed0, batch16,
새 AdamW, warmup100/cosine1,000step. 나머지와 planning loss=0은 동일.

같은 LAST1000 normal full-tune vx/history 및 occupancy/lane을 먼저 비교한다.
고정 tune370/11세션 paired 분석과 네 영상 대조도 같은 표본으로 재실행한다.
앞선 vx .1m/s screening과 temporal-control 보고 범위를 유지한다.
best(history) 값은 별도로 기록하되 best-vs-last를 섞지 않는다.
fixed가 작은 fitting만 좋아지고 정상 heldout 인지가 나빠지면 공통 정책으로 자동 확대하지 않는다.

## 다음 단계

이 테스트는 최종 G×S 토너먼트가 아니다. 양쪽 모든 결과와 한 seed라는 한계를 보존한다.
P1에는 동일한 common initialization과 하나의 학습 정책을 고정한다.
최종 val136/기존 val38은 이번 선택에 사용하지 않는다. GPU4–7에는 접근하지 않는다.
