# P0 보완 대조 실험 — 사전 고정 프로토콜

2026-09-07. 최신 OPEN_ISSUE 준수 및 장기 목표 아래 실행한다.
G×S 네 팔 전에 P0 감사에서 발견한 두 문제를 각각 분리 검증한다.
새 결과를 본 뒤 이 파일의 기준을 소급해서 바꾸지 않는다.

## 근거

- 인지는 학습됨: 고정 tune370에서 initial→last 객체 IoU .0342→.4342, lane .2079→.4755.
- best common은 기존 full-tune history 기준으로 고른 step750이다. 새370 표본으로 재선택하지 않는다.
- best vx MAE1.1443은 train median상수3.6860보다 좋지만, 현재영상 반복1.1606/시간반전1.1602와 거의 같다.
- ax는 median상수보다 낫지 않고 stop p>=.5 recall은0이다. 이것은 확률정보가 전혀 없다는 뜻은 아니다.
- 16표본 경로 피팅: 평가모드 D3 .5222, 첫2초 batch-stat 정규화D3 .0502이나 3초 오차 약5m.
  전체 query collapse/gradient 단절은 없다. 마지막2점이 train batch-stat D3의88.3%다.
- 기존 고/저해상도 feature 혼합은 ±8px 수평 이동 진단에서 올바른 offset율 약13%,
  같은 저해상도 영상으로 추출하면 약96%였다. 이는 feature 정합 진단이지 planning 개선 증거가 아니다.

## 쌍 A — XY 내부 단위에 따른 최적화

GPU0: 출력 단위(1,1), GPU1: (10,5).
같은 `p0_joint_fit_rawtime_s0/last.pth`, 같은16 train/12 tune, seed0, batch8,
새 AdamW, headLR1e-4/backbone1e-5, warmup20/cosine500step.
uncertainty/보조손실/공식 목표는 변경하지 않는다.

scale을 곱하기 전에 마지막 XY Linear의 weight와 bias를 역으로 rescale하여
초기 물리 좌표를 보존한다. 외부 위치 보정/goal 외삽이 아니라 neural head의 좌표 단위다.
모델 출력은 여전히 FP32이고 제출값은 그 출력 그대로다. optimizer moments는 두 팔 모두 새로 시작한다.

사전 기대조건: 초기 예측 차이 <1e-4m, 데이터 전체 순서 hash 동일.
판정은 같은 마지막500step checkpoint의 **정상 평가모드 train16**을 우선 쓴다.
train D3<=.15 및 3초 오차<=1m에 도달하는지 보고하고, 첫2초가 대조보다 .02m 이상 나빠지면
스케일을 자동 채택하지 않는다. 작은 tune12를 일반화 성능이나 최종 checkpoint 선택 근거로 삼지 않는다.
두 팔 모두 좋아지면 일정 연장 효과와 단위 변경 효과를 구분하고, 불필요한 변경은 채택하지 않는다.

## 쌍 B — motion용 현재 영상 feature 정합

GPU2 high_feature, GPU3 low_feature.
두 팔 모두 같은 common best750에서 시작, train54,810/tune1,998, seed0, batch16,
새 AdamW, 같은LR/보조손실, warmup100/cosine1,000step. planning loss는0.

두 팔 모두 현재6장을 처리하고, 같은 current front 저해상도1장+과거4장을 한 batch로 추가 인코딩한다.
현재 저해상도는 antialias=True bilinear로 만든다. 두 팔의 계산량/BatchNorm 입력은 동일하다.
raw motion으로 전달하는 현재 특징만 고해상도 feature 축소 vs 저해상도 직접 feature로 다르다.
scene feature의 입력, GT, 제공 정렬행렬, goal, time offsets는 바꾸지 않는다.
raw motion에는 goal/pose를 입력하지 않는다. 추가 current encoding도 전체 latency에 포함한다.

정상 평가모드에서 **같은 마지막1,000step**의 vx/history 오차를 우선 paired 비교한다.
기존 pretrain best 선택(history)도 별도로 보고하되 best-vs-last를 혼합하지 않는다.
고정 tune370 교란검사를 재실행한다. 정상 vx MAE .1m/s 이상 감소와 정확한 history의 실질적 이점이
같이 나타나는지를 screening한다. 이 작은 개발 기준을 공식 수상권 기준이라고 부르지 않는다.
11개 실제 세션의 paired bootstrap 불확실성을 병기한다. 실패하면 정확한 flow/상태 supervision과
손실 조건을 별도 검토하며, 영상 정보 자체가 없다는 결론으로 비약하지 않는다.

## 운영·공정성

설정: `configs/motiondrive_v2/p0_repair_pairs_r1_s0.json`.
GPU0–3만 사용한다. 기존 run/artifact는 덮어쓰지 않는다.
각 run의 모델 설정/초기 state hash/데이터 row hash/누적 batch 순서 hash를 기록한다.
쌍 A는 마지막 Linear 재매개화로 parameter hash가 달라도 초기 함수가 같아야 한다.
쌍 B는 초기 parameter hash까지 같아야 한다.
단일 팔만 중간 resume하여 sample 순서를 어긋나게 만들지 않는다.
전체 forward 감사/3090 실측은 채택할 구조에서 다시 시행한다.
P1의 goal/state 네 판은 이 보완 결과를 검토한 뒤 하나의 공통 초기값·구조를 고정하여 실행한다.
