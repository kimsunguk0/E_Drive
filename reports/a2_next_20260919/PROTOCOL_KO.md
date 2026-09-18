# A2 다음 실험 / 2026-09-19

사용자 실행안을 따른다. 이전 assistant의 motion만 0.2→0.02/2,000 update 제안은 대체한다.
G는 기존 occupancy/lane/motion 전체 묶음 ×1.0 대 ×0.25, LEN은 0.25 그대로다.
S는 고정 source마다 영상과 고정 metadata에서 한 개의 bounded 2D offset을 예측한다.
G와 S의 변경은 첫 run에서 결합하지 않는다. 새 head/λ/해상도/모델 스윕은 하지 않는다.

## 최신 계보와 대조

현재 수치상 가장 낮은 DEV A2는 QREFINE terminal 0.164251769704다.
이전 A2 대비 확실한 개선이라고 주장하지 않되, 사용자 지시대로 G의 공통 부모로 동결했다.
G0/G1은 이 checkpoint 전체를 strict load하고 새 optimizer와 seed1/epoch0 sampler로 시작한다.
양쪽 3,426 update, backbone 1e-6/head 1e-5, warmup100, cosine, B16/micro8, fixed BN/BF16/flip0.5.
첫 평가는 같은 부모 전체 V0 replay(0 update)를 공유하고 1,142/2,284/3,426에 각각 평가한다.
주 비교는 terminal G1−G0 및 각 parent 대비다. 선택된 중간 best는 독립 검증이 아니다.

S는 single-head BASE-NOM의 기존 upstream initializer와 전체 공동 학습 20,554 update를 쓴다.
완료된 BASE-NOM 0.165510648966을 control로 재사용하되 공통 초기 tensor와 sample-order SHA를 검증한다.
S에 QREFINE 또는 G의 loss 계수를 넣지 않는다. nominal 제공 status 입력과 기존 실측 supervision은 분리한다.
S 기본: hidden64, ±2 feature cell, offset 마지막 weight/bias0, source 수/grid/카메라/history 유지.
같은 이동 위치에서 key와 value를 읽는다. 원래 invisible source는 복구하지 않고 바깥으로 이동하면 mask한다.
Camera/level one-hot, time, height, normalized projection coordinates만 offset metadata로 사용한다.
Status/goal/conditioned query/scene은 offset predictor의 인자가 아니다.
투영 위치에는 기존 pose alignment가 반영되므로 전체 모듈을 pose-independent라 부르지 않는다.

## 진단 결과와 실행 판단

고정 train sampler의 32 effective batch(512행), micro8×2 누적, 동일 augmentation과 forward에서
PREFIX, PREFIX+LEN, 기존 가중 auxiliary 묶음의 gradient를 분리했다. 모델 parameter/buffer hash 동일,
optimizer update0, full-batch mask normalizer 사용. scene의 보조 전용 head는 shared 그룹에서 제외했다.

| 공유 그룹 | AUX/main norm 중앙값 | p90 | cosine 중앙값 | 음의 cosine |
|---|---:|---:|---:|---:|
| Backbone/FPN | 1.0183 | 1.5620 | 0.0108 | 15/32 |
| Shared scene | 0.0404 | 0.0718 | -0.0040 | 17/32 |
| A2 status query MLP | 0.0047 | 0.0127 | 0.3861 | 13/32 |

backbone에서 크기가 작지 않고 반대 방향도 반복돼 G0/G1의 짧은 대조를 진행한다.
scene의 작은 gradient와 부호만으로 그 경로가 주요 병목이라고 하지 않는다.
이것은 raw pre-clip gradient이며 AdamW의 실제 업데이트나 V0 개선을 예측하는 증거는 아니다.
negative uncertainty NLL의 scalar 크기를 gradient 우세로 해석하지 않는다.

## 비교와 결과 보존

DEV train 83,700행/310scene, V0 1,998행/37scene/11session을 유지한다.
PREFIX는 [11,11,5,5,2,2]/36이며 prefix L2_1s/2s/3s를 endpoint로 오해하지 않는다.
일반 주행 및 첫 2초 기여, 같은 GT tangent mask의 종/횡 오차, 인지 지표, session paired delta를 함께 본다.
S의 offset 크기/saturation/invalid 비율은 고정 train probe와 V0에서 따로 측정한다.
모든 G/S train/평가에서 FULL weight/teacher/cache를 사용하지 않는다.
완료된 A2 FULL은 재시작하지 않는다. MR FULL 공식 점수는0.18596892793122946;
A2 FULL의0.088934는 학습 포함 V0 값이며 서버 성능이 아니다.

GPU0=G0, GPU1=G1, GPU2=S 본 학습, GPU3=진단/변경 경로 검사에 사용한다.
다른 작업을 중단하지 않고 실제 점유를 확인한 뒤 실행한다. GPU4–7 제외.
S 검사/비용 측정이 실패하면 수정 완료 전 본 학습을 시작하지 않는다.
실제 실행은 launch_receipt.json과 manifest로 확인한다. 이 문서 자체는 시작/완료 증명이 아니다.
서버 제출은 자동 실행하지 않는다. 유효한 DEV 승자만 별도 FULL 이전/배포 후보로 검토한다.
