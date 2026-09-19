# A2: 세밀한 motion 대응 / 공개 backbone에서 새 학습

사용자가 2026-09-19에 승인한 1·2번을 독립 실행한다. 신규 FULL이나 제출은 포함하지 않는다.

|Arm|유일한 주 변경|초기값|대조|
|---|---|---|---|
|A2-C2F-MOTION|영상 기반 coarse-to-fine motion 특징 경로 추가|성공한 Q10 계보 upstream + 기존 QREFINE 재구성|완료 QREFINE 20,554 update|
|A2-FRESH-NUIM|ETRI 학습 가중치 상속 제거|공개 nuImages ResNet50 trunk, 나머지 새 초기화|같은 구조·새 update 예산의 완료 QREFINE|

Fresh arm은 과거 ETRI 학습 노출량까지 같은 대조가 아니다. 동일 신규 예산에서 초기화
레시피를 비교하며, 20,554 update의 미수렴 결과를 scratch 학습의 상한으로 해석하지 않는다.
두 arm을 합치지 않고, 기존 QREFINE terminal이나 FULL 가중치에서 이어 학습하지 않는다.

## 고정한 학습 계약

- DEV train310 83,700행 / V0 37 scenes 1,998행, 기존 split 및 row SHA 유지.
- Batch16 / microbatch8 / BF16 / fixed BN / flip .5 / nominal provided status.
- Backbone LR 5e-6, 나머지 5e-5, AdamW, warmup200, cosine20,554, clip5.
- 기존 PREFIX + LEN .25 + occupancy/lane/motion .2 및 uncertainty 유지.
- 평가·저장: 3,426 / 6,852 / 10,278 / 13,704 / 17,130 / 20,554.
- 주 비교는 terminal. 중간값을 선택하면 같은 V0 선택이라는 사실을 별도 표시.
- FULL 가중치·특징·teacher·적응 통계를 DEV에 반입하지 않는다.

## 1. Coarse-to-fine

기존 native front 768x432 현재+H4 pass의 C1(stride4)을 재사용한다. 추가 RGB frame,
추가 backbone forward 또는 외부 flow 모델은 없다. 기존 stride8/16 motion 경로도 유지한다.

1. 기존 motion projection의 stride16 descriptor에서 radius4 cosine correlation을 계산한다.
2. 유효 후보만 temperature .07 softmax 후 영상 기반 대응 위치를 추정한다.
3. 대응 변위를 stride4 단위로 4배 변환하고, 각 위치 주변 radius2의 영상 특징을 다시 읽는다.
4. 실제 위치는 `history(x + flow(x) + local_offset)`이다. 공간적으로 변하는 warp를
   먼저 만든 뒤 unfold하는 다른 연산과 구분한다. 좌표·sampling·correlation은 FP32다.
5. Fine cost/visual descriptor/영상 추정 flow·confidence/visibility를 작은 CNN으로 처리하고,
   기존 correlation pair map에 합친 뒤 기존 motion pooling과 head/planner를 사용한다.
6. 마지막 feature 출력층은 0 초기화한다. 기존 QREFINE 함수가 처음에 정확히 보존되고,
   이후 기존 최종 planning 및 state/history supervision으로 새 경로를 학습한다.

새 분기는 raw status·goal·pose를 받지 않는다. status는 기존 공통 scene query에서만
사용하며, 동일 scene을 occupancy/lane/planner가 읽는다. 이 설명은 구조의 실제 의존성이며
운영국의 개별 승인 주장이 아니다.

## 2. Fresh nuImages

공개 파일 `ckpt/backbones/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth`
SHA256 `4096396018c0cf59fbe0eb1afe6e269f4676b34460bed5eedde5d7680d58bb4e`.
318개 trunk tensor를 전부 로드한다. Compact FPN, scene/QREFINE, motion/state/history,
planner, A2 status query는 새로 초기화한다. FPN 사전학습을 했다고 주장하지 않는다.
기존 control manifest에서 읽는 것은 model_config뿐이다. 모든 가중치와 BN statistics의
출처는 공개 trunk 또는 새 초기화이며, 별도 initializer의 SHA와 load 범위를 기록한다.

## 검사와 실행

`preflight.py`: zero-init FP32/BF16 전 출력 parity, 합성 translation/flip/ramp/visibility,
실제 PREFIX gradient 경로, status/goal-motion 불변, 동일 scene 소비, strict fresh reload,
전체 B1 공식 동일 FlopCounterMode 및 B200 시간을 확인한다. RTX4090 시간으로 환산하지 않는다.

`launch.py --smoke`: GPU0/1 각각 2 update 및 V0 전체 평가로 실제 trainer 경로를 검사한다.
`launch.py`: 같은 GPU에 각각 20,554 update를 시작한다. 기록·checkpoint는 별도 경로에 보존한다.
이미 존재하는 run/log/receipt를 덮어쓰지 않는다. GPU2는 사전 검사, GPU3은 평가 여유 슬롯이다.

기록: `reports/a2_motion_fresh_20260919/`.
대형 산출물: `work_dirs/a2_motion_fresh_20260919/`.
