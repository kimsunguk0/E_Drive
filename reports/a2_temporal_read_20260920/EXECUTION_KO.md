# A2 시점별 motion read 실행

2026-09-20 사용자 승인: 직전 제안의 **1번만 실행**. 가중치 평균(2번), FULL, 추가 continuation, 제출은 이번 실행에 포함하지 않는다.

## 변경과 대조

- 새 run: `A2-TEMPORAL-READ-s1`.
- 공통 scene/QREFINE, 기존 pooled motion/state/history, direct XY head는 유지한다.
- 기존 decoded waypoint query가 시간축을 유지한 `4×192×128` motion memory를 별도 attention으로 읽고, 그 출력을 기존 XY head 앞 feature에 더한다.
- 새 read는 LayerNorm + 128차원/4-head cross-attention이며 마지막 output projection만 0 초기화한다. 새 모듈은 66,560 parameters다.
- Raw status/goal/pose 입력을 planner나 motion/state로 추가하지 않는다. 새 memory는 기존 정렬 전 영상과 시간/위치 embedding이다. Query에는 기존 scene의 간접 조건이 포함되므로 전체 경로를 goal-independent라고 부르지 않는다.
- 실제 대조: 완료 FRESH-NUIM-s1의 **20,554-update terminal PREFIX 0.158707259**. 선택된 FRESH-CONT 0.153684060은 추가 학습을 거친 실용 후보이며 같은 예산의 구조 대조가 아니다.

## 고정 학습 조건

|항목|설정|
|---|---|
|초기값|기존 FRESH 공개 nuImages R50 trunk + random nontrunk, 모든 기존 tensor 동일|
|DEV train/V0|310scene/83,700행, 37scene/1,998행|
|Status 입력|기존 nominal causal producer, state/history supervision 정의 유지|
|Budget|20,554 update, batch16/micro8, seed1|
|LR|backbone5e-6, 기존/신규 head5e-5|
|Schedule|warmup200, 기존 cosine, fresh AdamW, weight decay0.01, clip5|
|Loss|기존 PREFIX + LEN0.25 + occupancy/lane/motion 가중0.2 및 기존 uncertainty|
|실행 정책|fixed BN, BF16 encoder/FP32 planner, 기존 flip0.5|
|예정 평가|3,426 / 6,852 / 10,278 / 13,704 / 17,130 / 20,554|
|주 판정|고정 terminal끼리. 중간점 선택은 reused V0 선택으로 별도 표시|

기존 control의 protocol/source/row SHA를 검사한다. 새 모듈 생성은 기존 RNG를 진행시키지 않는다. 학습 로그의 rolling row SHA를 control과 비교한다. 기존 control을 다시 학습하지 않는다.

## 완료한 사전 검사

[preflight.json](preflight.json)에 원시 측정값이 있다. Preflight는 train fixture 2행을 사용했으며 검사 중 수정한 가중치를 실제 학습 초기값으로 쓰지 않는다.

- 모든 기존 parameter/buffer 및 생성 후 CPU RNG 동일.
- FP32/BF16의 scene, occupancy/lane, motion/state/history, 최종 XY 및 기존 total loss 차이 **0**.
- 새 output projection에 실제 PREFIX gradient 발생. 출력층을 시험용으로 한 번 연 뒤 Q/K/V, query/memory norm, 영상 backbone에 유한 gradient 확인.
- 새 reader만 격리한 backward에서 네 시점 memory 모두에 gradient 발생.
- 실제로 작동하도록 연 scene status condition을 변경하면 scene이 바뀌고, motion pair/pooled/state/history 출력은 차이 **0**. Goal 변경도 동일 경계 확인.
- Occupancy/lane/planner가 같은 scene 사용. 전체 batch 대비 microbatch loss 합 차이 `9.54e-7`.
- 기존 flip 왕복 및 새 모델 strict reload의 B1 XY 차이 **0**.
- 동일 전체-forward counter: FRESH **729.991777G**, 새 모델 **730.044861G**.
- B200 B1 forward median: 대조19.46ms / 새 모델19.58ms. 작은 fixture 측정이며 RTX4090 인증이 아니다.

본 학습 전 별도 2-update smoke를 거친다. 실제 시작 PID/GPU/시각은 `launch_main.json`, 초기 건강 상태는 `launch_health.json`에 기록한다. 완료/중간 결과는 [RESULTS_KO.md](RESULTS_KO.md)와 `results.json`을 확인한다. CPU collector는 다음 학습을 자동 시작하지 않는다.
