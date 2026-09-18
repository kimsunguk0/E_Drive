# A2 제공 command 비교 — 2026-09-19

사용자 지시: “이번엔 command 넣어본거 해 봐”. GPU0–3 사용 권한을 유지한다.
이번 신규 학습은 **A2-COMMAND-NOM-s1 한 개**, GPU0에서 시행한다.
완료된 BASE-NOM을 matched control로 재사용하므로 대조군을 중복 학습하지 않는다.
GPU3는 사전 검사와 완료 후 고정 inference에 사용한다. 기존 GPU4–7 작업에는 개입하지 않는다.

## 변경과 비교

현재 raw `command.parquet`의 6종 의미 지시를 one-hot으로 입력한다.
LANE_KEEP / TURN_LEFT / TURN_RIGHT / LANE_CHANGE_L / LANE_CHANGE_R / U_TURN.
192개 파라미터의 bias 없는 선형 projection을 0으로 초기화한다.
32차원 delta를 기존 A2 **공통 scene query**에 더한다.
카메라의 key/value, sampler, source 수, direct planner, motion/state/history는 기존과 같다.
같은 최종 scene tensor를 occupancy·lane·planner가 읽는다.
제공 command/status/goal을 별도의 planner token, value, motion 입력으로 전달하지 않는다.
Command와 goal 모두 scene의 영상 선택에 영향을 주므로 전체 planner는 이 조건들의 간접 영향을 받는다.

`vad_cmd`는 사용하지 않는다. 기존 ego cache의 vad_cmd는 GT 3초 말단 횡변위로 만든 3분류이며,
이번 입력은 원본 parquet의 의미 command6종이다. 두 의미가 다르므로 섞지 않는다.
현재 cache producer는 pose/GT/future XY를 읽어 command를 생성하지 않는다.

기준은 완료된 single-head BASE-NOM terminal PREFIX **0.165510648966**.
기존 최선 fixed terminal QREFINE **0.164251769704**도 실용 비교표에 함께 둔다.
G1 중간 checkpoint **0.163882487793**은 같은 V0에서 선택한 개발 후보이며 독립 확인값이 아니다.
이번 arm은 command 효과를 우선 분리하기 위해 QREFINE/G1/sampling을 결합하지 않는다.

## 학습 계약

|항목|설정|
|---|---|
|DEV train / V0|83,700 / 1,998행, 기존 split 그대로|
|초기값|r0_init_tplus.pth, 기존 MR fuse 재구성|
|기존 공통 파라미터 SHA|116f67a476b5ab519ec4384d6e15d5f5e36e83d1c89812f733b056ba5874593f|
|Update / effective batch / microbatch|20,554 / 16 / 8|
|Backbone / head LR|5e-6 / 5e-5, fresh AdamW|
|Warmup / schedule|200 / 기존 cosine|
|Loss|기존 PREFIX+LEN0.25+기존 occupancy/lane/motion 묶음|
|Status 입력|기존 nominal causal producer, supervision 정의 그대로|
|BN / precision / flip|fixed / BF16 / 0.5|
|평가|3,426마다 및20,554 terminal; 모든 예정 예측 보존|
|Primary 비교|같은 학습 예산 terminal COMMAND−BASE|

기존 command 없는 공통 tensor SHA, split/rows, loss, LR, status producer가 일치한다.
새 모듈은 RNG를 소비하지 않는다. Dedicated sample stream의 로그 SHA를 BASE와 비교한다.
좌우 반전은 TURN_LEFT↔TURN_RIGHT, LANE_CHANGE_L↔LANE_CHANGE_R,
LANE_KEEP→자기 자신, U_TURN→자기 자신으로 처리한다.
FULL 산출물/teacher/cache를 DEV로 가져오지 않는다.

## 완료한 사전 검사

원본 command와 frame timestamp를 exact join한 train+tune85,698행이 기존 meta와 전부 일치했다.
공식 test 제공 command1,125개는 동일 parser로 읽었으며 unknown/missing0.
입력 policy와 raw 파일 hash는 입력 cache manifest 및 INPUT_POLICY.json에 보존한다.
두 실제 train 영상에서 zero-init FP32/BF16 모든 공통 출력 차이0.
Command PREFIX gradient가 존재하며, nonzero command 조건에서 scene/plan이 변하고
motion/state/history 차이는0. 세 shared consumer가 같은 tensor를 사용한다.
Timestamp 중복/누락, 6종 flip, exception 후 context 정리, B1 전체 FLOPs 검사를 통과했다.
2-update smoke와 V01,998행 evaluation 완료, nonfinite0.
첫 학습 batch의 모든 loss와 row SHA가 BASE의 첫 batch와 정확히 같았다.

FLOPs: BASE729,815,616,192 → COMMAND729,815,616,576.
B200 B1 median 약18.88ms는 RTX4090 측정값이 아니다. RTX4090 latency는 미측정이다.
위 사전검사는 기능 확인이며 command 성능 개선을 의미하지 않는다.

## 평가와 해석

공식 PREFIX=[11,11,5,5,2,2]/36. L2_1s/2s/3s는 앞2/4/6점 평균이다.
전체, 첫2초, nonstop, 제공 의미 command, 별도 GT3초 기하 그룹, 종/횡 투영오차를 비교한다.
Command GT 기하 그룹을 학습 label이나 추가 입력으로 만들지 않는다.
11session paired bootstrap20,000회(seed0)를 사용하며 반복 사용 DEV/단일 seed 한계를 명시한다.
V0 의미 좌회전39행은1session, 우회전42행은3session이고 U_TURN0행이다.
따라서 좌회전 subgroup 점수 하나로 회전 일반화를 확정하지 않는다.
완료 후 같은 checkpoint에서 command를 모두 LANE_KEEP으로 바꾼 고정 진단을 추가한다.
이는 해당 모델의 command 의존성을 보는 inference 개입이며 독립 학습 control이 아니다.

## 규정 근거와 범위

OPEN_ISSUE.md Q1은 제공 command와 vad_cmd 모두 사용 가능하다고 답했다.
Q7/Q8 및 마지막 구조 질의의 답변은 공통 인지 feature 단계의 간접 활용과
planner 내부 query 입력을 구분한다. 이번 구조는 기존 A2의 공통 scene 경계를 유지한다.
공지2에 따른 최종 제출 코드 심사를 사전 승인된 것으로 주장하지 않는다.

## 실행 후 기록

launch_receipt.json: 실제 PID/시작시각/source commit/GPU.
runtime/train.log와 run manifest: 학습 상태.
watcher_status.json: 완료 대기 및 후처리 상태.
terminal_results.json / results.csv / RESULTS_KO.md: 고정 terminal 비교.
command_counterfactual.json: 올바른 지시 대비 LANE_KEEP 고정 진단.
run_artifact_index.json: 가중치·예측·보고서 경로/크기/SHA.
자동 watcher는 한 번의 비교와 기존 승인된 GitHub 기록만 수행한다.
추가 학습/FULL/공식 업로드는 자동 실행하지 않는다.
