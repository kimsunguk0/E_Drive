# Backbone / decoder / agent 세 실험과 공통 대조군

사용자 요청: "3 가지 다 해 봐". 사용 GPU는 0,1,2,3이다. 이 문서는 DEV 학습 명세이며 FULL 이전과 공식 업로드를 예약하지 않는다.

## 고정 비교

부모: 완료된 `M-NATIVE-s1/ckpt_step10277.pth`, SHA256 `8bc43364c05c899a563438f34538845ed0c33a969ab7e0cb6786e407ed9cbadf`. 부모 plain V0 PREFIX는 **0.144597662**다. FULL 계보의 가중치/feature/teacher를 사용하지 않는다.

|GPU|Run|단일 변경|배포 graph|
|---|---|---|---|
|0|C-CTRL-s1|동일 모델 continuation|기존 M-NATIVE|
|1|C-R101-s1|R50 stage3에 residual block 17개 추가|3/4/23/3 block의 R101 깊이|
|2|C-DECSPLIT-s1|길이/방향의 query와 decoder까지 별도 학습|하나의 모델·하나의 길이/방향 누적 XY|
|3|C-AGENT-s1|shared scene의 agent 미래 변위 보조 감독|보조 head를 제거한 기존 M-NATIVE|

모든 arm은 같은 current/history image, native1152 motion 입력, H4 causal nominal status, scene query, motion/state/history, PREFIX+LEN0.25+기존 occupancy/lane/motion uncertainty objective를 유지한다. C-AGENT만 loss가 하나 추가된다. 각 treatment−CTRL과 각 treatment−고정 부모를 함께 보고한다.

## 변경의 의미

### C-R101

기존 stem과 stage1/2/4 및 stage3의 첫6block은 부모 tensor를 그대로 보존한다. 추가17block의 내부 가중치/BN 통계는 부모 stage3 마지막 block에서 복사하고 마지막 BN affine gamma/beta를0으로 만든다. 기존 block 출력은 ReLU 이후이므로 새 residual block이 초기에는 identity다. 학습 중 새 block과 기존 trunk 모두 학습한다. 공개 R101 checkpoint를 새로 받거나 더 강한 공개 사전학습이라고 주장하지 않는다.

기존 `model_config.backbone_arch=resnet50`은 부모 생성 계약이고 **실제 graph는 capacity_execution_config의 actual_backbone=resnet101_grown_3_4_23_3**로 명시한다. 전용 strict loader가 이 구조를 구성하며 signature로 다른 graph 로딩을 거부한다. 일반 R50 loader로 배포하지 않는다.

### C-DECSPLIT

기존 SplitRead의 length_read/heading_read와 head에 더해 query 및 Transformer decoder를 두 벌로 복사한다. 각 경로가 같은 영상 기반 scene/motion/predicted-state memory를 읽는다. 추가 raw status/goal/pose 입력이 없고, 두 checkpoint를 조합하는 모델이 아니다. 초기 FP32/BF16 계획이 부모와 같다.

### C-AGENT

Shared scene에서 12채널(6시점×XY)의 dense field를 예측하는 작은 학습 전용 head다. 네트워크 forward에는 GT center/trajectory를 전달하지 않는다. Loss에서만 실제 현재 agent center 위치의 예측 field를 bilinear sampling하여 미래 변위와 비교한다. 따라서 head의 입력은 다른 인지 head/planner가 사용하는 공통 scene뿐이다.

대상은 원본 train310의 Car/Pedestrian/Cyclist이며 동일 obj_id와 class의 +5/+10/+15/+20/+25/+30 frame 위치를 사용한다. **미래 world 위치−현재 world 위치**를 **현재 ego 회전**으로 변환한다. 미래 ego pose는 이 target을 만드는 데 필요하지 않다. 현재/미래 num_points>0, 현재 center가 scene grid 내부이며 camera-visible인 경우만 감독한다. 미래 ID가 없으면 해당 시점만 invalid다.

좌표는 float64로 변환한 뒤 float32로 저장한다. 128개 저장 슬롯 중 실제 최대64개라 객체 절단은0이다. Valid future point 수는 시점별625491/615987/606921/596700/584271/569396이며 5,526행은 유효 agent target이 없다. 이 행도 기존 planning/인지 학습에서 제외하지 않는다.

Loss는 displacement/10m의 SmoothL1(beta0.1), full effective batch의 valid agent-time-XY 개수로 정규화한다. 계수0.1, 처음500update 선형 ramp다. 기존 task/LEN 계수는 그대로다. Head는 배포 export에서 제거하고 student를 strict reload해 계획 출력 동등성을 확인한다. 이는 기존 lane 기하 target이나 GT ego 진행량 loss의 재명명이 아니다.

## 학습 예산

- 모든 arm **10,277 추가 update**, effective batch16, microbatch8, seed1.
- 동일 DEV train310/83,700행. 평가 V0는37scene/1,998행/11session.
- Fresh AdamW, weight decay0.01, warmup100, 고정 horizon cosine, fixed-BN running statistics, grad clip5.
- 기존 trunk peak LR1e-6, 기존 FPN/head 및 native branch1e-5.
- C-R101 추가 identity block LR1e-5. C-AGENT 신규 head LR5e-5.
- FP32 planning/geometry와 BF16 image path는 기존 정책 유지.
- 평가/저장: **0 / 3,426 / 6,852 / 10,277**. Terminal이 주 비교이며 중간 best 선택은 별도 표시.
- 고정 train probe256행을 기존 producer로 평가한다.
- 현 schedule을 중간에 바꾸지 않는다. 추가 budget 필요성은 terminal 뒤 곡선을 보고 대조군과 함께 검토한다. 자동 sweep/자동 FULL은 없다.

## 실제 사전 검증

네 모델 모두 세 실제 train 입력에서 FP32/BF16의 plan/scene/motion/state/history/인지 출력 차이가 부모 대비0이다. 전체 V0의 초기 PREFIX도0.144597664로 같다. Status/goal만 바꿨을 때 image-only motion/state/history의 차이는0이며 같은 final scene을 occupancy/lane/planner가 사용한다.

5update production trainer smoke에서 추가 block의 BN/conv, 별도 query, agent head의 앞/뒤 layer가 업데이트됐다. Sample 순서·기존 RGB 증강·native detail RGB의 hash가 네 arm에서 같다. 같은 smoke student를 새 프로세스로 strict 로딩한 출력 차이도0이다. Smoke weights/optimizer를 본 학습 초기값에 반입하지 않는다.

초기 C-AGENT 검사는 수치/전체V0 검증 및 결과 저장 뒤 process 종료에서 `__cxa_finalize/libnvrtc.so.13` SIGSEGV가 한 번 발생했다. 원인은 확정하지 않았다. 이후 별도5update 학습·strict reload·student export 검사는 모두 정상 종료했다. 오류를 숨기는 exit override를 넣지 않았고 `initial_teardown_incident.json`에 원문과 후속 증거를 보존했다.

Agent loss의 full/microbatch loss와 gradient 차이는0이고, all-invalid+NaN target도 유한한 graph-connected0이다. 추가 좌표/flip/target cache checks는 `agent_target_checks.json`, 4090 전체forward 비용은 `RTX4090_cost.json`에 기록한다. 명시한 검사들이 개별 모델에 대한 운영국 승인 인증을 뜻하지는 않는다.

## 결과/상태 위치

- `reports/a2_capacity_dynamics_20260922/results.csv`, `RESULTS_KO.md`, `result_step*.json`: 전체·시점·그룹·종횡·세션 비교.
- `reports/a2_capacity_dynamics_20260922/protocol_C-*.json`: 실제 고정 recipe/입력/초기값/source hash.
- `reports/a2_capacity_dynamics_20260922/runtime/main_orchestrator.json`: PID·GPU·실제 착수 시각.
- `work_dirs/a2_capacity_dynamics_20260922/C-*-s1/`: metrics·가중치·예측·train probe.

서버0.133684828과 이번 DEV를 환산하지 않는다. 최종 후보는 planning PREFIX를 우선하되 일반주행/정지/출발·종횡·인지·배포비용의 손익을 함께 판단한다.

## 4090 비용과 최종 코드 기록

동일 실제 DEV 입력 두 개, B1, 각 warmup30/repeat200에서 측정했다. 이미지 경로 BF16, planner FP32의 전체 model-forward이며 파일 읽기/전처리/CPU→GPU 전송은 제외했다. 공식 제출 adapter 전체 검수 완료를 뜻하지 않는다.

|Arm|FLOPs(G)|중앙값(ms)|최대 p95(ms)|
|---|---:|---:|---:|
|C-CTRL|1444.924|49.39|49.47|
|C-R101|2607.683|64.58|64.77|
|C-DECSPLIT|1445.376|50.17|50.22|
|C-AGENT, head 제거|1444.924|49.36|49.46|

4090 호스트의 오래된 CDI 라이브러리 경로 때문에 기본 Docker GPU 주입이 실패했다. 호스트 설정을 변경하지 않고 해당 컨테이너에 GPU device와 현재 driver library만 명시한 runc 실행으로 측정했다. 실행 환경은 RTX4090_execution.json에 보존한다.

Smoke 이후 agent의 로그 total에 weighted agent loss까지 표시하도록 한 줄을 수정했다. 실제 반환 loss/gradient는 수정 전후 정확히 같으며, CTRL/AGENT·ramp 시작/끝/terminal 조건을 확인했다. Smoke 당시 source SHA는 유지하고 최종 SHA와 변경 증거를 logging_only_amendment.json에 기록했다. 이전 smoke의 total은 common objective, 본 학습의 total은 실제 전체 objective다.
