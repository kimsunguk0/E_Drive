# 기존 A2 대비 H4-PROGRESS의 변경과 남은 병목

2026-09-21. 소스 progress_model.py, temporal_model.py, scene_extensions.py와 완료된 DEV result_step20554.json을 재확인했다. 이 문서의 성능값은 진행 중인 FULL의 점수가 아니다.

## 구조·학습 변경

|항목|기존 A2-DIRECT|선택한 A2-H4-PROGRESS|
|---|---|---|
|공통 구조|ResNet50/FPN, shared scene, motion encoder, direct planner|같은 계열 유지. 현재6camera·front 과거4시점 유지|
|초기화|기존 MR과 같은 r0_init_tplus(q10_flip50) 계보|공개 nuImages ResNet50 trunk, 나머지 새 초기값에서 공동 학습한 FRESH 계보|
|Scene 조회|한 번의 image-source attention|QREFINE: 첫 시각 read로 query를 갱신하고 두 번째 시각 read를 더함. Scene MH4/SIDE는 이번 graph에 없음|
|Motion 전달|4시점 pair feature를 시간 축에서 합쳐 192 token으로 planner에 전달|기존 pooled 경로 유지 + 합치기 전4×192 token을 각 출력 query가 별도로 attention read|
|출력|6×2 absolute XY 직접 회귀|6개 구간의 양수 길이와 방향을 NN으로 예측하고 누적해 absolute XY 구성|
|제공 status 시점|11pose시점 fit, nominal 재평가/학습 정책으로 개선해 온 계보|실제로 RGB를 소비하는 [-10,-5,-2,-1,0] 다섯 pose만 nominal fit. Train·DEV·raw parser 동일|

H4라는 이름이 과거 영상을 처음 추가했다는 뜻은 아니다. 과거4시점은 기존에도 있었고 새 temporal read는 합치기 전 시점별 정보를 planner가 읽게 한다. QREFINE의 scene attention과 temporal read의 4-head attention은 서로 다른 모듈이다.

Progress의 length = 5*softplus(raw_length+bias), heading = raw_heading이다. 모델이 예측한 구간 벡터 length*[cos(heading),sin(heading)]를 누적해 XY로 출력한다. 제공 vx/ax를 수식 적분하거나, GT가 고른 후보를 쓰거나, 기존 XY에 scalar residual을 붙이는 모델이 아니다. Heading은 제한하지 않아 출력 방향으로 후진·회전을 표현할 수 있으며, 길이는 양수지만 정확한0은 보장하지 않는다. Serving에서 다시 cumsum하면 안 된다.

제공 status는 shared scene query에만 사용한다. Goal은 기존 scene 조건, pose는 정렬 및 같은 시점 status producer에 사용한다. Motion feature와 예측 state/history의 forward에는 제공 status를 새로 넣지 않는다. 같은 최종 scene을 occupancy·lane·planner가 읽고 기존 auxiliary/PREFIX/LEN은 유지한다. 전체 계획은 scene을 통한 외부 조건의 간접 영향을 받는다. 이는 구현 설명이며 운영국 개별 승인 주장이 아니다.

## 점수가 낮아진 이유와 한계

기존 A2-DIRECT nominal 재평가0.164455314 → H4-PROGRESS0.151178860은 약8.07% 차이다. 이 비교에는 초기값·scene/temporal 조회·status 정책·출력 표현이 함께 달라졌으므로 한 변경의 효과로 분해할 수 없다.

마지막 출력 구조의 matched 비교는 같은 H4 입력·FRESH 초기값·TemporalRead·20,554 update를 쓴 다음 두 모델이다.

|그룹|행 수|H4-DIRECT|H4-PROGRESS|
|---|---:|---:|---:|
|전체|1,998|0.154213031|0.151178860|
|일반 주행|1,875|0.148017019|0.154890508|
|출발|24|0.473660201|0.358556517|
|정지 유지|99|0.194120005|0.030609127|

전체는1.97% 개선했지만 일반 주행은4.64% 악화했다. 출발·정지의 전체 기여 개선 -0.001383/-0.008102가 일반 주행의 +0.006450 악화를 상쇄한 결과다. Session paired95%CI는0 포함,5/11session 개선이다. 같은 재사용 DEV의 한 seed 결과다.

## 최대 병목

- H4-PROGRESS 전체 PREFIX0.151178860 중 일반 주행 기여0.145355206, **96.15%**. 출발·정지를 모두 완벽하게 고쳐도 이 가정에서는0.145355가 남아0.12가 되지 않는다.
- 전체 첫2초 포인트 기여0.107898569, **71.37%**. 일반 주행 자체의 첫2초 기여도 DIRECT 대비 악화했다. 전체 첫2초 개선을 일반 주행 개선으로 해석할 수 없다.
- 동일 GT tangent mask의 종방향 절대오차0.122230, 횡방향0.068343이다. 둘은 PREFIX에 더해지는 성분이 아니다. 종방향이 여전히 크지만 출력 변경에서 횡방향도0.059775→0.068343으로 악화했다.
- 일반 주행 첫2초의 공통 진행속도형 오차 MAE는0.091975→0.089525m/s로 소폭 감소했으나, 구간마다 변하는 성분 RMS는0.092476→0.104884m/s로 증가했다. 정확한 현재 v0 오차가 아니라 미래 계획의 구간 길이 오차를0.5초로 나눈 진단이다.

따라서 현재 남은 문제는 일반 주행의 미세한 진행량·시간별 배분·방향 정밀도다. 이전 CONT/TEMPORAL 종합 진단에서 큰 가감속이 없는 일반 주행도 오차의 큰 비중을 차지했다. 그 과거 집계 비율을 새 PROGRESS의 재계산값으로 부르지 않는다. '가감속 타이밍만의 문제'라는 단일 원인 주장은 자료가 뒷받침하지 않는다.

가장 유력한 설명은 시각 motion의 metric 거리/진행량 표현과 이를 미래 계획으로 옮기는 관계의 정밀도·일반화 부족이다. Image matching, depth/scale, temporal phase, decoder 중 어느 하나가 유일 원인인지 분리해 입증하지 못했다. 정답 좌표·배포 입력·loss 정규화의 재현 검사는 이미 수행됐으나 GT 센서/노출 동기화의 독립적 정확도를 인증한 것은 아니다. 0.15를 구조의 상한이나 노이즈 바닥으로 선언하지 않는다.

## RTX4090 접속 및 실측

사용자가 알려준 chi@192.168.10.102 접속 성공. RTX4090(driver595.91.07), md-v2:4090 Docker, PyTorch2.7.1+cu128 동작 확인. 이전 NVML 불일치는 /home/a의 별도 PC에서 발생했다.

선택된 DEV H4-PROGRESS 가중치, 실제 raw train fixture2개, B1 BF16(+FP32 planner), 각 warmup30/repeat200:

|Clip|전체 forward median|p95|
|---|---:|---:|
|fixture_000|26.087ms|26.152ms|
|fixture_001|26.058ms|26.113ms|

모든 current/history image encoder를 포함한다. JPEG/Parquet 읽기·전처리·host-to-device transfer는 이 forward timing에서 제외한다. 서버 elapsed_ms가 아니다. Raw 입력 tensor SHA는 B200과 모두 일치하며 FP32 예측 최대 좌표 차이는0.000001907m다.

이 실측은 FULL과 같은 inference graph의 DEV 가중치 결과다. 완료된 FULL terminal의 독립 재현·시간 검사는 Downloads 복사 이후 자동 실행하도록 연결한다. 기존 B200 학습 및 수집/복사 process는 그대로 둔다. 실행 중인 다른 4090 job이 있으면 중단하지 않고 대기한다. 새 학습·추가 제출은 없다.

근거: RTX4090_preflight.json, RTX4090_DEV_fp32.json, RTX4090_DEV_bf16_timing.json, ../a2_progress_h4_20260920/TERMINAL_REVIEW_KO.md 및 result_step20554.json.
