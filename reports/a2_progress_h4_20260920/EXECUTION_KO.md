# H4 status 정합 + 구간 진행량/방향 학습

2026-09-20. 사용자 “방법 찾아서 해보자”에 따라 두 모델을 같은 조건으로 비교한다. 이 문서는 실행 명세이며 본 학습 점수나 0.12 도달을 주장하지 않는다. 실제 진행/종료는 `results.json`, `RESULTS_KO.md`와 launch receipt를 따른다.

## 지금 이 실험을 하는 근거

직전 종합 진단에서 일반 주행이 기존 평균 예측 오차의 약94%, 첫2초가 약74%였다. 첫2초 속도 변화가 작은 일반 주행도 전체 오차의60.8%를 차지했다. 남은 진행량 오차를 가감속 타이밍 하나로 설명할 수 없다. 새 TemporalRead는 실제로 시각 motion을 사용하고 횡방향 오차를 낮췄으나 종방향 절대오차는0.88% 늘었다.

따라서 같은 시각 motion read를 유지하면서 **여섯 구간의 진행량과 방향을 직접 학습하는 출력 구조**를 한 번 비교한다. 새로운 관측 정보를 더하는 실험은 아니다. 영상으로 수정 부호를 구분할 수 없다는 문제가 남으면 이 구조도 실패할 수 있다. 표현 구조와 최적화가 병목이라는 가설을 검증한다.

과거 P×V bank/selector, base plan의 scalar residual, raw provided status를 motion/state에 전달한 A3와 다르다. 이번 출력은 연속값이며 후보 선택·GT tangent·base trajectory 보정이 없다. 이전 A3에서 짧게 중단한 factorized 실험을 이 경계의 완료 대조로 재사용하지 않는다.

## 두 arm

|항목|A2-H4-DIRECT|A2-H4-PROGRESS|
|---|---|---|
|공통 구조|FRESH+QREFINE+TemporalRead|동일|
|출력|6개 absolute XY|6개 양의 구간 길이와 heading → absolute XY|
|출력 숫자/파라미터 수|12 / 26,652,536|동일|
|Status 입력|실제 RGB가 소비되는 -10,-5,-2,-1,0 pose만 사용|동일|
|초기값|공개 nuImages R50 trunk + 기존 FRESH random nontrunk|동일 tensor|
|추가 ETRI 부모 학습|없음|없음|
|주 비교|20,554 update terminal|동일 시점 DIRECT와 비교|
|배정 GPU|0|1|

기존 status는 연속11시점 pose를 사용했지만 실제 영상은5시점이었다. OPEN_ISSUE 질문2/4의 시점 조건에 대응하도록 새 producer를 분리했다. 새 producer는 각 frame index 차이에0.1초를 곱한 시간, 정확히5개 full pose의 quadratic fit, full rank/condition<100/위치 fit RMS<0.25m를 사용한다. 이전 최소6점/최대간격 guard를 그대로 적용한다고 주장하지 않는다. supervision의 실측 timestamp 정의는 바꾸지 않는다.

고정 checkpoint의 새 status 입력 평가(1,998행, 추가 optimizer update0):

|기존 모델|이전11시점 status|새5시점 status|
|---|---:|---:|
|CONT 선택step5710|0.153684060|0.153511500|
|TEMPORAL 고정terminal|0.154119411|0.153980107|
|QREFINE 고정terminal|0.164251767|0.164260786|

이 변경만으로 큰 개선은 없었으며 신규 학습의 입력 기준을 맞추는 목적이다. 기존 모델들은 이전 입력으로 학습했으므로 새 PROGRESS의 순수 대조군이 될 수 없다. 원본 가중치·기존 FULL 제출 ZIP은 덮어쓰지 않았다.

## 새 출력과 학습

모델이 시각 scene/motion을 읽은 query마다 두 raw 출력을 낸다.

`length_j = 5 * softplus(raw_length_j + log(expm1(0.25/5)))`

`heading_j = raw_heading_j`

`p_k = sum_{j<=k}(length_j * [cos(heading_j), sin(heading_j)])`

출력은 FP32로 합성한 absolute XY다. 제공 status의 수식 적분이 아니라 NN이 직접 예측한 미래 구간 길이·방향의 좌표 표현이다. Serving에서 다시 cumsum하면 안 된다. Heading 범위를 제한하지 않아 후진·곡선을 표현할 수 있다. Softplus는 작은 진행량에 접근하지만 정확한0 출력을 보장하지 않는다. 현재 정지라는 이유로 미래 경로를 강제로0으로 만드는 gate도 없다.

두 모델의 공통 초기 trainable tensor와 RNG stream은 같다. 그러나 출력 해석·물리 단위가 달라 **초기 궤적은 다르다**. Head 마지막 projection을 임의로0으로 만들지 않았고, 이 차이를 완전한 함수 parity라고 설명하지 않는다. 추가 loss나 velocity/heading GT를 넣지 않고 기존 최종 PREFIX, LEN0.25, occupancy/lane/motion auxiliary를 그대로 사용한다. 이 실험은 출력 구조와 그 최적화 특성을 함께 비교한다.

DEV train83,700 / V0 1,998행, seed1, batch16/micro8,20,554 update, backbone5e-6/head5e-5,200warmup cosine, AdamW weight decay0.01,clip5,fixedBN,BF16(backbone)/FP32(planner),flip0.5. 평가3426/6852/10278/13704/17130/20554. 두 arm sample stream을 로그 SHA로 비교한다. 중간 best는 재사용 V0 선택값으로 별도 기록하고 주 판정은 terminal이다.

## 정보 경계와 확인

Provided status는 기존 공통 scene query에만 조건으로 들어간다. Raw goal/status/pose를 새 planner token·value·gate로 전달하지 않는다. 원시 motion/state/history는 RGB+시간 경로를 유지하며, 최종 scene tensor를 occupancy/lane/planner가 공유한다. Scene에는 기존 goal/status의 간접 영향이 있으므로 전체 모델이 goal/status-independent라고 부르지 않는다. 이 경계와 새5시점 coverage 검사는 개별 구조의 운영국 승인을 대신하지 않는다.

실제 raw clip8개의 새 status와 캐시 값 차이0, fit pose시점과 소비 영상시점 일치, 사용하지 않는 과거/미래 pose를NaN으로 바꿔도 불변, 필수 pose 누락/중복 거부를 확인했다. 상태 입력을 바꿔도 motion/state/history는 고정되며 scene은 반응한다. 공통 초기 feature의 FP32/BF16 차이0, 길이/방향 채널과 영상 backbone까지 PREFIX gradient, 직선/곡선/반전/후진/정지 근접/출발, effective-batch loss, strict export를 검사했다. 검사 중 바꾼 가중치는 본 학습에 쓰지 않는다.

공식 방식 FlopCounterMode B1 전체 forward는 두 모델 모두730,044,861,120 FLOPs. B200 B1 중앙값 약19.5ms이며 RTX4090 측정값이 아니다. Counter가 세지 않는 소규모 elementwise 합성 연산까지 물리적으로0비용이라는 뜻도 아니다.

## 결과 판단과 재현

1. PROGRESS − 동일 step DIRECT: 진행량/방향 출력 구조의 실제 효과.
2. 새 모델 − 보존한 H4 입력 CONT 단일 후보0.153511500: 실용적인 후보 개선 여부. 학습량/계보 차이는 구분.
3. 첫2초·일반 주행·구간 진행량·종/횡·정지/출발·인지/state/history를 함께 기록. 특정 loss나 train fitting만으로 성공 판정하지 않음.

Source: `experiments/a2_progress_h4_20260920/`. 새 입력 캐시: `data/etri/motiondrive_v2/a2_h4_status_20260920/`(대형/데이터는Git제외). Run: `work_dirs/a2_progress_h4_20260920/`. Checkpoint/예측은 원격 보존, hash와 결과표를Git에 기록한다. `preflight.py` → `launch.py --smoke` 두 arm → `verify_smoke.py` → `launch.py` 두 arm. CPU `collect.py --publish`는 예정 평가를 수집하고 완료 결과를 공개 미러에 커밋·푸시하며 새 학습/FULL/제출은 시작하지 않는다.

기존0.146197790은 다른 입력 정책의 세 저장 예측 평균 DEV 결과다. 이번 단일 모델의 보장선이나 서버 환산식이 아니다. 현재 실측 공식 서버 기준은 MR-FULL0.185968928이며, 이번 변경의 서버 점수는 미측정이다.
