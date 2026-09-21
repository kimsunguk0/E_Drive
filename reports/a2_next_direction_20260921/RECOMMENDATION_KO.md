# 0.133685 이후 다음 공격안 검토 — 2026-09-21

상태: 최신 코드·완료 기록 검토, 저장 DEV 예측 재계산, 고정 가중치 sampling 정밀도 평가만 수행했다. 신규 optimizer update, 새 본 학습 예약, 공식 제출은 하지 않았다.

## 현재 판단

이번 CTRL/VECTOR/FINE/SHARED768은 같은 DEV 부모의 3,426-update continuation이다. 전체 DEV 데이터 약 0.655회 노출, backbone LR 1e-6 / head LR 1e-5였다. VECTOR/FINE의 추가 효과는 matched CTRL보다 좋지 않았다. SHARED768도 이 적응 조건에서는 악화했다. 이를 모든 예산·초기값에서의 구조적 한계로 일반화하지 않는다.

CTRL의 DEV 0.151178860 -> 0.150285502를 근거로, 별도 FULL terminal에 같은 레시피의 4,156-update stage2를 적용했다. DEV를 FULL weights로 검증한 것은 아니다. FULL stage2의 서버 성능은 아직 미측정이다. 서버 0.13 초반 또는 0.12를 예측할 근거로 사용하지 않는다.

공식 0.1336848279459137에서 0.12에는 약 10.24%, 0.11에는 약 17.72%의 감소가 필요하다. 현재 continuation의 DEV 개선은 약 0.59%다. 더 큰 변화를 검증할 필요가 있다.

## 기존 기록에서 유지되는 근거

- 공식 첫 2초 포인트는 PREFIX의 70.19%다. 뒤쪽 두 포인트는 29.81%다.
- FULL in-fit V0에서는 일반 주행이 98.15%를 기여한다. 미래 첫 2초 진행속도 변화 범위가 0.5m/s 이하인 일반 주행도 전체 점수의 62.51%다. 감가속 타이밍만을 유일 원인으로 삼을 수 없다.
- 전체 횡방향 절대오차의 약 76%가 표본이 많은 5도 미만 굽음의 행에서 나온다. FULL과 DEV에서 비슷한 관찰이다. 의미 좌/우회전만의 문제가 아니다.
- PROGRESS의 정지/출발 이득과 DIRECT의 일반 주행/방향 이득은 공존한다. 두 모델 차이만으로 공유 decoder의 gradient 충돌을 입증하지는 못한다.
- 현재 모델은 현재 객체 footprint와 raster lane을 학습한다. 주변 agent 미래 궤적 loss는 없다. 기존 agent cache는 옛 330-scene 계보의 99,000행, 최대40객체, FP16 좌표여서 현재 DEV/FULL 정밀 감독으로 그대로 재사용하지 않는다.

## 이번 새 진단

모든 저장 예측 비교는 같은 DEV 1,998행/11세션, row/GT 일치와 SHA 확인 후 수행했다.

|구성|DEV PREFIX|해석|
|---|---:|---|
|CTRL 실제 terminal|0.150285502|현재 추가 학습 대조|
|CTRL 예측 길이 + DIRECT 예측 방향|0.144266220|GT 선택 없는 두 저장 예측 재조합. 단일 학습 모델 결과 아님|
|GT 길이 + CTRL 예측 방향|0.065358001|진단용 GT 성분 치환. 실현 가능한 모델이나 최적 oracle 아님|
|CTRL 예측 길이 + 유효구간 GT 방향|0.117381183|GT 구간길이>0.05m에서만 방향 치환. 진단용|

진행량 정밀도의 개선 여지가 여전히 크고 방향도 무시할 수 없다는 관찰이다. 두 GT 치환 이득은 가산적 원인 분해가 아니며, 영상으로 해당 성분을 얼마나 예측할 수 있는지 알려주지 않는다.

### 저비용 가설 하나는 실제로 확인 후 후순위로 내림

Scene sampler가 FP32 투영 grid를 BF16 feature dtype으로 바꾼다는 코드를 확인했다. GPU1에서 CTRL 가중치를 고정하고, 나머지는 동일하게 둔 채 sampling만 FP32로 계산한 후 원래 output dtype으로 돌리는 비교를 했다.

- 같은 프로세스 baseline PREFIX: 0.150285498.
- FP32 sampling PREFIX: 0.150304091.
- 차이: +0.000018593, session paired95%CI [-0.000010715, +0.000044600].
- Motion/state/history 출력은 첫 V0 배치의 두 조건에서 동일했다. Parameter/buffer hash는 평가 전후 동일, optimizer update0.
- 첫 실행의 saved-run max-XY parity 문턱1e-5m는 0.000112534m 차이로 실패했다. 이를 숨기지 않고 재실행에서 기존 cross-process/device의1mm 문턱과 실제 재현 차이를 기록했다. 새 비교의 기준은 같은 프로세스 baseline이며 saved-run bit-exact 재현으로 설명하지 않는다.

추론 sampling 정밀도만 바꾸는 것은 큰 개선책의 근거가 없다. FP32로 처음부터 학습하는 모든 설정까지 기각하는 결과는 아니지만 지금 우선 투자하지 않는다.

## 권고 1 — 길이와 방향의 query/decoder 학습 경로 분리

하나의 공유 image backbone, 공통 scene, raw temporal motion을 유지한다. 기존 여섯 decoded query에서 길이/방향을 함께 출력하는 대신 길이와 방향이 각각 query/decoder/temporal read를 가지게 한다. 마지막에는 하나의 progress trajectory만 생성한다. 독립 checkpoint 앙상블 또는 sample별 GT selector를 제출하는 방식이 아니다.

근거는 예측 성분 재조합의 0.144266이다. 단일 모델에서도 그 수치가 재현된다고 주장하지 않는다. VECTOR는 목적함수를 바꿨고 FINE은 같은 motion map을 추가 조회했다. 두 방법은 길이/방향의 표현·최적화 경로를 분리하는 실험을 대신하지 않는다.

이번에는 0.655회 저LR 적응만으로 최종 판단하지 않는다. 공개 upstream에서 부모와 동일한 DEV 전체 학습 예산 20,554 update를 기본으로 삼고, 초기화·RNG·sample stream·기존 loss를 대조한다. 조건이 정확히 맞으면 기존 H4-PROGRESS control을 재사용한다. 새로운 총 예산이나 LR를 택하면 새로운 matched control도 필요하다.

기존 공통 scene을 인지 head와 planner 양쪽에서 읽는다. Raw status/pose/goal 사용 위치는 A2 그대로다. 분리 자체가 정보나 성능을 보장하지 않으며, 서로 다른 모델의 보완성이 공유 decoder 충돌의 인과 증거는 아니다.

## 권고 2 — 원본 RGB의 전방 현재/H4를1152로 학습

주력 관측 변경은 motion 전방 현재+H4 다섯 장을 원본 JPEG에서 동일 undistort/crop/FOV로1152x648로 생성하는 것이다. 현재 scene6-camera768 및 과거 scene384는 우선 유지한다. 768 cache를 보간해 확대하는 방식은 새 정보 비교가 아니다.

FINE은 기존768 영상에서 이미 계산된 특징의 읽기를 바꿨다. SHARED768은 이미 계산한768 과거 특징을 scene에 재사용했다. 이번 안은 영상에 남아 있던 실제 세부 정보가 motion에 들어간다. 과거 MR native-detail 비교와 같은 종류의 정보 검증이지만, 이번 PROGRESS에1152 입력을 학습한 결과는 아직 없다.

기존 비용 시제품: 전체 forward 약41.29ms,1,136.4G FLOPs. 이는 정확도 결과나 최종 새 가중치의 제출 비용 인증이 아니다. 같은 물리 범위의 대응을 보려면1152에 맞는 correlation radius/좌표계를 확인한다. Radius 변경에 따른 fuse 초기화도 실험 변경에 포함된다.

가장 깔끔한 대조는 같은1152 canvas/radius/초기화에서 768로 먼저 축소했다가 확대한 영상과 원본1152 영상을 비교하는 것이다. 단순 기존768 모델 비교만 하면 해상도·matching 구조가 함께 바뀐 실용 비교라고 명시한다. 원본 캐시 생성 시간과 디스크 용량은 시작 전에 확인해야 하며, 시제품 latency만으로 전체 작업시간을 약속하지 않는다.

여기도 공개 초기값부터 기존 전체 공동 학습 예산을 적용하는 것이 기본이다. 원본 terminal에 극히 작은 LR만 적용한 적응 결과로 표현의 효용을 끝내지 않는다. 현재6-camera1152는 그다음 scene 관측 후보이며, 처음부터 둘을 합치지 않는다.

## 권고 3 — 공통 장면의 감독을 실제로 더 풍부하게 만들기

이것은 더 큰 설계 변경 후보이고, 첫 두 실험과 한 run에 합치지 않는다.

우선순위는 일반 주행에 넓게 적용되는 연속 차선 기하다. 현재64x48의1채널 raster만 감독하는 것에서, 같은 공통 scene에 가까운 차선까지의 sub-cell 상대 위치와 선의 방향을 학습시키는 경로를 검토한다. 현재 ego 기준으로 만든 map geometry는 loss 정답으로만 쓴다. 교차로/여러 선의 대응과 양방향 선의 방향 표현을 명시해야 하며, 주행 경로를 정답 차선에 후처리로 붙이지 않는다.

주변 agent 미래 궤적은 다음 새 동적 감독 후보다. 기존 footprint를 넘어 현재 ego 좌표계에서의 미래 이동/위치를 shared scene이 예측하도록 학습시키는 방향이다. Obj_id/time/validity를 현재 split과 대조하고, 미래 ego 좌표계에서의 상대 운동을 현재 ego 기준 이동과 혼동하지 않는다. Future GT는 loss에만 사용하고 추론에서는 예측 visual feature를 쓴다.

SparseDrive와 UniAD의 detection/mapping/motion을 planning에 연결하는 설계는 이 방향의 문헌 근거다. 그 논문의 L2를 이 대회의 기대 점수로 바꾸지 않는다. 현재의 평범한 주행 오차가 주로 주변 차량 때문이라는 증거도 아직 없어 agent task를 곧바로 최우선 해결책으로 단정하지 않는다.

참고: https://arxiv.org/abs/2405.19620 , https://arxiv.org/abs/2212.10156

## 실행 순서와 판단

1. 분리 decoder와 원본1152 motion을 독립 주력 후보로 준비한다. 추론모델은 단일 모델/가중치 하나, A2 경계 유지.
2. 비교에 맞는 control을 확보하고 기존 성공한 전체 공동 학습 예산으로 판단한다. 개발 V0는 계속 같은 고정 지표로 사용하며 반복 개발 편향을 기록한다.
3. 일반 주행 첫2초, 길이·방향, 최종 PREFIX를 함께 본다. 보조 loss 하락이나 GT 치환 수치로 후보 승격하지 않는다.
4. 실제 DEV 이득이 확인된 변경만 결합을 검토하고 FULL에 이전한다. 10~18% 개선을 예측하거나 DEV-서버 고정 차이로 환산하지 않는다.
5. VECTOR 계수 스윕, FINE head/token 증설, SHARED768 무계획 연장을 자동 시작하지 않는다. 새 감독은 준비와 유효성에 따라 한 종류씩 검증한다.

현재 가장 직접적인 근거는 길이/방향의 모델 간 보완성이고, 더 큰 정보 증가 후보는 원본 영상 해상도다. 두 가설 모두 결과를 만들어야 하며 0.11~0.12를 보장하는 방법을 발견한 것은 아니다.

## 근거 파일

- reports/a2_progress_fourarm_20260921/result_step3426.json
- reports/a2_progress_fourarm_20260921/TERMINAL_LENGTH_HEADING_REVIEW.json
- reports/a2_after_submission_20260921/RESULTS_KO.md
- reports/a2_full_error_audit_20260921/RESULTS_KO.md
- reports/a2_next_direction_20260921/COMPONENT_RECOMPOSITION.json
- reports/a2_next_direction_20260921/FROZEN_FP32_SAMPLING.json
- experiments/a2_next_direction_20260921/frozen_sampling_review.py

새 진단의 재현 오차와 처음 실패한 parity 판정은 FROZEN_FP32_SAMPLING.json에 함께 기록했다.
