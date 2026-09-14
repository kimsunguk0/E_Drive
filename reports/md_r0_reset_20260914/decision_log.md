# 의사결정 로그 — R0 재검증 라운드

## P0 — R0 고정 (2026-09-14)

1. **무엇을 한 가지 바꿨는가**: 아무것도 학습하지 않았다. R0의 실체를 실측 해시로 고정했다.
2. **무엇을 유지했는가**: `q10_q10_flip50_s0/last.pth`와 git `ef49bad`의 31개 pinned 소스 파일.
3. **실제 값**: checkpoint `2e2fca25…`, 모델상태 `af609201…`, 파라미터 26,536,197개(버퍼 포함,
   trainable 26,426,936). backbone은 **resnet50**. pinned 31개 파일 **전부 변경 없음**.
4. **확인된 것**: A2 provided-status 경로가 체크포인트에 **존재하지 않는다**(marker 일치 0개).
   **확인되지 않은 것**: 서버 제출 이력 — `server_submission_id: null`, `score_type: UNKNOWN`.
5. **다음**: P1 평가 연결.

## P1 — train / tune / raw-B1 연결

1. **바꾼 것**: 없음. 같은 체크포인트를 5가지 조건으로 평가했다.
2. **유지한 것**: trainer의 `--eval-only` 경로, nominal 시간, fixed BN, 증강 OFF.
3. **실제 값**:

   | 평가 | 행 | D3 |
   |---|---:|---:|
   | `train_full` (증강 OFF) | 54,810 | **0.104942** |
   | `train_probe` (48/세션) | 3,456 | 0.108241 |
   | `tune_legacy` (B4) | 1,998 | **0.258544** |
   | `tune_legacy_repeat` | 1,998 | 동일 (차이 정확히 0) |
   | `tune_B1` | 1,998 | 0.258544 (차이 3.0e-9) |
   | `raw_B1_fixture` | 8 clip | 입력·출력 모두 **bitwise 동일** |

4. **확인된 것**:
   - 보고값 0.25854426847343837이 **비트 단위로 재현**된다.
   - **반복 실행 차이 = 0**. B1/B4 차이 = 3e-9. 따라서 이후 실험에서 3e-9를 넘는 차이는 실재하는 차이다.
   - 로드맵 §4.4의 두 정의가 충돌하지 않는다: `mean(ADE1,ADE2,ADE3)`와 `[11,11,5,5,2,2]/36` 가중합은
     **행 단위로 동일**하다(최대 차이 float32 수준). 안내서의 단순평균 ADE6은 별도로 저장했다(0.281).
   - 원본 JPEG/parquet에서 구성한 배포 어댑터 입력이 학습 경로 입력과 bitwise 같고,
     **모델 출력 궤적도 8개 clip 전부 정확히 같다**(최대 XY 차 0.0 m). 기존 감사가 남겨둔
     `model_forward_performed: false` 공백을 닫았다.
   - train_probe는 train_full보다 0.0033 높다. 균등 추출이 약간 비관적이지만 곡선 대용으로 쓸 수 있다.
   - **train/tune 비 2.46** (0.1049 → 0.2585).
5. **확인되지 않은 것**: 서버 점수, RTX4090 시간·연산량, 공식 test clip(이 노드에 없음).
   제출물은 `DEPLOYMENT_PENDING`.
6. **다음**: P2 계약검사.

## P2 — 공통 trainer 계약검사 (9종 전부 통과)

1. **바꾼 것**: 없음. 기존 `compute_loss` / `masked_mean` / optimizer 규칙을 그대로 호출해 검사했다.
2. **실제 값**:
   - **microbatch 동치**: B16 dense 대 micro 1/2/8/16에서 총손실·각 항·gradient 최대 차이 **2.5e-7**
     (float32 수준). B15 홀수, 완전 무효 행 포함, horizon 부분 무효 조건 모두 포함.
   - **FP64 reduction 대수**: 정확히 일치. 완전히 마스크된 microbatch의 기여는 **정확히 0**.
   - **negative control**: 신규 loss가 하던 방식(microbatch 자체 분모)으로 계산하면 8청크에서
     **8.16배** 누적된다. DYN/M8 실패 모드를 회귀검사로 보존했다.
   - **optimizer**: trainable 텐서 298개가 정확히 두 그룹에 한 번씩. backbone 159개(23,508,032원소,
     5e-6) / head 139개(2,918,904원소, 5e-5).
     **FPN의 lateral·output conv 5개 텐서는 backbone이 아니라 head 그룹(5e-5)이다.** 로드맵 §7.2가
     기록하라고 한 항목이며, "backbone LR"이라는 표현이 FPN을 포함하지 않는다.
   - **BN**: 53개 모듈 전부 eval 고정, affine 가중치는 학습 대상 유지.
   - **지표 항등식**: pred=GT → 0, 전 시점 (1,0) 오차 → 1, 등속 1 m/s 오차 → 1.25.
     `weighted_d3`는 내부에서 float32로 캐스팅하므로 항등식은 float32 정밀도로 성립한다(기록함).
   - **입력 whitelist**: 모델 forward 시그니처가 6개 입력과 정확히 일치. GT·status 누출 없음.
   - **FP32 투영**: bf16 autocast 아래에서도 planner 출력 dtype이 float32.
   - **증강**: flip 2회 적용이 항등(최대 차 0.0), trainer가 epoch마다 `set_epoch` 호출.
3. **다음**: P3 분할.

## P3 — H / Tplus 고정

1. **바꾼 것**: reserve 136 scene을 성능을 보지 않고 H와 A로 나눴다.
   규칙 `sha256("edrive-r0-holdout-v1|" + session_id)` 오름차순 첫 10 세션.
2. **유지한 것**: tune37 **완전히 동일**(새 manifest의 tune 목록과 tune rows_sha256이 R0와 일치).
   T203 arm의 train rows_sha256도 R0와 **동일**(`75ebfad6…`) — 대조군이 같은 행 집합을 쓴다는 증거.
3. **실제 값**:
   - 조상 계보 7단계 전부 train203만 학습했고 뿌리는 공개 nuImages backbone이다
     → reserve 31 세션 **전부 적격**.
   - H: 10 세션 / 29 scene / 7,830행. A: 21 세션 / 107 scene / 28,890행.
   - Tplus: 310 scene / **83,700행** = 기존의 1.527배.
   - supervision: geometry_v2 판본은 rawtime 판본과 **전역 calibration.npz 하나만** 다르다.
     그 보정 calibration으로 A·H scene을 새로 만들고, 기존 240 scene의 .npz는 하드링크로 그대로 두었다.
     기존 scene을 같은 입력으로 다시 만들면 **배열이 bitwise 재현**된다(패리티 검사 통과).
   - 초기화: 가중치를 바꾸지 않고 확대 split을 선언하는 artifact를 따로 만들었다.
     trainer의 split 계보 검사는 **삭제하지 않았다**.
4. **확인된 것**: 확대 split이 train을 추가만 하고 tune을 건드리지 않으며 val이 기존 val의 부분집합이라는
   envelope 11개 항목 전부 통과.
   **확인되지 않은 것**: 시간 분리가 공간·경로 분리를 뜻하지는 않는다.
   `historical_val38` 중 29 scene이 A로 들어가 학습된다 — 그 집합은 과거 계열에서 반복 평가된 이력이 있다.
   따라서 H는 blind set이 아니라 조상-학습-제외 확인 집합이다.
5. **다음**: E1 두 arm.

## E1 — 기존 train 대 확대 train (진행 중)

1. **바꾸는 것 한 가지**: 학습 데이터 풀만. 모델·손실·optimizer·schedule·증강·seed·평가집합·계산량 동일.
2. **예산**: 20,554 update, effective batch 16, warmup 200 + cosine, 3,426마다 평가·저장.
   T203는 자기 데이터를 6.00회, EXP는 3.93회 본다.
3. **기동 진단**(§5.3): 두 arm 모두 loss 유한, LR이 계획대로 상승(step 150에서 4e-5),
   grad_norm T203 11.9–54.6 / EXP 13.5–60.2, step당 0.79초, peak 5.36 GB.
4. **아직 없는 것**: 결과.
5. **다음**: 계획된 6개 평가 시점의 곡선을 받고, 그 다음에야 E2(LR) 또는 E3(command) 중 **하나만** 고른다.

## E1 — 결과 (2026-09-14, 완료)

1. **바꾼 것 한 가지**: 학습 scene 풀만. 모델·손실·optimizer·schedule·증강·seed·평가집합·update 수 동일.
   T203의 train row SHA는 R0와 동일, tune row SHA는 세 모델 모두 동일.
2. **유지한 것**: 같은 초기화 artifact(R0 가중치 bitwise 동일), 20,554 update, batch 16,
   backbone 5e-6 / head 5e-5, warmup 200 + cosine, BN fixed, flip 0.5, command OFF, 제공 status 없음.
3. **실제 값**:
   - V0: R0 0.258544 / T203 **0.267920** / EXP **0.225592** (B1도 1e-8 이내 동일)
   - 쌍체 CI: R0→EXP **−0.032953** [−0.0592, −0.0169], 10/11 세션 /
     T203→EXP **−0.042329** [−0.0645, −0.0294], **11/11 세션** / R0→T203 +0.009376, CI가 0 포함
   - train probe(동일 3,456행): R0 0.1082 / **T203 0.0735** / EXP 0.0859
   - vx MAE: R0 1.000 / T203 1.096 / **EXP 0.873**
   - forward: **clip당 1회**, **782.2 GFLOPs**(컷오프 7,053 대비 9.0배 여유)
4. **확인된 것**:
   - **확대 train만 운영 기준(Δ ≤ −0.010 그리고 CI 상한 < 0)을 넘었다.** `PRIORITY_CANDIDATE`.
   - **기존 203 scene에 계산량을 더 붓는 것은 V0를 개선하지 않는다.** T203는 train 적합이
     세 모델 중 가장 좋으면서 V0는 가장 나쁘다 — 정체의 원인이 용량·최적화가 아니라
     학습 분포 다양성이었다는 직접 증거다.
   - state head가 데이터만으로 1.10 → 0.87로 내려갔다. "표현 한계"라는 이전 결론은
     "이 데이터에서의 표현 한계"로 좁혀야 한다.
   **확인되지 않은 것**: H는 열지 않았다. V0는 반복 사용된 개발 집합이다. 단일 seed다.
   EXP 곡선은 끝점에서도 하강 중이므로 20,554 update가 EXP의 수렴점이라는 증거는 없다.
5. **다음 한 가지**: seed 1 복제 **쌍**(두 arm 모두)이 실행 중이다. 그 결과 전에는
   E2/E3로 넘어가지 않고, 구조 확대도 하지 않는다. E1-EXP-LONG(31,388 update = EXP 자기
   데이터 6회 노출)은 조건에 해당하지만 자동 실행하지 않고 승인 대기다.

## 작업 A — seed 1 복제 (2026-09-15, 완료): `REPLICATED`

1. **바꾼 것 한 가지**: shuffle·증강 seed만. 기준 arm과 변경 arm을 **쌍으로** 재실행했다.
2. **유지한 것**: 같은 R0 초기 가중치, 같은 V0 row(네 run과 R0 모두 식별자 일치), 같은
   loss·optimizer·update 예산·precision·증강 정책.
3. **실제 값**: EXP−T203 terminal이 seed 0 **−0.042329**, seed 1 **−0.041095**.
   양쪽 다 **11/11 세션 개선**. pooled seed 평균 −0.041712, CI [−0.06324, −0.02882].
   EXP terminal 자체가 0.225592 / 0.225622로 0.00003 차이.
4. **확인된 것**: §2.3 조건 충족 — 두 seed에서 EXP가 낮고 한 세션만의 결과가 아니다.
   train probe 방향도 재현됐다(T203가 train을 더 맞히고 V0는 EXP가 낫다).
   **확인되지 않은 것**: V0는 반복 사용 집합이고, seed 두 개는 모집단 추정이 아니다. H 미개봉.
5. **다음**: 복제가 지지하므로 작업 C 착수.

## 작업 B — 제출 준비 (완료, 업로드 없음)

1. **바꾼 것**: 없음. EXP terminal을 그대로 공식 test 1,125 clip에 적용했다.
2. **유지한 것**: 배포용 GT-free 어댑터. tar를 풀지 않고 메모리에서 읽는다.
   `command.parquet`은 모든 clip에 있지만 읽지 않는다.
3. **실제 값**: 1,125/1,125 clip, 파일 SHA `4e73a341…`, 312 KB, 22분.
   재-cumsum 시 평균 끝점 22.58 m → 79.07 m(3.50배)로 검사가 잡는다.
   clip 간 상태 격리 차이 0.0 m.
4. **확인된 것**: 파일과 형식은 준비됐다. **확인되지 않은 것**: 마감 시각·잔여 제출 횟수는
   계정 화면 미확인(안내서 문서값 9/23, 5회). RTX4090 시간 없음.
5. **다음**: 승인 시 업로드. 승인 전 quota 소비 없음.

## 작업 C — EXP-LONG (진행 중)

1. **바꾸는 것 한 가지**: 학습 예산과 cosine horizon. 구조·loss·LR·입력·증강은 E1 그대로.
2. **예산**: `ceil(6 × 83,700 / 16)` = **31,388 update**. 확대 데이터 기준 6.0001회 노출로,
   T203의 6.000회와 같은 epoch 조건이다. terminal에 step만 덧붙인 재개가 **아니라**
   처음부터 31,388 horizon의 cosine으로 새로 시작한 run이다.
3. **따라서 short/long 비교는 "학습 시간과 schedule horizon을 함께 늘린 recipe 비교"**이며
   노출 횟수 하나의 인과 효과로 읽지 않는다.
4. **아직 없는 것**: 결과.
5. **다음**: 계획된 평가 시점(3,426 배수 + terminal 31,388)의 V0·probe 곡선을 받고
   §4.5 표로 판정한다. LR×2·command·새 supervision은 같이 넣지 않는다.
