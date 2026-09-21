# A2-H4-PROGRESS: CONTROL / VECTOR / FINE / SHARED768 통합 실행 명세

작성: 2026-09-21  
상태: **실행 준비 명세. 이 문서 작성 과정에서 신규 학습·원격 실행·제출은 하지 않았다.**  
보존할 실제 서버 성능: **A2-H4-PROGRESS FULL, PREFIX 0.1336848279459137**.

## 0. 이번 결정 — 앞선 계획의 우선순위를 이 문서로 통합

**동일 DEV 부모에서 4-arm을 병렬 비교한다.**

| 논리 슬롯 | Run | 변경 하나 | 나머지 |
|---|---|---|---|
| 0 | P-CTRL | 같은 조건의 추가 학습 | 기존 LENGTH 0.25 |
| 1 | P-VECTOR | LENGTH를 VECTOR 0.25로 교체 | 기존 inference graph |
| 2 | P-FINE | 24×32 fine temporal memory read 추가 | 기존 LENGTH 0.25 |
| 3 | P-SHARED768 | motion용 raw 768 history FPN을 scene에도 공유 | 기존 LENGTH 0.25 |

물리 GPU 번호는 착수 시 사용 가능 여부를 확인해 배정한다. 이 문서는 GPU를 예약한 기록이 아니다.

- 네 모델은 **비교 후보 네 개**다. 최종 추론은 선택한 **단일 모델·단일 최종 가중치**만 사용한다.
- VECTOR/FINE/SHARED의 첫 결합 run, 독립 expert 앙상블, GATE3는 만들지 않는다.
- 현재 scene1152, 길이/방향 decoder 분리, 신규 history, recurrent BEV는 첫 파동에서 제외한다.
- 기존 FULL 0.13368483 checkpoint·제출 파일·코드·서버 결과를 불변 보존한다.
- 새로운 학습 결과는 아직 없다. CPU 계약검사와 비용 시제품의 PASS를 정확도 개선으로 바꾸어 쓰지 않는다.

**이번 우선순위 변경 이유:** 직전에는 SHARED768을 가장 먼저 두고 FINE을 뒤로 미뤘다. 최신 첨부에서는 FINE 시제품의 초기 parity와 작은 비용이 실제로 확인됐다고 보고한다. 따라서 SHARED를 버리는 것이 아니라, FINE의 구현 위험이 줄어든 근거를 반영해 세 축을 한 공통 CONTROL 옆에서 함께 시험한다.

## 1. 근거와 검증 범위

### 1.1 최신 자료가 보고한 것

사용자가 제공한 `PRO 제안 검토: interval VECTOR + FINE-READ — 2026-09-21`에 따르면:

- 핵심 source 4개 SHA가 기존 참조 ZIP과 B200 코드에서 일치.
- VECTOR를 실제 common loss/uncertainty 설정에 연결한 검사 통과.
- 검토용 FINE 시제품은 pre-pool map [4,128,54,96]에서 직접 24×32 memory를 구성.
- FINE 추가 파라미터 66,560개, 추가 backbone 0회.
- FINE 비용은 730.044861G → 730.256544G, 동일 조건 baseline 25.768/25.784ms → 26.161/26.144ms.
- FULL 가중치와 raw train fixture 두 개의 초기 FP32/BF16 출력 차이는 0.
- **Production factory·strict export를 갖춘 학습 모델로 완료한 상태는 아님. 학습 launcher도 아직 변경하지 않음.**
- 사용자가 밝힌 공개 미러 commit은 `9e2273ec`.

이는 **첨부 보고서가 보고한 결과**다. 이 문서 작성자가 최신 commit, FINE 시제품, 실제 GPU 검사를 독립 재실행한 것은 아니다.

### 1.2 이번에 직접 한 작업

- 최신 첨부 전문과 앞선 VECTOR/FINE 계획 및 SHARED 계획 대조.
- `review_73f9ea4.zip`의 DEV H4-PROGRESS manifest·experiment·run index·평가 결과 읽기.
- 보존된 PROGRESS/TemporalRead 소스 읽기.
- 추가 학습 노출량, FULL 이전 update, loss 규모 비율, 고정 방향에서의 최적 길이, 비용 증가량 재계산.
- 아래 실행 계약과 machine-readable **설정 명세** 작성. 실제 repository patch 또는 실행 가능한 launcher는 작성하지 않았다.

## 2. 최신 리뷰에서 수용할 해석 정정

### 2.1 일반 주행의 초기 구간도 개선 대상이다

전체 DEV의 0~1초 interval-vector 오차는 DIRECT 0.063260 → PROGRESS 0.060327m이나, **일반 주행만 보면 0.059676 → 0.061642m로 악화**했다.

따라서 ‘초기 일반 주행은 해결됐고 뒤만 고치면 된다’는 해석을 하지 않는다. 현재 모델이 얻은 **전체 초기 정확도와 정지/출발의 장점은 보존 대상으로 감시**하되, 일반 주행의 첫 구간도 개선 대상이다.

마지막 구간에서는 평균 각도 오차가 낮아졌어도 radial/angular 제곱오차 항이 둘 다 증가했다. 평균 angle 하나를 위치 횡오차나 방향 문제 전체의 대표값으로 사용하지 않는다. 이 항들은 interval-vector의 제곱오차 분해이지 PREFIX의 가산 분해가 아니다.

### 2.2 VECTOR는 방향 신호와 loss 배분을 함께 바꾼다

기존 DEV 출력에서 LENGTH 평균 0.075478m, VECTOR 평균 0.102670m, 비율 약 1.360이다. λ=0.25를 그대로 쓰는 비교는 타당하지만 **동일 loss 크기·동일 gradient budget의 순수 방향 대조가 아니다.**

첫 run에서 1.36으로 나눠 새 λ를 만들거나 V0 기준 λ 탐색을 하지 않는다. raw LENGTH와 VECTOR를 모두 모니터링할 수 있으나 backward에는 지정된 항 하나만 들어간다.

### 2.3 방향 오류가 남으면 VECTOR가 길이 축소를 선호할 수 있다

고정된 예측 방향과 GT 길이 L에서:

`min_{l>=0} ||l*u_pred - L*u_gt||`의 최적해는 `max(0, L*cos(delta_theta))`.

GT 길이 5m, 각도 오차 30°이면 4.330m, 45°이면 3.536m다. 이는 방향을 수정할 수 없는 조건의 수학적 절충이며, **실제 joint 학습이 반드시 그 해로 간다는 예측은 아니다.** 최종 위치 loss에도 관련 절충이 있을 수 있어 VECTOR만의 고유 결함이라고 하지 않는다.

따라서 signed interval-length error, 길이 MAE, 최종 PREFIX를 함께 본다. 사후에 길이를 늘리는 보정, GT heading 입력, 수동 curve scale은 추가하지 않는다.

## 3. 공통 부모·입력 계약

### 3.1 DEV 부모를 명확히 고정

역사적 원본 색인에서 확인한 부모:

- Run: `A2-H4-PROGRESS-s1`
- Checkpoint: `work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1/ckpt_step20554.pth`
- Checkpoint SHA256: `18128b1af6624333a00baa493a0601459804e8dfb660bf4dd7dbf60f381f2958`
- 학습 step: 20,554
- 저장된 DEV PREFIX: 약 **0.15117886**
- 학습: Tplus310, 83,700행
- V0: 1,998행, 11세션, 미학습 평가
- Status 정책: `five_RGB_consumed_pose_times_nominal_frame_dt`
- Train row SHA: `e20e4541834331a5932bdb66a2b03248ba2eb30ebd4ad685c2045ceb5b1a0471`
- V0 row SHA: `1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88`

위 SHA는 첨부 색인의 기록이다. 체크포인트 대형 파일은 이번 환경에 없었으므로 착수 노드에서 실제 bytes hash를 한 번 확인한다. **이것은 부모의 부모인 public initializer SHA와 다르다.**

FULL 가중치로 비용 시제품을 검사한 행위 자체와 DEV 학습 계보는 다르다. DEV는 새 모델 인스턴스를 만들고 위 DEV 부모만 로드한다. 시제품 객체에 남은 FULL weight·BN buffer·feature를 반입하지 않는다.

### 3.2 고정할 입력·출력

- 현재 6-camera 768×432.
- Front H4: -0.1/-0.2/-0.5/-1.0초.
- Motion 현재+H4: 768×432.
- H4 nominal status producer와 기존 state/history supervision은 그대로 유지.
- 제공 status/goal은 기존 A2 shared-scene 조건 경로에만 사용. 새 motion/read/planner 인자로 보내지 않음.
- Fine K/V와 shared history는 조건화 전 영상 feature에서 온다.
- Fine의 query는 기존 scene을 읽은 query이므로 전체 fine 출력이 goal-independent라고 설명하지 않는다.
- 기존 length/heading neural output과 FP32 cumulative absolute XY, [B,6,2] 유지.
- 제출 adapter에서 두 번째 cumsum 금지.
- 단일 학습 모델로 최종 궤적 하나. 독립 checkpoint를 묶는 제출 방식은 제외.

## 4. 실제 학습 설정 — 3,426으로 새로 통일

**명시적인 계획 변경:** 직전 두 실행안은 6,000 update, 신규 FINE read LR 5e-5를 제안했다. 최신 첨부의 3,426과 신규 read 1e-5는 같은 설정이 아니다. 이번 첫 스크린은 아래로 통일한다.

| 항목 | 네 arm 공통 |
|---|---|
| Stage-2 추가 optimizer update | **3,426** |
| Seed | **1** |
| Effective batch | **16** |
| Microbatch / eval batch | **8 / 8**, 부모와 동일. 변경 필요 시 공통 적용·기록 |
| 초기값 | 위 DEV terminal의 가중치만 로드 |
| Optimizer | **Fresh AdamW**, 부모 moment 미복원 |
| Trunk LR | **1e-6** |
| 기존 FPN·scene·motion·planner·head LR | **1e-5** |
| 신규 FINE read LR | **1e-5**, 이번에는 별도 5e-5 사용하지 않음 |
| Weight decay / grad clip | 0.01 / 5.0 |
| Warmup | **100 update** |
| Schedule | Stage 시작에서 설정한 **3,426-horizon cosine-to-zero** |
| BN / freeze 정책 | 기존 fixed BN, joint 학습. 새 freeze 정책 없음 |
| Precision | 부모와 동일: BF16 영상 경로, FP32 planner·출력·loss |
| 기존 task 가중치 | occ/lane/motion 0.2, uncertainty 설정 유지 |
| Augmentation | 기존 photometric·flip p=0.5 및 status flip 유지 |
| 평가·보존 시점 | Stage **0 / 1,142 / 2,284 / 3,426** |
| 주 비교 checkpoint | **Terminal 3,426** |

### 노출량과 계수 의미

`3,426 × 16 / 83,700 = 0.654910`회 노출이다. **Tplus 전체 1epoch가 아니다.** 이전 6,000은 약 1.147회였다.

따라서 ‘작은 LR로 약 0.655회 추가 노출하는 적응 스크린’으로 해석한다. 짧은 설정에서 개선이 없었다고 FINE·고해상도 특징의 원리적 효용을 기각하지 않는다. 반대로 자동으로 LR나 step 스윕을 재개하지도 않는다.

Stage step은 0부터 시작하며 parent step=20,554를 metadata로 따로 보존한다. 구형 resume 로직이 parent step을 그대로 가져와 3,426 update 루프를 건너뛰거나 LR=0을 승계하지 않게 한다. 필요하면 총 누적 step=20,554+stage step도 별도 기록한다.

### RNG·stream 통제

새 FINE 모듈 생성이 sample/augmentation RNG를 바꾸지 않게 분리한다. 같은 row hash는 같은 **순서**를 보장하지 않으므로 sample-order digest를 별도로 남긴다. 같은 sample에 대한 augmentation도 공유된 결정론적 정책으로 확인한다.

## 5. Arm별 구현 계약

### 5.1 CONTROL

기존 inference graph와 `common + 0.25*LENGTH` 그대로 추가 학습한다.

- Step0 plan은 DEV 부모와 일치해야 한다.
- 단순 continuation이 좋아지거나 나빠지는 기준이다.
- 세 변경이 CONTROL보다 좋아도 원래 부모보다 나쁘면 바로 승격하지 않는다.

### 5.2 VECTOR

`common + 0.25*LENGTH`를 **`common + 0.25*VECTOR`로 교체**한다.

`VECTOR = sum_complete_rows sum_6_intervals ||delta_pred-delta_gt|| / (6*C_full)`.

- Common은 기존 PREFIX 및 인지/state/history loss를 포함한다.
- 분모는 full effective batch의 `plan_complete`, 원점 p0=(0,0).
- 불완전 행·NaN GT는 norm 계산 전 기존 검증된 규칙으로 처리.
- `train_shared_dynamics.patched_runtime()`가 설치한 LENGTH wrapper 바깥에 VECTOR를 씌우지 않는다. **실제 실행 시점의 원본 common 함수에 지정 항 하나만 연결**한다.
- 같은 고정 출력에서 `total_vec-total_len = .25*(VECTOR-LENGTH)`이고 나머지 common parts는 같아야 한다.
- Step0 **prediction**은 CONTROL과 같고, **loss total**은 달라도 정상이다.
- 기존 통합 테스트를 사용한다. 새 unit-test 체계를 다시 만들지 않는다.

### 5.3 FINE-READ

흐름:

`correlation_fuse map [B*4,128,54,96]`

- 기존 경로: 12×16 pool → 기존 motion/state/history/coarse temporal read.
- 추가 경로: **원본 pre-pool map에서 직접 24×32 pool** → [B,4,768,128].
- 위치·네 시점 embedding → 3,072-token memory.
- 기존 coarse read 이후의 여섯 **구간 출력 query**가 추가 cross-attention.
- 기존 progress readout과 cumsum으로 최종 XY.

조건:

- 12×16 token을 확대해 fine이라고 부르지 않는다.
- Backbone 및 correlation_fuse를 다시 계산하지 않는다.
- 기존 config.motion_grid 전체를 24×32로 바꾸지 않는다.
- 같은 forward의 autograd-connected map을 사용한다. `.detach()`·CPU cache·persistent cross-clip memory 금지.
- New Q/K/V는 정상 초기화, **output projection만 0**.
- 첫 backward에서 내부 Q/K/V gradient가 0인 것은 이 초기화에서 가능하다. Output projection이 업데이트된 이후 Q/K/V 및 map으로 gradient가 이어지는지 검사한다.
- 같은 read dropout·dtype·위치/시간 정의를 시제품과 유지. 새 별도 normalization/geometry 변화를 섞지 않는다.
- 시제품은 추가 파라미터 66,560을 보고한다. Production에서 수가 달라지면 diff·이유를 기록한다.
- Forward hook을 쓴다면 capture가 재사용·누적되지 않도록 lifecycle을 명확히 한다. 가능하면 명시적 tensor 반환을 사용한다.
- 새 read를 학습한 뒤 ‘off’ 개입은 진단일 뿐, off 조건으로 다시 학습한 control을 대신하지 않는다.

### 5.4 SHARED-HISTORY768

- 현재 scene 6-camera768은 유지.
- 별도 scene용 현재front축소본+history384 encoder 호출을 제거.
- Motion encoder 이전의 **raw 768 FPN의 history slice**를 기존 scene encoder에 전달.
- Motion 입력·matching·time memory 자체는 기존대로 유지.
- Scene에서 raw shared tensor를 in-place 수정하거나 status-conditioned 값을 motion에 되돌리지 않는다.
- Shared history를 scene으로 보낼 때 detach/no_grad를 붙이지 않는다. Scene와 motion gradient가 공유 encoder로 합산돼야 한다.
- Crop/FOV·H4순서·projection/normalized-grid 기준을 유지한다. 새로운 sampling dtype 변경을 숨겨 넣지 않는다.
- 기존384 대비 step0 출력 차이는 허용된다. 비교의 실제 변경 자체다.
- 기대 parity는 **‘별도768 인코딩’ 대 ‘동일768 raw FPN 재사용’**이다. 이를 기존384 대비 parity와 혼동하지 않는다.
- 기존 가중치 key가 같아도 serving graph가 달라지므로 explicit configuration을 저장한다.

## 6. 본 학습 전에 남은 작업 — 기존 검사를 반복하는 것이 아니라 production 연결

첨부가 통과를 보고한 CPU 검사와 시제품 benchmark를 다시 모두 요청하지 않는다. 다음 연결만 마친다.

1. **하나의 arm-aware factory/loader/export**가 네 구성을 정확히 재구성하도록 만든다. Run 이름 substring으로 arm/pose 정책을 추측하지 않는다.
2. 공통 launcher가 각 arm의 올바른 auxiliary를 단 한 번 설치하고 optimizer 생성 전 모든 새 parameter를 등록하는지 확인한다.
3. DEV 부모를 로드한 각 실제 train forward/backward에서 5~20 update smoke. 이 짧은 smoke 뒤 원래 DEV 부모에서 새 본 run을 시작한다. Smoke update를 한 arm만 본 학습에 이어 쓰지 않는다.
4. 저장 → 새 프로세스 → explicit config 기반 strict load → 같은 fixture 출력 비교를 수행한다. 시험 당시 Python hook·monkeypatch만 남아 재로드 시 사라지지 않게 한다.

필수 명시 config 예:

- `architecture_id`: 기존 PROGRESS 또는 fine variant.
- `fine_read_enabled`, `fine_grid=[24,32]`, read 위치.
- `scene_history_feature_source`: separate_low384 / shared_raw_fpn768.
- `interval_auxiliary`: length / vector. 추론 무관해도 재현용 기록.
- `status_policy`, history offset, image sizes, precision.
- parent SHA, 실행 source 전체 SHA, 변경 파일 hash, dataset row hash.

잘못된 feature-source flag에서도 strict state load가 통과할 수 있으므로 config 검사도 fail-closed로 둔다. Prototype에서 FULL로 검사했다는 사실을 DEV 부모 parity의 완료로 대신하지 않는다.

## 7. 평가·판정 — 최종 PREFIX와 두 기준을 함께 사용

각 예정 시점마다 **동일한 1,998행**을 평가한다.

### 필수 두 대비

`Delta_ctrl = PREFIX_arm_terminal - PREFIX_control_terminal`

`Delta_parent = PREFIX_arm_terminal - PREFIX_frozen_DEV_parent`

첫째는 변경의 추가 이득, 둘째는 현재 후보를 실제로 교체할 가치다.

### 결과표

| 범주 | 보존할 항목 |
|---|---|
| 주 지표 | PREFIX, Delta_ctrl, Delta_parent |
| 시간별 | point L2 6개, L2_1s/2s/3s, 첫/중간/마지막 두 point의 PREFIX 기여 |
| 주행 조건 | 기존 producer의 nonstop/depart/steady, 행 수와 전체 기여 |
| 길이·방향 | interval vector MAE, length MAE, **signed length error 6개**, 동일 GT mask의 종/횡 위치 오차 |
| 집중도 | session별 Delta_ctrl와 전체 기여, 최대 영향 세션을 제외했을 때의 민감도 |
| 학습 상태 | 동일 train probe, LR, 주요 gradient/update, common loss/aux 항 |
| 기존 보조 task | occupancy/lane/state/history 지표 |

주행 조건과 moving mask는 모델별 prediction이 아니라 동일 GT/고정 producer로 고정한다. 예측 길이를 줄여 방향 통계에서 빠지는 행이 생기지 않도록 분모를 기록한다. 방향 정의가 어려운 예측은 삭제해 성능을 미화하지 않고 별도 count를 남긴다. **이런 행도 공식 PREFIX에는 모두 남긴다.**

VECTOR의 ‘회전에서 경로가 짧아짐’은 signed 길이 오차와 최종 XY로 읽는다. GT heading 차이가 줄었는지만 보고 판단하지 않는다. 일반 주행 첫 1초도 계속 기록한다.

### 판정 규칙

- Terminal을 주 비교로 사전 고정한다. 중간 best는 보존하고 별도 열에 보고하되 terminal인 것처럼 바꾸지 않는다.
- V0는 이미 반복 사용한 개발 집합이다. Session paired CI는 개발 진단이지 모집단/서버 승리 보장이 아니다.
- **CI가 0을 포함한다는 이유만으로 모든 작은 개선을 자동 폐기하지 않는다.** 효과 크기·세션 손익·경로 변화·배포 비용과 함께 채택 판단.
- **고정 -0.005 필터는 두지 않는다.** 현재 서버에서 .129까지 간격은 .004685이지만 DEV에서 그만큼 내려간다고 서버도 같은 만큼 내려가지는 않는다.
- Arm이 CONTROL보다 좋아도 부모보다 나쁘면 이번 첫 후보로는 유지하지 않는다.
- CONTROL이 가장 좋다면 continuation 레시피 자체도 유효한 후보로 인정한다. 새 이름의 변경을 강제로 고르지 않는다.
- 보조 loss만 감소하거나 train만 좋아지는 경우는 실제 planning 개선으로 채택하지 않는다.
- 세 변경이 서로 다른 조건에서 좋아지더라도 개선폭을 합산해 서버 점수를 예측하지 않는다.

### 3,426 이후를 자동 연장하지 않는다

이 짧은 run에서 신규 fine read가 충분히 적응했는지는 gradient/update와 계획된 곡선으로 해석한다. 내부 gradient가 이어지지 않는 연결 오류는 구현 문제이고, 적응은 됐으나 V0가 나쁜 것은 이 레시피의 결과다.

Terminal 이후 `--steps 6000`만 바꿔 cosine을 즉석 연장하지 않는다. 추가 비교가 필요하면 stage-3 restart인지 다른 총 horizon의 재실행인지 명확히 정하고, 같은 조건의 control을 둔다. 첫 결과 전에는 이를 예약하거나 자동 기동하지 않는다.

## 8. FULL 이전 — DEV 승자만, 원본은 그대로 보존

현재 FULL은 완료·제출된 성능 기준이다. 이를 DEV 초기값으로 사용하지 않는다.

DEV 결과에서 선택한 **하나의 변경 또는 CONTROL continuation 레시피**를 현재 제출 FULL terminal의 **복사본**에 적용하는 별도 stage2로 이전한다. 다른 입력을 사용하는 별도 학습 모델과 출력 평균하지 않는다.

같은 추가 노출량 기준:

`ceil(3426 * 101520 / 83700) = 4156 update`.

- 이전 6,000→7,278 이전값을 이번 3,426 실험에 사용하지 않는다.
- 실제 DEV/FULL row 수가 바뀌면 다시 계산한다.
- Full stage2의 optimizer는 fresh AdamW, 최대 LR·loss·graph는 선택한 DEV와 같다.
- Warmup은 100 update로 고정하고, 총 cosine horizon은 4,156으로 설정한다. 이식 방식은 새로운 FULL 최적 레시피라는 증명이 아니다.
- FULL의 옛 V0/H는 in-fit이다. Full checkpoint selection이나 일반화 성능 판정에 쓰지 않는다.
- Full stage2의 선택은 사전에 정한 terminal. 새 best 선택 규칙이 필요하면 실행 전에 DEV에서 결정하고 선택 사실을 명시한다.
- VECTOR는 학습 objective만 바뀌므로 graph 비용은 부모와 같지만 새 가중치·출력 재현은 확인한다.
- FINE/SHARED는 graph/config가 바뀌므로 최종 모델의 raw B1, 실제 전체 forward, FLOPs, export를 확인한다.
- 모든 후보의 최종 ZIP/Docker는 실제 config에서 단일 checkpoint로 복구 가능해야 한다.
- Test 1,125 clip 전체의 token·shape·finite·중복·상태 격리 및 `__flops__`를 기존 package 검사로 확인한다. 업로드는 사용자 결정과 실제 잔여 횟수를 따른다.

두 변경 결합은 실제 독립 이득이 확인된 뒤의 선택지다. 이 첫 라운드의 기본 산출물 또는 자동 후속 run으로 넣지 않는다. 전체 일정을 결합 검증 때문에 지연시키지 않는다.

## 9. 산출물 최소 목록

각 run별로 다음을 남긴다.

- `experiment.json` / `manifest.json`: parent/source/data/arm/config/seed 및 실제 LR 그룹.
- `metrics.jsonl`: stage step과 parent step 구분, loss·LR·finite·sample-order digest.
- 예정 시점 checkpoint 및 `eval_step*.json`: row key·pred/GT/valid를 포함한 실제 DEV 예측.
- `smoke_and_reload.json`: 실제 runtime wrapper·factory·export·초기화 검사.
- `decision.json`: terminal 두 대비, 세션/조건별 손익, 코드/출력 회귀, 후속 결정.
- `candidate_registry.json`: 최종 단계에서 선택한 모델/graph/weight/serving/결과 연결.

최신 첨부에서 이미 만들었다는 INDEPENDENT_INTERVALS / VECTOR_INTEGRATION / VECTOR_REFERENCE_REPLAY / FINE_READ_PAIRED_REVIEW / ANALYTIC_CHECKS / PROVENANCE는 원본 경로·SHA로 링크해 재사용한다. 동일 검사를 위한 새 장문 보고서부터 만들지 않는다.

## 10. 작업자에게 전달할 지시문

> 최신 검토 결과를 통합해 CONTROL/VECTOR/FINE/SHARED768 네 arm만 실행한다. 기준은 DEV A2-H4-PROGRESS-s1 step20554이며, 제출 FULL을 DEV에 가져오지 않는다. 학습은 동일 seed1, effective batch16, micro/eval batch8, fresh AdamW, trunk1e-6/나머지 및 신규FINE1e-5, warmup100, 3426-step cosine로 맞춘다. 이는 이전 6000-step 계획을 줄인 첫 적응 스크린이다.
>
> 검토용 FINE 시제품을 명시적인 model factory/export에 연결하고, VECTOR는 원본 common loss의 LENGTH를 대체하라. SHARED는 조건화 전768 history FPN을 detach 없이 scene에 전달하라. 같은 data/augmentation stream을 유지하고, 원래384와 공유768의 step0 차이는 legitimate 변경으로 기록하라. 새 프로세스 checkpoint 재로드와 짧은 실제 DEV smoke가 끝나면 추가 리뷰 대기 없이 본 학습을 시작한다.
>
> 평가 시점은0/1142/2284/3426, 주 판정은 terminal의 전체 PREFIX이며 CONTROL과 frozen parent를 둘 다 비교한다. 일반 주행 첫1초, signed 구간 길이, 회전 짧아짐, 세션 집중도도 기록하라. 처음부터 변경을 합치거나 LR·λ를 바꾸지 마라. 결과 후 승자 하나만 현재 FULL 복사본에 stage2로 옮기며, 동일 노출량은4156 update다. 기존 서버0.13368483 제출물은 보존하고, 새 제출은 단일 모델/단일 최종 가중치로만 한다.

## 11. 근거 파일

- 최신 첨부: `PRO 제안 검토: interval VECTOR + FINE-READ — 2026-09-21` (복사본 `SOURCE_CODEX_REVIEW_KO.md`).
- 직전 계획: `A2_8931739c_SharedHistory_Review_and_Plan_20260921.md`.
- 이전 VECTOR/FINE 계획: `A2_H4_PROGRESS_013368_to_012_SingleModel_Plan_20260921.md`.
- 부모 원본 기록: `review_73f9ea4.zip` 내부의 `A2-H4-PROGRESS-s1/RUN_INDEX_ENTRY.json`, `experiment.json`, `latest_eval.json`.
- 이번 package의 `arithmetic.json`, `protocol.json`, `provenance.json`.

`protocol.json`은 repository launcher의 지원 인자를 보장하는 실행 파일이 아니라 **옮겨 구현할 설정 계약**이다. 소스에 존재하지 않는 CLI를 이미 실행 가능하다고 제시하지 않는다. 실제 착수 source SHA와 checkpoint bytes는 작업 환경에서 확정한다.
