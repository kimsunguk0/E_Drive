# ETRI MotionDrive — 4908989 이후 재설계

작성: 2026-09-07. 근거 커밋: `490898943b267c22ae34141a930608494f99963e`.
작업루트: `/NHNHOME/data/sukim/adcl`.

이 문서는 새 전략과 실행 명세다. 아래 새 모델의 성능·지연은 아직 측정하지 않았다.
기존 설계/코드/체크포인트는 보존한다. 이번 작업은 조사와 설계 문서 작성이며,
학습 실행이나 제출 변경을 뜻하지 않는다.

## 최신 운영 답변 반영 — 영상 추론 status 사용

2026-09-07 사용자가 전달한 추가 운영 답변을 반영한다. 이번 확인 시 서버의
`h200_latest/OPEN_ISSUE.md` 말미에는 이 추가 답변이 없었으므로, 아래는
**사용자 전달 내용**으로 출처를 구분한다. 기존 질의응답 원문은 변경하지 않는다.

- 영상으로부터 네트워크가 직접 추론한 history/status를 planning에 사용하는 것은 허용된다.
  영상에서 파생된 정보라는 것이 허용 근거다.
- planner가 영상 기반 feature 없이 bbox/vectorized map 등의 출력값만 받는 구조는
  영상과 궤적이 하나의 네트워크로 이어지지 않으므로 이번 챌린지 범위에서 제외된다.

따라서 영상으로 예측한 속도/가속도/회전율/과거 움직임을 auxiliary에만 제한할
규정상 이유는 없다. **실제 영상 feature 경로와 함께 planner 입력으로 사용한다.**
제공 pose/status를 직접 넣는 것, goal로 후보 좌표를 생성/수정하는 것을 새로 허용한
답변은 아니므로 해당 제한은 유지한다. 전체 모듈 공동 학습이나 전 구간 gradient
전달 의무까지 이 답변에서 추론하지 않는다.

## 1. 결정

주력은 **정렬 전 영상의 운동 대응을 보존하는 dual-stream video encoder +
image-only 다중 5초 궤적 decoder + 고정된 출력 행만 고르는 goal selector**로 바꾼다.

목표를 “더 큰 bank에서 GT D3 순위를 잘 맞히기”로 한정하지 않는다.
지금 관측 가능한 초기 운동, 장면이 허용하는 미래 행동, 제공 goal이 지정하는
목적지를 서로 다른 문제로 다룬다. 작은 current-only 모델로 처음부터 다시
주행 표현을 배우게 하지도 않는다.

기존 T4/A0/selector는 약 47ms의 복구 가능한 제출 후보와 실험 대조군으로 남긴다.
새 bank는 이미 만들어졌으므로 버리지 않되, bank 확대 자체를 돌파구로 간주하지 않는다.
목표는 1위지만, 현재 증거만으로 0.09 도달이나 수상 심사 통과를 보장할 수 없다.

## 2. 현재까지 확실히 아는 것과 모르는 것

| 근거 | 관측 | 해석의 범위 |
|---|---|---|
| 현재 A0 + learned selector | val38 realized 0.2522, shortlist oracle 0.1764 | 반복 사용한 로컬 검증; 최종 test 예상값 아님 |
| 전체 wrapper 3090 | median 46.997ms, p99 47.044ms | 현재 모델 실측; 새 모델/4090의 실측 아님 |
| train medoid K14,279 | full oracle 0.055445 | 충분한 후보를 모두 알 때의 해당 split 하한 |
| 모델 top128 parent 아래 | oracle 0.057182, 평균 약 1,806 child | routing은 유망; fine scorer와 shortlist 출력은 미완 |
| child logit 상속 후 selector 재학습 | N48 realized 0.2794 | child 고유 영상 신호가 없는 실험의 실패 |
| Gaussian GT-score noise | 상관 약 0.55에서 N48 realized 0.2426 | 잡음 모형 한정 진단; 보편적인 상관 요구선 아님 |
| r4/r5 scorer 변형 | 대체로 수천분의 몇 개선 | 기존 표현 위의 손실·head 조정에서 큰 이득 미발견 |
| current-only → T4 | 큰 개선 | 이 구현에서 과거 영상이 중요하다는 실증 |
| T7, aligned VO, logit KD | 시험한 구성에서 실패 | 모든 history/VO/표현 전이가 불가능하다는 증거 아님 |

2026-09-07 공개 leaderboard에서 E2E 1위는 0.0924106, 3위는 0.1375893이다.
[공식 leaderboard](https://dxchallenge.ai.kr/competitions/8/leaderboard).
이는 공개 L2 순위이며 최종 속도·코드 심사 결과를 뜻하지 않는다.
서로 다른 split의 `val bank oracle 0.0996`과 `test 0.0924`를 비교하여
수학적 달성 불가능을 주장하면 안 된다. 다만 현재 구조의 여유가 부족한 근거다.

### 2.1 “공식가중”을 둘로 분리한다

운영 답변의 공식 시간가중은 0.5초 간격 6개 점에 대해
`w = [11, 11, 5, 5, 2, 2] / 36`이다.
첫 1초가 61.1%, 첫 2초가 88.9%다.

기존 `val_weight`는 별개다. `scripts/etri_split.py:179` 부근에서
test의 command × speed bin × goal-distance bin 분포에 맞춘 자체 importance weight를
만든다. 따라서 기존 0.2522는 **공식 시간가중 D3 + 자체 sample weight** 점수다.
공식 test 점수를 그대로 재현하는 estimator라고 단정할 수 없다.

새 모델의 학습/선택은 train 내부에서 정한 기준으로 한다. test-derived weight로
튜닝하지 않는다. 보고에는 다음을 나란히 남긴다.

1. 모든 평가 frame의 공식 시간가중 D3 평균.
2. 기존 proxy-weighted D3 — 과거 실험과의 비교용 보조 지표.
3. 시나리오/주행 세션별 평균, 정지/가속/감속/회전별 오차와 표본수.

설명회 FAQ는 test 라벨 생성·GT 역추정 및 test를 이용한 학습/최적화를 금지한다.
공개된 goal을 추론 시 허용된 선택 용도로 쓰는 것과 test 기반 최적화를 구별한다.

## 3. 이번에 코드에서 확인한 두 가지 정보 병목

### 3.1 움직임을 읽기 전에 정렬한다

`scripts/sparse_scoredrive.py:633`의 과거 투영은 `lidar2img @ hist_T`다.
현재 후보의 같은 세계점을 과거 영상에서도 표본화한다.
`:506`의 temporal MLP는 그 뒤의 `f_now, f_hist, 차이, 곱`을 받는다.
VO head도 `:643`에서 정렬된 feature 차이를 받는다.

정적 지점 X의 feature가 시점에 강건하다면 두 표본은 모두 h(X)에 가깝다.
장면 내용을 정렬하는 데는 유리하지만, 자차 운동을 알려 주는 원래 픽셀 이동을
명시적으로 보존하지 않는다. 카메라/높이 평균(`:450`)과 FPN 평균(`:495`)도
어디에서 어디로 움직였는지의 대응 정보를 더 압축한다.

**이것은 구조적 가설이지 원인 확정이 아니다.** 원근·가시성·동적 물체·정렬 오차를
통해 운동 신호가 남는다. 다만 기존 VO 실패는 원본 영상 correspondence의 실패와
같지 않다. 동일 encoder를 써서 정렬 전/후를 대조해야 한다.

### 3.2 선택 직전에 영상 정보를 scalar 하나로 압축한다

`scripts/train_row_selector.py:93`의 selector 입력은 후보 기하, goal/command,
후보당 visual logit 하나의 정규화·순위 파생량이다. temporal embedding과
운동 불확실성은 전달되지 않는다.

goal이 비슷한 여러 경로의 첫 1초 진행량을 선택할 때, 필요한 정보가 scalar
logit에 모두 보존되어 있다는 근거는 없다. 더 정확한 unconditional D3 순위뿐 아니라,
**후보별 영상 근거를 보존한 뒤 목표를 참고해 행을 선택하는 방식**을 검증해야 한다.

한 장면에서 좌회전/직진 등 여러 미래가 가능하면 image-only generator는 제공 goal에
따른 실제 선택 의도까지 알 수 없다. 단일 기록 GT와 다른 모든 후보를 강하게 벌주면
모드가 줄어든다. 생성은 plausible future 분포, 선택은 goal-conditioned row decision으로
나눈다. 이는 생성기에 goal을 넣자는 제안이 아니다.

## 4. 구조 재설계 전에 고칠 기하·평가 기준

### 4.1 SE(2) 대신 정확한 SE(3) 정렬 대조

`scripts/sparse_common.py:182`의 `history_transforms()`는 xy+yaw만 사용하고
z/roll/pitch를 버린다. raw pose와 label 변환은 full SE(3) 기반이다.

이번 read-only 표본 검사: val 38 scene × frame 30/75/120/165/210/255
(frame>=30인 2Hz val index를 scene별 `[::9]`로 추출),
과거 0.5/1.0/1.5초, 10개 GT waypoint를 현재 z=0 평면에 두고 과거 카메라에 투영.
768×432에서 두 방식 모두 보이는 표본 6,010개 기준 투영 위치 차이:

| median | p90 | p95 | p99 | >4px | >8px |
|---:|---:|---:|---:|---:|---:|
| 2.18px | 6.28px | 7.94px | 12.94px | 24.1% | 4.9% |

이는 full SE(3)와 기존 근사의 차이이며 실제 대응점 GT에 대한 reprojection error는
아니다. z=0 지면 가정도 별도 오차를 가진다. 이 수치로 성능 개선을 확정하지 않는다.
학습·평가 양쪽을 바꾼 paired 실험을 한다. 기존 ckpt의 inference만 바꾸면 분포가 달라진다.

정확한 변환: `T_current_to_history = inv(T_world_from_history) @ T_world_from_current`.
카메라 축/ego 축/좌우 부호/이미지 resize 좌표계는 기존 projection unit test와 교차한다.
pose는 이 장면 정렬 분기에만 사용한다. 새 raw-motion 분기로 행렬이나 이동량을 넘기지 않는다.

### 4.2 timestamp와 운동 label 정의

`fut5[:,:6] == fut`, `fut5[:,9] == goal`은 내부 변환 일관성의 증거다.
영상 노출 시각과 pose timestamp의 물리적 동기화까지 증명하지는 않는다.
train raw timestamp, 파일 frame mapping, optical displacement와 pose 변화의 지연을 점검한다.
test의 미래 정답을 추정하거나 그에 맞춰 offset을 고르지 않는다.

기존 `kin_feats()`는 직전 0.5초 평균 속력을 현재 순간 속력처럼 사용한다.
등가속 조건에서는 현재 속력이 이 값보다 `0.25*a`만큼 다를 수 있으며,
곡선 주행에서는 이동 거리와 현재 ego-x 변위도 다르다.
따라서 과거 GT 운동학 0.2082 역시 정확한 현재 상태를 아는 oracle는 아니다.
운동 label은 구간 평균변위와 현재 상태 추정치를 분리하고 train에서 노이즈를 측정한다.
이 검사는 진단/학습 label용이며 제공 pose로 제출 궤적을 외삽하지 않는다.

### 4.3 연속 주행 세션 단위 검증

raw 376 scenario는 14개 날짜에 걸쳐 있고, 한 날짜에 115개가 몰려 있다.
이번 timestamp 조사에서 val38 중 25개는 train scenario 시작 시각과 30초 이내,
32개는 60초 이내, 37개는 300초 이내다. 이는 파일 중복의 증거는 아니지만,
30초 clip만 분리하면 같은 연속 주행 구간이 양쪽에 놓일 수 있다.

날짜/주행 연속시간/공간 경로로 그룹을 만들고 인접 clip을 같은 fold에 둔다.
시간 gap의 임계값은 label 성능을 보기 전에 정하고 route overlap도 점검한다.
frame 단위 bootstrap 대신 세션 단위 paired bootstrap을 사용한다.

빠른 pilot은 고정 grouped split 하나에서 시작한다. 유망한 구조만 3 seeds와
복수 grouped folds로 확장한다. 이미 검토한 val38을 “새로운 untouched test”라고 부르지 않는다.

selector 학습용 upstream prediction은 OOF로 만든다. **전체 pipeline 일반화 점수**를
주장할 때는 outer-heldout이 bank/scorer/selector 학습에 모두 제외되어야 한다.
outer-train 내부 cross-fit으로 selector train 특징을 만들고 outer-heldout에서 최종 평가한다.
selector-only CV와 전체 pipeline nested OOF는 이름과 비용을 분리한다.

## 5. 새 모델 명세

```text
제공된 영상: 현재 6-camera + 과거의 필요한 원본 frame
    ├─ Scene stream: 도로/차선/신호/선행차/교차로의 공간 특징
    │    └─ 과거 정보 사용 시 정확한 SE(3)로 정렬 가능
    └─ Motion stream: 정렬 전 이미지 쌍의 correspondence + 시간간격
         └─ 영상 운동 latent + 예측 history/status + confidence
              (제공 pose/status 입력 없음)
                         ↓
           image-only joint trajectory decoder
       [B,N,10,2] 완성 5초 후보 + 후보별 visual descriptor
                         ↓  후보 확정, goal의 역방향 경로 차단
          goal/command → row selector → index 하나
                         ↓
                 선택한 행의 첫 6점 제출
```

### 5.1 Scene stream

기존 split의 정보 압축 진단에서는 현재 T4 encoder를 활용하여 변경 원인을 분리한다.
새 grouped split의 일반화 실험에서는 public pretrained 또는 해당 fold의 train만 본
checkpoint를 사용한다. 기존 T4가 새 holdout을 학습했다면 그 가중치를 가져오지 않는다.
bank/parent routing도 같은 원칙으로 다시 만든다. 기존 bank를 그대로 쓴 결과는
legacy diagnostic으로 표시하며 새로운 end-to-end 일반화 점수에 섞지 않는다.
최종 모델에서 dense BEV를 필수로 두지 않는다. multi-camera image feature + 적은 수의
scene/candidate tokens로 공간 문맥을 읽어도 된다. 다만 후보 지면점 평균만으로
신호등·선행차·진입 우선권을 다 안다고 가정하지 않는다.
이 scene feature는 후보 planner가 직접 읽는 실질적인 입력으로 유지한다.
perception 결과를 bbox/map 리스트로만 저장한 뒤 그것만 입력받는 별도 planner로
대체하지 않는다. 사용되지 않는 feature를 인자로 추가하는 형식적 연결도 하지 않는다.

장면 표현이 다음 병목으로 확인되면 이미 가진 driving-pretrained backbone 및
object/lane auxiliary supervision을 같은 encoder 예산에서 시험한다.
기존 map-distance scalar 실패는 scene-level perception supervision의 실패가 아니다.
처음부터 새 backbone 전체를 ranking loss 하나로 재학습하는 경로는 우선하지 않는다.

### 5.2 Motion stream — 원본 영상 대응을 직접 보존

시작 camera는 front이며 필요 시 실제 가시성/texture를 보고 side camera를 추가한다.
입력 간격은 `t0` 대비 `-0.1, -0.2, -0.5초`를 우선 비교한다. 저속에서는 긴 간격,
큰 움직임/occlusion에서는 짧은 간격이 유리할 수 있으므로 하나를 단정하지 않는다.
단순히 6-camera T7을 모두 고해상도로 추가하지 않는다.
설명회와 raw 데이터에서 과거 3초+현재의 10Hz 31frame 제공을 확인했으므로
-0.1/-0.2초 영상은 입력 후보로 사용할 수 있다. 실제 제출 adaptor도 이 frame들을
누락 없이 읽도록 확인한다. front-only는 운동 probe의 범위다.
제출 모델은 카메라 입력 규격과 과거 사용 시 영상 의무를 별도로 준수하며,
처리한 모든 시점/카메라의 연산을 지연 측정에 포함한다.

spatial correlation/cost volume 또는 추적된 대응점이 feature pooling보다 먼저다.
픽셀 위치, 카메라 ID, 시간간격, match confidence를 보존한다.
camera calibration과 지면 제약은 metric scale 확보에 쓸 수 있지만 지면 경사,
pitch 변화, 동적 물체, texture 부족을 반드시 평가한다. 다중 카메라라 해도
필요한 stereo overlap이 실제로 있는지 확인 없이 metric scale을 가정하지 않는다.

학습 label: 과거 구간 Δtranslation/Δrotation, 정지/비정지, 영상 matching 및 uncertainty.
제공 pose는 label 또는 scene 정렬에만 쓰고 raw-motion input/ROI 선택/normalization에
들어가지 않는다. 추론 시 planner에는 **영상 feature + 영상 운동 latent +
영상에서 예측한 명시적 history/status 및 confidence**를 전달할 수 있다.
예를 들어 v/a/yaw-rate/과거 변위의 추정값을 임베딩해 scene feature와 결합한다.
최신 답변에 따라 scalar v/a head를 진단·auxiliary 전용으로 제한했던 방침은 정정한다.
이는 네트워크가 영상에서 예측한 값이며, 제공 ego status를 임베딩한 값과 다르다.
planner를 상태 숫자만 받는 구조로 바꾸지 않고 영상 feature 경로를 유지한다.

정밀도와 효과는 별도 문제다. 같은 후보/데이터/encoder 조건에서
`scene feature only`, `scene + motion latent`,
`scene + motion latent + predicted status`를 비교하여 명시적 상태 입력의 순효과를 본다.
GT status 입력은 별도 표기된 진단 상한에만 사용하고 배포 체크포인트/API에서 제외한다.

초기 독립 검사는 고정 calibration + 정적 지면 픽셀 matching + robust ground-motion fit으로
실제 metric 정보량을 본다. 이는 진단 baseline이며 제출용 규칙 궤적 생성기가 아니다.
학습형 branch의 초기화/teacher 후보는 공개 optical-flow pretrained 모델이다.
[RAFT 공식 구현](https://github.com/princeton-vl/RAFT)은 명시적 correspondence/correlation을,
[SEA-RAFT 공식 구현](https://github.com/princeton-vl/SEA-RAFT)은 flow와 uncertainty 출력을 제공한다.
이 자료들은 ETRI metric 속도 정확도나 3090 지연을 보장하지 않는다.
외부 데이터와 weights의 라이선스·출처를 확인/기록한다. 무거운 모델은 train-only teacher로
쓸 수 있지만, tiny student의 정확도와 시간을 다시 측정해야 한다.

### 5.3 궤적 decoder — 경로 모양과 시간 진행을 함께 학습

최종 지향은 하나의 속도로 모든 후보를 고정하는 모델이 아니다.
현재 운동 posterior 주변에서 가속/감속/정지 전환과 회전의 **joint modes**를 만든다.
예를 들어 mode별 진행량 `s(t)`와 곡률/방향 spline을 neural decoder가 예측하고,
모델 내부에서 0.5초 간격 10개 좌표로 디코딩한다. stop/delayed-start 모드를 포함한다.
무조건적인 path × velocity Cartesian product는 불가능한 조합을 만들 수 있어 피한다.

초기 구현 비교는 두 가지로 제한한다.

- **Conservative head:** 실제 5초 medoid bank의 child-specific visual scorer.
  같은 parent의 child마다 같은 logit을 상속하지 않고, 후보 시간진행 descriptor와
  motion/scene latent를 함께 읽는다. model top64/128 parent routing을 사용한다.
- **Main head:** 작은 수의 image-conditioned spline/control modes와 image-only refinement.
  고정 bank의 초기속도 양자화에 묶이지 않는다. goal을 보기 전에 좌표가 완성된다.

두 head는 동일한 새 encoder/motion latent에서 비교한다. bank를 키우는 효과와
운동 정보를 보존하는 효과를 혼동하지 않는다. N은 12/24/32를 검증해 선택하며,
N을 늘렸다는 이유만으로 품질 향상을 주장하지 않는다.

Spline은 먼저 CPU로 4/6/8 knot representation fitting을 검사한다.
GT를 샘플별로 fit한 값은 **표현 재구성 하한**이지 learned candidate oracle이 아니다.
정지·출발·회전의 재구성 오차와 복잡도를 함께 보고, 충분히 작지 않으면 고정 medoid
fine-scoring head를 유지한다. 임의로 다항식 차수만 키워 0을 맞히지 않는다.

### 5.4 학습 목적

공식 D3는 주 목표다. tail 3.5–5초는 goal 선택을 위한 미래 구조 보조 목표다.
3초 정확도를 희생하여 5초 전체 평균을 낮추는 checkpoint는 채택하지 않는다.

초기 제안:
`L = L_D3(best matched mode) + beta*L_tail + L_mode_probability + gamma*L_motion + eta*L_scene`.

beta/gamma/eta는 고정 보편값이 아니라 train/tune에서 정할 소규모 후보다.
3초-only 대조군을 유지한다. L_scene은 실제 추가할 perception head가 있을 때만 쓴다.
전체 mode에 expected-D3를 강하게 걸어 다양성을 줄이는 손실은 주력으로 두지 않는다.
GT는 단일 실현 미래이므로 나머지 모든 mode를 물리적으로 틀린 미래라고 간주하지 않는다.
다양성은 무조건 밀어내는 좌표 repulsion보다 train의 endpoint/stop/turn 모드 coverage로 본다.

5초 train label에 goal과 같은 endpoint가 포함되는 것은 미래 예측의 정상적인 supervision이다.
test goal을 생성기에 넣거나 test 미래를 label로 만드는 것과 다르다.
tail auxiliary가 D3를 해치면 beta를 낮추거나 train-only medoid tail 부착 head를 backup으로 쓴다.

### 5.5 Selector — 좌표 생성과 분리된, 풍부한 근거를 가진 행 선택

generator 출력:
`candidate_xy_abs_5s [B,N,10,2]`, `candidate_ids [B,N]`,
`visual_logits [B,N]`, `candidate_visual_descriptor [B,N,d]`.

selector는 위의 **완성 출력**과 goal/command로 N개 score를 만들고 index 하나만 반환한다.
candidate feature는 영상 운동/장면과 해당 후보의 정합도·uncertainty를 보존한다.
goal-query로 candidate를 새로 생성하거나 좌표를 수정하지 않는다.
별도 모듈로 학습하고 generator를 freeze/detach하여 selector의 goal-dependent gradient가
generator로 흘러가지 않게 한다. 후보 재생성 loop, weighted coordinate average,
goal endpoint snap, 선택 후 velocity 보정은 없다.

이는 운영 답변의 “모델의 여러 출력 중 선택” 원칙에 맞추려는 설계이지 사전 승인서가 아니다.
기존 scalar-logit selector가 보수적 backup이다. 새 descriptor가 어떤 정보를 담고
어디서 생성되는지 코드와 counterfactual test로 명확히 제시한다.

## 6. 첫 실험 — 큰 학습보다 정보 가설부터 판정

### Gate 0: CPU, 기준 정리

1. 공식 시간가중/자체 sample weight를 분리한 evaluator.
2. grouped split + timestamp/projection 검증; SE(3) alignment unit test.
3. 같은 shortlist에서 scalar selector와 candidate-embedding selector 비교용 OOF dump 명세.
4. spline 표현 재구성 및 큰 bank의 routing/shortlist 분해. GT fit은 진단으로만 표시.

### Gate 1: B200 GPU 0–3에 할당할 첫 네 작업

| GPU | 작업 | 직접 답할 질문 |
|---|---|---|
| 0 | 기존 T4 + SE(2), 고정된 새 split의 대조 재학습/평가 | 비교 기준을 재현하는가? |
| 1 | GPU0와 동일하되 정확한 SE(3) 정렬 | 기하 근사가 실제 순위/선택 오차를 만들었는가? |
| 2 | 정렬 전 correspondence → 과거 metric-motion probe | 원본 영상에서 유용한 정밀 운동을 읽을 수 있는가? |
| 3 | 기존 split/후보 고정, scalar vs visual-descriptor row selector paired 진단 | scalar 압축이 선택 손실의 원인인가? |

GPU2는 미래 S3 회귀를 하지 않는다. 동일 encoder aligned descriptor와 raw correspondence를
같은 label/split에서 비교한다. pretrained matching baseline은 별도 이름으로 보고한다.
GPU3는 후보 집합/좌표/상위 encoder를 고정한다. 새 motion 결과가 없더라도 현재
temporal candidate embedding으로 정보 압축 가설을 먼저 검정할 수 있다.
이는 legacy split의 pilot이다. 새 grouped OOF 결과가 필요할 때는 GPU0/1의
fold-trained upstream dump가 나온 뒤 GPU3 selector 학습을 수행한다.
OOF 특징이 아직 없는 상태에서 네 작업이 모두 독립적인 최종 검증이라고 주장하지 않는다.
GPU 4–7은 사용하지 않는다. 이 문서는 위 작업을 이미 실행했다는 보고가 아니다.

### Gate 2: 유망한 신호를 결합하고 encoder를 고정한 head 대조

Gate 1 이후에만 동일 raw-motion/scene encoder로 (a) A0,
(b) child fine scorer, (c) dynamic spline decoder를 비교한다.
selector는 각 출력 분포에 맞춰 train-only OOF로 다시 학습한다.
새 bank에 기존 selector를 그대로 적용한 분포 이탈을 재발시키지 않는다.

우선 실용적인 screening 기준은 paired realized 개선 >=0.01로 미리 고정한다.
이 값은 통계법칙이 아니라 GPU 시간을 아끼기 위한 개발 gate다.
효과가 작은 경우 seed/세션 bootstrap CI로 판단하고, 좋은 seed 하나로 채택하지 않는다.
motion MAE만 개선되고 실제 oracle/regret가 그대로면 그 branch를 확대하지 않는다.

### Gate 3: 1위를 노릴 가치가 있는지 판정

공식 metric을 쓰는 동일한 heldout protocol에서 오차를
`representation/bank floor + proposal coverage loss + selection regret`로 분해한다.
dynamic decoder는 CPU GT-fit floor와 실제 image-generated oracle를 별도로 표시한다.

1차 통합 목표: 현재 기준 대비 >=0.03 절대 개선, 여러 grouped splits에서 방향 일치.
다음 목표: learned shortlist oracle <=0.10, realized <=0.15를 **실제 영상 추론**으로 확인.
1위 도전 단계의 예산 예시는 0.05 floor + 0.02 coverage + 0.02 regret = 0.09다.
이는 달성한 수치나 test 예측치가 아니라 어디까지 개선이 필요한지 보이는 설계 예산이다.
0.20 근처에 머무르면 “조금 더 학습하면 1위”라고 보고하지 않는다.

## 7. 필요한 정밀도를 과소평가하지 않는다

직선 주행에서 순수한 초기속도 오차만 가정하면 공식 시간가중 D3 민감도는
`sum(w*t)*abs(delta_v) = 1.25*abs(delta_v)`다.
순수 등가속도 오차는 `0.5*sum(w*t^2)*abs(delta_a) = 1.0486*abs(delta_a)`다.
각각 다른 오류를 0으로 둔 국소적인 민감도이며 일반 오차의 독립 합 공식이 아니다.

따라서 0.1m/s의 metric 속도 오차만으로도 약 0.125m가 될 수 있다.
“VO 오차 몇 %면 충분하다” 또는 “goal이 있으니 해결된다”는 근거가 없다.
실제 목표는 편향·정지 전환·시간 정렬을 포함한 초기 진행량의 정밀도와
goal 선택으로 회수 가능한 multimodal uncertainty다.
영상 속도 0.03–0.05m/s 수준은 작은 D3 기여를 위한 어려운 목표 규모이지
일반 VO 논문이 보장하는 수치가 아니다.

보고할 motion 지표: 구간 변위 MAE/RMSE, 속도 환산 MAE/RMSE 및 bias,
속도/가속/정지/회전별 성능, confidence calibration, 최종 D3 증감.
고상관만으로 정밀 추정을 주장하지 않는다. MAE를 Gaussian 가정 없이 sigma로 바꾸지 않는다.

## 8. 지연·규정·재현성

- 3090 기존 46.997ms는 실측이다. 새 motion/decoder 포함 시간은 미측정이다.
- 새로운 전체 forward의 개발 목표는 3090 <=80–90ms로 여유를 두는 것이다.
  3090/4090 FLOPS 비례값은 참고 범위일 뿐 4090 통과 보증이 아니다.
- preprocessing/format conversion만 제외한다. 원본 영상 encoder, flow/matching model,
  history forward, proposal, selector를 모두 포함한다. 여러 함수로 나누어도 측정에서 숨기지 않는다.
- batch1, 정해진 해상도, warmup, inference_mode, CUDA events + synchronize,
  실제 제출 precision/reset policy로 측정한다. median/p95/p99와 CPU wall time도 기록한다.
- goal/command 교란 시 raw-image forward로 candidates/ids/descriptors가 불변인지 확인한다.
- row selector는 독립 저장된 generator 출력에서 index만 반환하고 첫 6점이 bitwise 동일해야 한다.
  dynamic 후보는 고정 bank와 비교하는 대신 **goal을 보기 전 생성·저장된 tensor**와 비교한다.
- 이미지 zero 단일 테스트나 강제 stop만으로 compliance를 증명하지 않는다.
  normal/shuffle/temporal shuffle/zero 및 goal-only 음성 대조를 실제 전체 모델로 검사한다.
- 학습/추론 어디에도 test 미래 라벨 사용, test trajectory lookup, 장면 ID 외우는 lookup,
  제공 ego/status를 planner feature로 입력하는 통로를 두지 않는다.
- 학습 라벨로 쓰는 pose, scene 정렬에 쓰는 pose, 영상에서 예측한 history/status를
  명확히 구분한다. 마지막 것은 최신 답변에 따라 planner 입력으로 허용된다.
- planner는 scene/video feature를 직접 사용한다. bbox/map 출력값만으로 planning을
  수행하거나, 사용하지 않는 영상 feature를 명목상 연결하는 구조로 바꾸지 않는다.
- `git SHA / config / bank+split+checkpoint SHA / label provenance / code-path / precision /
  seed / selected step / metric definition / latency script`를 한 manifest에 남긴다.
- 실제 수상 허용 여부는 운영국의 코드 심사 사항이다. 내부 PASS를 공식 승인이라 부르지 않는다.

## 9. 이번 전략에서 하지 않을 것

1. same aligned representation 위에 loss/head 조합만 계속 추가하기.
2. 현재 T7 실패를 근거로 모든 긴 history를 닫거나, 반대로 근거 없이 T7부터 재시도하기.
3. rank correlation 0.55를 보편적인 성공 gate로 삼기.
4. 14k bank에 inherited logit을 붙여 놓고 bank 확장 실패라 결론 내리기.
5. goal/현재 상태만으로 계산한 경로를 신경망 wrapper에 넣어 규정을 피하기.
6. 이름이 강한 diffusion/BEV 모델을 전면 이식하고 motion 관측·metric scale 문제는 미루기.
   Diffusion은 새 표현이 검증된 뒤 동등한 후보 수/forward 예산에서 decoder 비교 후보로만 둔다.

## 10. 성공/실패 시 다음 방향

| 실험 결과 | 해석 | 다음 |
|---|---|---|
| SE(3)만으로 개선 | 일부는 표현 이전의 정렬 문제 | 모든 신모델의 정렬 기준으로 채택 |
| raw motion은 정확, A0 realized 개선 | 영상 운동 정보가 실제 선택에 유용 | dynamic/child head로 coverage 개선 |
| raw motion은 정확, A0에서는 미개선 | 후보 floor 또는 selector 전달이 병목일 수 있음 | 후보 descriptor/확장 bank 고정 비교 |
| raw motion 부정확, rich selector 개선 | 기존 latent에도 쓸 정보가 scalar보다 많음 | 해당 selector를 먼저 제출 후보로 확보 |
| ground matching만 실패 | 지면 가정/동적 물체/scale 문제일 수 있음 | 실패 원인별 raw learned correspondence 대조; VO 전체 기각 금지 |
| motion와 rich selector 모두 미개선 | 핵심 새 가설이 지지되지 않음 | 장면 perception/행동 전환 supervision을 우선 검정, 대규모 decoder 학습 중단 |
| GT-fit 좋고 learned proposals 나쁨 | 표현 용량이 아닌 학습/관측 문제 | knot/mode 수 확대 대신 신호/조건부 목표 재검토 |
| 전체 <=0.15에 접근 | 공개 1위 도전 준비 단계 | 3 seeds, nested grouped OOF, 최종 speed/compliance, 정식 제출 검증 |

최종 권고: **원본 영상에서 움직임을 보존하고, 그 정보를 마지막 선택까지 잃지 않는 것**을
먼저 증명한다. 그다음 고정 속도 bank의 한계를 image-conditioned 완성 경로로 줄인다.
이번에는 새 구조의 좋은 이름이나 train fitting이 아니라,
heldout realized 개선이 다음 GPU 투입을 결정하게 한다.

## 근거 파일

- `PHASE6_BANK_SELECTOR.md` — 4908989 최신 정정/실험표.
- `PHASE5EFG_SELECTOR.md` — selector, history, VO, profile 실험.
- `scripts/sparse_scoredrive.py:432` — spatial sampling/averaging.
- `scripts/sparse_scoredrive.py:506` — temporal fusion.
- `scripts/sparse_scoredrive.py:633` — pose alignment/VO branch.
- `scripts/sparse_common.py:182` — SE(2) history transform.
- `scripts/train_row_selector.py:53` — 과거 구간 속력 외삽 진단.
- `scripts/train_row_selector.py:93` — scalar visual feature selector.
- `scripts/etri_split.py:179` — 자체 sample importance weights.
- `h200_latest/OPEN_ISSUE.md` — 최신 운영 답변; 이전 완화 답변보다 최신 제한을 우선.
- 2026-09-07 사용자 전달 추가 운영 답변 — 영상에서 네트워크가 추론한 history/status
  planning 사용 허용, 영상 feature 없이 bbox/vectorized map 출력만 받는 planner 제외.
- `adcl_pre.zip`, slide 43/47 — 채점식, 외부 공개 데이터 라이선스/출처, test 학습 금지.
