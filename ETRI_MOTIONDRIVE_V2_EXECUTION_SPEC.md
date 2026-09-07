# ETRI MotionDrive V2 — 최신 OPEN_ISSUE 기반 실행 설계

작성: 2026-09-07. 설계 기준 코드: `7f2752c` / 실험 근거: `4908989`까지.
작업루트: `/NHNHOME/data/sukim/adcl`.

상태: **설계 확정, 신규 모델 구현·학습·지연 측정은 아직 하지 않음.**
확정은 실행할 구조와 검정 순서를 뜻하며, 성능 달성이나 운영국의 개별 코드 승인이 아니다.
이 문서는 이전 `ETRI_MOTIONDRIVE_REDESIGN_20260907.md`의 주력 구조·실험 순서를 대체한다.
기존 데이터/기하/통계 감사와 제출 백업은 유지한다.

## 0. 이번에 달라진 결정

**주력 A: goal-conditioned shared visual perception + image-inferred motion/state +
direct 3-second trajectory planner.**

goal은 실제 공통 BEV/scene feature를 형성하는 perception module에서만 사용한다.
planner에는 그 연속적인 영상 특징, 원본 영상에서 읽은 motion feature,
영상 네트워크가 예측한 history/status를 준다. planner에 goal token/query를 직접 주지 않는다.
처음 출력은 `[B,6,2]`의 3초 궤적 하나다. 고정 bank, NMS, 5초 후보, 외부 selector는 없다.

**백업 B: goal-blind complete-5s candidate generator + goal row-only selector.**
현재 T4/A0/selector의 코드·가중치·실측을 보존한다. 주력의 실패나 코드 심사 위험에
대비한 독립 실행 경로이며, 주력 A와 goal 불변성 규칙을 혼용하지 않는다.

새로운 규정은 5초 label의 필요조건을 바꾼다. 주력 A는 3초를 직접 학습할 수 있다.
5초 tail은 향후 3초 성능을 실제 개선할 때만 켜는 auxiliary다. 초기에는 off다.

## 1. 규정 원문과 출처

사용자가 제공한 업데이트 전문을 프로젝트 루트 `OPEN_ISSUE.md`에 원문 그대로 보관한다.
원본 첨부: `05ae29c7-4e99-43a6-acee-2440bc53819e/pasted-text.txt`.
SHA256: `ba8b50097450612b985a250fc7f75027467d1da1140555fdc155926efc079006`.
옛 `h200_latest/OPEN_ISSUE.md`는 역사 기록으로 보존한다.

| 항목 | 업데이트 원문 | 설계 결정 |
|---|---|---|
| 일부 카메라 사용 | 질문1: 6개 중 일부 사용 가능 | 현재 6개, 과거는 우선 전방만 처리 가능 |
| command/vad_cmd | 질문1: 둘 다 사용 가능 | 첫 2×2에서는 비활성화; 후속에는 공통 scene condition/백업 selector에 사용 |
| 과거 영상 | 질문4: 과거3초 안에서 수/간격 자유 | 10Hz의 짧은 간격도 활용, 모든 사용 시점 영상 처리 |
| 제공 history/status | 질문7: planner 직접/단순 embedding 금지; 공통 특징의 간접 활용 허용 | 주력에서는 이 간접 입력도 사용하지 않고 영상 예측을 택함 |
| goal → 공통 BEV/scene feature | 질문8 재질문에 명시적 긍정 답변 | **주력의 허용 경로** |
| goal → planner attention query | 우리 구조에 대한 답변에서 불허 | module 이름만 바꾸거나 query 위치만 옮기는 방식 제외 |
| goal → 완성 후보 중 선택 | 질문8 및 우리 질문 답변에서 허용 | 백업 경로 |
| 영상에서 예측한 history/status → planner | 질문10 명시적 허용 | 실제 planner 입력으로 채택 가능 |
| 시간 | 질문4: reset 이후 모든 model forward 누적 | 영상/motion/scene/planner/필요 selector 모두 측정 |
| 후처리 | 질문1/5: 제출 형식 변환만; 위치 보정 불가 | 좌표 보정·goal snap·외삽 후처리 없음 |

직전 대화에서 사용자가 별도로 전달한 답변은 “영상 feature 없이 bbox/vectorized map
출력만 받는 planner는 챌린지 범위 제외”다. 이 문장은 이번 284줄 첨부에는 없으므로
질문10의 인용문에 섞지 않는다. 이 추가 답변도 설계 제약으로 유지한다.

### 해석 정정

1. “goal은 선택기에만 사용 가능”은 규정 전체가 아니라 **백업 B의 보수적 설계**다.
2. “goal 변경 시 후보/출력 bitwise 불변”은 B에만 적용한다. A의 feature와 궤적은
   허용된 공통 특징 경로를 통해 goal에 따라 달라질 수 있다.
3. “status는 planner에 입력 불가”가 아니라 **제공 status와 영상 예측 status**를 구분한다.
4. 실제 다중-task supervision은 우리 설계의 근거다. 운영국이 특정 개수의 auxiliary
   head나 전 모듈 joint training을 필수로 요구했다고 쓰지 않는다.
5. exact-zero 영상 테스트 하나가 허용의 필요충분조건은 아니다. 거절된 우리 구조도
   해당 테스트를 만족했다. 실질적 영상 기여와 코드상의 정보 흐름을 확인한다.

## 2. 왜 기존 5초 bank 구조를 주력에서 내리는가

기존 구조는 goal을 늦게 사용하기 위해 먼저 goal 없이 큰 bank를 정렬하고,
12개 정도로 줄인 다음 5초 endpoint로 선택해야 했다. 그 과정에 세 가지 손실이 있다.

- 고정 bank의 좌표/초기속도 양자화.
- goal을 모르는 순위 모델이 실제 의도에 맞는 후보를 미리 버리는 coverage 손실.
- 후보당 scalar logit에서 실제 3초 궤적을 고르는 selection regret.

Q8 허용 경로를 쓰면, 공통 영상 표현이 목적지를 참고한 상태에서 planner가
공식 채점 대상 3초 좌표를 직접 학습할 수 있다. 위의 세 손실을 **구조상 강제하지 않는다.**
이는 자동으로 성능이 좋아진다는 뜻은 아니다. 회귀의 조건부 평균화,
영상 운동 정밀도, 표현 일반화는 새로운 실제 검정 대상이다.

기존 0.2522는 공식 시간가중 + 자체 sample proxy weight를 적용한 반복 검증값이다.
직접 예측 모델과는 같은 frame·같은 metric으로 다시 비교한다.
기존 bank oracle 0.0996은 새 continuous decoder의 하한이 아니다.

## 3. 주력 A의 정보 흐름

```text
현재 6-camera ── shared image backbone/FPN ───────┐
과거 전방 영상 ── shared image backbone/FPN ───────┤
                                                  ↓
고정 calibration + 필요 SE(3) ── 실제 spatial scene encoder
goal xy ────────────────────── shared scene enhancement
                                                  ↓ Z_scene
                              ┌───────────────────┼───────────────────┐
                              ↓                   ↓                   ↓
                     current object occupancy   lane geometry      planner
                                                                    ↑
정렬 전 전방 image features ── correspondence ── M_visual ──────────────┤
                                      └─ inferred history/status ────┘
                                                                    ↓
                                                 3초 좌표 [B,6,2]
                                                                    ↓
                                                   제출 형식 변환만
```

planner는 영상 feature `Z_scene`, `M_visual`을 직접 사용한다. bbox/차선 리스트만
받지 않는다. 전체 모델에는 goal이 입력되지만 planner 호출에는 직접 들어가지 않는다.
따라서 A를 “goal-free 모델”이라고 부르지 않는다.

## 4. 첫 구현의 tensor와 모듈

### 4.1 영상 입력 — V2.0 고정안

- 현재: 6-camera, 각 768×432.
- 과거: 전방 camera만 t=-0.1/-0.2/-0.5/-1.0s, 각 384×216.
- 물리적인 이미지 10장. 일부 카메라만 사용하는 것은 Q1 허용 범위다.
- 현재 전방 이미지는 motion용으로 feature를 재사용하거나 필요한 해상도로 모델 내
  변환한다. 그 추가 neural forward가 있으면 전부 지연에 포함한다.
- 과거 범위는 제공된 3초 이내, 실제 timestamp 간격을 사용한다.
- 첫 판에서 T7 전체 카메라나 dense sequential BEV rollout을 넣지 않는다.

위 간격은 시작 설정이지 최적값이 아니다. 입력 수를 바꾸는 ablation은 첫 2×2 이후다.
V2.0의 feature 부족이 확인되면 -1.5s/측면 과거 영상을 추가 비교한다.

motion correspondence의 좌표 기준은 과거 영상의 384×216으로 통일한다.
첫 구현은 현재 전방 FPN feature를 동일한 물리 시야의 과거 feature 크기로 bilinear
resample하여 대응시킨다. resize는 `align_corners=False`로 고정한다.
crop/undistort 이후 같은 시야인지 확인하고, intrinsics의 resize scale 및 feature stride,
pixel-center convention을 반영한 image↔feature roundtrip을 검사한다.
현재 768-pixel 변위와 과거 384-pixel 변위를 같은 수치로 비교하지 않는다.
증강은 첫 판에서 같은 카메라의 시간쌍에 일관된 photometric 변환을 적용하고,
기하 증강은 calibration/label 변환 검증 전까지 사용하지 않는다.

### 4.2 Backbone / shared scene

첫 초기화는 현재 보유한 공개 nuImages R50:

`ckpt/backbones/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth`.

CPU 실제 로드 확인: 318 state entries 주입, unexpected 0, backbone missing 0.
기존 `ResNet34FPN128(arch="resnet50")`를 재사용할 수 있다.
**128-channel compact FPN은 새 초기값**이므로 전체 trunk가 주행 사전학습 완료라고
표현하지 않는다. 기존 nuImages 이식 실험은 오히려 나빴으므로 성능 우위도 가정하지 않는다.
이번에는 실제 scene supervision과 새로운 조건 경로를 같은 초기값에서 비교한다.

scene output의 시작 규격은 `[B,64*48,128]`, ego-frame의 고정 spatial grid다.
초기 범위 x=[-10,70]m, y=[-32,32]m. train label coverage를 확인한 뒤 4판 시작 전에
범위를 고정한다. 출력 궤적을 이 grid 중심으로 양자화하지 않는다.

- calibration-aware multi-view sampling으로 image evidence를 읽는다.
- scene용 history alignment는 full SE(3). xy+yaw-only 근사는 대조 진단에만 남긴다.
- 카메라/높이/scale 정보를 학습 전에 모두 평균해서 지우지 않는다.
- 2개 정도의 local/window attention 또는 경량 spatial mixing layer를 사용한다.
  3,072-token global self-attention을 여러 층 반복하는 설계는 첫 버전에 넣지 않는다.
- 넓은 수용영역의 영상 context도 보존하여 lane point만으로 신호·선행차를 판단하지 않는다.

### 4.3 Goal의 정확한 위치: 실제 shared scene enhancement

`SharedSceneEncoder(image_features, calibration, alignment, goal_xy)` 안에서만 goal을 받는다.
고정 spatial grid의 영상 evidence를 어떤 위치/카메라에서 읽고 통합할지 조건화한다.
출력 `Z_scene`은 아래의 실제 perception heads와 planner가 **동일하게 공유**한다.

V2.0 조건화 연산은 고정 scene cell q에 대한 image evidence cross-attention으로 고정한다.
cell별 표본 f_qj는 calibration으로 정한 camera/height/scale의 영상 특징이다.

```text
g_used = goal_xy               if G=ON else constant_zero_xy
Q_q = Q_image(Z_base[q]) + Q_context(phi(cell_xy[q], g_used))
a_qj = softmax_j(Q_q · K_image(f_qj) / sqrt(d) + visibility_mask)
Z_scene = SpatialRefine(Z_base + sum_j(a_qj * V_image(f_qj)))
```

Q_q는 고정 공간 cell의 **인식 query**이지 trajectory/waypoint query가 아니다.
goal은 attention 가중치 형성에만 관여하고 `Q_context` 자체를 Z에 residual로 더하지 않는다.
scene cell/표본/모듈 폭은 G ON/OFF에서 같다. OFF에서도 동일 module을 실행하되
실제 goal 대신 상수만 사용하므로 goal 외의 용량/샘플링 차이를 줄인다.
이후 §4.6의 별도 waypoint query에는 Q_context나 g_used를 전달하지 않는다.
attention score/softmax는 안정적인 FP32 연산을 사용한다. 모든 표본이 invisible인
cell은 masked softmax NaN을 내지 않도록 명시적으로 zero evidence를 반환한다.

구현 금지:

- goal embedding을 `Z_scene` 끝에 별도 token으로 붙이기.
- goal을 Z의 별도 채널/skip/residual로 그대로 복사하여 planner에 전달하기.
- 6-waypoint/trajectory query를 먼저 만든 뒤 goal attention을 적용하고 이 모듈을
  이름만 “scene encoder”로 부르기.
- goal 또는 goal-derived 경로를 이미지 sampling의 유일한 후보 경로로 만들기.
- goal로 직접 만든 궤적에 작은 영상 residual만 더하는 구조.

scene output은 waypoint가 아닌 공간 장면 표현이다. goal은 spatial image-evidence
통합의 조건으로 사용하고, 값은 영상 특징에서 형성한다. 단, 이 분리만으로 승인된다는
주장은 하지 않는다. 실제 공간 인식과 공통 특징 사용이 핵심이다.

첫 버전은 현재 object occupancy와 lane-line geometry heads를 Z에 연결한다.
학습에서 유효한 supervision과 비상수 예측, heldout perception 품질을 확인한다.
이 두 작은 head는 최초 inference wrapper에서도 실행/반환하여 실제 경로를 명확히 한다.
그 scalar/raster 출력으로 Z를 대체하지 않는다.

### 4.4 실제 인식 supervision — 이름만 붙인 auxiliary 제외

확인한 raw `train/20260112-105434.tar`:

- `annotation/object.parquet`: timestamp, class, object ID, xyz, heading, box size,
  velocity, num_points 등. 현재 non-ego object의 oriented footprint로 occupancy target 생성.
- `annotation/map.parquet`: 이 scene에서 `lane_id`, XYZ `points`만 존재.
  lane-line raster 또는 distance target 생성. **drivable-area polygon label이 있다고 하지 않는다.**

다른 map variant에는 추가 class 정보가 있을 수 있으므로 schema를 명시적으로 normalize한다.
Q8 경로의 의미를 보이기 위해 현재 시점의 장면을 예측한다. 기존 future candidate
occupancy 배열을 이름만 바꿔 current-scene supervision으로 사용하지 않는다.

converter의 ego축/box heading 관례, raw map의 좌표계, timestamp를 unit test로 확인한다.
가능하면 raw global annotation을 해당 frame의 full SE(3) ego좌표로 일관되게 변환한다.
`num_points`/annotation validity/가시성/공간 범위를 반영하고 unknown을 임의로 free-space
음성 정답으로 바꾸지 않는다. ego vehicle 자체는 occupancy target에서 제외한다.

### 4.5 Motion / inferred history / state

raw temporal feature는 scene alignment **이전**에 분기한다.
`MotionEncoder(current_front_features, past_front_features, timestamps, fixed_calibration)`.
provided pose/status, hist_T, goal, command를 이 module에 입력하지 않는다.

첫 구조는 multi-scale local correspondence/correlation + confidence aggregation이다.
대응의 픽셀 위치와 시간간격을 보존하며 전역 feature 차이 평균만으로 VO를 만들지 않는다.

- `M_visual`: `[B,Nm,128]`의 영상 운동 tokens. Nm 시작값 192.
- `history_hat`: 과거 4시점의 current-relative dx/dy와 회전(sin/cos), `[B,4,4]`.
- `state_hat`: 예측 vx/vy/ax/ay/yaw-rate와 stop probability 및 uncertainty.
- `history_hat/state_hat`는 Q10에 따라 planner에서 직접 사용할 수 있다.

train pose는 지도 label이다. history 변위/회전을 먼저 학습하고, timestamp를 반영한
상태 label과 불확실성을 함께 학습한다. 순간 가속도를 거친 2차 차분 하나로 정의해
노이즈를 정답처럼 강제하지 않는다. causal 구간 fit/annotation validity를 검증한다.
GT 상태를 planner에 넣는 teacher forcing은 초기 2×2에서 쓰지 않는다.
train/inference 모두 실제 `state_hat`를 사용하여 상태 오류에 적응시킨다.

공개 optical-flow 모델은 원본 영상 대응의 독립 기준/학습용 teacher로 활용할 수 있다.
[SEA-RAFT 공식 구현](https://github.com/princeton-vl/SEA-RAFT)은 flow와 uncertainty를
출력하지만 ETRI metric 속도 정확도나 3090 지연을 보장하지 않는다. 공개 weights/license와
사용 데이터 출처를 기록한다. teacher가 train-only이면 inference에 넣지 않는다.

### 4.6 Planner: 6개의 학습된 waypoint query, 연속 좌표 직접 예측

서명:

```python
plan_abs = planner(
    scene_features=Z_scene,       # [B,3072,128], 실제 shared visual features
    motion_features=M_visual,     # [B,Nm,128], goal/pose-free image motion
    predicted_state=state_hat,    # image inference only
    predicted_history=history_hat,
)                                # [B,6,2], 0.5/1/1.5/2/2.5/3s
```

학습된 시간 query 6개가 Z/M와 영상 예측 상태를 읽는 2-layer 소형 decoder로 시작한다.
query의 출처는 learned embeddings/고정 시간 인덱스이지 goal/제공 status가 아니다.
마지막 MLP는 ego-frame 누적 xy를 직접 출력한다. 출력의 고정 bank row 제약은 없다.
abs→inc 변환은 제출 규격상 필요할 때만 수행하며 roundtrip과 축/단위를 검증한다.

최종 state/history 및 xy projection head와 planning loss는 FP32로 계산한다.
backbone의 bf16 autocast와 출력 좌표의 정밀도를 구분한다. 예를 들어 bf16으로
이미 반올림된 20m 부근 출력은 이후 float32 cast만으로 정밀도가 복구되지 않는다.
좌표/head 입력부터 autocast를 끄고 float32 연산을 하며 배포에도 같은 경로를 쓴다.

현재 상태를 단순 등속·등가속 외삽한 곡선으로 최종 출력하지 않는다.
신호/차량/도로 영상 특징을 읽는 학습형 decoder가 미래 위치를 정한다.

초기 head는 단일 조건부 궤적이다. 회전/정지 전환에서 모드 평균화가 실제로 확인되면
그때 3–6 learned modes + model probability head를 비교한다. 첫 판부터 diffusion,
14k fine ranking, NMS, selector를 한꺼번에 추가하지 않는다.

## 5. Loss / 학습 순서

### 5.1 공식 목표

`L_plan = sum_t w_t * ||plan_abs[t] - GT3[t]||_2`,
`w = [11,11,5,5,2,2]/36`.

첫 1초 61.1%, 첫 2초 88.9%가 반영된다. 검증은 이 정확한 metric을 사용한다.
최초 objective:

`L = L_plan + alpha_occ L_current_occ + alpha_lane L_lane + alpha_motion L_motion`.

L_current_occ는 유효영역의 balanced BCE 계열, L_lane은 line probability/distance loss,
L_motion은 normalized history/state regression + uncertainty/stop loss다.
loss는 유효 sample/pixel 수로 평균한다. 각 auxiliary의 단위·scale을 train-only pilot에서
정하고 네 판 모두 같은 값을 쓴다. 목표 gradient가 묻히지 않는지도 기록한다.
숫자 하나를 보편적 최적 weight라고 제시하지 않는다.

**5초 tail beta=0, smoothness=0, GT-logit KD=0**으로 시작한다.
과도한 smoothness로 실제 가감속을 지우거나, 이미 실패한 logit KD를 반복하지 않는다.
3초 성능이 안정된 이후에만 tail auxiliary를 paired 비교한다.

### 5.2 P0 — 데이터/latency/실제 perception 준비

1. 통일된 coordinates/timestamp/projection 검사와 shared scene raster 구현.
2. 실제 raw-video batch를 쓰는 inference skeleton의 3090 전체 시간 측정.
3. train subset overfit/heldout probe로 lane/occupancy/motion label과 학습 경로 검증.
4. 각 fold의 train만 사용해 public pretrained에서 common perception/motion 초기값 생성.
   네 판은 같은 초기값을 복제한다. 기존 full330 ETRI ckpt를 새 holdout에 재사용하지 않는다.

초기 optimizer 제안은 AdamW, backbone LR 1e-5, 새 FPN/scene/motion/head LR 1e-4,
weight decay 0.01, bf16, grad clip 5, batch 16/단일 GPU다. OOM이면 batch/accumulation을
전체 팔에 동일하게 조정한다. 이는 시작값이며 loss scale/수렴은 짧은 pilot에서 확인한다.
기존 Stage1–3 전체 freeze 처방을 기계적으로 재적용하지 않는다.
R50 전체가 이미 충분한 표현이라는 가정도 하지 않는다.

### 5.3 P1 — GPU 0–3, 2×2 요인 실험

첫 표는 **goal의 공통 scene condition 여부 G × 영상 예측 status의 planner 입력 여부 S**다.
모든 팔은 같은 원본 영상 입력, scene/motion features, 상태 auxiliary supervision,
perception heads, direct 3초 decoder, split/seed/schedule을 사용한다.

| GPU | G: goal→shared scene | S: predicted state/history→planner | 목적 |
|---|---|---|---|
| 0 | OFF | OFF | 공통 기준 |
| 1 | ON | OFF | 허용된 goal 간접 입력의 순효과 |
| 2 | OFF | ON | 영상 예측 상태를 명시적으로 전달하는 효과 |
| 3 | ON | ON | 실제 주력 조합과 상호작용 |

S=OFF에서도 동일 motion feature M과 동일 state prediction auxiliary는 유지한다.
차이는 planner가 명시적 예측 상태/역사 tensor를 받는가뿐이다. OFF에서는 해당 입력을
mask한다. 따라서 이 표는 raw motion의 존재 자체가 아니라 **명시적 상태 전달의 순효과**를 잰다.
raw correlation 대 aligned-diff의 정보량 검사는 P0 또는 이후 별도 paired probe다.

G=OFF는 scene condition을 neutral 값으로 고정하고 goal 입력을 읽지 않는다.
첫 2×2에서 command/vad_cmd는 전부 OFF로 하여 다른 navigation 정보가 goal 효과를
대신하지 않도록 한다. 이 선택은 규정상 command 금지가 아니라 실험 통제다.
주력 선택 후 command를 같은 scene module에 추가하는 paired 실험을 한다.

G=OFF 직접 회귀는 경로 의도가 더 모호하다. G의 이득을 backbone 일반화 개선으로
혼동하지 않는다. 모든 팔에서 최종 realized D3와 perception/motion 품질을 함께 보고한다.
GPU 4–7은 사용하지 않는다. 이 문서 작성 시 네 판을 실행하지 않았다.

### 5.4 P2 — 복제/단순화/실제 경쟁력

- 첫 판의 유망한 구성과 가장 가까운 대조군만 3 seeds로 복제한다.
- 주력 A vs 보존된 B는 같은 metric과 실제 입력별 전체 지연으로 비교한다.
- 명시적 status가 무효이면 S를 제거해도 된다. 허용된다는 것이 꼭 써야 한다는 뜻은 아니다.
- 회귀의 조건부 모드 평균화가 확인되면 작은 multimodal continuous decoder를 추가 비교한다.
- 5초 auxiliary와 command 효과는 주력 구조를 먼저 확정한 뒤 각각 분리한다.

## 6. 검증 protocol과 개발 gate

### 데이터 분리

- 기존 376 scenario는 시간상 인접 clip이 많으므로 주행 세션/공간 경로 단위로 묶는다.
- grouped holdout 정의와 gap 기준을 성능을 보기 전에 고정한다.
- backbone/fold pretraining/normalization/GT bank 모두 해당 holdout을 보지 않아야 한다.
- 반복 사용한 val38은 historical report set이지 untouched test가 아니다.
- public test에서 pseudo-label, 미래 역추적, scene lookup, 학습/최적화를 하지 않는다.

### 지표

- 1차: official temporal-weighted D3의 frame 평균.
- 병기: 기존 proxy-weighted 값, session 평균, stop/accel/decel/turn별 값과 n.
- 세션 단위 paired confidence interval; frame 독립 bootstrap이나 best-seed 선택 금지.
- 직접 예측 A에는 shortlist oracle/regret를 필수 점수로 붙이지 않는다. 그것은 B의 분해다.
- A의 endpoint/첫1초·2초 오차와 longitudinal/lateral 성분, multimodal averaging 실패를 본다.

### 단계적 gate — 목표이지 달성값 아님

1. 데이터/정렬/출력 format 일치, finite loss/gradient, perception 실제 학습을 먼저 통과.
2. 2×2의 paired realized 차이를 본다. 개발 screening 기준은 0.01 절대 개선으로
   미리 정하되 불확실성이 크면 반복 검증한다. 단순히 mean만 낮다고 확정하지 않는다.
3. 기존 B 대비 지속적인 개선과 실측 속도를 확보한 모델만 제출 후보로 올린다.
4. 0.15 부근 이하에 접근하는지 확인하고, 그다음 0.09 수준을 겨냥한 오차 예산을 세운다.
   서로 다른 val/test 숫자로 1위 가능/불가능을 수학적으로 단정하지 않는다.

0.1m/s의 순수 초기속도 오차는 직선 국소 가정에서 weighted D3 약 0.125m를 만들 수 있다.
그래서 motion high correlation이나 train fitting만으로 1위를 기대하지 않는다.
shared goal feature 또한 초기 타이밍 오차를 자동 해결하지 않는다.

## 7. Compliance tests — 트랙별로 다르게

### A: shared-goal-feature direct planner

1. Planner signature/runtime hooks: goal xy, goal embedding, supplied status/history의 직접
   입력 또는 숨은 global cache/closure 접근이 없어야 한다. image-predicted state는 허용한다.
2. `Z_scene/M/state_hat`를 고정하고 외부 goal만 바꾸면 planner 출력은 동일해야 한다.
   반면 영상부터 전체 forward를 다시 실행하면 Z와 경로가 바뀔 수 있다. 이를 구별한다.
3. 실제 shared Z에서 occupancy/lane heads와 planning이 작동하고 학습되는지 확인한다.
   dummy head, Z의 명목상 전달, goal 전용 숨은 channel을 쓰지 않는다.
4. 이미지 normal/shuffle/temporal shuffle/constant/zero를 실제 전체 pipeline으로 재실행한다.
   이미지 불일치 시 오차/인지/상태 추정의 변화를 보고 영상의 실질 기여를 검증한다.
5. goal, 제공 pose/status, 이미지 feature를 각각 분리해 state_hat의 출처를 검사한다.
   raw motion branch는 동일 영상에서 goal/provided status 변화에 불변이어야 한다.
6. GT history/status 교체·생략 테스트로 planner 숫자 경로의 우회 입력을 찾는다.
   scene용 alignment 변경에 따른 Z 변화는 별도로 취급한다.
7. 출력은 neural planner가 만든 6점을 제출 형식으로 변환한 것과 동일해야 한다.
   제출 후 좌표/속도/stop/endpoint 보정 금지.
8. 중요한 경계: goal이 shared feature를 거쳐 영향을 주는 것은 허용 범주지만,
   코드가 그 범주에 실제로 속하는지 최종 판단은 운영국이 한다.

### B: goal-blind candidates + row selector

기존 goal 교란→candidates/logits/ids 불변, 완성 row identity, 첫6점 slice,
goal 기반 좌표 보정 금지, 전체 raw forward 이미지 대조를 유지한다.
N=12 등은 우리 개발 선택이지 공식 후보 수 제한이라고 부르지 않는다.
selector OOF는 upstream 포함 검증인지 selector-only인지 구분한다.

## 8. Latency 명세

V2.0 시간은 아직 미측정이다. 개발 예산은 전체 3090 median<=80ms,
p95<=90ms로 잡고 공식 4090의 100ms 무벌점 경계에 여유를 지향한다.
이 수치 관계는 보장이 아니며 4090은 실제 해당 환경에서 검증해야 한다.

포함: 모든 사용 영상 encoder, scene alignment 이후 neural 처리, motion correspondence
network, state heads, shared scene heads, planner, 필요한 learned selector.
외부 모델로 분리해 놓았다는 이유로 flow/VO forward를 측정에서 제외하지 않는다.
본체에 없는 사전 image feature cache를 추론 입력으로 삼지 않는다.

batch1, 실제 입력 해상도/precision, warmup, inference_mode, CUDA events와
`torch.cuda.synchronize()`, median/p95/p99를 기록한다. reset 이후 전체 경로를 측정한다.
CPU wall-time도 별도 기록한다. 공식 제출 규격의 FLOPs 기준도 별도로 검사한다.

지연이 초과하면 먼저 구조 profiling 결과를 보고 image resolution/history 수/scene token
수를 줄인다. 4개의 accuracy 팔을 서로 다른 backbone으로 돌려 요인 비교를 깨뜨리지 않는다.
R50가 예산을 넘으면 모든 팔에 같은 R34 등 대안을 적용하고 초기화 차이를 명시한다.

## 9. 구현할 파일과 실행 순서

아래는 명세이며 아직 구현된 파일 목록이 아니다.

1. `scripts/build_scene_supervision_v2.py`: timestamp/coordinates 검증, current occupancy/lane rasters.
2. `scripts/build_grouped_split_v2.py`: 세션 그룹·split·출처 SHA 고정.
3. `models/motiondrive_v2/scene_encoder.py`: 실제 shared spatial features와 G condition.
4. `models/motiondrive_v2/motion_encoder.py`: raw correspondence와 영상 추론 state/history.
5. `models/motiondrive_v2/planner.py`: 6-query direct3s decoder, G direct input 없음.
6. `scripts/train_motiondrive_v2.py`: 동일 초기화의 2×2, exact metric, 주기 checkpoint.
7. `scripts/audit_motiondrive_v2.py`: A/B별 정보 흐름·실제 이미지 counterfactual tests.
8. `scripts/benchmark_motiondrive_v2.py`: 전체 forward, 3090 기준 측정.

데이터 unit test와 latency skeleton부터 만들고, P0를 통과한 뒤 GPU 0–3 학습을 시작한다.
이번 설계 요청에 대한 작업은 문서와 규정 snapshot까지이며 GPU 학습을 임의 실행하지 않는다.

## 10. 근거와 미확정 사항

확인한 코드/자산:

- `OPEN_ISSUE.md` 질문1/4/7/8/10 및 우리 구조 질문 답변 — 허용 범주.
- `scripts/sparse_scoredrive.py:81`, `scripts/train_sparse_scoredrive.py:39` — R50/FPN과 loader.
- `src/etri_vad/ETRI_E2E_Driving_Challenge/tools/data_converter/etri_vad_converter.py:150`
  — object labels; heading/좌표 관례 주의.
- 같은 repo `projects/mmdet3d_plugin/datasets/etri_vad_dataset.py:180` — map의 ego-frame 변환.
- `PHASE5DE_RESULTS.md` — 기존 nuImages 이식과 candidate occupancy 실험의 부정적 결과.
- `PHASE6_BANK_SELECTOR.md` — 현 백업 성능/지연 및 기존 평가의 한계.
- [VAD 공식 구현](https://github.com/hustvl/VAD) — 재사용 가능한 공개 코드의 출처.

미확정: raw correspondence의 실제 metric 정밀도, goal-conditioned shared perception의
heldout 이득, single-output 회귀의 모드 평균화 크기, 신모델의 3090/4090 지연,
최종 운영국 코드 심사. 이 항목을 달성/승인 완료로 표시하지 않는다.

**최종 선택: 주력은 허용된 공통 goal conditioning과 영상 추론 상태를 이용한
직접 3초 planning. 5초 후보-선택 구조는 필수가 아니라 백업이다.**
