# ETRI-ScoreDrive 상세 설계 및 실행 명세

작성일: 2026-09-03 KST  
대상: 2026 자율주행 AI 챌린지 과제 3 E2E Driving  
목표: 규정 위반 위험 없이 Top 3, 최종 목표 1위  
실행 노드: `/NHNHOME/data/sukim/adcl`, NVIDIA B200 8장 중 GPU 0–3 사용

이 문서는 아이디어 메모가 아니라 구현, 학습, 평가, 제출 판정을 위한 기준 문서다.
수치나 구조가 다른 문서와 충돌하면 최신 운영국 답변은 `h200_latest/OPEN_ISSUE.md`,
현재 실험 상태는 `h200_latest/PROJECT_STATUS.md`, 새 모델 설계는 이 문서를 따른다.

---

## 1. 한 문장 결론

최종 모델은 **goal, command, ego status를 전혀 받지 않는 영상 전용 후보 생성·순위화
모델**이어야 한다. 모델은 5초까지 완성된 복수 후보와 영상 점수를 출력하고, 모델 밖의
선택기는 영상 점수로 먼저 줄인 매우 작은 후보 집합에서만 제공된 command와 5초 goal을
사용해 후보의 **index만 선택**한다. 공식 제출은 선택된 후보의 앞 3초, 즉 첫 6개
waypoint만 반환한다.

핵심 우선순위는 다음과 같다.

1. 현재 K=1024 soft-CE의 top-1 변별력부터 고친다.
2. 기존 3초 후보를 보존한 채 3.5–5.0초 extension을 붙여 goal 선택용 의미를 만든다.
3. 최종적으로 path geometry × velocity profile factorization과 sparse image sampling으로
   정확도와 4090 지연시간을 동시에 맞춘다.
4. diffusion은 후보 coverage가 다시 병목으로 확인될 때만 1-step residual refiner로
   검토한다. 현재 1순위가 아니다.

---

## 2. 현재 기준선과 설계를 강제하는 실측 사실

### 2.1 리더보드와 우리 위치

2026-09-03 확인값이다.

| 항목 | 값 |
|---|---:|
| 1위 L2 | 0.130536583 |
| 3위 L2 | 0.142403011 |
| 우리 제출 `v2a_goal` test L2 | 0.236238922 |
| 같은 모델 val L2 | 0.23696 |
| val-test gap | -0.3% |

`v2a_goal`은 val-test 대리지표 확인용 제출이며 goal이 planner query에 직접 들어가므로
최종 수상 심사 기준으로는 제출 불가다. 이 모델의 점수는 목표값 비교에만 사용한다.

### 2.2 현재 규정 준수 K=1024 모델

768×432, 2Hz anchor, epoch 4 기준이다.

| 지표 | L2 |
|---|---:|
| top-1 | 0.5393 |
| oracle@3 | 0.2664 |
| oracle@6 | 0.2071 |
| oracle@20 | 0.1408 |
| oracle@100 | 0.1005 |
| 전체 K oracle | 약 0.0996 |

가장 가까운 anchor의 중앙 순위는 7위이고, top-6 포함률은 43.8%, top-20 포함률은
75.6%, 정확한 최인접 anchor의 top-1 적중률은 9.4%다. 후보 coverage는 이미 1위권
수준인데 모델이 상단 순서를 못 정한다.

768 epoch 1에서 epoch 4로 세 epoch를 더 학습했을 때 top-1은 0.5429에서 0.5393으로
0.7%만 개선됐지만 oracle@6은 0.3148에서 0.2071로 34% 개선됐다. 현재 soft target
cross entropy는 정답을 상위권으로 올리지만 1등으로 분리하는 압력이 부족하다.

### 2.3 해상도 실험

동일 epoch 1 비교다.

| 지표 | 768 | 1536 | 변화 |
|---|---:|---:|---:|
| top-1 | 0.5429 | 0.5202 | -4.2% |
| oracle@3 | 0.3851 | 0.3730 | -3.1% |
| oracle@6 | 0.3148 | 0.2939 | -6.6% |
| oracle@20 | 0.2024 | 0.1897 | -6.3% |

1536은 후보 순위와 coverage를 조금 개선하지만 `top-1 - oracle@6` 선택 gap은 거의
같다. 게다가 3090에서 전체 7-frame forward가 768은 590.6ms, 1536은 1108.2ms였다.
1536을 최종 배포 backbone으로 쓰지 않고 teacher 또는 front-camera 국소 고해상도
ablation으로만 사용한다.

### 2.4 5초 endpoint 선택 진단

train-only로 각 3초 anchor에 5초 endpoint 중앙값을 붙인 오프라인 진단이다.

| 선택 방식 | M=1 | M=3 | M=6 | M=10 | M=20 |
|---|---:|---:|---:|---:|---:|
| 현재 768 epoch 4 영상 순위 후 goal 선택 | 0.5393 | 0.3296 | 0.2741 | 0.2601 | 0.2517 |
| 이상적인 GT 순위 후 train-only endpoint 선택 | 0.0996 | 0.1220 | 0.1439 | 0.1711 | 0.2118 |

full K를 goal만으로 고르면 0.6888이다. 이는 다음 두 결론을 준다.

- goal은 좋은 영상 shortlist 안에서만 유용하다.
- 영상 ranker가 좋아질수록 M은 작아야 한다. 기본값은 M=3이다.

### 2.5 지연시간이 정확도만큼 중요하다

최종 점수는 다음 식이다.

```text
Error = L2 × (1 + max(0, T_infer - 100) / 200)
```

`T_infer`는 RTX 4090에서 stream reset 이후 최종 출력까지 실행된 모든 model forward의
누적시간이다. 과거 frame을 일곱 번 forward하면 일곱 번이 모두 더해진다.

| 4090 T_infer | 3위 Error 0.1424를 넘기 위한 최대 raw L2 |
|---:|---:|
| 100ms 이하 | 0.1424 |
| 125ms | 0.1266 |
| 150ms | 0.1139 |
| 200ms | 0.0949 |
| 300ms | 0.0712 |

현재 VAD 7-frame 구조는 3090 590.6ms, 4090 추정 320–400ms라서 그대로는 수상권이
어렵다. 최종 ScoreDrive의 절대 목표는 4090 전체 forward 100ms 이하, 3090 실측
proxy 150ms 이하이다.

---

## 3. 규정 위협 모델과 절대 금지선

### 3.1 운영국 최종 답변의 의미

운영국은 다음 구조도 금지라고 명시했다.

- goal을 attention query에만 사용한다.
- attention value는 영상 feature만 사용한다.
- 영상이 0이면 출력이 정확히 0이다.

이유는 goal이 trajectory를 생성하는 planner 내부에 들어가 생성에 관여하기 때문이다.
따라서 `goal-conditioned`, `command-conditioned`, `status-conditioned`라는 이름이 붙은
planner는 값 경로가 영상 전용이어도 사용하지 않는다.

운영국이 명시적으로 허용한 가장 안전한 경계는 다음뿐이다.

> 영상 모델이 이미 생성한 여러 완성 출력 중 하나를 goal 등으로 선택

최종 판단은 코드 심사이고 이의제기가 없으므로, 문언상 우회 가능해 보이는 구조도
사용하지 않는다. 이 설계도 사전 허가를 보장하는 것은 아니며, 알려진 답변 안에서
심사 위험을 가장 낮추는 보수적 구조다.

### 3.2 추론 모델 API에서 구조적으로 금지한다

프로덕션 모델의 추론 signature에는 아래 값이 존재하면 안 된다.

- `goal`, `target_point`, `ego_fut_goal`
- `command`, `vad_cmd`
- `ego_his_trajs`, `ego_lcf_feat`, velocity, acceleration, yaw rate
- 과거 pose의 원본 또는 임베딩

권장 API는 다음과 같다.

```python
outputs = model(
    current_images,       # [B, 6, 3, 432, 768]
    history_images,       # 선택: [B, 6, 3, 216, 384], t=-0.5s
    camera_calibration,
    history_alignment,    # 선택: 고정 기하변환에만 사용
)

# outputs
candidate_xy_5s   # [B, N, 10, 2], 이미 완성된 누적 좌표
visual_logits     # [B, N], goal/command와 무관
candidate_ids     # [B, N], 고정 bank index
```

여기서 `N`은 전체 virtual bank가 아니다. image-only coarse/fine ranking을 이미 끝낸
`N=3`의 작은 출력 목록이며, route-balanced ablation도 최대 12개까지만 허용한다. full
bank는 모델 내부 탐색 공간일 뿐 외부 goal selector에 노출하지 않는다.

학습 시 GT trajectory가 loss 함수에 들어가는 것은 label 사용이므로 허용된다. 중요한
것은 inference graph와 후보 생성 graph가 goal/status를 받지 않는다는 점이다.

### 3.3 상대 pose의 유일한 허용 용도

과거 frame을 사용할 경우 상대 pose는 과거 image feature와 현재 좌표를 정렬하는 고정
행렬 곱에만 사용한다.

```text
current candidate points
    -- fixed T_history_from_current --> history camera projection
    -- image feature sampling --> visual evidence
```

상대 pose 수치 또는 그 임베딩을 MLP, attention query/key/value, ranker에 넣지 않는다.
raw speed/status도 만들지 않는다. 정렬 행렬에는 gradient가 없고 저장된 feature와
concat하지 않는다.

### 3.4 selector가 할 수 있는 것과 없는 것

허용 범위:

- 영상 모델이 완성한 후보 index 중 하나 선택
- 제공된 `vad_cmd`로 정적 route bucket 선택
- 선택 가능한 작은 후보 집합 안에서 5초 endpoint와 goal의 거리로 index 선택
- 선택된 10개 waypoint 중 첫 6개를 제출 규격으로 잘라내기

금지 범위:

- goal 방향으로 후보 회전, 이동, scale 조정
- 3초/5초 비율로 trajectory 늘리기 또는 줄이기
- 둘 이상의 후보 interpolation, 평균, spline blending
- goal과 status로 새로운 waypoint 계산
- 영상 점수보다 먼저 full K를 goal로 검색
- 선택 뒤 속도, stop, endpoint를 규칙으로 보정

### 3.5 필수 불변성 테스트

같은 image input에 goal/command만 바꾼 counterfactual batch를 만든다.

```text
candidate_xy_5s: bitwise identical
visual_logits:   bitwise identical
candidate_ids:   bitwise identical
selected index:  변경 가능
selected path:   반드시 candidate_xy_5s의 한 row와 bitwise identical
```

`selected path`가 candidate 두 개의 중간값이거나 epsilon이라도 바뀌면 실패다.

---

## 4. 데이터 사용 명세

### 4.1 split

| split | scenario | 용도 |
|---|---:|---|
| train | 330 | 모든 weight, candidate bank, tail cluster 학습 |
| mini-val | 8 | loss 계수, M, threshold, LR 선택 |
| final val | 38 | 최종 보고와 제출 gate |
| test | 1,125 clips | 추론만, 학습·통계 적합 금지 |

candidate bank, endpoint median, route bucket threshold, loss temperature 등 데이터에서
구한 모든 값은 train 330만으로 만든다. mini-val은 hyperparameter 선택에만 사용한다.
final val에서 scale, threshold, affine 보정을 적합하지 않는다.

### 4.2 현재 데이터 구조

train scenario는 300개 main frame과 앞뒤 pose 범위를 가지며, 각 main frame에 대해
미래 5초까지 train label을 만들 수 있다. 영상은 6개 camera다.

```text
/NHNHOME/data/sukim/adcl/
├── train/                       # 376개 raw tar
├── test/                        # 1,125개 raw tar
├── cache/etri_768/              # 376개 scenario, 768×432 cache
├── data/etri/meta_train/        # annotation/calibration/meta
├── dense_vocab_v1/              # 2026-09-02 H200 최신 모델 소스
├── h200_latest/                 # 최신 문서와 K=1024 anchor
├── checkpoints/anchorvocab_k1024_wide/epoch_4.pth
└── env/venv/
```

PKL 안의 image path는 `/tmp/pm97/data/etri/...` 절대경로다. B200에서는 이미 symlink로
재구성했고 train 376개, PKL 2개, sample image, SHA256 검증을 통과했다.

### 4.3 현재 3초 label

`etri_vad_converter.py`는 다음 상수를 쓴다.

```text
TRAJ_STEP = 5       # raw 10Hz에서 0.5초
FUT_TS = 6          # 0.5, 1.0, ..., 3.0초
```

`ego_positions_in_frame`이 현재 ego frame으로 미래 global pose를 변환하고, 저장 label은
증분 좌표다. metric과 candidate assignment 전에 반드시 `cumsum`하여 누적 좌표로
변환한다.

### 4.4 새 5초 label

새 converter는 원본 변환식을 바꾸지 않고 `FUT_TS_5S = 10`만 별도 field로 추가한다.

```python
future_ids = frame_id + 5 * np.arange(11)
future_abs, valid = ego_positions_in_frame(data, frame_id, future_ids)
future_inc = np.diff(future_abs[:, :2], axis=0)

info['gt_ego_fut_trajs_5s'] = future_inc.astype(np.float32)   # [10, 2]
info['gt_ego_fut_masks_5s'] = valid[1:].astype(np.float32)   # [10]
info['gt_ego_fut_goal_5s'] = future_abs[-1, :2].astype(np.float32)
```

필수 검증:

- 새 field의 첫 6개 증분이 기존 `gt_ego_fut_trajs`와 exact 또는 `1e-6` 이내 일치
- `cumsum(new[:6])`이 기존 3초 누적 trajectory와 일치
- 330 train의 frame≥30, frame%5==0 anchor가 17,820개
- train main frame 모두 future 5초 mask가 유효한지 수량 보고
- 좌표축 x/y, 회전 방향, 단위가 기존 metric crosscheck를 통과

### 4.5 auxiliary label은 입력이 아니다

train pose에서 계산한 speed, acceleration, stop/creep class는 visual feature를 학습시키는
label로만 쓸 수 있다. inference 입력으로 연결하지 않는다.

권장 auxiliary label:

- `progress_1s`, `progress_2s`, `progress_3s`, `progress_5s`
- 0.5초별 이동거리와 누적 arc length
- stop `<0.25m/3s`, creep, moving class
- future heading change와 route geometry class
- 선행 객체 존재/거리, lane evidence는 train object/map label로만 supervision

map annotation과 object annotation의 odometry 기준 차이로 최대 약 1.6m drift가 관측됐으므로
map auxiliary loss는 낮은 weight로 시작하고 planning loss를 지배하게 두지 않는다.

---

## 5. 후보 표현: 3초 점수를 지키면서 5초 의미를 얻는 방법

### 5.1 왜 10개 waypoint를 균등 최적화하면 안 되는가

공식 평가는 첫 3초만 본다. 5초 10개 waypoint를 균등 L2로 학습하면 다음 이유로 첫
3초가 나빠질 수 있다.

- loss gradient의 40%가 평가되지 않는 3.5–5.0초에 쓰인다.
- 먼 미래의 multimodality와 label noise가 초기 waypoint를 끌어당긴다.
- 제한된 candidate 수가 3초 coverage보다 tail 다양성에 할당된다.
- 5초 joint clustering이 현재 K=1024의 3초 quantization을 악화시킬 수 있다.

따라서 3초가 주목적이고 5초는 navigation tag이자 auxiliary target이다.

### 5.2 Milestone A: anchor-preserving multi-tail bank

가장 먼저 구현할 안전한 bank다.

1. 기존 train-only K=1024 3초 anchor를 그대로 보존한다.
2. 모든 train anchor를 공식 D3 metric으로 가장 가까운 3초 anchor `k`에 배정한다.
3. 각 sample의 3.5–5.0초 tail을 3초 endpoint 기준 상대 좌표로 만든다.
4. 표본 수가 충분한 anchor는 tail을 1–3개 cluster로 나눈다.
5. 표본이 적은 anchor는 동일 route/진행거리 이웃의 train-only median tail로 fallback한다.
6. 완성 후보 `A[k,r]`의 첫 6개는 원래 K=1024 anchor와 bitwise 동일하고, 마지막
   4개만 extension이다.

이 방식은 5초 정보를 추가해도 첫 3초 candidate oracle을 절대 바꾸지 않는다.

tail descriptor 권장값:

```text
[Δx_3.5, Δy_3.5, ..., Δx_5.0, Δy_5.0,
 endpoint_dx, endpoint_dy, heading_change, tail_arc_length]
```

초기 설정:

- anchor별 `min_count=12`
- 최대 tail mode `R=3`
- exact stop anchor는 10 waypoint 전부 exact zero인 후보를 반드시 index 0으로 유지
- 전체 후보 수가 2,048을 넘으면 낮은 빈도 mode를 병합

### 5.3 Milestone B: path geometry × velocity profile factorization

최종 bank는 SparseDriveV2의 좋은 아이디어인 geometry와 progress 분리를 가져오되 ego
status 입력은 제거한다.

각 5초 GT 누적 trajectory를 다음처럼 분해한다.

```text
s_i = Σ_{j≤i} ||p_j - p_{j-1}||          # 누적 arc length
S   = max(s_10, ε)
r_i = s_i / S                             # 시간별 normalized progress
q(u) = trajectory를 normalized arc length u∈[0,1]에서 재표본한 뒤 좌표를 S로 나눈 curve
```

- geometry bank `P`: `q(u)`를 route-stratified k-medoids로 cluster
- velocity bank `V`: `[S, r_1, ..., r_10]`을 stop/creep/moving stratified cluster
- 조합 후보: `C_{p,v}(t_i) = S_v × q_p(r_{v,i})`

arc length 기반이므로 급회전과 U-turn에서도 단순 x축 정규화보다 안정적이다. stop은
정규화하지 않고 별도 exact-zero profile로 둔다.

CPU oracle sweep:

```text
P ∈ {64, 128, 256, 512}
V ∈ {16, 32, 64, 128}
```

bank 채택 조건:

- final val을 보기 전에 train/mini-val로 P,V 결정
- 3초 oracle이 기존 K=1024 oracle보다 0.003 이상 나빠지면 기각
- 5초 endpoint/route coverage가 anchor-preserving bank보다 좋아야 함
- stop, creep, left, right, lane-keep, U-turn 희귀 mode별 coverage 보고

전체 `P×V`를 항상 fine score하지 않는다. coarse scorer가 path 12–16개, velocity
4–8개를 뽑은 뒤 64–128개 조합만 fine score한다.

### 5.4 후보 좌표는 언제 완성되는가

goal과 command를 보기 전에 다음이 모두 끝나야 한다.

1. path/velocity 선택
2. 조합과 10개 waypoint 생성
3. image-only residual이 있다면 residual 적용
4. visual score 계산
5. image shortlist 생성

그 이후 goal selector는 index만 선택한다. residual refiner를 넣더라도 goal/command를
받지 않으며, selector 뒤에는 절대 적용하지 않는다.

---

## 6. 최종 모델 구조

모델명: `ETRI-ScoreDrive`  
설계 목표: dense BEV 전체를 매 frame 갱신하지 않고 후보 경로가 실제로 지나가는 image
feature만 sparse sampling한다.

### 6.1 입력

v1 기본 입력:

| 입력 | 크기 | 비고 |
|---|---|---|
| current 6-camera | 768×432 | 필수, 동일 backbone |
| history t=-0.5s 6-camera | 384×216 | visual motion용, shared backbone |
| calibration | camera intrinsic/extrinsic | projection에만 사용 |
| relative pose | t=-0.5→0 transform | 고정 좌표 정렬에만 사용 |

7개 frame을 순차 forward하지 않는다. current와 선택한 history를 한 번의 model forward
안에서 batch로 encode한다. history가 효과 없으면 제거해 current-only로 제출한다.

history 확장 순서는 `-0.5s` 한 장을 먼저 검증하고, 유의한 개선이 있을 때만 `-1.0s`를
추가한다. 과거를 사용한 모든 시점의 영상은 함께 입력한다.

### 6.2 image encoder

기본안:

- ResNet34 또는 동급 latency backbone
- current: C3/C4 feature, FPN 128 channels
- history: 동일 weight, 저해상도 C3/C4
- BN은 SyncBN 대신 frozen BN 또는 GroupNorm ablation
- 최종 제출에서는 detection/map용 dense decoder를 제거

대안은 ResNet18, ConvNeXt-Tiny, EfficientNet 계열을 latency-quality Pareto로 비교한다.
backbone 선택 기준은 ImageNet 이름값이 아니라 3090 전체 wrapper latency와 val top-1이다.

### 6.3 sparse multi-camera path sampler

각 후보 waypoint를 ego ground point로 보고 camera calibration으로 각 image plane에
projection한다.

```text
candidate point (x,y,z_ground)
  -> ego-to-camera projection
  -> visibility mask
  -> FPN level별 bilinear sample
  -> camera/scale aggregation
```

history feature는 candidate point를 `T_history_from_current`로 먼저 옮긴 뒤 같은 방식으로
sample한다. relative transform 값은 feature로 concat하지 않는다.

초기 sampling 예산:

- 10 waypoint × 2 FPN levels
- projection 주변 4 offset point
- 보이는 camera만 aggregate
- aggregation dimension 128

camera aggregation은 goal-free learned attention 또는 masked mean/max를 사용한다. 후보
geometry embedding을 score에 직접 더하는 camera-free prior는 피한다. geometry는 주로
sampling address와 정적 bank metadata에만 쓴다.

### 6.4 visual motion branch

종방향 오차가 현재 전체 오차의 77–86%이므로 가장 중요한 branch다.

권장 연산:

1. current/history sparse sample을 동일 candidate location에 정렬한다.
2. `[f_now, f_hist, f_now-f_hist, f_now*f_hist]`를 작은 bias-free MLP에 넣는다.
3. path별 진행 가능성, stop/creep, velocity profile logits을 예측한다.

raw ego speed는 입력하지 않는다. train pose speed는 auxiliary supervision으로만 쓴다.
영상에서 선행차량, 정체, 교차로, optical motion을 읽도록 만든다.

### 6.5 coarse-to-fine scorer

#### Coarse path scorer

- geometry bank P의 sparse image evidence를 조회
- path logits `[B,P]`
- top 12–16 path 선택

#### Coarse velocity scorer

- global front visual motion과 sparse temporal evidence 사용
- velocity logits `[B,V]`
- top 4–8 profile 선택
- stop/creep profile을 shortlist에 최소 1개 보존

#### Fine pair scorer

- 선택된 path×velocity 64–128개를 실제 10-waypoint candidate로 조합
- 조합 candidate 위치에서 image feature를 다시 sparse sample
- local evidence, temporal motion, global scene context를 합쳐 final visual score 출력

모든 top-k는 visual score로만 결정한다. goal, command, status는 coarse/fine 어느 단계에도
들어가지 않는다.

### 6.6 residual head

v1에서는 residual을 끈다. fixed bank + ranker가 먼저 수상권에 들어가는지 확인한다.

추후 허용 조건:

- image-only 입력
- candidate별 최대 residual norm 제한
- 10-waypoint 후보를 selector 전에 완성
- residual on/off로 3초 L2 개선이 0.005 이상
- latency 증가 10ms 이하
- goal counterfactual 후보 bitwise invariant

Diffusion residual은 이 단계에서만 1-step 또는 truncated 2-step으로 비교한다. 현재
K=1024 oracle 0.0996이므로 처음부터 diffusion proposal generator를 쓰는 것은 계산 낭비다.

---

## 7. 외부 selector 상세

### 7.1 기본 알고리즘

selector 입력은 모델이 완성한 후보와 visual score, 그리고 공식 제공 goal/command다.

```python
def select_completed_candidate(candidates_5s, visual_logits, route_meta,
                               vad_cmd, goal_xy, m=3):
    # 1) 모델의 image-only 순위가 항상 먼저다.
    ranked = stable_descending_argsort(visual_logits)

    # 2) 기본은 global top-M. route-balanced 방식은 별도 사전등록 ablation이다.
    shortlist = ranked[:m]

    # 3) command는 shortlist 안의 정적 route metadata를 필터링하는 데만 사용한다.
    compatible = [i for i in shortlist if route_meta[i] matches vad_cmd]
    pool = compatible if compatible else shortlist[:1]

    # 4) 이미 완성된 5초 endpoint가 goal에 가장 가까운 row의 index만 선택한다.
    selected = min(pool, key=lambda i: norm(candidates_5s[i, 9] - goal_xy))
    return candidates_5s[selected, :6]
```

실제 구현은 vectorized하되 의미는 위와 같아야 한다.

### 7.2 기본 M=3

ideal visual ranking에서 M=1은 0.0996, M=3은 0.1220, M=6은 0.1439였다. goal 선택
오차가 M과 함께 증가하므로 기본값은 3이다. M은 `{1,2,3,4,6}`을 mini-val에서 한 번
선택하고 final val을 본 뒤 다시 바꾸지 않는다.

### 7.3 route-balanced 대안

global top-6의 85.6%가 같은 횡방향 갈래였으므로 다음 대안을 비교한다.

- 모델이 goal과 무관하게 route bucket별 top-3 후보를 출력
- selector가 `vad_cmd` bucket을 고름
- 해당 bucket의 top-3 안에서 endpoint goal 선택

이 구조도 출력 후보와 score는 goal/command에 bitwise invariant여야 한다. global top-3보다
mini-val 개선이 명확할 때만 채택한다.

### 7.4 영상 증거가 약할 때의 fallback

full-K goal-only가 되지 않도록 visual confidence가 낮을 때 goal pool을 넓히지 않는다.
기본 fallback은 visual top-1이다. zero/constant image에서 logits가 동률이면 stable tie-break로
index 0 exact-stop을 선택한다.

---

## 8. Loss 재설계

### 8.1 공식 3초 거리

후보 c와 GT b의 누적 3초 거리는 다음이다.

```text
D[b,c] = Σ_i w_i ||C[c,i] - G[b,i]||₂
w = [11, 11, 5, 5, 2, 2] / 36
```

candidate와 GT가 증분이면 반드시 둘 다 `cumsum` 후 계산한다. mask 처리 방식은 기존
challenge metric 구현과 동일하게 유지하고, 5초 label이 불완전한 sample은 tail loss에서
제외한다.

### 8.2 현재 loss의 문제

현재 구현은 다음 full-K soft CE다.

```python
target_prob = softmax(-D / 0.1)
loss = -(target_prob * log_softmax(logits)).sum()
```

K=1024 전체에 확률을 뿌려 비슷한 후보 집합을 학습하지만, top-1과 hard negative의
순서를 직접 강제하지 않는다.

### 8.3 권장 loss 구성

#### A. Positive-set probability loss

```text
Dmin = min_c D[c]
P = {c | D[c] ≤ Dmin + ε_abs}
L_pos = -log Σ_{c∈P} softmax(logit/τ_score)[c]
```

초기값: `ε_abs=0.05m`, `τ_score=0.1`.

단 하나의 cluster index만 정답으로 두면 거의 같은 후보끼리 불필요하게 경쟁하므로
metric상 같은 positive set 전체의 확률 질량을 올린다.

#### B. Hard-negative pairwise ranking

positive는 P 안에서 score가 가장 높은 후보, negative는 visual top-32 중 P 밖이고
`D ≥ Dmin + 0.10m`인 후보를 쓴다.

```text
margin(p,n) = m0 + β × clamp(Dn-Dp, 0, dmax)
L_rank = mean softplus(score_n - score_p + margin(p,n))
```

초기값: `m0=0.2`, `β=0.5`, `dmax=1.0`.

#### C. Expected official L2

full K가 아니라 다음 union에서 계산한다.

```text
S = positive set ∪ GT-nearest 16 ∪ visual top-32
q = softmax(score[S] / τ_exp)
L_exp = Σ_{c∈S} q[c] × D[c]
```

초기값: `τ_exp=0.1`.

#### D. Path/velocity coarse supervision

GT와 가장 가까운 composed candidate의 path id와 velocity id를 coarse label로 사용한다.
positive-set CE를 각각 적용하며 full candidate score의 gradient가 coarse scorer에도 흐르게
한다.

#### E. 5초 tail auxiliary

공식 3초 score와 분리한다.

```text
L_tail = mean_{i=7..10} ||C_selected_or_soft[i] - G5[i]||
```

초기 weight는 `λ_tail ∈ {0.05, 0.10, 0.20}`이고 mini-val에서 선택한다. 3초 top-1이
0.003 이상 악화되면 tail weight를 낮추거나 gradient를 tail head로만 차단한다.

### 8.4 초기 총 loss

```text
L = 1.00 L_pos
  + 0.50 L_rank
  + 0.25 L_exp
  + 0.20 L_path
  + 0.20 L_velocity
  + 0.10 L_stop_creep
  + 0.10 L_tail
  + auxiliary perception losses
```

이 계수는 시작점이지 확정값이 아니다. 한 번에 전부 켜지 않고 아래 loss tournament로
각 항의 실질 이득을 확인한다.

### 8.5 loss tournament

B200 GPU 0–3을 DDP 한 판보다 독립 네 판에 쓰는 단계다.

| GPU | 실험 | 목적 |
|---:|---|---|
| 0 | `L_pos` | hard target 대비 positive-set 효과 |
| 1 | `L_pos + L_rank` | top-1 직접 분리 효과 |
| 2 | `L_pos + L_rank + L_exp` | metric 직접 최적화 효과 |
| 3 | GPU2 + velocity/stop auxiliary | 종방향 시각 신호 효과 |

동일 초기 checkpoint, seed, sample order, update 수를 사용한다. epoch이 아니라 동일 processed
sample 수로 비교한다.

---

## 9. 학습 전략

### 9.1 Stage 0: bank oracle 검증

GPU 학습 전에 CPU로 끝낸다.

- legacy K=1024 3초 oracle 재현
- anchor-preserving tail bank oracle
- path×velocity P/V sweep
- speed/route/stop bucket별 coverage
- M별 ideal-selector upper bound

이 단계에서 3초 oracle이 나쁜 bank는 모델로 보상하려 하지 않고 폐기한다.

### 9.2 Stage 1: cached/frozen ranker tournament

현재 epoch-4 trunk에서 candidate sampling feature를 cache하거나 trunk를 freeze한다.

- loss 구조만 빠르게 비교
- 최소 3 seed는 최종 2개 variant에만 실행
- model selection은 `top1 L2`와 `top1-oracle@3 gap`
- training loss나 oracle 단독으로 checkpoint를 고르지 않음

### 9.3 Stage 2: end-to-end fine-tuning

가장 좋은 ranker 하나만 image encoder까지 연다.

권장 LR 시작점:

| module | LR |
|---|---:|
| new scorer / velocity / tail head | 2e-4 또는 4e-4 |
| sparse sampler projection | 1e-4 |
| FPN | 2e-5 |
| backbone stage 4 | 1e-5 |
| backbone stage 1–3 | 초기 freeze |

global batch가 4에서 16으로 바뀐다고 LR을 자동으로 4배 올리지 않는다. head LR은
`{2e-4, 4e-4, 8e-4}` mini-val 짧은 sweep으로 선택한다.

### 9.4 Stage 3: visual history

current-only checkpoint에서 시작한다.

1. t=-0.5s 저해상도 history 추가
2. 모든 weight와 schedule을 같게 둔 paired comparison
3. 종방향 L2와 stop/creep bucket이 개선되는지 확인
4. raw L2 개선이 0.005 미만이거나 3090 latency가 15ms 넘게 늘면 제거

### 9.5 Stage 4: distillation과 ensemble

1536 teacher는 후보 ranking distribution과 visual progress label을 768 student에 distill한다.
teacher의 goal-conditioned output은 사용하지 않는다.

최종 ensemble은 별도 full model 여러 개를 실행하면 latency를 초과하므로 금지한다.
대신 다음만 허용한다.

- SWA 또는 checkpoint averaging
- 한 shared trunk 위 작은 ranker head 2–3개의 logit 평균
- latency 증가가 5ms 이하인 경우

---

## 10. B200 GPU 0–3 실행 규칙

### 10.1 검증된 환경

```text
GPU: NVIDIA B200 183,359 MiB × 8
사용: GPU 0,1,2,3
Topology: GPU 0–3 동일 NUMA 0, 상호 NV18
PyTorch: 2.7.1+cu128
Compute capability: sm_100
NCCL: 2.26.2
```

검증 결과:

- MultiScaleDeformableAttention CUDA forward/backward 통과
- 4-GPU NCCL collective 정합성 통과
- 실제 VAD 4-GPU 4 iteration 학습 완주
- GPU당 peak VRAM 약 20.9GB
- warm iteration 0.91–1.45s, global batch 16
- NaN/NanSkip 없음

### 10.2 필수 환경변수

이 노드에서는 blocking NCCL communicator 초기화가 멈췄다. 반드시 다음을 사용한다.

```bash
export ADCL_BASE=/NHNHOME/data/sukim/adcl
export ADCL_REPO=$ADCL_BASE/dense_vocab_v1
export ADCL_PY=$ADCL_BASE/env/venv/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_NCCL_USE_COMM_NONBLOCKING=1
export OMP_NUM_THREADS=4
export PYTHONPATH=$ADCL_REPO
```

DDP config에는 현재 꺼진 auxiliary head가 있으므로 다음이 필요하다.

```python
find_unused_parameters = True
```

최종 ScoreDrive에서 미사용 head를 제거한 뒤에는 `False`로 되돌려 DDP overhead를 없앤다.

### 10.3 4-GPU DDP launch template

```bash
cd "$ADCL_REPO"

TORCH_NCCL_USE_COMM_NONBLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMP_NUM_THREADS=4 \
"$ADCL_BASE/env/venv/bin/torchrun" \
  --standalone --nproc_per_node=4 \
  tools/train.py projects/configs/ScoreDrive/<CONFIG>.py \
  --launcher pytorch \
  --work-dir "$ADCL_BASE/work_dirs/<RUN>" \
  --no-validate --deterministic --seed 0
```

production은 `setsid nohup`으로 분리하고 PID를 명시적으로 기록한다. `pkill -f`는 사용하지
않는다.

### 10.4 batch와 schedule

현재 2Hz train anchor는 17,820개다.

| samples/GPU | global batch | iter/epoch | 용도 |
|---:|---:|---:|---|
| 1 | 4 | 4,455 | H200 batch-4와 update 수 비교 |
| 2 | 8 | 2,228 | 중간 안정성 |
| 4 | 16 | 1,114 | 최종 throughput 기본값 |

기존 batch-4 warmup 500 iteration은 2,000 sample이다. global batch 16에서는 동일 sample
기준 warmup이 약 125 iteration이다. cosine schedule과 checkpoint 간격도 iteration 수가
아니라 processed sample/epoch 기준으로 맞춘다.

### 10.5 I/O

첫 smoke iteration은 Lustre cold-read 때문에 data time이 약 13초였고 이후 0.15–0.71초로
떨어졌다. 장시간 학습에서 data time 비율이 30%를 넘으면 다음 순서로 대응한다.

1. worker를 rank당 1→2→4로 sweep
2. current/history cache를 `/tmp` 로컬 1.2TB tmpfs 또는 로컬 scratch에 stage
3. small file metadata 병목을 WebDataset/tar shard로 줄임
4. GPU utilization과 data time을 함께 기록

---

## 11. 평가 표준

### 11.1 매 checkpoint 필수 지표

```text
official top-1 L2
oracle@1, @3, @6, @20
top1 - oracle@3 gap
GT-nearest candidate top-M hit rate
median / p90 GT candidate rank
1s, 2s, 3s L2
x/y axis decomposition
stop / creep / moving / speed bucket
left / right / lane-keep / U-turn bucket
scenario-cluster bootstrap 95% CI
```

frame≥30, deployment-faithful depth를 사용한다. 평가 index는 원시 PKL이 아니라
`ds.data_infos` 정렬 결과에서 만든다.

### 11.2 model selection 기준

우선순위:

1. official top-1 L2
2. `top1 - oracle@3` gap
3. 시나리오 bootstrap 유의성
4. 4090 예상 Error score
5. latency/FLOPs

oracle만 좋아지는 checkpoint는 선택하지 않는다. 차이가 ±0.001 이내면 동률로 본다.

### 11.3 단계별 gate

| 단계 | 통과 조건 |
|---|---|
| bank | D3 oracle ≤ legacy oracle + 0.003 |
| loss 1차 | 동일 sample에서 기존 대비 top-1 10% 이상 개선 |
| loss 채택 | top1-oracle@3 gap 20% 이상 축소 |
| history | 종방향/stop 개선, 전체 L2 -0.005 이상 |
| final val | raw L2 ≤0.140, 목표 ≤0.128 |
| latency | 3090 full wrapper ≤150ms, 최종 4090 ≤100ms 목표 |
| compliance | 모든 구조/불변성 gate PASS |

### 11.4 제출 횟수 사용 조건

이미 1회 사용했고 남은 제출은 최대 4회로 본다. 다음을 모두 만족하기 전에는 제출하지
않는다.

- 규정 준수 코드 계보
- final val L2가 최소 0.145 이하
- val-test gap 검증 방식 유지
- latency penalty를 반영한 예상 Error가 현재 3위권과 경쟁 가능
- submission JSON sanity/trajectory plot/FLOPs/compliance gate 통과

---

## 12. Compliance test suite

### C1. Signature audit

production model의 `forward`, `simple_test`, planner/scorer 호출 graph에서 goal/command/status
인자가 없어야 한다.

### C2. Goal/command counterfactual

동일 image에 goal/command만 100회 무작위 변경한다. 모든 candidate와 visual logit이
bitwise 동일해야 한다.

### C3. Selector row identity

최종 output 10-point path가 candidate tensor의 정확히 한 row와 bitwise 동일해야 한다.
첫 6개를 자르는 것 외 좌표 연산이 없어야 한다.

### C4. Image ablation

- image-zero
- constant image
- clip shuffle
- current shuffle
- history shuffle
- camera permutation with calibration matched/mismatched
- image feature channel mean/shuffle

원본 대비 L2, candidate index change rate, rank correlation을 모두 보고한다. 단순히 zero
출력이 0이라는 한 가지 지표만으로 규정 준수를 주장하지 않는다.

### C5. Goal-only negative control

visual logits를 제거하거나 shuffle했을 때 selector 성능이 크게 붕괴해야 한다. full-K
goal-only L2도 보고하고 production 경로에는 절대 연결하지 않는다.

### C6. Pose-path audit

relative pose가 좌표 projection 행렬 곱 외에 tensor concat, embedding, attention 입력으로
들어가는지 정적 grep과 runtime hook으로 확인한다.

### C7. Reset/depth audit

각 test clip 시작 시 state를 초기화한다. 선택한 history frame 수와 간격이 train/eval/test에서
동일해야 한다.

### C8. Timing boundary

stream reset 후 model의 모든 forward를 합산한다. 일부 history feature를 미리 계산한 뒤
타이머 밖으로 빼지 않는다.

---

## 13. Latency/FLOPs 설계 예산

목표 4090 100ms를 다음처럼 나눈다.

| 구성 | 4090 목표 |
|---|---:|
| current 6-camera backbone/FPN | 45ms |
| low-res history backbone/FPN | 15ms |
| projection + sparse sampling | 12ms |
| coarse path/velocity scoring | 5ms |
| fine 64–128 candidate scoring | 10ms |
| tensor assembly/output | 3ms |
| 여유 | 10ms |

실측은 3090에서 다음 프로토콜로 한다.

```text
real 6-camera tensors
실제 선택 history 포함
batch=1
model.eval(), inference_mode, AMP
50회 warmup
CUDA event와 wall clock 동시 측정
각 반복 전후 torch.cuda.synchronize()
median, p95, peak allocated/reserved VRAM
pre/post-processing 제외, 모든 model forward 포함
```

목표 proxy:

- 3090 median ≤150ms
- 3090 p95 ≤165ms
- peak VRAM ≤18GB, 4090 24GB 안전
- FLOPs ≤7053G는 하드 컷이며, 실제 설계 목표는 1000G 이하

---

## 14. 구현 파일 구조

기존 `dense_vocab_v1`을 직접 계속 덧대지 않고 새 계보를 만든다.

```text
/NHNHOME/data/sukim/adcl/scoredrive/
├── projects/
│   ├── configs/ScoreDrive/
│   │   ├── scoredrive_bank_ablation.py
│   │   ├── scoredrive_rank_v1.py
│   │   ├── scoredrive_temporal_v1.py
│   │   └── scoredrive_final.py
│   └── mmdet3d_plugin/ScoreDrive/
│       ├── __init__.py
│       ├── model.py
│       ├── image_encoder.py
│       ├── sparse_path_sampler.py
│       ├── candidate_bank.py
│       ├── path_velocity_head.py
│       ├── fine_ranker.py
│       ├── losses.py
│       └── outputs.py
├── tools/
│   ├── data_converter/etri_5s_labels.py
│   ├── etri_scoredrive_train.py
│   └── etri_scoredrive_submit.py
├── scripts/
│   ├── build_5s_bank.py
│   ├── sweep_factorized_bank.py
│   ├── eval_ranking.py
│   ├── eval_selector.py
│   ├── compliance_gate.py
│   ├── latency_full_wrapper.py
│   └── render_experiment_table.py
├── tests/
│   ├── test_metric_exact.py
│   ├── test_3s_prefix_preserved.py
│   ├── test_goal_invariance.py
│   ├── test_selector_row_identity.py
│   ├── test_pose_alignment_only.py
│   └── test_reset_and_timing.py
└── artifacts/
    ├── banks/
    ├── manifests/
    └── reports/
```

새 계보 시작 시 baseline snapshot을 git commit하고, 생성 artifact에는 입력 PKL hash,
script hash, split hash, random seed를 manifest로 저장한다.

---

## 15. 구현 순서와 예상 판정

### Phase A — 3초 ranking loss, 가장 먼저

1. 현재 `AnchorGroundedVocabularyDecoder.vocabulary_loss`를 새 loss 모듈로 분리
2. 기존 K=1024, 모델 구조, 데이터, checkpoint 고정
3. GPU 0–3 loss tournament
4. top-1과 gap이 줄어드는지 확인

성공 시: 현재 진단이 맞고 다음 단계 진행.  
실패 시: score architecture 또는 visual speed feature가 주병목이므로 dense BEV 학습을
더 돌리지 않고 sparse temporal scorer로 바로 이동.

### Phase B — anchor-preserving 5초 extension

1. 10-waypoint train label 생성
2. K1024별 multi-tail bank 생성
3. 첫 6개 bitwise prefix test
4. image top-M 후 goal endpoint selector 평가

성공 기준: ideal M=3 약 0.122 재현, 실제 M=3이 top-1보다 명확히 개선.

### Phase C — factorized bank

1. P/V CPU oracle sweep
2. coarse path/velocity head
3. fine pair scorer
4. legacy bank와 동일 loss 조건 비교

factorized bank가 oracle 또는 실제 top-1을 개선하지 않으면 복잡도 때문에 채택하지 않는다.

### Phase D — latency-first ScoreDrive trunk

1. ResNet34 current-only sparse sampler
2. 3090 latency 측정
3. 기존 frozen VAD feature와 accuracy 비교
4. t=-0.5 low-res visual motion 추가

### Phase E — final fine-tune/distill

1. best bank + best rank loss + best temporal branch 고정
2. B200 4-GPU DDP full run
3. 3 seed 중 best single 또는 SWA
4. final val/compliance/latency/submit gate

---

## 16. 중단해야 하는 실패 패턴

- oracle@M만 좋아지고 top-1이 2회 연속 거의 그대로
- 5초 tail loss를 켠 뒤 첫 3초 top-1 또는 oracle이 0.003 이상 악화
- M을 늘릴수록 goal selector가 좋아진다는 이유로 full-K 검색으로 확대
- image-zero만 통과하고 clip-shuffle/feature-shuffle가 거의 무반응
- velocity head가 visual history를 shuffle해도 무반응
- 1536이 768 대비 raw L2 개선보다 latency penalty가 큼
- final 3090 proxy가 150ms를 넘는데 raw L2가 0.11보다 높음
- final val에서 threshold/scale을 다시 적합하려는 시도
- test input 통계로 bank, stop threshold, route prior를 수정
- goal/command가 scorer 함수 인자에 다시 등장

---

## 17. 현재 B200 재현 자산

```text
최신 소스
/NHNHOME/data/sukim/adcl/dense_vocab_v1

B200용 현재 config
/NHNHOME/data/sukim/adcl/dense_vocab_v1/projects/configs/VAD/
VAD_etri_anchorvocab_k1024_wide_b200.py

K=1024 anchor
/NHNHOME/data/sukim/adcl/h200_latest/artifacts/dense_vocab/
etri_train330_vocab_k1024.npy
sha256 04c5f2fc20dffd489907d2f7a9e550a11ee2895974c6f276736d70bb3b074ced

768 epoch-4 checkpoint
/NHNHOME/data/sukim/adcl/checkpoints/anchorvocab_k1024_wide/epoch_4.pth
sha256 2ec201e4461e3544dcd239d1065b6b5c7cbf7cb1f37b02f77565618cfd12ac53

환경
/NHNHOME/data/sukim/adcl/env/venv/bin/python

4-GPU smoke log
/NHNHOME/data/sukim/adcl/work_dirs/
anchorvocab_k1024_wide_b200_smoke/torchrun_smoke3.log

단일 scenario forward 결과
/NHNHOME/data/sukim/adcl/results/b200_anchorvocab_ep4_smoke.json
```

---

## 18. Definition of Done

최종 후보는 아래를 전부 만족해야 제출 가능하다.

- [ ] model inference signature에 goal/command/status가 없다.
- [ ] candidate와 visual score가 goal/command counterfactual에 bitwise invariant다.
- [ ] selector output은 완성 candidate 한 row와 bitwise 동일하다.
- [ ] 3초 prefix metric이 공식 metric crosscheck와 일치한다.
- [ ] train-only bank와 artifact manifest가 재현 가능하다.
- [ ] final val raw L2 ≤0.140, 목표 ≤0.128이다.
- [ ] scenario bootstrap에서 개선이 유의하다.
- [ ] image/feature/temporal ablation에서 영상의 실질 기여가 확인된다.
- [ ] 3090 full-wrapper median ≤150ms이고 4090 100ms 목표가 타당하다.
- [ ] FLOPs 7053G 미만이다.
- [ ] stream reset부터 최종 출력까지 모든 forward가 timing에 포함된다.
- [ ] test 데이터를 학습, threshold fitting, 연결 복원에 쓰지 않았다.
- [ ] 제출 JSON, 좌표 누적/증분, waypoint 수, NaN/Inf 검사가 통과한다.
- [ ] 제출 코드에서 금지된 보정, interpolation, goal-conditioned planner가 제거됐다.

최종 성공 판단은 raw L2 하나가 아니라 **규정 준수 × top-1 정확도 × 4090 latency**의
곱이다. 현재 가장 큰 즉시 개선점은 GPU 규모가 아니라 ranking loss이며, B200 4장은
loss/architecture 가설을 빠르게 기각하고 최종 한 구조를 충분히 학습하는 데 사용한다.
