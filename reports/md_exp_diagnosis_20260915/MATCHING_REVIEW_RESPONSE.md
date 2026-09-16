# matching 리뷰 대응 — 정정, 재측정, MR 대조 착수

## 1. 제 과한 주장을 정정한다

"−0.1 s 이동량이 cell의 1/11이므로 sub-cell → 근접 관측 추가도 무가치"는 **틀린 추론이다.**
0.0925 cell의 이동이 feature와 이웃 correlation 값에 0으로 나타난다는 보장이 없다.
연속 feature와 연속 readout은 격자보다 작은 변위를 회귀할 수 있다
(RAFT가 1/8 격자에서 bilinear lookup과 연속 flow update로 하는 일이 그 예다.
이는 stride = 최소 분해능이라는 등식에 대한 반례이지 우리 D3 성능의 증거가 아니다).

**정확한 결론**: 짧은 간격의 이동이 feature 격자 간격보다 작으므로
**미세 변위 민감도를 개선할 가치가 있다.** "정보가 없다", "H6-NEAR는 무가치"는 채택하지 않는다.

## 2. 초과율을 다시 계산했다 — 이전 수치는 기준이 틀렸다

이전 28.1%는 유클리드 norm과 fine stride 하나만 썼다. 5×5 사각 window라면 **축별** 판정이고
level별로 stride가 다르다. 양쪽 view에서 모두 valid한 점만 세도록 고쳤다.
대상도 test 40 clip이 아니라 **V0(tune) 행 120개**로 옮겼다(test 수치는 기하 참고로만 보존).

fine 탐색 ±16 half-res px, coarse ±32 half-res px.

| Δt | 중앙값 du / dv | p90 du / dv | **fine 초과** | **coarse 초과** | 둘 다 초과 |
|---|---|---|---:|---:|---:|
| 0.1 s | 0.93 / 0.72 | 11.2 / 6.9 | 5.2% | **0.0%** | 0.0% |
| 0.2 s | 1.78 / 1.35 | 20.3 / 11.8 | 14.7% | 2.6% | 2.6% |
| 0.5 s | 3.87 / 2.73 | 39.5 / 20.9 | 26.9% | 14.4% | 14.4% |
| 1.0 s | 6.57 / 4.07 | 57.3 / 28.1 | **36.8%** | **20.8%** | 20.8% |

**정정**: 1.0 s의 초과율은 28.1%가 아니라 fine 36.8% / coarse 20.8%다.
그리고 **0.1 s는 coarse window를 전혀 벗어나지 않는다**(0.0%).
전진 운동이라 수평 성분(du)이 수직(dv)을 지배한다.

이는 **기하학적 지원 검사**이지 matching 성공률이 아니다. 텍스처·가림·회전·지면 가정은
밖에 있고, 초과율을 D3 원인 비율로 쓰지 않으며, 범위 밖 이동에 cost volume이
단조 반응한다는 보장도 없다. 이 분포에 sampler나 모델 선택을 맞추지 않는다. H는 쓰지 않았다.

## 3. MR 대조 — 설계와 사전 검사

**질문**: 같은 temporal 관측과 같은 matching graph에서, 실제 고해상도 세부가
LEN의 계획 진행량 오차를 줄이는가?

| | MR-LOWDETAIL | MR-NATIVE |
|---|---|---|
| motion canvas | 768×432 | 768×432 |
| 입력 detail | 768 → **384 bottleneck** → 768 | 768 그대로 |
| fine / coarse grid | 54×96 / 27×48 (stride 8 / 16) | 동일 |
| radius / bins | **4 / 81** 각 level | 동일 |
| 탐색 범위(원본 px) | fine ±32, coarse ±64 | 동일 |
| scene branch | 기존 384×216, 자체 backbone pass | 동일 |
| 나머지 | R0 초기값, Tplus, seed 0, 20,554 update, batch 16/micro 2, flip·photometric·nominal Δt, **length λ=0.25** | 동일 |

radius 4가 **기존 탐색 범위를 그대로 유지**하면서 양자화만 4배 세밀하게 만든다 —
해상도만 올리고 radius를 2로 두면 원본 기준 탐색 범위가 절반이 되는 함정을 피한다.

축소는 loader의 실제 계약을 그대로 따른다: PIL BILINEAR, **jitter와 정규화 이전**.
MR-NATIVE는 384를 키우는 것이 아니라 cache의 768 데이터를 실제로 소비한다.

### 사전 검사 전부 통과 (`mr_smoke.json`)

- **detail이 실제로 다르다**: 고주파 에너지 native 0.0580 대 lowdetail 0.0283 (2.05배)
- **scene branch 불변**: history 입력이 216×384로 유지됨
- **flip 구멍을 막았다**: 새 canvas 키 두 개는 `flip_item`이 모르는 키라 **거울 처리 없이
  통과했을 것**이다(검사로 확인). wrapper로 front 카메라와 같은 수평 flip을 적용하고
  이중 flip 항등을 확인했다
- correlation 2회, 둘 다 radius 4 / **81 bins**, 54×96과 27×48
- 탐색 범위 실측: fine ±32, coarse ±64 원본 px — 원래와 동일
- `motion_encoder.correlation_fuse.0`: 178 → 290 in-channel, 334,208 파라미터, **재초기화**
- 비용: microbatch 2 forward+backward 중앙값 **0.090 s**, peak 8.46 GB

### step-zero parity를 주장하지 않는다

fuse conv이 재초기화되므로 step-zero state SHA가 R0와 같을 수 없다. 이를 숨기지 않고,
trainer의 pinned SHA 검사를 **측정값으로 대체**하되 실제 보장은 따로 강제했다 —
**재초기화된 두 텐서를 제외한 모든 텐서가 R0와 bitwise 동일**함을 로드 시 assert한다.
experiment 기록에 `mr_step_zero_state.sha256_is_measured_not_pinned = true`로 남긴다.

### 착수

`MR-NATIVE-s0` (GPU 5), `MR-LOWDETAIL-s0` (GPU 6). step 50의 sample_order_sha256이
`427bfa95…`로 E1-EXP·LEN과 **동일**하다. 약 6시간 예상.

## 4. 판정 기준 (사전 고정)

주 비교 **MR-NATIVE − MR-LOWDETAIL**, 실용 비교 각 MR − 기존 LEN.

- V0 전체 D3와 시점별 L2가 중심. 다섯 조건별 D3와 **전체 평균 기여**를 분리해 본다
  (constant는 기본 label이지 엄밀한 정속이 아니다).
- `mean|b|`, 구간 길이 MAE, 방향 오차, history/readout은 보조 진단.
- 고속 밴드는 `gt_first_interval_progress_speed`로 표기.
- 같은 T0 probe와 V0를 함께 기록한다. **train만 개선되면 채택하지 않는다.**
- native가 lowdetail보다 **그리고** LEN보다 좋아야 native detail의 실용적 이득을 주장한다.
- 두 MR이 함께 좋아지면 **matching graph 변경의 효과일 수 있다** — native detail만의
  이득이라고 하지 않는다.
- native에서 state/history만 좋아지고 D3가 아니면 planning 활용은 미확인으로 닫는다.
- 효과가 없으면 **이 graph·학습 설정의 실패**로 기록한다. "영상 속도 추정의 원리적 한계"로
  확대하지 않는다.

H6-NEAR/LONG은 이번 결과만으로 폐기하지 않는다. 동시에 실행하지 않고, matching을 바꾼 결과를
해석한 뒤 관측 확대 순서를 정한다. H는 열지 않았다.
