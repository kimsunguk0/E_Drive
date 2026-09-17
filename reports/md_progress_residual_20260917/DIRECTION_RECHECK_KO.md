# MotionDrive 방향 재점검 — local-minimum 탈출판

작성일: 2026-09-17 19시대 KST

## 결론

`FRONT residual → SIDE → AUX → denoise → A2 warmup`은 같은 등록 MR 출력에 작은
보정기를 붙이는 계열 안에서 움직였다. scalar oracle이 `0.080906`인데도 실제 일반
주행 개선이 `0.001` 아래였으므로, 이 계열에서 head 크기·카메라·loss를 조금씩 바꾸는
것은 주공격안이 될 수 없다.

새 방향은 두 문제를 동시에 끊는다.

1. causal status는 planner/residual의 raw 인자가 아니라 occupancy·lane·state/history·
   planning이 공유하는 **영상 feature 형성 단계**에만 사용한다.
2. planner의 최종 XY를 **경로 방향과 진행량으로 구조적으로 분리**하고, 기존 XY head가
   종방향 오차를 대신 흡수하지 못하게 한다.

GPU 1에는 같은 MR 초기화에서 A2 query를 처음부터 함께 학습하는 `A2-DIRECT`, GPU 3에는
status-conditioned shared image feature와 연속 `delta_v+delta_a` progress planner를 함께
학습하는 `A3-FP-VA`를 시작했다. GPU 0의 FULL 제출 안전망과 GPU 2의 honest old203→new107
producer는 유지한다.

## 목표와 현재 간극

| 항목 | 값 |
|---|---:|
| 등록 MR V0 PREFIX | 0.191002 |
| 등록 MR 서버 | 0.197988 |
| 현재 3위 서버 | 0.130537 |
| 서버 절대 감소 필요 | 0.067451 |
| 한 점 환산 DEV 공격 목표 | 약 0.12355 |

공식 PREFIX는 여섯 waypoint에 `[11,11,5,5,2,2]/36`의 가중치를 둔다. 서버 점수에서
0.5/1.0초가 `0.069466`, 1.5/2.0초가 `0.077504`, 2.5/3.0초가 `0.051017`을 차지한다.
후반만 고쳐서는 3위에 갈 수 없고, 첫 2초의 진행량을 함께 고쳐야 한다.

## 작은 residual 계열을 닫는 근거

| 실험/진단 | 전체 V0 | 일반 주행 1,875행 |
|---|---:|---:|
| 등록 MR | 0.191002 | 0.196742 |
| deployable scalar oracle | **0.080906** | **0.085242** |
| deployable `delta_v+delta_a` oracle | **0.063959** | - |
| FRONT joint step2616 final/base | 0.193752 / 0.197317 | residual 이득 0.000389 |
| SIDE-S-AUX joint step2616 final/base | 0.193830 / 0.197238 | residual 이득 0.000278 |
| SIDE-S-AUX-DN step2616 final/base | 0.194594 / 0.210242 | 최종은 등록보다 악화 |

마지막 A2 shared-scene warmup도 두 seed에서 같은 패턴이었다.

| seed | base | final | final-base | nonstop 개선 | 평균 `|c|` |
|---|---:|---:|---:|---:|---:|
| 1 | 0.191341 | 0.189591 | -0.001750 | -0.000960 | 0.00960 m/s |
| 2 | 0.190334 | 0.189498 | -0.000836 | -0.000257 | 0.00394 m/s |

개선은 steady 99행과 depart 24행에 집중됐고 점수 대부분인 nonstop은 거의 그대로였다.
oracle 평균 `|c|`는 약 `0.142 m/s`인데 head는 3~7%만 사용했다. 이 결과로 A2의 긴 joint,
motion-gain adapter, 더 큰 MLP를 연속 실행하지 않는다.

## warmup 실패가 status 자체의 실패는 아니다

과거 깨끗한 비교에서 동일한 flip50 조건의 제공-status A2는 `0.228957`, status 경로를
제거한 Q10 모델은 `0.258544`였다. 제공 status가 shared query에서 빠지는 비용은
`+0.02959`, 11-session CI도 0을 포함하지 않았다. 반면 현재 warmup은 완성된 등록 MR에
zero-init query/refiner를 붙이고 base를 동결한 뒤 1,000 update만 학습했다.

따라서 정확한 결론은 다음과 같다.

- 사후 보정 warmup은 일반 주행 residual을 배우지 못했다.
- status-conditioned shared perception은 처음부터 planner와 함께 co-adapt할 때 의미 있는
  이득을 낸 전력이 있다.
- 이를 현재 MR 해상도·데이터 확대와 결합한 fresh 학습은 아직 실행된 적이 없었다.

이 때문에 `A2-DIRECT`를 작은 보정기가 아닌 20,554-update fresh control로 다시 둔다.

## 현재 구조가 가진 두 개의 학습 병목

### 1. monolithic XY가 progress loss를 흡수한다

기존 residual 최종식은 대략 다음이다.

```text
final = base_XY + detached_basis(base_XY) * correction
```

기저만 detach해도 `base_XY`의 직접 gradient는 남는다. joint 학습에서는 기존 XY head가
진행량 오차를 계속 흡수할 수 있고, 새 correction head는 작은 계수에 머문다. 실제 A2
scene seed들의 평균 계수가 oracle보다 한 자릿수 이상 작았던 현상과 일치한다.

### 2. in-sample base가 correction 분포를 지운다

등록 MR은 train probe `0.089975`, V0 `0.191002`다. scalar oracle `|c|`도 train에서
`0.04745 m/s`, V0에서 `0.14160 m/s`다. 등록 base가 이미 맞힌 같은 train 행으로 보정기를
학습하면 새 세션에서 필요한 큰 부호·크기를 거의 보지 못한다.

GPU 2의 old203 producer는 old203만 학습하고 한 번도 보지 않은 new107을 예측한다. 이 출력은
후속 progress 학습에 honest error distribution을 주기 위한 것이며, 그 자체가 최종 모델은
아니다.

## A3-FP-VA의 실제 구조

### 정보 경로

```text
causal pose status(t-1.0s..t)
          ├─> shared scene-attention query
          └─> multiplicative gates on image FPN features
                         │
camera/current/history image features
                         ↓
       shared scene + motion representation
          ├─> occupancy / lane
          ├─> image state / history
          └─> planner decoder
                    ├─> proposal directions
                    └─> continuous progress head
```

progress head에는 raw goal, pose, provided status가 없다. planner가 이미 읽은 image-derived
decoded feature와 모델 자신의 detached proposal 좌표/구간 길이만 들어간다. status gate는
곱셈형이며 초기값 1이라 status 자체가 image value를 새로 만들지 않는다.

### 경로/진행량 분리

proposal 구간을 `delta p_k`, 단위 방향을 `u_k`, 길이를 `ell_k`라 하면:

```text
u_k = delta p_k / ||delta p_k||
ell0_k = stop_gradient(||delta p_k||)
ell_k = max(0, ell0_k + 0.5 * (delta_v + delta_a * t_mid_k))
p_k = cumulative_sum(u_k * ell_k)
```

proposal에는 방향 gradient만 남고 길이 gradient는 차단된다. 따라서 종방향 loss가 기존
XY magnitude 경로로 빠질 수 없고 progress head가 실제로 풀어야 한다. 정지 proposal은
ego-forward `+x`, 짧은 segment는 가장 가까운 유효 predicted tangent를 사용한다. 모든
연산은 neural forward 안에 있고 GT나 후보 선택 후처리가 없다.

### 과거 C/P×V와 다른 점

과거 P×V는 512 path × 128 velocity의 65,025개 고정 후보를 32차원 context로 argmax하는
분류 문제였다. bank oracle `0.104887` 또는 다른 C 진단의 더 낮은 oracle이 있어도 selector
regret와 train–tune 과적합이 컸다. A3는 candidate bank, shortlist, argmax가 없는 연속
저차원 회귀이고, 이미 강한 MR의 경로 방향을 보존한다.

## 규정 판단

`OPEN_ISSUE.md` Q6은 과거 pose로 현재 status를 계산하는 것을 허용한다. Q7은 status를
planner에 직접 또는 단순 임베딩으로 넣는 것을 금지하면서 여러 task의 공통 특징을 향상하는
간접 사용은 허용한다. A3는 후자를 구현한다.

다만 detach나 planner 함수 signature가 정보 의존성을 지우는 것은 아니다. 전체 계획은 causal
status와 shared-feature goal conditioning의 간접 영향을 받는다. protocol에 이 경로를 그대로
기록했고, 최종 허용 판단이 코드 심사에 있다는 경계도 유지한다.

## 검증과 현재 GPU 배치

새 factorized geometry 단위검사 5개가 통과했다. 두 arm의 실제 2-step smoke도 종료됐고,
첫 optimizer step의 planning loss가 둘 다 `0.6594299078`로 같았다. 즉 zero-init query,
multiplicative gates, progress output이 초기 MR 함수를 보존했다. A3 progress 계수는 2-step에서
0이 아닌 값으로 변해 gradient 연결도 확인됐다.

| GPU | 현재 실행 | 역할 |
|---|---|---|
| 0 | `MR-NATIVE-FULL-s1`, 24,931 update | 제출 안전망; 내부 tune은 in-fit 진단뿐 |
| 1 | `A2-DIRECT-s1`, 20,554 update | status shared-query fresh control |
| 2 | `OOF-MR-T203-s1`, 20,554 update | old203→new107 honest error producer |
| 3 | `A3-FP-VA-s1`, 20,554 update | shared dynamics + 강제 path/progress 분리 주력 |

등록 MR seed1의 비교 곡선은 step 3426/6852/10278/13704/17130/20554에서 각각
`0.254811/0.218241/0.222304/0.206502/0.192196/0.191002`다. 새 두 run도 같은 시점의
plain V0만 비교한다. 중간 곡선이 비단조였으므로 한 checkpoint만 보고 조기 종료하지 않는다.

## 다음 판정

- A2가 크게 좋아지고 A3가 뒤처지면 shared-status co-adaptation은 유지하고, OOF 출력으로
  scalar factorization을 다시 학습한다.
- A3가 A2를 이기면 scalar/두 번째 seed로 재현한 뒤 FULL 계보를 만든다.
- 둘 다 등록 MR 근처면 작은 adapter 탐색으로 돌아가지 않는다. 그때는 OOF에서 correction이
  실제로 학습 가능한지 판정하고, 불가능하면 이번 데이터에서 0.13대 근거가 없다고 인정한다.

현재 `0.13`을 뒷받침하는 실현 모델은 아직 없다. 다만 oracle 표현력, 과거 A2의 `-0.0296`,
그리고 direct base gradient 누수라는 구조적 결함을 한꺼번에 겨냥하는 첫 실험이 지금의 A3다.
이는 최근 작은 residual 변형들과 다른 축이며, 3위 격차를 노릴 수 있는 최소한의 구조 변경이다.
