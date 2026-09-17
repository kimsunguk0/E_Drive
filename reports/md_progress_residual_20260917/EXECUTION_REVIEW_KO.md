# MotionDrive: SIDE + progress residual 실행 전 최종 리뷰

작성일: 2026-09-17

## 0. 결론과 검증 범위

사용자가 제공한 네 가지 수정안에 대한 리뷰다. 새 커밋·학습 실행·배포형 oracle의 원시 배열을 이번에 검증한 것은 아니다. 기존 아키텍처 발췌와 Q&A 보존 문서를 읽었고, residual 충분조건, detach의 직접 경로, 영길이 tangent, 음수 진행량의 합성 예제를 계산했다.

**유지:** FULL 병행, 기존 MR + front/side temporal + 저차원 progress residual 주력, 준비되지 않은 recurrent BEV는 보류.

**실행 전 변경:**
1. Oracle은 배포에서 실제 사용하는 tangent·범위 제한·저속 처리와 동일한 함수에서 계산한다. Scalar oracle 값 하나만으로 두 계수로 자동 확장하지 않는다.
2. Residual head가 무엇을 수정하는지 알 수 있도록 detached base-plan 표현을 입력하는 것을 권한다. 이것은 raw goal/status 입력은 아니지만 goal-conditioned base의 간접 영향은 남는다.
3. `B.detach()`는 base planner나 shared backbone을 동결하지 않는다. freeze 범위와 joint 단계가 별도로 필요하다.
4. 새 static masking, 여러 pose loss, scene 재구성을 한 번에 필수화하지 않는다. 기존 supervision을 새 관측에 연결하고, 필요한 추가 loss 한 종류의 정규화·가중치를 명시한다.
5. FULL의 0.17 예상, 3.93회가 최적이라는 표현은 삭제한다. 검증된 레시피를 최종 학습 규모로 이전하는 것이지 점수 보장이 아니다.

이 확인을 위해 FULL을 보류하지 않는다. 같은 geometry 함수의 offline oracle 및 unit test가 끝나면 FRONT/SIDE 두 run을 시작한다. 새로운 대규모 감사나 모델 탐색을 추가하지 않는다.

## 1. 목표와 oracle

공식 PREFIX는 유지한다.

```
w = [11,11,5,5,2,2] / 36
t = [.5,1,1.5,2,2.5,3]
D = mean_n sum_k w[k] ||p[n,k] - y[n,k]||_2
L2_3s = mean of all six point errors
```

고정된 같은 배포형 기저 B에 대해 `p(c)=p_base+B*c`, `||B[k]||<=t[k]`이면

```
D(c_hat) <= D(c_star) + 1.25 * mean(|c_hat-c_star|)
```

동일 B에서 oracle=0.0875라는 추가 조건 아래 충분조건은:

| 목표 PREFIX | 충분한 residual MAE 상한 |
|---|---:|
| 0.15 | 0.0500 m/s |
| 0.124 | 0.0292 m/s |

필요조건도, 모델의 다음 단계 착수 문턱도 아니다. 로컬 0.124는 공격 목표로는 사용할 수 있지만, 서버에서 특정 점수를 보장하지 않는다. 이전 모델 한 번의 val–server 차이를 상수로 적용하지 않는다.

### Offline 계산은 한 producer로 제한

저장된 같은 base prediction에서 아래를 계산한다.

- 기존 ego-x scalar oracle: 과거 0.0875와의 정의 비교용.
- 실제 배포형 predicted-path tangent의 scalar oracle.
- 같은 tangent의 두 계수 oracle `(c_v,c_a)`.
- 실제 smoothing, fallback, cap, 진행량 clipping이 있는 경우 모두 적용한 최종 oracle.

정지·미세 이동 행을 숫자를 좋게 만들기 위해 제외하지 않는다. 문제 행의 처리와 개수를 별도 기록한다. coefficient의 최적화는 최종 PREFIX를 기준으로 하며, XY least-squares 해를 PREFIX 최적 oracle이라고 부르지 않는다.

Scalar oracle이 0.11~0.12라고 자동으로 두 계수를 학습하지 않는다. 두 계수에서 실제 oracle 개선이 있는지 확인한다. 방향·곡률 오류가 남는 경우 계수 추가가 별 도움이 없을 수 있다. 더 낮은 oracle이 곧 실제로 더 쉽게 예측되는 head를 뜻하지 않는다.

두 계수의 한 가지 일관된 정의는 다음이다.

```
Bv[k] = sum_{j<=k} dt * u[j]
Ba[k] = sum_{j<=k} dt * ((t[j-1]+t[j])/2) * u[j]
p_final[k] = p_base[k] + Bv[k]*cv + Ba[k]*ca
```

직선에서는 `Bv=t*u`, `Ba=.5*t^2*u`. 이때 보수적인 오차 계수는 각각 1.25초, 1.048611초²다. 여기에 등장하는 cv/ca는 base 계획에 대한 보정 계수이지 실제 현재 속도·가속도 그 자체가 아니다.

## 2. Residual head의 조건 입력

권장 형태:

```
F_raw_motion = front(/side) pair features
F_visual = goal/pose-conditioning 이전 시각 특징
p_base = 기존 goal-conditioned 공통 scene을 소비한 MR 예측

z_plan = plan_encoder(stop_gradient(p_base))
c = residual_head(F_raw_motion, F_visual, z_plan)
B = stop_gradient(stable_basis(p_base))
p_final = p_base + B*c
```

Residual은 base output에 조건부다. 같은 GT가 10m/s인 직선 예제에서 base가 10.1m/s이면 -0.1, 9.9m/s이면 +0.1이 필요하다. 영상만으로 base가 내놓은 값을 다시 암묵적으로 알아내게 하기보다, 예측 궤적의 길이·시간별 위치를 작은 학습 표현으로 제공하는 것을 권한다.

이 변경은 사용자 제안의 **엄격히 goal-free인 residual**보다 의존성을 넓힌다. raw goal 숫자나 provided ego state를 전달하지는 않지만, base plan을 통한 goal 영향은 존재한다. `detach`는 gradient를 끊는 것이지 정보 의존성을 제거하는 것이 아니다. 따라서 `no direct raw goal/status input`과 `goal independent`를 구분한다. 최종 규정 적합성의 자동 승인을 의미하지 않는다.

계속 엄격한 goal-independent coefficient head를 선택하려면 그 제한을 명시하고, base prediction이 달라질 때의 residual 불일치를 감수하는 별도 설계로 다룬다. 반대로 goal-conditioned scene를 쓰면 pure image-only residual이라는 설명을 하지 않는다.

## 3. Tangent와 gradient

### Detach와 freeze를 구분

`B.detach()`는 `dB/dp_base` 경로만 막는다. `p_final=p_base+B*c`의 직접 경로는 남아 있다. Shared backbone을 학습하면 base plan은 계속 바뀔 수 있다.

초기 함수 parity를 위해 residual 마지막 projection은 0으로 초기화한다. 필요하면 짧은 warm-up 동안 base 계산 경로 전체를 고정하고 새 분기를 적합한 후 joint 단계로 전환한다. Planner 파라미터만 freeze하고 shared backbone을 학습하는 것은 base prediction 동결이 아니다.

처음 마지막 projection이 0이면 내부 새 layer의 첫 gradient가 0일 수 있으므로, 첫 backward에서 모든 layer가 nonzero여야 한다고 요구하지 않는다. 몇 update 후 분기 학습이 실제로 진행되는지 확인한다.

### 저속·회전의 처리

- smoothing은 무조건 켜야 하는 기능이 아니다. 곡선 방향을 왜곡할 수 있다. 고정한 smoothing을 oracle·train·inference 모두에서 동일하게 사용한다.
- 짧은 segment에서 유효한 인접/누적 방향을 참조한다면 이는 **예측 경로만으로** 결정한다.
- `dp/max(||dp||,eps)`만 쓰면 완전한 zero base에서 B가 0이 된다. 이때 어떠한 c도 출발을 만들 수 없다. 별도의 all-zero fallback 정책을 명시한다.
- 음수 residual이 예측 segment 길이를 지나치게 줄이면 해당 segment 방향이 반전될 수 있다. 이 모델이 그런 반전을 허용할지, 비음수 progress 제약을 둘지 명시한다. 임의 GT stop gate는 금지한다.
- cap/clipping을 추가하면 oracle도 반드시 같은 제약으로 다시 계산한다. 제약 없는 oracle과 제약 있는 학습기를 비교하지 않는다.

### 범위 제한

Train 자료와 사전에 고정한 물리/설계 범위를 이용해 첫 cap을 정하고, V0에서는 saturation과 bounded oracle degradation을 진단한다. V0 p95/p99로 cap을 정하는 것도 개발상 튜닝이므로 금지라고 할 것은 없지만 독립 확인이라고 부르지 않는다. tiny train residual 분포 때문에 cap을 너무 좁게 만들지 않는다.

`tanh` cap은 큰 출력 폭주를 제한하지만, 저속에서 방향 반전을 자동 방지하지는 않는다. 또한 cap 부근에서는 gradient가 작아질 수 있으므로 단순히 작게 제한할수록 안전한 학습이라는 해석은 피한다.

## 4. 관측과 supervision

### 유지할 구조

- 기존 front H4 유지.
- 선택한 좌우 시야의 짧은 history 추가.
- 같은 카메라 내부에서 current/past 비교 후 camera ID, 수정된 K/extrinsic, 현재 기준 두 시점/Δt로 통합.
- dynamic ego pose/status는 새 motion forward 입력에 넣지 않는다. 학습 정답으로 쓰는 상대 운동과 고정 calibration을 구분한다.
- 기존 scene/motion의 연속 feature와 base planner를 없애지 않는다.

### 0.577의 해석

제공된 state-head MAE는 해당 head·target·평가에서 정확도가 부족하다는 관측이다. 모든 pooled feature가 metric motion을 얻을 수 없다는 증명은 아니다. 기존 모델에도 state/history supervision이 있으므로, 같은 target에 이름만 다른 loss를 더하는 것이 새로운 정보가 되지는 않는다.

첫 FRONT/SIDE 비교에서는 새 관측이 기존 motion/history task와 최종 PREFIX에서 실제 gradient를 받도록 연결한다. 별도 상대 운동 loss가 필요하면 한 종류, 고정 정규화, 같은 두 arm의 weight를 명시한다. 0.1초 displacement와 1초 displacement를 단위·count 조정 없이 합산하지 않는다.

### 정적 배경 가중치는 조건부 선택

기존 안정적인 image-only predicted confidence/mask가 있다면 soft weighting으로 활용할 수 있다. 그러나 없으면 별도 detector/tracker 개발을 첫 실험의 선행조건으로 만들지 않는다.

- 객체 존재는 객체 이동과 다르다. 주차된 차량도 정적 단서다.
- 완전히 가리지 않고 soft weight와 유효 관측을 보존한다.
- GT mask를 학습의 실제 feature 입력으로 쓰고 추론에서는 predicted mask로 바꾸는 불일치를 만들지 않는다. GT는 mask/auxiliary supervision으로 사용할 수 있다.
- goal-conditioned/pose-conditioned scene 출력에서 만든 mask를 residual branch에 전달하면 그 입력 의존성도 전파된다. detach는 이를 없애지 않는다.
- ego-motion용 downweighting과 scene의 주행 문맥은 분리한다. 선행차를 ego-motion 추정에서 낮게 가중한다고 planning scene에서도 제거하지 않는다.

## 5. 학습 objective와 잔차 target

최종 학습은 항상 배포 경로에서 생성한 `p_final`에 PREFIX를 적용한다. 기존 LEN 등 유지하는 loss도 base와 final 중 어느 출력에 적용되는지 고정한다. 원칙적으로 최종 plan의 loss를 쓰며, base에 원래 plan loss를 계속 유지하려면 그 추가 가중치를 명시한다. 기존 plan loss를 한 번 더 더해 총량이 조용히 바뀌지 않게 한다.

Base가 바뀌는 joint 학습에서 고정 residual target cache를 사용하지 않는다. 최종 XY를 직접 감독하면 per-step oracle label이 필수는 아니다. coefficient label을 사용하려면 current detached base에서 target을 갱신한다.

다만 target stale 문제가 없어져도 in-sample residual 문제는 남는다. 이미 학습한 train에서 base 오차가 작고 V0에서는 더 크다면, 새 head가 경험하는 수정량 분포가 다르다. 동일 producer로 train-probe/V0의 oracle coefficient 및 실제 correction 분포를 기록하되 새로운 대형 out-of-fold 프로젝트는 지금 시작하지 않는다.

평가에는 다음 네 출력을 구분해 저장한다.

1. 등록된 MR 기준 예측.
2. 학습된 새 모델의 `p_base` (c=0에 해당하는 내부 출력).
3. 동일 모델의 실제 `p_final`.
4. 동일 모델의 current p_base와 배포 geometry에 대한 oracle (진단 전용).

2는 독립적으로 재학습한 baseline이 아니므로 causal ablation으로 과장하지 않는다. 모델이 joint 학습으로 base만 개선한 것인지, 보정이 실질적으로 기여하는지 이해하는 보조 자료다.

## 6. FULL과 실행 배치

유효 full row가 101,520일 때:

```
U_full = ceil(20554 * 101520 / 83700) = 24931
```

이 수치는 MR의 성공한 초기값·레시피를 동일하게 옮기는 조건이다. 현재 MR terminal에 다시 24,931회를 추가하는 것과 다르다. 실제 row 수와 initializer는 manifest에서 확정한다. 3.929회 노출은 알려진 최적값이 아니라 성공한 설정이다.

FULL에서 `0.17대 가능성이 크다`는 예측은 현재 자료로 뒷받침되지 않는다. 안전망 역할은 맞지만 성능값은 미확정이다. FULL의 checkpoint, teacher, feature cache, data-fitted 통계를 DEV에 가져오지 않는다. FULL이 V0/H를 학습했어도 별도의 DEV 계보가 이를 학습하지 않았다면 DEV 검증은 유지할 수 있다.

| 슬롯 | 실행 | 기준 |
|---|---|---|
| GPU 0 | 검증된 MR FULL | 즉시 진행, 고정 full-fit terminal |
| GPU 1 | FRONT + scalar residual | 아래 SIDE와 같은 base·residual·supervision·공통 학습 budget |
| GPU 4 | FRONT H4 + SIDE + 같은 residual | 새 관측 패키지의 실제 최종 D3 |

FRONT만 짧게, SIDE만 길게 학습한 종점끼리 side 효과로 비교하지 않는다. 공통 budget endpoint에서 주 비교를 하고, 그 이후 추가 학습이 있으면 별도 단계로 표기한다.

Recurrent BEV는 다른 두 run을 시작했다는 이유만으로 자동 착수하지 않는다. 현재 계보에 준비된 구현이 없으므로 현 공격안 결과와 잔여 시간을 보고 결정한다. 이미 돌아가는 HR/SIDE 작업이 있다면 중복 실행·재시작하지 않는다.

## 7. 짧은 preflight와 판단 기준

Preflight는 하나의 수정 geometry 코드에서 처리한다.

- c=0 초기 plan parity.
- straight/curve/정지/출발/미세 segment의 finite 출력·gradient.
- mirror 및 좌우 camera 교환 일관성.
- negative correction과 cap 처리.
- full-batch/microbatch의 loss와 gradient.
- 실제 camera/pair count, tensor shape, 입력 whitelist.
- GT 제거 상태의 raw inference와 전체 forward 비용.

성공은 실제 V0 PREFIX 감소로 판정한다. Oracle, coefficient MAE, state/history 오차는 원인 해석에 사용한다. 개선이 확인되면 해당 레시피만 full-fit으로 이전한다. Oracle이 낮다는 이유만으로 run을 반복하거나, 유의성 판정 하나만으로 실용적 작은 개선을 버리지 않는다.

**최종 결정:** FULL은 시작하고, 배포형 geometry/oracle 점검이 끝나는 대로 FRONT/SIDE 한 쌍을 학습한다. 그 뒤에는 추가 설계 문서가 아니라 실제 예측 PREFIX로 다음 결정을 내린다.

## 출처 및 산술

- 사용자가 제공한 네 가지 수정안: 본 리뷰의 직접 대상.
- 업로드 `MotionDrive_architecture_reference_32da576.md`: 과거 shared backbone / state/history / scene 경로 발췌. 최신 구현 재검증은 아님.
- 업로드 `MotionDrive_common_feature_improvement_sources_20260911.md`: 보존 Q&A 및 현재 occupancy target 범위.
- 업로드 `MotionDrive_Temporal_Residual_Proposal_Review_20260917.md`: 직전 제안과 조건부 oracle bound.
- PyTorch Tensor.detach 공식 문서: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html
- scikit-learn StackingRegressor 공식 문서: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html
- 본 회차 계산: `residual_reply_review_20260917/check_math.py`, `verification.json`. 실제 모델 결과를 재현한 파일이 아니라 합성/대수 검증이다.
