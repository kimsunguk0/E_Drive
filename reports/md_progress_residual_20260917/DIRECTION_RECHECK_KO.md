# MotionDrive 방향 재점검

작성일: 2026-09-17 18시대 KST

## 최종 판정

현재 방향은 절반만 맞다. `MR FULL`은 제출 안전망으로 유지하고, scalar progress residual의
기하와 base-plan conditioning도 유지한다. 그러나 **FRONT/SIDE residual 자체가 주력 해법이라는
가설은 이미 약해졌다.** 지금 확인된 병목은 새 세션에서 base plan이 실제보다 빠른지 느린지를
판별할 종방향 관측이다.

가장 근거가 강한 다음 축은 Q6·Q7의 허용 범위 안에서 causal pose status를 planner나 residual에
직접 넣지 않고, **여러 task가 공유하는 영상 scene-attention query에만 조건으로 사용하는 A2**다.
이것을 1,000-step 동결-base 화면으로 먼저 검증한다. 준비되지 않은 recurrent BEV, 계수 수 증가,
더 큰 residual MLP는 지금의 병목에 직접 답하지 못하므로 보류한다.

## 3위까지의 실제 간극

| 항목 | 값 |
|---|---:|
| 등록 MR V0 PREFIX | 0.191002 |
| 등록 MR 서버 | 0.197988 |
| 현재 3위 서버 | 0.130537 |
| 서버 기준 절대 간극 | 0.067451 |
| 서버 기준 상대 감소 필요 | 34.1% |

등록 모델 한 점에서 관측한 서버−V0 차이 `+0.006986`을 그대로 가정할 때 3위에 필요한 DEV는
약 `0.12355`다. 표본 하나로 만든 환산이므로 보장선은 아니지만, 0.001~0.005 개선을 모으는
수준으로는 3위에 닿지 않는다는 판단에는 충분하다.

공식 지표는 여섯 waypoint 오차에 `[11,11,5,5,2,2]/36`을 주는 것과 같다. 서버 점수의
구간별 기여는 다음과 같다.

| waypoint 구간 | 점수 기여 |
|---|---:|
| 0.5초·1.0초 | 0.069466 |
| 1.5초·2.0초 | 0.077504 |
| 2.5초·3.0초 | 0.051017 |

따라서 후반 두 점만 고쳐도 0.146971이 남는다. 3위를 노리려면 첫 2초의 진행량도 함께
개선해야 한다. scalar progress correction은 모든 시점의 진행량을 함께 움직이므로 이 지표와
맞지만, 계수의 부호를 맞혀야만 한다.

## 병목을 특정한 실측

| 진단 | 전체 PREFIX/상관 | nonstop 1,875행 |
|---|---:|---:|
| 등록 base | 0.191002 | 0.196742 |
| deployable scalar oracle | **0.080906** | **0.085242** |
| deployable `delta_v+delta_a` oracle | 0.063959 | - |
| FRONT warmup final | 0.190236 | 사실상 변화 없음 |
| FRONT joint step 2616 final/base | 0.193752 / 0.197317 | residual 순기여 −0.000389 |

Scalar 하나의 표현력으로도 목표보다 훨씬 낮은 oracle이 나온다. `delta_a`가 없어서 막힌 것이
아니다. FRONT joint는 residual이 base 악화 중 일부를 되돌렸지만, 등록 모델보다 최종 점수가
나빠졌고 일반 주행은 거의 고치지 못했다. 이 계보는 종료했다.

같은 V0 1,998행에서 session을 하나씩 완전히 빼고 ridge 진단을 다시 계산했다. 입력 plan은
base XY 12개와 예측 구간 길이 6개이며, status를 붙인 경우에도 held-out session의 행은 fit에
한 번도 들어가지 않는다.

| LOSO 입력 | residual 계수 상관 | 계수 MAE | correction 적용 PREFIX |
|---|---:|---:|---:|
| base plan만 | 0.158 | 0.142 m/s | 0.191262 |
| base plan + 영상 예측 state | **0.067** | 0.146 m/s | **0.195840** |
| base plan + causal pose status | **0.711** | **0.093 m/s** | **0.139176** |

nonstop만 보면 영상 예측 state의 상관은 `−0.002`, causal pose status는 `0.747`이다. 더 직접적인
`vx - base 첫 구간 속도`와 oracle 계수의 상관도 각각 `−0.070`과 `0.624`다. 즉 현재 이미지
state head에는 필요한 상대 속도 신호가 없고, 허용된 causal status에는 큰 신호가 있다. 이것은
A2가 성공한다는 보장은 아니지만, 어디에 정보가 있고 없는지는 명확히 가른다.

재현 파일:

- `experiments/md_progress_residual_20260917/analyze_status_signal.py`
- `reports/md_progress_residual_20260917/status_signal_gap.json`

## VO와 더 많은 카메라에 대한 판정

과거 `vo_pred_val.npz`의 속도 상관 `0.991`은 pose-free 영상 운동의 증거가 아니었다. 실제
`hist_T`를 identity로 바꾸면 상관이 `0.02~0.08`로 붕괴했고, 과거 이미지를 현재 이미지나
shuffle 이미지로 바꿔도 실제 `hist_T`가 있으면 `0.88~0.93`이 남았다. 모델이 영상 대응보다
pose가 주입한 warp를 주로 읽었다.

독립적인 RAFT + 고정 calibration + ground-plane fit도 120행에서 다음 결과였다.

| 항목 | 값 |
|---|---:|
| 0.1초 translation-x MAE | 0.306 m |
| 환산 속도 MAE | **3.059 m/s** |
| 속도 상관 | 0.600 |
| `추정속도-base속도` 대 oracle 상관 | **−0.131** |

따라서 rule-based RAFT correction은 닫는다. SIDE 영상이나 optical flow 자체가 무가치하다는
뜻은 아니다. 현재 마감에서 필요한 약 0.05~0.1 m/s급 부호 판별과 거리가 너무 크다는 뜻이다.
RAFT를 쓰더라도 이후 learned motion representation의 보조 teacher 후보일 뿐, 바로 제출할
계산식은 아니다.

6-camera recurrent BEV도 지금 시작하지 않는다. 장면 기억과 주변 객체에는 유용할 수 있지만,
pose 정렬은 정적 배경의 ego-motion 흔적을 지우기도 한다. 현재 확인된 문제인 미세 종방향
오차를 여섯 카메라 memory가 해결한다는 직접 증거가 없고, 구현·학습·latency 변수가 너무 많다.

## A2의 정확한 정보 경로와 규정 판단

`OPEN_ISSUE.md` Q6은 과거 pose로 현재 ego status를 계산하는 것을 허용한다. Q7은 raw
status/과거 궤적을 planner에 직접 또는 단순 임베딩으로 주는 것을 금지하고, 여러 task의
공통 특징을 향상하는 간접 활용은 허용한다.

A2의 경로는 다음으로 고정했다.

```text
causal pose status (vx, vy, ax, ay, yaw_rate)
                  -> zero-init query delta
camera backbone -> shared scene cross-attention -> shared scene raster
                                               -> occupancy / lane / planner

base plan + image visual/motion features -> scalar residual
```

Status는 planner 인자, residual-head 인자, attention value가 아니다. attention으로 어느 영상
value를 읽을지 정하는 shared query만 바꾸고, 그 결과 scene raster를 occupancy·lane·planning이
함께 사용한다. 이 경로는 Q7의 허용 설명과 가장 직접적으로 맞춘 구현이다. 최종 허용 판단은
코드 심사에 있으므로 protocol에는 실제 의존성을 그대로 기록했다.

새 split의 train 83,700행과 tune 1,998행에서 status 5필드는 전부 valid/finite다. 값은 미래가
아닌 `t-1.0s...t` pose의 causal quadratic fit이다. 등록 MR과의 실제 full-resolution preflight에서
zero-init A2 base/final plan은 bitwise parity였고 최대 차이는 `0`, 초기 residual 계수도 정확히
`0`이었다.

## 현재 GPU 배치와 중단 기준

| GPU | 실행 | 판단 규칙 |
|---|---|---|
| 0 | 376-scene / 101,520-row MR FULL, 24,931 update | 제출 안전망. tune이 fit에 포함돼 eval은 선택 지표로 사용 금지 |
| 1 | STATUS-A2-S 1,000-step frozen-base warmup | final-base 개선 ≥0.005와 nonstop 개선 확인 후에만 joint |
| 2 | STATUS-A2-S seed 2, 1,000-step frozen-base warmup | seed 1과 같은 gate; 재현성 확인 |
| 3 | SIDE-S-AUX-DN joint | GPU2와 같은 step 2616 gate |

FULL 첫 1회 노출(step 6345)의 내부 tune 값은 `0.184706`이었다. 그 1,998행이 이미 FULL 학습에
포함되어 있으므로 일반화나 서버 예상치가 아니다. FULL은 정해 둔 3.93 exposure를 끝까지
수행하고 최종 제출 후보로만 취급한다.

STATUS-A2-S warmup의 비교 기준은 같은 등록 base에서 시작한 FRONT-S warmup `0.190236`이다.
0.001 안팎 변화는 seed/noise와 구분하기 어렵다. A2가 명확히 통하면 동일 checkpoint에서
joint로 전환하고, 이후 shared conditioning을 더 넓히는 것이 이번 마감의 주 공격안이다.

SIDE-S-AUX는 step 2616에서 final/base `0.193830/0.197238`, residual 순기여 `−0.003408`을
냈지만 nonstop 개선은 `0.000278`뿐이었다. 개선이 steady/depart 소수 표본에 몰려 gate에
미달했으므로 step 2751에서 종료하고 GPU 2를 A2 재현에 배정했다.

SIDE 두 모델이 step 2616에서 일반 주행을 못 고치면 추가 camera·denoising·두 번째 계수의
우선순위를 내린다. A2까지 실패한다면 현재 계보에서 3위 가능성은 낮다. 그 경우 남은 큰
연구축은 pose-free correspondence를 학습하는 motion branch지만, 작은 MLP나 학습 연장으로
해결됐다고 간주하지 않는다.

## 실행 결론

1. FULL은 계속 돌려 안전망을 확보한다.
2. A2 warmup을 가장 중요한 독립 검증으로 본다.
3. SIDE 두 run은 이미 정한 첫 joint endpoint까지만 공정하게 비교한다.
4. scalar oracle이 충분하므로 `delta_a`, 더 큰 selector, 같은 계보 ensemble은 중단한다.
5. recurrent BEV와 rule-based VO는 이번 6일 주력에서 제외한다.

현재 가장 큰 놓침은 모델 크기나 카메라 수가 아니다. **실제 causal status에는 base plan의
종방향 오차를 판별할 신호가 있는데, 현재 영상 state/motion 표현은 그 신호를 새 세션으로
일반화하지 못한다.** 남은 시간에는 이 신호를 규정상 허용된 shared perception 경로로 쓰는
방법이 실제 PREFIX를 낮추는지 검증하는 것이 가장 합리적이다.
