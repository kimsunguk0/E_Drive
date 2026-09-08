# MotionDrive v2 P1–P8 회고 및 P8 종결 보고서

상태: **P8까지 실행·독립 감사를 완료하고 사용자 지시에 따라 정지**. P8 이후의 새 학습·fit·모델 forward는 수행하지 않는다.

## 먼저 보는 결론

P1–P8은 1위 달성을 입증하지 못했다. 같은 reused grouped tune1998에서 수치상 가장 낮은 실제 학습 family mean은 P8 matched control의 D3 **0.324335916 m**다. 이는 historical P7 control 0.327602512 m보다 0.003266597 m 낮지만 bitwise replay나 matched architecture contrast가 아니고 CI도 0을 포함하므로 새 개선·배포 채택의 증거가 아니다. P8 wider-history는 두 seed에서 모두 matched control보다 나빴고 평균 **+0.010017613 m**, shared-session CI **[-0.010807711, +0.027507205]**였다. 따라서 보편적 해악을 입증한 것은 아니지만 사전 KEEP의 `두 seed 모두 W<C` 조건과 전체 conjunctive gate는 실패했다.

진단은 단일 원인 이야기를 지지하지 않는다. 영상에서 추정한 compact state/history의 tune 오차가 train보다 크게 악화했고, 그 예측값만 쓰는 작은 MLP는 실패했다. 동시에 frozen planner 안에서 predicted state/history를 GT21로 바꿔도 계획은 거의 움직이지 않았다. 이 두 현상은 모두 관찰된 후보 제약이며, 어느 하나가 단독 또는 주된 원인임은 입증되지 않았다.

## 공통 데이터·모델·지표

- 현재 실험 계열은 grouped train **54,810 rows / 203 scenes / 72 sessions**, tune **1,998 rows / 37 scenes / 11 sessions**를 사용한다. old ScoreDrive train330/val38과 다른 집합이다.
- old ScoreDrive Phase A–E는 별도 역사적 계열이며 이 P1–P8 표에 섞지 않는다. 특히 old train330 bank, val38 점수, 5 s candidate-selection 구조를 현재 grouped MotionDrive 결과로 재해석하지 않는다.
- backbone은 nuImages ResNet-50 계열이고, 여섯 current camera와 front history를 처리한다. P8 C/W는 여섯 current view와 downsampled current front, 네 past front를 합쳐 호출당 정확히 11 image encoding을 유지한다.
- 공식 내부 D3는 t=0.5…3.0 s의 6개 누적 waypoint L2에 `[11,11,5,5,2,2]/36`을 적용한 frame-weighted 평균이며, ADE1/2/3 평균과 동치다. 가중치의 61.1%가 첫 1 s 두 점, 88.9%가 첫 2 s 네 점에 놓인다. 단순 직선 근사에서 constant-velocity error의 D3 계수는 `Σw·t=5/4`, constant-acceleration error의 계수는 `Σw·0.5t²=151/144`다. 같은 ordered tune1998을 사용한 P4–P8 값은 내부 산술 비교가 가능하다.
- tune은 여러 단계에서 반복 사용됐다. 따라서 bootstrap은 row/session 불확실성 요약이지 independent-holdout 일반화 인증이 아니다. official clip-final server score와도 동일하지 않다.
- final136 row는 **현재 P1–P8** 학습·평가·분석에서 격리했다. 그러나 grouped manifest 자체가 `Historical labels were previously inspected; no split is advertised as untouched.`라고 명시하므로 historically pristine holdout이라고 부르지 않는다. P4+의 public init은 old ETRI checkpoint 직접 상속을 제거했지만 사람의 과거 실험 지식까지 제거하지 않는다. 공유 NPZ 파일에 다른 split bytes가 존재할 수 있으므로 '파일 bytes 자체에 접근하지 않았다'는 더 강한 주장도 하지 않는다.
- 아래 실제 모델 점수와 privileged GT/oracle bound를 분리한다. 후자는 배포 성능이나 학습된 multimodal upper bound가 아니다.

### P7/P8 실제 정보 경로

P7/P8 forward는 모든 argument가 RGB-only인 모델이 아니다. 입력은 RGB, `lidar2img` calibration, provided full-SE3 current→past transforms, time, goal을 포함한다. pose/calibration은 scene image sampling/alignment에 쓰이고 provided 5 s goal은 shared-scene auxiliary input으로 들어간다. raw motion estimator 자체는 image feature와 time만 받는다. planner는 continuous scene/motion token, 192 motion token, image-predicted compact state/history를 받지만 raw goal이나 provided status/history argument를 직접 받지 않는다. P1–P8 production planning target/output은 3 s의 6×XY waypoint이며, old ScoreDrive처럼 5 s candidate bank를 만들고 고르거나 crop하지 않는다. P7의 5 s bank는 사후 diagnostic일 뿐 production trajectory head나 tail loss가 아니다. P8은 past-history 시간 계약을 넓혔고 future prediction horizon은 바꾸지 않았다. `state_hat[5]`는 raw stop logit이고, 명시적으로 sigmoid를 적용한 진단만 probability라고 부른다. GT21 진단은 compact predicted-state/history branch만 치환했고 geometry-conditioned image route와 192 motion token은 그대로였다.

이 입력 경로가 내부의 보수적 규정 해석과 맞는다는 기술 논증은 organizer의 code-review 승인과 동일하지 않다. 실제 제출 적합성은 별도 확인이 필요하다.

## 단계별 요약

| 단계 | 좁은 질문 | 핵심 결과 | 사전 판정 |
|---|---|---|---|
| P1 | goal-conditioned perception G와 planner state/history query S의 효과 | G가 큰 개선; S는 G와 함께일 때 -0.009609 m | S 크기 기준 미달 |
| P2 | geometry/time repair와 입력 의존성 | 3개 clean arm 진단, 1개 SIGSEGV; image 의존성 큼 | 4-arm clean gate 실패 |
| P3 | frozen trunk에서 22D predicted state/history query | -0.008037 m, CI<0 | -0.010 m 기준 미달 |
| P4 | 새 grouped split의 2-seed fresh baseline | 0.372222 / 0.366957, 평균 0.369590 m | 기준선 확립 |
| P5 | original plan과 zero plan selector | 실현 이득 거의 0, 모든 CI가 0 포함 | reject |
| P6 | stop BCE 한 번의 class rebalance | B-C 평균 +0.000278 m | KEEP 실패 |
| P7 | ego-centered zero-slot C 대 real-5s-goal prior G | C 평균 0.327603, G 평균 0.429685 m | G reject; C working baseline |
| P8 | 0.1 s를 2.0 s history로 교체 | C 평균 0.324336, W 평균 0.334354 m; 두 seed 모두 W>C | KEEP 실패 |

## P1 — goal/state-query 2×2 screening

P1은 shared visual perception에만 goal을 넣는 G와, 영상이 예측한 state/history를 planner query에 명시적으로 주는 S를 분리했다. 이미지 계산, motion auxiliary, decoder, train rows, 공통 P0 초기값과 optimizer schedule은 고정했다.

| arm | G | S | LAST6000 D3 (m) |
|---|---|---|---:|
| G0S0 | off | off | 0.74784425 |
| G1S0 | on | off | 0.38165237 |
| G0S1 | off | on | 0.78536011 |
| G1S1 | on | on | 0.37204318 |

matched contrast는 G\|S0 -0.36619188 m, G\|S1 -0.41331692 m, S\|G0 +0.03751586 m, S\|G1 -0.00960919 m, interaction -0.04712505 m였다. 이 original setting에서는 goal conditioning이 지배적인 개선이었다. S는 G가 없을 때 악화했고 G와 함께일 때 소폭 개선했지만, 사전 screening 기준 -0.010 m에 미달했다. 네 supervisor는 모두 실제 `rc=0`이었다.

이 결과는 P7의 특정 real-goal Gaussian source-prior 악화와 모순되지 않는다. P1 G와 P7 G는 구조와 개입 위치가 다른 별도 intervention이다. P1은 단일 seed, raw-time/original wrong-rear geometry, 반복 tune 조건이므로 배포 가능한 일반 goal 효과가 아니다.

근거: `MOTIONDRIVE_V2_P1_PROTOCOL.md` SHA `9635fe45c04b42130867df14173772cb1421bde0bd8ed494fde4009a23b80ae2`; `reports/p1_last6000_raw_gs_analysis.json` SHA `c87ea5732b6f2054cb2c1b53504159d7c296446cb94331a1d62b8ef45df59cdd`.

## P2 — geometry/time 및 입력 의존성 진단

P2 전체 2×2 clean-exit gate는 실패했다. C1T0는 LAST3000 저장·평가 후 SIGSEGV로 종료했다(`actual rc=-11`, supervisor 139). checkpoint SHA `a3687d87...`, logged D3 0.44330748은 보존됐지만 정상 arm 결과로 편입하거나 누락값을 대체하지 않았다.

| 조건 | C0T0 raw | C0T1 nominal | C1T1 corrected geometry + nominal |
|---|---:|---:|---:|
| normal D3 | 0.36713314 | 0.36617518 | 0.44546200 |
| image shuffle | 2.12626990 | 2.11602130 | 2.60409496 |
| repeat current | 0.38814173 | 0.38545122 | 0.49038522 |
| reverse history | 0.36825017 | 0.36714039 | 0.45071615 |

image shuffle의 큰 악화는 영상 의존성을 확인한다. repeat-current는 약 0.019–0.045 m 악화했지만 reverse-history는 약 0.001–0.005 m만 변했다. 둘 다 분포 교란이므로 history의 이론적 가치나 개선 가능량이 아니다. C0 raw→nominal의 frame-weighted 차이는 -0.000958 m였고 session-equal 집계에서는 부호가 바뀌었다. C1T1과 C0T1의 +0.079287 m 차이는 wrong-rear geometry로 학습된 checkpoint의 3,000-step 적응 결과이지 corrected geometry from-scratch ceiling이 아니다.

사후 진단에서 neural yaw-rate는 GT yaw-rate와 약 0.946, 미래 lateral time coefficient와 약 0.817 상관이었지만 plan의 대응 coefficient는 약한 음의 상관과 GT variance의 약 0.5%만 보였다. ax MAE도 zero predictor보다 약 4–5% 낮았을 뿐이다. 이것은 planner 정보 전달을 조사할 근거이지 특정 모듈 결함의 증명이 아니다.

근거: `MOTIONDRIVE_V2_P2_DIAGNOSTIC_RESULTS_20260907.md` SHA `0eb063addce76d935f880852523f82c7665255b333ae74a6fd0fd91b5a83ade4`; `reports/p2_three_clean_arm_diagnostic_execution.json` SHA `aa8e75383b1ef3b77cfbaafdcb1aba8bbe2a0330ef5f72a49577a8b2782268ef`; failed C1T0 audit SHA `90ee08e117abc454c2b61776b851e7cc17f97163f84517a8bc4f5dd6cd65be43`.

## P3 — frozen planner query adapter

P3은 repaired/nominal frozen trunk 위에서 기존 planner와 query adapter를 합친 약 603k trainable parameter만 1,000 updates 학습해 zero query와 predicted 22D state/history query를 비교했다. 첫 r1 두 실행은 2.76e-8 initial-D3 equality check 때문에 step0 `rc=1`로 실패했고 별도 보존됐다. 수정된 r2는 둘 다 `rc=0`, 동일 initializer·row order, frozen auxiliary exact를 만족했다.

LAST1000 D3는 control 0.44597742 m, state 0.43794071 m로 delta -0.00803671 m(-1.802%), shared 11-session bootstrap CI [-0.01390354, -0.00272441]였다. effect를 0이라고 할 수 없지만, 사전 크기 기준은 <=-0.010 m였으므로 gate는 실패했고 기준을 사후 완화하지 않았다.

근거: `MOTIONDRIVE_V2_P3_RESULTS_20260907.md` SHA `d2872b2a98b7de5406b32abb206f3256e875601a3e701439da246fd4b7dd023d`; paired SHA `05fc105d248e0e48fb4e1b85b9a528f0656f7d30201e0b7936ca11410637f8ee`; execution SHA `eff779cc5a2088d6f29f3ea35cd2e5e16de62b24a19b0a2c8b62efb6a548abdf`.

## P4 — fresh grouped baseline

P4는 common public I0→각 seed own P0 LAST2000→fresh optimizer joint LAST6000의 2-seed 기준선을 만들었다. seed0 D3 0.3722216174 m, seed1 0.3669574755 m, 평균 0.3695895465 m였다. 과거 supervisor 기록의 `rc=1`은 이후 unrelated global-HEAD drift 때문이며 그대로 둔다. strict artifact audit와 canonical eval은 `rc=0`이었다.

근거: `MOTIONDRIVE_V2_P4_JOINT_RESULTS_20260908.md` SHA `6929153f`; `reports/p4_joint_results_summary_20260908.json` SHA `4449acfc`; seed eval SHA `cffa9cc8b12f5f9812af7b3fd3631e74597e291b02aecd1cc14d8ec8a1aeba2f`, `0e71bef199931cb1916abdc819ca78cf75d8cd7055439bd084ca975bb610c137`.

## P5 — zero/original selector

P5는 frozen P4 full-forward feature로 exact original plan과 exact zero plan 중 하나를 고르는지 물었다. 두 base×세 head seed의 CPU LAST1000에서 base0은 zero 선택 0개·delta 0, base1은 4–5개만 선택해 -7.84e-5∼-1.03e-4 m였다. 모든 11-session CI가 0을 포함했다. train fit은 강했지만 selector의 `p_zero` probability/calibration이 tune에서 붕괴해 fixed selector를 reject했고 threshold rescue를 하지 않았다. 이것은 neural raw stop logit의 probability라는 뜻이 아니다.

exact-GT oracle 약 0.30019/0.29989 m는 privileged two-candidate bound이며 선택 가능하거나 배포 가능한 성능이 아니다. 5 s goal/3 s stationary mismatch는 소수 12/99에만 해당해 대부분의 stop 오차를 설명하지 못했다.

근거: `reports/p5_zero_selector_head_results_20260908_ops.json` SHA `20fbcd5e`; source commit `79ec0d1`.

## P6 — 단일 stop-loss rebalance

P6은 train-valid N0=50,829, N1=3,981에서 w0=0.5391607, w1=6.8839488인 한 번의 global class-weighted BCE를 시험했다. 각 P4 LAST에서 fresh 1,000-update control/balanced continuation을 실행했다.

| base | original | control C | balanced B |
|---|---:|---:|---:|
| 0 | 0.3722216 | 0.3697885 | 0.3683529 |
| 1 | 0.3669575 | 0.3628463 | 0.3648386 |
| mean | 0.3695895 | 0.3663174 | 0.3665958 |

mean B-C는 +0.0002784 m, CI [-0.0023184, 0.0043583]; B-original은 -0.0029938 m, CI [-0.0082540, 0.0009267]였다. KEEP는 false였고 weight/threshold search를 하지 않았다. 첫 analyzer의 NumPy serialization `rc=1`/output absent와 reviewed serialization-only `rc=0`을 분리 보존했다.

근거: `MOTIONDRIVE_V2_P6_STOP_BALANCE_PROTOCOL.md` SHA `d591372b`; execution SHA `6cc460d4`; result SHA `9caf1de8`.

## P7 — cross-cell goal routing과 후속 국소화

P7 C의 새 slot은 neutral global attention이 아니다. 모든 destination query에 ego-centered fixed Gaussian source prior를 제공한다. G는 이 source-prior 중심을 real 5 s goal로 옮긴다. 기존 per-cell goal path는 둘 다 켜져 있다.

| base | P4 | P7 C | P7 G |
|---|---:|---:|---:|
| 0 | 0.372222 | 0.330586 | 0.424389 |
| 1 | 0.366957 | 0.324619 | 0.434980 |
| mean | 0.369590 | 0.327603 | 0.429685 |

G-C는 +0.102082 m, CI [+0.072843, +0.130415]로 gate를 명확히 실패했다. C-P4는 -0.041987 m, CI [-0.065010, -0.025029]로 복제된 exploratory working baseline이지만 새 cross-cell capacity와 ego-centered prior를 함께 도입하므로 두 구성요소를 분리할 수 없다. P7은 own P0에서 joint6000을 다시 학습했으므로 P4 joint LAST의 단순 continuation이라고 부르지 않는다. P1 goal 효과와 P7 G 악화는 서로 다른 구조의 개입이다.

후속 진단은 다음과 같다.

- current-train-only K1024 bank proxy는 direct C 0.327603→endpoint 0.330341 m, net +0.002739 m로 두 seed 모두 악화했다. full-K 0.114860과 M12 0.192710은 privileged bound다.
- matched 24D MLP는 tune에서 GT compact 0.088974 m, predicted compact 0.574126 m, direct C 0.327603 m였다. train GT/predicted는 0.089136/0.230761 m였다. 제공된 5 s goal은 두 arm에 같은 입력이다. privileged/direct-input 성격은 GT compact state/history와 continuous-image planner path 부재에서 오며, in-sample·distribution·22D confound도 있어 ceiling이나 pure decoder 진단이 아니다.
- frozen same-forward GT21 교체는 metric state 5개+history 16개를 GT로 바꾸고 predicted raw stop logit은 그대로 둔 intervention이다. D3 delta -2.8077e-5 m, shared CI [-3.6473e-5, -1.6300e-5], plan movement 약 5.316e-5 m였다. 이 21개 corrected component에 as-trained frozen planner가 거의 반응하지 않았다는 뜻이지, entire 22D/status token 또는 stop-token reliance를 검사한 것도 아니고 adaptation 뒤 precise state가 쓸모없다는 증명도 아니다. GT 치환의 OOD 성격도 남는다.
- cached inference에서 predicted vx MAE는 train 약 0.31 m/s에서 tune 약 0.98 m/s, 1 s history position error는 약 0.38 m에서 약 0.99 m로 악화했다. estimator generalization gap과 planner insensitivity가 동시에 관찰된다.

이 cached-inference 오차는 training-log loss가 아니라, 이미 감사된 train/tune prediction cache를 같은 row identity의 GT와 결합한 물리 단위 집계다.

| seed | split | vx bias / MAE / RMSE (m/s) | ax bias / MAE / RMSE (m/s²) | history XY mean error at 0.1 / 0.2 / 0.5 / 1.0 s (m) |
|---|---|---|---|---|
| 0 | train | +0.060588 / 0.308690 / 0.407067 | -0.004835 / 0.225759 / 0.329257 | 0.051554 / 0.088090 / 0.196239 / 0.378865 |
| 0 | tune | +0.015318 / 0.983957 / 1.270727 | +0.001371 / 0.280321 / 0.393056 | 0.108649 / 0.204419 / 0.499498 / 0.995452 |
| 1 | train | +0.061130 / 0.313408 / 0.412297 | -0.009075 / 0.228200 / 0.334490 | 0.051100 / 0.088474 / 0.198371 / 0.382769 |
| 1 | tune | +0.035739 / 0.973682 / 1.261078 | -0.005041 / 0.271563 / 0.381812 | 0.107946 / 0.201521 / 0.493964 / 0.987297 |

근거: P7 result commit `4dcb240`, results MD SHA `5d6c473a`, execution SHA `d7270a1a`; bank MD SHA `955f0993`; 24D result SHA `6edbb58e`, execution SHA `d13b399e`; GT21 execution SHA `fd7b73e0`.

GT compact 0.089 m와 predicted compact 0.574 m의 큰 차이는 이 compact estimator/MLP 경로의 train→tune 정밀도 문제가 실제임을 보인다. 그러나 full P7-C 0.328 m가 predicted-compact MLP보다 훨씬 좋다는 사실은 이 fixed 24D MLP로 full visual/scene/motion planning route를 대체할 근거가 없고 full route가 더 낮은 D3를 달성했다는 데까지만 말한다. 그 차이가 추가 정보, architecture/inductive bias, optimization 중 무엇 때문인지는 식별하지 못한다. 반대로 GT21을 같은 frozen planner에 넣었을 때 거의 안 움직였다는 사실도 as-trained compact branch의 local sensitivity가 작다는 데 한정된다. 두 진단을 estimator-only 또는 planner-only 단일 원인으로 축약하지 않는다.

source에서 생각할 수 있지만 아직 입증되지 않은 설명은 세 가지다. 첫째, radius2 local correlation과 pooled motion head가 metric/large-displacement 정밀도를 제한할 수 있다. 둘째, compact 22D는 richer scene/motion route 옆의 한 token이므로 영향이 작을 수 있지만 token 수만으로 attention dilution을 증명할 수 없다. 셋째, direct learned decoder는 explicit kinematic integration이 아니어서 auxiliary estimator가 좋아져도 D3가 자동 개선되지 않을 수 있다. 이것들은 후속 실행 제안이나 확정 원인이 아니라, 확인된 train→tune estimator gap과 tiny frozen-GT21 sensitivity를 해석할 때 남는 가설이다.

## P8 — wider-history matched C/W

### 질문과 계약

C는 `[0.1,0.2,0.5,1.0] s`, W는 `[0.2,0.5,1.0,2.0] s` history를 쓴다. W는 0.1 s image를 2.0 s로 바꾸면서 image, current-to-past SE(3), raw time offset, aligned pose-derived history target을 함께 바꾼다. 따라서 W 실패를 '긴 history가 쓸모없다'고 일반화할 수 없다.

두 arm 모두 11 encodings, architecture, parameter count, loss, normalization, optimizer, fixed BN, BF16 encoder/FP32 planner를 고정했다. curriculum은 public I0→arm별 P0 2,000 updates→own LAST2000 weights only→동일 seed의 새 13-key zero branch→fresh optimizer→joint 6,000 updates→terminal tune1998 한 번이다. P0의 step0+250 간격 aux eval/BEST는 비선택 진단이고 joint에는 initial/intermediate/BEST eval이 없다.

protocol SHA `81e547b64e17468de79c4043ebd27092ff4707f50e6223dbe0d45fe6be202f10`; actual joint source manifest SHA `4dcf2a95d619b52d9d61d375da108e3bc54c5e642789e8d6420e76372463dd3c`; driver SHA `c8aa61732be8499104e415ff626ccb63860464899b852d86f23072a1ce19a6d4`.

seed0 P0는 parent source manifest `6682ea02544fc8a39e5d78bdf90bf9735682b51f3f5c7f88db93d85729d77f75`와 driver `984acbc043f92ba12f3e8a35e1a57549a4dade13965e4683d36926f8c2a3d949`를 실행했다. seed1 P0와 모든 joint는 child manifest `4dcf...`/driver `c8aa...`를 실행했다. 유일한 driver 차이는 joint sidecar `model_config`의 tuple/list를 complete-tree canonical JSON으로 비교하는 compute-neutral validator 보정이고 P0 compute closure는 유지됐다. parent를 seed1 실행 source로 표기하지 않는다.

C/W overlay SHA는 각각 `4b6094cbd016bb0c050a0fc275456573c50b90bb0849bc6ab7aad279f2beffe2`, `55e03d202d2d2dfcbe1c3eef7462e0d6f270bd0ab067e075a649e21ba092189c`다. 203 train+37 tune scenes, 54,810+1,998 rows이고 final row를 사용하지 않았다. C overlay의 네 temporal array는 모든 선택 row에서 legacy와 bitwise equal한 뒤 W를 만들었다.

official-test filename-only audit는 1,125 archive 모두에서 6 camera의 `-30..0` 연속 frame 이름을 확인했다. 따라서 -1/-2가 제공되지 않는다는 문제는 없다. baseline stride5는 sampling choice다. 이 audit는 image contents나 official Docker mount를 주장하지 않는다.

### P8-C와 historical P7-C의 관계

P8-C는 P8-W의 matched control이지만 P7-C의 bitwise replay는 아니다. public-I0 model state, base seed, train/tune row, nominal C offset, high-level recipe와 모든 logged P0/joint sample-order SHA는 같다. C overlay temporal arrays도 legacy-bitwise-equivalent다. 그러나 P8은 새 explicit overlay/source path에서 P0를 다시 학습했고 P0 state가 달라져 joint initial full state도 달라졌다. P0 step1 값은 같지만 step10부터 갈라진다. P8 branch는 별도 SHA로 pin됐지만 P7 manifest에는 직접 비교 가능한 branch-substate SHA가 없다. 그러므로 seed별 P7-C→P8-C 차이를 nondeterminism이나 한 원인 효과라고 부르지 않는다. 두 seed family mean은 P8-C가 historical P7-C보다 -0.003266597 m였고 paired session CI [-0.00949736, +0.00459369]로 0을 포함한다. 수치상 최저 family mean이라는 사실과 architecture 개선·복제 성공·배포 채택은 구분한다. 이 비교의 exact 증거는 P7 P0 manifest/metrics SHA `9e5ba7ce89c0c1a9e4fe49476ff72c63b451f2658766e74e84c7336a4591fd98`/`a0574abe945e6df32c9c8dcdf16a3d04f0fdb2a87e19eac34f103e08d95a3196`, P7-C manifest/metrics `cf287d05ce6b2f4476c07ac5c2253cd9ecb3227bcc01404adbbba8463bb66de0`/`60772877bc0098436fd3d7d6f0f51fd885cf1bf1b3fc6f5897f50283c2f7e5b8`, P8-C P0 manifest/metrics `5125c32956672914175eaf9bf65cd7f21300cc78e64ddc7eddc8997433d31ad8`/`bde67dc6cacf45b38022eca8a5397af2ff555a7ac0494efacdcf3f2b4c5b3b62`, P8-C joint manifest/metrics `5604928fab7db3ead1e73fbe7d2b21efe4d4e22fcb40d0fbebee9cdd6d297df5`/`b30c8d3adfa30525cede2f8d0c5a098f49275b709e31553fe11aace7b2314129`다.

### seed0 terminal

| arm | actual rc | joint D3 | elapsed | LAST SHA | final-eval SHA |
|---|---:|---:|---:|---|---|
| C | 0 | 0.3325562676 | 7,248.20 s | `64fbe491f0c2a473f9be2a327472f24ee32d7b7a80ea40783dc7e35e32ab3b6c` | `c135f59b171727150113cba26e4572d86e0ea14261391c98eeaf62128970fdbf` |
| W | 0 | 0.3403412984 | 7,623.38 s | `9fefbc2c4c17edf647ef19aa527b20c43b50e50af413f8c0b52911cef3ff4368` | `8b9b322e0b03543e4dc0a6843a4fe5b1aac936e756c29b6231909af39cda22d4` |

| arm | ADE1 | ADE2 | ADE3 | vx MAE | ax MAE |
|---|---:|---:|---:|---:|---:|
| C | 0.194330 | 0.322389 | 0.480950 | 0.967355 | 0.275188 |
| W | 0.192972 | 0.332403 | 0.495648 | 1.007546 | 0.256574 |

ADE는 저장된 1,998개 `pred_abs_xy`/`gt_abs_xy`에서 first2/4/6 누적 L2를 read-only 재집계했다. device별 norm arithmetic 때문에 표의 세 ADE 평균과 저장된 official D3 사이에 최대 수 e-9∼e-7 수준 차이가 있으며, 판정에는 저장된 official D3를 쓴다. terminal report에는 stop target/bucket은 있지만 aggregate predicted stop probability metric은 없으므로 새 stop 성능값을 만들지 않는다.

W-C는 +0.0077850308 m로 W가 악화했다. seed0만의 fixed 11-session bootstrap CI는 [-0.02081071, 0.03006110]이다. session046 제외 delta -0.00393223은 influence 진단일 뿐 새 판정이 아니다. 정확한 mask는 `steady-stop = GT stop-positive & max 6-future displacement<=0.2 m`(n=99), `stop-depart = GT stop-positive & >0.2 m`(n=24), `nonstop = GT stop-negative`(n=1875)다. 세 bucket의 W-C는 +0.208890/+0.066297/-0.003582 m였다.

공통 lag만 의미를 맞춰 C index1/2/3과 W index0/1/2를 비교하면 W-C history position error는 0.2/0.5/1.0 s에서 +0.014464/+0.028972/+0.051604 m 악화했다. C-only 0.1 s와 W-only 2.0 s는 직접 짝지어 해석하지 않는다. 둘 다 step6000, nonfinite0, 601 train logs, 동일 final sample-order SHA, terminal eval 정확히 1회였다.

### seed1

두 P0는 actual `rc=0`, exact step2000, optimizer289 states 모두 finite/step2000, branch key0, initial0+250..2000 eval 9회, 모든 201 sample-order SHA 일치였다. C/W LAST SHA는 `b036e1b65f70167f48045b0f7324941905e8eb9e6e2efa0c5c27ddb679a0f0b2`/`5bbcd7b369c68daf34fbcfe2842e4bba08f8ef702aef41b14881a861028c0d75`; model-state SHA는 `b16d5d9941a36934de84a714c1e216a6cf8e14b37639b3ed17b600fd437742ec`/`7bf2f2c39fc20194239d2327731b31112f57f9f99e17a3b899c3cfb0bef8d050`; manifest SHA는 `7ad9ad3fcb9fc69a34d2f9935a0625f634ce19cefb601920d34e99ac92eb6c56`/`4c13c96aa2779200f86272a93696cbc51cc5fba11330373c0eebc3fc2be55465`다.

own-P0 joint CPU preflight C SHA `d78337973557113448d2dd8559b096f4a6bfb98999f2783b9a73bf3c8c0277cd`, W SHA `8e4a692a3ea270a49f9ca13490c09e81be53b76268c56ec99ca83e9a39d61f9a`는 둘 다 `rc=0`이었다. 13 missing keys, 동일 branch SHA `68ee0d727e8f3142b600ad0da77075d51717990e7d9a6579f07923a67d1ba246`, fresh optimizer step0를 검증했다.

두 joint도 actual `rc=0`, exact step6000, nonfinite0, optimizer 298 states 모두 finite/step6000, 601 train logs와 terminal tune1998 정확히 한 번, 모든 logged sample-order SHA 일치를 만족했다.

| arm | actual rc | joint D3 | elapsed | LAST SHA | final-eval SHA |
|---|---:|---:|---:|---|---|
| C | 0 | 0.3161155640 | 4,531.26 s | `ed90ed8e2e0d13a3b2bd12a25a36f56870f6e142de20284618fdadaa4e60b0b6` | `313c73200406f57651a91c4edd33630ef31ba648fc6dd15b830b52dffcd8e34c` |
| W | 0 | 0.3283657600 | 4,534.64 s | `831b552b9c0a5e491d7ae62387b96fc13c9a51efe948668271bf8ae8a5afe11c` | `489fddd87f942d88875b281d576add12d1b65e82d7a2fa03ca0d5e6264071c5b` |

| arm | ADE1 | ADE2 | ADE3 | vx MAE | ax MAE |
|---|---:|---:|---:|---:|---:|
| C | 0.196029 | 0.309899 | 0.442419 | 0.977010 | 0.276701 |
| W | 0.190429 | 0.323070 | 0.471598 | 1.015310 | 0.256471 |

seed1 W-C는 +0.0122501960 m였다. 공통 lag 0.2/0.5/1.0 s의 history error도 +0.015380/+0.018949/+0.039589 m 악화했다.

P0 wall time은 seed0 C/W 1,283.35/1,281.15 s, seed1 C/W 3,490.11/3,457.49 s였다. seed1 P0는 seed0 joint와 승인된 GPU sharing 중 실행돼 더 느렸으므로 속도 차이를 모델 품질로 해석하지 않는다. 네 P0 모두 peak allocated/reserved 5,320,917,504/5,922,357,248 bytes였다. joint wall time은 seed0 C/W 7,248.20/7,623.38 s, seed1 C/W 4,531.26/4,534.64 s였다. joint peak allocated는 모두 5,360,902,656 bytes, seed0 peak reserved는 5,949,620,224/5,947,523,072 bytes, seed1은 7,226,785,792/7,228,882,944 bytes였다. allocator cap 12,000 MiB와 launch reserve 8,192 MiB를 유지했고 pressure/nonfinite event는 없었다.

### 사전 판정

KEEP는 (1) 두 seed 모두 W<C, (2) mean W-C<=-0.015 m, (3) 두 seed에 같은 11-session draw를 적용한 bootstrap 95% upper<0을 모두 요구했다. 실제 C/W mean은 0.324335916/0.334353529 m, W-C는 +0.010017613 m, shared-session bootstrap CI는 [-0.010807711, +0.027507205]였다. 두 seed 관측값이 모두 W>C이므로 조건 (1)과 전체 conjunctive gate는 실패했다. CI가 0을 포함하므로 wider contract가 보편적으로 해롭다고 증명한 것은 아니다.

두 seed를 합친 exact mask 진단에서 W-C는 steady-stop(n=99) +0.1772373 m, stop-depart(n=24) +0.0220606 m, nonstop(n=1875) +0.00103426 m였다. session046 제외 +0.00095381 m는 영향도 확인일 뿐 새 판정이 아니다. 공통 lag 0.2/0.5/1.0 s의 두-seed mean history error는 +0.0149219/+0.0239602/+0.0455967 m 악화했다. 이 결과는 0.1 s 제거와 2.0 s 추가를 포함한 전체 time-contract bundle의 결과이며 'long history가 무조건 해롭다'는 분해 결론이 아니다.

## 지연시간: 정확도와 분리된 증거

P4 RTX3090 CUDA-forward median은 37.1036/36.9889 ms였다. trained P7-C seed0의 고정 B1/8 real clips/11 encodings/warm20+50×8 조건에서 CUDA-event median 38.6888 ms, synchronized host-wall forward median 38.7164 ms였다. H2D와 pre/post를 포함한 full pipeline preprocessing은 제외됐다. 둘 다 RTX3090·해당 wrapper 경계의 forward-only 측정이며 official RTX4090 latency나 정확도/adoption 증거가 아니다. trained canary execution SHA는 `7556d501`이다.

## 설계·운영 판단에서 부족했던 점

- official deployment의 camera/time/input contract를 P1 전에 먼저 감사했어야 했다. wrong-rear geometry와 raw/nominal time 문제를 뒤늦게 분리해 초기 결과의 해석 비용이 커졌다.
- P2 corrected arm은 wrong-geometry checkpoint의 continuation이었고 fresh corrected-from-public 비교가 아니었다. 그 차이를 corrected geometry의 한계처럼 읽을 수 없었다.
- auxiliary state/history에 gradient가 연결된다는 사실을 planner가 그 정보를 실질적으로 이용한다는 증거로 너무 일찍 취급했다. 실제 같은-forward GT21 sensitivity 검사는 P7 후반에야 수행됐다.
- privileged GT compact MLP 0.089 m는 정보 존재 가능성을 보였을 뿐, image에서 같은 정밀도로 추정 가능하거나 현재 full model이 그 representation을 활용할 수 있음을 보이지 않았다.
- 같은 tune1998에서 설계 선택과 진단을 반복했으므로 후기 단계의 수치와 bootstrap은 exploratory/optimistic하다. 독립 validation을 대신할 수 없다.
- strict global-HEAD, 극미세 float equality, tuple/list type equality 같은 guard가 계산 전에 false operational failure를 만들었다. 이런 실패를 과학 결과와 분리 보존한 것은 필요했지만, 처음부터 computational source와 wrapper provenance를 더 좁게 설계했어야 했다.

이 회고는 특정 universal redesign이 필수라고 단정하거나 다음 방법의 성공을 약속하지 않는다.

## 실패·보정 원장

| 단계 | 실제 사건 | 처리와 경계 |
|---|---|---|
| P2 | C1T0 SIGSEGV, child -11/supervisor139 | 4-arm gate FAIL 유지; saved metric을 clean arm으로 편입하지 않음 |
| P3 | r1 두 arm step0 rc1, 2.76e-8 equality check | 실패 보존; reviewed r2 rc0와 분리 |
| P4 | historical supervisors rc1, later HEAD drift | artifact audit/eval rc0와 분리; rc 재작성 없음 |
| P6 | 첫 analyzer NumPy serialization rc1/output absent | serialization-only correction 후 rc0; scientific recipe 불변 |
| P7 zero-init 3090 | wrapper NameError rc1 before model/CUDA/timing | 최소 import fix 뒤 fresh retry; 실패 relabel 없음 |
| trained P7 3090 | 첫 SSH rc255 pre-Docker | 단 한 번의 later bounded retry rc0; timing 실패로 해석 안 함 |
| P8 overlay test | shared moving test가 unfinished driver를 import해 collection rc2 | frozen producer4 + 독립 producer tests와 all-row parity를 실제 gate로 사용; rc2 보존 |
| P8 old golden | repo-root packaging rc1, tuple/list config rc1 | 두 pre-forward 실패 보존; canonical JSON-only fix 후 CPU FP32 exact replay PASS |
| P8 seed0 joint preflight | C/W rc1, checkpoint tuple 대 sidecar list | model_config만 canonical 비교; 나머지 lineage strict, retry rc0 |
| P8 seed0 joint launch | C/W rc1, CVD가 Python에 scope되지 않음 | pre-CUDA/output0 보존; 동일 argv env-scope 수정 후 rc0 |
| P8 seed1 W preflight | shell cwd/redirection rc1 before Python | output0/pre-model/pre-CUDA 보존; 동일 Python argv fresh rerun rc0 |

## 핵심 증거 색인

| 범위 | 경로 | SHA256 |
|---|---|---|
| P4 결과 문서 | `MOTIONDRIVE_V2_P4_JOINT_RESULTS_20260908.md` | `6929153fbae05f8ed09a250016a46955da030345a78759be6509aa3ab91c56f7` |
| P4 summary | `reports/p4_joint_results_summary_20260908.json` | `4449acfc5d7066d4a055412f6c8036b98cf6d31d26a13a026d0990ec8f5d3ffc` |
| P5 selector | `reports/p5_zero_selector_head_results_20260908_ops.json` | `20fbcd5ea12caaa1c752b6ec2182fbae85e2713f5df6b22f2d66177bbb9e5ef3` |
| P6 protocol | `MOTIONDRIVE_V2_P6_STOP_BALANCE_PROTOCOL.md` | `d591372b672cbd698c936e0fcd4aa7191762ca6389edce085ccc0b695140d772` |
| P6 execution | `reports/p6_stop_balance_execution_20260908_ops.json` | `6cc460d4dcfacfe819a38de944ffe2bb03a25ed3bb609ebc45542b0e29118d2f` |
| P7 결과 문서 | `MOTIONDRIVE_V2_P7_GOAL_ROUTING_RESULTS_20260908.md` | `5d6c473a66db94048c3da50a262e48c819c55f4e0df0c246cf9f54d841952aaa` |
| P7 execution | `reports/p7_goal_routing_execution_20260908_ops.json` | `d7270a1ac03cc32eb07c99599c6b19e39201a1ab2e692aa510c303c138244cca` |
| P7 bank 결과 | `MOTIONDRIVE_V2_P7_TRAIN203_BANK_FEASIBILITY_RESULTS_20260908.md` | `955f099310354394d6a73d3f970c6aff9d96d4448bb1def5646bbf64bc06682e` |
| P7 bank execution | `reports/p7_train203_bank_feasibility_execution_20260908_ops.json` | `cf1d9b722c544cd3cacdd8c7e51d0c59e21f241829fe43649fd9d77d63d63e3c` |
| P7 24D fit execution | `reports/p7_c_state_sufficiency_fit_execution_20260908_ops.json` | `d13b399e7380b83ed0bf3ddb5b58dd075386aea5bd899227de0e9097247adf99` |
| P7 GT21 execution | `reports/p7_c_gt21_execution_20260908_ops.json` | `fd7b73e07e37a81c4352e6b0a01a93f8c7f08817a5ce812f8bdceefa852a374d` |
| train/tune estimator audit | `reports/p7_c_train_tune_state_history_error_20260908_ops.md` | `0a8242e9dfbf1569144cd5c1db22d58923caca12fde4c123c8b652af878bc962` |
| metric reconciliation | `reports/motiondrive_v2_official_metric_reconciliation_20260908.md` | `f801c62e708f758ab8ff63629ee5c652b7e0e5282d4d873496bd10ad3ff2e987` |
| P8 protocol | `reports/p8_wide_history_fixed_protocol_draft_20260908.md` | `81e547b64e17468de79c4043ebd27092ff4707f50e6223dbe0d45fe6be202f10` |
| P8 joint source | `reports/p8_wide_history_joint_source_manifest_20260908_ops.json` | `4dcf2a95d619b52d9d61d375da108e3bc54c5e642789e8d6420e76372463dd3c` |
| P8 terminal analyzer | `reports/analyze_motiondrive_p8_terminal_20260908_ops.py` | `9cf8ae8cfc21b47f4f444d6f5e61bc2c63749ba768ae18f6b11c31f5dedd8c01` |
| P8 terminal analysis | `reports/p8_terminal_analysis_20260908_ops.json` | `025343bfed19e3256395c791fed747c47a186f26138f0095bf9dd685f169742d` |
| P8 terminal execution | `reports/p8_terminal_execution_20260908_ops.json` | `17fb90ba0f2f2090111e3413e6e3d59a2254e483a353072c135adb326314a80e` |

## 종결 경계

P8 seed1 두 joint run은 terminal audit까지 완료했고 관련 owned PID는 모두 종료됐다. P9 tune-cache process는 정지 신호 전 자연 종료했지만, P9 train cache·fit·성능 해석·retry는 수행하지 않는다. 새로운 experiment, benchmark, training, extraction은 이 보고서 범위 밖이다.

최종 상태는 '목표 달성'이 아니다. 수치상 최저 tested actual family mean은 reused tune에서 P8-C 약 0.324 m지만, final136 성능이나 official leaderboard 1위는 입증되지 않았고 automatic adoption도 하지 않았다. 사용자 지시에 따라 P8까지 증거를 고정하고 실행을 일시 정지한다.
