# MotionDrive-v2 P6 global stop-balance 사전등록 초안

상태: **FROZEN PREREGISTRATION — 구현 독립 검토 PASS; 실제 실행 증거는 별도 receipt에 기록**  
작성일: 2026-09-08 KST  
문서 소유자: operations  
실험 성격: 반복 사용된 tune에서 수행하는 제한적 진단. 최종 holdout, 제출 성능, 규정 준수 또는 1위 성능의 증거가 아니다.

## 1. 질문과 범위

P6는 두 개의 immutable P4 joint LAST6000 base 각각에서 기존 stop BCE를 그대로 둔 control(C)과, **train 전체 valid stop label의 고정 class weight**만 적용한 balance arm(B)을 비교한다. 목표는 P4의 정지 구간 악화가 global class imbalance 보정에 일관되게 반응하는지 확인하는 것이다.

다음은 바꾸지 않는다.

- model architecture와 forward graph
- goal, image, state, history 및 time 입력
- train/tune split과 row 순서의 모집단
- stop target 정의 `state_target[:, 5]` 및 validity mask
- planning·occupancy·lane·motion target과 loss
- trainable parameter 집합, fixed-BN 정책, C1 geometry, nominal time
- BF16 encoder / FP32 planner 정밀도 정책
- P4 base checkpoint bytes와 기존 P4 artifact/record

P6는 새로운 모델 구조, selector, threshold, cross-cell reweighting 또는 배포 변경을 시험하지 않는다.

## 2. 고정 base와 provenance

| base | immutable warm start | checkpoint SHA-256 | completed sidecar SHA-256 |
|---|---|---|---|
| 0 | `work_dirs/motiondrive_v2/p4_fresh_joint_s0/last.pth` | `3e9ae5aa6ca89f4eb9a72ba38355dfdbec4ccaa38eabbebb3526fa3019404478` | `001b822dde79024f0e2cd52b5b99cb1096bbab5bfd09ef2d95e963ad8dc6e3de` |
| 1 | `work_dirs/motiondrive_v2/p4_fresh_joint_s1/last.pth` | `c937f2f772c78f99cf3cfa5b1a96614267e20a9a667a157f4ada284940f9672e` | `49dc09c8e7840a059ae6bcadba1c490052847c678dc81d5524ee39f9da159aad` |

공통 데이터 provenance:

- split: `data/etri/motiondrive_v2/grouped_split_rawtime.json`, SHA-256 `f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936`
- supervision: `data/etri/motiondrive_v2/train_tune_geometry_v2/supervision_manifest.json`, SHA-256 `ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93`
- calibration: `data/etri/motiondrive_v2/train_tune_geometry_v2/calibration.npz`, SHA-256 `8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961`
- train inventory/global-count population: 203 scenes, 72 sessions, 54,810 rows; row SHA-256 `75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854`
- tune: 37 scenes, 11 sessions, 1,998 rows; row SHA-256 `1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88`
- final: 접근 0

P4 training source origin은 `86620b4ffc7e6838b49cf83b5be789eba12d8027`이다. P4 supervisor의 원래 rc1/global-HEAD drift 기록은 성공으로 바꾸지 않는다. Immutable LAST의 별도 CPU artifact audit와 evaluation rc0 provenance는 `MOTIONDRIVE_V2_P4_JOINT_RESULTS_20260908.md`, `reports/p4_joint_results_summary_20260908.json`, `reports/p4_joint_head_drift_audit_20260908/`에 보존되어 있다.

P6 구현 source commit은 preregistration commit 직후 실제 SHA를 실행 receipt에 기록한다. 동결된 changed/new source 파일은 다음과 같다.

- `scripts/motiondrive_v2_training.py`: SHA-256 `3929538dcc58c786b242680e84ca768ecb3674ec7bfbbd784a8143fd7d1aed70`
- `scripts/train_motiondrive_v2.py`: SHA-256 `61b889f9f600814ba3145cd98d50f1f8483cc72c9589054d7baafc35e920a10a`
- `scripts/run_motiondrive_v2_stop_balance_continuation.py`: SHA-256 `94167d50d46b5e91d529c0ac96924bf27934dd9c3e99c2cdc2bc3c803b7c81de`
- `tests/test_motiondrive_v2_stop_balance_continuation.py`: SHA-256 `abc3997f1781e618cebdada9178f440a4f22133a38ac383efa2b81c10313406d`

독립 label-only 집계의 실측 provenance:

- global valid stop count: `N=54,810`, `N_stop0=50,829`, `N_stop1=3,981`, invalid `0`
- fixed global weights: `w0=0.5391607153396683`, `w1=6.883948756593821`
- canonical two-level `(scene,row,state_target[:,5],state_valid[:,5])` digest: `d587ff2c8cbab445434123c1b3b194295d7c8e30513ba2c62aa538c82098a475`
- dataset-row-order target SHA-256: `fa635fd1f251b24db7678aa37372952a795626033c503aaf2939319d309c2123`
- dataset-row-order valid-mask SHA-256: `3e755a03d0f3d3aea0fe4a619beeded168bb0ea4d0c06243f3052d7c58293d78`
- `reports/p6_global_stop_train_counts_20260908_ops.json`, SHA-256 `9904a31458e11c35b942a429ca8af547e2bef644c34f149399d34c465d7497bc`
- `reports/p6_global_stop_train_counts_20260908_ops.receipt.json`, SHA-256 `b64b2336b7174a34382eeb334bfa365dd63859251e4d319376fbffe1aca9d355`

이 count는 54,810행 train inventory 전체에서 한 번 계산한 고정 전역 통계다. 각 continuation이 실제 소비하는 것은 `1,000 updates × logical batch 16 = 16,000` sampled examples이며, 54,810행 전체를 한 epoch 소비한다는 뜻이 아니다.

## 3. 2 bases × 2 arms

실행 행렬은 네 개이며 전부 LAST1000 결과를 보고한다.

| base | arm | 변경점 |
|---|---|---|
| 0 | C | 기존 P4 stop BCE reduction 그대로 |
| 0 | B | global train-valid class-weighted stop BCE |
| 1 | C | 기존 P4 stop BCE reduction 그대로 |
| 1 | B | global train-valid class-weighted stop BCE |

각 base 안에서 C와 B는 같은 warm-start model weights, seed, train row order 및 deterministic augmentation order를 쓴다. Base 0과 base 1은 서로 다른 P4 base를 유지하며 교차 초기화하지 않는다. 어느 arm도 optimizer, scheduler 또는 RNG state를 P4 checkpoint에서 resume하지 않는다.

각 실행은 최종 sample-order SHA를 기록하며, 같은 base의 C/B sample-order SHA는 exact equality여야 한다. 불일치하면 paired 비교를 fail closed한다.

### B arm의 유일한 수학적 변경

Train 전체의 valid `state_target[:, 5]`를 한 번만 세어 다음 고정 weight를 만든다.

```text
N = N_stop0 + N_stop1
w_c = N / (2 * N_c),  c in {0, 1}
```

- invalid target은 count와 loss에서 제외한다.
- weight는 전체 train count로 실행 전에 고정한다.
- minibatch별 재계산이나 balance는 하지 않는다.
- tune/final label로 weight를 계산하지 않는다.
- 기존 stop BCE의 effective coefficient `0.04`는 C와 B 모두 그대로다.
- C와 B 사이의 그 밖의 loss·normalization·optimizer·model 차이는 허용하지 않는다.

## 4. 공통 학습 계약

- warm start: 해당 base의 model weights only
- optimizer: fresh AdamW, step 0에서 시작, weight decay `0.01`
- original parameter groups의 절반 LR: head `5e-5`, backbone `5e-6`
- updates: 정확히 1,000
- warmup: 100 updates linear, 이후 cosine decay
- logical batch 16, microbatch 2
- gradient clip 5
- fixed BN running statistics; affine와 기존 trainable weights는 그대로 학습
- C1 geometry, nominal time
- BF16 encoder, FP32 planner/head
- train inventory 54,810 rows, stride 1; 실제 consumption은 1,000×16=16,000 sampled examples
- evaluation은 LAST1000 생성 뒤 tune 37 scenes/11 sessions/1,998 rows를 정확히 한 번만 수행
- LAST1000이 유일한 primary checkpoint이며 BEST 생성/선택을 하지 않는다.

금지:

- final holdout 접근
- tune threshold fitting
- seed, arm 또는 checkpoint 중 좋은 결과만 선택
- schedule 연장, 추가 seed, 재시작 또는 stop-loss 추가 tuning
- 결과를 보고 자동으로 새로운 실험·배포·3090 측정에 진입

## 5. 실행 순서와 자원 보호

1. P6 source freeze와 review/CPU test를 통과하고 exact new source pin을 기록한다.
2. GPU4/5 UUID, free memory, foreign process와 네 run-dir 부재를 확인한다.
3. base 0의 C와 B를 GPU4/5에 하나씩 병렬 실행한다. 고정 mapping은 C→GPU4, B→GPU5다.
4. 두 arm의 actual OS rc, native child rc, strict LAST1000, PID 부재와 input/source 불변을 확인한 뒤에만 base 1로 넘어간다.
5. base 1도 같은 arm→GPU mapping과 절차로 실행한다.
6. 네 실행을 모두 보고한 뒤 멈춘다.

GPU4 UUID는 `GPU-4b804d68-fd61-af14-393a-573c533d5006`, GPU5 UUID는 `GPU-1e9aea73-4e6b-2cb5-b788-f3d99e6dc6f8`이다. GPU0–3의 foreign 작업은 보존하고 GPU6/7은 사용하지 않는다. 종료·pressure 처리는 own parent/child PID에만 적용한다.

### 메모리 정책

P6는 cache job의 값을 관성적으로 복사하는 것이 아니라, 같은 full-joint architecture와 batch16/micro2를 쓴 P4 실측을 근거로 다음을 유지한다.

- allocator cap: 12,000 MiB
- launch reserve: 8,192 MiB
- P4 base0/base1 peak allocated: 각각 5,074.416 MiB
- P4 base0/base1 peak reserved: 각각 5,648 MiB
- P4 supervisor preflight free: 182,632 MiB; minimum observed free: 176,060 MiB
- P4 pressure event/OOM: 없음

12,000 MiB cap은 측정 peak-reserved의 약 2.13배이며, loss weighting만 바뀌는 P6에서 activation graph는 동일하다. Cap 또는 batch/microbatch 조건을 바꾸려면 새 실험명과 프로토콜이 필요하다.

P4 supervisor 원본은 외부 global-HEAD drift 때문에 rc1을 보존한다. 위 수치는 그 원인을 memory failure로 재해석하지 않으며, immutable source866 training log에 기록된 실제 allocator 관측값으로만 사용한다.

## 6. 평가와 보고 계약

네 LAST1000 모두에 대해 다음을 같은 tune 1,998행에서 보고한다.

- official weighted D3 전체값
- 같은 base의 `B − C`
- 같은 base의 `B − original P4 LAST6000`
- 같은 base의 `C − original P4 LAST6000`
- steady 99 / depart 24 / nonstop 1,875 breakdown
- session046 제외 결과
- ZERO/selector를 쓰지 않는 원 planning output의 동일 metric 정의
- source/input/LAST/report SHA, actual OS/native rc, optimizer step, finite, fixed-BN 및 PID 부재

통계 보고:

- base seed 0과 1을 각각 개별 보고
- 두 seed의 단순 mean과 range 보고
- `B − C`는 11-session paired cluster bootstrap CI를 보고
- 네 LAST tune report의 1,998개 `row/scenario/session/frame` 순서와 GT가 모두 exact인지 먼저 확인하며, 하나라도 다르면 fail closed한다.
- bootstrap은 기존 P4 analyzer 계약대로 `repeats=10000`, `seed=20260908`, NumPy linear 2.5%/97.5% quantile을 쓴다.
- mean `B − C`는 row별 float64 delta `0.5 * ((B0 − C0) + (B1 − C1))`를 먼저 만든 뒤, 11개 sorted session의 같은 resample indices를 매 draw에 적용한다.
- estimator는 frame-weighted다. 선택된 session의 delta 합을 선택된 session의 frame 수 합으로 나눈다.
- 3,996행/22 cluster로 concatenate하지 않고, seed 자체를 bootstrap하지 않는다.
- mean `B − original P4`의 descriptive CI도 같은 shared-session construction으로 보고한다.
- 특정 seed를 고르거나 CI 계산에서 제외하지 않는다.

## 7. 사전등록 채택 기준

B를 후속 검토 대상으로 채택하려면 모두 만족해야 한다.

1. base seed 0과 1 각각에서 B가 같은-base C보다 directional benefit을 보인다.
2. base seed 0과 1 각각에서 B가 해당 original P4 LAST6000보다 directional benefit을 보인다.
3. 두 seed 평균 B D3가 original P4 같은-base 평균보다 최소 `0.01` 낮다.
4. shared-clustering 11-session paired CI의 `B − C` upper bound가 0보다 작다.

KEEP 판정은 각 base의 `B − C < 0`, 각 base의 `B − original < 0`, 두 seed mean의 `B − original <= -0.01`, mean `B − C` cluster-CI upper `< 0`을 모두 만족할 때만 가능하다. Metric, delta, loss, checkpoint 또는 CI에 nonfinite가 하나라도 있거나 exact row/GT/source/input 검사가 실패하면 fail closed한다.

미충족 시 이 고정 P6는 효과 부족으로 기록하고 stop-loss weight, threshold, seed 또는 schedule을 추가 tuning하지 않는다. 충족하더라도 반복 사용된 tune에서의 exploratory 결과일 뿐이며 first-place, independent holdout 또는 규정 PASS로 표현하지 않는다. Stop-only oracle ceiling이 약 0.30이라는 진단은 개선 가능성의 하한 reference일 뿐, P6 하나로 목표 성능이나 1위를 입증하지 못한다.

## 8. 동기 진단과 해석 경계

P5-Z fixed selector는 base0 세 head seed 모두 ZERO 0/Δ0이었고, base1도 ZERO 4–5개와 약 `7.8e-5`–`1.03e-4` D3 감소에 그쳤으며 모든 11-session CI가 0을 포함했다. 따라서 P5-Z는 채택하지 않는다.

5초 goal/3초 GT timing 진단은 steady 99행 모두가 `max GT norm through 3s <= 0.2 m`임을 확인했다. 그중 87행은 5초 provided goal도 `<=0.2 m`이고, 이 87행이 stationary D3 합의 base0 85.34%, base1 83.84%를 차지했다. 5초 goal이 0.2m보다 먼 행은 12개다. 즉 endpoint/timing 불일치는 일부에 존재하지만 stationary 오류 대부분의 원인이라는 증거는 아니다.

진단 provenance:

- `reports/p5_zero_selector_head_results_20260908_ops.json`
- `reports/p5_zero_selector_headseed0_posthoc_20260908.json`
- `reports/p5_goal5s_stationary_timing_diagnostic_20260908_ops.json`, SHA-256 `fc9e58bb2a51518b6f50dd35a94d38fd447472f1f5b006e895cb3278ef7484e9`
- `reports/p5_goal5s_stationary_timing_diagnostic_20260908_ops.receipt.json`, SHA-256 `34ba8ae5e57fbbcfc01be811eb591fc5ef69afe73763df1a51ab9ee99463fc18`

이 evidence는 P6의 동기일 뿐 stop imbalance가 원인이라는 증명은 아니다. Cross-cell 또는 session-specific proposal은 본 프로토콜 밖으로 미룬다.

## 9. 속도·배포 경계

Model graph가 같더라도 P6 checkpoint의 실제 latency, parity, state-hash 및 배포 smoke를 측정하기 전에는 architecture-speed 변화 없음이나 배포 적합성을 주장하지 않는다. 이 사전등록은 추가 3090 측정, export 또는 submission 변경을 승인하지 않는다.
