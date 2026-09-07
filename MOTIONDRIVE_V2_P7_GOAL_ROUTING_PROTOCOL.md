# MotionDrive-v2 P7 goal-routing 사전등록 초안

상태: **MAIN6000/3090 EXECUTION-GATE CANDIDATE — 3-update plumbing PASS; 각 launch는 해당 preflight 완료와 root 조건을 따른다**  
작성일: 2026-09-08 KST  
문서 소유자: operations  
실험 성격: 반복 사용된 tune에서 수행하는 제한적 진단. 최종 holdout, 제출 성능, 규정 준수 또는 1위 성능의 증거가 아니다.

## 1. 질문과 고정 비교

P7은 각 seed의 own P0 LAST2000에서 같은 joint6000 학습을 다시 시작해, 새 goal-routing branch의 유무만 비교한다.

| base | arm | 새 branch 입력 |
|---|---|---|
| 0 | C | 새 slot을 exact zero로 입력 |
| 0 | G | raw provided goal을 새 distance-score에만 사용 |
| 1 | C | 새 slot을 exact zero로 입력 |
| 1 | G | raw provided goal을 새 distance-score에만 사용 |

C도 기존 모델의 per-cell goal 입력은 실제 값을 그대로 사용한다. C에서 zero가 되는 것은 새 branch slot뿐이다. G의 raw goal도 새 distance-score branch에만 들어가며 기존 입력 경로를 대체하거나 변경하지 않는다.

같은 base의 C/G는 checkpoint weights, 새 branch 초기값, RNG, sample order, augmentation order 및 optimizer/scheduler 초기 상태가 같아야 한다. Base 간 seed를 섞거나 한 seed의 결과로 다른 seed를 생략하지 않는다.

## 2. Immutable 초기화와 provenance

두 arm은 optimizer state를 이어받지 않고 해당 seed의 P0 LAST2000 **model weights only**에서 시작한다.

| base | P0 warm start | checkpoint SHA-256 | model-state SHA-256 | completed manifest SHA-256 |
|---|---|---|---|---|
| 0 | `work_dirs/motiondrive_v2/p4_fresh_pretrain_s0/last.pth` | `8e91c30947a040f267f01f0642db3f2725900545deeb6d4087c795b5eb398a0c` | `f69e1c52ccd4b9908130ffe02170c0f08bff21712c634fee064517009a6c2c7d` | `9e5ba7ce89c0c1a9e4fe49476ff72c63b451f2658766e74e84c7336a4591fd98` |
| 1 | `work_dirs/motiondrive_v2/p4_fresh_pretrain_s1/last.pth` | `6067ba9cc543e6e1be837d85883696bc7ad12c3da9b970edba3008e30eff83c5` | `d595f0788fbb5ae31ea2eaa2f3344ac300028756dcfe8f313ec1dbb761673e0d` | `385ea53269e96d73c7d66f4196bc2e1205b0c6b7e9c7db537f983bc7aa37c421` |

공통 public-only I0는 `work_dirs/motiondrive_v2/p4_public_init_s0.pth`, file SHA-256 `06d2e68e15ca00d2c0f9ed3c965e198db603fe7e80007c39319d76eba9832e5a`, model-state SHA-256 `7ac28a8f26dc796be70ddb78edff1e1c128c64f43b5982747ba419d5d5200683`이다. Producer Git은 `1947cb33134dba9559c14f1e8c47301c6ef52c7e`, P0 training Git은 `806f0fa8c7447a7f66f59462977d87e00b492d0c`이다.

P0는 G0S0/plan-loss 0으로 2,000 updates를 수행했다. Planner parameter optimizer state는 존재하지만 planning gradient signal은 없었다. AdamW weight decay 때문에 planner tensor 일부가 I0에서 달라졌으므로 P0 planner가 I0와 byte-identical하다고 표현하지 않는다.

공통 데이터 provenance:

- split: `data/etri/motiondrive_v2/grouped_split_rawtime.json`, SHA-256 `f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936`
- C1 supervision manifest: `data/etri/motiondrive_v2/train_tune_geometry_v2/supervision_manifest.json`, SHA-256 `ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93`
- calibration: `data/etri/motiondrive_v2/train_tune_geometry_v2/calibration.npz`, SHA-256 `8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961`
- train: 203 scenes, 72 sessions, 54,810 rows; row SHA-256 `75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854`
- tune: 37 scenes, 11 sessions, 1,998 rows; row SHA-256 `1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88`
- final: 접근 0

## 3. 새 branch 고정 사양

새 branch는 다음 값으로 고정한다.

- branch width `d=32`
- spatial pooling `4×4`
- 32-dimensional cosine Q/K with cosine-score scale/temperature `8.0`
- distance scales `sigma=(10, 32/3) m`
- raw provided goal은 이 새 distance-score 계산에만 사용
- full branch 내부에서 autocast를 disable하여 normalization, Q/K/V/position/output projections, contractions, metric distance score와 masked softmax를 FP32로 실행; residual만 원 scene dtype으로 반환
- 새 branch는 non-backbone optimizer group에 포함
- 새 branch를 포함한 모든 원 joint trainable parameter를 학습

동결된 production/smoke source는 다음과 같다.

| path | SHA-256 |
|---|---|
| `models/motiondrive_v2/config.py` | `a3c54cfd4016bb91c1325d6799ea0cffc36c60e7363ee1a88f33536f84ac24b3` |
| `models/motiondrive_v2/cross_cell_goal_residual.py` | `92153cc2418ff6ae8091ff5f5b9cf32463d3c507e01ba75eb1275366571f0d60` |
| `models/motiondrive_v2/scene_encoder.py` | `99cad3d3fe52ba76842d7dbd03754e09b1d24928e2a05522b8f69b42de1d9808` |
| `scripts/train_motiondrive_v2.py` | `52eac2a28ba12f4724b7f18a3c2d6889241a58ec5b665ef44bd0e2f5966268d2` |
| `scripts/run_motiondrive_v2_p7_goal_routing.py` | `75b6908715a9181309ec586a18f3026518b419c922d1ec67b2a9f729d5a4c8cc` |
| `tests/test_motiondrive_v2_p7_goal_routing.py` | `5b4da5e640999c51eaf71ecb0bb7c9e18b90109af148c1365a40f81c6a304e99` |
| `scripts/smoke_motiondrive_v2_p7_training.py` | `82bc1832a3c7d728f522ea0679c8cc7e7f27932a73b041c87a6de146cfaa52fb` |
| `tests/test_motiondrive_v2_p7_training_smoke.py` | `bc2c58ee4e640a491e22b0b5843b0f4ee45a2511b33772a638ffa85cddc547b8` |

Production 13-file runtime closure는 `reports/p7_goal_routing_source_manifest_20260908_ops.json`, SHA-256 `880c3cc36ababe31c58376ab815a239e337f130fc9b373a041fea6ad4a8d1ea4`에 exact set으로 고정한다. Production source commit은 `64e92694ccbe0a3c5a581e421e16a884b9c5a088`이며, 뒤의 기록-only commit 때문에 전체 HEAD가 달라지는 것은 runtime source drift가 아니다. Branch manifest의 canonical fields는 `attention_precision: fp32_autocast_disabled`와 `new_value_projection_adds_goal_or_position: false`이며 다른 이름으로 바꾸거나 누락하지 않는다.

## 4. 공통 joint6000 학습 계약

P7은 원 P4 joint6000 recipe를 유지한다.

- planning loss weight `1`
- 기존 occupancy/lane/history/state auxiliary losses 유지
- 기존 unweighted stop loss 유지; P6 global balance는 사용하지 않음
- AdamW, weight decay `0.01`
- non-backbone/head LR `1e-4`, backbone LR `1e-5`
- 200-update linear warmup 후 cosine schedule
- gradient clip `5`
- logical batch `16`, microbatch `2`
- fixed BN running statistics; affine 및 기존 trainable weights는 학습
- C1 geometry, nominal time
- BF16 encoder, FP32 planner
- 각 arm 정확히 6,000 updates
- own P0 model weights only warm start; fresh optimizer/scheduler step 0
- LAST6000 생성 뒤 tune 1,998행/11 sessions를 정확히 한 번 평가

금지:

- BEST checkpoint 생성 또는 선택
- 중간 tune 평가
- schedule 연장, 추가 seed, threshold fitting 또는 결과 기반 재시작
- final holdout 접근
- 한 seed 또는 좋은 checkpoint만 고르는 보고
- 결과를 보고 자동으로 counterfactual, deployment, export 또는 3090 측정에 진입

## 5. 실행 무결성 및 자원 계획

P7 source file SHA allowlist는 위 source manifest와 각 execution receipt에 기록한다. 전체 Git HEAD equality를 외부 정상 commit에 대한 실패 조건으로 사용하지 않는다. 대신 다음을 각 arm의 시작 전/종료 후 exact 검사한다.

- P7 source allowlist bytes
- P0 checkpoint와 manifest
- split, C1 supervision manifest 및 calibration
- C/G 동일 초기 model-state SHA와 새 branch 초기 state SHA
- C/G 동일 sample-order SHA와 RNG/augmentation contract
- actual parent/child PID, native OS rc, nonfinite count, optimizer step 6000
- fixed-BN state, LAST strict CPU load, input/source before-after equality
- tune 평가 정확히 1회, 1,998 rows/11 sessions

제안 실행 순서는 한 번에 두 arm이다.

1. wave 1: GPU4=`base0/C`, GPU5=`base0/G`
2. 두 arm의 terminal·LAST·PID absence·source/input 불변과 자원 회수를 확인
3. wave 2: GPU4=`base1/C`, GPU5=`base1/G`
4. 네 arm 결과를 모두 고정한 뒤 멈춤

Wave 사이 gate는 technical completion, artifact integrity와 resource validation만 확인한다. Base0의 정확도나 treatment 방향이 나쁘다는 이유로 base1을 생략하지 않는다. Genuine OOM, nonfinite, source/input drift 또는 소유 process failure가 있을 때만 중단해 root 판단을 기다리며, base0 결과로 schedule·loss·seed·branch를 바꾸거나 연장하지 않는다.

네 arm을 GPU4/5에 두 개씩 나누는 이유는 단순 memory capacity 부족이 아니다. 기존 joint 계열의 단일 process peak reserved는 약 5.6 GiB이고 각 process cap은 12,000 MiB라 B200 183,359 MiB에는 두 process/card도 명목상 들어간다. 그러나 같은 card의 두 trainer는 allocator reserve, cuDNN workspace와 kernel scheduling을 서로 교란해 paired wall time·peak memory를 비교하기 어렵고, pressure/OOM 발생 시 원인과 소유 PID 귀속도 흐린다. GPU당 하나씩 두 wave로 실행하면 현재 검증된 single-owner guard를 그대로 쓰며 C/G의 resource 조건을 대칭으로 유지한다.

GPU4 UUID는 `GPU-4b804d68-fd61-af14-393a-573c533d5006`, GPU5 UUID는 `GPU-1e9aea73-4e6b-2cb5-b788-f3d99e6dc6f8`이다. GPU0–3의 외부 작업은 조회 외에 건드리지 않고 GPU6/7은 사용하지 않는다. Pressure/OOM 처리는 own parent/child PID에만 적용하며 자동 설정 변경이나 재시작을 하지 않는다.

Base0 C/G 3-update smoke의 실제 peak reserved는 양 arm 각각 `5,949,620,224 B` (`5,674 MiB`), peak allocated는 `5,364,743,680 B`였다. 이는 짧은 plumbing 관측이지 6,000-update peak의 보장은 아니다. Main도 고정 `12,000 MiB` allocator cap과 `8,192 MiB` launch reserve를 사용한다. Cap 또는 batch/microbatch를 바꾸면 새 protocol revision이 필요하다.

### 독립 3-update plumbing smoke

Main6000 전에 별도 `scripts/smoke_motiondrive_v2_p7_training.py`로 base0 C/G를 GPU4/5에서 각각 정확히 3 updates만 실행한다. Final recipe의 batch16/micro2, fresh AdamW, LR group, warmup200의 첫 3 factor, fixedBN, C1/nominal, BF16 encoder와 full-FP32 새 branch를 그대로 쓴다. Train loader만 만들고 tune/final loader·평가·checkpoint를 만들지 않는다. Q/K/V/output projection gradient의 finite/nonzero 전개, finite total gradient, optimizer step3, source/input 불변, memory cap `12,000 MiB`와 reserve `8,192 MiB`, actual parent/child rc 및 PID absence를 기록한다. 이 ephemeral state는 main initialization에 재사용하지 않고, accuracy 또는 학습 결과로 보고하지 않는다.

이 smoke는 완료됐다. C/G 모두 actual child와 unified SSH exit `0`, exact 3 updates, 동일 initial state와 48-row sample order, finite loss/gradient, fixed-BN 불변 및 optimizer step3을 통과했다. Update 1에서는 zero-init output projection만 nonzero gradient이고, updates 2–3에서는 Q/K/V/output 모두 finite nonzero gradient였다. 결과는 `reports/p7_goal_routing_canary_b0_{control,goal}_20260908_ops/canary.json`, 실행 receipt는 `reports/p7_goal_routing_canary_b0_execution_20260908_ops.json` SHA-256 `c07e51b6cb09791a26273256a241f3b4992db8be1872757fec631b7fcb27e68b`이다. 이는 plumbing PASS이며 accuracy 결과가 아니다.

### Main6000 exact operator matrix

각 row는 `/usr/bin/python`, logical `--gpu 0`, `--workers 4`, `--cuda-memory-limit-mib 12000`, `--cuda-min-free-mib 8192`, 위 source manifest/SHA를 사용한다. `CUDA_VISIBLE_DEVICES`와 `--expected-physical-gpu-uuid`는 같은 표의 UUID여야 한다. 모든 run-dir은 launch 직전까지 absent여야 한다.

| wave | physical GPU | arm | base seed | P0 checkpoint / manifest | run-dir |
|---|---:|---|---:|---|---|
| 1 | 4 | `control_zero_slot` | 0 | `p4_fresh_pretrain_s0/last.pth` / `manifest.json` | `work_dirs/motiondrive_v2/p7_goal_routing_b0_control_last6000` |
| 1 | 5 | `goal_real_slot` | 0 | `p4_fresh_pretrain_s0/last.pth` / `manifest.json` | `work_dirs/motiondrive_v2/p7_goal_routing_b0_goal_last6000` |
| 2 | 4 | `control_zero_slot` | 1 | `p4_fresh_pretrain_s1/last.pth` / `manifest.json` | `work_dirs/motiondrive_v2/p7_goal_routing_b1_control_last6000` |
| 2 | 5 | `goal_real_slot` | 1 | `p4_fresh_pretrain_s1/last.pth` / `manifest.json` | `work_dirs/motiondrive_v2/p7_goal_routing_b1_goal_last6000` |

정확한 wrapper CLI shape는 다음과 같고 `<...>`는 위 표와 §2의 exact base별 path/SHA/UUID로만 치환한다.

```text
CUDA_VISIBLE_DEVICES=<PHYSICAL_UUID> PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/python \
scripts/run_motiondrive_v2_p7_goal_routing.py \
  --arm <control_zero_slot|goal_real_slot> --base-seed <0|1> \
  --init <OWN_P0_LAST> --expected-init-sha256 <OWN_P0_FILE_SHA256> \
  --run-manifest <OWN_P0_MANIFEST> \
  --expected-run-manifest-sha256 <OWN_P0_MANIFEST_SHA256> \
  --source-manifest reports/p7_goal_routing_source_manifest_20260908_ops.json \
  --expected-source-manifest-sha256 880c3cc36ababe31c58376ab815a239e337f130fc9b373a041fea6ad4a8d1ea4 \
  --data-root /NHNHOME/data/sukim/adcl \
  --split-manifest data/etri/motiondrive_v2/grouped_split_rawtime.json \
  --supervision-root data/etri/motiondrive_v2/train_tune_geometry_v2 \
  --run-dir <NEW_EXACT_RUN_DIR> --expected-physical-gpu-uuid <PHYSICAL_UUID> \
  --gpu 0 --workers 4 --cuda-memory-limit-mib 12000 --cuda-min-free-mib 8192
```

Driver가 생성하는 trainer argv는 §4의 fixed recipe를 exact 적용한다. Wave 2는 wave 1의 technical terminal/integrity/resource gate 후 실행하며 accuracy를 보고 생략하지 않는다.

## 6. 평가·통계 보고

네 LAST6000을 모두 개별 보고한다.

원 P4 same-base immutable 비교 report는 다음으로 고정한다.

- base0: `reports/p4_joint_full_tune_eval_s0_20260908_ops/normal_full_tune.json`, SHA-256 `cffa9cc8b12f5f9812af7b3fd3631e74597e291b02aecd1cc14d8ec8a1aeba2f`
- base1: `reports/p4_joint_full_tune_eval_s1_20260908_ops/normal_full_tune.json`, SHA-256 `0e71bef199931cb1916abdc819ca78cf75d8cd7055439bd084ca975bb610c137`

이 두 파일은 downstream metric reference이며 training input이나 checkpoint 선택 기준이 아니다.

- official weighted D3 전체값
- 각 base의 `G − C`
- 각 base의 `G − original P4 LAST6000`
- 두 seed의 G/C mean과 range
- steady 99 / depart 24 / nonstop 1,875
- session046 제외
- goal in-grid / out-of-grid breakdown과 count. In-grid는 raw metric goal에서 `-10 <= x <= 70` 및 `-32 <= y <= 32`를 모두 만족하는 경우다. 경계는 inclusive이고 clipping하지 않으며, 그 외는 out-of-grid다. 이 분류는 descriptive report에만 쓰고 선택·threshold·loss에 쓰지 않는다.
- source/input/LAST/report SHA, actual OS/native rc, optimizer step, finite, fixed-BN 및 PID 부재

통계 계약:

- 네 report의 1,998개 row/scenario/session/frame 순서와 GT가 exact여야 하며 불일치하면 fail closed
- mean `G − C`는 row별 float64 delta `0.5*((G0-C0)+(G1-C1))`로 구성
- sorted 11 sessions에 같은 cluster resample을 적용하는 shared-session paired bootstrap
- `repeats=10000`, `seed=20260908`, NumPy default linear 2.5%/97.5% quantile
- frame-weighted estimator; 3,996행/22 cluster concatenate 또는 seed bootstrap 금지
- 특정 seed, arm, checkpoint 또는 subgroup만 골라 primary 결론을 만들지 않음

## 7. 사전등록 KEEP 기준

G를 후속 채택 검토 대상으로 유지하려면 모두 만족해야 한다.

1. base0과 base1 각각 `G < C`
2. base0과 base1 각각 `G < original P4`
3. 두 seed mean `G − C <= -0.015`
4. shared 11-session paired-bootstrap의 mean `G − C` 95% CI upper `< 0`
5. 두 seed mean G official weighted D3 `<= 0.34`

하나라도 실패하거나 metric/source/input/row alignment가 nonfinite 또는 불일치이면 fail closed한다. `.30`은 더 강한 장기 목표이지 P7의 예측, 하한 또는 성공 약속이 아니다. P5의 약 `.30` 값은 P4 path와 exact ZERO 두 후보를 GT로 완벽 선택한 oracle 하한 진단이며, P7 전체 모델 학습의 성능 하한과 동일하지 않다.

## 8. 후속 채택 gate와 해석 경계

KEEP 수치 조건을 만족해도 다음은 별도 승인·실측 전까지 PENDING이다.

- image counterfactual
- goal counterfactual
- value/score counterfactual
- 실제 RTX 3090에서 전체 11-image forward latency·parity·state-hash·smoke
- deployment/export 및 submission 변경

새 branch의 단위·통합 테스트는 구현 계약을 확인할 뿐 규정 심사, 공식 성능, 독립 holdout 또는 경쟁 1위 통과를 보장하지 않는다. Reused tune 결과는 exploratory evidence로만 해석한다.

Zero-init canary는 새 branch의 full-graph 실행, 초기 default-function equivalence와 latency 경계만 확인한다. 이것만으로 learned goal use나 accuracy를 입증하지 않는다. 고정된 nonzero-`W_o` unit intervention은 score-only 및 constant/zero-value invariants를 확인할 수 있지만, 학습된 checkpoint의 image/goal/value counterfactual은 여전히 별도 adoption gate다.

## 9. RTX 3090 full-forward latency canary 계약

이 절은 두 단계를 구분한다. 먼저 own-P0에서 deterministic하게 조립한 zero-init C/G graph의 technical latency canary를 수행할 수 있다. 이는 learned P7 checkpoint가 아니며 full graph 실행·초기 C/G function equivalence·architecture latency만 본다. P7 KEEP 뒤의 learned deployment canary는 별도 checkpoint/export 검토와 root 승인이 필요한 후속 단계다. 어느 단계도 현재 timing 결과를 주장하지 않는다.

### A. Zero-init own-P0 technical canary

- 대상은 `base0/C`, `base0/G`, `base1/C`, `base1/G` 네 deterministic CPU assemblies다.
- 각 base의 exact P0 checkpoint와 completed sidecar를 직접 SHA pin하고, frozen P7 driver의 strict old-key load allowlist와 seed로 새 branch를 조립한다.
- 각 base C/G의 assembled model-state SHA가 exact 동일해야 하며, 이 SHA는 B200 CPU preflight에서 얻어 3090 CLI에 외부 pin한다.
- Historical P4 exporter나 source manifest를 P7 것으로 재라벨하지 않고 deployment bundle을 만들지 않는다.
- Actual wrapper `scripts/smoke_motiondrive_v2_p7_deployment.py` SHA-256 `74e25593fec5f187e54f6ea5732aa4c120b81ad786b9089c88d09b503273eae3`와 test `tests/test_motiondrive_v2_p7_deployment.py` SHA-256 `6da672b0d6b891570d56935c9a78bf9cd5ba16ec76b06ec031e64152ab1509ff`는 main training source 밖의 별도 validation 도구다. 독립 검토에서 dedicated 11 tests, adjacent P7/serving/export 239 tests, compile/help가 actual rc0였고 code blocker는 0이었다. 실제 3090 실행은 별도 source/input/container preflight가 통과해야 한다. 이 gate는 main scientific source/recipe의 launch gate와 분리한다.

제안 CLI shape는 다음과 같다. 실제 source/test SHA와 manifest SHA는 wrapper 독립 검토 뒤 freeze한다.

```text
python scripts/smoke_motiondrive_v2_p7_deployment.py \
  --init <OWN_P0_LAST2000> --expected-init-sha256 <P0_SHA256> \
  --run-manifest <OWN_P0_SIDECAR> --expected-run-manifest-sha256 <SIDECAR_SHA256> \
  --expected-initial-model-state-sha256 <ASSEMBLED_SHA256> \
  --training-source-manifest <P7_TRAIN_SOURCE_MANIFEST> \
  --expected-training-source-manifest-sha256 <SHA256> \
  --validation-source-manifest <P7_VALIDATION_SOURCE_MANIFEST> \
  --expected-validation-source-manifest-sha256 <SHA256> \
  --arm <control_zero_slot|goal_real_slot> --base-seed <0|1> \
  --fixture-root <READ_ONLY_TRAIN8_FIXTURE> --reference <READ_ONLY_REFERENCE_PT> \
  --calibration <READ_ONLY_CALIBRATION_NPZ> \
  --geometry-contract <READ_ONLY_C1_MANIFEST> \
  --expected-physical-gpu-uuid GPU-8768d7a1-6e1b-1aae-4774-248c9d28b887 \
  --device cuda:0 --precision bf16 --warmup 20 --repeats 50 \
  --output <NEW_REPORT_JSON>
```

P0 sidecar는 dead metadata가 아니라 actual ordinary file로 직접 읽어 exact SHA와 completed/step2000/source/data lineage를 검사한다. Training source manifest와 validation/timing source manifest는 Git provenance를 분리하고 각각 exact file closure를 검증한다.

### B. Learned deployment canary

P7 KEEP 이후 별도 승인되면 C/G와 base seed 효과를 숨기지 않도록 네 LAST6000 deployment artifact를 각각 측정한다. 이 단계의 P7-specific exporter/bundle schema는 아직 PENDING이며 historical P4 exporter를 검토 없이 재사용하지 않는다.

### 측정 대상과 고정 조건

- zero-init 또는 learned 단계에서 지정된 네 C/G×base graph를 모두 개별 측정
- batch size 1의 완전한 model forward; current 6-camera encodings와 current-front/history 4 encodings을 합친 총 11 image encodings
- BF16 image encoder, FP32 planner와 새 FP32 goal-attention branch
- feature cache, partial-forward API 또는 cached backbone output 사용 금지
- preregistered corrected-geometry train8 fixture의 동일 8 clips만 사용; final holdout read 0
- clip마다 warmup 20, timed repeats 50; bundle마다 timed full forward 총 400회
- 각 repeat에서 `torch.cuda.synchronize`, CUDA start/end events, full `model(**inputs)`, synchronize, elapsed event time 순서
- preprocessing, checkpoint/config/source loading, H2D, D2H, JSON serialization과 postprocessing은 timed interval 밖에 두고 별도 wall-time field로 기록
- TF32 off, cuDNN benchmark off, cuDNN deterministic on, deterministic algorithms on
- network none, read-only root filesystem, source/fixture/bundle mounts read-only, 새 report mount만 read-write

Checkpoint와 config loading은 측정 전에 반드시 수행하고 evidence에는 포함하되 latency sample에는 포함하지 않는다. Zero-init 단계는 P0 LAST2000/sidecar와 assembled state를 검사한다. Learned 단계는 LAST6000 step/phase, completed sidecar, source checkpoint, complete model config, branch arm/base, FP32 attention 설정 및 input contract를 외부 SHA receipt와 exact 비교한다. 모든 input은 ordinary non-symlink file이어야 하며 load 전후 SHA가 같아야 한다.

### Branch 실행 증거

C와 G 모두 같은 goal-attention module graph를 실제 실행한다. C는 새 branch slot 입력만 zero이고 기존 per-cell goal은 real이며, G는 raw goal distance score를 제공한다. 초기 또는 학습된 output projection `W_o`가 zero라고 해도 Q/K projection, cosine score, scale `8.0`, distance term, softmax/value aggregation과 output projection kernel을 건너뛰지 않는다. Eager forward에 arm별 conditional skip을 두지 않는다.

실제 canary는 untimed instrumentation으로 C와 G 각각에서 branch와 Q/K/V/position/output modules, 두 contraction `einsum`, masked softmax가 호출되고 projection/contraction/score/softmax의 실제 dtype이 FP32임을 확인한다. 원 scene 입력과 residual이 합쳐진 branch 반환 dtype도 기록한다. Python hooks/wrappers는 module/function execution과 tensor dtype/count를 증명할 뿐 exact CUDA kernel identity를 증명한다고 표현하지 않는다. Instrumentation은 `finally`에서 원 함수를 복원하고 timed forwards에는 설치하지 않는다. Zero `W_o` 때문에 출력 기여가 0일 수는 있지만 branch execution을 생략한 latency로 인정하지 않는다. Compiler constant folding, graph specialization 또는 branch bypass가 관측되면 fail closed한다.

### Source·input·runtime pin

- 실행 직전 P7 training Git, validation/export Git과 exact source allowlist를 서로 구분해 기록
- frozen P7 source archive SHA, file count와 source manifest SHA를 기록하고 container 안팎에서 before/after exact 검사
- zero-init 단계는 P0 checkpoint/sidecar/assembled state SHA, learned 단계는 LAST/sidecar/export artifact/model-state SHA를 기록
- train8 fixture manifest, reference tensor, calibration, geometry contract 및 raw 8-clip tree SHA를 P4 계약과 동일하게 고정
- Docker image `adcl-latency-runtime:cu128`, image ID `sha256:7c0f4aaf98f2f3391d30d426e6217f524c8717aaa6b7d431fcd6f71740aa15c7`
- GPU UUID `GPU-8768d7a1-6e1b-1aae-4774-248c9d28b887`; launch 전후 idle/compute PID/container 상태 기록
- Python, PyTorch, CUDA build, driver, GPU name, autocast, TF32와 deterministic flags를 child process에서 직접 기록

새 validation source manifest와 learned 단계의 export receipt는 각각 독립 review 뒤 exact path/SHA로 채운다. P4 source manifest나 과거 37 ms 결과를 P7 source 또는 P7 timing으로 복사하지 않는다.

### 제안 launch와 증거물

Host parent는 Docker image ID와 all mount sources를 pin한 뒤 한 container만 실행하며, actual host PID/container ID/child PID/argv/start/end/OS rc와 stdout/stderr SHA를 별도 immutable execution receipt에 쓴다. 동일 GPU에서 네 graph를 순차 실행하고 각 container가 끝나 PID/container absence와 GPU memory recovery를 확인한 뒤 다음 graph를 시작한다. 실패를 자동 retry하거나 조건을 바꾸지 않는다.

각 report/receipt는 다음을 포함한다.

- 8 clips × 20 warmup, 8 × 50 measured count exact
- clip별 및 pooled CUDA median/mean/std/min/max/p90/p95/p99와 wall-time 통계
- H2D와 D2H/postprocess 별도 통계, timed-boundary 문자열
- all 11 encodings, BF16 encoder, FP32 planner/attention hook proof
- complete output finite/shape checks, raw/reference parity, A/B/A stateless repeat, model state before=after
- source/bundle/fixture/raw bytes before=after, no final read, no cache
- peak allocated/reserved, pre/post GPU memory/process/container state
- actual native OS rc; 실패와 성공을 덮어쓰거나 재라벨하지 않음

사전등록 timing gate는 각 측정 대상에서 pooled CUDA-event full-forward median과 p95가 모두 실제 RTX 3090에서 `100 ms` 미만인 것이다. 하나라도 미달하면 latency gate는 실패한다. 이 기준은 learned accuracy·compliance gate와 독립이며, 과거 P4의 약 37 ms를 P7 측정값으로 대체하지 않는다.

이 canary는 latency·parity·statelessness 증거일 뿐 P7 정확도, 규정 또는 production 채택을 단독 승인하지 않는다.

## 10. 복구점과 기존 결과 보존

- P4 original training source Git: `86620b4ffc7e6838b49cf83b5be789eba12d8027`
- P4 joint supervisor는 외부 global-HEAD-only drift 때문에 original outer rc1을 보존하며 clean training rc0로 재라벨하지 않음
- P4 independent artifact/eval evidence: `MOTIONDRIVE_V2_P4_JOINT_RESULTS_20260908.md`, `reports/p4_joint_results_summary_20260908.json`, `reports/p4_joint_head_drift_audit_20260908/`
- P6 training source/recovery Git: `56753e775674204ab0b9241513d61dd819ec2ff5`
- P6 final record recovery Git: `2dbbaff5ca1c8da80ca858f195f2f939c07cd95e`
- P7 production source commit: `64e92694ccbe0a3c5a581e421e16a884b9c5a088`
- P7 source/preflight record commit: `ba6700b4c5781c232abeaa5d5412e87e0b545425`
- P7 3-update smoke record commit: `e7a595475af218e8230dfc291942f705e15b2af8`
- P6 protocol/results: `MOTIONDRIVE_V2_P6_STOP_BALANCE_PROTOCOL.md`, `reports/p6_stop_balance_execution_20260908_ops.json`, `reports/p6_stop_balance_results_20260908_ops.json`
- P6 fixed global-balance recipe는 preregistered KEEP를 실패해 불채택이며 P7에 포함하지 않음

기존 P0/P4/P6 checkpoint, sidecar, raw report 및 실패 기록은 수정·삭제하지 않는다. P7 source를 추가할 때도 외부 untracked/staged 파일과 large checkpoint/cache를 Git에 포함하지 않는다.

## 11. 2026-09-08 KST 준비 시점 자원 점검

B200 `/NHNHOME/data/sukim/adcl`:

- Last verified Git HEAD after P7 smoke records `e7a595475af218e8230dfc291942f705e15b2af8`, tracked clean
- 72 logical CPUs
- RAM 2,317,624 MiB total, 2,213,354 MiB available, swap 0
- load average `4.84 / 5.30 / 11.60`
- data filesystem 220 T total, 138 T available, 38% used
- `iostat` unavailable; 설치하지 않았으며 이 점검에서 정량 disk-throughput은 측정하지 않음
- GPU4/GPU5 각각 0 MiB used, 182,632 MiB free, utilization 0%, compute PID 없음
- GPU0–3에는 외부 `cosmos_distillation` processes가 있고 그대로 보존
- system Python 3.12.3, PyTorch `2.10.0a0+b4e4ee81d3.nv25.12`, CUDA 13.1; CPU probe에서 CUDA initialized false

RTX 3090 host read-only 점검:

- SSH endpoint `intern@192.168.10.182` (기본 SSH config/key; key 내용 기록 금지)
- GPU `GPU-8768d7a1-6e1b-1aae-4774-248c9d28b887`, NVIDIA GeForce RTX 3090
- 130 MiB used, 23,986 MiB free, utilization 0%, compute PID 없음
- running Docker container 없음
- runtime source `/home/intern/adcl_latency/runtime` 존재
- Docker image `adcl-latency-runtime:cu128`, image ID `sha256:7c0f4aaf98f2f3391d30d426e6217f524c8717aaa6b7d431fcd6f71740aa15c7`

이 점검은 현재 자원 여유가 4-run을 GPU4/5에 두 개씩 두 wave로 배치할 수 있음을 시사한다. 그러나 실행 가능성 판단일 뿐 launch 승인이 아니며, implementation freeze·독립 review·run-dir absence·즉시 재점검과 root의 별도 승인이 모두 필요하다. 이 초안 작성 중 GPU training, model forward, 3090 container 실행 및 remote write는 0이다.
