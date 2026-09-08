# MotionDrive V2 shared-status A1 fixed protocol (2026-09-08)

Status: preregistration candidate; no A1 overlay, training, evaluation, or claim exists yet.

## Question and boundary

Does a legitimate causal provided ego status improve adaptation when it modulates the
continuous image-derived shared scene raster, rather than entering the planner as a
raw token or trajectory shortcut? This is a bounded continuation screen, not a
from-scratch comparison or global upper bound.

The experiment uses current grouped train203 (54,810 rows) and tune37 (1,998 rows)
only. Final136 is inaccessible. It changes no camera count, history frames, labels,
heads, losses, planner signature, raw-motion estimator, goal path, or output horizon.

## Fixed paired arms

Both arms use seed 0, the exact P7 control b0 LAST6000 parent, identical new parameter
initialization, sample order, augmentations, parameter count and training schedule.

- `zero`: supplied status is exactly five zeros. After step 0 it may learn a constant
  channel gain through the final gamma bias, so it is an extra-capacity control rather
  than branch-off.
- `provided_causal_5d`: `(vx,vy,ax,ay,yaw_rate)` is computed from exactly 11 past/current
  ego poses at nominal 10 Hz, current-10 through current. Fixed physical scales are
  `[10,5,3,3,.5]`.

The ego-pose parquet may be read in full for the identity join, but only the 11 causal
poses are passed to the status helper. Future values, goal, command, HD labels, plan
labels and cached `state_target` are not used to form this input. The train and terminal
evaluation adapters consume the same frozen status overlay; its valid and invalid row
counts are recorded.

## Fusion and information route

The one new module is inserted after image attention, global image context and the
existing P7 zero-slot cross-cell residual, but before the existing two SpatialMix
refinement blocks. For scene raster `F[B,128,48,64]`:

`gamma = MLP(status/scales)`, with `5 -> 32 -> 128`, GELU and biases;
`F' = F * (1 + tanh(gamma))`.

The final gamma Linear weight and bias are both exact zero at initialization. The whole
new branch executes in FP32 with outer autocast disabled, and only `F'` is cast back.
There is no additive beta/status feature, direct planner status argument, status query,
trajectory residual, conditional skip, raw-goal addition, command or HD-label route.
The same fused raster feeds occupancy, lane and planning scene features. Existing
image-derived state/history and 192 motion tokens continue unchanged.

## Initialization and optimization

Parent checkpoint SHA256 is
`6145255ab2b1efd3deb169b873896e9b1b623ca49f11e5a46a71b138e3b0355e`, model-state
SHA256 `e79d545bbb09b4109c783a8cf4c861772bc5e52c62004ef5b37dc4ca63e938a2`, and completed
sidecar SHA256 `cf287d05ce6b2f4476c07ac5c2253cd9ecb3227bcc01404adbbba8463bb66de0`.
Loading is weights-only and the optimizer is fresh. All original and new parameters are
trainable under the existing fixed-BN-running-statistics policy.

Training is exactly 2,000 optimizer updates, logical batch 16/microbatch 2, BF16 outer
autocast, AdamW weight decay .01, backbone LR `5e-6`, other/new-head LR `5e-5`, warmup
100 then cosine decay, global grad clip 5. Losses remain plan 1, occupancy .2, lane .2,
motion .2 with unchanged uncertainty and unweighted stop handling. A rolling LAST
checkpoint is persisted at steps 500/1000/2000; none is evaluated or selected.
Exactly one tune evaluation occurs at terminal step 2000 and LAST2000 is the report.

## Required gates before execution

- frozen source, split, supervision, calibration, parent and overlay identities;
- exact 54,810/1,998 row identities, finite status and zero invalid rows;
- identical complete initial model and fusion-state SHA across arms;
- exact full parent outputs at initial identity for both arms;
- status-context restoration after success and exception, and reentrant/stale rejection;
- train and terminal-evaluation input adapters preserve the selected overlay status;
- raw-record versus overlay status parity and invariance to future/goal/GT mutations;
- unchanged planner/raw-motion signatures, fused-raster use by occupancy/lane/planning,
  finite gradients from each, initial zero-arm final-bias gradient, and later provided
  input-weight/status-conditioned activation.

GPU assignment is fixed: zero on physical GPU0 UUID
`GPU-5d2254f9-41a7-62dd-2b38-de82459acb24`, provided on physical GPU1 UUID
`GPU-041334c0-089c-6ff5-b0b5-59ff445fa015`, each isolated as logical CUDA device 0.

No result threshold or adoption decision is added here. Any result is specific to this
parent continuation and must not be described as a status-only planner or P1--P8 result.

Before either run, a separate output directory containing `smoke_only` receives exactly
two optimizer updates with the final recipe and no tune evaluation. Its rolling LAST is
explicitly non-scientific and must never initialize or be reported as an A1 trained run.
