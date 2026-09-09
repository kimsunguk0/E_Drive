# MotionDrive V2: image-derived ego-state precision screen (S1)

Protocol frozen before any training result. Date: 2026-09-09 KST.
Scope: one paired 2-arm 2,000-update continuation on idle physical GPUs 5 and 6.
No Goal change, no final136 access, no automatic follow-on search.
This is an internal tune experiment, not a leaderboard score.

## Question

The planner is handed an image-predicted ego state that is far worse than what
the same network demonstrably knows. Does repairing that estimate, and making
the planner actually consume it, reduce official D3?

This tests one bounded recipe. It does not test a single-frame information
limit, nor the general value of ego-state conditioning.

## Why this experiment (pre-registered diagnosis)

Measured on the parent checkpoint over tune1998 with the official metric:

- official D3 0.2939324227.
- Along-track (timing) weighted error 0.2023 vs cross-track 0.1331; the
  longitudinal:lateral RMS split is 83:17. Re-parameterizing the model's own
  predicted path by the ground-truth arclength gives D3 0.0765. Path geometry
  is not the binding constraint; the longitudinal profile is.
- `state_hat` vx MAE 1.015 m/s, while `history_hat` position MAE by past offset
  is [0.112, 0.211, 0.510, 1.006] m, and the plan's own first waypoint implies
  v0 with MAE 0.277 m/s. The planner is not using `state_hat`.
- A kinematic reference curve fit on train203 and evaluated on tune1998 maps
  ego-state accuracy to achievable D3: sigma 1.0 -> 0.308, 0.3 -> 0.229,
  0.0 -> 0.181. The model behaves as if it knew v0 to only ~0.87 m/s.
  This curve is a sizing diagnostic. The mapping it embodies is itself a
  prohibited construction and is never used as a model, a target, or a
  post-processing correction.

Structural hypothesis: in `models/motiondrive_v2/motion_encoder.py`,
`state_hat` is regressed from `motion`, which is the per-timestep pair tokens
collapsed by a softmax over time. That collapse destroys the displacement-to-dt
association velocity estimation needs. `history_hat` keeps per-timestep
evidence and is accurate. The two interventions are coupled and are therefore
screened as one arm: a better estimate is inert if the planner ignores it, and
forcing the planner onto today's estimate would inject a 1.015 m/s error.

## Compliance boundary

Governing rulings in `OPEN_ISSUE.md`: provided ego status, provided pose or
past trajectory, and the target point must not enter the planner directly or as
a simple embedding (Q7); a network's own image-inferred history/status may be
used in the planner (Q10). Design of record: `ETRI_MOTIONDRIVE_V2_EXECUTION_SPEC.md`
sections 4.5 and 4.6.

Both arms must satisfy, and tests must assert:

- `MotionEncoder.forward` takes exactly `(current_front_levels,
  history_front_levels, time_offsets)`. No pose, status, goal or command
  argument is reachable.
- `Planner.forward` keeps the signature `(scene_features, motion_features,
  predicted_state, predicted_history)`. Every quantity it consumes is produced
  by the image network in the same forward pass.
- No provided-status overlay, `ego_cache` kinematic field, or goal value is read
  inside either module.
- No post-forward trajectory correction of any kind.

## Data

- grouped train203 (54,810 rows, stride 1, frame >= 30) and tune37
  (1,998 rows, 11 sessions, stride 5, frame >= 30). Unchanged.
- Split SHA256 `f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936`.
- Train row SHA256 `75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854`.
- Tune row SHA256 `1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88`.
- final136 rows are not selected, trained, evaluated or analyzed.

## Parent and arms

Parent: `work_dirs/motiondrive_v2/pv_screen_b0_direct_last2000/last.pth`,
SHA256 `a72a958a097d895afcfa82075814765c76c96f59f5f59e93e21504be83c8b14d`,
reported tune D3 0.2939324227. Pin checkpoint, model-state and sidecar hashes
before launch. Never initialize from a smoke checkpoint.

1. `continuation_control` — physical GPU 5. Bitwise-identical continuation of
   the current code path with the new config flags off.
2. `state_precision` — physical GPU 6. Per-timestep, dt-aware state head plus
   planner query conditioning on `state_hat`/`history_hat`.

Both: seed 0, identical row order, identical augmentation, fresh optimizer,
2,000 updates, batch 16 / microbatch 2, bf16 backbone with FP32 planning/loss,
backbone LR 5e-6, head LR 5e-5, warmup 100, weight decay 0.01, gradient clip 5,
occ/lane/motion weights 0.2, fixed BN statistics, all upstream parameters
trainable. Evaluate once at terminal step 2,000. Training checkpoints are
recovery and diagnostic artifacts, not a license for tune selection.

## Pre-registered gates

Primary (KEEP): `state_precision` minus `continuation_control` official D3
<= -0.010 m, and ADE1 not worse.

Mechanism (must also hold for the causal claim, not for KEEP):
`state_hat` vx MAE improves by >= 0.30 m/s against the contemporaneous control.
If D3 improves while vx MAE does not, the result is recorded as an unexplained
capacity effect and the stated mechanism is not supported.

Diagnostic expectation, not a gate: the reference curve above sizes the
available move at roughly -0.06 m. A smaller realized effect does not
retroactively relax the KEEP threshold, and a larger one does not license
skipping replication.

Report both terminal results including failure. No threshold is relaxed after
seeing the numbers. No weight, LR or arm search follows a failed gate.

## Verification and reporting

Before launch: source manifest pinned and verified, complete key-load check,
control-arm bitwise output equality against the pre-change implementation,
step-zero equality between arms, exact row identity, finite gradients, real
TRAIN smoke, paired row-order check, GPU memory headroom, live recheck that
physical GPUs 5 and 6 are idle, and foreign GPU 0-4 work left untouched.
Preserve dirty files; explicit-path commits; no push. Record real exit codes
with durable process ownership and no automatic retries.

Primary report: terminal tune D3 and ADE1 per arm, plus the 11-session paired
bootstrap CI. Also record `state_hat` vx/vy/ax/ay/yaw-rate MAE, history
position MAE by offset, plan-implied v0 MAE, and the along-track/cross-track
decomposition, so the mechanism is separable from the outcome.

tune1998 has been reused across P1-P8 and this screen. Bootstrap intervals are
row/session uncertainty summaries, not independent-holdout certification, and
are not the official clip-final server score.

## Amendment A1 (recorded 2026-09-09 KST, before any arm was launched; no result existed)

The KEEP threshold stays -0.010 m. Rationale, pre-registered: contemporaneous
same-parent arms in this repository differ by about 0.001-0.002 m (the three
early-precision arms scored 0.2987348, 0.2988911, 0.2989621), so -0.010 m is
several times the paired noise, and it is the same threshold used in P1 and P3,
which keeps this screen comparable with the earlier stages.

Reading tiers are interpretation only and do not move the gate:

- delta <= -0.030 m: the stated mechanism is materially realized; proceed
  directly to the VO-precision stage on this checkpoint.
- -0.030 m < delta <= -0.010 m: KEEP. The recipe is carried forward, but the
  partial realization is reported as such and is not described as the full
  effect the reference curve sizes.
- delta > -0.010 m: reject this bounded recipe. No weight, LR, or arm search
  follows, and the threshold is not relaxed.

The reference curve sizes an idealized -0.069 m (image ego-state reaching
sigma 0.3 AND the planner fully exploiting it). It is a diagnostic expectation,
not a gate, and a smaller realized effect does not retroactively change any
threshold above.
