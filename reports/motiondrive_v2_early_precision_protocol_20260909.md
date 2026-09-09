# MotionDrive V2 early-precision screen (2026-09-09)

Status: fixed implementation protocol; no run is authorized by this document.

## Question and parent

This single-seed, three-arm screen asks whether 0--1 second precision benefits
from retaining ordered visual-motion evidence or from an explicit local interval
objective. Every arm starts weights-only from the exact completed A2 PROVIDED
seed-0 LAST2000 checkpoint, uses a fresh optimizer, and retains its learned shared
status-query route. The screen does not establish a seed distribution or an
architecture upper bound.

## Arms

1. `continuation_control`: unchanged A2 PROVIDED graph and unchanged loss.
2. `ordered_motion_residual`: at every one of the 192 motion-grid sites, concatenate
   the four already time-encoded 128D image-correlation tokens in fixed
   `[.1,.2,.5,1.0]` order. Apply `LayerNorm(512) -> Linear(512,64) -> GELU ->
   Linear(64,128)`. The final weight and bias are exact zero initially. Add the
   result to the weighted motion token before the existing `token_refine` and
   state head. No provided status, pose, goal, command, or label enters this
   branch. Parent outputs are exact at initialization.
3. `early_delta_aux`: unchanged A2 PROVIDED graph. Retain official D3 as the
   primary loss and add `0.2 * early_delta`. In FP32, `early_delta` is SmoothL1
   with beta `0.1 m`, averaged over the four XY components of
   `p(.5)-origin` and `p(1.0)-p(.5)` for complete six-point rows. Its denominator
   is the same full-effective-batch `plan_complete` count passed unchanged to
   every microbatch. Raw and weighted terms are logged.

There is no combined arm, capacity placebo, stop-head change, path-progress
decoder, new goal route, final136 access, or post-forward trajectory correction.

## Fixed continuation recipe

All arms use seed 0, the same train54,810/tune1,998 rows, nominal history times,
A2 PROVIDED status overlay, shuffle order and augmentation policy. Training is
2,000 updates with batch16/microbatch2, BF16 outer autocast, fixed BN running
statistics, fresh AdamW, backbone LR `5e-6`, other/new LR `5e-5`, weight decay
`.01`, warmup100 then cosine, and global clip5. Existing plan/occupancy/lane/motion
weights remain `1/.2/.2/.2`. Checkpoints at 500/1000/2000 are nonselective;
LAST2000 and one terminal tune evaluation are the only result.

The temporal branch is constructed without advancing the parent/control Torch
RNG stream. Exact parent loading, zero-init output parity, sample order, and
immutable input/source hashes are execution gates.

## Interpretation

Primary comparison is each treatment's terminal overall D3 against the
contemporaneous continuation control. Secondary descriptive outputs are ADE1,
individual .5/1.0-second errors, 1.5--3.0-second errors, and existing GT-defined
stop/depart buckets. A screening signal requires both at least `0.005 m` lower
overall D3 and at least `0.005 m` lower ADE1 than control. Early improvement with
worse overall D3 is mixed/inconclusive, not a winner.

The tune set is reused and is not an untouched holdout or leaderboard set. A
positive ordered-motion result supports this added ordered-mixer branch. Its
extra capacity and final bias mean the screen does not isolate ordering from
capacity or prove temporal collapse was the cause. A positive delta-loss result
supports changed gradient geometry; it does not prove a missing physical-state
input. Replication or a combined arm requires a separate decision and is not
automatically authorized.
