# MotionDrive V2 long-training protocol (2026-09-09)

Protocol frozen before any arm was launched. Four arms on idle physical GPUs
2 / 5 / 6 / 7. No architecture change. This measures a training budget, not an
intervention.

## Question

Every screened intervention since P1 has failed or been marginal, while
continued training has moved the score every time: the most recent 4,000-update
matched continuation alone gave 0.293932 -> 0.280640 (-0.013292) with no change
at all. Cumulative training on this lineage is only about four to five epochs
(1 epoch = 3,425 updates at batch 16 over 54,810 rows). No long run has ever
been performed.

This asks two things: how far does the current best configuration go when the
training budget stops being the binding constraint, and where does it turn over.

## Why a clean holdout is required

tune1998 (11 sessions) has been reused across P1-P8 and every screen since.
Locating a turnover point on it would compound that reuse. Two arms therefore
train on a reduced train set and validate on sessions never trained on.

Session-level holdout, pinned in
`reports/motiondrive_v2_longrun_holdout_split_20260909_ops.json`
(SHA256 `3f9cbb84839b783b9055a38b80bf6ae46f707629ec3a1890755efb3f08842dfd`):
every 6th of the 72 train sessions by sorted name, first 12 taken.
12 sessions / 32 scenes held out (15.8% of train scenes); 60 sessions /
171 scenes remain for training. Whole sessions are held out, so no scene from a
validation session appears in training.

## Parent

`work_dirs/motiondrive_v2/controlflow_b0_direct_last4000/last.pth`
- checkpoint SHA256 `7e1b3f3a52e2703210edf24f546d0cb62b91723952076482866d89204bbb9449`
- sidecar SHA256 `94fae557ff2ffadbd4dc7079cd228245bba9172c4d249a5498d2ca7fa1f31653`
- reported tune D3 0.280640, step 4000, status completed

This is the `direct` arm of the control-flow screen: the unchanged production
planner. The A2 shared-status query route and every existing scene, motion,
geometry, goal, occupancy, lane, state, history and stop route are unchanged and
remain trainable exactly as in the parent.

## Arms

| arm | GPU | physical UUID | train | periodic eval |
|---|---|---|---|---|
| `long_s0` | 2 | `GPU-4f5a3e60-3c76-6719-33ed-8c8b8bd41565` | train203, 72 sessions | tune1998 |
| `long_s1` | 5 | `GPU-1e9aea73-4e6b-2cb5-b788-f3d99e6dc6f8` | train203, seed 1 | tune1998 |
| `holdout_s0` | 6 | `GPU-cebed831-81f7-e364-93f3-59f6fe131e75` | 60 sessions | the 12 held-out sessions |
| `holdout_s1` | 7 | `GPU-88f41d8a-19f3-9d95-ea0c-cc5b1a42c9c8` | 60 sessions, seed 1 | the 12 held-out sessions |

## Fixed recipe

20 epochs = 68,500 updates. Fresh AdamW, backbone LR 5e-6, head LR 5e-5,
weight decay 0.01, warmup 100 then cosine decay through exactly update 68,500,
gradient clip 5, batch 16 / microbatch 2, fixed BatchNorm running statistics,
bf16 trunk with FP32 planning and coordinate head, auxiliary weights 0.2,
all parent parameters trainable. Evaluate and checkpoint every 3,425 updates
(one epoch). Identical recipe across all four arms; only seed and the scene
lists differ.

Approximately 17.6 hours per arm at the measured 1.08 step/s.

## Pre-registration

- **No checkpoint is selected on tune1998.** The tune curve is recorded for
  observation only. The reported number for the `long_*` arms is the terminal
  step-68,500 value.
- If a best checkpoint is nominated at all, it is nominated on the held-out
  12 sessions by the `holdout_*` arms, never on tune.
- The `holdout_*` arms train on a smaller set and are therefore **not**
  comparable to the 0.280640 lineage. They exist to locate the turnover and to
  measure the train-to-unseen-session gap, nothing else.
- final136 rows are not selected, trained, evaluated or analyzed.
- The parent has already been through several cosine cycles, so "20 epochs" is
  additional budget; cumulative is roughly 25 epochs. This is not a clean
  from-scratch epoch curve and will not be described as one.

## What counts as a result

Three deliverables, all reported whatever they show:

1. The tune1998 curve for `long_s0/s1`: does D3 keep falling past 0.280640,
   and where does it flatten.
2. The held-out-session curve for `holdout_s0/s1`: where does an uncontaminated
   validation turn over. If it turns over well before the tune curve flattens,
   the tune curve is not trustworthy for budgeting and that is the finding.
3. The seed spread at matched steps, which bounds how much of any movement is
   seed noise rather than budget.

No threshold is set for "success". A flat curve is as informative as a falling
one: it would end the training-budget line and force the remaining effort onto
the acceleration-profile half, where the diagnostics of 2026-09-09 place a floor
of 0.1434 against a path ceiling of 0.0765.

## Verification and operations

Before launch: source manifest pinned and verified, complete key-load check,
parent hash check, real TRAIN smoke, GPU memory headroom with a live recheck,
and confirmation that physical GPUs 2/5/6/7 are idle. Foreign work on other GPUs
is left untouched. Durable process ownership, real exit codes recorded, no
automatic retries. Preserve dirty files; explicit-path commits; no push.
