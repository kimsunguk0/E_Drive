# P7-C matched state/history+goal sufficiency diagnostic

Status: implementation candidate for CPU review. This document authorizes no GPU
forward, fit, final-validation access, model change, or deployment claim.

## Question and fixed pair

On the current grouped train203/tune37 split, how much future-path information is
available in the exact compact state/history semantics when those values are
pose-derived ground truth versus the frozen P7-C image model's own predictions?

For base seeds 0 and 1, train two separate but matched tiny predictors:

- GT: physical `vx,vy,ax,ay,yaw_rate`, binary stop `0/1`, and four
  `(dx,dy,sin(yaw),cos(yaw))` history rows at `0.1,0.2,0.5,1.0 s`.
- Pred: the same first five physical fields, `sigmoid(raw state_hat[5])`, and
  `history_hat[4,4]` from the frozen P7-C model.
- Both receive the same raw provided 5-second `goal_xy`. There is no no-goal arm.

This is 24 values. GT binary stop is never logit-transformed, and the predicted
raw stop logit is never compared to it as though it were a continuous physical
quantity. One canonical GT24 train54810 matrix defines the shared normalizer,
including goal. Divide first by the fixed scale
`[10,5,3,3,.5,1] + 4*[10,5,1,1] + [80,64]`, then compute float64 population
mean/std on train only with a `1e-6` standard-deviation floor. Apply those same
frozen arrays to GT/Pred and both seeds. Tune never affects normalization.

## Inputs and extraction

Use only current train54810 (stride1) and tune1998 (stride5) rows from grouped
split SHA `f4e0f30c...`, C1 supervision `ba1ba04e...`, calibration `8bd130de...`,
and ego cache `d35a69bb...`. Final136 is unavailable to the CLI.

No P7-C per-row train state/history cache exists. Its terminal tune records store
state6 but omit history16. P4 caches are not substitutes. A later separately
authorized extraction must therefore run each immutable P7-C base through the
full, unmodified model forward with `augment=False`, nominal C times, BF16 encoder
and FP32 state/history/planner heads. Train batch is fixed at 8 and tune batch at
4. Tune row/GT/plan/state must match the immutable P7-C terminal report bitwise;
history is taken from that same forward. The first batch is repeated, model state
is hashed before/after, and no optimizer/backward is created. A motion-only
shortcut is forbidden because none has been established as exact.

The historical P7 source manifest remains pinned for training lineage. Extraction
also pins the reviewed default-C runtime manifest and verifies the exact subset
of model, backbone, data, training, and temporal-contract files it imports. The
checkpoint is assembled directly with its external model config and loaded
strictly; no unpinned audit/construction helper participates.

## Fixed tiny-predictor recipe

For each seed, GT and Pred use identical initialization and sample permutations:

- `24 -> 512 -> 512 -> 12`, GELU and LayerNorm after each hidden layer;
- AdamW, LR `1e-3`, weight decay `1e-4`, batch `1024`;
- fixed 60 epochs, exactly `60*ceil(54810/1024)=3240` updates;
- per-update cosine schedule with `T_max=3240`;
- SmoothL1 beta `0.1` on the six physical cumulative XY points;
- LAST60 only, then exactly one tune1998 evaluation with official FP32 D3.

Report final in-sample train D3 explicitly as TRAIN, tune per-horizon ADE and D3,
and Pred-minus-GT within seed. No epoch, seed, threshold, architecture, or
hyperparameter selection is allowed. This reuses the e26-family architecture and
recipe on a different grouped split; it is not a reproduction of old `.0901`.
The D3 and all three horizon summaries are computed from the same single terminal
tune forward pass (two batches at batch size 1024), not by a second evaluation.

## Interpretation boundary

The GT arm is a privileged, prohibited diagnostic and is not a mathematical
ceiling. P7-C predictions on its training rows are in-sample model outputs, so
train fit can be optimistic. The raw-goal tiny MLP also differs from the legal
P7-C shared-scene goal route: even a strong Pred result establishes compact
information sufficiency only, not decoder isolation, image-use compliance,
deployability, or an architecture adoption case. No dense scene/motion tokens
are tested here, and no final/generalization claim is permitted.
