# P9 frozen-body late-state residual protocol

Status: local implementation candidate only. No cache extraction, fitting, GPU
execution, or result exists.

P9 freezes each trained P7-C seed's complete image model. A narrow extractor
computes the current six-camera plus five front-temporal encodings once and
captures the existing FP32 planner decoder output immediately before
`xy_head`: `[B,6,128]`. Applying the unchanged `xy_head` and fixed output scale
must reconstruct the immutable P7-C plan exactly. Saved `state_hat[6]`,
`history_hat[4,4]`, row/scenario/session/frame and baseline plan must also match
the already verified P7-C prediction/GT artifacts. Only grouped train 54,810
and tune 1,998 are accessible; final-136 is absent.

The trainable module is identical in all arms: LayerNorm of the frozen
image-conditioned 128-vector, concatenated with compact22 after the existing
physical scales and fixed waypoint time `t/3`, followed by a shared
`151->256->128->2` GELU MLP. Its final Linear weight and bias start exactly
zero, so the initial corrected plan equals the frozen baseline. Corrected plan
is `base_plan + residual` in physical metres. There is no raw goal input,
planner-query change, feature-extractor gradient, optimizer state resume, or
coordinate post-processing.

For each base seed, three same-sized arms use identical module initialization
and sample order:

- A `no_compact`: exactly zero 22-vector;
- B `predicted22`: physical predicted state6, including raw stop logit, plus
  predicted history16;
- C `gt21`: GT physical continuous state5 and GT history16, while preserving
  the same predicted raw stop logit. This arm is privileged and non-deployable.

Each arm uses batch 1024, 60 epochs = 3,240 updates, AdamW lr `1e-3`, weight
decay `1e-4`, per-step cosine schedule, gradient clip 5, official D3 training
loss and LAST60 only. Tune is run exactly once after training. Every tune row's
baseline/corrected plan and FP32 official D3 is saved; there is no intermediate
evaluation, BEST, threshold, grid, or arm/seed selection.

The frozen decoder feature is not image-only: it already contains the parent
planner's predicted-status token and goal-conditioned shared scene. “No raw
goal” means only that the new residual receives no additional/direct goal.

`B-A` isolates added compact predicted-state access beyond equal residual
capacity. `C-B` is the direct privileged precision diagnostic. A gain does not establish generalization or compliance;
flat/harm may reflect the frozen decoder representation or train/tune behavior.
The experiment does not prove token dilution, estimator information bounds, or
that GT inputs are deployable.

Preregistered primary gate for B−A: both seeds negative, two-seed mean at most
`-0.010`, and the shared-11-session 10,000-resample CI upper bound below zero.
Exploratory secondary A−parent and B−parent gates each require both seeds
negative, mean at most `-0.015`, and CI upper bound below zero. B adoption also
requires its B−parent gate, not merely B−A. C has no performance gate. The
already reused tune split makes every result exploratory; no arm can be chosen
or retuned from it. Full raw-image A/B adapter, full-graph latency and compliance
checks remain separate adoption gates; the adapter rejects privileged C.
The legal raw-image adapter keeps the complete parent in evaluation mode with
all parent parameters frozen, preserves all eleven parent image encodings, and
adds the metre residual exactly once after the existing plan scale. Cache and
fit receipts bind the executed extractor helper, residual module, bootstrap
helper, and P9 runner bytes separately; they do not relabel inherited P7 source.
