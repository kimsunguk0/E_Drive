# P7-C GT21 same-forward diagnostic protocol

Status: implementation-only candidate. No GPU execution or result exists.

This is a privileged diagnostic, never a candidate output or submission. For
each of the canonical grouped-split tune 1,998 rows and each trained P7-C seed,
the frozen model computes the six current-camera and five front-temporal image
encodings exactly once. The resulting scene and motion tensors are shared by
two calls to the unchanged FP32 planner:

1. baseline: the model's own physical `state_hat[0:5]`, raw stop logit
   `state_hat[5]`, and physical `history_hat[4,4]`;
2. GT21: replace only the five continuous state fields and sixteen history
   fields with exact valid C1 pose/timestamp-derived targets. Preserve the
   model's own raw stop logit bitwise.

Both calls therefore retain identical continuous image scene/motion features,
the existing per-cell goal-conditioned scene path, model weights and planner.
No raw goal is passed to the planner. There is no optimizer, backward, weight
change, threshold, row selection, final-136 access, or extra model forward.
Batch is 4; tune order, identities and baseline tensors must match the frozen
P7-C terminal report and the separately frozen GT/prediction artifacts exactly.

The output stores row/scenario/session/frame, GT plan, baseline and GT21 plan,
per-row official FP32 D3, and the preserved raw stop logit. These records allow
only paired 11-session descriptive analysis. Each seed report records every
session's row count and baseline/GT21/paired-difference sums and means. A joint
two-seed shared-session bootstrap is deferred until both immutable outputs
exist; it is not computed or selected inside either extraction run.

Interpretation is preregistered: improvement supports that the current planner
can use more precise continuous status. Flat or worse is inconclusive because
the frozen planner was jointly learned on its estimator's input distribution.
Neither outcome proves status-token dilution, an image-information bound, or
deployable performance; GT21 is unavailable at inference.
