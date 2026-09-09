# MotionDrive V2: fixed joint path × velocity head screen

Protocol frozen before the new bank's tune oracle or any head-training result.
Date: 2026-09-09 KST. Scope: one CPU coverage gate, then at most three paired
2,000-update runs if the gate passes. No Goal, final136 access, or automatic
follow-on search. This is an internal tune experiment, not a leaderboard score.

## Question

Can a learned joint path/timing representation improve the existing direct
regressor, and does a small neural per-candidate residual recover quantization
error? This tests a new-head training recipe, not a single-frame information
limit or the theoretical superiority of classification over regression.

## Data and bank gate

- Retain grouped train203 (54,810 rows, stride 1, frame >= 30) and tune37
  (1,998 rows, 11 sessions, stride 5, frame >= 30).
- Split SHA256: `f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936`.
- Train row SHA256: `75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854`.
- Tune row SHA256: `1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88`.
- Fit only train203 labels. Physical reading of a shared NPZ is not permission
  to select final136 rows. Do not load or reuse any old train330/train300 bank.
- One fixed factorization, seed 20260909: 512 normalized arc-length geometries
  and 127 nonzero time-progress profiles plus a dedicated exact-stop profile.
  Compose one zero trajectory plus 512 * 127 combinations = 65,025 candidates.
- Reuse the corrected physical-progress clustering recipe: globally standardized
  cumulative physical distance, official prefix weights and fixed tail weights;
  do not independently z-score the normalized progress dimensions per stratum.
  Axis prototypes are train medoids; composed trajectories are synthetic pairs,
  not intact observed training trajectories. Preserve axis source IDs/support.
- Save and freeze full 10-waypoint/5-second bank bytes and provenance before
  opening tune labels. Exact finite row-zero stop and deterministic indexing are
  mandatory. Report duplicate candidate counts; do not silently change IDs.
- Gate: full-bank tune official-temporal-weighted D3 oracle <= 0.11486043,
  the current clean grouped K1024 reference. Otherwise stop; no P/V sweep or
  tune-driven refit. Endpoint/tail oracles are diagnostics, not selection rules.
- The 5-second tail defines the bank, but this screen's head loss and emitted
  plan concern the first six points. It does not validate learned tail prediction.

## Common model and three arms

Use the exact completed seed-0 early-precision continuation-control LAST2000
checkpoint (reported tune D3 0.2987348186). Pin checkpoint, model-state and
sidecar hashes before launching; never initialize from a smoke checkpoint.

1. `direct`: ordinary continuation of the existing direct regression head.
2. `pv`: exhaustive learned joint scoring of all 65,025 fixed candidate rows.
3. `pv_residual`: the identical scorer plus bounded neural per-candidate offsets.

Retain the learned scene/motion encoders, planner decoder, shared goal route,
and A2 provided-status shared-query route. The new heads receive the existing
continuous planner latent, with no new raw goal, pose, or status arguments.
This whole model is not goal-free: goal/status already condition shared image
features. Do not describe this as purely image-only generation.

Joint scoring uses compact path/profile descriptors and learned interacting
keys, not attention from every candidate to the full BEV memory. Use FP32
bounded cosine logits and a capped learnable scale. No scalar-speed pruning,
hierarchy, goal pruning, or nearest-regression shortlist. Select with stable
first-index `argmax` over the entire same bank used for the oracle.

The residual is a compact image-conditioned per-candidate neural head, with
zero-initialized final projection and a fixed `0.5 * tanh(raw)` metre bound per
coordinate. It completes candidates inside model forward before argmax. There
is no external goal-dependent coordinate correction. Save selected base and
completed trajectories separately. The residual arm equals pure selection at
step zero; both selector arms have identical scorer initialization.

## Optimization and interpretation

- All upstream parameters remain trainable; fixed BN statistics. One paired
  optimization seed (0); same data order, augmentation, fresh optimizer and
  2,000 updates. Batch 16 / microbatch 2, bf16 backbone and FP32 planning/loss.
- Reuse the fixed continuation recipe: backbone LR 5e-6, head LR 5e-5,
  warmup 100, weight decay 0.01, gradient clip 5; occ/lane/motion weights 0.2.
- Direct arm: existing official D3 loss.
- Candidate arms: expected candidate D3 under the predicted full joint softmax
  plus 0.1 soft cross-entropy against `softmax(-base_bank_D3 / 0.1)`.
  Residual targets remain based on the fixed bank, not moving corrected labels.
  Compute distances/targets online; no expensive unused train-oracle cache.
- Retain complete-six-point masking and full-logical-batch normalizers across
  microbatches. Do not supervise on tune or select a best checkpoint from tune.
- Evaluate once at terminal step 2,000. Training checkpoints are recovery and
  diagnostic artifacts, not a license for further tune selection.
- The direct head is mature while selectors are new, and parameter counts and
  loss functions differ. Equal updates do not remove this confound. Failure
  only rejects this bounded recipe; success still needs independent replication.

## Verification and reporting

Before launch: source/parent/bank pins, complete key-load checks, direct parent
output equality, selector/residual step-zero equality, exact row identity,
finite gradients, actual TRAIN smoke, paired row-order check, and GPU memory
headroom. Use idle physical GPUs 0 / 1 / 5, one arm each, after live recheck;
leave foreign GPU4 work untouched. Preserve dirty files; explicit-path commits,
no push. Record real exit codes with durable process ownership and no retries.

Primary report: terminal tune D3 and ADE1. Also save per-row candidate IDs,
base/corrected plans, full-bank oracle gap, and 11-session paired uncertainty.
Report trainability/entropy/selected-mode diversity to distinguish mode collapse
from coverage failure. Minimum promising screen: D3 and ADE1 both improve by
>= 0.005 m against the contemporaneous direct arm; this is not final adoption.
Residual-vs-selection deltas must be shown separately, even if only the hybrid
wins. Keep all three terminal results, including failed/negative outcomes.
