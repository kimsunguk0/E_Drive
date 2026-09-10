# SparseDriveV2 ETRI screen — 2026-09-10

User authorization: physical GPUs **0, 1, 4** for this work. Other processes and
devices remain outside the experiment. The code lives in the isolated
`codex/sparsedrivev2-20260910` worktree; the original main worktree is preserved.

## Question and fixed evaluation

Can a pretrained factorized path/velocity scorer improve ETRI planning D3?
The target near 0.15 is a research objective, not an assumed result.

- Keep original train203 (72 sessions, 54,810 rows) and tune37 (11 sessions,
  1,998 rows). CPU raw-timestamp audit found no cross-split input/target time
  overlap. Existing reserve136 is not evaluated during this screen.
- Metric is the frame mean of six Euclidean waypoint errors weighted
  `[11,11,5,5,2,2]/36`; no test-derived sample weights.
- Existing terminal `longrun_long_s1/last.pth` is the fixed direct reference.
  Re-evaluation verifies the new worktree/data path. Its weights are not a
  public initializer and cannot initialize any fold that excludes its training
  sessions.
- The candidate bank uses only allowed training scenes and labels. The full
  bank oracle, oracle after learned pruning, and actual selected trajectory D3
  are separate quantities. Validation GT is only used for metrics.
- A confirmation split (12 held-out sessions inside train203) is also frozen.
  Its models, banks, normalizers and any teachers must exclude those sessions
  from their first ETRI update. Old subset-continuation checkpoints do not meet
  this condition.

## First bounded experiments

1. Verify existing terminal direct reference on unchanged tune1998 (GPU 0).
2. Build a small factorized bank, then scale coverage only as oracle results
   warrant (GPU 4 / CPU). Use native train future paths, with validity masks;
   do not extend short paths by copying endpoints as valid geometry.
3. Adapt official NAVSIMv1 SparseDriveV2 checkpoint to ETRI cameras/calibration
   and D3. Preserve reusable public tensor weights and report every missing or
   replaced tensor. Keep candidate coordinates fixed during inference.
4. Run matched short training pilots using zero status and causal-status
   selection (GPUs 0 and 1 after the reference evaluation). Command slots are
   zero in both first pilots; no future goal is read. Both use identical bank,
   seed, row population, input resolution and optimizer budget. These pilots
   test optimization and input dependence; they are not sufficient to rank
   architecture against a much longer-trained direct reference.

Provided status, when enabled, is the existing hash-verified causal nominal
pose fit (vx, vy, ax, ay). It may score/prune/index the precomputed complete
trajectory bank, but may not numerically change its coordinates. This is the
selection-only design motivated by OPEN_ISSUE Q8/Q9. The implementation and
image contribution must be inspected; no blanket official approval is claimed.

## Evidence and next decisions

Record source/checkpoint/data/bank hashes, device, seed, actual command and
process status for every run. Require finite forward/backward and valid
complete-bank row selection before scaling training. Preserve failed attempts.

Report terminal and best pilot scores separately. Candidate coverage can be
assessed before full training, but low oracle is not achievable-model evidence.
If full-bank oracle is too high, revisit vocabulary allocation before training
a large scorer. If pruning destroys coverage, improve coarse supervision or
retain more candidates. If shortlist is strong but actual D3 is poor, focus on
image-conditioned ranking and its metric target.

After a successful screen, repeat the selected recipe from public-only
initialization on the frozen confirmation split. Measure complete inference
cost early and on final deployment hardware before making submission claims.
