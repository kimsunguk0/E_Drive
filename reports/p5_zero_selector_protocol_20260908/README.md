# P5-Z zero-vs-move selector preregistration — 2026-09-08

## Status

This document preregisters a bounded offline diagnostic.  It does not authorize
GPU execution, deployment, a submission change, or a compliance/performance
claim.  Cache and head implementation SHA-256 values remain `PENDING` until the
implementation and its tests finish independent review.  Any material change
to the feature contract, target, optimizer, schedule, split use, or report
contract requires a new protocol and experiment name.

P5-Z asks only whether a small offline readout of an otherwise frozen P4 model
can distinguish an exact-zero path from that same P4 forward's original path.
It is not a new end-to-end model and it is not a license to force current-stop
rows to zero.

## Immutable bases and provenance

The two bases are fixed before any P5-Z cache or head training:

| base | P4 LAST6000 SHA-256 | model-state SHA-256 | completed sidecar SHA-256 |
|---|---|---|---|
| 0 | `3e9ae5aa6ca89f4eb9a72ba38355dfdbec4ccaa38eabbebb3526fa3019404478` | `4f5ea00863d0a99c0aba4931df3865a5f6cab4160347e1f96abe97c1ca96995f` | `001b822dde79024f0e2cd52b5b99cb1096bbab5bfd09ef2d95e963ad8dc6e3de` |
| 1 | `c937f2f772c78f99cf3cfa5b1a96614267e20a9a667a157f4ada284940f9672e` | `02d3e53b5cbc89ac0a2f6a73dd6d3e736f6fe0af30fef20499429448de702fa0` | `49dc09c8e7840a059ae6bcadba1c490052847c678dc81d5524ee39f9da159aad` |

Training-source provenance is
`86620b4ffc7e6838b49cf83b5be789eba12d8027`.  Independent normal evaluation
used validation-source commit
`095e49d4e7be36802bad2f5923b873e65115e7fc`; the repository at protocol
creation was `53e702c9aeb553ef16cd9369023078e7e46dfe40`.  These are separate provenance
fields.  A whole-HEAD equality check must not replace the source contract.

The canonical runtime contract is the exact 22-key `source.file_sha256` mapping
stored in both original child receipts:
`logs/motiondrive_v2/p4_fresh_joint_s0.child.json` and
`logs/motiondrive_v2/p4_fresh_joint_s1.child.json`.  The two mappings are JSON
identical and attribute those bytes to training commit `86620b4...`.  P5-Z must
verify exactly that keyset and those hashes before and after caching.  New P5-Z
helpers are separately pinned; they are not falsely attributed to the old
training commit.

The fixed data contract is:

- grouped split SHA-256
  `f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936`;
- C1 supervision manifest SHA-256
  `ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93`;
- canonical calibration SHA-256
  `8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961`;
- train: 54,810 rows, 203 scenes, 72 sessions, stride 1, row SHA-256
  `75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854`;
- tune: 1,998 rows, 37 scenes, 11 sessions, stride 5, row SHA-256
  `1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88`;
- final validation and test: zero access.

## Cache and feature contract

Each base is fully frozen in eval mode.  Cache inference uses the original
normal P4 input whitelist, nominal time, fixed BN state, a BF16 image encoder
and FP32 planner.  It performs exactly one hooked full forward per cached row;
the hook is removed in `finally`.  The first batch additionally receives one
unhooked parity forward.  Model state, BN state, inputs, checkpoint, sidecar,
data and source hashes must match before and after.

The selector input is exactly 790 FP32 values from the same full forward:

1. `planner.xy_head`'s immediate pre-hook input after `decoded.float()`, shape
   `[6,128]`, flattened to 768 values;
2. neural `state_hat`, shape `[6]`;
3. neural `history_hat`, shape `[4,4]`, flattened to 16 values.

Raw goal, pose, time, row/scenario/session identity, GT, labels, candidate
costs, and any post-hoc bucket indicator are forbidden selector inputs.  The
decoded hidden state may contain the ordinary goal context already used by the
frozen P4 planner; no new raw goal input is added.

Train cache batch is 8 and tune cache batch is 4.  The tune cache must reproduce
the immutable normal evaluator's ordered identity, GT, P4 path and stored D3
row by row.  Cache manifests report checkpoint/sidecar/source/data/cache hashes,
row hashes, exact feature shapes/dtypes, hook count/removal, first-batch parity,
same-forward equality, model/BN/input immutability, and
`final_val_accessed: false`; the separate parent execution receipt records the
actual child/parent process return codes and PID lifecycle.

Pilot mode must still validate the complete fixed split inventory and full row
SHA before selecting the ordered prefix.  Its manifest must state
`execution_scope: ordered_first8_pilot`, `pilot_samples: 8`,
`full_cache_completed: false`, and `data.cached_n: 8`.  Default
`--pilot-samples 0` means full extraction and is not authorized by this
protocol revision.

## Candidates, target, and decision

Candidate order is fixed: class 0 is an exact `[6,2]` zero path and class 1 is
the unmodified P4 path from the same forward.  Costs use the existing official
weighted D3 implementation with weights `[11,11,5,5,2,2]/36`.

The train-only target is ZERO exactly when `Dzero < Dmove`; equality is MOVE.
At inference, select ZERO exactly when `logit_zero > logit_move`; equality is
MOVE.  No probability threshold is fitted or tuned.  The final two-logit Linear
layer starts with all weights zero and bias `[-2,+2]`, so initial hard decisions
are MOVE and initial hard D3 must equal the frozen base exactly.

## Frozen head recipe

For each base, train head seeds 0, 1, and 2 independently, producing all six
LAST1000 artifacts.  The fixed objective is

`cross_entropy(label) + 0.25 * mean(sum(softmax(logits) * [Dzero,Dmove])) / C_scale`,

where `C_scale = max(mean train Dzero, 1e-3)` is computed from train only.
Optimizer is AdamW, learning rate `1e-3`, weight decay `0.01`, logical batch
128, 1,000 updates, 100-update linear warmup then cosine decay, and pre-clip
gradient norm limit 5.  Only the selector head trains; cached features and both
candidate paths remain fixed.

Every LAST1000 is primary.  There is no BEST selection, early stopping,
extension, bonus epoch, head-seed selection, base selection, or tune
hyperparameter/threshold tuning.  Tune evaluation happens once for each of the
six LAST artifacts only after its training completes.  Final implementation,
test and source-manifest SHA-256 values will replace these placeholders before
any GPU cache run:

- cache helper SHA-256:
  `42749aebe6ae1d56a1d55348193c9bb6d92357b5870122dfc8ad303c9bd921d1`;
- cache tests SHA-256:
  `0fc686be9bb50b3d9b86c3ed15fe12071d70a900f4b1db36b564af01265b9128`;
- head trainer SHA-256: `PENDING_REVIEW`;
- head tests SHA-256: `PENDING_REVIEW`;
- exact runtime-22 source manifest
  `reports/p5_zero_selector_protocol_20260908/runtime22_source_manifest.json`
  SHA-256:
  `98478a88022981396f7d3a6ffb2f7f2e815797ca60b482a1d7c645c83aac4ae2`.

## Preregistered reporting

Report all six LAST1000 results, never only the better head seed.  For each run
report actual OS/supervisor return codes, nonfinite count, optimizer steps,
initial and LAST head hashes, loss trace, actual learning rate, gradient norms,
input/cache/source hashes, and proof that the base model/cache did not change.

For each run and then as base-wise three-seed mean/range, report:

- original-base D3, selector D3 and delta;
- ZERO decision count/rate;
- false/misselection cost relative to the per-row two-candidate oracle;
- current-stop 123, future-steady 99, future-departing 24, and non-stop 1,875
  breakdowns;
- all 11 tune sessions and the aggregate with session 046 excluded;
- paired session-cluster bootstrap intervals using the 11 sessions, explicitly
  noting that 11 clusters give broad uncertainty.

The stop/steady/depart/session breakdowns are post-training diagnostics only.
They cannot fit a threshold, choose a head, or gate a checkpoint.  Existing
CPU oracle2 values, `0.3001864` for base 0 and `0.2998887` for base 1, use GT to
choose per row and are unattainable lower-bound references, not targets or pass
gates.  Their maximum recoverable gain is concentrated in session 046
(approximately 64.5%/66.2% for the two bases), so session-level and
session-046-excluded reporting is mandatory.

P5-Z remains an offline diagnostic.  It does not establish production routing,
zero/shuffle robustness, latency parity, official submission improvement,
regulatory compliance, or leaderboard performance.

## Proposed cache pilot procedure — not yet authorized

After the cache blocker fixes, exact helper/test review and a separate root GPU
approval, the first proposed pilot is **base-0 train8 plus tune8 on physical
GPU 4, and base-1 train8 plus tune8 on physical GPU 5 only**.  Full
train54,810/tune1,998 extraction remains `PENDING` and requires a separate
approval.  The reviewed helper must provide explicit `--pilot-samples 8`
deterministic-prefix mode, preregister the selected row IDs and row SHA before
any forward, and refuse pilot output paths for any other count.  The tune8 rows
must match the corresponding immutable normal evaluator identities, GT, P4
paths and D3 exactly.  The cache-main argv passed by the guarded child is
expected to have these forms, with reviewed hashes and the final pilot flag
substituted:

```text
/usr/bin/python scripts/cache_motiondrive_v2_zero_selector_features.py
  --checkpoint work_dirs/motiondrive_v2/p4_fresh_joint_s0/last.pth
  --expected-checkpoint-sha256 3e9ae5aa6ca89f4eb9a72ba38355dfdbec4ccaa38eabbebb3526fa3019404478
  --run-manifest work_dirs/motiondrive_v2/p4_fresh_joint_s0/manifest.json
  --expected-run-manifest-sha256 001b822dde79024f0e2cd52b5b99cb1096bbab5bfd09ef2d95e963ad8dc6e3de
  --base-seed 0 --data-root /NHNHOME/data/sukim/adcl
  --split-manifest data/etri/motiondrive_v2/grouped_split_rawtime.json
  --supervision-root data/etri/motiondrive_v2/train_tune_geometry_v2
  --split train --source-manifest reports/p5_zero_selector_protocol_20260908/runtime22_source_manifest.json
  --expected-source-manifest-sha256 98478a88022981396f7d3a6ffb2f7f2e815797ca60b482a1d7c645c83aac4ae2
  --out work_dirs/motiondrive_v2/p5_zero_selector_pilot_b0_train8.pt
  --workers 4 --batch 8 --pilot-samples 8
  --device cuda:0
```

```text
/usr/bin/python scripts/cache_motiondrive_v2_zero_selector_features.py
  --checkpoint work_dirs/motiondrive_v2/p4_fresh_joint_s0/last.pth
  --expected-checkpoint-sha256 3e9ae5aa6ca89f4eb9a72ba38355dfdbec4ccaa38eabbebb3526fa3019404478
  --run-manifest work_dirs/motiondrive_v2/p4_fresh_joint_s0/manifest.json
  --expected-run-manifest-sha256 001b822dde79024f0e2cd52b5b99cb1096bbab5bfd09ef2d95e963ad8dc6e3de
  --base-seed 0 --data-root /NHNHOME/data/sukim/adcl
  --split-manifest data/etri/motiondrive_v2/grouped_split_rawtime.json
  --supervision-root data/etri/motiondrive_v2/train_tune_geometry_v2
  --split tune --source-manifest reports/p5_zero_selector_protocol_20260908/runtime22_source_manifest.json
  --expected-source-manifest-sha256 98478a88022981396f7d3a6ffb2f7f2e815797ca60b482a1d7c645c83aac4ae2
  --out work_dirs/motiondrive_v2/p5_zero_selector_pilot_b0_tune8.pt
  --workers 4 --batch 4 --pilot-samples 8
  --tune-report reports/p4_joint_full_tune_eval_s0_20260908_ops/normal_full_tune.json
  --expected-tune-report-sha256 cffa9cc8b12f5f9812af7b3fd3631e74597e291b02aecd1cc14d8ec8a1aeba2f
  --device cuda:0
```

The two base-1 commands are identical except that they use physical GPU 5,
`--base-seed 1`, checkpoint
`work_dirs/motiondrive_v2/p4_fresh_joint_s1/last.pth` with SHA-256
`c937f2f772c78f99cf3cfa5b1a96614267e20a9a667a157f4ada284940f9672e`,
sidecar `work_dirs/motiondrive_v2/p4_fresh_joint_s1/manifest.json` with SHA-256
`49dc09c8e7840a059ae6bcadba1c490052847c678dc81d5524ee39f9da159aad`,
outputs `p5_zero_selector_pilot_b1_train8.pt` and
`p5_zero_selector_pilot_b1_tune8.pt`, and tune report
`reports/p4_joint_full_tune_eval_s1_20260908_ops/normal_full_tune.json` with
SHA-256
`0e71bef199931cb1916abdc819ca78cf75d8cd7055439bd084ca975bb610c137`.

No new launcher framework is proposed.  A report-local one-shot parent will
compose the already reviewed primitives: physical-GPU UUID/free-memory query,
single-UUID `CUDA_VISIBLE_DEVICES` mapping to logical `cuda:0`, allocator
configuration before model allocation (`12,000 MiB` cap and `8,192 MiB`
reserve), and a five-second pressure check.  It owns one exact `Popen` child;
only that child may receive TERM and then KILL after the existing grace period.
Foreign PIDs and process groups are never signalled.  GPU 0–3 remain untouched
and GPU 6/7 remain forbidden.

Before launch, require the reviewed helper/source hashes, base/data hashes,
absent new output/manifest/record paths, the expected GPU-4/GPU-5 UUIDs, no
compute PID on either assigned GPU, and at least `12,000+8,192 MiB` free on
each.  Each report-local parent owns at most one exact child at a time; train8
must reach actual exit and PID absence before that base's tune8 child starts.
The child sets the allocator cap
before allocating the model and then calls the cache main with the argv above.
Record the unified SSH handle, parent
and child PID/PPID, physical UUID/logical mapping, start time, periodic memory,
pressure event, native child return code, parent return code, stdout, cache and
manifest SHA-256, strict post-run cache validation, actual PID absence and final
GPU memory.  A timeout or observer disconnect is not a failure/restart signal;
native nonzero is preserved and never retried automatically.

This proposed command is preparation only.  Pilot8 itself, full-cache
extraction, deployment, Git commit, and head training all remain forbidden at
this revision.
