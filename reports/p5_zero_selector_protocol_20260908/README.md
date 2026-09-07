# P5-Z zero-vs-move selector preregistration — 2026-09-08

## Status

This document preregistered a bounded offline diagnostic.  It did not itself
authorize GPU execution, deployment, a submission change, or a
compliance/performance claim; later bounded diagnostic executions were
authorized separately and are recorded in **Actual progress** below.  Any material change
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
six LAST artifacts only after its training completes.  The frozen
implementation, test, and source-manifest SHA-256 values are:

- cache helper SHA-256:
  `7fdc279f9221d5bc5704e6354cb60bb2aa0aefe9bf9e9b0adf05d21a69e7124b`;
- cache tests SHA-256:
  `e9d9f58e4576f8b5c1577eed959017a31774026fe630ffdd5dccaeb0ba6d2360`;
- head trainer SHA-256:
  `03bff7af9a9a55dc630e9223d566b5774a5744d36a99bf573cf6374587b11f06`;
- head tests SHA-256:
  `f2183a65b3fb365c5c75ee4e8db56e2fc3ccdae23b84b742c8d5866fc391dd4d`;
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

## Actual progress — separately authorized diagnostic work

This section records later root-authorized executions without retroactively
changing the preregistration above.

- The CPU-only two-candidate oracle used the immutable tune1,998 reports and no
  new forward.  Base-0 D3 had a GT-selected lower bound of `0.3001864` from
  `0.3722216`; base 1 had `0.2998887` from `0.3669575`.  Session 046 accounted
  for about 64.5%/66.2% of the recoverable gain.  The report is
  `reports/p4_zero_move_oracle_train_stationary_20260908/oracle_train_distribution.json`
  (SHA-256 `65c5cd0295aa89c5a6ff5085d0ab961f95b5ecb36a23b09e5f3e6080f61b2d35`),
  with command receipt SHA-256
  `c91aa7665901f1225f970a76964742c5c14da0a96ac4a7a97f51cccf2230761d`.
  This is a GT oracle ceiling diagnostic, not an attainable gate.
- A CPU-only fixed 0.5 coordinate-mean ensemble, with no model forward, gave
  D3 `0.3661828813`: delta `-0.0060387358` versus base 0 and
  `-0.0007745944` versus base 1.  It has not been adopted.  Report SHA-256 is
  `d1047ec664d17c213b09815ae81f9df0758223b0db3d72ca466bbb259d3a1fb9`;
  receipt SHA-256 is
  `27ca35cab21d8f8a53e12b4f8275a5c88f57e405ed7dd4314f56ebfee69be3e9`.
- The first bounded cache pilot used commit
  `e1a4594bbd86f39d11db2a3ac7487f7abd4bd58d`.  Both ordered train8 children
  exited 0; both tune8 children exited 1 at the strict P4 plan/GT bitwise
  gate, before tune output creation.  The original child argv fixes the
  executable as `/NHNHOME/data/sukim/adcl/env/venv/bin/python`; the exact
  Torch `2.7.1+cu128` and CUDA `12.8` values come from a later CPU-only probe
  of that same venv, not from the failed child stdout.  The failure summary is
  `reports/p5_zero_selector_pilot_failure_20260908_ops.json` (SHA-256
  `f7724b18f668a7cd6fad377bc0df55c5ef1f03e575edd97be70d21f13db77ba6`)
  and the reference-runtime CPU probe SHA-256 is
  `59c0fec62e6aff0fa4e8ba0470abdef10cc09442c9d84bec1cec1633864719db`.
- Retry 1 used `/usr/bin/python`, Python 3.12.3, Torch
  `2.10.0a0+b4e4ee81d3.nv25.12`, and CUDA `13.1`, checked directly by both
  parent and child before evaluation.  Both train8 children again exited 0;
  both tune8 children exited 1 at row 14730.  Ground truth was bitwise exact.
  The P4 path differed in 8 elements for base 0 and 10 for base 1, with
  maximum absolute difference `1.9073486328125e-6`.  The preserved evidence
  directories are `reports/p5_zero_selector_pilot_retry1_b{0,1}_20260908/`;
  their execution SHA-256 values are
  `2d6920b946de2b6867b149bb13d8026f89290e2ba707d33bfec8747341e65ef1`
  and `e490a9b9b198601826c4ccf99c31cd4ea155b729c1e63479a98773b0751aca9a`.
- To separate the original evaluator from cache-path effects, a separately
  authorized base-0/GPU4 replay ran the unmodified evaluator on the first tune
  scene (54 rows), using the original batch 4/BF16/nominal/seed 0/normal
  contract and exact `/usr/bin/python` runtime.  Child, supervisor, and OS
  return codes were 0.  Identity-joined plan, GT, and D3 were bitwise exact
  for the first 52 batch-matched rows, including the first eight rows.  The
  final two rows used a different terminal-batch composition and are reported
  separately.  Raw replay report SHA-256 is
  `4910dd2452362fa8e53d7db467986929ac389c409c5a7caa00b7a0a42ae0e002`
  (path `reports/p5_zero_selector_reference_replay_b0_20260908_ops/normal_scene019_54.json`);
  comparison SHA-256 is
  `9effe126ca3c56bfd794a3473aac65a584f83e0f606e3a005f89f6a957a0da6a`
  and its receipt SHA-256 is
  `878ea66da6ad0ecf6f167d9f4576479e6ee98024510eb6a5460d368b971a98f0`.
- The frozen cache and head tests passed together on CPU: 52 tests, actual
  exit 0.  A separately authorized base-0/GPU4 seven-forward diagnostic then
  exited 0.  Its evaluator `requires_grad=True` plain-hook-plain path was
  repeatable and bitwise exact to the immutable report.  Its
  `requires_grad=False` hook-plain-hook path was internally repeatable but
  differed from the immutable report in 37 plan elements (maximum absolute
  difference `5.7220458984375e-6`) and all four D3 values (maximum absolute
  difference `8.940696716308594e-7`); GT remained exact.  The direct cache
  hooked-first output and 790-value selector feature were bitwise exact to the
  evaluator's `requires_grad=False` hooked output.  Inputs and model
  construction/state/config/parameter/buffer contracts were exact.  This
  identifies `requires_grad_(False)` as the repeatable cache-specific numerical
  branch in this run; hook presence, hook order, input loading, and model
  construction were not the differing branch.  The immutable diagnostic is
  `reports/p5_zero_selector_forward_delta_b0_20260908_ops/diagnostic.json`
  (SHA-256 `2f659d3d860904aa780e500c5abeac64ec610cc3767c84372ed8898f571cc8c5`),
  with execution receipt SHA-256
  `0257e26efd597860561e1eef01316519e113fba8938060172340bb684e9c3217`
  and child receipt SHA-256
  `315a752517c59895017f98bd5958f04e9c5d535eeec76c8f4e58a918909e19b9`.
  Independent review rechecked the exact seven-forward count and all pinned
  pre/post, state, input, and constructor evidence.  No selector-head training
  has run.  No tolerance was introduced, no ensemble was adopted, and full
  train54,810/tune1,998 cache extraction, head training, deployment, and
  final-validation access remain unauthorized.

Throughout these diagnostics the exact runtime-22 contract, P4 checkpoints,
sidecars, data artifacts, and immutable evaluation reports remained pinned;
the new diagnostic helpers are separately attributed rather than relabelled as
the original training source.
