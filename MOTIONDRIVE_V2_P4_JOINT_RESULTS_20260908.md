# MotionDrive V2 P4 fresh joint results — 2026-09-08

## Status and scope

P4 produced two immutable `LAST6000` joint checkpoints and evaluated each once
on the fixed normal tune split (1,998 frames, 37 scenes, 11 sessions).  The
evaluation contract was C1 geometry, nominal time, BF16, batch 4, four workers,
and same-forward motion predictions.  No final-validation or test labels were
accessed, and no best-of-two seed selection was performed.

The primary LAST6000 tune results are:

| run | official equal-frame D3 | checkpoint SHA-256 |
|---|---:|---|
| seed 0 | 0.3722216174 | `3e9ae5aa6ca89f4eb9a72ba38355dfdbec4ccaa38eabbebb3526fa3019404478` |
| seed 1 | 0.3669574755 | `c937f2f772c78f99cf3cfa5b1a96614267e20a9a667a157f4ada284940f9672e` |
| descriptive mean | 0.3695895465 | not a checkpoint |

The two-seed range is `0.0052641419`.  Two seeds do not establish a seed
distribution.  The paired seed1-minus-seed0 difference is `-0.0052641419`,
with an 11-session cluster-bootstrap 95% interval of
`[-0.0175989518, 0.0028907967]`; selecting seed 1 alone would be optimistic.

## Execution-status separation

The original training supervisors both remain **actual rc1**, with
`completion_evidence: null`.  The in-process trainer body had already written
completed step-6000 sidecars (`nonfinite_count: 0`) and LAST checkpoints before
the post-training global-HEAD guard observed a shared-repository commit.  There
is no separate trainer subprocess OS return code.  These preserved records are
not relabelled as clean training exits.

An independent **CPU artifact audit exited rc0**.  It strictly loaded both full
models, verified finite step-6000 optimizer state, fixed BN state, each seed's
own P0 LAST lineage, completed sidecars and receipts, and confirmed that all 22
runtime sources and all three pinned data inputs still matched their launch
snapshots.  This rc0 establishes artifact validity under the documented
global-HEAD-only exception; it does not change either training rc1.

Both subsequent **GPU evaluation supervisors and evaluator children exited
rc0**, with no pressure event and no residual owned PID.  Their protocols were
preregistered before any forward and record `selection_performed: false` and
`final_val_accessed: false`.

## Source lineage and HEAD-drift exception

- Training source: `86620b4ffc7e6838b49cf83b5be789eba12d8027`.
- Shared HEAD changed during training to
  `0fad18f5dc8a9b9343e03ccd11c42b2b8b10e678` and then
  `095e49d4e7be36802bad2f5923b873e65115e7fc`.
- Those commits only added `LEADERBOARD_DIAGNOSIS_20260908.md`,
  `scripts/bench_sparsedrive_3090.py`, `scripts/e25_banned_baselines.py`, and
  `scripts/e26_pose_only_ceiling.py`; they did not alter the frozen 22-file P4
  runtime or the three pinned data files.
- Evaluation source: `095e49d4e7be36802bad2f5923b873e65115e7fc`.
  Both protocols record the same runtime file hashes as training.

The original supervisor records, child receipts, completed sidecars, and
checkpoints remain byte-preserved and retain their training-source attribution.

## Interpretation

The historical P3 C1 reference `0.4379407` is higher by `0.0657191` and
`0.0709832` for the two P4 seeds.  This is a competitiveness screen, not a
geometry-only causal effect, independent holdout, leaderboard, or first-place
claim.  The historical C0/T1 result `0.3661751849` used the same ordered tune
rows and has no discovered final-data leakage, but its initialization, training
budget, geometry history, and checkpoint-selection lineage differ.  It is a
valid historical reference, not a condition-matched causal control; P4 differs
from it by `+0.0060464` and `+0.0007823` D3 for seeds 0 and 1.

CPU paired analysis localizes the remaining error: compared with historical
C0/T1, the two-seed P4 mean improves on non-stop frames but regresses strongly
on 123 stop frames.  Same-forward state outputs also show stop-state and
planning errors co-occurring, but this association does not establish a causal
mechanism.  The `stop` bucket is defined from the current-frame state and does
not assert that the vehicle remains stopped for the following three seconds.
These subgroup observations are descriptive.  The full frame-level analysis is
intentionally not added to Git; it is retained as
`reports/p4_actual_results_analysis_20260908/p4_results_analysis.json`, SHA-256
`5101a7d535cdc9ad2fe2bef13976d03b8545f1c8fb97d5d5a1f5ef6b859c796f`.

Real RTX 3090 deployment checks completed for both bundles.  Each timing run
encoded all 11 images (six high-resolution and five low-resolution), used batch
1 with no feature cache, warmed up 20 times per clip, and measured 50 full
forwards on each of eight preregistered clips (400 samples).  Seed 0 CUDA
latency was median `37.1036 ms`, p95 `37.1981 ms`; seed 1 was median
`36.9889 ms`, p95 `37.0690 ms`.  Raw-reference parity passed bitwise on eight
clips, BF16 encoder/FP32 planner precision and all 11 encodings were observed,
ABA statelessness and pre/post model-state hashes passed, and the actual smoke
commands exited 0.  These are model-forward-only timings: preprocessing, H2D,
and postprocessing are excluded.  No older latency number is carried forward.

The separate 97-forward-per-bundle goal-response runs exited 0 and are
descriptive diagnostics only: they compute no GT accuracy metric and define no
compliance or pass threshold.

## Immutable evidence

| evidence | repository-relative or external path | SHA-256 |
|---|---|---|
| independent artifact audit | `reports/p4_joint_head_drift_audit_20260908/audit.json` | `bb8b64b27979dc9382a190d3c1bb78696a3ad51f3a989927f8075a3ce375f200` |
| audit command receipt | `reports/p4_joint_head_drift_audit_20260908/command_receipt.json` | `e3f84941a51b04b4c84952537f3035a351acd1d5f14745ace027ae50ebe964ce` |
| seed-0 evaluator report | `reports/p4_joint_full_tune_eval_s0_20260908_ops/normal_full_tune.json` | `cffa9cc8b12f5f9812af7b3fd3631e74597e291b02aecd1cc14d8ec8a1aeba2f` |
| seed-0 protocol | `reports/p4_joint_full_tune_eval_s0_20260908_ops/normal_full_tune.protocol.json` | `29118cf3e3f55ab1aa12b9a17271418456d82c8708090c4d816b4ea7a70f2c9d` |
| seed-0 OS execution receipt | `reports/p4_joint_full_tune_eval_s0_20260908_ops/execution.json` | `8f31107d54875510d284fae4e3cf0753c3d11932846492336f2ceb2ed1802537` |
| seed-0 child receipt | `reports/p4_joint_full_tune_eval_s0_20260908_ops/child_receipt.json` | `73789889759d66451441b426b63437fa94d713e3db6a42781d2caa3e3842bcf3` |
| seed-1 evaluator report | `reports/p4_joint_full_tune_eval_s1_20260908_ops/normal_full_tune.json` | `0e71bef199931cb1916abdc819ca78cf75d8cd7055439bd084ca975bb610c137` |
| seed-1 protocol | `reports/p4_joint_full_tune_eval_s1_20260908_ops/normal_full_tune.protocol.json` | `3ddc8de4e9b5fcc7a970bb3e41c11aecb53f106f67211243188fc24a3990fe6c` |
| seed-1 OS execution receipt | `reports/p4_joint_full_tune_eval_s1_20260908_ops/execution.json` | `7226c0e94f2d5a96f9ed61494c8f3bc21efaae49c6d594e9bb7c8c404cba86b1` |
| seed-1 child receipt | `reports/p4_joint_full_tune_eval_s1_20260908_ops/child_receipt.json` | `cfe253b0ec5081f7e2f4634b142d208812555587a1d2a37e6e0c854828d2e579` |
| paired-analysis summary | `reports/p4_actual_results_analysis_20260908/README.md` | `831f277d503a85c434466c07971ab8584b48f55ef1c972563e581a97cddf40fd` |
| seed-0 RTX 3090 smoke/timing | `reports/p4_s0_smoke_timing.json` | `7c4418edf979e1f3aa624bfdc923702e71beb9b0a91c0a66f3526d2cc36feca6` |
| seed-1 RTX 3090 smoke/timing | `reports/p4_s1_smoke_timing.json` | `233b36935e895eb91a9c2b6d41d8b28ff16106c10bb437bc10ca72ad3c68e706` |
| seed-0 goal-response diagnostic | `reports/p4_s0_goal_response.json` | `d3ad8fb6eb7880903a820b6c05943d274eb0c8f6927d828b69e487abef11777a` |
| seed-1 goal-response diagnostic | `reports/p4_s1_goal_response.json` | `709cc146a8dceedbfc992a389a9c8031fb157f4ef2d47cb4f7ffb346ec4a15a7` |
| RTX 3090 execution receipt | `reports/p4_3090_execution_receipt.json` | `cbdaa45d22ead73d1c30b245abb10ef9d5bfa575db9ff513000f7646df3a27f4` |
| fixed grouped split | `data/etri/motiondrive_v2/grouped_split_rawtime.json` | `f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936` |

Large checkpoints and frame-level evaluator/analysis dumps are referenced by
path and digest only; they are not added to Git.

## Deployment status

The isolated deployment bundles have SHA-256
`3acc4d30b17bf6ff3ad45d334d081a20e89dba9e31a022ee3c9984c9c7ec0129`
(seed 0) and
`edfd1bb738b878653b16b546336112f47f5bd100be044826f1727b5e15bd9e49`
(seed 1).  They are referenced rather than added to Git.  The real RTX 3090
smoke/parity/timing checks described above are complete.  No additional
training, schedule extension, seed, final evaluation, or checkpoint-selection
change is authorized by this record.
