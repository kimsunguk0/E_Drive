# MotionDrive V2 P4 LAST6000 actual tune analysis — 2026-09-08

## Evidence status

This is a CPU-only analysis of the two immutable `normal` tune evaluator dumps.
It performs no model forward and does not read final validation/test data.

- P4 training source: `86620b4ffc7e6838b49cf83b5be789eba12d8027`
- Evaluation source: `095e49d4e7be36802bad2f5923b873e65115e7fc`
- Both evaluation supervisors and evaluator children exited 0.
- Both original training supervisors remain actual rc1 because the global HEAD
  guard fired after the trainer body wrote its completed step-6000 artifacts.
  The independent CPU artifact audit exited 0, but neither that audit nor this
  analysis relabels the original rc1 records as clean training exits.
- The frozen analyzer was not edited. Direct path invocation first exited 1
  before reading inputs (`ModuleNotFoundError: scripts`); invocation as
  `python3 -m scripts.analyze_motiondrive_v2_p4_results` then exited 0.

The completed analysis is `p4_results_analysis.json`, SHA-256
`5101a7d535cdc9ad2fe2bef13976d03b8545f1c8fb97d5d5a1f5ef6b859c796f`.

## Primary P4 result

All values use the fixed 1,998-frame, 37-scene, 11-session tune split, C1
geometry, nominal time, BF16, batch 4, and the official equal-frame D3.

| result | official D3 | 11-session cluster-bootstrap 95% interval |
|---|---:|---:|
| seed 0 LAST6000 | 0.3722216174 | [0.3076913595, 0.4369865694] |
| seed 1 LAST6000 | 0.3669574755 | [0.3061245336, 0.4284722040] |
| two-seed descriptive mean | 0.3695895465 | not a population estimate |

The seed range is `0.0052641419`. The paired seed1-minus-seed0 difference is
`-0.0052641419`, session-cluster 95% interval
`[-0.0175989518, 0.0028907967]`. Two seeds do not establish a seed
distribution, and selecting only seed1 would be optimistic.

## Historical C0 reference — not a causal control

The optional C0/T1 LAST3000 dump has the exact same ordered rows and GT and an
official D3 of `0.3661751849`. It differs in initialization, geometry, training
source/schedule and checkpoint selection, so these paired numbers are only
historical localization:

| contrast | frame-weighted delta | 11-session cluster-bootstrap 95% interval |
|---|---:|---:|
| P4 seed0 minus C0 | +0.0060464326 | [-0.0573859843, 0.0447834071] |
| P4 seed1 minus C0 | +0.0007822907 | [-0.0604558217, 0.0371104641] |

Neither interval distinguishes P4 from this historical reference. This is not
a geometry, initialization, or training-budget effect estimate.

## Where P4 is better and worse relative to that reference

Both seeds show the same main failure structure. On the 1,875 non-stop frames,
seed0/seed1 improve by `-0.0356630/-0.0350498` D3. On the 123 stop frames they
regress by `+0.6418609/+0.5470028`. Thus a small subgroup cancels most of the
non-stop gain.

| GT-state bucket | n | P4 two-seed mean | C0 | descriptive delta |
|---|---:|---:|---:|---:|
| stop | 123 | 1.1375680 | 0.5431361 | +0.5944319 |
| accel | 203 | 0.3990437 | 0.4323235 | -0.0332799 |
| decel | 186 | 0.3930949 | 0.4088950 | -0.0158000 |
| cruise | 1,486 | 0.2990562 | 0.3371441 | -0.0380879 |

The two largest P4 scenario errors are repeatable across seeds:
`20260210-101004` is 36/54 stop frames and scores `1.188731/1.155613`
(C0 `0.520914`); `20260210-100854` is 28/54 stop frames and scores
`0.926885/0.899771` (C0 `0.516681`). Scenario/session effects are heterogeneous:
each seed is descriptively lower than C0 on 19/37 scenarios, while the absolute
session-cluster intervals remain broad.

P4 trades lateral accuracy for longitudinal error. Averaged across the two P4
seeds, weighted absolute longitudinal error is `0.3374435 m`, `+0.0596731 m`
relative to C0, while weighted absolute lateral error is `0.0814055 m`,
`-0.0647252 m` relative to C0. The P4 signed longitudinal error grows positive
from about `+0.034 m` at 0.5 s to `+0.233 m` at 3 s, whereas C0 remains near
zero; this is compatible with a forward-distance bias, not proof of its cause.

The two-seed mean waypoint-L2 deltas versus C0 at 0.5, 1.0, 1.5, 2.0, 2.5,
and 3.0 s are respectively
`[+0.021244, +0.023619, +0.007124, -0.024282, -0.065387, -0.077004] m`.
Accordingly, cumulative ADE deltas at 1/2/3 s are
`[+0.022431, +0.006926, -0.019114] m`: P4 is worse early and better late in
this non-causal comparison, while the official metric weights early points
most heavily.

The evaluator dumps also contain state predictions from the same forward.
Overall vx MAE is `0.9805/0.9999 m/s` for seed0/seed1. On stop frames the GT
mean speed is `0.0104 m/s`, but predicted mean speed is `1.1525/1.1120 m/s`;
within those 123 frames, the Pearson association between D3 and predicted
speed is `0.842/0.876`. This shows that stop-state and planning errors co-occur.
It does not establish that the predicted state token caused the plan error or
that it is the only failure mechanism.

## Immutable input receipts

| input | SHA-256 |
|---|---|
| seed0 evaluator report | `cffa9cc8b12f5f9812af7b3fd3631e74597e291b02aecd1cc14d8ec8a1aeba2f` |
| seed0 protocol | `29118cf3e3f55ab1aa12b9a17271418456d82c8708090c4d816b4ea7a70f2c9d` |
| seed0 evaluation execution | `8f31107d54875510d284fae4e3cf0753c3d11932846492336f2ceb2ed1802537` |
| seed0 evaluator child receipt | `73789889759d66451441b426b63437fa94d713e3db6a42781d2caa3e3842bcf3` |
| seed1 evaluator report | `0e71bef199931cb1916abdc819ca78cf75d8cd7055439bd084ca975bb610c137` |
| seed1 protocol | `3ddc8de4e9b5fcc7a970bb3e41c11aecb53f106f67211243188fc24a3990fe6c` |
| seed1 evaluation execution | `7226c0e94f2d5a96f9ed61494c8f3bc21efaae49c6d594e9bb7c8c404cba86b1` |
| seed1 evaluator child receipt | `cfe253b0ec5081f7e2f4634b142d208812555587a1d2a37e6e0c854828d2e579` |
| independent source/artifact audit | `bb8b64b27979dc9382a190d3c1bb78696a3ad51f3a989927f8075a3ce375f200` |
| audit command receipt | `e3f84941a51b04b4c84952537f3035a351acd1d5f14745ace027ae50ebe964ce` |
| historical C0 report | `68bc0ed92a82dda05d5ae3c522d9532a6b4f238eaf508d1320f9f226ad2107d3` |
| fixed grouped split | `f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936` |

CPU schema tests and this CPU analysis validate report integrity and statistics;
they are not new model forwards, deployment parity, latency, final-validation,
or leaderboard evidence.
