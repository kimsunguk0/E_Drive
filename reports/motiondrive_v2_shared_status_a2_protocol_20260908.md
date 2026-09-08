# MotionDrive V2 shared-status A2 query protocol (2026-09-08)

Status: fixed opt-in proposal; no A2 training or result exists yet.

## Question and boundary

Does the same legitimate causal provided status used in A1 help when it changes the
32D context of image-attention queries before image values are selected? A2 is a
bounded continuation screen, not a planner replacement or global upper bound. It
changes no image inputs, goal route, raw-motion estimator, planner signature, labels,
heads, losses, history contract, output horizon, or P1--P8 artifact.

The first authorized comparison is seed 0 `zero` versus `provided_causal_5d`, both
initialized weights-only from the exact P7-C seed-0 LAST6000. The runner also pins the
corresponding P7-C seed-1 parent for a separately approved matched replication; it must
never mix a checkpoint or sidecar across seeds.

## Fixed arms and route

Both arms use the frozen A1 nominal-time train54,810/tune1,998 status overlay. `zero`
receives five exact zeros. `provided_causal_5d` receives causal
`(vx,vy,ax,ay,yaw_rate)` from current-10 through current pose records, with fixed scales
`[10,5,3,3,.5]`.

The only new model module is a `5→32→32` GELU MLP. Its final Linear weight and bias are
both exactly zero initially. In an autocast-disabled FP32 scope it computes
`delta(status/scales)`, which is added to `scene_encoder.query_context` before the
existing image attention. The result is cast back to the parent dtype. Status changes
attention queries/scores only; it is not an image value, scene-value beta, planner
argument, status token, trajectory residual, goal, command, or HD-label path. ZERO is
the same-size extra-capacity control and may learn a constant query shift after step 0.

## Parent, optimization, and evaluation

Seed 0 parent identities are checkpoint
`6145255ab2b1efd3deb169b873896e9b1b623ca49f11e5a46a71b138e3b0355e`, model state
`e79d545bbb09b4109c783a8cf4c861772bc5e52c62004ef5b37dc4ca63e938a2`, and sidecar
`cf287d05ce6b2f4476c07ac5c2253cd9ecb3227bcc01404adbbba8463bb66de0`.
Seed 1 counterparts are checkpoint
`8ffb429b17820b63477987f5c7de151dfa9e0e2b12378e247aed0b0e41de2563`, model state
`bfcd5e8f807013452d71bf00e824926d89f2a0582809336d2a144aa17e571b29`, and sidecar
`11854d7a9f828b8b4b6710679fd420f5c8d243d1322231d723137419aa48e80f`.

Within each seed, arms share complete assembled initialization, parameter count,
shuffle order, augmentation, and status overlay. Optimization is the unchanged A1
continuation recipe: fresh AdamW, backbone LR `5e-6`, other/new LR `5e-5`, weight decay
`.01`, warmup 100 then cosine, global clip 5, BF16 outer autocast, fixed BN running
statistics, batch16/microbatch2, and 2,000 updates. Plan/occupancy/lane/motion losses
remain `1/.2/.2/.2` with unchanged uncertainty behavior. Checkpoints are written at
500, 1000, and 2000 without selection. Exactly one tune evaluation occurs at step
2000 and LAST2000 is the only result.

## Execution gates

Before training: exact source/parent/split/supervision/calibration/overlay identities;
train54,810/tune1,998 exact rows and all-valid statuses; parent tensors unchanged;
query branch as the only missing state; within-seed arm-identical assembled and query
state hashes; initial full-parent output equality; train and terminal adapters carrying
the selected status; status-context cleanup/reentrancy tests; and a two-update finite
GPU smoke with no tune evaluation. Each invocation is one seed and one arm. The arm-to-
physical-GPU mapping remains A1's fixed GPU0 ZERO / GPU1 PROVIDED mapping.

Final136, new status definitions, planner inputs, additional conditioning variants,
checkpoint/epoch selection, and hyperparameter sweeps are prohibited. Any conclusion
is specific to this P7-C continuation and reused tune split.
