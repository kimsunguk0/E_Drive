# MotionDrive v2 P7 goal-routing results (2026-09-08 KST)

## Outcome

All four preregistered runs completed 6,000 updates and exactly one 1,998-row tune evaluation. All model and optimizer tensors were finite, all optimizer states were at step 6000, paired C/G sample orders were identical within each base, source and inputs were unchanged, and all unified SSH commands returned 0. The final CPU analysis used stored predictions only; it performed no model forward and did not access the final holdout.

The preregistered G KEEP gate is **false**. G is worse than both C and original P4 on both bases, its two-base mean is above 0.34, and the shared 11-session interval is strictly on the adverse side. No threshold, seed, checkpoint, extension, or follow-on model was selected.

| base | original P4 | C: ego-prior slot | G: real-goal slot | G-C | G-P4 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.372222 | 0.330586 | 0.424389 | +0.093803 | +0.052168 |
| 1 | 0.366957 | 0.324619 | 0.434980 | +0.110361 | +0.068023 |
| mean | 0.369590 | 0.327603 | 0.429685 | +0.102082 | +0.060095 |

Shared-session bootstrap results use one rowwise two-base mean delta, the same 11 sorted session clusters per draw, 10,000 draws, seed 20260908, and frame-weighted aggregation. G-C is `+0.102082`, 95% CI `[+0.072843,+0.130415]`; G-P4 is `+0.060095`, CI `[+0.034558,+0.082090]`.

## Error localization

The shared G-C delta is `-0.243666` on steady 99 rows, `+0.035859` on departing 24 rows, and `+0.121185` on nonstop 1,875 rows. It remains `+0.101498` after excluding session 046. It is `+0.087204` for 1,648 in-grid goals and `+0.172139` for 350 out-of-grid goals. Thus the steady subset improves, while the much larger nonstop population dominates the adverse total. Base-0 path decomposition further shows the degradation is predominantly longitudinal; auxiliary history/state/occupancy/lane metrics remain comparable, so this is not evidence of a broad auxiliary collapse.

Training minibatch D3 is not the answer: in base 0, the final ten logged training points favored G (`0.176877` versus C `0.186061`), while the terminal fixed tune evaluation strongly favored C. This is why no causal or performance claim is drawn from training-log loss ordering.

## C as an exploratory working baseline

C improves against each own-base original P4 report: base 0 delta `-0.041635`, 11-session CI `[-0.066958,-0.022123]`; base 1 delta `-0.042339`, CI `[-0.063602,-0.026822]`. The rowwise two-base mean delta is `-0.041987`, shared-session CI `[-0.065010,-0.025029]`, and mean C D3 is `0.327603`.

This is a replicated exploratory working baseline, not the preregistered G gate and not an official adoption claim. C is not neutral global attention. The new branch gives every destination query a fixed ego-centered Gaussian source prior; G moves that source-prior center to the provided 5-second goal. Consequently, C-P4 confounds added capacity, the ego-centered prior, and continuation optimization. G-C tests moving that prior center from ego to the 5-second endpoint, not general goal usefulness.

## Evidence

- Four-run execution receipt: `reports/p7_goal_routing_execution_20260908_ops.json`.
- Frozen analyzer output: `reports/p7_goal_routing_results_20260908_ops.json`, SHA-256 `7f17f64a40495060d9b7e252604d34b8ae91134a587ba91d54ff4d83c6ff27f0`.
- Detailed completed base-0 paired analysis: `reports/p7_goal_routing_wave1_interim_20260908_ops.md`.
- Official metric interpretation: `reports/motiondrive_v2_official_metric_reconciliation_20260908.md`.
- Protocol: `MOTIONDRIVE_V2_P7_GOAL_ROUTING_PROTOCOL.md`, SHA-256 `aa85ad0634fb65dfac0cd6490242fbbd319ad98c0baa081bd6702886c56db1ad`.

The earlier P5 approximately 0.30 oracle was a ground-truth selector over the fixed P4 and ZERO candidates. It is not a performance lower bound for P7 and is not mixed with this result.
