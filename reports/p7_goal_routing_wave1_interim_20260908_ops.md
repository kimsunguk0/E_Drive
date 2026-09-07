# MotionDrive v2 P7 wave-1 interim result (2026-09-08 KST)

## Scope and status

This is a read-only analysis of the completed base-0 control (`C`) and real-goal (`G`) runs. It uses the same 1,998 ordered tune rows and their stored predictions; it did not run another forward pass and did not access the final holdout. Base-1 replication was still running when this report was written.

The preregistered KEEP rule requires both seeds to satisfy `G < C` and `G < original P4`. Verified base 0 violates both inequalities, so that exact KEEP rule is already false and base 1 cannot rescue it. Base 1 nevertheless continues unchanged because the protocol prescribes both seeds. This is not an adoption decision for `C` and not a universal causal claim.

## Immutable evidence

| Artifact | Path | SHA-256 |
|---|---|---|
| original P4 base-0 tune report | `reports/p4_joint_full_tune_eval_s0_20260908_ops/normal_full_tune.json` | `cffa9cc8b12f5f9812af7b3fd3631e74597e291b02aecd1cc14d8ec8a1aeba2f` |
| P7 base-0 C final eval | `work_dirs/motiondrive_v2/p7_goal_routing_b0_control_last6000/final_eval.json` | `67279fe06120ef752d44c915dbc01d022244b94d60ba27577ed62d1bace1afd0` |
| P7 base-0 C manifest | `work_dirs/motiondrive_v2/p7_goal_routing_b0_control_last6000/manifest.json` | `cf287d05ce6b2f4476c07ac5c2253cd9ecb3227bcc01404adbbba8463bb66de0` |
| P7 base-0 C LAST6000 | `work_dirs/motiondrive_v2/p7_goal_routing_b0_control_last6000/last.pth` | `6145255ab2b1efd3deb169b873896e9b1b623ca49f11e5a46a71b138e3b0355e` |
| P7 base-0 C train metrics | `work_dirs/motiondrive_v2/p7_goal_routing_b0_control_last6000/metrics.jsonl` | `60772877bc0098436fd3d7d6f0f51fd885cf1bf1b3fc6f5897f50283c2f7e5b8` |
| P7 base-0 G final eval | `work_dirs/motiondrive_v2/p7_goal_routing_b0_goal_last6000/final_eval.json` | `675b4d4337410e508ad3c0fb1dc2a00eb3ddc7f9cfca7ad334c749b27fa13d18` |
| P7 base-0 G manifest | `work_dirs/motiondrive_v2/p7_goal_routing_b0_goal_last6000/manifest.json` | `814b55535b860d5bd7cbd72b549a4951359bbd2569d5689a26356589068e9417` |
| P7 base-0 G LAST6000 | `work_dirs/motiondrive_v2/p7_goal_routing_b0_goal_last6000/last.pth` | `ccf138c7a3485e99e13cfdd229f2b0a4e0122b8eabfb4404979dac1b00c7d309` |
| P7 base-0 G train metrics | `work_dirs/motiondrive_v2/p7_goal_routing_b0_goal_last6000/metrics.jsonl` | `7f813146823dfc176fbea10b66969377882c26ee5daf21bc66ed35dcdba52cce` |

The three reports join exactly on all 1,998 ordered `(scenario,row,session,frame)` identities and ground-truth paths. Goal groups use the same joined rows from `/tmp/pm97/data/etri/ego_cache.npz`, SHA-256 `d35a69bb229b2d4736aff7dddefd3522ddbd346e112283e2ce58ce3986a023bd`. Runtime source provenance remains the pinned 13-file closure at source Git `64e92694ccbe0a3c5a581e421e16a884b9c5a088`; launch HEAD differences contain only out-of-closure records.

## Terminal tune result

All values below are frame-weighted over the fixed tune population. D3 is the registered equal-frame mean of cumulative ADE at 1, 2, and 3 seconds.

| Metric | original P4 | P7 C | P7 G | G - C | C - P4 |
|---|---:|---:|---:|---:|---:|
| D3 | 0.372222 | 0.330586 | 0.424389 | +0.093803 | -0.041635 |
| ADE 1 s | 0.237763 | 0.202848 | 0.290568 | +0.087720 | -0.034915 |
| ADE 2 s | 0.371913 | 0.322830 | 0.433883 | +0.111053 | -0.049083 |
| ADE 3 s | 0.506989 | 0.466081 | 0.548718 | +0.082637 | -0.040908 |
| weighted absolute longitudinal error | 0.340630 | 0.296556 | 0.396532 | +0.099976 | -0.044074 |
| weighted absolute lateral error | 0.079683 | 0.084108 | 0.080884 | -0.003224 | +0.004425 |

The G degradation is predominantly longitudinal, not lateral.

## Paired localization

Each contribution is `sum(row delta) / 1998`, so group contributions add to the overall paired delta.

| Fixed group | n | P4 D3 | C D3 | G D3 | G-C group mean | G-C contribution |
|---|---:|---:|---:|---:|---:|---:|
| future steady | 99 | 1.212492 | 1.028565 | 0.759689 | -0.268876 | -0.013323 |
| departing | 24 | 1.071579 | 0.935933 | 1.025366 | +0.089433 | +0.001074 |
| nonstop | 1,875 | 0.318904 | 0.285984 | 0.398993 | +0.113009 | +0.106052 |
| goal norm <= 0.2 m | 91 | 1.149433 | 0.968102 | 0.706240 | -0.261862 | -0.011927 |
| goal norm > 0.2 m | 1,907 | 0.335134 | 0.300165 | 0.410940 | +0.110775 | +0.105730 |
| in grid (`x` in [-10,70], `y` in [-32,32], inclusive) | 1,648 | 0.387164 | 0.337661 | 0.414485 | +0.076824 | +0.063366 |
| out of grid | 350 | 0.301862 | 0.297275 | 0.471025 | +0.173751 | +0.030437 |

For out-of-grid rows, G-C ADE deltas at 1/2/3 seconds are `+0.099391/+0.186497/+0.235364`. Session 046 contributes `+0.027088` to G-C, but excluding it still gives G-C `+0.091424`; the result is not a session-046-only artifact. G is worse than C in 9 of 11 session means and better in sessions 092 and 112.

## Auxiliary and training-log context

Terminal auxiliary metrics do not show a broad collapse. C/G history errors by offset are `[0.10865,0.20442,0.49950,0.99545]` / `[0.11026,0.20376,0.49174,0.97452]`; state MAE `(vx,vy,ax,ay,yawrate)` is `[0.98396,0.03684,0.28032,0.08399,0.00796]` / `[0.97053,0.03899,0.28005,0.07809,0.00758]`; occupancy/lane IoU is `0.42879/0.48377` / `0.43355/0.49140`.

The final ten logged training points (steps 5910 through 6000) have mean minibatch planning D3 `0.186061` for C and `0.176877` for G. These are augmented training minibatches, not tune evaluation. Their ordering opposite to the terminal tune result is evidence against using train D3 as the performance answer; it is not a train-fit or generalization proof.

## Bounded interpretation

Because C and G share the same base-0 P0 weights, initial assembled state, RNG/data order, optimizer recipe, and terminal evaluation rows, the paired evidence localizes the difference to the slot value used by the same new routing branch under this run. C is not neutral global attention: its zero slot creates a fixed ego-centered Gaussian source prior for every destination query, whereas G moves that prior center to the provided 5-second goal. Therefore C-P4 confounds added capacity, the ego-centered prior, and continuation optimization; G-C tests moving the source-prior center from ego to the 5-second endpoint, not general goal usefulness. The observed pattern is consistent with a planning-specific routing/generalization failure: G improves the small steady/near-zero subset and keeps auxiliary metrics comparable, but shifts the much larger non-stop and non-zero-goal population adversely, especially longitudinally. It does not establish a universal mechanism, isolate a single internal cause, or justify thresholding, retuning, extending training, or adopting C. Base-1 replication and the preregistered four-run analysis remain required for the final report.
