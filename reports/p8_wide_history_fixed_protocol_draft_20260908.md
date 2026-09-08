# P8 wider-history C/W paired experiment — fixed protocol draft

Status: **preregistered design plus local CPU-validated implementation candidate**. The
separately frozen history-overlay producer may be run only through its operator
gate (C all-row parity before W). This document itself authorizes no GPU execution,
training, tune metric, or final-row access.

## Question and single variable

Does replacing the least separated past image at 0.1 s with a 2.0 s past image improve longitudinal timing while keeping the number of image encodings and every model/loss choice fixed?

| arm | history frame offsets at 10 Hz | model time offsets |
|---|---|---|
| C | `[1,2,5,10]` | `[0.1,0.2,0.5,1.0]` s |
| W | `[2,5,10,20]` | `[0.2,0.5,1.0,2.0]` s |

This changes the temporal input **and its aligned history supervision policy**. It is not an isolated information-only ablation: history images, current-to-past SE(3), raw time offsets, and pose-derived history targets change together. The model's corresponding history prediction remains image-derived; the target is not. Current images, row identities, GT plan, state/stop/occupancy/lane targets, calibration, goal, losses, normalization scales, and architecture remain fixed. In particular, `HISTORY_SCALE=[10,5,1,1]` scales `(x,y,sin(yaw),cos(yaw))` components, not time, and is not retuned.

The actual filename-only official-test audit (`reports/motiondrive_v2_official_test_filename_inventory_20260908_ops.json`, SHA256 `4709c9614b3054c46d2703f879df5ad5df8a0ee661337b89f4372e571790619c`) found all integer camera frames `-30..0` in all six cameras for all 1,125 archives, so both input contracts are available. Member contents were not opened by that audit.

## Architecture and computation held fixed

- Six current views at 768x432 plus one temporal pass containing downsampled current front and four past front images at 384x216: exactly 11 image encodings in both arms.
- Same ResNet-50, low-feature motion source, C1 geometry, nominal-time policy, BF16 encoder/FP32 planner, fixed BN, batch16/microbatch2.
- Same existing per-cell real-goal conditioning and P7 control `cross_cell_goal_mode=zero`; no raw goal is added to planner or value tensors.
- Same motion encoder, scene encoder, planner, single `[6,2]` output, loss weights, uncertainty, unweighted stop BCE, parameter count, optimizer groups, output scales `[10,5]`, and final evaluator.
- The model already receives current observed motion through four image correlations, 192 motion tokens, predicted state, and predicted history. This experiment changes their temporal evidence span; it does not add a motion head.

## History-only supervision overlays

Never modify or overwrite the immutable C1 supervision root. Build two new isolated overlays, one per temporal contract, using only current train203 and tune37 identities. Each overlay contains exact `row`, `frame`, `history_transforms`, `history_target`, `history_valid`, and raw `time_offsets`; all state/stop/occupancy/lane/calibration/plan fields continue to come from the pinned C1 base.

The overlay builder reads only each selected scene's timestamps and ego pose, uses full SE(3), and records their file SHAs, split SHA, ego-cache SHA, base-supervision SHA, offsets, selected row SHA, and every output SHA. It must reject missing/extra/duplicate/out-of-order rows and must not index final scenes.

Mandatory barrier before W generation: generate the C overlay through the new path and prove all four temporal arrays bitwise equal to the existing C1 arrays for exact train54810 and tune1998 selected rows. Freeze both overlay manifests before any training. Both arms then consume the same overlay-loading code path. There is no silent fallback to base histories, on-the-fly offset substitution, or acceptance of an old supervision manifest under W.

Shared compressed/source files may contain other split bytes; only allowed train/tune identities may be indexed, transformed, summarized, or emitted. No final metric or selection is performed.

## Matched training curriculum

All four pipelines begin from the same pinned public I0 model state. C and W are rerun under one reviewed source/runtime; existing P4/P7 C checkpoints are historical references, not the paired control.

For each base seed 0 and 1:

1. Run an arm-specific P0 auxiliary pretrain for exactly 2,000 updates from public I0, with the original P4 architecture: `goal_on=false`, `state_on=false`, `cross_cell_goal_mode=disabled`, planning loss weight 0. Preserve the original fixed P4 P0 monitoring cadence: one auxiliary tune evaluation at step0, then evaluation and save every 250 updates through step2000. The historical history-position-MAE BEST artifact may be emitted as a nonselective diagnostic, but it is never chosen or consumed; LAST2000 is the only joint initializer.
2. Load that arm's own P0 model weights only. Do not resume optimizer or step.
3. Attach the P7 C zero-slot cross-cell branch for the first time, with a deterministic base-specific initialization. The branch state must be bitwise identical between C and W for the same base seed. Other initial weights legitimately differ because each P0 learned from its arm's temporal contract.
4. Start a fresh joint optimizer at step0 and run exactly 6,000 updates. Use the original P7-C joint recipe: head/non-backbone `1e-4`, backbone `1e-5`, warmup200 cosine, weight decay `.01`, gradient clip5, batch16/microbatch2, fixed BN, BF16/FP32, all joint parameters trainable.
5. Save LAST6000 and run the canonical full tune1998 evaluation exactly once. No intermediate joint tune/BEST, extension, early stopping, threshold, loss, scale, or history-span selection. Final data remain unopened.

Within a seed, C/W use identical row order, augmentation RNG, optimizer recipe, and branch initialization. The past pixels differ by design, while current pixels and their augmentation are exact. Physical GPU assignments are counterbalanced: seed0 C/GPU4 and W/GPU5; seed1 C/GPU5 and W/GPU4. Inter-wave checks are technical integrity gates only, never performance-dependent decisions.

Total proposed work is four P0 runs plus four joint runs, 32,000 optimizer updates. Runtime is to be estimated from the preserved P4 P0 and P7 joint receipts; no GPU reservation or launch follows from this draft.

## Backward-compatible implementation boundary

The old contract remains the default and must reproduce its current serialized contract and behavior exactly. New temporal metadata use defaulted config fields so old checkpoint configs restore C; W checkpoints must explicitly record offsets and seconds. Historical P4/P6/P7 source-pin tests remain unchanged and may intentionally reject the new source tree.

Minimal intended source surface after approval:

- Add validated temporal-contract fields to the model/run configuration; only exact C or W is accepted in this experiment.
- Add a new history-overlay builder and an optional dataset overlay loader. Default `None` retains the old C1 loading path byte-for-byte.
- Thread the selected nominal seconds through training and its one evaluator instead of a hidden module constant; default remains C.
- Parameterize the raw test adapter/input/serving contract by the reviewed temporal contract, defaulting to C. Pose rows already cover all `-30..0`; image reads remain exactly six current plus four selected past.
- Add a narrow experimental driver that pins I0, overlays, source closure, recipes, resource identity, paired initialization, and terminal evaluation. Do not weaken old exporters or provenance validators to accept the new tree.

## CPU and pre-launch gates

- Default-disabled/default-C source regression: old model state keys, outputs, serialized input contract, image read order, and supervision sample tensors remain exact.
- Synthetic and real-overlay C parity: exact row/frame and bitwise equality for history transforms/targets/valid/raw dt over train54810+tune1998 before W is emitted.
- W geometry: requested frames `[t-2,t-5,t-10,t-20]`, full-SE3 current-to-past convention, ordered raw dt, and exact nominal `[.2,.5,1.,2.]`; reject stale C overlay/bundle/fixture.
- Cross-arm invariants: identical allowed rows, GT plan, state/stop/occ/lane targets, calibration, current images, model graph/parameter count, optimizer groups, and source/input hashes.
- Public-I0 exact load; P0 branch absent; P0 optimizer step0→2000; own-arm LAST weights-only into joint; fresh joint optimizer step0→6000; same-base C/W new-branch key set and branch state SHA exact.
- The common 0.2/0.5/1.0 s history diagnostics may be compared by aligned slot meaning. The C-only 0.1 s and W-only 2.0 s diagnostics stay descriptive; no scale or loss retuning follows.
- Raw/test-shaped adapter and model spy: exact four history images/times/transforms, no provided status, six-input signature, and exactly 11 encodings. Actual trained latency is a later separate gate.
- All manifests and inputs remain immutable before/after; actual OS return codes and process/GPU release are preserved without relabeling failures.

## Preregistered result rule

On the two paired seed terminal tune reports, proceed only if all hold:

1. `W D3 < paired C D3` for both seeds;
2. two-seed mean `W-C <= -0.015 m`;
3. shared 11-session whole-cluster bootstrap (10,000 draws, seed `20260908`)
   resamples the same sorted 11 session indices once per draw and applies that
   draw jointly to both seeds. The estimator stays frame-weighted: divide the
   sampled-session delta sums by the sampled-session frame counts. Do not
   concatenate the two seeds as 22 clusters or bootstrap seeds. Its upper 95%
   CI must be `< 0`.

`mean W D3 <= 0.30 m` is an aspiration, not a forecast or required gate. Existing P4/P7 values do not substitute for the new paired C. Tune is reused for this preregistered decision, so any later proposal chosen from these results is exploratory; no final/leaderboard/generalization claim is made.
