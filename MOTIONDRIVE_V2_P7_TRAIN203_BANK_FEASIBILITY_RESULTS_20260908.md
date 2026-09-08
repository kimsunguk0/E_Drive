# MotionDrive-v2 P7 train203 bank feasibility result (2026-09-08)

## Decision

The fixed ONE-K1024/M12 proxy is **not adopted**. The preregistered goal-endpoint selector improves over the prefix-nearest proxy, but it remains worse than the untouched P7 C prediction for both base seeds and in the two-base mean. No bank-size, shortlist-size, NMS, threshold, selector, or training sweep follows from this result.

This is a CPU-only, reused-tune diagnostic. It is not a learned multimodal predictor, a production selector, a final-holdout result, or evidence of official adoption.

## Frozen recipe and separation

- Bank: one exact-zero trajectory plus 1,023 unique medoids selected only from the current grouped-split train inventory (54,810 rows, 203 scenes).
- Fit coordinates: first six future XY points (12 dimensions), scaled by the square root of official horizon weights `[11, 11, 5, 5, 2, 2] / 36`; no `StandardScaler`.
- Fit: `MiniBatchKMeans`, seed `20260908`, batch 4,096, `n_init=3`, `max_iter=200`, `max_no_improvement=25`, `reassignment_ratio=0`, k-means++.
- Evaluation: immutable C0/C1 predictions, a fixed M12 prefix-nearest shortlist, and two fixed selectors (prefix-nearest and provided 5 s goal-endpoint-nearest).
- The bank manifest and all four bank artifact hashes were frozen after fit and before tune predictions or labels were opened.
- No final rows, GPU, neural training, new model forward, or post-outcome rule change was used.

## Results

All values are canonical equal-frame D3 over the same 1,998 tune rows.

| Base | Direct C | Prefix-nearest M12 | Goal-endpoint-nearest | Prefix - C | Endpoint - Prefix | Endpoint - C |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.33058614 | 0.34622130 | 0.33384800 | +0.01563519 | -0.01237331 | +0.00326188 |
| 1 | 0.32461891 | 0.34288949 | 0.32683477 | +0.01827060 | -0.01605474 | +0.00221586 |
| Mean | 0.32760252 | 0.34455539 | 0.33034138 | +0.01695287 | -0.01421401 | +0.00273887 |

The decomposition is therefore: mapping the direct C path onto the fixed prefix shortlist costs **+0.01695287 D3**; the fixed endpoint rule recovers **0.01421401 D3** of that loss; the realized endpoint proxy still finishes **+0.00273887 D3 worse than direct C**. Endpoint and prefix selectors choose different candidates on 1,663/1,998 rows for each base.

The M12 GT oracle is 0.19271011 D3 and the full-K metricwise oracle is 0.11486043 D3. These are non-deployable hindsight/coverage bounds. They are not realized selectors, learned multimodal upper bounds, or evidence that increasing K/M would improve the fixed proxy.

## Immutable evidence

- Frozen script: `scripts/analyze_motiondrive_v2_train203_bank_feasibility.py`, SHA256 `b302c0db66baf4e6483fa8e2498a2c6f152f012747a553abb34ce6eb0474e33b`.
- Test: `tests/test_motiondrive_v2_train203_bank_feasibility.py`, SHA256 `3047bd87b16145d8be87a0474807f0671652b696ac438620e57a739d2eb465b1`; B200 actual result: 25 passed, exit 0.
- Fit handle/PID/exit: `45036` / `2310428` / 0. Evaluate handle/PID/exit: `78783` / `2311220` / 0. Both PIDs were absent after exit.
- Bank manifest: `reports/p7_train203_bank_feasibility_20260908_ops/bank/bank_manifest.json`, SHA256 `33d7e93c3c776363dc3636fca127a9b6a7bffbc622b6bfea61e9e5a030b8b16c`.
- Bank artifacts: `bank.npy` `ee0e2267d29bb2253c5e0542d52b01cc5a0537ee66af10b55188afa3605aa469`; `cluster_ids.npy` `bf999e769fad0c3d6bd32b775165395531c8cbc1574dabdba7c08b0a9c00447d`; `cluster_support.npy` `2e18a7217d0e74d650d73c5aebcb56a474661737be874663d216c7b67315236d`; `source_rows.npy` `56718f763c84b3ea8d6b242d47f9f3298f0f0456bfd7cfbe30f32f01ef7c1ed4`.
- Fit receipt: `reports/p7_train203_bank_feasibility_fit_receipt_20260908_ops.json`, SHA256 `74dc42f7dd29c32bccddd2470b5f3b768c39f62b71ddec058f4e6f7ca8b8579f`.
- Evaluation output: `reports/p7_train203_bank_feasibility_20260908_ops/tune_feasibility.json`, SHA256 `32947a5bb83df6d12d8ab47427b854c761d3fcd10d4c62480dbd2e9b8e5585c7`.
- Execution receipt: `reports/p7_train203_bank_feasibility_execution_20260908_ops.json`, SHA256 `cf1d9b722c544cd3cacdd8c7e51d0c59e21f241829fe43649fd9d77d63d63e3c`.

The first train-only uniqueness probe exited 1 because of a NumPy void-view shape error; it produced no bank and did not open tune or final data. The corrected read-only probe exited 0 and found 54,810 distinct nonzero full-10 trajectories. This failed probe is preserved in the execution receipt and is not relabeled.
