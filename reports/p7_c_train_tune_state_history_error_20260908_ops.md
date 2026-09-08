# P7-C cached TRAIN/TUNE state and history error audit — 2026-09-08

This is a read-only aggregate over the already-audited P7-C prediction caches and
ground-truth packs. It is inference error on canonical TRAIN/TUNE rows, not a
training-log loss, a new model forward, a fit, or an evaluation variant. No final
validation rows were accessed.

## Immutable inputs and method

- GT TRAIN 54,810: `gt_train.pt`, SHA-256
  `96eb0d2c9fa895a79e62f85f5c4a626185930790ec778ee6eeef0ce055cb46ec`
- GT TUNE 1,998: `gt_tune.pt`, SHA-256
  `8548f5df313eac452cc1ba34c894d5f68bcc1d7337d6122fe7012f6a7d485ea2`
- base0 predictions: TRAIN
  `5b06a432912324312bae97b4a14f796585a86b9532e2c57edc18694948e4699c`,
  TUNE `dc7e547345938a00407f0a739cb188849cf31f372499ad9ec029b39cc74bdad7`
- base1 predictions: TRAIN
  `c5e6efe92e593abb4bc2781fb610660a76853a35ac35139bf35355486b62ac9c`,
  TUNE `8b5fb837ceb1c75c36d388b891e76f17df574e072103ae92f7158b2ff2d92e6f`

For every seed/split, the prediction and GT `rows` tensors were required to be
bitwise equal. All selected validity masks were true. State residual is
`prediction - target`; bias is its mean, MAE is mean absolute residual, and RMSE
is the square root of mean squared residual. History error is the row-mean 2D
Euclidean distance between the first two history channels at each fixed offset.
All reductions used float64 after loading the stored float32 tensors on CPU.

## Results

| seed | split | vx bias | vx MAE | vx RMSE | ax bias | ax MAE | ax RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | TRAIN | +0.060588 | 0.308690 | 0.407067 | -0.004835 | 0.225759 | 0.329257 |
| 0 | TUNE | +0.015318 | 0.983957 | 1.270727 | +0.001371 | 0.280321 | 0.393056 |
| 1 | TRAIN | +0.061130 | 0.313408 | 0.412297 | -0.009075 | 0.228200 | 0.334490 |
| 1 | TUNE | +0.035739 | 0.973682 | 1.261078 | -0.005041 | 0.271563 | 0.381812 |

`vx` is in m/s and `ax` in m/s².

| seed | split | history 0.1 s | 0.2 s | 0.5 s | 1.0 s |
|---|---|---:|---:|---:|---:|
| 0 | TRAIN | 0.051554 | 0.088090 | 0.196239 | 0.378865 |
| 0 | TUNE | 0.108649 | 0.204419 | 0.499498 | 0.995452 |
| 1 | TRAIN | 0.051100 | 0.088474 | 0.198371 | 0.382769 |
| 1 | TUNE | 0.107946 | 0.201521 | 0.493964 | 0.987297 |

History values are mean XY Euclidean errors in meters.

The estimator itself has a large TRAIN-to-TUNE gap, especially for `vx` and
longer history offsets. This is evidence of estimator generalization error, but
does not by itself prove that this is the sole cause of the downstream Pred24
MLP gap.
