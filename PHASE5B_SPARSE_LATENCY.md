# Phase 5B — current-only sparse trunk latency gate

## Decision

**PASS / VERY GOOD.** The complete current-only skeleton takes **13.003 ms
median in AMP FP16** and **18.923 ms in FP32** on the RTX 3090. Both are far
below the 100 ms no-penalty boundary and leave substantial budget for training
features or one lightweight history frame.

This is a latency/structure result only. The model has random weights and makes
**no accuracy claim**.

## Measured structure

One top-level `forward(images, lidar2img)` contains all of the following:

1. six current (`t0`) camera images at 768×432, batched through one shared
   ResNet-34;
2. a 128-channel FPN at strides 8 and 16;
3. actual ETRI ego/lidar-to-camera projection of all A0 K=1024 × 10 waypoints,
   sampled at z=0 m and z=1 m;
4. sparse bilinear feature aggregation and image/kinematics compatibility
   scoring;
5. stable score3+nms9 selection; and
6. construction of the four compliant A0 outputs (`abs`, `inc`, local visual
   logits, candidate IDs), with full-K hidden.

There is no dense BEV, object/map head, recurrent `prev_bev`, past trajectory,
goal, command, or status input. Image decode/normalization is outside the
model-forward timing, consistently with the existing latency protocol.

## Real-data fixture and projection

- Annotation: `etri_val38_goal.pkl`, scenario `20260112-134847`, frame 30
- Annotation SHA256: `a622f0bb3098a55b07823b99ba2f7b872927a8e4fca675d7afe72dd9fb3091e9`
- A0 deploy SHA256: `154da8bcfd05904e5859ba32a27e3f282378a7bbf991d80af6ded5d3ed5eb1b0`
- Fixture SHA256: `0c03eaf2067dbe1dc36c3b1e75ce9b48551b29a4a5afa5520aa765fb6ebe5264`
- Geometry: stored undistorted intrinsic, 1920×1080 camera-specific crop,
  followed by 0.4 scale to the 768×432 cache
- Waypoints visible in at least one camera/height: **93.14%**
- Visible fraction over all camera/height rays: **14.89%** (expected because
  forward paths lie primarily in the front camera)

This is not an arbitrary UV benchmark: both images and calibration are from a
real sample, and projection mirrors the deployed `CachedImageGeometry` path.

## RTX 3090 results

PyTorch 2.7.1+cu128, 20 warm-up iterations, 60 measured repetitions,
`torch.cuda.synchronize()` immediately before and after every measurement.
Inputs were already resident on GPU, and every total measurement invokes the
entire model exactly once.

| precision | median | p90 | p95 | p99 | std | peak alloc / reserved | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| AMP FP16 | **13.003 ms** | 13.013 | 13.025 | 13.107 | 0.029 | 266.6 / 452 MiB | VERY GOOD |
| FP32 | **18.923 ms** | 18.938 | 18.939 | 19.103 | 0.044 | 362.4 / 442 MiB | VERY GOOD |

Non-additive isolated medians:

| component | AMP FP16 | FP32 |
|---|---:|---:|
| backbone + FPN | 9.463 ms | 15.097 ms |
| real projection + sparse sampling + scorer | 1.365 ms | 1.634 ms |
| stable shortlist + complete-candidate API | 2.045 ms | 2.089 ms |

The old seven-forward dense VAD measured about 586.6 ms on the same RTX 3090.
The AMP skeleton is approximately **45× faster**. Its latency multiplier is
exactly **1.0×** because total forward time is below 100 ms.

## Size and compute

- Trainable parameters: **21,744,320**
- Hook-counted convolution + linear compute: **306.50 GFLOPs** using
  2 FLOPs/MAC
- The estimate excludes grid sampling, projection, stable sort, and elementwise
  operations; their actual elapsed time is already included in the 13.003 ms
  end-to-end forward measurement.
- Even the partial analytic count is about 23× below the 7,053 GFLOP cutoff.

## Structural gates

- C1 goal-free signature: PASS (`forward(images, lidar2img)` only)
- C3 exact candidate row and immutable first-six prefix: PASS
- C4 exact-zero input produces all-zero logits; stable first row is candidate
  zero: PASS
- Output shapes: abs/inc `[1,12,10,2]`, logits/IDs `[1,12]`: PASS
- Stable score3+nms9 matches the established NumPy implementation for random
  logits and exact ties: PASS
- Full-K scores/candidates absent from the returned dictionary: PASS
- Docker remained stopped; the 3090 measurement used the extracted host Python
  runtime directly.

## Interpretation and next gate

The Phase-D latency hypothesis is validated. Training should proceed on this
current-only model before adding history. ResNet-34/FPN and candidate scoring
may change accuracy after training, but trained weights do not change the
measured tensor shapes or main compute graph.

Recommended order:

1. train the current-only skeleton with the Phase-A loss and A0 outputs;
2. measure dev accuracy and recalibrate the external selector;
3. add one low-resolution `t=-0.5 s` branch only if motion accuracy needs it;
4. retain a hard 100 ms deployment gate after every structural change.
