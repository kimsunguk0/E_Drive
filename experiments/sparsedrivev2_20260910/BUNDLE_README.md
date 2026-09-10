# Primary SparseDriveV2 + relative CE inference artifact

This package contains the chosen primary image model, fixed bank, and final relative score head. Its output is one complete bank row: six cumulative rear-axle ego XY points in metres at 0.5 through 3.0 seconds. Provided causal state and goal affect candidate selection; coordinates are never refined or constructed from them.

Original checkpoint identities:

- Image model, terminal 2000: `bfe29df9f510c3ff76f36f41688a56faaadcc5c07071063504f4e244fcb29aa1`.
- CE head, terminal 2000: `5ec5bc6286610d35f7239c0f0cb41caee3c5fd1cc0b033cb384c04bbb3aed374`.
- Train-only P1024/V1024 bank: `4aff9b40696f91383bfbec377f388e6209d619fa509e51a2019b19af977b5e04`.

The two packaged checkpoints remove only optimizer state. Their export receipts bind original/exported SHA, exact state tensor fingerprints, and unchanged remaining metadata. `artifact_manifest.json` records all package file hashes and the new export SHA values. Training inputs, ego caches, feature caches and original checkpoint files are not required at runtime. `artifacts/cache_manifest.json` is small provenance metadata; no associated NPY/GT files are opened.

Use the existing verified server environment from this package directory:

```bash
CUDA_VISIBLE_DEVICES='' /NHNHOME/data/sukim/adcl/env/venv/bin/python predict_one.py --device cpu --verify-only
CUDA_VISIBLE_DEVICES=0 /NHNHOME/data/sukim/adcl/env/venv/bin/python predict_one.py --device cuda:0 --clip /path/to/raw_clip --output /path/to/prediction.json
```

Use an allocated GPU. `prediction.json` contains `trajectory` with shape 6 by 2, the selected bank row ID and provenance. This utility does not produce or publish a challenge submission. Do not apply `cumsum` to its output. For a multi-clip service, construct `FrozenBankDriver` once and reuse `predict_clip`; per-process CLI loading/compilation is not steady-state inference latency.

Each raw clip directory contains `calibration.parquet`, `ego_pose.parquet`, and current JPEGs `camera_front_left/frame_0.jpg`, `camera_front/frame_0.jpg`, `camera_front_right/frame_0.jpg`. Calibration has the official six camera rows; pose records are -30 through 0 and +50. The adapter uses the -10..0 poses at nominal 10 Hz for causal state, and the provided +50 XYZ transformed by current pose for goal features. It performs undistortion/crop/768 resize, preserves the training JPEG Q95 roundtrip, then RGB/ImageNet 512 by 256 normalization. Three independent camera workers reuse calibration maps. OpenCV one-thread mode is set before library import for this process.

The selected model achieved original tune1998 D3 **0.1305221511 at B=1** and 0.1307486145 at B=8. B1 vs B8 changed 227 selected IDs and 213 XY rows under BF16, so the B1 result is the relevant deployment check. Cached-head vs live B8 matched every row/XY/ID exactly. This is repeated-use tune performance, not hidden-test or unseen confirmation performance.

B200 B1, 50 distinct raw tune inputs: model-only mean 16.951 ms/p95 17.147 ms; memory-resident raw JPEGs through CPU XY result mean 58.028 ms/p95 59.320 ms. The latter includes warm calibrated image processing, pose fit, H2D, native model and exact-bank checks. It excludes filesystem/parquet/tar reads, cold maps, checkpoint loading and extension compilation. These are B200 results, not RTX 4090 measurements.

Torch registered operations were 137.65641728 GFLOPs; adding the documented all-camera native DFA upper bound of 38.191104 G once gives 175.84752128 G. MAC/FMA counts as two. This registered-counter-plus-native-bound figure is separate from a final official `__flops__` integration and excludes unregistered scalar elementwise operations. No projection or interpolation subset is double counted.

`requirements.txt` records direct runtime versions; `environment_packages.json` records the full verified environment. The original environment uses Python 3.10.20, Torch 2.7.1+cu128 and torchvision 0.22.1+cu128. CUDA compilation additionally requires a compatible toolkit/C++ compiler; the package includes the original Apache-2.0 CUDA/C++ source and license. The loader emits the audited current-CUDA-stream launch correction into its build directory. Development testing used B200 sm100. Rebuild and validate for RTX 4090 sm89; no sm100-only binary is shipped as portable.

To reproduce the export, run the captured `build_deployment_bundle.py` recipe from its original worktree path recorded in the manifest, with a new `--output` directory. It checks all selected original artifact hashes, removes optimizer only, reloads and fingerprints every state tensor, copies the pinned source files and writes this package manifest. Final challenge wrapper integration, target-GPU memory/timing and hidden-test performance remain unverified.
