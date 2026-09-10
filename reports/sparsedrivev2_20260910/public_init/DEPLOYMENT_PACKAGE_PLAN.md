# SparseDriveV2 ETRI raw deployment and measurement boundary

Prepared 2026-09-10. Read-only train/tune validation, followed by an explicitly assigned bounded GPU1 measurement; no external submission, hidden test evaluation or edits to running training sources were performed for this task. The callable raw adapter and strict checkpoint loader are implemented. The parent selected the dense2000_none terminal checkpoint for the bounded measurement; a standalone submission CLI/bundle and final target-device latency remain outstanding.

## Implemented callable entry

`experiments/sparsedrivev2_20260910/deployment.py` provides:

```python
from experiments.sparsedrivev2_20260910.deployment import FrozenBankDriver

driver = FrozenBankDriver.from_training_checkpoint(
    selected_checkpoint,
    expected_sha256=selected_checkpoint_sha256,
    bank_path=selected_bank_npz,
    device="cuda:0", precision="bf16", backend="native",
)
trajectory = driver.predict_clip(raw_clip_directory)  # float32 numpy [6, 2]
```

The directory contains official `calibration.parquet`, `ego_pose.parquet`, and `camera_front_left/frame_0.jpg`, `camera_front/frame_0.jpg`, `camera_front_right/frame_0.jpg`. `predict_records(calibration_records, pose_records, image_loader)` supports the same contract without a filesystem adapter. The output is cumulative current-ego XY in metres at 0.5, 1, 1.5, 2, 2.5 and 3 seconds. Do **not** apply another cumulative sum.

The loader checks an explicitly selected checkpoint SHA on the same open file handle used by `torch.load(..., weights_only=True)`, its manifest, training step, score/input modes, bank SHA, frozen public-model/goal-wrapper source hashes, and every embedded bank buffer, then uses `load_state_dict(strict=True)`. It preserves public initialization provenance `330072b133981f77b0b18b669216ad50b211552a82ea45ac52d507ec9021a735`; passing `public_checkpoint=` additionally rehashes that public file. Without that optional file the recorded provenance is verified against the known digest, not independently re-derived by replaying training. Runtime uses trained weights directly and does not need the public 536 MB checkpoint.

All candidate coordinates and the selected trajectory are checked against the embedded fixed bank. The returned six points are an exact bank row; status/goal influence candidate filtering and scores only. These checks, including their synchronization cost, are part of the current callable entry and must be included when timing it.

## Raw input and cache parity

- Cameras: current front-left/front/front-right, RGB, ImageNet mean/std, final 512 by 256.
- Geometry: raw calibration Euler `xyz` degrees and camera-to-ego translation metres; full inverse camera-to-ego transform. Official physical ego origin is the rear axle centre, x forward, y left, z up (official briefing slide 33).
- Pixels: raw JPEG decode using OpenCV; calibration undistortion; 1920 by 1080 crop (front bottom, side cameras top); OpenCV INTER_AREA to 768 by 432; **JPEG quality 95 encode/PIL RGB decode**; PIL BILINEAR to 512 by 256; normalization.
- Projection: existing `models/motiondrive_v2_inputs.py` calibrated/cropped 768 by 432 matrix, then row 0 times 512/768 and row 1 times 256/432. The frozen matrix precision/order is preserved.
- Status: official pose records -10 through 0 only, nominal 10 Hz, current full SE3 origin. Least squares uses `[1, t, .5*t*t]` with a free intercept. Its XY first and second coefficients give `[vx,vy,ax,ay]`; 8D input is `[0,0,0,0,vx,vy,ax,ay]`. Residual RMSE threshold 0.25 m matches the training causal overlay. No timestamp field or future trajectory label is read. `zero` mode supplies all-zero status.
- Goal: optional provided +50 XYZ expressed using current pose rotation, `(R0.T @ (p50-p0))[:2]`, supplied to the selector as `goal_xy`. Future Euler angles are ignored. `goal_mode=none` never uses the +50 coordinates; a CPU test sets them to NaN and confirms identical causal inputs. The official pose-table schema still contains the +50 record.
- Geometry remap arrays may be cached by calibration digest. Images, features and pose state are not cached across clips.
- The default adapter uses a persistent three-worker camera pool. Image loading remains ordered on the caller thread; each worker owns its image buffers and only reads shared remap arrays. A lock protects geometry-cache replacement, and each request retains its own immutable geometry tuple. `camera_workers=1` provides the sequential reference; `close()`/context-manager cleanup releases workers. No runtime `cv2.setNumThreads` call is made.

`test_deployment_cpu.py` validates four train and four tune rows against the frozen `PlanDataset`. Raw 1920 by 1536 JPEGs are read from the original `train/<scene>.tar`; the `/tmp/pm97/.../train` image paths are cache symlinks and were correctly rejected as raw images by the adapter's size check.

Results in `public_init/deployment_cpu.json`: all 8 normalized image tensors, status vectors, goals and projection matrices are exactly equal; 24/24 re-encoded JPEG files have the same SHA as archived cache JPEGs. The disk interface, repeated input, disabled-goal path and pilot500 strict-load/CPU forward passed. Pilot500 selected candidate ID 216867 with an exact bank row. The optional-goal two-step canary strict load also passed (`deployment_goal_load_cpu.json`); this is a loader check, not a competitive score claim.

Skipping the Q95 roundtrip changes RGB pixels at 512 by 256: mean absolute difference over cameras 0.71459/255, per-camera mean range 0.46480–0.91459/255, maximum 23/255; mean PSNR 47.277 dB, minimum 45.452 dB. Preserve the roundtrip now. Any removal or serving-library change requires full validation parity/accuracy testing first.

An earlier serial adapter's sampled raw preprocessing CPU times were 87.9–172.6 ms with four Torch/OpenCV threads. Its single raw CPU grid-backend forward was 4.115 s; these older measurements prove functionality and identify the preprocessing bottleneck, not final device latency.

The optimized adapter was measured for 32 interleaved iterations per configuration (`raw_preprocessing_parallel_cpu.json`). JPEG bytes were already in memory; parquet/tar I/O and the model were excluded. At process startup the profile set `OPENCV_FOR_THREADS_NUM=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=4`; Torch used four threads, with no OpenCV global setter. All eight train/tune inputs and camera provenance were exactly equal to sequential and cache references. Eight concurrent requests on one adapter also passed exact equality.

| CPU raw preparation | Mean ms | p50 ms | p95 ms | Average CPU cores |
|---|---:|---:|---:|---:|
| Sequential, warm maps | 117.01 | 116.86 | 121.34 | 1.16 |
| Three camera workers, warm maps | **41.48** | **41.25** | **43.81** | 3.23 |
| Sequential, cold maps | 212.00 | 209.69 | 222.82 | 1.08 |
| Three camera workers, cold maps | 137.56 | 135.45 | 146.69 | 1.69 |

Warm includes pixels, pose fit, normalization and input validation; cold additionally includes adapter/pool creation and all six calibration-map constructions, with cleanup excluded. The process had affinity to 72 CPUs and host load average changed from 3.01 to 3.66. This achieves the requested sub-50 ms warm preprocessing diagnostic on the B200 host, while cold-map cost remains material. Pin the OpenCV process-start configuration in the deployment launcher and remeasure under actual concurrent host load. These are not RTX 4090 or final judge latency claims.

## Official interface and measurement evidence

Local official repository: `/NHNHOME/data/sukim/adcl/src/etri_vad/ETRI_E2E_Driving_Challenge`.

- `tools/data_converter/etri_test_converter.py` defines the official relative frame/pose/calibration contracts and creates VAD metadata. Its history uses -30 through 0 at five-frame stride and DT=0.1.
- `tools/etri_test_submit.py` is a **VAD-specific** config/checkpoint/ann-file/out runner, with stream reset per clip and command-mode selection followed by cumulative sum of VAD displacement outputs. It is not a universal custom-model Python signature. A custom bundle should emit the same clip-key JSON values but must use this model's already cumulative XY directly.
- `tools/measure_flops.py` builds the model on CUDA/eval, runs one warmup input, constructs/collates/scatters the measured input outside `FlopCounterMode`, then counts one forward. The baseline's previous BEV is present after warmup; this SparseDriveV2 variant has no temporal BEV. The tool sums Global operator counts and writes integer `__flops__` to submission JSON. It does not time raw JPEG/parquet preprocessing, model loading or extension compilation.
- The local guideline `ETRI_E2E_Driving_Challenge_Guideline/site/test.html` FLOP section (around lines 843–869) names PyTorch FlopCounterMode and the 7053 GFLOP cutoff. Its final-score section (around line 918) gives `L2 * (1 + max(0, T_infer_ms - 100) / 200)`.
- The inspected public code and guideline do not fully specify whether the final judge's latency boundary includes raw image/parquet decoding or cold calibration-map creation. Until clarified by the actual test harness, report model-only, full warm raw input and full cold raw input separately. Do not infer the final boundary from the FLOP tool's preprocessing exclusion.

The previous pilot500 B200 cached-input benchmark, 27.30 ms mean / 28.77 ms p95 for B=1, includes cached JPEG decode/resize/normalize, H2D and forward; it excludes raw undistortion/re-encoding and is **not** an RTX 4090 latency result. Official briefing describes RTX 4090 and an RTX PRO 6000 alternative for models exceeding 24 GB; target hardware and final timing remain to be checked.

The parent-selected `dense2000_none_s0_v1/last.pth`, SHA `bfe29df9f510c3ff76f36f41688a56faaadcc5c07071063504f4e244fcb29aa1`, was subsequently strictly loaded on assigned B200 GPU1. `profile_deployment_gpu.py` measured 50 distinct tune rows at B=1, native DFA and BF16, after extension/model warmup. All 50 raw inputs were exactly equal to their cached training recipe; every shortlisted and selected trajectory was an exact bank row. GPU1 was released after the test.

| B200 stage, 50 raw tune samples | Mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|
| Warm raw preprocessing, three workers | 40.098 | 39.836 | 41.399 |
| Host to device | 0.335 | 0.334 | 0.347 |
| Model forward only | 16.193 | 16.184 | 16.247 |
| Bank equality checks and output copy | 0.141 | 0.140 | 0.147 |
| Complete measured boundary | **56.769** | **56.496** | **58.205** |

The complete boundary starts with three raw JPEG byte strings in memory and ends with six XY points on CPU. It includes calibrated image processing/Q95/normalization, causal fit, H2D, native model, and exact-bank checks. It excludes parquet/tar/filesystem reads, cold calibration maps, checkpoint loading and extension compilation. Peak CUDA allocated/reserved memory was 1,425,444,352 / 1,545,601,024 bytes in this audit; these are process allocator peaks, not total board memory. See `public_init/deployment_gpu_dense2000_none.json`. No RTX 4090 timing inference follows from this B200 result.

## Native DFA analytic FLOP supplement

`dfa_flops_supplement.py` and `public_init/dfa_flops_supplement.json` document a CPU arithmetic calculation because the PyBind CUDA kernel is invisible to ordinary Torch operator counting. This is not the model's official `__flops__`.

Convention: scalar multiply/add/subtract/divide/exp each count as 1; MAC/FMA counts as 2. Comparisons, index arithmetic, floor, conversion and memory traffic are excluded. Exp is a nominal scalar operation, not an instruction-cost estimate. Per valid camera-point, per feature level, per channel, the source executes 21 arithmetic FLOPs: feature-coordinate conversion 4, bilinear offsets 4, four interpolation coefficients 4, four-neighbour weighted value 7, learned weight and accumulation 2. The last 9 are a subset of 21 and must not be added twice.

| DFA stage | Anchors | Points per anchor | Query points | Native kernel all-camera upper GFLOPs |
|---|---:|---:|---:|---:|
| First path stage | 1024 | 500 | 512000 | 33.030144 |
| Second path stage | 128 | 500 | 64000 | 4.128768 |
| Final trajectory stage | 200 | 80 | 16000 | 1.032192 |
| Total | | | 592000 | **38.191104** |

Each anchor point uses five fixed heights and two learned XY offsets. There are 3 cameras, 4 feature levels and 256 channels. The native arithmetic count is `21 * valid_camera_points * 4 * 256`; the table assumes all three camera projections are valid, so it is a conservative upper bound. The kernel itself checks strict normalized `0 < u,v < 1`; an exact data-dependent count must count that condition, not just the preceding depth/weight mask.

Distinct surrounding Torch operations are listed for coverage inspection: projection matmul 0.056832 G (normally already counted as bmm/mm), projection divisions 0.007104 G scalar ops, stable-softmax nominal 0.227317184 G scalar ops, native-output point reduction 0.151205888 G additions, plus small XY-offset/camera-embedding/residual additions. Linear keypoint/weight/camera/output projections, convolutions, attention and all other model operations belong in the measured Torch counter.

In the current Torch 2.7.1 registry, mm/addmm/bmm/baddbmm/_scaled_mm are registered; the inspected registry has no softmax, grid-sampler or native multi-head-attention entries. Actual dispatch still must be measured: use the same model mode/precision/backend/input as inference and report raw counter output plus a clearly separate conservative native supplement. Do not add projection matmul twice, and do not silently replace missing whole-model operations with this DFA-only supplement. Changing velocity-bank size leaves the three DFA table shapes unchanged at the same filtering counts, but changes other attention/embedding operations.

Actual B200 BF16 dispatch for the selected terminal checkpoint was measured with official-style `FlopCounterMode` and its no-op multi-grad-hook compatibility workaround. Raw registered count was **137.65147648 GFLOPs**: convolution 97.429487616 G, addmm 37.527867392 G, flash attention 2.637234176 G, bmm 0.056887296 G. Thus flash attention was counted and the projection matmul was already included in bmm. Adding the native-DFA all-camera upper bound exactly once gives **175.84258048 GFLOPs**; no projection or interpolation subset was double counted. This is a registered-counter total plus documented native bound, not a claim that every scalar Torch elementwise operation was counted. The audit wrote no submission JSON or `__flops__` field.

## Self-contained package plan and remaining checks

Preserve this file hierarchy so the native-op loader can resolve its source directory:

```text
bundle/
  experiments/sparsedrivev2_20260910/{public_model,goal_selector,deployment}.py
  models/{__init__,motiondrive_v2_inputs,motiondrive_v2_input_contract,motiondrive_v2_temporal_contract}.py
  third_party/SparseDriveV2/LICENSE
  third_party/SparseDriveV2/navsim/agents/sparsedrive/ops/src/
    deformable_aggregation.cpp
    deformable_aggregation_cuda.cu
  artifacts/selected_checkpoint.pth
  artifacts/frozen_bank.npz
  provenance.json
  requirements.lock
  run_submission.py  # still to implement against final judge wrapper
```

No training NPZ/labels, split caches, `PlanDataset`, pose annotations, optimizer or train-only scripts are required during runtime. Current training checkpoint files contain optimizer state; a later reviewed export should retain the trained `model`, source/bank/public provenance and deployment manifest while stripping optimizer state, recording a new artifact SHA. This export and the final JSON CLI are not implemented in this bounded task.

Known working dependencies include Python 3.10.20, Torch 2.7.1+cu128, torchvision 0.22.1+cu128, timm 0.6.13, OpenCV 4.8.1 and Pillow 12.2.0, plus NumPy/SciPy/PyArrow and Ninja/compiler/CUDA toolchain. `public_init/deployment_environment_packages.json` captures all 203 installed package versions for deriving the actual deployment lock. Preserve Apache-2.0 source license and public code pin `696ef77924eb9e0a4b4047d013a50e9854bfa026`. The task-local CUDA copy differs only by explicitly launching on PyTorch's current CUDA stream; record both original and generated source hashes.

The B200 compiled extension targets sm100. A final RTX 4090 bundle must build from source for sm89 (or ship a separately validated compatible binary); do not copy the B200 binary as if portable. Build and warm the extension before steady-state timing. Calibrated raw preprocessing must retain Q95 parity under the deployed OpenCV/Pillow versions.

Remaining before claiming deployable challenge performance: complete any still-running agreed checkpoint comparisons; run strict export/load and full raw-input accuracy parity; validate the FLOP reporting convention for the custom op against the final wrapper; measure memory and B=1 latency on the target GPU under the final harness boundary; then create the clip-key submission JSON with integer `__flops__`. The status/goal design keeps complete fixed candidate rows unchanged and matches the interpretation of the organizers' Q8 candidate-selection exception; this report does not assert a separate official approval of this specific implementation.
