# SparseDriveV2 public initialization and ETRI adapter

The public model is loaded and runnable. This is an implementation/initialization
result; no competitive ETRI accuracy is established by these smoke checks.

## Provenance

- Code: <https://github.com/swc-17/SparseDriveV2>, commit
  `696ef77924eb9e0a4b4047d013a50e9854bfa026`.
- Checkpoint: `wenchaosun/SparseDriveV2/sparsedrive_navsimv1_92p2.ckpt`,
  Hugging Face revision `42c15531aa6cb18c7360dd0377cfc1c221280597`.
- Downloaded bytes: 536,373,288. SHA256:
  `330072b133981f77b0b18b669216ad50b211552a82ea45ac52d507ec9021a735`.
- `torch.load(..., weights_only=True)` succeeds. All 400 checkpoint tensors,
  50,370,143 elements, load with the original bank. All 41,808,827 learnable
  parameter elements are reused. Frozen bank tensors account for most of the
  difference; BatchNorm running statistics are also preserved.
- Source license is retained at `third_party/SparseDriveV2/LICENSE` (Apache 2.0).
  `public_model.py` is an explicitly modified architectural adaptation.
- Runtime: `/NHNHOME/data/sukim/adcl/env/venv/bin/python`, torch 2.7.1+cu128,
  torchvision 0.22.1+cu128, timm 0.6.13. No global package installation or downgrade.

## Preserved algorithm

The adapter retains the ResNet34+FPN, 8D status encoder, path and velocity positional
MLPs, both decoder layers, path candidate deformable feature aggregation, velocity
image attention, top-k factorized filtering, full trajectory reconditioning DFA,
imitation scorer, and every public NAVSIM metric head. Parameter names and shapes
match the checkpoint. The path bank is `[P,50,3]` (XY and heading); velocity has 8
half-second values. Internal trajectories contain 8 XY+heading poses, preserving
the trajectory DFA offset projection shape. ETRI outputs the first six XY poses.

The native CUDA aggregation source is reused. Its two launches are changed in
task-local copies to specify `at::cuda::getCurrentCUDAStream()`. Original pinned
source files remain unchanged. The extension builds in the vendor's
`.etri_extension/torch_<version>` folder and does not install into site-packages.
The Python reference uses bilinear `grid_sample(align_corners=False)` and reproduces
the native strict `(0,1)` sample bounds. CUDA/reference outputs and gradients are
checked, including execution on a non-default stream.

The original training-only GridMask augmentation is disabled; the dataset owns
augmentation. MultiheadAttention uses the equivalent `need_weights=False` path.
NAVSIM PDM label/scorer code is excluded from model forward; ETRI losses belong
outside forward.

## ETRI contract

- Current RGB images only, camera order front-left/front/front-right, selected
  from ETRI six-camera indices `[2,0,1]`. No image history is synthesized.
- Size 512x256 (width,height), normalized RGB mean `[.485,.456,.406]` and standard
  deviation `[.229,.224,.225]`. This matches the public RGB statistics. Root's
  cached 768x432 images are resized and each projection matrix's first two rows
  scaled by the corresponding width/height factors.
- `lidar2img[B,3,4,4]` maps current ego XYZ metres to resized pixel homogeneous
  coordinates. `image_hw` is `[height,width]`, `[B,2]`, or `[B,3,2]`.
- Default status is eight zeros. Optional status has command[4], velocity[2],
  acceleration[2]. Root's first causal-selection arm keeps command slots zero
  and uses independently hash-verified causal pose-derived state.
- Status can affect attention/scoring and bank indices. It never changes candidate
  coordinates. Candidates and final trajectory are exact frozen `traj_vocab`
  rows; there is no coordinate residual, interpolation, shift, scale, or rotation
  at inference.
- ETRI's primary bank uses 50 points at 2m spacing to cover 100m while preserving
  every learned tensor shape. DFA consumes actual metric coordinates, so it has
  no hard-coded 1m spacing assumption. The positional MLP nevertheless sees a
  different coordinate distribution than the public 50m bank; weight compatibility
  alone does not imply accuracy.
- Four original bank tensors are intentionally replaced when `bank_path` is
  supplied. Every remaining tensor must match; unexpected missing/shape-mismatched
  keys cause a hard error. Bank masks restrict final selection to valid six-point
  complete trajectories. Velocity pruning prefers candidates valid for at least
  one retained path. The bank should include a valid stopping candidate.
- Default ETRI score is the pretrained **imitation head logit**. The original
  NAVSIM-v1 product of predicted metrics is exposed separately as `navsim_scores`
  or selectable with `score_mode='navsim_v1'`. Public PDMS=92.22 is not an ETRI D3
  result and should not be treated as one.

## Usage

```python
model, coverage = PublicSparseDriveV2.from_public_checkpoint(
    'checkpoints/sparsedrivev2_20260910/public/sparsedrive_navsimv1_92p2.ckpt',
    bank_path='cache/sparsedrivev2_20260910/bank/p256_v128_native100m_v8.npz',
    backend='native', score_mode='imitation')
out = model(images=images, lidar2img=projection, image_hw=image_hw, status=status)
```

Output includes `trajectory[B,6,2]`, `candidate_xy[B,K,6,2]`, `scores[B,K]`,
`candidate_valid[B,K]`, `candidate_ids[B,K]`, `selected_candidate_id[B]`,
`imitation_scores`, `navsim_scores`, `metric_logits`, and a two-element `coarse`
list. Each coarse entry contains `path_scores`, `velocity_scores`, `path_ids`,
and `velocity_ids` **before** its stage's pruning. IDs index the global immutable
bank; full row ID is `path_id * V + velocity_id`. External D3 loss should respect
`candidate_valid` when creating soft targets.

Run provenance/coverage and synthetic numerical checks from the worktree root:

```bash
CUDA_VISIBLE_DEVICES=1 TORCH_CUDA_ARCH_LIST=10.0 \
/NHNHOME/data/sukim/adcl/env/venv/bin/python -m \
experiments.sparsedrivev2_20260910.bootstrap_public --backward
```

## Verification

`bootstrap.json`: CPU complete-decoder forward with reduced bank, native vs
reference CUDA numerical/gradient parity, full public-bank 512x256 GPU forward
and backward, finite gradients, and exact bank identity under varying status.
One warmed B200 B1 model-only forward took 0.01596s, forward+backward 0.2784s,
peak allocated memory 1.96GB. These are synthetic local smoke timings and exclude
image decode, transfer, deployment packaging, and final challenge hardware.

`bf16_smoke.json`: B2 with BF16 autocast forward/backward passes; scores are BF16,
candidate/output coordinates remain FP32, all 264 exercised gradient tensors
are finite, and selected candidates exactly match frozen bank rows.

`public_coverage.json` and `checkpoint_inventory.json`: complete public key/shape
inventory and reused-key accounting. ETRI bank replacement reports are separate.

The full NAVSIM Python environment, PDM simulation, and datasets are not required
to execute this adapter. The pinned CUDA source and license must remain available
when reproducing or packaging its native backend.
