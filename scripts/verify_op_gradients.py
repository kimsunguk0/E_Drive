#!/usr/bin/env python
"""Verify the backward pass of SparseDrive's custom deformable_aggregation op.

WHY THIS EXISTS
---------------
Loading the official checkpoint and running eval validates the *forward* path
only. The gradients, the optimizer and the schedule are exercised solely by
training. A custom CUDA op whose backward kernel is subtly wrong is a real bug
class: the loss still decreases (the forward is fine, and a wrong gradient is
often still correlated with the right one), so nothing looks broken until the
final metric comes in low — by which point weeks of GPU time are gone.

We built this op ourselves for sm_90 with a toolchain the upstream authors never
used (torch 2.1.2 / CUDA 12.1 / gcc 11 instead of torch 1.13 / CUDA 11.6), so
"upstream tested it" is not evidence about *our* binary.

WHAT IT DOES
------------
Reimplements deformable_aggregation in pure PyTorch straight from the CUDA
source (ops/src/deformable_aggregation_cuda.cu), lets autograd differentiate the
reference, and compares against the custom op:

    forward output
    grad w.r.t. mc_ms_feat          (bilinear scatter, atomicAdd)
    grad w.r.t. sampling_location   (the chain rule through h_im/w_im)
    grad w.r.t. weights

A correct float32 op agrees with a float64 reference to ~1e-6 relative error.
A wrong one is off by O(1) — there is no ambiguous middle ground.

    python scripts/verify_op_gradients.py

Exits non-zero if any check fails. Deliberately tiny (< 100 MB VRAM, ~seconds)
so it is safe to run on a shared GPU alongside training.
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/pm97/workspace/sukim/adcl/src/SparseDrive")
from projects.mmdet3d_plugin.ops import DeformableAggregationFunction as DAF  # noqa: E402


def reference(mc_ms_feat, spatial_shape, scale_start_index, sampling_location,
              weights, num_groups):
    """Pure-PyTorch deformable_aggregation, transcribed from the .cu file.

    Shapes (as implied by the kernel's index arithmetic):
        mc_ms_feat        [bs, num_feat, num_embeds]
        spatial_shape     [num_cams, num_scale, 2]   (h, w)
        scale_start_index [num_cams, num_scale]
        sampling_location [bs, num_anchors, num_pts, num_cams, 2]  (loc_w, loc_h)
        weights           [bs, num_anchors, num_pts, num_cams, num_scale, num_groups]
        ->  output        [bs, num_anchors, num_embeds]
    """
    bs, num_feat, num_embeds = mc_ms_feat.shape
    num_cams, num_scale, _ = spatial_shape.shape
    _, num_anchors, num_pts, _, _ = sampling_location.shape
    ch_per_group = num_embeds // num_groups

    loc_w = sampling_location[..., 0]
    loc_h = sampling_location[..., 1]
    # The kernel returns early unless BOTH coords are strictly inside (0, 1).
    valid = (loc_w > 0) & (loc_w < 1) & (loc_h > 0) & (loc_h < 1)

    out = mc_ms_feat.new_zeros(bs, num_anchors, num_embeds)
    for cam in range(num_cams):
        for scale in range(num_scale):
            h = int(spatial_shape[cam, scale, 0])
            w = int(spatial_shape[cam, scale, 1])
            start = int(scale_start_index[cam, scale])

            # [bs, num_embeds, h, w]
            fmap = (mc_ms_feat[:, start:start + h * w, :]
                    .reshape(bs, h, w, num_embeds)
                    .permute(0, 3, 1, 2))

            # h_im = loc_h*h - 0.5 / w_im = loc_w*w - 0.5 is exactly
            # grid_sample's align_corners=False mapping for grid = 2*loc - 1,
            # and the kernel's out-of-range guards are zero padding.
            gw = 2.0 * loc_w[:, :, :, cam] - 1.0
            gh = 2.0 * loc_h[:, :, :, cam] - 1.0
            grid = torch.stack([gw, gh], dim=-1).reshape(bs, num_anchors, num_pts, 2)

            sampled = F.grid_sample(fmap, grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)
            # [bs, num_anchors, num_pts, num_embeds]
            sampled = sampled.permute(0, 2, 3, 1)

            m = valid[:, :, :, cam].unsqueeze(-1).to(sampled.dtype)
            wt = weights[:, :, :, cam, scale, :]                    # [bs,a,p,g]
            wt = wt.repeat_interleave(ch_per_group, dim=-1)          # [bs,a,p,e]
            out = out + (sampled * wt * m).sum(dim=2)
    return out


def build_case(device, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    bs, num_cams, num_scale, num_embeds, num_groups = 2, 2, 2, 8, 2
    num_anchors, num_pts = 3, 4
    shapes = [[(5, 7), (3, 4)], [(6, 5), (2, 3)]]  # [cam][scale]

    spatial_shape = torch.tensor(shapes, dtype=torch.int32, device=device)
    starts, acc = [], 0
    for cam in range(num_cams):
        row = []
        for scale in range(num_scale):
            row.append(acc)
            acc += shapes[cam][scale][0] * shapes[cam][scale][1]
        starts.append(row)
    scale_start_index = torch.tensor(starts, dtype=torch.int32, device=device)
    num_feat = acc

    feat = torch.randn(bs, num_feat, num_embeds, generator=g).to(device=device, dtype=dtype)
    # Spread locations over [-0.2, 1.2] so both the interior path and the
    # early-return guard get exercised, including points right at the border.
    loc = (torch.rand(bs, num_anchors, num_pts, num_cams, 2, generator=g) * 1.4 - 0.2
           ).to(device=device, dtype=dtype)
    wts = torch.randn(bs, num_anchors, num_pts, num_cams, num_scale, num_groups,
                      generator=g).to(device=device, dtype=dtype)
    return feat, spatial_shape, scale_start_index, loc, wts, num_groups


def relerr(a, b):
    denom = b.abs().max().clamp_min(1e-12)
    return ((a - b).abs().max() / denom).item()


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable"); return 1
    dev = "cuda"
    torch.manual_seed(0)

    # --- custom op, float32 (the op casts everything to float internally) ----
    f32 = build_case(dev, torch.float32)
    feat, ss, ssi, loc, wts, ng = f32
    feat32 = feat.clone().requires_grad_(True)
    loc32 = loc.clone().requires_grad_(True)
    wts32 = wts.clone().requires_grad_(True)
    out_op = DAF.apply(feat32, ss, ssi, loc32, wts32)

    # A fixed non-uniform upstream gradient: a uniform one can mask index bugs
    # that permute contributions between positions.
    gcpu = torch.Generator(device="cpu").manual_seed(7)
    gout = torch.randn(*out_op.shape, generator=gcpu).to(dev)
    out_op.backward(gout)

    # --- reference, float64 -------------------------------------------------
    feat64 = feat.double().clone().requires_grad_(True)
    loc64 = loc.double().clone().requires_grad_(True)
    wts64 = wts.double().clone().requires_grad_(True)
    out_ref = reference(feat64, ss, ssi, loc64, wts64, ng)
    out_ref.backward(gout.double())

    checks = [
        ("forward output          ", out_op.detach().double(), out_ref.detach()),
        ("grad mc_ms_feat         ", feat32.grad.double(), feat64.grad),
        ("grad sampling_location  ", loc32.grad.double(), loc64.grad),
        ("grad weights            ", wts32.grad.double(), wts64.grad),
    ]

    print("=" * 68)
    print(" deformable_aggregation: custom CUDA op  vs  float64 PyTorch 레퍼런스")
    print("=" * 68)
    TOL = 1e-5
    bad = 0
    for name, a, b in checks:
        e = relerr(a, b)
        ok = e < TOL
        bad += (not ok)
        print(f"  {name} 상대오차 {e:9.3e}   {'OK' if ok else '*** FAIL ***'}")
        if not ok:
            print(f"      op  범위 [{a.min():.6g}, {a.max():.6g}]")
            print(f"      ref 범위 [{b.min():.6g}, {b.max():.6g}]")

    # Second seed: a single lucky configuration proves little.
    print("-" * 68)
    for seed in (1, 2, 3):
        feat, ss, ssi, loc, wts, ng = build_case(dev, torch.float32, seed=seed)
        a1 = feat.clone().requires_grad_(True)
        a2 = loc.clone().requires_grad_(True)
        a3 = wts.clone().requires_grad_(True)
        o = DAF.apply(a1, ss, ssi, a2, a3)
        g2 = torch.randn(*o.shape, generator=torch.Generator("cpu").manual_seed(seed)).to(dev)
        o.backward(g2)
        b1 = feat.double().clone().requires_grad_(True)
        b2 = loc.double().clone().requires_grad_(True)
        b3 = wts.double().clone().requires_grad_(True)
        r = reference(b1, ss, ssi, b2, b3, ng)
        r.backward(g2.double())
        es = [relerr(o.detach().double(), r.detach()),
              relerr(a1.grad.double(), b1.grad),
              relerr(a2.grad.double(), b2.grad),
              relerr(a3.grad.double(), b3.grad)]
        worst = max(es)
        bad += (worst >= TOL)
        print(f"  seed {seed}: 최대 상대오차 {worst:9.3e}   "
              f"{'OK' if worst < TOL else '*** FAIL ***'}")

    print("=" * 68)
    if bad:
        print(f"결과: {bad} 개 실패 — backward 커널이 틀렸습니다")
        return 1
    print("결과: 전부 통과 — forward/backward 모두 정확")
    return 0


if __name__ == "__main__":
    sys.exit(main())
