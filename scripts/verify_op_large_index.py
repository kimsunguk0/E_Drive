#!/usr/bin/env python
"""Verify deformable_aggregation in the regime the int64 patch exists for.

WHY THIS EXISTS
---------------
verify_op_gradients.py checks correctness at tiny sizes -- where the ORIGINAL
int32 kernel is also correct. It therefore says nothing about the only situation
the int64 patch was written for: num_kernels past 2^31.

That gap matters now. At samples_per_gpu=64 the map head launches

    num_kernels = bs * num_anchors * num_pts * num_cams * num_scale * num_embeds
                = 64 * 100  * 300 * 6 * 4 * 256   ~= 1.18e10

which is 5.5x INT32_MAX. And since switching to that configuration, grad_norm sits
at ~3.5x the official run's (peaking at 9.9x) even though loss tracks it to ~1.04x.
A kernel that is subtly wrong only at large indices would look exactly like that.

METHOD
------
A float64 PyTorch reference at this scale is not memory-feasible, so instead we
use a self-consistency test that stays inside the already-verified regime:

    output(batch of B)  ==  concat over b of output(batch of 1)

Each single-sample call has num_kernels below INT32_MAX -- i.e. in the range where
verify_op_gradients.py already proved the kernel correct against a float64
reference -- while the full-batch call is past it. The op is per-sample
independent (every index in the kernel is derived from batch_index), so exact
agreement is required, not merely approximate. Gradients are checked the same way.

Runs the same comparison for forward and for all three gradients, and reports
whether the tested configuration actually exceeds INT32_MAX (otherwise the test
proves nothing).

    python scripts/verify_op_large_index.py

Sized to stay under ~1 GB of VRAM.
"""
import sys

import torch

sys.path.insert(0, "/home/pm97/workspace/sukim/adcl/src/SparseDrive")
from projects.mmdet3d_plugin.ops import DeformableAggregationFunction as DAF  # noqa: E402

INT32_MAX = 2 ** 31 - 1


def build(bs, num_anchors, num_pts, cams, scales, embeds, groups, hw=8, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    shapes = [[(hw, hw)] * scales for _ in range(cams)]
    spatial_shape = torch.tensor(shapes, dtype=torch.int32, device="cuda")
    starts, acc = [], 0
    for c in range(cams):
        row = []
        for s in range(scales):
            row.append(acc)
            acc += hw * hw
        starts.append(row)
    scale_start_index = torch.tensor(starts, dtype=torch.int32, device="cuda")

    feat = torch.randn(bs, acc, embeds, generator=g).cuda()
    loc = (torch.rand(bs, num_anchors, num_pts, cams, 2, generator=g) * 1.2 - 0.1).cuda()
    wts = torch.randn(bs, num_anchors, num_pts, cams, scales, groups, generator=g).cuda()
    return feat, spatial_shape, scale_start_index, loc, wts


def run(feat, ss, ssi, loc, wts, gout):
    a = feat.clone().requires_grad_(True)
    b = loc.clone().requires_grad_(True)
    c = wts.clone().requires_grad_(True)
    out = DAF.apply(a, ss, ssi, b, c)
    out.backward(gout)
    return out.detach(), a.grad, b.grad, c.grad


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable"); return 1

    # Chosen so the full batch exceeds INT32_MAX while each single sample does not.
    BS, A, P, C, S, E, G = 4, 100, 1024, 6, 4, 256, 1
    full = BS * A * P * C * S * E
    one = full // BS
    print("=" * 70)
    print(" deformable_aggregation: 대규모 인덱스 자기일관성 검증")
    print("=" * 70)
    print(f"  구성: bs={BS} anchors={A} pts={P} cams={C} scales={S} embeds={E}")
    print(f"  num_kernels (전체 배치): {full:,}  = INT32_MAX x {full/INT32_MAX:.2f}")
    print(f"  num_kernels (1 샘플)   : {one:,}  = INT32_MAX x {one/INT32_MAX:.2f}")
    if full <= INT32_MAX:
        print("  *** 이 구성은 INT32_MAX 를 넘지 않아 검증 의미가 없음 ***")
        return 1
    if one > INT32_MAX:
        print("  *** 1 샘플도 INT32_MAX 를 넘어 기준(reference)이 될 수 없음 ***")
        return 1
    print(f"  참고: 실제 bs=64 map head 는 약 1.18e10 = INT32_MAX x 5.5")

    feat, ss, ssi, loc, wts = build(BS, A, P, C, S, E, G)
    gcpu = torch.Generator(device="cpu").manual_seed(7)
    gout = torch.randn(BS, A, E, generator=gcpu).cuda()

    print(f"\n  VRAM 사용: {torch.cuda.memory_allocated()/2**20:.0f} MiB")

    o_full, gf_full, gl_full, gw_full = run(feat, ss, ssi, loc, wts, gout)

    # per-sample reference, each below INT32_MAX
    o_c, gf_c, gl_c, gw_c = [], [], [], []
    for b in range(BS):
        o, gf, gl, gw = run(feat[b:b+1], ss, ssi, loc[b:b+1], wts[b:b+1], gout[b:b+1])
        o_c.append(o); gf_c.append(gf); gl_c.append(gl); gw_c.append(gw)
    o_c = torch.cat(o_c); gf_c = torch.cat(gf_c)
    gl_c = torch.cat(gl_c); gw_c = torch.cat(gw_c)

    def rel(a, b):
        d = b.abs().max().clamp_min(1e-12)
        return ((a - b).abs().max() / d).item()

    checks = [
        ("forward output        ", o_full, o_c),
        ("grad mc_ms_feat       ", gf_full, gf_c),
        ("grad sampling_location", gl_full, gl_c),
        ("grad weights          ", gw_full, gw_c),
    ]
    print()
    TOL = 1e-6      # per-sample independence means this should be ~exact
    bad = 0
    for name, a, b in checks:
        e = rel(a, b)
        ok = e < TOL
        bad += (not ok)
        print(f"  {name}  상대오차 {e:9.3e}   {'OK' if ok else '*** FAIL ***'}")
        if not ok:
            nz = (a - b).abs()
            print(f"      불일치 원소 {int((nz > 0).sum()):,} / {a.numel():,}")
            print(f"      전체배치 범위 [{a.min():.6g}, {a.max():.6g}]")
            print(f"      샘플별   범위 [{b.min():.6g}, {b.max():.6g}]")

    print("=" * 70)
    if bad:
        print(f"결과: {bad} 개 실패 — 대규모 인덱스에서 커널이 틀렸습니다")
        return 1
    print("결과: 전부 통과 — INT32_MAX 초과 영역에서도 정확")
    return 0


if __name__ == "__main__":
    sys.exit(main())
