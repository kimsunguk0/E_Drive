#!/usr/bin/env python
"""Phase 6 gate — verify the restored environment is actually usable for training.

Deliberately strict: every CUDA op is executed on a real cuda tensor, because a
successful CPU-only `import mmcv` says nothing about whether the compiled
extension carries kernels for this GPU's arch. A wheel built with a narrowed
TORCH_CUDA_ARCH_LIST imports fine and then fails at the first forward.

Exit 0 = all required checks passed. Exit 1 = at least one required check failed.
Optional checks (SparseDrive op, flash-attn) report WARN and do not fail the run.

VRAM budget: every allocation here is a few MB. The GPU is shared with another
user's training job, so nothing in this file may allocate at scale.
"""
import argparse
import os
import re
import sys
import traceback

EXPECT_TORCH = "2.1.2"
EXPECT_TV = "0.16.2"
EXPECT_MMCV = "1.7.2"
EXPECT_MMDET = "2.28.2"
EXPECT_NUMPY = "1.23.5"
EXPECT_CAP = (9, 0)          # H200 NVL; override with --allow-any-cap elsewhere
EXPECT_ARCHS = {"86", "89", "90"}

results = []


def check(name, required=True):
    """Decorator: run fn, record PASS/FAIL/WARN, never let an exception escape."""
    def deco(fn):
        try:
            detail = fn()
            results.append((("PASS", name, detail or "")))
        except Exception as e:
            status = "FAIL" if required else "WARN"
            tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
            results.append((status, name, f"{type(e).__name__}: {e} | {tb}"))
        return fn
    return deco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-any-cap", action="store_true",
                    help="skip the sm_90 assert (use when verifying on 3090/4090)")
    ap.add_argument("--skip-gpu", action="store_true",
                    help="version/import checks only; no CUDA kernels launched")
    args = ap.parse_args()

    # ---------- versions ----------
    @check("numpy == %s" % EXPECT_NUMPY)
    def _numpy():
        import numpy as np
        assert np.__version__ == EXPECT_NUMPY, (
            f"numpy is {np.__version__}, expected {EXPECT_NUMPY}. "
            "numpy 2.x silently breaks mmcv/mmdet ABI.")
        return np.__version__

    @check("torch == %s (+cu121)" % EXPECT_TORCH)
    def _torch():
        import torch
        assert torch.__version__.startswith(EXPECT_TORCH), \
            f"torch is {torch.__version__}, expected {EXPECT_TORCH}"
        assert torch.version.cuda == "12.1", \
            f"torch built against CUDA {torch.version.cuda}, expected 12.1"
        return f"{torch.__version__} / cuda {torch.version.cuda}"

    @check("torchvision == %s" % EXPECT_TV)
    def _tv():
        import torchvision
        assert torchvision.__version__.startswith(EXPECT_TV), \
            f"torchvision is {torchvision.__version__}, expected {EXPECT_TV}"
        return torchvision.__version__

    @check("mmcv-full == %s" % EXPECT_MMCV)
    def _mmcv_ver():
        import mmcv
        assert mmcv.__version__ == EXPECT_MMCV, \
            f"mmcv is {mmcv.__version__}, expected {EXPECT_MMCV}"
        return mmcv.__version__

    @check("mmdet == %s" % EXPECT_MMDET)
    def _mmdet_ver():
        import mmdet
        assert mmdet.__version__ == EXPECT_MMDET, \
            f"mmdet is {mmdet.__version__}, expected {EXPECT_MMDET}"
        return mmdet.__version__

    @check("motmetrics tracking-eval path works with installed pandas")
    def _motmetrics():
        # motmetrics 1.1.3 calls Series.iteritems(), which pandas 2.0 removed.
        # A bare `import motmetrics` still succeeds, so the only honest check is
        # to actually compute idf1 — that is the code path nuScenes tracking
        # eval (AMOTA/AMOTP) goes through.
        import pandas as pd
        import motmetrics as mm
        acc = mm.MOTAccumulator(auto_id=True)
        acc.update([1, 2], [1, 2], [[0.1, 0.9], [0.9, 0.1]])
        acc.update([1, 2], [1, 2], [[0.2, 0.8], [0.8, 0.2]])
        mh = mm.metrics.create()
        s = mh.compute(acc, metrics=["num_frames", "mota", "motp", "idf1"], name="t")
        assert float(s["idf1"].iloc[0]) == 1.0, s.to_string()
        return f"pandas {pd.__version__}, motmetrics {mm.__version__}, idf1 computed"

    @check("nuscenes-devkit imports")
    def _devkit():
        import nuscenes
        from nuscenes.eval.common.config import config_factory  # noqa: F401
        return getattr(nuscenes, "__version__", "1.1.10")

    # ---------- GPU availability ----------
    @check("torch.cuda.is_available()")
    def _cuda():
        import torch
        if args.skip_gpu:
            raise RuntimeError("skipped by --skip-gpu")
        assert torch.cuda.is_available(), "CUDA not available"
        return f"{torch.cuda.device_count()} device(s): {torch.cuda.get_device_name(0)}"

    if not args.skip_gpu:
        @check("compute capability == %s" % (EXPECT_CAP,), required=not args.allow_any_cap)
        def _cap():
            import torch
            cap = torch.cuda.get_device_capability()
            if not args.allow_any_cap:
                assert cap == EXPECT_CAP, f"capability {cap}, expected {EXPECT_CAP}"
            return str(cap)

        # ---------- the checks that actually matter ----------
        @check("mmcv _ext.so embeds sm_86/sm_89/sm_90 cubins")
        def _archs():
            # Check the extension WE built, not torch's own wheel. The official
            # torch 2.1.2+cu121 wheel ships sm_50..sm_86 + sm_90 and no sm_89,
            # which is fine for torch (an sm_86 cubin runs on sm_89 by CUDA's
            # minor-version binary-compat rule) but says nothing about our
            # extensions. What actually matters is that mmcv was built with
            # TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX", so assert that directly by
            # reading the cubins out of the .so.
            import glob
            import shutil
            import subprocess

            import mmcv
            so = glob.glob(os.path.join(os.path.dirname(mmcv.__file__), "_ext*.so"))
            assert so, "mmcv/_ext*.so not found — mmcv was built without ops"
            cuobjdump = shutil.which("cuobjdump")
            if not cuobjdump:
                raise RuntimeError("cuobjdump not on PATH (activate the env first)")
            r = subprocess.run([cuobjdump, "--list-elf", so[0]],
                               capture_output=True, text=True, timeout=180)
            # Lines look like "ELF file  1: mmdeform_conv_cuda.sm_86.cubin", so
            # the arch is embedded mid-token — search, don't startswith.
            got = set(re.findall(r"sm_(\d+)", r.stdout))
            missing = EXPECT_ARCHS - got
            assert not missing, (
                f"mmcv _ext.so carries archs {sorted(got)} but is missing "
                f"{sorted(missing)}. 4090 (sm_89) is the organizer's scoring GPU — "
                "rebuild with TORCH_CUDA_ARCH_LIST='8.6;8.9;9.0+PTX'.")
            return "sm_" + ", sm_".join(sorted(got))

        @check("mmcv.ops.modulated_deform_conv2d CUDA forward")
        def _mdc():
            import torch
            from mmcv.ops import modulated_deform_conv2d
            N, Cin, Cout, H, W, K = 1, 4, 4, 8, 8, 3
            x = torch.randn(N, Cin, H, W, device="cuda")
            w = torch.randn(Cout, Cin, K, K, device="cuda")
            b = torch.zeros(Cout, device="cuda")
            off = torch.zeros(N, 2 * K * K, H, W, device="cuda")
            mask = torch.ones(N, K * K, H, W, device="cuda")
            # modulated_deform_conv2d IS ModulatedDeformConv2dFunction.apply, and
            # autograd Function.apply rejects keyword arguments, so every option
            # has to be positional:
            #   (input, offset, mask, weight, bias, stride, padding, dilation,
            #    groups, deform_groups)
            y = modulated_deform_conv2d(x, off, mask, w, b, 1, 1, 1, 1, 1)
            torch.cuda.synchronize()
            assert y.shape == (N, Cout, H, W), y.shape
            assert torch.isfinite(y).all(), "non-finite output"
            return f"out {tuple(y.shape)}"

        @check("mmcv.ops.MultiScaleDeformableAttention CUDA forward")
        def _msda():
            import torch
            from mmcv.ops import MultiScaleDeformableAttention
            embed, heads, levels, points = 32, 4, 2, 4
            m = MultiScaleDeformableAttention(
                embed_dims=embed, num_heads=heads, num_levels=levels,
                num_points=points, batch_first=True).cuda().eval()
            spatial_shapes = torch.tensor([[8, 8], [4, 4]], dtype=torch.long, device="cuda")
            num_keys = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum())
            level_start_index = torch.cat((
                spatial_shapes.new_zeros((1,)),
                (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]))
            B, Q = 1, 6
            query = torch.rand(B, Q, embed, device="cuda")
            value = torch.rand(B, num_keys, embed, device="cuda")
            ref = torch.rand(B, Q, levels, 2, device="cuda")
            with torch.no_grad():
                out = m(query=query, value=value, reference_points=ref,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index)
            torch.cuda.synchronize()
            assert out.shape == (B, Q, embed), out.shape
            assert torch.isfinite(out).all(), "non-finite output"
            return f"out {tuple(out.shape)}"

        @check("SparseDrive deformable_aggregation CUDA forward")
        def _sparsedrive():
            # torch MUST be imported before the ext: the .so links libc10.so and
            # relies on torch having already loaded it into the process.
            import torch
            import deformable_aggregation_ext as dae

            # Shapes taken from the real call site,
            # projects/mmdet3d_plugin/models/blocks.py:145 (DAF(*feature_maps,
            # points_2d, weights)):
            #   mc_ms_feat        [bs, num_feat, num_embeds]      float32
            #   spatial_shape     [num_cams, num_levels, 2]       int32
            #   scale_start_index [num_cams, num_levels]          int32
            #   sampling_location [bs, num_anchor, num_pts, num_cams, 2]
            #   weights   [bs, num_anchor, num_pts, num_cams, num_levels, num_groups]
            #   output            [bs, num_anchor, num_embeds]
            bs, cams, levels = 1, 6, 4
            embeds, groups = 32, 4
            anchors, pts = 10, 13
            hw = [(16, 16), (8, 8), (4, 4), (2, 2)]

            shape = torch.tensor([[list(s) for s in hw]] * cams,
                                 dtype=torch.int32, device="cuda")
            per_cam = [h * w for h, w in hw]
            starts, off = [], 0
            for _ in range(cams):
                row = []
                for n in per_cam:
                    row.append(off)
                    off += n
                starts.append(row)
            start_index = torch.tensor(starts, dtype=torch.int32, device="cuda")
            num_feat = off

            feat = torch.rand(bs, num_feat, embeds, device="cuda")
            # keep sampling locations inside [0,1] so the bilinear sampler is
            # exercised rather than the out-of-bounds early-out
            loc = torch.rand(bs, anchors, pts, cams, 2, device="cuda")
            wts = torch.rand(bs, anchors, pts, cams, levels, groups, device="cuda")

            out = dae.deformable_aggregation_forward(feat, shape, start_index, loc, wts)
            torch.cuda.synchronize()
            assert out.shape == (bs, anchors, embeds), out.shape
            assert torch.isfinite(out).all(), "non-finite output"
            assert out.abs().sum().item() > 0, "output is all zeros — kernel did no work"
            return f"out {tuple(out.shape)}, num_feat={num_feat}"

        @check("flash-attn (optional)", required=False)
        def _fa():
            import flash_attn
            return flash_attn.__version__

    # ---------- report ----------
    width = max(len(n) for _, n, _ in results)
    print("=" * (width + 60))
    print("verify_env.py — %s" % os.environ.get("CONDA_PREFIX", "(no CONDA_PREFIX)"))
    print("=" * (width + 60))
    n_fail = n_warn = 0
    for status, name, detail in results:
        mark = {"PASS": "✓", "FAIL": "✗", "WARN": "!"}[status]
        print(f" {mark} {status:4}  {name:<{width}}  {detail}")
        n_fail += status == "FAIL"
        n_warn += status == "WARN"
    print("=" * (width + 60))
    print(f" {len(results) - n_fail - n_warn} passed, {n_warn} warn (optional), {n_fail} FAILED")
    if n_fail:
        print("\nENVIRONMENT NOT READY — see FAIL lines above.")
        return 1
    print("\nENVIRONMENT READY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
