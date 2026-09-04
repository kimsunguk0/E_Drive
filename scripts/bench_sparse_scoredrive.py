#!/usr/bin/env python3
"""Measure one complete six-camera SparseScoreDrive forward on CUDA."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sparse_scoredrive import SparseScoreDrive, count_parameters


def stats_ms(values):
    a = np.asarray(values, np.float64)
    return {
        "median_ms": float(np.median(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std()),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "repeats": int(a.size),
    }


def timed(fn, repeats):
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
    return stats_ms(values)


def conv_linear_flops(model: nn.Module, images, lidar2img, autocast_ctx):
    """Hook-based conv/linear FLOPs; excludes grid_sample/sort/elementwise ops."""
    total = 0
    handles = []

    def conv_hook(module, inputs, output):
        nonlocal total
        out = output
        batch = out.shape[0]
        out_h, out_w = out.shape[-2:]
        kernel = module.kernel_size[0] * module.kernel_size[1]
        mac = batch * out.shape[1] * out_h * out_w * (module.in_channels // module.groups) * kernel
        total += 2 * mac

    def linear_hook(module, inputs, output):
        nonlocal total
        total += 2 * output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    with torch.inference_mode(), autocast_ctx():
        model(images, lidar2img)
    for handle in handles:
        handle.remove()
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--precision", choices=("fp32", "amp_fp16"), default="amp_fp16")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the latency gate")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")
    fixture = np.load(args.fixture, allow_pickle=False)
    images = torch.from_numpy(fixture["images"])[None].to(device)
    lidar2img = torch.from_numpy(fixture["lidar2img"])[None].to(device)
    model = SparseScoreDrive(args.bank).eval().to(device)

    enabled = args.precision == "amp_fp16"
    def autocast_ctx():
        return torch.autocast("cuda", dtype=torch.float16, enabled=enabled)

    with torch.inference_mode():
        for _ in range(args.warmup):
            with autocast_ctx():
                model(images, lidar2img)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        def full_forward():
            with autocast_ctx():
                model(images, lidar2img)

        total = timed(full_forward, args.repeats)
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

        with autocast_ctx():
            levels, p4 = model.encode_images(images)
            evidence = images.abs().amax(dim=(1, 2, 3, 4)) > 0
            logits = model.score_features(
                levels, p4, lidar2img, images.shape[-2:], evidence)

        def encode_only():
            with autocast_ctx():
                model.encode_images(images)

        def score_only():
            with autocast_ctx():
                model.score_features(
                    levels, p4, lidar2img, images.shape[-2:], evidence)

        def candidate_api_only():
            with autocast_ctx():
                model.build_candidates(logits)

        encode = timed(encode_only, args.repeats)
        score = timed(score_only, args.repeats)
        candidate_api = timed(candidate_api_only, args.repeats)

    flops = conv_linear_flops(model, images, lidar2img, autocast_ctx)
    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "precision": args.precision,
        "input_shape": list(images.shape),
        "warmup": args.warmup,
        "one_top_level_forward": True,
        "total_forward": total,
        "breakdown_non_additive": {
            "backbone_fpn": encode,
            "projection_sampling_scorer": score,
            "stable_shortlist_candidate_api": candidate_api,
        },
        "peak_allocated_mib": peak_alloc / (1024 ** 2),
        "peak_reserved_mib": peak_reserved / (1024 ** 2),
        "parameters": count_parameters(model),
        "conv_linear_flops": int(flops),
        "conv_linear_gflops": flops / 1.0e9,
        "flops_note": "2 FLOPs/MAC; excludes grid_sample, projection, sort, and elementwise ops",
        "weights": "random Phase-5B skeleton; no accuracy claim",
        "fixture_manifest": str(fixture["manifest"]),
    }
    median = total["median_ms"]
    if median <= 100:
        result["gate"] = "VERY_GOOD_LE_100MS"
    elif median <= 150:
        result["gate"] = "KEEP_100_TO_150MS"
    elif median <= 200:
        result["gate"] = "SHRINK_150_TO_200MS"
    else:
        result["gate"] = "REDESIGN_GT_200MS"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
