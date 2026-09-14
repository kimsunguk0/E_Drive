#!/usr/bin/env python3
"""Forward cost of the R0 family: how many forwards, how many FLOPs, how long.

This is a B200 measurement and a FLOPs count.  It is NOT an RTX4090 number and
is not written into the Error Score; the guide's time penalty must be measured
on the official device.  What it does settle is the structural question: how
many model forwards one clip costs, and whether the operation count is anywhere
near the FLOPs cut-off.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models import motiondrive_v2_inputs as adapter
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
import train_motiondrive_v2 as trainer

FIXTURE_ROOT = ROOT / "data/etri/motiondrive_v2/deploy_fixture_train8"
OUT = ROOT / "reports/md_r0_reset_20260914/forward_cost.json"
# Printed guide p.13: the FLOPs cut-off is three times the baseline; the number
# recorded in this project's compliance notes for that cut-off is 7053 GFLOPs.
FLOPS_CUTOFF_G = 7053.0


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--init", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()

    payload = torch.load(args.init, map_location="cpu", weights_only=False)
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    model = MotionDriveV2(config)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(f"cuda:{args.gpu}")
    model.to(device).eval()

    prepared = adapter.prepare_clip_inputs(FIXTURE_ROOT / "fixture_000")
    inputs = {k: v.to(device) for k, v in prepared.inputs.items()}

    # how many model forwards does one clip cost?
    calls = {"n": 0}
    original_forward = MotionDriveV2.forward

    def counting_forward(self, *a, **kw):
        calls["n"] += 1
        return original_forward(self, *a, **kw)

    MotionDriveV2.forward = counting_forward
    try:
        with torch.no_grad(), trainer.autocast(device, args.precision):
            model(**inputs)
    finally:
        MotionDriveV2.forward = original_forward
    forwards_per_clip = calls["n"]

    with torch.no_grad(), trainer.autocast(device, args.precision):
        for _ in range(args.warmup):
            model(**inputs)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(**inputs)
            end.record()
            torch.cuda.synchronize(device)
            timings.append(start.elapsed_time(end))
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

    with torch.no_grad(), trainer.autocast(device, args.precision):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    with_flops=True) as profile:
            model(**inputs)
    flops = sum(event.flops for event in profile.key_averages() if event.flops)
    top = sorted(profile.key_averages(), key=lambda e: e.flops or 0, reverse=True)[:8]

    timings = np.asarray(timings, dtype=np.float64)
    result = {
        "schema_version": 1,
        "label": args.label,
        "checkpoint": str(Path(args.init).resolve()),
        "checkpoint_sha256": trainer.sha256(args.init),
        "device": torch.cuda.get_device_name(device),
        "device_is_not_the_scored_device": True,
        "precision": args.precision, "batch": 1,
        "forwards_per_clip": forwards_per_clip,
        "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
        "latency_ms": {
            "median": float(np.median(timings)), "mean": float(timings.mean()),
            "p90": float(np.percentile(timings, 90)), "p95": float(np.percentile(timings, 95)),
            "p99": float(np.percentile(timings, 99)),
            "min": float(timings.min()), "max": float(timings.max()),
            "std": float(timings.std(ddof=1)), "repeats": int(args.repeats),
        },
        "peak_memory_mib": {"allocated": peak_alloc / 2 ** 20, "reserved": peak_reserved / 2 ** 20},
        "profiler_flops_total": int(flops),
        "profiler_gflops_total": flops / 1e9,
        "flops_cutoff_gflops": FLOPS_CUTOFF_G,
        "flops_headroom_x": FLOPS_CUTOFF_G / (flops / 1e9) if flops else None,
        "top_flops_operators": [{"name": e.key, "gflops": (e.flops or 0) / 1e9,
                                 "calls": e.count} for e in top],
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "limitations": [
            "B200 timing is not the RTX4090 time used in the Error Score and must not be substituted for it.",
            "The profiler counts convolution and matmul FLOPs; elementwise and layout operations are not included.",
            "One clip, batch 1, warm cache; JPEG decode and preprocessing are outside the measured region, matching the operators' answer that only model forward is timed.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    existing[args.label] = result
    OUT.write_text(json.dumps(existing, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in
                      ("label", "device", "forwards_per_clip", "latency_ms",
                       "profiler_gflops_total", "flops_headroom_x", "peak_memory_mib")},
                     indent=1))


if __name__ == "__main__":
    main()
