#!/usr/bin/env python3
"""RTX 4090 forward-time measurement for the MotionDrive V2 candidate.

The preserved Q&A defines T_infer as the accumulated time of every model forward
from the reset up to the final trajectory. This model consumes its four past
frames inside one top-level forward, so the accumulation is over a single call;
that single call still runs several image encodings internally, and the whole
call is what is timed here.

Input preparation (tar read, JPEG decode, undistort, resize) is outside the
timed region, matching the operators' answer that only the model forward is
measured.

This is our own measurement, not the organisers' one.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import torch

CODE = Path("/work/code")
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "scripts"))

from models import motiondrive_v2_inputs as adapter
from models.motiondrive_v2_inputs import CALIBRATION_COLUMNS, POSE_COLUMNS
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
import motiondrive_v2_training as mt

FLOPS_CUTOFF_G = 7053.0


def clip_from_tar(path: Path):
    import pyarrow.parquet as pq
    with tarfile.open(path) as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        token = sorted({n.split("/", 1)[0] for n in members})[0]

        def read(name):
            return archive.extractfile(members[f"{token}/{name}"]).read()

        calibration = pq.read_table(io.BytesIO(read("calibration.parquet")),
                                    columns=list(CALIBRATION_COLUMNS)).to_pylist()
        poses = pq.read_table(io.BytesIO(read("ego_pose.parquet")),
                              columns=list(POSE_COLUMNS)).to_pylist()
        prepared = adapter.prepare_clip_from_records(
            calibration, poses, lambda camera, frame: read(f"{camera}/frame_{frame}.jpg"))
    return token, prepared


def autocast(device, precision):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def time_forward(model, inputs, device, precision, warmup, repeats):
    with torch.no_grad(), autocast(device, precision):
        for _ in range(warmup):
            model(**inputs)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(**inputs)
            end.record()
            torch.cuda.synchronize(device)
            timings.append(start.elapsed_time(end))
    return np.asarray(timings, dtype=np.float64)


def summarize(timings):
    return {"median_ms": float(np.median(timings)), "mean_ms": float(timings.mean()),
            "p90_ms": float(np.percentile(timings, 90)),
            "p95_ms": float(np.percentile(timings, 95)),
            "p99_ms": float(np.percentile(timings, 99)),
            "min_ms": float(timings.min()), "max_ms": float(timings.max()),
            "std_ms": float(timings.std(ddof=1)), "repeats": int(len(timings))}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--init", required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--reference-submission", help="the B200 submission json, for cross-check")
    args = parser.parse_args()

    payload = torch.load(args.init, map_location="cpu", weights_only=False)
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    model = MotionDriveV2(config)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device("cuda:0")
    model.to(device).eval()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

    reference = json.loads(Path(args.reference_submission).read_text()) if args.reference_submission else {}

    calls = {"n": 0}
    original = MotionDriveV2.forward

    def counting(self, *a, **kw):
        calls["n"] += 1
        return original(self, *a, **kw)

    per_clip = []
    for clip_path in args.clips:
        token, prepared = clip_from_tar(Path(clip_path))
        inputs = {k: v.to(device) for k, v in prepared.inputs.items()}

        calls["n"] = 0
        MotionDriveV2.forward = counting
        try:
            with torch.no_grad(), autocast(device, "bf16"):
                output = model(**inputs)["plan_abs"].float().cpu().numpy()[0]
        finally:
            MotionDriveV2.forward = original

        entry = {"clip": token, "forwards_per_clip": calls["n"],
                 "prediction_xy": output.tolist()}
        if token in reference:
            b200 = np.asarray(reference[token], dtype=np.float64)
            entry["cross_check_vs_b200"] = {
                "max_abs_xy_diff_m": float(np.abs(output - b200).max()),
                "mean_abs_xy_diff_m": float(np.abs(output - b200).mean()),
                "b200_endpoint_m": float(np.linalg.norm(b200[-1])),
                "rtx4090_endpoint_m": float(np.linalg.norm(output[-1])),
            }
        for precision in ("bf16", "fp32"):
            timings = time_forward(model, inputs, device, precision, args.warmup, args.repeats)
            entry[precision] = summarize(timings)
            entry[precision]["peak_memory_mib"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
        per_clip.append(entry)
        print(json.dumps({"clip": token, "bf16_median_ms": entry["bf16"]["median_ms"],
                          "fp32_median_ms": entry["fp32"]["median_ms"]}), flush=True)

    token, prepared = clip_from_tar(Path(args.clips[0]))
    inputs = {k: v.to(device) for k, v in prepared.inputs.items()}
    with torch.no_grad(), autocast(device, "bf16"):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    with_flops=True) as profile:
            model(**inputs)
    flops = sum(event.flops for event in profile.key_averages() if event.flops)

    bf16_medians = [c["bf16"]["median_ms"] for c in per_clip]
    payload_out = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(device),
        "driver_reported_by_torch": torch.version.cuda,
        "torch": torch.__version__,
        "measured_by": "this project, inside a pinned container; not the organisers' measurement",
        "t_infer_definition": (
            "accumulated time of every model forward from reset to the final trajectory; "
            "this model consumes its four past frames inside ONE top-level forward, so the "
            "accumulation is over a single call, which internally runs several image encodings"),
        "timed_region": "model forward only; tar read, JPEG decode, undistort and resize are outside it",
        "forwards_per_clip": per_clip[0]["forwards_per_clip"],
        "batch": 1, "warmup": args.warmup, "repeats_per_clip": args.repeats,
        "clips": len(per_clip),
        "bf16_median_ms_across_clips": {"min": float(min(bf16_medians)),
                                        "max": float(max(bf16_medians))},
        "profiler_gflops_total": flops / 1e9,
        "flops_cutoff_gflops": FLOPS_CUTOFF_G,
        "flops_headroom_x": FLOPS_CUTOFF_G / (flops / 1e9) if flops else None,
        "flops_note": ("torch profiler counts convolution and matmul only; it is a reference "
                       "figure for the registered operator range, not an official tool result"),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "checkpoint": str(Path(args.init).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(args.init).read_bytes()).hexdigest(),
        "model_state_sha256": mt.tensor_state_sha256(payload["model"]),
        "per_clip": per_clip,
        "limitations": [
            "Our own timing on our own 4090; the organisers measure on their environment.",
            "Warm cache, batch 1, deterministic cudnn, TF32 disabled to match the training-time settings.",
            "The score's time term also depends on how the official harness brackets the forward.",
        ],
    }
    Path(args.out).write_text(json.dumps(payload_out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: payload_out[k] for k in
                      ("device", "forwards_per_clip", "bf16_median_ms_across_clips",
                       "profiler_gflops_total", "flops_headroom_x")}, indent=1))


if __name__ == "__main__":
    main()
