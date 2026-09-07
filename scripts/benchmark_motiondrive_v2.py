#!/usr/bin/env python3
"""Measure the complete V2 forward, including motion, perception and planner."""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from audit_motiondrive_v2 import (ROOT, autocast_context, common_parser, construct_model,
                                 load_batch, model_inputs, output_checks, sha256)


def summarize_ms(values):
    array = np.asarray(values, np.float64)
    if not array.size:
        raise ValueError("at least one timing sample is required")
    return {"median_ms": float(np.median(array)), "mean_ms": float(array.mean()),
            "p90_ms": float(np.percentile(array, 90)), "p95_ms": float(np.percentile(array, 95)),
            "p99_ms": float(np.percentile(array, 99)), "std_ms": float(array.std()),
            "min_ms": float(array.min()), "max_ms": float(array.max())}


def benchmark(model, batch, *, precision="bf16", warmup=20, repeats=50):
    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    device = batch["images"].device
    if device.type != "cuda":
        raise ValueError("official timing benchmark requires CUDA; CPU is not a substitute")
    if batch["images"].shape[0] != 1:
        raise ValueError("latency benchmark requires batch size 1")
    torch.cuda.set_device(device)
    model.eval()
    inputs = model_inputs(batch)
    with torch.inference_mode(), autocast_context(device, precision):
        for _ in range(warmup):
            output = model(**inputs)
        torch.cuda.synchronize(device)
        del output
        torch.cuda.reset_peak_memory_stats(device)
        cuda_ms, wall_ms = [], []
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            wall_start = time.perf_counter()
            start.record()
            output = model(**inputs)
            end.record()
            torch.cuda.synchronize(device)
            wall_ms.append((time.perf_counter() - wall_start) * 1000)
            cuda_ms.append(start.elapsed_time(end))
        checks = output_checks(output, 1)
    return {"cuda": summarize_ms(cuda_ms), "wall": summarize_ms(wall_ms),
            "cuda_samples_ms": cuda_ms, "wall_samples_ms": wall_ms,
            "warmup": warmup, "repeats": repeats,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "output_checks": checks, "output_shapes": {k: list(v.shape) for k, v in output.items() if torch.is_tensor(v)},
            "output_dtypes": {k: str(v.dtype) for k, v in output.items() if torch.is_tensor(v)},
            "all_output_checks_pass": all(checks.values())}


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--tf32", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.benchmark = False
    model = construct_model(args)
    batch, source = load_batch(args.batch, device=args.device, seed=args.seed)
    result = benchmark(model, batch, precision=args.precision, warmup=args.warmup, repeats=args.repeats)
    cfg = getattr(model, "config", getattr(model, "cfg", None))
    config = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else repr(cfg)
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,utilization.gpu",
                              "--format=csv"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        smi = "unavailable"
    result.update({"input_source": source, "checkpoint": args.checkpoint,
                   "model_load": model.audit_load_metadata,
                   "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
                   "weight_status": "checkpoint_loaded" if args.checkpoint else "random_initialization",
                   "device": args.device, "gpu": torch.cuda.get_device_name(args.device),
                   "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                   "torchvision": importlib.metadata.version("torchvision"),
                   "cudnn": torch.backends.cudnn.version(), "python": platform.python_version(),
                   "precision": args.precision, "tf32": bool(args.tf32), "config": config,
                   "nvidia_smi_after": smi,
                   "input_shapes": {k: list(v.shape) for k, v in model_inputs(batch).items()},
                   "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                                      [*sorted((ROOT / "models/motiondrive_v2").glob("*.py")),
                                       ROOT / "scripts/sparse_scoredrive.py",
                                       Path(__file__).resolve(), ROOT / "scripts/audit_motiondrive_v2.py"]
                                      if p.is_file()},
                   "included": "All current/history image forwards, scene, motion, state, occupancy/lane heads, planner.",
                   "excluded": "File loading, preprocessing outside the model, host-to-device transfer; no feature cache used.",
                   "production_latency_gate_final": False,
                   "notice": "Real-input provenance and deployment-equivalent wrapper must be validated before final acceptance."})
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps({"cuda": result["cuda"], "wall": result["wall"],
                      "peak_allocated_bytes": result["peak_allocated_bytes"],
                      "input_source": source, "output_checks_pass": result["all_output_checks_pass"],
                      "model_load": model.audit_load_metadata,
                      "report": str(target)}, indent=2))
    return 0 if result["all_output_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
