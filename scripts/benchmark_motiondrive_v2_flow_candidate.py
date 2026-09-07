#!/usr/bin/env python3
"""Research-only RAFT-small component timing. Never a V2 inference/accuracy score."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import torch
from torch.nn import functional as F

REFERENCE_SHA = "4fb192ff6dd80ad44c8374b64afe55d49112327174ee8f7534ef4591cd2de1f3"
WEIGHTS_URL = "https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth"
WEIGHTS_PREFIX = "01064c6d"
UPDATES = (4, 8, 12)
WARMUP, REPEATS = 20, 50


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_pairs(batch):
    """Only two image keys are consulted; labels/pose/goal/status cannot flow here."""
    current = batch["images"][0:1, 0].float()
    history = batch["history_images"][0].float()
    if current.ndim != 4 or history.ndim != 4 or current.shape[1] != 3 or history.shape[:2] != (4, 3):
        raise ValueError("Expected one current front and four historical RGB frames")
    current = F.interpolate(current, size=history.shape[-2:], mode="bilinear",
                            align_corners=False, antialias=True)
    mean = current.new_tensor([.485, .456, .406])[None, :, None, None]
    std = current.new_tensor([.229, .224, .225])[None, :, None, None]
    # Undo ImageNet normalization, then apply RAFT's [0,1] -> [-1,1]. No clipping.
    current = 2 * (current * std + mean) - 1
    history = 2 * (history * std + mean) - 1
    if not torch.isfinite(current).all() or not torch.isfinite(history).all():
        raise ValueError("Nonfinite image inputs")
    if max(float(current.abs().max()), float(history.abs().max())) > 1.00001:
        raise ValueError("Input is not an ImageNet-normalized RGB image")
    return {1: (current.contiguous(), history[:1].contiguous()),
            4: (current.expand(4, -1, -1, -1).contiguous(), history.contiguous())}


def validate_outputs(outputs, batch_size, updates):
    if not isinstance(outputs, list) or len(outputs) != updates:
        raise ValueError("RAFT must return one flow tensor per update")
    for flow in outputs:
        if flow.shape != (batch_size, 2, 216, 384) or flow.dtype != torch.float32:
            raise ValueError("Unexpected flow shape or precision")
        if not torch.isfinite(flow).all():
            raise ValueError("Nonfinite flow")
    return {"list_length": len(outputs), "shape": list(outputs[-1].shape),
            "dtype": str(outputs[-1].dtype), "all_updates_finite": True,
            "final_min_pixels": float(outputs[-1].min()), "final_max_pixels": float(outputs[-1].max())}


def summary_ms(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != REPEATS or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Expected 50 positive finite event measurements")
    return {"median_ms": float(np.median(values)), "p90_ms": float(np.percentile(values, 90)),
            "p95_ms": float(np.percentile(values, 95)), "max_ms": float(values.max()),
            "std_population_ms": float(values.std(ddof=0)), "repeats": len(values)}


def query_compute(uuid):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return sorted({int(line.split(",")[0]) for line in result.stdout.splitlines()
                   if line.strip() and line.split(",")[1].strip() == uuid})


def foreign_pids(pids, own_pid):
    return sorted(set(pids) - {own_pid})


def gpu_info():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,driver_version",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    rows = [[part.strip() for part in line.split(",")] for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or rows[0][0] != "0":
        raise RuntimeError("Only the single approved GPU0 may be visible")
    index, uuid, name, memory, driver = rows[0]
    return {"index": int(index), "uuid": uuid, "name": name, "memory_used_mib": int(memory), "driver": driver}


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to replace a measurement")
    if torch.cuda.is_initialized():
        raise RuntimeError("Use a fresh process")
    start = time.time()
    import torchvision
    import torchvision.models.optical_flow.raft as raft_source
    import torchvision.models.optical_flow._utils as flow_utils
    import torchvision.transforms._presets as presets
    from torchvision.models.optical_flow import raft_small

    paths = [Path(__file__), Path(args.reference), Path(args.weights),
             Path(inspect.getfile(raft_source)), Path(inspect.getfile(flow_utils)), Path(inspect.getfile(presets))]
    before = {str(p): sha256(p) for p in paths}
    if before[str(Path(args.reference))] != REFERENCE_SHA:
        raise ValueError("Immutable train8 reference mismatch")
    if not before[str(Path(args.weights))].startswith(WEIGHTS_PREFIX):
        raise ValueError("Official weight SHA prefix mismatch")
    hardware = gpu_info()
    initial_compute = query_compute(hardware["uuid"])
    if initial_compute or hardware["memory_used_mib"] >= 1000:
        raise RuntimeError("GPU is not idle: no compute PID and memory <1000 MiB required")
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    batch, metadata = reference["batch"], reference["metadata"]
    if (tuple(batch["images"].shape) != (8, 6, 3, 432, 768) or
            tuple(batch["history_images"].shape) != (8, 4, 3, 216, 384)):
        raise ValueError("Unexpected immutable train8 pixel contract")
    identity = metadata["samples"][0]
    if identity["scenario"] != batch["scenario"][0] or identity["frame"] != int(batch["frame"][0]):
        raise ValueError("Reference clip identity disagrees with export metadata")
    pairs_cpu = prepare_pairs(batch)
    # The fixture reader alone sees metadata. The network receives ONLY these RGB pairs.
    del reference, batch
    model = raft_small(weights=None, progress=False).eval()
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    load_result = model.load_state_dict(state, strict=True)
    del state
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != 990162:
        raise RuntimeError("Unexpected RAFT-small architecture")
    flags_initial = {"cudnn_benchmark": torch.backends.cudnn.benchmark,
                     "cudnn_deterministic": torch.backends.cudnn.deterministic,
                     "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                     "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                     "float32_matmul_precision": torch.get_float32_matmul_precision(),
                     "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = model.to(device="cuda:0", dtype=torch.float32)
    pairs = {n: (a.to("cuda:0"), b.to("cuda:0")) for n, (a, b) in pairs_cpu.items()}
    torch.cuda.synchronize()
    own = os.getpid()
    after_init = query_compute(hardware["uuid"])
    if after_init != [own]:
        raise RuntimeError("Expected host PID namespace and only this GPU process; refusing ambiguous ownership")
    stop = threading.Event()
    errors, observations = [], []

    def poll():
        while not stop.is_set():
            try:
                pids = query_compute(hardware["uuid"])
                observations.append({"unix_time": time.time(), "pids": pids})
                if foreign_pids(pids, own):
                    errors.append("External GPU compute PID appeared: " + repr(pids))
                    return
            except Exception as exc:
                errors.append(repr(exc))
                return
            stop.wait(.5)

    monitor = threading.Thread(target=poll, daemon=True)
    monitor.start()

    def healthy():
        if errors:
            raise RuntimeError(errors[0])

    results = []
    try:
        with torch.inference_mode():
            for pair_count in (1, 4):
                for updates in UPDATES:
                    healthy()
                    first, second = pairs[pair_count]
                    for _ in range(WARMUP):
                        healthy()
                        flows = model(first, second, num_flow_updates=updates)
                        del flows
                    torch.cuda.synchronize()
                    start_alloc = torch.cuda.memory_allocated()
                    start_reserved = torch.cuda.memory_reserved()
                    torch.cuda.reset_peak_memory_stats()
                    times = []
                    for _ in range(REPEATS):
                        healthy()
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        begin.record()
                        flows = model(first, second, num_flow_updates=updates)
                        end.record()
                        torch.cuda.synchronize()
                        times.append(begin.elapsed_time(end))
                        validation = validate_outputs(flows, pair_count, updates)
                        del flows
                    healthy()
                    row = {"pair_count": pair_count, "num_flow_updates": updates,
                           **summary_ms(times), "event_ms": times, "output_validation": validation,
                           "memory_allocated_before_bytes": start_alloc,
                           "memory_reserved_before_bytes": start_reserved,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                           "memory_allocated_after_bytes": torch.cuda.memory_allocated(),
                           "memory_reserved_after_bytes": torch.cuda.memory_reserved()}
                    results.append(row)
                    print(json.dumps({k: v for k, v in row.items() if k != "event_ms"}), flush=True)
    finally:
        stop.set()
        monitor.join(timeout=10)
    healthy()
    final_compute = query_compute(hardware["uuid"])
    if foreign_pids(final_compute, own):
        raise RuntimeError("External GPU work detected after measurements; discard results")
    after = {str(p): sha256(p) for p in paths}
    if before != after:
        raise RuntimeError("Source, reference, or weight changed")
    report = {"status": "completed", "research_only": True, "not_v2_tinfer": True,
              "not_velocity_or_planning_accuracy": True, "license_review_complete": False,
              "argv": vars(args), "pid": own, "gpu": hardware, "initial_compute_pids": initial_compute,
              "after_cuda_init_compute_pids": after_init, "final_compute_pids": final_compute,
              "compute_monitor": {"interval_seconds": .5, "observations": observations,
                                  "errors": errors, "signals_sent": False},
              "reference_clip_identity": identity, "image_keys_only": ["images[:,0]", "history_images"],
              "forward_inputs": ["current_front_rgb", "historical_front_rgb"],
              "provided_goal_pose_status_labels_forwarded": False,
              "pair_direction": "current t0 -> past [-1,-2,-5,-10]; one-pair uses -1",
              "weights_url": WEIGHTS_URL, "weight_sha256": before[str(Path(args.weights))],
              "strict_load": {"missing": load_result.missing_keys, "unexpected": load_result.unexpected_keys},
              "parameter_count": parameter_count, "source_sha256_before": before, "source_sha256_after": after,
              "environment": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                              "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
              "pytorch_flags_initial": flags_initial,
              "pytorch_flags_effective": {"cudnn_benchmark": False, "cudnn_allow_tf32": False,
                                          "matmul_allow_tf32": False, "autocast": False},
              "precision": "float32 with TF32 disabled", "warmup_per_condition": WARMUP,
              "repeats_per_condition": REPEATS, "conditions": results,
              "measurement_scope": "Every full torchvision RAFT forward: both image feature encodings, full all-pairs correlation pyramid, context encoder, all recurrent updates and all output upsampling. No feature cache or warm-start flow.",
              "excluded": ["CPU preprocessing", "H2D transfers", "model/weights load", "output validation"],
              "preprocessing": "current768 -> history384 F.interpolate bilinear align_corners=False antialias=True; inverse ImageNet normalization; map RGB[0,1] to [-1,1]; no further crop/pad/resize/clamp",
              "wall_start_unix": start, "wall_finish_unix": time.time(),
              "limitations": ["Component timing on one fixed training clip, not full V2 inference.",
                              "No optical-flow GT, metric velocity, planning, or generalization evaluation.",
                              "GPU occupancy is sampled every 0.5 s; shorter external processes could escape observation.",
                              "Memory includes model/input/correlation buffers and allocator history across conditions.",
                              "Public weight license review is not complete; research-only and not an adoption decision."]}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
