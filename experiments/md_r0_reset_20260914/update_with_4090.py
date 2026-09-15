#!/usr/bin/env python3
"""Fold the RTX 4090 measurement into the candidate and submission registries."""
import json
from pathlib import Path

ROOT = Path("/NHNHOME/data/sukim/adcl")
REPORTS = ROOT / "reports/md_r0_reset_20260914"
rtx = json.loads((REPORTS / "rtx4090_forward_cost.json").read_text())

medians = [c["bf16"]["median_ms"] for c in rtx["per_clip"]]
p95s = [c["bf16"]["p95_ms"] for c in rtx["per_clip"]]
worst_p95 = max(p95s)
cross = [c["cross_check_vs_b200"] for c in rtx["per_clip"] if "cross_check_vs_b200" in c]

timing = {
    "device": rtx["device"],
    "measured_by": rtx["measured_by"],
    "t_infer_definition": rtx["t_infer_definition"],
    "timed_region": rtx["timed_region"],
    "forwards_per_clip": rtx["forwards_per_clip"],
    "batch": 1, "precision": "bf16",
    "warmup": rtx["warmup"], "repeats_per_clip": rtx["repeats_per_clip"],
    "clips": rtx["clips"],
    "median_ms_range": [min(medians), max(medians)],
    "worst_p95_ms": worst_p95,
    "fp32_median_ms_range": [min(c["fp32"]["median_ms"] for c in rtx["per_clip"]),
                             max(c["fp32"]["median_ms"] for c in rtx["per_clip"])],
    "peak_memory_mib": max(c["bf16"]["peak_memory_mib"] for c in rtx["per_clip"]),
    "time_penalty": {
        "formula": "Error = L2 * (1 + max(0, T_ms - 100) / 200)",
        "penalty_start_ms": 100,
        "multiplier_at_worst_p95": 1.0 + max(0.0, (worst_p95 - 100.0) / 200.0),
        "reading": ("the measured forward is far under the 100 ms point where the time term "
                    "starts, so on this measurement the time term is inert"),
        "caveat": ("this is our measurement on our own 4090; the organisers measure in their "
                   "environment and may bracket the forward differently"),
    },
    "flops": {
        "profiler_gflops_total": rtx["profiler_gflops_total"],
        "cutoff_gflops": rtx["flops_cutoff_gflops"],
        "headroom_x": rtx["flops_headroom_x"],
        "note": rtx["flops_note"],
    },
    "port_cross_check_vs_b200_submission": {
        "clips": len(cross),
        "max_abs_xy_diff_m": max(c["max_abs_xy_diff_m"] for c in cross) if cross else None,
        "mean_abs_xy_diff_m": max(c["mean_abs_xy_diff_m"] for c in cross) if cross else None,
        "reading": ("the container reproduces the B200 trajectories to a few millimetres; the "
                    "residual is bf16 kernel differences between architectures, not a port error"),
    },
    "container": {
        "dockerfile": "experiments/md_r0_reset_20260914/rtx4090/Dockerfile",
        "script": "experiments/md_r0_reset_20260914/rtx4090/measure_4090.py",
        "status": "BUILT_AND_RUN",
        "base": "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04",
        "torch": rtx["torch"],
        "host": "RTX 4090 workstation with driver only; CUDA toolkit supplied by the image",
    },
    "artifact": "reports/md_r0_reset_20260914/rtx4090_forward_cost.json",
}

candidate_path = REPORTS / "candidate_registry.json"
candidate = json.loads(candidate_path.read_text())
if candidate["checkpoint_sha256"] != rtx["checkpoint_sha256"]:
    raise SystemExit("the timed checkpoint is not the registered candidate")
candidate["checks"]["forward_cost"]["rtx4090_measured"] = True
candidate["checks"]["rtx4090_timing"] = timing
candidate["pending"] = [p for p in candidate["pending"]
                        if "RTX4090" not in p and "Error Score time term" not in p]
candidate_path.write_text(json.dumps(candidate, indent=1, sort_keys=True) + "\n")

submission_path = REPORTS / "submission_registry.json"
submission = json.loads(submission_path.read_text())
submission["entries"]["E1-EXP_terminal"]["rtx4090_timing"] = timing
submission["pending"] = [p for p in submission["pending"] if "RTX4090" not in p]
submission_path.write_text(json.dumps(submission, indent=1, sort_keys=True) + "\n")

repro_path = REPORTS / "reproduction_manifest.json"
repro = json.loads(repro_path.read_text())
repro["docker"] = {
    "dockerfile": "experiments/md_r0_reset_20260914/rtx4090/Dockerfile",
    "status": "BUILT_AND_RUN_ON_RTX4090",
    "image": "md-v2:4090",
    "note": ("built on a workstation that had only the NVIDIA driver, no CUDA toolkit and no "
             "docker; the image supplies the CUDA runtime and the pinned wheels"),
}
repro["rtx4090_timing"] = timing
repro_path.write_text(json.dumps(repro, indent=1, sort_keys=True) + "\n")

print(json.dumps({"median_ms_range": timing["median_ms_range"],
                  "worst_p95_ms": timing["worst_p95_ms"],
                  "penalty_multiplier": timing["time_penalty"]["multiplier_at_worst_p95"],
                  "flops_headroom_x": timing["flops"]["headroom_x"],
                  "port_max_diff_m": timing["port_cross_check_vs_b200_submission"]["max_abs_xy_diff_m"]},
                 indent=1))
