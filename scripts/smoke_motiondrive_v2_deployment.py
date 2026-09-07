#!/usr/bin/env python3
"""Research-only strict bundle -> raw train8 -> complete serving smoke.

This is not a submission, accuracy evaluation, benchmark or checkpoint selector.
The existing labeled reference is opened only by this diagnostic reader. Only
its six explicitly adapted input tensors can reach the model. Existing source,
reference, raw clips and bundle are never modified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import motiondrive_v2_inputs as adapter
from models.motiondrive_v2_serving import (
    absolute_plan_to_list, forward_clip, load_deployment_bundle,
)
from scripts.audit_motiondrive_v2_deploy_inputs import (
    CALIBRATION_SHA256, FIXTURE_MANIFEST_SHA256, REFERENCE_SHA256,
    byte_equal, file_sha, make_reference_inputs, tree_sha,
    validate_identity, verify_raw_files,
)
from scripts.export_motiondrive_v2_inference import C1_SUPERVISION_SHA256
from scripts.motiondrive_v2_training import tensor_state_sha256

PLAN_ATOL = 1e-5  # Fixed before actual execution; rtol=0, never auto-enlarged.
EXPECTED_ENCODINGS = [[6, 3, 432, 768], [5, 3, 216, 384]]


def gpu_snapshot(*, require_idle=False):
    query = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"]
    rows = subprocess.check_output(query, text=True).strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError("This smoke requires the single-GPU 3090 host")
    fields = [part.strip() for part in rows[0].split(",")]
    if len(fields) != 5 or fields[0] != "0":
        raise RuntimeError("Unexpected GPU query format")
    used, free, total = map(float, fields[2:])
    if not all(np.isfinite([used, free, total])) or min(used, free) < 0 or total <= 0:
        raise RuntimeError("Invalid GPU memory observation")
    command = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
               "--format=csv,noheader"]
    processes = []
    for row in subprocess.check_output(command, text=True).strip().splitlines():
        uuid, pid, name = [part.strip() for part in row.split(",", 2)]
        processes.append({"gpu_uuid": uuid, "pid": int(pid), "name": name})
    foreign = [row for row in processes if row["pid"] != os.getpid()]
    if foreign or require_idle and (processes or used >= 1000):
        raise RuntimeError(f"GPU is not uncontended; no process will be stopped: {processes}")
    return {"observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "gpu_uuid": fields[1], "used_mib": used, "free_mib": free,
            "total_mib": total, "processes": processes, "foreign_processes": foreign}


def compare_plans(actual, expected):
    if (actual.shape != (6, 2) or expected.shape != (6, 2) or
            actual.dtype != torch.float32 or expected.dtype != torch.float32 or
            actual.device.type != "cpu" or expected.device.type != "cpu" or
            not torch.isfinite(actual).all() or not torch.isfinite(expected).all()):
        raise ValueError("Only finite CPU FP32 absolute [6,2] plans can be compared")
    difference = (actual.double() - expected.double()).abs()
    return {"bitwise_equal": byte_equal(actual, expected), "max_abs_m": float(difference.max()),
            "atol_m": PLAN_ATOL, "rtol": 0., "pass": float(difference.max()) <= PLAN_ATOL}


def observed_forward(model, inputs, contract, *, device, precision):
    """Hooks observe exactly one public call and real image encodings; no timing."""
    calls, encodings, observed_outputs = [], [], []

    def before(module, args, kwargs):
        if args or set(kwargs) != set(adapter.INPUT_KEYS):
            raise RuntimeError("Full forward received positional or non-whitelisted inputs")
        calls.append({key: list(value.shape) for key, value in kwargs.items()})

    def encoded(module, args):
        encodings.append(list(args[0].shape))

    def after(module, args, output):
        shapes = {"plan_abs": (1, 6, 2), "state_hat": (1, 6), "history_hat": (1, 4, 4),
                  "scene_features": (1, 3072, 128), "motion_features": (1, 192, 128),
                  "occ_logits": (1, 1, 64, 48), "lane_logits": (1, 1, 64, 48)}
        result = {}
        for name, shape in shapes.items():
            value = output.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise RuntimeError(f"Full forward returned invalid {name}")
            if name in ("plan_abs", "state_hat", "history_hat") and value.dtype != torch.float32:
                raise RuntimeError(f"Final head is not FP32: {name}")
            result[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "finite": True}
        result["plan_abs"]["values"] = output["plan_abs"].detach().cpu()[0].tolist()
        observed_outputs.append(result)

    handles = [model.register_forward_pre_hook(before, with_kwargs=True),
               model.backbone_fpn.register_forward_pre_hook(encoded),
               model.register_forward_hook(after)]
    try:
        plan = forward_clip(model, inputs, contract, device=device, precision=precision)
    finally:
        for handle in handles:
            handle.remove()
    if len(calls) != 1 or len(observed_outputs) != 1 or encodings != EXPECTED_ENCODINGS:
        raise RuntimeError(f"Expected one complete forward with 6+5 image encodings: {len(calls)}, {encodings}")
    encoded = absolute_plan_to_list(plan)
    restored = torch.tensor(json.loads(json.dumps(encoded, allow_nan=False)), dtype=torch.float32)
    if not byte_equal(plan, restored) or encoded != observed_outputs[0]["plan_abs"]["values"]:
        raise RuntimeError("ABS serialization changed neural output coordinates")
    return plan, {"full_forward_count": len(calls), "input_shapes": calls[0],
                  "backbone_input_shapes": encodings, "image_encodings": 11,
                  "outputs": observed_outputs[0], "absolute_json_roundtrip_bitwise": True,
                  "timing_measured": False}


def smoke(args):
    if torch.cuda.is_initialized():
        raise RuntimeError("Start a fresh process; validate CPU artifacts before CUDA")
    fixture = Path(args.fixture_root).resolve()
    manifest_path = fixture / "fixture_manifest.json"
    paths = {"bundle": Path(args.bundle), "reference": Path(args.reference),
             "raw_manifest": manifest_path, "calibration": Path(args.calibration),
             "geometry_contract": Path(args.geometry_contract)}
    pinned = {"bundle": args.expected_bundle_sha256, "reference": REFERENCE_SHA256,
              "raw_manifest": FIXTURE_MANIFEST_SHA256, "calibration": CALIBRATION_SHA256,
              "geometry_contract": C1_SUPERVISION_SHA256}
    before = {name: file_sha(path) for name, path in paths.items()}
    if before != pinned:
        raise ValueError(f"Artifact hashes differ from pinned inputs: {before}")
    source_manifest = json.loads(Path(args.source_manifest).read_text())
    source_before = {name: file_sha(ROOT / name) for name in source_manifest["files"]}
    if source_before != source_manifest["files"]:
        raise ValueError("Staged runtime source SHA mismatch")
    reference = torch.load(paths["reference"], map_location="cpu", weights_only=True)
    reference_before = tree_sha(reference)
    manifest = json.loads(manifest_path.read_text())
    clips = validate_identity(reference, manifest)
    raw_before = verify_raw_files(fixture, clips)
    with np.load(paths["calibration"], allow_pickle=False) as archive:
        calibration = torch.from_numpy(archive["lidar2img"].copy())
    expected_inputs, adaptation = make_reference_inputs(reference["batch"], calibration)
    contract = adapter.input_contract()
    prepared_clips, comparisons = [], []
    for i, clip in enumerate(clips):
        prepared = adapter.prepare_clip_inputs(fixture / clip["clip_id"])
        if prepared.metadata["input_contract"] != contract:
            raise ValueError("Complete adapter contract differs, including crop/pixel description")
        expected = {name: value[i:i + 1] for name, value in expected_inputs.items()}
        comparison = adapter.compare_input_fixture(prepared, {"inputs": expected,
            "metadata": {"input_contract": contract}}, projection_atol=1e-4, pose_atol=1e-5)
        if not comparison["all_pass"]:
            raise RuntimeError(f"Raw/reference input mismatch before model forward: {clip['clip_id']}")
        prepared_clips.append(prepared)
        comparisons.append(comparison)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    observations = [gpu_snapshot(require_idle=True)]
    model, loaded_contract = load_deployment_bundle(paths["bundle"],
        expected_bundle_sha256=args.expected_bundle_sha256, device=args.device)
    if loaded_contract != contract:
        raise ValueError("Bundle and complete adapter contract differ")
    if (model.config.motion_input_mode != "low_feature" or not model.config.goal_on or
            not model.config.state_on):
        raise ValueError("This smoke is specifically the full C1T1 G1S1 low_feature canary")
    with paths["bundle"].open("rb") as stream:
        bundle = torch.load(stream, map_location="cpu", weights_only=True)
    if bundle["step"] != 2 or Path(bundle["deployment_provenance"]["run_dir"]).name != "p2_shared_memory_probe_s0":
        raise ValueError("Smoke must not silently replace the explicitly authorized two-step canary")
    state_before = tensor_state_sha256(model.state_dict())
    if state_before != tensor_state_sha256(bundle["model"]):
        raise RuntimeError("Strict loader changed model tensors")
    torch.cuda.reset_peak_memory_stats(args.device)
    results, first_plan = [], None
    for i, (clip, prepared) in enumerate(zip(clips, prepared_clips)):
        observations.append(gpu_snapshot())
        raw_plan, raw_call = observed_forward(model, prepared.inputs, contract,
            device=args.device, precision=args.precision)
        observations.append(gpu_snapshot())
        reference_inputs = {name: value[i:i + 1] for name, value in expected_inputs.items()}
        reference_plan, reference_call = observed_forward(model, reference_inputs, contract,
            device=args.device, precision=args.precision)
        equality = compare_plans(raw_plan, reference_plan)
        if not equality["pass"]:
            raise RuntimeError(f"Raw/reference ABS outputs disagree: {clip['clip_id']} {equality}")
        if first_plan is None:
            first_plan = raw_plan.clone()
        results.append({"clip_id": clip["clip_id"], "source_scene": clip["source_scene"],
            "source_frame": clip["source_frame"], "mapping_is_diagnostic_only": True,
            "input_comparison": comparisons[i], "raw_forward": raw_call,
            "reference_forward": reference_call, "output_comparison": equality})
    observations.append(gpu_snapshot())
    repeated_plan, repeated_call = observed_forward(model, prepared_clips[0].inputs, contract,
        device=args.device, precision=args.precision)
    repeat = compare_plans(repeated_plan, first_plan)
    if not repeat["pass"]:
        raise RuntimeError("A/other clips/A stateless output check failed")
    torch.cuda.synchronize(args.device)
    memory = {"allocated_bytes": torch.cuda.memory_allocated(args.device),
              "reserved_bytes": torch.cuda.memory_reserved(args.device),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device)}
    state_after = tensor_state_sha256(model.state_dict())
    after = {name: file_sha(path) for name, path in paths.items()}
    source_after = {name: file_sha(ROOT / name) for name in source_before}
    raw_after = verify_raw_files(fixture, clips)
    if (state_before != state_after or before != after or source_before != source_after or
            raw_before != raw_after or reference_before != tree_sha(reference)):
        raise RuntimeError("Model state or immutable input/source artifact changed during smoke")
    import cv2, PIL, pyarrow, scipy, torchvision
    return {"schema_version": 1, "status": "completed_pass", "pid": os.getpid(),
        "purpose": "Two-step canary packaging/input/full-forward/ABS smoke; not a final submission candidate",
        "not_accuracy_or_latency_evaluation": True, "official_submission_performed": False,
        "precision": args.precision, "device": args.device,
        "input_contract": contract, "training_source": bundle["manifest"]["git_sha"],
        "canary_checkpoint_sha256": bundle["source_checkpoint"]["sha256"],
        "checkpoint_step": bundle["step"], "source_manifest": source_manifest,
        "artifacts_before": before, "artifacts_after": after,
        "source_sha256_before": source_before, "source_sha256_after": source_after,
        "raw_file_count": len(raw_before), "raw_files_before": raw_before, "raw_files_after": raw_after,
        "reference_adaptation": adaptation, "reference_and_excluded_labels_unchanged": True,
        "model_state_sha256_before": state_before, "model_state_sha256_after": state_after,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "total_full_forward_calls": 17, "clips": results,
        "interleaved_repeat": {"comparison": repeat, "forward": repeated_call},
        "gpu_observations": observations, "memory_not_latency": memory,
        "environment": {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "cuda_build": torch.version.cuda,
            "opencv": cv2.__version__, "pillow": PIL.__version__, "numpy": np.__version__,
            "pyarrow": pyarrow.__version__, "scipy": scipy.__version__,
            "gpu_name": torch.cuda.get_device_name(args.device),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "mountinfo": [line for line in Path('/proc/self/mountinfo').read_text().splitlines()
                          if any(' ' + mount + ' ' in line for mount in
                                 ('/source', '/runtime', '/inputs', '/bundle', '/reference/p1_gradient_train8.pt', '/reports'))]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "expected-bundle-sha256", "fixture-root", "reference", "calibration",
                 "geometry-contract", "source-manifest", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=["cuda:0"], default="cuda:0")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing smoke report")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        report = smoke(args)
    except Exception as error:
        report = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                  "traceback": traceback.format_exc(), "pid": os.getpid(),
                  "not_accuracy_or_latency_evaluation": True}
    report.update(argv=sys.argv, started_at=started,
                  ended_at=dt.datetime.now(dt.timezone.utc).isoformat())
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output), "pid": os.getpid(),
                      "sha256": file_sha(output)}), flush=True)
    if report["status"] != "completed_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
