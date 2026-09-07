#!/usr/bin/env python3
"""P4 LAST6000 deployment smoke and full-forward RTX 3090 benchmark.

This is a post-training diagnostic, not checkpoint selection, accuracy
evaluation, or submission.  It consumes one SHA-pinned deployment bundle and
the preregistered corrected-geometry train8 fixture.  Preprocessing, H2D and
postprocessing are reported separately; CUDA events enclose only the complete
``model(**six_inputs)`` call.  No feature cache or partial-forward API is used.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback
from typing import Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models import motiondrive_v2_inputs as adapter
from models.motiondrive_v2_serving import (
    absolute_plan_to_list, load_deployment_bundle, validate_bundle_contract,
    validate_clip_inputs,
)
from scripts.audit_motiondrive_v2 import output_checks
from scripts.audit_motiondrive_v2_deploy_inputs import (
    CALIBRATION_SHA256, FIXTURE_MANIFEST_SHA256, REFERENCE_SHA256,
    file_sha, make_reference_inputs, tree_sha, validate_identity,
    verify_raw_files,
)
from scripts.benchmark_motiondrive_v2 import summarize_ms
from scripts.export_motiondrive_v2_inference import (
    C1_SUPERVISION_SHA256, validate_complete_config,
)
from scripts.motiondrive_v2_training import tensor_state_sha256
from scripts.smoke_motiondrive_v2_deployment import (
    compare_plans, gpu_snapshot, observed_forward,
)

P4_STEP = 6000
P4_CONFIG = {"backbone_arch": "resnet50", "goal_on": True, "state_on": True,
             "motion_input_mode": "low_feature", "plan_output_scale": [10., 5.],
             "n_history": 4}
EXPECTED_ENCODINGS = [[6, 3, 432, 768], [5, 3, 216, 384]]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_int(value):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def read_pinned_bundle(path, expected_sha256):
    """Hash before and after safe tensor-only loading; never infer a SHA."""
    require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256),
            "Explicit lowercase bundle SHA256 required")
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Bundle must be an ordinary file")
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        require(digest.hexdigest() == expected_sha256, "Bundle SHA256 mismatch")
        stream.seek(0)
        bundle = torch.load(stream, map_location="cpu", weights_only=True)
        stream.seek(0)
        after = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            after.update(block)
        require(after.hexdigest() == expected_sha256, "Bundle changed while loading")
    return bundle


def validate_p4_bundle(bundle, *, expected_checkpoint_sha256,
                       expected_run_manifest_sha256):
    """Add P4 identity to the generic deployment-contract validation."""
    validate_bundle_contract(bundle)
    manifest = bundle.get("manifest", {})
    arguments = manifest.get("arguments", {})
    source = bundle.get("source_checkpoint", {})
    provenance = bundle.get("deployment_provenance", {})
    actual_steps = {"bundle": bundle.get("step"), "source": source.get("step"),
                    "checkpoint": provenance.get("checkpoint_step"),
                    "completed": provenance.get("completed_run_step"),
                    "arguments": arguments.get("steps")}
    require(all(type(value) is int and value == P4_STEP for value in actual_steps.values()),
            f"P4 requires completed LAST6000 throughout: {actual_steps}")
    require(arguments.get("phase") == "joint", "P4 deployment requires joint phase")
    config = manifest.get("model_config", {})
    # The generic validator checks only serving-related fields. Complete config
    # validation must still happen before the first CUDA API call, not later in
    # load_deployment_bundle.
    validate_complete_config(config)
    for name, expected in P4_CONFIG.items():
        actual = config.get(name)
        if name == "plan_output_scale" and isinstance(actual, tuple):
            actual = list(actual)
        if name in ("goal_on", "state_on"):
            require(type(actual) is bool and actual is expected,
                    f"P4 model_config.{name} mismatch: {actual!r}")
        else:
            require(actual == expected, f"P4 model_config.{name} mismatch: {actual!r}")
    plan_weight = manifest.get("loss_weights", {}).get("plan")
    require(type(plan_weight) in (int, float) and plan_weight == 1,
            f"P4 joint loss_weights.plan must be 1: {plan_weight!r}")
    training_git = manifest.get("git_sha")
    require(isinstance(training_git, str) and re.fullmatch(r"[0-9a-f]{40}", training_git),
            "P4 training source Git SHA missing")
    selected = provenance.get("selected_checkpoint_sha256")
    require(isinstance(selected, str) and re.fullmatch(r"[0-9a-f]{64}", selected),
            "P4 selected source checkpoint SHA missing")
    evidence = provenance.get("evidence", {})
    sidecar = evidence.get("completed_run_manifest", {})
    sidecar_sha = sidecar.get("sha256") if isinstance(sidecar, Mapping) else None
    require(isinstance(sidecar_sha, str) and re.fullmatch(r"[0-9a-f]{64}", sidecar_sha),
            "P4 completed sidecar evidence SHA missing")
    require(selected == expected_checkpoint_sha256,
            "P4 checkpoint SHA differs from the explicit external receipt")
    require(sidecar_sha == expected_run_manifest_sha256,
            "P4 completed sidecar SHA differs from the explicit external receipt")
    require(source.get("sha256") == selected
            and evidence.get("source_checkpoint", {}).get("sha256") == selected,
            "P4 selected/source/evidence checkpoint SHA mismatch")
    require(source.get("training_git_sha") == training_git
            and provenance.get("training_git_sha") == training_git,
            "P4 packaged training Git receipts disagree")
    return {"checkpoint_step": P4_STEP, "phase": "joint", "model_config":
            {name: config[name] for name in P4_CONFIG},
            "training_source_git_sha": training_git,
            "source_checkpoint_sha256": selected,
            "completed_run_manifest_sha256": sidecar_sha,
            "original_run_dir": provenance.get("run_dir"),
            "evaluation_source_git_must_equal_training_source": False}


def validate_source_manifest(manifest, root=ROOT):
    require(isinstance(manifest, Mapping) and isinstance(manifest.get("files"), Mapping)
            and bool(manifest["files"]), "Evaluation source manifest files required")
    before = {}
    for name, expected in manifest["files"].items():
        path = Path(root) / name
        require(path.is_file() and file_sha(path) == expected,
                f"Evaluation source differs from manifest: {name}")
        before[name] = expected
    source_git = manifest.get("source_git_sha") or manifest.get("observed_head")
    require(isinstance(source_git, str) and re.fullmatch(r"[0-9a-f]{40}", source_git),
            "Invalid evaluation source Git receipt")
    return before, source_git


def require_parity(comparisons):
    require(len(comparisons) == 8 and all(row.get("all_pass") is True for row in comparisons),
            "All eight corrected raw/reference input comparisons must pass")


def validate_aba(first, middle, repeated):
    repeat = compare_plans(first, repeated)
    require(repeat["pass"], "A/B/A stateless repeat changed the A output")
    return {"a_repeat": repeat, "a_vs_b_bitwise_different": not torch.equal(first, middle)}


def _all_tensor_dtypes(value):
    if isinstance(value, torch.Tensor):
        return [str(value.dtype)]
    if isinstance(value, (tuple, list)):
        return [dtype for item in value for dtype in _all_tensor_dtypes(item)]
    if isinstance(value, Mapping):
        return [dtype for item in value.values() for dtype in _all_tensor_dtypes(item)]
    return []


def precision_and_encoding_probe(model, inputs, contract, device):
    """Untimed hooks prove two R50 calls and FP32 planner internals."""
    trace = {"backbone_output_dtypes": [], "decoder_input_dtypes": [], "xy_head_input_dtypes": []}
    handles = [model.backbone_fpn.register_forward_hook(
        lambda module, args, output: trace["backbone_output_dtypes"].append(_all_tensor_dtypes(output))),
        model.planner.decoder.register_forward_pre_hook(
            lambda module, args: trace["decoder_input_dtypes"].append(_all_tensor_dtypes(args))),
        model.planner.xy_head.register_forward_pre_hook(
            lambda module, args: trace["xy_head_input_dtypes"].append(_all_tensor_dtypes(args)))]
    try:
        plan, observed = observed_forward(model, inputs, contract, device=device, precision="bf16")
    finally:
        for handle in handles:
            handle.remove()
    require(observed["backbone_input_shapes"] == EXPECTED_ENCODINGS,
            "Expected full current6 plus current-front/past4 R50 encodings")
    backbone_dtypes = [dtype for call in trace["backbone_output_dtypes"] for dtype in call]
    require(len(trace["backbone_output_dtypes"]) == 2 and backbone_dtypes,
            "Both R50 calls must produce tensor outputs")
    planner_dtypes = [dtype for group in (trace["decoder_input_dtypes"], trace["xy_head_input_dtypes"])
                      for call in group for dtype in call]
    require(planner_dtypes and all(dtype == "torch.float32" for dtype in planner_dtypes),
            f"Planner decoder/head inputs must be FP32: {planner_dtypes}")
    trace.update(backbone_input_shapes=observed["backbone_input_shapes"], image_encodings=11,
                 full_forward_count=1, plan_dtype=str(plan.dtype),
                 encoder_autocast_dtype="torch.bfloat16",
                 planner_required_dtype="torch.float32")
    return plan, trace


def measure_full_forward_samples(model, gpu_inputs, *, device, warmup, repeats):
    """CUDA-event core kept separate so ordering has a CPU mock regression test."""
    cuda_ms, wall_ms = [], []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(warmup):
            output = model(**gpu_inputs)
        torch.cuda.synchronize(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            wall_start = time.perf_counter()
            start.record()
            output = model(**gpu_inputs)
            end.record()
            torch.cuda.synchronize(device)
            wall_ms.append((time.perf_counter() - wall_start) * 1000)
            cuda_ms.append(start.elapsed_time(end))
    return output, cuda_ms, wall_ms


def timed_full_forward(model, cpu_inputs, *, device, warmup, repeats):
    """Time only complete model calls; report H2D and serialization separately."""
    require(type(warmup) is int and warmup > 0 and type(repeats) is int and repeats > 0,
            "warmup/repeats must be positive integers")
    device = torch.device(device)
    require(device.type == "cuda" and device.index is not None,
            "P4 timing requires an explicit CUDA device")
    validate_clip_inputs(cpu_inputs, adapter.input_contract())
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    gpu_inputs = {name: value.to(device=device, non_blocking=False)
                  for name, value in cpu_inputs.items()}
    torch.cuda.synchronize(device)
    h2d_ms = (time.perf_counter() - started) * 1000
    output, cuda_ms, wall_ms = measure_full_forward_samples(
        model, gpu_inputs, device=device, warmup=warmup, repeats=repeats)
    checks = output_checks(output, 1)
    require(all(checks.values()), "Complete forward output checks failed")
    post_started = time.perf_counter()
    plan = output["plan_abs"].detach()[0].to(device="cpu").contiguous()
    encoded = absolute_plan_to_list(plan)
    post_ms = (time.perf_counter() - post_started) * 1000
    return {"cuda_samples_ms": cuda_ms, "wall_samples_ms": wall_ms,
            "cuda": summarize_ms(cuda_ms), "wall": summarize_ms(wall_ms),
            "warmup": warmup, "repeats": repeats, "h2d_ms": h2d_ms,
            "postprocess_d2h_and_json_ms": post_ms, "output_checks": checks,
            "plan_abs": encoded,
            "timing_boundary": "CUDA events enclose only complete model(**six_inputs); preprocessing/H2D/D2H/JSON excluded"}


def run(args):
    require(not torch.cuda.is_initialized(), "Start fresh; validate CPU artifacts before CUDA")
    require(args.device == "cuda:0" and args.precision == "bf16",
            "P4 RTX3090 protocol is fixed to cuda:0 and BF16 encoder/FP32 planner")
    paths = {"bundle": Path(args.bundle), "reference": Path(args.reference),
             "raw_manifest": Path(args.fixture_root) / "fixture_manifest.json",
             "calibration": Path(args.calibration), "geometry_contract": Path(args.geometry_contract),
             "source_manifest": Path(args.source_manifest)}
    pinned = {"bundle": args.expected_bundle_sha256, "reference": REFERENCE_SHA256,
              "raw_manifest": FIXTURE_MANIFEST_SHA256, "calibration": CALIBRATION_SHA256,
              "geometry_contract": C1_SUPERVISION_SHA256}
    before = {name: file_sha(path) for name, path in paths.items()}
    require(all(before[name] == expected for name, expected in pinned.items()),
            f"Pinned P4 artifacts differ: {before}")
    bundle = read_pinned_bundle(paths["bundle"], args.expected_bundle_sha256)
    p4 = validate_p4_bundle(
        bundle, expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_run_manifest_sha256=args.expected_run_manifest_sha256)
    source_manifest = json.loads(paths["source_manifest"].read_text())
    source_before, evaluation_git = validate_source_manifest(source_manifest)
    reference = torch.load(paths["reference"], map_location="cpu", weights_only=True)
    reference_before = tree_sha(reference)
    raw_manifest = json.loads(paths["raw_manifest"].read_text())
    clips = validate_identity(reference, raw_manifest)
    raw_before = verify_raw_files(args.fixture_root, clips)
    with np.load(paths["calibration"], allow_pickle=False) as archive:
        calibration = torch.from_numpy(archive["lidar2img"].copy())
    expected_inputs, adaptation = make_reference_inputs(reference["batch"], calibration)
    contract = adapter.input_contract()
    prepared, reference_inputs, comparisons, preprocessing_ms = [], [], [], []
    adapter.cv2.setNumThreads(1)
    for index, clip in enumerate(clips):
        started = time.perf_counter()
        item = adapter.prepare_clip_inputs(Path(args.fixture_root) / clip["clip_id"])
        preprocessing_ms.append((time.perf_counter() - started) * 1000)
        require(item.metadata["input_contract"] == contract, "Adapter input contract changed")
        expected = {name: value[index:index + 1] for name, value in expected_inputs.items()}
        comparison = adapter.compare_input_fixture(item, {"inputs": expected,
            "metadata": {"input_contract": contract}}, projection_atol=1e-4, pose_atol=1e-5)
        comparisons.append(comparison)
        prepared.append(item)
        reference_inputs.append(expected)
    require_parity(comparisons)

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    gpu_observations = [gpu_snapshot(require_idle=True)]
    model, loaded_contract = load_deployment_bundle(paths["bundle"],
        expected_bundle_sha256=args.expected_bundle_sha256, device=args.device)
    require(loaded_contract == contract, "Bundle and adapter contracts differ")
    state_before = tensor_state_sha256(model.state_dict())
    _, precision_trace = precision_and_encoding_probe(model, prepared[0].inputs, contract, args.device)
    untimed_calls, output_parity = [], []
    for clip, item, expected in zip(clips, prepared, reference_inputs):
        raw_plan, raw_call = observed_forward(
            model, item.inputs, contract, device=args.device, precision="bf16")
        reference_plan, reference_call = observed_forward(
            model, expected, contract, device=args.device, precision="bf16")
        comparison = compare_plans(raw_plan, reference_plan)
        require(comparison["pass"],
                f"Raw/reference complete-forward parity failed: {clip['clip_id']}")
        untimed_calls.append({"clip_id": clip["clip_id"], "raw": raw_call,
                              "reference": reference_call})
        output_parity.append({"clip_id": clip["clip_id"], **comparison})
    a1, _ = observed_forward(model, prepared[0].inputs, contract, device=args.device, precision="bf16")
    b, _ = observed_forward(model, prepared[1].inputs, contract, device=args.device, precision="bf16")
    a2, _ = observed_forward(model, prepared[0].inputs, contract, device=args.device, precision="bf16")
    aba = validate_aba(a1, b, a2)

    torch.cuda.reset_peak_memory_stats(args.device)
    clip_timings = []
    for clip, item in zip(clips, prepared):
        timing = timed_full_forward(model, item.inputs, device=args.device,
                                    warmup=args.warmup, repeats=args.repeats)
        clip_timings.append({"clip_id": clip["clip_id"], "source_scene": clip["source_scene"],
                             "source_frame": clip["source_frame"], **timing})
    all_cuda = [value for row in clip_timings for value in row["cuda_samples_ms"]]
    all_wall = [value for row in clip_timings for value in row["wall_samples_ms"]]
    state_after = tensor_state_sha256(model.state_dict())
    require(state_before == state_after, "Model state changed across P4 smoke/benchmark")
    gpu_observations.append(gpu_snapshot())
    after = {name: file_sha(path) for name, path in paths.items()}
    source_after = {name: file_sha(ROOT / name) for name in source_before}
    raw_after = verify_raw_files(args.fixture_root, clips)
    require(before == after and source_before == source_after and raw_before == raw_after
            and reference_before == tree_sha(reference), "Immutable P4 input/source changed")
    return {"schema_version": 1, "status": "completed_pass", "pid": os.getpid(),
        "purpose": "P4 LAST6000 corrected-train8 deployment smoke and RTX3090 full-forward timing",
        "not_accuracy_evaluation": True, "not_checkpoint_selection": True,
        "not_the_old_p1_32ms_measurement": True, "p4_measurement_completed": True,
        "p4_bundle_receipt": p4,
        "evaluation_source": {"manifest": source_manifest, "git_sha": evaluation_git,
                              "must_equal_training_git": False,
                              "sha256_before": source_before, "sha256_after": source_after},
        "input_contract": contract, "reference_adaptation": adaptation,
        "fixture": {"manifest_sha256": FIXTURE_MANIFEST_SHA256, "clip_count": 8,
                    "all_parity_pass": True, "comparisons": comparisons,
                    "preprocessing_wall_ms": preprocessing_ms,
                    "preprocessing_summary_ms": summarize_ms(preprocessing_ms)},
        "untimed_full_forward_checks": untimed_calls,
        "raw_reference_output_parity": output_parity,
        "precision_and_encoding_probe": precision_trace,
        "aba_stateless": aba, "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "timing_protocol": {"warmup_per_clip": args.warmup, "repeats_per_clip": args.repeats,
            "clip_count": 8, "total_warmup_full_forwards": 8 * args.warmup,
            "total_timed_full_forwards": 8 * args.repeats,
            "aggregation": "per-clip summaries plus pooled equal-count samples over all eight preregistered clips",
            "model_forward_only": True, "preprocessing_h2d_postprocessing_excluded": True,
            "feature_cache_used": False, "batch_size": 1},
        "pooled_timing": {"cuda": summarize_ms(all_cuda), "wall": summarize_ms(all_wall),
                          "cuda_samples_ms": all_cuda, "wall_samples_ms": all_wall},
        "clip_timings": clip_timings,
        "memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device)},
        "gpu_observations": gpu_observations,
        "artifacts_sha256_before": before, "artifacts_sha256_after": after,
        "raw_files_before": raw_before, "raw_files_after": raw_after,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(args.device),
                        "precision": "bf16", "planner_precision": "fp32",
                        "tf32": False, "deterministic_algorithms": True}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("bundle", "expected-bundle-sha256", "expected-checkpoint-sha256",
                 "expected-run-manifest-sha256", "fixture-root", "reference", "calibration",
                 "geometry-contract", "source-manifest", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--repeats", type=positive_int, default=50)
    args = parser.parse_args(argv)
    output = Path(args.output).absolute()
    if os.path.lexists(output) or not output.parent.is_dir():
        raise FileExistsError("P4 report must be a new file in an existing directory")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        report = run(args)
    except Exception as error:
        report = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                  "traceback": traceback.format_exc(), "pid": os.getpid(),
                  "not_accuracy_evaluation": True, "p4_measurement_completed": False}
    report.update(argv=sys.argv if argv is None else [str(Path(__file__)), *argv],
                  started_at=started, ended_at=dt.datetime.now(dt.timezone.utc).isoformat())
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "pid": os.getpid(), "sha256": file_sha(output)}), flush=True)
    return 0 if report["status"] == "completed_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
