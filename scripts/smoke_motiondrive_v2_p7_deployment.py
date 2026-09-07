#!/usr/bin/env python3
"""P7 zero-init assembly smoke and RTX3090 full-forward timing.

This canary deterministically assembles one reviewed P7 C/G graph from its
SHA-pinned own-seed P0 LAST2000 on CPU, then times only complete batch-one
model calls on the fixed corrected-geometry train8 fixture. It does not create
a deployment bundle, read final, select a checkpoint, or claim learned goal use.
"""
from __future__ import annotations

import argparse
import datetime as dt
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
from models.motiondrive_v2 import cross_cell_goal_residual as p7_residual
from scripts.audit_motiondrive_v2_deploy_inputs import (
    CALIBRATION_SHA256, FIXTURE_MANIFEST_SHA256, REFERENCE_SHA256,
    file_sha, make_reference_inputs, tree_sha, validate_identity, verify_raw_files,
)
from scripts.benchmark_motiondrive_v2 import summarize_ms
from scripts.export_motiondrive_v2_inference import C1_SUPERVISION_SHA256
from scripts.motiondrive_v2_training import tensor_state_sha256
from scripts.run_motiondrive_v2_p7_goal_routing import (
    ARM_TO_MODE, _build_initialized_model, validate_p0,
    validate_source_manifest as validate_training_source_manifest,
)
from scripts.smoke_motiondrive_v2_deployment import compare_plans, gpu_snapshot, observed_forward
from scripts.smoke_motiondrive_v2_p4_deployment import (
    positive_int, require, require_parity, timed_full_forward,
)

EXPECTED_GPU_UUID = "GPU-8768d7a1-6e1b-1aae-4774-248c9d28b887"
P0 = {
    0: {
        "file_sha256": "8e91c30947a040f267f01f0642db3f2725900545deeb6d4087c795b5eb398a0c",
        "model_state_sha256": "f69e1c52ccd4b9908130ffe02170c0f08bff21712c634fee064517009a6c2c7d",
        "manifest_sha256": "9e5ba7ce89c0c1a9e4fe49476ff72c63b451f2658766e74e84c7336a4591fd98",
    },
    1: {
        "file_sha256": "6067ba9cc543e6e1be837d85883696bc7ad12c3da9b970edba3008e30eff83c5",
        "model_state_sha256": "d595f0788fbb5ae31ea2eaa2f3344ac300028756dcfe8f313ec1dbb761673e0d",
        "manifest_sha256": "385ea53269e96d73c7d66f4196bc2e1205b0c6b7e9c7db537f983bc7aa37c421",
    },
}
EXPECTED_ENCODINGS = [[6, 3, 432, 768], [5, 3, 216, 384]]
TRAINING_SOURCE_FILES = {
    "models/motiondrive_v2/__init__.py", "models/motiondrive_v2/config.py",
    "models/motiondrive_v2/cross_cell_goal_residual.py", "models/motiondrive_v2/model.py",
    "models/motiondrive_v2/motion_encoder.py", "models/motiondrive_v2/planner.py",
    "models/motiondrive_v2/scene_encoder.py", "scripts/build_grouped_split_v2.py",
    "scripts/motiondrive_v2_data.py", "scripts/motiondrive_v2_training.py",
    "scripts/run_motiondrive_v2_p7_goal_routing.py", "scripts/sparse_scoredrive.py",
    "scripts/train_motiondrive_v2.py",
}
VALIDATION_SOURCE_FILES = {
    "models/__init__.py", "models/motiondrive_v2/__init__.py",
    "models/motiondrive_v2/config.py", "models/motiondrive_v2/cross_cell_goal_residual.py",
    "models/motiondrive_v2/model.py", "models/motiondrive_v2/motion_encoder.py",
    "models/motiondrive_v2/planner.py", "models/motiondrive_v2/scene_encoder.py",
    "models/motiondrive_v2_serving.py",
    "models/motiondrive_v2_input_contract.py", "models/motiondrive_v2_inputs.py",
    "scripts/audit_motiondrive_v2.py", "scripts/audit_motiondrive_v2_deploy_inputs.py",
    "scripts/benchmark_motiondrive_v2.py", "scripts/build_grouped_split_v2.py",
    "scripts/export_motiondrive_v2_inference.py", "scripts/motiondrive_v2_data.py",
    "scripts/motiondrive_v2_training.py",
    "scripts/run_motiondrive_v2_p7_goal_routing.py", "scripts/smoke_motiondrive_v2_deployment.py",
    "scripts/smoke_motiondrive_v2_p4_deployment.py",
    "scripts/smoke_motiondrive_v2_p7_deployment.py", "scripts/sparse_scoredrive.py",
    "scripts/train_motiondrive_v2.py",
}


def lowercase_hash(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def canonical_gpu_uuid(value: object) -> str:
    """Normalize NVIDIA's optional display prefix, never device identity."""
    require(isinstance(value, str), "GPU UUID must be a string")
    body = value[4:] if value.startswith("GPU-") else value
    require(re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", body) is not None,
            "Malformed GPU UUID")
    return body.lower()


def require_expected_gpu(snapshot: Mapping, expected_uuid: str) -> dict:
    require(canonical_gpu_uuid(expected_uuid) == canonical_gpu_uuid(EXPECTED_GPU_UUID),
            "CLI expected GPU differs from the preregistered RTX3090")
    require(canonical_gpu_uuid(snapshot.get("gpu_uuid")) == canonical_gpu_uuid(expected_uuid),
            "Observed device differs from the exact expected RTX3090")
    result = dict(snapshot)
    result.update(canonical_gpu_uuid=canonical_gpu_uuid(snapshot["gpu_uuid"]),
                  expected_gpu_uuid=EXPECTED_GPU_UUID)
    return result


def validate_validation_source_manifest(path: str | Path, expected_sha256: str) -> tuple[dict, dict]:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Validation source manifest must be ordinary")
    require(lowercase_hash(expected_sha256, 64) and file_sha(path) == expected_sha256,
            "Validation source manifest SHA mismatch")
    manifest = json.loads(path.read_text())
    require(set(manifest) == {"schema_version", "git_sha", "file_sha256"}
            and manifest["schema_version"] == 1 and lowercase_hash(manifest["git_sha"], 40),
            "Validation source manifest schema mismatch")
    files = manifest["file_sha256"]
    require(isinstance(files, dict) and set(files) == VALIDATION_SOURCE_FILES
            and all(isinstance(name, str) and lowercase_hash(value, 64)
                    for name, value in files.items()),
            "Validation source closure must be the exact runtime set")
    before = {name: file_sha(ROOT / name) for name in files}
    require(before == files, "Staged validation source differs from manifest")
    return manifest, before


def assemble_p7(args) -> tuple[torch.nn.Module, dict]:
    """Validate own P0 and construct the reviewed zero-init P7 graph on CPU."""
    require(args.expected_init_sha256 == P0[args.base_seed]["file_sha256"]
            and args.expected_run_manifest_sha256 == P0[args.base_seed]["manifest_sha256"],
            "CLI P0 file/sidecar SHA does not match selected base")
    source = validate_training_source_manifest(
        args.training_source_manifest, args.expected_training_source_manifest_sha256)
    require(set(source["file_sha256"]) == TRAINING_SOURCE_FILES,
            "Training source manifest does not contain exact P7 closure")
    payload, p0 = validate_p0(args.init, args.expected_init_sha256, args.run_manifest,
                              args.expected_run_manifest_sha256, args.base_seed)
    require(p0["model_state_sha256"] == P0[args.base_seed]["model_state_sha256"],
            "Own-P0 model tensor SHA mismatch")
    mode = ARM_TO_MODE[args.arm]
    model, missing, initial_sha = _build_initialized_model(payload, args.base_seed, mode)
    other_mode = "real" if mode == "zero" else "zero"
    paired, paired_missing, paired_sha = _build_initialized_model(payload, args.base_seed, other_mode)
    require(missing == paired_missing and initial_sha == paired_sha,
            "P7 C/G deterministic initial state differs")
    require(initial_sha == args.expected_initial_model_state_sha256,
            "P7 assembled model-state SHA differs from CPU preregistration")
    require(model.config.cross_cell_goal_mode == mode and model.config.goal_on is True
            and model.config.state_on is True and model.config.backbone_arch == "resnet50"
            and model.config.motion_input_mode == "low_feature"
            and list(model.config.plan_output_scale) == [10.0, 5.0],
            "P7 assembled complete config mismatch")
    branch = model.scene_encoder.cross_cell_goal_residual
    require(branch is not None and branch.output.bias is None
            and not bool(branch.output.weight.count_nonzero()), "P7 zero-init branch mismatch")
    require(branch.COSINE_SCALE == 8.0
            and torch.equal(branch.sigma_m.detach().cpu(),
                            torch.tensor([10.0, 32.0 / 3.0], dtype=torch.float32)),
            "P7 attention scale/sigma mismatch")
    require(all(parameter.requires_grad for parameter in model.parameters()),
            "P7 assembled joint parameters must remain trainable")
    del paired, payload
    model.eval()
    return model, {"arm": args.arm, "mode": mode, "base_seed": args.base_seed,
                   "p0": p0, "training_source": source,
                   "expected_missing_keys": missing,
                   "initial_model_state_sha256": initial_sha,
                   "paired_other_mode_initial_model_state_sha256": paired_sha,
                   "weights_only_p0_init": True, "fresh_optimizer_created": False,
                   "zero_output_weight": True}


def branch_execution_probe(model, inputs: Mapping, contract: Mapping, *, arm: str,
                           device: str) -> tuple[torch.Tensor, dict]:
    """Observe untimed module/function calls and dtypes, never kernel identity."""
    branch = getattr(getattr(model, "scene_encoder", None), "cross_cell_goal_residual", None)
    require(branch is not None, "P7 branch must be enabled")
    calls, children = [], {name: [] for name in
                           ("query_image", "key_image", "value_image", "position", "output")}
    contractions = {"bqd,bkd->bqk": [], "bqk,bkd->bqd": []}
    softmax_calls = []

    def before(module, args):
        require(len(args) == 3, "P7 branch must receive scene, goal and visibility")
        scene, goal, visible = args
        expected_goal = inputs["goal_xy"] if arm == "goal_real_slot" else torch.zeros_like(inputs["goal_xy"])
        calls.append({"scene_shape": list(scene.shape), "scene_dtype": str(scene.dtype),
                      "goal_shape": list(goal.shape), "goal_dtype": str(goal.dtype),
                      "goal_matches_arm_input": bool(torch.equal(goal.detach().cpu(), expected_goal)),
                      "visible_shape": list(visible.shape), "visible_dtype": str(visible.dtype),
                      "visible_count": int(visible.sum().item()),
                      "outer_cuda_autocast_enabled": bool(torch.is_autocast_enabled("cuda"))})

    def after(module, args, output):
        require(isinstance(output, torch.Tensor) and tuple(output.shape) == (1, 3072, 128)
                and bool(torch.isfinite(output).all()), "P7 branch output invalid")
        calls[-1].update(output_shape=list(output.shape), output_dtype=str(output.dtype))

    def child_after(name):
        def hook(module, args, output):
            require(isinstance(output, torch.Tensor) and bool(torch.isfinite(output).all()),
                    f"P7 branch {name} output invalid")
            children[name].append({"shape": list(output.shape), "dtype": str(output.dtype)})
        return hook

    original_einsum = torch.einsum
    original_masked_softmax = p7_residual.masked_softmax

    def observed_einsum(equation, *operands):
        output = original_einsum(equation, *operands)
        if equation in contractions:
            contractions[equation].append({
                "operand_shapes": [list(value.shape) for value in operands],
                "operand_dtypes": [str(value.dtype) for value in operands],
                "output_shape": list(output.shape), "output_dtype": str(output.dtype),
                "cuda_autocast_enabled": bool(torch.is_autocast_enabled("cuda")),
            })
        return output

    def observed_masked_softmax(scores, valid, dim=-1):
        output = original_masked_softmax(scores, valid, dim=dim)
        softmax_calls.append({
            "score_shape": list(scores.shape), "score_dtype": str(scores.dtype),
            "valid_shape": list(valid.shape), "valid_dtype": str(valid.dtype),
            "output_shape": list(output.shape), "output_dtype": str(output.dtype),
            "dim": int(dim), "cuda_autocast_enabled": bool(torch.is_autocast_enabled("cuda")),
        })
        return output

    handles = [branch.register_forward_pre_hook(before), branch.register_forward_hook(after)]
    handles += [getattr(branch, name).register_forward_hook(child_after(name)) for name in children]
    torch.einsum = observed_einsum
    p7_residual.masked_softmax = observed_masked_softmax
    try:
        plan, full = observed_forward(model, inputs, contract, device=device, precision="bf16")
    finally:
        torch.einsum = original_einsum
        p7_residual.masked_softmax = original_masked_softmax
        for handle in handles:
            handle.remove()
    counts = {name: len(rows) for name, rows in children.items()}
    require(len(calls) == 1 and calls[0]["goal_matches_arm_input"],
            "P7 branch call/arm goal evidence mismatch")
    require(counts == {"query_image": 1, "key_image": 1, "value_image": 1,
                       "position": 2, "output": 1},
            f"P7 branch component call evidence mismatch: {counts}")
    require(all(row["dtype"] == "torch.float32" for rows in children.values() for row in rows),
            f"P7 residual projections must execute in FP32: {children}")
    require({name: len(rows) for name, rows in contractions.items()}
            == {"bqd,bkd->bqk": 1, "bqk,bkd->bqd": 1},
            f"P7 residual contraction evidence mismatch: {contractions}")
    require(all(dtype == "torch.float32" for rows in contractions.values() for row in rows
                for dtype in [*row["operand_dtypes"], row["output_dtype"]]),
            f"P7 residual contractions must execute in FP32: {contractions}")
    require(len(softmax_calls) == 1
            and softmax_calls[0]["score_dtype"] == "torch.float32"
            and softmax_calls[0]["valid_dtype"] == "torch.bool"
            and softmax_calls[0]["output_dtype"] == "torch.float32",
            f"P7 masked softmax must execute in FP32: {softmax_calls}")
    require(calls[0]["output_dtype"] == calls[0]["scene_dtype"],
            "P7 residual must return the original scene dtype")
    require(full.get("image_encodings") == 11
            and full.get("backbone_input_shapes") == EXPECTED_ENCODINGS,
            "P7 probe must execute complete 11-encoding model")
    return plan, {"module_execution_observed": True, "module_call_count": 1,
                  "component_module_calls": children, "call": calls[0],
                  "contraction_function_calls": contractions,
                  "masked_softmax_function_calls": softmax_calls,
                  "full_forward": full, "timing_measured": False,
                  "cuda_kernel_identity_observed": False,
                  "fp32_projection_dtype_observed": True,
                  "fp32_contraction_dtype_observed": True,
                  "fp32_score_softmax_dtype_observed": True,
                  "instrumentation_removed_before_timing": True,
                  "zero_output_weight_does_not_authorize_branch_bypass": True,
                  "learned_goal_use_or_accuracy_proven": False}


def latency_gate(summary: Mapping) -> dict:
    median, p95 = summary.get("median_ms"), summary.get("p95_ms")
    require(type(median) is float and type(p95) is float
            and np.isfinite([median, p95]).all(), "Finite float median/p95 required")
    result = {"threshold_ms_exclusive": 100.0,
              "median_below_100ms": median < 100.0,
              "p95_below_100ms": p95 < 100.0}
    result["pass"] = result["median_below_100ms"] and result["p95_below_100ms"]
    return result


def run(args) -> dict:
    require(not torch.cuda.is_initialized(), "Start fresh; CPU validation must precede CUDA")
    require(args.device == "cuda:0" and args.precision == "bf16"
            and args.warmup == 20 and args.repeats == 50,
            "P7 RTX3090 protocol requires cuda:0, BF16, warmup20 and repeat50")
    require(canonical_gpu_uuid(args.expected_physical_gpu_uuid)
            == canonical_gpu_uuid(EXPECTED_GPU_UUID), "CLI GPU UUID is not fixed RTX3090")
    paths = {"init": Path(args.init), "run_manifest": Path(args.run_manifest),
             "training_source_manifest": Path(args.training_source_manifest),
             "validation_source_manifest": Path(args.validation_source_manifest),
             "reference": Path(args.reference),
             "raw_manifest": Path(args.fixture_root) / "fixture_manifest.json",
             "calibration": Path(args.calibration), "geometry_contract": Path(args.geometry_contract)}
    pinned = {"init": args.expected_init_sha256,
              "run_manifest": args.expected_run_manifest_sha256,
              "training_source_manifest": args.expected_training_source_manifest_sha256,
              "validation_source_manifest": args.expected_validation_source_manifest_sha256,
              "reference": REFERENCE_SHA256, "raw_manifest": FIXTURE_MANIFEST_SHA256,
              "calibration": CALIBRATION_SHA256, "geometry_contract": C1_SUPERVISION_SHA256}
    require(all(path.is_file() and not path.is_symlink() for path in paths.values()),
            "Every P7 canary input must be an ordinary non-symlink file")
    before = {name: file_sha(path) for name, path in paths.items()}
    require(before == pinned, f"Pinned P7 canary artifacts differ: {before}")
    validation_manifest, validation_before = validate_validation_source_manifest(
        paths["validation_source_manifest"], args.expected_validation_source_manifest_sha256)
    model, assembly = assemble_p7(args)

    reference = torch.load(paths["reference"], map_location="cpu", weights_only=True)
    reference_before = tree_sha(reference)
    raw_manifest = json.loads(paths["raw_manifest"].read_text())
    clips = validate_identity(reference, raw_manifest)
    raw_before = verify_raw_files(args.fixture_root, clips)
    require(len(clips) == 8, "P7 requires fixed eight-clip fixture")
    with np.load(paths["calibration"], allow_pickle=False) as archive:
        calibration = torch.from_numpy(archive["lidar2img"].copy())
    expected_inputs, adaptation = make_reference_inputs(reference["batch"], calibration)
    contract = adapter.input_contract()
    prepared, reference_inputs, comparisons, preprocessing_ms = [], [], [], []
    adapter.cv2.setNumThreads(1)
    for index, clip in enumerate(clips):
        started = time.perf_counter()
        item = adapter.prepare_clip_inputs(Path(args.fixture_root) / clip["clip_id"])
        preprocessing_ms.append((time.perf_counter() - started) * 1000.0)
        require(item.metadata["input_contract"] == contract, "Adapter input contract changed")
        expected = {name: value[index:index + 1] for name, value in expected_inputs.items()}
        comparison = adapter.compare_input_fixture(
            item, {"inputs": expected, "metadata": {"input_contract": contract}},
            projection_atol=1e-4, pose_atol=1e-5)
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
    float32_matmul_precision = torch.get_float32_matmul_precision()
    gpu_observations = [require_expected_gpu(
        gpu_snapshot(require_idle=True), args.expected_physical_gpu_uuid)]
    model.to(args.device).eval()
    state_before = tensor_state_sha256(model.state_dict())
    require(state_before == args.expected_initial_model_state_sha256,
            "CPU-to-GPU transfer changed assembled P7 state")
    _, branch_probe = branch_execution_probe(
        model, prepared[0].inputs, contract, arm=args.arm, device=args.device)

    parity = []
    for clip, item, expected in zip(clips, prepared, reference_inputs):
        raw_plan, raw_call = observed_forward(model, item.inputs, contract,
                                              device=args.device, precision="bf16")
        reference_plan, reference_call = observed_forward(model, expected, contract,
                                                          device=args.device, precision="bf16")
        comparison = compare_plans(raw_plan, reference_plan)
        require(comparison["pass"], f"P7 raw/reference output mismatch: {clip['clip_id']}")
        parity.append({"clip_id": clip["clip_id"], **comparison,
                       "raw_call": raw_call, "reference_call": reference_call})
    a1, _ = observed_forward(model, prepared[0].inputs, contract, device=args.device, precision="bf16")
    b, _ = observed_forward(model, prepared[1].inputs, contract, device=args.device, precision="bf16")
    a2, _ = observed_forward(model, prepared[0].inputs, contract, device=args.device, precision="bf16")
    aba = {"a_repeat": compare_plans(a1, a2), "a_vs_b_bitwise_different": not torch.equal(a1, b)}
    require(aba["a_repeat"]["pass"], "P7 A/B/A stateless repeat failed")

    torch.cuda.reset_peak_memory_stats(args.device)
    clip_timings = []
    for clip, item in zip(clips, prepared):
        timing = timed_full_forward(model, item.inputs, device=args.device,
                                    warmup=args.warmup, repeats=args.repeats)
        clip_timings.append({"clip_id": clip["clip_id"], "source_scene": clip["source_scene"],
                             "source_frame": clip["source_frame"], **timing})
    all_cuda = [sample for row in clip_timings for sample in row["cuda_samples_ms"]]
    all_wall = [sample for row in clip_timings for sample in row["wall_samples_ms"]]
    cuda_summary, wall_summary = summarize_ms(all_cuda), summarize_ms(all_wall)

    state_after = tensor_state_sha256(model.state_dict())
    require(state_before == state_after, "P7 model state changed during canary")
    gpu_observations.append(require_expected_gpu(gpu_snapshot(), args.expected_physical_gpu_uuid))
    after = {name: file_sha(path) for name, path in paths.items()}
    validation_after = {name: file_sha(ROOT / name) for name in validation_before}
    raw_after = verify_raw_files(args.fixture_root, clips)
    require(before == after and validation_before == validation_after
            and raw_before == raw_after and reference_before == tree_sha(reference),
            "P7 immutable source/input changed during canary")

    return {"schema_version": 1, "status": "completed_measurement", "pid": os.getpid(),
            "purpose": "P7 zero-init own-P0 RTX3090 full-forward latency/parity canary",
            "not_learned_p7_checkpoint": True, "learned_goal_use_or_accuracy_proven": False,
            "not_checkpoint_selection": True, "official_submission_performed": False,
            "final_holdout_accessed": False, "assembly": assembly,
            "validation_source": {"manifest": validation_manifest,
                                  "manifest_sha256": args.expected_validation_source_manifest_sha256,
                                  "sha256_before": validation_before, "sha256_after": validation_after},
            "input_contract": contract, "reference_adaptation": adaptation,
            "fixture": {"manifest_sha256": FIXTURE_MANIFEST_SHA256, "clip_count": 8,
                        "all_input_parity_pass": True, "comparisons": comparisons,
                        "preprocessing_wall_ms": preprocessing_ms},
            "branch_execution_probe": branch_probe, "raw_reference_output_parity": parity,
            "aba_stateless": aba, "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "timing_protocol": {"batch_size": 1, "image_encodings": 11,
                "warmup_per_clip": 20, "repeats_per_clip": 50, "clip_count": 8,
                "total_warmup_full_forwards": 160, "total_timed_full_forwards": 400,
                "model_forward_only": True, "feature_cache_used": False,
                "checkpoint_config_preprocessing_h2d_d2h_postprocessing_excluded": True,
                "cuda_events_and_synchronize": True},
            "pooled_timing": {"cuda": cuda_summary, "wall": wall_summary,
                              "cuda_samples_ms": all_cuda, "wall_samples_ms": all_wall},
            "timing_gate": latency_gate(cuda_summary), "clip_timings": clip_timings,
            "memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device)},
            "gpu_observations": gpu_observations,
            "artifacts_sha256_before": before, "artifacts_sha256_after": after,
            "raw_files_before": raw_before, "raw_files_after": raw_after,
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "cuda_build": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(args.device),
                            "encoder_precision": "bf16", "planner_precision": "fp32",
                            "new_residual_projection_dtype_observed": "fp32",
                            "tf32_cuda_matmul": torch.backends.cuda.matmul.allow_tf32,
                            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                            "float32_matmul_precision": float32_matmul_precision,
                            "deterministic_algorithms": True}}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("init", "expected-init-sha256", "run-manifest",
                 "expected-run-manifest-sha256", "expected-initial-model-state-sha256",
                 "training-source-manifest", "expected-training-source-manifest-sha256",
                 "validation-source-manifest", "expected-validation-source-manifest-sha256",
                 "fixture-root", "reference", "calibration", "geometry-contract", "output",
                 "expected-physical-gpu-uuid"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--arm", choices=tuple(ARM_TO_MODE), required=True)
    parser.add_argument("--base-seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--repeats", type=positive_int, default=50)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = arguments(argv)
    for name in ("expected_init_sha256", "expected_run_manifest_sha256",
                 "expected_initial_model_state_sha256",
                 "expected_training_source_manifest_sha256",
                 "expected_validation_source_manifest_sha256"):
        require(lowercase_hash(getattr(args, name), 64), f"Malformed required SHA: {name}")
    output = Path(args.output).absolute()
    if os.path.lexists(output) or not output.parent.is_dir():
        raise FileExistsError("P7 report must be a new file in an existing directory")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        report = run(args)
    except Exception as error:
        report = {"schema_version": 1, "status": "failed",
                  "error_type": type(error).__name__, "error": str(error),
                  "traceback": traceback.format_exc(), "pid": os.getpid(),
                  "not_accuracy_evaluation": True, "final_holdout_accessed": False,
                  "timing_gate_pass": False}
    report.update(argv=sys.argv if argv is None else [str(Path(__file__)), *argv],
                  started_at=started, ended_at=dt.datetime.now(dt.timezone.utc).isoformat())
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "pid": os.getpid(), "sha256": file_sha(output)}), flush=True)
    return 0 if report["status"] == "completed_measurement" else 1


if __name__ == "__main__":
    raise SystemExit(main())
