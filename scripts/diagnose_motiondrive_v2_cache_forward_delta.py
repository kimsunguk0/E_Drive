#!/usr/bin/env python3
"""Isolate evaluator-vs-cache first-batch forward differences without mutation.

This bounded diagnostic compares one exact tune batch4 across the immutable
evaluator constructor and the cache direct constructor.  It separates input
layout, constructor metadata, requires-grad freezing, and an xy_head pre-hook.
It never changes a checkpoint/runtime file and never accesses final validation.
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_motiondrive_v2 import construct_model
from cache_motiondrive_v2_zero_selector_features import (
    CALIBRATION_SHA256, P4_CHECKPOINT_SHA256, P4_GIT_SHA,
    P4_RUN_MANIFEST_SHA256, SPLIT_SHA256, SUPERVISION_SHA256,
    file_sha, forward_with_decoded, load_frozen_model,
    read_pinned_json, validate_source_manifest,
)
from evaluate_motiondrive_v2_planning import (
    ImageCounterfactualDataset, planning_model_inputs,
)
from motiondrive_v2_data import MotionDriveDataset
from motiondrive_v2_training import MODEL_INPUTS, model_inputs, tensor_state_sha256, to_device, weighted_d3
from train_motiondrive_v2 import autocast as training_autocast

BATCH = 4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update((str(value.dtype) + "\0" + str(tuple(value.shape)) + "\0").encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_receipt(value):
    require(isinstance(value, torch.Tensor), "Tensor receipt requires a tensor")
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "contiguous": value.is_contiguous(),
            "finite": bool(torch.isfinite(value).all()) if value.is_floating_point() else None,
            "tensor_sha256": tensor_sha(value)}


def tensor_delta(left, right):
    require(isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor),
            "Tensor comparison requires tensors")
    same_contract = left.shape == right.shape and left.dtype == right.dtype
    exact = same_contract and torch.equal(left, right)
    result = {"same_shape_dtype": bool(same_contract), "bitwise_equal": bool(exact),
              "left": tensor_receipt(left), "right": tensor_receipt(right)}
    if same_contract and left.is_floating_point():
        cpu_left, cpu_right = left.detach().cpu(), right.detach().cpu()
        difference = (cpu_left.float() - cpu_right.float()).abs()
        mask = cpu_left != cpu_right
        first = torch.nonzero(mask, as_tuple=False)[0].tolist() if mask.any() else None
        result.update(max_abs=float(difference.max()), different_elements=int(mask.sum()),
                      first_mismatch_index=first,
                      left_at_first=(float(cpu_left[tuple(first)]) if first is not None else None),
                      right_at_first=(float(cpu_right[tuple(first)]) if first is not None else None))
    return result


def batch_receipt(batch):
    return {key: tensor_receipt(batch[key]) for key in (*MODEL_INPUTS, "gt_plan", "plan_valid")}


def compare_batches(left, right):
    compared = {key: tensor_delta(left[key], right[key]) for key in (*MODEL_INPUTS, "gt_plan", "plan_valid")}
    identities = {}
    for key in ("row", "frame"):
        identities[key] = tensor_delta(left[key], right[key])
    for key in ("scenario", "session_id"):
        identities[key] = {"bitwise_equal": list(left[key]) == list(right[key]),
                           "left": list(left[key]), "right": list(right[key])}
    return {"tensors": compared, "identities": identities,
            "all_tensor_values_exact": all(row["bitwise_equal"] for row in compared.values()),
            "all_identity_values_exact": all(row["bitwise_equal"] for row in identities.values())}


def model_receipt(model, constructor):
    modules = dict(model.named_modules())
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameters = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                         "stride": list(value.stride())} for name, value in model.named_parameters()}
    buffers = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                      "stride": list(value.stride())} for name, value in model.named_buffers()}
    contract_sha = lambda value: hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"constructor": constructor, "state_sha256": tensor_state_sha256(model.state_dict()),
            "config": dataclasses.asdict(model.config),
            "model_training": model.training,
            "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()
                                             if parameter.requires_grad),
            "requires_grad_name_count": len(trainable),
            "requires_grad_names_sha256": contract_sha(trainable),
            "module_count": len(modules),
            "training_module_count": sum(module.training for module in modules.values()),
            "parameter_contract_count": len(parameters),
            "parameter_contract_sha256": contract_sha(parameters),
            "buffer_contract_count": len(buffers),
            "buffer_contract_sha256": contract_sha(buffers),
            "planner_plan_output_scale": list(model.planner.config.plan_output_scale),
            "audit_load_metadata": getattr(model, "audit_load_metadata", None)}


def output_receipt(output):
    return {key: tensor_receipt(value) for key, value in output.items() if isinstance(value, torch.Tensor)}


def output_delta(left, right):
    tensor_keys = sorted(key for key in set(left) & set(right)
                         if isinstance(left[key], torch.Tensor) and isinstance(right[key], torch.Tensor))
    return {"same_tensor_keys": tensor_keys == sorted(key for key, value in left.items()
                                                       if isinstance(value, torch.Tensor))
            == sorted(key for key, value in right.items() if isinstance(value, torch.Tensor)),
            "tensors": {key: tensor_delta(left[key], right[key]) for key in tensor_keys}}


def normal_records(document):
    if isinstance(document.get("conditions"), dict):
        records = document["conditions"].get("normal", {}).get("records")
    else:
        records = document.get("records")
    require(isinstance(records, list) and len(records) >= BATCH, "Reference needs at least first batch4")
    return records


def validate_reference_pair(immutable, rerun, checkpoint_sha):
    left, right = normal_records(immutable), normal_records(rerun)
    keys = ("row", "scenario", "session", "frame", "pred_abs_xy", "gt_abs_xy", "d3")
    require(all(all(a.get(key) == b.get(key) for key in keys)
                for a, b in zip(left[:BATCH], right[:BATCH])),
            "Immutable evaluator and evaluator rerun first4 differ")
    protocol = immutable.get("protocol", {})
    require(protocol.get("checkpoint_sha256") == checkpoint_sha
            and protocol.get("checkpoint_step") == 6000
            and protocol.get("data", {}).get("split_sha256") == SPLIT_SHA256
            and protocol.get("arguments", {}).get("batch") == BATCH
            and protocol.get("arguments", {}).get("precision") == "bf16"
            and protocol.get("arguments", {}).get("time_input") == "nominal"
            and immutable.get("final_val_accessed") is False,
            "Immutable evaluator protocol differs")
    return left[:BATCH]


def compare_output_to_reference(output, gt, records):
    plan = output["plan_abs"].detach().cpu()
    reference_plan = torch.tensor([row["pred_abs_xy"] for row in records], dtype=torch.float32)
    reference_gt = torch.tensor([row["gt_abs_xy"] for row in records], dtype=torch.float32)
    observed_d3 = weighted_d3(plan, gt.detach().cpu().float())
    reference_d3 = torch.tensor([row["d3"] for row in records], dtype=torch.float32)
    return {"plan": tensor_delta(plan, reference_plan),
            "ground_truth": tensor_delta(gt.detach().cpu().float(), reference_gt),
            "d3": tensor_delta(observed_d3, reference_d3)}


def plain_forward(model, inputs, device):
    with torch.inference_mode(), training_autocast(device, "bf16"):
        return model(**inputs)


def run_evaluator_path(args, batch, device, records):
    torch.manual_seed(args.base_seed)
    model = construct_model(SimpleNamespace(checkpoint=args.checkpoint, config_json=None,
                                            goal_on=None, state_on=None, device=args.device))
    before = model_receipt(model, "scripts.audit_motiondrive_v2.construct_model")
    state_before = tensor_state_sha256(model.state_dict())
    inputs = planning_model_inputs(to_device(batch, device), "nominal")
    grad_true_plain_a = plain_forward(model, inputs, device)
    grad_true_hook, grad_true_hidden = forward_with_decoded(
        model, inputs, training_autocast(device, "bf16"))
    grad_true_plain_repeat = plain_forward(model, inputs, device)
    state_after_grad_true = tensor_state_sha256(model.state_dict())
    model.requires_grad_(False)
    frozen_receipt = model_receipt(model, "same evaluator model after requires_grad_(False)")
    grad_false_hook_a, grad_false_hidden_a = forward_with_decoded(
        model, inputs, training_autocast(device, "bf16"))
    grad_false_plain = plain_forward(model, inputs, device)
    grad_false_hook_repeat, grad_false_hidden_repeat = forward_with_decoded(
        model, inputs, training_autocast(device, "bf16"))
    state_after_grad_false = tensor_state_sha256(model.state_dict())
    require(state_before == state_after_grad_true == state_after_grad_false,
            "Evaluator-path model state changed during inference")
    result = {
        "constructor_before_freeze": before,
        "constructor_after_freeze": frozen_receipt,
        "input_receipt": {key: tensor_receipt(value) for key, value in inputs.items()},
        "forward_order": ["grad_true_plain_A", "grad_true_hook", "grad_true_plain_repeat_A",
                          "grad_false_hook_A", "grad_false_plain", "grad_false_hook_repeat_A"],
        "forwards": {
            "grad_true_plain_A": output_receipt(grad_true_plain_a),
            "grad_true_hook": output_receipt(grad_true_hook),
            "grad_true_plain_repeat_A": output_receipt(grad_true_plain_repeat),
            "grad_false_hook_A": output_receipt(grad_false_hook_a),
            "grad_false_plain": output_receipt(grad_false_plain),
            "grad_false_hook_repeat_A": output_receipt(grad_false_hook_repeat)},
        "comparisons": {
            "grad_true_plain_to_hook": output_delta(grad_true_plain_a, grad_true_hook),
            "grad_true_plain_repeat": output_delta(grad_true_plain_a, grad_true_plain_repeat),
            "grad_false_hook_to_plain": output_delta(grad_false_hook_a, grad_false_plain),
            "grad_false_hook_repeat": output_delta(grad_false_hook_a, grad_false_hook_repeat),
            "requires_grad_true_vs_false_plain": output_delta(grad_true_plain_a, grad_false_plain),
            "requires_grad_true_vs_false_hook": output_delta(grad_true_hook, grad_false_hook_a),
            "grad_true_plain_to_immutable_reference": compare_output_to_reference(
                grad_true_plain_a, batch["gt_plan"], records),
            "grad_false_hook_to_immutable_reference": compare_output_to_reference(
                grad_false_hook_a, batch["gt_plan"], records)},
        "hook_hidden": {
            "grad_true": tensor_receipt(grad_true_hidden),
            "grad_false_A": tensor_receipt(grad_false_hidden_a),
            "grad_false_repeat_A": tensor_receipt(grad_false_hidden_repeat),
            "grad_true_vs_false": tensor_delta(grad_true_hidden, grad_false_hidden_a),
            "grad_false_repeat": tensor_delta(grad_false_hidden_a, grad_false_hidden_repeat)},
        "state_sha256": {"before": state_before, "after_grad_true_P_H_P": state_after_grad_true,
                         "after_grad_false_H_P_H": state_after_grad_false},
        "full_forward_count": 6,
    }
    cpu_grad_false_hook = {key: value.detach().cpu() for key, value in grad_false_hook_a.items()
                           if isinstance(value, torch.Tensor)}
    cpu_grad_false_feature = grad_false_hidden_a.detach().cpu()
    del model
    torch.cuda.empty_cache()
    return result, cpu_grad_false_hook, cpu_grad_false_feature


def run_cache_direct_path(args, batch, device, records):
    torch.manual_seed(args.base_seed)
    model, _ = load_frozen_model(args.checkpoint, args.expected_checkpoint_sha256,
                                 args.run_manifest, args.expected_run_manifest_sha256,
                                 args.base_seed, device)
    before = model_receipt(model, "cache.load_frozen_model direct MotionDriveV2 strict load")
    state_before = tensor_state_sha256(model.state_dict())
    inputs = model_inputs(to_device(batch, device), time_input="nominal")
    # Match the failed cache's actual order: the feature-producing hooked call is first.
    hooked, hidden = forward_with_decoded(model, inputs, training_autocast(device, "bf16"))
    state_after = tensor_state_sha256(model.state_dict())
    require(state_before == state_after, "Direct cache model state changed during inference")
    result = {
        "constructor": before,
        "input_receipt": {key: tensor_receipt(value) for key, value in inputs.items()},
        "forward_order": ["hooked_first_actual_cache_order"],
        "forwards": {"hooked_first_actual_cache_order": output_receipt(hooked)},
        "comparisons": {"hooked_to_immutable_reference": compare_output_to_reference(
                            hooked, batch["gt_plan"], records)},
        "hook_hidden": tensor_receipt(hidden),
        "state_sha256": {"before": state_before, "after": state_after},
        "full_forward_count": 1,
    }
    cpu_hooked = {key: value.detach().cpu() for key, value in hooked.items()
                  if isinstance(value, torch.Tensor)}
    cpu_feature = hidden.detach().cpu()
    del model
    torch.cuda.empty_cache()
    return result, cpu_hooked, cpu_feature


def write_new_json(path, value):
    path = Path(path)
    require(not path.exists() and not path.is_symlink() and path.parent.is_dir(),
            "New output in an existing directory required")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--expected-run-manifest-sha256", required=True)
    parser.add_argument("--base-seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--immutable-report", required=True)
    parser.add_argument("--expected-immutable-report-sha256", required=True)
    parser.add_argument("--evaluator-rerun-report", required=True)
    parser.add_argument("--expected-evaluator-rerun-report-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"), default="cuda:0")
    args = parser.parse_args(argv)
    require(args.workers >= 0, "workers must be nonnegative")
    require(args.expected_checkpoint_sha256 == P4_CHECKPOINT_SHA256[args.base_seed]
            and args.expected_run_manifest_sha256 == P4_RUN_MANIFEST_SHA256[args.base_seed],
            "Wrong fixed P4 base artifacts")
    require(all(valid_sha(getattr(args, name)) for name in (
        "expected_source_manifest_sha256", "expected_immutable_report_sha256",
        "expected_evaluator_rerun_report_sha256")), "Explicit lowercase SHA pins required")
    return args


def main(argv=None):
    args = arguments(argv)
    paths = {"checkpoint": Path(args.checkpoint), "run_manifest": Path(args.run_manifest),
             "split": Path(args.split_manifest),
             "supervision": Path(args.supervision_root) / "supervision_manifest.json",
             "calibration": Path(args.supervision_root) / "calibration.npz",
             "source_manifest": Path(args.source_manifest),
             "immutable_report": Path(args.immutable_report),
             "evaluator_rerun_report": Path(args.evaluator_rerun_report)}
    expected = {"checkpoint": args.expected_checkpoint_sha256,
                "run_manifest": args.expected_run_manifest_sha256,
                "split": SPLIT_SHA256, "supervision": SUPERVISION_SHA256,
                "calibration": CALIBRATION_SHA256,
                "source_manifest": args.expected_source_manifest_sha256,
                "immutable_report": args.expected_immutable_report_sha256,
                "evaluator_rerun_report": args.expected_evaluator_rerun_report_sha256}
    before = {name: file_sha(path) for name, path in paths.items()}
    require(before == expected, "Pinned diagnostic input SHA mismatch")
    source = read_pinned_json(paths["source_manifest"], expected["source_manifest"])
    validate_source_manifest(source)
    immutable = read_pinned_json(paths["immutable_report"], expected["immutable_report"])
    rerun = read_pinned_json(paths["evaluator_rerun_report"], expected["evaluator_rerun_report"])
    records = validate_reference_pair(immutable, rerun, expected["checkpoint"])
    device = torch.device(args.device)
    torch.manual_seed(args.base_seed)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    common = dict(data_root=args.data_root, split_manifest=args.split_manifest, split="tune",
                  supervision_root=args.supervision_root, min_frame=30, frame_stride=5,
                  max_samples=0, augment=False, seed=args.base_seed)
    evaluator_dataset = ImageCounterfactualDataset(MotionDriveDataset(**common), "normal")
    cache_dataset = MotionDriveDataset(**common)
    evaluator_batch = next(iter(DataLoader(evaluator_dataset, batch_size=BATCH, shuffle=False,
                                           num_workers=args.workers, pin_memory=True)))
    cache_batch = next(iter(DataLoader(cache_dataset, batch_size=BATCH, shuffle=False,
                                      num_workers=args.workers, pin_memory=True)))
    input_comparison = compare_batches(evaluator_batch, cache_batch)
    require(input_comparison["all_tensor_values_exact"]
            and input_comparison["all_identity_values_exact"], "Evaluator/cache first batch inputs differ")
    evaluator_cpu_inputs = planning_model_inputs(evaluator_batch, "nominal")
    direct_cpu_inputs = model_inputs(cache_batch, time_input="nominal")
    model_input_comparison = {
        key: tensor_delta(evaluator_cpu_inputs[key], direct_cpu_inputs[key]) for key in MODEL_INPUTS}
    require(all(row["bitwise_equal"] for row in model_input_comparison.values()),
            "Evaluator/cache model-input helpers differ")
    evaluator, evaluator_hooked, evaluator_feature = run_evaluator_path(
        args, evaluator_batch, device, records)
    direct, direct_hooked, direct_feature = run_cache_direct_path(
        args, cache_batch, device, records)
    report = {
        "schema_version": 1, "status": "completed_bounded_forward_path_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(), "base_seed": args.base_seed,
        "scope": ("exactly seven full forwards on one first tune batch4: same evaluator model "
                  "grad-true P-H-P then grad-false H-P-H, then direct cache hooked-first once"),
        "environment": {"pid": os.getpid(), "python": sys.executable,
                        "python_version": sys.version.split()[0], "torch": str(torch.__version__),
                        "cuda_runtime": torch.version.cuda, "device": str(device),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "reference_join": {"immutable_report_sha256": expected["immutable_report"],
                           "evaluator_rerun_report_sha256": expected["evaluator_rerun_report"],
                           "first4_plan_gt_d3_bitwise_exact": True},
        "inputs_sha256_before": before, "input_pipeline_comparison": input_comparison,
        "model_input_helper_comparison": model_input_comparison,
        "coordinate_path": ("model plan_abs cumulative XY compared directly; no absolute/incremental "
                            "coordinate conversion or round trip"),
        "evaluator_path": evaluator, "cache_direct_path": direct,
        "constructor_state_delta": {
            "state_hash_equal": evaluator["constructor_before_freeze"]["state_sha256"]
                                == direct["constructor"]["state_sha256"],
            "config_equal": evaluator["constructor_before_freeze"]["config"]
                            == direct["constructor"]["config"],
            "parameter_contract_equal": evaluator["constructor_before_freeze"]["parameter_contract_sha256"]
                                        == direct["constructor"]["parameter_contract_sha256"],
            "buffer_contract_equal": evaluator["constructor_before_freeze"]["buffer_contract_sha256"]
                                     == direct["constructor"]["buffer_contract_sha256"]},
        "cross_constructor_hooked_output": output_delta(evaluator_hooked, direct_hooked),
        "cross_constructor_selector_feature": tensor_delta(evaluator_feature, direct_feature),
        "full_forward_count": evaluator["full_forward_count"] + direct["full_forward_count"],
        "final_val_accessed": False, "checkpoint_or_runtime_modified": False,
        "tolerance_gate_changed": False,
    }
    require(report["full_forward_count"] == 7, "Bounded diagnostic must execute exactly seven forwards")
    report["inputs_sha256_after"] = {name: file_sha(path) for name, path in paths.items()}
    require(report["inputs_sha256_after"] == before, "Diagnostic inputs changed")
    write_new_json(args.out, report)
    print(json.dumps({"status": report["status"], "out": str(Path(args.out).resolve()),
                      "out_sha256": file_sha(args.out), "full_forward_count": report["full_forward_count"],
                      "final_val_accessed": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
