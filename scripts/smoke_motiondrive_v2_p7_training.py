#!/usr/bin/env python3
"""Three-update P7 plumbing smoke; never a checkpoint or scientific run.

The probe reads train labels/images only, performs exactly three logical
batch16/micro2 optimizer updates, and writes one isolated diagnostic JSON.  It
does not instantiate a tune/final loader, evaluate D3, or save reusable model
or optimizer state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_motiondrive_v2_p7_goal_routing as p7
import train_motiondrive_v2 as trainer
from motiondrive_v2_training import (LossWeights, build_loss_normalizers, compute_loss,
                                     model_inputs, set_training_mode, tensor_state_sha256,
                                     to_device)

UPDATES = 3
LOGICAL_BATCH = 16
MICROBATCH = 2


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def lr_factor(update_index: int) -> float:
    """Original joint6000 warmup200+cosine factor for one-indexed update."""
    require(type(update_index) is int and 1 <= update_index <= UPDATES,
            "Canary update index must be 1..3")
    warm = min(1., update_index / 200.)
    progress = max(0., ((update_index - 1) - 200) / (6000 - 200))
    return warm * .5 * (1. + np.cos(np.pi * min(1., progress)))


def optimizer_for(model):
    backbone, other = [], []
    branch_names = []
    for name, parameter in model.named_parameters():
        is_backbone = name.startswith("backbone_fpn.") and not name.startswith(
            ("backbone_fpn.lat", "backbone_fpn.out"))
        (backbone if is_backbone else other).append(parameter)
        if name.startswith("scene_encoder.cross_cell_goal_residual."):
            branch_names.append(name)
            require(not is_backbone, "P7 branch entered the backbone optimizer group")
    require(branch_names and all(parameter.requires_grad for parameter in model.parameters()),
            "P7 smoke requires every joint parameter trainable")
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": 1e-5, "base_lr": 1e-5},
        {"params": other, "lr": 1e-4, "base_lr": 1e-4},
    ], weight_decay=.01)
    require(not optimizer.state, "P7 canary optimizer must start fresh at step0")
    return optimizer, branch_names


def gradient_receipt(branch, update_index: int) -> dict:
    parameters = {
        "output": branch.output.weight,
        "query": branch.query_image.weight,
        "key": branch.key_image.weight,
        "value": branch.value_image.weight,
    }
    result = {}
    for name, parameter in parameters.items():
        gradient = parameter.grad
        require(gradient is not None and bool(torch.isfinite(gradient).all()),
                f"P7 {name} gradient missing/nonfinite")
        result[name] = {"nonzero": bool(torch.count_nonzero(gradient)),
                        "l2": float(gradient.detach().float().norm())}
    require(result["output"]["nonzero"], "P7 zero-init output did not receive first-path gradient")
    if update_index == 1:
        require(not any(result[name]["nonzero"] for name in ("query", "key", "value")),
                "P7 closed Q/K/V received a gradient before output opened")
    else:
        require(all(result[name]["nonzero"] for name in ("query", "key", "value")),
                "P7 Q/K/V did not receive gradients after output opened")
    return result


def compare_arm_receipts(control: dict, goal: dict) -> dict:
    """CPU postcheck for two separately preserved same-base canary reports."""
    for report in (control, goal):
        require(report.get("status") == "completed_three_update_canary_not_training_result",
                "Incomplete P7 canary report")
    require(control["arm"] == "control_zero_slot" and goal["arm"] == "goal_real_slot"
            and control["base_seed"] == goal["base_seed"], "P7 canary pair identity mismatch")
    for field in ("initial_model_state_sha256", "sample_order_sha256", "sample_rows"):
        require(control[field] == goal[field], f"P7 canary C/G {field} mismatch")
    return {"same_base_initial_state_exact": True, "same_data_order_exact": True,
            "updates_each": UPDATES, "scientific_result": False}


def atomic_new_json(path: Path, payload: dict) -> None:
    require(not path.exists() and not path.is_symlink(), "Refusing to overwrite canary report")
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("x") as stream:
        stream.write(serialized); stream.flush(); os.fsync(stream.fileno())
    os.link(temp, path); temp.unlink()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--arm", choices=tuple(p7.ARM_TO_MODE), required=True)
    parser.add_argument("--base-seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--expected-init-sha256", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--expected-run-manifest-sha256", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-smoke-source-sha256", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-physical-gpu-uuid", required=True)
    parser.add_argument("--gpu", type=int, choices=(0,), default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cuda-memory-limit-mib", type=int, default=12000)
    parser.add_argument("--cuda-min-free-mib", type=int, default=8192)
    parser.set_defaults(preflight_only=False)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    require(p7.file_sha(Path(__file__).resolve()) == args.expected_smoke_source_sha256,
            "P7 smoke source SHA mismatch")
    output_dir = Path(args.output_dir).resolve()
    require("p7_goal_routing_canary" in output_dir.name and not output_dir.exists(),
            "P7 smoke requires a new unmistakable canary output directory")
    source = p7.validate_source_manifest(args.source_manifest, args.expected_source_manifest_sha256)
    data = p7.validate_data_contract(args.data_root, args.split_manifest,
                                     args.supervision_root, args.base_seed)
    payload, p0 = p7.validate_p0(args.init, args.expected_init_sha256, args.run_manifest,
                                 args.expected_run_manifest_sha256, args.base_seed)
    mode = p7.ARM_TO_MODE[args.arm]
    initialized = p7.prepare_initialization(payload, args.base_seed, mode)
    runtime = p7.validate_runtime_namespace(args)
    device = torch.device("cuda:0")
    memory_policy = trainer.configure_cuda_memory(
        device, args.cuda_memory_limit_mib, args.cuda_min_free_mib)
    model, missing, initial_sha = p7._build_initialized_model(payload, args.base_seed, mode)
    require(missing == initialized["expected_missing_state_keys"]
            and initial_sha == initialized["initial_model_state_sha256"],
            "P7 canary assembly differs from production preflight")
    branch = model.scene_encoder.cross_cell_goal_residual
    require(not bool(branch.output.weight.count_nonzero()), "P7 canary Wo must begin zero")
    optimizer, branch_names = optimizer_for(model)
    initial_bn = {name: value.clone() for name, value in model.state_dict().items()
                  if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)

    from motiondrive_v2_data import MotionDriveDataset
    dataset = MotionDriveDataset(data_root=args.data_root, split_manifest=args.split_manifest,
                                 supervision_root=args.supervision_root, split="train",
                                 min_frame=30, frame_stride=1, max_samples=0,
                                 augment=True, seed=args.base_seed)
    generator = torch.Generator().manual_seed(args.base_seed)
    loader = DataLoader(dataset, batch_size=LOGICAL_BATCH, shuffle=True, num_workers=args.workers,
                        pin_memory=True, generator=generator, worker_init_fn=trainer.worker_seed,
                        drop_last=False, persistent_workers=False)
    weights = LossWeights(plan=1., occupancy=.2, lane=.2, motion=.2, uncertainty=True)
    rows, receipts, branch_call_count, branch_input_dtypes, planner_input_dtypes = [], [], 0, [], []
    def branch_pre(_module, values):
        nonlocal branch_call_count
        branch_call_count += 1; branch_input_dtypes.append(str(values[0].dtype))
    handles = [branch.register_forward_pre_hook(branch_pre),
               model.planner.xy_head.register_forward_pre_hook(
                   lambda _module, values: planner_input_dtypes.append(str(values[0].dtype)))]
    started = time.monotonic()
    try:
        for update_index, raw in enumerate(loader, 1):
            if update_index > UPDATES:
                break
            require(len(raw["images"]) == LOGICAL_BATCH, "P7 canary requires full logical batch16")
            rows.extend(int(value) for value in raw["row"])
            set_training_mode(model, "fixed")
            require(not any(module.training for module in model.modules()
                            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)),
                    "P7 canary fixed-BN policy failed")
            factor = float(lr_factor(update_index))
            for group in optimizer.param_groups: group["lr"] = group["base_lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            normalizers = to_device(build_loss_normalizers(raw), device)
            losses = []
            for start in range(0, LOGICAL_BATCH, MICROBATCH):
                trainer.check_cuda_headroom(device, args.cuda_min_free_mib)
                batch = to_device(trainer.slice_batch(raw, start, start + MICROBATCH), device)
                with trainer.autocast(device, "bf16"):
                    output = model(**model_inputs(batch, time_input="nominal"))
                require(output["plan_abs"].dtype == output["state_hat"].dtype
                        == output["history_hat"].dtype == torch.float32,
                        "P7 canary planner/motion outputs must be FP32")
                loss, _ = compute_loss(output, batch, weights, normalizers=normalizers)
                require(loss.dtype == torch.float32 and bool(torch.isfinite(loss)),
                        "P7 canary loss must be finite FP32")
                loss.backward(); losses.append(float(loss.detach()))
            gradients = gradient_receipt(branch, update_index)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            require(bool(torch.isfinite(norm)), "P7 canary gradient norm nonfinite")
            optimizer.step()
            require(bool(branch.output.weight.count_nonzero()), "P7 canary Wo did not open")
            receipts.append({"update": update_index, "lr_factor": factor,
                             "backbone_lr": optimizer.param_groups[0]["lr"],
                             "head_lr": optimizer.param_groups[1]["lr"],
                             "microbatch_losses": losses, "grad_norm_before_clip": float(norm),
                             "branch_gradients": gradients})
    finally:
        for handle in handles: handle.remove()
    require(len(receipts) == UPDATES and branch_call_count == UPDATES * (LOGICAL_BATCH // MICROBATCH),
            "P7 canary did not execute exactly 3x8 branch forwards")
    require(planner_input_dtypes and set(planner_input_dtypes) == {"torch.float32"},
            "P7 canary planner did not receive FP32")
    require(branch_input_dtypes and set(branch_input_dtypes) <= {"torch.bfloat16", "torch.float32"},
            "P7 canary branch input dtype unexpected")
    final_bn = {name: value.detach().cpu() for name, value in model.state_dict().items()
                if name in initial_bn}
    require(initial_bn.keys() == final_bn.keys()
            and all(torch.equal(initial_bn[key], final_bn[key]) for key in initial_bn),
            "P7 canary fixed-BN buffers changed")
    final_sha = tensor_state_sha256(model.state_dict())
    require(final_sha != initial_sha, "P7 canary optimizer did not update model state")
    optimizer_steps, optimizer_tensors_finite = set(), True
    for state in optimizer.state.values():
        step = state.get("step")
        if isinstance(step, torch.Tensor):
            step = step.item()
        if step is not None:
            optimizer_steps.add(int(step))
        for value in state.values():
            if isinstance(value, torch.Tensor):
                optimizer_tensors_finite &= bool(torch.isfinite(value).all())
    require(optimizer_steps == {UPDATES} and optimizer_tensors_finite,
            "P7 canary optimizer state is nonfinite or not at step3")
    sample_order_sha = hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "status": "completed_three_update_canary_not_training_result",
        "scientific_result": False, "eligible_as_training_initializer": False,
        "checkpoint_written": False, "arm": args.arm, "branch_mode": mode,
        "base_seed": args.base_seed, "updates": UPDATES, "logical_batch": LOGICAL_BATCH,
        "microbatch": MICROBATCH, "sample_rows": rows, "sample_order_sha256": sample_order_sha,
        "initial_model_state_sha256": initial_sha, "final_ephemeral_model_state_sha256": final_sha,
        "p0": p0, "source": source, "data": data, "runtime": runtime,
        "memory_policy": memory_policy, "cuda_memory": trainer.cuda_memory_snapshot(device),
        "elapsed_seconds": time.monotonic() - started, "missing_new_state_keys": missing,
        "branch_parameter_names": branch_names, "branch_calls": branch_call_count,
        "optimizer_state_entries": len(optimizer.state),
        "optimizer_state_steps": sorted(optimizer_steps),
        "optimizer_tensors_finite": optimizer_tensors_finite,
        "branch_input_dtypes": sorted(set(branch_input_dtypes)),
        "planner_input_dtypes": sorted(set(planner_input_dtypes)), "receipts": receipts,
        "fixed_bn_buffers_unchanged": True, "tune_evaluations": 0,
        "final_validation_accessed": False,
        "boundary": "Three-update plumbing/memory evidence only; never initialize a later run from this process.",
    }
    atomic_new_json(output_dir / "canary.json", report)
    print(json.dumps({"status": report["status"], "output": str(output_dir / "canary.json"),
                      "sample_order_sha256": sample_order_sha,
                      "peak_reserved_bytes": report["cuda_memory"]["peak_reserved_bytes"]},
                     sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
