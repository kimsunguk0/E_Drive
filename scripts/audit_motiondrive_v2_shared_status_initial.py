#!/usr/bin/env python3
"""Audit A1 initial identity and shared-head gradients on one TRAIN microbatch.

This is a backward-only diagnostic.  It never steps an optimizer, evaluates
the tune split, or writes a checkpoint.  The unmodified parent receives only
its original six model inputs; status is passed only to the opt-in A1 models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_motiondrive_v2_shared_status_a1 as a1
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from motiondrive_v2_data import MotionDriveDataset
from motiondrive_v2_shared_status_data import SharedStatusDataset, shared_status_model_inputs
from motiondrive_v2_training import LossWeights, compute_loss, model_inputs, tensor_state_sha256, to_device
from train_motiondrive_v2 import autocast, seed_all, slice_batch, worker_seed


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    require(not path.exists() and not path.is_symlink(), "refusing to overwrite audit output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    require(not temporary.exists() and not temporary.is_symlink(), "stale audit temporary output")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def tensor_summary(outputs: dict[str, torch.Tensor]) -> dict:
    require(outputs and all(isinstance(value, torch.Tensor) for value in outputs.values()),
            "model output must be a nonempty tensor mapping")
    return {
        "tree_sha256": tensor_state_sha256(outputs),
        "keys": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype),
                  "finite": bool(torch.isfinite(value).all())}
            for key, value in sorted(outputs.items())
        },
    }


def compare_outputs(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> dict:
    require(set(reference) == set(candidate), "initial output keyset changed")
    differences = {}
    for key in sorted(reference):
        left, right = reference[key], candidate[key]
        require(left.shape == right.shape and left.dtype == right.dtype,
                f"initial output schema changed: {key}")
        difference = (left.float() - right.float()).abs()
        differences[key] = float(difference.max()) if difference.numel() else 0.0
        require(torch.equal(left, right), f"initial A1 output differs from parent: {key}")
    return {"bitwise_equal": True, "max_abs_by_key": differences,
            "max_abs_all": max(differences.values(), default=0.0)}


def final_gamma_gradients(model, outputs, batch) -> dict:
    weights = LossWeights(plan=1.0, occupancy=1.0, lane=1.0,
                          motion=0.0, uncertainty=False)
    _total, parts = compute_loss(outputs, batch, weights)
    final = model.shared_status_fusion.status_mlp[-1]
    result = {}
    names = (("plan", "plan_d3"), ("occupancy", "occ_bce"), ("lane", "lane_bce"))
    for index, (name, part) in enumerate(names):
        grads = torch.autograd.grad(parts[part], (final.weight, final.bias),
                                    retain_graph=index + 1 < len(names), allow_unused=False)
        norms = {"weight_l2": float(torch.linalg.vector_norm(grads[0].float())),
                 "bias_l2": float(torch.linalg.vector_norm(grads[1].float()))}
        require(all(math.isfinite(value) and value > 0 for value in norms.values()),
                f"{name} does not reach both final gamma parameters")
        result[name] = {**norms, "loss": float(parts[part].detach())}
    return result


def cuda_identity(expected_uuid: str) -> dict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(visible == expected_uuid and "," not in visible,
            "audit requires one UUID-isolated CUDA device")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "audit requires exactly one visible CUDA device")
    raw = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    actual = raw if raw.startswith("GPU-") else "GPU-" + raw
    require(actual == expected_uuid, "observed physical GPU UUID mismatch")
    return {"cuda_visible_devices": visible, "observed_uuid_raw": raw,
            "actual_physical_gpu_uuid": actual, "device": "cuda:0"}


def first_train_microbatch(args, overlay):
    base = MotionDriveDataset(
        data_root=args.data_root, split_manifest=args.split_manifest, split="train",
        supervision_root=args.supervision_root, min_frame=30, frame_stride=1,
        max_samples=0, augment=True, seed=0, history_contract="control")
    wrapped = SharedStatusDataset(base, "train", overlay, "provided_causal_5d")
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(wrapped, batch_size=16, shuffle=True, num_workers=4,
                        pin_memory=True, generator=generator, worker_init_fn=worker_seed,
                        drop_last=False, persistent_workers=False)
    logical = next(iter(loader))
    require(len(logical["images"]) == 16, "first logical TRAIN batch must contain 16 rows")
    micro = slice_batch(logical, 0, 2)
    logical_rows = np.asarray(logical["row"], dtype="<i8")
    micro_rows = np.asarray(micro["row"], dtype="<i8")
    return micro, {
        "logical_batch_rows": logical_rows.tolist(),
        "logical_batch_rows_sha256": hashlib.sha256(logical_rows.tobytes()).hexdigest(),
        "microbatch_rows": micro_rows.tolist(),
        "microbatch_rows_sha256": hashlib.sha256(micro_rows.tobytes()).hexdigest(),
        "logical_batch": 16, "audited_microbatch": 2, "epoch": 0,
    }


def original_model(payload, device):
    seed_all(0)
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    model = MotionDriveV2(config)
    model.load_state_dict(payload["model"], strict=True)
    require(tensor_state_sha256(model.state_dict()) == a1.EXPECTED_PARENT_MODEL_SHA256,
            "strict original parent state mismatch")
    model.to(device).eval()
    return model


def a1_model(payload, device):
    model, prepared = a1.prepare_model(payload, seed=0)
    model.to(device).eval()
    return model, prepared


def run(args):
    require(sha256(__file__) == args.expected_script_sha256,
            "audit script SHA mismatch")
    source = a1.validate_source_manifest(args.source_manifest,
                                         args.expected_source_manifest_sha256)
    payload, parent = a1.validate_parent(args)
    data, overlay = a1.validate_data(args)
    # The shared ego-cache container is opened for row identity.  State the
    # scientific boundary precisely rather than implying no physical read.
    data.pop("final_validation_accessed", None)
    data["final136_rows_selected_or_analyzed"] = 0
    runtime = cuda_identity(args.expected_physical_gpu_uuid)
    immutable = [args.init, args.init_manifest, args.source_manifest, args.split_manifest,
                 Path(args.supervision_root) / "supervision_manifest.json",
                 Path(args.supervision_root) / "calibration.npz",
                 Path(args.status_overlay_root) / "overlay_manifest.json",
                 Path(args.status_overlay_root) / "train.npz",
                 Path(args.status_overlay_root) / "tune.npz"]
    before = {str(Path(path).resolve()): sha256(path) for path in immutable}
    cpu_batch, rows = first_train_microbatch(args, overlay)
    device = torch.device("cuda:0")
    batch = to_device(cpu_batch, device)
    base_inputs = model_inputs(batch, time_input="nominal")
    status = batch["provided_status5"]
    require(status.shape == (2, 5) and bool(torch.isfinite(status).all()),
            "audited provided status is invalid")

    base = original_model(payload, device)
    base_state_before = tensor_state_sha256(base.state_dict())
    with torch.no_grad(), autocast(device, "bf16"):
        reference_gpu = base(**base_inputs)
    reference = {key: value.detach().cpu() for key, value in reference_gpu.items()}
    base_state_after = tensor_state_sha256(base.state_dict())
    require(base_state_before == base_state_after == a1.EXPECTED_PARENT_MODEL_SHA256,
            "original parent changed during forward")
    reference_summary = tensor_summary(reference)
    del reference_gpu, base
    torch.cuda.empty_cache()

    arms = {}
    complete_initial_states = []
    fusion_initial_states = []
    for arm in a1.ARMS:
        model, prepared = a1_model(payload, device)
        complete_initial_states.append(prepared["initial_model_state_sha256"])
        fusion_initial_states.append(prepared["fusion_state_sha256"])
        model_state_before = tensor_state_sha256(model.state_dict())
        supplied = torch.zeros_like(status) if arm == "zero" else status
        inputs = {**base_inputs, "provided_status5": supplied}
        with autocast(device, "bf16"):
            output_gpu = model(**inputs)
            gradients = final_gamma_gradients(model, output_gpu, batch)
        output = {key: value.detach().cpu() for key, value in output_gpu.items()}
        comparison = compare_outputs(reference, output)
        model_state_after = tensor_state_sha256(model.state_dict())
        require(model_state_before == model_state_after,
                f"{arm} weights/buffers changed during backward-only audit")
        arms[arm] = {
            "supplied_status_sha256": tensor_state_sha256({"status5": supplied}),
            "prepared": prepared,
            "model_state_before_and_after": model_state_before,
            "output": tensor_summary(output),
            "parent_output_comparison": comparison,
            "final_gamma_gradients": gradients,
        }
        del output_gpu, output, model
        torch.cuda.empty_cache()
    require(len(set(complete_initial_states)) == len(set(fusion_initial_states)) == 1,
            "ZERO and PROVIDED initial states differ")

    after = {str(Path(path).resolve()): sha256(path) for path in immutable}
    require(before == after, "immutable audit inputs changed")
    a1.validate_source_manifest(args.source_manifest, args.expected_source_manifest_sha256)
    return {
        "schema_version": 1,
        "status": "completed_backward_only_initial_audit",
        "script": str(Path(__file__).resolve()),
        "script_sha256": args.expected_script_sha256,
        "source": source, "parent": parent, "data": data, "runtime": runtime,
        "batch": rows,
        "reference": {"model_state_before_and_after": base_state_before,
                      "output": reference_summary,
                      "status_passed_to_parent_body": False},
        "arms": arms,
        "paired_initial_model_state_sha256": complete_initial_states[0],
        "paired_initial_fusion_state_sha256": fusion_initial_states[0],
        "optimizer_created": False, "optimizer_step": False,
        "tune_rows_evaluated": 0, "final136_rows_selected_or_analyzed": 0,
        "immutable_inputs_before_and_after": before,
    }


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--init", required=True)
    parser.add_argument("--init-manifest", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--status-overlay-root", required=True)
    parser.add_argument("--expected-status-overlay-sha256", required=True)
    parser.add_argument("--expected-physical-gpu-uuid", required=True)
    parser.add_argument("--expected-script-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.set_defaults(arm="provided_causal_5d")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    atomic_json(Path(args.output).resolve(), run(args))


if __name__ == "__main__":
    main()
