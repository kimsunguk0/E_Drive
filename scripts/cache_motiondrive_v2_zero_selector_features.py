#!/usr/bin/env python3
"""Cache frozen-P4 features for the offline zero-vs-move selector probe.

This is a diagnostic cache, not a production gate.  It performs one normal
P4 forward per cached sample and exposes only the FP32 tensor immediately
before ``planner.xy_head`` plus the same-forward neural state/history outputs.
Ground truth, costs, and row identity are stored separately and are never
selector features.  The final validation split is unavailable from this CLI.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from motiondrive_v2_data import MotionDriveDataset
from motiondrive_v2_training import (MODEL_INPUTS, model_inputs, tensor_state_sha256,
                                     to_device, weighted_d3)

P4_GIT_SHA = "86620b4ffc7e6838b49cf83b5be789eba12d8027"
SPLIT_SHA256 = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
SUPERVISION_SHA256 = "ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93"
CALIBRATION_SHA256 = "8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961"
INPUT_CONTRACT_SHA256 = "d45db0b11b5db19894ccc316eaa53039d4cae40118c45860aabf5eb8af375278"
RUNTIME_SHA256 = {
    "models/motiondrive_v2/__init__.py": "3a3e6c72029bda6edee171a07b863a1739dcd87717765ae3a6af48dbe007a47d",
    "models/motiondrive_v2/config.py": "f03d75c4d7dd4a500c009cf2adf082f75b90c268ef03da22c747a2dd17df7f8b",
    "models/motiondrive_v2/model.py": "d5507cf773342daea8cba0396232b5a0768d8ec84b080106552715b7dfb2b3e6",
    "models/motiondrive_v2/motion_encoder.py": "80c84387a0337c6bfcf700db8b49a3e761f1610f8f5e6a3830c4ade58069eacd",
    "models/motiondrive_v2/planner.py": "94811887bf7edd72f4796d83fb1193c5b87b83d2313eafdc50daabab4e8f6599",
    "models/motiondrive_v2/scene_encoder.py": "2263e2281b0175309575a977db89158da1d44511b5177a398eafa713b6893d8d",
    "models/motiondrive_v2_query_adapter.py": "49eef489ac20f35f2884bcfbb2315d1328f182932b709f964e2baae6e270eab2",
    "scripts/audit_motiondrive_v2.py": "fc10bba5c068c2bd46c84a436fc5357d82d931c16f565efc69d95049718ce06b",
    "scripts/build_grouped_split_v2.py": "5d3019cf730fe56a27ab7f204ee3f15bc8a4ffaef5df9fbf0635f086c84a49c1",
    "scripts/evaluate_motiondrive_v2_planning.py": "41c0a3e4b80173fbc6b740d3401949c6e5f9e09087e0d2378ad84dd216af8a07",
    "scripts/evaluate_motiondrive_v2_shared.py": "7b49e8f5c3a5b0169e039627953a7ce22b961d9e3d3c86aacb29afe768e6c5ba",
    "scripts/export_motiondrive_v2_inference.py": "7f93f716edd05f966960309b779d535a10649a9bea5188741545a53ff2a2ef2d",
    "scripts/initialize_motiondrive_v2_public.py": "673750b051d4ee20fd0792275b281c757e43b5dbb90d01c06547c2f7a83452cc",
    "scripts/launch_motiondrive_v2_trials.py": "8590f231e14985eac100e13b47b00832fb4a8e5a26e05717411ace637ff10826",
    "scripts/motiondrive_v2_data.py": "83b6c74177c131116f48feb968aa8f96821b5e820823843e8b2e97ba7af171ca",
    "scripts/motiondrive_v2_training.py": "148ce7a6c16bb5c604a75d3a15f8ff150b7bf136b2015c5c0d3c02bcf05d326a",
    "scripts/run_motiondrive_v2_fresh_trial.py": "08f2ed7dcd6df8fb9252620d263426dd813f5ee7c317b738f29bf0e0259974d2",
    "scripts/run_motiondrive_v2_query_trial.py": "2ac750de3b252c4a14178ca1772f6a831cc84825b96266f56eb43f6956db68ff",
    "scripts/sparse_scoredrive.py": "3d8e89824d813c07588a64c6fb1a44600681d54adebe8af3807c3f9ad521e7db",
    "scripts/supervise_motiondrive_v2_job.py": "4a970d80fe2cdc4116f6aebf4848f416478ddcbcaf57765e3d4e5788f9d87218",
    "scripts/train_motiondrive_v2.py": "3a209687203630cb09be94ebbc717c4f74c3e2570138fb4bf992bbc59c7be6f5",
    "scripts/train_motiondrive_v2_query_adapter.py": "330a999b08f3dcca9f05da6158570bb54f3452e03a40c7916e431f315e02fe5c",
}
RUNTIME_FILES = frozenset(RUNTIME_SHA256)
P4_CHECKPOINT_SHA256 = {
    0: "3e9ae5aa6ca89f4eb9a72ba38355dfdbec4ccaa38eabbebb3526fa3019404478",
    1: "c937f2f772c78f99cf3cfa5b1a96614267e20a9a667a157f4ada284940f9672e",
}
P4_RUN_MANIFEST_SHA256 = {
    0: "001b822dde79024f0e2cd52b5b99cb1096bbab5bfd09ef2d95e963ad8dc6e3de",
    1: "49dc09c8e7840a059ae6bcadba1c490052847c678dc81d5524ee39f9da159aad",
}
EXPECTED = {
    "train": {"n": 54810, "scenes": 203, "sessions": 72, "stride": 1},
    "tune": {"n": 1998, "scenes": 37, "sessions": 11, "stride": 5},
}
P4_STEP = 6000
CACHE_BATCH = {"train": 8, "tune": 4}
WAYPOINTS, HIDDEN_CHANNELS = 6, 128
FEATURE_COMPONENTS = (
    {"name": "planner_decoded_fp32", "shape": [6, 128], "start": 0, "stop": 768},
    {"name": "predicted_state_hat_fp32", "shape": [6], "start": 768, "stop": 774},
    {"name": "predicted_history_hat_fp32", "shape": [4, 4], "start": 774, "stop": 790},
)
FEATURE_DIM = 790
CANDIDATE_ORDER = ("zero_path", "original_p4_path")
ZERO, MOVE = 0, 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_sha256(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def runtime_receipt(device):
    return {"python_executable": sys.executable, "python_version": sys.version.split()[0],
            "torch_version": str(torch.__version__), "torch_cuda_runtime": torch.version.cuda,
            "device": str(device), "cuda_initialized": torch.cuda.is_initialized()}


def ensure_new(path):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"Refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_new_bytes(path, value):
    path = Path(path)
    ensure_new(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"Refusing to overwrite output: {path}") from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_new_json(path, value):
    atomic_new_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def read_pinned_json(path, expected_sha):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Ordinary JSON input required: {path}")
    require(valid_sha(expected_sha) and file_sha(path) == expected_sha,
            f"JSON SHA256 mismatch: {path}")
    value = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"Nonfinite JSON constant: {value}")))
    require(isinstance(value, dict), "JSON input must be a mapping")
    return value


def validate_source_manifest(manifest, root=ROOT):
    require(isinstance(manifest, Mapping)
            and isinstance(manifest.get("source_git_sha"), str)
            and re.fullmatch(r"[0-9a-f]{40}", manifest["source_git_sha"])
            and manifest.get("runtime_origin_git_sha") == P4_GIT_SHA,
            "Validation Git and exact P4 runtime origin Git receipts are required")
    files = manifest.get("files")
    require(isinstance(files, Mapping) and dict(files) == RUNTIME_SHA256 and len(files) == 22,
            "Exact 22-file P4 training runtime allowlist required")
    observed = {}
    for name, expected in files.items():
        require(isinstance(name, str) and valid_sha(expected), "Invalid source allowlist entry")
        path = Path(root) / name
        require(path.is_file() and not path.is_symlink() and file_sha(path) == expected,
                f"Pinned runtime source differs: {name}")
        observed[name] = expected
    contract = Path(root) / "models/motiondrive_v2_input_contract.py"
    require(contract.is_file() and file_sha(contract) == INPUT_CONTRACT_SHA256,
            "Pinned deployment input-contract source differs")
    return observed


def _json_equal(a, b):
    return json.dumps(a, sort_keys=True, separators=(",", ":"), allow_nan=False) == \
           json.dumps(b, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_checkpoint_payload(payload, sidecar, *, base_seed, checkpoint_sha,
                                run_manifest_sha):
    require(base_seed in P4_CHECKPOINT_SHA256, "base_seed must be 0 or 1")
    require(checkpoint_sha == P4_CHECKPOINT_SHA256[base_seed], "Wrong pinned P4 LAST SHA")
    require(run_manifest_sha == P4_RUN_MANIFEST_SHA256[base_seed], "Wrong pinned P4 sidecar SHA")
    require(isinstance(payload, dict) and payload.get("step") == P4_STEP,
            "P4 LAST6000 checkpoint required")
    manifest = payload.get("manifest")
    require(isinstance(manifest, dict) and isinstance(sidecar, dict), "Training manifests required")
    args, config = manifest.get("arguments", {}), manifest.get("model_config", {})
    required_args = {"phase": "joint", "steps": P4_STEP, "seed": base_seed,
                     "precision": "bf16", "time_input": "nominal", "bn_policy": "fixed"}
    require(all(args.get(key) == value for key, value in required_args.items()),
            "Checkpoint is not the fixed P4 joint recipe")
    required_config = {"backbone_arch": "resnet50", "goal_on": True, "state_on": True,
                       "motion_input_mode": "low_feature", "n_history": 4}
    require(all(config.get(key) == value for key, value in required_config.items())
            and list(config.get("plan_output_scale", ())) == [10., 5.],
            "Checkpoint is not complete G1S1/R50/low_feature/scale10,5")
    require(manifest.get("git_sha") == P4_GIT_SHA and sidecar.get("git_sha") == P4_GIT_SHA,
            "Training Git lineage differs")
    require(manifest.get("split_sha256") == SPLIT_SHA256
            and manifest.get("supervision_manifest_sha256") == SUPERVISION_SHA256,
            "C1 data lineage differs")
    require(manifest.get("data_counts") == {"train": 54810, "eval": 1998},
            "Full train/tune counts required")
    require(manifest.get("loss_weights", {}).get("plan") == 1., "Joint plan loss must be one")
    require(sidecar.get("status") == "completed" and sidecar.get("step") == P4_STEP,
            "Completed LAST6000 sidecar required")
    for key in ("arguments", "model_config", "git_sha", "split_sha256",
                "supervision_manifest_sha256", "data_counts", "train_rows_sha256",
                "eval_rows_sha256", "time_input", "time_input_policy", "loss_weights"):
        require(key in manifest and key in sidecar and _json_equal(manifest[key], sidecar[key]),
                f"Checkpoint/sidecar mismatch: {key}")
    state = payload.get("model")
    require(isinstance(state, dict) and state, "Checkpoint model state required")
    for name, tensor in state.items():
        require(isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all(),
                f"Nonfinite/non-tensor model state: {name}")
    return manifest


def load_frozen_model(checkpoint, expected_sha, run_manifest, expected_manifest_sha,
                      base_seed, device):
    checkpoint, run_manifest = Path(checkpoint), Path(run_manifest)
    require(checkpoint.is_file() and not checkpoint.is_symlink(), "Ordinary checkpoint required")
    require(valid_sha(expected_sha) and file_sha(checkpoint) == expected_sha,
            "Checkpoint SHA256 mismatch")
    sidecar = read_pinned_json(run_manifest, expected_manifest_sha)
    # Trusted user-owned local trainer checkpoint; RNG state requires full pickle load.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    manifest = validate_checkpoint_payload(payload, sidecar, base_seed=base_seed,
                                           checkpoint_sha=expected_sha,
                                           run_manifest_sha=expected_manifest_sha)
    config = MotionDriveV2Config(**manifest["model_config"])
    model = MotionDriveV2(config)
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False).eval().to(device)
    require(not any(parameter.requires_grad for parameter in model.parameters()),
            "Base model must be completely frozen")
    require(file_sha(checkpoint) == expected_sha and file_sha(run_manifest) == expected_manifest_sha,
            "Checkpoint lineage changed while loading")
    return model, manifest


def dataset_inventory(dataset):
    spec = EXPECTED[dataset.split]
    scenes = {str(dataset.scene_names[row]) for row in dataset.rows}
    sessions = {dataset.manifest["scene_to_session"][scene] for scene in scenes}
    require(len(dataset) == spec["n"] and len(scenes) == spec["scenes"]
            and len(sessions) == spec["sessions"], f"Wrong fixed {dataset.split} inventory")
    return {"n": len(dataset), "scenes": len(scenes), "sessions": len(sessions),
            "rows_sha256": rows_sha256(dataset.rows)}


def compose_selector_feature(decoded, state_hat, history_hat):
    require(all(isinstance(value, torch.Tensor) for value in (decoded, state_hat, history_hat)),
            "Selector feature components must be tensors")
    b = decoded.shape[0] if decoded.ndim else -1
    require(decoded.shape == (b, WAYPOINTS, HIDDEN_CHANNELS)
            and state_hat.shape == (b, 6) and history_hat.shape == (b, 4, 4),
            "Unexpected selector feature component shape")
    require(all(value.dtype == torch.float32 for value in (decoded, state_hat, history_hat)),
            "Selector features must already be FP32")
    require(all(torch.isfinite(value).all() for value in (decoded, state_hat, history_hat)),
            "Selector features must be finite")
    feature = torch.cat((decoded.flatten(1), state_hat, history_hat.flatten(1)), dim=1)
    require(feature.shape == (b, FEATURE_DIM), "Selector feature dimension mismatch")
    return feature


def forward_with_decoded(model, inputs, autocast_context):
    """One production forward with a temporary pre-hook; handle always removed."""
    captured = []

    def capture(module, values):
        require(len(values) == 1 and isinstance(values[0], torch.Tensor),
                "xy_head must receive exactly one tensor")
        captured.append(values[0].detach())

    handle = model.planner.xy_head.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode(), autocast_context:
            output = model(**inputs)
    finally:
        handle.remove()
    require(len(captured) == 1, "planner.xy_head must execute exactly once per full forward")
    decoded = captured[0]
    require(decoded.dtype == torch.float32, "xy_head pre-input must be decoded.float()")
    feature = compose_selector_feature(decoded, output["state_hat"], output["history_hat"])
    return output, feature


def exact_output_equal(left, right):
    require(set(left) == set(right), "Repeated forward output keys differ")
    for key in left:
        if isinstance(left[key], torch.Tensor):
            require(isinstance(right[key], torch.Tensor)
                    and left[key].dtype == right[key].dtype
                    and left[key].shape == right[key].shape
                    and torch.equal(left[key], right[key]),
                    f"Hook changed P4 output: {key}")
    return True


def selector_targets(plan, gt):
    require(plan.shape == gt.shape and plan.ndim == 3 and plan.shape[1:] == (6, 2),
            "Plans must be matching [B,6,2]")
    zero = torch.zeros_like(plan)
    candidates = torch.stack((zero, plan), dim=1)
    zero_cost, move_cost = weighted_d3(zero, gt), weighted_d3(plan, gt)
    costs = torch.stack((zero_cost, move_cost), dim=1).float()
    # Strictly cheaper zero only; exact cost ties retain the original P4 path.
    labels = torch.where(zero_cost < move_cost,
                         torch.full_like(zero_cost, ZERO, dtype=torch.long),
                         torch.full_like(zero_cost, MOVE, dtype=torch.long))
    return candidates.float(), costs, labels


def _autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext
    return nullcontext()


def _batch_ids(batch):
    size = len(batch["row"])
    return [{"row": int(batch["row"][i]), "scenario": str(batch["scenario"][i]),
             "session": str(batch["session_id"][i]), "frame": int(batch["frame"][i])}
            for i in range(size)]


def diagnostic_bucket(gt_plan, state_target, state_valid):
    """Preregistered descriptive GT buckets; never a selector input."""
    require(gt_plan.ndim == 3 and gt_plan.shape[1:] == (6, 2)
            and state_target.ndim == 2 and state_target.shape[1:] == (6,)
            and state_valid.shape == state_target.shape, "Invalid GT bucket tensors")
    future_max = torch.linalg.vector_norm(gt_plan.float(), dim=-1).max(dim=1).values
    valid_speed = state_valid[:, :2].bool().all(dim=1) & torch.isfinite(state_target[:, :2]).all(dim=1)
    current_stop = valid_speed & (torch.linalg.vector_norm(state_target[:, :2].float(), dim=1) < .2)
    return ["steady" if bool(stop and distance <= .2) else
            "depart" if bool(stop) else "nonstop"
            for stop, distance in zip(current_stop.cpu(), future_max.cpu())]


def validate_tune_reference(report, *, base_seed, checkpoint_sha, inventory):
    require(isinstance(report, dict) and report.get("status") == "completed"
            and report.get("precision") == "bf16" and report.get("precision_requested") == "bf16"
            and report.get("time_input") == "nominal"
            and report.get("selection_performed") is False
            and report.get("final_val_accessed") is False, "Immutable completed BF16 tune report required")
    protocol = report.get("protocol", {})
    args = protocol.get("arguments", {})
    require(protocol.get("status") == "preregistered_before_any_forward"
            and protocol.get("checkpoint_sha256") == checkpoint_sha
            and protocol.get("checkpoint_step") == P4_STEP
            and protocol.get("data", {}).get("split_sha256") == SPLIT_SHA256
            and protocol.get("data", {}).get("receiver_count") == EXPECTED["tune"]["n"]
            and protocol.get("data", {}).get("receiver_rows_sha256") == inventory["rows_sha256"],
            "Tune reference checkpoint/data protocol differs")
    required_args = {"split": "tune", "frame_stride": 5, "max_samples": 0,
                     "scenes": None, "batch": 4, "precision": "bf16",
                     "time_input": "nominal", "conditions": ["normal"], "seed": base_seed}
    require(all(args.get(key) == value for key, value in required_args.items()),
            "Tune reference is not exact normal batch4 evaluation")
    require(report.get("model_load", {}).get("checkpoint_sha256") == checkpoint_sha,
            "Tune reference model checkpoint differs")
    normal = report.get("conditions", {}).get("normal", {})
    records = normal.get("records")
    require(set(report.get("conditions", {})) == {"normal"}
            and isinstance(records, list) and len(records) == EXPECTED["tune"]["n"],
            "Tune reference must contain exactly normal tune1998 records")
    return records


def compare_tune_batch(reference, offset, batch, plan, gt):
    rows = _batch_ids(batch)
    for index, identity in enumerate(rows):
        record = reference[offset + index]
        require(all(record.get(key) == identity[key] for key in ("row", "scenario", "session", "frame")),
                "Tune reference row identity/order differs")
        expected_plan = torch.as_tensor(record.get("pred_abs_xy"), dtype=torch.float32)
        expected_gt = torch.as_tensor(record.get("gt_abs_xy"), dtype=torch.float32)
        observed_plan, observed_gt = plan[index].detach().cpu(), gt[index].detach().cpu().float()
        require(expected_plan.shape == observed_plan.shape == (6, 2)
                and expected_gt.shape == observed_gt.shape == (6, 2)
                and torch.isfinite(expected_plan).all() and torch.isfinite(expected_gt).all(),
                "Tune reference plan/GT must be finite [6,2]")
        plan_equal, gt_equal = torch.equal(observed_plan, expected_plan), torch.equal(observed_gt, expected_gt)
        if not plan_equal or not gt_equal:
            def difference(left, right):
                mask = left != right
                first = torch.nonzero(mask, as_tuple=False)[0].tolist() if mask.any() else None
                return {"bitwise_equal": bool(not mask.any()),
                        "different_elements": int(mask.sum()),
                        "max_abs": float((left - right).abs().max()),
                        "first_mismatch_index": first,
                        "observed_at_first": (float(left[tuple(first)]) if first is not None else None),
                        "reference_at_first": (float(right[tuple(first)]) if first is not None else None)}
            print(json.dumps({"event": "tune_reference_bitwise_mismatch",
                              "identity": identity,
                              "plan": difference(observed_plan, expected_plan),
                              "ground_truth": difference(observed_gt, expected_gt)},
                             allow_nan=False), flush=True)
        require(plan_equal and gt_equal,
                "Tune report and cache forward/GT are not bitwise identical")
        expected_d3 = float(record.get("d3"))
        observed_d3 = float(weighted_d3(plan[index:index + 1], gt[index:index + 1]).item())
        require(np.isfinite(expected_d3) and observed_d3 == expected_d3,
                "Tune report D3 differs from cache forward")
    return len(rows)


def cache_split(model, loader, device, *, state_sha, tune_reference=None):
    features, candidates, costs, labels, targets, ids, buckets, seen_rows = [], [], [], [], [], [], [], []
    first_verified = False
    for cpu_batch in loader:
        batch = to_device(cpu_batch, device)
        inputs = model_inputs(batch, time_input="nominal")
        output, feature = forward_with_decoded(model, inputs, _autocast(device))
        plan, gt = output.get("plan_abs"), batch.get("gt_plan")
        plan_valid = batch.get("plan_valid")
        require(isinstance(plan, torch.Tensor) and plan.dtype == torch.float32
                and plan.shape == gt.shape and plan.shape[1:] == (6, 2)
                and isinstance(plan_valid, torch.Tensor)
                and plan_valid.shape == plan.shape[:-1] and plan_valid.bool().all()
                and torch.isfinite(plan).all() and torch.isfinite(gt).all(),
                "All six finite same-forward FP32 plan/GT points must be valid")
        cand, cost, label = selector_targets(plan, gt)
        if tune_reference is not None:
            compare_tune_batch(tune_reference, len(ids), cpu_batch, plan, gt)
        if not first_verified:
            with torch.inference_mode(), _autocast(device):
                repeated = model(**inputs)
            exact_output_equal(output, repeated)
            require(tensor_state_sha256(model.state_dict()) == state_sha,
                    "Hook/repeated forward changed frozen P4 state")
            first_verified = True
        features.append(feature.cpu())
        candidates.append(cand.cpu())
        costs.append(cost.cpu())
        labels.append(label.cpu())
        targets.append(gt.float().cpu())
        ids.extend(_batch_ids(cpu_batch))
        buckets.extend(diagnostic_bucket(cpu_batch["gt_plan"], cpu_batch["state_target"],
                                         cpu_batch["state_valid"]))
        seen_rows.extend(int(value) for value in cpu_batch["row"])
    payload = {"features": torch.cat(features), "candidate_plans": torch.cat(candidates),
               "candidate_costs": torch.cat(costs), "labels": torch.cat(labels),
               "gt_plan": torch.cat(targets), "rows": torch.tensor(seen_rows, dtype=torch.int64)}
    require(payload["features"].shape == (len(ids), FEATURE_DIM), "Incomplete feature cache")
    require(tune_reference is None or len(ids) == len(tune_reference),
            "Tune reference/cache length differs")
    require(tensor_state_sha256(model.state_dict()) == state_sha, "Frozen P4 state changed")
    return payload, ids, buckets


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--expected-run-manifest-sha256", required=True)
    parser.add_argument("--base-seed", required=True, type=int, choices=(0, 1))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--split", required=True, choices=("train", "tune"))
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--tune-report")
    parser.add_argument("--expected-tune-report-sha256")
    parser.add_argument("--pilot-samples", type=int, choices=(0, 8), default=0,
                        help="8 caches the ordered first eight rows only; 0 is the full fixed split")
    parser.add_argument("--device", choices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"), default="cuda:0")
    args = parser.parse_args(argv)
    if args.batch is None:
        args.batch = CACHE_BATCH[args.split]
    require(args.batch == CACHE_BATCH[args.split] and args.workers >= 0,
            "P5-Z requires train batch8 and exact tune batch4")
    require((args.tune_report is not None) == (args.split == "tune")
            and (args.expected_tune_report_sha256 is not None) == (args.split == "tune"),
            "Tune requires its immutable normal full-tune report; train forbids it")
    return args


def main(argv=None):
    args = arguments(argv)
    output = Path(args.out).expanduser().absolute()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    ensure_new(output)
    ensure_new(manifest_path)
    paths = {"checkpoint": Path(args.checkpoint), "run_manifest": Path(args.run_manifest),
             "split": Path(args.split_manifest),
             "supervision": Path(args.supervision_root) / "supervision_manifest.json",
             "calibration": Path(args.supervision_root) / "calibration.npz",
             "source_manifest": Path(args.source_manifest)}
    expected = {"checkpoint": args.expected_checkpoint_sha256,
                "run_manifest": args.expected_run_manifest_sha256,
                "split": SPLIT_SHA256, "supervision": SUPERVISION_SHA256,
                "calibration": CALIBRATION_SHA256,
                "source_manifest": args.expected_source_manifest_sha256}
    if args.split == "tune":
        paths["tune_report"] = Path(args.tune_report)
        expected["tune_report"] = args.expected_tune_report_sha256
    require(all(valid_sha(value) for value in expected.values()), "All input SHA256 pins are required")
    before = {name: file_sha(path) for name, path in paths.items()}
    require(before == expected, "Pinned input artifact SHA256 mismatch")
    source_document = read_pinned_json(paths["source_manifest"], expected["source_manifest"])
    source_files = validate_source_manifest(source_document)
    # Match the immutable evaluator's seed point before device/model creation.
    torch.manual_seed(args.base_seed)
    device = torch.device(args.device)
    require(device.type == "cuda", "Actual feature caching requires CUDA BF16 inference")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    model, training_manifest = load_frozen_model(
        paths["checkpoint"], expected["checkpoint"], paths["run_manifest"],
        expected["run_manifest"], args.base_seed, device)
    state_sha = tensor_state_sha256(model.state_dict())
    spec = EXPECTED[args.split]
    dataset = MotionDriveDataset(
        data_root=args.data_root, split_manifest=args.split_manifest, split=args.split,
        supervision_root=args.supervision_root, min_frame=30, frame_stride=spec["stride"],
        max_samples=0, augment=False, seed=args.base_seed)
    inventory = dataset_inventory(dataset)
    expected_rows = training_manifest["train_rows_sha256" if args.split == "train" else "eval_rows_sha256"]
    require(inventory["rows_sha256"] == expected_rows, "Dataset row order differs from P4 training")
    tune_reference = None
    if args.split == "tune":
        tune_report = read_pinned_json(paths["tune_report"], expected["tune_report"])
        tune_reference = validate_tune_reference(tune_report, base_seed=args.base_seed,
                                                 checkpoint_sha=expected["checkpoint"],
                                                 inventory=inventory)
    selected_n = args.pilot_samples or len(dataset)
    selected_rows = np.asarray(dataset.rows[:selected_n], dtype=np.int64)
    selected_dataset = Subset(dataset, range(selected_n)) if args.pilot_samples else dataset
    if tune_reference is not None:
        tune_reference = tune_reference[:selected_n]
    loader = DataLoader(selected_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=True, drop_last=False)
    payload, ids, buckets = cache_split(model, loader, device, state_sha=state_sha,
                                        tune_reference=tune_reference)
    require(rows_sha256(payload["rows"].numpy()) == rows_sha256(selected_rows),
            "Observed cache row order differs")
    metadata = {
        "schema_version": 1, "status": "completed", "purpose": "P5-Z offline diagnostic cache",
        "created_utc": datetime.now(timezone.utc).isoformat(), "split": args.split,
        "execution_scope": "ordered_first8_pilot" if args.pilot_samples else "full_fixed_split",
        "pilot_samples": args.pilot_samples, "full_cache_completed": args.pilot_samples == 0,
        "base_seed": args.base_seed, "checkpoint_step": P4_STEP,
        "checkpoint_sha256": expected["checkpoint"],
        "completed_run_manifest_sha256": expected["run_manifest"],
        "training_git_sha": P4_GIT_SHA, "evaluation_source_git_sha": source_document["source_git_sha"],
        "runtime": runtime_receipt(device),
        "source_manifest_sha256": expected["source_manifest"], "source_files": source_files,
        "data": {**inventory, "split_sha256": SPLIT_SHA256,
                 "supervision_manifest_sha256": SUPERVISION_SHA256,
                 "canonical_calibration_sha256": CALIBRATION_SHA256,
                 "ego_cache_sha256": dataset.cache_sha, "augment": False,
                 "frame_stride": spec["stride"], "min_frame": 30, "max_samples": 0,
                 "batch": args.batch, "cached_n": selected_n,
                 "cached_rows_sha256": rows_sha256(selected_rows),
                 "selection": "ordered first eight rows" if args.pilot_samples else "all ordered rows"},
        "model": {"mode": "eval", "all_parameters_frozen": True, "fixed_bn": True,
                  "time_input": "nominal", "precision": "bf16_encoder_fp32_planner",
                  "state_sha256_before_after": state_sha,
                  "feature_producing_hooked_forwards_per_cached_sample": 1,
                  "extra_unhooked_forwards": {"first_batch_only": 1, "cached_or_trained_on": False},
                  "hook": "planner.xy_head forward_pre_hook; exactly decoded.float()"},
        "immutable_tune_report_sha256": expected.get("tune_report"),
        "immutable_tune_rowwise_plan_gt_d3_bitwise_verified": args.split == "tune",
        "immutable_tune_rows_compared": selected_n if args.split == "tune" else 0,
        "feature_contract": {"dtype": "float32", "dimension": FEATURE_DIM,
                             "components": list(FEATURE_COMPONENTS),
                             "allowed": [item["name"] for item in FEATURE_COMPONENTS],
                             "forbidden": ["raw_goal", "pose", "time", "row_or_identity", "ground_truth", "candidate_cost"],
                             "derived_planner_hidden_may_contain_normal_p4_goal_context": True},
        "candidate_order": list(CANDIDATE_ORDER), "class_order": {"zero": ZERO, "move": MOVE},
        "label_rule": "ZERO iff D3(zero,GT) < D3(P4,GT); ties MOVE",
        "metric": "official cumulative-XY D3 weights [11,11,5,5,2,2]/36",
        "ids": ids, "diagnostic_buckets": buckets,
        "diagnostic_bucket_definition": {
            "steady": "current GT speed <0.2m/s and max six-future-GT position norm <=0.2m",
            "depart": "current GT speed <0.2m/s and max six-future-GT position norm >0.2m",
            "nonstop": "current GT speed >=0.2m/s or current speed invalid"},
        "diagnostic_buckets_are_selector_features": False,
        "final_val_accessed": False, "selection_performed": False,
        "production_api_modified": False, "accuracy_latency_compliance_certified": False,
    }
    payload["metadata"] = {key: value for key, value in metadata.items()
                           if key not in ("ids", "diagnostic_buckets")}
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise ValueError(f"Refusing to overwrite output: {output}") from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    metadata["cache_sha256"] = file_sha(output)
    metadata["cache_bytes"] = output.stat().st_size
    metadata["inputs_sha256_before"] = before
    metadata["inputs_sha256_after"] = {name: file_sha(path) for name, path in paths.items()}
    require(metadata["inputs_sha256_after"] == before, "Input artifacts changed during caching")
    require(validate_source_manifest(source_document) == source_files, "Runtime source changed during caching")
    atomic_new_json(manifest_path, metadata)
    print(json.dumps({"status": "completed", "cache": str(output),
                      "cache_sha256": metadata["cache_sha256"], "manifest": str(manifest_path),
                      "manifest_sha256": file_sha(manifest_path), "rows": len(ids),
                      "split": args.split, "execution_scope": metadata["execution_scope"],
                      "gpu_cache_execution": True,
                      "diagnostic_only": True, "final_val_accessed": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
