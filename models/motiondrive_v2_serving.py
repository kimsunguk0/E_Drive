"""One stateless, complete V2 forward and lossless absolute-XY serialization.

The caller selects a SHA-pinned deployment bundle and prepares one raw clip
with the GT-free adapter. This module never chooses checkpoints, opens dataset
clips, constructs trajectories from metadata, or submits to the challenge.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import torch

from .motiondrive_v2_input_contract import (
    INPUT_KEYS, NOMINAL_SECONDS, input_contract, require_input_contract,
)

SHAPES = {
    "images": (1, 6, 3, 432, 768),
    "history_images": (1, 4, 3, 216, 384),
    "lidar2img": (1, 6, 4, 4),
    "history_transforms": (1, 4, 4, 4),
    "time_offsets": (1, 4),
    "goal_xy": (1, 2),
}


def validate_bundle_contract(bundle: Mapping) -> None:
    """Verify packaged declarations, not unavailable remote training artifacts.

    A separately supplied SHA pins the entire bundle in load_deployment_bundle.
    Original B200 files were checked by the exporter, not re-opened on serving.
    These checks do not imply accuracy, clean training exit or organizer approval.
    """
    from scripts.export_motiondrive_v2_inference import (
        C1_CANONICAL_SHA256, C1_SUPERVISION_SHA256, RAWTIME_SPLIT_SHA256,
    )
    if (not isinstance(bundle, Mapping) or bundle.get("format") != "motiondrive_v2_inference" or
            bundle.get("format_version") != 1 or bundle.get("input_contract") != input_contract()):
        raise ValueError("A verified geometry_v2/nominal inference bundle is required")
    provenance = bundle.get("deployment_provenance", {})
    manifest, source = bundle.get("manifest", {}), bundle.get("source_checkpoint", {})
    selected = provenance.get("selected_checkpoint_sha256")
    if (provenance.get("mode") != "geometry-v2-nominal" or
            provenance.get("status") != "input_contract_lineage_verified" or
            not isinstance(selected, str) or re.fullmatch(r"[0-9a-f]{64}", selected) is None or
            source.get("sha256") != selected or type(bundle.get("step")) is not int or
            bundle["step"] <= 0 or source.get("step") != bundle["step"] or
            provenance.get("checkpoint_step") != bundle["step"]):
        raise ValueError("Deployment bundle checkpoint lineage is missing or inconsistent")
    evidence = provenance.get("evidence", {})
    for name, expected in (("geometry_edition", "geometry_v2"), ("time_input", "nominal"),
                           ("canonical_calibration_sha256", C1_CANONICAL_SHA256),
                           ("supervision_manifest_sha256", C1_SUPERVISION_SHA256),
                           ("split_sha256", RAWTIME_SPLIT_SHA256)):
        if provenance.get(name) != expected:
            raise ValueError(f"Contradictory or missing packaged provenance: {name}")
    completed = provenance.get("completed_run_step")
    if (type(completed) is not int or completed < bundle["step"] or
            completed != manifest.get("arguments", {}).get("steps")):
        raise ValueError("Packaged completed-run step contradicts the selected checkpoint")
    for role, expected in (("source_checkpoint", selected), ("canonical_calibration", C1_CANONICAL_SHA256),
                           ("supervision_manifest", C1_SUPERVISION_SHA256), ("split_manifest", RAWTIME_SPLIT_SHA256)):
        if not isinstance(evidence.get(role), Mapping) or evidence[role].get("sha256") != expected:
            raise ValueError(f"Missing or wrong packaged input lineage: {role}")
    from scripts.motiondrive_v2_training import time_input_policy
    if (manifest.get("time_input") != "nominal" or
            manifest.get("arguments", {}).get("time_input") != "nominal" or
            manifest.get("time_input_policy") != time_input_policy("nominal") or
            manifest.get("supervision_manifest_sha256") != C1_SUPERVISION_SHA256 or
            manifest.get("split_sha256") != RAWTIME_SPLIT_SHA256 or
            manifest.get("model_config", {}).get("n_history") != 4):
        raise ValueError("Packaged training settings disagree with deployment inputs")


def explicit_device(value):
    device = torch.device(value)
    if not (device.type == "cpu" and device.index is None or
            device.type == "cuda" and device.index is not None):
        raise ValueError("Use cpu or an explicit CUDA device such as cuda:0")
    return device


def load_deployment_bundle(path, *, expected_bundle_sha256: str, device="cpu"):
    """Load only an explicitly SHA-pinned safe tensor bundle, with strict config.

    This never promotes a generic P0/P1 export to the corrected serving contract.
    No architecture overrides, inferred defaults, optimizer or feature cache.
    """
    from scripts.export_motiondrive_v2_inference import validate_complete_config
    from .motiondrive_v2 import MotionDriveV2
    device = explicit_device(device)
    if not isinstance(expected_bundle_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_bundle_sha256) is None:
        raise ValueError("Explicit lowercase bundle SHA256 is required")
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != expected_bundle_sha256:
            raise ValueError("Deployment bundle SHA256 mismatch")
        stream.seek(0)
        bundle = torch.load(stream, map_location="cpu", weights_only=True)
        # Protect against in-place modification between hashing and loading.
        stream.seek(0)
        after = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            after.update(block)
        if after.hexdigest() != expected_bundle_sha256:
            raise ValueError("Deployment bundle changed during loading")
    validate_bundle_contract(bundle)
    config = validate_complete_config(bundle["manifest"]["model_config"])
    state = bundle.get("model")
    if (not isinstance(state, Mapping) or not state or
            any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values())):
        raise ValueError("Deployment model state must contain finite tensors")
    with torch.random.fork_rng(devices=[]):
        model = MotionDriveV2(config)
        target = model.state_dict()
        if set(state) != set(target) or any(state[name].dtype != target[name].dtype for name in target):
            raise ValueError("Strict serving must not silently cast checkpoint parameter dtypes")
        model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, json.loads(json.dumps(bundle["input_contract"], allow_nan=False))


def validate_clip_inputs(inputs: Mapping, contract: Mapping) -> None:
    """Reject labels, cached features, raw timestamps and unexpected batch sizes."""
    require_input_contract(contract)
    if not isinstance(inputs, Mapping) or set(inputs) != set(INPUT_KEYS):
        raise ValueError("Exactly the six raw-clip model input tensors are required")
    for name, shape in SHAPES.items():
        value = inputs[name]
        if (not isinstance(value, torch.Tensor) or value.shape != shape or
                value.device.type != "cpu" or value.dtype != torch.float32 or
                value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError(f"Expected finite detached CPU float32 {name}{shape}")
    nominal = torch.tensor([NOMINAL_SECONDS], dtype=torch.float32)
    if not torch.equal(inputs["time_offsets"].contiguous().view(torch.uint8), nominal.view(torch.uint8)):
        raise ValueError("Deployment time_offsets must be exactly nominal float32")


def forward_clip(model, inputs: Mapping, contract: Mapping, *, device="cpu", precision="fp32"):
    """Run the public model forward ONCE; it contains all eleven image encodings.

    No forward_parts/plan_from_features fast path, feature caching, coordinate
    correction or status replacement exists here. CPU checking/H2D/D2H are
    outside the model-forward timing boundary; this function is not a timer.
    Model construction and deployment-bundle lineage checks belong to its loader.
    """
    validate_clip_inputs(inputs, contract)
    if precision not in ("fp32", "bf16"):
        raise ValueError("Only explicit fp32 or bf16 inference is supported")
    device = explicit_device(device)
    if model.training or any(module.training for module in model.modules()):
        raise ValueError("The complete model must already be in evaluation mode")
    for value in (*model.parameters(), *model.buffers()):
        if value.device.type != device.type or (device.index is not None and value.device.index != device.index):
            raise ValueError("Model must already be loaded on the explicitly requested device")
    tensors = {key: inputs[key].to(device=device, non_blocking=False) for key in INPUT_KEYS}
    amp = (contextlib.nullcontext() if precision == "fp32" else
           torch.autocast(device_type=device.type, dtype=torch.bfloat16))
    with torch.inference_mode(), amp:
        outputs = model(**tensors)
    if not isinstance(outputs, Mapping) or "plan_abs" not in outputs:
        raise ValueError("The complete model must return its absolute plan_abs")
    plan = outputs["plan_abs"]
    if (not isinstance(plan, torch.Tensor) or plan.shape != (1, 6, 2) or
            plan.dtype != torch.float32 or not torch.isfinite(plan).all()):
        raise ValueError("Final neural coordinates must be finite FP32 [1,6,2]")
    return plan.detach()[0].to(device="cpu").contiguous()


def absolute_plan_to_list(plan: torch.Tensor) -> list[list[float]]:
    """Official six absolute ego-frame XY points; NO cumsum, clamp or resampling."""
    if (not isinstance(plan, torch.Tensor) or plan.shape != (6, 2) or
            plan.device.type != "cpu" or plan.dtype != torch.float32 or
            not torch.isfinite(plan).all()):
        raise ValueError("Expected finite CPU FP32 absolute [6,2] coordinates")
    return plan.detach().tolist()
