#!/usr/bin/env python3
"""Export a trusted local V2 training checkpoint without optimizer/RNG state.

The COMPLETE saved manifest.model_config is required: no architecture, motion
mode, output unit, or G/S setting is filled from current defaults. The resulting
CPU tensor bundle loads unchanged through audit_motiondrive_v2.construct_model
and the benchmark's shared loader. This is packaging, not model conversion.

Training checkpoints contain Python/NumPy RNG pickles. Only use --checkpoint
with a trusted, user-owned local file; loading it uses weights_only=False.
The exported bundle itself can be opened with weights_only=True.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config


def file_sha256(path):
    with open(path, "rb") as stream:
        return stream_sha256(stream)


def stream_sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def validate_complete_config(saved):
    """Require every current config field explicitly, including default values."""
    if not isinstance(saved, dict):
        raise ValueError("manifest.model_config must be a complete dictionary")
    required = {field.name for field in dataclasses.fields(MotionDriveV2Config)}
    missing, extra = required - saved.keys(), saved.keys() - required
    if missing or extra:
        raise ValueError(f"Incomplete/unknown model_config: missing={sorted(missing)}, "
                         f"unknown={sorted(extra)}; explicit reviewed migration required")
    for name in ("goal_on", "state_on"):
        if type(saved[name]) is not bool:
            raise ValueError(f"model_config.{name} must be an explicit boolean")
    for name in ("channels", "n_history", "correlation_channels", "correlation_radius",
                 "scene_attention_channels", "scene_chunk_size", "planner_layers", "planner_heads"):
        value = saved[name]
        if type(value) is not int or value < (0 if name == "correlation_radius" else 1):
            raise ValueError(f"Invalid integer model_config.{name}")
    for name in ("grid_size", "motion_grid"):
        value = saved[name]
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(type(x) is not int or x <= 0 for x in value)):
            raise ValueError(f"model_config.{name} must contain two positive integers")
    for name in ("x_range", "y_range", "heights", "plan_output_scale"):
        value = saved[name]
        if (not isinstance(value, (list, tuple)) or not value
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
            raise ValueError(f"Invalid finite numeric model_config.{name}")
        if name != "heights" and len(value) != 2:
            raise ValueError(f"model_config.{name} must have two entries")
        if name.endswith("_range") and value[0] >= value[1]:
            raise ValueError(f"model_config.{name} must be increasing")
    if saved["backbone_arch"] not in ("resnet34", "resnet50"):
        raise ValueError("Unsupported explicit backbone_arch")
    # Dataclass validation also checks head divisibility, positive scales/mode.
    return MotionDriveV2Config(**saved)


def prepare_bundle(checkpoint, *, source_path, source_sha256, source_bytes):
    """Validate a trainer payload, then keep weights and JSON-only provenance."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a trainer checkpoint dictionary")
    manifest = checkpoint.get("manifest")
    if not isinstance(manifest, dict) or "model_config" not in manifest:
        raise ValueError("Required manifest.model_config is missing; no default fallback")
    config = validate_complete_config(manifest["model_config"])
    step = checkpoint.get("step")
    if type(step) is not int or step < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    split_sha = manifest.get("split_sha256")
    if not isinstance(split_sha, str) or re.fullmatch(r"[0-9a-fA-F]{64}", split_sha) is None:
        raise ValueError("Required split_sha256 lineage is missing/invalid")
    try:
        # Trainer manifests are JSON records. This strips e.g. TorchVersion's
        # str subclass and disallows tensors/custom pickles in deployment metadata.
        clean_manifest = json.loads(json.dumps(manifest, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Manifest must be finite JSON-only metadata") from exc
    # Canonical explicit full settings; tuple/list representation is immaterial.
    clean_manifest["model_config"] = dataclasses.asdict(config)
    state = checkpoint.get("model")
    if (not isinstance(state, dict) or not state
            or any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
                   for k, v in state.items())):
        raise ValueError("Checkpoint model must be a nonempty tensor state dictionary")
    # Strict validation catches missing/unexpected keys and shape mismatches.
    # CPU construction only; no CUDA, optimizer, export trace, or model forward.
    with torch.random.fork_rng(devices=[]):
        validation_model = MotionDriveV2(config)
        validation_model.load_state_dict(state, strict=True)
    del validation_model
    # Clones remove view/backing-storage aliases; preserve every tensor's dtype.
    cpu_state = {name: value.detach().to(device="cpu").contiguous().clone()
                 for name, value in state.items()}
    source = {"path": str(source_path), "sha256": source_sha256,
              "bytes": int(source_bytes), "step": step,
              "split_sha256": split_sha,
              "training_git_sha": clean_manifest.get("git_sha")}
    bundle = {"format": "motiondrive_v2_inference", "format_version": 1,
              "model": cpu_state, "manifest": clean_manifest, "step": step,
              "source_checkpoint": source,
              "inference_export": {"script_sha256": file_sha256(__file__),
                                   "weights_device": "cpu", "weights_dtype_preserved": True,
                                   "optimizer_and_rng_excluded": True,
                                   "complete_config_required": True}}
    if "epoch" in checkpoint:
        epoch = checkpoint["epoch"]
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Checkpoint epoch must be a nonnegative integer")
        bundle["epoch"] = source["epoch"] = epoch
    return bundle


def export_checkpoint(checkpoint_path, output_path):
    """Publish atomically without replacing any existing target, even in a race.

    The output parent must already exist. A same-directory temporary file is
    linked to the final name only after serialization/fsync, then removed.
    """
    source = Path(checkpoint_path).expanduser().resolve(strict=True)
    target = Path(output_path).expanduser().absolute()
    if not source.is_file():
        raise ValueError("--checkpoint must be an existing trusted local file")
    if target.resolve(strict=False) == source:
        raise ValueError("Input checkpoint overwrite is forbidden")
    if os.path.lexists(target):
        raise FileExistsError(f"Output already exists: {target}")
    if not target.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    with source.open("rb") as stream:
        before = os.fstat(stream.fileno())
        source_sha = stream_sha256(stream)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source checkpoint changed while being read; use a stable checkpoint")
    bundle = prepare_bundle(checkpoint, source_path=source,
                            source_sha256=source_sha, source_bytes=before.st_size)
    del checkpoint
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", prefix=f".{target.name}.",
                                         suffix=".tmp", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(bundle, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Unlike rename/replace, link NEVER overwrites a target created meanwhile.
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"output": str(target), "output_sha256": file_sha256(target),
            "output_bytes": target.stat().st_size, "source_checkpoint": bundle["source_checkpoint"],
            "model_config": bundle["manifest"]["model_config"],
            "tensor_count": len(bundle["model"]), "optimizer_and_rng_excluded": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Trusted user-owned local training checkpoint; loads Python pickle")
    parser.add_argument("--out", required=True, help="New bundle path in an existing directory")
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.out), indent=2))


if __name__ == "__main__":
    main()
