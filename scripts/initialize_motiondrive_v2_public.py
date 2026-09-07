#!/usr/bin/env python3
"""Create a zero-ETRI-update public-R50 initialization, on CPU only.

The public backbone is SHA-pinned and fully injected; compact FPN/scene/motion/
planner retain their seed-zero initialization. This is NOT a trained P0 or an
inference/submission bundle. Its empty optimizer supports weights-only --init,
not --resume. No dataset, labels, image tensors, or GPU forward are read/run.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from scripts.export_motiondrive_v2_inference import validate_complete_config
from scripts.motiondrive_v2_training import (LossWeights, set_training_mode,
    tensor_state_sha256, time_input_policy)

PUBLIC_SHA256 = "4096396018c0cf59fbe0eb1afe6e269f4676b34460bed5eedde5d7680d58bb4e"
SPLIT_SHA256 = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
PUBLIC_BACKBONE_ENTRIES = 318
SEED = 0
SOURCE_FILES = (
    "scripts/initialize_motiondrive_v2_public.py",
    "models/__init__.py", "models/motiondrive_v2/__init__.py",
    "models/motiondrive_v2/config.py", "models/motiondrive_v2/model.py",
    "models/motiondrive_v2/planner.py", "models/motiondrive_v2/scene_encoder.py",
    "models/motiondrive_v2/motion_encoder.py", "scripts/sparse_scoredrive.py",
    "scripts/export_motiondrive_v2_inference.py", "scripts/motiondrive_v2_training.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_cpu_only():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "CPU initialization requires CUDA_VISIBLE_DEVICES='' explicitly")
    require(not torch.cuda.is_initialized(), "CUDA must not be initialized")


def source_snapshot(expected_git_sha=None):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if expected_git_sha is not None:
        require(isinstance(expected_git_sha, str) and re.fullmatch(r"[0-9a-f]{40}", expected_git_sha),
                "Expected full lowercase Git commit SHA")
        require(commit == expected_git_sha, "Initializer Git commit mismatch")
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=ROOT, text=True)
        require(not dirty.strip(), "Pinned initialization requires tracked-clean sources")
    return {"git_sha": commit, "file_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES}}


def backbone_key(name):
    return name.startswith(tuple("backbone_fpn." + prefix for prefix in
                                 ("stem.", "layer1.", "layer2.", "layer3.", "layer4.")))


def strict_cpu_state(model, state):
    target = model.state_dict()
    require(isinstance(state, dict) and set(target) == set(state), "Complete model state keys required")
    for name, value in state.items():
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and
                value.layout == torch.strided and value.dtype == target[name].dtype and
                value.shape == target[name].shape and bool(torch.isfinite(value).all()),
                f"Invalid finite CPU state/shape/dtype: {name}")
    model.load_state_dict(state, strict=True)


def verify_sources(public, split, sources, expected_git_sha):
    assert_cpu_only()
    require(sha256(public) == PUBLIC_SHA256, "Public R50 source SHA mismatch/changed")
    require(sha256(split) == SPLIT_SHA256, "Pinned grouped split SHA mismatch/changed")
    require(source_snapshot(expected_git_sha) == sources, "Initialization source files changed")


def prepare_public_payload(public_checkpoint, split_manifest, expected_git_sha=None):
    """Validate inputs and build CPU tensors only; no file publication."""
    assert_cpu_only()
    public = Path(public_checkpoint).resolve(strict=True)
    split = Path(split_manifest).resolve(strict=True)
    require(public.is_file() and split.is_file(), "Public checkpoint and split must be files")
    sources = source_snapshot(expected_git_sha)
    verify_sources(public, split, sources, expected_git_sha)
    config = MotionDriveV2Config(backbone_arch="resnet50", goal_on=False, state_on=False,
                               motion_input_mode="low_feature", plan_output_scale=(10., 5.))
    validate_complete_config(config.to_dict())
    with torch.random.fork_rng(devices=[]):
        # Seed only the CPU generator; do not initialize or perturb CUDA RNG.
        torch.random.default_generator.manual_seed(SEED)
        model = MotionDriveV2(config).cpu()
        initial_nonbackbone = {name: value.detach().clone() for name, value in model.state_dict().items()
                               if not backbone_key(name)}
        report = model.load_pretrained_backbone(public)
        require(report.get("injected") == PUBLIC_BACKBONE_ENTRIES and
                report.get("unexpected") == [] and report.get("nonhead_missing") == [] and
                report.get("fpn_initialized_from_checkpoint") is False,
                f"Incomplete public R50 backbone injection: {report}")
        require(sum(backbone_key(name) for name in model.state_dict()) == PUBLIC_BACKBONE_ENTRIES,
                "Unexpected R50 backbone state inventory")
        state = {name: value.detach().cpu().contiguous().clone() for name, value in model.state_dict().items()}
        nonbackbone = {name: state[name] for name in initial_nonbackbone}
        require(tensor_state_sha256(nonbackbone) == tensor_state_sha256(initial_nonbackbone),
                "Public loading altered the fresh FPN/scene/motion/planner tensors")
        bn_count = set_training_mode(model, "fixed")
        require(all(not module.training for module in model.modules()
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)), "BN statistics must be fixed")
        require(all(p.requires_grad for p in model.parameters()), "Initialization must not freeze weights")
        # A new instance must accept every tensor with no missing keys or casts.
        validation_model = MotionDriveV2(validate_complete_config(config.to_dict())).cpu()
        strict_cpu_state(validation_model, state)
        initial_sha = tensor_state_sha256(state)
        require(tensor_state_sha256(validation_model.state_dict()) == initial_sha,
                "Full strict restoration changed tensor bytes")
    verify_sources(public, split, sources, expected_git_sha)
    manifest = {
        "schema_version": 1, "git_sha": sources["git_sha"], "source": sources,
        "model_config": config.to_dict(), "split_sha256": SPLIT_SHA256,
        "arguments": {"phase": "pretrain", "goal_on": 0, "state_on": 0, "arch": "resnet50",
                      "motion_input_mode": "low_feature", "plan_output_scale": [10., 5.],
                      "bn_policy": "fixed", "time_input": "nominal", "seed": SEED,
                      "pretrained": str(public), "split_manifest": str(split),
                      "init": None, "resume": None, "allow_unpretrained": False},
        "time_input": "nominal", "time_input_policy": time_input_policy("nominal"),
        "pretrained_sha256": PUBLIC_SHA256, "public_checkpoint_sha256": PUBLIC_SHA256,
        "public_source": {"path": str(public), "sha256": PUBLIC_SHA256,
                          "bytes": public.stat().st_size, "scope": "public nuImages R50 backbone only"},
        "load_report": {"backbone": report}, "initialization_seed": SEED, "seed": SEED,
        "initial_model_state_sha256": initial_sha,
        "initial_parameter_count": sum(p.numel() for p in model.parameters()),
        "nonbackbone_initial_state_sha256": tensor_state_sha256(nonbackbone),
        "nonbackbone_seed_initialization_preserved": True,
        "bn_training": {"policy": "fixed", "modules": bn_count,
                        "affine_and_backbone_weights_trainable": True,
                        "running_statistics_source": "public pretrained checkpoint; zero ETRI updates"},
        "loss_weights": dataclasses.asdict(LossWeights(plan=0.)),
        "step": 0, "status": "public_initialization_only", "etri_optimizer_updates": 0,
        "etri_optimizer_steps": 0,
        "etri_samples_seen": 0, "labels_read": False, "image_data_read": False,
        "split_file_usage": "SHA-only lineage check; no scene/label selection",
        "plan_output_scale_origin": "constructor config (10,5), no inverse reparameterization",
        "optimizer_state_initialized": False, "compatible_usage": "weights-only --init, NOT --resume",
        "torch": str(torch.__version__), "device": "cpu", "cuda_initialized": False,
        "source_files_unchanged": True,
    }
    json.dumps(manifest, allow_nan=False)
    return {"model": state, "step": 0, "epoch": 0, "optimizer": {}, "manifest": manifest}


def new_path(value, suffixes):
    given = Path(value).absolute()
    require(not os.path.lexists(given), f"Refusing existing output: {given}")
    result = given.resolve()
    require(not os.path.lexists(result) and result.parent.is_dir() and result.suffix in suffixes,
            f"Expected a new output with existing parent: {result}")
    return result


def initialize_public_checkpoint(public_checkpoint, split_manifest, output, report, expected_git_sha=None):
    """Exclusively create checkpoint/report; never overwrite or remove any file.

    A failure during publication may leave a new incomplete artifact. It is not
    approved for use without a successful report and exit; no automatic deletion
    or replacement is attempted. Choose a new path for any retry.
    """
    assert_cpu_only()
    output_path, report_path = new_path(output, (".pt", ".pth")), new_path(report, (".json",))
    require(output_path != report_path, "Checkpoint/report paths must differ")
    payload = prepare_public_payload(public_checkpoint, split_manifest, expected_git_sha)
    public, split = Path(public_checkpoint).resolve(), Path(split_manifest).resolve()
    sources = payload["manifest"]["source"]
    verify_sources(public, split, sources, expected_git_sha)
    with output_path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    # Published tensors must survive the exact safe-pickle CPU round trip.
    restored = torch.load(output_path, map_location="cpu", weights_only=True)
    require(restored["step"] == 0 and restored["optimizer"] == {} and
            tensor_state_sha256(restored["model"]) == payload["manifest"]["initial_model_state_sha256"],
            "Published initialization failed the CPU tensor roundtrip")
    verify_sources(public, split, sources, expected_git_sha)
    result = {
        "status": "public_cpu_initialization_complete", "checkpoint_path": str(output_path),
        "checkpoint_sha256": sha256(output_path), "checkpoint_bytes": output_path.stat().st_size,
        "initial_model_state_sha256": payload["manifest"]["initial_model_state_sha256"],
        "public_checkpoint_sha256": PUBLIC_SHA256, "split_sha256": SPLIT_SHA256,
        "source": sources, "manifest": payload["manifest"],
        "strict_full_model_load_verified": True, "safe_cpu_roundtrip_verified": True,
        "all_source_sha_before_after_equal": True, "etri_optimizer_updates": 0,
        "gpu_used": False, "pid": os.getpid(), "argv": list(sys.argv),
        "accuracy_or_deployment_claim": False,
    }
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--public-checkpoint", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-git-sha")
    args = parser.parse_args(argv)
    result = initialize_public_checkpoint(args.public_checkpoint, args.split_manifest,
                                         args.output, args.report, args.expected_git_sha)
    print(json.dumps({key: result[key] for key in ("status", "checkpoint_path", "checkpoint_sha256",
                                                 "initial_model_state_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
