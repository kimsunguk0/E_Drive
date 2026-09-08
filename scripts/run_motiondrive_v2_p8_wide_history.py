#!/usr/bin/env python3
"""Run one preregistered P8 control/wide P0 or joint stage.

This wrapper binds the temporal overlay and initializer before delegating to
the existing trainer.  P0 retains the historical auxiliary-evaluation cadence;
joint training attaches the P7 zero-slot branch and evaluates tune once at 6000.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_grouped_split_v2 import sha256, validate_manifest
from models.motiondrive_v2_temporal_contract import temporal_contract
from motiondrive_v2_training import tensor_state_sha256

EXPECTED_SPLIT_SHA256 = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
EXPECTED_SUPERVISION_SHA256 = "ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93"
EXPECTED_CALIBRATION_SHA256 = "8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961"
EXPECTED_TRAIN_ROWS_SHA256 = "75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854"
EXPECTED_TUNE_ROWS_SHA256 = "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88"
EXPECTED_I0_FILE_SHA256 = "06d2e68e15ca00d2c0f9ed3c965e198db603fe7e80007c39319d76eba9832e5a"
EXPECTED_I0_MODEL_SHA256 = "7ac28a8f26dc796be70ddb78edff1e1c128c64f43b5982747ba419d5d5200683"
ALLOWED_PHYSICAL_GPU_UUIDS = {
    "GPU-4b804d68-fd61-af14-393a-573c533d5006",
    "GPU-1e9aea73-4e6b-2cb5-b788-f3d99e6dc6f8",
}
GPU_ASSIGNMENTS = {
    (0, "control"): "GPU-4b804d68-fd61-af14-393a-573c533d5006",
    (0, "wide"): "GPU-1e9aea73-4e6b-2cb5-b788-f3d99e6dc6f8",
    (1, "control"): "GPU-1e9aea73-4e6b-2cb5-b788-f3d99e6dc6f8",
    (1, "wide"): "GPU-4b804d68-fd61-af14-393a-573c533d5006",
}
SOURCE_FILES = {
    "models/motiondrive_v2_temporal_contract.py",
    "models/motiondrive_v2/__init__.py", "models/motiondrive_v2/config.py",
    "models/motiondrive_v2/cross_cell_goal_residual.py", "models/motiondrive_v2/model.py",
    "models/motiondrive_v2/motion_encoder.py", "models/motiondrive_v2/planner.py",
    "models/motiondrive_v2/scene_encoder.py", "scripts/build_grouped_split_v2.py",
    "scripts/build_motiondrive_v2_history_overlay.py", "scripts/motiondrive_v2_data.py",
    "scripts/motiondrive_v2_training.py", "scripts/run_motiondrive_v2_p8_wide_history.py",
    "scripts/sparse_scoredrive.py", "scripts/train_motiondrive_v2.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path) -> str:
    return sha256(str(path))


def validate_source_manifest(path: str, expected_sha256: str) -> dict:
    source_path = Path(path).resolve()
    require(len(expected_sha256) == 64 and file_sha(source_path) == expected_sha256,
            "P8 source-manifest SHA mismatch")
    manifest = json.loads(source_path.read_text())
    require(set(manifest) == {"schema_version", "git_sha", "file_sha256"}
            and manifest["schema_version"] == 1
            and isinstance(manifest["git_sha"], str) and len(manifest["git_sha"]) == 40,
            "P8 source manifest schema mismatch")
    files = manifest["file_sha256"]
    require(isinstance(files, dict) and set(files) == SOURCE_FILES,
            "P8 source manifest must contain the exact runtime closure")
    require(all(isinstance(value, str) and len(value) == 64 for value in files.values()),
            "P8 source hash malformed")
    actual = {name: file_sha(ROOT / name) for name in SOURCE_FILES}
    require(actual == files, "P8 runtime source differs from pinned source manifest")
    return {"path": str(source_path), "sha256": expected_sha256,
            "git_sha": manifest["git_sha"], "file_sha256": files}


def validate_runtime_namespace(args) -> dict:
    require(args.gpu == 0 and args.workers == 4 and args.cuda_memory_limit_mib == 12000
            and args.cuda_min_free_mib == 8192,
            "P8 fixed logical-GPU/worker/memory contract mismatch")
    require(args.expected_physical_gpu_uuid == GPU_ASSIGNMENTS[(args.base_seed, args.arm)],
            "P8 physical GPU does not match the preregistered counterbalanced assignment")
    if args.preflight_only:
        return {"gpu_used": False, "logical_gpu": 0,
                "expected_physical_gpu_uuid": args.expected_physical_gpu_uuid}
    namespace = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(namespace == args.expected_physical_gpu_uuid and "," not in namespace
            and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "P8 requires one UUID-isolated CUDA device")
    raw_uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    actual_uuid = raw_uuid if raw_uuid.startswith("GPU-") else "GPU-" + raw_uuid
    require(actual_uuid == args.expected_physical_gpu_uuid, "P8 physical GPU UUID mismatch")
    return {"gpu_used": True, "logical_gpu": 0, "cuda_visible_devices": namespace,
            "expected_physical_gpu_uuid": args.expected_physical_gpu_uuid,
            "observed_physical_gpu_uuid_raw": raw_uuid, "actual_physical_gpu_uuid": actual_uuid}


def validate_loaded_overlays(datasets) -> dict[str, int]:
    validated_rows = {}
    for split_name, dataset in datasets:
        scenes = sorted(set(dataset.scene_names[dataset.rows]))
        loaded = [dataset._history_overlay(scene) for scene in scenes]
        require(all(item is not None for item in loaded),
                f"P8 {split_name} overlay unexpectedly fell back to base history")
        validated_rows[split_name] = sum(len(item["row"]) for item in loaded)
        require(validated_rows[split_name] == len(dataset),
                f"P8 {split_name} overlay internal row coverage mismatch")
    return validated_rows


def validate_data(args) -> dict:
    split_path = Path(args.split_manifest).resolve()
    supervision = Path(args.supervision_root).resolve()
    overlay = Path(args.history_overlay_root).resolve() / "overlay_manifest.json"
    require(file_sha(split_path) == EXPECTED_SPLIT_SHA256, "P8 split SHA mismatch")
    require(file_sha(supervision / "supervision_manifest.json") == EXPECTED_SUPERVISION_SHA256,
            "P8 C1 supervision SHA mismatch")
    require(file_sha(supervision / "calibration.npz") == EXPECTED_CALIBRATION_SHA256,
            "P8 calibration SHA mismatch")
    require(file_sha(overlay) == args.expected_history_overlay_sha256,
            "P8 temporal overlay SHA mismatch")
    split = json.loads(split_path.read_text())
    validate_manifest(split)
    require(len(split["splits"]["train"]) == 203 and len(split["splits"]["tune"]) == 37,
            "P8 requires train203/tune37")
    overlay_manifest = json.loads(overlay.read_text())
    expected_scenes = set(split["splits"]["train"]) | set(split["splits"]["tune"])
    artifacts = overlay_manifest.get("artifacts")
    require(isinstance(artifacts, dict) and set(artifacts) == expected_scenes,
            "P8 overlay artifact inventory must equal train203+tune37")
    artifact_paths = []
    for scene in sorted(expected_scenes):
        spec = artifacts[scene]
        require(isinstance(spec, dict) and spec.get("file") == f"{scene}.npz"
                and isinstance(spec.get("sha256"), str) and len(spec["sha256"]) == 64,
                f"P8 overlay artifact declaration malformed: {scene}")
        artifact_path = Path(args.history_overlay_root).resolve() / spec["file"]
        require(artifact_path.parent == Path(args.history_overlay_root).resolve()
                and file_sha(artifact_path) == spec["sha256"],
                f"P8 overlay artifact SHA mismatch: {scene}")
        artifact_paths.append(str(artifact_path))
    from motiondrive_v2_data import MotionDriveDataset
    common = dict(data_root=args.data_root, split_manifest=str(split_path),
                  supervision_root=str(supervision), min_frame=30, max_samples=0,
                  seed=args.base_seed, history_contract=args.arm,
                  history_overlay_root=args.history_overlay_root,
                  expected_history_overlay_sha256=args.expected_history_overlay_sha256)
    train = MotionDriveDataset(split="train", frame_stride=1, augment=True, **common)
    tune = MotionDriveDataset(split="tune", frame_stride=5, augment=False, **common)
    train_sha = hashlib.sha256(np.asarray(train.rows, dtype="<i8").tobytes()).hexdigest()
    tune_sha = hashlib.sha256(np.asarray(tune.rows, dtype="<i8").tobytes()).hexdigest()
    require((len(train), train_sha) == (54810, EXPECTED_TRAIN_ROWS_SHA256),
            "P8 train rows mismatch")
    require((len(tune), tune_sha) == (1998, EXPECTED_TUNE_ROWS_SHA256),
            "P8 tune rows mismatch")
    validated_rows = validate_loaded_overlays((("train", train), ("tune", tune)))
    return {"split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "supervision_manifest_sha256": EXPECTED_SUPERVISION_SHA256,
            "calibration_sha256": EXPECTED_CALIBRATION_SHA256,
            "history_overlay_manifest_sha256": args.expected_history_overlay_sha256,
            "history_overlay_artifact_paths": artifact_paths,
            "train_rows": 54810, "train_rows_sha256": train_sha,
            "tune_rows": 1998, "tune_rows_sha256": tune_sha,
            "internally_validated_overlay_rows": validated_rows,
            "image_getitem_calls": 0, "final_validation_accessed": False}


def validate_initializer(args) -> tuple[dict, dict]:
    init_path = Path(args.init).resolve()
    require(file_sha(init_path) == args.expected_init_sha256,
            "P8 initializer checkpoint SHA mismatch")
    payload = torch.load(init_path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict) and {"model", "optimizer", "step", "manifest"} <= set(payload),
            "P8 initializer payload incomplete")
    model_sha = tensor_state_sha256(payload["model"])
    embedded = payload["manifest"]
    if args.stage == "pretrain":
        require(args.expected_init_sha256 == EXPECTED_I0_FILE_SHA256
                and args.expected_init_model_state_sha256 == EXPECTED_I0_MODEL_SHA256
                and model_sha == EXPECTED_I0_MODEL_SHA256 and payload["step"] == 0
                and not payload["optimizer"].get("state", {}),
                "P8 P0 must start from the exact optimizer-free common public I0")
        require(args.init_manifest is None and args.expected_init_manifest_sha256 is None,
                "P8 public I0 does not accept a fabricated sidecar")
    else:
        require(args.init_manifest and args.expected_init_manifest_sha256,
                "P8 joint requires its own completed P0 sidecar")
        sidecar_path = Path(args.init_manifest).resolve()
        require(file_sha(sidecar_path) == args.expected_init_manifest_sha256,
                "P8 P0 sidecar SHA mismatch")
        sidecar = json.loads(sidecar_path.read_text())
        protocol = sidecar.get("experimental_protocol", {})
        config, saved_args = sidecar.get("model_config", {}), sidecar.get("arguments", {})
        require(sidecar.get("status") == "completed" and sidecar.get("step") == 2000
                and payload["step"] == 2000 and protocol.get("name") == "p8_wide_history"
                and protocol.get("stage") == "pretrain" and protocol.get("arm") == args.arm,
                "P8 joint initializer must be its own completed LAST2000 P0")
        require(config.get("history_contract") == args.arm
                and config.get("cross_cell_goal_mode") == "disabled"
                and config.get("goal_on") is False and config.get("state_on") is False
                and saved_args.get("phase") == "pretrain" and saved_args.get("seed") == args.base_seed,
                "P8 P0 architecture/seed/history lineage mismatch")
        require(sidecar.get("load_report", {}).get("common_checkpoint_sha256") == EXPECTED_I0_FILE_SHA256
                and sidecar.get("initial_model_state_sha256") == EXPECTED_I0_MODEL_SHA256,
                "P8 P0 does not trace to the common public I0")
        immutable_fields = ("git_sha", "arguments", "model_config", "loss_weights", "split_sha256",
                            "history_overlay_manifest_sha256", "time_input", "time_input_policy",
                            "initial_model_state_sha256", "load_report", "experimental_protocol")
        require(all(payload["manifest"].get(key) == sidecar.get(key) for key in immutable_fields),
                "P8 P0 embedded checkpoint and sidecar lineage mismatch")
    require(model_sha == args.expected_init_model_state_sha256,
            "P8 initializer model-state SHA mismatch")
    return payload, {"checkpoint_path": str(init_path),
                     "checkpoint_sha256": args.expected_init_sha256,
                     "model_state_sha256": model_sha, "step": int(payload["step"]),
                     "sidecar_path": str(Path(args.init_manifest).resolve()) if args.init_manifest else None,
                     "sidecar_sha256": args.expected_init_manifest_sha256,
                     "weights_only": True}


def branch_state_sha(model) -> tuple[list[str], str]:
    state = {name: value for name, value in model.state_dict().items()
             if name.startswith("scene_encoder.cross_cell_goal_residual.")}
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return sorted(state), digest.hexdigest()


def prepare_model(payload: Mapping, args) -> dict:
    import train_motiondrive_v2
    from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
    train_motiondrive_v2.seed_all(args.base_seed)
    config = train_motiondrive_v2.initialization_configuration(
        payload["manifest"], goal_on=0 if args.stage == "pretrain" else 1,
        state_on=0 if args.stage == "pretrain" else 1, explicit_arch="resnet50",
        cross_cell_goal_mode="disabled" if args.stage == "pretrain" else "zero",
        history_contract=args.arm, allow_legacy_history_override=args.stage == "pretrain")
    model = MotionDriveV2(MotionDriveV2Config(**config))
    if args.stage == "pretrain":
        model.load_state_dict(payload["model"], strict=True)
        missing, branch_sha = [], None
    else:
        incompatible = model.load_state_dict(payload["model"], strict=False)
        missing = list(incompatible.missing_keys)
        require(not incompatible.unexpected_keys and missing
                and all(key.startswith("scene_encoder.cross_cell_goal_residual.") for key in missing),
                "P8 joint P0 load must miss exactly the new branch state")
        require(all(torch.equal(model.state_dict()[key], value)
                    for key, value in payload["model"].items()),
                "P8 P0 existing tensors did not load exactly")
        branch = model.scene_encoder.cross_cell_goal_residual
        require(branch.output.bias is None and not bool(branch.output.weight.count_nonzero()),
                "P8 joint residual must start bias-free and exactly zero")
        branch_keys, branch_sha = branch_state_sha(model)
        require(branch_keys == sorted(missing),
                "P8 missing keys must equal the complete branch state")
        require(args.expected_branch_state_sha256 == branch_sha,
                "P8 deterministic branch-state SHA mismatch")
        branch_parameter_names = [name for name, _ in model.named_parameters()
                                  if name.startswith("scene_encoder.cross_cell_goal_residual.")]
        require(branch_parameter_names and all(not name.startswith("backbone_fpn.")
                                               for name in branch_parameter_names),
                "P8 branch parameters must all enter the existing non-backbone group")
    require(all(parameter.requires_grad for parameter in model.parameters()),
            "P8 all stage parameters must remain trainable")
    return {"initial_model_state_sha256": tensor_state_sha256(model.state_dict()),
            "expected_missing_state_keys": missing, "branch_state_sha256": branch_sha,
            "branch_parameter_names": branch_parameter_names if args.stage == "joint" else [],
            "branch_optimizer_group": "head" if args.stage == "joint" else None}


def build_experiment(args, source, data, init, prepared) -> dict:
    temporal = temporal_contract(args.arm)
    return {"schema_version": 1, "name": "p8_wide_history", "stage": args.stage,
            "arm": args.arm,
            "temporal_contract": {"name": temporal.name,
                                  "frame_offsets": list(temporal.frame_offsets),
                                  "nominal_seconds": list(temporal.nominal_seconds)},
            "history_overlay_manifest_sha256": args.expected_history_overlay_sha256,
            "expected_initial_checkpoint_sha256": init["checkpoint_sha256"],
            "expected_p0_model_state_sha256": init["model_state_sha256"],
            "expected_initial_model_state_sha256": prepared["initial_model_state_sha256"],
            "expected_branch_state_sha256": prepared["branch_state_sha256"],
            "expected_optimizer_groups": [{"name": "backbone", "base_lr": 1e-5},
                                          {"name": "head", "base_lr": 1e-4}],
            "expected_missing_state_keys": prepared["expected_missing_state_keys"],
            "train_data": {"rows": data["train_rows"], "rows_sha256": data["train_rows_sha256"]},
            "tune_data": {"rows": data["tune_rows"], "rows_sha256": data["tune_rows_sha256"]},
            "source": dict(source), "fresh_optimizer_step_zero": True,
            "all_model_parameters_trainable": True, "p0_branch_disabled": True,
            "joint_branch_zero": True, "final_validation_accessed": False,
            "branch": {"mode": "disabled" if args.stage == "pretrain" else "zero",
                       "sigma_m": [10., 32. / 3.], "pool_size": 4, "source_cells": 192,
                       "destination_cells": 3072, "attention_dim": 32, "cosine_scale": 8.,
                       "attention_precision": "fp32_autocast_disabled",
                       "goal_enters_distance_score_only": True,
                       "new_value_projection_adds_goal_or_position": False,
                       "output_bias": False, "output_weight_zero_initialized_at_joint": True}}


def trainer_argv(args) -> list[str]:
    steps = 2000 if args.stage == "pretrain" else 6000
    interval = 250 if args.stage == "pretrain" else 6000
    return ["--data-root", str(Path(args.data_root).resolve()),
            "--split-manifest", str(Path(args.split_manifest).resolve()),
            "--supervision-root", str(Path(args.supervision_root).resolve()),
            "--history-overlay-root", str(Path(args.history_overlay_root).resolve()),
            "--expected-history-overlay-sha256", args.expected_history_overlay_sha256,
            "--history-contract", args.arm, "--run-dir", str(Path(args.run_dir).resolve()),
            "--phase", args.stage if args.stage == "pretrain" else "joint",
            "--goal-on", "0" if args.stage == "pretrain" else "1",
            "--state-on", "0" if args.stage == "pretrain" else "1",
            "--cross-cell-goal-mode", "disabled" if args.stage == "pretrain" else "zero",
            "--gpu", "0", "--seed", str(args.base_seed), "--steps", str(steps),
            "--batch", "16", "--microbatch", "2", "--eval-batch", "4",
            "--workers", "4", "--eval-every", str(interval), "--save-every", str(interval),
            "--log-every", "10", "--lr", "0.0001", "--backbone-lr", "0.00001",
            "--weight-decay", "0.01", "--warmup", "200", "--grad-clip", "5",
            "--alpha-occ", "0.2", "--alpha-lane", "0.2", "--alpha-motion", "0.2",
            "--uncertainty", "1", "--precision", "bf16", "--time-input", "nominal",
            "--bn-policy", "fixed", "--init", str(Path(args.init).resolve()),
            "--arch", "resnet50", "--motion-input-mode", "low_feature",
            "--train-stride", "1", "--eval-stride", "5", "--max-train-samples", "0",
            "--max-eval-samples", "0", "--eval-split", "tune",
            "--cuda-memory-limit-mib", "12000", "--cuda-min-free-mib", "8192"]


def immutable_paths(args, data) -> list[str]:
    paths = [args.init, args.source_manifest, args.split_manifest,
             str(Path(args.supervision_root) / "supervision_manifest.json"),
             str(Path(args.supervision_root) / "calibration.npz"),
             str(Path(args.history_overlay_root) / "overlay_manifest.json"),
             *data["history_overlay_artifact_paths"]]
    if args.init_manifest:
        paths.append(args.init_manifest)
    return paths


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stage", choices=("pretrain", "joint"), required=True)
    parser.add_argument("--arm", choices=("control", "wide"), required=True)
    parser.add_argument("--base-seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--expected-init-sha256", required=True)
    parser.add_argument("--expected-init-model-state-sha256", required=True)
    parser.add_argument("--init-manifest")
    parser.add_argument("--expected-init-manifest-sha256")
    parser.add_argument("--expected-branch-state-sha256")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--history-overlay-root", required=True)
    parser.add_argument("--expected-history-overlay-sha256", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu", type=int, choices=(0,), default=0)
    parser.add_argument("--expected-physical-gpu-uuid", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cuda-memory-limit-mib", type=int, default=12000)
    parser.add_argument("--cuda-min-free-mib", type=int, default=8192)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    for value in (args.expected_init_sha256, args.expected_init_model_state_sha256,
                  args.expected_source_manifest_sha256, args.expected_history_overlay_sha256):
        require(isinstance(value, str) and len(value) == 64, "P8 artifact pins require full SHA256")
    if args.stage == "pretrain":
        require(args.expected_branch_state_sha256 is None, "P8 P0 has no cross-cell branch")
    else:
        require(isinstance(args.expected_branch_state_sha256, str)
                and len(args.expected_branch_state_sha256) == 64,
                "P8 joint requires the preregistered branch-state SHA")
    runtime = validate_runtime_namespace(args)
    source = validate_source_manifest(args.source_manifest, args.expected_source_manifest_sha256)
    data = validate_data(args)
    payload, init = validate_initializer(args)
    prepared = prepare_model(payload, args)
    experiment = build_experiment(args, source, data, init, prepared)
    command = trainer_argv(args)
    if args.preflight_only:
        print(json.dumps({"status": "completed_cpu_preflight_only", "stage": args.stage,
                          "arm": args.arm, "runtime": runtime, "source": source, "data": data,
                          "initializer": init, "prepared": prepared, "experiment": experiment,
                          "trainer_argv": command, "cuda_initialized": torch.cuda.is_initialized()},
                         sort_keys=True, allow_nan=False), flush=True)
        return
    immutable = immutable_paths(args, data)
    before = {str(Path(path).resolve()): file_sha(path) for path in immutable}
    import train_motiondrive_v2
    train_motiondrive_v2.run_training(command, experiment=experiment)
    after = {str(Path(path).resolve()): file_sha(path) for path in immutable}
    require(before == after, "Immutable P8 inputs changed during training")
    validate_source_manifest(args.source_manifest, args.expected_source_manifest_sha256)


if __name__ == "__main__":
    main()
