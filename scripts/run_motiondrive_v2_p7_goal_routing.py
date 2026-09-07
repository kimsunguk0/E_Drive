#!/usr/bin/env python3
"""P7 paired joint training with one score-only cross-cell goal slot.

This fail-closed driver warm-starts model weights from the matching immutable
P0 LAST2000 and delegates the unchanged losses, optimizer construction,
microbatching, and one terminal detailed tune evaluation to the existing
trainer.  It does not access final validation or select a checkpoint.
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
from motiondrive_v2_training import tensor_state_sha256

EXPECTED_SPLIT_SHA256 = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
EXPECTED_SUPERVISION_SHA256 = "ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93"
EXPECTED_CALIBRATION_SHA256 = "8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961"
EXPECTED_TRAIN_ROWS_SHA256 = "75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854"
EXPECTED_TUNE_ROWS_SHA256 = "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88"
EXPECTED_I0_FILE_SHA256 = "06d2e68e15ca00d2c0f9ed3c965e198db603fe7e80007c39319d76eba9832e5a"
EXPECTED_I0_MODEL_SHA256 = "7ac28a8f26dc796be70ddb78edff1e1c128c64f43b5982747ba419d5d5200683"
P0_TRAINING_GIT_SHA = "806f0fa8c7447a7f66f59462977d87e00b492d0c"
P0_ARTIFACTS = {
    0: {"checkpoint_sha256": "8e91c30947a040f267f01f0642db3f2725900545deeb6d4087c795b5eb398a0c",
        "sidecar_sha256": "9e5ba7ce89c0c1a9e4fe49476ff72c63b451f2658766e74e84c7336a4591fd98",
        "model_state_sha256": "f69e1c52ccd4b9908130ffe02170c0f08bff21712c634fee064517009a6c2c7d"},
    1: {"checkpoint_sha256": "6067ba9cc543e6e1be837d85883696bc7ad12c3da9b970edba3008e30eff83c5",
        "sidecar_sha256": "385ea53269e96d73c7d66f4196bc2e1205b0c6b7e9c7db537f983bc7aa37c421",
        "model_state_sha256": "d595f0788fbb5ae31ea2eaa2f3344ac300028756dcfe8f313ec1dbb761673e0d"},
}
ARM_TO_MODE = {"control_zero_slot": "zero", "goal_real_slot": "real"}
ALLOWED_PHYSICAL_GPU_UUIDS = {
    "GPU-4b804d68-fd61-af14-393a-573c533d5006",
    "GPU-1e9aea73-4e6b-2cb5-b788-f3d99e6dc6f8",
}
SOURCE_FILES = {
    "models/motiondrive_v2/__init__.py",
    "models/motiondrive_v2/config.py",
    "models/motiondrive_v2/cross_cell_goal_residual.py",
    "models/motiondrive_v2/model.py",
    "models/motiondrive_v2/motion_encoder.py",
    "models/motiondrive_v2/planner.py",
    "models/motiondrive_v2/scene_encoder.py",
    "scripts/build_grouped_split_v2.py",
    "scripts/motiondrive_v2_data.py",
    "scripts/motiondrive_v2_training.py",
    "scripts/run_motiondrive_v2_p7_goal_routing.py",
    "scripts/sparse_scoredrive.py",
    "scripts/train_motiondrive_v2.py",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: str | Path) -> str:
    return sha256(str(path))


def validate_runtime_namespace(args) -> dict:
    require(args.gpu == 0 and args.workers == 4 and args.cuda_memory_limit_mib == 12000
            and args.cuda_min_free_mib == 8192,
            "P7 fixed logical-GPU/worker/memory safety contract mismatch")
    require(args.expected_physical_gpu_uuid in ALLOWED_PHYSICAL_GPU_UUIDS,
            "P7 expected physical GPU UUID must be the authorized GPU4 or GPU5")
    if args.preflight_only:
        return {"gpu_used": False, "expected_physical_gpu_uuid": args.expected_physical_gpu_uuid,
                "logical_gpu": 0, "resource_contract_checked": True}
    namespace = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(namespace == args.expected_physical_gpu_uuid and "," not in namespace
            and torch.cuda.is_available()
            and torch.cuda.device_count() == 1,
            "P7 execution requires one UUID-isolated visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    raw_uuid = str(getattr(properties, "uuid", ""))
    actual_uuid = raw_uuid if raw_uuid.startswith("GPU-") else "GPU-" + raw_uuid
    require(actual_uuid == args.expected_physical_gpu_uuid,
            "P7 visible physical GPU UUID mismatch")
    return {"gpu_used": True, "cuda_visible_devices": namespace, "logical_gpu": 0,
            "expected_physical_gpu_uuid": args.expected_physical_gpu_uuid,
            "observed_physical_gpu_uuid_raw": raw_uuid,
            "actual_physical_gpu_uuid": actual_uuid, "resource_contract_checked": True}


def validate_source_manifest(path: str, expected_sha256: str) -> dict:
    source_path = Path(path).resolve()
    require(len(expected_sha256) == 64 and file_sha(source_path) == expected_sha256,
            "P7 source-manifest SHA mismatch")
    manifest = json.loads(source_path.read_text())
    require(set(manifest) == {"schema_version", "git_sha", "file_sha256"}
            and manifest["schema_version"] == 1,
            "P7 source manifest schema mismatch")
    git_sha = manifest["git_sha"]
    require(isinstance(git_sha, str) and len(git_sha) == 40
            and all(c in "0123456789abcdef" for c in git_sha),
            "P7 source Git must be lowercase SHA1")
    files = manifest["file_sha256"]
    require(isinstance(files, dict) and set(files) == SOURCE_FILES,
            "P7 source manifest must contain the exact runtime closure")
    require(all(isinstance(value, str) and len(value) == 64
                and all(c in "0123456789abcdef" for c in value)
                for value in files.values()), "P7 source file SHA malformed")
    actual = {name: file_sha(ROOT / name) for name in SOURCE_FILES}
    require(actual == files, "P7 runtime source differs from the pinned manifest")
    return {"path": str(source_path), "sha256": expected_sha256,
            "git_sha": git_sha, "file_sha256": files}


def validate_data_contract(data_root: str, split_manifest: str,
                           supervision_root: str, seed: int) -> dict:
    split_path = Path(split_manifest).resolve()
    supervision_path = Path(supervision_root).resolve() / "supervision_manifest.json"
    calibration_path = Path(supervision_root).resolve() / "calibration.npz"
    require(file_sha(split_path) == EXPECTED_SPLIT_SHA256, "P7 split SHA mismatch")
    require(file_sha(supervision_path) == EXPECTED_SUPERVISION_SHA256,
            "P7 C1 supervision manifest SHA mismatch")
    require(file_sha(calibration_path) == EXPECTED_CALIBRATION_SHA256,
            "P7 C1 calibration SHA mismatch")
    split = json.loads(split_path.read_text())
    validate_manifest(split)
    require(len(split["splits"]["train"]) == 203 and len(split["splits"]["tune"]) == 37,
            "P7 requires train203/tune37")
    train_sessions = {split["scene_to_session"][x] for x in split["splits"]["train"]}
    tune_sessions = {split["scene_to_session"][x] for x in split["splits"]["tune"]}
    final_sessions = {split["scene_to_session"][x] for x in split["splits"]["val"]}
    require((len(train_sessions), len(tune_sessions), len(final_sessions)) == (72, 11, 31)
            and not (train_sessions & tune_sessions or train_sessions & final_sessions
                     or tune_sessions & final_sessions), "P7 session split contract mismatch")
    from motiondrive_v2_data import MotionDriveDataset
    common = dict(data_root=data_root, split_manifest=str(split_path),
                  supervision_root=supervision_root, min_frame=30, max_samples=0, seed=seed)
    train = MotionDriveDataset(split="train", frame_stride=1, augment=True, **common)
    tune = MotionDriveDataset(split="tune", frame_stride=5, augment=False, **common)
    train_rows_sha = hashlib.sha256(np.asarray(train.rows, dtype="<i8").tobytes()).hexdigest()
    tune_rows_sha = hashlib.sha256(np.asarray(tune.rows, dtype="<i8").tobytes()).hexdigest()
    require(len(train) == 54810 and train_rows_sha == EXPECTED_TRAIN_ROWS_SHA256,
            "P7 train54810 row order mismatch")
    require(len(tune) == 1998 and tune_rows_sha == EXPECTED_TUNE_ROWS_SHA256,
            "P7 tune1998 row order mismatch")
    return {"split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "supervision_manifest_sha256": EXPECTED_SUPERVISION_SHA256,
            "calibration_sha256": EXPECTED_CALIBRATION_SHA256,
            "train_rows": 54810, "train_rows_sha256": train_rows_sha,
            "tune_rows": 1998, "tune_rows_sha256": tune_rows_sha,
            "train_scenes": 203, "tune_scenes": 37,
            "train_sessions": 72, "tune_sessions": 11,
            "final_sessions_declared_not_accessed": 31,
            "image_getitem_calls": 0, "final_validation_accessed": False}


def validate_p0(path: str, expected_sha256: str, sidecar_path: str,
                expected_sidecar_sha256: str, base_seed: int) -> tuple[dict, dict]:
    checkpoint_path, manifest_path = Path(path).resolve(), Path(sidecar_path).resolve()
    pinned = P0_ARTIFACTS[base_seed]
    require(expected_sha256 == pinned["checkpoint_sha256"]
            and expected_sidecar_sha256 == pinned["sidecar_sha256"],
            "P7 CLI P0 pins do not match the preregistered own-seed artifacts")
    require(file_sha(checkpoint_path) == expected_sha256, "P7 P0 LAST SHA mismatch")
    require(file_sha(manifest_path) == expected_sidecar_sha256, "P7 P0 sidecar SHA mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sidecar = json.loads(manifest_path.read_text())
    require(isinstance(payload, dict) and set(("model", "optimizer", "step", "manifest")) <= set(payload),
            "P7 P0 payload incomplete")
    embedded = payload["manifest"]
    args, config = embedded.get("arguments", {}), embedded.get("model_config", {})
    require(payload["step"] == 2000 and args.get("phase") == "pretrain"
            and args.get("steps") == 2000 and args.get("seed") == base_seed,
            "P7 initializer must be own-seed P0 LAST2000")
    require(embedded.get("git_sha") == sidecar.get("git_sha") == P0_TRAINING_GIT_SHA,
            "P7 P0 trainer Git mismatch")
    require(sidecar.get("status") == "completed" and sidecar.get("step") == 2000,
            "P7 P0 sidecar must be completed")
    require(config.get("backbone_arch") == "resnet50" and config.get("goal_on") is False
            and config.get("state_on") is False and config.get("motion_input_mode") == "low_feature"
            and list(config.get("plan_output_scale", ())) == [10., 5.],
            "P7 P0 model config mismatch")
    require(embedded.get("loss_weights", {}).get("plan") == 0.,
            "P7 P0 must have zero planning-loss coefficient")
    require(args.get("time_input") == "nominal" and args.get("bn_policy") == "fixed"
            and args.get("precision") == "bf16" and args.get("batch") == 16
            and args.get("microbatch") == 2, "P7 P0 execution contract mismatch")
    require(embedded.get("split_sha256") == EXPECTED_SPLIT_SHA256
            and embedded.get("supervision_manifest_sha256") == EXPECTED_SUPERVISION_SHA256,
            "P7 P0 data lineage mismatch")
    require(sidecar.get("load_report", {}).get("common_checkpoint_sha256") == EXPECTED_I0_FILE_SHA256
            and sidecar.get("initial_model_state_sha256") == EXPECTED_I0_MODEL_SHA256,
            "P7 P0 did not consume the common public I0")
    model_sha = tensor_state_sha256(payload["model"])
    require(model_sha == pinned["model_state_sha256"], "P7 P0 model-state SHA mismatch")
    return payload, {"checkpoint_path": str(checkpoint_path), "checkpoint_sha256": expected_sha256,
                     "sidecar_path": str(manifest_path), "sidecar_sha256": expected_sidecar_sha256,
                     "base_seed": base_seed, "step": 2000, "phase": "pretrain",
                     "model_state_sha256": model_sha, "training_git_sha": P0_TRAINING_GIT_SHA,
                     "common_public_i0_sha256": EXPECTED_I0_FILE_SHA256,
                     "common_public_i0_model_state_sha256": EXPECTED_I0_MODEL_SHA256,
                     "weights_only_for_p7": True}


def _build_initialized_model(payload: Mapping, base_seed: int, mode: str):
    import train_motiondrive_v2
    from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
    train_motiondrive_v2.seed_all(base_seed)
    config = train_motiondrive_v2.initialization_configuration(
        payload["manifest"], goal_on=1, state_on=1, explicit_arch="resnet50",
        cross_cell_goal_mode=mode)
    model = MotionDriveV2(MotionDriveV2Config(**config))
    incompatible = model.load_state_dict(payload["model"], strict=False)
    missing = list(incompatible.missing_keys)
    require(not incompatible.unexpected_keys and missing
            and all(key.startswith("scene_encoder.cross_cell_goal_residual.") for key in missing),
            "P7 P0 load must miss only new branch state")
    state = model.state_dict()
    require(all(torch.equal(state[key], value) for key, value in payload["model"].items()),
            "P7 P0 existing state did not load exactly")
    branch = model.scene_encoder.cross_cell_goal_residual
    require(branch.output.bias is None and not bool(branch.output.weight.count_nonzero()),
            "P7 residual output must start bias-free and exactly zero")
    require(torch.equal(branch.sigma_m.detach().cpu(),
                        torch.tensor((10., 32. / 3.), dtype=torch.float32))
            and branch.COSINE_SCALE == 8., "P7 fixed geometry/temperature mismatch")
    require(all(parameter.requires_grad for parameter in model.parameters()),
            "P7 all joint parameters must remain trainable")
    return model, missing, tensor_state_sha256(state)


def prepare_initialization(payload: Mapping, base_seed: int, mode: str) -> dict:
    model, missing, state_sha = _build_initialized_model(payload, base_seed, mode)
    other_mode = "real" if mode == "zero" else "zero"
    paired, paired_missing, paired_sha = _build_initialized_model(payload, base_seed, other_mode)
    require(missing == paired_missing and state_sha == paired_sha,
            "P7 C/G new-branch initialization or model state differs")
    backbone, other = [], []
    for name, parameter in model.named_parameters():
        is_backbone = name.startswith("backbone_fpn.") and not name.startswith(
            ("backbone_fpn.lat", "backbone_fpn.out"))
        (backbone if is_backbone else other).append((name, parameter))
    branch_names = [name for name, _ in other
                    if name.startswith("scene_encoder.cross_cell_goal_residual.")]
    require(branch_names and not any(name.startswith("scene_encoder.cross_cell_goal_residual.")
                                     for name, _ in backbone),
            "P7 branch must use the existing non-backbone optimizer group")
    optimizer = torch.optim.AdamW([
        {"params": [p for _, p in backbone], "lr": 1e-5, "base_lr": 1e-5},
        {"params": [p for _, p in other], "lr": 1e-4, "base_lr": 1e-4},
    ], weight_decay=.01)
    require(not optimizer.state, "P7 optimizer must begin at step0")
    del optimizer, paired
    return {"initial_model_state_sha256": state_sha,
            "paired_other_mode_initial_model_state_sha256": paired_sha,
            "expected_missing_state_keys": missing,
            "branch_parameter_names": branch_names,
            "branch_parameter_count": sum(p.numel() for n, p in model.named_parameters()
                                          if n in set(branch_names)),
            "branch_optimizer_group": "head", "fresh_optimizer_step": 0}


def build_experiment(arm: str, p0: Mapping, initialized: Mapping,
                     data: Mapping, source: Mapping) -> dict:
    mode = ARM_TO_MODE[arm]
    return {
        "schema_version": 1, "name": "p7_cross_cell_goal_routing", "arm": arm,
        "last_only_final_eval": True,
        "expected_initial_model_state_sha256": initialized["initial_model_state_sha256"],
        "expected_p0_model_state_sha256": p0["model_state_sha256"],
        "expected_optimizer_groups": [{"name": "backbone", "base_lr": 1e-5},
                                      {"name": "head", "base_lr": 1e-4}],
        "expected_missing_state_keys": initialized["expected_missing_state_keys"],
        "train_data": {"rows": data["train_rows"],
                       "rows_sha256": data["train_rows_sha256"]},
        "tune_data": {"rows": data["tune_rows"],
                      "rows_sha256": data["tune_rows_sha256"]},
        "source": dict(source), "fresh_optimizer_step_zero": True,
        "all_model_parameters_joint_trainable": True, "existing_goal_path_on": True,
        "planner_signature_unchanged": True,
        "branch": {"mode": mode, "sigma_m": [10., 32. / 3.],
                   "sigma_selection": "fixed_geometry_not_tuned", "pool_size": 4,
                   "source_cells": 192, "destination_cells": 3072, "attention_dim": 32,
                   "cosine_scale": 8., "attention_precision": "fp32_autocast_disabled",
                   "goal_enters_distance_score_only": True,
                   "new_value_projection_adds_goal_or_position": False, "output_bias": False,
                   "output_weight_zero_initialized": True},
    }


def trainer_argv(args, mode: str) -> list[str]:
    return [
        "--data-root", str(Path(args.data_root).resolve()),
        "--split-manifest", str(Path(args.split_manifest).resolve()),
        "--supervision-root", str(Path(args.supervision_root).resolve()),
        "--run-dir", str(Path(args.run_dir).resolve()), "--phase", "joint",
        "--goal-on", "1", "--state-on", "1", "--cross-cell-goal-mode", mode,
        "--gpu", str(args.gpu), "--seed", str(args.base_seed), "--steps", "6000",
        "--batch", "16", "--microbatch", "2", "--eval-batch", "4",
        "--workers", str(args.workers), "--eval-every", "6000", "--save-every", "6000",
        "--log-every", "10", "--lr", "0.0001", "--backbone-lr", "0.00001",
        "--weight-decay", "0.01", "--warmup", "200", "--grad-clip", "5",
        "--alpha-occ", "0.2", "--alpha-lane", "0.2", "--alpha-motion", "0.2",
        "--uncertainty", "1", "--precision", "bf16", "--time-input", "nominal",
        "--bn-policy", "fixed", "--init", str(Path(args.init).resolve()),
        "--arch", "resnet50", "--motion-input-mode", "low_feature",
        "--train-stride", "1", "--eval-stride", "5", "--max-train-samples", "0",
        "--max-eval-samples", "0", "--eval-split", "tune",
        "--cuda-memory-limit-mib", str(args.cuda_memory_limit_mib),
        "--cuda-min-free-mib", str(args.cuda_min_free_mib),
    ]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--arm", choices=tuple(ARM_TO_MODE), required=True)
    parser.add_argument("--base-seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--expected-init-sha256", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--expected-run-manifest-sha256", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu", type=int, choices=(0,), default=0,
                        help="Logical CUDA 0 inside an operator-owned physical GPU namespace")
    parser.add_argument("--expected-physical-gpu-uuid", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cuda-memory-limit-mib", type=int, default=0)
    parser.add_argument("--cuda-min-free-mib", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    for value in (args.expected_init_sha256, args.expected_run_manifest_sha256,
                  args.expected_source_manifest_sha256):
        require(len(value) == 64, "P7 expected artifact SHAs must be full SHA256")
    runtime = validate_runtime_namespace(args)
    source = validate_source_manifest(args.source_manifest, args.expected_source_manifest_sha256)
    data = validate_data_contract(args.data_root, args.split_manifest,
                                  args.supervision_root, args.base_seed)
    payload, p0 = validate_p0(args.init, args.expected_init_sha256, args.run_manifest,
                              args.expected_run_manifest_sha256, args.base_seed)
    mode = ARM_TO_MODE[args.arm]
    initialized = prepare_initialization(payload, args.base_seed, mode)
    experiment = build_experiment(args.arm, p0, initialized, data, source)
    if args.preflight_only:
        print(json.dumps({"status": "completed_cpu_preflight_only", "arm": args.arm,
                          "p0": p0, "initialized": initialized, "data": data,
                          "source": source, "runtime": runtime,
                          "trainer_argv": trainer_argv(args, mode),
                          "cuda_initialized": torch.cuda.is_initialized()},
                         sort_keys=True, allow_nan=False), flush=True)
        return
    immutable = [args.init, args.run_manifest, args.source_manifest, args.split_manifest,
                 str(Path(args.supervision_root) / "supervision_manifest.json"),
                 str(Path(args.supervision_root) / "calibration.npz")]
    before = {str(Path(path).resolve()): file_sha(path) for path in immutable}
    import train_motiondrive_v2
    train_motiondrive_v2.run_training(trainer_argv(args, mode), experiment=experiment)
    after = {str(Path(path).resolve()): file_sha(path) for path in immutable}
    require(before == after, "Immutable P7 inputs changed during training")
    validate_source_manifest(args.source_manifest, args.expected_source_manifest_sha256)


if __name__ == "__main__":
    main()
