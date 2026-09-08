#!/usr/bin/env python3
"""Run the fixed shared-status A1 continuation screen.

This opt-in wrapper leaves every P1--P8 production file unchanged.  It adds a
beta-free channel FiLM immediately before the existing shared scene refinement,
and supplies either zero or nominal-time causal provided status through one
frozen train/tune overlay.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_grouped_split_v2 import sha256, validate_manifest
from models.motiondrive_v2.shared_status import install_shared_status, shared_status_state
from motiondrive_v2_shared_status_data import (
    SharedStatusDataset, load_status_overlay, shared_status_model_inputs,
)
from motiondrive_v2_training import tensor_state_sha256


NAME = "shared_status_a1"
ARMS = ("zero", "provided_causal_5d")
EXPECTED_SPLIT_SHA256 = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
EXPECTED_SUPERVISION_SHA256 = "ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93"
EXPECTED_CALIBRATION_SHA256 = "8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961"
EXPECTED_PARENT_FILE_SHA256 = "6145255ab2b1efd3deb169b873896e9b1b623ca49f11e5a46a71b138e3b0355e"
EXPECTED_PARENT_MODEL_SHA256 = "e79d545bbb09b4109c783a8cf4c861772bc5e52c62004ef5b37dc4ca63e938a2"
EXPECTED_PARENT_MANIFEST_SHA256 = "cf287d05ce6b2f4476c07ac5c2253cd9ecb3227bcc01404adbbba8463bb66de0"
EXPECTED_ROWS = {
    "train": (54810, "75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854"),
    "tune": (1998, "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88"),
}
GPU_ASSIGNMENTS = {
    "zero": "GPU-5d2254f9-41a7-62dd-2b38-de82459acb24",
    "provided_causal_5d": "GPU-041334c0-089c-6ff5-b0b5-59ff445fa015",
}
SOURCE_FILES = {
    "models/motiondrive_v2_temporal_contract.py",
    "models/motiondrive_v2/__init__.py", "models/motiondrive_v2/config.py",
    "models/motiondrive_v2/cross_cell_goal_residual.py", "models/motiondrive_v2/model.py",
    "models/motiondrive_v2/motion_encoder.py", "models/motiondrive_v2/planner.py",
    "models/motiondrive_v2/scene_encoder.py", "models/motiondrive_v2/shared_status.py",
    "scripts/build_grouped_split_v2.py", "scripts/motiondrive_v2_data.py",
    "scripts/motiondrive_v2_training.py", "scripts/train_motiondrive_v2.py",
    "scripts/evaluate_motiondrive_v2_planning.py", "scripts/audit_motiondrive_v2.py",
    "scripts/export_motiondrive_v2_inference.py", "models/motiondrive_v2_input_contract.py",
    "scripts/sparse_scoredrive.py",
    "scripts/motiondrive_v2_shared_status_data.py",
    "scripts/build_motiondrive_v2_shared_status_overlay.py",
    "scripts/run_motiondrive_v2_shared_status_a1.py",
    "tests/test_motiondrive_v2_shared_status_a1.py",
    "tests/test_motiondrive_v2_shared_status_a1_review.py",
    "reports/motiondrive_v2_shared_status_a1_protocol_20260908.md",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows_sha(rows) -> str:
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def validate_source_manifest(path, expected_sha256):
    path = Path(path).resolve()
    require(len(expected_sha256) == 64 and sha256(path) == expected_sha256,
            "A1 source-manifest SHA mismatch")
    manifest = json.loads(path.read_text())
    require(set(manifest) == {"schema_version", "git_sha", "file_sha256"}
            and manifest["schema_version"] == 1
            and isinstance(manifest["git_sha"], str) and len(manifest["git_sha"]) == 40
            and isinstance(manifest["file_sha256"], dict)
            and set(manifest["file_sha256"]) == SOURCE_FILES,
            "A1 source manifest must be the exact runtime closure")
    actual = {name: sha256(ROOT / name) for name in sorted(SOURCE_FILES)}
    require(actual == manifest["file_sha256"], "A1 runtime source differs from manifest")
    return {"path": str(path), "sha256": expected_sha256,
            "git_sha": manifest["git_sha"], "file_sha256": actual}


def validate_runtime(args):
    require(args.seed == 0 and args.gpu == 0 and args.workers == 4
            and args.cuda_memory_limit_mib == 12000 and args.cuda_min_free_mib == 8192,
            "A1 seed/logical-GPU/worker/memory contract mismatch")
    expected = GPU_ASSIGNMENTS[args.arm]
    require(args.expected_physical_gpu_uuid == expected,
            "A1 physical GPU assignment mismatch")
    if args.preflight_only:
        return {"gpu_used": False, "expected_physical_gpu_uuid": expected,
                "cuda_initialized": torch.cuda.is_initialized()}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(visible == expected and "," not in visible and torch.cuda.is_available()
            and torch.cuda.device_count() == 1, "A1 requires one UUID-isolated CUDA device")
    raw = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    actual = raw if raw.startswith("GPU-") else "GPU-" + raw
    require(actual == expected, "A1 observed physical GPU UUID mismatch")
    return {"gpu_used": True, "logical_gpu": 0, "cuda_visible_devices": visible,
            "observed_physical_gpu_uuid_raw": raw, "actual_physical_gpu_uuid": actual}


def validate_parent(args):
    checkpoint_path = Path(args.init).resolve()
    sidecar_path = Path(args.init_manifest).resolve()
    require(sha256(checkpoint_path) == EXPECTED_PARENT_FILE_SHA256
            and sha256(sidecar_path) == EXPECTED_PARENT_MANIFEST_SHA256,
            "A1 requires the exact P7-C b0 LAST6000 and sidecar")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sidecar = json.loads(sidecar_path.read_text())
    require(isinstance(payload, dict) and {"model", "step", "manifest"} <= set(payload)
            and payload["step"] == 6000 and sidecar.get("step") == 6000
            and sidecar.get("status") == "completed",
            "A1 parent is not a completed P7-C LAST6000")
    require(tensor_state_sha256(payload["model"]) == EXPECTED_PARENT_MODEL_SHA256,
            "A1 parent model-state SHA mismatch")
    config = payload["manifest"].get("model_config", {})
    protocol = payload["manifest"].get("experimental_protocol", {})
    require(config.get("goal_on") is True and config.get("state_on") is True
            and config.get("cross_cell_goal_mode") == "zero"
            and config.get("motion_input_mode") == "low_feature"
            and config.get("history_contract", "control") == "control"
            and protocol.get("name") == "p7_cross_cell_goal_routing"
            and protocol.get("arm") == "control_zero_slot",
            "A1 parent architecture/protocol mismatch")
    require(payload["manifest"].get("split_sha256") == EXPECTED_SPLIT_SHA256,
            "A1 parent split lineage mismatch")
    return payload, {"checkpoint": str(checkpoint_path),
                     "checkpoint_sha256": EXPECTED_PARENT_FILE_SHA256,
                     "model_state_sha256": EXPECTED_PARENT_MODEL_SHA256,
                     "sidecar": str(sidecar_path),
                     "sidecar_sha256": EXPECTED_PARENT_MANIFEST_SHA256,
                     "step": 6000, "weights_only": True,
                     "fresh_optimizer": True,
                     "producer_git_sha": payload["manifest"].get("git_sha")}


def validate_data(args):
    split_path = Path(args.split_manifest).resolve()
    supervision = Path(args.supervision_root).resolve()
    require(sha256(split_path) == EXPECTED_SPLIT_SHA256,
            "A1 grouped split SHA mismatch")
    require(sha256(supervision / "supervision_manifest.json") == EXPECTED_SUPERVISION_SHA256
            and sha256(supervision / "calibration.npz") == EXPECTED_CALIBRATION_SHA256,
            "A1 immutable C1 supervision/calibration mismatch")
    split = json.loads(split_path.read_text())
    validate_manifest(split)
    require(len(split["splits"]["train"]) == 203 and len(split["splits"]["tune"]) == 37,
            "A1 requires current train203/tune37")
    overlay_manifest, overlay = load_status_overlay(
        args.status_overlay_root, args.expected_status_overlay_sha256)
    from motiondrive_v2_data import MotionDriveDataset
    common = dict(data_root=args.data_root, split_manifest=str(split_path),
                  supervision_root=str(supervision), min_frame=30, max_samples=0,
                  seed=0, history_contract="control")
    train = MotionDriveDataset(split="train", frame_stride=1, augment=True, **common)
    tune = MotionDriveDataset(split="tune", frame_stride=5, augment=False, **common)
    for name, dataset in (("train", train), ("tune", tune)):
        expected_n, expected_sha = EXPECTED_ROWS[name]
        require((len(dataset), rows_sha(dataset.rows)) == (expected_n, expected_sha),
                f"A1 {name} base rows mismatch")
        SharedStatusDataset(dataset, name, overlay, args.arm)
    all_status = np.concatenate([overlay[name]["status5"] for name in ("train", "tune")])
    require(np.isfinite(all_status).all(), "A1 status contains invalid rows")
    return {"split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "supervision_manifest_sha256": EXPECTED_SUPERVISION_SHA256,
            "calibration_sha256": EXPECTED_CALIBRATION_SHA256,
            "status_overlay_manifest_sha256": args.expected_status_overlay_sha256,
            "status_overlay_manifest": overlay_manifest,
            "train_rows": 54810, "train_rows_sha256": EXPECTED_ROWS["train"][1],
            "tune_rows": 1998, "tune_rows_sha256": EXPECTED_ROWS["tune"][1],
            "status_valid_rows": int(len(all_status)), "status_invalid_rows": 0,
            "status_valid_fraction": 1.0, "final_validation_accessed": False}, overlay


def fusion_state_sha(model):
    return tensor_state_sha256(shared_status_state(model))


def prepare_model(payload, seed=0):
    import train_motiondrive_v2 as trainer
    from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
    trainer.seed_all(seed)
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    model = install_shared_status(MotionDriveV2(config))
    incompatible = model.load_state_dict(payload["model"], strict=False)
    missing = list(incompatible.missing_keys)
    expected = [name for name in model.state_dict() if name.startswith("shared_status_fusion.")]
    require(not incompatible.unexpected_keys and sorted(missing) == sorted(expected),
            "A1 parent load must miss exactly the new FiLM state")
    require(all(torch.equal(model.state_dict()[name], value)
                for name, value in payload["model"].items()),
            "A1 parent tensors did not load exactly")
    final = model.shared_status_fusion.status_mlp[-1]
    require(not bool(final.weight.count_nonzero()) and not bool(final.bias.count_nonzero()),
            "A1 FiLM must begin at exact identity")
    return model, {"initial_model_state_sha256": tensor_state_sha256(model.state_dict()),
                   "fusion_state_sha256": fusion_state_sha(model),
                   "missing_keys": missing,
                   "fusion_parameter_names": [name for name, _ in model.named_parameters()
                                                if name.startswith("shared_status_fusion.")],
                   "initial_identity": True}


def build_experiment(args, source, data, parent, prepared):
    return {"schema_version": 1, "name": NAME, "arm": args.arm, "seed": 0,
            "parent": parent, "source": source,
            "status_overlay_manifest_sha256": args.expected_status_overlay_sha256,
            "train_data": {"rows": data["train_rows"], "rows_sha256": data["train_rows_sha256"]},
            "tune_data": {"rows": data["tune_rows"], "rows_sha256": data["tune_rows_sha256"]},
            "expected_initial_model_state_sha256": prepared["initial_model_state_sha256"],
            "expected_fusion_state_sha256": prepared["fusion_state_sha256"],
            "expected_missing_state_keys": prepared["missing_keys"],
            "expected_optimizer_groups": [{"name": "backbone", "base_lr": 5e-6},
                                          {"name": "head", "base_lr": 5e-5}],
            "fusion": {"location": "after image/global/cross-cell scene; before SpatialMix",
                       "formula": "F*(1+tanh(gamma))", "beta_or_additive_value": False,
                       "status_fields": ["vx", "vy", "ax", "ay", "yaw_rate"],
                       "status_scale": [10., 5., 3., 3., .5], "mlp": [5, 32, 128],
                       "precision": "fp32_autocast_disabled", "final_weight_and_bias_zero": True},
            "fresh_optimizer_step_zero": True, "all_original_parameters_trainable": True,
            "fixed_bn_running_statistics": True,
            "smoke_only_never_training_initializer": bool(args.smoke_only),
            "terminal_tune_evaluations": 0 if args.smoke_only else 1,
            "final_validation_accessed": False}


def _add_status(inputs, batch):
    status = batch.get("provided_status5")
    require(isinstance(status, torch.Tensor) and status.ndim == 2 and status.shape[1] == 5,
            "provided_status5 [B,5] is required")
    return {**inputs, "provided_status5": status}


@contextlib.contextmanager
def patched_training_runtime(overlay, arm, expected_parent_model_sha,
                             expected_fusion_sha, expected_missing):
    """Narrow instance/data/input patches; always restore module globals."""
    import models.motiondrive_v2 as model_api
    import motiondrive_v2_data as data_api
    import train_motiondrive_v2 as trainer
    import evaluate_motiondrive_v2_planning as planning_eval
    originals = {
        "model": model_api.MotionDriveV2, "dataset": data_api.MotionDriveDataset,
        "train_inputs": trainer.model_inputs, "eval_inputs": planning_eval.planning_model_inputs,
        "protocol": trainer._validate_experimental_protocol,
        "runtime": trainer._validate_experimental_runtime,
        "loader": trainer._load_initial_model_state,
        "schedule": trainer._training_schedule_actions,
    }

    def model_factory(config=None):
        return install_shared_status(originals["model"](config))

    def dataset_factory(**kwargs):
        split = kwargs.get("split")
        require(split in ("train", "tune"), "A1 permits only train/tune datasets")
        return SharedStatusDataset(originals["dataset"](**kwargs), split, overlay, arm)

    def train_inputs(batch, time_input="raw", nominal_history_seconds=(.1, .2, .5, 1.)):
        return shared_status_model_inputs(originals["train_inputs"], batch,
                                          time_input=time_input,
                                          nominal_history_seconds=nominal_history_seconds)

    def eval_inputs(batch, time_input="raw"):
        return _add_status(originals["eval_inputs"](batch, time_input), batch)

    def validate_protocol(experiment):
        require(isinstance(experiment, dict) and experiment.get("name") == NAME,
                "A1 experimental protocol missing")
        require(experiment["arm"] == arm and experiment["seed"] == 0
                and experiment["expected_fusion_state_sha256"] == expected_fusion_sha,
                "A1 experimental protocol mismatch")
        return None

    def validate_runtime(args, experiment):
        smoke = experiment["smoke_only_never_training_initializer"]
        expected = {"phase": "joint", "goal_on": 1, "state_on": 1,
                    "steps": 2 if smoke else 2000,
                    "batch": 16, "microbatch": 2, "eval_batch": 4,
                    "eval_every": 2000, "save_every": 500,
                    "lr": 5e-5, "backbone_lr": 5e-6, "weight_decay": .01,
                    "warmup": 100, "grad_clip": 5., "alpha_occ": .2,
                    "alpha_lane": .2, "alpha_motion": .2, "uncertainty": 1,
                    "precision": "bf16", "time_input": "nominal", "bn_policy": "fixed",
                    "arch": "resnet50", "motion_input_mode": "low_feature",
                    "cross_cell_goal_mode": "zero", "history_contract": "control",
                    "train_stride": 1, "eval_stride": 5,
                    "max_train_samples": 0, "max_eval_samples": 0, "eval_split": "tune"}
        require(all(getattr(args, key) == value for key, value in expected.items())
                and args.init and not args.resume and not args.pretrained and not args.eval_only
                and not args.train_scenes and not args.eval_scenes,
                "A1 fixed continuation recipe mismatch")

    def load_initial(model, common, experiment=None):
        require(tensor_state_sha256(common["model"]) == expected_parent_model_sha,
                "A1 trainer parent model-state mismatch")
        incompatible = model.load_state_dict(common["model"], strict=False)
        require(list(incompatible.missing_keys) == expected_missing
                and not incompatible.unexpected_keys,
                "A1 trainer parent load missing-key mismatch")
        require(fusion_state_sha(model) == expected_fusion_sha,
                "A1 trainer FiLM initialization mismatch")
        final = model.shared_status_fusion.status_mlp[-1]
        require(not bool(final.weight.count_nonzero()) and not bool(final.bias.count_nonzero()),
                "A1 trainer FiLM no longer begins at identity")
        return {"shared_status_parent_load": {"strict_existing_keys": True,
                "missing_keys": expected_missing, "unexpected_keys": [],
                "weights_only": True, "parent_model_state_sha256": expected_parent_model_sha,
                "fusion_state_sha256": expected_fusion_sha}}

    def schedule(step, args, experiment):
        if experiment["smoke_only_never_training_initializer"]:
            return False, False
        return step == 2000, step in (500, 1000, 2000)

    model_api.MotionDriveV2 = model_factory
    data_api.MotionDriveDataset = dataset_factory
    trainer.model_inputs = train_inputs
    planning_eval.planning_model_inputs = eval_inputs
    trainer._validate_experimental_protocol = validate_protocol
    trainer._validate_experimental_runtime = validate_runtime
    trainer._load_initial_model_state = load_initial
    trainer._training_schedule_actions = schedule
    try:
        yield {"train_model_inputs": train_inputs, "terminal_model_inputs": eval_inputs}
    finally:
        model_api.MotionDriveV2 = originals["model"]
        data_api.MotionDriveDataset = originals["dataset"]
        trainer.model_inputs = originals["train_inputs"]
        planning_eval.planning_model_inputs = originals["eval_inputs"]
        trainer._validate_experimental_protocol = originals["protocol"]
        trainer._validate_experimental_runtime = originals["runtime"]
        trainer._load_initial_model_state = originals["loader"]
        trainer._training_schedule_actions = originals["schedule"]


def trainer_argv(args):
    steps = 2 if args.smoke_only else 2000
    return ["--data-root", str(Path(args.data_root).resolve()),
            "--split-manifest", str(Path(args.split_manifest).resolve()),
            "--supervision-root", str(Path(args.supervision_root).resolve()),
            "--run-dir", str(Path(args.run_dir).resolve()),
            "--phase", "joint", "--goal-on", "1", "--state-on", "1",
            "--cross-cell-goal-mode", "zero", "--history-contract", "control",
            "--gpu", "0", "--seed", "0", "--steps", str(steps),
            "--batch", "16", "--microbatch", "2", "--eval-batch", "4",
            "--workers", "4", "--eval-every", "2000", "--save-every", "500",
            "--log-every", "1" if args.smoke_only else "10",
            "--lr", "0.00005", "--backbone-lr", "0.000005",
            "--weight-decay", "0.01", "--warmup", "100", "--grad-clip", "5",
            "--alpha-occ", "0.2", "--alpha-lane", "0.2", "--alpha-motion", "0.2",
            "--uncertainty", "1", "--precision", "bf16", "--time-input", "nominal",
            "--bn-policy", "fixed", "--init", str(Path(args.init).resolve()),
            "--arch", "resnet50", "--motion-input-mode", "low_feature",
            "--train-stride", "1", "--eval-stride", "5",
            "--max-train-samples", "0", "--max-eval-samples", "0",
            "--eval-split", "tune", "--cuda-memory-limit-mib", "12000",
            "--cuda-min-free-mib", "8192"]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--init-manifest", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--status-overlay-root", required=True)
    parser.add_argument("--expected-status-overlay-sha256", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    parser.add_argument("--gpu", type=int, choices=(0,), default=0)
    parser.add_argument("--expected-physical-gpu-uuid", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cuda-memory-limit-mib", type=int, default=12000)
    parser.add_argument("--cuda-min-free-mib", type=int, default=8192)
    parser.add_argument("--expected-initial-state-sha256")
    parser.add_argument("--expected-fusion-state-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Exactly two optimizer updates, no evaluation; artifact is never an initializer")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    require(not (args.preflight_only and args.smoke_only),
            "A1 CPU preflight and two-update GPU smoke are separate modes")
    if args.smoke_only:
        require("smoke_only" in Path(args.run_dir).name,
                "A1 smoke output directory must be unmistakably named smoke_only")
    runtime = validate_runtime(args)
    source = validate_source_manifest(args.source_manifest, args.expected_source_manifest_sha256)
    payload, parent = validate_parent(args)
    data, overlay = validate_data(args)
    _model, prepared = prepare_model(payload, args.seed)
    experiment = build_experiment(args, source, data, parent, prepared)
    command = trainer_argv(args)
    if args.preflight_only:
        print(json.dumps({"status": "completed_cpu_preflight_only", "arm": args.arm,
                          "runtime": runtime, "source": source, "data": data,
                          "parent": parent, "prepared": prepared,
                          "experiment": experiment, "trainer_argv": command,
                          "cuda_initialized": torch.cuda.is_initialized()},
                         sort_keys=True, allow_nan=False), flush=True)
        return
    require(args.expected_initial_state_sha256 == prepared["initial_model_state_sha256"]
            and args.expected_fusion_state_sha256 == prepared["fusion_state_sha256"],
            "A1 launch requires the reviewed assembled/fusion state pins")
    immutable = [args.init, args.init_manifest, args.source_manifest, args.split_manifest,
                 str(Path(args.supervision_root) / "supervision_manifest.json"),
                 str(Path(args.supervision_root) / "calibration.npz"),
                 str(Path(args.status_overlay_root) / "overlay_manifest.json"),
                 str(Path(args.status_overlay_root) / "train.npz"),
                 str(Path(args.status_overlay_root) / "tune.npz")]
    before = {str(Path(path).resolve()): sha256(path) for path in immutable}
    import train_motiondrive_v2 as trainer
    with patched_training_runtime(overlay, args.arm, EXPECTED_PARENT_MODEL_SHA256,
                                  prepared["fusion_state_sha256"], prepared["missing_keys"]):
        trainer.run_training(command, experiment=experiment)
    after = {str(Path(path).resolve()): sha256(path) for path in immutable}
    require(before == after, "A1 immutable inputs changed during training")
    validate_source_manifest(args.source_manifest, args.expected_source_manifest_sha256)


if __name__ == "__main__":
    main()
