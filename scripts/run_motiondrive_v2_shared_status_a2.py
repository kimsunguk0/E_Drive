#!/usr/bin/env python3
"""Run the opt-in shared-status A2 pre-attention query screen.

This wrapper reuses the frozen A1 data and trainer adapters.  Its only model
change is a zero-initialized status delta on the 32D scene query context before
image attention.  It never forwards raw status to the planner or image values.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_motiondrive_v2_shared_status_a1 as a1
from build_grouped_split_v2 import sha256
from models.motiondrive_v2.shared_status_query import (
    install_shared_status_query,
    shared_status_query_state,
)
from motiondrive_v2_training import tensor_state_sha256


NAME = "shared_status_a2_query"
ARMS = a1.ARMS
PARENTS = {
    0: {
        "checkpoint_sha256": "6145255ab2b1efd3deb169b873896e9b1b623ca49f11e5a46a71b138e3b0355e",
        "model_state_sha256": "e79d545bbb09b4109c783a8cf4c861772bc5e52c62004ef5b37dc4ca63e938a2",
        "sidecar_sha256": "cf287d05ce6b2f4476c07ac5c2253cd9ecb3227bcc01404adbbba8463bb66de0",
    },
    1: {
        "checkpoint_sha256": "8ffb429b17820b63477987f5c7de151dfa9e0e2b12378e247aed0b0e41de2563",
        "model_state_sha256": "bfcd5e8f807013452d71bf00e824926d89f2a0582809336d2a144aa17e571b29",
        "sidecar_sha256": "11854d7a9f828b8b4b6710679fd420f5c8d243d1322231d723137419aa48e80f",
    },
}
GPU_ASSIGNMENTS = a1.GPU_ASSIGNMENTS
SOURCE_FILES = set(a1.SOURCE_FILES) | {
    "models/motiondrive_v2/shared_status_query.py",
    "scripts/run_motiondrive_v2_shared_status_a2.py",
    "tests/test_motiondrive_v2_shared_status_query.py",
    "tests/test_motiondrive_v2_shared_status_a2.py",
    "reports/motiondrive_v2_shared_status_a2_protocol_20260908.md",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_source_manifest(path, expected_sha256):
    path = Path(path).resolve()
    require(len(expected_sha256) == 64 and sha256(path) == expected_sha256,
            "A2 source-manifest SHA mismatch")
    manifest = json.loads(path.read_text())
    require(set(manifest) == {"schema_version", "git_sha", "file_sha256"}
            and manifest["schema_version"] == 1
            and isinstance(manifest["git_sha"], str) and len(manifest["git_sha"]) == 40
            and isinstance(manifest["file_sha256"], dict)
            and set(manifest["file_sha256"]) == SOURCE_FILES,
            "A2 source manifest must be the exact inherited runtime closure")
    actual = {name: sha256(ROOT / name) for name in sorted(SOURCE_FILES)}
    require(actual == manifest["file_sha256"], "A2 runtime source differs from manifest")
    return {"path": str(path), "sha256": expected_sha256,
            "git_sha": manifest["git_sha"], "file_sha256": actual}


def validate_runtime(args):
    require(args.seed in PARENTS and args.gpu == 0 and args.workers == 4
            and args.cuda_memory_limit_mib == 12000 and args.cuda_min_free_mib == 8192,
            "A2 seed/logical-GPU/worker/memory contract mismatch")
    expected = GPU_ASSIGNMENTS[args.arm]
    require(args.expected_physical_gpu_uuid == expected,
            "A2 physical GPU assignment mismatch")
    if args.preflight_only:
        return {"gpu_used": False, "expected_physical_gpu_uuid": expected,
                "cuda_initialized": torch.cuda.is_initialized()}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(visible == expected and "," not in visible and torch.cuda.is_available()
            and torch.cuda.device_count() == 1, "A2 requires one UUID-isolated CUDA device")
    raw = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    actual = raw if raw.startswith("GPU-") else "GPU-" + raw
    require(actual == expected, "A2 observed physical GPU UUID mismatch")
    return {"gpu_used": True, "logical_gpu": 0, "cuda_visible_devices": visible,
            "observed_physical_gpu_uuid_raw": raw, "actual_physical_gpu_uuid": actual}


def validate_parent(args):
    expected = PARENTS[args.seed]
    checkpoint_path = Path(args.init).resolve()
    sidecar_path = Path(args.init_manifest).resolve()
    require(sha256(checkpoint_path) == expected["checkpoint_sha256"]
            and sha256(sidecar_path) == expected["sidecar_sha256"],
            "A2 requires the exact seed-matched P7-C LAST6000 and sidecar")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sidecar = json.loads(sidecar_path.read_text())
    require(isinstance(payload, dict) and {"model", "step", "manifest"} <= set(payload)
            and payload["step"] == 6000 and sidecar.get("step") == 6000
            and sidecar.get("status") == "completed",
            "A2 parent is not a completed P7-C LAST6000")
    require(tensor_state_sha256(payload["model"]) == expected["model_state_sha256"],
            "A2 parent model-state SHA mismatch")
    config = payload["manifest"].get("model_config", {})
    protocol = payload["manifest"].get("experimental_protocol", {})
    require(config.get("goal_on") is True and config.get("state_on") is True
            and config.get("cross_cell_goal_mode") == "zero"
            and config.get("motion_input_mode") == "low_feature"
            and config.get("history_contract", "control") == "control"
            and protocol.get("name") == "p7_cross_cell_goal_routing"
            and protocol.get("arm") == "control_zero_slot"
            and payload["manifest"].get("split_sha256") == a1.EXPECTED_SPLIT_SHA256,
            "A2 parent architecture/protocol/split mismatch")
    return payload, {"checkpoint": str(checkpoint_path),
                     "checkpoint_sha256": expected["checkpoint_sha256"],
                     "model_state_sha256": expected["model_state_sha256"],
                     "sidecar": str(sidecar_path),
                     "sidecar_sha256": expected["sidecar_sha256"],
                     "step": 6000, "base_seed": args.seed,
                     "weights_only": True, "fresh_optimizer": True,
                     "producer_git_sha": payload["manifest"].get("git_sha")}


def query_state_sha(model):
    return tensor_state_sha256(shared_status_query_state(model))


def prepare_model(payload, seed):
    import train_motiondrive_v2 as trainer
    from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
    trainer.seed_all(seed)
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    model = install_shared_status_query(MotionDriveV2(config))
    incompatible = model.load_state_dict(payload["model"], strict=False)
    missing = list(incompatible.missing_keys)
    expected = [name for name in model.state_dict()
                if name.startswith("shared_status_query_fusion.")]
    require(not incompatible.unexpected_keys and sorted(missing) == sorted(expected),
            "A2 parent load must miss exactly the new query state")
    require(all(torch.equal(model.state_dict()[name], value)
                for name, value in payload["model"].items()),
            "A2 parent tensors did not load exactly")
    final = model.shared_status_query_fusion.status_mlp[-1]
    require(not bool(final.weight.count_nonzero()) and not bool(final.bias.count_nonzero()),
            "A2 query delta must begin at exact identity")
    return model, {"initial_model_state_sha256": tensor_state_sha256(model.state_dict()),
                   "query_state_sha256": query_state_sha(model),
                   "missing_keys": missing,
                   "query_parameter_names": [name for name, _ in model.named_parameters()
                                               if name.startswith("shared_status_query_fusion.")],
                   "initial_identity": True}


def build_experiment(args, source, data, parent, prepared):
    return {"schema_version": 1, "name": NAME, "arm": args.arm, "seed": args.seed,
            "parent": parent, "source": source,
            "status_overlay_manifest_sha256": args.expected_status_overlay_sha256,
            "train_data": {"rows": data["train_rows"],
                           "rows_sha256": data["train_rows_sha256"]},
            "tune_data": {"rows": data["tune_rows"],
                          "rows_sha256": data["tune_rows_sha256"]},
            "expected_initial_model_state_sha256": prepared["initial_model_state_sha256"],
            "expected_query_state_sha256": prepared["query_state_sha256"],
            "expected_missing_state_keys": prepared["missing_keys"],
            "expected_optimizer_groups": [{"name": "backbone", "base_lr": 5e-6},
                                          {"name": "head", "base_lr": 5e-5}],
            "conditioning": {"location": "scene query_context before image attention",
                "formula": "query_context + delta(status5 / fixed_scales)",
                "changes_attention_values": False, "direct_planner_status_argument": False,
                "status_fields": ["vx", "vy", "ax", "ay", "yaw_rate"],
                "status_scale": [10., 5., 3., 3., .5], "mlp": [5, 32, 32],
                "precision": "fp32_autocast_disabled", "final_weight_and_bias_zero": True},
            "fresh_optimizer_step_zero": True, "all_original_parameters_trainable": True,
            "fixed_bn_running_statistics": True,
            "smoke_only_never_training_initializer": bool(args.smoke_only),
            "terminal_tune_evaluations": 0 if args.smoke_only else 1,
            "final_validation_accessed": False}


@contextlib.contextmanager
def patched_training_runtime(overlay, arm, seed, expected_parent_model_sha,
                             expected_query_sha, expected_missing):
    import models.motiondrive_v2 as model_api
    import motiondrive_v2_data as data_api
    import train_motiondrive_v2 as trainer
    import evaluate_motiondrive_v2_planning as planning_eval
    originals = {"model": model_api.MotionDriveV2,
                 "dataset": data_api.MotionDriveDataset,
                 "train_inputs": trainer.model_inputs,
                 "eval_inputs": planning_eval.planning_model_inputs,
                 "protocol": trainer._validate_experimental_protocol,
                 "runtime": trainer._validate_experimental_runtime,
                 "loader": trainer._load_initial_model_state,
                 "schedule": trainer._training_schedule_actions}

    def model_factory(config=None):
        return install_shared_status_query(originals["model"](config))

    def dataset_factory(**kwargs):
        split = kwargs.get("split")
        require(split in ("train", "tune"), "A2 permits only train/tune datasets")
        return a1.SharedStatusDataset(originals["dataset"](**kwargs), split, overlay, arm)

    def train_inputs(batch, time_input="raw", nominal_history_seconds=(.1, .2, .5, 1.)):
        return a1.shared_status_model_inputs(originals["train_inputs"], batch,
                                             time_input=time_input,
                                             nominal_history_seconds=nominal_history_seconds)

    def eval_inputs(batch, time_input="raw"):
        return a1._add_status(originals["eval_inputs"](batch, time_input), batch)

    def validate_protocol(experiment):
        require(isinstance(experiment, dict) and experiment.get("name") == NAME
                and experiment.get("arm") == arm and experiment.get("seed") == seed
                and experiment.get("expected_query_state_sha256") == expected_query_sha,
                "A2 experimental protocol mismatch")

    def validate_experiment_runtime(runtime_args, experiment):
        smoke = experiment["smoke_only_never_training_initializer"]
        expected = {"phase": "joint", "goal_on": 1, "state_on": 1,
                    "steps": 2 if smoke else 2000, "batch": 16, "microbatch": 2,
                    "eval_batch": 4, "eval_every": 2000, "save_every": 500,
                    "lr": 5e-5, "backbone_lr": 5e-6, "weight_decay": .01,
                    "warmup": 100, "grad_clip": 5., "alpha_occ": .2,
                    "alpha_lane": .2, "alpha_motion": .2, "uncertainty": 1,
                    "precision": "bf16", "time_input": "nominal", "bn_policy": "fixed",
                    "arch": "resnet50", "motion_input_mode": "low_feature",
                    "cross_cell_goal_mode": "zero", "history_contract": "control",
                    "train_stride": 1, "eval_stride": 5, "max_train_samples": 0,
                    "max_eval_samples": 0, "eval_split": "tune"}
        require(all(getattr(runtime_args, key) == value for key, value in expected.items())
                and runtime_args.seed == seed and runtime_args.init
                and not runtime_args.resume and not runtime_args.pretrained
                and not runtime_args.eval_only and not runtime_args.train_scenes
                and not runtime_args.eval_scenes,
                "A2 fixed continuation recipe mismatch")

    def load_initial(model, common, experiment=None):
        require(tensor_state_sha256(common["model"]) == expected_parent_model_sha,
                "A2 trainer parent model-state mismatch")
        incompatible = model.load_state_dict(common["model"], strict=False)
        require(list(incompatible.missing_keys) == expected_missing
                and not incompatible.unexpected_keys,
                "A2 trainer parent load missing-key mismatch")
        require(query_state_sha(model) == expected_query_sha,
                "A2 trainer query initialization mismatch")
        final = model.shared_status_query_fusion.status_mlp[-1]
        require(not bool(final.weight.count_nonzero()) and not bool(final.bias.count_nonzero()),
                "A2 trainer query delta no longer begins at identity")
        return {"shared_status_query_parent_load": {"strict_existing_keys": True,
                "missing_keys": expected_missing, "unexpected_keys": [], "weights_only": True,
                "parent_model_state_sha256": expected_parent_model_sha,
                "query_state_sha256": expected_query_sha}}

    def schedule(step, runtime_args, experiment):
        if experiment["smoke_only_never_training_initializer"]:
            return False, False
        return step == 2000, step in (500, 1000, 2000)

    model_api.MotionDriveV2 = model_factory
    data_api.MotionDriveDataset = dataset_factory
    trainer.model_inputs = train_inputs
    planning_eval.planning_model_inputs = eval_inputs
    trainer._validate_experimental_protocol = validate_protocol
    trainer._validate_experimental_runtime = validate_experiment_runtime
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
    command = a1.trainer_argv(args)
    seed_index = command.index("--seed") + 1
    command[seed_index] = str(args.seed)
    return command


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
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--gpu", type=int, choices=(0,), default=0)
    parser.add_argument("--expected-physical-gpu-uuid", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cuda-memory-limit-mib", type=int, default=12000)
    parser.add_argument("--cuda-min-free-mib", type=int, default=8192)
    parser.add_argument("--expected-initial-state-sha256")
    parser.add_argument("--expected-query-state-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    require(not (args.preflight_only and args.smoke_only),
            "A2 CPU preflight and two-update GPU smoke are separate modes")
    if args.smoke_only:
        require("smoke_only" in Path(args.run_dir).name,
                "A2 smoke output directory must be unmistakably named smoke_only")
    runtime = validate_runtime(args)
    source = validate_source_manifest(args.source_manifest, args.expected_source_manifest_sha256)
    payload, parent = validate_parent(args)
    data, overlay = a1.validate_data(args)
    model, prepared = prepare_model(payload, args.seed)
    experiment = build_experiment(args, source, data, parent, prepared)
    command = trainer_argv(args)
    if args.preflight_only:
        print(json.dumps({"status": "completed_cpu_preflight_only", "arm": args.arm,
                          "seed": args.seed, "runtime": runtime, "source": source,
                          "data": data, "parent": parent, "prepared": prepared,
                          "experiment": experiment, "trainer_argv": command,
                          "cuda_initialized": torch.cuda.is_initialized()},
                         sort_keys=True, allow_nan=False), flush=True)
        return
    require(args.expected_initial_state_sha256 == prepared["initial_model_state_sha256"]
            and args.expected_query_state_sha256 == prepared["query_state_sha256"],
            "A2 launch requires reviewed assembled/query state pins")
    immutable = [args.init, args.init_manifest, args.source_manifest, args.split_manifest,
                 str(Path(args.supervision_root) / "supervision_manifest.json"),
                 str(Path(args.supervision_root) / "calibration.npz"),
                 str(Path(args.status_overlay_root) / "overlay_manifest.json"),
                 str(Path(args.status_overlay_root) / "train.npz"),
                 str(Path(args.status_overlay_root) / "tune.npz")]
    before = {str(Path(path).resolve()): sha256(path) for path in immutable}
    import train_motiondrive_v2 as trainer
    with patched_training_runtime(overlay, args.arm, args.seed,
                                  PARENTS[args.seed]["model_state_sha256"],
                                  prepared["query_state_sha256"], prepared["missing_keys"]):
        trainer.run_training(command, experiment=experiment)
    after = {str(Path(path).resolve()): sha256(path) for path in immutable}
    require(before == after, "A2 immutable inputs changed during training")
    validate_source_manifest(args.source_manifest, args.expected_source_manifest_sha256)


if __name__ == "__main__":
    main()
