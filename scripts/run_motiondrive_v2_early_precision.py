#!/usr/bin/env python3
"""Run the fixed three-arm MotionDrive V2 early-precision screen."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_motiondrive_v2_shared_status_a1 as a1
import run_motiondrive_v2_shared_status_a2 as a2
from build_grouped_split_v2 import sha256
from models.motiondrive_v2.early_precision import (
    OrderedTemporalMotionEncoder,
    install_ordered_temporal_encoder,
    ordered_residual_state,
)
from models.motiondrive_v2.shared_status_query import install_shared_status_query
from motiondrive_v2_training import tensor_state_sha256


NAME = "motiondrive_v2_early_precision"
ARMS = ("continuation_control", "ordered_motion_residual", "early_delta_aux")
PARENT = {
    "checkpoint_sha256": "2cd1241979da921756c714f2b16ab97d4a63d0586cf666fc57960560107bbfea",
    "model_state_sha256": "96c16d73f8ee1de790680edd45da08539395920e90eec9487ade789e5040ad9a",
    "sidecar_sha256": "37a06fbd939a293304458bbf9601ee7117f5772323395fb698b5183031f58312",
    "source_manifest_sha256": "d140972b85e8424066284d93945cad54351b6dc0cd3a266ec67a10e208a52418",
}
GPU_ASSIGNMENTS = {
    "continuation_control": "GPU-4b804d68-fd61-af14-393a-573c533d5006",
    "ordered_motion_residual": "GPU-1e9aea73-4e6b-2cb5-b788-f3d99e6dc6f8",
    "early_delta_aux": "GPU-4b804d68-fd61-af14-393a-573c533d5006",
}
DELTA_BETA_METRES = .1
DELTA_ALPHA = .2
SOURCE_FILES = set(a2.SOURCE_FILES) | {
    "models/motiondrive_v2/early_precision.py",
    "scripts/run_motiondrive_v2_early_precision.py",
    "tests/test_motiondrive_v2_early_precision.py",
    "reports/motiondrive_v2_early_precision_protocol_20260909.md",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_source_manifest(path, expected_sha256):
    path = Path(path).resolve()
    require(len(expected_sha256) == 64 and sha256(path) == expected_sha256,
            "early-precision source-manifest SHA mismatch")
    manifest = json.loads(path.read_text())
    require(set(manifest) == {"schema_version", "git_sha", "file_sha256"}
            and manifest["schema_version"] == 1
            and isinstance(manifest["git_sha"], str) and len(manifest["git_sha"]) == 40
            and isinstance(manifest["file_sha256"], dict)
            and set(manifest["file_sha256"]) == SOURCE_FILES,
            "early-precision source manifest must be the exact runtime closure")
    actual = {name: sha256(ROOT / name) for name in sorted(SOURCE_FILES)}
    require(actual == manifest["file_sha256"], "early-precision runtime source differs from manifest")
    return {"path": str(path), "sha256": expected_sha256,
            "git_sha": manifest["git_sha"], "file_sha256": actual}


def validate_runtime(args):
    require(args.seed == 0 and args.gpu == 0 and args.workers == 4
            and args.cuda_memory_limit_mib == 12000 and args.cuda_min_free_mib == 8192,
            "early-precision runtime contract mismatch")
    expected = GPU_ASSIGNMENTS[args.arm]
    require(args.expected_physical_gpu_uuid == expected,
            "early-precision physical GPU assignment mismatch")
    if args.preflight_only:
        return {"gpu_used": False, "expected_physical_gpu_uuid": expected,
                "cuda_initialized": torch.cuda.is_initialized()}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(visible == expected and "," not in visible and torch.cuda.is_available()
            and torch.cuda.device_count() == 1,
            "early-precision run requires one UUID-isolated CUDA device")
    raw = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    actual = raw if raw.startswith("GPU-") else "GPU-" + raw
    require(actual == expected, "early-precision observed physical GPU UUID mismatch")
    return {"gpu_used": True, "logical_gpu": 0, "actual_physical_gpu_uuid": actual}


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def validate_parent(args):
    checkpoint = Path(args.init).resolve()
    sidecar_path = Path(args.init_manifest).resolve()
    require(sha256(checkpoint) == PARENT["checkpoint_sha256"]
            and sha256(sidecar_path) == PARENT["sidecar_sha256"],
            "requires exact A2 PROVIDED seed0 LAST2000 and sidecar")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sidecar = json.loads(sidecar_path.read_text())
    require(isinstance(payload, dict) and {"model", "step", "manifest"} <= set(payload)
            and payload["step"] == 2000 and sidecar.get("step") == 2000
            and sidecar.get("status") == "completed",
            "parent must be completed A2 PROVIDED LAST2000")
    require(tensor_state_sha256(payload["model"]) == PARENT["model_state_sha256"],
            "A2 parent model-state SHA mismatch")
    embedded = payload["manifest"]
    protocol = embedded.get("experimental_protocol", {})
    require(protocol.get("name") == a2.NAME
            and protocol.get("arm") == "provided_causal_5d"
            and protocol.get("seed") == 0
            and embedded.get("split_sha256") == a1.EXPECTED_SPLIT_SHA256
            and embedded.get("history_overlay_manifest_sha256") is None,
            "A2 PROVIDED parent protocol mismatch")
    for field in ("model_config", "loss_weights", "time_input", "time_input_policy",
                  "split_sha256", "load_report", "experimental_protocol"):
        require(_canonical(embedded.get(field)) == _canonical(sidecar.get(field)),
                f"A2 parent embedded/sidecar mismatch: {field}")
    require(any(name.startswith("shared_status_query_fusion.") for name in payload["model"])
            and not any(name.startswith("shared_status_fusion.") for name in payload["model"]),
            "parent must contain A2 query conditioning and no A1 FiLM")
    return payload, {"checkpoint": str(checkpoint), **PARENT, "step": 2000,
                     "weights_only": True, "fresh_optimizer": True}


def make_model(config, arm):
    # Import the immutable implementation, not the package export temporarily
    # replaced by patched_runtime's trainer factory.
    from models.motiondrive_v2.model import MotionDriveV2
    model = install_shared_status_query(MotionDriveV2(config))
    if arm == "ordered_motion_residual":
        install_ordered_temporal_encoder(model)
    return model


def prepare_model(payload, arm):
    import train_motiondrive_v2 as trainer
    from models.motiondrive_v2 import MotionDriveV2Config
    trainer.seed_all(0)
    model = make_model(MotionDriveV2Config(**payload["manifest"]["model_config"]), arm)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    missing = list(incompatible.missing_keys)
    expected = ([name for name in model.state_dict()
                 if name.startswith("motion_encoder.ordered_temporal_residual.")]
                if arm == "ordered_motion_residual" else [])
    require(not incompatible.unexpected_keys and sorted(missing) == sorted(expected),
            "A2 parent must miss exactly the optional ordered residual")
    require(all(torch.equal(model.state_dict()[name], value)
                for name, value in payload["model"].items()),
            "A2 parent tensors did not load exactly")
    residual_sha = None
    if expected:
        final = model.motion_encoder.ordered_temporal_residual.output_projection
        require(not bool(final.weight.count_nonzero()) and not bool(final.bias.count_nonzero()),
                "ordered residual must begin at exact zero")
        residual_sha = tensor_state_sha256(ordered_residual_state(model))
    return model, {"initial_model_state_sha256": tensor_state_sha256(model.state_dict()),
                   "ordered_residual_state_sha256": residual_sha,
                   "missing_keys": missing,
                   "parameter_count": sum(p.numel() for p in model.parameters()),
                   "initial_parent_output_exact": True}


def early_delta_smoothl1(outputs, batch, normalizers=None):
    pred = outputs["plan_abs"].float()
    target = batch["gt_plan"].float()
    if pred.shape != target.shape or pred.shape[-2:] != (6, 2):
        raise ValueError("early delta requires matching [B,6,2] plans")
    valid = batch.get("plan_valid", torch.ones_like(pred[..., 0], dtype=torch.bool))
    if valid.shape != pred.shape[:2]:
        raise ValueError("plan_valid must be [B,6]")
    complete = valid.bool().all(-1)
    safe_target = torch.where(complete[:, None, None], target, torch.zeros_like(target))
    origin = torch.zeros_like(pred[:, :1])
    pred_delta = torch.cat([pred[:, :1] - origin, pred[:, 1:2] - pred[:, :1]], 1)
    gt_delta = torch.cat([safe_target[:, :1] - origin,
                          safe_target[:, 1:2] - safe_target[:, :1]], 1)
    value = F.smooth_l1_loss(pred_delta, gt_delta, beta=DELTA_BETA_METRES,
                             reduction="none").mean((1, 2))
    if normalizers is None:
        denominator = complete.sum()
    else:
        denominator = normalizers["plan_complete"]
    return (value * complete.to(value.dtype)).sum() / denominator.clamp_min(1)


def loss_with_early_diagnostic(original, alpha, outputs, batch, weights, **kwargs):
    total, parts = original(outputs, batch, weights, **kwargs)
    raw = early_delta_smoothl1(outputs, batch, kwargs.get("normalizers"))
    weighted = raw * alpha
    if alpha:
        total = total + weighted
        parts = {**parts, "total": total}
    else:
        # Preserve the exact original total tensor/value in non-loss arms.
        parts = dict(parts)
    parts["early_delta_smoothl1"] = raw
    parts["early_delta_weighted"] = weighted
    return total, parts


def build_experiment(args, source, data, parent, prepared):
    return {"schema_version": 1, "name": NAME, "arm": args.arm, "seed": 0,
            "source": source, "parent": parent,
            "status_overlay_manifest_sha256": args.expected_status_overlay_sha256,
            "train_data": {"rows": data["train_rows"], "rows_sha256": data["train_rows_sha256"]},
            "tune_data": {"rows": data["tune_rows"], "rows_sha256": data["tune_rows_sha256"]},
            "expected_initial_model_state_sha256": prepared["initial_model_state_sha256"],
            "expected_ordered_residual_state_sha256": prepared["ordered_residual_state_sha256"],
            "expected_missing_state_keys": prepared["missing_keys"],
            "expected_optimizer_groups": [{"name": "backbone", "base_lr": 5e-6},
                                          {"name": "head", "base_lr": 5e-5}],
            "ordered_motion": {"enabled": args.arm == "ordered_motion_residual",
                "source": "four ordered time-encoded image-correlation tokens per spatial site",
                "shape": [4, 192, 128], "mixer": [512, 64, 128],
                "insertion": "weighted motion before token_refine and state head",
                "provided_status_goal_pose_inputs": False, "final_weight_and_bias_zero": True},
            "early_delta": {"alpha": DELTA_ALPHA if args.arm == "early_delta_aux" else 0.,
                "beta_metres": DELTA_BETA_METRES, "increments": ["origin_to_0.5s", "0.5s_to_1.0s"],
                "components": 4, "mask": "complete_six_point_rows",
                "denominator": "full_effective_batch_plan_complete"},
            "fresh_optimizer_step_zero": True, "all_parameters_trainable": True,
            "fixed_bn_running_statistics": True,
            "smoke_only_never_training_initializer": bool(args.smoke_only),
            "terminal_tune_evaluations": 0 if args.smoke_only else 1,
            "final_validation_accessed": False}


@contextlib.contextmanager
def patched_runtime(arm, overlay, expected_parent_sha, expected_missing,
                    expected_residual_sha):
    import models.motiondrive_v2 as model_api
    import motiondrive_v2_data as data_api
    import train_motiondrive_v2 as trainer
    import evaluate_motiondrive_v2_planning as planning_eval
    originals = {"model": model_api.MotionDriveV2, "dataset": data_api.MotionDriveDataset,
                 "train_inputs": trainer.model_inputs,
                 "eval_inputs": planning_eval.planning_model_inputs,
                 "protocol": trainer._validate_experimental_protocol,
                 "runtime": trainer._validate_experimental_runtime,
                 "loader": trainer._load_initial_model_state,
                 "schedule": trainer._training_schedule_actions,
                 "loss": trainer.compute_loss}

    def model_factory(config=None):
        from models.motiondrive_v2 import MotionDriveV2Config
        if config is None:
            config = MotionDriveV2Config()
        return make_model(config, arm)

    def dataset_factory(**kwargs):
        require(kwargs.get("split") in ("train", "tune"), "only train/tune datasets are permitted")
        return a1.SharedStatusDataset(originals["dataset"](**kwargs), kwargs["split"],
                                      overlay, "provided_causal_5d")

    def train_inputs(batch, time_input="raw", nominal_history_seconds=(.1, .2, .5, 1.)):
        return a1.shared_status_model_inputs(originals["train_inputs"], batch,
                                             time_input=time_input,
                                             nominal_history_seconds=nominal_history_seconds)

    def eval_inputs(batch, time_input="raw"):
        return a1._add_status(originals["eval_inputs"](batch, time_input), batch)

    def validate_protocol(experiment):
        require(experiment.get("name") == NAME and experiment.get("arm") == arm
                and experiment.get("seed") == 0,
                "early-precision experimental protocol mismatch")

    def validate_experimental_runtime(runtime_args, experiment):
        expected = {"phase": "joint", "goal_on": 1, "state_on": 1,
                    "steps": 2 if experiment["smoke_only_never_training_initializer"] else 2000,
                    "batch": 16, "microbatch": 2, "eval_batch": 4,
                    "eval_every": 2000, "save_every": 500,
                    "lr": 5e-5, "backbone_lr": 5e-6, "weight_decay": .01,
                    "warmup": 100, "grad_clip": 5., "alpha_occ": .2,
                    "alpha_lane": .2, "alpha_motion": .2, "uncertainty": 1,
                    "precision": "bf16", "time_input": "nominal", "bn_policy": "fixed",
                    "arch": "resnet50", "motion_input_mode": "low_feature",
                    "cross_cell_goal_mode": "zero", "history_contract": "control",
                    "train_stride": 1, "eval_stride": 5, "max_train_samples": 0,
                    "max_eval_samples": 0, "eval_split": "tune"}
        require(all(getattr(runtime_args, key) == value for key, value in expected.items())
                and runtime_args.seed == 0 and runtime_args.init
                and not runtime_args.resume and not runtime_args.pretrained
                and not runtime_args.eval_only and not runtime_args.train_scenes
                and not runtime_args.eval_scenes,
                "fixed early-precision continuation recipe mismatch")

    def load_initial(model, common, experiment=None):
        require(tensor_state_sha256(common["model"]) == expected_parent_sha,
                "trainer A2 parent state mismatch")
        incompatible = model.load_state_dict(common["model"], strict=False)
        require(list(incompatible.missing_keys) == expected_missing
                and not incompatible.unexpected_keys,
                "trainer optional residual missing-key mismatch")
        if expected_residual_sha is not None:
            require(tensor_state_sha256(ordered_residual_state(model)) == expected_residual_sha,
                    "trainer ordered residual initialization mismatch")
        require(all(torch.equal(model.state_dict()[name], value)
                    for name, value in common["model"].items()),
                "trainer parent tensors changed during load")
        return {"early_precision_parent_load": {"strict_existing_keys": True,
                "missing_keys": expected_missing, "unexpected_keys": [],
                "weights_only": True, "parent_model_state_sha256": expected_parent_sha,
                "ordered_residual_state_sha256": expected_residual_sha}}

    def schedule(step, runtime_args, experiment):
        if experiment["smoke_only_never_training_initializer"]:
            return False, False
        return step == 2000, step in (500, 1000, 2000)

    alpha = DELTA_ALPHA if arm == "early_delta_aux" else 0.
    def compute_loss(outputs, batch, weights, **kwargs):
        return loss_with_early_diagnostic(originals["loss"], alpha,
                                          outputs, batch, weights, **kwargs)

    model_api.MotionDriveV2 = model_factory
    data_api.MotionDriveDataset = dataset_factory
    trainer.model_inputs = train_inputs
    planning_eval.planning_model_inputs = eval_inputs
    trainer._validate_experimental_protocol = validate_protocol
    trainer._validate_experimental_runtime = validate_experimental_runtime
    trainer._load_initial_model_state = load_initial
    trainer._training_schedule_actions = schedule
    trainer.compute_loss = compute_loss
    try:
        yield
    finally:
        model_api.MotionDriveV2 = originals["model"]
        data_api.MotionDriveDataset = originals["dataset"]
        trainer.model_inputs = originals["train_inputs"]
        planning_eval.planning_model_inputs = originals["eval_inputs"]
        trainer._validate_experimental_protocol = originals["protocol"]
        trainer._validate_experimental_runtime = originals["runtime"]
        trainer._load_initial_model_state = originals["loader"]
        trainer._training_schedule_actions = originals["schedule"]
        trainer.compute_loss = originals["loss"]


def trainer_argv(args):
    command = a1.trainer_argv(args)
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
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    parser.add_argument("--gpu", type=int, choices=(0,), default=0)
    parser.add_argument("--expected-physical-gpu-uuid", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cuda-memory-limit-mib", type=int, default=12000)
    parser.add_argument("--cuda-min-free-mib", type=int, default=8192)
    parser.add_argument("--expected-initial-state-sha256")
    parser.add_argument("--expected-ordered-residual-state-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    require(not (args.preflight_only and args.smoke_only), "preflight and smoke are separate")
    if args.smoke_only:
        require("smoke_only" in Path(args.run_dir).name, "smoke directory must be explicit")
    runtime = validate_runtime(args)
    source = validate_source_manifest(args.source_manifest, args.expected_source_manifest_sha256)
    payload, parent = validate_parent(args)
    data_args = argparse.Namespace(**{**vars(args), "arm": "provided_causal_5d"})
    data, overlay = a1.validate_data(data_args)
    _model, prepared = prepare_model(payload, args.arm)
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
            and args.expected_ordered_residual_state_sha256
            == prepared["ordered_residual_state_sha256"],
            "launch requires reviewed initial/residual state pins")
    immutable = [args.init, args.init_manifest, args.source_manifest, args.split_manifest,
                 str(Path(args.supervision_root) / "supervision_manifest.json"),
                 str(Path(args.supervision_root) / "calibration.npz"),
                 str(Path(args.status_overlay_root) / "overlay_manifest.json"),
                 str(Path(args.status_overlay_root) / "train.npz"),
                 str(Path(args.status_overlay_root) / "tune.npz")]
    before = {str(Path(path).resolve()): sha256(path) for path in immutable}
    import train_motiondrive_v2 as trainer
    with patched_runtime(args.arm, overlay, PARENT["model_state_sha256"],
                         prepared["missing_keys"], prepared["ordered_residual_state_sha256"]):
        trainer.run_training(command, experiment=experiment)
    after = {str(Path(path).resolve()): sha256(path) for path in immutable}
    require(before == after, "immutable inputs changed during training")
    validate_source_manifest(args.source_manifest, args.expected_source_manifest_sha256)


if __name__ == "__main__":
    main()
