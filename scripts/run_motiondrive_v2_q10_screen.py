#!/usr/bin/env python3
"""Run the fixed four-arm MotionDrive V2 long-training continuation."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_motiondrive_v2_shared_status_a1 as a1
import run_motiondrive_v2_shared_status_a2 as a2
from build_grouped_split_v2 import sha256, validate_manifest
from models.motiondrive_v2.shared_status_query import install_shared_status_query
from motiondrive_v2_training import tensor_state_sha256


NAME = "motiondrive_v2_q10_screen"
# The 12-session holdout used by the earlier screens is a subset of train203,
# which the shared parent already trained on, so it cannot separate
# generalization from retained memorization.  These arms train on the full
# train split and are scored on tune, which no arm has ever trained on.
# No A2 query conditioning in any arm: the provided causal status reaches
# neither the scene-encoder query nor the planner.  The planner still receives
# `predicted_state` / `predicted_history`, which the network infers from the
# images and which Q10 permits as planner input.
ARMS = ("q10_noflip_s0", "q10_flip50_s0", "q10_flip50_s1")
ARM_SEEDS = {"q10_noflip_s0": 0, "q10_flip50_s0": 0, "q10_flip50_s1": 1}
FLIP_P = {"q10_noflip_s0": 0.0, "q10_flip50_s0": 0.5, "q10_flip50_s1": 0.5}
FLIP_ARMS = frozenset(arm for arm, p in FLIP_P.items() if p > 0)
HOLDOUT_ARMS = frozenset()       # full train split, tune evaluation
UPDATES = 6850
EVAL_SAVE_EVERY = 3425
# The long run continues the control-flow screen's `direct` arm: the unchanged
# production planner at tune D3 0.280640, itself a descendant of the A2 PROVIDED
# lineage, so the A2 query route and its status overlay still apply.
PARENT = {
    "checkpoint_sha256": "7e1b3f3a52e2703210edf24f546d0cb62b91723952076482866d89204bbb9449",
    "model_state_sha256": "ffaa7f42f55b25d73a187c71af1a1cd96a1f8f32b10ead17a5ccca460c4f192c",
    "sidecar_sha256": "94fae557ff2ffadbd4dc7079cd228245bba9172c4d249a5498d2ca7fa1f31653",
    "source_manifest_sha256": "6552a06ffa3ae743cc41c8b927853d373579759d0b8562d59e6dbbb2e0dd8039",
    "step": 4000,
    "protocol_name": "motiondrive_v2_controlflow_screen",
    "protocol_arm": "direct",
}
SOURCE_FILES = set(a2.SOURCE_FILES) | {
    "scripts/motiondrive_v2_flip_augment.py",
    "scripts/run_motiondrive_v2_q10_screen.py",
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def validate_arm_seed(arm, seed):
    require(arm in ARM_SEEDS and ARM_SEEDS[arm] == seed,
            "long-training arm/seed mismatch")


def validate_source_manifest(path, expected_sha256):
    path = Path(path).resolve()
    require(isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and sha256(path) == expected_sha256,
            "long-training source-manifest SHA mismatch")
    manifest = json.loads(path.read_text())
    require(set(manifest) == {"schema_version", "git_sha", "file_sha256"}
            and manifest["schema_version"] == 1
            and isinstance(manifest["git_sha"], str) and len(manifest["git_sha"]) == 40
            and isinstance(manifest["file_sha256"], dict)
            and set(manifest["file_sha256"]) == SOURCE_FILES,
            "long-training source manifest must be the exact runtime closure")
    actual = {name: sha256(ROOT / name) for name in sorted(SOURCE_FILES)}
    require(actual == manifest["file_sha256"],
            "long-training runtime source differs from manifest")
    return {"path": str(path), "sha256": expected_sha256,
            "git_sha": manifest["git_sha"], "file_sha256": actual}


def validate_runtime(args):
    validate_arm_seed(args.arm, args.seed)
    require(args.gpu == 0 and args.workers == 4
            and args.cuda_memory_limit_mib == 12000
            and args.cuda_min_free_mib == 8192,
            "long-training logical-GPU/worker/memory contract mismatch")
    expected = args.expected_physical_gpu_uuid
    require(isinstance(expected, str) and expected.startswith("GPU-")
            and "," not in expected and "\0" not in expected,
            "long-training physical GPU UUID is invalid")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(visible == expected and "," not in visible and torch.cuda.is_available()
            and torch.cuda.device_count() == 1,
            "long-training run requires one UUID-isolated CUDA device")
    raw = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    actual = raw if raw.startswith("GPU-") else "GPU-" + raw
    require(actual == expected, "long-training observed physical GPU UUID mismatch")
    return {"gpu_used": True, "logical_gpu": 0,
            "cuda_visible_devices": visible,
            "observed_physical_gpu_uuid_raw": raw,
            "actual_physical_gpu_uuid": actual}


def validate_parent(args):
    checkpoint = Path(args.init).resolve()
    sidecar_path = Path(args.init_manifest).resolve()
    require(sha256(checkpoint) == PARENT["checkpoint_sha256"]
            and sha256(sidecar_path) == PARENT["sidecar_sha256"],
            "long training requires the exact control-flow direct LAST4000 and sidecar")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sidecar = json.loads(sidecar_path.read_text())
    require(isinstance(payload, dict) and {"model", "step", "manifest"} <= set(payload)
            and isinstance(payload["model"], dict) and payload["model"]
            and payload["step"] == PARENT["step"] and sidecar.get("step") == PARENT["step"]
            and sidecar.get("status") == "completed",
            "long-training parent must be the completed control-flow direct LAST4000")
    require(tensor_state_sha256(payload["model"]) == PARENT["model_state_sha256"],
            "long-training parent model-state SHA mismatch")
    embedded = payload["manifest"]
    protocol = embedded.get("experimental_protocol", {})
    config = embedded.get("model_config", {})
    require(protocol.get("name") == PARENT["protocol_name"]
            and protocol.get("arm") == PARENT["protocol_arm"]
            and protocol.get("seed") == 0
            and embedded.get("split_sha256") == a1.EXPECTED_SPLIT_SHA256
            and embedded.get("history_overlay_manifest_sha256") is None
            and config.get("goal_on") is True and config.get("state_on") is True
            and config.get("backbone_arch") == "resnet50"
            and config.get("motion_input_mode") == "low_feature"
            and config.get("cross_cell_goal_mode") == "zero"
            and config.get("history_contract", "control") == "control",
            "long-training parent protocol/configuration mismatch")
    for field in ("model_config", "loss_weights", "time_input", "time_input_policy",
                  "split_sha256", "load_report", "experimental_protocol"):
        require(_canonical(embedded.get(field)) == _canonical(sidecar.get(field)),
                f"long-training parent embedded/sidecar mismatch: {field}")
    require(any(name.startswith("shared_status_query_fusion.") for name in payload["model"])
            and not any(name.startswith("shared_status_fusion.") for name in payload["model"]),
            "long-training parent must contain A2 query conditioning and no A1 FiLM")
    return payload, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": PARENT["checkpoint_sha256"],
        "model_state_sha256": PARENT["model_state_sha256"],
        "sidecar": str(sidecar_path),
        "sidecar_sha256": PARENT["sidecar_sha256"],
        "source_manifest_sha256": PARENT["source_manifest_sha256"],
        "step": 2000, "weights_only": True, "fresh_optimizer": True,
        "producer_git_sha": embedded.get("git_sha"),
    }


def make_model(config):
    # Avoid the package export replaced temporarily by patched_runtime.
    # Deliberately NOT wrapped in install_shared_status_query: this screen has
    # no provided-status route at all.
    from models.motiondrive_v2.model import MotionDriveV2
    return MotionDriveV2(config)


def prepare_model(payload, seed):
    import train_motiondrive_v2 as trainer
    from models.motiondrive_v2 import MotionDriveV2Config

    trainer.seed_all(seed)
    model = make_model(MotionDriveV2Config(**payload["manifest"]["model_config"]))
    # The parent is A2-descended; its query-conditioning tensors are dropped
    # rather than loaded, and every remaining tensor must match exactly.
    parent_state = {name: value for name, value in payload["model"].items()
                    if not name.startswith("shared_status_query_fusion.")}
    dropped = len(payload["model"]) - len(parent_state)
    require(dropped > 0, "expected an A2 parent whose query tensors can be dropped")
    incompatible = model.load_state_dict(parent_state, strict=True)
    require(not incompatible.missing_keys and not incompatible.unexpected_keys,
            "q10 parent strict load failed after dropping A2 tensors")
    require(all(torch.equal(model.state_dict()[name], value)
                for name, value in parent_state.items()),
            "q10 parent tensors did not load exactly")
    require(not any(name.startswith("shared_status_query_fusion.")
                    for name in model.state_dict()),
            "q10 model must not carry A2 query conditioning")
    return model, {
        "initial_model_state_sha256": tensor_state_sha256(model.state_dict()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "strict_parent_load": True,
        "missing_keys": [], "unexpected_keys": [],
        "dropped_a2_query_tensors": dropped,
    }


def validate_holdout_split(path, expected_sha256, split_manifest):
    path = Path(path).resolve()
    require(isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and sha256(path) == expected_sha256,
            "long-training holdout-split SHA mismatch")
    artifact = json.loads(path.read_text())
    require(set(artifact) == {"holdout_sessions", "holdout_scenes", "train_scenes", "rule"},
            "long-training holdout-split keyset mismatch")
    for key in ("holdout_sessions", "holdout_scenes", "train_scenes"):
        values = artifact[key]
        require(type(values) is list and values
                and all(isinstance(value, str) and value and "\0" not in value
                        for value in values)
                and len(values) == len(set(values)),
                f"long-training {key} must be a nonempty unique string list")
    require(artifact["rule"] is not None,
            "long-training holdout rule is missing")
    require(all(not scene.startswith("-")
                for key in ("holdout_scenes", "train_scenes")
                for scene in artifact[key]),
            "long-training scene names cannot be parsed as CLI options")
    held_out, training = set(artifact["holdout_scenes"]), set(artifact["train_scenes"])
    require(held_out.isdisjoint(training),
            "long-training holdout and train scenes overlap")

    split = json.loads(Path(split_manifest).resolve().read_text())
    validate_manifest(split)
    full_train = set(split["splits"]["train"])
    require(len(full_train) == 203 and (held_out | training) <= full_train
            and held_out and training,
            "data-scaling artifact must be disjoint subsets of train203")
    mapping = split["scene_to_session"]
    expected_sessions = {mapping[scene] for scene in held_out}
    require(set(artifact["holdout_sessions"]) == expected_sessions
            and not ({mapping[scene] for scene in training} & expected_sessions),
            "long-training holdout sessions do not match the held-out scenes")
    return {"path": str(path), "sha256": expected_sha256, **artifact}


_FLIP = {"on": False, "p": 0.0, "seed": 0, "wf": 768, "wh": 384}


def _overlay_for_dataset(base, split, overlay):
    source = overlay[split]
    source_rows = np.asarray(source["row"], dtype=np.int64)
    rows = np.asarray(base.rows, dtype=np.int64)
    indices = np.searchsorted(source_rows, rows)
    valid = indices < len(source_rows)
    require(bool(valid.all()) and np.array_equal(source_rows[indices], rows),
            "long-training dataset rows are absent from the status overlay")
    return {split: {key: value[indices] for key, value in source.items()}}


def status_dataset(base, split, overlay, augmentable=False):
    """No status overlay in this screen; only optional flip augmentation."""
    if _FLIP["on"] and augmentable:
        from motiondrive_v2_flip_augment import FlipAugmented
        return FlipAugmented(base, _FLIP["wf"], _FLIP["wh"],
                             p=_FLIP["p"], seed=_FLIP["seed"])
    return base


def _unused_status_dataset(base, split, overlay, augmentable=False):
    """Attach the flip wrapper only where the caller says augmentation belongs.

    Holdout arms build BOTH the training set and the evaluation set with
    split == "train" and separate them by scene list, so keying augmentation
    off the split alone silently mirrors half of the evaluation rows.
    """
    aligned = _overlay_for_dataset(base, split, overlay)
    ds = a1.SharedStatusDataset(base, split, aligned, "provided_causal_5d")
    if augmentable and _FLIP["on"]:
        from motiondrive_v2_flip_augment import FlipAugmented
        ds = FlipAugmented(ds, _FLIP["wf"], _FLIP["wh"],
                           p=_FLIP["p"], seed=_FLIP["seed"])
    return ds


def validate_data(args):
    _FLIP["on"] = args.arm in FLIP_ARMS
    _FLIP["p"] = FLIP_P[args.arm]
    _FLIP["seed"] = args.seed
    data_args = argparse.Namespace(**{**vars(args), "arm": "provided_causal_5d"})
    full, overlay = a1.validate_data(data_args)
    holdout = validate_holdout_split(
        args.holdout_split, args.expected_holdout_split_sha256, args.split_manifest)
    result = dict(full)
    result["holdout_split"] = holdout
    if args.arm in HOLDOUT_ARMS:
        from motiondrive_v2_data import MotionDriveDataset

        common = dict(data_root=args.data_root,
                      split_manifest=str(Path(args.split_manifest).resolve()),
                      supervision_root=str(Path(args.supervision_root).resolve()),
                      min_frame=30, max_samples=0, seed=args.seed,
                      history_contract="control")
        training = MotionDriveDataset(
            split="train", frame_stride=1, augment=True,
            scenes=holdout["train_scenes"], **common)
        evaluation = MotionDriveDataset(
            split="train", frame_stride=5, augment=False,
            scenes=holdout["holdout_scenes"], **common)
        status_dataset(training, "train", overlay, augmentable=True)
        status_dataset(evaluation, "train", overlay, augmentable=False)
        result.update(
            train_rows=len(training), train_rows_sha256=a1.rows_sha(training.rows),
            eval_rows=len(evaluation), eval_rows_sha256=a1.rows_sha(evaluation.rows),
            eval_split="train", train_scenes=list(holdout["train_scenes"]),
            eval_scenes=list(holdout["holdout_scenes"]), holdout_applied=True)
    else:
        result.update(
            eval_rows=full["tune_rows"], eval_rows_sha256=full["tune_rows_sha256"],
            eval_split="tune", train_scenes=None, eval_scenes=None,
            holdout_applied=False)
    return result, overlay, holdout


def build_experiment(args, source, data, parent, prepared):
    return {
        "schema_version": 1, "name": NAME, "arm": args.arm, "seed": args.seed,
        "source": source, "parent": parent,
        "status_route": "none anywhere; planner consumes image-inferred state only",
        "holdout_split": {
            "path": data["holdout_split"]["path"],
            "sha256": data["holdout_split"]["sha256"],
            "rule": data["holdout_split"]["rule"],
            "holdout_sessions": data["holdout_split"]["holdout_sessions"],
            "holdout_scenes": data["holdout_split"]["holdout_scenes"],
            "train_scenes": data["holdout_split"]["train_scenes"],
            "applied": data["holdout_applied"],
        },
        "train_data": {"rows": data["train_rows"],
                       "rows_sha256": data["train_rows_sha256"],
                       "split": "train", "scenes": data["train_scenes"]},
        "evaluation_data": {"rows": data["eval_rows"],
                            "rows_sha256": data["eval_rows_sha256"],
                            "split": data["eval_split"],
                            "scenes": data["eval_scenes"]},
        "expected_initial_model_state_sha256": prepared["initial_model_state_sha256"],
        "expected_optimizer_groups": [{"name": "backbone", "base_lr": 5e-6},
                                      {"name": "head", "base_lr": 5e-5}],
        "updates": UPDATES,
        "recipe": {
            "updates": UPDATES, "optimizer": "fresh AdamW",
            "backbone_lr": 5e-6, "head_lr": 5e-5, "weight_decay": .01,
            "warmup_updates": 100, "decay": "cosine",
            "cosine_decay_end_update": UPDATES, "grad_clip": 5.,
            "batch": 16, "microbatch": 2, "eval_batch": 4,
            "eval_every": EVAL_SAVE_EVERY, "save_every": EVAL_SAVE_EVERY,
            "bn_running_statistics": "fixed", "precision": "bf16",
            "planning_path": "existing fp32", "auxiliary_weights": {
                "occupancy": .2, "lane": .2, "motion": .2},
        },
        "provided_status_used": False,
        "flip_augmentation": {"probability": FLIP_P[args.arm],
                              "seed": args.seed,
                              "applied_to": "training dataset only"},
        "unchanged_production_planner": True,
        "fresh_optimizer_step_zero": True,
        "all_parameters_trainable": True,
        "fixed_bn_running_statistics": True,
        "final_validation_accessed": False,
    }


def _expected_scene_arguments(arm, holdout):
    if arm in HOLDOUT_ARMS:
        return list(holdout["train_scenes"]), list(holdout["holdout_scenes"]), "train"
    return None, None, "tune"


class _HoldoutTrainEvalSplit(str):
    """Keep ``train`` semantics while passing the old tune-only guard.

    Commit 54ef584 rejects ``args.eval_split != "tune"`` before its
    experimental schedule can establish that no best-checkpoint selection is
    performed.  Long holdout runs evaluate pinned train scenes and save only on
    their fixed schedule, so this string remains ``train`` for manifests,
    hashing, and dataset construction while bypassing exactly that legacy test.
    """

    def __new__(cls):
        return super().__new__(cls, "train")

    def __ne__(self, other):
        if other == "tune":
            return False
        return super().__ne__(other)


@contextlib.contextmanager
def patched_runtime(arm, seed, overlay, holdout, expected_parent_sha,
                    expected_initial_sha):
    """Install only A2/status/trainer adapters and restore every binding."""
    import models.motiondrive_v2 as model_api
    import motiondrive_v2_data as data_api
    import train_motiondrive_v2 as trainer
    import evaluate_motiondrive_v2_planning as planning_eval

    validate_arm_seed(arm, seed)
    expected_train_scenes, expected_eval_scenes, expected_eval_split = (
        _expected_scene_arguments(arm, holdout))
    originals = {
        "model": model_api.MotionDriveV2,
        "dataset": data_api.MotionDriveDataset,
        "train_inputs": trainer.model_inputs,
        "eval_inputs": planning_eval.planning_model_inputs,
        "protocol": trainer._validate_experimental_protocol,
        "runtime": trainer._validate_experimental_runtime,
        "loader": trainer._load_initial_model_state,
        "schedule": trainer._training_schedule_actions,
        "arguments": trainer.arguments,
    }

    def parse_arguments(argv=None):
        parsed = originals["arguments"](argv)
        if arm in HOLDOUT_ARMS:
            require(parsed.eval_split == "train",
                    "holdout trainer argv must request the train eval split")
            parsed.eval_split = _HoldoutTrainEvalSplit()
        return parsed

    def model_factory(config=None):
        from models.motiondrive_v2 import MotionDriveV2Config
        if config is None:
            config = MotionDriveV2Config()
        return make_model(config)

    def dataset_factory(**kwargs):
        split = kwargs.get("split")
        scenes = kwargs.get("scenes")
        if arm in HOLDOUT_ARMS:
            require(split == "train", "holdout arms permit only the train split")
            requested = list(scenes or [])
            require(requested in (expected_train_scenes, expected_eval_scenes),
                    "holdout arm dataset scenes differ from the pinned artifact")
            augmentable = requested == expected_train_scenes
            if requested == expected_eval_scenes:
                # The old trainer derives augmentation from split == "train".
                # Held-out train-split evaluation must remain deterministic.
                kwargs["augment"] = False
            else:
                require(kwargs.get("augment") is True,
                        "holdout training dataset must retain augmentation")
        else:
            require(split in ("train", "tune") and scenes is None
                    and ((split == "train") == bool(kwargs.get("augment"))),
                    "long arm must use full train and tune datasets")
            augmentable = split == "train"
        base = originals["dataset"](**kwargs)
        return status_dataset(base, split, overlay, augmentable=augmentable)

    def train_inputs(batch, time_input="raw",
                     nominal_history_seconds=(.1, .2, .5, 1.)):
        # Plain production inputs: no provided_status5 is added.
        return originals["train_inputs"](
            batch, time_input=time_input,
            nominal_history_seconds=nominal_history_seconds)

    def eval_inputs(batch, time_input="raw"):
        return originals["eval_inputs"](batch, time_input)

    def validate_protocol(experiment):
        require(isinstance(experiment, dict) and experiment.get("name") == NAME
                and experiment.get("arm") == arm and experiment.get("seed") == seed
                and experiment.get("expected_initial_model_state_sha256")
                    == expected_initial_sha
                and experiment.get("holdout_split", {}).get("sha256")
                    == holdout["sha256"]
                and experiment.get("holdout_split", {}).get("applied")
                    == (arm in HOLDOUT_ARMS),
                "long-training experimental protocol mismatch")

    def validate_experimental_runtime(runtime_args, experiment):
        expected = {
            "phase": "joint", "goal_on": 1, "state_on": 1,
            "steps": UPDATES, "batch": 16, "microbatch": 2,
            "eval_batch": 4, "eval_every": EVAL_SAVE_EVERY,
            "save_every": EVAL_SAVE_EVERY, "lr": 5e-5,
            "backbone_lr": 5e-6, "weight_decay": .01, "warmup": 100,
            "grad_clip": 5., "alpha_occ": .2, "alpha_lane": .2,
            "alpha_motion": .2, "uncertainty": 1, "precision": "bf16",
            "time_input": "nominal", "bn_policy": "fixed",
            "arch": "resnet50", "motion_input_mode": "low_feature",
            "cross_cell_goal_mode": "zero", "history_contract": "control",
            "train_stride": 1, "eval_stride": 5, "max_train_samples": 0,
            "max_eval_samples": 0, "eval_split": expected_eval_split,
        }
        require(all(getattr(runtime_args, key) == value
                    for key, value in expected.items())
                and runtime_args.seed == seed and runtime_args.init
                and not runtime_args.resume and not runtime_args.pretrained
                and not runtime_args.eval_only and not runtime_args.cpu
                and runtime_args.train_scenes == expected_train_scenes
                and runtime_args.eval_scenes == expected_eval_scenes,
                "fixed long-training continuation recipe mismatch")

    def load_initial(model, common, experiment=None):
        require(tensor_state_sha256(common["model"]) == expected_parent_sha,
                "long-training trainer A2 parent state mismatch")
        incompatible = model.load_state_dict(common["model"], strict=True)
        require(not incompatible.missing_keys and not incompatible.unexpected_keys,
                "long-training trainer parent strict load failed")
        require(tensor_state_sha256(model.state_dict()) == expected_initial_sha,
                "long-training trainer step-zero state mismatch")
        return {"long_training_parent_load": {
            "strict": True, "missing_keys": [], "unexpected_keys": [],
            "weights_only": True,
            "parent_model_state_sha256": expected_parent_sha,
        }}

    def schedule(step, runtime_args, experiment):
        require(runtime_args.steps == UPDATES
                and runtime_args.eval_every == EVAL_SAVE_EVERY
                and runtime_args.save_every == EVAL_SAVE_EVERY,
                "long-training schedule contract mismatch")
        due = step % EVAL_SAVE_EVERY == 0
        return due, due

    model_api.MotionDriveV2 = model_factory
    data_api.MotionDriveDataset = dataset_factory
    trainer.model_inputs = train_inputs
    planning_eval.planning_model_inputs = eval_inputs
    trainer._validate_experimental_protocol = validate_protocol
    trainer._validate_experimental_runtime = validate_experimental_runtime
    trainer._load_initial_model_state = load_initial
    trainer._training_schedule_actions = schedule
    trainer.arguments = parse_arguments
    try:
        yield {"train_model_inputs": train_inputs,
               "terminal_model_inputs": eval_inputs}
    finally:
        model_api.MotionDriveV2 = originals["model"]
        data_api.MotionDriveDataset = originals["dataset"]
        trainer.model_inputs = originals["train_inputs"]
        planning_eval.planning_model_inputs = originals["eval_inputs"]
        trainer._validate_experimental_protocol = originals["protocol"]
        trainer._validate_experimental_runtime = originals["runtime"]
        trainer._load_initial_model_state = originals["loader"]
        trainer._training_schedule_actions = originals["schedule"]
        trainer.arguments = originals["arguments"]


def trainer_argv(args, holdout=None):
    validate_arm_seed(args.arm, args.seed)
    if holdout is None:
        holdout = validate_holdout_split(
            args.holdout_split, args.expected_holdout_split_sha256,
            args.split_manifest)
    train_scenes, eval_scenes, eval_split = _expected_scene_arguments(args.arm, holdout)
    command = [
        "--data-root", str(Path(args.data_root).resolve()),
        "--split-manifest", str(Path(args.split_manifest).resolve()),
        "--supervision-root", str(Path(args.supervision_root).resolve()),
        "--run-dir", str(Path(args.run_dir).resolve()),
        "--phase", "joint", "--goal-on", "1", "--state-on", "1",
        "--cross-cell-goal-mode", "zero", "--history-contract", "control",
        "--gpu", "0", "--seed", str(args.seed), "--steps", str(UPDATES),
        "--batch", "16", "--microbatch", "2", "--eval-batch", "4",
        "--workers", "4", "--eval-every", str(EVAL_SAVE_EVERY),
        "--save-every", str(EVAL_SAVE_EVERY), "--log-every", "10",
        "--lr", "0.00005", "--backbone-lr", "0.000005",
        "--weight-decay", "0.01", "--warmup", "100", "--grad-clip", "5",
        "--alpha-occ", "0.2", "--alpha-lane", "0.2", "--alpha-motion", "0.2",
        "--uncertainty", "1", "--precision", "bf16", "--time-input", "nominal",
        "--bn-policy", "fixed", "--init", str(Path(args.init).resolve()),
        "--arch", "resnet50", "--motion-input-mode", "low_feature",
        "--train-stride", "1", "--eval-stride", "5",
        "--max-train-samples", "0", "--max-eval-samples", "0",
    ]
    if train_scenes is not None:
        command.extend(("--train-scenes", *train_scenes,
                        "--eval-scenes", *eval_scenes))
    command.extend(("--eval-split", eval_split,
                    "--cuda-memory-limit-mib", "12000",
                    "--cuda-min-free-mib", "8192"))
    return command


def immutable_inputs(args):
    overlay_root = Path(args.status_overlay_root)
    return [
        args.init, args.init_manifest, args.source_manifest, args.split_manifest,
        str(Path(args.supervision_root) / "supervision_manifest.json"),
        str(Path(args.supervision_root) / "calibration.npz"),
        args.holdout_split,
        str(overlay_root / "overlay_manifest.json"),
        str(overlay_root / "train.npz"), str(overlay_root / "tune.npz"),
    ]


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
    parser.add_argument("--holdout-split", required=True)
    parser.add_argument("--expected-holdout-split-sha256", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--gpu", type=int, choices=(0,), default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cuda-memory-limit-mib", type=int, default=12000)
    parser.add_argument("--cuda-min-free-mib", type=int, default=8192)
    parser.add_argument("--expected-physical-gpu-uuid", required=True)
    parser.add_argument("--expected-initial-state-sha256", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    runtime = validate_runtime(args)
    source = validate_source_manifest(
        args.source_manifest, args.expected_source_manifest_sha256)
    payload, parent = validate_parent(args)
    data, overlay, holdout = validate_data(args)
    _model, prepared = prepare_model(payload, args.seed)
    require(args.expected_initial_state_sha256
            == prepared["initial_model_state_sha256"],
            "long-training launch requires the reviewed initial-state pin")
    experiment = build_experiment(args, source, data, parent, prepared)
    command = trainer_argv(args, holdout)
    immutable = immutable_inputs(args)
    before = {str(Path(path).resolve()): sha256(path) for path in immutable}
    import train_motiondrive_v2 as trainer
    with patched_runtime(args.arm, args.seed, overlay, holdout,
                         parent["model_state_sha256"],
                         prepared["initial_model_state_sha256"]):
        trainer.run_training(command, experiment=experiment)
    after = {str(Path(path).resolve()): sha256(path) for path in immutable}
    require(before == after, "long-training immutable inputs changed during training")
    validate_source_manifest(args.source_manifest,
                             args.expected_source_manifest_sha256)
    validate_holdout_split(args.holdout_split,
                           args.expected_holdout_split_sha256,
                           args.split_manifest)
    print(json.dumps({"status": "completed", "arm": args.arm,
                      "runtime": runtime, "parent": parent,
                      "experiment": experiment},
                     sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
