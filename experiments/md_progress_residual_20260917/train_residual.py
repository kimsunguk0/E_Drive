#!/usr/bin/env python3
"""Train FRONT/SIDE progress refiners from the registered MR-NATIVE-s1 model."""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
BASE_DIR = ROOT / "experiments/md_r0_reset_20260914"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_grouped_split_v2 import sha256
from motiondrive_v2_training import tensor_state_sha256
from motiondrive_v2_training import (HISTORY_SCALE, STATE_SCALE, masked_mean,
                                     regression_loss, weighted_d3)
import matching_resolution as mr
from progress_residual import (ResidualMotionDriveV2, SIDE_HISTORY_KEY, SIDE_OFFSETS,
                               SideHistoryDataset, wrap_flip_item)


SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json"
SUPERVISION = ROOT / "data/etri/motiondrive_v2/r0reset_tplus_geometry_v2"
MR_INIT = ROOT / "work_dirs/md_r0_reset_20260914/MR-NATIVE-s1/last.pth"
REPORT_DIR = ROOT / "reports/md_progress_residual_20260917"
TRAIN_ROWS = 83_700
TUNE_ROWS = 1_998
BATCH = 16
MICROBATCH = 8
EVAL_BATCH = 8
WORKERS = 8
NEW_MODULE_LR = 5e-5
LENGTH_LAMBDA = .25
CAP = 1.0
SIDE_AUX_LAMBDA = .5
DENOISE_LAMBDA = .5
NEW_MODULE_SEED = 20260917
WARMUP_STEPS = 1_000
JOINT_STEPS = math.ceil(2 * TRAIN_ROWS / BATCH)  # 10,463, two complete exposures
JOINT_EVAL_EVERY = math.ceil(.5 * TRAIN_ROWS / BATCH)
CUDA_MEMORY_LIMIT_MIB = 170_000

ARMS = {
    "FRONT-S": {"side": False, "coefficients": 1, "side_aux": False, "denoise": False},
    "SIDE-S": {"side": True, "coefficients": 1, "side_aux": False, "denoise": False},
    "SIDE-VA": {"side": True, "coefficients": 2, "side_aux": False, "denoise": False},
    "SIDE-S-AUX": {"side": True, "coefficients": 1, "side_aux": True, "denoise": False},
    "SIDE-VA-AUX": {"side": True, "coefficients": 2, "side_aux": True, "denoise": False},
    "SIDE-S-AUX-DN": {"side": True, "coefficients": 1, "side_aux": True, "denoise": True},
    "SIDE-VA-AUX-DN": {"side": True, "coefficients": 2, "side_aux": True, "denoise": True},
}
STAGES = {
    "warmup": {"steps": WARMUP_STEPS, "eval_every": WARMUP_STEPS,
               "backbone_lr": 0., "base_head_lr": 0.},
    "joint": {"steps": JOINT_STEPS, "eval_every": JOINT_EVAL_EVERY,
              "backbone_lr": 5e-7, "base_head_lr": 5e-6},
}


def rows_sha(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def raw_datasets(seed):
    from motiondrive_v2_data import MotionDriveDataset
    common = dict(data_root="/tmp/pm97", split_manifest=str(SPLIT),
                  supervision_root=str(SUPERVISION), min_frame=30,
                  max_samples=0, seed=seed, history_contract="control")
    train = MotionDriveDataset(split="train", frame_stride=1, augment=False, **common)
    tune = MotionDriveDataset(split="tune", frame_stride=5, augment=False, **common)
    if len(train) != TRAIN_ROWS or len(tune) != TUNE_ROWS:
        raise ValueError(f"unexpected dataset rows: {len(train)}/{len(tune)}")
    return train, tune


def experiment(arm, stage, seed, initializer, train, tune):
    spec, schedule = ARMS[arm], STAGES[stage]
    payload = torch.load(initializer, map_location="cpu", weights_only=False)
    return {
        "schema_version": 1,
        "name": "md_progress_residual_20260917",
        "arm": arm,
        "stage": stage,
        "seed": seed,
        "question": ("does short side temporal observation improve a base-plan-conditioned "
                     "neural progress correction under the official PREFIX loss?"),
        "initializer": {
            "path": str(initializer), "checkpoint_sha256": sha256(initializer),
            "model_state_sha256": tensor_state_sha256(payload["model"]),
            "weights_only": True,
        },
        "split_manifest": {"path": str(SPLIT), "sha256": sha256(SPLIT)},
        "supervision": {"path": str(SUPERVISION),
                        "sha256": sha256(SUPERVISION / "supervision_manifest.json")},
        "train_data": {"rows": len(train), "rows_sha256": rows_sha(train.rows),
                       "scenes": len(set(map(str, train.scene_names[train.rows])))},
        "tune_data": {"rows": len(tune), "rows_sha256": rows_sha(tune.rows),
                      "scenes": len(set(map(str, tune.scene_names[tune.rows])))},
        "expected_initial_model_state_sha256": "measured-after-strict-load",
        "expected_optimizer_groups": [
            {"name": "backbone", "base_lr": schedule["backbone_lr"]},
            {"name": "head", "base_lr": schedule["base_head_lr"]},
        ],
        "optimizer_groups": [
            {"name": "base_backbone", "base_lr": schedule["backbone_lr"]},
            {"name": "base_nonbackbone", "base_lr": schedule["base_head_lr"]},
            {"name": "new_progress_and_side", "base_lr": NEW_MODULE_LR},
        ],
        "recipe": {
            "steps": schedule["steps"], "eval_every": schedule["eval_every"],
            "batch": BATCH, "microbatch": MICROBATCH, "eval_batch": EVAL_BATCH,
            "workers": WORKERS, "warmup": 200, "weight_decay": .01,
            "grad_clip": 5., "precision": "bf16", "bn_policy": "fixed",
            "time_input": "nominal", "length_auxiliary_lambda": LENGTH_LAMBDA,
            "cuda_memory_limit_mib": CUDA_MEMORY_LIMIT_MIB,
            "terminal_fixed_before_results": True,
        },
        "architecture": {
            "registered_base": "MR-NATIVE-s1 step20554",
            "front_history_seconds": [.1, .2, .5, 1.],
            "side_enabled": spec["side"],
            "side_absolute_auxiliary": spec["side_aux"],
            "side_absolute_auxiliary_lambda": SIDE_AUX_LAMBDA if spec["side_aux"] else 0.,
            "progress_plan_denoising": spec["denoise"],
            "progress_plan_denoising_lambda": DENOISE_LAMBDA if spec["denoise"] else 0.,
            "progress_plan_noise_cap": ([.5] if spec["coefficients"] == 1 else [.5, .2])
                                       if spec["denoise"] else [],
            "side_cameras": (["camera_front_right", "camera_front_left"]
                             if spec["side"] else []),
            "side_history_seconds": ([.1, .2, .5] if spec["side"] else []),
            "coefficient_count": spec["coefficients"],
            "coefficient_semantics": (["delta_v"] if spec["coefficients"] == 1
                                      else ["delta_v", "delta_a"]),
            "coefficient_cap_abs": CAP,
            "base_plan_input": "stop-gradient predicted XY plus predicted interval lengths",
            "geometry": ("detached nearest-valid predicted tangent; all-zero fallback ego +x; "
                         "nonnegative corrected interval lengths; direct base-plan gradient retained"),
            "raw_goal_pose_provided_status_to_residual": False,
            "goal_dependency": "indirect through detached goal-conditioned base plan",
            "side_motion_dynamic_pose_input": False,
            "side_calibration": "current fixed lidar2img only",
            "final_inside_neural_forward": True,
            "new_module_seed": NEW_MODULE_SEED,
            "zero_initialised": ["progress_refiner.output",
                                 "side_state_delta", "side_history_delta"] if spec["side"]
                                else ["progress_refiner.output"],
        },
        "comparison_contract": {
            "same_initializer": True, "same_rows": True, "same_sample_seed": True,
            "same_schedule": True, "same_final_prefix_loss": True,
            "front_vs_side_scalar": ["FRONT-S", "SIDE-S"],
            "side_scalar_vs_two_coefficient": ["SIDE-S", "SIDE-VA"],
            "side_aux_scalar_vs_two_coefficient": ["SIDE-S-AUX", "SIDE-VA-AUX"],
            "side_denoise_scalar_vs_two_coefficient": ["SIDE-S-AUX-DN", "SIDE-VA-AUX-DN"],
        },
        "provided_status_used": False,
        "checkpoint_selection": "report every fixed endpoint; primary comparison at common terminal",
    }


def trainer_argv(arm, stage, seed, initializer, run_dir):
    schedule = STAGES[stage]
    return [
        "--data-root", "/tmp/pm97", "--split-manifest", str(SPLIT),
        "--supervision-root", str(SUPERVISION), "--run-dir", str(run_dir),
        "--phase", "joint", "--goal-on", "1", "--state-on", "1",
        "--cross-cell-goal-mode", "zero", "--history-contract", "control",
        "--gpu", "0", "--seed", str(seed), "--steps", str(schedule["steps"]),
        "--batch", str(BATCH), "--microbatch", str(MICROBATCH),
        "--eval-batch", str(EVAL_BATCH), "--workers", str(WORKERS),
        "--eval-every", str(schedule["eval_every"]),
        "--save-every", str(schedule["eval_every"]), "--log-every", "25",
        "--lr", str(schedule["base_head_lr"]),
        "--backbone-lr", str(schedule["backbone_lr"]),
        "--weight-decay", ".01", "--warmup", "200", "--grad-clip", "5",
        "--alpha-occ", ".2", "--alpha-lane", ".2", "--alpha-motion", ".2",
        "--uncertainty", "1", "--precision", "bf16", "--time-input", "nominal",
        "--bn-policy", "fixed", "--init", str(initializer), "--arch", "resnet50",
        "--motion-input-mode", "low_feature", "--train-stride", "1",
        "--eval-stride", "5", "--max-train-samples", "0", "--max-eval-samples", "0",
        "--eval-split", "tune", "--cuda-memory-limit-mib", str(CUDA_MEMORY_LIMIT_MIB),
        "--cuda-min-free-mib", "8192",
    ]


def wrap_side_auxiliary_loss(original, lam):
    """Supervise side-only motion features even when the base fits train rows."""
    if lam <= 0:
        raise ValueError("side auxiliary weight must be positive")

    def compute_loss(outputs, batch, weights, *, normalizers=None, stop_class_weights=None):
        total, parts = original(outputs, batch, weights, normalizers=normalizers,
                                stop_class_weights=stop_class_weights)
        required = ("side_state_aux_hat", "side_history_aux_hat")
        if any(name not in outputs for name in required):
            raise KeyError("side auxiliary arm did not emit its direct motion predictions")
        state_valid = batch.get("state_valid")
        history_valid = batch.get("history_valid")
        state = regression_loss(
            outputs["side_state_aux_hat"][..., :5], batch["state_target"][..., :5],
            None if state_valid is None else state_valid[..., :5], STATE_SCALE,
            normalizer=None if normalizers is None else normalizers["state_valid"])
        # The denominator deliberately remains the full four-offset count.  This
        # makes microbatch contributions additive and gives the three-offset side
        # auxiliary a fixed 3/4 weighting without changing trainer normalizers.
        history = regression_loss(
            outputs["side_history_aux_hat"], batch["history_target"][:, :len(SIDE_OFFSETS)],
            None if history_valid is None else history_valid[:, :len(SIDE_OFFSETS)],
            HISTORY_SCALE,
            normalizer=None if normalizers is None else normalizers["history_valid"])
        auxiliary = state + history
        total = total + lam * auxiliary
        parts = dict(parts)
        parts.update(side_state_aux=state, side_history_aux=history,
                     side_motion_aux=auxiliary, total=total)
        return total, parts

    return compute_loss


def wrap_progress_denoise_loss(original, lam):
    """Teach the shared refiner to undo synthetic base-plan progress errors."""
    if lam <= 0:
        raise ValueError("progress denoise weight must be positive")

    def compute_loss(outputs, batch, weights, *, normalizers=None, stop_class_weights=None):
        total, parts = original(outputs, batch, weights, normalizers=normalizers,
                                stop_class_weights=stop_class_weights)
        if "progress_denoise_plan" not in outputs or "plan_base_abs" not in outputs:
            raise KeyError("denoise arm did not emit its reconstructed and base plans")
        valid = batch.get("plan_valid", torch.ones_like(
            outputs["progress_denoise_plan"][..., 0], dtype=torch.bool)).bool().all(-1)
        error = weighted_d3(outputs["progress_denoise_plan"],
                            outputs["plan_base_abs"].detach())
        denoise = masked_mean(
            error, valid,
            normalizer=None if normalizers is None else normalizers["plan_complete"])
        total = total + lam * denoise
        parts = dict(parts)
        parts.update(progress_denoise=denoise, total=total)
        return total, parts

    return compute_loss


@contextlib.contextmanager
def patched_runtime(arm, stage, seed, declared, run_dir):
    import models.motiondrive_v2 as model_api
    import models.motiondrive_v2.model as model_module
    import motiondrive_v2_data as data_api
    import motiondrive_v2_flip_augment as flip_api
    import evaluate_motiondrive_v2_planning as eval_api
    import train_motiondrive_v2 as trainer
    from motiondrive_v2_flip_augment import FlipAugmented
    from length_auxiliary import wrap_compute_loss

    spec, schedule = ARMS[arm], STAGES[stage]
    originals = {
        "dataset": data_api.MotionDriveDataset,
        "flip": flip_api.flip_item,
        "train_inputs": trainer.model_inputs,
        "eval_inputs": eval_api.planning_model_inputs,
        "model_api": model_api.MotionDriveV2,
        "model_module": model_module.MotionDriveV2,
        "loader": trainer._load_initial_model_state,
        "protocol": trainer._validate_experimental_protocol,
        "runtime": trainer._validate_experimental_runtime,
        "schedule": trainer._training_schedule_actions,
        "checkpoint": trainer.atomic_checkpoint,
        "json": trainer.atomic_json,
        "loss": trainer.compute_loss,
        "adamw": torch.optim.AdamW,
    }
    state = {"model": None, "new_parameter_ids": set()}

    def dataset_factory(**kwargs):
        split = kwargs.get("split")
        base_dataset = originals["dataset"](**kwargs)
        wrapped = mr.MotionCanvasDataset(base_dataset, "native")
        if spec["side"]:
            wrapped = SideHistoryDataset(wrapped)
        if split == "train":
            if kwargs.get("augment") is not True:
                raise ValueError("residual training must preserve augmentation")
            wrapped = FlipAugmented(wrapped, 768, 384, p=.5, seed=seed)
        elif split != "tune" or kwargs.get("augment"):
            raise ValueError("residual lineage uses train/tune only")
        return wrapped

    def train_inputs(batch, **kwargs):
        inputs = mr.model_inputs_with_canvas(originals["train_inputs"], batch, **kwargs)
        if spec["side"]:
            inputs[SIDE_HISTORY_KEY] = batch[SIDE_HISTORY_KEY]
        return inputs

    def eval_inputs(batch, **kwargs):
        inputs = mr.model_inputs_with_canvas(originals["eval_inputs"], batch, **kwargs)
        if spec["side"]:
            inputs[SIDE_HISTORY_KEY] = batch[SIDE_HISTORY_KEY]
        return inputs

    def model_factory(config=None):
        if config is None:
            raise ValueError("residual model requires checkpoint-restored config")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(NEW_MODULE_SEED)
            built = ResidualMotionDriveV2(
                config, coefficient_count=spec["coefficients"],
                side_enabled=spec["side"], side_auxiliary=spec["side_aux"],
                progress_denoise=spec["denoise"], cap=CAP)
        prefixes = ("progress_refiner.", "side_motion_encoder.", "side_calibration.",
                    "side_state_delta.", "side_history_delta.",
                    "side_state_aux.", "side_history_aux.")
        state["model"] = built
        state["new_parameter_ids"] = {
            id(parameter) for name, parameter in built.named_parameters()
            if name.startswith(prefixes)}
        return built

    def load_initial(model, common, experiment=None):
        expected_sha = declared["initializer"]["model_state_sha256"]
        if tensor_state_sha256(common["model"]) != expected_sha:
            raise ValueError("initializer model tensors differ from the pinned protocol")
        if stage == "warmup":
            incompatible = model.load_state_dict(common["model"], strict=False)
            expected_missing = sorted(set(model.state_dict()).difference(common["model"]))
            if sorted(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
                raise ValueError("MR warmup load did not miss exactly the new residual modules")
            loaded = {name: value for name, value in model.state_dict().items()
                      if name in common["model"]}
            if tensor_state_sha256(loaded) != tensor_state_sha256(common["model"]):
                raise ValueError("existing MR tensors changed during residual installation")
        else:
            model.load_state_dict(common["model"], strict=True)
            expected_missing = []
        if bool(model.progress_refiner.output.weight.count_nonzero()) \
                or bool(model.progress_refiner.output.bias.count_nonzero()):
            if stage == "warmup":
                raise ValueError("new progress output must begin exactly at zero")
        measured = tensor_state_sha256(model.state_dict())
        declared["expected_initial_model_state_sha256"] = measured
        declared["initial_load"] = {
            "strict_existing_keys": True, "missing_new_keys": expected_missing,
            "measured_model_state_sha256": measured,
            "base_frozen_by_zero_lr": stage == "warmup",
        }
        return {"progress_residual_load": declared["initial_load"]}

    def adamw_factory(groups, **kwargs):
        groups = list(groups)
        if len(groups) != 2 or state["model"] is None:
            raise ValueError("unexpected base optimizer groups")
        backbone = list(groups[0]["params"])
        other = list(groups[1]["params"])
        new_ids = state["new_parameter_ids"]
        new = [parameter for parameter in other if id(parameter) in new_ids]
        base_other = [parameter for parameter in other if id(parameter) not in new_ids]
        if not new or len(new) + len(base_other) != len(other):
            raise ValueError("new-module optimizer partition failed")
        custom = [
            {"params": backbone, "lr": schedule["backbone_lr"],
             "base_lr": schedule["backbone_lr"]},
            {"params": base_other, "lr": schedule["base_head_lr"],
             "base_lr": schedule["base_head_lr"]},
            {"params": new, "lr": NEW_MODULE_LR, "base_lr": NEW_MODULE_LR},
        ]
        return originals["adamw"](custom, **kwargs)

    def validate_protocol(value):
        if value is not declared or value.get("name") != "md_progress_residual_20260917" \
                or value.get("arm") != arm or value.get("stage") != stage \
                or value.get("provided_status_used") is not False:
            raise ValueError("progress residual protocol mismatch")
        return None

    def validate_runtime(args, value):
        expected = {
            "steps": schedule["steps"], "batch": BATCH, "microbatch": MICROBATCH,
            "eval_batch": EVAL_BATCH, "workers": WORKERS,
            "eval_every": schedule["eval_every"], "save_every": schedule["eval_every"],
            "lr": schedule["base_head_lr"], "backbone_lr": schedule["backbone_lr"],
            "weight_decay": .01, "warmup": 200, "grad_clip": 5.,
            "alpha_occ": .2, "alpha_lane": .2, "alpha_motion": .2,
            "precision": "bf16", "time_input": "nominal", "bn_policy": "fixed",
            "arch": "resnet50", "motion_input_mode": "low_feature",
            "cross_cell_goal_mode": "zero", "history_contract": "control",
            "train_stride": 1, "eval_stride": 5, "max_train_samples": 0,
            "max_eval_samples": 0, "eval_split": "tune", "seed": seed,
        }
        bad = {key: (getattr(args, key), value) for key, value in expected.items()
               if getattr(args, key) != value}
        if bad or args.resume or args.pretrained or args.eval_only or args.cpu \
                or args.train_scenes is not None or args.eval_scenes is not None:
            raise ValueError(f"progress residual runtime mismatch: {bad}")

    def schedule_actions(step, args, value):
        due = step == args.steps or step % schedule["eval_every"] == 0
        return due, due

    def checkpoint(path, payload):
        originals["checkpoint"](path, payload)
        if Path(path).name == "last.pth":
            light = {key: payload[key] for key in
                     ("model", "manifest", "step", "epoch", "best_metric")}
            originals["checkpoint"](Path(run_dir) / f"ckpt_step{payload['step']}.pth", light)

    def atomic_json(path, payload):
        originals["json"](path, payload)
        if Path(path).name == "latest_eval.json" and "step" in payload:
            originals["json"](Path(run_dir) / f"eval_step{payload['step']}.json", payload)

    flip_api.flip_item = wrap_flip_item(mr.wrap_flip_item(originals["flip"]))
    data_api.MotionDriveDataset = dataset_factory
    trainer.model_inputs = train_inputs
    eval_api.planning_model_inputs = eval_inputs
    model_api.MotionDriveV2 = model_factory
    model_module.MotionDriveV2 = model_factory
    trainer._load_initial_model_state = load_initial
    trainer._validate_experimental_protocol = validate_protocol
    trainer._validate_experimental_runtime = validate_runtime
    trainer._training_schedule_actions = schedule_actions
    trainer.atomic_checkpoint = checkpoint
    trainer.atomic_json = atomic_json
    loss = wrap_compute_loss(originals["loss"], LENGTH_LAMBDA)
    if spec["side_aux"]:
        loss = wrap_side_auxiliary_loss(loss, SIDE_AUX_LAMBDA)
    if spec["denoise"]:
        loss = wrap_progress_denoise_loss(loss, DENOISE_LAMBDA)
    trainer.compute_loss = loss
    torch.optim.AdamW = adamw_factory
    try:
        yield
    finally:
        torch.optim.AdamW = originals["adamw"]
        trainer.compute_loss = originals["loss"]
        trainer.atomic_json = originals["json"]
        trainer.atomic_checkpoint = originals["checkpoint"]
        trainer._training_schedule_actions = originals["schedule"]
        trainer._validate_experimental_runtime = originals["runtime"]
        trainer._validate_experimental_protocol = originals["protocol"]
        trainer._load_initial_model_state = originals["loader"]
        model_module.MotionDriveV2 = originals["model_module"]
        model_api.MotionDriveV2 = originals["model_api"]
        eval_api.planning_model_inputs = originals["eval_inputs"]
        trainer.model_inputs = originals["train_inputs"]
        data_api.MotionDriveDataset = originals["dataset"]
        flip_api.flip_item = originals["flip"]


def run_stage(arm, stage, seed, initializer, run_dir, dry_run=False):
    train, tune = raw_datasets(seed)
    declared = experiment(arm, stage, seed, initializer, train, tune)
    command = trainer_argv(arm, stage, seed, initializer, run_dir)
    summary = {"arm": arm, "stage": stage, "seed": seed,
               "initializer": str(initializer), "run_dir": str(run_dir),
               "train_rows": len(train), "tune_rows": len(tune),
               "steps": STAGES[stage]["steps"], "microbatch": MICROBATCH,
               "side": ARMS[arm]["side"], "coefficients": ARMS[arm]["coefficients"]}
    print("PLAN " + json.dumps(summary, sort_keys=True), flush=True)
    if dry_run:
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    protocol_path = REPORT_DIR / f"protocol_{arm}_{stage}.json"
    protocol_path.write_text(json.dumps(declared, indent=1, sort_keys=True) + "\n")
    import train_motiondrive_v2 as trainer
    with patched_runtime(arm, stage, seed, declared, run_dir):
        trainer.run_training(command, experiment=declared)
    Path(run_dir, "experiment.json").write_text(
        json.dumps(declared, indent=1, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--gpu", type=int, choices=range(4), required=True,
                        help="recorded physical GPU; CUDA_VISIBLE_DEVICES must expose only it")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--root-run-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--joint-only", action="store_true",
                        help="resume the fixed joint stage from a completed warmup/last.pth")
    args = parser.parse_args()
    root = Path(args.root_run_dir).resolve()
    warmup = root / "warmup"
    joint = root / "joint"
    if not args.joint_only:
        run_stage(args.arm, "warmup", args.seed, MR_INIT, warmup, args.dry_run)
    if args.dry_run and not args.joint_only:
        print("PLAN " + json.dumps({
            "arm": args.arm, "stage": "joint", "seed": args.seed,
            "initializer": str(warmup / "last.pth"), "run_dir": str(joint),
            "steps": JOINT_STEPS, "microbatch": MICROBATCH,
            "side": ARMS[args.arm]["side"],
            "coefficients": ARMS[args.arm]["coefficients"],
        }, sort_keys=True), flush=True)
        return
    if not (warmup / "last.pth").exists():
        raise FileNotFoundError("warmup did not produce last.pth")
    # The next stage runs in the same process.  Release the warmup allocator
    # reservation before the trainer performs its fresh-process headroom gate.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_stage(args.arm, "joint", args.seed, warmup / "last.pth", joint, args.dry_run)
    if args.dry_run:
        return
    (root / "completed.json").write_text(json.dumps({
        "arm": args.arm, "seed": args.seed, "physical_gpu": args.gpu,
        "warmup": str(warmup), "joint": str(joint),
    }, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
