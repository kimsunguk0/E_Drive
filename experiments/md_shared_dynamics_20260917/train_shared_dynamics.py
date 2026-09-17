#!/usr/bin/env python3
"""Train fresh MR shared-status and factorized-progress models on train310."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
BASE_DIR = ROOT / "experiments/md_r0_reset_20260914"
PROGRESS_DIR = ROOT / "experiments/md_progress_residual_20260917"
HERE = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "scripts", BASE_DIR, PROGRESS_DIR, HERE):
    sys.path.insert(0, str(path))

from build_grouped_split_v2 import sha256
from factorized_model import SharedDynamicsMotionDriveV2
from motiondrive_v2_training import tensor_state_sha256
import matching_resolution as mr


SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json"
SUPERVISION = ROOT / "data/etri/motiondrive_v2/r0reset_tplus_geometry_v2"
INITIALIZER = ROOT / "work_dirs/md_r0_reset_20260914/r0_init_tplus.pth"
REPORT_DIR = ROOT / "reports/md_shared_dynamics_20260917"
TRAIN_ROWS = 83_700
TUNE_ROWS = 1_998
UPDATES = 20_554
EVAL_EVERY = 3_426
BATCH = 16
MICROBATCH = 8
EVAL_BATCH = 8
WORKERS = 8
HEAD_LR = 5e-5
BACKBONE_LR = 5e-6
LENGTH_LAMBDA = 0.25
CUDA_MEMORY_LIMIT_MIB = 170_000
ARMS = ("A2-DIRECT", "A3-FP-S", "A3-FP-VA")
PROVIDED_STATUS_KEY = "provided_status5"


def rows_sha(rows) -> str:
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


class CausalStatusDataset(torch.utils.data.Dataset):
    """Expose the pose-derived causal fit as an explicit inference input."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def set_epoch(self, epoch):
        return self.base.set_epoch(epoch)

    def __getitem__(self, index):
        item = self.base[index]
        state, valid = item.get("state_target"), item.get("state_valid")
        if not isinstance(state, torch.Tensor) or state.shape != (6,):
            raise ValueError("causal status source must be state_target [6]")
        if not isinstance(valid, torch.Tensor) or valid.shape != (6,) \
                or not bool(valid[:5].all()):
            raise ValueError("causal status requires five valid fields")
        item[PROVIDED_STATUS_KEY] = state[:5].clone()
        return item


def raw_datasets(seed: int):
    from motiondrive_v2_data import MotionDriveDataset
    common = dict(data_root="/tmp/pm97", split_manifest=str(SPLIT),
                  supervision_root=str(SUPERVISION), min_frame=30,
                  max_samples=0, seed=seed, history_contract="control")
    train = MotionDriveDataset(split="train", frame_stride=1, augment=False, **common)
    tune = MotionDriveDataset(split="tune", frame_stride=5, augment=False, **common)
    if len(train) != TRAIN_ROWS or len(tune) != TUNE_ROWS:
        raise ValueError(f"unexpected train/tune rows: {len(train)}/{len(tune)}")
    return train, tune


def build_experiment(arm: str, seed: int, train, tune, smoke: bool):
    payload = torch.load(INITIALIZER, map_location="cpu", weights_only=False)
    coefficient_count = SharedDynamicsMotionDriveV2.VALID_ARMS[arm]
    return {
        "schema_version": 1,
        "name": "md_shared_dynamics_factorized_20260917",
        "arm": arm,
        "seed": seed,
        "smoke": bool(smoke),
        "question": ("can causal status improve the shared image representation when it is "
                     "optimized with the registered MR graph from the start, and does forcing "
                     "the final planner to separate path direction from progress improve V0?"),
        "initializer": {
            "path": str(INITIALIZER),
            "checkpoint_sha256": sha256(INITIALIZER),
            "model_state_sha256": tensor_state_sha256(payload["model"]),
            "source": "q10_flip50 initializer used by registered MR",
        },
        "split_manifest": {"path": str(SPLIT), "sha256": sha256(SPLIT)},
        "supervision": {
            "path": str(SUPERVISION),
            "manifest": str(SUPERVISION / "supervision_manifest.json"),
            "sha256": sha256(SUPERVISION / "supervision_manifest.json"),
        },
        "train_data": {"rows": len(train), "rows_sha256": rows_sha(train.rows),
                       "scenes": len(set(train.scene_names[train.rows].tolist()))},
        "tune_data": {"rows": len(tune), "rows_sha256": rows_sha(tune.rows),
                      "scenes": len(set(tune.scene_names[tune.rows].tolist()))},
        "provided_status_used": True,
        "information_route": {
            "source": "causal quadratic pose fit using t-1.0s through t; no future pose",
            "fields": ["vx", "vy", "ax", "ay", "yaw_rate"],
            "raw_status_direct_planner_input": False,
            "shared_query": True,
            "shared_multiplicative_image_feature_conditioning": arm.startswith("A3-"),
            "shared_consumers": ["occupancy", "lane", "image-state/history", "planning"],
            "progress_head_inputs": ["planner decoded image evidence",
                                     "detached model proposal coordinates/lengths"],
            "progress_head_raw_goal_pose_status_inputs": False,
        },
        "planner": {
            "factorized": bool(coefficient_count),
            "coefficient_count": coefficient_count,
            "coefficient_semantics": ([] if not coefficient_count else
                                      (["delta_v"] if coefficient_count == 1
                                       else ["delta_v", "delta_a"])),
            "direction_source": "current neural proposal segments",
            "proposal_length_gradient": ("ordinary direct XY" if not coefficient_count
                                         else "stopped; proposal receives direction gradient only"),
            "composition": "neural forward cumulative direction times nonnegative progress",
            "candidate_bank_or_selector": False,
        },
        "recipe": {
            "updates": 2 if smoke else UPDATES,
            "eval_every": 2 if smoke else EVAL_EVERY,
            "batch": BATCH, "microbatch": MICROBATCH, "eval_batch": EVAL_BATCH,
            "head_lr": HEAD_LR, "backbone_lr": BACKBONE_LR,
            "warmup": 200, "weight_decay": 0.01, "grad_clip": 5.0,
            "flip_probability": 0.5, "precision": "bf16",
            "interval_length_auxiliary_lambda": LENGTH_LAMBDA,
            "correlation_radius": 4, "motion_canvas": "native 768x432",
        },
        "comparison_contract": {
            "same_initializer_as_registered_mr": True,
            "same_train310_tune37_split": True,
            "same_seed_and_sample_order_across_arms": True,
            "same_full_update_budget": not smoke,
        },
        "expected_initial_model_state_sha256": "set_by_strict_loader_before_optimizer",
        "expected_optimizer_groups": [
            {"name": "backbone", "base_lr": BACKBONE_LR},
            {"name": "head", "base_lr": HEAD_LR},
        ],
    }


def trainer_argv(seed: int, run_dir: Path, smoke: bool):
    steps, every = (2, 2) if smoke else (UPDATES, EVAL_EVERY)
    return [
        "--data-root", "/tmp/pm97", "--split-manifest", str(SPLIT),
        "--supervision-root", str(SUPERVISION), "--run-dir", str(run_dir),
        "--phase", "joint", "--goal-on", "1", "--state-on", "1",
        "--cross-cell-goal-mode", "zero", "--history-contract", "control",
        "--gpu", "0", "--seed", str(seed), "--steps", str(steps),
        "--batch", str(BATCH), "--microbatch", str(MICROBATCH),
        "--eval-batch", str(EVAL_BATCH), "--workers", str(WORKERS),
        "--eval-every", str(every), "--save-every", str(every), "--log-every", "1" if smoke else "50",
        "--lr", str(HEAD_LR), "--backbone-lr", str(BACKBONE_LR),
        "--weight-decay", ".01", "--warmup", "200", "--grad-clip", "5",
        "--alpha-occ", ".2", "--alpha-lane", ".2", "--alpha-motion", ".2",
        "--uncertainty", "1", "--precision", "bf16", "--time-input", "nominal",
        "--bn-policy", "fixed", "--init", str(INITIALIZER), "--arch", "resnet50",
        "--motion-input-mode", "low_feature", "--train-stride", "1",
        "--eval-stride", "5", "--max-train-samples", "0",
        "--max-eval-samples", "0", "--eval-split", "tune",
        "--cuda-memory-limit-mib", str(CUDA_MEMORY_LIMIT_MIB),
        "--cuda-min-free-mib", "8192",
    ]


@contextlib.contextmanager
def patched_runtime(arm: str, seed: int, declared: dict, run_dir: Path, smoke: bool):
    import models.motiondrive_v2 as model_api
    import models.motiondrive_v2.model as model_module
    import motiondrive_v2_data as data_api
    import motiondrive_v2_flip_augment as flip_api
    import evaluate_motiondrive_v2_planning as eval_api
    import train_motiondrive_v2 as trainer
    from motiondrive_v2_flip_augment import FlipAugmented
    from length_auxiliary import wrap_compute_loss

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
    }
    construction: dict[str, object] = {}

    def dataset_factory(**kwargs):
        split = kwargs.get("split")
        base = originals["dataset"](**kwargs)
        wrapped = CausalStatusDataset(mr.MotionCanvasDataset(base, "native"))
        if split == "train":
            if kwargs.get("augment") is not True:
                raise ValueError("shared-dynamics train dataset must preserve augmentation")
            wrapped = FlipAugmented(wrapped, 768, 384, p=0.5, seed=seed)
        elif split != "tune" or kwargs.get("augment"):
            raise ValueError("shared-dynamics lineage permits train/tune only")
        return wrapped

    def add_inputs(original, batch, **kwargs):
        inputs = mr.model_inputs_with_canvas(original, batch, **kwargs)
        status = batch.get(PROVIDED_STATUS_KEY)
        if not isinstance(status, torch.Tensor) or status.ndim != 2 or status.shape[1] != 5:
            raise ValueError("provided_status5 batch input is missing")
        inputs[PROVIDED_STATUS_KEY] = status
        return inputs

    def train_inputs(batch, **kwargs):
        return add_inputs(originals["train_inputs"], batch, **kwargs)

    def eval_inputs(batch, **kwargs):
        return add_inputs(originals["eval_inputs"], batch, **kwargs)

    def model_factory(config=None):
        if config is None:
            raise ValueError("shared-dynamics model requires an explicit config")
        model = SharedDynamicsMotionDriveV2(config, arm=arm)
        rebuild = mr.rebuild_correlation_fuse(model, mr.NEW_RADIUS)
        construction["model"] = model
        construction["rebuild"] = rebuild
        return model

    def load_initial(model, common, experiment=None):
        if tensor_state_sha256(common["model"]) != declared["initializer"]["model_state_sha256"]:
            raise ValueError("initializer state differs from the pinned protocol")
        state = dict(common["model"])
        dropped = sorted(name for name in state
                         if name.startswith("motion_encoder.correlation_fuse.0."))
        for name in dropped:
            state.pop(name)
        incompatible = model.load_state_dict(state, strict=False)
        expected_missing = sorted(set(model.state_dict()).difference(state))
        if sorted(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise ValueError("fresh shared-dynamics load has an unexpected key mismatch")
        loaded = {name: model.state_dict()[name] for name in state}
        if tensor_state_sha256(loaded) != tensor_state_sha256(state):
            raise ValueError("fresh shared-dynamics load changed a retained parent tensor")
        query_final = model.shared_status_query_fusion.status_mlp[-1]
        if bool(query_final.weight.count_nonzero()) or bool(query_final.bias.count_nonzero()):
            raise ValueError("shared status query must start at exact identity")
        if model.shared_status_feature_conditioner is not None:
            final = model.shared_status_feature_conditioner.network[-1]
            if bool(final.weight.count_nonzero()) or bool(final.bias.count_nonzero()):
                raise ValueError("shared feature conditioner must start at exact identity")
        if model.progress_head is not None:
            final = model.progress_head.output[-1]
            if bool(final.weight.count_nonzero()) or bool(final.bias.count_nonzero()):
                raise ValueError("continuous progress output must start at exact zero")
        measured = tensor_state_sha256(model.state_dict())
        declared["expected_initial_model_state_sha256"] = measured
        declared["initial_load"] = {
            "strict_parent_except_mr_fuse": True,
            "dropped_parent_keys": dropped,
            "missing_new_or_rebuilt_keys": expected_missing,
            "measured_model_state_sha256": measured,
            "mr_rebuild": construction["rebuild"],
        }
        return {"shared_dynamics_initial_load": declared["initial_load"]}

    def validate_protocol(value):
        if value is not declared or value.get("name") != declared["name"] \
                or value.get("arm") != arm or value.get("provided_status_used") is not True:
            raise ValueError("shared-dynamics experimental protocol mismatch")

    def validate_runtime(args, value):
        steps, every = (2, 2) if smoke else (UPDATES, EVAL_EVERY)
        expected = {
            "steps": steps, "batch": BATCH, "microbatch": MICROBATCH,
            "eval_batch": EVAL_BATCH, "workers": WORKERS,
            "eval_every": every, "save_every": every,
            "lr": HEAD_LR, "backbone_lr": BACKBONE_LR, "weight_decay": 0.01,
            "warmup": 200, "grad_clip": 5.0, "alpha_occ": 0.2,
            "alpha_lane": 0.2, "alpha_motion": 0.2, "uncertainty": 1,
            "precision": "bf16", "time_input": "nominal", "bn_policy": "fixed",
            "arch": "resnet50", "motion_input_mode": "low_feature",
            "cross_cell_goal_mode": "zero", "history_contract": "control",
            "train_stride": 1, "eval_stride": 5,
            "max_train_samples": 0,
            "max_eval_samples": 0,
            "eval_split": "tune", "seed": seed,
        }
        bad = {key: (getattr(args, key), wanted) for key, wanted in expected.items()
               if getattr(args, key) != wanted}
        if bad or args.resume or args.pretrained or args.eval_only or args.cpu \
                or args.train_scenes is not None or args.eval_scenes is not None:
            raise ValueError(f"shared-dynamics runtime mismatch: {bad}")

    def schedule_actions(step, args, value):
        due = step == args.steps or step % args.eval_every == 0
        return due, due

    def checkpoint(path, payload):
        originals["checkpoint"](path, payload)
        if Path(path).name == "last.pth":
            light = {key: payload[key] for key in
                     ("model", "manifest", "step", "epoch", "best_metric")}
            originals["checkpoint"](run_dir / f"ckpt_step{payload['step']}.pth", light)

    def atomic_json(path, payload):
        originals["json"](path, payload)
        if Path(path).name == "latest_eval.json" and "step" in payload:
            originals["json"](run_dir / f"eval_step{payload['step']}.json", payload)

    flip_api.flip_item = mr.wrap_flip_item(originals["flip"])
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
    trainer.compute_loss = wrap_compute_loss(originals["loss"], LENGTH_LAMBDA)
    try:
        yield
    finally:
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


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--gpu", type=int, choices=range(4), required=True,
                        help="physical GPU recorded in the protocol; expose it as CUDA device 0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    train, tune = raw_datasets(args.seed)
    declared = build_experiment(args.arm, args.seed, train, tune, args.smoke)
    summary = {
        "arm": args.arm, "seed": args.seed, "physical_gpu": args.gpu,
        "run_dir": str(run_dir), "train_rows": len(train), "tune_rows": len(tune),
        "steps": declared["recipe"]["updates"], "microbatch": MICROBATCH,
        "provided_status_route": declared["information_route"],
        "planner": declared["planner"],
    }
    print("PLAN " + json.dumps(summary, sort_keys=True), flush=True)
    if args.dry_run:
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = REPORT_DIR / f"protocol_{args.arm}_s{args.seed}{'_smoke' if args.smoke else ''}.json"
    protocol.write_text(json.dumps(declared, indent=1, sort_keys=True) + "\n")
    import train_motiondrive_v2 as trainer
    with patched_runtime(args.arm, args.seed, declared, run_dir, args.smoke):
        trainer.run_training(trainer_argv(args.seed, run_dir, args.smoke), experiment=declared)
    (run_dir / "experiment.json").write_text(
        json.dumps(declared, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
