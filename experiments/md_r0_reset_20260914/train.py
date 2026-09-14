#!/usr/bin/env python3
"""E1 - existing train (T203) versus expanded train (Tplus), same R0, same budget.

The only difference between the two arms is which scenes the training dataset
draws from.  Model, loss, optimizer, schedule, augmentation, seed, evaluation
set and compute budget are identical, and both arms start from the SAME
rebound R0 initializer.

Training itself is `scripts/train_motiondrive_v2.py`; this file only pins the
protocol, installs the flip augmentation on the training dataset and keeps a
checkpoint at every planned evaluation point so §7.4's best-on-V0 selection is
possible after the fact.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_grouped_split_v2 import sha256
from motiondrive_v2_training import tensor_state_sha256

NAME = "md_r0_reset_e1_data_scale"
BASE_SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_rawtime.json"
NEW_SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json"
SUPERVISION = ROOT / "data/etri/motiondrive_v2/r0reset_tplus_geometry_v2"
INIT = ROOT / "work_dirs/md_r0_reset_20260914/r0_init_tplus.pth"
REGISTRY = ROOT / "reports/md_r0_reset_20260914/baseline_registry.json"

ARMS = ("E1-T203", "E1-EXP", "E1-EXP-LONG")
EXPANDED_ARMS = ("E1-EXP", "E1-EXP-LONG")
N0_ROWS = 54810                                   # existing train split
TPLUS_ROWS = 83700                                # expanded train split
BATCH = 16
MICROBATCH = 2
# E1 held compute fixed at <= 6 exposures of the EXISTING train split.
# The LONG arm instead gives the EXPANDED split the same 6 exposures, so its
# cosine horizon is longer from step one rather than being appended later.
ARM_UPDATES = {
    "E1-T203": math.ceil(6 * N0_ROWS / BATCH),    # 20554
    "E1-EXP": math.ceil(6 * N0_ROWS / BATCH),     # 20554
    "E1-EXP-LONG": math.ceil(6 * TPLUS_ROWS / BATCH),   # 31388
}
EVAL_EVERY = math.ceil(N0_ROWS / BATCH)           # 3426, one T203 exposure
WARMUP = 200
FLIP_P = 0.5
FLIP_WIDTHS = (768, 384)
BACKBONE_LR = 5e-6
HEAD_LR = 5e-5


def rows_sha(rows) -> str:
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def build_datasets(arm, seed):
    from motiondrive_v2_data import MotionDriveDataset
    base = json.loads(BASE_SPLIT.read_text())
    common = dict(data_root="/tmp/pm97", split_manifest=str(NEW_SPLIT),
                  supervision_root=str(SUPERVISION), min_frame=30, max_samples=0,
                  seed=seed, history_contract="control")
    scenes = None if arm in EXPANDED_ARMS else sorted(base["splits"]["train"])
    training = MotionDriveDataset(split="train", frame_stride=1, augment=False,
                                  scenes=scenes, **common)
    evaluation = MotionDriveDataset(split="tune", frame_stride=5, augment=False, **common)
    return scenes, training, evaluation


def experiment_rows(dataset):
    return len(dataset)


def build_experiment(arm, seed, scenes, training, evaluation):
    updates = ARM_UPDATES[arm]
    registry = json.loads(REGISTRY.read_text())
    payload = torch.load(INIT, map_location="cpu", weights_only=False)
    initial_sha = tensor_state_sha256(payload["model"])
    if initial_sha != registry["model_state_sha256"]:
        raise SystemExit("initializer tensors differ from the pinned R0 registry")
    return {
        "schema_version": 1, "name": NAME, "arm": arm, "seed": seed,
        "question": "does the same R0 recipe improve V0 when more same-domain sessions are added?",
        "initializer": {"path": str(INIT), "checkpoint_sha256": sha256(INIT),
                        "model_state_sha256": initial_sha,
                        "source_run": registry["run_name"],
                        "source_checkpoint_sha256": registry["checkpoint_sha256"]},
        "split_manifest": {"path": str(NEW_SPLIT), "sha256": sha256(NEW_SPLIT)},
        "supervision": {"path": str(SUPERVISION),
                        "sha256": sha256(SUPERVISION / "supervision_manifest.json")},
        "train_scenes": scenes,
        "train_data": {"rows": len(training), "rows_sha256": rows_sha(training.rows),
                       "scenes": len(set(training.scene_names[training.rows]))},
        "tune_data": {"rows": len(evaluation), "rows_sha256": rows_sha(evaluation.rows),
                      "scenes": len(set(evaluation.scene_names[evaluation.rows]))},
        "expected_initial_model_state_sha256": initial_sha,
        "expected_optimizer_groups": [{"name": "backbone", "base_lr": BACKBONE_LR},
                                      {"name": "head", "base_lr": HEAD_LR}],
        "recipe": {"updates": updates, "eval_every": EVAL_EVERY, "warmup": WARMUP,
                   "batch": BATCH, "microbatch": MICROBATCH, "eval_batch": 4,
                   "optimizer": "fresh AdamW", "backbone_lr": BACKBONE_LR,
                   "head_lr": HEAD_LR, "weight_decay": .01, "grad_clip": 5.0,
                   "decay": "cosine", "cosine_horizon_updates": updates,
                   "precision": "bf16", "time_input": "nominal",
                   "bn_running_statistics": "fixed",
                   "auxiliary_weights": {"occupancy": .2, "lane": .2, "motion": .2},
                   "flip_probability": FLIP_P, "command_input": "off",
                   "provided_status_used": False},
        "exposures_of_own_train": updates * BATCH / experiment_rows(training),
        "budget_note": (
            "E1's two arms share one update count, so the expanded arm sees each of its "
            "samples fewer times: a fixed-compute data-scale contrast, not an equal-epoch "
            "one. The LONG arm instead gives the expanded split the same six exposures, "
            "with the cosine horizon set to its own total from step one. Comparing LONG "
            "with the short expanded run is therefore a schedule-and-budget comparison, "
            "not the causal effect of exposure count alone."),
        "checkpoint_selection": "lowest V0 official_d3 among planned eval steps; ties go to the earlier step",
        "provided_status_used": False,
    }


def trainer_argv(arm, seed, run_dir, scenes, gpu):
    command = [
        "--data-root", "/tmp/pm97",
        "--split-manifest", str(NEW_SPLIT),
        "--supervision-root", str(SUPERVISION),
        "--run-dir", str(Path(run_dir).resolve()),
        "--phase", "joint", "--goal-on", "1", "--state-on", "1",
        "--cross-cell-goal-mode", "zero", "--history-contract", "control",
        "--gpu", str(gpu), "--seed", str(seed), "--steps", str(ARM_UPDATES[arm]),
        "--batch", str(BATCH), "--microbatch", str(MICROBATCH), "--eval-batch", "4",
        "--workers", "4", "--eval-every", str(EVAL_EVERY),
        "--save-every", str(EVAL_EVERY), "--log-every", "50",
        "--lr", str(HEAD_LR), "--backbone-lr", str(BACKBONE_LR),
        "--weight-decay", "0.01", "--warmup", str(WARMUP), "--grad-clip", "5",
        "--alpha-occ", "0.2", "--alpha-lane", "0.2", "--alpha-motion", "0.2",
        "--uncertainty", "1", "--precision", "bf16", "--time-input", "nominal",
        "--bn-policy", "fixed", "--init", str(INIT),
        "--arch", "resnet50", "--motion-input-mode", "low_feature",
        "--train-stride", "1", "--eval-stride", "5",
        "--max-train-samples", "0", "--max-eval-samples", "0",
        "--eval-split", "tune",
        "--cuda-memory-limit-mib", "12000", "--cuda-min-free-mib", "8192",
    ]
    if scenes is not None:
        command.extend(("--train-scenes", *scenes))
    return command


@contextlib.contextmanager
def patched_runtime(arm, seed, scenes, experiment, run_dir):
    import models.motiondrive_v2 as model_api
    import motiondrive_v2_data as data_api
    import train_motiondrive_v2 as trainer
    from motiondrive_v2_flip_augment import FlipAugmented

    run_dir = Path(run_dir).resolve()
    originals = {
        "dataset": data_api.MotionDriveDataset,
        "protocol": trainer._validate_experimental_protocol,
        "runtime": trainer._validate_experimental_runtime,
        "schedule": trainer._training_schedule_actions,
        "checkpoint": trainer.atomic_checkpoint,
        "json": trainer.atomic_json,
    }

    def dataset_factory(**kwargs):
        split, requested = kwargs.get("split"), kwargs.get("scenes")
        if split == "tune":
            if requested is not None or kwargs.get("augment"):
                raise ValueError("the tune evaluation set takes no scene filter and no augmentation")
            return originals["dataset"](**kwargs)
        if split != "train":
            raise ValueError("E1 uses only the train and tune splits")
        expected = None if arm in EXPANDED_ARMS else scenes
        if (list(requested) if requested else None) != (list(expected) if expected else None):
            raise ValueError("training dataset scenes differ from the pinned arm")
        if kwargs.get("augment") is not True:
            raise ValueError("the training dataset must keep augmentation on")
        base = originals["dataset"](**kwargs)
        # the flip stream follows the training seed, as in the original screens
        return FlipAugmented(base, *FLIP_WIDTHS, p=FLIP_P, seed=seed)

    def validate_protocol(declared):
        if not isinstance(declared, dict) or declared.get("name") != NAME \
                or declared.get("arm") != arm or declared.get("seed") != seed \
                or declared.get("provided_status_used") is not False:
            raise ValueError("E1 experimental protocol mismatch")
        return None

    def validate_runtime(runtime_args, declared):
        expected = {"phase": "joint", "goal_on": 1, "state_on": 1,
                    "steps": ARM_UPDATES[arm],
                    "batch": BATCH, "microbatch": MICROBATCH, "eval_batch": 4,
                    "eval_every": EVAL_EVERY, "save_every": EVAL_EVERY,
                    "lr": HEAD_LR, "backbone_lr": BACKBONE_LR, "weight_decay": .01,
                    "warmup": WARMUP, "grad_clip": 5., "alpha_occ": .2,
                    "alpha_lane": .2, "alpha_motion": .2, "uncertainty": 1,
                    "precision": "bf16", "time_input": "nominal", "bn_policy": "fixed",
                    "arch": "resnet50", "motion_input_mode": "low_feature",
                    "cross_cell_goal_mode": "zero", "history_contract": "control",
                    "train_stride": 1, "eval_stride": 5, "max_train_samples": 0,
                    "max_eval_samples": 0, "eval_split": "tune"}
        bad = {k: (getattr(runtime_args, k), v) for k, v in expected.items()
               if getattr(runtime_args, k) != v}
        if bad:
            raise ValueError(f"E1 recipe mismatch: {bad}")
        if (runtime_args.seed != seed or not runtime_args.init or runtime_args.resume
                or runtime_args.pretrained or runtime_args.eval_only or runtime_args.cpu):
            raise ValueError("E1 runtime flags mismatch")
        expected_scenes = None if arm in EXPANDED_ARMS else list(scenes)
        if runtime_args.train_scenes != expected_scenes or runtime_args.eval_scenes is not None:
            raise ValueError("E1 scene arguments mismatch")

    def schedule(step, runtime_args, declared):
        if (runtime_args.steps != ARM_UPDATES[arm] or runtime_args.eval_every != EVAL_EVERY
                or runtime_args.save_every != EVAL_EVERY):
            raise ValueError("E1 schedule contract mismatch")
        due = step % EVAL_EVERY == 0 or step == runtime_args.steps
        return due, due

    def atomic_checkpoint(path, data):
        originals["checkpoint"](path, data)
        # Keep a weights-only snapshot at every planned point so the
        # best-on-V0 checkpoint can be evaluated after the run.
        if Path(path).name == "last.pth":
            light = {"model": data["model"], "manifest": data["manifest"],
                     "step": data["step"], "epoch": data["epoch"],
                     "best_metric": data["best_metric"]}
            originals["checkpoint"](run_dir / f"ckpt_step{int(data['step'])}.pth", light)

    def atomic_json(path, data):
        originals["json"](path, data)
        if Path(path).name == "latest_eval.json" and isinstance(data, dict) and "step" in data:
            originals["json"](run_dir / f"eval_step{int(data['step'])}.json", data)

    data_api.MotionDriveDataset = dataset_factory
    trainer._validate_experimental_protocol = validate_protocol
    trainer._validate_experimental_runtime = validate_runtime
    trainer._training_schedule_actions = schedule
    trainer.atomic_checkpoint = atomic_checkpoint
    trainer.atomic_json = atomic_json
    try:
        yield
    finally:
        data_api.MotionDriveDataset = originals["dataset"]
        trainer._validate_experimental_protocol = originals["protocol"]
        trainer._validate_experimental_runtime = originals["runtime"]
        trainer._training_schedule_actions = originals["schedule"]
        trainer.atomic_checkpoint = originals["checkpoint"]
        trainer.atomic_json = originals["json"]


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenes, training, evaluation = build_datasets(args.arm, args.seed)
    experiment = build_experiment(args.arm, args.seed, scenes, training, evaluation)
    command = trainer_argv(args.arm, args.seed, args.run_dir, scenes, args.gpu)
    summary = {"arm": args.arm, "seed": args.seed, "gpu": args.gpu,
               "run_dir": str(Path(args.run_dir).resolve()),
               "train_rows": experiment["train_data"]["rows"],
               "train_scenes": experiment["train_data"]["scenes"],
               "train_rows_sha256": experiment["train_data"]["rows_sha256"],
               "tune_rows": experiment["tune_data"]["rows"],
               "tune_rows_sha256": experiment["tune_data"]["rows_sha256"],
               "updates": ARM_UPDATES[args.arm], "eval_every": EVAL_EVERY,
               "exposures_of_own_train": (ARM_UPDATES[args.arm] * BATCH
                                          / experiment["train_data"]["rows"])}
    print("PLAN " + json.dumps(summary), flush=True)
    if args.dry_run:
        return

    del training, evaluation
    import train_motiondrive_v2 as trainer
    run_dir = Path(args.run_dir).resolve()
    # The trainer refuses a non-empty run directory, so the protocol copy is
    # written next to the reports first and into the run directory afterwards.
    plan_path = ROOT / "reports/md_r0_reset_20260914" / f"experiment_{args.arm}.json"
    plan_path.write_text(json.dumps(experiment, indent=1, sort_keys=True) + "\n")
    with patched_runtime(args.arm, args.seed, scenes, experiment, run_dir):
        trainer.run_training(command, experiment=experiment)
    (run_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=1, sort_keys=True) + "\n")
    print("DONE " + json.dumps({"arm": args.arm, "run_dir": str(run_dir)}), flush=True)


if __name__ == "__main__":
    main()
