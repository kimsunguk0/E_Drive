#!/usr/bin/env python3
"""Final-fit MR-NATIVE on the 376 unique challenge scenes.

This is deliberately a new fit from the pinned R0 initializer.  It does not
continue the 20,554-step MR terminal.  The existing train/tune/val membership
and supervision provenance remain intact; a small union dataset presents each
unique scene once to the trainer.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
BASE_DIR = ROOT / "experiments/md_r0_reset_20260914"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(BASE_DIR))

import train as base
import motiondrive_v2_data as data_api


ARM = "MR-NATIVE-FULL"
NAME = "md_mr_native_fullfit_20260917"
SEED = 1
FULL_SPLITS = ("train", "tune", "val")
FULL_SCENES = 376
FULL_ROWS = 101_520
UPDATES = 24_931
EVAL_EVERY = math.ceil(FULL_ROWS / base.BATCH)  # diagnostic only; overlaps fit data
MICROBATCH = 8
CUDA_MEMORY_LIMIT_MIB = 170_000


class UniqueSplitUnion:
    """Concatenate disjoint manifest splits while preserving global row IDs."""

    def __init__(self, datasets):
        if not datasets:
            raise ValueError("at least one dataset is required")
        self.datasets = tuple(datasets)
        self.ends = np.cumsum([len(d) for d in self.datasets]).tolist()
        self.rows = np.concatenate([d.rows for d in self.datasets]).astype(np.int64, copy=False)
        first = self.datasets[0]
        self.arr = first.arr
        self.scene_names = first.scene_names
        self.image_root = first.image_root
        self.history_offsets = first.history_offsets
        self.augment = first.augment
        self.seed = first.seed
        self.epoch = 0
        seen = set()
        for dataset in self.datasets:
            names = set(map(str, dataset.scene_names[dataset.rows]))
            overlap = seen.intersection(names)
            if overlap:
                raise ValueError(f"full-fit split overlap: {sorted(overlap)[:4]}")
            seen.update(names)
            if (dataset.arr is not first.arr and
                    not np.array_equal(dataset.arr["frame"], first.arr["frame"])):
                raise ValueError("union children do not share the same ego cache")
        if len(seen) != FULL_SCENES or len(self.rows) != FULL_ROWS:
            raise ValueError(f"expected {FULL_SCENES} scenes/{FULL_ROWS} rows, "
                             f"got {len(seen)}/{len(self.rows)}")
        if len(np.unique(self.rows)) != len(self.rows):
            raise ValueError("full-fit union contains duplicate row IDs")

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        child = bisect.bisect_right(self.ends, index)
        start = 0 if child == 0 else self.ends[child - 1]
        return self.datasets[child][index - start]

    def __getattr__(self, name):
        if name in {"datasets", "ends", "rows", "arr", "scene_names", "epoch"}:
            raise AttributeError(name)
        return getattr(self.datasets[0], name)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        for dataset in self.datasets:
            dataset.set_epoch(epoch)


def make_union(original_dataset, kwargs):
    if kwargs.get("split") != "train" or kwargs.get("scenes") is not None:
        raise ValueError("FULL training expects the unfiltered logical train split")
    children = []
    for split in FULL_SPLITS:
        child_kwargs = dict(kwargs)
        child_kwargs["split"] = split
        children.append(original_dataset(**child_kwargs))
    return UniqueSplitUnion(children)


def build_datasets(seed):
    common = dict(data_root="/tmp/pm97", split_manifest=str(base.NEW_SPLIT),
                  supervision_root=str(base.SUPERVISION), min_frame=30,
                  max_samples=0, seed=seed, history_contract="control")
    training = make_union(data_api.MotionDriveDataset,
                          dict(split="train", frame_stride=1, augment=False, **common))
    # This is an explicitly in-fit diagnostic.  It is never used for checkpoint
    # or recipe selection and remains separate from every DEV lineage.
    evaluation = data_api.MotionDriveDataset(split="tune", frame_stride=5,
                                             augment=False, **common)
    return training, evaluation


def configure_base_module():
    base.NAME = NAME
    if ARM not in base.ARMS:
        base.ARMS = (*base.ARMS, ARM)
    if ARM not in base.EXPANDED_ARMS:
        base.EXPANDED_ARMS = (*base.EXPANDED_ARMS, ARM)
    base.MR_DETAIL[ARM] = "native"
    base.LENGTH_LAMBDA[ARM] = base.LENGTH_LAMBDA_MR
    base.ARM_UPDATES[ARM] = UPDATES
    base.EVAL_EVERY = EVAL_EVERY
    base.MICROBATCH = MICROBATCH


def replace_option(command, option, value):
    index = command.index(option)
    command[index + 1] = str(value)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_base_module()
    training, evaluation = build_datasets(SEED)
    experiment = base.build_experiment(ARM, SEED, None, training, evaluation)
    experiment.update({
        "name": NAME,
        "question": "final fit of the registered MR recipe on every unique challenge scene",
        "full_fit": {
            "logical_splits": list(FULL_SPLITS),
            "unique_scenes": FULL_SCENES,
            "rows": FULL_ROWS,
            "historical_val_is_subset_of_val": True,
            "dev_weights_or_caches_imported": False,
            "evaluation_is_in_fit_diagnostic": True,
            "checkpoint_selection_from_evaluation": False,
            "schedule_transfer": "20554 * 101520 / 83700, ceiling",
            "not_a_continuation": True,
        },
    })
    experiment["recipe"]["microbatch"] = MICROBATCH
    experiment["recipe"]["cuda_memory_limit_mib"] = CUDA_MEMORY_LIMIT_MIB
    experiment["recipe"]["eval_role"] = "overlapping diagnostic only"
    experiment["checkpoint_selection"] = "fixed terminal step 24931; no in-fit metric selection"
    command = base.trainer_argv(ARM, SEED, args.run_dir, None, args.gpu)
    replace_option(command, "--cuda-memory-limit-mib", CUDA_MEMORY_LIMIT_MIB)
    summary = {
        "arm": ARM, "seed": SEED, "gpu": args.gpu,
        "run_dir": str(Path(args.run_dir).resolve()),
        "train_rows": len(training),
        "train_scenes": len(set(map(str, training.scene_names[training.rows]))),
        "updates": UPDATES, "eval_every": EVAL_EVERY,
        "microbatch": MICROBATCH,
        "exposures": UPDATES * base.BATCH / len(training),
    }
    print("PLAN " + json.dumps(summary, sort_keys=True), flush=True)
    if args.dry_run:
        return

    original_dataset = data_api.MotionDriveDataset

    def full_dataset_factory(**kwargs):
        if kwargs.get("split") == "train":
            return make_union(original_dataset, kwargs)
        return original_dataset(**kwargs)

    plan_path = ROOT / "reports/md_r0_reset_20260914" / f"experiment_{ARM}-s{SEED}.json"
    plan_path.write_text(json.dumps(experiment, indent=1, sort_keys=True) + "\n")
    import train_motiondrive_v2 as trainer
    data_api.MotionDriveDataset = full_dataset_factory
    try:
        with base.patched_runtime(ARM, SEED, None, experiment, args.run_dir):
            trainer.run_training(command, experiment=experiment)
    finally:
        data_api.MotionDriveDataset = original_dataset
    run_dir = Path(args.run_dir).resolve()
    (run_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
