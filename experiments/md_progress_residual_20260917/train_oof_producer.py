#!/usr/bin/env python3
"""Train one honest old-203 MR producer for residual stacking.

The deployed MR was fitted on the expanded 310-scene pool.  A refiner trained
on its in-sample predictions mostly sees tiny errors.  This producer is fitted
only on the ancestral 203-scene split, leaving the 107 newly absorbed scenes
untouched so they can supply real out-of-fit base-plan errors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path("/NHNHOME/data/sukim/adcl")
BASE_DIR = ROOT / "experiments/md_r0_reset_20260914"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(BASE_DIR))

import train as base
from build_grouped_split_v2 import sha256


ARM = "MR-NATIVE-T203-OOF"
UPDATES = 20_554
REPORT_DIR = ROOT / "reports/md_progress_residual_20260917"


def configure_base() -> None:
    # Reuse the already audited MR training implementation without editing its
    # historical experiment registry.  This arm is deliberately absent from
    # EXPANDED_ARMS, so build_datasets selects the prior train203 scene list.
    base.NAME = "md_progress_oof_producer_20260917"
    base.ARMS = (ARM,)
    base.MR_DETAIL[ARM] = "native"
    base.LENGTH_LAMBDA[ARM] = base.LENGTH_LAMBDA_MR
    base.ARM_UPDATES[ARM] = UPDATES
    # Evaluate and save only at the fixed terminal.  V0 never selects a
    # producer checkpoint.
    base.EVAL_EVERY = UPDATES


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--physical-gpu", type=int, choices=range(4), required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configure_base()
    prior = json.loads(base.BASE_SPLIT.read_text())
    expanded = json.loads(base.NEW_SPLIT.read_text())
    prior_train = set(prior["splits"]["train"])
    expanded_train = set(expanded["splits"]["train"])
    held_out = sorted(expanded_train - prior_train)
    if len(prior_train) != 203 or len(held_out) != 107:
        raise ValueError(f"unexpected old/new scene partition: {len(prior_train)}/{len(held_out)}")
    if prior_train & set(held_out):
        raise ValueError("OOF producer train scenes overlap the residual-fit pool")

    scenes, training, evaluation = base.build_datasets(ARM, args.seed)
    if set(scenes) != prior_train or len(training) != base.N0_ROWS:
        raise ValueError("producer dataset is not exactly the prior train203 split")
    experiment = base.build_experiment(ARM, args.seed, scenes, training, evaluation)
    experiment["question"] = (
        "can real errors on 107 scenes excluded from this producer train a progress "
        "refiner that generalizes to unseen sessions?")
    experiment["checkpoint_selection"] = "fixed terminal step 20554; V0 never selects producer"
    experiment["oof_contract"] = {
        "producer_train_scenes": len(prior_train),
        "producer_train_rows": len(training),
        "producer_train_split": str(base.BASE_SPLIT),
        "producer_train_split_sha256": sha256(base.BASE_SPLIT),
        "residual_fit_scenes_excluded_from_producer": len(held_out),
        "residual_fit_scene_names": held_out,
        "expanded_split": str(base.NEW_SPLIT),
        "expanded_split_sha256": sha256(base.NEW_SPLIT),
        "scene_overlap": 0,
        "terminal_only": True,
        "purpose": "producer only; do not submit this checkpoint",
    }
    experiment["runtime"] = {
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices_must_equal": str(args.physical_gpu),
        "trainer_logical_gpu": 0,
    }
    command = base.trainer_argv(ARM, args.seed, args.run_dir, scenes, gpu=0)
    summary = {
        "arm": ARM,
        "seed": args.seed,
        "physical_gpu": args.physical_gpu,
        "run_dir": str(Path(args.run_dir).resolve()),
        "train_rows": len(training),
        "train_scenes": len(prior_train),
        "held_out_scenes": len(held_out),
        "updates": UPDATES,
        "eval_every": base.EVAL_EVERY,
        "exposures": UPDATES * base.BATCH / len(training),
    }
    print("PLAN " + json.dumps(summary, sort_keys=True), flush=True)
    if args.dry_run:
        return

    del training, evaluation
    import train_motiondrive_v2 as trainer
    run_dir = Path(args.run_dir).resolve()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = REPORT_DIR / f"protocol_{ARM}_s{args.seed}.json"
    plan_path.write_text(json.dumps(experiment, indent=1, sort_keys=True) + "\n")
    with base.patched_runtime(ARM, args.seed, scenes, experiment, run_dir):
        trainer.run_training(command, experiment=experiment)
    (run_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=1, sort_keys=True) + "\n")
    print("DONE " + json.dumps({"arm": ARM, "run_dir": str(run_dir)}), flush=True)


if __name__ == "__main__":
    main()
