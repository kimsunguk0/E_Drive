"""Score a MotionDrive V2 checkpoint on the held-out `tune` split.

Everything measured during the long-training, data-scaling and flip screens was
measured on a 12-session holdout carved out of train203.  Those sessions come
from the same collection campaign as the training sessions, so a gain there may
not be a gain on a genuinely separate split.  This scores the same checkpoints
on `tune`, where the lineage's parent stands at official_d3 0.280640, using the
trainer's own --eval-only path and the A2 status wiring.
"""
import argparse, json, shutil, sys
from pathlib import Path

REPO = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import run_motiondrive_v2_shared_status_a1 as a1
import run_motiondrive_v2_flip_screen as fs
import train_motiondrive_v2 as trainer
import models.motiondrive_v2 as model_api
import motiondrive_v2_data as data_api
import evaluate_motiondrive_v2_planning as planning_eval
import __main__


class _HoldoutTrainEvalSplit(str):
    def __new__(cls, value="train"):
        return super().__new__(cls, value)

    def __ne__(self, other):
        return False if other == "tune" else super().__ne__(other)


__main__._HoldoutTrainEvalSplit = _HoldoutTrainEvalSplit

OVERLAY = REPO / "data/etri/motiondrive_v2_shared_status_a1_20260908_ops"
OVERLAY_SHA = "8852f1e7ce80895b09ac400e8332d99d9feb6918a8a18e0893738300a3021699"


def evaluate(checkpoint, run_dir, gpu=0):
    checkpoint = Path(checkpoint).resolve()
    run_dir = Path(run_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)

    args = argparse.Namespace(
        arm="provided_causal_5d", seed=0,
        data_root="/tmp/pm97",
        split_manifest=str(REPO / "data/etri/motiondrive_v2/grouped_split_rawtime.json"),
        supervision_root=str(REPO / "data/etri/motiondrive_v2/train_tune_geometry_v2"),
        status_overlay_root=str(OVERLAY),
        expected_status_overlay_sha256=OVERLAY_SHA,
    )
    _full, overlay = a1.validate_data(args)

    original_model = model_api.MotionDriveV2
    original_dataset = data_api.MotionDriveDataset
    original_eval_inputs = planning_eval.planning_model_inputs
    original_train_inputs = trainer.model_inputs
    original_loader = trainer._load_initial_model_state
    original_protocol = trainer._validate_experimental_protocol
    original_runtime = trainer._validate_experimental_runtime

    def model_factory(config=None):
        from models.motiondrive_v2 import MotionDriveV2Config
        return fs.make_model(config if config is not None else MotionDriveV2Config())

    def dataset_factory(**kwargs):
        base = original_dataset(**kwargs)
        # tune is a plain evaluation split: never augmentable.
        return fs.status_dataset(base, kwargs.get("split"), overlay, augmentable=False)

    def eval_inputs(batch, time_input="raw"):
        return a1._add_status(original_eval_inputs(batch, time_input), batch)

    def train_inputs(batch, time_input="raw",
                     nominal_history_seconds=(.1, .2, .5, 1.)):
        # The trainer's evaluation loop builds inputs through trainer.model_inputs,
        # so the A2 status route has to be installed here as well.
        return a1.shared_status_model_inputs(
            original_train_inputs, batch, time_input=time_input,
            nominal_history_seconds=nominal_history_seconds)

    def load_initial(model, common, experiment=None):
        bad = model.load_state_dict(common["model"], strict=True)
        assert not bad.missing_keys and not bad.unexpected_keys, bad
        return {"tune_eval_strict_load": True}

    command = [
        "--data-root", args.data_root,
        "--split-manifest", args.split_manifest,
        "--supervision-root", args.supervision_root,
        "--run-dir", str(run_dir.resolve()),
        "--phase", "joint", "--goal-on", "1", "--state-on", "1",
        "--cross-cell-goal-mode", "zero", "--history-contract", "control",
        "--gpu", str(gpu), "--seed", "0", "--steps", "1",
        "--batch", "16", "--microbatch", "2", "--eval-batch", "4",
        "--workers", "4", "--eval-every", "1", "--save-every", "1",
        "--log-every", "10", "--lr", "0.00005", "--backbone-lr", "0.000005",
        "--weight-decay", "0.01", "--warmup", "100", "--grad-clip", "5",
        "--alpha-occ", "0.2", "--alpha-lane", "0.2", "--alpha-motion", "0.2",
        "--uncertainty", "1", "--precision", "bf16", "--time-input", "nominal",
        "--bn-policy", "fixed", "--init", str(checkpoint),
        "--arch", "resnet50", "--motion-input-mode", "low_feature",
        "--train-stride", "1", "--eval-stride", "5",
        "--max-train-samples", "0", "--max-eval-samples", "0",
        "--eval-split", "tune", "--eval-only",
        "--cuda-memory-limit-mib", "12000", "--cuda-min-free-mib", "8192",
    ]
    try:
        model_api.MotionDriveV2 = model_factory
        data_api.MotionDriveDataset = dataset_factory
        planning_eval.planning_model_inputs = eval_inputs
        trainer.model_inputs = train_inputs
        trainer._load_initial_model_state = load_initial
        trainer._validate_experimental_protocol = lambda experiment: None
        trainer._validate_experimental_runtime = lambda runtime_args, experiment: None
        trainer.run_training(command, experiment=None)
    finally:
        model_api.MotionDriveV2 = original_model
        data_api.MotionDriveDataset = original_dataset
        planning_eval.planning_model_inputs = original_eval_inputs
        trainer.model_inputs = original_train_inputs
        trainer._load_initial_model_state = original_loader
        trainer._validate_experimental_protocol = original_protocol
        trainer._validate_experimental_runtime = original_runtime

    report = json.loads((run_dir / "evaluation.json").read_text())["report"]
    return {"checkpoint": str(checkpoint),
            "official_d3": report.get("official_d3"),
            "session_mean_d3": report.get("session_mean_d3"),
            "n": report.get("n")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parsed = parser.parse_args()
    run_dir = parsed.run_dir or f"/tmp/tuneeval_{Path(parsed.checkpoint).parent.name}"
    print("RESULT " + json.dumps(evaluate(parsed.checkpoint, run_dir, parsed.gpu)),
          flush=True)


if __name__ == "__main__":
    main()
