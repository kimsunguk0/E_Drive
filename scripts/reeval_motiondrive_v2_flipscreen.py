"""Re-evaluate the flip screen's checkpoints on the clean (never-flipped) holdout.

The flip-screen launcher used to attach `FlipAugmented` whenever
``split == "train"``.  Holdout arms build BOTH the training set and the
evaluation set with ``split == "train"`` (dataset_factory separates them by
scene list, not by split) and `FlipAugmented.__getitem__` mirrors with p=0.5
with no train/eval switch, so `flip60`'s reported holdout score was measured on
a 50%-mirrored holdout while `noflip60`'s was measured on the clean one.  The
launcher now takes an explicit `augmentable` flag; this driver reuses that
fixed wiring and the trainer's `--eval-only` path.

Three cases are evaluated.  Two are positive controls that must reproduce what
the launcher recorded; the third is the comparison the screen was supposed to
make.
"""
import argparse, json, shutil, sys
from pathlib import Path

REPO = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import run_motiondrive_v2_flip_screen as fs
import train_motiondrive_v2 as trainer
import __main__


class _HoldoutTrainEvalSplit(str):
    """Unpickling shim: the launcher ran as __main__ and pickled this marker
    into every checkpoint's arguments; str subclasses rebuild as cls(value)."""

    def __new__(cls, value="train"):
        return super().__new__(cls, value)

    def __ne__(self, other):
        return False if other == "tune" else super().__ne__(other)


__main__._HoldoutTrainEvalSplit = _HoldoutTrainEvalSplit

OVERLAY = REPO / "data/etri/motiondrive_v2_shared_status_a1_20260908_ops"
OVERLAY_SHA = "8852f1e7ce80895b09ac400e8332d99d9feb6918a8a18e0893738300a3021699"
HOLDOUT = REPO / "reports/motiondrive_v2_longrun_holdout_split_20260909_ops.json"
HOLDOUT_SHA = "3f9cbb84839b783b9055a38b80bf6ae46f707629ec3a1890755efb3f08842dfd"

# (arm, holdout label, mirror the evaluation rows, value the launcher recorded)
CASES = (("noflip60", "clean", False, 0.1301326504978567),
         ("flip60", "clean", False, None),
         ("flip60", "mirrored", True, 0.1794939921108178))


def make_args(arm, run_dir, checkpoint=None):
    checkpoint = (Path(checkpoint) if checkpoint
                  else REPO / f"work_dirs/motiondrive_v2/flipscreen_{arm}/last.pth")
    if not checkpoint.exists():
        raise SystemExit(f"missing checkpoint {checkpoint}")
    return argparse.Namespace(
        arm=arm, seed=0, gpu=0, workers=4,
        init=str(checkpoint),
        init_manifest=str(checkpoint.parent / "manifest.json"),
        data_root="/tmp/pm97",
        split_manifest=str(REPO / "data/etri/motiondrive_v2/grouped_split_rawtime.json"),
        supervision_root=str(REPO / "data/etri/motiondrive_v2/train_tune_geometry_v2"),
        status_overlay_root=str(OVERLAY),
        expected_status_overlay_sha256=OVERLAY_SHA,
        holdout_split=str(HOLDOUT),
        expected_holdout_split_sha256=HOLDOUT_SHA,
        run_dir=str(run_dir),
        cuda_memory_limit_mib=12000, cuda_min_free_mib=8192,
        history_overlay_root=None, expected_history_overlay_sha256=None,
    )


def run_case(arm, label, mirror_eval, checkpoint=None):
    run_dir = Path(f"/tmp/reeval_{arm}_{label}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    args = make_args(arm, run_dir, checkpoint)
    _data, overlay, holdout = fs.validate_data(args)  # sets _FLIP from the arm
    fs._FLIP["on"] = mirror_eval
    command = fs.trainer_argv(args, holdout) + ["--eval-only"]

    def load_initial(model, common, experiment=None):
        bad = model.load_state_dict(common["model"], strict=True)
        assert not bad.missing_keys and not bad.unexpected_keys, bad
        return {"reeval_strict_load": True}

    original_status_dataset = fs.status_dataset

    def forced(base, split, ov, augmentable=False):
        # The fixed launcher never augments the evaluation set.  The mirrored
        # control deliberately overrides that to reproduce the buggy reading.
        return original_status_dataset(base, split, ov, augmentable=mirror_eval)

    with fs.patched_runtime(arm, 0, overlay, holdout, "", ""):
        fs.status_dataset = forced
        trainer._validate_experimental_protocol = lambda experiment: None
        trainer._validate_experimental_runtime = lambda runtime_args, experiment: None
        trainer._load_initial_model_state = load_initial
        try:
            trainer.run_training(command, experiment=None)
        finally:
            fs.status_dataset = original_status_dataset

    report = json.loads((run_dir / "evaluation.json").read_text())["report"]
    return {"arm": arm, "checkpoint": str(args.init),
            "holdout": label, "mirrored_eval": mirror_eval,
            "official_d3": report.get("official_d3"),
            "session_mean_d3": report.get("session_mean_d3"),
            "n": report.get("n")}


def main():
    # `--checkpoint PATH LABEL` scores any checkpoint on the clean holdout,
    # borrowing an arm's scene wiring; otherwise run the three pinned cases.
    if sys.argv[1:2] == ["--checkpoint"]:
        path, label = sys.argv[2], sys.argv[3]
        row = run_case("noflip60", label, False, checkpoint=path)
        print("RESULT " + json.dumps(row), flush=True)
        return
    selected = sys.argv[1:]
    rows = []
    for arm, label, mirror_eval, recorded in CASES:
        if selected and f"{arm}:{label}" not in selected:
            continue
        row = run_case(arm, label, mirror_eval)
        if recorded is not None:
            row["launcher_recorded"] = recorded
            row["reproduced"] = abs(row["official_d3"] - recorded) < 1e-4
        rows.append(row)
        print("RESULT " + json.dumps(row), flush=True)
    print("SUMMARY " + json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
