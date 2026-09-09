from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_motiondrive_v2_long_training as screen


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_split(tmp_path: Path) -> tuple[Path, Path, dict]:
    scenes = [f"scene_{index:03d}" for index in range(203)]
    sessions = {scene: f"session_{index:03d}"
                for index, scene in enumerate(scenes)}
    split = {
        "splits": {"train": scenes, "tune": [], "val": [],
                   "historical_val": []},
        "scene_to_session": sessions,
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    artifact = {
        "holdout_sessions": [sessions[scene] for scene in scenes[:3]],
        "holdout_scenes": scenes[:3],
        "train_scenes": scenes[3:],
        "rule": {"method": "unit-test session holdout"},
    }
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps(artifact))
    return split_path, holdout_path, artifact


def namespace(tmp_path: Path, arm="long_s0", seed=0):
    split, holdout, artifact = pinned_split(tmp_path)
    args = SimpleNamespace(
        arm=arm, init="parent.pth", init_manifest="parent.json",
        source_manifest="source.json", expected_source_manifest_sha256="a" * 64,
        data_root="data", split_manifest=str(split), supervision_root="supervision",
        status_overlay_root="overlay", expected_status_overlay_sha256="b" * 64,
        holdout_split=str(holdout),
        expected_holdout_split_sha256=digest(holdout), run_dir="run",
        seed=seed, gpu=0, workers=4, cuda_memory_limit_mib=12000,
        cuda_min_free_mib=8192, expected_physical_gpu_uuid="GPU-test",
        expected_initial_state_sha256="c" * 64,
    )
    return args, artifact


def launcher_cli(tmp_path: Path):
    args, _artifact = namespace(tmp_path)
    return [
        "--arm", args.arm, "--init", args.init,
        "--init-manifest", args.init_manifest,
        "--source-manifest", args.source_manifest,
        "--expected-source-manifest-sha256", args.expected_source_manifest_sha256,
        "--data-root", args.data_root, "--split-manifest", args.split_manifest,
        "--supervision-root", args.supervision_root,
        "--status-overlay-root", args.status_overlay_root,
        "--expected-status-overlay-sha256", args.expected_status_overlay_sha256,
        "--holdout-split", args.holdout_split,
        "--expected-holdout-split-sha256", args.expected_holdout_split_sha256,
        "--run-dir", args.run_dir, "--seed", str(args.seed), "--gpu", "0",
        "--workers", "4", "--cuda-memory-limit-mib", "12000",
        "--cuda-min-free-mib", "8192",
        "--expected-physical-gpu-uuid", args.expected_physical_gpu_uuid,
        "--expected-initial-state-sha256", args.expected_initial_state_sha256,
    ]


@pytest.mark.parametrize("option", [
    "--status-overlay-root", "--expected-status-overlay-sha256",
])
def test_status_overlay_arguments_are_required(tmp_path, option):
    cli = launcher_cli(tmp_path)
    index = cli.index(option)
    del cli[index:index + 2]
    with pytest.raises(SystemExit):
        screen.arguments(cli)


def test_launcher_has_exact_surface_and_recipe_is_not_tunable(tmp_path):
    args = screen.arguments(launcher_cli(tmp_path))
    assert set(vars(args)) == {
        "arm", "init", "init_manifest", "source_manifest",
        "expected_source_manifest_sha256", "data_root", "split_manifest",
        "supervision_root", "status_overlay_root", "expected_status_overlay_sha256",
        "holdout_split", "expected_holdout_split_sha256", "run_dir", "seed",
        "gpu", "workers", "cuda_memory_limit_mib", "cuda_min_free_mib",
        "expected_physical_gpu_uuid", "expected_initial_state_sha256",
    }
    with pytest.raises(SystemExit):
        screen.arguments(launcher_cli(tmp_path) + ["--steps", "1"])

    holdout = screen.validate_holdout_split(
        args.holdout_split, args.expected_holdout_split_sha256,
        args.split_manifest)
    command = screen.trainer_argv(args, holdout)
    import train_motiondrive_v2 as trainer
    parsed = trainer.arguments(command)
    expected = {
        "phase": "joint", "goal_on": 1, "state_on": 1, "steps": 68500,
        "batch": 16, "microbatch": 2, "eval_batch": 4, "workers": 4,
        "eval_every": 3425, "save_every": 3425, "lr": 5e-5,
        "backbone_lr": 5e-6, "weight_decay": .01, "warmup": 100,
        "grad_clip": 5., "alpha_occ": .2, "alpha_lane": .2,
        "alpha_motion": .2, "precision": "bf16", "time_input": "nominal",
        "bn_policy": "fixed", "arch": "resnet50",
        "motion_input_mode": "low_feature", "cross_cell_goal_mode": "zero",
        "history_contract": "control", "train_stride": 1, "eval_stride": 5,
        "max_train_samples": 0, "max_eval_samples": 0,
    }
    assert all(getattr(parsed, key) == value for key, value in expected.items())
    assert parsed.init and not parsed.resume and not parsed.pretrained
    assert parsed.uncertainty == 1 and not parsed.eval_only and not parsed.cpu
    assert "--status-overlay-root" not in command
    assert "--expected-status-overlay-sha256" not in command


@pytest.mark.parametrize(("arm", "seed"), [
    ("long_s0", 1), ("long_s1", 0),
    ("holdout_s0", 1), ("holdout_s1", 0),
])
def test_arm_seed_disagreement_is_rejected(tmp_path, arm, seed):
    args, artifact = namespace(tmp_path, arm, seed)
    with pytest.raises(ValueError, match="arm/seed mismatch"):
        screen.trainer_argv(args, artifact)


@pytest.mark.parametrize(("arm", "seed", "holdout_applied"), [
    ("long_s0", 0, False), ("long_s1", 1, False),
    ("holdout_s0", 0, True), ("holdout_s1", 1, True),
])
def test_scene_lists_and_eval_split_follow_arm(tmp_path, arm, seed, holdout_applied):
    args, artifact = namespace(tmp_path, arm, seed)
    holdout = screen.validate_holdout_split(
        args.holdout_split, args.expected_holdout_split_sha256,
        args.split_manifest)
    import train_motiondrive_v2 as trainer
    parsed = trainer.arguments(screen.trainer_argv(args, holdout))
    if holdout_applied:
        assert parsed.train_scenes == artifact["train_scenes"]
        assert parsed.eval_scenes == artifact["holdout_scenes"]
        assert parsed.eval_split == "train"
    else:
        assert parsed.train_scenes is None and parsed.eval_scenes is None
        assert parsed.eval_split == "tune"


@pytest.mark.parametrize(("arm", "seed"), [
    ("long_s0", 0), ("long_s1", 1),
    ("holdout_s0", 0), ("holdout_s1", 1),
])
def test_holdout_sha_is_verified_for_every_arm(tmp_path, arm, seed):
    args, _artifact = namespace(tmp_path, arm, seed)
    args.expected_holdout_split_sha256 = "0" * 64
    with pytest.raises(ValueError, match="holdout-split SHA mismatch"):
        screen.trainer_argv(args)


def test_status_overlay_is_row_aligned_for_holdout_subsets():
    class Base:
        rows = np.asarray([2, 5], dtype=np.int64)
        arr = {"frame": np.asarray([0, 0, 20, 0, 0, 50, 0, 0], dtype=np.int64)}
        scene_names = np.asarray(["x"] * 8)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"row": torch.tensor(int(self.rows[index]))}

        def set_epoch(self, _epoch):
            return None

    overlay = {"train": {
        "row": np.asarray([1, 2, 5, 7], dtype=np.int64),
        "frame": np.asarray([10, 20, 50, 70], dtype=np.int64),
        "status5": np.arange(20, dtype=np.float32).reshape(4, 5),
    }}
    wrapped = screen.status_dataset(Base(), "train", overlay)
    assert torch.equal(wrapped[0]["provided_status5"],
                       torch.from_numpy(overlay["train"]["status5"][1]))
    assert torch.equal(wrapped[1]["provided_status5"],
                       torch.from_numpy(overlay["train"]["status5"][2]))


def test_patched_runtime_installs_status_wiring_and_fully_restores(monkeypatch):
    import models.motiondrive_v2 as model_api
    import motiondrive_v2_data as data_api
    import train_motiondrive_v2 as trainer
    import evaluate_motiondrive_v2_planning as planning_eval

    class Base:
        rows = np.asarray([2], dtype=np.int64)
        arr = {"frame": np.asarray([0, 0, 20], dtype=np.int64)}
        scene_names = np.asarray(["x", "x", "scene"])

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return {"base": torch.tensor(1.)}

        def set_epoch(self, _epoch):
            return None

    base = Base()
    monkeypatch.setattr(data_api, "MotionDriveDataset", lambda **_kwargs: base)
    monkeypatch.setattr(
        trainer, "model_inputs",
        lambda _batch, time_input="raw", nominal_history_seconds=(.1, .2, .5, 1.):
            {"train_base": (time_input, nominal_history_seconds)})
    monkeypatch.setattr(planning_eval, "planning_model_inputs",
                        lambda _batch, time_input="raw": {"eval_base": time_input})
    overlay = {name: {
        "row": np.asarray([2], dtype=np.int64),
        "frame": np.asarray([20], dtype=np.int64),
        "status5": np.asarray([[1., 2., 3., 4., 5.]], dtype=np.float32),
    } for name in ("train", "tune")}
    holdout = {"sha256": "h" * 64, "train_scenes": ["train"],
               "holdout_scenes": ["held"]}

    def bindings():
        return (model_api.MotionDriveV2, data_api.MotionDriveDataset,
                trainer.model_inputs, planning_eval.planning_model_inputs,
                trainer._validate_experimental_protocol,
                trainer._validate_experimental_runtime,
                trainer._load_initial_model_state,
                trainer._training_schedule_actions, trainer.arguments)

    before = bindings()
    status = torch.tensor([[1., 2., 3., 4., 5.]])
    with pytest.raises(RuntimeError, match="sentinel"):
        with screen.patched_runtime("long_s0", 0, overlay, holdout, "p", "i"):
            dataset = data_api.MotionDriveDataset(split="train", augment=True)
            assert isinstance(dataset, screen.a1.SharedStatusDataset)
            assert torch.equal(dataset[0]["provided_status5"], status[0])
            batch = {"provided_status5": status}
            assert trainer.model_inputs(batch)["provided_status5"] is status
            assert planning_eval.planning_model_inputs(batch)["provided_status5"] is status
            assert bindings() != before
            raise RuntimeError("sentinel")
    assert bindings() == before


def test_holdout_runtime_preserves_train_split_and_bypasses_only_legacy_guard(tmp_path):
    args, artifact = namespace(tmp_path, "holdout_s0", 0)
    holdout = {"sha256": args.expected_holdout_split_sha256, **artifact}
    import train_motiondrive_v2 as trainer
    command = screen.trainer_argv(args, holdout)
    original_arguments = trainer.arguments
    with screen.patched_runtime("holdout_s0", 0,
                                {"train": {}, "tune": {}},
                                holdout, "p", "i"):
        parsed = trainer.arguments(command)
        assert parsed.eval_split == "train"
        assert not (parsed.eval_split != "tune")
    assert trainer.arguments is original_arguments


def test_holdout_runtime_disables_augmentation_only_for_heldout_evaluation(monkeypatch):
    import motiondrive_v2_data as data_api

    class Base:
        rows = np.asarray([2], dtype=np.int64)
        arr = {"frame": np.asarray([0, 0, 20], dtype=np.int64)}
        scene_names = np.asarray(["x", "x", "scene"])

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return {}

        def set_epoch(self, _epoch):
            return None

    calls = []
    monkeypatch.setattr(
        data_api, "MotionDriveDataset",
        lambda **kwargs: (calls.append(kwargs.copy()) or Base()))
    overlay = {"train": {
        "row": np.asarray([2], dtype=np.int64),
        "frame": np.asarray([20], dtype=np.int64),
        "status5": np.zeros((1, 5), dtype=np.float32),
    }}
    holdout = {"sha256": "h" * 64, "train_scenes": ["train"],
               "holdout_scenes": ["held"]}
    with screen.patched_runtime("holdout_s0", 0, overlay, holdout, "p", "i"):
        data_api.MotionDriveDataset(
            split="train", scenes=["held"], augment=True)
        data_api.MotionDriveDataset(
            split="train", scenes=["train"], augment=True)
    assert calls[0]["augment"] is False
    assert calls[1]["augment"] is True


def test_runtime_validator_and_schedule_assert_the_fixed_recipe(tmp_path):
    args, artifact = namespace(tmp_path)
    holdout = {"sha256": args.expected_holdout_split_sha256, **artifact}
    import train_motiondrive_v2 as trainer
    parsed = trainer.arguments(screen.trainer_argv(args, holdout))
    with screen.patched_runtime("long_s0", 0, {"train": {}, "tune": {}},
                                holdout, "p", "i"):
        trainer._validate_experimental_runtime(parsed, {})
        assert trainer._training_schedule_actions(3424, parsed, {}) == (False, False)
        assert trainer._training_schedule_actions(3425, parsed, {}) == (True, True)
        assert trainer._training_schedule_actions(68500, parsed, {}) == (True, True)
        parsed.steps = 1
        with pytest.raises(ValueError, match="fixed long-training"):
            trainer._validate_experimental_runtime(parsed, {})


def test_overlay_and_holdout_artifacts_are_immutable_inputs(tmp_path):
    args, _artifact = namespace(tmp_path)
    immutable = {str(path) for path in screen.immutable_inputs(args)}
    assert {"overlay/overlay_manifest.json", "overlay/train.npz",
            "overlay/tune.npz", args.holdout_split} <= immutable


def test_source_closure_contains_a2_lineage_and_new_launcher_test():
    assert {"models/motiondrive_v2/shared_status_query.py",
            "scripts/run_motiondrive_v2_shared_status_a1.py",
            "scripts/run_motiondrive_v2_shared_status_a2.py",
            "scripts/run_motiondrive_v2_long_training.py",
            "tests/test_motiondrive_v2_long_training.py"} <= screen.SOURCE_FILES
    assert not any(name.startswith("work_dirs/") for name in screen.SOURCE_FILES)
