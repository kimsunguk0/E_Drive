"""CPU-only P8 temporal-contract and orchestration regressions."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from models.motiondrive_v2_temporal_contract import CONTROL, WIDE, temporal_contract
from scripts import build_motiondrive_v2_history_overlay as overlay_builder
from scripts import export_motiondrive_v2_inference as exporter
from scripts import motiondrive_v2_data as data_module
from scripts import motiondrive_v2_training as training
from scripts import run_motiondrive_v2_p8_wide_history as p8
from scripts import train_motiondrive_v2 as trainer


def hash_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def temporal_dict(name):
    contract = temporal_contract(name)
    return {"name": name, "frame_offsets": list(contract.frame_offsets),
            "nominal_seconds": list(contract.nominal_seconds)}


def protocol(stage="joint", arm="wide"):
    return {"schema_version": 1, "name": "p8_wide_history", "stage": stage, "arm": arm,
            "temporal_contract": temporal_dict(arm),
            "history_overlay_manifest_sha256": "a" * 64,
            "expected_initial_checkpoint_sha256": "b" * 64,
            "expected_p0_model_state_sha256": "c" * 64,
            "expected_initial_model_state_sha256": "d" * 64,
            "expected_branch_state_sha256": None if stage == "pretrain" else "f" * 64,
            "expected_optimizer_groups": [{"name": "backbone", "base_lr": 1e-5},
                                          {"name": "head", "base_lr": 1e-4}],
            "expected_missing_state_keys": ([] if stage == "pretrain" else
                ["scene_encoder.cross_cell_goal_residual.output.weight"]),
            "train_data": {"rows": 54810, "rows_sha256": p8.EXPECTED_TRAIN_ROWS_SHA256},
            "tune_data": {"rows": 1998, "rows_sha256": p8.EXPECTED_TUNE_ROWS_SHA256},
            "source": {"sha256": "e" * 64}, "fresh_optimizer_step_zero": True,
            "all_model_parameters_trainable": True, "p0_branch_disabled": True,
            "joint_branch_zero": True, "final_validation_accessed": False,
            "branch": {"mode": "disabled" if stage == "pretrain" else "zero",
                       "sigma_m": [10., 32. / 3.], "pool_size": 4, "source_cells": 192,
                       "destination_cells": 3072, "attention_dim": 32, "cosine_scale": 8.,
                       "attention_precision": "fp32_autocast_disabled",
                       "goal_enters_distance_score_only": True,
                       "new_value_projection_adds_goal_or_position": False,
                       "output_bias": False, "output_weight_zero_initialized_at_joint": True}}


def runtime_args(stage="joint", arm="wide"):
    steps = 2000 if stage == "pretrain" else 6000
    interval = 250 if stage == "pretrain" else 6000
    return SimpleNamespace(
        phase=stage if stage == "pretrain" else "joint",
        goal_on=0 if stage == "pretrain" else 1,
        state_on=0 if stage == "pretrain" else 1, steps=steps, batch=16, microbatch=2,
        eval_batch=4, eval_every=interval, save_every=interval, lr=1e-4,
        backbone_lr=1e-5, weight_decay=.01, warmup=200, grad_clip=5., alpha_occ=.2,
        alpha_lane=.2, alpha_motion=.2, uncertainty=1, precision="bf16",
        time_input="nominal", bn_policy="fixed", arch="resnet50",
        motion_input_mode="low_feature", cross_cell_goal_mode=(
            "disabled" if stage == "pretrain" else "zero"), history_contract=arm,
        train_stride=1, eval_stride=5, max_train_samples=0, max_eval_samples=0,
        eval_split="tune", init="init.pth", resume=None, pretrained=None,
        eval_only=False, train_scenes=None, eval_scenes=None,
        history_overlay_root="overlay", expected_history_overlay_sha256="a" * 64)


def test_fixed_control_and_wide_temporal_contracts_only():
    assert CONTROL.frame_offsets == (1, 2, 5, 10)
    assert WIDE.frame_offsets == (2, 5, 10, 20)
    assert WIDE.nominal_seconds == (.2, .5, 1., 2.)
    with pytest.raises(ValueError):
        temporal_contract("sweep")


def test_default_config_is_exact_historical_control_and_wide_is_explicit():
    control = MotionDriveV2Config()
    assert control.history_contract == "control"
    assert control.history_frame_offsets == CONTROL.frame_offsets
    wide = MotionDriveV2Config(history_contract="wide", history_frame_offsets=WIDE.frame_offsets,
                               nominal_history_seconds=WIDE.nominal_seconds)
    assert wide.history_contract == "wide"
    with pytest.raises(ValueError, match="fixed recipe"):
        MotionDriveV2Config(history_contract="wide")


def test_legacy_checkpoint_migrates_control_but_only_neutral_path_may_acquire_wide():
    saved = {"model_config": MotionDriveV2Config().to_dict()}
    for key in ("history_contract", "history_frame_offsets", "nominal_history_seconds"):
        saved["model_config"].pop(key)
    control = trainer.initialization_configuration(saved, goal_on=0, state_on=0,
                                                   history_contract="control")
    assert control["history_contract"] == "control"
    with pytest.raises(ValueError, match="neutral public initializer"):
        trainer.initialization_configuration(saved, goal_on=0, state_on=0,
                                             history_contract="wide")
    wide = trainer.initialization_configuration(saved, goal_on=0, state_on=0,
                                                history_contract="wide",
                                                allow_legacy_history_override=True)
    assert tuple(wide["history_frame_offsets"]) == WIDE.frame_offsets


def test_trained_control_contract_cannot_change_or_be_partial():
    saved = {"model_config": MotionDriveV2Config().to_dict()}
    with pytest.raises(ValueError, match="cannot be overridden"):
        trainer.initialization_configuration(saved, goal_on=1, state_on=1,
                                             history_contract="wide")
    partial = {"model_config": dict(saved["model_config"])}
    partial["model_config"].pop("nominal_history_seconds")
    with pytest.raises(ValueError, match="partial"):
        trainer.initialization_configuration(partial, goal_on=1, state_on=1,
                                             history_contract="control")


def test_export_complete_legacy_control_migration_and_strict_wide_config():
    legacy = MotionDriveV2Config().to_dict()
    for key in ("history_contract", "history_frame_offsets", "nominal_history_seconds"):
        legacy.pop(key)
    migrated = exporter.validate_complete_config(legacy)
    assert migrated.history_contract == "control"
    partial = MotionDriveV2Config(history_contract="wide", history_frame_offsets=WIDE.frame_offsets,
                                  nominal_history_seconds=WIDE.nominal_seconds).to_dict()
    partial.pop("nominal_history_seconds")
    with pytest.raises(ValueError, match="Incomplete"):
        exporter.validate_complete_config(partial)


def test_nominal_model_input_uses_selected_contract_without_shape_change():
    batch = {key: torch.zeros(3, 1) for key in training.MODEL_INPUTS}
    batch["images"] = torch.zeros(3, 6, 3, 2, 2)
    batch["time_offsets"] = torch.randn(3, 4)
    wide = training.model_inputs(batch, "nominal", WIDE.nominal_seconds)
    assert wide["time_offsets"].shape == (3, 4)
    assert torch.equal(wide["time_offsets"][0], torch.tensor(WIDE.nominal_seconds))
    assert training.time_input_policy("nominal", WIDE.nominal_seconds)[
        "nominal_history_seconds"] == list(WIDE.nominal_seconds)


@pytest.mark.parametrize("stage", ["pretrain", "joint"])
@pytest.mark.parametrize("arm", ["control", "wide"])
def test_p8_protocol_and_fixed_runtime_recipe(stage, arm):
    item = protocol(stage, arm)
    assert trainer._validate_experimental_protocol(item) is None
    trainer._validate_experimental_runtime(runtime_args(stage, arm), item)
    bad = runtime_args(stage, arm)
    bad.steps += 1
    with pytest.raises(ValueError, match="fixed stage recipe"):
        trainer._validate_experimental_runtime(bad, item)


def test_p8_protocol_rejects_temporal_or_missing_key_mutation():
    for mutation in ("seconds", "missing"):
        item = protocol()
        if mutation == "seconds":
            item["temporal_contract"]["nominal_seconds"][0] = .1
        else:
            item.pop("expected_p0_model_state_sha256")
        with pytest.raises(ValueError):
            trainer._validate_experimental_protocol(item)


def test_p8_schedule_preserves_original_p0_monitoring_but_joint_is_terminal_only():
    args = SimpleNamespace(steps=2000, eval_every=250, save_every=250)
    pretrain = protocol("pretrain", "control")
    assert trainer._training_schedule_actions(250, args, pretrain) == (True, True)
    assert trainer._training_schedule_actions(251, args, pretrain) == (False, False)
    args = SimpleNamespace(steps=6000, eval_every=6000, save_every=6000)
    joint = protocol("joint", "control")
    assert trainer._training_schedule_actions(250, args, joint) == (False, False)
    assert trainer._training_schedule_actions(6000, args, joint) == (True, False)


def test_p8_trainer_argv_keeps_count_and_recipe_fixed():
    args = SimpleNamespace(stage="joint", arm="wide", data_root="data",
        split_manifest="split", supervision_root="sup", history_overlay_root="overlay",
        expected_history_overlay_sha256="a" * 64, run_dir="run", base_seed=1, init="p0")
    argv = p8.trainer_argv(args)
    parsed = trainer.arguments(argv)
    trainer._validate_experimental_runtime(parsed, protocol("joint", "wide"))
    assert parsed.history_contract == "wide" and parsed.cross_cell_goal_mode == "zero"
    assert (parsed.batch, parsed.microbatch, parsed.steps) == (16, 2, 6000)


def test_temporal_array_generation_uses_exact_selected_offsets_and_pose_targets():
    frames = np.arange(0, 41, dtype=np.int64)
    times = frames.astype(np.float64) / 10.
    xyz = np.column_stack((times, np.zeros_like(times), np.zeros_like(times)))
    poses = data_module.full_pose_matrices(xyz, np.zeros((len(frames), 3)))
    cache_frames = np.asarray([30], np.int64)
    arrays = overlay_builder.temporal_arrays(cache_frames, [0], frames, times, poses,
                                             WIDE.frame_offsets)
    assert arrays["history_transforms"].dtype == np.float32
    assert np.array_equal(arrays["time_offsets"], np.asarray([WIDE.nominal_seconds], np.float32))


def make_overlay_object(tmp_path, *, frame=30, bad=None):
    scene = "scene"
    arrays = {"row": np.asarray([0], np.int64), "frame": np.asarray([frame], np.int64),
              "history_transforms": np.broadcast_to(np.eye(4, dtype=np.float32), (1, 4, 4, 4)).copy(),
              "history_target": np.zeros((1, 4, 4), np.float32),
              "history_valid": np.ones((1, 4, 4), np.bool_),
              "time_offsets": np.asarray([WIDE.nominal_seconds], np.float32)}
    if bad == "dtype":
        arrays["time_offsets"] = arrays["time_offsets"].astype(np.float64)
    path = tmp_path / f"{scene}.npz"
    np.savez(path, **arrays)
    spec = {"split": "train", "file": path.name, "sha256": p8.file_sha(path), "rows": 1,
            "rows_sha256": hash_array(np.asarray([0], dtype="<i8")),
            "arrays_sha256": {key: hash_array(value) for key, value in arrays.items()},
            "source_sha256": {"timestamps": "a" * 64, "ego_pose": "b" * 64}}
    if bad == "digest_missing":
        spec["arrays_sha256"].pop("time_offsets")
    obj = data_module.MotionDriveDataset.__new__(data_module.MotionDriveDataset)
    obj.history_overlay_manifest = {"artifacts": {scene: spec}}
    obj.history_overlay_root = tmp_path
    obj.split = "train"
    obj.scene_names = np.asarray([scene])
    obj.arr = {"frame": np.asarray([30])}
    obj._supervision = lambda _: {"row": np.asarray([0]), "frame": np.asarray([30])}
    return obj, scene


@pytest.mark.parametrize("bad", ["dtype", "digest_missing", "frame"])
def test_overlay_consumer_rejects_dtype_incomplete_digest_and_base_identity(tmp_path, bad):
    obj, scene = make_overlay_object(tmp_path, frame=31 if bad == "frame" else 30,
                                     bad=None if bad == "frame" else bad)
    with pytest.raises(ValueError):
        obj._history_overlay(scene)


def test_overlay_consumer_accepts_exact_six_array_contract(tmp_path):
    obj, scene = make_overlay_object(tmp_path)
    loaded = obj._history_overlay(scene)
    assert set(loaded) == {"row", "frame", "history_transforms", "history_target",
                           "history_valid", "time_offsets", "row_lookup"}


def getitem_dataset(contract_name, *, overlay):
    temporal = temporal_contract(contract_name)
    obj = data_module.MotionDriveDataset.__new__(data_module.MotionDriveDataset)
    obj.rows = np.asarray([2, 5, 9], np.int64)
    obj.scene_names = np.asarray(["scene"] * 10)
    obj.arr = {"frame": np.arange(30, 40), "scen_idx": np.zeros(10, np.int64),
               "goal": np.zeros((10, 2), np.float32), "fut": np.zeros((10, 6, 2), np.float32)}
    obj.history_offsets = np.asarray(temporal.frame_offsets)
    obj.history_contract = temporal
    obj.augment, obj.seed, obj.epoch = True, 17, 3
    obj.lidar2img = np.broadcast_to(np.eye(4, dtype=np.float32), (6, 4, 4)).copy()
    obj.manifest = {"scene_to_session": {"scene": "session"}}
    n = len(obj.rows)
    base = {"row": obj.rows.copy(), "frame": obj.arr["frame"][obj.rows],
            "frame_lookup": {int(obj.arr["frame"][row]): i for i, row in enumerate(obj.rows)},
            "history_transforms": np.stack([np.full((4, 4, 4), row, np.float32) for row in obj.rows]),
            "history_target": np.stack([np.full((4, 4), row, np.float32) for row in obj.rows]),
            "history_valid": np.ones((n, 4, 4), np.bool_),
            "time_offsets": np.tile(np.asarray(CONTROL.nominal_seconds, np.float32), (n, 1)),
            "state_target": np.zeros((n, 6), np.float32), "state_valid": np.ones((n, 6), np.bool_),
            "occ_target": np.zeros((n, 1, 64, 48), np.float32),
            "occ_valid": np.ones((n, 1, 64, 48), np.bool_),
            "lane_target": np.zeros((n, 1, 64, 48), np.float32),
            "lane_valid": np.ones((n, 1, 64, 48), np.bool_), "proxy_weight": np.ones(n)}
    if overlay:
        history = {key: np.asarray(base[key]).copy()
                   for key in ("row", "frame", "history_transforms", "history_target",
                               "history_valid", "time_offsets")}
        history["time_offsets"][:] = temporal.nominal_seconds
        history["row_lookup"] = {int(row): i for i, row in enumerate(obj.rows)}
        obj._history_overlay = lambda _: history
    else:
        obj._history_overlay = lambda _: None
    obj._supervision = lambda _: base

    def image(_scene, camera, frame, _size, jitter):
        # Includes the shared per-row/front jitter draw without allocating pixels.
        value = float(frame) + (float(np.sum(jitter)) if jitter is not None else 0.)
        return torch.full((3, 1, 1), value if camera == data_module.CAMERA_ORDER[0] else value + 100.)
    obj._image = image
    return obj


def test_control_overlay_getitem_matches_legacy_at_first_last_and_stride_middle():
    legacy = getitem_dataset("control", overlay=False)
    overlaid = getitem_dataset("control", overlay=True)
    for index in (0, 1, 2):
        old, new = legacy[index], overlaid[index]
        assert (old["row"], old["frame"]) == (new["row"], new["frame"])
        for key in ("history_transforms", "history_target", "history_valid", "time_offsets"):
            assert torch.equal(old[key], new[key])


def test_common_physical_history_frames_and_current_augmented_images_are_exact():
    control = getitem_dataset("control", overlay=True)[1]
    wide = getitem_dataset("wide", overlay=True)[1]
    assert torch.equal(control["images"], wide["images"])
    # C slots .2/.5/1s map to W slots .2/.5/1s with the same front jitter.
    assert torch.equal(control["history_images"][1:], wide["history_images"][:3])
    assert not torch.equal(control["history_images"][0], wide["history_images"][0])


def test_runtime_namespace_accepts_raw_or_prefixed_authorized_uuid(monkeypatch):
    args = SimpleNamespace(gpu=0, workers=4, cuda_memory_limit_mib=12000,
                           cuda_min_free_mib=8192, preflight_only=False,
                           base_seed=0, arm="control",
                           expected_physical_gpu_uuid=p8.GPU_ASSIGNMENTS[(0, "control")])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", args.expected_physical_gpu_uuid)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    raw = args.expected_physical_gpu_uuid.removeprefix("GPU-")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid=raw))
    assert p8.validate_runtime_namespace(args)["actual_physical_gpu_uuid"] == args.expected_physical_gpu_uuid
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(uuid=args.expected_physical_gpu_uuid))
    p8.validate_runtime_namespace(args)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="different"))
    with pytest.raises(ValueError, match="UUID mismatch"):
        p8.validate_runtime_namespace(args)


def test_runtime_namespace_rejects_swapped_but_authorized_gpu():
    args = SimpleNamespace(gpu=0, workers=4, cuda_memory_limit_mib=12000,
                           cuda_min_free_mib=8192, preflight_only=True,
                           base_seed=0, arm="control",
                           expected_physical_gpu_uuid=p8.GPU_ASSIGNMENTS[(0, "wide")])
    with pytest.raises(ValueError, match="counterbalanced"):
        p8.validate_runtime_namespace(args)


def test_source_closure_contains_new_contract_overlay_and_driver():
    assert {"models/motiondrive_v2_temporal_contract.py",
            "scripts/build_motiondrive_v2_history_overlay.py",
            "scripts/run_motiondrive_v2_p8_wide_history.py"} <= p8.SOURCE_FILES
    assert not any("p6" in name for name in p8.SOURCE_FILES)


def test_immutable_inventory_includes_every_overlay_artifact():
    args = SimpleNamespace(init="init", source_manifest="source", split_manifest="split",
                           supervision_root="sup", history_overlay_root="overlay",
                           init_manifest="p0_manifest")
    paths = p8.immutable_paths(args, {"history_overlay_artifact_paths": ["a.npz", "b.npz"]})
    assert paths[-3:] == ["a.npz", "b.npz", "p0_manifest"]


def test_preflight_forces_internal_overlay_validation_before_training():
    class Fake:
        rows = np.asarray([0])
        scene_names = np.asarray(["scene"])
        def __len__(self):
            return 1
        def _history_overlay(self, scene):
            raise ValueError(f"malformed {scene}")
    with pytest.raises(ValueError, match="malformed scene"):
        p8.validate_loaded_overlays((("train", Fake()),))


@pytest.fixture(scope="module")
def branchless_payload():
    trainer.seed_all(0)
    config = MotionDriveV2Config(backbone_arch="resnet50", goal_on=False, state_on=False,
                                 plan_output_scale=(10., 5.), motion_input_mode="low_feature",
                                 cross_cell_goal_mode="disabled")
    model = MotionDriveV2(config)
    saved = config.to_dict()
    for key in ("history_contract", "history_frame_offsets", "nominal_history_seconds"):
        saved.pop(key)
    payload = {"model": {key: value.detach().clone() for key, value in model.state_dict().items()},
               "optimizer": {"state": {}}, "step": 0,
               "manifest": {"model_config": saved, "arguments": {"phase": "pretrain"}}}
    del model
    return payload


def test_exact_public_style_legacy_state_can_seed_wide_p0_without_new_keys(branchless_payload):
    args = SimpleNamespace(stage="pretrain", arm="wide", base_seed=0,
                           expected_branch_state_sha256=None)
    prepared = p8.prepare_model(branchless_payload, args)
    assert prepared["expected_missing_state_keys"] == []
    assert prepared["branch_state_sha256"] is None
    assert prepared["initial_model_state_sha256"] == training.tensor_state_sha256(
        branchless_payload["model"])


def test_joint_adds_only_complete_zero_branch_and_places_params_in_head_group(branchless_payload):
    payload = copy.deepcopy(branchless_payload)
    temporal = temporal_contract("wide")
    payload["manifest"]["model_config"].update(
        history_contract="wide", history_frame_offsets=temporal.frame_offsets,
        nominal_history_seconds=temporal.nominal_seconds)
    payload["step"] = 2000
    trainer.seed_all(0)
    config = trainer.initialization_configuration(
        payload["manifest"], goal_on=1, state_on=1, explicit_arch="resnet50",
        cross_cell_goal_mode="zero", history_contract="wide")
    model = MotionDriveV2(MotionDriveV2Config(**config))
    incompatible = model.load_state_dict(payload["model"], strict=False)
    keys, expected_branch_sha = p8.branch_state_sha(model)
    assert set(keys) == set(incompatible.missing_keys) and not incompatible.unexpected_keys
    del model
    args = SimpleNamespace(stage="joint", arm="wide", base_seed=0,
                           expected_branch_state_sha256=expected_branch_sha)
    prepared = p8.prepare_model(payload, args)
    assert set(prepared["expected_missing_state_keys"]) == set(keys)
    assert prepared["branch_state_sha256"] == expected_branch_sha
    assert prepared["branch_optimizer_group"] == "head"
    assert prepared["branch_parameter_names"]


def test_joint_initializer_rejects_mixed_checkpoint_and_sidecar(tmp_path):
    payload = {"model": {"tiny": torch.ones(1)}, "optimizer": {}, "step": 2000}
    temporal = temporal_contract("wide")
    config = MotionDriveV2Config().to_dict()
    config.update(history_contract="wide", history_frame_offsets=temporal.frame_offsets,
                  nominal_history_seconds=temporal.nominal_seconds,
                  cross_cell_goal_mode="disabled", goal_on=False, state_on=False)
    embedded = {"git_sha": "1" * 40, "arguments": {"phase": "pretrain", "seed": 0},
                "model_config": config, "loss_weights": {"plan": 0.},
                "split_sha256": p8.EXPECTED_SPLIT_SHA256,
                "history_overlay_manifest_sha256": "a" * 64,
                "time_input": "nominal",
                "time_input_policy": training.time_input_policy("nominal", WIDE.nominal_seconds),
                "initial_model_state_sha256": p8.EXPECTED_I0_MODEL_SHA256,
                "load_report": {"common_checkpoint_sha256": p8.EXPECTED_I0_FILE_SHA256},
                "experimental_protocol": {"name": "p8_wide_history", "stage": "pretrain",
                                          "arm": "wide"}}
    payload.update(step=2000, manifest=embedded)
    checkpoint = tmp_path / "last.pth"
    torch.save(payload, checkpoint)
    sidecar = copy.deepcopy(embedded)
    sidecar.update(status="completed", step=2000)
    sidecar["loss_weights"] = {"plan": 0., "tampered": True}
    sidecar_path = tmp_path / "manifest.json"
    sidecar_path.write_text(json.dumps(sidecar))
    args = SimpleNamespace(stage="joint", arm="wide", base_seed=0, init=str(checkpoint),
        expected_init_sha256=p8.file_sha(checkpoint),
        expected_init_model_state_sha256=training.tensor_state_sha256(payload["model"]),
        init_manifest=str(sidecar_path), expected_init_manifest_sha256=p8.file_sha(sidecar_path))
    with pytest.raises(ValueError, match="embedded checkpoint and sidecar"):
        p8.validate_initializer(args)


def test_main_cpu_preflight_path_builds_complete_protocol_without_cuda(monkeypatch, capsys):
    branch_sha = "f" * 64
    monkeypatch.setattr(p8, "validate_runtime_namespace", lambda args: {"gpu_used": False})
    monkeypatch.setattr(p8, "validate_source_manifest", lambda *args: {"sha256": "s" * 64})
    monkeypatch.setattr(p8, "validate_data", lambda args: {
        "train_rows": 54810, "train_rows_sha256": p8.EXPECTED_TRAIN_ROWS_SHA256,
        "tune_rows": 1998, "tune_rows_sha256": p8.EXPECTED_TUNE_ROWS_SHA256,
        "history_overlay_artifact_paths": []})
    monkeypatch.setattr(p8, "validate_initializer", lambda args: (
        {"model": {}}, {"checkpoint_sha256": "b" * 64, "model_state_sha256": "c" * 64}))
    monkeypatch.setattr(p8, "prepare_model", lambda *args: {
        "initial_model_state_sha256": "d" * 64,
        "expected_missing_state_keys": ["scene_encoder.cross_cell_goal_residual.output.weight"],
        "branch_state_sha256": branch_sha})
    argv = ["--stage", "joint", "--arm", "wide", "--base-seed", "0",
            "--init", "p0", "--expected-init-sha256", "b" * 64,
            "--expected-init-model-state-sha256", "c" * 64,
            "--init-manifest", "manifest", "--expected-init-manifest-sha256", "e" * 64,
            "--expected-branch-state-sha256", branch_sha,
            "--source-manifest", "source", "--expected-source-manifest-sha256", "a" * 64,
            "--data-root", "data", "--split-manifest", "split",
            "--supervision-root", "sup", "--history-overlay-root", "overlay",
            "--expected-history-overlay-sha256", "a" * 64, "--run-dir", "run",
            "--expected-physical-gpu-uuid", p8.GPU_ASSIGNMENTS[(0, "wide")], "--preflight-only"]
    was_initialized = torch.cuda.is_initialized()
    p8.main(argv)
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "completed_cpu_preflight_only"
    assert output["experiment"]["expected_branch_state_sha256"] == branch_sha
    assert torch.cuda.is_initialized() == was_initialized
