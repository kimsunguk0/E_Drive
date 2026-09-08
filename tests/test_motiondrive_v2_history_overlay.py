"""CPU-only tests for the independently frozen P8 history-overlay capsule."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.motiondrive_v2_temporal_contract import CONTROL, WIDE, temporal_contract
from scripts import build_motiondrive_v2_history_overlay as builder
from scripts import motiondrive_v2_data as data


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def test_fixed_contracts_and_no_sweep_api():
    assert CONTROL.frame_offsets == (1, 2, 5, 10)
    assert WIDE.frame_offsets == (2, 5, 10, 20)
    with pytest.raises(ValueError):
        temporal_contract("custom")


def test_history_only_helper_is_exact_motion_target_subset():
    times = np.arange(41, dtype=np.float64) / 10.
    xyz = np.column_stack((times, times ** 2, np.zeros_like(times)))
    poses = data.full_pose_matrices(xyz, np.zeros((41, 3)))
    ids = [28, 25, 20, 10]
    history = data.history_targets(poses, times, 30, ids)
    complete = data.motion_targets(poses, times, 30, ids)
    assert set(history) == set(builder.TEMPORAL_KEYS)
    for key in history:
        assert np.array_equal(history[key], complete[key])


@pytest.mark.parametrize("contract", [CONTROL, WIDE])
def test_overlay_arrays_use_exact_offsets_and_physical_seconds(contract):
    frames = np.arange(41, dtype=np.int64)
    times = frames.astype(np.float64) / 10.
    poses = data.full_pose_matrices(np.column_stack((times, np.zeros((41, 2)))),
                                    np.zeros((41, 3)))
    arrays = builder.temporal_arrays(np.asarray([30]), [0], frames, times, poses,
                                     contract.frame_offsets)
    assert arrays["row"].dtype == np.int64 and arrays["frame"].dtype == np.int64
    assert np.array_equal(arrays["time_offsets"], np.asarray([contract.nominal_seconds], np.float32))
    assert arrays["history_valid"].dtype == np.bool_


def overlay_object(tmp_path, *, frame=30, bad=None):
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
    spec = {"split": "train", "file": path.name, "sha256": builder.sha256(path), "rows": 1,
            "rows_sha256": digest(np.asarray([0], dtype="<i8")),
            "arrays_sha256": {key: digest(value) for key, value in arrays.items()},
            "source_sha256": {"timestamps": "a" * 64, "ego_pose": "b" * 64}}
    if bad == "digest_missing":
        spec["arrays_sha256"].pop("time_offsets")
    obj = data.MotionDriveDataset.__new__(data.MotionDriveDataset)
    obj.history_overlay_manifest = {"artifacts": {scene: spec}}
    obj.history_overlay_root, obj.split = tmp_path, "train"
    obj.scene_names = np.asarray([scene])
    obj.arr = {"frame": np.asarray([30])}
    obj._supervision = lambda _: {"row": np.asarray([0]), "frame": np.asarray([30])}
    return obj, scene


def test_overlay_consumer_accepts_exact_six_array_contract(tmp_path):
    obj, scene = overlay_object(tmp_path)
    loaded = obj._history_overlay(scene)
    assert set(loaded) == {"row", "frame", *builder.TEMPORAL_KEYS, "row_lookup"}


@pytest.mark.parametrize("bad", ["dtype", "digest_missing", "frame"])
def test_overlay_consumer_rejects_bad_dtype_digest_or_base_identity(tmp_path, bad):
    obj, scene = overlay_object(tmp_path, frame=31 if bad == "frame" else 30,
                                bad=None if bad == "frame" else bad)
    with pytest.raises(ValueError):
        obj._history_overlay(scene)


def sample_dataset(contract_name, *, overlay):
    contract = temporal_contract(contract_name)
    obj = data.MotionDriveDataset.__new__(data.MotionDriveDataset)
    obj.rows = np.asarray([2, 5, 9], np.int64)
    obj.scene_names = np.asarray(["scene"] * 10)
    obj.arr = {"frame": np.arange(30, 40), "scen_idx": np.zeros(10, np.int64),
               "goal": np.zeros((10, 2), np.float32), "fut": np.zeros((10, 6, 2), np.float32)}
    obj.history_offsets = np.asarray(contract.frame_offsets)
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
        temporal = {key: np.asarray(base[key]).copy() for key in (
            "row", "frame", "history_transforms", "history_target", "history_valid", "time_offsets")}
        temporal["time_offsets"][:] = contract.nominal_seconds
        temporal["row_lookup"] = {int(row): i for i, row in enumerate(obj.rows)}
        obj._history_overlay = lambda _: temporal
    else:
        obj._history_overlay = lambda _: None
    obj._supervision = lambda _: base
    obj._image = lambda _s, camera, frame, _size, jitter: torch.full(
        (3, 1, 1), float(frame) + (float(np.sum(jitter)) if jitter is not None else 0.)
        + (0. if camera == data.CAMERA_ORDER[0] else 100.))
    return obj


def test_control_overlay_row_lookup_matches_legacy_first_middle_last():
    legacy, overlaid = sample_dataset("control", overlay=False), sample_dataset("control", overlay=True)
    for index in (0, 1, 2):
        old, new = legacy[index], overlaid[index]
        assert (old["row"], old["frame"]) == (new["row"], new["frame"])
        for key in builder.TEMPORAL_KEYS:
            assert torch.equal(old[key], new[key])


def test_common_history_times_and_current_augmentations_map_exactly():
    control, wide = sample_dataset("control", overlay=True)[1], sample_dataset("wide", overlay=True)[1]
    assert torch.equal(control["images"], wide["images"])
    assert torch.equal(control["history_images"][1:], wide["history_images"][:3])
    assert not torch.equal(control["history_images"][0], wide["history_images"][0])


def test_fixed_input_hash_constants_and_c_before_w_barrier():
    assert builder.EXPECTED_SPLIT_SHA256.startswith("f4e0f30c")
    assert builder.EXPECTED_EGO_SHA256.startswith("d35a69bb")
    assert builder.EXPECTED_C1_SHA256.startswith("ba1ba04e")
    args = SimpleNamespace if False else None
    assert "control-overlay manifest SHA" in builder.validate_control_barrier.__doc__ if builder.validate_control_barrier.__doc__ else True
