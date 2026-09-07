"""CPU-only unit contracts; synthetic fixtures never stand in for real-data gates."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_grouped_split_v2 import (group_scenes, group_raw_scene_bounds,
                                    make_manifest, rebind_raw_sessions, sha256,
                                    validate_manifest)
from build_scene_supervision_v2 import (camera_visible, object_corners_world,
                                        rasterize_objects, rasterize_lanes,
                                        transform_points)
from motiondrive_v2_data import (CAMERA_ORDER, GRID_SHAPE, MotionDriveDataset,
                                 calibration_from_info, full_pose_matrices,
                                 grid_centers, motion_targets, verify_canonical_calibration)


def test_end_to_start_gap_and_quarantine():
    scenes = ["20260101-100000", "20260101-100130", "20260101-100301",
              "20260101-100500", "20260101-100700", "20260101-100900"]
    groups = group_scenes(scenes, 60)
    assert groups[0] == scenes[:2]
    assert groups[1] == scenes[2:3]
    m = make_manifest(scenes, [scenes[1]], tune_fraction=.2)
    validate_manifest(m)
    assert set(scenes[:2]) <= set(m["splits"]["val"])
    assert not set(m["splits"]["train"]) & set(m["splits"]["val"])
    m["splits"]["train"].append(scenes[0])
    with pytest.raises(ValueError, match="Overlapping"):
        validate_manifest(m)


def test_raw_timestamps_not_ambiguous_filename_and_cross_split_fail():
    names = ["20260205-03916", "20260205-03948", "20260205-04020",
             "20260205-100000", "20260205-110000", "20260205-120000"]
    bounds = {s: {"start_seconds": start, "end_seconds": start + 31.875}
              for s, start in zip(names, [0, 31.875, 63.75, 10000, 14000, 18000])}
    assert group_raw_scene_bounds(bounds)[0] == names[:3]
    previous = make_manifest(names, [names[-1]])
    previous["splits"] = {"train": names[:3], "tune": names[3:5], "val": names[5:], "historical_val": names[5:]}
    previous["scene_to_session"] = {s: s for s in names}
    fixed = rebind_raw_sessions(previous, bounds)
    assert fixed["splits"] == previous["splits"]
    assert len({fixed["scene_to_session"][s] for s in names[:3]}) == 1
    previous["splits"]["train"].remove(names[1])
    previous["splits"]["tune"].append(names[1])
    with pytest.raises(ValueError, match="CROSSES_PRIMARY_SPLITS"):
        rebind_raw_sessions(previous, bounds)


def test_motion_causal_constant_acceleration_and_full_se3():
    t = np.arange(21) * .1
    xyz = np.column_stack((2 * t + .5 * 3 * t ** 2, .2 * t, .1 * t))
    rpy = np.zeros((len(t), 3))
    poses = full_pose_matrices(xyz, rpy)
    got = motion_targets(poses, t, 20, [19, 18, 15, 10])
    np.testing.assert_allclose(got["state_target"][:5], [8, .2, 3, 0, 0], atol=2e-5)
    assert got["state_valid"].all()
    assert got["state_target"][-1] == 0
    np.testing.assert_allclose(got["time_offsets"], [.1, .2, .5, 1.])
    np.testing.assert_allclose(got["history_target"][:, 2:], np.tile([0, 1], (4, 1)))
    # Full alignment contains vertical displacement; old xy+yaw approximation did not.
    assert np.max(np.abs(got["history_transforms"][:, 2, 3])) > .09
    for h, idx in enumerate([19, 18, 15, 10]):
        identity = poses[idx] @ got["history_transforms"][h] @ np.linalg.inv(poses[20])
        np.testing.assert_allclose(identity, np.eye(4), atol=1e-6)
    changed = poses.copy()
    changed[20, :3, 3] += [1, 0, 0]
    with pytest.raises(ValueError, match="strictly causal"):
        motion_targets(changed, t, 10, [9, 8, 5, 11])


def test_future_samples_cannot_affect_state_targets():
    t = np.arange(31) * .1
    xyz = np.column_stack((3 * t, np.zeros_like(t), np.zeros_like(t)))
    poses = full_pose_matrices(xyz, np.zeros((31, 3)))
    a = motion_targets(poses, t, 20, [19, 18, 15, 10])
    poses[21:, :3, 3] = 9999
    b = motion_targets(poses, t, 20, [19, 18, 15, 10])
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_grid_axes_cell_centers_and_ego_box_exclusion():
    xy = grid_centers()
    assert xy.shape == (64, 48, 2)
    np.testing.assert_allclose(xy[0, 0], [-9.375, -31.33333333])
    rec = {"class": "Car", "x[m]": 10, "y[m]": 0, "z[m]": 0,
           "heading[rad]": 0, "width[m]": 4, "length[m]": 2, "num_points": 10}
    corners = object_corners_world(rec)
    np.testing.assert_allclose(np.ptp(corners, axis=0), [4, 2, 0])
    visible = np.ones(GRID_SHAPE, bool)
    target, valid = rasterize_objects([rec], np.eye(4), visible)
    assert (target.astype(bool) & valid).sum() > 0
    assert (~target.astype(bool) & valid).sum() > 0
    assert valid.mean() < .1
    target, valid = rasterize_objects([{**rec, "class": "ego"}], np.eye(4), visible)
    assert not target.any() and not valid.any()
    target, valid = rasterize_objects([{**rec, "num_points": 0}], np.eye(4), visible)
    assert not target.any() and not valid.any()


def test_invisible_and_unknown_are_not_negative_labels():
    rec = {"class": "Car", "x[m]": 10, "y[m]": 0, "z[m]": 0,
           "heading[rad]": 0, "width[m]": 4, "length[m]": 2, "num_points": 10}
    _, mask = rasterize_objects([rec], np.eye(4), np.zeros(GRID_SHAPE, bool))
    assert not mask.any()
    lines = [np.asarray([[0, 0, 0], [20, 0, 0]], float)]
    lane, valid = rasterize_lanes(lines, np.eye(4), np.ones(GRID_SHAPE, bool))
    assert lane.any() and valid.any()
    assert valid.mean() < .3


def test_calibration_inverse_convention_and_front_crop():
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler("xyz", [.2, -.1, .7]).as_matrix()
    cam = {"sensor2lidar_rotation": rot, "sensor2lidar_translation": [2, 1, .5],
           "cam_intrinsic": np.asarray([[1000, 0, 960], [0, 1000, 700], [0, 0, 1]]),
           "image_width": 1920, "image_height": 1536}
    got = calibration_from_info({"cams": {c: cam for c in CAMERA_ORDER}})
    point = np.asarray([10., 5, 2, 1])
    local = rot.T @ (point[:3] - cam["sensor2lidar_translation"])
    raw = cam["cam_intrinsic"] @ local
    expected = .4 * (raw[:2] / raw[2] - [0, 456])
    proj = got[0] @ point
    np.testing.assert_allclose(proj[:2] / proj[2], expected, atol=1e-4)
    rear = got[CAMERA_ORDER.index("camera_rear_wide")] @ point
    np.testing.assert_allclose(rear[:2] / rear[2], expected, atol=1e-4)
    side = got[CAMERA_ORDER.index("camera_front_left")] @ point
    np.testing.assert_allclose(side[:2] / side[2], .4 * raw[:2] / raw[2], atol=1e-4)


def test_derived_canonical_calibration_sha_is_required_and_verified(tmp_path):
    path = tmp_path / "calibration.npz"
    np.savez_compressed(path, lidar2img=np.tile(np.eye(4, dtype=np.float32), (6, 1, 1)))
    verify_canonical_calibration({"schema_version": 1}, path)
    verify_canonical_calibration({}, path)
    valid = {"schema_version": 2, "canonical_calibration_sha256": sha256(path)}
    verify_canonical_calibration(valid, path)
    with pytest.raises(ValueError, match="lacks canonical"):
        verify_canonical_calibration({"schema_version": 2}, path)
    for version in (1, 2):
        with pytest.raises(ValueError, match="provenance SHA mismatch"):
            verify_canonical_calibration({"schema_version": version,
                                          "canonical_calibration_sha256": "bad"}, path)
    np.savez_compressed(path, lidar2img=np.zeros((6, 4, 4), np.float32))
    with pytest.raises(ValueError, match="provenance SHA mismatch"):
        verify_canonical_calibration(valid, path)


def test_dataset_shapes_and_no_gt_state_alias(tmp_path):
    scenes = [f"20260101-{hour:02d}0000" for hour in range(10, 16)]
    manifest = make_manifest(scenes, [scenes[-1]])
    sp = tmp_path / "split.json"
    sp.write_text(json.dumps(manifest))
    scene = manifest["splits"]["train"][0]
    ec = tmp_path / "ego.npz"
    np.savez(ec, scenarios=np.asarray([scene]), scen_idx=[0], frame=[30],
             goal=np.zeros((1, 2), np.float32), fut=np.zeros((1, 6, 2), np.float32))
    sup = tmp_path / "sup"
    sup.mkdir()
    np.savez(sup / "calibration.npz", lidar2img=np.broadcast_to(np.eye(4), (6, 4, 4)))
    contract = dict(split_manifest_sha256=sha256(sp), ego_cache_sha256=sha256(ec),
                    grid_shape=[64, 48], grid_extent=[-10., 70., -32., 32.], history_frame_offsets=[1, 2, 5, 10])
    (sup / "supervision_manifest.json").write_text(json.dumps(contract))
    (sup / f"{scene}.json").write_text(json.dumps({**contract, "scene": scene}))
    arrays = dict(frame=[30], row=[0], history_transforms=np.broadcast_to(np.eye(4), (1, 4, 4, 4)).astype(np.float32),
                  time_offsets=np.asarray([[.1, .2, .5, 1]], np.float32),
                  history_target=np.zeros((1, 4, 4), np.float32), history_valid=np.ones((1, 4, 4), bool),
                  state_target=np.zeros((1, 6), np.float32), state_valid=np.ones((1, 6), bool))
    for name in ("occ", "lane"):
        arrays[name + "_target"] = np.zeros((1, 1, 64, 48), np.float32)
        arrays[name + "_valid"] = np.ones((1, 1, 64, 48), bool)
    np.savez(sup / f"{scene}.npz", **arrays)
    for camera in CAMERA_ORDER:
        folder = tmp_path / "images" / scene / camera
        folder.mkdir(parents=True)
        for frame in ([30, 29, 28, 25, 20] if camera == CAMERA_ORDER[0] else [30]):
            Image.new("RGB", (768, 432), color=(90, 120, 180)).save(folder / f"{frame:08d}.jpg")
    ds = MotionDriveDataset(tmp_path, sp, supervision_root=sup, ego_cache=ec,
                            image_root=tmp_path / "images", scenes=[scene])
    sample = ds[0]
    assert sample["images"].shape == (6, 3, 432, 768)
    assert sample["history_images"].shape == (4, 3, 216, 384)
    assert sample["history_transforms"].shape == (4, 4, 4)
    assert sample["state_target"].shape == (6,)
    assert "predicted_state" not in sample and "predicted_history" not in sample
    assert sample["occ_valid"].dtype.__str__() == "torch.bool"
    wrong = dict(contract, split_manifest_sha256="bad")
    (sup / "supervision_manifest.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="provenance SHA mismatch"):
        MotionDriveDataset(tmp_path, sp, supervision_root=sup, ego_cache=ec,
                           image_root=tmp_path / "images", scenes=[scene])
