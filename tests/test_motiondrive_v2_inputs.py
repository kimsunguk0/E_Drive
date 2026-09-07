"""Synthetic CPU contract tests; actual train-cache parity is a separate gate."""
import copy
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import motiondrive_v2_inputs as adapter


def calibration_rows():
    return [{"camera_name": name, "K": [1100., 0., 960., 0., 1100., 768., 0., 0., 1.],
             "distortion": [0., 0., 0., 0., 0.], "is_fisheye": False,
             "image_width": 1920, "image_height": 1536, "euler": [2., -3., i * 15.],
             "translation": [1.2, -.3, 1.5]} for i, name in enumerate(adapter.CAMERA_ORDER)]


def pose_rows():
    return [{"frame": frame, "x": 8. + frame * .25, "y": -4. + frame * .02,
             "z": 2. + frame * .01, "roll": .07 + frame * .0002,
             "pitch": -.03 + frame * .0001, "yaw": .4 + frame * .001}
            for frame in sorted(adapter.POSE_FRAMES)]


@pytest.fixture(scope="module")
def synthetic_pixels():
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        ys = np.arange(1536, dtype=np.uint16)[:, None]
        xs = np.arange(1920, dtype=np.uint16)[None, :]
        raw = np.empty((1536, 1920, 3), dtype=np.uint8)
        raw[..., 0] = (xs // 7 + ys // 17) % 256
        raw[..., 1] = (ys // 5) % 256
        raw[..., 2] = (xs // 11) % 256
        ok, encoded = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 92])
        assert ok
        yield encoded.tobytes()
    finally:
        cv2.setNumThreads(old_threads)


@pytest.fixture(scope="module")
def prepared(synthetic_pixels):
    reads = []

    def loader(camera, frame):
        reads.append((camera, frame))
        return synthetic_pixels

    output = adapter.prepare_clip_from_records(calibration_rows(), pose_rows(), loader)
    return output, reads


def test_prepare_has_exact_six_inputs_ten_images_shapes_and_nominal_dt(prepared):
    result, reads = prepared
    expected_reads = [(camera, 0) for camera in adapter.CAMERA_ORDER]
    expected_reads += [("camera_front", frame) for frame in (-1, -2, -5, -10)]
    assert reads == expected_reads
    assert set(result.inputs) == set(adapter.INPUT_KEYS)
    shapes = {"images": (1, 6, 3, 432, 768), "history_images": (1, 4, 3, 216, 384),
              "lidar2img": (1, 6, 4, 4), "history_transforms": (1, 4, 4, 4),
              "time_offsets": (1, 4), "goal_xy": (1, 2)}
    for name, tensor in result.inputs.items():
        assert tuple(tensor.shape) == shapes[name]
        assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
        assert not tensor.requires_grad and torch.isfinite(tensor).all()
    assert torch.equal(result.inputs["time_offsets"], torch.tensor([[.1, .2, .5, 1.]]))
    assert len(result.metadata["images"]) == 10
    assert result.metadata["input_contract"]["geometry_edition"] == "geometry_v2"
    assert result.metadata["input_contract"]["time_input"] == "nominal"


def test_actual_pipeline_matches_independent_q95_pil_reference(prepared, synthetic_pixels):
    result, _ = prepared
    row = calibration_rows()[0]
    raw = cv2.imdecode(np.frombuffer(synthetic_pixels, np.uint8), cv2.IMREAD_COLOR)
    k = np.array(row["K"]).reshape(3, 3)
    distortion = np.array(row["distortion"])
    new_k = cv2.getOptimalNewCameraMatrix(k, distortion, (1920, 1536), alpha=0)[0]
    maps = cv2.initUndistortRectifyMap(k, distortion, None, new_k, (1920, 1536), cv2.CV_16SC2)
    corrected = cv2.remap(raw, *maps, cv2.INTER_LINEAR)
    for index, y in ((0, 456), (1, 0), (3, 456)):
        resized = cv2.resize(corrected[y:y + 1080, :1920], (768, 432), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        image = Image.open(io.BytesIO(encoded.tobytes())).convert("RGB")
        pixels = np.asarray(image, np.float32) / 255.
        expected = torch.from_numpy(((pixels - np.array([.485, .456, .406], np.float32)) /
                                     np.array([.229, .224, .225], np.float32)).transpose(2, 0, 1).copy())
        assert torch.equal(result.inputs["images"][0, index], expected)
        if index == 0:
            pixels = np.asarray(image.resize((384, 216), Image.Resampling.BILINEAR), np.float32) / 255.
            expected_history = torch.from_numpy(((pixels - adapter.MEAN) / adapter.STD).transpose(2, 0, 1).copy())
            assert all(torch.equal(result.inputs["history_images"][0, j], expected_history) for j in range(4))
    assert not torch.equal(result.inputs["images"][0, 0], result.inputs["images"][0, 1])


def test_projection_uses_degrees_full_extrinsic_and_both_bottom_crops(prepared):
    result, _ = prepared
    for i, row in enumerate(calibration_rows()):
        name = row["camera_name"]
        meta = result.metadata["camera_geometry"][name]
        y = 456 if i in (0, 3) else 0
        assert meta["crop_xy"] == [0, y]
        camera_to_ego = np.eye(4)
        camera_to_ego[:3, :3] = Rotation.from_euler("xyz", row["euler"], degrees=True).as_matrix()
        camera_to_ego[:3, 3] = row["translation"]
        pixel_shift = np.eye(4)
        pixel_shift[1, 2] = -y
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = meta["new_K"]
        expected = (np.diag([.4, .4, 1., 1.]) @ pixel_shift @ intrinsic @ np.linalg.inv(camera_to_ego)).astype(np.float32)
        assert np.array_equal(result.inputs["lidar2img"][0, i].numpy(), expected)


def test_pose_full_se3_current_to_past_and_goal_no_yaw_only_shortcut():
    rows = pose_rows()
    alignment, goal = adapter.pose_geometry(rows)
    matrices = {}
    for row in rows:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler("xyz", [row[k] for k in ("roll", "pitch", "yaw")]).as_matrix()
        matrix[:3, 3] = [row[k] for k in ("x", "y", "z")]
        matrices[row["frame"]] = matrix
    for i, frame in enumerate(adapter.PAST_FRAMES):
        np.testing.assert_allclose(matrices[frame] @ alignment[i], matrices[0], atol=1e-6, rtol=0)
    goal_reference = (np.linalg.inv(matrices[0]) @ matrices[50])[:2, 3]
    np.testing.assert_allclose(goal, goal_reference, atol=1e-6, rtol=0)
    yaw_only = Rotation.from_euler("z", .4).as_matrix().T @ (matrices[50][:3, 3] - matrices[0][:3, 3])
    assert np.max(np.abs(goal - yaw_only[:2])) > .005


def test_future_orientation_and_unused_past_rows_cannot_change_model_geometry():
    rows = pose_rows()
    expected_h, expected_goal = adapter.pose_geometry(rows)
    modified = copy.deepcopy(rows)
    for row in modified:
        if row["frame"] == 50:
            row.update(roll=float("nan"), pitch=float("nan"), yaw=float("nan"))
        elif row["frame"] not in (*adapter.PAST_FRAMES, 0):
            row.update(x=1e8, y=-1e8, z=1e8, roll=1e3, pitch=1e3, yaw=1e3)
    h, goal = adapter.pose_geometry(modified)
    assert np.array_equal(h, expected_h) and np.array_equal(goal, expected_goal)
    modified[-1]["x"] += 2.
    new_h, new_goal = adapter.pose_geometry(modified)
    assert np.array_equal(new_h, expected_h) and not np.array_equal(new_goal, expected_goal)


@pytest.mark.parametrize("change", ["timestamp", "status", "future3s", "missing", "duplicate"])
def test_pose_rejects_extra_fields_future_labels_or_nonofficial_rows(change):
    rows = pose_rows()
    if change in ("timestamp", "status"):
        rows[0][change] = 123.
    elif change == "future3s":
        rows[0]["frame"] = 30
    elif change == "missing":
        rows.pop()
    else:
        rows[0]["frame"] = rows[1]["frame"]
    with pytest.raises(ValueError):
        adapter.pose_geometry(rows)


@pytest.mark.parametrize("field,value", [("camera_name", "camera_unknown"), ("K", [0.]),
                                        ("image_width", 1800), ("is_fisheye", "False")])
def test_invalid_calibration_fails_closed(field, value):
    rows = calibration_rows()
    rows[0][field] = value
    with pytest.raises(ValueError):
        adapter.build_camera_geometry(rows)


def test_fisheye_geometry_and_corrupt_image_failures(synthetic_pixels):
    rows = calibration_rows()
    rows[0]["is_fisheye"] = True
    cameras = adapter.build_camera_geometry(rows)
    expected = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        np.array(rows[0]["K"]).reshape(3, 3), np.zeros(4), (1920, 1536), np.eye(3), balance=0.)
    assert np.array_equal(cameras[0].new_intrinsic, expected)
    with pytest.raises(ValueError, match="Invalid encoded"):
        adapter.cache_compatible_image(b"not-an-image", cameras[0])
    with pytest.raises(TypeError, match="JPEG bytes"):
        adapter.cache_compatible_image(np.zeros(10), cameras[0])
    tiny = cv2.imencode(".jpg", np.zeros((4, 4, 3), np.uint8))[1].tobytes()
    with pytest.raises(ValueError, match="dimensions"):
        adapter.cache_compatible_image(tiny, cameras[0])


def test_parity_separates_bitwise_pixels_from_rounded_matrix_tolerance(prepared):
    result, _ = prepared
    reference = {name: tensor.clone() for name, tensor in result.inputs.items()}
    fixture = {"inputs": reference, "metadata": {"input_contract": adapter.input_contract()}}
    report = adapter.compare_input_fixture(result, fixture)
    assert report["all_pass"] and all(row["bitwise_equal"] for row in report["inputs"].values())
    # A real archived-canonical effect can be about 5.8e-5; this is not pixel equality.
    reference["lidar2img"][0, 0, 0, 0] += 6e-5
    report = adapter.compare_input_fixture(result, fixture)
    assert report["all_pass"] and not report["inputs"]["lidar2img"]["bitwise_equal"]
    assert not adapter.compare_input_fixture(result, fixture, projection_atol=0)["all_pass"]
    reference["images"][0, 0, 0, 0, 0] += .001
    report = adapter.compare_input_fixture(result, fixture)
    assert not report["all_pass"] and not report["inputs"]["images"]["pass"]


def test_parity_bitwise_claim_distinguishes_signed_zero(prepared):
    result, _ = prepared
    local = adapter.PreparedClip(dict(result.inputs), result.metadata)
    local.inputs["images"] = torch.zeros_like(local.inputs["images"])
    reference = dict(local.inputs)
    reference["images"] = -local.inputs["images"]
    assert torch.equal(reference["images"], local.inputs["images"])
    report = adapter.compare_input_fixture(local, {"inputs": reference,
        "metadata": {"input_contract": adapter.input_contract()}})
    assert not report["all_pass"]
    assert report["inputs"]["images"]["max_abs"] == 0.
    assert not report["inputs"]["images"]["bitwise_equal"]


def test_stateless_interleaved_clip_preparation_a_b_a(prepared, synthetic_pixels):
    a, _ = prepared
    rows_b = pose_rows()
    rows_b[-1]["x"] += 3.
    b = adapter.prepare_clip_from_records(calibration_rows(), rows_b, lambda *_: synthetic_pixels)
    a_repeat = adapter.prepare_clip_from_records(calibration_rows(), pose_rows(), lambda *_: synthetic_pixels)
    for name in adapter.INPUT_KEYS:
        assert torch.equal(a.inputs[name], a_repeat.inputs[name])
        assert (not torch.equal(a.inputs[name], b.inputs[name])) if name == "goal_xy" else torch.equal(a.inputs[name], b.inputs[name])


@pytest.mark.parametrize("change", ["old_geometry", "raw_time", "labels", "missing_contract"])
def test_parity_rejects_legacy_or_labeled_fixture(prepared, change):
    result, _ = prepared
    fixture = {"inputs": dict(result.inputs), "metadata": {"input_contract": adapter.input_contract()}}
    if change == "old_geometry":
        fixture["metadata"]["input_contract"]["geometry_edition"] = "geometry_v1"
    elif change == "raw_time":
        fixture["metadata"]["input_contract"]["time_input"] = "raw"
    elif change == "labels":
        fixture["inputs"]["gt_plan"] = torch.zeros(1, 6, 2)
    else:
        fixture["metadata"] = {}
    with pytest.raises(ValueError):
        adapter.compare_input_fixture(result, fixture)


def test_official_directory_reads_whitelisted_tables_and_only_ten_jpegs(tmp_path, monkeypatch, synthetic_pixels):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    clip = tmp_path / "fixture_000"
    clip.mkdir()
    calib_path, pose_path = clip / "calibration.parquet", clip / "ego_pose.parquet"
    pq.write_table(pa.Table.from_pylist(calibration_rows()), calib_path)
    poses = pose_rows()
    # Such optional columns are not loaded; no timestamp is required or forwarded.
    for row in poses:
        row["timestamp"] = "must not reach adapter"
        row["status"] = "must not reach adapter"
    pq.write_table(pa.Table.from_pylist(poses), pose_path)
    expected_files = {calib_path, pose_path}
    for camera, frame in [(camera, 0) for camera in adapter.CAMERA_ORDER] + [("camera_front", frame) for frame in adapter.PAST_FRAMES]:
        directory = clip / camera
        directory.mkdir(exist_ok=True)
        path = directory / f"frame_{frame}.jpg"
        path.write_bytes(synthetic_pixels)
        expected_files.add(path)
    (clip / "command.parquet").write_bytes(b"invalid; must never be opened")
    (clip / "annotations.parquet").write_bytes(b"invalid; must never be opened")
    reads, table_reads = [], []
    original_bytes, original_table = Path.read_bytes, pq.read_table

    def checked_bytes(path):
        if path.is_relative_to(clip):
            assert path in expected_files
            reads.append(path)
        return original_bytes(path)

    def checked_table(path, *, columns):
        table_reads.append((Path(path).name, tuple(columns)))
        return original_table(path, columns=columns)

    monkeypatch.setattr(Path, "read_bytes", checked_bytes)
    monkeypatch.setattr(pq, "read_table", checked_table)
    result = adapter.prepare_clip_inputs(clip)
    assert table_reads == [("calibration.parquet", adapter.CALIBRATION_COLUMNS), ("ego_pose.parquet", adapter.POSE_COLUMNS)]
    assert len([path for path in reads if path.suffix == ".jpg"]) == 10
    assert result.metadata["clip_id"] == "fixture_000"
    assert set(result.inputs) == set(adapter.INPUT_KEYS)
    assert result.metadata["input_contract"]["provided_status_or_timestamp_input"] is False
