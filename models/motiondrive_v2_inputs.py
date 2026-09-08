"""GT-free, CPU-only input adapter for one official raw ETRI test-shaped clip.

This module does not import a dataset, load a checkpoint, run a network, or emit
a submission. Its only forward inputs are six normalized images, four earlier
front images, calibration, scene-alignment transforms, nominal time offsets and
the provided goal. Pose-derived state/history targets are deliberately absent.

Raw calibration reconstruction is mathematically equivalent to geometry_v2,
but is not claimed bitwise equal to matrices rounded in older pickle/NPZ files.
The cache pixel pipeline includes its JPEG-quality-95 encode/PIL-decode step.
Actual pixel parity must still be checked with the serving OpenCV/PIL build.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, __version__ as PIL_VERSION
from scipy.spatial.transform import Rotation
import torch

from .motiondrive_v2_input_contract import (
    CAMERA_ORDER, PAST_FRAMES, POSE_FRAMES, NOMINAL_SECONDS, INPUT_KEYS,
    CURRENT_WH, HISTORY_WH, CROP_WH, input_contract, require_input_contract,
)
from .motiondrive_v2_temporal_contract import temporal_contract

CALIBRATION_COLUMNS = ("camera_name", "K", "distortion", "is_fisheye", "image_width",
                       "image_height", "euler", "translation")
POSE_COLUMNS = ("frame", "x", "y", "z", "roll", "pitch", "yaw")
MEAN = np.asarray([.485, .456, .406], np.float32)
STD = np.asarray([.229, .224, .225], np.float32)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite_array(value, shape, name):
    array = np.asarray(value, np.float64)
    if array.size != int(np.prod(shape)):
        raise ValueError(f"{name} must have shape {shape}")
    array = array.reshape(shape)
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite {name}")
    return array


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value != integer:
        raise ValueError(f"{name} must be an integer")
    return integer


@dataclass
class CameraGeometry:
    name: str
    raw_wh: tuple[int, int]
    crop_xy: tuple[int, int]
    new_intrinsic: np.ndarray
    cached_intrinsic: np.ndarray
    lidar2img: np.ndarray
    map1: np.ndarray
    map2: np.ndarray


def build_camera_geometry(records: Sequence[Mapping]) -> tuple[CameraGeometry, ...]:
    """Calibration Euler xyz is in degrees; camera translation is in ego metres."""
    records = list(records)
    if len(records) != 6:
        raise ValueError("Exactly six camera calibration rows are required")
    rows = {}
    for row in records:
        if set(row) != set(CALIBRATION_COLUMNS):
            raise ValueError("Calibration records must contain only the official calibration columns")
        name = row["camera_name"]
        if name not in CAMERA_ORDER or name in rows:
            raise ValueError(f"Unknown or duplicate calibration camera: {name}")
        rows[name] = row
    result = []
    for name in CAMERA_ORDER:
        row = rows[name]
        width, height = (_integer(row[k], k) for k in ("image_width", "image_height"))
        if width < CROP_WH[0] or height < CROP_WH[1]:
            raise ValueError("Raw image is smaller than the fixed 1920x1080 crop")
        intrinsic = _finite_array(row["K"], (3, 3), "K")
        distortion = np.asarray(row["distortion"], np.float64).reshape(-1)
        if not np.isfinite(distortion).all():
            raise ValueError("Nonfinite camera distortion")
        if not isinstance(row["is_fisheye"], (bool, np.bool_)):
            raise ValueError("is_fisheye must be a boolean")
        fisheye = bool(row["is_fisheye"])
        if len(distortion) < 4 or (not fisheye and len(distortion) not in (4, 5, 8, 12, 14)):
            raise ValueError("Unsupported camera distortion coefficients")
        size = (width, height)
        if fisheye:
            new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                intrinsic, distortion[:4], size, np.eye(3), balance=0.)
            maps = cv2.fisheye.initUndistortRectifyMap(
                intrinsic, distortion[:4], np.eye(3), new_k, size, cv2.CV_16SC2)
        else:
            new_k = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, size, alpha=0)[0]
            maps = cv2.initUndistortRectifyMap(intrinsic, distortion, None, new_k, size, cv2.CV_16SC2)
        if not np.isfinite(new_k).all():
            raise ValueError("Undistortion produced nonfinite intrinsics")
        camera_to_ego = np.eye(4)
        camera_to_ego[:3, :3] = Rotation.from_euler(
            "xyz", _finite_array(row["euler"], (3,), "camera euler"), degrees=True).as_matrix()
        camera_to_ego[:3, 3] = _finite_array(row["translation"], (3,), "camera translation")
        crop_xy = ((width - CROP_WH[0]) // 2,
                   height - CROP_WH[1] if name in ("camera_front", "camera_rear_wide") else 0)
        crop = np.eye(4)
        crop[0, 2], crop[1, 2] = -crop_xy[0], -crop_xy[1]
        k4 = np.eye(4)
        k4[:3, :3] = new_k
        resize = np.diag([CURRENT_WH[0] / CROP_WH[0], CURRENT_WH[1] / CROP_WH[1], 1., 1.])
        projection = resize @ crop @ k4 @ np.linalg.inv(camera_to_ego)
        cached_k = (resize @ crop @ k4)[:3, :3]
        result.append(CameraGeometry(name, size, crop_xy, new_k, cached_k,
                                     projection.astype(np.float32), *maps))
    return tuple(result)


def pose_geometry(records: Sequence[Mapping], history_contract: str = "control") -> tuple[np.ndarray, np.ndarray]:
    """Read only current/past RPY and goal XYZ; future orientation is never used."""
    rows = {}
    for row in records:
        if set(row) != set(POSE_COLUMNS):
            raise ValueError("Pose records must contain only frame/x/y/z/roll/pitch/yaw; no status or timestamps")
        frame = _integer(row["frame"], "frame")
        if frame in rows:
            raise ValueError(f"Duplicate pose frame: {frame}")
        rows[frame] = row
    if set(rows) != POSE_FRAMES:
        raise ValueError("Official pose table must contain exactly frames -30..0 and +50")

    def pose(frame):
        row = rows[frame]
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler("xyz", _finite_array(
            [row[k] for k in ("roll", "pitch", "yaw")], (3,), "ego RPY radians")).as_matrix()
        matrix[:3, 3] = _finite_array([row[k] for k in ("x", "y", "z")], (3,), "ego XYZ")
        return matrix

    current = pose(0)
    temporal = temporal_contract(history_contract)
    past_frames = tuple(-offset for offset in temporal.frame_offsets)
    past = np.stack([pose(frame) for frame in past_frames])
    alignment = np.linalg.inv(past) @ current
    goal_xyz = _finite_array([rows[50][k] for k in ("x", "y", "z")], (3,), "provided goal XYZ")
    goal = (current[:3, :3].T @ (goal_xyz - current[:3, 3]))[:2]
    return alignment.astype(np.float32), goal.astype(np.float32)


def cache_compatible_image(raw_jpeg: bytes, geometry: CameraGeometry) -> tuple[Image.Image, dict]:
    """Recreate the unaugmented 768 cache in memory; never write a derived image."""
    if not isinstance(raw_jpeg, bytes):
        raise TypeError("Image loader must return encoded JPEG bytes")
    raw = cv2.imdecode(np.frombuffer(raw_jpeg, np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError("Invalid encoded JPEG")
    if raw.shape[:2] != geometry.raw_wh[::-1]:
        raise ValueError(f"Raw JPEG dimensions disagree with calibration for {geometry.name}")
    undistorted = cv2.remap(raw, geometry.map1, geometry.map2, cv2.INTER_LINEAR)
    x, y = geometry.crop_xy
    cropped = undistorted[y:y + CROP_WH[1], x:x + CROP_WH[0]]
    resized = cv2.resize(cropped, CURRENT_WH, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Q95 JPEG encoding failed")
    encoded_bytes = encoded.tobytes()
    with Image.open(io.BytesIO(encoded_bytes)) as image:
        rgb = image.convert("RGB")
    return rgb, {"raw_jpeg_sha256": _sha(raw_jpeg), "reconstructed_cache_jpeg_sha256": _sha(encoded_bytes),
                 "decoded_cache_rgb_sha256": _sha(np.asarray(rgb, np.uint8).tobytes())}


def normalize_image(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    if image.size != CURRENT_WH:
        raise ValueError("Expected the decoded 768x432 cache image")
    if size not in (CURRENT_WH, HISTORY_WH):
        raise ValueError("Only current768/history384 image sizes are supported")
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    x = np.asarray(image, np.float32) / 255.
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy())


@dataclass
class PreparedClip:
    inputs: dict[str, torch.Tensor]
    metadata: dict


def prepare_clip_from_records(calibration_rows: Sequence[Mapping], ego_pose_rows: Sequence[Mapping],
                              image_loader: Callable[[str, int], bytes],
                              history_contract: str = "control") -> PreparedClip:
    """Fixture-friendly core. The loader is called for exactly ten allowed images.

    No arbitrary model-input dictionary or future label is accepted. There is no
    persistent clip state: interleaving calls A/B/A cannot reuse neural features.
    """
    temporal = temporal_contract(history_contract)
    alignments, goal = pose_geometry(ego_pose_rows, temporal.name)
    cameras = build_camera_geometry(calibration_rows)
    pixels, history, image_provenance = [], [], {}
    for camera in cameras:
        image, provenance = cache_compatible_image(image_loader(camera.name, 0), camera)
        pixels.append(normalize_image(image, CURRENT_WH))
        image_provenance[f"{camera.name}/frame_0.jpg"] = provenance
    for frame in (-offset for offset in temporal.frame_offsets):
        image, provenance = cache_compatible_image(image_loader(CAMERA_ORDER[0], frame), cameras[0])
        history.append(normalize_image(image, HISTORY_WH))
        image_provenance[f"{CAMERA_ORDER[0]}/frame_{frame}.jpg"] = provenance
    inputs = {
        "images": torch.stack(pixels).unsqueeze(0),
        "history_images": torch.stack(history).unsqueeze(0),
        # Historical sampling uses this SAME current768 geometry; do not halve K again.
        "lidar2img": torch.from_numpy(np.stack([camera.lidar2img for camera in cameras])).unsqueeze(0),
        "history_transforms": torch.from_numpy(alignments).unsqueeze(0),
        "time_offsets": torch.tensor([temporal.nominal_seconds], dtype=torch.float32),
        "goal_xy": torch.from_numpy(goal).unsqueeze(0),
    }
    if any(tensor.dtype != torch.float32 or not torch.isfinite(tensor).all() for tensor in inputs.values()):
        raise ValueError("Prepared model inputs must be finite CPU float32")
    metadata = {"input_contract": input_contract(temporal.name), "cpu_only": True,
                "opencv_version": cv2.__version__, "pillow_version": PIL_VERSION,
                "adapter_source_sha256": _sha(Path(__file__).read_bytes()),
                "images": image_provenance,
                "camera_geometry": {camera.name: {"raw_wh": list(camera.raw_wh),
                    "crop_xy": list(camera.crop_xy), "new_K": camera.new_intrinsic.tolist(),
                    "K_cache": camera.cached_intrinsic.tolist()} for camera in cameras},
                "limitations": ["Pixel hashes are reconstruction evidence, not proof of an unprovided cache reference.",
                                "Raw-calibration projection can differ from archived rounded NPZ by about 5.8e-5; compare explicitly.",
                                "No model forward, inferred-state check, latency measurement or submission is performed."]}
    return PreparedClip(inputs, metadata)


def prepare_clip_inputs(clip_dir: str | Path, history_contract: str = "control") -> PreparedClip:
    """Read official test-shaped files only; command/status/annotations are ignored.

    Parquet projection reads only these named columns, not any optional timestamp
    or label column. Direct record callers must supply the same strict schema.
    The directory may contain all 31-frame images; only the ten required JPEGs
    are accessed. No recursive discovery or training-dataset loader is used.
    """
    import pyarrow.parquet as pq
    directory = Path(clip_dir)
    calibration_path, pose_path = directory / "calibration.parquet", directory / "ego_pose.parquet"
    calibration = pq.read_table(calibration_path, columns=list(CALIBRATION_COLUMNS)).to_pylist()
    poses = pq.read_table(pose_path, columns=list(POSE_COLUMNS)).to_pylist()

    def read_image(camera, frame):
        return (directory / camera / f"frame_{frame}.jpg").read_bytes()

    result = prepare_clip_from_records(calibration, poses, read_image, history_contract)
    result.metadata.update(clip_id=directory.name, clip_dir=str(directory.resolve()),
                           source_sha256={"calibration.parquet": _sha(calibration_path.read_bytes()),
                                          "ego_pose.parquet": _sha(pose_path.read_bytes())})
    return result


def compare_input_fixture(prepared: PreparedClip, fixture: Mapping, *, projection_atol=1e-4,
                          pose_atol=1e-5) -> dict:
    """CPU parity report against a separately exported train-only input fixture.

    Expected format: {inputs: six batched tensors, metadata: {input_contract: ...}}.
    Passing a labeled batch or legacy raw-time/old-rear fixture is rejected.
    Matrix tolerance is not a claim of bitwise or universal pixel-coordinate parity.
    """
    for tolerance in (projection_atol, pose_atol):
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Parity tolerances must be finite and nonnegative")
    prepared_contract = prepared.metadata.get("input_contract")
    fixture_contract = fixture.get("metadata", {}).get("input_contract")
    require_input_contract(prepared_contract)
    require_input_contract(fixture_contract)
    if prepared_contract != fixture_contract:
        raise ValueError("Prepared input and fixture temporal contracts differ")
    reference = fixture.get("inputs", {})
    if set(reference) != set(INPUT_KEYS) or set(prepared.inputs) != set(INPUT_KEYS):
        raise ValueError("Parity fixture must contain exactly six model inputs, not a labeled batch")
    results = {}
    for name in INPUT_KEYS:
        actual, expected = prepared.inputs[name], reference[name]
        if not isinstance(expected, torch.Tensor) or actual.device.type != "cpu" or expected.device.type != "cpu":
            raise ValueError("Parity is a CPU tensor-only check")
        if actual.dtype != torch.float32 or expected.dtype != torch.float32 or actual.shape != expected.shape:
            raise ValueError(f"Fixture shape/dtype mismatch for {name}")
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise ValueError(f"Nonfinite fixture values for {name}")
        tolerance = projection_atol if name == "lidar2img" else pose_atol if name in ("history_transforms", "goal_xy") else 0.
        delta = (actual.double() - expected.double()).abs()
        maximum = float(delta.max())
        bitwise = torch.equal(actual.detach().contiguous().view(torch.uint8),
                              expected.detach().contiguous().view(torch.uint8))
        strict_pixels_or_time = name in ("images", "history_images", "time_offsets")
        results[name] = {"shape": list(actual.shape), "bitwise_equal": bitwise,
                         "max_abs": maximum, "mean_abs": float(delta.mean()), "atol": tolerance,
                         "rtol": 0., "pass": bitwise if strict_pixels_or_time else maximum <= tolerance}
    return {"all_pass": all(row["pass"] for row in results.values()), "inputs": results,
            "input_contract": dict(prepared_contract),
            "scope": "Train-fixture CPU input parity only; no generalization or model-output claim",
            "projection_note": "Absolute matrix-entry tolerance; not universal projected pixel error or bitwise equality."}
