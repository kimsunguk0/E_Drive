"""GT-free raw-input adapter for the temporal SDV2 comparison.

No checkpoint, bank, dataset, label, auxiliary cache or status cache is loaded.
Only constant camera geometry is cached. Pose-derived status is computed solely
for explicit common_status=True, and the provided goal is returned separately
from model inputs for a final completed-candidate selector.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from threading import Lock
from typing import Callable, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from models.motiondrive_v2_inputs import (
    CALIBRATION_COLUMNS, POSE_COLUMNS, build_camera_geometry, cache_compatible_image,
)

CAMERAS = ("camera_front_left", "camera_front", "camera_front_right")
FRONT_INDEX = 1
HISTORY_FRAMES = (-1, -5)
IMAGE_WH = (512, 256)
MEAN = np.asarray((.485, .456, .406), np.float32)
STD = np.asarray((.229, .224, .225), np.float32)
MODEL_KEYS = frozenset(("images", "lidar2img", "image_hw", "history_images", "time_offsets"))


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def _require(value, message):
    if not value:
        raise ValueError(message)


def _selected_pose_rows(records, wanted):
    """Inspect values only for the requested frames; unrelated pose values unused."""
    _require(records is not None, "Requested status/goal needs provided pose records")
    wanted, selected = set(wanted), {}
    for record in records:
        frame = record["frame"]
        try:
            parsed = int(frame)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Pose frame must be an integer") from exc
        _require(not isinstance(frame, (bool, np.bool_)) and parsed == frame, "Pose frame must be an integer")
        if parsed not in wanted:
            continue
        _require(set(record) == set(POSE_COLUMNS), "Selected pose records require only official frame/XYZ/RPY columns")
        _require(parsed not in selected, "Duplicate requested pose frame")
        selected[parsed] = record
    _require(set(selected) == wanted, "Missing requested pose frame")
    return selected


def causal_status4_from_records(records):
    """Exact nominal -10..0 past-only fit; returns physical vx/vy/ax/ay."""
    rows = _selected_pose_rows(records, range(-10, 1))
    selected = [rows[i] for i in range(-10, 1)]
    xyz = np.asarray([[r[k] for k in ("x", "y", "z")] for r in selected], np.float64)
    rpy = np.asarray([[r[k] for k in ("roll", "pitch", "yaw")] for r in selected], np.float64)
    _require(np.isfinite(xyz).all() and np.isfinite(rpy).all(), "Nonfinite causal pose")
    poses = np.broadcast_to(np.eye(4), (11, 4, 4)).copy()
    poses[:, :3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    poses[:, :3, 3] = xyz
    relative = np.linalg.inv(poses[-1]) @ poses
    time = np.arange(-10, 1, dtype=np.float64) / 10.
    design = np.column_stack((np.ones_like(time), time, .5 * time * time))
    coefficients = np.linalg.lstsq(design, relative[:, :2, 3], rcond=None)[0]
    status = np.asarray([*coefficients[1], *coefficients[2]], np.float32)
    residual = relative[:, :2, 3] - design @ coefficients
    _require(np.linalg.cond(design) < 100 and np.isfinite(status).all()
             and np.sqrt(np.mean(residual ** 2)) < .25, "Invalid causal nominal state fit")
    return status


def provided_goal_from_records(records):
    """Current full-SE3 transform of provided +50 XYZ; no future RPY read."""
    rows = _selected_pose_rows(records, (0, 50))
    current, future = rows[0], rows[50]
    xyz = np.asarray([current[k] for k in ("x", "y", "z")], np.float64)
    rpy = np.asarray([current[k] for k in ("roll", "pitch", "yaw")], np.float64)
    goal = np.asarray([future[k] for k in ("x", "y", "z")], np.float64)
    _require(np.isfinite(xyz).all() and np.isfinite(rpy).all() and np.isfinite(goal).all(),
             "Nonfinite provided goal/current pose")
    return (Rotation.from_euler("xyz", rpy).as_matrix().T @ (goal - xyz))[:2].astype(np.float32)


@dataclass
class PreparedTemporalInput:
    inputs: dict[str, torch.Tensor]
    selector_inputs: dict[str, torch.Tensor]
    metadata: dict

    @property
    def goal_xy(self):
        return self.selector_inputs.get("goal_xy")


class TemporalRawInputAdapter:
    def __init__(self, history_mode="real", common_status=False, goal_mode="none", camera_workers=3):
        _require(history_mode in ("repeat", "real"), "history_mode must be repeat or real")
        _require(isinstance(common_status, bool), "common_status must be an explicit bool")
        _require(goal_mode in ("none", "selection"), "goal_mode must be none or selection")
        _require(camera_workers in (1, 3), "camera_workers must be 1 or 3")
        self.history_mode, self.common_status, self.goal_mode = history_mode, common_status, goal_mode
        self.camera_workers = camera_workers
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="temporal_pixels") if camera_workers == 3 else None
        self._geometry_key, self._geometry = None, None
        self._geometry_lock = Lock()
        self._closed = False
        dependency = inspect.getsourcefile(cache_compatible_image)
        self.pixel_dependency_sha256 = file_sha256(dependency) if dependency and Path(dependency).is_file() else None

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _prepare_camera(geometry, raw_bytes):
        # Mandatory same undistort/crop -> 768 Q95 JPEG/PIL decode -> 512 resize.
        rgb, receipt = cache_compatible_image(raw_bytes, geometry)
        rgb = rgb.resize(IMAGE_WH, Image.Resampling.BILINEAR)
        value = (np.asarray(rgb, np.float32) / 255. - MEAN) / STD
        return torch.from_numpy(value.transpose(2, 0, 1).copy()), receipt

    def prepare_records(self, calibration_rows: Sequence[Mapping], pose_rows,
                        image_loader: Callable[[str, int], bytes]) -> PreparedTemporalInput:
        _require(not self._closed, "TemporalRawInputAdapter is closed")
        calibration_rows = list(calibration_rows)
        raw_key = json.dumps(calibration_rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        key = hashlib.sha256(raw_key).hexdigest()
        with self._geometry_lock:
            if self._geometry_key != key:
                all_geometry = {g.name: g for g in build_camera_geometry(calibration_rows)}
                self._geometry = tuple(all_geometry[name] for name in CAMERAS)
                self._geometry_key = key
            geometry = self._geometry
        requests = [(g, 0) for g in geometry]
        if self.history_mode == "real":
            requests += [(geometry[FRONT_INDEX], frame) for frame in HISTORY_FRAMES]
        # Caller-owned I/O is serial, since arbitrary loaders need not be thread-safe.
        raw_images = [image_loader(g.name, frame) for g, frame in requests]
        pending = [self._pool.submit(self._prepare_camera, g, raw) for (g, _), raw in zip(requests, raw_images)] if self._pool else None
        processed = [job.result() for job in pending] if pending else [
            self._prepare_camera(g, raw) for (g, _), raw in zip(requests, raw_images)]
        current = torch.stack([value for value, _ in processed[:3]])
        history = (torch.stack([value for value, _ in processed[3:]]) if self.history_mode == "real"
                   else current[FRONT_INDEX].unsqueeze(0).repeat(2, 1, 1, 1))
        matrices = []
        for g in geometry:
            matrix = g.lidar2img.copy()
            matrix[0] *= IMAGE_WH[0] / 768.
            matrix[1] *= IMAGE_WH[1] / 432.
            matrices.append(matrix)
        inputs = {"images": current[None], "history_images": history[None],
                  "lidar2img": torch.from_numpy(np.stack(matrices))[None],
                  "image_hw": torch.tensor([[256., 512.]], dtype=torch.float32),
                  "time_offsets": torch.tensor([[.1, .5]], dtype=torch.float32)}
        # Do not even iterate pose_rows for A/B with goal disabled.
        selected_pose_rows = None
        pose_frames_used = set(range(-10, 1)) if self.common_status else set()
        if self.goal_mode == "selection":
            pose_frames_used.update((0, 50))
        if pose_frames_used:
            selected_pose_rows = _selected_pose_rows(pose_rows, pose_frames_used)
        if self.common_status:
            inputs["perception_status"] = torch.from_numpy(
                causal_status4_from_records(selected_pose_rows.values()))[None]
        selector_inputs = {}
        if self.goal_mode == "selection":
            selector_inputs["goal_xy"] = torch.from_numpy(
                provided_goal_from_records(selected_pose_rows.values()))[None]
        expected_keys = MODEL_KEYS | ({"perception_status"} if self.common_status else set())
        _require(set(inputs) == expected_keys and "status" not in inputs and "goal_xy" not in inputs,
                 "Prepared model-input boundary mismatch")
        _require(all(t.dtype == torch.float32 and torch.isfinite(t).all()
                     for t in [*inputs.values(), *selector_inputs.values()]), "Nonfinite prepared inputs")
        image_receipts = {}
        for (g, frame), (_, receipt) in zip(requests, processed):
            image_receipts[f"{g.name}/frame_{frame}.jpg"] = {
                **receipt, "crop_xy": list(g.crop_xy), "raw_wh": list(g.raw_wh),
                "K_cache": g.cached_intrinsic.tolist()}
        used_pose_values = []
        for frame in sorted(pose_frames_used):
            fields = ("x", "y", "z") if frame == 50 else ("x", "y", "z", "roll", "pitch", "yaw")
            used_pose_values.append({"frame": frame, **{k: float(selected_pose_rows[frame][k]) for k in fields}})
        pose_digest = (hashlib.sha256(json.dumps(used_pose_values, sort_keys=True, allow_nan=False).encode()).hexdigest()
                       if used_pose_values else None)
        metadata = {"input_contract": "temporal-sdv2-etri-raw-input-v1", "history_mode": self.history_mode,
                    "camera_order": list(CAMERAS), "image_wh": list(IMAGE_WH),
                    "history_frames": list(HISTORY_FRAMES), "time_offsets": [.1, .5],
                    "raw_image_reads": len(requests), "images": image_receipts,
                    "calibration_value_sha256": key, "pose_value_sha256": pose_digest,
                    "pose_frames_used": sorted(pose_frames_used), "common_status": self.common_status,
                    "causal_status_computed": self.common_status, "planner_status_input_present": False,
                    "goal_mode": self.goal_mode, "goal_only_in_selector_inputs": True,
                    "future_pose_orientation_used": False, "pose_alignment_used": False,
                    "state_definition": "nominal -10..0 causal quadratic fit in current ego XY" if self.common_status else None,
                    "geometry_cache": "full-calibration-keyed remap geometry only", "image_feature_cache": False,
                    "camera_workers": self.camera_workers, "pixel_dependency_sha256": self.pixel_dependency_sha256,
                    "labels_read": False, "aux_cache_read": False, "status_cache_read": False,
                    "model_or_checkpoint_loaded": False}
        return PreparedTemporalInput(inputs, selector_inputs, metadata)

    def prepare_clip(self, clip_dir):
        """Official test-shaped disk bundle; A/B goal-none need no pose file."""
        import pyarrow.parquet as pq
        directory = Path(clip_dir)
        calibration_path = directory / "calibration.parquet"
        calibration = pq.read_table(calibration_path, columns=list(CALIBRATION_COLUMNS)).to_pylist()
        wanted = set(range(-10, 1)) if self.common_status else set()
        if self.goal_mode == "selection":
            wanted.update((0, 50))
        poses = None
        if wanted:
            poses = pq.read_table(directory / "ego_pose.parquet", columns=list(POSE_COLUMNS),
                                  filters=[("frame", "in", sorted(wanted))]).to_pylist()
        prepared = self.prepare_records(calibration, poses, lambda camera, frame:
                                        (directory / camera / f"frame_{frame}.jpg").read_bytes())
        prepared.metadata["source_sha256"] = {"calibration.parquet": file_sha256(calibration_path)}
        prepared.metadata["pose_file_opened"] = bool(wanted)
        prepared.metadata["pose_parquet_frames_selected"] = sorted(wanted)
        return prepared
