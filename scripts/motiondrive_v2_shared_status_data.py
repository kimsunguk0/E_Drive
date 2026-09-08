"""Causal nominal-time status contract and opt-in dataset adapter for A1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from motiondrive_v2_data import full_pose_matrices, motion_targets

STATUS_FIELDS = ("vx", "vy", "ax", "ay", "yaw_rate")
NOMINAL_FRAMES = tuple(range(-10, 1))
NOMINAL_TIMES = np.arange(-10, 1, dtype=np.float64) / 10.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_sha(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def causal_status5_from_pose_matrices(pose11) -> np.ndarray:
    """Use exactly 11 causal poses at nominal 10 Hz; no future input exists."""
    poses = np.asarray(pose11, dtype=np.float64)
    require(poses.shape == (11, 4, 4) and np.isfinite(poses).all(),
            "causal status requires finite [11,4,4] poses")
    result = motion_targets(poses, NOMINAL_TIMES, 10, (9, 8, 5, 0))
    require(bool(np.asarray(result["state_valid"][:5]).all()), "causal status fit is invalid")
    status = np.asarray(result["state_target"][:5], dtype=np.float32)
    require(status.shape == (5,) and np.isfinite(status).all(), "causal status is invalid")
    return status


def causal_status5_from_pose_records(records: Sequence[Mapping], current_frame: int = 0) -> np.ndarray:
    """Serving-format adapter; reads only current and ten preceding pose rows."""
    required = {"frame", "x", "y", "z", "roll", "pitch", "yaw"}
    selected = {}
    wanted = {current_frame + offset for offset in NOMINAL_FRAMES}
    for item in records:
        if not required <= set(item):
            raise ValueError("pose records lack frame/XYZ/RPY")
        frame = int(item["frame"])
        if frame in wanted:
            require(frame not in selected, "duplicate causal pose frame")
            selected[frame] = item
    require(set(selected) == wanted, "exact causal pose frames are required")
    rows = [selected[current_frame + offset] for offset in NOMINAL_FRAMES]
    xyz = np.asarray([[row[k] for k in ("x", "y", "z")] for row in rows], np.float64)
    rpy = np.asarray([[row[k] for k in ("roll", "pitch", "yaw")] for row in rows], np.float64)
    return causal_status5_from_pose_matrices(full_pose_matrices(xyz, rpy))


def load_status_overlay(root, expected_manifest_sha256: str) -> tuple[dict, dict[str, dict]]:
    from build_grouped_split_v2 import sha256
    root = Path(root).resolve()
    manifest_path = root / "overlay_manifest.json"
    require(sha256(manifest_path) == expected_manifest_sha256, "shared-status manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text())
    expected = {"schema_version", "status", "name", "split_manifest_sha256",
                "ego_cache_sha256", "source_root", "causal_contract", "splits", "artifacts",
                "source_files", "source_files_sha256"}
    require(set(manifest) == expected and manifest["schema_version"] == 1
            and manifest["status"] == "frozen" and manifest["name"] == "shared_status_a1",
            "shared-status manifest schema mismatch")
    contract = manifest["causal_contract"]
    require(contract == {"fields": list(STATUS_FIELDS), "frames": list(NOMINAL_FRAMES),
                         "seconds": NOMINAL_TIMES.tolist(),
                         "method": "motion_targets causal quadratic fit; nominal 10Hz",
                         "future_values_used": False,
                         "ego_pose_file_all_rows_read_for_identity_join": True,
                         "goal_command_hd_files_read": False},
            "shared-status causal contract mismatch")
    require(manifest["splits"] == {
        "train": {"rows": 54810,
                  "rows_sha256": "75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854"},
        "tune": {"rows": 1998,
                 "rows_sha256": "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88"}},
            "shared-status split population contract mismatch")
    source_files = manifest["source_files"]
    require(isinstance(source_files, dict) and len(source_files) == 240
            and all(set(value) == {"ego_pose_sha256", "timestamps_sha256"}
                    and all(isinstance(digest, str) and len(digest) == 64
                            for digest in value.values())
                    for value in source_files.values()),
            "shared-status source-file inventory mismatch")
    source_digest = hashlib.sha256(json.dumps(
        source_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(source_digest == manifest["source_files_sha256"],
            "shared-status source-file digest mismatch")
    loaded = {}
    for split in ("train", "tune"):
        spec = manifest["artifacts"].get(split)
        require(isinstance(spec, dict) and set(spec) == {"file", "sha256", "rows", "rows_sha256",
                                                        "status_sha256", "valid_rows"},
                f"shared-status {split} declaration malformed")
        path = root / spec["file"]
        require(path.parent == root and sha256(path) == spec["sha256"],
                f"shared-status {split} artifact SHA mismatch")
        with np.load(path, allow_pickle=False) as source:
            require(set(source.files) == {"row", "frame", "status5"},
                    f"shared-status {split} keys mismatch")
            data = {key: source[key] for key in source.files}
        n = len(data["row"])
        require(data["row"].dtype == np.int64 and data["frame"].dtype == np.int64
                and data["status5"].dtype == np.float32 and data["status5"].shape == (n, 5)
                and np.isfinite(data["status5"]).all() and len(np.unique(data["row"])) == n,
                f"shared-status {split} array contract mismatch")
        require((n, array_sha(data["row"]), array_sha(data["status5"]), spec["valid_rows"])
                == (spec["rows"], spec["rows_sha256"], spec["status_sha256"], n),
                f"shared-status {split} digests/count mismatch")
        loaded[split] = data
    return manifest, loaded


class SharedStatusDataset(torch.utils.data.Dataset):
    """Add only the frozen status tensor to an unchanged MotionDrive dataset."""

    def __init__(self, base, split: str, overlay: dict[str, dict], mode: str):
        require(mode in ("zero", "provided_causal_5d"), "unknown shared-status arm")
        self.base, self.mode = base, mode
        data = overlay[split]
        require(np.array_equal(np.asarray(base.rows, np.int64), data["row"]),
                "shared-status rows differ from the base dataset")
        base_frames = np.asarray(base.arr["frame"][base.rows], np.int64)
        require(np.array_equal(base_frames, data["frame"]),
                "shared-status frames differ from the base dataset")
        self.status = torch.from_numpy(data["status5"].copy())
        self.rows, self.arr, self.scene_names = base.rows, base.arr, base.scene_names

    def __getattr__(self, name):
        # Preserve the full base-dataset interface used by trainer provenance,
        # calibration and supervision bookkeeping.
        if name in {"base", "mode", "status", "rows", "arr", "scene_names"}:
            raise AttributeError(name)
        return getattr(self.base, name)

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        return self.base.set_epoch(epoch)

    def __getitem__(self, index):
        item = self.base[index]
        status = self.status[index]
        item["provided_status5"] = torch.zeros_like(status) if self.mode == "zero" else status.clone()
        return item


def shared_status_model_inputs(base_model_inputs, batch, *, time_input="raw",
                               nominal_history_seconds=(.1, .2, .5, 1.)):
    inputs = base_model_inputs(batch, time_input=time_input,
                              nominal_history_seconds=nominal_history_seconds)
    status = batch.get("provided_status5")
    require(isinstance(status, torch.Tensor) and status.ndim == 2 and status.shape[1] == 5,
            "provided_status5 [B,5] is required")
    inputs["provided_status5"] = status
    return inputs
