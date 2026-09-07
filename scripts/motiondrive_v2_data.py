"""MotionDrive V2 inputs and strict, explicitly masked training targets.

GT pose/status are targets or geometry only. This module never constructs a
planner input from GT state. All images come from the undistorted 768x432 cache.
"""
from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

CAMERA_ORDER = ("camera_front", "camera_front_right", "camera_front_left",
                "camera_rear_wide", "camera_rear_left", "camera_rear_right")
HISTORY_OFFSETS = np.asarray([1, 2, 5, 10], dtype=np.int64)
GRID_SHAPE = (64, 48)
GRID_EXTENT = (-10., 70., -32., 32.)
MEAN = np.asarray([.485, .456, .406], np.float32)
STD = np.asarray([.229, .224, .225], np.float32)


def grid_centers(shape=GRID_SHAPE, extent=GRID_EXTENT):
    nx, ny = shape
    x0, x1, y0, y1 = extent
    x = x0 + (np.arange(nx) + .5) * (x1 - x0) / nx
    y = y0 + (np.arange(ny) + .5) * (y1 - y0) / ny
    return np.stack(np.meshgrid(x, y, indexing="ij"), -1)


def calibration_from_info(info):
    """Match actual etri_768 cache: front/rear_wide bottom, four side views top.

    Archived canonical NPZs remain unchanged; this conversion does not repair
    an old on-disk edition. New versions must verify actual cache metadata.
    """
    out = []
    for name in CAMERA_ORDER:
        cam = info["cams"][name]
        rotation = np.asarray(cam["sensor2lidar_rotation"], np.float64)
        translation = np.asarray(cam["sensor2lidar_translation"], np.float64)
        world_to_cam = np.eye(4)
        world_to_cam[:3, :3] = np.linalg.inv(rotation)
        world_to_cam[:3, 3] = -np.linalg.inv(rotation) @ translation
        k = np.eye(4)
        k[:3, :3] = cam["cam_intrinsic"]
        ox = (int(cam["image_width"]) - 1920) // 2
        oy = int(cam["image_height"]) - 1080 if name in ("camera_front", "camera_rear_wide") else 0
        crop = np.eye(4)
        crop[0, 2], crop[1, 2] = -ox, -oy
        resize = np.diag([.4, .4, 1., 1.])
        out.append(resize @ crop @ k @ world_to_cam)
    return np.asarray(out, dtype=np.float32)


def load_calibration(path):
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            arr = z["lidar2img"]
    else:
        with path.open("rb") as f:
            payload = pickle.load(f)  # trusted, user-owned dataset conversion file
        arr = calibration_from_info(payload["infos"][0])
    if arr.shape != (6, 4, 4) or not np.isfinite(arr).all():
        raise ValueError("Invalid canonical calibration")
    return arr.astype(np.float32)


def verify_canonical_calibration(contract, path):
    """Derived geometry must pin the actual NPZ, not only its source PKL."""
    from build_grouped_split_v2 import sha256
    expected = contract.get("canonical_calibration_sha256")
    if contract.get("schema_version", 1) >= 2 and expected is None:
        raise ValueError("Derived supervision lacks canonical calibration SHA")
    # Older schema1 editions were recorded without a derived-file checksum.
    # Preserve historical loading, but verify any checksum that is declared.
    if expected is not None and (not isinstance(expected, str) or sha256(path) != expected):
        raise ValueError("Canonical calibration provenance SHA mismatch")


def full_pose_matrices(xyz, rpy):
    from scipy.spatial.transform import Rotation
    xyz, rpy = np.asarray(xyz, np.float64), np.asarray(rpy, np.float64)
    out = np.broadcast_to(np.eye(4), (len(xyz), 4, 4)).copy()
    out[:, :3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    out[:, :3, 3] = xyz
    return out


def motion_targets(poses, times_seconds, current_index, past_indices):
    """Full SE(3) alignment; causal quadratic fit over <=1 second for state.

    state = vx,vy,ax,ay,yaw_rate,stop. Fit time is centered at current timestamp;
    zero intercept is NOT imposed (otherwise pose noise at t0 biases derivatives).
    Units m/s, m/s², rad/s; stop threshold = causal fitted planar speed <.2m/s.
    """
    poses, ts = np.asarray(poses, np.float64), np.asarray(times_seconds, np.float64)
    now = int(current_index)
    ids = np.asarray(past_indices, np.int64)
    if (ids >= now).any() or (ids < 0).any():
        raise ValueError("History must be strictly causal")
    rel = np.linalg.inv(poses[now]) @ poses[ids]
    alignment = np.linalg.inv(poses[ids]) @ poses[now]
    yaw = np.arctan2(rel[:, 1, 0], rel[:, 0, 0])
    history = np.column_stack((rel[:, :2, 3], np.sin(yaw), np.cos(yaw)))
    dt = ts[now] - ts[ids]
    valid_history = np.isfinite(history).all(1) & np.isfinite(dt) & (dt > 0) & (dt <= 3.001)
    if not valid_history.all():
        raise ValueError("Nonfinite/out-of-window pose history")
    fit_ids = np.where((ts <= ts[now]) & (ts >= ts[now] - 1.001))[0]
    fit_ids = fit_ids[fit_ids <= now]
    state, state_valid = np.zeros(6, np.float32), np.zeros(6, bool)
    if len(fit_ids) >= 6:
        fit_rel = np.linalg.inv(poses[now]) @ poses[fit_ids]
        t = ts[fit_ids] - ts[now]
        design = np.column_stack((np.ones_like(t), t, .5 * t * t))
        gap_ok = np.all(np.diff(ts[fit_ids]) > 0) and np.max(np.diff(ts[fit_ids])) <= .151
        if gap_ok and np.linalg.cond(design) < 100:
            position = np.linalg.lstsq(design, fit_rel[:, :2, 3], rcond=None)[0]
            angle = np.unwrap(np.arctan2(fit_rel[:, 1, 0], fit_rel[:, 0, 0]))
            angular = np.linalg.lstsq(design, angle, rcond=None)[0]
            state = np.asarray([*position[1], *position[2], angular[1], np.linalg.norm(position[1]) < .2], np.float32)
            residual = fit_rel[:, :2, 3] - design @ position
            valid = np.isfinite(state).all() and np.sqrt(np.mean(residual ** 2)) < .25
            state_valid[:] = valid
    return dict(history_transforms=alignment.astype(np.float32),
                history_target=history.astype(np.float32),
                history_valid=np.repeat(valid_history[:, None], 4, axis=1),
                time_offsets=dt.astype(np.float32), state_target=state,
                state_valid=state_valid)


class MotionDriveDataset(Dataset):
    def __init__(self, data_root, split_manifest, split="train", supervision_root=None,
                 *, image_root=None, ego_cache=None, min_frame=30, frame_stride=1,
                 max_samples=0, augment=False, seed=0, allow_missing_supervision=False,
                 calibration_path=None, scenes=None, frames=None):
        self.root = Path(data_root)
        with open(split_manifest) as f:
            self.manifest = json.load(f)
        from build_grouped_split_v2 import sha256, validate_manifest
        validate_manifest(self.manifest)
        if split not in self.manifest["splits"]:
            raise KeyError(split)
        if frame_stride < 1 or min_frame < int(HISTORY_OFFSETS.max()):
            raise ValueError("frame_stride>=1 and min_frame>=10 required")
        cache = Path(ego_cache) if ego_cache else self.root / "data/etri/ego_cache.npz"
        if not cache.exists() and ego_cache is None:
            cache = Path("/tmp/pm97/data/etri/ego_cache.npz")
        with np.load(cache, allow_pickle=False) as z:
            self.arr = {k: z[k] for k in ("scenarios", "scen_idx", "frame", "goal", "fut")}
        self.split_sha = sha256(split_manifest)
        self.cache_sha = sha256(cache)
        names = self.arr["scenarios"].astype(str)[self.arr["scen_idx"]]
        mask = np.isin(names, self.manifest["splits"][split])
        if scenes is not None:
            requested = set(map(str, scenes))
            if not requested <= set(self.manifest["splits"][split]):
                raise ValueError("Explicit scenes must belong to the requested split")
            mask &= np.isin(names, list(requested))
        mask &= (self.arr["frame"] >= min_frame) & (self.arr["frame"] % frame_stride == 0)
        if frames is not None:
            mask &= np.isin(self.arr["frame"], list(frames))
        self.rows = np.flatnonzero(mask)
        if max_samples:
            # Evenly sample the complete split; never silently only its first scene.
            self.rows = self.rows[np.linspace(0, len(self.rows) - 1, min(max_samples, len(self.rows)), dtype=int)]
        if not len(self.rows):
            raise ValueError(f"Empty split {split}")
        self.image_root = Path(image_root) if image_root else self.root / "cache/etri_768"
        self.supervision_root = Path(supervision_root) if supervision_root else self.root / "data/etri/motiondrive_v2"
        with (self.supervision_root / "supervision_manifest.json").open() as f:
            contract = json.load(f)
        if contract.get("split_manifest_sha256") != self.split_sha:
            raise ValueError("Supervision split provenance SHA mismatch")
        if contract.get("ego_cache_sha256") != self.cache_sha:
            raise ValueError("Supervision ego-cache provenance SHA mismatch")
        if contract.get("grid_shape") != list(GRID_SHAPE) or contract.get("grid_extent") != list(GRID_EXTENT):
            raise ValueError("Supervision grid geometry mismatch")
        if contract.get("history_frame_offsets") != HISTORY_OFFSETS.tolist():
            raise ValueError("Supervision history offsets mismatch")
        self.augment, self.seed, self.epoch = bool(augment), int(seed), 0
        self.allow_missing_supervision = allow_missing_supervision
        canonical_path = self.supervision_root / "calibration.npz"
        verify_canonical_calibration(contract, canonical_path)
        cp = Path(calibration_path) if calibration_path else self.supervision_root / "calibration.npz"
        self.lidar2img = load_calibration(cp)
        if not np.array_equal(self.lidar2img, load_calibration(self.supervision_root / "calibration.npz")):
            raise ValueError("Override calibration differs from supervision geometry")
        self.scene_names = names
        self.split = split
        self._check_files()

    def _check_files(self):
        missing = [s for s in sorted(set(self.scene_names[self.rows]))
                   if not (self.supervision_root / f"{s}.npz").exists()]
        if missing and not self.allow_missing_supervision:
            raise FileNotFoundError(f"Missing supervision for {len(missing)} scenes: {missing[:4]}")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    @lru_cache(maxsize=4)
    def _supervision(self, scene):
        path = self.supervision_root / f"{scene}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Geometry/motion targets missing even in skeleton mode: {path}")
        with np.load(path, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
        with path.with_suffix(".json").open() as f:
            report = json.load(f)
        if report.get("scene") != scene or report.get("split_manifest_sha256") != self.split_sha or report.get("ego_cache_sha256") != self.cache_sha:
            raise ValueError(f"Scene provenance mismatch: {scene}")
        if "row" not in out or len(out["row"]) != len(out["frame"]) or len(np.unique(out["frame"])) != len(out["frame"]):
            raise ValueError(f"Scene row/frame index malformed: {scene}")
        row = out["row"].astype(np.int64)
        if (row < 0).any() or (row >= len(self.scene_names)).any():
            raise ValueError(f"Scene row out of bounds: {scene}")
        if not np.array_equal(self.arr["frame"][row], out["frame"]) or not np.all(self.scene_names[row] == scene):
            raise ValueError(f"Scene rows do not match ego-cache ordering: {scene}")
        out["frame_lookup"] = {int(f): i for i, f in enumerate(out["frame"])}
        return out

    def _image(self, scene, camera, frame, size, jitter):
        path = self.image_root / scene / camera / f"{frame:08d}.jpg"
        with Image.open(path) as f:
            im = f.convert("RGB")
        if im.size != (768, 432):
            raise ValueError(f"Cache geometry mismatch: {path}: {im.size}")
        if size != im.size:
            im = im.resize(size, Image.Resampling.BILINEAR)
        if jitter is not None:
            for cls, factor in zip((ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color), jitter):
                im = cls(im).enhance(float(factor))
        x = np.asarray(im, np.float32) / 255.
        return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy())

    def __getitem__(self, index):
        row = int(self.rows[index])
        frame, scene = int(self.arr["frame"][row]), str(self.scene_names[row])
        supervision = self._supervision(scene)
        if frame not in supervision["frame_lookup"]:
            raise KeyError(f"Supervision missing requested frame: {scene}/{frame}")
        si = supervision["frame_lookup"][frame]
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + row)
        jitter = rng.uniform(.9, 1.1, (6, 3)) if self.augment else [None] * 6
        # Same front-camera photometric parameters across all temporal images.
        images = torch.stack([self._image(scene, cam, frame, (768, 432), jitter[c]) for c, cam in enumerate(CAMERA_ORDER)])
        history = torch.stack([self._image(scene, CAMERA_ORDER[0], frame - int(offset), (384, 216), jitter[0]) for offset in HISTORY_OFFSETS])
        result = dict(images=images, history_images=history,
                      lidar2img=torch.from_numpy(self.lidar2img.copy()),
                      goal_xy=torch.from_numpy(self.arr["goal"][row].astype(np.float32)),
                      gt_plan=torch.from_numpy(self.arr["fut"][row].astype(np.float32)),
                      plan_valid=torch.ones(6, dtype=torch.bool),
                      row=row, scen_idx=int(self.arr["scen_idx"][row]), frame=frame,
                      scenario=scene, session_id=self.manifest["scene_to_session"][scene],
                      proxy_weight=torch.tensor(float(supervision.get("proxy_weight", np.ones(len(supervision["frame"])))[si]), dtype=torch.float32))
        for key in ("history_transforms", "time_offsets", "history_target", "history_valid",
                    "state_target", "state_valid", "occ_target", "lane_target", "occ_valid", "lane_valid"):
            if key not in supervision:
                raise KeyError(f"{scene}: missing required {key}")
            value = supervision[key][si]
            result[key] = torch.from_numpy(np.asarray(value).copy())
        return result
