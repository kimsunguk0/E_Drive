"""ETRI current-camera inputs and separate plan labels for the V2 screen.

Only the original train/tune populations are enabled by default. Explicit row
files may remove samples, but cannot add samples outside the requested split.
Provided causal state is an opt-in selection input; cache vel/acc are not used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

CAMERAS = ("camera_front_left", "camera_front", "camera_front_right")
CAMERA_INDICES = (2, 0, 1)
MEAN = np.asarray((.485, .456, .406), dtype=np.float32)
STD = np.asarray((.229, .224, .225), dtype=np.float32)
TIME_WEIGHTS = (11 / 36, 11 / 36, 5 / 36, 5 / 36, 2 / 36, 2 / 36)
EXPECTED_SPLIT_SHA = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
EXPECTED_EGO_SHA = "d35a69bb229b2d4736aff7dddefd3522ddbd346e112283e2ce58ce3986a023bd"
EXPECTED_CALIBRATION_SHA = "8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961"


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_sha(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def d3(pred, gt):
    if pred.shape != gt.shape or pred.shape[-2:] != (6, 2):
        raise ValueError("D3 requires matching [...,6,2] cumulative XY metres")
    error = torch.linalg.vector_norm(pred.float() - gt.float(), dim=-1)
    return (error * error.new_tensor(TIME_WEIGHTS)).sum(-1)


class PlanDataset(Dataset):
    def __init__(self, base, split_manifest, split, *, ego_cache="/tmp/pm97/data/etri/ego_cache.npz",
                 stride=None, rows_file=None, image_size=(512, 256), augment=False, seed=0,
                 status_mode="zero", goal_mode="none", limit=0):
        if split not in ("train", "tune"):
            raise ValueError("This screen does not open reserve/test populations")
        if status_mode not in ("zero", "causal_selection"):
            raise ValueError("Unknown status mode")
        if goal_mode not in ("none", "selection"):
            raise ValueError("Unknown goal mode")
        self.base = Path(base).resolve()
        self.split_manifest = Path(split_manifest).resolve()
        if file_sha(self.split_manifest) != EXPECTED_SPLIT_SHA:
            raise ValueError("Use the audited primary manifest; restrict folds with explicit rows")
        self.manifest = json.loads(self.split_manifest.read_text())
        self.split = split
        self.status_mode = status_mode
        self.goal_mode = goal_mode
        self.augment, self.seed, self.epoch = bool(augment), int(seed), 0
        self.image_size = tuple(image_size)
        if len(self.image_size) != 2 or min(self.image_size) < 32:
            raise ValueError("Invalid image size")
        self.ego_cache = Path(ego_cache).resolve()
        if file_sha(self.ego_cache) != EXPECTED_EGO_SHA:
            raise ValueError("Ego cache differs from the audited label source")
        with np.load(self.ego_cache, allow_pickle=False) as z:
            self.scenarios = z["scenarios"].astype(str)
            self.scen_idx = z["scen_idx"].astype(np.int64)
            self.frames = z["frame"].astype(np.int64)
            self.gt = z["fut"].astype(np.float32)
            self.goal = z["goal"].astype(np.float32) if goal_mode == "selection" else None
        names = self.scenarios[self.scen_idx]
        stride = int(stride or (1 if split == "train" else 5))
        if stride < 1:
            raise ValueError("Stride must be positive")
        mask = np.isin(names, self.manifest["splits"][split]) & (self.frames >= 30)
        mask &= (self.frames % stride == 0)
        eligible = np.flatnonzero(mask)
        if rows_file is not None:
            p = Path(rows_file)
            if p.suffix == ".npy":
                requested = np.load(p, allow_pickle=False)
            else:
                payload = json.loads(p.read_text())
                requested = np.asarray(payload["rows"] if isinstance(payload, dict) else payload)
            requested = np.asarray(requested)
            if requested.dtype.kind not in "iu":
                raise ValueError("Explicit rows must contain integer indices")
            requested = requested.astype(np.int64)
            if (requested.ndim != 1 or len(np.unique(requested)) != len(requested)
                    or not np.isin(requested, eligible).all()):
                raise ValueError("Explicit rows must be unique members of the requested population")
            eligible = np.sort(requested)
        self.allowed_rows = eligible.copy()
        if limit:
            eligible = eligible[np.linspace(0, len(eligible) - 1, min(limit, len(eligible)), dtype=int)]
        if not len(eligible):
            raise ValueError("Empty dataset")
        self.rows = eligible
        if self.gt.shape[1:] != (6, 2) or not np.isfinite(self.gt[self.rows]).all():
            raise ValueError("Invalid future targets")
        if self.goal is not None and (self.goal.shape != (len(self.gt), 2)
                                     or not np.isfinite(self.goal[self.rows]).all()):
            raise ValueError("Invalid provided goal input")
        self.image_root = self.base / "cache/etri_768"
        self.calibration_path = self.base / "data/etri/motiondrive_v2/train_tune_geometry_v2/calibration.npz"
        if file_sha(self.calibration_path) != EXPECTED_CALIBRATION_SHA:
            raise ValueError("Calibration differs from the verified cache crop geometry")
        with np.load(self.calibration_path, allow_pickle=False) as z:
            projection = z["lidar2img"].astype(np.float32)
        if projection.shape != (6, 4, 4) or not np.isfinite(projection).all():
            raise ValueError("Invalid calibration")
        projection = projection[list(CAMERA_INDICES)].copy()
        projection[:, 0, :] *= self.image_size[0] / 768
        projection[:, 1, :] *= self.image_size[1] / 432
        self.projection = torch.from_numpy(projection)
        self.status = np.zeros((len(self.rows), 8), dtype=np.float32)
        self.status_source = None
        if status_mode == "causal_selection":
            root = self.base / "data/etri/motiondrive_v2_shared_status_a1_20260908_ops"
            contract = json.loads((root / "overlay_manifest.json").read_text())
            if (contract["ego_cache_sha256"] != file_sha(self.ego_cache)
                    or contract["split_manifest_sha256"] != file_sha(self.split_manifest)
                    or contract["causal_contract"]["future_values_used"]):
                raise ValueError("Causal status provenance mismatch")
            path = root / contract["artifacts"][split]["file"]
            if file_sha(path) != contract["artifacts"][split]["sha256"]:
                raise ValueError("Causal status file changed")
            with np.load(path, allow_pickle=False) as z:
                source_rows, source_frames, status5 = z["row"], z["frame"], z["status5"]
            lookup = {int(row): i for i, row in enumerate(source_rows)}
            ids = np.asarray([lookup[int(row)] for row in self.rows], dtype=np.int64)
            if not np.array_equal(source_frames[ids], self.frames[self.rows]):
                raise ValueError("Causal status row/frame mismatch")
            # Native NAVSIM slots: command[4], velocity[2], acceleration[2].
            # Command remains zero; no inferred VAD command or future goal is read.
            self.status[:, 4:] = status5[ids, :4]
            if not np.isfinite(self.status).all():
                raise ValueError("Invalid causal status")
            self.status_source = {"path": str(path), "sha256": file_sha(path),
                                  "command": "zero", "fields": ["vx", "vy", "ax", "ay"],
                                  "use": "fixed complete trajectory selection only"}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def provenance(self):
        names = self.scenarios[self.scen_idx[self.rows]]
        return {"split": self.split, "rows": len(self.rows), "rows_sha256": rows_sha(self.rows),
                "allowed_rows": len(self.allowed_rows), "allowed_rows_sha256": rows_sha(self.allowed_rows),
                "scenes": sorted(set(names.tolist())),
                "sessions": sorted({self.manifest["scene_to_session"][s] for s in names}),
                "split_manifest": str(self.split_manifest), "split_sha256": file_sha(self.split_manifest),
                "ego_cache": str(self.ego_cache), "ego_cache_sha256": file_sha(self.ego_cache),
                "calibration": str(self.calibration_path), "calibration_sha256": file_sha(self.calibration_path),
                "camera_order": list(CAMERAS), "image_wh": list(self.image_size),
                "status_mode": self.status_mode, "status_source": self.status_source,
                "goal_mode": self.goal_mode,
                "goal_source": None if self.goal is None else {
                    "path": str(self.ego_cache), "field": "goal", "frame": "current ego XY metres",
                    "use": "provided goal for fixed complete trajectory selection only"},
                "normalization": {"rgb": True, "mean": MEAN.tolist(), "std": STD.tolist()},
                "target": "cumulative XY, current ego x-forward y-left, metres, .5..3s",
                "metric_weights": list(TIME_WEIGHTS)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = int(self.rows[index])
        scene = str(self.scenarios[self.scen_idx[row]])
        frame = int(self.frames[row])
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + row)
        images = []
        for camera in CAMERAS:
            path = self.image_root / scene / camera / f"{frame:08d}.jpg"
            with Image.open(path) as source:
                image = source.convert("RGB")
            if image.size != (768, 432):
                raise ValueError(f"Unexpected cached image geometry: {path}")
            image = image.resize(self.image_size, Image.Resampling.BILINEAR)
            if self.augment:
                for transform in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                    image = transform(image).enhance(float(rng.uniform(.9, 1.1)))
            arr = (np.asarray(image, dtype=np.float32) / 255. - MEAN) / STD
            images.append(torch.from_numpy(arr.transpose(2, 0, 1).copy()))
        result = {"images": torch.stack(images), "lidar2img": self.projection.clone(),
                "image_hw": torch.tensor(self.image_size[::-1], dtype=torch.float32),
                "status": torch.from_numpy(self.status[index].copy()),
                "gt_plan": torch.from_numpy(self.gt[row].copy()), "row": row,
                "frame": frame, "scenario": scene,
                "session": self.manifest["scene_to_session"][scene]}
        if self.goal is not None:
            result["goal_xy"] = torch.from_numpy(self.goal[row].copy())
        return result


def model_inputs(batch, goal_selection=False):
    """Explicit deployed inputs; labels and sample identifiers cannot flow through."""
    keys = ("images", "lidar2img", "image_hw", "status")
    if goal_selection:
        keys += ("goal_xy",)
    return {k: batch[k] for k in keys}
