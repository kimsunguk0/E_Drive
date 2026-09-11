"""Temporal wrapper over the frozen SDV2 PlanDataset; no inference pose warp.

All arms retain identical current images, labels and row identities. Only the
source of two past-front images differs. Causal status is loss-only by default;
an explicit common_status=True maps it to perception_status, never planner status.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

BASE_SOURCE_SHA256 = "54bcd88b1c2567d8aab3337fd82f5e6be50a4950e51957ba1df2908185f1eddb"
LAGS = (1, 5)
FRONT_INDEX = 1
MEAN = np.asarray((.485, .456, .406), np.float32)
STD = np.asarray((.229, .224, .225), np.float32)
GRID_SHAPE = (64, 48)
GRID_EXTENT = (-10., 70., -32., 32.)
MODEL_INPUT_KEYS = ("images", "lidar2img", "image_hw", "history_images", "time_offsets")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def rows_sha(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def _base_source_sha(dataset):
    path = inspect.getsourcefile(type(dataset))
    return file_sha(path) if path is not None and Path(path).is_file() else None


def _front_jitter(dataset, row):
    if not dataset.augment:
        return None
    rng = np.random.default_rng(dataset.seed + dataset.epoch * 1000003 + row)
    # Frozen PlanDataset visits front_left, front, front_right; each draws B/C/C.
    return rng.uniform(.9, 1.1, (3, 3))[FRONT_INDEX]


def _read_image(path, image_size, jitter):
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.size != (768, 432):
        raise ValueError(f"Unexpected cached image geometry: {path}")
    image = image.resize(tuple(image_size), Image.Resampling.BILINEAR)
    if jitter is not None:
        for cls, factor in zip((ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color), jitter):
            image = cls(image).enhance(float(factor))
    arr = (np.asarray(image, dtype=np.float32) / 255. - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def current_visibility(projection, image_hw):
    """Conservative existing raster visibility recipe, restricted to current3."""
    p = np.asarray(projection, dtype=np.float64)
    hw = np.asarray(image_hw, dtype=np.float64)
    if p.shape != (3, 4, 4) or hw.shape != (2,) or not np.isfinite(p).all() or not (hw > 0).all():
        raise ValueError("Expected finite current3 projection and [height,width]")
    x = -10. + (np.arange(64) + .5) * 80. / 64.
    y = -32. + (np.arange(48) + .5) * 64. / 48.
    xx, yy = np.meshgrid(x, y, indexing="ij")
    visible = np.zeros(GRID_SHAPE, bool)
    for height in (0., 1.):
        points = np.stack((xx, yy, np.full_like(xx, height), np.ones_like(xx)), -1)
        q = np.einsum("cij,xyj->cxyi", p, points)
        z = q[..., 2]
        u, v = q[..., 0] / np.where(z > .05, z, 1.), q[..., 1] / np.where(z > .05, z, 1.)
        visible |= ((z > .05) & (u >= 0) & (u < hw[1]) & (v >= 0) & (v < hw[0])).any(0)
    return visible[None]


class TemporalPlanDataset(Dataset):
    def __init__(self, base_dataset, *, history_mode="real", auxiliary=False,
                 supervision_root=None, causal_status_root=None):
        if history_mode not in ("repeat", "real"):
            raise ValueError("history_mode must be repeat or real")
        if base_dataset.status_mode != "zero":
            raise ValueError("Underlying planner status must remain constant zero")
        if base_dataset.split not in ("train", "tune"):
            raise ValueError("Only primary train/tune populations are supported")
        if tuple(base_dataset.image_size) != (512, 256):
            raise ValueError("This matched experiment fixes current/history image size to 512x256")
        status = np.asarray(base_dataset.status)
        if status.shape != (len(base_dataset.rows), 8) or status.dtype != np.float32 or np.any(status != 0):
            raise ValueError("Underlying status tensor must be float32 [N,8] exact zero")
        source_sha = _base_source_sha(base_dataset)
        if base_dataset.augment and source_sha != BASE_SOURCE_SHA256:
            raise ValueError("Photometric RNG replay requires the pinned PlanDataset source")
        self.dataset = base_dataset
        self.history_mode, self.auxiliary = history_mode, bool(auxiliary)
        self.base_source_sha256 = source_sha
        # Hash large immutable sources once, never on per-scene cache misses.
        self.expected_split_sha = file_sha(base_dataset.split_manifest)
        self.expected_ego_sha = file_sha(base_dataset.ego_cache)
        self.expected_calibration_sha = file_sha(base_dataset.calibration_path)
        self.rows = np.asarray(base_dataset.rows, np.int64).copy()
        self.rows.setflags(write=False)
        self.supervision_root = Path(supervision_root) if supervision_root else (
            Path(base_dataset.base) / "data/etri/motiondrive_v2/train_tune_geometry_v2")
        self.causal_status_root = Path(causal_status_root) if causal_status_root else (
            Path(base_dataset.base) / "data/etri/motiondrive_v2_shared_status_a1_20260908_ops")
        self.visibility = current_visibility(base_dataset.projection, base_dataset.image_size[::-1])
        self.aux_source = None
        if self.auxiliary:
            p = self.supervision_root / "supervision_manifest.json"
            m = json.loads(p.read_text())
            if (m["split_manifest_sha256"] != self.expected_split_sha
                    or m["ego_cache_sha256"] != self.expected_ego_sha
                    or m.get("canonical_calibration_sha256", m.get("calibration_sha256")) != self.expected_calibration_sha
                    or tuple(m["grid_shape"]) != GRID_SHAPE
                    or tuple(m["grid_extent"]) != GRID_EXTENT):
                raise ValueError("Auxiliary supervision provenance/geometry mismatch")
            self.aux_source = {"directory": str(self.supervision_root), "manifest_sha256": file_sha(p),
                               "fields": ["occ_target", "occ_valid", "lane_target", "lane_valid"],
                               "valid_mask": "original annotation valid AND current3 z0/z1 camera visibility",
                               "visible_cells": int(self.visibility.sum()),
                               "cache_max_scenes_per_worker": 256,
                               "cache_raster_storage": "four bool arrays; convert two binary targets to float32 only at item read"}
        self.causal_status4, self.causal_source = self._load_causal_status()

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def __getattr__(self, name):
        # Keep trainer metadata interface; do not copy mutable epoch state.
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def _load_causal_status(self):
        p = self.causal_status_root / "overlay_manifest.json"
        m = json.loads(p.read_text())
        c = m["causal_contract"]
        if (m["split_manifest_sha256"] != self.expected_split_sha
                or m["ego_cache_sha256"] != self.expected_ego_sha
                or c["future_values_used"]
                or c["fields"][:4] != ["vx", "vy", "ax", "ay"]
                or c["frames"] != list(range(-10, 1))
                or not np.array_equal(np.asarray(c["seconds"]), np.arange(-10, 1) / 10.)):
            raise ValueError("Past-only nominal causal status contract mismatch")
        spec = m["artifacts"][self.dataset.split]
        artifact = self.causal_status_root / spec["file"]
        if artifact.parent.resolve() != self.causal_status_root.resolve() or file_sha(artifact) != spec["sha256"]:
            raise ValueError("Causal status artifact path/hash mismatch")
        with np.load(artifact, allow_pickle=False) as z:
            source_rows, source_frames, source_status = z["row"], z["frame"], z["status5"]
        if source_rows.ndim != 1 or source_rows.dtype.kind not in "iu" or len(np.unique(source_rows)) != len(source_rows):
            raise ValueError("Causal status rows must be unique integers")
        if rows_sha(source_rows) != spec["rows_sha256"] or source_status.shape != (len(source_rows), 5):
            raise ValueError("Causal status population mismatch")
        lookup = {int(row): i for i, row in enumerate(source_rows)}
        try:
            positions = np.asarray([lookup[int(row)] for row in self.rows])
        except KeyError as exc:
            raise ValueError("Requested row absent from causal overlay") from exc
        if not np.array_equal(source_frames[positions], self.dataset.frames[self.rows]):
            raise ValueError("Causal state row/frame mismatch")
        state4 = source_status[positions, :4].astype(np.float32, copy=True)
        if not np.isfinite(state4).all():
            raise ValueError("Nonfinite causal state")
        return state4, {"manifest_sha256": file_sha(p), "path": str(artifact), "sha256": file_sha(artifact),
                        "rows_sha256": rows_sha(self.rows), "fields": ["vx", "vy", "ax", "ay"],
                        "definition": "exact 11 causal pose frames -10..0, nominal 10Hz quadratic fit, current ego XY",
                        "units": ["m/s", "m/s", "m/s^2", "m/s^2"], "future_values_used": False,
                        "use": "loss-only default; explicit common_status maps solely to perception_status"}

    @lru_cache(maxsize=256)
    def _aux(self, scene):
        p = self.supervision_root / (scene + ".npz")
        report = json.loads(p.with_suffix(".json").read_text())
        if (report["scene"] != scene or report["split_manifest_sha256"] != self.expected_split_sha
                or report["ego_cache_sha256"] != self.expected_ego_sha):
            raise ValueError("Auxiliary scene provenance mismatch")
        keys = ("row", "frame", "occ_target", "occ_valid", "lane_target", "lane_valid")
        with np.load(p, allow_pickle=False) as z:
            out = {key: z[key] for key in keys}
        rows, frames = out["row"], out["frame"]
        if rows.dtype.kind not in "iu" or rows.ndim != 1 or len(np.unique(rows)) != len(rows):
            raise ValueError("Auxiliary rows must be unique integers")
        if (np.any(rows < 0) or np.any(rows >= len(self.dataset.frames))
                or not np.array_equal(self.dataset.frames[rows], frames)
                or not np.all(self.dataset.scenarios[self.dataset.scen_idx[rows]].astype(str) == scene)):
            raise ValueError("Auxiliary global row/scene/frame mismatch")
        for key in keys[2:]:
            if out[key].shape != (len(rows), 1, *GRID_SHAPE):
                raise ValueError("Auxiliary raster shape mismatch")
            if key.endswith("valid") and out[key].dtype != np.bool_:
                raise ValueError("Auxiliary validity must remain boolean")
            if key.endswith("target") and (not np.isfinite(out[key]).all() or not np.isin(out[key], (0, 1)).all()):
                raise ValueError("Auxiliary labels must be finite binary rasters")
            # Binary labels need no float storage between reads (~3.16 MiB/scene).
            if key.endswith("target"):
                out[key] = out[key].astype(np.bool_, copy=False)
            out[key].setflags(write=False)
        out["lookup"] = {int(row): i for i, row in enumerate(rows)}
        return out

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        row = int(self.rows[index])
        expected_scene = str(self.dataset.scenarios[self.dataset.scen_idx[row]])
        expected_frame = int(self.dataset.frames[row])
        if (int(item["row"]) != row or item["scenario"] != expected_scene
                or int(item["frame"]) != expected_frame):
            raise ValueError("Underlying item row/scene/frame mismatch")
        if item["images"].shape != (3, 3, 256, 512) or not torch.isfinite(item["images"]).all():
            raise ValueError("Current images must be finite [3,3,256,512]")
        if item["status"].shape != (8,) or torch.any(item["status"] != 0):
            raise ValueError("Underlying item exposed nonzero planner status")
        if self.history_mode == "repeat":
            history = item["images"][FRONT_INDEX].unsqueeze(0).repeat(len(LAGS), 1, 1, 1)
        else:
            jitter = _front_jitter(self.dataset, row)
            images = []
            for lag in LAGS:
                if expected_frame - lag < 0:
                    raise ValueError("History would precede the cached scene")
                path = Path(self.dataset.image_root) / expected_scene / "camera_front" / f"{expected_frame - lag:08d}.jpg"
                images.append(_read_image(path, self.dataset.image_size, jitter))
            history = torch.stack(images)
        item["history_images"] = history
        item["time_offsets"] = torch.tensor((.1, .5), dtype=torch.float32)
        # Distinct tensors clarify the explicit input vs loss boundary.
        item["causal_status4"] = torch.from_numpy(self.causal_status4[index].copy())
        item["state_target"] = item["causal_status4"].clone()
        item["state_valid"] = torch.ones(4, dtype=torch.bool)
        if self.auxiliary:
            z = self._aux(expected_scene)
            try:
                at = z["lookup"][row]
            except KeyError as exc:
                raise ValueError("Current row absent from auxiliary cache") from exc
            if int(z["frame"][at]) != expected_frame:
                raise ValueError("Auxiliary current-row frame mismatch")
            for key in ("occ_target", "lane_target"):
                item[key] = torch.from_numpy(z[key][at].astype(np.float32, copy=True))
            for key in ("occ_valid", "lane_valid"):
                item[key] = torch.from_numpy((z[key][at] & self.visibility).copy())
        return item

    def provenance(self):
        p = dict(self.dataset.provenance())
        p["temporal_data"] = {"source_sha256": file_sha(__file__), "base_source_sha256": self.base_source_sha256,
                              "history_mode": self.history_mode, "frame_lags": list(LAGS),
                              "time_offsets": [.1, .5], "time_source": "constant nominal seconds",
                              "history_camera": "camera_front", "front_index": FRONT_INDEX,
                              "history_wh": [512, 256], "pose_alignment_used": False,
                              "augmentation": "same row/epoch front B-C-C factors for current and both history images",
                              "model_input_keys_default": list(MODEL_INPUT_KEYS),
                              "common_status_opt_in_key": "perception_status",
                              "goal_use": "outer sample only; final selector caller must pass separately",
                              "causal_status": self.causal_source, "auxiliary": self.aux_source}
        return p


def model_inputs(batch, common_status=False):
    if not isinstance(common_status, bool):
        raise ValueError("common_status must be an explicit bool")
    result = {key: batch[key] for key in MODEL_INPUT_KEYS}
    if common_status:
        state = batch["causal_status4"]
        if state.shape[-1:] != (4,) or not torch.isfinite(state).all():
            raise ValueError("perception_status must be finite [...,4]")
        result["perception_status"] = state
    return result
