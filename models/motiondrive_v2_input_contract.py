"""Dependency-free input contract shared by CPU packaging and raw-clip serving.

Importing this module must not initialize Torch, OpenCV, Pillow or CUDA.
It describes inputs; it does not create them or certify checkpoint lineage.
"""
from __future__ import annotations

from typing import Mapping

from .motiondrive_v2_temporal_contract import temporal_contract

CAMERA_ORDER = ("camera_front", "camera_front_right", "camera_front_left",
                "camera_rear_wide", "camera_rear_left", "camera_rear_right")
PAST_FRAMES = (-1, -2, -5, -10)
POSE_FRAMES = frozenset(range(-30, 1)) | {50}
NOMINAL_SECONDS = (.1, .2, .5, 1.)
INPUT_KEYS = ("images", "history_images", "lidar2img", "history_transforms", "time_offsets", "goal_xy")
CURRENT_WH, HISTORY_WH, CROP_WH = (768, 432), (384, 216), (1920, 1080)


def input_contract(history_contract: str = "control") -> dict:
    """Serving bundles must explicitly bind checkpoints to this new contract."""
    temporal = temporal_contract(history_contract)
    result = {"schema_version": 1, "geometry_edition": "geometry_v2", "time_input": "nominal",
            "camera_order": list(CAMERA_ORDER), "current_wh": list(CURRENT_WH),
            "history_wh": list(HISTORY_WH), "history_frames": [-x for x in temporal.frame_offsets],
            "nominal_seconds": list(temporal.nominal_seconds), "dtype": "float32",
            "pose_alignment": "inv(E_past) @ E_current; full SE3; scene branch only",
            "goal": "(R_current.T @ (p_plus50 - p_current))[:2]",
            "crop": "front and rear_wide bottom; other four top; horizontal center 1920x1080",
            "pixels": "OpenCV undistort/INTER_AREA768 -> JPEG Q95 -> PIL RGB -> ImageNet normalization",
            "history_pixels": "PIL BILINEAR384 from decoded Q95 768 image, before normalization",
            "raw_image_reads": 10, "cross_clip_feature_cache": False,
            "provided_status_or_timestamp_input": False, "future_labels_input": False,
            "projection_precision": "raw calibration float64 -> final float32; archived NPZ rounding may differ"}
    # Preserve the historical serialized contract byte-for-byte at the default.
    if temporal.name != "control":
        result["history_contract"] = temporal.name
    return result


def require_input_contract(contract: Mapping, history_contract: str | None = None) -> None:
    """Do not silently treat a raw-time/old-geometry checkpoint as deployment-ready."""
    if not isinstance(contract, Mapping):
        raise ValueError("Explicit geometry_v2/nominal input contract is required")
    if history_contract is None:
        history_contract = contract.get("history_contract", "control") if isinstance(contract, Mapping) else "control"
    expected_contract = input_contract(history_contract)
    if set(contract) != set(expected_contract):
        raise ValueError("Incompatible input contract keyset")
    for name, expected in expected_contract.items():
        if name in ("schema_version", "geometry_edition", "time_input", "camera_order",
                    "current_wh", "history_wh", "history_frames", "nominal_seconds", "dtype"):
            actual = contract.get(name)
            if isinstance(expected, list) and isinstance(actual, tuple):
                actual = list(actual)
            if actual != expected:
                raise ValueError(f"Incompatible input contract {name}: {actual!r} != {expected!r}")
