#!/usr/bin/env python3
"""Deployment-side input preparation for the MR graph.

The MR motion branch needs a 768x432 front-camera canvas for the current frame
and for each of the four past frames. The existing adapter already decodes every
one of those images at exactly that geometry; it then downsamples the past ones
to 384x216 for the scene branch. This module keeps the 768x432 version as well,
so no new image is read and no new geometry is introduced.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import motiondrive_v2_inputs as adapter
from models.motiondrive_v2_input_contract import CURRENT_WH, HISTORY_WH
from models.motiondrive_v2_temporal_contract import temporal_contract

BOTTLENECK_WH = (384, 216)


def motion_canvas(image, detail):
    """One canvas tensor from an already-decoded 768x432 cache image."""
    if detail == "lowdetail":
        from PIL import Image
        image = image.resize(BOTTLENECK_WH, Image.Resampling.BILINEAR)
        image = image.resize(CURRENT_WH, Image.Resampling.BILINEAR)
    elif detail != "native":
        raise ValueError("detail must be native or lowdetail")
    return adapter.normalize_image(image, CURRENT_WH)


def prepare_mr_clip_from_records(calibration_rows, ego_pose_rows, image_loader,
                                 history_contract="control", detail="native"):
    prepared = adapter.prepare_clip_from_records(
        calibration_rows, ego_pose_rows, image_loader, history_contract)
    temporal = temporal_contract(history_contract)
    front = adapter.build_camera_geometry(calibration_rows)[0]
    name = adapter.CAMERA_ORDER[0]

    def canvas(frame):
        image, _ = adapter.cache_compatible_image(image_loader(name, frame), front)
        return motion_canvas(image, detail)

    current = canvas(0)
    history = torch.stack([canvas(-offset) for offset in temporal.frame_offsets])
    prepared.inputs["motion_current"] = current.unsqueeze(0)
    prepared.inputs["motion_history"] = history.unsqueeze(0)

    # For native detail the current canvas IS the front camera the scene branch
    # already received, so this is an identity that must hold exactly. It is the
    # cheapest available check that the canvas geometry did not drift.
    front_from_scene = prepared.inputs["images"][0, 0]
    prepared.metadata["mr"] = {
        "detail": detail,
        "motion_current_wh": list(CURRENT_WH),
        "motion_history_frames": len(temporal.frame_offsets),
        "current_canvas_equals_scene_front": bool(torch.equal(front_from_scene, current)),
        "current_canvas_max_abs_diff_vs_scene_front": float((front_from_scene - current).abs().max()),
        "images_decoded": "the same ten images the base adapter reads; none added",
        "note": ("history_images stays 384x216 for the scene branch; the motion branch "
                 "gets the 768x432 version of those same decoded frames"),
    }
    return prepared


def prepare_mr_clip_inputs(clip_dir, history_contract="control", detail="native"):
    import pyarrow.parquet as pq
    from models.motiondrive_v2_inputs import CALIBRATION_COLUMNS, POSE_COLUMNS
    clip_dir = Path(clip_dir)
    calibration = pq.read_table(clip_dir / "calibration.parquet",
                                columns=list(CALIBRATION_COLUMNS)).to_pylist()
    poses = pq.read_table(clip_dir / "ego_pose.parquet",
                          columns=list(POSE_COLUMNS)).to_pylist()

    def read(camera, frame):
        return (clip_dir / camera / f"frame_{frame}.jpg").read_bytes()

    return prepare_mr_clip_from_records(calibration, poses, read, history_contract, detail)
