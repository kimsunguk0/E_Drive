#!/usr/bin/env python3
"""Make a non-versioned Phase-5B fixture from one real ETRI frame/calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from sparse_scoredrive import CAMERA_ORDER, build_cached_lidar2img


MEAN = np.asarray([123.675, 116.28, 103.53], np.float32).reshape(1, 1, 3)
STD = np.asarray([58.395, 57.12, 57.375], np.float32).reshape(1, 1, 3)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--index", type=int, default=30)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ann = Path(args.ann)
    data = pickle.loads(ann.read_bytes())
    info = data["infos"][args.index]
    images = []
    source_files = []
    for name in CAMERA_ORDER:
        original = Path(info["cams"][name]["data_path"])
        # Cache layout is <root>/<scenario>/<camera>/<frame>.jpg.
        scenario = original.parts[-3]
        cached = Path(args.cache_root) / scenario / name / original.name
        rgb = np.asarray(Image.open(cached).convert("RGB"), dtype=np.float32)
        if rgb.shape != (432, 768, 3):
            raise ValueError(f"unexpected cached image {cached}: {rgb.shape}")
        images.append(((rgb - MEAN) / STD).transpose(2, 0, 1))
        source_files.append({"path": str(cached), "sha256": sha256(cached)})
    lidar2img = build_cached_lidar2img(info)
    manifest = {
        "ann": str(ann),
        "ann_sha256": sha256(ann),
        "index": args.index,
        "scene_token": info["scene_token"],
        "frame_idx": int(info["frame_idx"]),
        "camera_order": list(CAMERA_ORDER),
        "source_files": source_files,
        "geometry": "undistorted intrinsic + crop(1920x1080) + scale(0.4)",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=np.asarray(images, np.float32),
        lidar2img=lidar2img,
        manifest=np.asarray(json.dumps(manifest, sort_keys=True)),
    )
    print(json.dumps({"out": str(out), "sha256": sha256(out), **manifest}, indent=2))


if __name__ == "__main__":
    main()
