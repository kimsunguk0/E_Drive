"""Offline future non-ego footprint targets in the CURRENT ego frame.

Future object boxes at +5 and +10 frames are rasterized with the current
frame's world->ego transform and the current frame's camera visibility, so the
target says where annotated objects will be, expressed where the model is now.
Everything reuses the pinned current-occupancy builder, which fixes the grid,
the corner convention including the documented width/length swap, and the
conservative support rule. Nothing here reaches a model input.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
LAGS = (5, 10)
LABEL_NAME = "future_annotated_object_footprint"


def load_builder():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "_scene_supervision", ROOT / "scripts/build_scene_supervision_v2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_scene_supervision"] = mod
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    spec.loader.exec_module(mod)
    return mod


def build_scene(bs, scene, source, rows, cache, lidar2img, min_frame=30):
    frames, timestamps, poses, grouped, _lines, _schema = bs.read_scene_metadata(source)
    lookup = {int(frame): i for i, frame in enumerate(frames)}
    visible = bs.camera_visible(lidar2img)
    selected = [int(r) for r in rows if cache["frame"][r] >= min_frame]
    out = {k: [] for k in ("row", "frame", "future_target", "future_valid",
                           "future_present", "future_seconds")}
    for row in selected:
        frame = int(cache["frame"][row])
        now = lookup[frame]
        world_to_ego = np.linalg.inv(poses[now])
        targets, valids, present, seconds = [], [], [], []
        for lag in LAGS:
            index = lookup.get(frame + lag)
            records = grouped.get(frame + lag, []) if index is not None else []
            has_objects = index is not None and any(
                str(r["class"]).lower() != "ego" for r in records)
            if not has_objects:
                targets.append(np.zeros(bs.GRID_SHAPE, np.float32))
                valids.append(np.zeros(bs.GRID_SHAPE, bool))
                present.append(False)
                seconds.append(np.float32("nan"))
                continue
            # Future boxes, current transform, current visibility.
            target, valid = bs.rasterize_objects(records, world_to_ego, visible)
            targets.append(target[0])
            valids.append(valid[0])
            present.append(True)
            seconds.append(np.float32((timestamps[index] - timestamps[now]) / 1000.0))
        out["row"].append(row)
        out["frame"].append(frame)
        out["future_target"].append(np.stack(targets))
        out["future_valid"].append(np.stack(valids))
        out["future_present"].append(np.array(present))
        out["future_seconds"].append(np.array(seconds, np.float32))
    arrays = {k: np.asarray(v) for k, v in out.items()}
    positives = [(arrays["future_target"][:, i].astype(bool) & arrays["future_valid"][:, i]).sum()
                 for i in range(len(LAGS))]
    negatives = [(~arrays["future_target"][:, i].astype(bool) & arrays["future_valid"][:, i]).sum()
                 for i in range(len(LAGS))]
    report = {"scene": scene, "rows": len(selected),
              "present_fraction": arrays["future_present"].mean(0).tolist(),
              "positive_valid_cells": [int(x) for x in positives],
              "negative_valid_cells": [int(x) for x in negatives],
              "seconds_median": np.nanmedian(arrays["future_seconds"], 0).tolist()}
    return arrays, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--supervision-root", required=True)
    p.add_argument("--meta-root", default=str(ROOT / "data/etri/meta_train"))
    p.add_argument("--ego-cache", default="/tmp/pm97/data/etri/ego_cache.npz")
    p.add_argument("--output", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "tune"])
    p.add_argument("--limit-scenes", type=int, default=0)
    a = p.parse_args()
    bs = load_builder()
    out = Path(a.output).resolve()
    if out.exists():
        raise SystemExit(f"output is immutable: {out}")
    split = json.loads(Path(a.split_manifest).read_text())
    cache = np.load(a.ego_cache, allow_pickle=False)
    calibration = np.load(Path(a.supervision_root) / "calibration.npz", allow_pickle=False)
    scen = cache["scen_idx"]
    names = cache["scenarios"].astype(str)
    started = time.time()
    out.mkdir(parents=True)
    reports, totals = [], np.zeros((len(LAGS), 2), np.int64)
    for name in a.splits:
        scenes = sorted(split["splits"][name])
        if a.limit_scenes:
            scenes = scenes[:a.limit_scenes]
        store = {}
        for scene in scenes:
            index = int(np.where(names == scene)[0][0])
            rows = np.flatnonzero(scen == index)
            lidar2img = calibration["lidar2img"]   # one shared rig for every scene
            arrays, report = build_scene(bs, scene, Path(a.meta_root) / scene, rows,
                                         cache, lidar2img)
            reports.append(report)
            if name == "train":      # the class weight is fixed from train only
                totals[:, 0] += np.array(report["positive_valid_cells"])
                totals[:, 1] += np.array(report["negative_valid_cells"])
            for k, v in arrays.items():
                store.setdefault(k, []).append(v)
            if len(reports) % 25 == 0:
                print(json.dumps({"scenes_done": len(reports)}), flush=True)
        merged = {k: np.concatenate(v, 0) for k, v in store.items()}
        order = np.argsort(merged["row"])
        merged = {k: v[order] for k, v in merged.items()}
        np.savez_compressed(out / f"{name}.npz", **merged)
    # Class weight is fixed once, from train only, and clipped as the spec asks.
    ratio = (totals[:, 1] / np.maximum(totals[:, 0], 1)).astype(float)
    manifest = {"schema": "future_footprint_target_v1", "label": LABEL_NAME,
                "lags_frames": list(LAGS), "grid_shape": list(bs.GRID_SHAPE),
                "grid_extent": list(bs.GRID_EXTENT),
                "coordinate_note": ("future boxes rasterized with the CURRENT world->ego "
                                    "transform and CURRENT camera visibility"),
                "support_note": bs.ANNOTATION_CONTRACT["support"],
                "class_weight_source": "train split only",
                "positive_valid_cells": totals[:, 0].tolist(),
                "negative_valid_cells": totals[:, 1].tolist(),
                "raw_neg_over_pos": ratio.tolist(),
                "pos_weight_clipped_1_20": np.clip(ratio, 1.0, 20.0).tolist(),
                "splits": a.splits, "scenes": len(reports),
                "elapsed_seconds": time.time() - started}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    (out / "per_scene.json").write_text(json.dumps(reports, indent=1, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
