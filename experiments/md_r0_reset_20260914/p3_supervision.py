#!/usr/bin/env python3
"""P3 - supervision targets for the expanded train split.

The geometry_v2 edition differs from the rawtime edition only in the single
global `calibration.npz` (the rear-wide projection fix); every per-scene file
was copied byte-identically.  So the A scenes are built with the SAME corrected
calibration and joined to the existing 241 scene files.  Nothing existing is
modified, and the split-provenance SHA check stays in force: the new root
carries the new split manifest's SHA.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_grouped_split_v2 import sha256

BASE_SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_rawtime.json"
NEW_SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json"
PARENT_ROOT = ROOT / "data/etri/motiondrive_v2/train_tune_geometry_v2"
OUT_ROOT = ROOT / "data/etri/motiondrive_v2/r0reset_tplus_geometry_v2"
PARITY_ROOT = ROOT / "data/etri/motiondrive_v2/r0reset_parity_probe"
REPORT = ROOT / "reports/md_r0_reset_20260914/supervision_report.json"
CACHE_META = ROOT / "cache/etri_768/cache_meta.json"
BUILDER = ROOT / "scripts/build_scene_supervision_v2.py"
PYTHON = ROOT / "env/venv/bin/python"
CAMERA_ORDER = ("camera_front", "camera_front_left", "camera_front_right",
                "camera_rear_left", "camera_rear_right", "camera_rear_wide")


def run_builder(output_root, scenes, calibration, split="train"):
    command = [str(PYTHON), str(BUILDER),
               "--split-manifest", str(NEW_SPLIT), "--split", split,
               "--output-root", str(output_root),
               "--calibration", str(calibration),
               "--scenes", ",".join(scenes)]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "scripts"), environment.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    completed = subprocess.run(command, capture_output=True, text=True, env=environment)
    if completed.returncode != 0:
        raise SystemExit(f"supervision builder failed:\n{completed.stderr[-4000:]}")
    return len(completed.stdout.strip().splitlines())


def npz_arrays_equal(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if sorted(a.files) != sorted(b.files):
            return False, {"keys_differ": True}
        diffs = {}
        for key in sorted(a.files):
            x, y = a[key], b[key]
            same = x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y)
            if not same:
                diffs[key] = {"shape": [list(x.shape), list(y.shape)],
                              "dtype": [str(x.dtype), str(y.dtype)]}
        return not diffs, diffs


def camera_metadata_matches(scenes, reference_scene):
    metadata = json.loads(CACHE_META.read_text())
    reference = {camera: metadata[reference_scene][camera] for camera in CAMERA_ORDER}
    mismatched = []
    for scene in scenes:
        entry = metadata.get(scene)
        if entry is None or any(entry.get(camera) != reference[camera] for camera in CAMERA_ORDER):
            mismatched.append(scene)
    return reference_scene, mismatched


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--parity-scenes", type=int, default=3)
    args = parser.parse_args()

    base = json.loads(BASE_SPLIT.read_text())
    new = json.loads(NEW_SPLIT.read_text())
    new_sha = sha256(NEW_SPLIT)
    parent_contract = json.loads((PARENT_ROOT / "supervision_manifest.json").read_text())
    added = sorted(set(new["splits"]["train"]) - set(base["splits"]["train"]))
    kept = sorted(set(base["splits"]["train"]) | set(base["splits"]["tune"]))

    reference_scene, mismatched = camera_metadata_matches(added, kept[0])
    result = {
        "schema_version": 1,
        "new_split_manifest": str(NEW_SPLIT), "new_split_manifest_sha256": new_sha,
        "parent_supervision_root": str(PARENT_ROOT),
        "parent_supervision_manifest_sha256": sha256(PARENT_ROOT / "supervision_manifest.json"),
        "parent_geometry_edition": parent_contract.get("geometry_edition"),
        "output_root": str(OUT_ROOT),
        "added_scenes": len(added), "kept_scenes": len(kept),
        "camera_metadata_reference_scene": reference_scene,
        "camera_metadata_mismatched_added_scenes": mismatched,
        "written": bool(args.write),
    }
    if mismatched:
        result["status"] = "BLOCKED_RIG_MISMATCH"
        REPORT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        raise SystemExit("added scenes do not share the train/tune camera metadata")

    if not args.write:
        REPORT.write_text(json.dumps(result | {"status": "dry_run"}, indent=1, sort_keys=True) + "\n")
        print(json.dumps({k: result[k] for k in
                          ("added_scenes", "kept_scenes",
                           "camera_metadata_mismatched_added_scenes")}, indent=1))
        return

    if OUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT_ROOT}")

    # 1. the corrected global calibration, byte-identical to the parent edition
    OUT_ROOT.mkdir(parents=True)
    shutil.copy2(PARENT_ROOT / "calibration.npz", OUT_ROOT / "calibration.npz")
    calibration_sha = sha256(OUT_ROOT / "calibration.npz")
    if calibration_sha != parent_contract["canonical_calibration_sha256"]:
        raise SystemExit("copied calibration does not match the parent canonical SHA")
    result["canonical_calibration_sha256"] = calibration_sha

    # 2. parity: rebuild a few EXISTING scenes with the same inputs and compare
    parity_scenes = kept[:args.parity_scenes]
    if PARITY_ROOT.exists():
        shutil.rmtree(PARITY_ROOT)
    PARITY_ROOT.mkdir(parents=True)
    shutil.copy2(PARENT_ROOT / "calibration.npz", PARITY_ROOT / "calibration.npz")
    run_builder(PARITY_ROOT, parity_scenes, PARITY_ROOT / "calibration.npz", split="train")
    parity = {}
    for scene in parity_scenes:
        same, diffs = npz_arrays_equal(PARITY_ROOT / f"{scene}.npz", PARENT_ROOT / f"{scene}.npz")
        parity[scene] = {"arrays_bitwise_identical": bool(same), "differences": diffs}
    result["parity_probe"] = {"scenes": parity_scenes, "results": parity,
                              "all_identical": all(v["arrays_bitwise_identical"]
                                                   for v in parity.values())}
    if not result["parity_probe"]["all_identical"]:
        result["status"] = "BLOCKED_PARITY"
        REPORT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        raise SystemExit("rebuilding an existing scene did not reproduce it bitwise")

    # 3. build the added scenes into the new root
    built = run_builder(OUT_ROOT, added, OUT_ROOT / "calibration.npz", split="train")
    result["built_scene_reports"] = built

    # 4. join the existing scene files by hard link (no copy, no modification)
    linked = 0
    for scene in kept:
        for suffix in (".npz", ".json"):
            source, target = PARENT_ROOT / f"{scene}{suffix}", OUT_ROOT / f"{scene}{suffix}"
            if target.exists():
                raise SystemExit(f"unexpected pre-existing {target}")
            os.link(source, target)
            linked += 1
    result["linked_files"] = linked

    # 5. the assembled contract: parent's schema-2 contract, new split SHA
    builder_contract = json.loads((OUT_ROOT / "supervision_manifest.json").read_text())
    if builder_contract["split_manifest_sha256"] != new_sha:
        raise SystemExit("builder wrote a different split SHA than the new manifest")
    for key in ("ego_cache_sha256", "grid_shape", "grid_extent",
                "history_frame_offsets", "state_target_order", "state_fit"):
        if builder_contract[key] != parent_contract[key]:
            raise SystemExit(f"builder contract diverges from the parent edition at {key}")
    contract = copy.deepcopy(parent_contract)
    contract["split_manifest_sha256"] = new_sha
    contract["r0_reset_expansion"] = {
        "purpose": "E1 expanded-train (Tplus) supervision",
        "parent_root": str(PARENT_ROOT),
        "parent_supervision_manifest_sha256": result["parent_supervision_manifest_sha256"],
        "parent_split_manifest_sha256": parent_contract["split_manifest_sha256"],
        "added_scenes": added,
        "added_scene_count": len(added),
        "kept_scene_count": len(kept),
        "kept_scene_files": "hard links to the parent edition; bytes unchanged",
        "builder": "scripts/build_scene_supervision_v2.py",
        "builder_sha256": sha256(BUILDER),
        "builder_calibration": "parent canonical calibration.npz (geometry_v2 corrected)",
        "canonical_calibration_sha256": calibration_sha,
        "parity_probe": result["parity_probe"],
        "note": ("geometry_v2 differs from the rawtime edition only in the global "
                 "calibration.npz, so building new scenes against that corrected "
                 "calibration reproduces the same edition for them."),
    }
    (OUT_ROOT / "supervision_manifest.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    result["supervision_manifest_sha256"] = sha256(OUT_ROOT / "supervision_manifest.json")

    # 6. the dataset must actually load the expanded split
    from motiondrive_v2_data import MotionDriveDataset
    counts = {}
    for split, stride in (("train", 1), ("tune", 5), ("val", 1)):
        dataset = MotionDriveDataset(
            data_root="/tmp/pm97", split_manifest=str(NEW_SPLIT), split=split,
            supervision_root=str(OUT_ROOT), min_frame=30, frame_stride=stride,
            augment=False, seed=0, history_contract="control")
        counts[split] = {"rows": len(dataset), "scenes": len(set(dataset.scene_names[dataset.rows]))}
    result["dataset_counts"] = counts
    result["status"] = "created_verified"

    REPORT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "added_scenes": len(added),
                      "parity_all_identical": result["parity_probe"]["all_identical"],
                      "dataset_counts": counts,
                      "supervision_manifest_sha256": result["supervision_manifest_sha256"]},
                     indent=1))


if __name__ == "__main__":
    main()
