#!/usr/bin/env python3
"""P3 - finish the expanded supervision root: add H, verify, write the report.

H needs targets too, otherwise the final confirmation in §10.3 cannot be run
later.  Building targets reads no model and reveals no performance.
"""
from __future__ import annotations

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
HOLDOUT_ROOT = ROOT / "data/etri/motiondrive_v2/r0reset_holdout_build"
REPORT = ROOT / "reports/md_r0_reset_20260914/supervision_report.json"
BUILDER = ROOT / "scripts/build_scene_supervision_v2.py"
PYTHON = ROOT / "env/venv/bin/python"


def run_builder(output_root, scenes, calibration, split):
    command = [str(PYTHON), str(BUILDER),
               "--split-manifest", str(NEW_SPLIT), "--split", split,
               "--output-root", str(output_root),
               "--calibration", str(calibration),
               "--scenes", ",".join(scenes)]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "scripts")])
    completed = subprocess.run(command, capture_output=True, text=True, env=environment)
    if completed.returncode != 0:
        raise SystemExit(f"builder failed:\n{completed.stderr[-4000:]}")
    return len(completed.stdout.strip().splitlines())


def npz_arrays_equal(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if sorted(a.files) != sorted(b.files):
            return False, {"keys_differ": True}
        diffs = {k: True for k in sorted(a.files)
                 if not (a[k].shape == b[k].shape and a[k].dtype == b[k].dtype
                         and np.array_equal(a[k], b[k]))}
        return not diffs, diffs


def main() -> None:
    base = json.loads(BASE_SPLIT.read_text())
    new = json.loads(NEW_SPLIT.read_text())
    new_sha = sha256(NEW_SPLIT)
    contract_path = OUT_ROOT / "supervision_manifest.json"
    contract = json.loads(contract_path.read_text())
    parent_contract = json.loads((PARENT_ROOT / "supervision_manifest.json").read_text())

    added = sorted(set(new["splits"]["train"]) - set(base["splits"]["train"]))
    kept = sorted(set(base["splits"]["train"]) | set(base["splits"]["tune"]))
    holdout = sorted(new["splits"]["val"])

    calibration = OUT_ROOT / "calibration.npz"
    calibration_sha = sha256(calibration)
    if calibration_sha != parent_contract["canonical_calibration_sha256"]:
        raise SystemExit("output calibration is not the parent canonical file")

    # parity probe, recomputed from what is on disk
    parity_scenes = sorted(p.stem for p in PARITY_ROOT.glob("*.npz") if p.stem != "calibration")
    parity = {}
    for scene in parity_scenes:
        same, diffs = npz_arrays_equal(PARITY_ROOT / f"{scene}.npz", PARENT_ROOT / f"{scene}.npz")
        parity[scene] = {"arrays_bitwise_identical": bool(same), "differing_arrays": sorted(diffs)}
    all_identical = bool(parity) and all(v["arrays_bitwise_identical"] for v in parity.values())
    if not all_identical:
        raise SystemExit(f"parity probe failed: {json.dumps(parity)}")

    # build H into its own root, then hard link it in
    missing = [s for s in holdout if not (OUT_ROOT / f"{s}.npz").exists()]
    built = 0
    if missing:
        if HOLDOUT_ROOT.exists():
            shutil.rmtree(HOLDOUT_ROOT)
        HOLDOUT_ROOT.mkdir(parents=True)
        shutil.copy2(calibration, HOLDOUT_ROOT / "calibration.npz")
        built = run_builder(HOLDOUT_ROOT, missing, HOLDOUT_ROOT / "calibration.npz", "val")
        for scene in missing:
            for suffix in (".npz", ".json"):
                os.link(HOLDOUT_ROOT / f"{scene}{suffix}", OUT_ROOT / f"{scene}{suffix}")

    expansion = dict(contract.get("r0_reset_expansion", {}))
    expansion.update({
        "parent_root": str(PARENT_ROOT),
        "parent_supervision_manifest_sha256": sha256(PARENT_ROOT / "supervision_manifest.json"),
        "parent_split_manifest_sha256": parent_contract["split_manifest_sha256"],
        "added_train_scenes": added, "added_train_scene_count": len(added),
        "holdout_scenes": holdout, "holdout_scene_count": len(holdout),
        "holdout_note": ("H targets are built so the single final confirmation can run; "
                         "no model has been evaluated on them"),
        "kept_scene_count": len(kept),
        "kept_scene_files": "hard links to the parent edition; bytes unchanged",
        "builder": "scripts/build_scene_supervision_v2.py",
        "builder_sha256": sha256(BUILDER),
        "builder_calibration": "parent canonical calibration.npz (geometry_v2 corrected)",
        "canonical_calibration_sha256": calibration_sha,
        "parity_probe": {"scenes": parity_scenes, "results": parity,
                         "all_identical": all_identical},
        "note": ("geometry_v2 differs from the rawtime edition only in the global "
                 "calibration.npz; new scenes built against that corrected "
                 "calibration therefore belong to the same edition."),
    })
    contract["split_manifest_sha256"] = new_sha
    contract["r0_reset_expansion"] = expansion
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")

    from motiondrive_v2_data import MotionDriveDataset
    counts = {}
    for split, stride in (("train", 1), ("tune", 5), ("val", 1)):
        dataset = MotionDriveDataset(
            data_root="/tmp/pm97", split_manifest=str(NEW_SPLIT), split=split,
            supervision_root=str(OUT_ROOT), min_frame=30, frame_stride=stride,
            augment=False, seed=0, history_contract="control")
        counts[split] = {"rows": len(dataset),
                         "scenes": int(len(set(dataset.scene_names[dataset.rows])))}
    # the legacy root must still serve the legacy split unchanged
    legacy = MotionDriveDataset(
        data_root="/tmp/pm97", split_manifest=str(BASE_SPLIT), split="train",
        supervision_root=str(PARENT_ROOT), min_frame=30, frame_stride=1,
        augment=False, seed=0, history_contract="control")

    report = {
        "schema_version": 1, "status": "created_verified",
        "new_split_manifest": str(NEW_SPLIT), "new_split_manifest_sha256": new_sha,
        "output_root": str(OUT_ROOT),
        "supervision_manifest_sha256": sha256(contract_path),
        "canonical_calibration_sha256": calibration_sha,
        "added_train_scenes": len(added), "kept_scenes": len(kept),
        "holdout_scenes": len(holdout), "holdout_scene_reports_built": built,
        "parity_probe": expansion["parity_probe"],
        "dataset_counts": counts,
        "legacy_root_unchanged_train_rows": len(legacy),
        "parent_supervision_manifest_sha256": expansion["parent_supervision_manifest_sha256"],
    }
    REPORT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("status", "added_train_scenes", "holdout_scenes",
                       "dataset_counts", "legacy_root_unchanged_train_rows",
                       "supervision_manifest_sha256")}
                     | {"parity_all_identical": all_identical}, indent=1))


if __name__ == "__main__":
    main()
