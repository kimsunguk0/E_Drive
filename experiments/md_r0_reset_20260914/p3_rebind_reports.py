#!/usr/bin/env python3
"""P3 - rebind the per-scene provenance of the kept supervision reports.

The loader checks `split_manifest_sha256` inside every per-scene JSON sidecar,
so the 240 scenes hard-linked from the parent edition still name the previous
split manifest.  Only that one field is rewritten, in a real file that replaces
the hard link; the .npz targets keep the parent's inode and bytes.  Every
original JSON hash is recorded so the change stays auditable.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_grouped_split_v2 import sha256

BASE_SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_rawtime.json"
NEW_SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json"
PARENT_ROOT = ROOT / "data/etri/motiondrive_v2/train_tune_geometry_v2"
OUT_ROOT = ROOT / "data/etri/motiondrive_v2/r0reset_tplus_geometry_v2"
REPORT = ROOT / "reports/md_r0_reset_20260914/supervision_report.json"


def main() -> None:
    base_sha, new_sha = sha256(BASE_SPLIT), sha256(NEW_SPLIT)
    base = json.loads(BASE_SPLIT.read_text())
    kept = sorted(set(base["splits"]["train"]) | set(base["splits"]["tune"]))

    rebound, npz_checked, unchanged = {}, 0, 0
    for scene in kept:
        target = OUT_ROOT / f"{scene}.json"
        report = json.loads(target.read_text())
        if report.get("scene") != scene:
            raise SystemExit(f"scene report names the wrong scene: {scene}")
        if report.get("split_manifest_sha256") == new_sha:
            unchanged += 1
            continue
        if report.get("split_manifest_sha256") != base_sha:
            raise SystemExit(f"unexpected prior split provenance in {scene}")
        prior_sha = sha256(target)
        report["split_manifest_sha256"] = new_sha
        target.unlink()
        target.write_text(json.dumps(report, indent=2) + "\n")
        rebound[scene] = {"prior_json_sha256": prior_sha, "json_sha256": sha256(target)}
        # the targets themselves must still be the parent's bytes
        if sha256(OUT_ROOT / f"{scene}.npz") != sha256(PARENT_ROOT / f"{scene}.npz"):
            raise SystemExit(f"target npz diverged from the parent edition: {scene}")
        npz_checked += 1

    # every scene the expanded splits need must now name the new manifest
    new = json.loads(NEW_SPLIT.read_text())
    required = sorted({s for key in ("train", "tune", "val") for s in new["splits"][key]})
    wrong = [s for s in required
             if json.loads((OUT_ROOT / f"{s}.json").read_text()).get("split_manifest_sha256") != new_sha]
    if wrong:
        raise SystemExit(f"{len(wrong)} scene reports still name another split: {wrong[:4]}")

    contract_path = OUT_ROOT / "supervision_manifest.json"
    contract = json.loads(contract_path.read_text())
    contract["r0_reset_expansion"]["kept_scene_files"] = (
        "targets are hard links to the parent edition (bytes unchanged); the per-scene "
        "JSON sidecars were replaced with copies whose split_manifest_sha256 names the "
        "expanded manifest")
    contract["r0_reset_expansion"]["kept_scene_report_rebind"] = {
        "prior_split_manifest_sha256": base_sha,
        "new_split_manifest_sha256": new_sha,
        "rebound_scene_count": len(rebound),
        "already_correct_scene_count": unchanged,
        "npz_verified_identical_to_parent": npz_checked,
        "changed_fields": ["split_manifest_sha256"],
        "scene_hashes": rebound,
    }
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")

    report_payload = json.loads(REPORT.read_text())
    report_payload["supervision_manifest_sha256"] = sha256(contract_path)
    report_payload["kept_scene_report_rebind"] = {
        "rebound": len(rebound), "already_correct": unchanged,
        "npz_verified_identical_to_parent": npz_checked,
        "all_required_scene_reports_name_new_split": True}
    REPORT.write_text(json.dumps(report_payload, indent=1, sort_keys=True) + "\n")

    # finally, actually read one row per split through the loader
    from motiondrive_v2_data import MotionDriveDataset
    sampled = {}
    for split, stride in (("train", 1), ("tune", 5), ("val", 1)):
        dataset = MotionDriveDataset(
            data_root="/tmp/pm97", split_manifest=str(NEW_SPLIT), split=split,
            supervision_root=str(OUT_ROOT), min_frame=30, frame_stride=stride,
            augment=False, seed=0, history_contract="control")
        first, last = dataset[0], dataset[len(dataset) - 1]
        sampled[split] = {"rows": len(dataset),
                          "first_scene": first["scenario"], "last_scene": last["scenario"],
                          "images": list(first["images"].shape)}
    print(json.dumps({"rebound_scene_reports": len(rebound),
                      "already_correct": unchanged,
                      "npz_verified_identical_to_parent": npz_checked,
                      "supervision_manifest_sha256": sha256(contract_path),
                      "loader_sample": sampled}, indent=1))


if __name__ == "__main__":
    main()
