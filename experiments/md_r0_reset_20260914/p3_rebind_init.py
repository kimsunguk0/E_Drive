#!/usr/bin/env python3
"""P3 - rebind the R0 initializer onto the expanded split manifest.

The trainer refuses an initializer whose manifest names a different split
manifest.  That check is kept, not deleted: instead this writes a NEW
initializer artifact whose model tensors are bitwise identical to R0 and whose
manifest declares the expanded split, together with the envelope proof that the
expanded split only ADDS training scenes and leaves tune untouched.

Both E1 arms start from this one artifact, so T203 and EXP differ in data only.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_grouped_split_v2 import sha256
from motiondrive_v2_training import tensor_state_sha256

BASE_SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_rawtime.json"
NEW_SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json"
SOURCE = ROOT / "work_dirs/motiondrive_v2/q10_q10_flip50_s0/last.pth"
TARGET = ROOT / "work_dirs/md_r0_reset_20260914/r0_init_tplus.pth"
REGISTRY = ROOT / "reports/md_r0_reset_20260914/baseline_registry.json"
REPORT = ROOT / "reports/md_r0_reset_20260914/init_rebind_report.json"
NEW_SUPERVISION = ROOT / "data/etri/motiondrive_v2/r0reset_tplus_geometry_v2"


def main() -> None:
    registry = json.loads(REGISTRY.read_text())
    base = json.loads(BASE_SPLIT.read_text())
    new = json.loads(NEW_SPLIT.read_text())
    base_sha, new_sha = sha256(BASE_SPLIT), sha256(NEW_SPLIT)

    envelope = {
        "tune_identical": base["splits"]["tune"] == new["splits"]["tune"],
        "train_is_superset": set(new["splits"]["train"]) >= set(base["splits"]["train"]),
        "added_train_scenes": len(set(new["splits"]["train"]) - set(base["splits"]["train"])),
        "removed_train_scenes": len(set(base["splits"]["train"]) - set(new["splits"]["train"])),
        "val_is_subset_of_prior_val": set(new["splits"]["val"]) <= set(base["splits"]["val"]),
        "added_train_scenes_come_from_prior_val": (
            set(new["splits"]["train"]) - set(base["splits"]["train"])
            <= set(base["splits"]["val"])),
        "tune_never_enters_train": not (set(new["splits"]["train"]) & set(new["splits"]["tune"])),
        "val_never_enters_train": not (set(new["splits"]["train"]) & set(new["splits"]["val"])),
        "prior_manifest_sha256_recorded": (
            new["r0_reset_envelope"]["prior_manifest_sha256"] == base_sha),
        "initializer_declared_split_sha256": registry["train_split_id"]["split_manifest_sha256"],
        "initializer_split_matches_prior": (
            registry["train_split_id"]["split_manifest_sha256"] == base_sha),
    }
    failures = [k for k, v in envelope.items() if v is False]
    if failures:
        REPORT.write_text(json.dumps({"status": "BLOCKED_ENVELOPE", "failures": failures,
                                      "envelope": envelope}, indent=1, sort_keys=True) + "\n")
        raise SystemExit(f"expanded split is not a safe envelope: {failures}")

    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source_state_sha = tensor_state_sha256(payload["model"])
    if source_state_sha != registry["model_state_sha256"]:
        raise SystemExit("source checkpoint tensors differ from the pinned R0 registry")

    manifest = copy.deepcopy(payload["manifest"])
    manifest["split_sha256"] = new_sha
    manifest["r0_reset_rebind"] = {
        "purpose": "E1 initializer; identical weights, expanded split declaration",
        "source_checkpoint": str(SOURCE),
        "source_checkpoint_sha256": registry["checkpoint_sha256"],
        "source_model_state_sha256": source_state_sha,
        "prior_split_manifest": str(BASE_SPLIT), "prior_split_manifest_sha256": base_sha,
        "new_split_manifest": str(NEW_SPLIT), "new_split_manifest_sha256": new_sha,
        "new_supervision_root": str(NEW_SUPERVISION),
        "new_supervision_manifest_sha256": sha256(NEW_SUPERVISION / "supervision_manifest.json"),
        "envelope": envelope,
        "weights_changed": False,
        "optimizer_state_dropped": True,
        "note": ("the trainer's initializer/split lineage check is untouched; this "
                 "artifact makes the lineage explicit instead of bypassing it"),
    }
    rebound = {"model": payload["model"], "manifest": manifest,
               "step": payload["step"], "epoch": payload.get("epoch", 0),
               "best_metric": payload.get("best_metric")}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        raise FileExistsError(f"Refusing to overwrite {TARGET}")
    torch.save(rebound, TARGET)

    check = torch.load(TARGET, map_location="cpu", weights_only=False)
    written_sha = tensor_state_sha256(check["model"])
    report = {
        "status": "created_verified",
        "target": str(TARGET), "target_sha256": sha256(TARGET),
        "model_state_sha256": written_sha,
        "model_state_identical_to_r0": written_sha == registry["model_state_sha256"],
        "declared_split_manifest_sha256": check["manifest"]["split_sha256"],
        "envelope": envelope,
        "source": str(SOURCE), "source_sha256": registry["checkpoint_sha256"],
    }
    if not report["model_state_identical_to_r0"]:
        raise SystemExit("rebound initializer changed the weights")
    REPORT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
