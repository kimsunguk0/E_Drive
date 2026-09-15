#!/usr/bin/env python3
"""Fact-find the command signal before any command experiment is specified.

Section 5 of the work order requires checking how command is actually provided,
its classes, its missingness and what a left/right flip would have to do to it,
BEFORE writing a spec.  This reads data only: no model, no training, no GPU.
"""
from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "reports/md_r0_reset_20260914/command_inventory.json"
TEST_ROOT = ROOT / "test"
META_TRAIN = Path("/tmp/pm97/data/etri/meta_train")
SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json"
TEST_SAMPLE = 200


def read_test_commands(limit):
    import pyarrow.parquet as pq
    rows = []
    for tar_path in sorted(TEST_ROOT.glob("*.tar"))[:limit]:
        with tarfile.open(tar_path) as archive:
            member = next((m for m in archive.getmembers()
                           if m.name.endswith("/command.parquet")), None)
            if member is None:
                rows.append({"clip": tar_path.stem, "missing": True})
                continue
            table = pq.read_table(io.BytesIO(archive.extractfile(member).read()))
            entry = {"clip": tar_path.stem, "missing": False,
                     "columns": table.column_names, "rows": table.num_rows}
            entry.update({name: table.column(name).to_pylist()
                          for name in table.column_names})
            rows.append(entry)
    return rows


def read_train_commands(scenes, limit):
    import pyarrow.parquet as pq
    found, missing, samples = 0, [], []
    for scene in scenes[:limit]:
        candidates = list((META_TRAIN / scene).glob("**/command.parquet"))
        if not candidates:
            missing.append(scene)
            continue
        found += 1
        if len(samples) < 5:
            table = pq.read_table(candidates[0])
            samples.append({"scene": scene, "path": str(candidates[0]),
                            "columns": table.column_names, "rows": table.num_rows,
                            "head": {c: table.column(c).to_pylist()[:5]
                                     for c in table.column_names}})
    return found, missing, samples


def main() -> None:
    split = json.loads(SPLIT.read_text())
    train_scenes = sorted(split["splits"]["train"])

    test_rows = read_test_commands(TEST_SAMPLE)
    present = [r for r in test_rows if not r["missing"]]
    columns = sorted({tuple(r["columns"]) for r in present})
    counts = {}
    for column in ("command", "vad_cmd"):
        values = [v for r in present for v in r.get(column, [])]
        counts[column] = {
            "n": len(values),
            "distinct": sorted({str(v) for v in values}),
            "distribution": dict(Counter(str(v) for v in values).most_common()),
            "nulls": sum(1 for v in values if v is None),
        }

    train_found, train_missing, train_samples = read_train_commands(train_scenes, 40)
    meta_listing = sorted({p.name for scene in train_scenes[:3]
                           for p in (META_TRAIN / scene).iterdir()}) if META_TRAIN.exists() else []

    report = {
        "schema_version": 1,
        "purpose": "pre-specification fact-find for a command experiment; no experiment started",
        "test_set": {
            "clips_inspected": len(test_rows),
            "clips_with_command_parquet": len(present),
            "clips_missing_command_parquet": sum(1 for r in test_rows if r["missing"]),
            "column_sets": [list(c) for c in columns],
            "rows_per_clip": sorted({r["rows"] for r in present}),
            "values": counts,
            "note": "one row per clip: the command at the current frame, not a per-frame series",
        },
        "train_side": {
            "scenes_inspected": min(40, len(train_scenes)),
            "scenes_with_command_parquet": train_found,
            "scenes_without": train_missing[:10],
            "meta_directory_contents_sample": meta_listing,
            "samples": train_samples,
        },
        "flip_mapping_requirement": {
            "issue": ("the training augmentation mirrors the ego frame about y = 0, so any "
                      "left/right command class has to be swapped in the same item, and a "
                      "double flip must return the original class"),
            "unresolved_until_classes_are_known": True,
        },
        "input_boundary_if_used": {
            "allowed_form": ("shared scene-feature conditioning only, as a small zero-init "
                             "residual on the scene query context, mirroring how goal is used"),
            "forbidden": ["raw command into the planner",
                          "command into the ego-state head or waypoint queries",
                          "filling a missing command from ground-truth future motion"],
            "control_design": ("CMD0 versus CMD1 must carry the identical module and parameter "
                               "count, differing only in whether the slots hold the real command "
                               "or null, so capacity is not the contrast"),
        },
        "status": "FACT_FIND_ONLY_NO_SPEC_NO_EXPERIMENT",
    }
    OUT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "test_clips_inspected": report["test_set"]["clips_inspected"],
        "test_with_command": report["test_set"]["clips_with_command_parquet"],
        "columns": report["test_set"]["column_sets"],
        "rows_per_clip": report["test_set"]["rows_per_clip"],
        "command_distribution": counts["command"]["distribution"],
        "vad_cmd_distribution": counts["vad_cmd"]["distribution"],
        "train_scenes_with_command": f'{train_found}/{min(40, len(train_scenes))}',
        "meta_dir_sample": meta_listing,
    }, indent=1))


if __name__ == "__main__":
    main()
