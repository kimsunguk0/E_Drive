#!/usr/bin/env python3
"""Submission registry: the R0 baseline entry plus the E1-EXP candidate."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/NHNHOME/data/sukim/adcl")
REPORTS = ROOT / "reports/md_r0_reset_20260914"
OUT = REPORTS / "submission_registry.json"


def main() -> None:
    baseline = json.loads((REPORTS / "baseline_registry.json").read_text())
    candidate = json.loads((REPORTS / "candidate_registry.json").read_text())
    official = json.loads((REPORTS / "submission/official_test_E1-EXP.validation.json").read_text())
    fixture = json.loads((REPORTS / "submission/fixture8_E1-EXP.validation.json").read_text())
    reproduction = json.loads((REPORTS / "reproduction_manifest.json").read_text())

    registry = {
        "schema_version": 2,
        "entries": {
            "R0": {
                "role": "baseline, preserved",
                "status": "NO_SUBMISSION_FILE_BUILT",
                "checkpoint": baseline["checkpoint_path"],
                "checkpoint_sha256": baseline["checkpoint_sha256"],
                "model_state_sha256": baseline["model_state_sha256"],
                "v0_official_d3": baseline["historical_tune_d3"],
                "server_submission_id": None, "server_score_type": "UNKNOWN",
                "note": ("a file can be built with the same writer on request; the candidate is "
                         "the better model, so quota is not spent on this one by default"),
            },
            "E1-EXP_terminal": {
                "role": "current candidate",
                "status": "SUBMISSION_FILE_BUILT_AND_VALIDATED_NOT_UPLOADED",
                "checkpoint": candidate["checkpoint_path"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "model_state_sha256": candidate["model_state_sha256"],
                "v0_official_d3": candidate["results"]["v0_b4_official_d3"],
                "submission_file": official["submission_file"],
                "submission_sha256": official["submission_sha256"],
                "submission_bytes": official["submission_bytes"],
                "clips_written": official["clips_written"],
                "clips_available": official["clips_available"],
                "build_seconds": official["elapsed_seconds"],
                "format": official["format"],
                "validation": official["checks"],
                "fixture_dry_run": {
                    "clips": fixture["clips"],
                    "submission_sha256": fixture["submission_sha256"],
                },
                "reproduction_manifest": str(REPORTS / "reproduction_manifest.json"),
                "server_submission_id": None, "server_score_type": "UNKNOWN",
            },
        },
        "writer": "experiments/md_r0_reset_20260914/submit_official_test.py",
        "inputs_read_per_clip": official["inputs"],
        "runtime_source_git_sha": reproduction["runtime_source_git_sha"],
        "upload_policy": {
            "uploaded": False,
            "quota_consumed": 0,
            "requires": ["user approval", "confirmed remaining submission count",
                         "confirmed current submission form and field naming"],
            "guide_document_values": {
                "submission_period_ends": "2026-09-23 per the printed guide page 1",
                "max_submissions_per_team": "5 per the printed guide page 4",
                "caveat": ("these are document values; the live deadline, time zone and actual "
                           "remaining count were not confirmed from the account screen"),
            },
        },
        "pending": [
            "No RTX4090 timing exists, so the Error Score time term cannot be computed.",
            "H confirmation is unopened.",
            "Seed 1 replication is still running.",
            "The candidate may change if EXP-LONG improves; a new candidate requires rebuilding "
            "this file and re-linking its SHA in this entry.",
        ],
    }
    OUT.write_text(json.dumps(registry, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"entries": list(registry["entries"]),
                      "candidate_file": registry["entries"]["E1-EXP_terminal"]["submission_file"],
                      "candidate_sha256": registry["entries"]["E1-EXP_terminal"]["submission_sha256"],
                      "clips": registry["entries"]["E1-EXP_terminal"]["clips_written"],
                      "uploaded": registry["upload_policy"]["uploaded"]}, indent=1))


if __name__ == "__main__":
    main()
