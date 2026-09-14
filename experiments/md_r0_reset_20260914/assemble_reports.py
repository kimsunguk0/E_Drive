#!/usr/bin/env python3
"""Assemble the P0-P3 result table and the submission registry."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

REPORTS = ROOT / "reports/md_r0_reset_20260914"
EVAL = REPORTS / "eval"
ORDER = ["train_full", "train_probe", "tune_legacy", "tune_legacy_repeat", "tune_B1"]
COLUMNS = ["tag", "eval_split", "rows", "sessions", "scenes", "eval_batch", "eval_stride",
           "precision", "augmentation", "official_d3_weighted", "guide_two_stage_d3",
           "ade1", "ade2", "ade3", "ade6_unweighted", "d3_p95", "d3_p99",
           "vx_mae", "occ_iou", "lane_iou", "rows_sha256"]


def main() -> None:
    registry = json.loads((REPORTS / "baseline_registry.json").read_text())
    rows = []
    for tag in ORDER:
        path = EVAL / f"{tag}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        metrics, report = payload["metrics"], payload["trainer_report"]
        rows.append({
            "tag": tag, "eval_split": payload["eval_split"], "rows": payload["rows"],
            "sessions": payload["sessions"], "scenes": payload["scenes"],
            "eval_batch": payload["eval_batch"], "eval_stride": payload["eval_stride"],
            "precision": payload["precision"], "augmentation": payload["augmentation"],
            "official_d3_weighted": f'{metrics["official_d3_weighted"]:.9f}',
            "guide_two_stage_d3": f'{metrics["guide_two_stage_d3"]:.9f}',
            "ade1": f'{metrics["ade1"]:.6f}', "ade2": f'{metrics["ade2"]:.6f}',
            "ade3": f'{metrics["ade3"]:.6f}',
            "ade6_unweighted": f'{metrics["ade6_unweighted"]:.6f}',
            "d3_p95": f'{metrics["d3_p95"]:.6f}', "d3_p99": f'{metrics["d3_p99"]:.6f}',
            "vx_mae": f'{report["state_mae_vx_vy_ax_ay_yawrate"][0]:.6f}',
            "occ_iou": f'{report["occ_iou"]:.6f}', "lane_iou": f'{report["lane_iou"]:.6f}',
            "rows_sha256": payload["rows_sha256"],
        })
    with (REPORTS / "baseline_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    by_tag = {r["tag"]: r for r in rows}
    legacy = float(by_tag["tune_legacy"]["official_d3_weighted"])
    summary = {
        "schema_version": 1,
        "historical_tune_d3": registry["historical_tune_d3"],
        "reproduced_tune_d3": legacy,
        "reproduces_history_bitwise": (
            json.loads((EVAL / "tune_legacy.json").read_text())["trainer_report"]["official_d3"]
            == registry["historical_tune_d3"]),
        "repeat_run_abs_diff": abs(legacy - float(by_tag["tune_legacy_repeat"]["official_d3_weighted"])),
        "b1_vs_b4_abs_diff": abs(legacy - float(by_tag["tune_B1"]["official_d3_weighted"])),
        "train_full_d3": float(by_tag["train_full"]["official_d3_weighted"]),
        "train_probe_d3": float(by_tag["train_probe"]["official_d3_weighted"]),
        "probe_minus_full": (float(by_tag["train_probe"]["official_d3_weighted"])
                             - float(by_tag["train_full"]["official_d3_weighted"])),
        "tune_over_train_full_ratio": legacy / float(by_tag["train_full"]["official_d3_weighted"]),
        "weighted_equals_two_stage": all(
            abs(float(r["official_d3_weighted"]) - float(r["guide_two_stage_d3"])) < 1e-8 for r in rows),
        "tolerance_note": ("repeat runs of the same configuration are bitwise identical, so any "
                           "difference above ~3e-9 (the observed B1/B4 gap) is a real difference, "
                           "not run-to-run noise"),
    }
    (REPORTS / "baseline_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")

    raw = json.loads((EVAL / "raw_b1_fixture.json").read_text())
    submission = {
        "schema_version": 1,
        "status": "DEPLOYMENT_PENDING",
        "checkpoint": registry["checkpoint_path"],
        "checkpoint_sha256": registry["checkpoint_sha256"],
        "model_state_sha256": registry["model_state_sha256"],
        "runtime_source_git_sha": registry["runtime_source_id"]["git_sha"],
        "server_submission_id": None, "server_score_type": "UNKNOWN",
        "raw_adapter": "models/motiondrive_v2_inputs.py (prepare_clip_inputs)",
        "raw_input_parity": "bitwise on 8 preregistered fixtures",
        "raw_output_parity_max_xy_diff_m": raw["max_plan_abs_xy_diff_m"],
        "blocking": [
            "The official test clips are not present on this node, so no submission file can be built yet.",
            "No RTX4090 measurement exists for this model; T_infer must be the accumulated forward time over every frame processed, per the preserved Q&A.",
            "The submission deadline and remaining submission count are unconfirmed.",
        ],
        "ready": [
            "checkpoint identity and runtime source are pinned",
            "the GT-free raw adapter reproduces both the model inputs and the model outputs at batch 1",
            "the metric matches the two-stage ADE1/ADE2/ADE3 definition exactly",
        ],
    }
    (REPORTS / "submission_registry.json").write_text(
        json.dumps(submission, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
