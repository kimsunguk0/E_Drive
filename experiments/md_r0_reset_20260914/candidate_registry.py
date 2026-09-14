#!/usr/bin/env python3
"""Register E1-EXP as a candidate, separately from the R0 baseline entry."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_grouped_split_v2 import sha256
from motiondrive_v2_training import tensor_state_sha256

REPORTS = ROOT / "reports/md_r0_reset_20260914"
RUN = ROOT / "work_dirs/md_r0_reset_20260914/E1-EXP"
CHECKPOINT = RUN / "ckpt_step20554.pth"
OUT = REPORTS / "candidate_registry.json"
# Identifiers carried in the work order; the local artifact is what decides.
EXPECTED = {
    "checkpoint_sha256": "019faf708f006895aaad7ae57029e625daad091bcfbfc4ceb74d47f9daad245f",
    "model_state_sha256": "d5b8f0aefa4f88684fb3184499457cbd507336813c40d0fe7006a7b29c93b228",
}


def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def main() -> None:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    experiment = json.loads((RUN / "experiment.json").read_text())
    manifest = json.loads((RUN / "manifest.json").read_text())
    final = json.loads((RUN / "final_eval.json").read_text())["report"]

    measured = {"checkpoint_sha256": sha256(CHECKPOINT),
                "model_state_sha256": tensor_state_sha256(payload["model"])}
    mismatch = {k: {"work_order": v, "measured": measured[k]}
                for k, v in EXPECTED.items() if measured[k] != v}

    paired = load(REPORTS / "e1_paired_analysis.json", {})
    negative = load(REPORTS / "negative_controls.json", {}).get("E1-EXP_terminal", {})
    raw_b1 = load(REPORTS / "eval/raw_b1_fixture_E1-EXP_terminal.json", {})
    cost = load(REPORTS / "forward_cost.json", {}).get("E1-EXP_terminal", {})
    submission = load(REPORTS / "submission/fixture8_E1-EXP.validation.json", {})
    probe = load(REPORTS / "eval/E1-EXP_probe_step20554.json", {})
    b1 = load(REPORTS / "eval/E1-EXP_tuneB1_step20554.json", {})

    entry = {
        "schema_version": 1,
        "candidate": "E1-EXP_terminal",
        "status": "CANDIDATE_LOCKED_FOR_SUBMISSION_PREP",
        "supersedes_baseline": False,
        "baseline_entry": "reports/md_r0_reset_20260914/baseline_registry.json (R0, preserved)",
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": measured["checkpoint_sha256"],
        "model_state_sha256": measured["model_state_sha256"],
        "work_order_identifier_check": {
            "expected": EXPECTED, "measured": measured,
            "matches": not mismatch, "mismatch": mismatch,
        },
        "step": int(payload["step"]),
        "model_config": manifest["model_config"],
        "training": {
            "launcher": "experiments/md_r0_reset_20260914/train.py",
            "runtime_source_git_sha": manifest["git_sha"],
            "initializer": experiment["initializer"],
            "split_manifest": experiment["split_manifest"],
            "supervision": experiment["supervision"],
            "train_data": experiment["train_data"],
            "tune_data": experiment["tune_data"],
            "recipe": experiment["recipe"],
            "elapsed_seconds": manifest["elapsed_seconds"],
            "nonfinite_count": manifest["nonfinite_count"],
            "optimizer_checkpoint_preserved": str(RUN / "last.pth"),
        },
        "information_contract": {
            "provided_numeric_ego_status": "not an input anywhere",
            "goal": "shared scene-feature conditioning only, never a planner input",
            "command_vad_cmd": "off",
            "past_relative_pose": "image alignment only; the frames it aligns are always fed as images",
            "not_rgb_only": ("past relative pose alignment remains, so this model must not be "
                             "described as free of all ego-motion information"),
            "output": "six cumulative absolute XY points in the current ego frame",
        },
        "results": {
            "v0_b4_official_d3": final["official_d3"],
            "v0_b1_official_d3": (b1.get("metrics") or {}).get("official_d3_weighted"),
            "v0_session_mean_d3": final["session_mean_d3"],
            "t0_train_probe_d3": (probe.get("metrics") or {}).get("official_d3_weighted"),
            "vx_mae": final["state_mae_vx_vy_ax_ay_yawrate"][0],
            "occ_iou": final["occ_iou"], "lane_iou": final["lane_iou"],
            "paired_vs_r0": (paired.get("comparisons") or {}).get("R0 -> E1-EXP"),
            "paired_vs_t203": (paired.get("comparisons") or {}).get("E1-T203 -> E1-EXP"),
            "checkpoint_selection": "best on V0 among planned steps == terminal (20554)",
        },
        "checks": {
            "raw_b1_input_output_parity": {
                "fixtures": raw_b1.get("fixtures"),
                "inputs_bitwise_equal": raw_b1.get("inputs_all_bitwise_equal"),
                "max_plan_abs_xy_diff_m": raw_b1.get("max_plan_abs_xy_diff_m"),
            },
            "negative_controls": {
                name: {"d3": value["official_d3"], "ratio_vs_normal": value["d3_ratio_vs_normal"]}
                for name, value in (negative.get("conditions") or {}).items()
            },
            "forward_cost": {
                "forwards_per_clip": cost.get("forwards_per_clip"),
                "profiler_gflops_total": cost.get("profiler_gflops_total"),
                "flops_cutoff_gflops": cost.get("flops_cutoff_gflops"),
                "b200_median_ms": (cost.get("latency_ms") or {}).get("median"),
                "rtx4090_measured": False,
            },
            "submission_writer": {
                "validated_on": submission.get("clips_root"),
                "clips": submission.get("clips"),
                "cumsum_not_reapplied_ratio": ((submission.get("checks") or {})
                                               .get("cumsum_not_reapplied") or {}).get("ratio"),
                "clip_state_isolation_pass": (submission.get("checks") or {}).get("clip_state_isolation_pass"),
                "submission_sha256": submission.get("submission_sha256"),
            },
        },
        "pending": [
            "H confirmation is unopened and stays unopened until the design is locked.",
            "The official test clips are not on this node, so no real submission file exists yet.",
            "No RTX4090 timing; the Error Score time term cannot be computed.",
            "Deadline and remaining submission count are unconfirmed; no upload has been made.",
            "Seed 1 replication pair is still running.",
        ],
    }
    OUT.write_text(json.dumps(entry, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "candidate": entry["candidate"],
        "identifiers_match_work_order": entry["work_order_identifier_check"]["matches"],
        "v0_b4": entry["results"]["v0_b4_official_d3"],
        "raw_b1_output_diff_m": entry["checks"]["raw_b1_input_output_parity"]["max_plan_abs_xy_diff_m"],
        "image_swap_ratio": (entry["checks"]["negative_controls"].get("image_swap_cross_scene") or {}).get("ratio_vs_normal"),
        "written": str(OUT)}, indent=1))


if __name__ == "__main__":
    main()
