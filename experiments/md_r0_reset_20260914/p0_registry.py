#!/usr/bin/env python3
"""P0 - pin the R0 baseline identity with measured values only.

Reads the existing R0 artifacts, recomputes hashes, checks that the A2
provided-status route is absent from the checkpoint, and writes
reports/md_r0_reset_20260914/baseline_registry.json.  No training, no eval.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from motiondrive_v2_training import tensor_state_sha256

RUN = ROOT / "work_dirs/motiondrive_v2/q10_q10_flip50_s0"
OUT = ROOT / "reports/md_r0_reset_20260914/baseline_registry.json"

# Names that would indicate the A2 provided-status query route is present.
A2_MARKERS = ("shared_status", "status_query", "provided_status", "status_delta")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads((RUN / "manifest.json").read_text())
    final_eval = json.loads((RUN / "final_eval.json").read_text())
    report = final_eval["report"]
    payload = torch.load(RUN / "last.pth", map_location="cpu", weights_only=False)

    state = payload["model"] if "model" in payload else payload["model_state"]
    marked = sorted(k for k in state if any(m in k for m in A2_MARKERS))
    params = sum(int(v.numel()) for v in state.values() if hasattr(v, "numel"))

    protocol = manifest["experimental_protocol"]
    source = protocol["source"]
    pinned = source["file_sha256"]
    drift = {}
    for rel, expected in sorted(pinned.items()):
        path = ROOT / rel
        actual = sha256(path) if path.exists() else None
        if actual != expected:
            drift[rel] = {"expected": expected, "actual": actual}

    registry = {
        "schema_version": 1,
        "written_by": "experiments/md_r0_reset_20260914/p0_registry.py",
        "run_name": "q10_q10_flip50_s0",
        "checkpoint_path": str(RUN / "last.pth"),
        "checkpoint_sha256": sha256(RUN / "last.pth"),
        "model_state_sha256": tensor_state_sha256(state),
        "checkpoint_keys": sorted(payload.keys()),
        "checkpoint_step": payload.get("step"),
        "parameter_count": params,
        "initial_model_state_sha256": manifest["initial_model_state_sha256"],
        "initial_parameter_count": manifest["initial_parameter_count"],
        "model_config": manifest["model_config"],
        "loss_weights": manifest["loss_weights"],
        "recipe": protocol["recipe"],
        "runtime_source_id": {
            "git_sha": source["git_sha"],
            "source_manifest": source["path"],
            "source_manifest_sha256": source["sha256"],
            "pinned_file_count": len(pinned),
            "pinned_file_drift": drift,
            "pinned_files_unchanged": not drift,
        },
        "parent": protocol["parent"],
        "train_split_id": {
            "split_manifest": manifest["arguments"]["split_manifest"],
            "split_manifest_sha256": manifest["split_sha256"],
            "split": "train",
            "scenes": 203,
            "sessions": 72,
            "rows": protocol["train_data"]["rows"],
            "rows_sha256": protocol["train_data"]["rows_sha256"],
            "train_scenes_argument": manifest["arguments"].get("train_scenes"),
        },
        "tune_split_id": {
            "split": "tune",
            "scenes": 37,
            "sessions": 11,
            "rows": protocol["evaluation_data"]["rows"],
            "rows_sha256": protocol["evaluation_data"]["rows_sha256"],
            "eval_stride": manifest["arguments"].get("eval_stride"),
        },
        "preprocessing_id": {
            "data_root": manifest["arguments"]["data_root"],
            "image_cache": "/NHNHOME/data/sukim/adcl/cache/etri_768",
            "supervision_root": manifest["arguments"]["supervision_root"],
            "supervision_manifest_sha256": manifest["supervision_manifest_sha256"],
            "time_input_policy": manifest["time_input_policy"],
            "geometry_edition": "cache_meta_rear_wide_v2",
        },
        "historical_tune_d3": report["official_d3"],
        "historical_eval_mode": {
            "eval_batch": protocol["recipe"]["eval_batch"],
            "precision": protocol["recipe"]["precision"],
            "bn_running_statistics": protocol["recipe"]["bn_running_statistics"],
            "time_input": manifest["time_input"],
            "eval_stride": manifest["arguments"].get("eval_stride"),
            "metric": manifest["metric"],
            "n_rows": report["n"],
            "n_sessions": report["n_sessions"],
            "augmentation": "off (evaluation dataset is never augmented)",
        },
        "historical_session_d3": report["session_d3"],
        "historical_auxiliary": {
            "state_mae_vx_vy_ax_ay_yawrate": report["state_mae_vx_vy_ax_ay_yawrate"],
            "history_position_mae_by_offset": report["history_position_mae_by_offset"],
            "occ_iou": report["occ_iou"],
            "lane_iou": report["lane_iou"],
        },
        "a2_provided_status_route": {
            "protocol_status_route": protocol["status_route"],
            "protocol_provided_status_used": protocol["provided_status_used"],
            "checkpoint_parameters_matching_a2_markers": marked,
            "absent": not marked,
        },
        "torch": manifest["torch"],
        "numpy": manifest["numpy"],
        "server_submission_id": None,
        "server_score_type": "UNKNOWN",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(registry, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "written": str(OUT),
        "checkpoint_sha256": registry["checkpoint_sha256"],
        "model_state_sha256": registry["model_state_sha256"],
        "parameter_count": params,
        "a2_absent": registry["a2_provided_status_route"]["absent"],
        "pinned_files_unchanged": registry["runtime_source_id"]["pinned_files_unchanged"],
        "historical_tune_d3": registry["historical_tune_d3"],
    }, indent=1))


if __name__ == "__main__":
    main()
