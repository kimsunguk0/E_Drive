#!/usr/bin/env python3
"""P1 - raw test-shaped B1 parity, inputs AND model outputs.

The existing audit proved the deployment adapter reproduces the six model
inputs bitwise from raw JPEG/parquet, but it never ran the model
(`model_forward_performed: false`).  This closes that gap: for each of the eight
preregistered fixtures it runs R0 at batch 1 on the raw-adapter inputs and on
the training-dataset inputs, and compares the produced trajectories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models import motiondrive_v2_inputs as adapter
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
import motiondrive_v2_training as mt
import train_motiondrive_v2 as trainer

BASE_SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_rawtime.json"
PARENT_SUPERVISION = ROOT / "data/etri/motiondrive_v2/train_tune_geometry_v2"
FIXTURE_ROOT = ROOT / "data/etri/motiondrive_v2/deploy_fixture_train8"
FIXTURE_MANIFEST = ROOT / "reports/motiondrive_v2_deploy_fixture_train8_manifest.json"
REGISTRY = ROOT / "reports/md_r0_reset_20260914/baseline_registry.json"
OUT = ROOT / "reports/md_r0_reset_20260914/eval/raw_b1_fixture.json"


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--init", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()

    registry = json.loads(REGISTRY.read_text())
    manifest = json.loads(FIXTURE_MANIFEST.read_text())
    clips = manifest["clips"]

    payload = torch.load(args.init, map_location="cpu", weights_only=False)
    if mt.tensor_state_sha256(payload["model"]) != registry["model_state_sha256"]:
        raise SystemExit("checkpoint tensors differ from the pinned R0 registry")
    device = torch.device(f"cuda:{args.gpu}")
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    model = MotionDriveV2(config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()

    from motiondrive_v2_data import MotionDriveDataset
    from torch.utils.data import DataLoader

    results = []
    for clip in clips:
        scene, frame = clip["source_scene"], int(clip["source_frame"])
        dataset = MotionDriveDataset(
            data_root="/tmp/pm97", split_manifest=str(BASE_SPLIT),
            split=clip["source_split"], supervision_root=str(PARENT_SUPERVISION),
            min_frame=30, frame_stride=1, augment=False, seed=0,
            history_contract="control", scenes=[scene], frames=[frame])
        if len(dataset) != 1:
            raise SystemExit(f"expected exactly one dataset row for {scene}/{frame}")
        batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
        dataset_inputs = mt.model_inputs(batch, time_input="nominal",
                                         nominal_history_seconds=config.nominal_history_seconds)
        raw_inputs = adapter.prepare_clip_inputs(FIXTURE_ROOT / clip["clip_id"]).inputs

        input_diffs = {}
        for name in adapter.INPUT_KEYS:
            left = dataset_inputs[name].float()
            right = raw_inputs[name].float()
            if left.shape != right.shape:
                raise SystemExit(f"{clip['clip_id']}/{name} shape mismatch {left.shape} {right.shape}")
            difference = (left - right).abs()
            input_diffs[name] = {"max_abs": float(difference.max()),
                                 "bitwise_equal": bool(torch.equal(left, right))}

        with torch.no_grad():
            with trainer.autocast(device, args.precision):
                from_dataset = model(**{k: v.to(device) for k, v in dataset_inputs.items()})["plan_abs"]
                from_raw = model(**{k: v.to(device) for k, v in raw_inputs.items()})["plan_abs"]
        gt = batch["gt_plan"].to(device).float()
        plan_diff = (from_dataset.float() - from_raw.float()).abs()
        results.append({
            "clip_id": clip["clip_id"], "source_scene": scene, "source_frame": frame,
            "input_diffs": input_diffs,
            "inputs_all_bitwise_equal": all(v["bitwise_equal"] for v in input_diffs.values()),
            "plan_max_abs_xy_diff_m": float(plan_diff.max()),
            "plan_mean_abs_xy_diff_m": float(plan_diff.mean()),
            "d3_dataset_path": float(mt.weighted_d3(from_dataset.float(), gt)),
            "d3_raw_adapter_path": float(mt.weighted_d3(from_raw.float(), gt)),
            "d3_abs_diff": float((mt.weighted_d3(from_dataset.float(), gt)
                                  - mt.weighted_d3(from_raw.float(), gt)).abs()),
            "plan_dataset_xy": from_dataset.float().cpu().tolist()[0],
            "plan_raw_xy": from_raw.float().cpu().tolist()[0],
        })

    plan_max = max(r["plan_max_abs_xy_diff_m"] for r in results)
    d3_max = max(r["d3_abs_diff"] for r in results)
    payload_out = {
        "schema_version": 1,
        "purpose": "raw test-shaped B1 input AND output parity for R0",
        "checkpoint": str(Path(args.init).resolve()),
        "checkpoint_sha256": registry["checkpoint_sha256"],
        "fixtures": len(results), "batch": 1, "precision": args.precision,
        "device": torch.cuda.get_device_name(device),
        "inputs_all_bitwise_equal": all(r["inputs_all_bitwise_equal"] for r in results),
        "max_plan_abs_xy_diff_m": plan_max,
        "max_d3_abs_diff": d3_max,
        "mean_d3_dataset_path": float(np.mean([r["d3_dataset_path"] for r in results])),
        "mean_d3_raw_adapter_path": float(np.mean([r["d3_raw_adapter_path"] for r in results])),
        "clips": results,
        "limitations": [
            "Eight preregistered TRAIN clips; this is transport parity, not generalization.",
            "No submission file is produced and no timing is claimed here.",
            "The raw adapter decodes JPEG independently; bf16 autocast still applies to both paths.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload_out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: payload_out[k] for k in
                      ("fixtures", "inputs_all_bitwise_equal", "max_plan_abs_xy_diff_m",
                       "max_d3_abs_diff", "mean_d3_dataset_path", "mean_d3_raw_adapter_path")},
                     indent=1))


if __name__ == "__main__":
    main()
