#!/usr/bin/env python3
"""Raw test-shaped B1 input AND output parity for the MR graph.

Mirrors p1_raw_b1 but for the matching-resolution model: the raw adapter must
reproduce the cached loader's tensors, including the two canvas tensors the MR
motion branch needs, and the model must agree on both paths.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments/md_r0_reset_20260914")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import motiondrive_v2_training as mt
import train_motiondrive_v2 as trainer
from models import motiondrive_v2_inputs as adapter
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
import matching_resolution as mr
import mr_deploy

FIXTURE_ROOT = ROOT / "data/etri/motiondrive_v2/deploy_fixture_train8"
FIXTURE_MANIFEST = ROOT / "reports/motiondrive_v2_deploy_fixture_train8_manifest.json"
BASE_SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_rawtime.json"
PARENT_SUPERVISION = ROOT / "data/etri/motiondrive_v2/train_tune_geometry_v2"
OUT = ROOT / "reports/md_r0_reset_20260914/eval/mr_raw_b1_fixture.json"
KEYS = tuple(adapter.INPUT_KEYS) + ("motion_current", "motion_history")


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--init", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--detail", choices=["native", "lowdetail"], default="native")
    parser.add_argument("--label", default="MR-NATIVE-s0")
    args = parser.parse_args()

    clips = json.loads(FIXTURE_MANIFEST.read_text())["clips"]
    payload = torch.load(args.init, map_location="cpu", weights_only=False)
    device = torch.device(f"cuda:{args.gpu}")
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    model = MotionDriveV2(config)
    rebuild = mr.rebuild_correlation_fuse(model, mr.NEW_RADIUS)
    mr.install(model, args.detail)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()

    from motiondrive_v2_data import MotionDriveDataset
    from torch.utils.data import DataLoader

    results = []
    for clip in clips:
        scene, frame = clip["source_scene"], int(clip["source_frame"])
        base = MotionDriveDataset(
            data_root="/tmp/pm97", split_manifest=str(BASE_SPLIT),
            split=clip["source_split"], supervision_root=str(PARENT_SUPERVISION),
            min_frame=30, frame_stride=1, augment=False, seed=0,
            history_contract="control", scenes=[scene], frames=[frame])
        dataset = mr.MotionCanvasDataset(base, args.detail)
        if len(dataset) != 1:
            raise SystemExit(f"expected exactly one dataset row for {scene}/{frame}")
        batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
        dataset_inputs = mr.model_inputs_with_canvas(
            mt.model_inputs, batch, time_input="nominal",
            nominal_history_seconds=config.nominal_history_seconds)
        raw_inputs = mr_deploy.prepare_mr_clip_inputs(
            FIXTURE_ROOT / clip["clip_id"], detail=args.detail).inputs

        diffs = {}
        for name in KEYS:
            left, right = dataset_inputs[name].float(), raw_inputs[name].float()
            if left.shape != right.shape:
                raise SystemExit(f"{clip['clip_id']}/{name} shape {left.shape} vs {right.shape}")
            diffs[name] = {"max_abs": float((left - right).abs().max()),
                           "bitwise_equal": bool(torch.equal(left, right))}

        with torch.no_grad(), trainer.autocast(device, args.precision):
            from_dataset = model(**{k: v.to(device) for k, v in dataset_inputs.items()})["plan_abs"]
            from_raw = model(**{k: v.to(device) for k, v in raw_inputs.items()})["plan_abs"]
        gt = batch["gt_plan"].to(device).float()
        plan_diff = (from_dataset.float() - from_raw.float()).abs()
        results.append({
            "clip_id": clip["clip_id"], "source_scene": scene, "source_frame": frame,
            "input_diffs": diffs,
            "inputs_all_bitwise_equal": all(v["bitwise_equal"] for v in diffs.values()),
            "plan_max_abs_xy_diff_m": float(plan_diff.max()),
            "plan_mean_abs_xy_diff_m": float(plan_diff.mean()),
            "d3_dataset_path": float(mt.weighted_d3(from_dataset.float(), gt)),
            "d3_raw_adapter_path": float(mt.weighted_d3(from_raw.float(), gt)),
        })

    payload_out = {
        "schema_version": 1,
        "purpose": "raw test-shaped B1 input AND output parity for the MR graph",
        "label": args.label, "detail": args.detail,
        "checkpoint": str(Path(args.init).resolve()),
        "checkpoint_sha256": trainer.sha256(args.init),
        "model_state_sha256": mt.tensor_state_sha256(payload["model"]),
        "graph": {"correlation_radius": mr.NEW_RADIUS, "rebuild": rebuild,
                  "motion_canvas_wh": list(mr.MOTION_WH)},
        "fixtures": len(results), "batch": 1, "precision": args.precision,
        "checked_keys": list(KEYS),
        "inputs_all_bitwise_equal": all(r["inputs_all_bitwise_equal"] for r in results),
        "max_plan_abs_xy_diff_m": max(r["plan_max_abs_xy_diff_m"] for r in results),
        "max_d3_abs_diff": max(abs(r["d3_dataset_path"] - r["d3_raw_adapter_path"]) for r in results),
        "per_fixture": results,
        "limitations": [
            "Fixtures are train-sourced clips shaped like the test input, not official test clips.",
            "Parity is about the input pipeline and the forward, not about accuracy.",
        ],
    }
    OUT.write_text(json.dumps(payload_out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: payload_out[k] for k in
                      ("fixtures", "inputs_all_bitwise_equal",
                       "max_plan_abs_xy_diff_m", "max_d3_abs_diff")}, indent=1))


if __name__ == "__main__":
    main()
