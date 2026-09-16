#!/usr/bin/env python3
"""Build and validate the official submission JSON from test-shaped clips.

The official example (tools/etri_test_submit.py) writes {clip_token: [[x, y] x 6]}
and gets there by cumsum-ing VAD's per-step deltas.  MotionDrive V2 already
returns cumulative absolute XY, so the cumsum must NOT be applied again; this
writer asserts that and records the check.

Runs through the same GT-free adapter as deployment: raw JPEG and parquet only.
"""
from __future__ import annotations

import argparse
import hashlib
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

REQUIRED = ("calibration.parquet", "ego_pose.parquet", "camera_front/frame_0.jpg")


def discover(root: Path):
    clips = []
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        if all((entry / name).exists() for name in REQUIRED):
            clips.append(entry)
    return clips


def predict(model, device, precision, clip_dir):
    prepared = (mr_deploy.prepare_mr_clip_inputs(clip_dir, detail=MR_DETAIL[0])
                if MR_DETAIL[0] else adapter.prepare_clip_inputs(clip_dir))
    inputs = {k: v.to(device) for k, v in prepared.inputs.items()}
    with torch.no_grad(), trainer.autocast(device, precision):
        plan = model(**inputs)["plan_abs"]
    return plan.float().cpu().numpy()[0], prepared.metadata


import matching_resolution as mr
import mr_deploy

MR_DETAIL = [None]


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--init", required=True)
    parser.add_argument("--clips-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mr-detail", choices=["native", "lowdetail"],
                        help="build with the matching-resolution graph")
    args = parser.parse_args()

    payload = torch.load(args.init, map_location="cpu", weights_only=False)
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    model = MotionDriveV2(config)
    MR_DETAIL[0] = args.mr_detail
    if args.mr_detail:
        mr.rebuild_correlation_fuse(model, mr.NEW_RADIUS)
        mr.install(model, args.mr_detail)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(f"cuda:{args.gpu}")
    model.to(device).eval()

    clips = discover(Path(args.clips_root))
    if args.limit:
        clips = clips[:args.limit]
    if not clips:
        raise SystemExit(f"no test-shaped clips under {args.clips_root}")

    submission, per_clip = {}, []
    for clip in clips:
        plan, metadata = predict(model, device, args.precision, clip)
        if plan.shape != (6, 2):
            raise SystemExit(f"{clip.name}: expected [6,2], got {plan.shape}")
        if not np.isfinite(plan).all():
            raise SystemExit(f"{clip.name}: non-finite coordinates")
        distance = np.linalg.norm(plan, axis=-1)
        steps = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), plan]), axis=0), axis=-1)
        submission[clip.name] = plan.astype(np.float64).tolist()
        per_clip.append({
            "clip": clip.name,
            "endpoint_distance_m": float(distance[-1]),
            "first_point_distance_m": float(distance[0]),
            "max_step_m": float(steps.max()),
            "cumulative_distance_is_monotone": bool(np.all(np.diff(distance) > -1e-6)),
            "adapter_source_sha256": metadata["adapter_source_sha256"],
        })

    # Absolute versus delta: a second cumsum would roughly double the endpoint.
    endpoints = np.asarray([c["endpoint_distance_m"] for c in per_clip])
    first = np.asarray([c["first_point_distance_m"] for c in per_clip])
    double_applied = np.asarray(
        [np.linalg.norm(np.cumsum(np.asarray(submission[c["clip"]]), axis=0)[-1])
         for c in per_clip])

    # Clip isolation: re-running the first clip after the others must be identical.
    replay, _ = predict(model, device, args.precision, clips[0])
    isolation = float(np.abs(replay - np.asarray(submission[clips[0].name])).max())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(submission))
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()

    validation = {
        "schema_version": 1,
        "label": args.label,
        "status": "validated_not_uploaded",
        "checkpoint": str(Path(args.init).resolve()),
        "checkpoint_sha256": trainer.sha256(args.init),
        "model_state_sha256": mt.tensor_state_sha256(payload["model"]),
        "clips_root": str(Path(args.clips_root).resolve()),
        "clips": len(submission),
        "submission_file": str(out_path.resolve()),
        "submission_sha256": digest,
        "submission_bytes": out_path.stat().st_size,
        "format": {
            "container": "json object keyed by clip directory name",
            "value": "[[x, y] x 6] float64, metres, current ego frame, 0.5 s steps to 3.0 s",
            "coordinates": "cumulative absolute positions as the model emits them",
        },
        "checks": {
            "all_shapes_6x2": True,
            "all_finite": True,
            "cumsum_not_reapplied": {
                "mean_endpoint_m": float(endpoints.mean()),
                "mean_endpoint_if_cumsum_were_applied_again_m": float(double_applied.mean()),
                "ratio": float(double_applied.mean() / max(endpoints.mean(), 1e-9)),
                "note": ("the model already returns cumulative absolute XY; applying the "
                         "example script's cumsum again would inflate the endpoint by this ratio"),
            },
            "first_point_is_half_second_step_not_full_path": {
                "mean_first_point_m": float(first.mean()),
                "mean_endpoint_m": float(endpoints.mean()),
            },
            "clip_state_isolation_max_abs_diff_m": isolation,
            "clip_state_isolation_pass": isolation == 0.0,
            "monotone_cumulative_distance_clips": int(sum(
                c["cumulative_distance_is_monotone"] for c in per_clip)),
        },
        "inputs": {
            "adapter": "models/motiondrive_v2_inputs.py prepare_clip_inputs",
            "reads": "calibration.parquet, ego_pose.parquet, ten JPEGs",
            "no_ground_truth_no_provided_status": True,
        },
        "per_clip": per_clip,
        "limitations": [
            "Validated on the clips under clips_root; running on the official test set is a separate step.",
            "No upload was performed and no submission quota was consumed.",
            "The exact official field naming must be confirmed against the current submission form.",
        ],
    }
    sidecar = out_path.with_suffix(".validation.json")
    sidecar.write_text(json.dumps(validation, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: validation[k] for k in
                      ("label", "clips", "submission_sha256", "checks")}, indent=1))


if __name__ == "__main__":
    main()
