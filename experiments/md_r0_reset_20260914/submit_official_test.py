#!/usr/bin/env python3
"""Run the candidate over the official test clips and write the submission JSON.

Each official clip is a tar of calibration.parquet, ego_pose.parquet,
command.parquet and the camera JPEGs.  The tars are read in memory and fed to
the same GT-free adapter used for deployment parity, so nothing is unpacked and
no training loader is involved.

command.parquet is present in the clips and is deliberately NOT read: this
model is trained with command off.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models import motiondrive_v2_inputs as adapter
from models.motiondrive_v2_inputs import CALIBRATION_COLUMNS, POSE_COLUMNS
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
import motiondrive_v2_training as mt
import train_motiondrive_v2 as trainer


def clip_from_tar(path: Path):
    """Read one official clip without unpacking it."""
    import pyarrow.parquet as pq
    with tarfile.open(path) as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        roots = {name.split("/", 1)[0] for name in members}
        if len(roots) != 1:
            raise ValueError(f"{path.name}: expected exactly one clip root, got {sorted(roots)}")
        token = roots.pop()
        if token != path.stem:
            raise ValueError(f"{path.name}: clip root {token} does not match the file name")

        def read(name):
            member = members.get(f"{token}/{name}")
            if member is None:
                raise FileNotFoundError(f"{path.name}: missing {name}")
            return archive.extractfile(member).read()

        calibration_bytes, pose_bytes = read("calibration.parquet"), read("ego_pose.parquet")
        calibration = pq.read_table(io.BytesIO(calibration_bytes),
                                    columns=list(CALIBRATION_COLUMNS)).to_pylist()
        poses = pq.read_table(io.BytesIO(pose_bytes), columns=list(POSE_COLUMNS)).to_pylist()
        requested = []

        def image_loader(camera, frame):
            requested.append(f"{camera}/frame_{frame}.jpg")
            return read(f"{camera}/frame_{frame}.jpg")

        prepared = adapter.prepare_clip_from_records(calibration, poses, image_loader)
    prepared.metadata.update(
        clip_id=token, clip_tar=str(path.resolve()),
        images_read=sorted(requested), members=len(members),
        source_sha256={"calibration.parquet": hashlib.sha256(calibration_bytes).hexdigest(),
                       "ego_pose.parquet": hashlib.sha256(pose_bytes).hexdigest()})
    return token, prepared


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--init", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    payload = torch.load(args.init, map_location="cpu", weights_only=False)
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    model = MotionDriveV2(config)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(f"cuda:{args.gpu}")
    model.to(device).eval()

    tars = sorted(Path(args.test_root).glob("*.tar"))
    if args.limit:
        tars = tars[:args.limit]
    if not tars:
        raise SystemExit(f"no clip tars under {args.test_root}")

    submission, per_clip, images_read = {}, [], set()
    started = time.monotonic()
    for index, tar in enumerate(tars, 1):
        token, prepared = clip_from_tar(tar)
        images_read.update(prepared.metadata["images_read"])
        inputs = {k: v.to(device) for k, v in prepared.inputs.items()}
        with torch.no_grad(), trainer.autocast(device, args.precision):
            plan = model(**inputs)["plan_abs"].float().cpu().numpy()[0]
        if plan.shape != (6, 2) or not np.isfinite(plan).all():
            raise SystemExit(f"{token}: bad prediction {plan.shape}")
        submission[token] = plan.astype(np.float64).tolist()
        distance = np.linalg.norm(plan, axis=-1)
        per_clip.append({"clip": token,
                         "endpoint_distance_m": float(distance[-1]),
                         "monotone": bool(np.all(np.diff(distance) > -1e-6))})
        if index % 100 == 0 or index == len(tars):
            print(json.dumps({"done": index, "of": len(tars),
                              "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)

    # clip isolation: the first clip re-run after all the others must be identical
    _, replay_prepared = clip_from_tar(tars[0])
    with torch.no_grad(), trainer.autocast(device, args.precision):
        replay = model(**{k: v.to(device) for k, v in replay_prepared.inputs.items()}
                       )["plan_abs"].float().cpu().numpy()[0]
    isolation = float(np.abs(replay - np.asarray(submission[tars[0].stem])).max())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(submission))
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()

    endpoints = np.asarray([c["endpoint_distance_m"] for c in per_clip])
    doubled = np.asarray([np.linalg.norm(np.cumsum(np.asarray(v), axis=0)[-1])
                          for v in submission.values()])
    validation = {
        "schema_version": 1, "label": args.label, "status": "built_not_uploaded",
        "checkpoint": str(Path(args.init).resolve()),
        "checkpoint_sha256": trainer.sha256(args.init),
        "model_state_sha256": mt.tensor_state_sha256(payload["model"]),
        "test_root": str(Path(args.test_root).resolve()),
        "clips_available": len(sorted(Path(args.test_root).glob("*.tar"))),
        "clips_written": len(submission),
        "submission_file": str(out_path.resolve()),
        "submission_sha256": digest, "submission_bytes": out_path.stat().st_size,
        "elapsed_seconds": time.monotonic() - started,
        "format": {"container": "json object keyed by clip token",
                   "value": "[[x, y] x 6] float64 metres, current ego frame, 0.5 s steps to 3.0 s",
                   "coordinates": "cumulative absolute positions as the model emits them"},
        "checks": {
            "all_shapes_6x2": True, "all_finite": True,
            "clip_token_matches_tar_name": True,
            "cumsum_not_reapplied": {
                "mean_endpoint_m": float(endpoints.mean()),
                "mean_endpoint_if_cumsum_were_applied_again_m": float(doubled.mean()),
                "ratio": float(doubled.mean() / max(endpoints.mean(), 1e-9)),
            },
            "clip_state_isolation_max_abs_diff_m": isolation,
            "clip_state_isolation_pass": isolation == 0.0,
            "monotone_cumulative_distance_clips": int(sum(c["monotone"] for c in per_clip)),
            "endpoint_distance_m": {"min": float(endpoints.min()), "p50": float(np.median(endpoints)),
                                    "max": float(endpoints.max())},
        },
        "inputs": {
            "adapter": "models/motiondrive_v2_inputs.py prepare_clip_from_records",
            "files_read_per_clip": sorted({name.split("/")[-2] + "/" + name.split("/")[-1]
                                           for name in images_read}),
            "parquets_read": ["calibration.parquet", "ego_pose.parquet"],
            "command_parquet_read": False,
            "provided_numeric_ego_status_used": False,
            "future_pose_used": "the +50 frame target point, as a scene-feature condition only",
        },
        "limitations": [
            "No upload was performed and no submission quota was consumed.",
            "Field naming must be confirmed against the current official submission form.",
            "No RTX4090 timing accompanies this file.",
        ],
    }
    out_path.with_suffix(".validation.json").write_text(
        json.dumps(validation, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: validation[k] for k in
                      ("label", "clips_written", "clips_available", "submission_sha256",
                       "submission_bytes", "elapsed_seconds", "checks")}, indent=1))


if __name__ == "__main__":
    main()
