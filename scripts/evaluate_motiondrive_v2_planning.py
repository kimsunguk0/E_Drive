#!/usr/bin/env python3
"""Tune-only direct-planner evaluation; never selects a checkpoint or a rule.

Uses the production dataset, training model-input whitelist and weighted_d3,
and audit checkpoint loader without configuration overrides. GT plans are six
cumulative positions in the current ego frame, NOT six displacement increments.
Final/historical validation is intentionally unavailable in this CLI.

Counterfactuals change images only. image_mismatch is an alias of image_shuffle:
a fixed scene derangement at the SAME frame replaces all six current cameras
and all four historical front images. Goal, calibration, poses and times stay
with the receiver. The protocol and donor map are published before any forward.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_motiondrive_v2 import construct_model, sha256
from export_motiondrive_v2_inference import stream_sha256, validate_complete_config
from motiondrive_v2_data import MotionDriveDataset
from motiondrive_v2_training import MODEL_INPUTS, TIME_WEIGHTS, model_inputs, to_device, weighted_d3
from train_motiondrive_v2 import autocast as training_autocast

CONDITIONS = ("normal", "image_shuffle", "repeat_current", "reverse_history")
BUCKET_NAMES = ("stop", "accel", "decel", "cruise", "unknown")
BUCKET_DEFINITION = {
    "version": 1, "source": "GT state_target only; never model-predicted state",
    "stop": "valid vx,vy and hypot(vx,vy) < 0.2 m/s",
    "accel": "not stop; valid ego-x acceleration ax >= 0.5 m/s^2",
    "decel": "not stop; valid ego-x acceleration ax <= -0.5 m/s^2",
    "cruise": "not stop; valid ego-x acceleration -0.5 < ax < 0.5 m/s^2",
    "unknown": "missing/nonfinite required GT state components",
    "priority": "stop first; mutually exclusive and exhaustive",
    "axes": "current ego x=longitudinal, y=lateral; not a trajectory-tangent frame",
}


def normalize_conditions(conditions):
    values = ["image_shuffle" if c == "image_mismatch" else c for c in conditions]
    if not values or len(set(values)) != len(values) or any(c not in CONDITIONS for c in values):
        raise ValueError("Conditions must be distinct supported values (mismatch aliases shuffle)")
    return values


def require_tune(split):
    if split != "tune":
        raise ValueError("Only tune is authorized; final/historical val requires a separately reviewed change")


def rows_sha256(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def scene_derangement(scenes, seed):
    """Sattolo permutation of sorted full-split scenes, independent of labels."""
    scenes = sorted(set(map(str, scenes)))
    if len(scenes) < 2:
        raise ValueError("Image shuffle requires at least two split scenes")
    donors = list(scenes)
    rng = np.random.default_rng(seed)
    for i in range(len(donors) - 1, 0, -1):
        j = int(rng.integers(0, i))
        donors[i], donors[j] = donors[j], donors[i]
    return dict(zip(scenes, donors))


def donor_indices(receiver, donors, scene_map):
    """Resolve every same-frame donor BEFORE inference; never silently drop rows."""
    lookup = {}
    for i, row in enumerate(donors.rows):
        key = (str(donors.scene_names[row]), int(donors.arr["frame"][row]))
        if key in lookup:
            raise ValueError(f"Duplicate donor scene/frame: {key}")
        lookup[key] = i
    result = []
    for row in receiver.rows:
        scene = str(receiver.scene_names[row])
        target_scene = scene_map.get(scene)
        if target_scene is None or target_scene == scene:
            raise ValueError(f"Missing or self donor for scene {scene}")
        key = (target_scene, int(receiver.arr["frame"][row]))
        if key not in lookup:
            raise ValueError(f"No donor at the same frame: {key}")
        result.append(lookup[key])
    return result


class ImageCounterfactualDataset(Dataset):
    """Wrap the existing dataset; only raw image tensors are replaced/reordered."""
    def __init__(self, receiver, condition="normal", donors=None, indices=None):
        self.receiver = receiver
        self.condition = normalize_conditions([condition])[0]
        self.donors, self.indices = donors, indices
        if self.condition == "image_shuffle" and (donors is None or indices is None or len(indices) != len(receiver)):
            raise ValueError("Image shuffle requires a fully resolved same-frame donor list")

    def __len__(self):
        return len(self.receiver)

    def __getitem__(self, index):
        original = self.receiver[index]
        sample = dict(original)
        sample.update(donor_scenario="", donor_session="", donor_frame=-1, donor_row=-1)
        if self.condition == "image_shuffle":
            donor = self.donors[self.indices[index]]
            if donor["scenario"] == original["scenario"] or donor["frame"] != original["frame"]:
                raise ValueError("Donor identity changed after protocol validation")
            sample["images"], sample["history_images"] = donor["images"], donor["history_images"]
            sample.update(donor_scenario=donor["scenario"], donor_session=donor["session_id"],
                          donor_frame=donor["frame"], donor_row=donor["row"])
        elif self.condition == "repeat_current":
            front = F.interpolate(original["images"][:1], size=original["history_images"].shape[-2:],
                                  mode="bilinear", align_corners=False, antialias=True)
            sample["history_images"] = front.expand_as(original["history_images"]).clone()
        elif self.condition == "reverse_history":
            sample["history_images"] = original["history_images"].flip(0)
        return sample


def per_frame_errors(pred, gt):
    if pred.ndim != 3 or pred.shape != gt.shape or pred.shape[1:] != (6, 2):
        raise ValueError("Expected matching cumulative absolute plans [B,6,2]")
    if not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
        raise FloatingPointError("Nonfinite planning prediction or GT")
    error = pred.float() - gt.float()
    distance = torch.linalg.vector_norm(error, dim=-1)
    weights = distance.new_tensor(TIME_WEIGHTS)
    return {"d3": weighted_d3(pred, gt), "point_l2": distance,
            "cumulative_ade_1_2_3s": torch.stack([distance[:, :n].mean(1) for n in (2, 4, 6)], 1),
            "longitudinal_error": error[..., 0], "lateral_error": error[..., 1],
            "weighted_abs_longitudinal_error": (error[..., 0].abs() * weights).sum(1),
            "weighted_abs_lateral_error": (error[..., 1].abs() * weights).sum(1)}


def state_bucket(state, valid):
    state, valid = np.asarray(state, float), np.asarray(valid, bool)
    if state.shape != (6,) or valid.shape != (6,):
        raise ValueError("GT state and mask must have six components")
    if not valid[:2].all() or not np.isfinite(state[:2]).all():
        return "unknown"
    if np.linalg.norm(state[:2]) < .2:
        return "stop"
    if not valid[2] or not np.isfinite(state[2]):
        return "unknown"
    return "accel" if state[2] >= .5 else "decel" if state[2] <= -.5 else "cruise"


def _aggregate(records):
    fields = ("point_l2", "cumulative_ade_1_2_3s", "weighted_abs_longitudinal_error",
              "weighted_abs_lateral_error")
    result = {"n": len(records), "official_d3": None}
    if not records:
        return {**result, **{k: None for k in fields}}
    result["official_d3"] = float(np.mean([r["d3"] for r in records]))
    result.update({key: np.asarray([r[key] for r in records], float).mean(0).tolist() for key in fields})
    return result


def summarize_records(records):
    by_session = {}
    for record in records:
        by_session.setdefault(record["session"], []).append(record)
    sessions = {name: _aggregate(rows) for name, rows in sorted(by_session.items())}
    return {**_aggregate(records), "n_scenes": len({r["scenario"] for r in records}),
            "n_sessions": len(sessions),
            "session_mean_d3": (float(np.mean([r["official_d3"] for r in sessions.values()])) if sessions else None),
            "session_d3": {name: row["official_d3"] for name, row in sessions.items()},
            "sessions": sessions}


@torch.inference_mode()
def evaluate_planning(model, loader, device, precision="bf16"):
    """Exact trainer metric/AMP path; no GT field can enter model(**inputs)."""
    model.eval()
    records = []
    for raw in loader:
        batch = to_device(raw, device)
        if batch["plan_valid"].shape != batch["gt_plan"].shape[:-1] or not batch["plan_valid"].bool().all():
            raise ValueError("Invalid six-point GT; refusing to change the official metric")
        with training_autocast(device, precision):
            output = model(**model_inputs(batch))
        pred, gt = output["plan_abs"], batch["gt_plan"]
        if pred.dtype != torch.float32:
            raise ValueError("plan_abs must already be FP32; evaluator will not repair output precision")
        errors = {k: v.cpu().tolist() for k, v in per_frame_errors(pred, gt).items()}
        pred, gt = pred.cpu().tolist(), gt.cpu().tolist()
        states, valid = raw["state_target"].cpu().tolist(), raw["state_valid"].cpu().tolist()
        for i in range(len(pred)):
            record = {"scenario": raw["scenario"][i], "session": raw["session_id"][i],
                      "frame": int(raw["frame"][i]), "row": int(raw["row"][i]),
                      "pred_abs_xy": pred[i], "gt_abs_xy": gt[i],
                      **{k: v[i] for k, v in errors.items()},
                      "gt_state": [float(v) if ok and np.isfinite(v) else None for v, ok in zip(states[i], valid[i])],
                      "gt_state_valid": [bool(v) for v in valid[i]],
                      "bucket": state_bucket(states[i], valid[i])}
            if raw.get("donor_scenario", [""] * len(pred))[i]:
                record["donor"] = {"scenario": raw["donor_scenario"][i],
                                   "session": raw["donor_session"][i],
                                   "frame": int(raw["donor_frame"][i]), "row": int(raw["donor_row"][i])}
            records.append(record)
    if not records:
        raise ValueError("Empty evaluation split")
    identities = [(r["scenario"], r["frame"]) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate scene/frame evaluation rows")
    return {"summary": summarize_records(records),
            "buckets": {name: summarize_records([r for r in records if r["bucket"] == name]) for name in BUCKET_NAMES},
            "records": records}


def ensure_new_output(path):
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite {path}")
    if not path.parent.is_dir():
        raise ValueError(f"Output parent must already exist: {path.parent}")


def write_new_json(path, value):
    ensure_new_output(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_identity(stat):
    return {"device": stat.st_dev, "inode": stat.st_ino,
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def checkpoint_manifest(path, split_sha):
    # Trusted local trainer checkpoints contain NumPy/Python RNG pickles.
    # Hash/read the SAME open file: best.pth may be atomically replaced by a
    # training process. Reject replacement before the shared loader is used.
    with open(path, "rb") as stream:
        identity = file_identity(os.fstat(stream.fileno()))
        checkpoint_sha = stream_sha256(stream)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=False)
        if file_identity(os.fstat(stream.fileno())) != identity:
            raise ValueError("Checkpoint changed while reading its manifest")
    manifest = saved.get("manifest", {}) if isinstance(saved, dict) else {}
    validate_complete_config(manifest.get("model_config"))
    if manifest.get("split_sha256") != split_sha:
        raise ValueError("Checkpoint and evaluation split lineage do not match")
    return {"checkpoint_sha256": checkpoint_sha, "checkpoint_file_identity": identity,
            "checkpoint_step": saved.get("step"),
            "checkpoint_manifest": json.loads(json.dumps(manifest)),
            "source_checkpoint": saved.get("source_checkpoint")}


def source_manifest():
    paths = [Path(__file__), ROOT / "scripts/motiondrive_v2_data.py",
             ROOT / "scripts/motiondrive_v2_training.py", ROOT / "scripts/train_motiondrive_v2.py",
             ROOT / "scripts/audit_motiondrive_v2.py", ROOT / "scripts/export_motiondrive_v2_inference.py",
             ROOT / "scripts/sparse_scoredrive.py", ROOT / "scripts/build_grouped_split_v2.py",
             *sorted((ROOT / "models/motiondrive_v2").glob("*.py"))]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_sha": commit, "tracked_changes": dirty,
            "file_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths}}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True, help="Trusted user-owned local checkpoint/bundle")
    parser.add_argument("--data-root", default=str(ROOT))
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--split", choices=("tune",), default="tune")
    parser.add_argument("--out", required=True, help="New JSON report; sibling .protocol.json is preregistered")
    parser.add_argument("--conditions", nargs="+", choices=(*CONDITIONS, "image_mismatch"), default=["normal"])
    parser.add_argument("--donor-seed", type=int, default=20260907)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--scenes", nargs="+", help="Explicit diagnostic receiver subset; donor map still uses full tune split")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"), default="cpu")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args(argv)
    require_tune(args.split)
    args.conditions = normalize_conditions(args.conditions)
    if args.batch < 1 or args.workers < 0 or args.frame_stride < 1 or args.max_samples < 0:
        raise ValueError("Invalid batch/workers/stride/sample count")
    return args


def main(argv=None):
    args = arguments(argv)
    # Do not resolve the final component: an existing dangling symlink is also
    # an existing output and must be rejected, not followed to a new target.
    output = Path(args.out).expanduser().absolute()
    protocol_path = output.with_suffix(".protocol.json")
    ensure_new_output(output)
    ensure_new_output(protocol_path)
    split_sha = sha256(args.split_manifest)
    header = checkpoint_manifest(args.checkpoint, split_sha)
    dataset_args = dict(data_root=args.data_root, split_manifest=args.split_manifest, split=args.split,
                        supervision_root=args.supervision_root, min_frame=30,
                        frame_stride=args.frame_stride, augment=False, seed=args.seed)
    receiver = MotionDriveDataset(**dataset_args, max_samples=args.max_samples, scenes=args.scenes)
    donors, indices, mapping = None, None, None
    if "image_shuffle" in args.conditions:
        donors = MotionDriveDataset(**dataset_args, max_samples=0)
        mapping = scene_derangement(receiver.manifest["splits"][args.split], args.donor_seed)
        indices = donor_indices(receiver, donors, mapping)
    used_scenes = set(map(str, receiver.scene_names[receiver.rows]))
    if donors is not None:
        used_scenes.update(str(donors.scene_names[donors.rows[i]]) for i in indices)
    supervision = Path(args.supervision_root)
    data_provenance = {"split_sha256": split_sha, "ego_cache_sha256": receiver.cache_sha,
                       "receiver_rows_sha256": rows_sha256(receiver.rows), "receiver_count": len(receiver),
                       "donor_rows_sha256": rows_sha256([donors.rows[i] for i in indices]) if indices is not None else None,
                       "image_root": str(receiver.image_root.resolve()),
                       "supervision_sha256": {name: sha256(supervision / name) for name in
                           ["supervision_manifest.json", "calibration.npz",
                            *[f"{s}{suffix}" for s in sorted(used_scenes) for suffix in (".npz", ".json")]]}}
    protocol = {"status": "preregistered_before_any_forward", "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                **header, "data": data_provenance, "source": source_manifest(),
                "donor_scene_map": mapping, "donor_policy": "full tune split Sattolo scene permutation; same frame; image tensors only",
                "bucket_definition": BUCKET_DEFINITION, "model_input_whitelist": list(MODEL_INPUTS),
                "nominal_waypoint_seconds": [.5, 1., 1.5, 2., 2.5, 3.],
                "target_time_semantics": "Official cached frame-offset targets preserved; actual raw timestamps can differ from nominal 0.5-second spacing",
                "metric": "mean cumulative ADE@1/2/3s; [11,11,5,5,2,2]/36; equal frame weights",
                "coordinate_format": "current ego-frame cumulative XY positions, metres; no cumsum or postprocessing",
                "caveats": ["Image counterfactuals break image/geometry consistency and are diagnostics, not automatic compliance proof.",
                            "Repeatedly used tune is not an untouched holdout; this evaluator performs no checkpoint/condition selection."],
                "selection_performed": False, "final_val_accessed": False}
    write_new_json(protocol_path, protocol)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
    model = construct_model(SimpleNamespace(checkpoint=args.checkpoint, config_json=None,
                                           goal_on=None, state_on=None, device=args.device))
    if model.audit_load_metadata["explicit_overrides"]:
        raise ValueError("Evaluation must restore checkpoint configuration without overrides")
    if model.audit_load_metadata["checkpoint_sha256"] != header["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after protocol preregistration")
    if file_identity(os.stat(args.checkpoint)) != header["checkpoint_file_identity"]:
        raise ValueError("Checkpoint was replaced after protocol preregistration; use an immutable snapshot")
    results = {}
    for condition in args.conditions:
        wrapped = ImageCounterfactualDataset(receiver, condition, donors, indices)
        loader = DataLoader(wrapped, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda")
        results[condition] = evaluate_planning(model, loader, device, args.precision)
        observed_rows = [r["row"] for r in results[condition]["records"]]
        if rows_sha256(observed_rows) != data_provenance["receiver_rows_sha256"]:
            raise ValueError("Observed evaluation rows differ from preregistration")
    report = {"status": "completed", "protocol_path": str(protocol_path), "protocol_sha256": sha256(protocol_path),
              "protocol": protocol, "model_load": model.audit_load_metadata,
              "precision": args.precision if device.type == "cuda" else "fp32",
              "precision_requested": args.precision, "torch": str(torch.__version__),
              "cuda_runtime": torch.version.cuda, "conditions": results,
              "selection_performed": False, "final_val_accessed": False}
    write_new_json(output, report)
    print(json.dumps({"report": str(output), "conditions": {k: v["summary"] for k, v in results.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
