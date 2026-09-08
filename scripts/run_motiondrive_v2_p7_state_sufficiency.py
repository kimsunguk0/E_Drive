#!/usr/bin/env python3
"""Matched P7-C state/history+goal sufficiency diagnostic.

This is an offline, non-submittable diagnostic.  It has three deliberately
separate stages: pack pose-derived train/tune labels, cache the image model's
own predictions with one unmodified full forward, then fit the same tiny MLP
to GT or predicted inputs.  The final validation split is not exposed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.motiondrive_v2 import MotionDriveV2
from motiondrive_v2_data import MotionDriveDataset
from motiondrive_v2_training import model_inputs, tensor_state_sha256, to_device, weighted_d3
from train_motiondrive_v2 import autocast as training_autocast


SCHEMA_VERSION = 1
SPLIT_SHA256 = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
SUPERVISION_SHA256 = "ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93"
CALIBRATION_SHA256 = "8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961"
EGO_SHA256 = "d35a69bb229b2d4736aff7dddefd3522ddbd346e112283e2ce58ce3986a023bd"
P7_SOURCE_MANIFEST_SHA256 = "880c3cc36ababe31c58376ab815a239e337f130fc9b373a041fea6ad4a8d1ea4"
RUNTIME_SOURCE_MANIFEST_SHA256 = "6682ea02544fc8a39e5d78bdf90bf9735682b51f3f5c7f88db93d85729d77f75"
RUNTIME_FILES = {
    "models/motiondrive_v2_temporal_contract.py",
    "models/motiondrive_v2/__init__.py", "models/motiondrive_v2/config.py",
    "models/motiondrive_v2/cross_cell_goal_residual.py", "models/motiondrive_v2/model.py",
    "models/motiondrive_v2/motion_encoder.py", "models/motiondrive_v2/planner.py",
    "models/motiondrive_v2/scene_encoder.py", "scripts/motiondrive_v2_data.py",
    "scripts/motiondrive_v2_training.py", "scripts/sparse_scoredrive.py",
    "scripts/train_motiondrive_v2.py",
}
VALIDATION_FILES = {
    "scripts/analyze_motiondrive_v2_p7_goal_routing_results.py":
        "4ba2ba0050bb74f6de462daf9ef9b5b31a8d3bb70779532465a2ce78fea0213b",
    "scripts/analyze_motiondrive_v2_p6_stop_balance_results.py":
        "e3ef5f7dac642d20aedafa71bcd4979aee4faaa077d2c234f910b736691da753",
    "scripts/analyze_motiondrive_v2_query_pair.py":
        "03e714e845f7c66a152b5e28449444e61b96d82541550747003fa58e71065651",
    "scripts/run_motiondrive_v2_p7_goal_routing.py":
        "75b6908715a9181309ec586a18f3026518b419c922d1ec67b2a9f729d5a4c8cc",
    "scripts/build_grouped_split_v2.py":
        "5d3019cf730fe56a27ab7f204ee3f15bc8a4ffaef5df9fbf0635f086c84a49c1",
}
ROWS = {
    "train": (54810, 1, "75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854"),
    "tune": (1998, 5, "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88"),
}
P7_CONTROL = {
    0: {
        "checkpoint_sha256": "6145255ab2b1efd3deb169b873896e9b1b623ca49f11e5a46a71b138e3b0355e",
        "manifest_sha256": "cf287d05ce6b2f4476c07ac5c2253cd9ecb3227bcc01404adbbba8463bb66de0",
        "final_eval_sha256": "67279fe06120ef752d44c915dbc01d022244b94d60ba27577ed62d1bace1afd0",
        "official_d3": 0.3305861224029754,
    },
    1: {
        "checkpoint_sha256": "8ffb429b17820b63477987f5c7de151dfa9e0e2b12378e247aed0b0e41de2563",
        "manifest_sha256": "11854d7a9f828b8b4b6710679fd420f5c8d243d1322231d723137419aa48e80f",
        "final_eval_sha256": "703f46945496e2de9d5a9386e8596aa0fb01d6f47e627f9c8db33b0b58b6f795",
        "official_d3": 0.3246189025291302,
    },
}
NOMINAL_SECONDS = (0.1, 0.2, 0.5, 1.0)
STATE_NAMES = ("vx", "vy", "ax", "ay", "yaw_rate", "stop_probability")
HISTORY_NAMES = tuple(
    f"history_{seconds:g}s_{component}"
    for seconds in NOMINAL_SECONDS for component in ("dx", "dy", "sin_yaw", "cos_yaw")
)
FEATURE_NAMES = STATE_NAMES + HISTORY_NAMES + ("goal_x_5s", "goal_y_5s")
FEATURE_SCALE = np.asarray(
    (10., 5., 3., 3., .5, 1.) + (10., 5., 1., 1.) * 4 + (80., 64.), dtype=np.float64)
N_EPOCHS = 60
BATCH_SIZE = 1024
N_STEPS = N_EPOCHS * math.ceil(ROWS["train"][0] / BATCH_SIZE)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_sha256(rows) -> str:
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def runtime_receipt():
    return {"python": platform.python_version(), "executable": sys.executable,
            "torch": str(torch.__version__), "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_initialized": torch.cuda.is_initialized()}


def tensor_tree_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return tensor_state_sha256(state)


def ensure_fresh(path: str | Path) -> Path:
    path = Path(path).resolve()
    require(not path.exists() and not path.is_symlink(), f"Refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path: str | Path, value) -> None:
    path = ensure_fresh(path)
    data = (json.dumps(to_native(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _atomic_bytes(path, data)


def atomic_torch(path: str | Path, value) -> None:
    path = ensure_fresh(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_bytes(path: Path, data: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def to_native(value):
    if isinstance(value, Mapping):
        return {str(key): to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_native(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, torch.Tensor):
        return to_native(value.detach().cpu().tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def read_json(path, expected_sha=None):
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink(), f"Ordinary JSON required: {path}")
    if expected_sha is not None:
        require(sha256(path) == expected_sha, f"JSON SHA mismatch: {path}")
    return json.loads(path.read_text(), parse_constant=lambda token: (_ for _ in ()).throw(
        ValueError(f"Nonfinite JSON token: {token}")))


def validate_common_inputs(args, split):
    split_path = Path(args.split_manifest).resolve()
    supervision = Path(args.supervision_root).resolve()
    require(sha256(split_path) == SPLIT_SHA256, "Grouped split SHA mismatch")
    require(sha256(supervision / "supervision_manifest.json") == SUPERVISION_SHA256,
            "C1 supervision manifest SHA mismatch")
    require(sha256(supervision / "calibration.npz") == CALIBRATION_SHA256,
            "C1 calibration SHA mismatch")
    ego_path = Path(args.data_root).resolve() / "data/etri/ego_cache.npz"
    require(sha256(ego_path) == EGO_SHA256, "Ego cache SHA mismatch")
    expected_n, stride, expected_rows_sha = ROWS[split]
    dataset = MotionDriveDataset(
        data_root=args.data_root, split_manifest=args.split_manifest, split=split,
        supervision_root=args.supervision_root, min_frame=30, frame_stride=stride,
        max_samples=0, augment=False, seed=0)
    require(len(dataset) == expected_n and rows_sha256(dataset.rows) == expected_rows_sha,
            f"Canonical {split} rows mismatch")
    return dataset


def collect_ground_truth(dataset):
    rows, state, state_valid, history, history_valid, goals, plans, plan_valid = ([] for _ in range(8))
    scenarios, sessions, frames = [], [], []
    for row_value in dataset.rows:
        row = int(row_value)
        scene, frame = str(dataset.scene_names[row]), int(dataset.arr["frame"][row])
        source = dataset._supervision(scene)
        require(frame in source["frame_lookup"], f"Missing supervision row: {scene}/{frame}")
        index = source["frame_lookup"][frame]
        require(int(source["row"][index]) == row, "Supervision row identity mismatch")
        rows.append(row)
        state.append(source["state_target"][index])
        state_valid.append(source["state_valid"][index])
        history.append(source["history_target"][index])
        history_valid.append(source["history_valid"][index])
        goals.append(dataset.arr["goal"][row])
        plans.append(dataset.arr["fut"][row])
        plan_valid.append(np.ones(6, dtype=np.bool_))
        scenarios.append(scene)
        sessions.append(dataset.manifest["scene_to_session"][scene])
        frames.append(frame)
    payload = {
        "rows": torch.as_tensor(np.asarray(rows, dtype=np.int64)),
        "state_target": torch.as_tensor(np.asarray(state, dtype=np.float32)),
        "state_valid": torch.as_tensor(np.asarray(state_valid, dtype=np.bool_)),
        "history_target": torch.as_tensor(np.asarray(history, dtype=np.float32)),
        "history_valid": torch.as_tensor(np.asarray(history_valid, dtype=np.bool_)),
        "goal_xy": torch.as_tensor(np.asarray(goals, dtype=np.float32)),
        "gt_plan": torch.as_tensor(np.asarray(plans, dtype=np.float32)),
        "plan_valid": torch.as_tensor(np.asarray(plan_valid, dtype=np.bool_)),
        "scenario": scenarios, "session": sessions,
        "frame": torch.as_tensor(np.asarray(frames, dtype=np.int64)),
    }
    validate_label_payload(payload, len(dataset), rows_sha256(dataset.rows))
    return payload


def validate_label_payload(payload, expected_n, expected_rows_sha):
    required = {"rows", "state_target", "state_valid", "history_target", "history_valid",
                "goal_xy", "gt_plan", "plan_valid", "scenario", "session", "frame"}
    require(set(payload) == required, "GT payload keys mismatch")
    shapes = {"rows": (expected_n,), "state_target": (expected_n, 6),
              "state_valid": (expected_n, 6), "history_target": (expected_n, 4, 4),
              "history_valid": (expected_n, 4, 4), "goal_xy": (expected_n, 2),
              "gt_plan": (expected_n, 6, 2), "plan_valid": (expected_n, 6),
              "frame": (expected_n,)}
    require(all(isinstance(payload[key], torch.Tensor) and tuple(payload[key].shape) == shape
                for key, shape in shapes.items()), "GT payload shapes mismatch")
    require(payload["rows"].dtype == torch.int64 and payload["frame"].dtype == torch.int64
            and payload["state_valid"].dtype == torch.bool
            and payload["history_valid"].dtype == torch.bool
            and payload["plan_valid"].dtype == torch.bool,
            "GT identity/mask dtype mismatch")
    floats = ("state_target", "history_target", "goal_xy", "gt_plan")
    require(all(payload[key].dtype == torch.float32 and torch.isfinite(payload[key]).all()
                for key in floats), "GT float dtype/finite mismatch")
    require(rows_sha256(payload["rows"].numpy()) == expected_rows_sha,
            "GT payload row order mismatch")
    require(len(payload["scenario"]) == len(payload["session"]) == expected_n,
            "GT string identity count mismatch")
    require(bool(payload["state_valid"].all()) and bool(payload["history_valid"].all())
            and bool(payload["plan_valid"].all()), "Probe requires all 22 fields and plan points valid")
    stop = payload["state_target"][:, 5]
    require(bool(((stop == 0) | (stop == 1)).all()), "GT stop must be exact binary")
    return True


def pack_labels(args):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "Label packing is CPU-only and requires explicitly empty CUDA visibility")
    dataset = validate_common_inputs(args, args.split)
    payload = collect_ground_truth(dataset)
    output = Path(args.out).resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    atomic_torch(output, payload)
    metadata = {
        "schema_version": SCHEMA_VERSION, "status": "completed", "kind": "ground_truth",
        "split": args.split, "rows": len(dataset), "rows_sha256": rows_sha256(dataset.rows),
        "artifact_sha256": sha256(output), "split_manifest_sha256": SPLIT_SHA256,
        "supervision_manifest_sha256": SUPERVISION_SHA256,
        "calibration_sha256": CALIBRATION_SHA256, "ego_cache_sha256": EGO_SHA256,
        "feature_semantics": list(FEATURE_NAMES), "image_reads": 0,
        "script_sha256": sha256(__file__),
        "runtime": runtime_receipt(),
        "allowed_splits": ["train", "tune"], "final_validation_accessed": False,
    }
    atomic_json(manifest_path, metadata)
    print(json.dumps({"status": "completed", "artifact": str(output),
                      "sha256": metadata["artifact_sha256"], "manifest": str(manifest_path)},
                     sort_keys=True))


def validate_p7_inputs(args):
    expected = P7_CONTROL[args.base_seed]
    require(sha256(args.checkpoint) == expected["checkpoint_sha256"], "P7-C LAST SHA mismatch")
    require(sha256(args.run_manifest) == expected["manifest_sha256"], "P7-C sidecar SHA mismatch")
    require(sha256(args.p7_source_manifest) == P7_SOURCE_MANIFEST_SHA256,
            "P7 training source-manifest SHA mismatch")
    require(sha256(args.runtime_source_manifest) == RUNTIME_SOURCE_MANIFEST_SHA256,
            "Reviewed runtime source-manifest SHA mismatch")
    require({name: sha256(ROOT / name) for name in VALIDATION_FILES} == VALIDATION_FILES,
            "P7 manifest validation helper source differs from reviewed bytes")
    # Validate both the historical P7 training record and current default-C runtime closure.
    from analyze_motiondrive_v2_p7_goal_routing_results import (
        validate_p7_source_manifest, validate_terminal_manifest)
    historical = read_json(args.p7_source_manifest, P7_SOURCE_MANIFEST_SHA256)
    pinned = validate_p7_source_manifest(historical, P7_SOURCE_MANIFEST_SHA256)
    manifest = read_json(args.run_manifest, expected["manifest_sha256"])
    validate_terminal_manifest(manifest, base_seed=args.base_seed, arm="control",
                               pinned_source=pinned, expected_official_d3=expected["official_d3"])
    runtime = read_json(args.runtime_source_manifest, RUNTIME_SOURCE_MANIFEST_SHA256)
    require(set(runtime) == {"schema_version", "git_sha", "file_sha256"}
            and runtime.get("schema_version") == 1
            and isinstance(runtime.get("file_sha256"), dict)
            and RUNTIME_FILES <= set(runtime["file_sha256"]),
            "Reviewed runtime source-manifest schema/closure mismatch")
    actual = {name: sha256(ROOT / name) for name in RUNTIME_FILES}
    require(actual == {name: runtime["file_sha256"][name] for name in RUNTIME_FILES},
            "P7-C extraction runtime source differs from reviewed closure")
    return manifest, runtime


def load_p7_control_model(checkpoint_path, run_manifest, device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict) and set(("model", "step", "manifest")) <= set(payload)
            and payload["step"] == 6000 and isinstance(payload["model"], dict),
            "P7-C checkpoint payload is not terminal LAST6000")
    embedded_config = json.dumps(payload["manifest"].get("model_config"), sort_keys=True,
                                 separators=(",", ":"), allow_nan=False)
    external_config = json.dumps(run_manifest.get("model_config"), sort_keys=True,
                                 separators=(",", ":"), allow_nan=False)
    require(embedded_config == external_config, "P7-C embedded/external model config mismatch")
    with torch.random.fork_rng(devices=[]):
        model = MotionDriveV2(run_manifest["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    require(not any(module.training for module in model.modules()), "P7-C model is not fully eval")
    return model


def validate_tune_reference(path, base_seed):
    expected = P7_CONTROL[base_seed]
    require(path is not None and sha256(path) == expected["final_eval_sha256"],
            "P7-C tune report SHA mismatch")
    payload = read_json(path, expected["final_eval_sha256"])
    require(set(payload) == {"report", "records"} and len(payload["records"]) == ROWS["tune"][0],
            "P7-C tune report schema/count mismatch")
    return payload["records"]


def validate_prediction_payload(payload, expected_n, expected_rows_sha, base_seed, split):
    require(set(payload) == {"rows", "pred_state", "pred_history", "pred_plan"},
            "Prediction payload keys mismatch")
    shapes = {"rows": (expected_n,), "pred_state": (expected_n, 6),
              "pred_history": (expected_n, 4, 4), "pred_plan": (expected_n, 6, 2)}
    require(all(isinstance(payload[key], torch.Tensor) and tuple(payload[key].shape) == shape
                for key, shape in shapes.items()), "Prediction payload shape mismatch")
    require(payload["rows"].dtype == torch.int64
            and all(payload[key].dtype == torch.float32 and torch.isfinite(payload[key]).all()
                    for key in ("pred_state", "pred_history", "pred_plan")),
            "Prediction dtype/finite mismatch")
    require(rows_sha256(payload["rows"].numpy()) == expected_rows_sha,
            "Prediction row order mismatch")
    require(base_seed in (0, 1) and split in ROWS, "Prediction base/split mismatch")
    return True


def extract_predictions(args):
    require(torch.cuda.is_available() and args.device == "cuda:0",
            "P7-C extraction requires one isolated CUDA device")
    manifest, runtime_source = validate_p7_inputs(args)
    dataset = validate_common_inputs(args, args.split)
    expected_n, stride, expected_rows_sha = ROWS[args.split]
    reference = validate_tune_reference(args.tune_report, args.base_seed) if args.split == "tune" else None
    require((args.tune_report is None) == (args.split == "train"),
            "Only tune extraction accepts the immutable terminal report")
    require(args.batch == (8 if args.split == "train" else 4), "Fixed extraction batch mismatch")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", None)
            and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
            and torch.cuda.device_count() == 1 and args.workers == 4,
            "P7-C extraction requires one isolated visible GPU and workers=4")
    random.seed(args.base_seed); np.random.seed(args.base_seed); torch.manual_seed(args.base_seed)
    torch.cuda.manual_seed_all(args.base_seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_p7_control_model(args.checkpoint, manifest, device)
    before_state = tensor_state_sha256(model.state_dict())
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=True, drop_last=False)
    rows, states, histories, plans = [], [], [], []
    for batch_index, raw in enumerate(loader):
        batch = to_device(raw, device)
        with torch.inference_mode(), training_autocast(device, "bf16"):
            output = model(**model_inputs(batch, time_input="nominal",
                                          nominal_history_seconds=NOMINAL_SECONDS))
        state = output["state_hat"].float().cpu()
        history = output["history_hat"].float().cpu()
        plan = output["plan_abs"].float().cpu()
        d3 = weighted_d3(output["plan_abs"], batch["gt_plan"]).cpu()
        require(state.shape[1:] == (6,) and history.shape[1:] == (4, 4)
                and plan.shape[1:] == (6, 2), "P7-C output shape mismatch")
        require(torch.isfinite(state).all() and torch.isfinite(history).all()
                and torch.isfinite(plan).all(), "P7-C output nonfinite")
        offset = len(rows)
        batch_rows = [int(value) for value in raw["row"]]
        if reference is not None:
            for index, row in enumerate(batch_rows):
                item = reference[offset + index]
                require(item.get("row") == row and item.get("scenario") == raw["scenario"][index]
                        and item.get("session") == raw["session_id"][index]
                        and item.get("frame") == int(raw["frame"][index]),
                        "Tune report identity mismatch")
                require(torch.equal(state[index], torch.as_tensor(item["pred_state"], dtype=torch.float32))
                        and torch.equal(plan[index], torch.as_tensor(item["pred_abs_xy"], dtype=torch.float32))
                        and torch.equal(batch["gt_plan"][index].float().cpu(),
                                        torch.as_tensor(item["gt_abs_xy"], dtype=torch.float32))
                        and torch.equal(batch["state_target"][index].float().cpu(),
                                        torch.as_tensor(item["gt_state"], dtype=torch.float32))
                        and torch.equal(batch["state_valid"][index].bool().cpu(),
                                        torch.as_tensor(item["gt_state_valid"], dtype=torch.bool)),
                        "Tune full-forward state/plan/GT state differs from terminal P7-C report")
                require(float(d3[index]) == float(item["d3"]),
                        "Tune full-forward official D3 differs from terminal P7-C report")
        if batch_index == 0:
            with torch.inference_mode(), training_autocast(device, "bf16"):
                repeated = model(**model_inputs(batch, time_input="nominal",
                                                nominal_history_seconds=NOMINAL_SECONDS))
            require(all(torch.equal(output[key], repeated[key])
                        for key in ("state_hat", "history_hat", "plan_abs")),
                    "Repeated first full forward differs")
        rows.extend(batch_rows)
        states.append(state); histories.append(history); plans.append(plan)
    payload = {"rows": torch.as_tensor(rows, dtype=torch.int64),
               "pred_state": torch.cat(states), "pred_history": torch.cat(histories),
               "pred_plan": torch.cat(plans)}
    validate_prediction_payload(payload, expected_n, expected_rows_sha, args.base_seed, args.split)
    after_state = tensor_state_sha256(model.state_dict())
    require(before_state == after_state, "P7-C model state changed during inference")
    output = Path(args.out).resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    atomic_torch(output, payload)
    metadata = {
        "schema_version": SCHEMA_VERSION, "status": "completed", "kind": "prediction",
        "diagnostic": "P7-C state/history sufficiency; non-submittable",
        "base_seed": args.base_seed, "split": args.split, "rows": expected_n,
        "rows_sha256": expected_rows_sha, "artifact_sha256": sha256(output),
        "checkpoint_sha256": P7_CONTROL[args.base_seed]["checkpoint_sha256"],
        "run_manifest_sha256": P7_CONTROL[args.base_seed]["manifest_sha256"],
        "p7_training_source_manifest_sha256": P7_SOURCE_MANIFEST_SHA256,
        "runtime_source_manifest_sha256": RUNTIME_SOURCE_MANIFEST_SHA256,
        "runtime_source": runtime_source, "script_sha256": sha256(__file__),
        "validation_source_sha256": VALIDATION_FILES,
        "model_state_sha256_before": before_state,
        "model_state_sha256_after": after_state, "mode": "eval",
        "context": "torch.inference_mode", "precision": "bf16_encoder_fp32_heads",
        "batch": args.batch, "workers": args.workers, "augment": False,
        "time_input": "nominal", "nominal_seconds": list(NOMINAL_SECONDS),
        "full_unmodified_forward": True, "motion_only_shortcut": False,
        "tune_terminal_state_plan_gt_bitwise_verified": args.split == "tune",
        "optimizer_created": False, "backward_called": False, "weights_updated": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "runtime": runtime_receipt(),
        "final_validation_accessed": False,
    }
    atomic_json(manifest_path, metadata)
    print(json.dumps({"status": "completed", "artifact": str(output),
                      "sha256": metadata["artifact_sha256"], "manifest": str(manifest_path)},
                     sort_keys=True))


def canonical_raw24(labels, predictions=None):
    state = labels["state_target"].float() if predictions is None else predictions["pred_state"].float()
    history = (labels["history_target"].float() if predictions is None
               else predictions["pred_history"].float())
    stop = state[:, 5:6] if predictions is None else state[:, 5:6].sigmoid()
    raw = torch.cat((state[:, :5], stop, history.flatten(1), labels["goal_xy"].float()), 1)
    require(raw.shape == (len(state), 24) and torch.isfinite(raw).all(), "Canonical24 invalid")
    if predictions is None:
        require(bool(((stop == 0) | (stop == 1)).all()), "GT stop must remain binary")
    return raw


def fit_shared_normalizer(gt_train_raw24):
    require(gt_train_raw24.ndim == 2 and gt_train_raw24.shape[1] == 24,
            "Normalizer requires canonical GT train [N,24]")
    scaled = gt_train_raw24.double() / torch.from_numpy(FEATURE_SCALE)[None]
    mean, std = scaled.mean(0), scaled.std(0, unbiased=False).clamp_min(1e-6)
    require(torch.isfinite(mean).all() and torch.isfinite(std).all() and bool((std > 0).all()),
            "Normalizer is nonfinite")
    return {"scale": torch.from_numpy(FEATURE_SCALE.copy()), "mean": mean, "std": std}


def apply_normalizer(raw24, normalizer):
    return ((raw24.double() / normalizer["scale"]) - normalizer["mean"]) / normalizer["std"]


def tiny_model():
    return nn.Sequential(nn.Linear(24, 512), nn.GELU(), nn.LayerNorm(512),
                         nn.Linear(512, 512), nn.GELU(), nn.LayerNorm(512),
                         nn.Linear(512, 12))


def model_state_sha(model):
    return tensor_tree_sha256(model.state_dict())


def evaluate_tiny(model, features, target):
    row_d3, distances = [], []
    with torch.inference_mode():
        for start in range(0, len(features), BATCH_SIZE):
            prediction = model(features[start:start + BATCH_SIZE].float()).reshape(-1, 6, 2)
            truth = target[start:start + BATCH_SIZE].float()
            row_d3.append(weighted_d3(prediction, truth))
            distances.append(torch.linalg.vector_norm(prediction - truth, dim=-1))
    row_d3, distances = torch.cat(row_d3), torch.cat(distances)
    row_d3_f64 = row_d3.numpy().astype(np.float64)
    distances_f64 = distances.numpy().astype(np.float64)
    return float(row_d3_f64.mean()), row_d3, {
        "ade1": float(distances_f64[:, :2].mean()),
        "ade2": float(distances_f64[:, :4].mean()),
        "ade3": float(distances_f64.mean()),
    }


def train_tiny(train_x, train_y, eval_x, eval_y, *, seed):
    require(train_x.shape == (ROWS["train"][0], 24)
            and train_y.shape == (ROWS["train"][0], 6, 2)
            and eval_x.shape == (ROWS["tune"][0], 24)
            and eval_y.shape == (ROWS["tune"][0], 6, 2), "Tiny fit fixed inventory mismatch")
    torch.manual_seed(seed)
    model = tiny_model().cpu()
    initial_sha = model_state_sha(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS)
    generator = torch.Generator().manual_seed(seed)
    order_digest = hashlib.sha256()
    step = 0
    model.train()
    x, y = train_x.float(), train_y.float()
    for _ in range(N_EPOCHS):
        order = torch.randperm(len(x), generator=generator)
        order_digest.update(order.numpy().astype("<i8", copy=False).tobytes())
        for start in range(0, len(x), BATCH_SIZE):
            index = order[start:start + BATCH_SIZE]
            prediction = model(x[index]).reshape(-1, 6, 2)
            loss = F.smooth_l1_loss(prediction, y[index], beta=.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step(); scheduler.step(); step += 1
    require(step == N_STEPS, "Tiny fit step count mismatch")
    model.eval()
    train_d3, _, _ = evaluate_tiny(model, train_x, train_y)
    tune_d3, tune_rows, tune_horizons = evaluate_tiny(model, eval_x, eval_y)
    return model, {"initial_model_state_sha256": initial_sha,
                   "final_model_state_sha256": model_state_sha(model), "steps": step,
                   "sample_order_sha256": order_digest.hexdigest(),
                   "train_d3_in_sample": train_d3, "tune_official_d3": tune_d3,
                   "tune_row_d3": tune_rows,
                   "tune_per_horizon_ade": tune_horizons}


def load_artifact(path, expected_kind, split):
    path = Path(path).resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == SCHEMA_VERSION and manifest.get("status") == "completed"
            and manifest.get("kind") == expected_kind and manifest.get("split") == split
            and manifest.get("rows") == ROWS[split][0]
            and manifest.get("rows_sha256") == ROWS[split][2]
            and manifest.get("artifact_sha256") == sha256(path)
            and manifest.get("script_sha256") == sha256(__file__), "Artifact manifest mismatch")
    if expected_kind == "ground_truth":
        require(manifest.get("split_manifest_sha256") == SPLIT_SHA256
                and manifest.get("supervision_manifest_sha256") == SUPERVISION_SHA256
                and manifest.get("calibration_sha256") == CALIBRATION_SHA256
                and manifest.get("ego_cache_sha256") == EGO_SHA256
                and manifest.get("image_reads") == 0
                and manifest.get("allowed_splits") == ["train", "tune"]
                and manifest.get("final_validation_accessed") is False,
                "GT artifact provenance mismatch")
    else:
        seed = int(manifest.get("base_seed", -1))
        require(seed in P7_CONTROL
                and manifest.get("checkpoint_sha256") == P7_CONTROL[seed]["checkpoint_sha256"]
                and manifest.get("run_manifest_sha256") == P7_CONTROL[seed]["manifest_sha256"]
                and manifest.get("p7_training_source_manifest_sha256") == P7_SOURCE_MANIFEST_SHA256
                and manifest.get("runtime_source_manifest_sha256") == RUNTIME_SOURCE_MANIFEST_SHA256
                and manifest.get("model_state_sha256_before") == manifest.get("model_state_sha256_after")
                and manifest.get("mode") == "eval" and manifest.get("context") == "torch.inference_mode"
                and manifest.get("precision") == "bf16_encoder_fp32_heads"
                and manifest.get("batch") == (8 if split == "train" else 4)
                and manifest.get("workers") == 4 and manifest.get("augment") is False
                and manifest.get("time_input") == "nominal"
                and manifest.get("nominal_seconds") == list(NOMINAL_SECONDS)
                and manifest.get("full_unmodified_forward") is True
                and manifest.get("motion_only_shortcut") is False
                and manifest.get("tune_terminal_state_plan_gt_bitwise_verified") is (split == "tune")
                and manifest.get("optimizer_created") is False
                and manifest.get("backward_called") is False
                and manifest.get("weights_updated") is False
                and manifest.get("final_validation_accessed") is False,
                "Prediction artifact provenance mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected_n, _, expected_rows_sha = ROWS[split]
    if expected_kind == "ground_truth":
        validate_label_payload(payload, expected_n, expected_rows_sha)
    else:
        validate_prediction_payload(payload, expected_n, expected_rows_sha,
                                    int(manifest["base_seed"]), split)
    return payload, manifest, {"artifact": str(path), "artifact_sha256": sha256(path),
                               "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path)}


def fit_probe(args):
    gt_train, _, gt_train_receipt = load_artifact(args.gt_train, "ground_truth", "train")
    gt_tune, _, gt_tune_receipt = load_artifact(args.gt_tune, "ground_truth", "tune")
    normalizer = fit_shared_normalizer(canonical_raw24(gt_train))
    normalizer_sha = tensor_tree_sha256(normalizer)
    gt_train_x = apply_normalizer(canonical_raw24(gt_train), normalizer).float()
    gt_tune_x = apply_normalizer(canonical_raw24(gt_tune), normalizer).float()
    runs, checkpoints, inputs = {}, {}, {"gt_train": gt_train_receipt, "gt_tune": gt_tune_receipt}
    for seed in (0, 1):
        pred_train, train_manifest, train_receipt = load_artifact(
            getattr(args, f"pred{seed}_train"), "prediction", "train")
        pred_tune, tune_manifest, tune_receipt = load_artifact(
            getattr(args, f"pred{seed}_tune"), "prediction", "tune")
        require(train_manifest["base_seed"] == tune_manifest["base_seed"] == seed,
                "Prediction base seed mismatch")
        require(torch.equal(pred_train["rows"], gt_train["rows"])
                and torch.equal(pred_tune["rows"], gt_tune["rows"]), "GT/pred row join mismatch")
        pred_train_x = apply_normalizer(canonical_raw24(gt_train, pred_train), normalizer).float()
        pred_tune_x = apply_normalizer(canonical_raw24(gt_tune, pred_tune), normalizer).float()
        pair = {}
        for arm, train_x, tune_x in (("gt", gt_train_x, gt_tune_x),
                                     ("pred", pred_train_x, pred_tune_x)):
            model, result = train_tiny(train_x, gt_train["gt_plan"], tune_x,
                                       gt_tune["gt_plan"], seed=seed)
            checkpoint_path = Path(args.output_dir).resolve() / f"base{seed}_{arm}_last60.pt"
            atomic_torch(checkpoint_path, {"model": model.state_dict(), "epoch": N_EPOCHS,
                                           "steps": N_STEPS, "seed": seed, "arm": arm,
                                           "normalizer_sha256": normalizer_sha})
            result["checkpoint"] = str(checkpoint_path)
            result["checkpoint_sha256"] = sha256(checkpoint_path)
            result.pop("tune_row_d3")
            pair[arm] = result
            checkpoints[f"base{seed}_{arm}"] = checkpoint_path
        require(pair["gt"]["initial_model_state_sha256"] == pair["pred"]["initial_model_state_sha256"]
                and pair["gt"]["sample_order_sha256"] == pair["pred"]["sample_order_sha256"],
                "Paired tiny runs did not share initialization and row order")
        pair["pred_minus_gt_tune_d3"] = pair["pred"]["tune_official_d3"] - pair["gt"]["tune_official_d3"]
        pair["p7_c_reference_tune_d3"] = P7_CONTROL[seed]["official_d3"]
        pair["pred_minus_p7_c_tune_d3"] = (
            pair["pred"]["tune_official_d3"] - P7_CONTROL[seed]["official_d3"])
        pair["gt_minus_p7_c_tune_d3"] = (
            pair["gt"]["tune_official_d3"] - P7_CONTROL[seed]["official_d3"])
        runs[f"base{seed}"] = pair
        inputs[f"pred{seed}_train"], inputs[f"pred{seed}_tune"] = train_receipt, tune_receipt
    report = {
        "schema_version": SCHEMA_VERSION, "status": "completed", "diagnostic_only": True,
        "non_submittable": True, "not_a_mathematical_ceiling": True,
        "feature_names": list(FEATURE_NAMES), "stop_contract": {
            "gt": "binary float 0/1", "prediction": "sigmoid(raw state_hat[5] logit)"},
        "goal_contract": "same raw provided 5-second goal XY in both arms",
        "normalizer": {"source": "one canonical GT24 grouped-train54810 matrix",
                       "shared_across_gt_pred_and_both_seeds": True,
                       "scale": normalizer["scale"], "mean": normalizer["mean"],
                       "std": normalizer["std"], "sha256": normalizer_sha},
        "recipe": {"architecture": "24-512-512-12 GELU+LayerNorm",
                   "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
                   "batch": BATCH_SIZE, "epochs": N_EPOCHS, "steps": N_STEPS,
                   "scheduler": "per-step CosineAnnealingLR T_max=3240",
                   "loss": "SmoothL1 beta=0.1", "selection": "LAST60 only"},
        "inputs": inputs, "runs": runs,
        "mean_tune": {
            "gt_d3": float(np.mean([runs[f"base{s}"]["gt"]["tune_official_d3"]
                                     for s in (0, 1)])),
            "pred_d3": float(np.mean([runs[f"base{s}"]["pred"]["tune_official_d3"]
                                       for s in (0, 1)])),
            "p7_c_reference_d3": float(np.mean([P7_CONTROL[s]["official_d3"]
                                                 for s in (0, 1)])),
        },
        "script_sha256": sha256(__file__),
        "runtime": runtime_receipt(),
        "boundaries": {"train_metrics_are_in_sample": True,
                       "raw_goal_path_differs_from_legal_P7_C_shared_scene_path": True,
                       "does_not_isolate_production_decoder": True,
                       "no_hyperparameter_or_epoch_selection": True,
                       "no_final_validation_access": True,
                       "no_model_forward_in_fit_stage": True,
                       "e26_family_on_different_grouped_split_not_e26_reproduction": True},
    }
    report_path = Path(args.output_dir).resolve() / "result.json"
    for receipt in inputs.values():
        require(sha256(receipt["artifact"]) == receipt["artifact_sha256"]
                and sha256(receipt["manifest"]) == receipt["manifest_sha256"],
                "Fit input changed during training")
    atomic_json(report_path, report)
    print(json.dumps({"status": "completed", "report": str(report_path),
                      "sha256": sha256(report_path)}, sort_keys=True))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    def data_args(p):
        p.add_argument("--data-root", required=True)
        p.add_argument("--split-manifest", required=True)
        p.add_argument("--supervision-root", required=True)
    pack = sub.add_parser("pack-labels")
    data_args(pack); pack.add_argument("--split", choices=("train", "tune"), required=True)
    pack.add_argument("--out", required=True)
    extract = sub.add_parser("extract")
    data_args(extract); extract.add_argument("--split", choices=("train", "tune"), required=True)
    extract.add_argument("--base-seed", type=int, choices=(0, 1), required=True)
    extract.add_argument("--checkpoint", required=True); extract.add_argument("--run-manifest", required=True)
    extract.add_argument("--p7-source-manifest", required=True)
    extract.add_argument("--runtime-source-manifest", required=True)
    extract.add_argument("--tune-report"); extract.add_argument("--out", required=True)
    extract.add_argument("--batch", type=int, required=True); extract.add_argument("--workers", type=int, default=4)
    extract.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    fit = sub.add_parser("fit")
    fit.add_argument("--gt-train", required=True); fit.add_argument("--gt-tune", required=True)
    for seed in (0, 1):
        fit.add_argument(f"--pred{seed}-train", dest=f"pred{seed}_train", required=True)
        fit.add_argument(f"--pred{seed}-tune", dest=f"pred{seed}_tune", required=True)
    fit.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if args.command == "pack-labels":
        pack_labels(args)
    elif args.command == "extract":
        extract_predictions(args)
    else:
        output_dir = Path(args.output_dir).resolve()
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
                "Tiny fitting is CPU-only and requires explicitly empty CUDA visibility")
        require(not output_dir.exists(), f"Refusing existing output directory: {output_dir}")
        output_dir.mkdir(parents=True)
        fit_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
