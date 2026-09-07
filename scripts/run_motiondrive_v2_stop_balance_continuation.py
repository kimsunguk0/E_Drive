#!/usr/bin/env python3
"""P6 paired continuation: only global train-stop BCE class balance changes.

This driver computes fixed class weights from label-only train supervision, then
invokes the existing trainer with a weights-only P4 LAST initialization and a
fresh optimizer.  It never opens final validation.  GPU execution requires a
separate operator gate; ``--count-only`` performs no model construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_grouped_split_v2 import sha256, validate_manifest
from motiondrive_v2_training import global_binary_class_weights, tensor_state_sha256

EXPECTED_SPLIT_SHA256 = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
EXPECTED_SUPERVISION_SHA256 = "ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93"
EXPECTED_TRAIN_ROWS = 54810
EXPECTED_TUNE_ROWS = 1998
EXPECTED_TRAIN_ROWS_SHA256 = "75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854"
EXPECTED_STOP_VALID = 54810
EXPECTED_STOP_NEGATIVE = 50829
EXPECTED_STOP_POSITIVE = 3981
EXPECTED_STOP_LABELS_SHA256 = "d587ff2c8cbab445434123c1b3b194295d7c8e30513ba2c62aa538c82098a475"
EXPECTED_STOP_TARGET_SHA256 = "fa635fd1f251b24db7678aa37372952a795626033c503aaf2939319d309c2123"
EXPECTED_STOP_VALID_SHA256 = "3e755a03d0f3d3aea0fe4a619beeded168bb0ea4d0c06243f3052d7c58293d78"
P4_TRAINING_GIT_SHA = "86620b4ffc7e6838b49cf83b5be789eba12d8027"
ARM_NAMES = ("control_unweighted", "balanced_global_train")
UNCHANGED_MODEL_DATA_SHA256 = {
    "models/motiondrive_v2/__init__.py": "3a3e6c72029bda6edee171a07b863a1739dcd87717765ae3a6af48dbe007a47d",
    "models/motiondrive_v2/config.py": "f03d75c4d7dd4a500c009cf2adf082f75b90c268ef03da22c747a2dd17df7f8b",
    "models/motiondrive_v2/model.py": "d5507cf773342daea8cba0396232b5a0768d8ec84b080106552715b7dfb2b3e6",
    "models/motiondrive_v2/motion_encoder.py": "80c84387a0337c6bfcf700db8b49a3e761f1610f8f5e6a3830c4ade58069eacd",
    "models/motiondrive_v2/planner.py": "94811887bf7edd72f4796d83fb1193c5b87b83d2313eafdc50daabab4e8f6599",
    "models/motiondrive_v2/scene_encoder.py": "2263e2281b0175309575a977db89158da1d44511b5177a398eafa713b6893d8d",
    "scripts/motiondrive_v2_data.py": "83b6c74177c131116f48feb968aa8f96821b5e820823843e8b2e97ba7af171ca",
    "scripts/sparse_scoredrive.py": "3d8e89824d813c07588a64c6fb1a44600681d54adebe8af3807c3f9ad521e7db",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: str | Path) -> str:
    return sha256(str(path))


def _sha_rows(rows) -> str:
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def validate_unchanged_model_data_sources(root: Path = ROOT) -> dict[str, str]:
    actual = {name: file_sha(root / name) for name in UNCHANGED_MODEL_DATA_SHA256}
    require(actual == UNCHANGED_MODEL_DATA_SHA256,
            "P6 model/goal/input/label source differs from immutable P4 runtime")
    return actual


def count_train_stop_labels(dataset) -> dict:
    """Count state_valid[:,5] labels without calling image-loading __getitem__."""
    require(getattr(dataset, "split", None) == "train", "Stop counts must use train split only")
    rows = np.asarray(dataset.rows, dtype=np.int64)
    require(rows.ndim == 1 and len(rows) == EXPECTED_TRAIN_ROWS,
            f"Expected complete train{EXPECTED_TRAIN_ROWS}, got {len(rows)}")
    require(len(np.unique(rows)) == len(rows), "Train rows must be unique")
    selected = set(map(int, rows.tolist()))
    seen: set[int] = set()
    labels: dict[int, tuple[str, bool, np.float32]] = {}
    canonical_by_scene: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    positive = negative = valid_count = 0
    scenes = sorted(set(map(str, dataset.scene_names[rows])))
    for scene in scenes:
        supervision = dataset._supervision(scene)
        source_rows = np.asarray(supervision.get("row"), dtype=np.int64)
        state_target = np.asarray(supervision.get("state_target"))
        state_valid = np.asarray(supervision.get("state_valid"))
        require(state_target.shape == state_valid.shape and state_target.ndim == 2
                and state_target.shape[1] == 6 and len(source_rows) == len(state_target),
                f"Malformed state supervision: {scene}")
        positions = np.flatnonzero(np.isin(source_rows, rows, assume_unique=False))
        for position in positions.tolist():
            row = int(source_rows[position])
            if row not in selected:
                continue
            require(row not in seen, f"Duplicate train supervision row: {row}")
            seen.add(row)
            is_valid = bool(state_valid[position, 5])
            target32 = np.float32(state_target[position, 5])
            labels[row] = (scene, is_valid, target32)
            if is_valid:
                target = float(target32)
                require(np.isfinite(target) and target in (0., 1.),
                        f"Valid stop target must be binary: row {row}")
                valid_count += 1
                positive += int(target >= .5)
                negative += int(target < .5)
        canonical_by_scene[scene] = (
            np.asarray(source_rows[positions], dtype="<i8"),
            np.asarray(state_target[positions, 5], dtype="<f4"),
            np.asarray(state_valid[positions, 5], dtype=np.uint8),
        )
    require(seen == selected, "Train supervision rows do not exactly cover selected rows")
    require(valid_count == positive + negative and positive > 0 and negative > 0,
            "Both global train stop classes must be present")
    label_digest = hashlib.sha256()
    for scene in scenes:
        scene_rows, scene_targets, scene_valid = canonical_by_scene[scene]
        require(all(labels[int(row)][0] == scene for row in scene_rows),
                "Canonical stop-label scene mismatch")
        scene_digest = hashlib.sha256()
        scene_digest.update(scene.encode("utf-8") + b"\0")
        scene_digest.update(np.asarray(scene_rows, dtype="<i8").tobytes())
        scene_digest.update(scene_targets.tobytes())
        scene_digest.update(scene_valid.tobytes())
        label_digest.update(scene.encode("utf-8") + b"\0" + scene_digest.hexdigest().encode("ascii") + b"\n")
    ordered_targets = np.asarray([labels[int(row)][2] for row in rows], dtype="<f4")
    ordered_valid = np.asarray([labels[int(row)][1] for row in rows], dtype=np.uint8)
    negative_weight, positive_weight = global_binary_class_weights(negative, positive)
    target_sha = hashlib.sha256(ordered_targets.tobytes()).hexdigest()
    valid_sha = hashlib.sha256(ordered_valid.tobytes()).hexdigest()
    return {
        "split": "train", "rows": len(rows), "rows_sha256": _sha_rows(rows),
        "scenes": len(scenes), "valid": valid_count, "negative": negative, "positive": positive,
        "labels_sha256": label_digest.hexdigest(),
        "labels_sha256_serialization": "per sorted scene sha256(scene UTF-8 + NUL + contiguous int64le rows + float32le target5 + uint8 valid5); aggregate sha256 of scene UTF-8 + NUL + lowercase scene digest + newline",
        "target5_sha256_dataset_row_order": target_sha,
        "valid5_sha256_dataset_row_order": valid_sha,
        "weights_negative_positive": [negative_weight, positive_weight],
        "target_definition": "state_valid[:,5] AND state_target[:,5]>=0.5; target generated from causal current planar speed <0.2 m/s",
        "image_getitem_calls": 0, "final_validation_accessed": False,
        "batch_recomputation": False,
    }


def make_train_dataset(data_root: str, split_manifest: str, supervision_root: str, seed: int):
    from motiondrive_v2_data import MotionDriveDataset
    return MotionDriveDataset(data_root=data_root, split_manifest=split_manifest,
                              supervision_root=supervision_root, split="train", min_frame=30,
                              frame_stride=1, max_samples=0, augment=True, seed=seed)


def validate_data_contract(data_root: str, split_manifest: str, supervision_root: str,
                           seed: int) -> tuple[dict, dict]:
    unchanged_sources = validate_unchanged_model_data_sources()
    split_path = Path(split_manifest).resolve()
    supervision_path = Path(supervision_root).resolve() / "supervision_manifest.json"
    require(file_sha(split_path) == EXPECTED_SPLIT_SHA256, "P6 split manifest SHA mismatch")
    require(file_sha(supervision_path) == EXPECTED_SUPERVISION_SHA256,
            "P6 C1 supervision manifest SHA mismatch")
    split = json.loads(split_path.read_text())
    validate_manifest(split)
    require(len(split["splits"]["train"]) == 203 and len(split["splits"]["tune"]) == 37,
            "Expected fixed train203/tune37 scene split")
    train_sessions = {split["scene_to_session"][x] for x in split["splits"]["train"]}
    tune_sessions = {split["scene_to_session"][x] for x in split["splits"]["tune"]}
    final_sessions = {split["scene_to_session"][x] for x in split["splits"]["val"]}
    require(len(train_sessions) == 72 and len(tune_sessions) == 11 and len(final_sessions) == 31,
            "Expected fixed 72/11/31 session split")
    require(not (train_sessions & tune_sessions or train_sessions & final_sessions or tune_sessions & final_sessions),
            "Session split leakage")
    dataset = make_train_dataset(data_root, str(split_path), supervision_root, seed)
    counts = count_train_stop_labels(dataset)
    require(counts["rows_sha256"] == EXPECTED_TRAIN_ROWS_SHA256, "P6 train row-order SHA mismatch")
    require(counts["target5_sha256_dataset_row_order"] == EXPECTED_STOP_TARGET_SHA256
            and counts["valid5_sha256_dataset_row_order"] == EXPECTED_STOP_VALID_SHA256,
            "P6 ordered raw stop target/valid digest mismatch")
    return counts, {"split_manifest_sha256": EXPECTED_SPLIT_SHA256,
                    "supervision_manifest_sha256": EXPECTED_SUPERVISION_SHA256,
                    "train_scene_count": 203, "train_session_count": 72,
                    "tune_scene_count": 37, "tune_session_count": 11,
                    "final_session_count_declared_not_accessed": 31,
                    "unchanged_p4_model_data_source_sha256": unchanged_sources}


def validate_p4_initialization(checkpoint_path: str, expected_checkpoint_sha256: str,
                               run_manifest_path: str, expected_run_manifest_sha256: str,
                               split_sha256: str, base_seed: int) -> dict:
    checkpoint_path = str(Path(checkpoint_path).resolve())
    run_manifest_path = str(Path(run_manifest_path).resolve())
    require(file_sha(checkpoint_path) == expected_checkpoint_sha256, "P4 LAST file SHA mismatch")
    require(file_sha(run_manifest_path) == expected_run_manifest_sha256, "P4 run-manifest file SHA mismatch")
    sidecar = json.loads(Path(run_manifest_path).read_text())
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict) and set(("model", "optimizer", "step", "manifest")) <= set(payload),
            "P4 LAST payload incomplete")
    require(payload["step"] == 6000, "P6 must start at P4 LAST6000")
    manifest = payload["manifest"]
    args, config = manifest.get("arguments", {}), manifest.get("model_config", {})
    require(manifest.get("git_sha") == P4_TRAINING_GIT_SHA, "P4 embedded training Git mismatch")
    require(sidecar.get("git_sha") == P4_TRAINING_GIT_SHA, "P4 sidecar training Git mismatch")
    require(sidecar.get("step") == 6000 and sidecar.get("status") == "completed",
            "P4 sidecar must record completed step6000 training")
    require(sidecar.get("load_report", {}).get("common_checkpoint_sha256")
            == manifest.get("load_report", {}).get("common_checkpoint_sha256"),
            "P4 checkpoint/sidecar initialization lineage mismatch")
    expected_args = {"phase": "joint", "steps": 6000, "goal_on": 1, "state_on": 1,
                     "motion_input_mode": "low_feature", "bn_policy": "fixed",
                     "time_input": "nominal", "precision": "bf16", "batch": 16,
                     "microbatch": 2, "seed": base_seed, "train_stride": 1,
                     "eval_stride": 5, "max_train_samples": 0, "max_eval_samples": 0}
    require(all(args.get(k) == v for k, v in expected_args.items()), "P4 training argument mismatch")
    require(config.get("backbone_arch") == "resnet50" and config.get("goal_on") is True
            and config.get("state_on") is True and config.get("motion_input_mode") == "low_feature"
            and list(config.get("plan_output_scale", ())) == [10., 5.], "P4 model config mismatch")
    require(manifest.get("split_sha256") == split_sha256, "P4 split lineage mismatch")
    require(manifest.get("loss_weights", {}).get("plan") == 1., "P4 joint loss mismatch")
    from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
    model = MotionDriveV2(MotionDriveV2Config(**config))
    incompatible = model.load_state_dict(payload["model"], strict=True)
    require(not incompatible.missing_keys and not incompatible.unexpected_keys,
            "P4 strict model load was not exact")
    require(all(torch.isfinite(v).all().item() for v in payload["model"].values()
                if torch.is_floating_point(v)), "P4 model contains nonfinite tensors")
    state_sha = tensor_state_sha256(payload["model"])
    require(tensor_state_sha256(model.state_dict()) == state_sha, "P4 strict-loaded state SHA mismatch")
    return {"checkpoint_path": checkpoint_path, "checkpoint_sha256": expected_checkpoint_sha256,
            "run_manifest_path": run_manifest_path,
            "run_manifest_sha256": expected_run_manifest_sha256,
            "training_git_sha": P4_TRAINING_GIT_SHA, "step": 6000,
            "base_seed": base_seed, "initial_model_state_sha256": state_sha,
            "strict_load_missing": [], "strict_load_unexpected": [],
            "supervisor_integrity_exception": "P4 computation completed; outer supervisor rc1 from later global HEAD drift is preserved and is not relabeled"}


def build_experiment(arm: str, counts: Mapping[str, object], initialization: Mapping[str, object],
                     source: Mapping[str, object]) -> dict:
    require(arm in ARM_NAMES, "Unknown P6 arm")
    weights = None if arm == "control_unweighted" else list(counts["weights_negative_positive"])
    return {
        "schema_version": 1, "name": "p6_global_stop_class_balance", "arm": arm,
        "last_only_final_eval": True, "stop_class_weights": weights,
        "expected_initial_model_state_sha256": initialization["initial_model_state_sha256"],
        "expected_optimizer_groups": [{"name": "backbone", "base_lr": 5e-6},
                                      {"name": "head", "base_lr": 5e-5}],
        "train_label_counts": {**{k: int(counts[k]) for k in ("rows", "valid", "negative", "positive")},
                               "rows_sha256": str(counts["rows_sha256"]),
                               "labels_sha256": str(counts["labels_sha256"])},
        "fresh_optimizer_step_zero": True,
        "all_model_parameters_joint_trainable": True,
        "stop_target_definition": str(counts["target_definition"]),
        "balanced_output_is_calibrated_posterior": False,
        "raw_stop_logit_is_planner_input": True,
        "source": dict(source),
    }


def trainer_argv(args, initialization: Mapping[str, object]) -> list[str]:
    return [
        "--data-root", str(Path(args.data_root).resolve()),
        "--split-manifest", str(Path(args.split_manifest).resolve()),
        "--supervision-root", str(Path(args.supervision_root).resolve()),
        "--run-dir", str(Path(args.run_dir).resolve()), "--phase", "joint",
        "--goal-on", "1", "--state-on", "1", "--gpu", str(args.gpu),
        "--seed", str(args.base_seed), "--steps", "1000", "--batch", "16",
        "--microbatch", "2", "--eval-batch", "4", "--workers", str(args.workers),
        "--eval-every", "1000", "--save-every", "1000", "--log-every", "10",
        "--lr", "0.00005", "--backbone-lr", "0.000005", "--weight-decay", "0.01",
        "--warmup", "100", "--grad-clip", "5", "--alpha-occ", "0.2",
        "--alpha-lane", "0.2", "--alpha-motion", "0.2", "--uncertainty", "1",
        "--precision", "bf16", "--time-input", "nominal", "--bn-policy", "fixed",
        "--init", initialization["checkpoint_path"], "--arch", "resnet50",
        "--motion-input-mode", "low_feature", "--train-stride", "1",
        "--eval-stride", "5", "--max-train-samples", "0", "--max-eval-samples", "0",
        "--eval-split", "tune", "--cuda-memory-limit-mib", str(args.cuda_memory_limit_mib),
        "--cuda-min-free-mib", str(args.cuda_min_free_mib),
    ]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--arm", choices=ARM_NAMES, required=True)
    parser.add_argument("--base-seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--expected-init-sha256", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--expected-run-manifest-sha256", required=True)
    parser.add_argument("--expected-stop-valid", type=int, required=True)
    parser.add_argument("--expected-stop-negative", type=int, required=True)
    parser.add_argument("--expected-stop-positive", type=int, required=True)
    parser.add_argument("--expected-stop-labels-sha256", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0,
                        help="Logical CUDA id inside an operator-owned CUDA_VISIBLE_DEVICES namespace")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cuda-memory-limit-mib", type=int, default=0)
    parser.add_argument("--cuda-min-free-mib", type=int, default=0)
    parser.add_argument("--count-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    require(len(args.expected_init_sha256) == 64 and len(args.expected_run_manifest_sha256) == 64,
            "Expected artifact SHAs must be full SHA256")
    counts, data = validate_data_contract(args.data_root, args.split_manifest,
                                          args.supervision_root, args.base_seed)
    expected_counts = {"valid": args.expected_stop_valid, "negative": args.expected_stop_negative,
                       "positive": args.expected_stop_positive,
                       "labels_sha256": args.expected_stop_labels_sha256}
    require(expected_counts == {"valid": EXPECTED_STOP_VALID, "negative": EXPECTED_STOP_NEGATIVE,
                                "positive": EXPECTED_STOP_POSITIVE,
                                "labels_sha256": EXPECTED_STOP_LABELS_SHA256},
            "P6 CLI stop-label pins differ from preregistered measured train counts")
    require(all(counts[key] == value for key, value in expected_counts.items()),
            "P6 externally pinned train stop counts/labels SHA mismatch")
    if args.count_only:
        print(json.dumps({"status": "completed_cpu_label_only", "counts": counts,
                          "data": data, "cuda_initialized": torch.cuda.is_initialized()},
                         sort_keys=True, allow_nan=False), flush=True)
        return
    initialization = validate_p4_initialization(
        args.init, args.expected_init_sha256, args.run_manifest,
        args.expected_run_manifest_sha256, data["split_manifest_sha256"], args.base_seed)
    source = {path: file_sha(ROOT / path) for path in (
        "scripts/run_motiondrive_v2_stop_balance_continuation.py",
        "scripts/train_motiondrive_v2.py", "scripts/motiondrive_v2_training.py")}
    source.update(data)
    experiment = build_experiment(args.arm, counts, initialization, source)
    before = {"init": file_sha(args.init), "run_manifest": file_sha(args.run_manifest),
              "split": file_sha(args.split_manifest),
              "supervision": file_sha(Path(args.supervision_root) / "supervision_manifest.json")}
    import train_motiondrive_v2
    train_motiondrive_v2.run_training(trainer_argv(args, initialization), experiment=experiment)
    after = {"init": file_sha(args.init), "run_manifest": file_sha(args.run_manifest),
             "split": file_sha(args.split_manifest),
             "supervision": file_sha(Path(args.supervision_root) / "supervision_manifest.json")}
    require(before == after, "Immutable P4/data inputs changed during P6")


if __name__ == "__main__":
    main()
