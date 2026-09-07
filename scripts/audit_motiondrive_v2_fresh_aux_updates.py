#!/usr/bin/env python3
"""CPU-only audit of auxiliary updates in a fresh MotionDrive V2 training stage.

This reads existing checkpoints and a trainer manifest.  It never reads the
dataset and never executes a model forward.  The optimizer-id mapping mirrors
the two parameter groups, predicate, and parameter order in
``scripts/train_motiondrive_v2.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from scripts.export_motiondrive_v2_inference import validate_complete_config
from scripts.motiondrive_v2_training import tensor_state_sha256


PRIMARY_AUXILIARIES = {
    "occupancy": ("scene_encoder.occ_head.",),
    "lane": ("scene_encoder.lane_head.",),
    "history": ("motion_encoder.history_head.",),
    "state": ("motion_encoder.state_head.",),
}
UNCERTAINTY_AUXILIARIES = {
    "history_uncertainty": ("motion_encoder.history_uncertainty_head.",),
    "state_uncertainty": ("motion_encoder.state_uncertainty_head.",),
}
STOP_WEIGHT = "motion_encoder.state_head.2.weight"
STOP_BIAS = "motion_encoder.state_head.2.bias"
PUBLIC_SHA256 = "4096396018c0cf59fbe0eb1afe6e269f4676b34460bed5eedde5d7680d58bb4e"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_cpu_only() -> None:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "CPU audit requires CUDA_VISIBLE_DEVICES='' explicitly")
    require(not torch.cuda.is_initialized(), "CUDA must not be initialized")


def input_file(value: str | Path, label: str) -> Path:
    given = Path(value).absolute()
    require(not given.is_symlink(), f"{label} must be a regular non-symlink file")
    path = given.resolve(strict=True)
    require(path.is_file(), f"{label} must be a regular non-symlink file")
    return path


def new_output(value: str | Path) -> Path:
    given = Path(value).absolute()
    require(not os.path.lexists(given), f"Refusing existing output: {given}")
    path = given.resolve()
    require(not os.path.lexists(path) and path.parent.is_dir() and path.suffix == ".json",
            "Output must be a new .json file with an existing parent")
    return path


def same_json(left, right) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def validate_expected_stage(arguments: Mapping, last_step: int, sidecar_step: int,
                            expected_phase: str, expected_steps: int) -> dict:
    require(type(expected_steps) is int and expected_steps > 0, "Expected steps must be a positive integer")
    require(expected_phase in ("joint", "pretrain"), "Expected phase must be joint or pretrain")
    require((expected_phase, expected_steps) in (("joint", 2), ("pretrain", 2000)),
            "Audit scope is limited to joint/2 canary or pretrain/2000")
    actual = {"manifest_phase": arguments.get("phase"),
              "manifest_argument_steps": arguments.get("steps"),
              "last_payload_step": last_step, "sidecar_step": sidecar_step}
    require(actual == {"manifest_phase": expected_phase,
                       "manifest_argument_steps": expected_steps,
                       "last_payload_step": expected_steps, "sidecar_step": expected_steps},
            "Manifest phase/steps and LAST payload must match the explicit expected stage")
    return {"expected": {"phase": expected_phase, "steps": expected_steps}, "actual": actual}


def validate_producer_consumer_lineage(initial_manifest: Mapping, last_manifest: Mapping,
                                        sidecar: Mapping, initial_file_sha: str,
                                        initial_tensor_sha: str) -> dict:
    producer_git = initial_manifest.get("git_sha")
    source = initial_manifest.get("source")
    require(isinstance(producer_git, str) and producer_git,
            "Public initializer producer git_sha required")
    require(isinstance(source, Mapping) and source.get("git_sha") == producer_git,
            "Public initializer source/producer git_sha mismatch")
    source_files = source.get("file_sha256")
    require(isinstance(source_files, Mapping) and bool(source_files)
            and all(isinstance(name, str) and isinstance(digest, str) and len(digest) == 64
                    and all(character in "0123456789abcdef" for character in digest)
                    for name, digest in source_files.items()),
            "Public initializer source file SHA mapping is missing or malformed")
    trainer_gits = (last_manifest.get("git_sha"), sidecar.get("git_sha"))
    require(all(isinstance(value, str) and value for value in trainer_gits)
            and trainer_gits[0] == trainer_gits[1],
            "LAST embedded and sidecar trainer git_sha mismatch")
    require(sidecar.get("load_report", {}).get("common_checkpoint_sha256") == initial_file_sha,
            "Trainer did not consume the supplied public initializer file SHA")
    require(sidecar.get("initial_model_state_sha256") == initial_tensor_sha,
            "Trainer initial tensor SHA does not match the supplied public initializer")
    require(initial_manifest.get("initial_model_state_sha256") == initial_tensor_sha,
            "Public initializer manifest does not identify its actual model tensors")
    return {"initializer_producer_git_sha": producer_git,
            "trainer_git_sha": trainer_gits[0],
            "producer_and_trainer_git_may_differ": True,
            "consumed_initializer_file_sha256": initial_file_sha,
            "consumed_initializer_tensor_sha256": initial_tensor_sha,
            "runtime_source_file_intersection": "unavailable: LAST/sidecar manifest has no source file hash mapping"}


def validate_public_initializer_manifest(manifest: Mapping) -> dict:
    require(manifest.get("status") == "public_initialization_only" and manifest.get("step") == 0,
            "Initial checkpoint must be the zero-step public initializer")
    require(manifest.get("public_checkpoint_sha256") == PUBLIC_SHA256
            and manifest.get("pretrained_sha256") == PUBLIC_SHA256,
            "Public backbone SHA provenance mismatch")
    require(manifest.get("etri_optimizer_steps") == 0
            and manifest.get("etri_optimizer_updates") == 0
            and manifest.get("labels_read") is False and manifest.get("image_data_read") is False,
            "Public initializer must contain zero ETRI training/data use")
    report = manifest.get("load_report", {}).get("backbone", {})
    require(report.get("injected") == 318 and report.get("unexpected") == []
            and report.get("nonhead_missing") == []
            and report.get("fpn_initialized_from_checkpoint") is False,
            "Public backbone load report mismatch")
    bn = manifest.get("bn_training", {})
    require(bn.get("policy") == "fixed" and bn.get("affine_and_backbone_weights_trainable") is True,
            "Public initializer fixed-BN policy mismatch")
    config = validate_complete_config(dict(manifest.get("model_config", {}))).to_dict()
    require(config["goal_on"] is False and config["state_on"] is False,
            "Public initializer must be G0S0")
    return config


def optimizer_parameter_name_groups(model: torch.nn.Module) -> list[list[str]]:
    """Mirror the original trainer's backbone/other construction exactly."""
    backbone, other = [], []
    for name, _ in model.named_parameters():
        is_backbone = name.startswith("backbone_fpn.") and not name.startswith(
            ("backbone_fpn.lat", "backbone_fpn.out"))
        (backbone if is_backbone else other).append(name)
    return [backbone, other]


def optimizer_id_to_name(model: torch.nn.Module, optimizer: Mapping) -> tuple[dict[int, str], list[dict]]:
    """Map serialized AdamW ids using the trainer's exact two-group order."""
    require(isinstance(optimizer, Mapping), "Optimizer checkpoint must be a mapping")
    saved_groups = optimizer.get("param_groups")
    require(isinstance(saved_groups, list) and len(saved_groups) == 2,
            "Expected the original trainer's two AdamW parameter groups")
    expected_groups = optimizer_parameter_name_groups(model)
    mapping: dict[int, str] = {}
    summaries = []
    next_id = 0
    for index, (saved, names) in enumerate(zip(saved_groups, expected_groups)):
        ids = saved.get("params") if isinstance(saved, Mapping) else None
        require(isinstance(ids, list) and len(ids) == len(names),
                f"Optimizer parameter group {index} length/order contract mismatch")
        require(all(type(value) is int for value in ids), "Optimizer parameter ids must be integers")
        expected_ids = list(range(next_id, next_id + len(names)))
        require(ids == expected_ids,
                f"Optimizer parameter group {index} serialized id order contract mismatch")
        next_id += len(names)
        require(not set(ids).intersection(mapping), "Duplicate optimizer parameter id")
        mapping.update(zip(ids, names))
        encoded = json.dumps(names, separators=(",", ":"), ensure_ascii=True).encode()
        summaries.append({"index": index, "role": "backbone" if index == 0 else "other",
                          "parameter_count": len(names),
                          "ordered_names_sha256": hashlib.sha256(encoded).hexdigest()})
    require(len(mapping) == sum(len(group) for group in expected_groups),
            "Incomplete optimizer parameter mapping")
    state = optimizer.get("state")
    require(isinstance(state, Mapping) and set(state).issubset(mapping),
            "Optimizer state contains an unmapped parameter id")
    return mapping, summaries


def squared_norm(tensor: torch.Tensor) -> float:
    return float(tensor.detach().double().square().sum())


def parameter_update_report(initial_state: Mapping[str, torch.Tensor],
                            last_state: Mapping[str, torch.Tensor], optimizer: Mapping,
                            id_to_name: Mapping[int, str], prefixes: tuple[str, ...]) -> dict:
    names = sorted(name for name in id_to_name.values() if name.startswith(prefixes))
    require(names, f"No parameters matched prefixes {prefixes}")
    name_to_id = {name: key for key, name in id_to_name.items()}
    delta_sq = moment_sq = 0.0
    changed, nonzero_moment, state_present = [], [], []
    all_finite = True
    optimizer_state = optimizer["state"]
    for name in names:
        before, after = initial_state[name], last_state[name]
        delta = after.detach().double() - before.detach().double()
        delta_sq += squared_norm(delta)
        all_finite = all_finite and bool(torch.isfinite(before).all()) and bool(torch.isfinite(after).all())
        if not torch.equal(before, after):
            changed.append(name)
        entry = optimizer_state.get(name_to_id[name])
        if entry is None:
            continue
        state_present.append(name)
        exp_avg = entry.get("exp_avg")
        require(isinstance(exp_avg, torch.Tensor) and exp_avg.shape == after.shape,
                f"Missing or malformed Adam exp_avg for {name}")
        all_finite = all_finite and bool(torch.isfinite(exp_avg).all())
        moment_finite = bool(torch.isfinite(exp_avg).all())
        moment_sq += squared_norm(exp_avg)
        if moment_finite and bool(torch.count_nonzero(exp_avg)):
            nonzero_moment.append(name)
    gradient_updated = sorted(set(changed).intersection(nonzero_moment))
    return {
        "prefixes": list(prefixes), "parameter_count": len(names),
        "parameters_with_optimizer_state": len(state_present),
        "actual_delta_count": len(changed),
        "finite_nonzero_exp_avg_count": len(nonzero_moment),
        "all_tensors_and_exp_avg_finite": all_finite,
        "delta_l2_norm": math.sqrt(delta_sq), "exp_avg_l2_norm": math.sqrt(moment_sq),
        "changed_parameters": changed, "nonzero_exp_avg_parameters": nonzero_moment,
        "changed_with_nonzero_exp_avg_parameters": gradient_updated,
        "actual_gradient_update_observed": bool(gradient_updated and all_finite),
    }


def stop_row_report(initial_state: Mapping[str, torch.Tensor], last_state: Mapping[str, torch.Tensor],
                    optimizer: Mapping, id_to_name: Mapping[int, str]) -> dict:
    """Inspect output row 5, which is uniquely supervised by stop BCE."""
    name_to_id = {name: key for key, name in id_to_name.items()}
    reports = []
    delta_sq = moment_sq = 0.0
    finite, changed, nonzero = True, 0, 0
    for name in (STOP_WEIGHT, STOP_BIAS):
        require(name in name_to_id and name in initial_state and name in last_state,
                f"Missing stop-head parameter: {name}")
        before, after = initial_state[name][5], last_state[name][5]
        entry = optimizer["state"].get(name_to_id[name])
        require(isinstance(entry, Mapping) and isinstance(entry.get("exp_avg"), torch.Tensor),
                f"Missing stop-head Adam exp_avg: {name}")
        moment = entry["exp_avg"][5]
        is_changed = not torch.equal(before, after)
        is_nonzero = bool(torch.count_nonzero(moment))
        is_finite = bool(torch.isfinite(before).all() and torch.isfinite(after).all()
                         and torch.isfinite(moment).all())
        changed += int(is_changed)
        nonzero += int(is_nonzero and is_finite)
        finite = finite and is_finite
        delta_sq += squared_norm(after.detach().double() - before.detach().double())
        moment_sq += squared_norm(moment)
        reports.append({"parameter": name, "output_index": 5, "actual_delta": is_changed,
                        "finite_nonzero_exp_avg": is_nonzero and is_finite})
    return {"definition": "state_head final output row 5; isolated stop-logit path",
            "rows": reports, "actual_delta_count": changed,
            "finite_nonzero_exp_avg_count": nonzero, "all_tensors_and_exp_avg_finite": finite,
            "delta_l2_norm": math.sqrt(delta_sq), "exp_avg_l2_norm": math.sqrt(moment_sq),
            "actual_gradient_update_observed": bool(changed and nonzero and finite)}


def validate_model_states(config: Mapping, initial_state: Mapping, last_state: Mapping) -> MotionDriveV2:
    model = MotionDriveV2(validate_complete_config(dict(config))).cpu()
    target = model.state_dict()
    require(set(initial_state) == set(last_state) == set(target), "Checkpoint model keys differ")
    for label, state in (("initial", initial_state), ("last", last_state)):
        for name, value in state.items():
            expected = target[name]
            require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                    and value.shape == expected.shape and value.dtype == expected.dtype
                    and bool(torch.isfinite(value).all()), f"Invalid {label} model tensor: {name}")
        model.load_state_dict(state, strict=True)
    return model


def audit(initial_checkpoint: str | Path, last_checkpoint: str | Path,
          manifest_path: str | Path, output: str | Path, *, expected_phase: str = "joint",
          expected_steps: int = 2) -> dict:
    assert_cpu_only()
    initial_path = input_file(initial_checkpoint, "Initial checkpoint")
    last_path = input_file(last_checkpoint, "LAST checkpoint")
    sidecar_path = input_file(manifest_path, "Manifest")
    output_path = new_output(output)
    require(len({initial_path, last_path, sidecar_path, output_path}) == 4, "Input/output paths must differ")
    before = {"initial_checkpoint": sha256(initial_path), "last_checkpoint": sha256(last_path),
              "manifest": sha256(sidecar_path)}
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    sidecar = json.loads(sidecar_path.read_text())
    require(isinstance(initial, Mapping) and isinstance(last, Mapping) and isinstance(sidecar, Mapping),
            "Checkpoint and manifest mappings required")
    initial_manifest, last_manifest = initial.get("manifest", {}), last.get("manifest", {})
    require(isinstance(initial_manifest, Mapping) and isinstance(last_manifest, Mapping),
            "Embedded checkpoint manifests required")
    initial_step, last_step, sidecar_step = initial.get("step"), last.get("step"), sidecar.get("step")
    require(type(initial_step) is int and initial_step == 0, "Initializer must be step zero")
    require(initial.get("optimizer") == {}, "Step-zero public initializer optimizer must be empty")
    require(type(last_step) is int and type(sidecar_step) is int,
            "LAST and sidecar optimizer steps must be integers")
    arguments = sidecar.get("arguments", {})
    require(isinstance(arguments, Mapping), "Manifest arguments mapping required")
    stage = validate_expected_stage(arguments, last_step, sidecar_step,
                                    expected_phase, expected_steps)
    require(sidecar.get("status") == "completed", "Trainer sidecar must be completed")
    require(Path(arguments.get("init", "")).resolve() == initial_path,
            "Manifest initializer path does not identify the supplied checkpoint")
    split_values = [initial_manifest.get("split_sha256"), last_manifest.get("split_sha256"),
                    sidecar.get("split_sha256")]
    require(all(isinstance(value, str) and value for value in split_values)
            and len(set(split_values)) == 1, "Initial/LAST/sidecar split_sha256 identity mismatch")
    for field in ("model_config", "arguments", "initial_model_state_sha256", "load_report",
                  "loss_weights", "supervision_manifest_sha256"):
        require(same_json(last_manifest.get(field), sidecar.get(field)),
                f"LAST embedded/sidecar identity mismatch: {field}")
    initial_state, last_state = initial.get("model"), last.get("model")
    require(isinstance(initial_state, Mapping) and isinstance(last_state, Mapping), "Model state mappings required")
    initial_tensor_sha = tensor_state_sha256(initial_state)
    producer_config = validate_public_initializer_manifest(initial_manifest)
    provenance = validate_producer_consumer_lineage(initial_manifest, last_manifest, sidecar,
                                                     before["initial_checkpoint"], initial_tensor_sha)
    trainer_config = validate_complete_config(dict(sidecar.get("model_config", {}))).to_dict()
    require(all(same_json(producer_config[key], trainer_config[key])
                for key in producer_config if key not in ("goal_on", "state_on")),
            "Trainer architecture differs from the public initializer beyond G/S flags")
    expected_flags = (False, False) if expected_phase == "pretrain" else (True, True)
    require((trainer_config["goal_on"], trainer_config["state_on"]) == expected_flags,
            "Trainer G/S flags do not match the audited stage")
    require(arguments.get("bn_policy") == "fixed"
            and sidecar.get("bn_training", {}).get("policy") == "fixed",
            "Trainer fixed-BN policy mismatch")
    model = validate_model_states(sidecar["model_config"], initial_state, last_state)
    bn_names = [name for name in initial_state
                if name.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    require(bool(bn_names) and all(torch.equal(initial_state[name], last_state[name]) for name in bn_names),
            "Fixed-BN running buffers changed")
    fixed_bn_sha = tensor_state_sha256({name: last_state[name] for name in bn_names})
    optimizer = last.get("optimizer")
    id_to_name, group_summaries = optimizer_id_to_name(model, optimizer)
    primary = {name: parameter_update_report(initial_state, last_state, optimizer, id_to_name, prefixes)
               for name, prefixes in PRIMARY_AUXILIARIES.items()}
    uncertainty = {name: parameter_update_report(initial_state, last_state, optimizer, id_to_name, prefixes)
                   for name, prefixes in UNCERTAINTY_AUXILIARIES.items()}
    stop = stop_row_report(initial_state, last_state, optimizer, id_to_name)
    passed = all(value["actual_gradient_update_observed"] for value in primary.values())
    after = {"initial_checkpoint": sha256(initial_path), "last_checkpoint": sha256(last_path),
             "manifest": sha256(sidecar_path)}
    require(after == before, "An audited input changed during the CPU audit")
    require(not torch.cuda.is_initialized(), "CPU audit initialized CUDA")
    result = {
        "schema_version": 1, "status": "passed" if passed else "failed",
        "primary_aux_update_gate_passed": passed,
        "primary_gate_definition": "each of occupancy/lane/history/state has >=1 same parameter with an actual delta and finite nonzero Adam exp_avg",
        "primary_auxiliaries": primary, "uncertainty_paths_report_only": uncertainty,
        "stop_path_report_only": stop,
        "optimizer_mapping": {"source": "train_motiondrive_v2.py named_parameters backbone/other predicate and order",
                              "groups": group_summaries, "mapped_parameter_count": len(id_to_name)},
        "stage_contract": stage,
        "producer_consumer_lineage": provenance,
        "lineage": {"initial_step": initial_step, "last_step": last_step,
                    "sidecar_step": sidecar_step,
                    "initializer_producer_git_sha": initial_manifest["git_sha"],
                    "trainer_git_sha": sidecar["git_sha"],
                    "split_sha256": sidecar["split_sha256"],
                    "supervision_manifest_sha256": sidecar.get("supervision_manifest_sha256"),
                    "initial_model_state_sha256": tensor_state_sha256(initial_state),
                    "last_model_state_sha256": tensor_state_sha256(last_state),
                    "fixed_bn_state_sha256": fixed_bn_sha},
        "inputs": {name: {"path": str(path), "sha256_before": before[name],
                           "sha256_after": after[name]}
                   for name, path in (("initial_checkpoint", initial_path),
                                      ("last_checkpoint", last_path), ("manifest", sidecar_path))},
        "scope": {"cpu_only": True, "cuda_initialized": False, "model_forward_performed": False,
                  "dataset_or_labels_read": False, "global_grad_norm_audited": False,
                  "os_exit_audited": False, "uncertainty_and_stop_are_not_primary_gate_terms": True},
    }
    json.dumps(result, allow_nan=False)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--last-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-phase", choices=("joint", "pretrain"), default="joint")
    parser.add_argument("--expected-steps", type=positive_int, default=2)
    args = parser.parse_args(argv)
    result = audit(args.initial_checkpoint, args.last_checkpoint, args.manifest, args.output,
                   expected_phase=args.expected_phase, expected_steps=args.expected_steps)
    print(json.dumps({"status": result["status"],
                      "primary_aux_update_gate_passed": result["primary_aux_update_gate_passed"],
                      "output": str(Path(args.output).absolute())}))
    return 0 if result["primary_aux_update_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
