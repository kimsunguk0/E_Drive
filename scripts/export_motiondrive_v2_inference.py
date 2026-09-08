#!/usr/bin/env python3
"""Export a trusted local V2 training checkpoint without optimizer/RNG state.

The COMPLETE saved manifest.model_config is required: no architecture, motion
mode, output unit, or G/S setting is filled from current defaults. The resulting
CPU tensor bundle loads unchanged through audit_motiondrive_v2.construct_model
and the benchmark's shared loader. This is packaging, not model conversion.

Training checkpoints contain Python/NumPy RNG pickles. Only use --checkpoint
with a trusted, user-owned local file; loading it uses weights_only=False.
The exported bundle itself can be opened with weights_only=True.

Generic export does NOT certify serving input compatibility. The optional
--deployment-contract geometry-v2-nominal mode requires the explicitly selected
checkpoint SHA and its original, completed run lineage on disk. It only binds
verified C1 geometry/nominal timing to the shared input contract; it does not
certify accuracy, latency, challenge compliance, or a clean OS process exit.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from models.motiondrive_v2_input_contract import input_contract


DEPLOYMENT_CONTRACT = "geometry-v2-nominal"
C1_CANONICAL_SHA256 = "8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961"
C1_SUPERVISION_SHA256 = "ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93"
RAWTIME_SPLIT_SHA256 = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"


def _required_sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Required lowercase SHA256 is missing/invalid: {name}")
    return value


def _json_form(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _lineage_path(value, root, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Explicit lineage path is required: {name}")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve(strict=True)


def _verified_file(path, expected_sha, evidence, role, *, parse_json=False):
    """Hash and parse the same bytes; retain a fingerprint for publish-time checks."""
    expected_sha = _required_sha(expected_sha, role)
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Lineage artifact is not a file: {role}")
    if parse_json:
        raw = path.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
    else:
        observed = file_sha256(path)
    if observed != expected_sha:
        raise ValueError(f"SHA256 mismatch for {role}: {observed} != {expected_sha}")
    evidence[role] = {"path": str(path), "sha256": observed, "bytes": path.stat().st_size}
    if parse_json:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON mapping for {role}")
        return parsed


def verify_deployment_lineage(checkpoint, *, source_path, source_sha256,
                              expected_checkpoint_sha256, deployment_contract):
    """Fail closed; never infer a serving contract from a run name or defaults.

    This is intentionally stricter than generic packaging. Relocated runs with
    unavailable original metadata/initializers require a separately reviewed
    provenance migration, not a path override that silently relabels weights.
    """
    if deployment_contract != DEPLOYMENT_CONTRACT:
        raise ValueError(f"Unsupported deployment contract: {deployment_contract}")
    expected = _required_sha(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    if source_sha256 != expected:
        raise ValueError("Selected checkpoint SHA256 does not match the source file")
    manifest = checkpoint.get("manifest", {})
    args = manifest.get("arguments", {})
    if not isinstance(args, dict):
        raise ValueError("Explicit training arguments are required")
    # Import only the training policy helper, never the OpenCV-dependent adapter.
    from scripts.motiondrive_v2_training import time_input_policy
    config = manifest.get("model_config", {})
    if (args.get("time_input") != "nominal" or manifest.get("time_input") != "nominal"
            or manifest.get("time_input_policy") != time_input_policy(
                "nominal", config.get("nominal_history_seconds", (.1, .2, .5, 1.)))):
        raise ValueError("Deployment requires consistent explicit nominal time_input/policy")
    if manifest.get("supervision_manifest_sha256") != C1_SUPERVISION_SHA256:
        raise ValueError("Deployment requires the verified C1 supervision manifest SHA256")
    if manifest.get("split_sha256") != RAWTIME_SPLIT_SHA256:
        raise ValueError("Deployment split SHA256 is not the fixed rawtime split")
    if manifest.get("model_config", {}).get("n_history") != 4:
        raise ValueError("Deployment input contract requires exactly four historical frames")
    if type(checkpoint.get("step")) is not int or checkpoint["step"] <= 0:
        raise ValueError("Deployment requires a trained checkpoint with positive step")
    root_value = args.get("data_root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise ValueError("Deployment lineage requires an absolute training data_root")
    root = Path(root_value).resolve(strict=True)
    run = _lineage_path(args.get("run_dir"), root, "run_dir")
    source = Path(source_path).resolve(strict=True)
    if source.parent != run:
        raise ValueError("Source checkpoint parent does not match its recorded run_dir")
    evidence = {}
    _verified_file(source, expected, evidence, "source_checkpoint")
    run_manifest = run / "manifest.json"
    final = _verified_file(run_manifest, file_sha256(run_manifest), evidence,
                           "completed_run_manifest", parse_json=True)
    # Embedded checkpoints retain status=running; only the final sidecar is
    # completed. Completion metadata is not evidence of the OS child exit code.
    final_step = final.get("step")
    if (final.get("status") != "completed" or type(final_step) is not int
            or final_step < checkpoint["step"] or type(args.get("steps")) is not int
            or final_step != args["steps"]):
        raise ValueError("Recorded run is not completed at its configured final step")
    stable_keys = ("arguments", "model_config", "time_input", "time_input_policy",
                   "split_sha256", "supervision_manifest_sha256", "load_report",
                   "initial_model_state_sha256", "git_sha", "train_rows_sha256",
                   "eval_rows_sha256", "data_counts", "bn_training", "loss_weights", "pid",
                   "pretrained_sha256")
    for name in stable_keys:
        if (name not in manifest or name not in final
                or _json_form(manifest[name]) != _json_form(final[name])):
            raise ValueError(f"Checkpoint/run manifest lineage mismatch: {name}")
    for name in ("initial_model_state_sha256", "train_rows_sha256", "eval_rows_sha256"):
        _required_sha(manifest[name], name)
    supervision = _lineage_path(args.get("supervision_root"), root, "supervision_root")
    contract = _verified_file(supervision / "supervision_manifest.json", C1_SUPERVISION_SHA256,
                              evidence, "supervision_manifest", parse_json=True)
    if (type(contract.get("schema_version")) is not int or contract["schema_version"] != 2
            or contract.get("geometry_edition") != "cache_meta_rear_wide_v2"
            or contract.get("canonical_calibration_sha256") != C1_CANONICAL_SHA256
            or contract.get("split_manifest_sha256") != RAWTIME_SPLIT_SHA256):
        raise ValueError("C1 geometry/split declaration does not match the serving contract")
    _verified_file(supervision / "calibration.npz", C1_CANONICAL_SHA256,
                   evidence, "canonical_calibration")
    split = _lineage_path(args.get("split_manifest"), root, "split_manifest")
    _verified_file(split, RAWTIME_SPLIT_SHA256, evidence, "split_manifest")
    load_report = manifest["load_report"]
    if not isinstance(load_report, dict):
        raise ValueError("Checkpoint initialization lineage must be a mapping")
    initializer = args.get("resume") or args.get("init")
    initializer_sha = load_report.get("common_checkpoint_sha256")
    if not initializer or not initializer_sha:
        raise ValueError("Explicit common-checkpoint initialization lineage is required")
    _verified_file(_lineage_path(initializer, root, "initializer"), initializer_sha,
                   evidence, "training_initializer")
    if args.get("pretrained"):
        _verified_file(_lineage_path(args["pretrained"], root, "pretrained"),
                       manifest.get("pretrained_sha256"), evidence, "pretrained_backbone")
    for record in (checkpoint, manifest):
        if "input_contract" in record and record["input_contract"] != input_contract():
            raise ValueError("Existing checkpoint input_contract conflicts with the verified contract")
    return {"mode": deployment_contract, "status": "input_contract_lineage_verified",
            "geometry_edition": "geometry_v2", "time_input": "nominal",
            "canonical_calibration_sha256": contract["canonical_calibration_sha256"],
            "supervision_manifest_sha256": manifest["supervision_manifest_sha256"],
            "split_sha256": manifest["split_sha256"], "run_dir": str(run),
            "training_git_sha": manifest["git_sha"],
            "selected_checkpoint_sha256": expected, "checkpoint_step": checkpoint["step"],
            "completed_run_step": final_step, "evidence": evidence,
            "contract_source_sha256": file_sha256(ROOT / "models/motiondrive_v2_input_contract.py"),
            "verified_scope": "C1 geometry and nominal model-input timing lineage only",
            "accuracy_latency_compliance_certified": False, "os_process_exit_verified": False}


def reverify_deployment_evidence(verification):
    """Reject a changed checkpoint, initializer, or sidecar before publication."""
    for role, item in verification["evidence"].items():
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Deployment lineage artifact changed before publication: {role}")
    if file_sha256(ROOT / "models/motiondrive_v2_input_contract.py") != verification["contract_source_sha256"]:
        raise ValueError("Input contract source changed before publication")


def file_sha256(path):
    with open(path, "rb") as stream:
        return stream_sha256(stream)


def stream_sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def validate_complete_config(saved):
    """Require complete current config or the one explicit legacy-control schema.

    Historical P4/P7 manifests predate serialized temporal fields.  Their exact
    old keyset migrates only to the fixed control contract; partial omissions or
    an implicit wide contract remain invalid.
    """
    if not isinstance(saved, dict):
        raise ValueError("manifest.model_config must be a complete dictionary")
    required = {field.name for field in dataclasses.fields(MotionDriveV2Config)}
    temporal = {"history_contract", "history_frame_offsets", "nominal_history_seconds"}
    if set(saved) == required - temporal:
        saved = {**saved, "history_contract": "control",
                 "history_frame_offsets": [1, 2, 5, 10],
                 "nominal_history_seconds": [.1, .2, .5, 1.]}
    missing, extra = required - saved.keys(), saved.keys() - required
    if missing or extra:
        raise ValueError(f"Incomplete/unknown model_config: missing={sorted(missing)}, "
                         f"unknown={sorted(extra)}; explicit reviewed migration required")
    for name in ("goal_on", "state_on"):
        if type(saved[name]) is not bool:
            raise ValueError(f"model_config.{name} must be an explicit boolean")
    for name in ("channels", "n_history", "correlation_channels", "correlation_radius",
                 "scene_attention_channels", "scene_chunk_size", "planner_layers", "planner_heads"):
        value = saved[name]
        if type(value) is not int or value < (0 if name == "correlation_radius" else 1):
            raise ValueError(f"Invalid integer model_config.{name}")
    for name in ("grid_size", "motion_grid"):
        value = saved[name]
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(type(x) is not int or x <= 0 for x in value)):
            raise ValueError(f"model_config.{name} must contain two positive integers")
    for name in ("x_range", "y_range", "heights", "plan_output_scale"):
        value = saved[name]
        if (not isinstance(value, (list, tuple)) or not value
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
            raise ValueError(f"Invalid finite numeric model_config.{name}")
        if name != "heights" and len(value) != 2:
            raise ValueError(f"model_config.{name} must have two entries")
        if name.endswith("_range") and value[0] >= value[1]:
            raise ValueError(f"model_config.{name} must be increasing")
    if saved["backbone_arch"] not in ("resnet34", "resnet50"):
        raise ValueError("Unsupported explicit backbone_arch")
    # Dataclass validation also checks head divisibility, positive scales/mode.
    return MotionDriveV2Config(**saved)


def prepare_bundle(checkpoint, *, source_path, source_sha256, source_bytes,
                   deployment_contract=None, expected_checkpoint_sha256=None):
    """Validate a trainer payload, then keep weights and JSON-only provenance."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a trainer checkpoint dictionary")
    manifest = checkpoint.get("manifest")
    if not isinstance(manifest, dict) or "model_config" not in manifest:
        raise ValueError("Required manifest.model_config is missing; no default fallback")
    config = validate_complete_config(manifest["model_config"])
    if deployment_contract is None and expected_checkpoint_sha256 is not None:
        raise ValueError("expected_checkpoint_sha256 requires an explicit deployment contract")
    verification = None
    if deployment_contract is not None:
        verification = verify_deployment_lineage(
            checkpoint, source_path=source_path, source_sha256=source_sha256,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            deployment_contract=deployment_contract)
        if source_bytes != verification["evidence"]["source_checkpoint"]["bytes"]:
            raise ValueError("Source checkpoint byte count does not match the actual file")
    step = checkpoint.get("step")
    if type(step) is not int or step < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    split_sha = manifest.get("split_sha256")
    if not isinstance(split_sha, str) or re.fullmatch(r"[0-9a-fA-F]{64}", split_sha) is None:
        raise ValueError("Required split_sha256 lineage is missing/invalid")
    try:
        # Trainer manifests are JSON records. This strips e.g. TorchVersion's
        # str subclass and disallows tensors/custom pickles in deployment metadata.
        clean_manifest = json.loads(json.dumps(manifest, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Manifest must be finite JSON-only metadata") from exc
    # Canonical explicit full settings; tuple/list representation is immaterial.
    clean_manifest["model_config"] = dataclasses.asdict(config)
    state = checkpoint.get("model")
    if (not isinstance(state, dict) or not state
            or any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
                   for k, v in state.items())):
        raise ValueError("Checkpoint model must be a nonempty tensor state dictionary")
    if verification is not None and any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Deployment checkpoint contains nonfinite model tensors")
    # Strict validation catches missing/unexpected keys and shape mismatches.
    # CPU construction only; no CUDA, optimizer, export trace, or model forward.
    with torch.random.fork_rng(devices=[]):
        validation_model = MotionDriveV2(config)
        validation_model.load_state_dict(state, strict=True)
    del validation_model
    # Clones remove view/backing-storage aliases; preserve every tensor's dtype.
    cpu_state = {name: value.detach().to(device="cpu").contiguous().clone()
                 for name, value in state.items()}
    source = {"path": str(source_path), "sha256": source_sha256,
              "bytes": int(source_bytes), "step": step,
              "split_sha256": split_sha,
              "training_git_sha": clean_manifest.get("git_sha")}
    bundle = {"format": "motiondrive_v2_inference", "format_version": 1,
              "model": cpu_state, "manifest": clean_manifest, "step": step,
              "source_checkpoint": source,
              "inference_export": {"script_sha256": file_sha256(__file__),
                                   "weights_device": "cpu", "weights_dtype_preserved": True,
                                   "optimizer_and_rng_excluded": True,
                                   "complete_config_required": True}}
    if "epoch" in checkpoint:
        epoch = checkpoint["epoch"]
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Checkpoint epoch must be a nonnegative integer")
        bundle["epoch"] = source["epoch"] = epoch
    if verification is not None:
        bundle["input_contract"] = input_contract(config.history_contract)
        bundle["deployment_provenance"] = verification
    return bundle


def export_checkpoint(checkpoint_path, output_path, *, deployment_contract=None,
                      expected_checkpoint_sha256=None):
    """Publish atomically without replacing any existing target, even in a race.

    The output parent must already exist. A same-directory temporary file is
    linked to the final name only after serialization/fsync, then removed.
    """
    source = Path(checkpoint_path).expanduser().resolve(strict=True)
    target = Path(output_path).expanduser().absolute()
    if not source.is_file():
        raise ValueError("--checkpoint must be an existing trusted local file")
    if target.resolve(strict=False) == source:
        raise ValueError("Input checkpoint overwrite is forbidden")
    if os.path.lexists(target):
        raise FileExistsError(f"Output already exists: {target}")
    if not target.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    with source.open("rb") as stream:
        before = os.fstat(stream.fileno())
        source_sha = stream_sha256(stream)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source checkpoint changed while being read; use a stable checkpoint")
    bundle = prepare_bundle(checkpoint, source_path=source,
                            source_sha256=source_sha, source_bytes=before.st_size,
                            deployment_contract=deployment_contract,
                            expected_checkpoint_sha256=expected_checkpoint_sha256)
    del checkpoint
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", prefix=f".{target.name}.",
                                         suffix=".tmp", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(bundle, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if deployment_contract is not None:
            reverify_deployment_evidence(bundle["deployment_provenance"])
        # Unlike rename/replace, link NEVER overwrites a target created meanwhile.
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    result = {"output": str(target), "output_sha256": file_sha256(target),
            "output_bytes": target.stat().st_size, "source_checkpoint": bundle["source_checkpoint"],
            "model_config": bundle["manifest"]["model_config"],
            "tensor_count": len(bundle["model"]), "optimizer_and_rng_excluded": True}
    if deployment_contract is not None:
        result["input_contract"] = bundle["input_contract"]
        result["deployment_provenance"] = bundle["deployment_provenance"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Trusted user-owned local training checkpoint; loads Python pickle")
    parser.add_argument("--out", required=True, help="New bundle path in an existing directory")
    parser.add_argument("--deployment-contract", choices=[DEPLOYMENT_CONTRACT], default=None,
                        help="Optional fail-closed serving-input lineage verification; generic export is unlabelled")
    parser.add_argument("--expected-checkpoint-sha256", default=None,
                        help="Explicitly selected checkpoint SHA256; required in deployment-contract mode")
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.out,
                                      deployment_contract=args.deployment_contract,
                                      expected_checkpoint_sha256=args.expected_checkpoint_sha256), indent=2))


if __name__ == "__main__":
    main()
