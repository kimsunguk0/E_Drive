#!/usr/bin/env python3
"""Describe P4 goal conditioning on preregistered train8 inputs.

This is a post-training counterfactual diagnostic, not an accuracy evaluation,
checkpoint selector, compliance certificate, or submission.  It runs the full
model for normal, normalized-zero, and cyclically shuffled images with four
goal choices per clip.  Goal-response magnitudes are descriptive: zero/nonzero
effects have no pass threshold and the controls are not realistic driving data.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback
from typing import Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models import motiondrive_v2_inputs as adapter
from models.motiondrive_v2_serving import load_deployment_bundle, validate_clip_inputs
from scripts.audit_motiondrive_v2 import output_checks
from scripts.audit_motiondrive_v2_deploy_inputs import (
    CALIBRATION_SHA256, FIXTURE_MANIFEST_SHA256, REFERENCE_SHA256,
    file_sha, make_reference_inputs, tree_sha, validate_identity,
    verify_raw_files,
)
from scripts.export_motiondrive_v2_inference import C1_SUPERVISION_SHA256
from scripts.motiondrive_v2_training import tensor_state_sha256
from scripts.smoke_motiondrive_v2_deployment import gpu_snapshot
from scripts.smoke_motiondrive_v2_p4_deployment import (
    read_pinned_bundle, validate_p4_bundle, validate_source_manifest,
)


SEED = 0
CONDITIONS = ("normal", "normalized_zero", "cyclic_shuffle")
GOAL_ROLES = ("own", "donor_1", "donor_2", "zero")
TRACKED_OUTPUTS = (
    "scene_features", "occ_logits", "lane_logits", "plan_abs",
    "motion_features", "history_hat", "state_hat",
)
GOAL_FREE_OUTPUTS = ("motion_features", "history_hat", "state_hat")
GOAL_RESPONSE_OUTPUTS = ("scene_features", "occ_logits", "lane_logits", "plan_abs")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def condition_plan(n_clips=8):
    """The 96 unique requested conditions; the final A repeat is extra."""
    require(type(n_clips) is int and n_clips == 8, "Goal response is train8-only")
    return [(condition, receiver, role) for condition in CONDITIONS
            for receiver in range(n_clips) for role in GOAL_ROLES]


def select_goal_donors(goals: torch.Tensor, clip_ids):
    """Pick two other observed goals by a deterministic distance/tie-break rule.

    Score = |norm(candidate)-norm(receiver)| + |candidate_x-receiver_x|.
    Ties then use norm difference, forward-x difference, clip id and index.
    Each donor goal was observed in a real train fixture, but that does not make
    it road-feasible for the receiver image.
    """
    clip_ids = list(map(str, clip_ids))
    require(isinstance(goals, torch.Tensor) and goals.shape == (8, 2)
            and goals.device.type == "cpu" and goals.dtype == torch.float32
            and torch.isfinite(goals).all(), "Goals must be finite CPU FP32 [8,2]")
    require(len(clip_ids) == 8 and len(set(clip_ids)) == 8,
            "Eight distinct clip ids are required")
    values = goals.double()
    norms = torch.linalg.vector_norm(values, dim=1)
    selections = []
    for receiver in range(8):
        candidates = []
        for donor in range(8):
            if donor == receiver:
                continue
            norm_delta = float(abs(norms[donor] - norms[receiver]))
            forward_delta = float(abs(values[donor, 0] - values[receiver, 0]))
            candidates.append(((norm_delta + forward_delta, norm_delta,
                                forward_delta, clip_ids[donor], donor), donor,
                               norm_delta, forward_delta))
        candidates.sort(key=lambda row: row[0])
        chosen = candidates[:2]
        selections.append({
            "receiver_index": receiver,
            "receiver_clip_id": clip_ids[receiver],
            "receiver_goal_xy": goals[receiver].tolist(),
            "donors": [{"rank": rank + 1, "index": row[1],
                        "clip_id": clip_ids[row[1]],
                        "goal_xy": goals[row[1]].tolist(),
                        "norm_delta_m": row[2], "forward_x_delta_m": row[3],
                        "combined_score_m": row[2] + row[3]}
                       for rank, row in enumerate(chosen)],
            "receiver_road_feasibility_guaranteed": False,
        })
    return selections


def goal_variants(goals, selection, receiver):
    require(selection["receiver_index"] == receiver and len(selection["donors"]) == 2,
            "Goal donor selection does not match receiver")
    return {
        "own": goals[receiver:receiver + 1].clone(),
        "donor_1": goals[selection["donors"][0]["index"]:
                         selection["donors"][0]["index"] + 1].clone(),
        "donor_2": goals[selection["donors"][1]["index"]:
                         selection["donors"][1]["index"] + 1].clone(),
        "zero": torch.zeros_like(goals[receiver:receiver + 1]),
    }


def make_condition_inputs(all_inputs, receiver, condition, goal):
    """Change only the declared images and goal; never mutate source tensors."""
    require(condition in CONDITIONS and len(all_inputs) == 8
            and type(receiver) is int and 0 <= receiver < 8,
            "Invalid train8 receiver/condition")
    base = all_inputs[receiver]
    require(set(base) == set(adapter.INPUT_KEYS), "Exactly six model inputs required")
    require(isinstance(goal, torch.Tensor) and goal.shape == (1, 2)
            and goal.device.type == "cpu" and goal.dtype == torch.float32
            and torch.isfinite(goal).all(), "Goal variant must be finite CPU FP32 [1,2]")
    result = dict(base)
    image_source = receiver
    if condition == "normalized_zero":
        result["images"] = torch.zeros_like(base["images"])
        result["history_images"] = torch.zeros_like(base["history_images"])
        image_source = None
    elif condition == "cyclic_shuffle":
        image_source = (receiver + 1) % 8
        result["images"] = all_inputs[image_source]["images"]
        result["history_images"] = all_inputs[image_source]["history_images"]
    result["goal_xy"] = goal.clone()
    return result, {"receiver_index": receiver, "image_source_index": image_source,
                    "only_goal_changed_within_condition": True,
                    "image_geometry_pairing_realistic": condition == "normal",
                    "negative_control": condition != "normal"}


def validate_tracked_outputs(outputs):
    checks = output_checks(outputs, 1)
    require(all(checks.values()), "Full model output contract failed")
    expected = {
        "scene_features": (1, 3072, 128), "occ_logits": (1, 1, 64, 48),
        "lane_logits": (1, 1, 64, 48), "plan_abs": (1, 6, 2),
        "motion_features": (1, 192, 128), "history_hat": (1, 4, 4),
        "state_hat": (1, 6),
    }
    result = {}
    for name, shape in expected.items():
        value = outputs.get(name)
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape
                and torch.isfinite(value).all(), f"Invalid full output: {name}")
        result[name] = value.detach().to(device="cpu").contiguous()
    return result


def full_forward(model, cpu_inputs, *, device="cuda:0", precision="bf16"):
    """One untimed complete forward; labels and partial-feature APIs are absent."""
    validate_clip_inputs(cpu_inputs, adapter.input_contract())
    require(device == "cuda:0" and precision == "bf16",
            "Goal diagnostic is fixed to cuda:0/BF16")
    tensors = {name: cpu_inputs[name].to(device=device, non_blocking=False)
               for name in adapter.INPUT_KEYS}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(**tensors)
    return validate_tracked_outputs(outputs)


def tensor_delta(actual, baseline, name):
    require(isinstance(actual, torch.Tensor) and isinstance(baseline, torch.Tensor)
            and actual.shape == baseline.shape and actual.device.type == "cpu"
            and baseline.device.type == "cpu" and torch.isfinite(actual).all()
            and torch.isfinite(baseline).all(), f"Invalid tensors for delta: {name}")
    difference = actual.double() - baseline.double()
    absolute = difference.abs()
    report = {"bitwise_equal": bool(torch.equal(actual, baseline)),
              "mean_abs": float(absolute.mean()),
              "rms": float(torch.sqrt(difference.square().mean())),
              "max_abs": float(absolute.max())}
    if name == "scene_features":
        per_cell = torch.linalg.vector_norm(difference, dim=-1)
        report["cell_l2_mean"] = float(per_cell.mean())
        report["cell_l2_max"] = float(per_cell.max())
    elif name == "plan_abs":
        waypoint = torch.linalg.vector_norm(difference, dim=-1)[0]
        report["waypoint_l2"] = waypoint.tolist()
        report["waypoint_l2_mean"] = float(waypoint.mean())
        report["waypoint_l2_max"] = float(waypoint.max())
    return report


def analyze_goal_variant(outputs, baseline):
    """Goal-free equality is a sanity gate; goal-response magnitude is not."""
    result = {name: tensor_delta(outputs[name], baseline[name], name)
              for name in TRACKED_OUTPUTS}
    invariant = {}
    for name in GOAL_FREE_OUTPUTS:
        invariant[name] = result[name]["bitwise_equal"]
        require(invariant[name], f"Goal unexpectedly changed goal-free output: {name}")
    return {"goal_response": {name: result[name] for name in GOAL_RESPONSE_OUTPUTS},
            "goal_free_invariance": {name: result[name] for name in GOAL_FREE_OUTPUTS},
            "goal_free_all_bitwise": all(invariant.values())}


def validate_repeat(first, repeated):
    results = {name: tensor_delta(repeated[name], first[name], name)
               for name in TRACKED_OUTPUTS}
    require(all(row["bitwise_equal"] for row in results.values()),
            "A/B/A repeat changed a tracked output")
    return {"all_tracked_outputs_bitwise": True, "outputs": results}


def summarize_records(records):
    """Descriptive summaries only; deliberately no sensitivity pass threshold."""
    result = {}
    for condition in CONDITIONS:
        rows = [row for row in records if row["condition"] == condition
                and row["goal_role"] != "own"]
        by_output = {}
        for name in GOAL_RESPONSE_OUTPUTS:
            field = "waypoint_l2_mean" if name == "plan_abs" else "rms"
            values = np.asarray([row["analysis"]["goal_response"][name][field]
                                 for row in rows], np.float64)
            by_output[name] = {"field": field, "n": int(values.size),
                               "mean": float(values.mean()),
                               "median": float(np.median(values)),
                               "min": float(values.min()), "max": float(values.max())}
        result[condition] = by_output
    return result


def run(args):
    require(not torch.cuda.is_initialized(), "Start fresh; validate CPU artifacts before CUDA")
    require(args.device == "cuda:0" and args.precision == "bf16",
            "Goal diagnostic is fixed to cuda:0/BF16")
    paths = {"bundle": Path(args.bundle), "reference": Path(args.reference),
             "raw_manifest": Path(args.fixture_root) / "fixture_manifest.json",
             "calibration": Path(args.calibration),
             "geometry_contract": Path(args.geometry_contract),
             "source_manifest": Path(args.source_manifest)}
    pinned = {"bundle": args.expected_bundle_sha256, "reference": REFERENCE_SHA256,
              "raw_manifest": FIXTURE_MANIFEST_SHA256,
              "calibration": CALIBRATION_SHA256,
              "geometry_contract": C1_SUPERVISION_SHA256}
    artifacts_before = {name: file_sha(path) for name, path in paths.items()}
    require(all(artifacts_before[name] == expected for name, expected in pinned.items()),
            "Pinned P4 goal-diagnostic artifacts differ")
    bundle = read_pinned_bundle(paths["bundle"], args.expected_bundle_sha256)
    p4 = validate_p4_bundle(
        bundle, expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_run_manifest_sha256=args.expected_run_manifest_sha256)
    source_manifest = json.loads(paths["source_manifest"].read_text())
    source_before, validation_git = validate_source_manifest(source_manifest)
    reference = torch.load(paths["reference"], map_location="cpu", weights_only=True)
    reference_before = tree_sha(reference)
    raw_manifest = json.loads(paths["raw_manifest"].read_text())
    clips = validate_identity(reference, raw_manifest)
    require(len(clips) == 8 and all(row.get("source_split") == "train" for row in clips),
            "Only the preregistered train8 fixture is allowed")
    raw_before = verify_raw_files(args.fixture_root, clips)
    with np.load(paths["calibration"], allow_pickle=False) as archive:
        calibration = torch.from_numpy(archive["lidar2img"].copy())
    reference_inputs, adaptation = make_reference_inputs(reference["batch"], calibration)
    prepared, input_parity = [], []
    adapter.cv2.setNumThreads(1)
    for index, clip in enumerate(clips):
        item = adapter.prepare_clip_inputs(Path(args.fixture_root) / clip["clip_id"])
        expected = {name: value[index:index + 1] for name, value in reference_inputs.items()}
        comparison = adapter.compare_input_fixture(
            item, {"inputs": expected, "metadata": {"input_contract": adapter.input_contract()}},
            projection_atol=1e-4, pose_atol=1e-5)
        require(comparison["all_pass"], f"Train8 input parity failed: {clip['clip_id']}")
        prepared.append(item.inputs)
        input_parity.append({"clip_id": clip["clip_id"], **comparison})

    goals = torch.cat([item["goal_xy"] for item in prepared], dim=0)
    clip_ids = [row["clip_id"] for row in clips]
    selections = select_goal_donors(goals, clip_ids)
    variants = [goal_variants(goals, selections[index], index) for index in range(8)]
    plan = condition_plan()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    gpu_observations = [gpu_snapshot(require_idle=True)]
    model, contract = load_deployment_bundle(
        paths["bundle"], expected_bundle_sha256=args.expected_bundle_sha256,
        device=args.device)
    require(contract == adapter.input_contract(), "Bundle/adapter contract differs")
    state_before = tensor_state_sha256(model.state_dict())
    records, baselines, first_a = [], {}, None
    for condition, receiver, role in plan:
        inputs, manipulation = make_condition_inputs(
            prepared, receiver, condition, variants[receiver][role])
        outputs = full_forward(model, inputs, device=args.device, precision=args.precision)
        if first_a is None:
            require((condition, receiver, role) == ("normal", 0, "own"),
                    "A/B/A anchor ordering changed")
            first_a = outputs
        key = (condition, receiver)
        if role == "own":
            baselines[key] = outputs
        require(key in baselines, "Own-goal baseline must precede donor goals")
        analysis = analyze_goal_variant(outputs, baselines[key])
        records.append({"condition": condition, "receiver_index": receiver,
                        "receiver_clip_id": clip_ids[receiver], "goal_role": role,
                        "goal_xy": variants[receiver][role].tolist()[0],
                        "manipulation": manipulation, "analysis": analysis})
    repeat_inputs, _ = make_condition_inputs(prepared, 0, "normal", variants[0]["own"])
    repeated_a = full_forward(model, repeat_inputs, device=args.device, precision=args.precision)
    aba = validate_repeat(first_a, repeated_a)
    state_after = tensor_state_sha256(model.state_dict())
    require(state_before == state_after, "Model state changed during goal diagnostic")
    gpu_observations.append(gpu_snapshot())

    artifacts_after = {name: file_sha(path) for name, path in paths.items()}
    source_after = {name: file_sha(ROOT / name) for name in source_before}
    raw_after = verify_raw_files(args.fixture_root, clips)
    require(artifacts_before == artifacts_after and source_before == source_after
            and raw_before == raw_after and reference_before == tree_sha(reference),
            "Immutable input/source changed during goal diagnostic")
    return {
        "schema_version": 1, "status": "completed_descriptive_diagnostic",
        "purpose": "P4 train8 goal-response counterfactual; no accuracy/compliance decision",
        "actual_gpu_diagnostic_completed": True, "condition_count": len(plan),
        "full_forward_count": len(plan) + 1, "batch_size": 1,
        "not_accuracy_evaluation": True, "no_gt_metric_computed": True,
        "not_checkpoint_selection": True, "not_compliance_certificate": True,
        "no_goal_response_pass_threshold": True,
        "negative_controls_are_not_realistic_driving_inputs": True,
        "p4_bundle_receipt": p4,
        "source": {"training_git_sha": p4["training_source_git_sha"],
                   "validation_git_sha": validation_git,
                   "git_equality_required": False,
                   "script_sha256": file_sha(__file__),
                   "manifest": source_manifest,
                   "file_sha256_before": source_before,
                   "file_sha256_after": source_after},
        "seed": SEED, "precision": args.precision, "device": args.device,
        "input_contract": contract, "reference_adaptation": adaptation,
        "input_parity": input_parity,
        "goal_selection_policy": {
            "score": "abs(norm(candidate)-norm(receiver)) + abs(candidate_x-receiver_x)",
            "tie_break": "norm_delta, forward_x_delta, donor clip_id, donor index",
            "donor_source": "two other actually observed preregistered train8 goals",
            "receiver_road_feasibility_guaranteed": False,
            "zero_goal_is_a_negative_control_not_a_feasible_goal_claim": True},
        "goal_selections": selections, "conditions": list(CONDITIONS),
        "goal_roles": list(GOAL_ROLES), "records": records,
        "descriptive_summary": summarize_records(records),
        "aba_stateless": aba,
        "goal_free_outputs_required_bitwise_across_goals": list(GOAL_FREE_OUTPUTS),
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "gpu_observations": gpu_observations,
        "artifacts_sha256_before": artifacts_before,
        "artifacts_sha256_after": artifacts_after,
        "raw_files_before": raw_before, "raw_files_after": raw_after,
        "environment": {"python": platform.python_version(),
                        "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(args.device),
                        "tf32": False, "deterministic_algorithms": True},
        "interpretation_limits": [
            "A goal observed for a donor clip is not guaranteed road-feasible for the receiver.",
            "Normalized-zero and cyclic-shuffle break the natural image distribution and/or geometry pairing.",
            "Goal response does not establish accuracy, causality, competition success, or organizer approval.",
            "This diagnostic does not authorize a model or training change."],
    }


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("bundle", "expected-bundle-sha256", "expected-checkpoint-sha256",
                 "expected-run-manifest-sha256", "fixture-root", "reference",
                 "calibration", "geometry-contract", "source-manifest", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    output = Path(args.output).absolute()
    if os.path.lexists(output) or not output.parent.is_dir():
        raise FileExistsError("Goal-response report must be new in an existing directory")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        report = run(args)
    except Exception as error:
        report = {"status": "failed", "error_type": type(error).__name__,
                  "error": str(error), "traceback": traceback.format_exc(),
                  "actual_gpu_diagnostic_completed": False,
                  "not_accuracy_evaluation": True,
                  "not_compliance_certificate": True}
    report.update(argv=sys.argv if argv is None else [str(Path(__file__)), *argv],
                  started_at=started, ended_at=dt.datetime.now(dt.timezone.utc).isoformat())
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "sha256": file_sha(output)}), flush=True)
    return 0 if report["status"] == "completed_descriptive_diagnostic" else 1


if __name__ == "__main__":
    raise SystemExit(main())
