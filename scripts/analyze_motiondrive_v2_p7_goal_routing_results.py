#!/usr/bin/env python3
"""CPU-only analysis of the preregistered P7 C/G x two-base terminal reports.

This consumes saved LAST6000 tune predictions and a pinned ego-cache solely to
recover the raw metric goal omitted by the frozen evaluator records.  It never
runs a model, reads final validation, selects a threshold, or tunes a subgroup.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import analyze_motiondrive_v2_p6_stop_balance_results as p6
from scripts import run_motiondrive_v2_p7_goal_routing as p7
from scripts.analyze_motiondrive_v2_query_pair import paired_cluster_bootstrap

EXPECTED_N = 1998
EXPECTED_SCENES = 37
EXPECTED_SESSIONS = 11
EXPECTED_BUCKETS = {"steady": 99, "depart": 24, "nonstop": 1875}
SESSION_046 = "session_046_20260210-100504"
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 20260908
EXPECTED_EGO_CACHE_SHA256 = "d35a69bb229b2d4736aff7dddefd3522ddbd346e112283e2ce58ce3986a023bd"
EXPECTED_P7_SOURCE_MANIFEST_SHA256 = "880c3cc36ababe31c58376ab815a239e337f130fc9b373a041fea6ad4a8d1ea4"
EXPECTED_INITIAL_MODEL_STATE_SHA256 = {
    0: "fc51a847cc5532e8010454d2452ea8233e0d1e560053d2217f34580f464db1c7",
    1: "4f480070eac81176b0a3945484186ccf79063806644be75c4afc41e3bff423a4",
}
EXPECTED_NEW_STATE_KEYS = [
    "scene_encoder.cross_cell_goal_residual.destination_xy",
    "scene_encoder.cross_cell_goal_residual.source_xy",
    "scene_encoder.cross_cell_goal_residual.position_scale",
    "scene_encoder.cross_cell_goal_residual.sigma_m",
    "scene_encoder.cross_cell_goal_residual.norm.weight",
    "scene_encoder.cross_cell_goal_residual.norm.bias",
    "scene_encoder.cross_cell_goal_residual.query_image.weight",
    "scene_encoder.cross_cell_goal_residual.key_image.weight",
    "scene_encoder.cross_cell_goal_residual.value_image.weight",
    "scene_encoder.cross_cell_goal_residual.position.0.weight",
    "scene_encoder.cross_cell_goal_residual.position.0.bias",
    "scene_encoder.cross_cell_goal_residual.position.2.weight",
    "scene_encoder.cross_cell_goal_residual.output.weight",
]
P4_REFERENCE_SHA256 = dict(p6.P4_REFERENCE_SHA256)
GRID_BOUNDS_M = {"x": (-10., 70.), "y": (-32., 32.), "inclusive": True}
ARM_TO_MODE = {"control": ("control_zero_slot", "zero"),
               "goal": ("goal_real_slot", "real")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_p7_source_manifest(payload, expected_sha256):
    require(expected_sha256 == EXPECTED_P7_SOURCE_MANIFEST_SHA256
            and set(payload) == {"schema_version", "git_sha", "file_sha256"}
            and payload.get("schema_version") == 1
            and isinstance(payload.get("git_sha"), str) and len(payload["git_sha"]) == 40
            and all(value in "0123456789abcdef" for value in payload["git_sha"])
            and isinstance(payload.get("file_sha256"), dict)
            and set(payload["file_sha256"]) == p7.SOURCE_FILES
            and all(p6.valid_sha(value) for value in payload["file_sha256"].values()),
            "Externally pinned P7 source manifest is invalid")
    return {"sha256": expected_sha256, "git_sha": payload["git_sha"],
            "file_sha256": dict(payload["file_sha256"])}


def validate_terminal_manifest(manifest, *, base_seed, arm, pinned_source,
                               expected_official_d3):
    expected_arm, mode = ARM_TO_MODE[arm]
    require(manifest.get("status") == "completed" and manifest.get("step") == 6000
            and manifest.get("nonfinite_count") == 0,
            "P7 manifest must be a finite completed LAST6000 run")
    args = manifest.get("arguments", {})
    expected_args = {
        "phase": "joint", "steps": 6000, "seed": base_seed, "goal_on": 1,
        "state_on": 1, "cross_cell_goal_mode": mode, "batch": 16,
        "microbatch": 2, "eval_batch": 4, "eval_every": 6000,
        "save_every": 6000, "precision": "bf16", "time_input": "nominal",
        "bn_policy": "fixed", "eval_split": "tune", "eval_stride": 5,
        "max_eval_samples": 0, "train_stride": 1, "max_train_samples": 0,
        "lr": 1e-4, "backbone_lr": 1e-5, "weight_decay": .01, "warmup": 200,
        "grad_clip": 5., "alpha_occ": .2, "alpha_lane": .2, "alpha_motion": .2,
        "uncertainty": 1, "workers": 4, "gpu": 0,
        "cuda_memory_limit_mib": 12000, "cuda_min_free_mib": 8192,
    }
    require(all(args.get(key) == value for key, value in expected_args.items())
            and args.get("train_scenes") is None and args.get("eval_scenes") is None
            and args.get("eval_only") is False and args.get("resume") is None,
            "P7 manifest runtime/data recipe mismatch")
    config = manifest.get("model_config", {})
    require(config.get("cross_cell_goal_mode") == mode
            and config.get("goal_on") is True and config.get("state_on") is True
            and config.get("backbone_arch") == "resnet50"
            and config.get("motion_input_mode") == "low_feature"
            and list(config.get("plan_output_scale", ())) == [10., 5.],
            "P7 terminal model configuration mismatch")
    experiment = manifest.get("experimental_protocol", {})
    expected_branch = {
        "mode": mode, "sigma_m": [10., 32. / 3.],
        "sigma_selection": "fixed_geometry_not_tuned", "pool_size": 4,
        "source_cells": 192, "destination_cells": 3072, "attention_dim": 32,
        "cosine_scale": 8., "attention_precision": "fp32_autocast_disabled",
        "goal_enters_distance_score_only": True,
        "new_value_projection_adds_goal_or_position": False, "output_bias": False,
        "output_weight_zero_initialized": True,
    }
    expected_experiment_keys = {
        "schema_version", "name", "arm", "last_only_final_eval",
        "expected_initial_model_state_sha256", "expected_p0_model_state_sha256",
        "expected_optimizer_groups", "expected_missing_state_keys", "train_data", "tune_data",
        "source", "fresh_optimizer_step_zero", "all_model_parameters_joint_trainable",
        "existing_goal_path_on", "planner_signature_unchanged", "branch",
    }
    require(set(experiment) == expected_experiment_keys
            and experiment.get("schema_version") == 1
            and experiment.get("name") == "p7_cross_cell_goal_routing"
            and experiment.get("arm") == expected_arm
            and experiment.get("last_only_final_eval") is True
            and experiment.get("fresh_optimizer_step_zero") is True
            and experiment.get("all_model_parameters_joint_trainable") is True
            and experiment.get("existing_goal_path_on") is True
            and experiment.get("planner_signature_unchanged") is True
            and experiment.get("branch") == expected_branch,
            "P7 experimental arm/precision protocol mismatch")
    source = experiment.get("source", {})
    launch_git = manifest.get("git_sha")
    require(set(source) == {"path", "sha256", "git_sha", "file_sha256"}
            and source.get("sha256") == pinned_source["sha256"]
            and source.get("git_sha") == pinned_source["git_sha"]
            and source.get("file_sha256") == pinned_source["file_sha256"]
            and isinstance(launch_git, str) and len(launch_git) == 40
            and all(value in "0123456789abcdef" for value in launch_git),
            "P7 source manifest identity mismatch")
    require(manifest.get("loss_weights") == {
        "plan": 1., "occupancy": .2, "lane": .2, "motion": .2, "uncertainty": True},
        "P7 must retain the original unweighted joint loss recipe")
    require(manifest.get("data_counts") == {"train": 54810, "eval": EXPECTED_N}
            and manifest.get("split_sha256") == p7.EXPECTED_SPLIT_SHA256
            and manifest.get("supervision_manifest_sha256") == p7.EXPECTED_SUPERVISION_SHA256
            and manifest.get("train_rows_sha256") == p7.EXPECTED_TRAIN_ROWS_SHA256
            and manifest.get("eval_rows_sha256") == p7.EXPECTED_TUNE_ROWS_SHA256,
            "P7 train/tune inventory mismatch")
    require(experiment.get("expected_optimizer_groups") == [
                {"name": "backbone", "base_lr": 1e-5},
                {"name": "head", "base_lr": 1e-4}]
            and experiment.get("expected_missing_state_keys") == EXPECTED_NEW_STATE_KEYS
            and experiment.get("train_data") == {
                "rows": 54810, "rows_sha256": p7.EXPECTED_TRAIN_ROWS_SHA256}
            and experiment.get("tune_data") == {
                "rows": EXPECTED_N, "rows_sha256": p7.EXPECTED_TUNE_ROWS_SHA256},
            "P7 optimizer/new-state/train-tune protocol mismatch")
    pinned_p0 = p7.P0_ARTIFACTS[base_seed]
    require(manifest.get("load_report", {}).get("common_checkpoint_sha256")
                == pinned_p0["checkpoint_sha256"]
            and experiment.get("expected_p0_model_state_sha256")
                == pinned_p0["model_state_sha256"], "P7 own-base P0 lineage mismatch")
    expected_initial = EXPECTED_INITIAL_MODEL_STATE_SHA256[base_seed]
    require(experiment.get("expected_initial_model_state_sha256") == expected_initial
            and manifest.get("initial_model_state_sha256") == expected_initial,
            "P7 preregistered initial model-state identity mismatch")
    require(type(expected_official_d3) is float
            and type(manifest.get("best_metric")) in (int, float)
            and float(manifest["best_metric"]) == expected_official_d3,
            "P7 manifest/report terminal metric binding mismatch")
    return {"source_manifest_sha256": pinned_source["sha256"],
            "production_source_git_sha": source.get("git_sha"),
            "launch_record_git_sha": launch_git,
            "source_and_launch_git_equal_informational": source.get("git_sha") == launch_git,
            "initial_model_state_sha256": expected_initial, "base_seed": base_seed,
            "arm": expected_arm, "mode": mode}


def validate_p7_report(payload, *, base_seed, arm):
    require(set(payload) == {"report", "records"}, "P7 final_eval wrapper schema mismatch")
    report, source = payload["report"], payload["records"]
    require(isinstance(report, dict) and isinstance(source, list)
            and len(source) == EXPECTED_N, "P7 final_eval rows missing")
    require(report.get("kind") == "eval" and report.get("step") == 6000
            and report.get("time_input") == "nominal" and report.get("n") == EXPECTED_N
            and report.get("n_sessions") == EXPECTED_SESSIONS
            and report.get("selection_definition") == "official_d3",
            "P7 terminal report contract mismatch")
    rows, predictions, ground_truth, stored = [], [], [], []
    for item in source:
        require(isinstance(item, dict), "Malformed P7 detailed record")
        identity = p6._identity(item)
        pred = p6._array(item.get("pred_abs_xy"), (6, 2), "P7 pred_abs_xy")
        gt = p6._array(item.get("gt_abs_xy"), (6, 2), "P7 gt_abs_xy")
        d3 = item.get("d3")
        require(type(d3) in (int, float) and np.isfinite(float(d3)) and float(d3) >= 0,
                "P7 stored D3 invalid")
        bucket = p6._bucket_from_saved_gt(item, gt)
        require(item.get("bucket") == bucket and item.get("stop_bucket") == bucket,
                "P7 saved bucket differs from GT")
        maximum = p6._max_gt_displacement(gt)
        require(type(item.get("max_gt_displacement_m")) in (int, float)
                and float(item["max_gt_displacement_m"]) == maximum,
                "P7 max GT displacement mismatch")
        state = p6._array(item.get("pred_state"), (6,), "P7 pred_state")
        gt_state = p6._array(item.get("gt_state"), (6,), "P7 gt_state")
        valid = item.get("gt_state_valid")
        require(isinstance(valid, list) and len(valid) == 6
                and all(type(value) is bool for value in valid)
                and valid[5] == item.get("stop_valid"), "P7 GT state mask mismatch")
        if item["stop_valid"]:
            require(float(gt_state[5]) == float(item.get("stop_target")),
                    "P7 stop target differs from GT state")
        rows.append({"identity": identity, "pred": pred, "gt": gt, "d3": float(d3),
                     "bucket": bucket, "gt_state": gt_state,
                     "gt_state_valid": tuple(valid), "stop_valid": item["stop_valid"],
                     "stop_target": item.get("stop_target"),
                     "stop_logit": float(state[5])})
        predictions.append(pred); ground_truth.append(gt); stored.append(float(d3))
    require(len({item["identity"] for item in rows}) == EXPECTED_N
            and len({item["identity"][1] for item in rows}) == EXPECTED_SCENES
            and len({item["identity"][2] for item in rows}) == EXPECTED_SESSIONS,
            "P7 row/scene/session inventory mismatch")
    counts = {name: sum(item["bucket"] == name for item in rows) for name in EXPECTED_BUCKETS}
    require(counts == EXPECTED_BUCKETS, f"P7 fixed tune buckets differ: {counts}")
    _, audit = p6._d3_audit(predictions, ground_truth, stored)
    official = report.get("official_d3")
    require(type(official) in (int, float) and float(official) == float(np.mean(stored)),
            "P7 official D3 differs from saved row mean")
    require(type(report.get("selection_metric")) in (int, float)
            and float(report["selection_metric"]) == float(official),
            "P7 terminal selection metric differs from official D3")
    audit.update(stored_official_d3=float(official), base_seed=base_seed, arm=arm)
    return rows, audit


def load_goals(raw, references, expected_sha256):
    require(expected_sha256 == EXPECTED_EGO_CACHE_SHA256,
            "P7 goal cache must use the preregistered ego-cache SHA")
    with np.load(io.BytesIO(raw), allow_pickle=False) as source:
        require(set(("scenarios", "scen_idx", "frame", "goal", "fut")) <= set(source.files),
                "Ego cache fields incomplete")
        scenarios = source["scenarios"].astype(str)
        scene_index = np.asarray(source["scen_idx"])
        frames = np.asarray(source["frame"])
        goals = np.asarray(source["goal"], dtype=np.float32)
        futures = np.asarray(source["fut"], dtype=np.float32)
    require(goals.ndim == 2 and goals.shape[1] == 2 and futures.shape == (len(goals), 6, 2)
            and scene_index.shape == frames.shape == (len(goals),)
            and np.isfinite(goals).all() and np.isfinite(futures).all(),
            "Ego cache goal/future arrays malformed")
    result = []
    for record in references:
        row, scenario, _session, frame = record["identity"]
        require(0 <= row < len(goals) and 0 <= int(scene_index[row]) < len(scenarios)
                and scenarios[int(scene_index[row])] == scenario and int(frames[row]) == frame
                and np.array_equal(futures[row], record["gt"]),
                "Ego-cache row/scenario/frame/future GT join mismatch")
        result.append(goals[row])
    result = np.stack(result).astype(np.float32, copy=False)
    inside = ((result[:, 0] >= GRID_BOUNDS_M["x"][0])
              & (result[:, 0] <= GRID_BOUNDS_M["x"][1])
              & (result[:, 1] >= GRID_BOUNDS_M["y"][0])
              & (result[:, 1] <= GRID_BOUNDS_M["y"][1]))
    return result, inside


def require_same_rows(reference, collections):
    p6.require_same_labels(reference, collections)


def _metric(values, mask):
    selected = np.asarray(values, np.float64)[np.asarray(mask, bool)]
    require(selected.size and np.isfinite(selected).all(), "Metric group empty/nonfinite")
    return {"n": int(selected.size), "official_d3": float(selected.mean())}


def _optional_metric(values, mask):
    selected = np.asarray(values, np.float64)[np.asarray(mask, bool)]
    require(np.isfinite(selected).all(), "Metric group nonfinite")
    return {"n": int(selected.size),
            "official_d3": float(selected.mean()) if selected.size else None}


def summarize(rows, in_grid):
    values = np.asarray([item["d3"] for item in rows], np.float64)
    buckets = np.asarray([item["bucket"] for item in rows])
    sessions = np.asarray([item["identity"][2] for item in rows])
    return {"all": _metric(values, np.ones(len(rows), bool)),
            "buckets": {name: _metric(values, buckets == name) for name in EXPECTED_BUCKETS},
            "excluding_session_046": _metric(values, sessions != SESSION_046),
            "goal_grid": {"in_grid": _optional_metric(values, in_grid),
                          "out_of_grid": _optional_metric(values, ~in_grid)}}


def compare(left, right, in_grid):
    left_values = np.asarray([item["d3"] for item in left], np.float64)
    right_values = np.asarray([item["d3"] for item in right], np.float64)
    sessions = np.asarray([item["identity"][2] for item in left])
    buckets = np.asarray([item["bucket"] for item in left])
    delta = left_values - right_values
    all_result = paired_cluster_bootstrap(delta, sessions, repeats=BOOTSTRAP_REPEATS,
                                          seed=BOOTSTRAP_SEED)
    return {"all": all_result,
            "buckets": {name: {"n": int((buckets == name).sum()),
                                "delta_d3": float(delta[buckets == name].mean())}
                        for name in EXPECTED_BUCKETS},
            "excluding_session_046": {
                "n": int((sessions != SESSION_046).sum()),
                "delta_d3": float(delta[sessions != SESSION_046].mean())},
            "goal_grid": {
                "in_grid": {"n": int(in_grid.sum()),
                            "delta_d3": float(delta[in_grid].mean()) if in_grid.any() else None},
                "out_of_grid": {"n": int((~in_grid).sum()),
                                "delta_d3": float(delta[~in_grid].mean())
                                if (~in_grid).any() else None}}}


def analyze(reports, references, in_grid):
    all_collections = {f"base{base}_{arm}": rows
                       for (base, arm), rows in reports.items()}
    all_collections["base1_original"] = references[1]
    require_same_rows(references[0], all_collections)
    variants, comparisons = {}, {}
    for base in (0, 1):
        control, goal, original = (reports[(base, "control")], reports[(base, "goal")],
                                   references[base])
        variants[f"base{base}"] = {"control": summarize(control, in_grid),
                                    "goal": summarize(goal, in_grid),
                                    "original_p4": summarize(original, in_grid)}
        comparisons[f"base{base}"] = {
            "goal_minus_control": compare(goal, control, in_grid),
            "goal_minus_original_p4": compare(goal, original, in_grid)}
    arrays = {name: [np.asarray([item["d3"] for item in rows], np.float64) for rows in pair]
              for name, pair in {
                  "goal": [reports[(0, "goal")], reports[(1, "goal")]],
                  "control": [reports[(0, "control")], reports[(1, "control")]],
                  "original_p4": [references[0], references[1]],
              }.items()}
    means = {name: [float(value.mean()) for value in values] for name, values in arrays.items()}
    mean_gc = .5 * ((arrays["goal"][0] - arrays["control"][0])
                    + (arrays["goal"][1] - arrays["control"][1]))
    mean_go = .5 * ((arrays["goal"][0] - arrays["original_p4"][0])
                    + (arrays["goal"][1] - arrays["original_p4"][1]))
    sessions = np.asarray([item["identity"][2] for item in references[0]])
    shared_gc = paired_cluster_bootstrap(mean_gc, sessions, repeats=BOOTSTRAP_REPEATS,
                                         seed=BOOTSTRAP_SEED)
    shared_go = paired_cluster_bootstrap(mean_go, sessions, repeats=BOOTSTRAP_REPEATS,
                                         seed=BOOTSTRAP_SEED)
    per_base_gc = [comparisons[f"base{base}"]["goal_minus_control"]["all"]["delta"]
                   for base in (0, 1)]
    per_base_go = [comparisons[f"base{base}"]["goal_minus_original_p4"]["all"]["delta"]
                   for base in (0, 1)]
    mean_goal = float(np.mean(means["goal"]))
    mean_gc_value = float(np.mean(means["goal"]) - np.mean(means["control"]))
    checks = {
        "both_bases_goal_better_than_control": all(value < 0 for value in per_base_gc),
        "both_bases_goal_better_than_original_p4": all(value < 0 for value in per_base_go),
        "mean_goal_minus_control_at_most_minus_0p015": mean_gc_value <= -.015,
        "shared_session_goal_minus_control_ci_upper_below_zero": shared_gc["ci95"][1] < 0,
        "mean_goal_d3_at_most_0p34": mean_goal <= .34,
    }
    return {"variants": variants, "comparisons": comparisons,
            "two_base_description": {
                name: {"values": values, "mean": float(np.mean(values)),
                       "min": float(np.min(values)), "max": float(np.max(values)), "n_bases": 2}
                for name, values in means.items()},
            "shared_session_bootstrap": {
                "goal_minus_control": shared_gc, "goal_minus_original_p4": shared_go,
                "same_resamples_for_both_comparisons": True,
                "construction": "rowwise 0.5*((base0 delta)+(base1 delta)); same 11 sessions",
                "repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED,
                "frame_iid": False, "seed_bootstrap": False},
            "shared_descriptive_groups": {
                "steady": {"n": EXPECTED_BUCKETS["steady"],
                           "goal_minus_control_d3": float(mean_gc[np.asarray(
                               [x["bucket"] == "steady" for x in references[0]])].mean()),
                           "goal_minus_original_p4_d3": float(mean_go[np.asarray(
                               [x["bucket"] == "steady" for x in references[0]])].mean())},
                "depart": {"n": EXPECTED_BUCKETS["depart"],
                           "goal_minus_control_d3": float(mean_gc[np.asarray(
                               [x["bucket"] == "depart" for x in references[0]])].mean()),
                           "goal_minus_original_p4_d3": float(mean_go[np.asarray(
                               [x["bucket"] == "depart" for x in references[0]])].mean())},
                "nonstop": {"n": EXPECTED_BUCKETS["nonstop"],
                            "goal_minus_control_d3": float(mean_gc[np.asarray(
                                [x["bucket"] == "nonstop" for x in references[0]])].mean()),
                            "goal_minus_original_p4_d3": float(mean_go[np.asarray(
                                [x["bucket"] == "nonstop" for x in references[0]])].mean())},
                "excluding_session_046": {
                    "n": int((sessions != SESSION_046).sum()),
                    "goal_minus_control_d3": float(mean_gc[sessions != SESSION_046].mean()),
                    "goal_minus_original_p4_d3": float(mean_go[sessions != SESSION_046].mean())},
                "goal_in_grid": {"n": int(in_grid.sum()), "goal_minus_control_d3":
                                 float(mean_gc[in_grid].mean()) if in_grid.any() else None,
                                 "goal_minus_original_p4_d3":
                                 float(mean_go[in_grid].mean()) if in_grid.any() else None},
                "goal_out_of_grid": {"n": int((~in_grid).sum()),
                                     "goal_minus_control_d3": float(mean_gc[~in_grid].mean())
                                     if (~in_grid).any() else None,
                                     "goal_minus_original_p4_d3": float(mean_go[~in_grid].mean())
                                     if (~in_grid).any() else None}},
            "preregistered_keep_gate": {
                "checks": checks, "keep": all(checks.values()),
                "mean_goal_minus_control": mean_gc_value, "mean_goal_d3": mean_goal,
                "meaning": "Reused-tune preregistered KEEP screen; not final, compliance, or deployment evidence"}}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("b0-control", "b0-goal", "b1-control", "b1-goal"):
        parser.add_argument(f"--{name}", nargs=4,
                            metavar=("FINAL_EVAL", "EVAL_SHA256", "MANIFEST", "MANIFEST_SHA256"),
                            required=True)
    for name in ("b0-reference", "b1-reference"):
        parser.add_argument(f"--{name}", nargs=2, metavar=("REPORT", "SHA256"), required=True)
    parser.add_argument("--ego-cache", nargs=2, metavar=("NPZ", "SHA256"), required=True)
    parser.add_argument("--p7-source-manifest", nargs=2, metavar=("MANIFEST", "SHA256"),
                        required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "P7 result analysis is CPU-only with CUDA_VISIBLE_DEVICES=''")
    inputs, reports, audits, lineages = {}, {}, {}, {}
    source_path, source_sha = args.p7_source_manifest
    source_payload, source_raw = p6.read_pinned_json(source_path, source_sha)
    pinned_source = validate_p7_source_manifest(source_payload, source_sha)
    inputs[str(Path(source_path).resolve())] = (source_sha, source_raw)
    for base, arm, spec in ((0, "control", args.b0_control), (0, "goal", args.b0_goal),
                            (1, "control", args.b1_control), (1, "goal", args.b1_goal)):
        eval_path, eval_sha, manifest_path, manifest_sha = spec
        payload, eval_raw = p6.read_pinned_json(eval_path, eval_sha)
        manifest, manifest_raw = p6.read_pinned_json(manifest_path, manifest_sha)
        reports[(base, arm)], audits[f"base{base}_{arm}"] = validate_p7_report(
            payload, base_seed=base, arm=arm)
        lineages[f"base{base}_{arm}"] = validate_terminal_manifest(
            manifest, base_seed=base, arm=arm, pinned_source=pinned_source,
            expected_official_d3=audits[f"base{base}_{arm}"]["stored_official_d3"])
        inputs[str(Path(eval_path).resolve())] = (eval_sha, eval_raw)
        inputs[str(Path(manifest_path).resolve())] = (manifest_sha, manifest_raw)
    references = {}
    for base, spec in enumerate((args.b0_reference, args.b1_reference)):
        path, digest = spec
        require(digest == P4_REFERENCE_SHA256[base], "Immutable P4 reference SHA mismatch")
        payload, raw = p6.read_pinned_json(path, digest)
        require(payload.get("protocol", {}).get("data", {}).get("ego_cache_sha256")
                == EXPECTED_EGO_CACHE_SHA256, "P4 reference ego-cache lineage mismatch")
        references[base], audits[f"p4_base{base}"] = p6.validate_p4_reference(payload, base)
        inputs[str(Path(path).resolve())] = (digest, raw)
    ego_path, ego_sha = args.ego_cache
    require(ego_sha == EXPECTED_EGO_CACHE_SHA256, "Explicit ego-cache SHA mismatch")
    require(Path(ego_path).is_file() and not Path(ego_path).is_symlink(),
            "Pinned ordinary ego-cache required")
    ego_raw = Path(ego_path).read_bytes()
    require(p6.sha256_bytes(ego_raw) == ego_sha, "Pinned ego-cache SHA mismatch")
    goals, in_grid = load_goals(ego_raw, references[0], ego_sha)
    inputs[str(Path(ego_path).resolve())] = (ego_sha, ego_raw)
    result = analyze(reports, references, in_grid)
    for path, (digest, raw) in inputs.items():
        require(Path(path).read_bytes() == raw and p6.sha256_bytes(raw) == digest,
                f"Input changed during analysis: {path}")
    output = {"schema_version": 1,
              "status": "completed_cpu_analysis_of_p7_terminal_reports",
              "scope": "exact four P7 LAST6000 tune reports plus two immutable P4 references",
              "inputs_sha256_before_after_exact": {path: value[0] for path, value in inputs.items()},
              "lineage": lineages, "numeric_d3_audit": audits, "analysis": result,
              "goal_grid_contract": {**GRID_BOUNDS_M, "clipping": False,
                  "source": "raw metric goal[row] from pinned ego cache",
                  "counts": {"in_grid": int(in_grid.sum()), "out_of_grid": int((~in_grid).sum())},
                  "goals_sha256": p6.sha256_bytes(goals.astype("<f4", copy=False).tobytes())},
              "bootstrap_contract": {"clusters": 11, "repeats": BOOTSTRAP_REPEATS,
                  "seed": BOOTSTRAP_SEED, "frame_weighted": True,
                  "same_draws_across_two_base_comparisons": True},
              "boundaries": {"synthetic_tests_are_not_accuracy_evidence": True,
                  "model_forward": False, "training": False, "threshold_selection": False,
                  "final_validation_accessed": False, "gpu_used": False,
                  "reused_tune_is_exploratory": True},
              "environment": {"python": sys.version, "numpy": np.__version__,
                  "torch": str(torch.__version__), "cuda_visible_devices": "",
                  "cuda_initialized": torch.cuda.is_initialized(),
                  "analyzer_sha256": p6.sha256_bytes(Path(__file__).read_bytes())}}
    p6.write_new_json(args.output, output)
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "keep": result["preregistered_keep_gate"]["keep"],
                      "gpu_used": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
