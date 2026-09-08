"""Synthetic CPU contracts only; no P7 model result or forward is produced."""
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import analyze_motiondrive_v2_p7_goal_routing_results as analysis


def row(index, *, session="s0", bucket="nonstop", d3=.3):
    gt = np.zeros((6, 2), np.float32)
    stop = bucket != "nonstop"
    if bucket == "depart": gt[-1, 0] = 1.
    pred = gt.copy(); pred[:, 0] += d3
    weights = np.asarray(analysis.p6.OFFICIAL_TIME_WEIGHTS, np.float32)
    exact_d3 = float((torch.linalg.vector_norm(torch.from_numpy(pred - gt), dim=-1)
                      * torch.from_numpy(weights)).sum())
    state = np.zeros(6, np.float32); state[5] = .2
    gt_state = np.zeros(6, np.float32); gt_state[5] = float(stop)
    return {"identity": (index, f"scene{index}", session, 30 + index),
            "pred": pred, "gt": gt, "d3": exact_d3, "bucket": bucket,
            "gt_state": gt_state, "gt_state_valid": (True,) * 6,
            "stop_valid": True, "stop_target": float(stop), "stop_logit": .2}


def rows(offset=0., sessions=11):
    result = []
    buckets = ("steady", "depart", "nonstop")
    for index in range(33):
        item = row(index, session=f"s{index % sessions}", bucket=buckets[index % 3], d3=.3)
        item["d3"] += offset
        result.append(item)
    return result


def test_keep_gate_exact_fixed_recipe_and_shared_session_bootstrap(monkeypatch):
    monkeypatch.setattr(analysis, "EXPECTED_BUCKETS", {"steady": 11, "depart": 11, "nonstop": 11})
    original0, original1 = rows(.06), rows(.06)
    reports = {(0, "control"): rows(.05), (1, "control"): rows(.05),
               (0, "goal"): rows(0.), (1, "goal"): rows(0.)}
    in_grid = np.asarray([index % 2 == 0 for index in range(33)])
    result = analysis.analyze(reports, {0: original0, 1: original1}, in_grid)
    gate = result["preregistered_keep_gate"]
    assert gate["keep"] and gate["mean_goal_minus_control"] == pytest.approx(-.05)
    assert set(gate["checks"]) == {
        "both_bases_goal_better_than_control", "both_bases_goal_better_than_original_p4",
        "mean_goal_minus_control_at_most_minus_0p015",
        "shared_session_goal_minus_control_ci_upper_below_zero", "mean_goal_d3_at_most_0p34"}
    bootstrap = result["shared_session_bootstrap"]
    assert bootstrap["repeats"] == 10000 and bootstrap["seed"] == 20260908
    assert bootstrap["goal_minus_control"]["clusters"] == 11
    assert bootstrap["same_resamples_for_both_comparisons"] is True


def test_keep_fails_without_all_preregistered_checks(monkeypatch):
    monkeypatch.setattr(analysis, "EXPECTED_BUCKETS", {"steady": 11, "depart": 11, "nonstop": 11})
    base = rows(0.)
    reports = {(0, "control"): rows(0.), (1, "control"): rows(0.),
               (0, "goal"): rows(0.), (1, "goal"): rows(0.)}
    result = analysis.analyze(reports, {0: base, 1: rows(0.)},
                              np.asarray([index % 2 == 0 for index in range(33)]))
    assert result["preregistered_keep_gate"]["keep"] is False


def test_goal_grid_uses_inclusive_metric_bounds_and_exact_join(monkeypatch):
    reference = []
    scenarios, indices, frames, goals, futures = [], [], [], [], []
    values = [(-10., -32.), (70., 32.), (-10.001, 0.), (0., 32.001)]
    for index, goal in enumerate(values):
        scenarios.append(f"scene{index}"); indices.append(index); frames.append(30 + index)
        goals.append(goal); futures.append(np.zeros((6, 2), np.float32))
        reference.append({"identity": (index, f"scene{index}", f"s{index}", 30 + index),
                          "gt": np.zeros((6, 2), np.float32)})
    stream = io.BytesIO()
    np.savez(stream, scenarios=np.asarray(scenarios), scen_idx=np.asarray(indices),
             frame=np.asarray(frames), goal=np.asarray(goals, np.float32),
             fut=np.asarray(futures, np.float32))
    digest = analysis.p6.sha256_bytes(stream.getvalue())
    monkeypatch.setattr(analysis, "EXPECTED_EGO_CACHE_SHA256", digest)
    loaded, inside = analysis.load_goals(stream.getvalue(), reference, digest)
    assert np.array_equal(loaded, np.asarray(goals, np.float32))
    assert inside.tolist() == [True, True, False, False]
    broken = list(reference); broken[0] = dict(broken[0], identity=(0, "wrong", "s0", 30))
    with pytest.raises(ValueError, match="join mismatch"):
        analysis.load_goals(stream.getvalue(), broken, digest)


def manifest(base=0, arm="control", source="a" * 64):
    expected_arm, mode = analysis.ARM_TO_MODE[arm]
    return {"status": "completed", "step": 6000, "nonfinite_count": 0,
            "git_sha": "ca50e75fd43e38d2c7833d409869f4e74f969010", "best_metric": .3,
            "arguments": {"phase": "joint", "steps": 6000, "seed": base,
                "goal_on": 1, "state_on": 1, "cross_cell_goal_mode": mode,
                "batch": 16, "microbatch": 2, "eval_batch": 4, "eval_every": 6000,
                "save_every": 6000, "precision": "bf16", "time_input": "nominal",
                "bn_policy": "fixed", "eval_split": "tune", "eval_stride": 5,
                "max_eval_samples": 0, "train_stride": 1, "max_train_samples": 0,
                "lr": 1e-4, "backbone_lr": 1e-5, "weight_decay": .01, "warmup": 200,
                "grad_clip": 5., "alpha_occ": .2, "alpha_lane": .2,
                "alpha_motion": .2, "uncertainty": 1, "workers": 4, "gpu": 0,
                "cuda_memory_limit_mib": 12000, "cuda_min_free_mib": 8192,
                "train_scenes": None, "eval_scenes": None, "eval_only": False,
                "resume": None},
            "model_config": {"cross_cell_goal_mode": mode, "goal_on": True,
                "state_on": True, "backbone_arch": "resnet50",
                "motion_input_mode": "low_feature", "plan_output_scale": [10., 5.]},
            "experimental_protocol": {"name": "p7_cross_cell_goal_routing",
                "schema_version": 1, "arm": expected_arm, "last_only_final_eval": True,
                "expected_initial_model_state_sha256":
                    analysis.EXPECTED_INITIAL_MODEL_STATE_SHA256[base],
                "fresh_optimizer_step_zero": True,
                "all_model_parameters_joint_trainable": True,
                "existing_goal_path_on": True, "planner_signature_unchanged": True,
                "branch": {"mode": mode, "sigma_m": [10., 32. / 3.],
                    "sigma_selection": "fixed_geometry_not_tuned", "pool_size": 4,
                    "source_cells": 192, "destination_cells": 3072, "attention_dim": 32,
                    "cosine_scale": 8., "attention_precision": "fp32_autocast_disabled",
                    "goal_enters_distance_score_only": True,
                    "new_value_projection_adds_goal_or_position": False,
                    "output_bias": False, "output_weight_zero_initialized": True},
                "source": {"path": "/p7/source_manifest.json",
                           "sha256": source,
                           "git_sha": "64e92694ccbe0a3c5a581e421e16a884b9c5a088",
                           "file_sha256": {name: "c" * 64 for name in analysis.p7.SOURCE_FILES}},
                "expected_p0_model_state_sha256": analysis.p7.P0_ARTIFACTS[base]["model_state_sha256"],
                "expected_optimizer_groups": [{"name": "backbone", "base_lr": 1e-5},
                                              {"name": "head", "base_lr": 1e-4}],
                "expected_missing_state_keys": list(analysis.EXPECTED_NEW_STATE_KEYS),
                "train_data": {"rows": 54810,
                    "rows_sha256": analysis.p7.EXPECTED_TRAIN_ROWS_SHA256},
                "tune_data": {"rows": analysis.EXPECTED_N,
                    "rows_sha256": analysis.p7.EXPECTED_TUNE_ROWS_SHA256}},
            "data_counts": {"train": 54810, "eval": analysis.EXPECTED_N},
            "split_sha256": analysis.p7.EXPECTED_SPLIT_SHA256,
            "supervision_manifest_sha256": analysis.p7.EXPECTED_SUPERVISION_SHA256,
            "train_rows_sha256": analysis.p7.EXPECTED_TRAIN_ROWS_SHA256,
            "eval_rows_sha256": analysis.p7.EXPECTED_TUNE_ROWS_SHA256,
            "initial_model_state_sha256": analysis.EXPECTED_INITIAL_MODEL_STATE_SHA256[base],
            "loss_weights": {"plan": 1., "occupancy": .2, "lane": .2,
                             "motion": .2, "uncertainty": True},
            "load_report": {"common_checkpoint_sha256":
                            analysis.p7.P0_ARTIFACTS[base]["checkpoint_sha256"]}}


def pinned_source():
    return {"sha256": "a" * 64,
            "git_sha": "64e92694ccbe0a3c5a581e421e16a884b9c5a088",
            "file_sha256": {name: "c" * 64 for name in analysis.p7.SOURCE_FILES}}


@pytest.mark.parametrize("mutation", ["status", "step", "source", "source_map", "seed", "mode",
                                       "rows", "p0", "initial", "expected_initial"])
def test_terminal_manifest_fails_closed(mutation):
    value = manifest()
    if mutation == "status": value["status"] = "running"
    elif mutation == "step": value["step"] = 5999
    elif mutation == "source": value["experimental_protocol"]["source"]["sha256"] = "d" * 64
    elif mutation == "source_map":
        value["experimental_protocol"]["source"]["file_sha256"][next(iter(analysis.p7.SOURCE_FILES))] = "d" * 64
    elif mutation == "seed": value["arguments"]["seed"] = 1
    elif mutation == "mode": value["model_config"]["cross_cell_goal_mode"] = "real"
    elif mutation == "rows": value["data_counts"]["eval"] -= 1
    elif mutation == "p0": value["load_report"]["common_checkpoint_sha256"] = "e" * 64
    elif mutation == "initial": value["initial_model_state_sha256"] = "e" * 64
    else: value["experimental_protocol"]["expected_initial_model_state_sha256"] = "e" * 64
    with pytest.raises(ValueError):
        analysis.validate_terminal_manifest(value, base_seed=0, arm="control",
                                            pinned_source=pinned_source(),
                                            expected_official_d3=.3)


def test_terminal_manifest_accepts_exact_control_and_goal():
    assert analysis.validate_terminal_manifest(
        manifest(), base_seed=0, arm="control", pinned_source=pinned_source(),
        expected_official_d3=.3)["source_and_launch_git_equal_informational"] is False
    assert analysis.validate_terminal_manifest(
        manifest(1, "goal"), base_seed=1, arm="goal",
        pinned_source=pinned_source(), expected_official_d3=.3)["mode"] == "real"
    with pytest.raises(ValueError, match="metric binding"):
        analysis.validate_terminal_manifest(
            manifest(), base_seed=0, arm="control", pinned_source=pinned_source(),
            expected_official_d3=.31)


def test_external_source_manifest_schema_and_map_are_exact(monkeypatch):
    monkeypatch.setattr(analysis, "EXPECTED_P7_SOURCE_MANIFEST_SHA256", "a" * 64)
    payload = {"schema_version": 1,
               "git_sha": "64e92694ccbe0a3c5a581e421e16a884b9c5a088",
               "file_sha256": {name: "c" * 64 for name in analysis.p7.SOURCE_FILES}}
    assert analysis.validate_p7_source_manifest(payload, "a" * 64) == pinned_source()
    payload["file_sha256"][next(iter(analysis.p7.SOURCE_FILES))] = "not-a-hash"
    with pytest.raises(ValueError, match="pinned"):
        analysis.validate_p7_source_manifest(payload, "a" * 64)


def test_row_and_gt_mismatch_fails_closed():
    reference = rows()
    other = rows(); other[3] = dict(other[3], gt=np.ones((6, 2), np.float32))
    with pytest.raises(ValueError, match="GT plan differs"):
        analysis.require_same_rows(reference, {"other": other})


def detailed_record(index, bucket):
    normalized = row(index, session=f"s{index % 2}", bucket=bucket)
    identity = normalized["identity"]
    return {"row": identity[0], "scenario": identity[1], "session": identity[2],
            "frame": identity[3], "pred_abs_xy": normalized["pred"].tolist(),
            "gt_abs_xy": normalized["gt"].tolist(), "d3": normalized["d3"],
            "bucket": bucket, "stop_bucket": bucket,
            "max_gt_displacement_m": analysis.p6._max_gt_displacement(normalized["gt"]),
            "pred_state": np.zeros(6, np.float32).tolist(),
            "gt_state": normalized["gt_state"].tolist(),
            "gt_state_valid": [True] * 6, "stop_valid": True,
            "stop_target": normalized["stop_target"], "proxy": 1.}


def test_terminal_report_fp32_audit_and_saved_row_schema(monkeypatch):
    monkeypatch.setattr(analysis, "EXPECTED_N", 6)
    monkeypatch.setattr(analysis, "EXPECTED_SCENES", 6)
    monkeypatch.setattr(analysis, "EXPECTED_SESSIONS", 2)
    monkeypatch.setattr(analysis, "EXPECTED_BUCKETS", {"steady": 2, "depart": 2, "nonstop": 2})
    records = [detailed_record(index, ("steady", "depart", "nonstop")[index % 3])
               for index in range(6)]
    mean = float(np.mean([item["d3"] for item in records]))
    payload = {"report": {"kind": "eval", "step": 6000, "time_input": "nominal",
                          "n": 6, "n_sessions": 2, "official_d3": mean,
                          "selection_metric": mean, "selection_definition": "official_d3"},
               "records": records}
    result, audit = analysis.validate_p7_report(payload, base_seed=0, arm="control")
    assert len(result) == 6 and audit["stored_official_d3"] == mean
    payload["records"][0]["d3"] += .01
    with pytest.raises(ValueError, match="FP32 recomputation"):
        analysis.validate_p7_report(payload, base_seed=0, arm="control")


def test_numpy_safe_new_json_roundtrip_and_no_overwrite(tmp_path):
    path = tmp_path / "result.json"
    analysis.p6.write_new_json(path, {"float": np.float32(.5), "int": np.int64(2),
                                      "array": np.asarray([1., 2.], np.float32)})
    assert json.loads(path.read_text()) == {"float": .5, "int": 2, "array": [1., 2.]}
    with pytest.raises(ValueError, match="overwrite"):
        analysis.p6.write_new_json(path, {"x": 1})


def test_cli_requires_exact_four_runs_references_goal_cache_and_source():
    with pytest.raises(SystemExit):
        analysis.arguments([])
    source = Path(analysis.__file__).read_text()
    assert "final_validation_accessed\": False" in source
    assert "model_forward\": False" in source
    assert 'add_argument("--threshold' not in source
    assert 'parser.add_argument("--p7-source-manifest"' in source
