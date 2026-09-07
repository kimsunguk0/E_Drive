import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_motiondrive_v2_tail_pair import (compare_probe_reports, first2_metric,
                                             initial_function_check, validate_initial_real_check,
                                             validate_run_pair)


def probe(error_by_time, step=500):
    splits = {}
    for split, n in (("train", 16), ("tune", 12)):
        gt = np.zeros((n, 6, 2))
        gt[..., 0] = np.arange(1, 7)
        pred = gt.copy()
        pred[..., 0] += np.asarray(error_by_time)
        splits[split] = {"predictions": pred.tolist(), "targets": gt.tolist(),
                         "records": [{"scenario": split, "frame": i} for i in range(n)]}
    # A tempting but irrelevant BEST report must not influence LAST analysis.
    return {"checkpoints": {"last": {"step": step, **splits}, "best": {"step": 100, **splits}}}


def manifests():
    a = {"status": "completed", "step": 500, "git_sha": "git", "split_sha256": "split", "train_rows_sha256": "train",
         "eval_rows_sha256": "eval", "supervision_manifest_sha256": "sup", "loss_weights": {"plan": 1},
         "data_counts": {"train": 16, "eval": 12}, "model_config": {"plan_output_scale": [1, 1]},
         "arguments": {"resume": None}, "load_report": {"common_checkpoint_sha256": "checkpoint"}}
    b = copy.deepcopy(a)
    b["model_config"]["plan_output_scale"] = [10, 5]
    logs = [{"kind": "train", "step": 1, "sample_order_sha256": "first"},
            {"kind": "train", "step": 500, "sample_order_sha256": "last"}]
    return a, b, logs


def test_last500_train_gates_ignore_best_and_tune_selection():
    result = compare_probe_reports(probe([.3] * 6), probe([.1] * 6))
    assert not result["gates"]["unit1_fit_pass"]
    assert result["gates"]["unit10x5_fit_pass"]
    assert result["screening_decision"] == "SCALED_PASSES_CONTROL_FAILS_ELIGIBLE_FOR_FOLLOWUP"
    with pytest.raises(ValueError, match="LAST500"):
        compare_probe_reports(probe([.3] * 6, step=400), probe([.1] * 6))


def test_early_harm_veto_applies_even_when_total_d3_improves():
    control = probe([.01, .01, .01, .01, 2, 2])
    scaled = probe([.04, .04, .04, .04, .1, .1])
    result = compare_probe_reports(control, scaled)
    assert result["paired_deltas_scaled_minus_control"]["train_d3"] < 0
    assert result["gates"]["unit10x5_fit_pass"]
    assert result["gates"]["first2_harm_over_0p02"]
    assert result["screening_decision"] == "DO_NOT_AUTO_ADOPT_SCALE_EARLY_HARM"


def test_both_pass_does_not_automatically_adopt_parameterization():
    result = compare_probe_reports(probe([.1] * 6), probe([.08] * 6))
    assert result["screening_decision"].startswith("BOTH_PASS_KEEP_CONTROL")


def test_first2_uses_prespecified_renormalized_official_weights():
    assert first2_metric([1, 2, 3, 4, 100, 100]) == pytest.approx((11 + 22 + 15 + 20) / 32)


def test_different_gt_or_row_order_is_rejected():
    a, b = probe([.1] * 6), probe([.1] * 6)
    b["checkpoints"]["last"]["train"]["records"].reverse()
    with pytest.raises(ValueError, match="identity/order"):
        compare_probe_reports(a, b)
    b = probe([.1] * 6)
    b["checkpoints"]["last"]["train"]["targets"][0][0][0] += .1
    with pytest.raises(ValueError, match="ground-truth"):
        compare_probe_reports(a, b)


def test_lineage_and_training_order_are_required():
    a, b, logs = manifests()
    assert validate_run_pair(a, b, logs, logs)["lineage_and_conditions_match"]
    bad_logs = copy.deepcopy(logs)
    bad_logs[-1]["sample_order_sha256"] = "different"
    with pytest.raises(ValueError, match="row order"):
        validate_run_pair(a, b, logs, bad_logs)
    b["train_rows_sha256"] = "bad"
    with pytest.raises(ValueError, match="lineage mismatch"):
        validate_run_pair(a, b, logs, logs)


def test_baseline_separates_schedule_and_scale_effects():
    old = probe([.5] * 6, step=1500)
    result = compare_probe_reports(probe([.2] * 6), probe([.1] * 6), old)
    assert result["schedule_extension_control_delta_d3"] == pytest.approx(-.3)
    assert result["paired_deltas_scaled_minus_control"]["train_d3"] == pytest.approx(-.1)


def test_initial_function_cpu_check_and_non_head_tampering(tmp_path):
    from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
    cfg = MotionDriveV2Config(channels=16, backbone_arch="resnet34", grid_size=(6, 4),
                              motion_grid=(3, 4), scene_attention_channels=8,
                              correlation_channels=8, planner_layers=1)
    model = MotionDriveV2(cfg).eval()
    a, b = tmp_path / "unit1.pth", tmp_path / "unit10x5.pth"
    torch.save({"model": model.state_dict(), "manifest": {"model_config": cfg.to_dict()}}, a)
    model.reparameterize_plan_output_scale((10, 5), preserve_function=True)
    torch.save({"model": model.state_dict(), "manifest": {"model_config": cfg.to_dict()}}, b)
    result = initial_function_check(a, b)
    assert result["pass"]
    assert result["cpu_tiny_input_max_abs_plan_difference_m"] < 1e-4
    with torch.no_grad():
        model.planner.waypoint_queries.add_(.1)
    torch.save({"model": model.state_dict(), "manifest": {"model_config": cfg.to_dict()}}, b)
    with pytest.raises(ValueError, match="non-output"):
        initial_function_check(a, b)


def test_real_initial_check_uses_coordinates_not_only_average_metric():
    a = probe([.1] * 6)["checkpoints"]["last"]["train"]
    b = copy.deepcopy(a)
    assert validate_initial_real_check({"unit1": a, "unit10x5": b})["pass"]
    # A single coordinate violation must fail even if a mean score barely moves.
    b["predictions"][0][5][1] += .001
    with pytest.raises(ValueError, match="prediction difference"):
        validate_initial_real_check({"unit1": a, "unit10x5": b})
