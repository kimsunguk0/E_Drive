"""동일 forward의 신경 상태 사후 집계: 합성 자료 CPU 검증만 수행한다."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import analyze_motiondrive_v2_motion_predictions as analysis


def fixture_document(conditions=("normal",)):
    n = 6
    t = analysis.TIMES
    states = [[0., 0., 0., 0., -.3, 1.], [3., .1, 1., .2, -.2, 0.],
              [4., -.1, -1., -.3, -.1, 0.], [5., 0., 0., .4, .1, 0.],
              [6., .2, .1, -.5, .2, 0.], [0., 0., 0., 0., .3, 1.]]
    records = []
    for i, state in enumerate(states):
        gt = np.column_stack(((i + 1) * t + .05 * i * t ** 2, -.1 * t + state[4] * t ** 2))
        pred = np.column_stack(((i + 1.1) * t + .02 * i * t ** 2, .2 * t + .5 * state[4] * t ** 2))
        mask = [True] * 6
        neural = np.asarray(state, np.float32)
        neural[:5] += np.array([1., -2., 3., -4., .05], np.float32)
        neural[5] = 3. if state[5] else -2.
        records.append({"row": i + 30, "frame": 30 + i, "scenario": f"scene{i // 3}",
                        "session": f"session{i // 3}", "pred_abs_xy": pred.tolist(), "gt_abs_xy": gt.tolist(),
                        "d3": float(analysis.official_d3(pred[None], gt[None])[0]),
                        "gt_state": list(state), "gt_state_valid": mask,
                        "bucket": analysis.state_bucket(state, mask), "pred_state": neural.tolist(),
                        "pred_history": (np.arange(16).reshape(4, 4) * .01 + i).tolist()})
    protocol = {"status": "preregistered_before_any_forward", "schema_version": 1,
                "arguments": {"split": "tune", "time_input": "nominal", "conditions": list(conditions),
                              "include_motion_predictions": True},
                "checkpoint_step": 6000, "checkpoint_sha256": "a" * 64,
                "data": {"split_sha256": "b" * 64, "ego_cache_sha256": "c" * 64,
                         "receiver_rows_sha256": analysis.digest(np.arange(30, 30 + n, dtype="<i8").tobytes()),
                         "receiver_count": n,
                         "supervision_sha256": {"supervision_manifest.json": "d" * 64, "calibration.npz": "e" * 64}},
                "source": {"file_sha256": {"scripts/evaluate_motiondrive_v2_planning.py": "f" * 64}},
                "bucket_definition": copy.deepcopy(analysis.BUCKET_DEFINITION),
                "model_input_whitelist": list(analysis.MODEL_INPUTS),
                "time_input": "nominal", "nominal_waypoint_seconds": analysis.TIMES.tolist(),
                "coordinate_format": "current ego-frame cumulative XY positions, metres; no cumsum or postprocessing",
                "motion_prediction_contract": analysis.motion_prediction_contract(),
                "selection_performed": False, "final_val_accessed": False}
    return {"status": "completed", "protocol": protocol, "time_input": "nominal",
            "selection_performed": False, "final_val_accessed": False,
            "conditions": {c: {"time_input": "nominal", "records": copy.deepcopy(records)} for c in conditions}}


def test_known_state_error_units_and_aggregate_only():
    result = analysis.analyze(fixture_document())
    stats = result["conditions"]["normal"]["all"]
    for name, error in zip(analysis.STATE_NAMES, (1., -2., 3., -4., .05)):
        assert stats["state"][name]["mae"] == pytest.approx(abs(error), abs=1e-6)
        assert stats["state"][name]["rmse"] == pytest.approx(abs(error), abs=1e-6)
        assert stats["state"][name]["bias_pred_minus_gt"] == pytest.approx(error, abs=1e-6)
        assert stats["state"][name]["pearson"] == pytest.approx(1., abs=1e-6)
    assert stats["state"]["yaw_rate"]["unit"] == "rad/s"
    assert result["aggregate_only"] is True and result["additional_forward_calls"] == 0
    encoded = json.dumps(result, allow_nan=False)
    for forbidden in ('"records"', '"pred_abs_xy"', '"gt_abs_xy"', '"pred_state" :'):
        assert forbidden not in encoded


def test_history_missing_gt_is_descriptive_not_accuracy():
    stats = analysis.analyze(fixture_document())["conditions"]["normal"]["all"]["history"]
    assert stats["gt_available"] is False and stats["accuracy"] is None
    assert np.asarray(stats["raw_pred_component_mean"]).shape == (4, 4)
    assert stats["raw_pred_component_mean"][0][0] == pytest.approx(2.5)


def test_zero_reference_and_state_dispersion_do_not_fit_or_modify_predictions():
    pred = np.array([-1., 0., 1.])
    gt = np.array([-2., 0., 2.])
    before = pred.copy(), gt.copy()
    result = analysis.regression_metrics(pred, gt)
    assert result["zero_reference_mae"] == pytest.approx(4 / 3)
    assert result["mae_relative_to_zero_reference"] == pytest.approx(.5)
    assert result["gt_std_ddof0"] == pytest.approx(np.sqrt(8 / 3))
    assert result["pred_std_ddof0"] == pytest.approx(np.sqrt(2 / 3))
    np.testing.assert_array_equal(pred, before[0])
    np.testing.assert_array_equal(gt, before[1])


@pytest.mark.parametrize("pred,gt", [([], []), ([0., 1.], [0., 0.])])
def test_zero_reference_empty_or_zero_denominator_returns_null(pred, gt):
    result = analysis.regression_metrics(pred, gt)
    assert result["mae_relative_to_zero_reference"] is None
    json.dumps(result, allow_nan=False)


def test_coefficients_axes_two_b_relation_and_correlations_are_distinguished():
    stats = analysis.analyze(fixture_document())["conditions"]["normal"]["all"]
    assert stats["temporal_coefficients"]["y"]["b"]["std_ratio_pred_gt"] == pytest.approx(.5)
    assert stats["temporal_coefficients"]["x"]["b"]["std_ratio_pred_gt"] == pytest.approx(.4)
    cor = stats["yaw_temporal_correlations"]
    assert cor["gt_yaw_vs_gt_future_y_b"]["pearson"] == pytest.approx(1.)
    assert "영상 추정 성능 아님" in cor["gt_yaw_vs_gt_future_y_b"]["meaning"]
    assert "정확도 아님" in cor["estimated_yaw_vs_predicted_future_y_b"]["meaning"]
    assert cor["gt_yaw_vs_estimated_yaw"]["pearson"] == pytest.approx(1., abs=1e-6)


def test_each_mask_controls_its_own_denominator_and_yaw_relations():
    doc = fixture_document()
    record = doc["conditions"]["normal"]["records"][0]
    for j in (1, 4, 5):
        record["gt_state_valid"][j] = False
        record["gt_state"][j] = None
    record["bucket"] = "unknown"
    stats = analysis.analyze(doc)["conditions"]["normal"]["all"]
    assert stats["state"]["vx"]["n_valid"] == 6
    assert stats["state"]["vy"]["n_valid"] == stats["stop"]["n_valid"] == 5
    assert stats["state"]["yaw_rate"]["n_invalid"] == 1
    assert stats["yaw_temporal_correlations"]["gt_yaw_vs_gt_future_y_b"]["n"] == 5
    assert stats["yaw_temporal_correlations"]["estimated_yaw_vs_gt_future_y_b"]["n"] == 6


def test_all_invalid_masks_and_empty_bucket_are_null_without_nan():
    doc = fixture_document()
    for r in doc["conditions"]["normal"]["records"]:
        r.update(gt_state=[None] * 6, gt_state_valid=[False] * 6, bucket="unknown")
    result = analysis.analyze(doc)["conditions"]["normal"]
    assert result["all"]["state"]["vx"]["mae"] is None
    assert result["all"]["stop"]["brier"] is None
    assert result["gt_buckets"]["stop"]["n"] == 0
    assert result["gt_buckets"]["stop"]["temporal_coefficients"]["y"]["b"] is None
    json.dumps(result, allow_nan=False)


def test_session_and_bucket_partitions_are_exhaustive():
    result = analysis.analyze(fixture_document())["conditions"]["normal"]
    assert sum(g["n"] for g in result["sessions"].values()) == 6
    assert sum(g["n"] for g in result["gt_buckets"].values()) == 6
    assert set(result["gt_buckets"]) == set(analysis.BUCKET_NAMES)
    assert sum(g["n"] * g["official_d3"] for g in result["sessions"].values()) / 6 == pytest.approx(result["all"]["official_d3"])


@pytest.mark.parametrize("logits,labels,auc", [([-2., 2.], [0, 1], 1.), ([2., -2.], [0, 1], 0.),
                                              ([0., 0.], [0, 1], .5), ([0., 1.], [1, 1], None),
                                              ([], [], None), ([1000., 1001.], [0, 1], 1.)])
def test_stop_auroc_ties_single_class_empty_extremes(logits, labels, auc):
    result = analysis.stop_metrics(logits, labels)
    assert result["auroc"] == auc
    json.dumps(result, allow_nan=False)


def test_stop_uses_raw_logits_then_sigmoid_not_clipping_or_binary_threshold():
    result = analysis.stop_metrics([-2., 3.], [0., 1.])
    probability = 1 / (1 + np.exp(-np.array([-2., 3.])))
    assert result["brier"] == pytest.approx(np.mean((probability - [0., 1.]) ** 2))
    assert result["raw_logit_min"] == -2. and result["raw_logit_max"] == 3.


def test_constant_pearson_and_coefficient_ratio_are_null():
    assert analysis.pearson([1., 1.], [2., 3.]) is None
    assert analysis.pearson([], []) is None
    assert analysis.pearson([1.], [2.]) is None
    assert analysis.coefficient_statistics([1., 2.], [0., 0.])["std_ratio_pred_gt"] is None


def test_normal_or_all_are_the_only_nontuned_condition_choices():
    doc = fixture_document(analysis.CONDITIONS)
    other = doc["conditions"]["repeat_current"]["records"][0]
    other["pred_state"][0] += 10.
    assert analysis.analyze(doc)["conditions_analyzed"] == ["normal"]
    result = analysis.analyze(doc, "all")
    assert result["conditions_analyzed"] == list(analysis.CONDITIONS)
    assert result["conditions"]["repeat_current"]["all"]["state"]["vx"]["mae"] > result["conditions"]["normal"]["all"]["state"]["vx"]["mae"]
    with pytest.raises(ValueError, match="normal 또는 all"):
        analysis.analyze(doc, "repeat_current")


@pytest.mark.parametrize("change", ["missing_optin", "missing_contract", "extra_forward", "stop_probability",
                                   "source_missing", "split_sha", "not_completed", "finalval", "wrong_time", "coordinate"])
def test_protocol_rejects_unverified_or_wrong_contract(change):
    doc = fixture_document()
    p = doc["protocol"]
    if change == "missing_optin": p["arguments"].pop("include_motion_predictions")
    elif change == "missing_contract": p.pop("motion_prediction_contract")
    elif change == "extra_forward": p["motion_prediction_contract"]["additional_forward_calls"] = 1
    elif change == "stop_probability": p["motion_prediction_contract"]["pred_state"]["units"][-1] = "probability"
    elif change == "source_missing": p["source"]["file_sha256"].clear()
    elif change == "split_sha": p["data"]["split_sha256"] = "bad"
    elif change == "not_completed": doc["status"] = "running"
    elif change == "finalval": doc["final_val_accessed"] = True
    elif change == "wrong_time": p["arguments"]["time_input"] = "raw"
    else: p["coordinate_format"] = "incremental"
    with pytest.raises(ValueError):
        analysis.analyze(doc)


@pytest.mark.parametrize("change", ["missing_prediction", "state_shape", "history_shape", "state_nan", "history_inf",
                                   "mask_type", "valid_null", "invalid_nonnull", "stop_target", "wrong_bucket",
                                   "duplicate_row", "row_order", "bad_identity", "wrong_d3", "unknown_history_gt"])
def test_record_guards_fail_closed(change):
    doc = fixture_document()
    records = doc["conditions"]["normal"]["records"]
    r = records[0]
    if change == "missing_prediction": r.pop("pred_state")
    elif change == "state_shape": r["pred_state"].pop()
    elif change == "history_shape": r["pred_history"].pop()
    elif change == "state_nan": r["pred_state"][0] = float("nan")
    elif change == "history_inf": r["pred_history"][0][0] = float("inf")
    elif change == "mask_type": r["gt_state_valid"][0] = 1
    elif change == "valid_null": r["gt_state"][0] = None
    elif change == "invalid_nonnull": r["gt_state_valid"][0] = False
    elif change == "stop_target": r["gt_state"][5] = .5
    elif change == "wrong_bucket": r["bucket"] = "cruise"
    elif change == "duplicate_row": records[1]["row"] = r["row"]
    elif change == "row_order": records.reverse()
    elif change == "bad_identity": r["row"] = True
    elif change == "wrong_d3": r["d3"] += .1
    else: r["gt_history"] = [[0.] * 4] * 4
    with pytest.raises(ValueError):
        analysis.analyze(doc)


@pytest.mark.parametrize("field", ["state", "mask", "identity", "plan"])
def test_all_conditions_require_identical_receiver_gt_and_masks(field):
    doc = fixture_document(("normal", "reverse_history"))
    r = doc["conditions"]["reverse_history"]["records"][0]
    if field == "state": r["gt_state"][4] += .1
    elif field == "mask": r["gt_state_valid"][4] = False; r["gt_state"][4] = None
    elif field == "identity": r["session"] = "another"; r["scenario"] = "another"
    else:
        r["gt_abs_xy"][0][0] += .1
        r["d3"] = float(analysis.official_d3([r["pred_abs_xy"]], [r["gt_abs_xy"]])[0])
    with pytest.raises(ValueError, match="불변성 위반"):
        analysis.analyze(doc, "all")


def test_file_sha_immutability_and_nonoverwrite(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(fixture_document()), encoding="utf-8")
    original = source.read_bytes()
    sha = analysis.digest(original)
    output = tmp_path / "aggregate.json"
    with pytest.raises(ValueError, match="SHA256 불일치"):
        analysis.analyze_file(source, output, "0" * 64)
    result = analysis.analyze_file(source, output, sha)
    assert result["source_report_sha256"] == sha and source.read_bytes() == original
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        analysis.analyze_file(source, output, sha)
    assert output.read_bytes() == before
    original_analyze = analysis.analyze

    def mutate(*args, **kwargs):
        result = original_analyze(*args, **kwargs)
        source.write_text("changed", encoding="utf-8")
        return result

    monkeypatch.setattr(analysis, "analyze", mutate)
    second = tmp_path / "second.json"
    with pytest.raises(ValueError, match="분석 중 입력 보고서 변경"):
        analysis.analyze_file(source, second, sha)
    assert not second.exists()


def test_cli_help_is_cpu_and_exposes_only_aggregate_analysis():
    result = subprocess.run([sys.executable, str(Path(analysis.__file__)), "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "--expected-source-sha256" in result.stdout and "{normal,all}" in result.stdout
    assert "--device" not in result.stdout and "--checkpoint" not in result.stdout
