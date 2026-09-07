"""사후 기술통계 정의·입력 보존 CPU 테스트. 학습/배포 경로를 만들지 않는다."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_motiondrive_v2_temporal_shape as analysis


def document(pred, gt):
    records = [{"row": i, "frame": 30 + i, "scenario": f"scene{i % 2}", "session": f"session{i % 2}",
                "pred_abs_xy": p.tolist(), "gt_abs_xy": g.tolist(), "d3": float(score)}
               for i, (p, g, score) in enumerate(zip(pred, gt, analysis.official_d3(pred, gt)))]
    return {"final_val_accessed": False, "time_input": "nominal",
            "protocol": {"arguments": {"split": "tune", "supervision_root": "mock"},
                         "checkpoint_step": 3000, "data": {}}, "conditions": {"normal": {"records": records}}}


def quadratic(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a[:, None, :] * analysis.TIMES[None, :, None] + b[:, None, :] * analysis.TIMES[None, :, None] ** 2


def test_known_quadratic_and_two_b_second_derivative_without_axis_mix():
    a = np.array([[2., -3.], [4., 8.]])
    b = np.array([[.5, -2.], [-1., 3.]])
    xy = quadratic(a, b)
    fitted = analysis.fit_coefficients(xy)
    np.testing.assert_allclose(fitted[:, 0], a, atol=1e-13)
    np.testing.assert_allclose(fitted[:, 1], b, atol=1e-13)
    np.testing.assert_allclose((xy[:, 2] - 2 * xy[:, 1] + xy[:, 0]) / .5 ** 2, 2 * fitted[:, 1], atol=1e-13)
    r = analysis.fit_residual_statistics(xy[:, :, 1])["t_plus_t_squared"]
    assert r["coordinate_rmse_m"] < 1e-13 and r["coordinate_max_abs_m"] < 1e-12


def test_linear_and_zero_variance_use_none_not_nan():
    xy = quadratic([[2., 3.], [2., 3.]], [[0., 0.], [0., 0.]])
    np.testing.assert_allclose(analysis.fit_coefficients(xy)[:, 1], 0, atol=1e-13)
    stats = analysis.coefficient_statistics([0., 0.], [0., 0.])
    assert stats["pearson"] is None and stats["std_ratio_pred_gt"] is None
    assert stats["variance_ratio_pred_gt"] is None
    zero = analysis.fit_residual_statistics(np.zeros((2, 6)))["t_plus_t_squared"]
    assert zero["residual_uncentered_energy_fraction"] is None
    assert zero["coordinate_rmse_m"] == zero["coordinate_max_abs_m"] == 0
    json.dumps(analysis.analyze(document(np.zeros((2, 6, 2)), np.zeros((2, 6, 2)))), allow_nan=False)


def test_sign_partitions_are_exhaustive_and_axes_are_separate():
    gt = quadratic([[0., 0.]] * 3, [[1., -1.], [2., 0.], [3., 2.]])
    pred = quadratic([[0., 0.]] * 3, [[.1, -.1], [.2, 0.], [.3, .2]])
    result = analysis.analyze(document(pred, gt))
    yp = result["gt_b_sign_partitions"]["y"]
    assert [yp[k]["n"] for k in ("negative", "exact_zero", "positive")] == [1, 1, 1]
    assert sum(x["n"] for x in yp.values()) == result["all"]["n"]
    assert sum(x["fraction_total_d3_sum"] for x in yp.values()) == pytest.approx(1.)
    for name in ("x", "y"):
        assert result["all"]["axes"][name]["quadratic_coefficient_b"]["std_ratio_pred_gt"] == pytest.approx(.1)
    assert result["all"]["axes"]["x"]["quadratic_coefficient_b"]["gt_mean"] == pytest.approx(2.)
    assert result["all"]["axes"]["y"]["quadratic_coefficient_b"]["gt_mean"] == pytest.approx(1 / 3)


def test_exact_official_weights_not_unweighted_ols():
    pred = np.zeros((1, 6, 2)); pred[0, :, 0] = [1, 2, 3, 4, 5, 6]
    target = np.zeros_like(pred)
    expected = (1.5 + 2.5 + 3.5) / 3
    assert analysis.official_d3(pred, target)[0] == pytest.approx(expected)
    stats = analysis.fit_residual_statistics(pred[:, :, 0])["t_only"]
    assert stats["rmse_denominator_coordinate_count"] == 6
    assert stats["original_energy_m2"] == pytest.approx(91)


@pytest.mark.parametrize("change", ["nan", "shape", "duplicate_row", "duplicate_frame", "wrong_d3", "bad_identity", "empty", "finalval"])
def test_guards_reject_invalid_records(change):
    values = np.zeros((2, 6, 2))
    doc = document(values, values)
    records = doc["conditions"]["normal"]["records"]
    if change == "nan": records[0]["gt_abs_xy"][0][0] = float("nan")
    elif change == "shape": records[0]["pred_abs_xy"] = [[0., 0.]] * 5
    elif change == "duplicate_row": records[1]["row"] = records[0]["row"]
    elif change == "duplicate_frame": records[1].update(scenario=records[0]["scenario"], frame=records[0]["frame"])
    elif change == "wrong_d3": records[0]["d3"] = 1.
    elif change == "bad_identity": records[0]["row"] = True
    elif change == "empty": records.clear()
    else: doc["final_val_accessed"] = True
    with pytest.raises(ValueError): analysis.analyze(doc)


def test_write_requires_pinned_immutable_source_and_never_overwrites(tmp_path, monkeypatch):
    source, out = tmp_path / "source.json", tmp_path / "out.json"
    xy = np.zeros((2, 6, 2))
    raw = json.dumps(document(xy, xy)).encode()
    source.write_bytes(raw)
    digest = analysis.sha256_bytes(raw)
    with pytest.raises(ValueError, match="SHA256"):
        analysis.analyze_file(source, out, "0" * 64)
    assert not out.exists()
    result = analysis.analyze_file(source, out, digest)
    assert source.read_bytes() == raw and result["source_report_sha256"] == digest
    previous = out.read_bytes()
    with pytest.raises(FileExistsError): analysis.analyze_file(source, out, digest)
    assert out.read_bytes() == previous
    other = tmp_path / "other.json"
    real_analyze = analysis.analyze
    def mutate(doc):
        result = real_analyze(doc)
        source.write_bytes(raw + b"\n")
        return result
    monkeypatch.setattr(analysis, "analyze", mutate)
    with pytest.raises(ValueError, match="분석 중"):
        analysis.analyze_file(source, other, digest)
    assert not other.exists()


def test_report_has_aggregates_only_and_explicit_descriptive_limits():
    xy = quadratic([[1., 2.], [2., 3.]], [[.3, -.2], [.1, .4]])
    result = analysis.analyze(document(xy, xy))
    text = json.dumps(result, allow_nan=False)
    assert "pred_abs_xy" not in text and "gt_abs_xy" not in text
    assert not result["per_sample_coefficients_or_fitted_paths_exported"]
    assert result["future_gt_descriptive_only"] and not result["deployment_correction_allowed"]
    assert not result["raw_or_additional_val_access"] and not result["gpu_used"]
