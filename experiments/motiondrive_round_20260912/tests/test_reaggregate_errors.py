"""The checks section 3.6 of the work order requires, plus the orthogonality law."""
import numpy as np
import pytest

from reaggregate_errors import (W, along_cross, arclength, progress_scale_oracle,
                                row_errors, bucket_table)


def test_coordinate_axis_norm():
    a = np.array([[3.0, 4.0], [0.0, 5.0]])
    assert np.allclose(np.linalg.norm(a, axis=-1), [5.0, 5.0])


def test_trajectory_norm_shape():
    x = np.zeros((7, 6, 2))
    assert np.linalg.norm(x, axis=-1).shape == (7, 6)


def test_perfect_prediction_is_zero():
    gt = np.random.default_rng(0).normal(size=(5, 6, 2))
    _, d3, _ = row_errors(gt.copy(), gt)
    assert np.allclose(d3, 0.0)


def test_unit_error_everywhere_gives_one():
    gt = np.zeros((4, 6, 2))
    pred = np.zeros((4, 6, 2))
    pred[..., 0] = 1.0
    _, d3, _ = row_errors(pred, gt)
    assert np.allclose(d3, 1.0)


def test_constant_speed_offset_gives_1p25():
    t = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    gt = np.stack([np.zeros_like(t), np.zeros_like(t)], -1)[None]
    pred = np.stack([t, np.zeros_like(t)], -1)[None]
    _, d3, _ = row_errors(pred, gt)
    assert np.isclose(d3[0], 1.25)


def test_d3_equals_mean_of_cumulative_ades():
    rng = np.random.default_rng(1)
    pred, gt = rng.normal(size=(9, 6, 2)), rng.normal(size=(9, 6, 2))
    _, d3, ade = row_errors(pred, gt)
    assert np.allclose(d3, ade.mean(-1), rtol=1e-12, atol=1e-12)


def test_stopped_segments_flagged_not_nan():
    gt = np.zeros((2, 6, 2))
    gt[1, :, 0] = np.array([1.0, 2.0, 3.0, 3.0, 3.0, 4.0])   # stalls mid-way
    pred = gt + 0.1
    along, cross, valid = along_cross(pred, gt)
    assert np.isfinite(along).all() and np.isfinite(cross).all()
    assert not valid[0].any()          # fully stopped row has no direction
    assert valid[1].sum() == 4         # two stalled segments drop out
    assert np.allclose(along[~valid], 0.0) and np.allclose(cross[~valid], 0.0)


def test_orthogonal_decomposition_on_valid_segments():
    rng = np.random.default_rng(2)
    gt = np.cumsum(rng.normal(size=(6, 6, 2)) + 1.0, axis=1)
    pred = gt + rng.normal(scale=0.3, size=gt.shape)
    along, cross, valid = along_cross(pred, gt)
    err = np.linalg.norm(pred - gt, axis=-1) ** 2
    assert np.allclose((along ** 2 + cross ** 2)[valid], err[valid])


def test_reversing_gt_is_not_forced_positive():
    gt = np.stack([-np.arange(1.0, 7.0), np.zeros(6)], -1)[None]
    pred = gt.copy()
    along, cross, valid = along_cross(pred, gt)
    assert valid.all()
    bearing = np.arctan2(gt[:, -1, 1], gt[:, -1, 0])
    assert np.isclose(abs(bearing[0]), np.pi)       # not clamped to ~0


def test_scale_grid_must_contain_identity():
    gt = np.cumsum(np.ones((3, 6, 2)), axis=1)
    with pytest.raises(ValueError):
        progress_scale_oracle(gt, gt, np.array([0.5, 0.9]))


def test_scaled_straight_line_clamp_is_recorded():
    """pred = 0.98 x GT on a straight line: the residual is clamping, not shape."""
    t = np.arange(1.0, 7.0)
    gt = np.stack([t, np.zeros_like(t)], -1)[None]
    pred = gt * 0.98
    out = progress_scale_oracle(pred, gt, np.linspace(0.5, 1.6, 111))
    assert out["d3"][0] > 0.0                       # clamping leaves error
    assert out["clamped_fraction"][0] > 0.0         # and it is reported as such
    assert out["scale"][0] > 1.0


def test_arclength_matches_manual():
    p = np.array([[[3.0, 4.0], [3.0, 8.0]]])
    assert np.allclose(arclength(p), [[5.0, 9.0]])


def test_group_counterfactual_does_not_add():
    d3 = np.array([1.0, 2.0, 3.0, 4.0])
    a = np.array([True, True, False, False])
    b = np.array([False, True, True, False])
    table = bucket_table(d3, {"a": a, "b": b})
    assert np.isclose(table["a"]["d3_if_group_were_exact"], (0 + 0 + 3 + 4) / 4)
    assert np.isclose(table["b"]["d3_if_group_were_exact"], (1 + 0 + 0 + 4) / 4)
