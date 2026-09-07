from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_motiondrive_v2_planning_errors import (TIMES, WEIGHTS, error_summary,
                                                    path_lengths, stop_bucket, temporal_fits)


@pytest.mark.parametrize("distance,expected", [(0., "continued_stop"), (.2, "continued_stop"),
                                             (.20001, "stop_boundary"), (.5, "stop_boundary"),
                                             (.50001, "delayed_start")])
def test_stop_partition_exact_boundaries(distance, expected):
    future = np.zeros((6, 2))
    future[-1, 0] = distance
    assert stop_bucket(np.zeros(6), np.ones(6, bool), future) == expected


def test_speed_boundary_and_invalid_state():
    state = np.zeros(6)
    state[0] = .2
    assert stop_bucket(state, np.ones(6, bool), np.zeros((6, 2))) == "moving"
    assert stop_bucket(state, np.zeros(6, bool), np.zeros((6, 2))) == "state_unknown"


def test_future_max_not_only_endpoint():
    gt = np.zeros((6, 2))
    gt[2, 1] = 1.
    assert stop_bucket(np.zeros(6), np.ones(6, bool), gt) == "delayed_start"


def test_oracle_t_and_t_squared_recover_different_shapes():
    error = np.stack([2 * TIMES, 3 * TIMES ** 2], axis=-1)[None]
    fits = temporal_fits(error)["models"]
    assert fits["t_only"]["longitudinal_explained_energy_fraction"] == pytest.approx(1.)
    assert fits["t_squared_only"]["lateral_explained_energy_fraction"] == pytest.approx(1.)
    assert fits["t_only"]["lateral_explained_energy_fraction"] < .99
    assert fits["t_plus_t_squared"]["joint_explained_energy_fraction"] == pytest.approx(1.)
    assert "coefficients" not in fits["t_only"]


def test_zero_error_explanation_is_undefined_not_nan():
    fits = temporal_fits(np.zeros((2, 6, 2)))
    assert fits["models"]["t_only"]["joint_explained_energy_fraction"] is None


def test_radial_distance_and_path_length_are_distinct():
    path = np.array([[[1., 0.], [0., 0.], [1., 0.], [0., 0.], [1., 0.], [0., 0.]]])
    assert np.linalg.norm(path[0, -1]) == 0
    assert path_lengths(path)[0, -1] == 6


def test_signed_vs_absolute_and_exhaustive_empty_bucket():
    pred = np.stack([np.ones((6, 2)), -np.ones((6, 2))])
    gt = np.zeros_like(pred)
    scores = np.linalg.norm(pred, axis=-1) @ WEIGHTS
    result = error_summary(pred, gt, scores, np.array(["a", "b"]), np.array(["s1", "s2"]), 2, scores.sum())
    assert result["weighted_signed_longitudinal_m"] == 0
    assert result["weighted_abs_longitudinal_m"] == pytest.approx(1.)
    assert result["timewise"][0]["radial_displacement"]["over_fraction"] == 1.
    empty = error_summary(pred[:0], gt[:0], scores[:0], np.array([]), np.array([]), 2, scores.sum())
    assert empty["n"] == 0 and empty["official_d3"] is None
