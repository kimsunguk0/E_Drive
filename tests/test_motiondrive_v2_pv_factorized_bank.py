from __future__ import annotations

import numpy as np
import pytest

from scripts import build_motiondrive_v2_pv_factorized_bank as bank


def simple_axes():
    path_xy = np.zeros((2, 10, 2), np.float32)
    path_xy[0, :, 0] = np.arange(1, 11, dtype=np.float32)
    path_xy[1, :, 1] = np.arange(1, 11, dtype=np.float32)
    _step, cumulative, total = bank.cumulative_geometry(path_xy)
    profiles = np.zeros((3, 11), np.float32)
    profiles[1, 0] = 10.
    profiles[1, 1:] = np.linspace(.1, 1., 10, dtype=np.float32)
    profiles[2, 0] = 5.
    profiles[2, 1:] = np.linspace(.1, 1., 10, dtype=np.float32)
    return path_xy, cumulative, total, profiles


def test_fixed_constants_and_sqrt_quota_are_bounded():
    assert bank.P_COUNT == 512
    assert bank.V_COUNT == 128
    assert bank.BANK_COUNT == 65025
    assert bank.FIT_SEED == 20260909
    assert bank.GATE_MAX_D3 == pytest.approx(.11486043)
    labels = np.asarray(["a"] * 100 + ["b"] * 25 + ["c"] * 4)
    counts, quotas = bank.allocate_sqrt_quotas(labels, ("a", "b", "c"), 12)
    assert counts == {"a": 100, "b": 25, "c": 4}
    assert sum(quotas.values()) == 12
    assert all(1 <= quotas[key] <= counts[key] for key in quotas)


def test_composition_is_path_major_with_dedicated_stop():
    path_xy, cumulative, total, profiles = simple_axes()
    trajectories, path_id, velocity_id = bank.compose_bank(
        path_xy, cumulative, total, profiles)
    assert trajectories.shape == (5, 10, 2)
    assert np.array_equal(path_id, np.asarray([-1, 0, 0, 1, 1], np.int32))
    assert np.array_equal(velocity_id, np.asarray([0, 1, 2, 1, 2], np.int32))
    assert np.array_equal(trajectories[0], np.zeros((10, 2), np.float32))
    assert np.allclose(trajectories[1], path_xy[0])
    assert np.allclose(trajectories[2], path_xy[0] * .5)
    assert np.allclose(trajectories[3], path_xy[1])
    assert np.allclose(trajectories[4], path_xy[1] * .5)
    diagnostics = bank.bank_diagnostics(trajectories, path_id, velocity_id, 2, 3)
    assert diagnostics == {"finite": True, "exact_zero_count": 1,
                           "duplicate_count": 0,
                           "path_major_cartesian_coverage": True}


def test_duplicate_compositions_are_reported_not_rejected():
    path_xy, cumulative, total, profiles = simple_axes()
    profiles[2] = profiles[1]
    trajectories, path_id, velocity_id = bank.compose_bank(
        path_xy, cumulative, total, profiles)
    diagnostics = bank.bank_diagnostics(trajectories, path_id, velocity_id, 2, 3)
    assert diagnostics["duplicate_count"] == 2
    assert diagnostics["exact_zero_count"] == 1


def test_exact_oracle_uses_weighted_euclidean_and_first_tie():
    candidate = np.zeros((3, 10, 2), np.float32)
    candidate[1, :, 0] = 1.
    candidate[2] = candidate[1]
    gt = np.zeros((2, 10, 2), np.float32)
    gt[0] = candidate[1]
    gt[1, :6, 0] = .4
    selected, oracle = bank.exact_oracle(gt, candidate, candidate_chunk=1)
    assert np.array_equal(selected, np.asarray([1, 0], np.int32))
    assert np.array_equal(oracle["d3"], np.asarray([0., .4], np.float32))
    assert np.array_equal(oracle["endpoint5"], np.asarray([0., 0.], np.float32))
    assert set(oracle) == {"d3", "endpoint5", "tail_3p5_to_5", "mean10"}


def test_manifest_contract_accepts_reported_duplicates():
    manifest = {
        "schema_version": 1,
        "status": "frozen_train203_factorized_bank_before_tune_gate",
        "scope": {"selected_fit_rows": "current train203 only",
                  "train_rows": bank.current.EXPECTED_TRAIN_ROWS,
                  "tune_rows_selected_or_used_for_fit": 0,
                  "final136_rows_selected_fit_evaluated_or_analyzed": 0,
                  "gpu_used": False, "neural_model_forward_or_training": False},
        "source": {"script_sha256": "a" * 64,
                   "train203_helper_sha256": bank.EXPECTED_HELPER_SHA256},
        "train": {"scenes": 203, "rows": bank.current.EXPECTED_TRAIN_ROWS,
                  "rows_sha256": bank.current.EXPECTED_TRAIN_ROWS_SHA256,
                  "selected_identity_sha256":
                  bank.current.EXPECTED_SELECTED_IDENTITY_SHA256["train"]},
        "recipe": {"fixed_seed": bank.FIT_SEED, "P": bank.P_COUNT,
                   "V": bank.V_COUNT, "candidate_count": bank.BANK_COUNT,
                   "exact_stop_rows": 1, "no_P_or_V_sweep": True,
                   "published_order":
                   "row0 exact stop; path-major p=0..511 then v=1..127"},
        "bank_diagnostics": {"exact_zero_count": 3, "duplicate_count": 7},
        "artifact": {"path": "factorized_bank_P512_V128.npz", "sha256": "b" * 64},
        "inputs": {"split": {"sha256": bank.current.EXPECTED_SPLIT_SHA256},
                   "ego": {"sha256": bank.current.EXPECTED_EGO_SHA256},
                   "ego5": {"sha256": bank.current.EXPECTED_EGO5_SHA256}},
        "arrays": {name: {"shape": shape, "dtype": dtype, "sha256": "c" * 64}
                   for name, (shape, dtype) in bank.EXPECTED_ARRAY_SCHEMA.items()},
    }
    bank.validate_manifest(manifest)


def test_arguments_expose_only_fixed_fit_and_gate(tmp_path):
    common = ["--split-manifest", str(tmp_path / "split.json"),
              "--ego-cache", str(tmp_path / "ego.npz"),
              "--ego5-cache", str(tmp_path / "ego5.npz"),
              "--expected-script-sha256", "a" * 64,
              "--output-dir", str(tmp_path / "out")]
    fit = bank.arguments(["fit", *common])
    assert fit.phase == "fit" and not hasattr(fit, "P")
    gate = bank.arguments(["gate", *common, "--bank-manifest", str(tmp_path / "bank.json"),
                           "--expected-bank-manifest-sha256", "b" * 64])
    assert gate.phase == "gate" and not hasattr(gate, "threshold")
