#!/usr/bin/env python3
"""Pure CPU unit tests for Phase 5A factorization helpers."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "sweep_factorized_bank.py"
SPEC = importlib.util.spec_from_file_location("phase5a", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class FactorizedBankTest(unittest.TestCase):
    def test_exact_reconstruction_of_piecewise_linear_path(self):
        traj = np.asarray(
            [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 1.0], [5.0, 2.0],
              [6.0, 3.0], [7.0, 4.0], [8.0, 5.0], [9.0, 5.5], [10.0, 6.0]]],
            dtype=np.float32,
        )
        _, cumulative, total = M.cumulative_geometry(traj)
        profile = np.concatenate([[total[0]], cumulative[0] / total[0]])[None].astype(np.float32)
        profiles = np.concatenate([np.zeros((1, 11), np.float32), profile], axis=0)
        bank, path_id, velocity_id = M.compose_bank(traj, cumulative, total, profiles)
        self.assertTrue(np.array_equal(bank[0], np.zeros((10, 2), np.float32)))
        np.testing.assert_allclose(bank[1], traj[0], atol=2e-6, rtol=0)
        self.assertEqual(path_id.tolist(), [-1, 0])
        self.assertEqual(velocity_id.tolist(), [0, 1])

    def test_quota_is_exact_and_preserves_rare_strata(self):
        labels = np.asarray(["straight"] * 100 + ["left"] * 4 + ["right"] * 2 + ["uturn"])
        quota = M.allocate_sqrt_quotas(labels, M.ROUTE_NAMES, 16)
        self.assertEqual(sum(quota.values()), 16)
        self.assertTrue(all(quota[name] >= 1 for name in M.ROUTE_NAMES))
        self.assertLessEqual(quota["uturn"], 1)

    def test_official_distance_prefers_exact_candidate(self):
        gt = np.zeros((2, 10, 2), dtype=np.float32)
        gt[1, :, 0] = np.arange(1, 11)
        bank = gt.copy()
        vectors = M.oracle_vectors(gt, bank, chunk=1)
        np.testing.assert_array_equal(vectors["d3"], np.zeros(2))
        np.testing.assert_array_equal(vectors["endpoint5"], np.zeros(2))
        np.testing.assert_array_equal(vectors["traj5"], np.zeros(2))


if __name__ == "__main__":
    unittest.main()
