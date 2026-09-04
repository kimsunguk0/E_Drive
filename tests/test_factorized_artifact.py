#!/usr/bin/env python3
"""Integration/provenance gates for the durable Phase 5A artifact."""

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


WORKTREE = Path(__file__).parents[1]
SCRIPT = WORKTREE / "scripts/sweep_factorized_bank.py"
SPEC = importlib.util.spec_from_file_location("phase5a_artifact", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

ROOT = Path("/NHNHOME/data/sukim/adcl")
ARTIFACT = ROOT / "data/etri/factorized_phase5a/factorized_P512_V128.npz"
EGO5 = ROOT / "data/etri/ego_cache_5s.npz"
SPLIT = ROOT / "data/etri/val_clips.npz"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FactorizedArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = np.load(ARTIFACT, allow_pickle=False)
        cls.ego = np.load(EGO5, allow_pickle=False)
        cls.split = np.load(SPLIT, allow_pickle=True)

    def test_shape_finite_and_exact_zero(self):
        bank = self.artifact["bank"]
        self.assertEqual(bank.shape, (65025, 10, 2))
        self.assertTrue(np.isfinite(bank).all())
        self.assertTrue(np.array_equal(bank[0], np.zeros((10, 2), np.float32)))
        self.assertEqual(self.artifact["geometry_traj"].shape, (512, 10, 2))
        self.assertEqual(self.artifact["velocity_profile"].shape, (128, 11))

    def test_all_prototypes_are_train_only(self):
        frame = self.ego["frame"]
        train = self.split["train_idx"]
        train = train[frame[train] >= 30]
        mini = set(self.split["minival_idx"].tolist())
        dev = set(self.split["val_idx"].tolist())
        allowed = set(train.tolist())
        geometry_rows = self.artifact["geometry_source_cache_row"]
        velocity_rows = self.artifact["velocity_source_cache_row"][1:]
        for row in np.concatenate([geometry_rows, velocity_rows]):
            self.assertIn(int(row), allowed)
            self.assertNotIn(int(row), mini)
            self.assertNotIn(int(row), dev)
        np.testing.assert_array_equal(self.artifact["geometry_traj"], self.ego["fut5"][geometry_rows])

    def test_velocity_profiles_match_source_rows(self):
        rows = self.artifact["velocity_source_cache_row"][1:]
        traj = self.ego["fut5"][rows]
        _, cumulative, total = M.cumulative_geometry(traj)
        expected = np.concatenate([total[:, None], cumulative / total[:, None]], axis=1)
        np.testing.assert_allclose(self.artifact["velocity_profile"][1:], expected, atol=2e-6, rtol=0)
        self.assertTrue(np.array_equal(self.artifact["velocity_profile"][0], np.zeros(11, np.float32)))

    def test_saved_cross_product_rows_recompose(self):
        p_count = 512
        bank = self.artifact["bank"]
        geometry = self.artifact["geometry_traj"]
        cumulative = self.artifact["geometry_cumulative_s"]
        total = self.artifact["geometry_total_s"]
        profile = self.artifact["velocity_profile"]
        for p, v in ((0, 1), (127, 31), (511, 127)):
            recomposed, _, _ = M.compose_bank(
                geometry[p : p + 1], cumulative[p : p + 1], total[p : p + 1],
                np.stack([np.zeros(11, np.float32), profile[v]]),
            )
            saved_row = 1 + (v - 1) * p_count + p
            np.testing.assert_array_equal(recomposed[1], bank[saved_row])
            self.assertEqual(int(self.artifact["path_id"][saved_row]), p)
            self.assertEqual(int(self.artifact["velocity_id"][saved_row]), v)

    def test_manifest_hashes_match_durable_inputs_and_script(self):
        self.assertEqual(str(self.artifact["source_ego5_sha256"]), digest(EGO5))
        self.assertEqual(str(self.artifact["source_split_sha256"]), digest(SPLIT))
        self.assertEqual(str(self.artifact["source_script_sha256"]), digest(SCRIPT))


if __name__ == "__main__":
    unittest.main()
