"""Numerical contracts that can silently corrupt a factorized bank."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import json

import numpy as np

SPEC = importlib.util.spec_from_file_location("bank", Path(__file__).with_name("bank.py"))
bank = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bank)


class BankContracts(unittest.TestCase):
    def test_stop_duplicates_reverse_and_invalid_path(self):
        points = np.array([[0., 0.], [0., 0.], [-1., 0.], [-1., 0.], [-3., 0.]])
        xy, mask, total = bank.sample_path(points, np.array([1., 2., 3., 4.]))
        np.testing.assert_array_equal(mask, [True, True, True, False])
        np.testing.assert_allclose(xy, [[-1., 0.], [-2., 0.], [-3., 0.], [0., 0.]])
        self.assertEqual(total, 3.)
        xy, mask, total = bank.sample_path(np.zeros((4, 2)), np.arange(1., 5.))
        self.assertFalse(mask.any())
        self.assertEqual(total, 0.)
        self.assertTrue(np.isfinite(xy).all())

    def test_absolute_speed_preserves_physical_curvature(self):
        # A right-angle path turns after exactly 2m, independent of velocity.
        paths = np.array([[[1., 0.], [2., 0.], [2., 1.], [2., 2.]]], np.float32)
        velocity = np.array([[0., 0., 0.], [2., 2., 2.], [4., 4., 4.]], np.float32)
        xy, valid, lengths = bank.compose(paths, velocity, allow_extrapolation=True)
        np.testing.assert_array_equal(xy[0, 0], np.zeros((3, 2)))
        np.testing.assert_allclose(xy[0, 1], [[1, 0], [2, 0], [2, 1]])
        np.testing.assert_allclose(xy[0, 2], [[2, 0], [2, 2], [2, 4]])
        np.testing.assert_array_equal(valid[0, 2], [True, True, False])
        self.assertEqual(lengths[0], 4.)

    def test_metric_is_official_prefix_ade_weight(self):
        errors = np.array([1., 2., 3., 4., 5., 6.])
        explicit = (errors[:2].mean() + errors[:4].mean() + errors.mean()) / 3
        self.assertAlmostEqual(float(errors @ bank.WEIGHTS), explicit, places=6)
        self.assertAlmostEqual(float(np.array([.1,.1,.2,.2,.3,.3]) @ bank.WEIGHTS), .15, places=6)

    def test_absolute_velocity_uses_each_interval(self):
        future = np.array([[[1., 0.], [3., 0.], [3., 0.], [2., 0.]]])
        np.testing.assert_allclose(bank.speeds(future), [[2., 4., 0., 2.]])

    def test_extraction_hash_and_partition_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.npz"
            rows = np.array([1, 4], dtype="<i8")
            bank.npz_new(path, rows=rows)
            manifest = {"partition": "tune", "artifact_sha256": bank.sha(path), "rows_sha256": bank.arr_sha(rows)}
            bank.json_new(str(path) + ".json", manifest)
            with self.assertRaisesRegex(ValueError, "expected train"):
                bank.verified_extraction(path, "train")
            data, _ = bank.verified_extraction(path, "tune")
            np.testing.assert_array_equal(data["rows"], rows)
            with path.open("ab") as stream:
                stream.write(b"mutation")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                bank.verified_extraction(path, "tune")

    def test_chunked_oracle_recovers_exact_valid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = np.stack([np.c_[np.arange(1,51), np.zeros(50)], np.c_[np.zeros(50), np.arange(1,51)]]).astype(np.float32)
            velocities = np.stack([np.zeros(8), np.full(8, 2.)]).astype(np.float32)
            traj, mask, _ = bank.compose(paths, velocities)
            bp = root / "bank.npz"
            bank.npz_new(bp, path_xy=paths, velocity8=velocities, traj_xy=traj[:,:,:6],
                         candidate_valid=mask[:,:,:6].all(-1), train_rows=np.array([1], dtype="<i8"))
            bm = {"partition":"train", "split_sha256":"same", "train_scenes":["train_scene"],
                  "train_rows_sha256":"train", "bank_sha256":bank.sha(bp)}
            bank.json_new(str(bp)+".json", bm)
            tp = root / "tune.npz"
            rows = np.array([100,101], dtype="<i8")
            bank.npz_new(tp, rows=rows, scene=np.array(["tune_scene"]*2), frame=np.array([30,35]),
                         path_xy=paths, path_mask=np.ones((2,50),bool), velocity8=velocities[[1,1]],
                         velocity=velocities[[1,1],:6], gt_xy=traj[:,1,:6])
            bank.json_new(str(tp)+".json", {"partition":"tune", "artifact_sha256":bank.sha(tp),
                          "rows_sha256":bank.arr_sha(rows), "split_sha256":"same", "scenes":["tune_scene"]})
            out = root / "oracle.npz"
            bank.oracle(SimpleNamespace(bank=str(bp), tune=str(tp), output=str(out), device="cpu",
                                        threads=1, rows_chunk=1, candidate_chunk=3))
            with np.load(out) as z:
                np.testing.assert_array_equal(z["winner"], [1,3])
                np.testing.assert_array_equal(z["d3_full_bank"], [0.,0.])
                np.testing.assert_array_equal(z["d3_coarse1x1"], [0.,0.])

    def test_progress6_fit_reuses_frozen_paths_and_preserves_eight_speeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = np.arange(20, dtype="<i8")
            stations = np.arange(1,51, dtype=np.float32)
            paths = np.stack([np.c_[stations, np.zeros(50)], np.c_[stations, stations*.01]]).astype(np.float32)
            rawv = np.repeat(np.linspace(.1, 4., 20, dtype=np.float32)[:,None],8,axis=1)
            rawv[:,6:] += 1.
            train = root/"train.npz"
            bank.npz_new(train, rows=rows, path_xy=np.repeat(paths[:1],20,axis=0), path_mask=np.ones((20,50),bool),
                         velocity8=rawv, path_stations=stations)
            trm = {"partition":"train", "artifact_sha256":bank.sha(train), "rows_sha256":bank.arr_sha(rows),
                   "scenes":["train_scene"], "split_sha256":"same"}
            bank.json_new(str(train)+".json", trm)
            frozen = root/"frozen.npz"
            bank.npz_new(frozen, path_xy=paths, path_stations=stations, path_support=np.array([10,10]))
            bank.json_new(str(frozen)+".json", {"bank_sha256":bank.sha(frozen), "train_rows_sha256":bank.arr_sha(rows),
                          "path_inertia":0.})
            output = root/"progress_bank.npz"
            bank.fit(SimpleNamespace(train=str(train),output=str(output),paths=2,velocities=4,path_bank=str(frozen),
                                     velocity_mode="progress6",threads=1,seed=42,max_iter=30))
            with np.load(output) as z:
                np.testing.assert_array_equal(z["path_xy"],paths)
                self.assertEqual(z["velocity8"].shape,(4,8))
                self.assertTrue((z["velocity8"] >= 0).all())
                np.testing.assert_array_equal(z["velocity8"][0],np.zeros(8))
                self.assertTrue((z["velocity8"][1:,6:] > 1.).all())
                self.assertTrue(z["candidate_valid"].all())


if __name__ == "__main__":
    unittest.main()
