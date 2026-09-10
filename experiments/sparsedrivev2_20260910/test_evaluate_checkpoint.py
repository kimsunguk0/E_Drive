"""CPU contracts for approval, immutable candidates, and training-eval parity."""
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

import data
import train
import evaluate_checkpoint as evaluator


class TinyDataset(Dataset):
    rows = np.arange(3, dtype=np.int64)

    def __len__(self):
        return 3

    def __getitem__(self, index):
        t = torch.arange(1., 7.)
        gt = torch.stack((index + .1 * (index + 1) * t,
                          index - .07 * t.square()), -1)
        return {"images": torch.full((3, 3, 2, 2), float(index)),
                "lidar2img": torch.eye(4).repeat(3, 1, 1), "image_hw": torch.tensor([2., 2.]),
                "status": torch.zeros(8), "gt_plan": gt,
                "row": index, "scenario": f"scene{index}", "session": f"session{index//2}"}


class TinySelector(nn.Module):
    def __init__(self):
        super().__init__()
        bank = torch.arange(4.).reshape(2, 2, 1, 1).expand(2, 2, 8, 3).clone()
        self._trajectory_head = SimpleNamespace(traj_vocab=bank, traj_mask=torch.ones(2, 2, 8))

    def forward(self, images, lidar2img, image_hw, status):
        batch = len(images)
        chosen = images[:, 0, 0, 0, 0].long()
        ids = torch.arange(4).expand(batch, -1)
        scores = -(ids - chosen[:, None]).float().abs()
        bank = self._trajectory_head.traj_vocab.flatten(0, 1)
        pids = torch.arange(2).expand(batch, -1)
        stage = {"path_ids": pids, "velocity_ids": pids,
                 "path_scores": torch.zeros(batch, 2), "velocity_scores": torch.zeros(batch, 2)}
        return {"trajectory": bank[chosen, :6, :2], "candidate_xy": bank[ids, :6, :2],
                "candidate_ids": ids, "selected_candidate_id": chosen,
                "candidate_valid": torch.ones(batch, 4, dtype=torch.bool), "scores": scores,
                "path_ids": pids, "velocity_ids": pids, "coarse": [stage, stage]}


class EvaluatorContracts(unittest.TestCase):
    def test_approval_must_exist_and_be_exact(self):
        expected = {"checkpoint_sha256": "checkpoint", "bank_sha256": "bank",
                    "train_rows_sha256": "train", "eval_rows_sha256": "held", "split_sha256": "split"}
        receipt = {"schema": "sparsedrivev2_confirmation_candidate_v1", "approved": True,
                   "frozen": True, "population": "confirmation12", **expected}
        evaluator.check_approval(receipt, expected)
        for key, replacement in (("approved", False), ("frozen", False), ("approved", 1),
                                 ("population", "tune"), ("checkpoint_sha256", "other"),
                                 ("bank_sha256", "other"), ("train_rows_sha256", "other"),
                                 ("eval_rows_sha256", "other"), ("split_sha256", "other")):
            with self.subTest(key=key, replacement=replacement), self.assertRaises(ValueError):
                evaluator.check_approval({**receipt, key: replacement}, expected)

    def test_missing_confirmation_approval_stops_before_checkpoint_load(self):
        with tempfile.TemporaryDirectory() as root:
            cp = Path(root) / "not_a_real_checkpoint"
            cp.write_bytes(b"synthetic guard fixture, never a model")
            with patch.object(torch, "load") as load, self.assertRaisesRegex(ValueError, "requires explicit"):
                evaluator.inspect_checkpoint(cp, population="confirmation12")
            load.assert_not_called()

    def test_rows_are_integer_sorted_unique(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "rows.npy"
            for rows in (np.array([1., 2.]), np.array([2, 1]), np.array([1, 1]), np.array([], np.int64)):
                np.save(path, rows)
                with self.assertRaises(ValueError):
                    evaluator.read_rows(path)
            np.save(path, np.array([1, 2], np.int64))
            np.testing.assert_array_equal(evaluator.read_rows(path), [1, 2])

    def test_cpu_parity_with_current_train_evaluate_and_coarse_dump(self):
        model, dataset = TinySelector(), TinyDataset()
        runtime = SimpleNamespace(train=train, data=data)
        args = SimpleNamespace(eval_batch=2, workers=0, precision="fp32", goal_mode="none")
        with tempfile.TemporaryDirectory() as root:
            with patch.object(train, "transfer", side_effect=lambda value: value), \
                 patch.object(train, "autocast", side_effect=lambda _: nullcontext()):
                expected = train.evaluate(model, dataset, args, Path(root), 5)
            arrays, result = evaluator.evaluate_model(model, dataset, runtime, device="cpu",
                                                      precision="fp32", batch_size=2, workers=0,
                                                      dump_coarse=True)
            parity = evaluator.check_reference(arrays, Path(root) / "eval_000005.npz")
            self.assertIn("pred", parity["bitwise_equal_keys"])
            self.assertIn("coarse0_path_ids", arrays)
            self.assertIn("coarse1_velocity_scores", arrays)
            self.assertGreater(result["official_d3"], 0)
            for key in ("official_d3", "shortlist_oracle_d3", "selection_regret", "point_l2_metres_05_to_30", "prefix_ade_1_2_3s"):
                self.assertEqual(result[key], expected[key])

    def test_modified_candidate_coordinates_are_rejected(self):
        model, dataset = TinySelector(), TinyDataset()
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
        output = model(**data.model_inputs(batch))
        output["candidate_xy"] = output["candidate_xy"].clone()
        output["candidate_xy"][0, 0, 0, 0] += .01
        with self.assertRaisesRegex(ValueError, "not fixed bank rows"):
            evaluator.verify_bank_output(model, output, SimpleNamespace(train=train))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
