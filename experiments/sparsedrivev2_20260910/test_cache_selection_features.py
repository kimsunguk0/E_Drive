"""CPU-only synthetic tests for cache schema and the GT/feature boundary."""
from pathlib import Path
from contextlib import nullcontext
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
import cache_selection_features as cache


class ToyDataset(Dataset):
    rows = np.arange(3, dtype=np.int64)
    scen_idx = np.arange(3, dtype=np.int64)
    scenarios = np.asarray(["a", "b", "c"])
    manifest = {"scene_to_session": {"a": "s0", "b": "s0", "c": "s1"}}

    def __init__(self, gt_shift=0.):
        self.gt_shift = gt_shift

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {"images": torch.zeros(3, 3, 2, 2), "lidar2img": torch.eye(4).repeat(3, 1, 1),
                "image_hw": torch.tensor([2., 2.]), "status": torch.tensor([0., 0., 0., 0., 2., 0., 0., 0.]),
                "goal_xy": torch.tensor([10., 1.]), "gt_plan": torch.full((6, 2), .2 + self.gt_shift),
                "row": index, "scenario": self.scenarios[index], "session": self.manifest["scene_to_session"][self.scenarios[index]]}


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        bank = (torch.arange(200.) / 100).reshape(20, 10, 1, 1).expand(20, 10, 8, 3).clone()
        self._trajectory_head = SimpleNamespace(traj_vocab=bank, traj_mask=torch.ones(20, 10, 8))
        self.received_goal = []

    def forward(self, images, lidar2img, image_hw, status, goal_xy=None):
        self.received_goal.append(goal_xy is not None)
        n = len(images)
        ids = torch.arange(200).expand(n, -1)
        selected = torch.zeros(n, dtype=torch.int64)
        bank = self._trajectory_head.traj_vocab.flatten(0, 1)
        return {"candidate_ids": ids, "candidate_xy": bank[ids, :6, :2],
                "selected_candidate_id": selected, "trajectory": bank[selected, :6, :2],
                "scores": -ids.float(), "candidate_valid": torch.ones(n, 200, dtype=torch.bool)}


def feature_builder(output, status, goal_xy):
    if set(output) != {"candidate_xy", "scores", "candidate_valid"}:
        raise AssertionError("Unexpected feature input; labels must stay outside this dictionary")
    n, k = output["scores"].shape
    feature = torch.zeros(n, k, 32)
    feature[..., :12] = output["candidate_xy"].flatten(-2)
    feature[..., 12:20] = status[:, None]
    feature[..., 20:22] = goal_xy[:, None]
    feature[..., 22] = output["scores"]
    return feature


class CacheContracts(unittest.TestCase):
    def test_labels_change_costs_without_changing_features_or_base_goal_routing(self):
        runtime = SimpleNamespace(data=data, train=train)
        plan = SimpleNamespace(goal_mode="none", train_rows=np.arange(3),
                               receipt={"checkpoint_sha256": "checkpoint", "bank_sha256": "bank"},
                               manifest={"arguments": {"status_mode": "causal_selection"}})
        meta = {"status_source": {"sha256": "causal_status"}, "ego_cache_sha256": "ego"}
        model = ToyModel()
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for i, shift in enumerate((0., 10.)):
                path = Path(root) / str(i)
                paths.append(path)
                dataset = ToyDataset(shift)
                record, artifacts = cache.cache_partition(path, "train", (dataset, dataset, dataset.rows, meta),
                                                           model, runtime, plan, feature_builder,
                                                           batch_size=2, workers=0, precision="fp32", device="cpu")
                self.assertEqual(record["rows"], 3)
                self.assertEqual(artifacts["train/features.npy"]["shape"], [3, 200, 32])
                self.assertEqual(artifacts["train/candidate_ids.npy"]["dtype"], "int64")
                self.assertEqual(artifacts["train/gt_xy.npy"]["dtype"], "float32")
            for name in ("features", "base_logits", "candidate_xy", "candidate_ids", "valid"):
                np.testing.assert_array_equal(np.load(paths[0] / "train" / f"{name}.npy"),
                                              np.load(paths[1] / "train" / f"{name}.npy"))
            self.assertFalse(np.array_equal(np.load(paths[0] / "train/costs.npy"), np.load(paths[1] / "train/costs.npy")))
            self.assertFalse(any(model.received_goal))

    def test_writer_rejects_incomplete_cache_and_existing_partition(self):
        with tempfile.TemporaryDirectory() as root:
            writer = cache.PartitionWriter(root, "train", np.arange(2), ["a", "b"], ["s", "s"])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                writer.finish({})
            with self.assertRaises(FileExistsError):
                cache.PartitionWriter(root, "train", np.arange(2), ["a", "b"], ["s", "s"])
            with self.assertRaisesRegex(ValueError, "Only train/tune"):
                cache.PartitionWriter(root, "held", np.arange(2), ["a", "b"], ["s", "s"])

    def test_tune_reference_requires_bitwise_training_evaluation_parity(self):
        runtime = SimpleNamespace(data=data, train=train)
        plan = SimpleNamespace(goal_mode="none", train_rows=np.arange(3),
                               receipt={"checkpoint_sha256": "checkpoint", "bank_sha256": "bank"},
                               manifest={"arguments": {"status_mode": "causal_selection"}})
        meta = {"status_source": {"sha256": "causal_status"}, "ego_cache_sha256": "ego"}
        dataset, model = ToyDataset(), ToyModel()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            args = SimpleNamespace(eval_batch=2, workers=0, precision="fp32", goal_mode="none")
            with patch.object(train, "transfer", side_effect=lambda x: x), \
                 patch.object(train, "autocast", side_effect=lambda _: nullcontext()):
                train.evaluate(model, dataset, args, root, 2000)
            record, _ = cache.cache_partition(root / "cache", "tune", (dataset, dataset, dataset.rows, meta),
                                              model, runtime, plan, feature_builder, batch_size=2, workers=0,
                                              precision="fp32", device="cpu", reference_predictions=root / "eval_002000.npz")
            self.assertEqual(set(record["reference_parity"]["bitwise_equal_keys"]),
                             {"rows", "pred", "d3", "shortlist_oracle", "candidate_id", "error_xy", "point_l2"})


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
