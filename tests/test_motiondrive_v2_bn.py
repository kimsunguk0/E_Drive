from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_motiondrive_v2_bn import ImagesOnlyDataset, image_passes, merge_moments, recalibrate_shared_bn


def test_moment_merge_includes_between_batch_variance():
    result = merge_moments([{"count": 2, "mean": [0.], "variance": [1.]},
                            {"count": 2, "mean": [2.], "variance": [1.]}])
    assert result["mean"].tolist() == [1.]
    assert result["variance"].tolist() == [2.]


def test_images_only_loader_never_requests_targets_or_full_getitem():
    class FakeSource:
        rows = np.array([0])
        arr = {"frame": np.array([30])}
        scene_names = np.array(["scene"])
        def __getitem__(self, _):
            raise AssertionError("Full target-bearing loader must not be called")
        def _image(self, scene, camera, frame, size, jitter):
            return torch.zeros(3, 2, 2)
    value = ImagesOnlyDataset(FakeSource())[0]
    assert set(value) == {"images", "history_images", "scenario", "frame", "row"}
    assert value["images"].shape == (6, 3, 2, 2)
    assert value["history_images"].shape == (4, 3, 2, 2)


def toy_model():
    model = nn.Module()
    model.backbone_fpn = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 3, 1))
    model.config = SimpleNamespace(motion_input_mode="legacy")
    return model


def batch():
    return {"images": torch.full((2, 6, 3, 4, 4), 4.),
            "history_images": torch.full((2, 4, 3, 2, 2), 2.),
            "scenario": ["a", "b"], "frame": torch.tensor([30, 35]), "row": torch.tensor([0, 1])}


def test_recalibration_only_updates_running_stats_and_restores_momentum():
    model = toy_model().eval()
    parameters = {name: value.clone() for name, value in model.named_parameters()}
    out = recalibrate_shared_bn(model, [batch()], torch.device("cpu"))
    bn = model.backbone_fpn[0]
    assert torch.equal(bn.running_mean, torch.full((3,), 3.))
    assert torch.equal(bn.running_var, torch.zeros(3))
    assert bn.num_batches_tracked == 2
    assert bn.momentum == .1
    assert not bn.training
    assert not out["goal_or_labels_used"]
    assert all(torch.equal(v, parameters[n]) for n, v in model.named_parameters())


def test_image_recalibration_rejects_goal_or_gt_fields():
    value = batch()
    value["goal_xy"] = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="whitelist"):
        image_passes(toy_model(), value, torch.device("cpu"))
