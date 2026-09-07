import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_motiondrive_v2_fit import summary, vector_statistics


def test_fit_metrics_and_collapse_detection():
    target = np.zeros((2, 6, 2))
    target[:, :, 0] = np.arange(1, 7)
    matched = summary(target, target)
    assert matched["d3"] == 0
    assert matched["collapsed_below_0p1m_on_gt_progress_over_1m"] == 0
    collapsed = np.ones_like(target)
    diagnostic = summary(collapsed, target)
    assert diagnostic["collapsed_below_0p1m_on_gt_progress_over_1m"] == 2
    assert diagnostic["pred_adjacent_distance_mean"] == [0] * 5
    assert diagnostic["lateral_mae_by_time"] == [1] * 6


def test_gradient_cosine_handles_unused_parameters():
    value = vector_statistics([torch.tensor([1., 0.]), None],
                              [torch.tensor([-2., 0.]), torch.tensor([0.])])
    assert value["cosine"] == -1
    assert value["aux_to_plan_norm_ratio"] == 2
