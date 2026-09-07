import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_motiondrive_v2_aux import (Accumulator, confusion_report, correlation, donor_mapping, evaluate_checkpoint,
                                         fixed_frames, history_condition)


def test_fixed_sampling_and_different_scene_donors():
    assert fixed_frames() == [30, 55, 85, 115, 145, 175, 205, 235, 265, 295]
    mapping = donor_mapping(["scene3", "scene1", "scene2"])
    assert all(source != donor for source, donor in mapping.items())
    assert set(mapping) == set(mapping.values())
    with pytest.raises(ValueError):
        donor_mapping(["one"])


def test_temporal_perturbations_leave_other_inputs_and_labels_unchanged():
    b = {"images": torch.randn(2, 6, 3, 16, 20), "history_images": torch.randn(2, 4, 3, 8, 10),
         "lidar2img": torch.randn(2, 6, 4, 4), "history_transforms": torch.randn(2, 4, 4, 4),
         "time_offsets": torch.tensor([[.1, .2, .5, 1.]]).repeat(2, 1), "goal_xy": torch.randn(2, 2),
         "donor_history_images": torch.randn(2, 4, 3, 8, 10), "state_target": torch.randn(2, 6)}
    for condition in ("normal", "repeat_current", "reverse_history", "cross_scene_history"):
        out = history_condition(b, condition)
        assert "state_target" not in out
        for k in ("images", "lidar2img", "history_transforms", "time_offsets", "goal_xy"):
            assert out[k] is b[k]
        if condition == "repeat_current":
            torch.testing.assert_close(out["history_images"][:, 0], out["history_images"][:, 3])
        if condition == "reverse_history":
            torch.testing.assert_close(out["history_images"][:, 0], b["history_images"][:, 3])
        if condition == "cross_scene_history":
            assert out["history_images"] is b["donor_history_images"]


def test_physical_metrics_and_stop_threshold():
    acc = Accumulator()
    state = torch.tensor([[1., 0, 0, 0, 0, -3], [3., 0, 0, 0, 0, 3]])
    history = torch.zeros(2, 4, 4)
    history[..., 3] = 1.
    target = torch.zeros_like(state)
    target[:, 5] = torch.tensor([0., 1.])
    gt_history = history.clone()
    gt_history[..., 0] = 2.
    batch = {"state_target": target, "state_valid": torch.ones_like(state, dtype=torch.bool),
             "history_target": gt_history, "history_valid": torch.ones_like(history, dtype=torch.bool),
             "scenario": ["a", "b"], "session_id": ["s1", "s2"], "frame": torch.tensor([30, 30])}
    acc.add({"state_hat": state, "history_hat": history}, batch, include_raster=False)
    r = acc.finish()
    assert r["state"]["vx_m_s"]["mae"] == 2.
    assert r["state"]["vx_m_s"]["rmse"] == pytest.approx(np.sqrt(5))
    assert r["history_position_mae_mean4_m"] == 2.
    assert r["stop"]["precision"] == 1. and r["stop"]["recall"] == 1.
    assert r["history"][0]["yaw_mae_rad"] == 0.


def test_empty_prediction_precision_is_not_invented():
    r = confusion_report(0, 0, 3, 7)
    assert r["precision"] is None
    assert r["recall"] == 0.
    assert r["accuracy"] == .7
    assert correlation(np.full(370, .173, dtype=np.float32), np.arange(370)) is None


def test_wrong_expected_step_fails_before_gpu(monkeypatch):
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"step": 750})
    with pytest.raises(ValueError, match="checkpoint step 불일치"):
        evaluate_checkpoint(Path("last.pth"), None, SimpleNamespace(expected_step=1000), "split", "supervision")
