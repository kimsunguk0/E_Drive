"""Independent reviewer regressions for the opt-in shared-status A1 helpers."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.motiondrive_v2.shared_status import (  # noqa: E402
    SharedCausalStatusFiLM,
    install_shared_status,
    shared_status_state,
)
from scripts.motiondrive_v2_data import full_pose_matrices  # noqa: E402
from scripts.motiondrive_v2_shared_status_data import (  # noqa: E402
    NOMINAL_FRAMES,
    causal_status5_from_pose_matrices,
    causal_status5_from_pose_records,
    shared_status_model_inputs,
)


class _Refine(nn.Module):
    def __init__(self):
        super().__init__()
        self.fail = False
        self.last_input = None

    def forward(self, value):
        self.last_input = value.detach().clone()
        if self.fail:
            raise RuntimeError("injected refine failure")
        return value + 0.25


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scene_encoder = nn.Module()
        self.scene_encoder.refine = _Refine()

    def forward(self, images, history_images, lidar2img, history_transforms,
                time_offsets, goal_xy):
        del history_images, lidar2img, history_transforms, time_offsets, goal_xy
        raster = images[:, :128]
        refined = self.scene_encoder.refine(raster)
        # Stand in for existing shared auxiliary and planning consumers.
        return {"occ": refined.mean((1, 2, 3)),
                "lane": refined.square().mean((1, 2, 3)),
                "plan": refined.flatten(1)[:, :12].reshape(-1, 6, 2)}


def _args(scene):
    batch = len(scene)
    return (scene, torch.empty(batch, 0), torch.empty(batch, 0),
            torch.empty(batch, 0), torch.empty(batch, 0), torch.empty(batch, 2))


def _pose_records(speed=2.0, acceleration=0.5):
    records = []
    for frame in (*NOMINAL_FRAMES, 1, 50):
        t = frame / 10.0
        records.append({"frame": frame,
                        "x": speed * t + 0.5 * acceleration * t * t,
                        "y": 0.2 * t, "z": 0.0,
                        "roll": 0.0, "pitch": 0.0, "yaw": 0.1 * t})
    return records


def test_initial_identity_same_state_and_zero_scene_has_no_status_value():
    torch.manual_seed(29)
    parent = _ToyModel()
    scene = torch.randn(2, 128, 8, 8)
    baseline = parent(*_args(scene))

    torch.manual_seed(101)
    zero = install_shared_status(copy.deepcopy(parent))
    torch.manual_seed(101)
    provided = install_shared_status(copy.deepcopy(parent))
    assert shared_status_state(zero).keys() == shared_status_state(provided).keys()
    for name, value in shared_status_state(zero).items():
        torch.testing.assert_close(value, shared_status_state(provided)[name], rtol=0, atol=0)

    status = torch.randn(2, 5)
    for model, supplied in ((zero, torch.zeros_like(status)), (provided, status)):
        result = model(*_args(scene), supplied)
        for key in baseline:
            torch.testing.assert_close(result[key], baseline[key], rtol=0, atol=0)

    film = SharedCausalStatusFiLM().eval()
    empty = torch.zeros(2, 128, 8, 8, dtype=torch.bfloat16)
    output = film(empty, torch.randn(2, 5))
    assert output.dtype == empty.dtype
    assert torch.count_nonzero(output) == 0


def test_zero_arm_can_learn_constant_gain_and_provided_weights_activate():
    torch.manual_seed(7)
    film = SharedCausalStatusFiLM()
    scene = torch.randn(3, 128, 3, 3)
    zero = torch.zeros(3, 5)
    film(scene, zero).square().mean().backward()
    last = film.status_mlp[-1]
    assert last.bias.grad is not None and torch.count_nonzero(last.bias.grad) > 0

    optimizer = torch.optim.SGD(film.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    status = torch.randn(3, 5)
    film(scene, status).square().mean().backward()
    first = film.status_mlp[0]
    assert first.weight.grad is not None and torch.count_nonzero(first.weight.grad) > 0
    assert not torch.equal(film(scene, status[:1].expand(3, -1)),
                           film(scene, -status[:1].expand(3, -1)))


def test_pose_status_is_nominal_causal_record_matrix_equal_and_fails_missing():
    records = _pose_records()
    from_records = causal_status5_from_pose_records(records)
    causal_rows = [next(row for row in records if row["frame"] == frame)
                   for frame in NOMINAL_FRAMES]
    xyz = np.asarray([[row[key] for key in ("x", "y", "z")] for row in causal_rows])
    rpy = np.asarray([[row[key] for key in ("roll", "pitch", "yaw")] for row in causal_rows])
    from_matrices = causal_status5_from_pose_matrices(full_pose_matrices(xyz, rpy))
    np.testing.assert_array_equal(from_records, from_matrices)

    mutated = copy.deepcopy(records)
    for row in mutated:
        if row["frame"] in (1, 50):
            row.update(x=1e9, y=-1e9, yaw=2.5)
    np.testing.assert_array_equal(from_records, causal_status5_from_pose_records(mutated))
    with pytest.raises(ValueError, match="exact causal pose frames"):
        causal_status5_from_pose_records([row for row in records if row["frame"] != -7])


def test_model_input_adapter_ignores_unrelated_ground_truth_mutations():
    def base_inputs(batch, **_kwargs):
        return {key: batch[key] for key in ("images", "goal_xy")}

    batch = {"images": torch.randn(2, 6, 3, 2, 2),
             "goal_xy": torch.randn(2, 2),
             "provided_status5": torch.randn(2, 5),
             "state_target": torch.randn(2, 6),
             "future_target": torch.randn(2, 6, 2)}
    expected = shared_status_model_inputs(base_inputs, batch)
    changed = dict(batch, state_target=torch.full((2, 6), 9999.0),
                   future_target=torch.full((2, 6, 2), -9999.0))
    actual = shared_status_model_inputs(base_inputs, changed)
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_refine_hook_uses_per_call_context_and_clears_after_exception():
    model = install_shared_status(_ToyModel())
    scene = torch.randn(1, 128, 8, 8)
    status = torch.randn(1, 5)
    model.scene_encoder.refine.fail = True
    with pytest.raises(RuntimeError, match="injected refine failure"):
        model(*_args(scene), status)
    assert model._shared_status_context["status"] is None

    model.scene_encoder.refine.fail = False
    result = model(*_args(scene), status)
    assert result["plan"].shape == (1, 6, 2)
    assert model._shared_status_context["status"] is None
    with pytest.raises(ValueError, match="already installed"):
        install_shared_status(model)

