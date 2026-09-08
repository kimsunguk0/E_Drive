"""Synthetic CPU regressions for opt-in shared-status query conditioning."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from models.motiondrive_v2.shared_status_query import (
    SharedCausalStatusQuery,
    install_shared_status_query,
    shared_status_query_state,
)


class _QueryContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(6, 32, bias=False)
        self.fail = False

    def forward(self, value):
        if self.fail:
            raise RuntimeError("injected query failure")
        return self.proj(value)


class _ToyParent(nn.Module):
    """The original forward and planner never accept provided status."""

    def __init__(self):
        super().__init__()
        self.scene_encoder = nn.Module()
        self.scene_encoder.query_context = _QueryContext()
        self.key = nn.Parameter(torch.randn(1, 1, 3, 32))

    def forward(self, images, history_images, lidar2img, history_transforms,
                time_offsets, goal_xy):
        query = self.scene_encoder.query_context(images)
        score = (query[:, :, None] * self.key).sum(-1)
        attention = score.softmax(-1)
        scene = (attention[..., None] * history_images).sum(2)
        return {"query": query, "scene": scene, "plan": scene[..., :2]}


def _inputs(batch=2, cells=4):
    query_source = torch.randn(batch, cells, 6)
    image_values = torch.randn(batch, cells, 3, 32)
    unused = torch.empty(0)
    original = (query_source, image_values, unused, unused, unused, unused)
    return original, torch.randn(batch, 5)


def test_initial_installed_outputs_equal_original_parent_exactly():
    torch.manual_seed(20260908)
    parent = _ToyParent().eval()
    original, status = _inputs()
    expected = parent(*original)
    torch.manual_seed(0)
    zero = install_shared_status_query(copy.deepcopy(parent)).eval()
    torch.manual_seed(0)
    provided = install_shared_status_query(copy.deepcopy(parent)).eval()
    assert shared_status_query_state(zero).keys() == shared_status_query_state(provided).keys()
    for name, value in shared_status_query_state(zero).items():
        torch.testing.assert_close(value, shared_status_query_state(provided)[name], rtol=0, atol=0)
    for model, supplied in ((zero, torch.zeros_like(status)), (provided, status)):
        actual = model(*original, supplied)
        assert actual.keys() == expected.keys()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_nonzero_query_delta_is_status_sensitive_without_planner_status_argument():
    torch.manual_seed(3)
    model = install_shared_status_query(_ToyParent()).eval()
    original, status = _inputs()
    with torch.no_grad():
        model.shared_status_query_fusion.status_mlp[-1].weight.fill_(0.02)
    positive = model(*original, status)
    negative = model(*original, -status)
    assert not torch.equal(positive["query"], negative["query"])
    assert not torch.equal(positive["plan"], negative["plan"])


def test_zero_image_values_zero_scene_but_not_a_whole_image_zero_claim():
    torch.manual_seed(5)
    model = install_shared_status_query(_ToyParent()).eval()
    original, status = _inputs()
    with torch.no_grad():
        model.shared_status_query_fusion.status_mlp[-1].bias.fill_(0.25)
    query_source, image_values, *rest = original
    result = model(query_source, torch.zeros_like(image_values), *rest, status)
    assert torch.count_nonzero(result["query"]) > 0
    assert torch.count_nonzero(result["scene"]) == 0
    # This proves only the attention value algebra in this fixture.  It is not
    # a claim that a real backbone maps a zero-valued image to exact zero.


def test_query_contract_rejects_bad_shape_device_and_nonfinite():
    fusion = SharedCausalStatusQuery()
    query = torch.randn(2, 7, 32)
    status = torch.randn(2, 5)
    assert fusion(query, status).shape == query.shape
    with pytest.raises(ValueError, match="B,Q,32"):
        fusion(query[..., :31], status)
    with pytest.raises(ValueError, match="B,5"):
        fusion(query, status[:, :4])
    changed = status.clone()
    changed[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fusion(query, changed)
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="one device"):
            fusion(query.cuda(), status)


def test_context_is_cleared_after_failure_and_reentrancy_is_rejected():
    model = install_shared_status_query(_ToyParent()).eval()
    original, status = _inputs()
    model.scene_encoder.query_context.fail = True
    with pytest.raises(RuntimeError, match="injected query failure"):
        model(*original, status)
    assert model._shared_status_query_context["status"] is None
    model.scene_encoder.query_context.fail = False
    assert model(*original, status)["plan"].shape == (2, 4, 2)
    assert model._shared_status_query_context["status"] is None
    with pytest.raises(ValueError, match="already installed"):
        install_shared_status_query(model)

    model._shared_status_query_context["status"] = status
    with pytest.raises(RuntimeError, match="reentrant"):
        model(*original, status)
    model._shared_status_query_context["status"] = None


def test_install_rejects_a1_fusion_stacking():
    model = _ToyParent()
    model.add_module("shared_status_fusion", nn.Identity())
    with pytest.raises(ValueError, match="cannot stack with A1"):
        install_shared_status_query(model)
