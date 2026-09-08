from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from models.motiondrive_v2_late_state_residual import (
    LateResidualP7Adapter, LateStateResidual, compact22)
from scripts import run_motiondrive_v2_p9_late_state_residual as p9


def test_module_shape_time_scale_and_exact_zero_mapping():
    module = LateStateResidual()
    decoded, status = torch.randn(3, 6, 128), torch.randn(3, 22)
    assert tuple(module.waypoint_time_over_3.tolist()) == pytest.approx(
        (1/6, 2/6, 3/6, 4/6, 5/6, 1.))
    assert module.mlp[0].in_features == 151
    assert torch.count_nonzero(module.mlp[-1].weight) == 0
    assert torch.count_nonzero(module.mlp[-1].bias) == 0
    assert torch.equal(module(decoded, status), torch.zeros(3, 6, 2))


def test_complete_branch_is_fp32_under_outer_bf16_autocast_and_backward_finite():
    module = LateStateResidual()
    with torch.no_grad():
        module.mlp[-1].weight.normal_()
    seen = []
    handles = [layer.register_forward_hook(lambda _m, _i, out: seen.append(out.dtype))
               for layer in (module.image_norm, module.mlp[0], module.mlp[2], module.mlp[4])]
    decoded = torch.randn(2, 6, 128, requires_grad=True)
    status = torch.randn(2, 22, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = module(decoded, status)
    for handle in handles: handle.remove()
    assert output.dtype == torch.float32 and seen == [torch.float32] * 4
    output.square().mean().backward()
    assert torch.isfinite(decoded.grad).all() and torch.isfinite(status.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in module.parameters())


def test_three_compact_arms_are_same_shape_and_gt21_preserves_raw_stop():
    pred_state, pred_history = torch.randn(4, 6), torch.randn(4, 4, 4)
    gt_state, gt_history = torch.randn(4, 6), torch.randn(4, 4, 4)
    a = compact22(pred_state, pred_history, arm="no_compact")
    b = compact22(pred_state, pred_history, arm="predicted22")
    c = compact22(pred_state, pred_history, arm="gt21", gt_state=gt_state,
                  gt_history=gt_history)
    assert a.shape == b.shape == c.shape == (4, 22) and not torch.count_nonzero(a)
    assert torch.equal(b[:, :6], pred_state)
    assert torch.equal(c[:, :5], gt_state[:, :5])
    assert torch.equal(c[:, 5], pred_state[:, 5])
    assert torch.equal(c[:, 6:], gt_history.flatten(1))


class DummyPlanner(nn.Module):
    def __init__(self):
        super().__init__(); self.xy_head = nn.Linear(128, 2, bias=False)
        with torch.no_grad():
            self.xy_head.weight.zero_()
            self.xy_head.weight[0, 0] = 1.
            self.xy_head.weight[1, 1] = 1.


class DummyParent(nn.Module):
    def __init__(self):
        super().__init__(); self.planner = DummyPlanner(); self.config = SimpleNamespace(plan_output_scale=(1., 1.))
        self.parts_calls = 0; self.plan_calls = 0
    def forward_parts(self, *args):
        self.parts_calls += 1
        b = len(args[0]); return {"scene_features": torch.randn(b, 2, 128),
            "motion_features": torch.randn(b, 3, 128), "state_hat": torch.randn(b, 6),
            "history_hat": torch.randn(b, 4, 4)}
    def plan_from_features(self, scene, motion, state, history):
        self.plan_calls += 1
        decoded = torch.arange(len(state) * 6 * 128, dtype=torch.float32).reshape(len(state), 6, 128)
        return self.planner.xy_head(decoded)[..., :2]


def test_capture_is_actual_existing_head_input_and_reconstructs_plan():
    parent = DummyParent(); parts = parent.forward_parts(torch.zeros(2, 1))
    decoded, plan = p9.capture_decoder_features(parent, parts)
    assert parent.plan_calls == 1 and decoded.shape == (2, 6, 128)
    assert torch.equal(decoded[..., :2], plan)
    assert not parent.planner.xy_head._forward_pre_hooks


def test_raw_image_adapter_preserves_full_parent_and_rejects_privileged_c():
    parent = DummyParent(); adapter = LateResidualP7Adapter(parent, "predicted22")
    adapter.train()
    assert not adapter.parent.training and adapter.residual.training
    assert all(not parameter.requires_grad for parameter in adapter.parent.parameters())
    inputs = (torch.zeros(2, 1),) * 6
    output = adapter(*inputs)
    assert parent.parts_calls == parent.plan_calls == 1
    assert set(output) == {"scene_features", "motion_features", "state_hat",
                           "history_hat", "plan_abs"}
    assert torch.equal(output["plan_abs"], torch.arange(2 * 6 * 128).reshape(2, 6, 128)[..., :2])
    assert not parent.planner.xy_head._forward_pre_hooks
    with pytest.raises(ValueError, match="only legal"):
        LateResidualP7Adapter(parent, "gt21")


def test_batch_join_rejects_identity_and_mask_changes():
    raw = {"row": torch.tensor([3]), "frame": torch.tensor([30]), "scenario": ["x"],
           "session_id": ["s"], "state_target": torch.zeros(1, 6),
           "history_target": torch.zeros(1, 4, 4), "gt_plan": torch.zeros(1, 6, 2),
           "state_valid": torch.ones(1, 6, dtype=torch.bool),
           "history_valid": torch.ones(1, 4, 4, dtype=torch.bool)}
    labels = {"rows": raw["row"], "frame": raw["frame"], "scenario": ["x"], "session": ["s"],
              **{key: raw[key] for key in ("state_target", "history_target", "gt_plan",
                                           "state_valid", "history_valid")}}
    pred = {"rows": raw["row"]}
    assert p9.validate_batch(raw, labels, pred, 0) == slice(0, 1)
    labels["history_valid"] = labels["history_valid"].clone(); labels["history_valid"][0, 0, 0] = False
    with pytest.raises(ValueError, match="frozen GT"):
        p9.validate_batch(raw, labels, pred, 0)


def test_fixed_recipe_steps_and_scales():
    module = LateStateResidual()
    assert p9.STEPS == 3240 and p9.ARMS == ("no_compact", "predicted22", "gt21")
    assert tuple(module.compact_scale.tolist()) == (
        10., 5., 3., 3., .5, 1., 10., 5., 1., 1., 10., 5., 1., 1.,
        10., 5., 1., 1., 10., 5., 1., 1.)
    closure = p9.validate_p9_sources()
    assert set(closure) == {
        "models/motiondrive_v2_late_state_residual.py",
        "scripts/analyze_motiondrive_v2_query_pair.py",
        "scripts/motiondrive_v2_training.py",
        "scripts/run_motiondrive_v2_p7_state_sufficiency.py",
    }


def test_train_has_same_init_order_one_terminal_eval_and_per_row_metric(monkeypatch):
    monkeypatch.setattr(p9, "EPOCHS", 2); monkeypatch.setattr(p9, "BATCH_SIZE", 2)
    monkeypatch.setattr(p9, "STEPS", 4)
    monkeypatch.setitem(p9.probe.ROWS, "tune", (4, 5, "x"))
    features, baseline, target = torch.randn(4, 6, 128), torch.randn(4, 6, 2), torch.randn(4, 6, 2)
    left_model, left = p9.train_residual(features, torch.zeros(4, 22), baseline, target,
        (features, torch.zeros(4, 22), baseline, target), seed=0)
    right_model, right = p9.train_residual(features, torch.randn(4, 22), baseline, target,
        (features, torch.randn(4, 22), baseline, target), seed=0)
    assert left["initial_model_state_sha256"] == right["initial_model_state_sha256"]
    assert left["sample_order_sha256"] == right["sample_order_sha256"]
    assert left["steps"] == right["steps"] == 4
    assert left["terminal_tune_forward_calls"] == right["terminal_tune_forward_calls"] == 2
    assert left["tune_d3"].shape == right["tune_d3"].shape == (4,)
    assert all(torch.isfinite(p).all() for p in left_model.parameters())


def test_shared_session_comparison_and_gate_are_fixed():
    sessions = np.asarray([f"s{i}" for i in range(11) for _ in range(2)])
    right = {0: np.ones(22), 1: np.ones(22)}
    left = {0: np.full(22, .98), 1: np.full(22, .98)}
    result = p9.compare_rows(left, right, sessions, -.01)
    assert result["mean_delta_d3"] == pytest.approx(-.02)
    assert result["shared_11_session_bootstrap"]["clusters"] == 11
    assert result["shared_11_session_bootstrap"]["repeats"] == 10000
    assert result["shared_11_session_bootstrap"]["seed"] == 20260908
    assert all(result["gate"].values())
    assert p9.compare_rows(left, right, sessions)["gate"] is None


def test_no_gate_json_round_trip_and_source_tamper_fail_closed(tmp_path, monkeypatch):
    sessions = np.asarray([f"s{i}" for i in range(11)])
    result = p9.compare_rows({0: np.zeros(11), 1: np.zeros(11)},
                             {0: np.ones(11), 1: np.ones(11)}, sessions)
    path = tmp_path / "result.json"
    p9.probe.atomic_json(path, result)
    assert p9.probe.read_json(path)["gate"] is None
    monkeypatch.setattr(p9, "MODULE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="source closure"):
        p9.validate_p9_sources()


def test_cli_has_only_train_tune_extract_and_one_aggregate_fit():
    source = open(p9.__file__).read()
    assert 'choices=("train", "tune")' in source and "final" not in p9.probe.ROWS
    assert 'for seed in (0, 1):' in source and 'for arm in ARMS:' in source
    assert '"single_terminal_tune_pass": True' in source
    assert '"per_row_outputs_saved": True' in source
