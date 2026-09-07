import sys
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from motiondrive_v2_training import (LossWeights, MODEL_INPUTS, balanced_raster_bce,
                                     compute_loss, model_inputs, set_training_mode,
                                     tensor_state_sha256, weighted_d3)


def test_official_metric_matches_cumulative_definition():
    pred = torch.randn(8, 6, 2)
    target = torch.randn(8, 6, 2)
    d = (pred - target).norm(dim=-1)
    reference = (d[:, :2].mean(1) + d[:, :4].mean(1) + d.mean(1)) / 3
    torch.testing.assert_close(weighted_d3(pred, target), reference)


def test_whitelist_excludes_ground_truth():
    batch = {key: torch.zeros(1) for key in MODEL_INPUTS}
    batch.update(gt_plan=torch.ones(1), state_target=torch.ones(1), supplied_status=torch.ones(1))
    assert set(model_inputs(batch)) == set(MODEL_INPUTS)


def test_empty_mask_bce_is_zero_with_finite_gradient():
    logits = torch.randn(2, 1, 3, 4, requires_grad=True)
    loss = balanced_raster_bce(logits, torch.zeros_like(logits), torch.zeros_like(logits, dtype=torch.bool))
    loss.backward()
    assert loss == 0
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() == 0


def test_loss_all_heads_receive_gradients_and_state_mask_works():
    out = {
        "plan_abs": torch.randn(2, 6, 2, requires_grad=True),
        "occ_logits": torch.randn(2, 1, 3, 4, requires_grad=True),
        "lane_logits": torch.randn(2, 1, 3, 4, requires_grad=True),
        "history_hat": torch.randn(2, 4, 4, requires_grad=True),
        "state_hat": torch.randn(2, 6, requires_grad=True),
    }
    batch = {
        "gt_plan": torch.zeros(2, 6, 2), "plan_valid": torch.ones(2, 6, dtype=torch.bool),
        "occ_target": torch.zeros(2, 1, 3, 4), "occ_valid": torch.ones(2, 1, 3, 4, dtype=torch.bool),
        "lane_target": torch.zeros(2, 1, 3, 4), "lane_valid": torch.ones(2, 1, 3, 4, dtype=torch.bool),
        "history_target": torch.zeros(2, 4, 4), "history_valid": torch.ones(2, 4, 4, dtype=torch.bool),
        "state_target": torch.zeros(2, 6), "state_valid": torch.ones(2, 6, dtype=torch.bool),
    }
    loss, parts = compute_loss(out, batch, LossWeights())
    loss.backward()
    for name, tensor in out.items():
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.abs().sum() > 0, name
    assert set(parts) >= {"plan_d3", "occ_bce", "lane_bce", "motion"}


def test_zero_error_has_finite_gradient():
    pred = torch.zeros(2, 6, 2, requires_grad=True)
    weighted_d3(pred, torch.zeros_like(pred)).mean().backward()
    assert torch.isfinite(pred.grad).all()


def test_eval_checkpoint_restores_non_tensor_flags():
    from train_motiondrive_v2 import restore_run_configuration
    args = SimpleNamespace(resume=None, goal_on=1, state_on=1, arch="resnet50", phase="joint")
    saved = {"model_config": {"goal_on": False, "state_on": False, "backbone_arch": "resnet34"},
             "arguments": {"phase": "pretrain"}}
    config = restore_run_configuration(args, saved, set())
    assert not config["goal_on"] and args.goal_on == 0 and args.state_on == 0
    assert args.phase == "pretrain" and args.arch == "resnet34"


def test_eval_rejects_explicit_checkpoint_flag_override():
    from train_motiondrive_v2 import restore_run_configuration
    args = SimpleNamespace(resume=None, goal_on=1, state_on=1, arch="resnet50", phase="joint")
    saved = {"model_config": {"goal_on": False, "state_on": False, "backbone_arch": "resnet50"},
             "arguments": {"phase": "joint"}}
    with pytest.raises(ValueError, match="incompatible explicit"):
        restore_run_configuration(args, saved, {"--goal-on"})


def test_init_preserves_architecture_and_only_switches_gs():
    from train_motiondrive_v2 import initialization_configuration
    saved = {"model_config": {"goal_on": False, "state_on": False,
             "backbone_arch": "resnet34", "channels": 32, "grid_size": [12, 8],
             "plan_output_scale": [10., 5.], "motion_input_mode": "low_feature"}}
    output = initialization_configuration(saved, goal_on=1, state_on=0)
    assert output["goal_on"] is True and output["state_on"] is False
    for key in ("backbone_arch", "channels", "grid_size", "plan_output_scale", "motion_input_mode"):
        assert output[key] == saved["model_config"][key]
    with pytest.raises(ValueError, match="backbone"):
        initialization_configuration(saved, goal_on=1, state_on=0, explicit_arch="resnet50")


def test_state_fingerprint_includes_scalar_bfloat_buffers_and_order_independent():
    state = {"a": torch.tensor(1), "b": torch.randn(2, 3).bfloat16()}
    original = tensor_state_sha256(state)
    assert tensor_state_sha256(dict(reversed(list(state.items())))) == original
    state["a"] += 1
    assert tensor_state_sha256(state) != original


def test_eval_restores_new_flags_and_rejects_implicit_mode_change():
    from train_motiondrive_v2 import restore_run_configuration
    args = SimpleNamespace(resume=None, goal_on=1, state_on=1, arch="resnet50", phase="joint",
                           motion_input_mode=None, plan_output_scale=None)
    saved = {"model_config": {"goal_on": True, "state_on": True, "backbone_arch": "resnet50",
                             "plan_output_scale": [10., 5.], "motion_input_mode": "low_feature"},
             "arguments": {"phase": "joint"}}
    restore_run_configuration(args, saved, set())
    assert args.motion_input_mode == "low_feature" and args.plan_output_scale == [10., 5.]
    args.plan_output_scale = [1., 1.]
    with pytest.raises(ValueError, match="incompatible explicit"):
        restore_run_configuration(args, saved, {"--plan-output-scale"})


def test_fixed_bn_keeps_statistics_but_trains_backbone_and_affine():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4),
                                torch.nn.GroupNorm(1, 4), torch.nn.Dropout(.1))
    before = {k: v.clone() for k, v in model[1].named_buffers()}
    assert set_training_mode(model, "fixed") == 1
    assert model.training and not model[1].training and model[2].training and model[3].training
    model(torch.randn(4, 3, 8, 8)).square().sum().backward()
    for i in (0, 1):
        assert model[i].weight.requires_grad and model[i].weight.grad.abs().sum() > 0
    assert all(torch.equal(before[k], v) for k, v in model[1].named_buffers())
    set_training_mode(model, "adaptive")
    model(torch.randn(4, 3, 8, 8))
    assert model[1].training and model[1].num_batches_tracked == 1


def test_fixed_bn_rejects_missing_statistics_and_unknown_policy():
    model = torch.nn.BatchNorm2d(4, track_running_stats=False)
    with pytest.raises(ValueError, match="running statistics"):
        set_training_mode(model, "fixed")
    with pytest.raises(ValueError, match="Unknown"):
        set_training_mode(model, "invalid")


def test_checkpoint_restores_bn_training_policy_not_current_default():
    from train_motiondrive_v2 import restore_run_configuration
    args = SimpleNamespace(resume=None, goal_on=1, state_on=1, arch="resnet50",
                           phase="joint", bn_policy="adaptive")
    saved = {"model_config": {"goal_on": True, "state_on": True, "backbone_arch": "resnet50"},
             "arguments": {"phase": "joint", "bn_policy": "fixed"}}
    restore_run_configuration(args, saved, set())
    assert args.bn_policy == "fixed"
    args.bn_policy = "adaptive"
    with pytest.raises(ValueError, match="incompatible explicit"):
        restore_run_configuration(args, saved, {"--bn-policy"})
