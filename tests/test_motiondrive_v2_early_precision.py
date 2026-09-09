from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.motiondrive_v2.config import MotionDriveV2Config
from models.motiondrive_v2.early_precision import (
    OrderedTemporalMotionEncoder,
    OrderedTemporalResidual,
    install_ordered_temporal_encoder,
)
from models.motiondrive_v2.motion_encoder import MotionEncoder
from scripts import run_motiondrive_v2_early_precision as screen


def config():
    return MotionDriveV2Config(backbone_arch="resnet34", goal_on=True, state_on=True,
                               cross_cell_goal_mode="zero", motion_input_mode="low_feature",
                               history_contract="control")


class EncoderOwner(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config()
        self.motion_encoder = MotionEncoder(self.config)


def encoder_inputs():
    torch.manual_seed(9)
    current = (torch.randn(1, 128, 3, 4), torch.randn(1, 128, 2, 3))
    history = (torch.randn(1, 4, 128, 3, 4), torch.randn(1, 4, 128, 2, 3))
    times = torch.tensor([[.1, .2, .5, 1.]], dtype=torch.float32)
    return current, history, times


def test_ordered_install_preserves_rng_and_parent_output_at_exact_zero():
    torch.manual_seed(101)
    owner = EncoderOwner().eval()
    current, history, times = encoder_inputs()
    with torch.no_grad():
        expected = owner.motion_encoder(current, history, times)
    before_rng = torch.get_rng_state().clone()
    install_ordered_temporal_encoder(owner)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert isinstance(owner.motion_encoder, OrderedTemporalMotionEncoder)
    final = owner.motion_encoder.ordered_temporal_residual.output_projection
    assert final.weight.count_nonzero() == 0 and final.bias.count_nonzero() == 0
    with torch.no_grad():
        actual = owner.motion_encoder(current, history, times)
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], expected[key]) for key in expected)


def test_ordered_residual_uses_fixed_time_major_order_per_spatial_site():
    module = OrderedTemporalResidual()
    tokens = torch.arange(1 * 4 * 3 * 128, dtype=torch.float32).reshape(1, 4, 3, 128)
    captured = {}
    handle = module.norm.register_forward_pre_hook(
        lambda _module, args: captured.setdefault("ordered", args[0].detach().clone()))
    module(tokens)
    handle.remove()
    expected = tokens.permute(0, 2, 1, 3).flatten(2)
    assert torch.equal(captured["ordered"], expected)
    assert captured["ordered"].shape == (1, 3, 512)


def test_ordered_residual_is_fp32_under_outer_bf16_and_final_gets_gradient():
    module = OrderedTemporalResidual()
    tokens = torch.randn(2, 4, 5, 128)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        initial = module(tokens)
    assert initial.dtype == tokens.dtype and torch.equal(initial, torch.zeros_like(initial))
    initial.sum().backward()
    assert module.output_projection.weight.grad is not None
    assert module.output_projection.bias.grad is not None
    assert torch.isfinite(module.output_projection.weight.grad).all()
    assert torch.isfinite(module.output_projection.bias.grad).all()
    assert module.output_projection.weight.grad.abs().sum() > 0
    assert module.output_projection.bias.grad.abs().sum() > 0


def plans(n=4):
    pred = torch.tensor([[[.2, -.1], [.5, .3], [9., 9.], [8., 8.], [7., 7.], [6., 6.]]],
                        dtype=torch.float32).repeat(n, 1, 1).requires_grad_()
    gt = torch.zeros_like(pred)
    batch = {"gt_plan": gt, "plan_valid": torch.ones(n, 6, dtype=torch.bool)}
    return pred, batch


def test_early_delta_uses_only_first_two_intervals_and_expected_formula():
    pred, batch = plans(1)
    value = screen.early_delta_smoothl1({"plan_abs": pred}, batch)
    delta = torch.stack([pred[:, 0], pred[:, 1] - pred[:, 0]], 1)
    expected = torch.nn.functional.smooth_l1_loss(
        delta, torch.zeros_like(delta), beta=.1, reduction="none").mean((1, 2)).mean()
    assert torch.equal(value, expected)
    value.backward()
    assert pred.grad[:, :2].abs().sum() > 0
    assert pred.grad[:, 2:].count_nonzero() == 0


def test_early_delta_mixed_mask_uneven_microbatches_match_value_and_gradient():
    generator = torch.Generator().manual_seed(17)
    pred_full = torch.randn(5, 6, 2, generator=generator, requires_grad=True)
    gt = torch.randn(5, 6, 2, generator=generator)
    valid = torch.ones(5, 6, dtype=torch.bool)
    valid[1, 4] = False
    valid[3, 0] = False
    gt[~valid.all(-1)] = float("nan")
    batch = {"gt_plan": gt, "plan_valid": valid}
    normalizers = {"plan_complete": torch.tensor(3)}
    full = screen.early_delta_smoothl1({"plan_abs": pred_full}, batch, normalizers)
    full.backward()
    full_grad = pred_full.grad.detach().clone()

    pred_split = pred_full.detach().clone().requires_grad_()
    pieces = []
    for start, stop in ((0, 3), (3, 5)):
        pieces.append(screen.early_delta_smoothl1(
            {"plan_abs": pred_split[start:stop]},
            {key: value[start:stop] for key, value in batch.items()}, normalizers))
    split = sum(pieces)
    split.backward()
    torch.testing.assert_close(split, full, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(pred_split.grad, full_grad, rtol=1e-6, atol=1e-7)
    assert pred_split.grad[[1, 3]].count_nonzero() == 0
    assert pred_split.grad[:, 2:].count_nonzero() == 0


def test_early_delta_all_invalid_is_differentiable_zero():
    pred, batch = plans(2)
    batch["plan_valid"].zero_()
    value = screen.early_delta_smoothl1(
        {"plan_abs": pred}, batch, {"plan_complete": torch.tensor(0)})
    assert value.item() == 0
    value.backward()
    assert pred.grad is not None and pred.grad.count_nonzero() == 0


def test_alpha_zero_returns_original_total_exactly_but_logs_diagnostic():
    pred, batch = plans(1)
    sentinel = pred.sum() * 0 + 3.25

    def original(*_args, **_kwargs):
        return sentinel, {"total": sentinel, "plan_d3": sentinel - 1}

    total, parts = screen.loss_with_early_diagnostic(
        original, 0., {"plan_abs": pred}, batch, object())
    assert total is sentinel and parts["total"] is sentinel
    assert parts["early_delta_weighted"].item() == 0


def test_delta_arm_adds_exact_fixed_weight_and_updates_logged_total():
    pred, batch = plans(1)
    base = pred.sum() * 0 + 2.

    def original(*_args, **_kwargs):
        return base, {"total": base}

    total, parts = screen.loss_with_early_diagnostic(
        original, screen.DELTA_ALPHA, {"plan_abs": pred}, batch, object())
    assert torch.equal(total, base + .2 * parts["early_delta_smoothl1"])
    assert parts["total"] is total


def test_data_validation_adapts_all_screen_arms_to_provided_status(monkeypatch):
    seen = []
    monkeypatch.setattr(screen, "validate_runtime", lambda args: {})
    monkeypatch.setattr(screen, "validate_source_manifest", lambda *_: {})
    monkeypatch.setattr(screen, "validate_parent", lambda args: ({}, {}))

    def validate(args):
        seen.append(args.arm)
        return ({"train_rows": 1, "train_rows_sha256": "t",
                 "tune_rows": 1, "tune_rows_sha256": "v"}, {})
    monkeypatch.setattr(screen.a1, "validate_data", validate)
    monkeypatch.setattr(screen, "prepare_model", lambda *_: (None, {
        "initial_model_state_sha256": "i", "ordered_residual_state_sha256": None,
        "missing_keys": [], "parameter_count": 1, "initial_parent_output_exact": True}))
    monkeypatch.setattr(screen, "build_experiment", lambda *_: {})
    monkeypatch.setattr(screen, "trainer_argv", lambda _args: [])
    monkeypatch.setattr(screen, "arguments", lambda _argv=None: SimpleNamespace(
        arm="ordered_motion_residual", preflight_only=True, smoke_only=False,
        seed=0, gpu=0, workers=4, cuda_memory_limit_mib=12000,
        cuda_min_free_mib=8192,
        expected_physical_gpu_uuid=screen.GPU_ASSIGNMENTS["ordered_motion_residual"],
        source_manifest="source.json", expected_source_manifest_sha256="s"))
    screen.main([])
    assert seen == ["provided_causal_5d"]


def test_patched_model_factory_constructs_without_recursive_package_export(monkeypatch):
    # Replacing the public package export reproduces trainer's patched API. The
    # immutable implementation imported by make_model must still be used once.
    import models.motiondrive_v2 as api
    original = api.MotionDriveV2
    calls = []
    monkeypatch.setattr(api, "MotionDriveV2", lambda *_args, **_kwargs: calls.append("recursive"))
    try:
        model = screen.make_model(config(), "continuation_control")
    finally:
        monkeypatch.setattr(api, "MotionDriveV2", original)
    assert isinstance(model, nn.Module) and calls == []
    assert hasattr(model, "shared_status_query_fusion")


def full_inputs():
    generator = torch.Generator().manual_seed(23)
    projection = torch.tensor([[48., 20., 0., 0.], [32., 0., -20., 0.],
                               [1., 0., 0., 0.], [0., 0., 0., 1.]])
    rand = lambda *shape: torch.randn(*shape, generator=generator)
    return dict(images=rand(1, 6, 3, 64, 96),
                history_images=rand(1, 4, 3, 32, 48),
                lidar2img=projection[None, None].repeat(1, 6, 1, 1),
                history_transforms=torch.eye(4)[None, None].repeat(1, 4, 1, 1),
                time_offsets=torch.tensor([[.1, .2, .5, 1.]], dtype=torch.float32),
                goal_xy=torch.tensor([[15., 3.]], dtype=torch.float32),
                provided_status5=rand(1, 5))


def test_full_a2_model_all_outputs_match_control_aux_and_zero_residual_under_bf16():
    torch.manual_seed(20260909)
    parent = screen.make_model(config(), "continuation_control").eval()
    parent_state = copy.deepcopy(parent.state_dict())
    models = []
    for arm in screen.ARMS:
        torch.manual_seed(800)
        model = screen.make_model(config(), arm).eval()
        incompatible = model.load_state_dict(parent_state, strict=False)
        expected = ([name for name in model.state_dict()
                     if name.startswith("motion_encoder.ordered_temporal_residual.")]
                    if arm == "ordered_motion_residual" else [])
        assert sorted(incompatible.missing_keys) == sorted(expected)
        assert not incompatible.unexpected_keys
        models.append(model)
    values = full_inputs()
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        outputs = [model(**values) for model in models]
    assert len(outputs[0]) >= 10
    assert all(output.keys() == outputs[0].keys() for output in outputs)
    for name in outputs[0]:
        assert all(torch.equal(output[name], outputs[0][name]) for output in outputs[1:]), name


def test_patched_runtime_factory_and_all_bindings_restore_on_normal_and_exception():
    import models.motiondrive_v2 as model_api
    import motiondrive_v2_data as data_api
    import train_motiondrive_v2 as trainer
    import evaluate_motiondrive_v2_planning as planning_eval
    def bindings():
        return (model_api.MotionDriveV2, data_api.MotionDriveDataset, trainer.model_inputs,
                planning_eval.planning_model_inputs, trainer._validate_experimental_protocol,
                trainer._validate_experimental_runtime, trainer._load_initial_model_state,
                trainer._training_schedule_actions, trainer.compute_loss)
    before = bindings()
    with screen.patched_runtime("ordered_motion_residual", {}, "p", [], "r"):
        model = model_api.MotionDriveV2(config())
        assert isinstance(model.motion_encoder, OrderedTemporalMotionEncoder)
        assert bindings() != before
    assert bindings() == before
    with pytest.raises(RuntimeError, match="sentinel"):
        with screen.patched_runtime("early_delta_aux", {}, "p", [], None):
            raise RuntimeError("sentinel")
    assert bindings() == before


def test_control_and_aux_cli_need_no_residual_pin():
    common = ["--arm", "continuation_control", "--init", "p", "--init-manifest", "m",
              "--source-manifest", "s", "--expected-source-manifest-sha256", "x",
              "--data-root", "d", "--split-manifest", "sp", "--supervision-root", "sup",
              "--status-overlay-root", "o", "--expected-status-overlay-sha256", "h",
              "--run-dir", "r", "--expected-physical-gpu-uuid",
              screen.GPU_ASSIGNMENTS["continuation_control"]]
    assert screen.arguments(common).expected_ordered_residual_state_sha256 is None


@pytest.mark.parametrize("arm,ordered,alpha", [
    ("continuation_control", False, 0.),
    ("ordered_motion_residual", True, 0.),
    ("early_delta_aux", False, .2),
])
def test_experiment_arm_contract(arm, ordered, alpha):
    args = SimpleNamespace(arm=arm, expected_status_overlay_sha256="o", smoke_only=False)
    prepared = {"initial_model_state_sha256": "i",
                "ordered_residual_state_sha256": "r" if ordered else None,
                "missing_keys": ["k"] if ordered else []}
    result = screen.build_experiment(
        args, {}, {"train_rows": 54810, "train_rows_sha256": "t",
                   "tune_rows": 1998, "tune_rows_sha256": "v"}, {}, prepared)
    assert result["ordered_motion"]["enabled"] is ordered
    assert result["early_delta"]["alpha"] == alpha
    assert result["ordered_motion"]["provided_status_goal_pose_inputs"] is False
