"""CPU structural tests for P7; these are not training or accuracy evidence."""
import dataclasses
import inspect
import json
from pathlib import Path
import types

import pytest
import torch

from models.motiondrive_v2 import MotionDriveV2Config
from models.motiondrive_v2.cross_cell_goal_residual import GoalScoredImageValueResidual
from models.motiondrive_v2.scene_encoder import SharedSceneEncoder
from scripts import motiondrive_v2_training as training
from scripts import run_motiondrive_v2_p7_goal_routing as p7
from scripts import train_motiondrive_v2 as trainer


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(mode="disabled"):
    return MotionDriveV2Config(backbone_arch="resnet50", motion_input_mode="low_feature",
                               goal_on=True, state_on=True, plan_output_scale=(10., 5.),
                               cross_cell_goal_mode=mode)


def protocol(arm="control_zero_slot", missing=None):
    mode = p7.ARM_TO_MODE[arm]
    return {
        "schema_version": 1, "name": "p7_cross_cell_goal_routing", "arm": arm,
        "last_only_final_eval": True, "expected_initial_model_state_sha256": "a" * 64,
        "expected_p0_model_state_sha256": "b" * 64,
        "expected_optimizer_groups": [{"name": "backbone", "base_lr": 1e-5},
                                      {"name": "head", "base_lr": 1e-4}],
        "expected_missing_state_keys": missing or [
            "scene_encoder.cross_cell_goal_residual.output.weight"],
        "train_data": {"rows": 54810, "rows_sha256": p7.EXPECTED_TRAIN_ROWS_SHA256},
        "tune_data": {"rows": 1998, "rows_sha256": p7.EXPECTED_TUNE_ROWS_SHA256},
        "source": {"manifest_sha256": "c" * 64}, "fresh_optimizer_step_zero": True,
        "all_model_parameters_joint_trainable": True, "existing_goal_path_on": True,
        "planner_signature_unchanged": True,
        "branch": {"mode": mode, "sigma_m": [10., 32. / 3.],
                   "sigma_selection": "fixed_geometry_not_tuned", "pool_size": 4,
                   "source_cells": 192, "destination_cells": 3072, "attention_dim": 32,
                   "cosine_scale": 8., "attention_precision": "fp32_autocast_disabled",
                   "goal_enters_distance_score_only": True,
                   "new_value_projection_adds_goal_or_position": False, "output_bias": False,
                   "output_weight_zero_initialized": True},
    }


def test_default_disabled_config_roundtrip_and_state_dict_are_legacy_shaped():
    cfg = MotionDriveV2Config(**dataclasses.asdict(config()))
    assert cfg.cross_cell_goal_mode == "disabled"
    encoder = SharedSceneEncoder(cfg)
    assert encoder.cross_cell_goal_residual is None
    assert not any("cross_cell_goal_residual" in key for key in encoder.state_dict())
    scene = torch.randn(2, 3072, 128)
    assert encoder._apply_cross_cell_goal_residual(
        scene, torch.ones(2, 3072, dtype=torch.bool), torch.randn(2, 2)) is scene


@pytest.mark.parametrize("bad", ["", "on", "goal", None])
def test_config_rejects_unknown_branch_mode(bad):
    with pytest.raises(ValueError):
        config(bad)


def test_fixed_geometry_temperature_parameter_and_mac_contract():
    layer = GoalScoredImageValueResidual()
    assert tuple(layer.sigma_m.tolist()) == pytest.approx((10., 32. / 3.))
    assert layer.COSINE_SCALE == 8. and layer.POOL_SIZE == 4
    assert layer.destination_xy.shape == (3072, 2) and layer.source_xy.shape == (192, 2)
    assert layer.parameter_accounting()["actual"] == layer.parameter_accounting()["total"] == 17824
    assert layer.mac_accounting_per_sample()["total"] == 68247552
    assert layer.output.bias is None and layer.value_image.bias is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_initialized_residual_preserves_exact_input_and_dtype(dtype):
    torch.manual_seed(8)
    layer = GoalScoredImageValueResidual()
    scene = torch.randn(1, 3072, 128).to(dtype)
    result = layer(scene, torch.tensor([[10., -4.]]), torch.ones(1, 3072, dtype=torch.bool))
    assert result.dtype == dtype and torch.equal(result, scene)


def test_entire_branch_stays_fp32_inside_outer_bf16_autocast(monkeypatch):
    """Observe projections and both contractions, not only explicit input casts."""
    torch.manual_seed(81)
    layer = GoalScoredImageValueResidual()
    with torch.no_grad():
        layer.output.weight.normal_(std=.02)
    projection_dtypes = {}
    handles = []
    def observe_projection(name):
        def hook(_module, _inputs, output):
            projection_dtypes.setdefault(name, output.dtype)
        return hook
    for name, module in (("query", layer.query_image), ("key", layer.key_image),
                         ("value", layer.value_image), ("position0", layer.position[0]),
                         ("position2", layer.position[2]), ("output", layer.output)):
        handles.append(module.register_forward_hook(observe_projection(name)))
    contractions = []
    original_einsum = torch.einsum

    def observed_einsum(equation, *operands):
        output = original_einsum(equation, *operands)
        contractions.append((equation, tuple(value.dtype for value in operands), output.dtype))
        return output

    monkeypatch.setattr(torch, "einsum", observed_einsum)
    scene = torch.randn(1, 3072, 128, dtype=torch.bfloat16, requires_grad=True)
    goal = torch.tensor([[10., -4.]], requires_grad=True)
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = layer(scene, goal)
        result.float().square().mean().backward()
    finally:
        for handle in handles:
            handle.remove()
    assert result.dtype == torch.bfloat16
    assert projection_dtypes and set(projection_dtypes.values()) == {torch.float32}
    assert [item[0] for item in contractions] == ["bqd,bkd->bqk", "bqk,bkd->bqd"]
    assert all(set(input_dtypes) == {torch.float32} and output_dtype == torch.float32
               for _, input_dtypes, output_dtype in contractions)
    assert torch.isfinite(scene.grad).all() and torch.isfinite(goal.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in layer.parameters())


def test_first_update_opens_output_then_qkv_position_and_norm_gradients():
    torch.manual_seed(31)
    layer = GoalScoredImageValueResidual()
    scene = torch.randn(1, 3072, 128)
    goal = torch.tensor([[10., 3.]])
    probe = torch.randn_like(scene)
    (layer(scene, goal) * probe).sum().backward()
    assert layer.output.weight.grad.count_nonzero().item() > 0
    for parameter in (layer.query_image.weight, layer.key_image.weight,
                      layer.value_image.weight, layer.position[0].weight, layer.norm.weight):
        assert parameter.grad is not None and parameter.grad.count_nonzero().item() == 0
    with torch.no_grad(): layer.output.weight.add_(-1e-4 * layer.output.weight.grad)
    layer.zero_grad(set_to_none=True)
    (layer(scene, goal) * probe).sum().backward()
    for parameter in (layer.query_image.weight, layer.key_image.weight,
                      layer.value_image.weight, layer.position[0].weight, layer.norm.weight):
        assert parameter.grad is not None and parameter.grad.count_nonzero().item() > 0


def test_open_output_constant_values_are_goal_invariant_and_zero_value_is_exact():
    torch.manual_seed(32)
    layer = GoalScoredImageValueResidual()
    with torch.no_grad(): layer.output.weight.normal_(std=.02)
    token = torch.randn(2, 1, 128)
    constant = token.expand(2, 3072, 128).clone()
    first = layer(constant, torch.tensor([[0., 0.], [0., 0.]]))
    second = layer(constant, torch.tensor([[40., 20.], [-5., -25.]]))
    torch.testing.assert_close(first, second, rtol=2e-6, atol=2e-6)
    with torch.no_grad(): layer.value_image.weight.zero_()
    assert torch.equal(layer(constant, torch.tensor([[40., 20.], [-5., -25.]])), constant)


def test_partial_mask_is_batch_independent():
    torch.manual_seed(33)
    layer = GoalScoredImageValueResidual()
    with torch.no_grad(): layer.output.weight.normal_(std=.02)
    scene = torch.randn(2, 3072, 128)
    goal = torch.tensor([[0., 0.], [50., 20.]])
    valid = torch.zeros(2, 3072, dtype=torch.bool)
    valid[0, :16] = True; valid[1, -16:] = True
    batched = layer(scene, goal, valid)
    separate = torch.cat([layer(scene[i:i + 1], goal[i:i + 1], valid[i:i + 1])
                          for i in range(2)])
    torch.testing.assert_close(batched, separate, rtol=1e-5, atol=2e-6)


def test_value_path_has_no_effective_bias_and_goal_only_occurs_in_score():
    layer = GoalScoredImageValueResidual()
    with torch.no_grad():
        layer.norm.bias.normal_(); layer.output.weight.normal_()
    zeros = torch.zeros(1, 3072, 128)
    assert torch.equal(layer(zeros, torch.tensor([[10., 0.]])), zeros)
    source = inspect.getsource(GoalScoredImageValueResidual.forward)
    assert "goal_xy_m" not in source.split("value =", 1)[1].split("content_score", 1)[0]
    assert "distance = ((self.source_xy.float()[None] - goal_xy_m.float()" in source


def test_all_invalid_is_exact_identity_and_backward_finite():
    layer = GoalScoredImageValueResidual()
    with torch.no_grad(): layer.output.weight.normal_()
    scene = torch.randn(1, 3072, 128, requires_grad=True)
    goal = torch.tensor([[10., 3.]], requires_grad=True)
    result = layer(scene, goal, torch.zeros(1, 3072, dtype=torch.bool))
    assert torch.equal(result, scene)
    result.square().mean().backward()
    assert torch.isfinite(scene.grad).all() and torch.isfinite(goal.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in layer.parameters())


def test_scene_integration_changes_only_new_slot_and_passes_raw_metres_and_mask():
    class Spy(torch.nn.Module):
        def __init__(self): super().__init__(); self.calls = []
        def forward(self, scene, goal, visible):
            self.calls.append((goal.clone(), visible.clone()))
            return scene
    goal = torch.tensor([[10., -4.]])
    visible = torch.zeros(1, 3072, dtype=torch.bool); visible[:, 17] = True
    encoders = []
    for mode in ("zero", "real"):
        torch.manual_seed(19)
        encoder = SharedSceneEncoder(config(mode)); encoder.cross_cell_goal_residual = Spy()
        encoder._apply_cross_cell_goal_residual(torch.randn(1, 3072, 128), visible, goal)
        encoders.append(encoder)
    assert encoders[0].config.goal_on is encoders[1].config.goal_on is True
    assert torch.equal(encoders[0].cross_cell_goal_residual.calls[0][0], torch.zeros_like(goal))
    assert torch.equal(encoders[1].cross_cell_goal_residual.calls[0][0], goal)
    assert torch.equal(encoders[1].cross_cell_goal_residual.calls[0][0], torch.tensor([[10., -4.]]))
    assert torch.equal(encoders[0].cross_cell_goal_residual.calls[0][1], visible)
    assert torch.equal(encoders[1].cross_cell_goal_residual.calls[0][1], visible)


def test_same_seed_zero_and_real_modes_have_identical_initial_state():
    torch.manual_seed(101); zero = SharedSceneEncoder(config("zero")).state_dict()
    torch.manual_seed(101); real = SharedSceneEncoder(config("real")).state_dict()
    assert zero.keys() == real.keys()
    assert all(torch.equal(zero[key], real[key]) for key in zero)


def test_enabled_arms_execute_branch_at_zero_output_and_keep_old_goal_response():
    torch.manual_seed(29)
    zero = SharedSceneEncoder(config("zero")).eval()
    torch.manual_seed(29)
    real = SharedSceneEncoder(config("real")).eval()
    batch, channels = 1, 128
    generator = torch.Generator().manual_seed(30)
    current = (torch.randn(batch, 6, channels, 8, 12, generator=generator),
               torch.randn(batch, 6, channels, 4, 6, generator=generator))
    history = (torch.randn(batch, 4, channels, 8, 12, generator=generator),
               torch.randn(batch, 4, channels, 4, 6, generator=generator))
    current_global = torch.randn(batch, 6, channels, 2, 3, generator=generator)
    projection = torch.tensor([[48., 20., 0., 0.], [32., 0., -20., 0.],
                               [1., 0., 0., 0.], [0., 0., 0., 1.]])
    lidar2img = projection[None, None].repeat(batch, 6, 1, 1)
    transforms = torch.eye(4)[None, None].repeat(batch, 4, 1, 1)
    times = torch.tensor([[.1, .2, .5, 1.]])
    calls = {"zero": 0, "real": 0}
    for name, encoder in (("zero", zero), ("real", real)):
        original = encoder.cross_cell_goal_residual.forward
        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)
        encoder.cross_cell_goal_residual.forward = counted
    args = (current, history, current_global, lidar2img, transforms, times)
    with torch.no_grad():
        zero_a = zero(*args, torch.tensor([[10., 0.]]), (64, 96))["scene_features"]
        zero_b = zero(*args, torch.tensor([[20., 3.]]), (64, 96))["scene_features"]
        real_a = real(*args, torch.tensor([[10., 0.]]), (64, 96))["scene_features"]
        real_b = real(*args, torch.tensor([[20., 3.]]), (64, 96))["scene_features"]
    assert calls == {"zero": 2, "real": 2}
    assert torch.equal(zero_a, real_a) and torch.equal(zero_b, real_b)
    assert not torch.equal(zero_a, zero_b) and not torch.equal(real_a, real_b)


def test_old_scene_state_plus_zero_residual_is_exact_legacy_full_output():
    torch.manual_seed(34)
    legacy = SharedSceneEncoder(config("disabled")).eval()
    old_state = {key: value.clone() for key, value in legacy.state_dict().items()}
    torch.manual_seed(99)
    enabled = SharedSceneEncoder(config("real")).eval()
    incompatible = enabled.load_state_dict(old_state, strict=False)
    assert not incompatible.unexpected_keys and incompatible.missing_keys
    assert all(key.startswith("cross_cell_goal_residual.") for key in incompatible.missing_keys)
    batch, channels = 1, 128
    generator = torch.Generator().manual_seed(35)
    current = (torch.randn(batch, 6, channels, 8, 12, generator=generator),
               torch.randn(batch, 6, channels, 4, 6, generator=generator))
    history = (torch.randn(batch, 4, channels, 8, 12, generator=generator),
               torch.randn(batch, 4, channels, 4, 6, generator=generator))
    current_global = torch.randn(batch, 6, channels, 2, 3, generator=generator)
    projection = torch.tensor([[48., 20., 0., 0.], [32., 0., -20., 0.],
                               [1., 0., 0., 0.], [0., 0., 0., 1.]])
    args = (current, history, current_global, projection[None, None].repeat(1, 6, 1, 1),
            torch.eye(4)[None, None].repeat(1, 4, 1, 1), torch.tensor([[.1, .2, .5, 1.]]),
            torch.tensor([[10., 2.]]), (64, 96))
    with torch.no_grad():
        before = legacy(*args)
        after = enabled(*args)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_integrated_all_invalid_branch_backward_is_finite():
    encoder = SharedSceneEncoder(config("real"))
    with torch.no_grad(): encoder.cross_cell_goal_residual.output.weight.normal_()
    scene = torch.randn(1, 3072, 128, requires_grad=True)
    goal = torch.tensor([[10., 2.]], requires_grad=True)
    result = encoder._apply_cross_cell_goal_residual(
        scene, torch.zeros(1, 3072, dtype=torch.bool), goal)
    assert torch.equal(result, scene)
    result.square().mean().backward()
    assert torch.isfinite(scene.grad).all() and torch.isfinite(goal.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in encoder.cross_cell_goal_residual.parameters())


def test_planner_and_model_forward_signatures_receive_no_new_raw_goal():
    from models.motiondrive_v2.model import MotionDriveV2
    from models.motiondrive_v2.planner import DirectTrajectoryPlanner
    planner = str(inspect.signature(DirectTrajectoryPlanner.forward))
    plan_from = str(inspect.signature(MotionDriveV2.plan_from_features))
    assert "goal" not in planner and "goal" not in plan_from
    assert list(inspect.signature(MotionDriveV2.forward).parameters)[1:] == [
        "images", "history_images", "lidar2img", "history_transforms", "time_offsets", "goal_xy"]


@pytest.mark.parametrize("arm", tuple(p7.ARM_TO_MODE))
def test_p7_protocol_and_fixed_runtime_recipe(arm):
    experiment = protocol(arm)
    assert trainer._validate_experimental_protocol(experiment) is None
    args = trainer.arguments(p7.trainer_argv(types.SimpleNamespace(
        data_root="/data", split_manifest="/split", supervision_root="/sup",
        run_dir="/run", init="/p0/last.pth", gpu=0, base_seed=1, workers=4,
        cuda_memory_limit_mib=12000, cuda_min_free_mib=8192), p7.ARM_TO_MODE[arm]))
    trainer._validate_experimental_runtime(args, experiment)
    assert args.steps == 6000 and args.warmup == 200 and args.eval_every == args.save_every == 6000
    assert args.lr == 1e-4 and args.backbone_lr == 1e-5 and args.init and not args.resume


@pytest.mark.parametrize("mutation", ["mode", "sigma", "precision", "goal_path", "optimizer", "train_rows", "tune_rows", "missing"])
def test_p7_protocol_fails_closed(mutation):
    value = protocol()
    if mutation == "mode": value["branch"]["mode"] = "real"
    elif mutation == "sigma": value["branch"]["sigma_m"] = [9., 9.]
    elif mutation == "precision": value["branch"]["attention_precision"] = "outer_autocast"
    elif mutation == "goal_path": value["existing_goal_path_on"] = False
    elif mutation == "optimizer": value["expected_optimizer_groups"][1]["base_lr"] = 2e-4
    elif mutation == "train_rows": value["train_data"]["rows"] = 8
    elif mutation == "tune_rows": value["tune_data"]["rows"] = 8
    else: value["expected_missing_state_keys"] = ["planner.weight"]
    with pytest.raises(ValueError): trainer._validate_experimental_protocol(value)


def test_old_checkpoint_load_allows_exact_new_branch_keys_only_and_zero_output():
    class Branch(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.extra = torch.nn.Linear(2, 2)
            self.output = torch.nn.Linear(2, 2, bias=False)
            torch.nn.init.zeros_(self.output.weight)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.old = torch.nn.Linear(2, 2)
            self.scene_encoder = torch.nn.Module()
            self.scene_encoder.cross_cell_goal_residual = Branch()
    model = Model()
    old = {key: value.clone() for key, value in model.state_dict().items()
           if not key.startswith("scene_encoder.cross_cell_goal_residual.")}
    missing = [key for key in model.state_dict() if key not in old]
    experiment = protocol(missing=missing)
    experiment["expected_p0_model_state_sha256"] = training.tensor_state_sha256(old)
    report = trainer._load_initial_model_state(model, {"model": old}, experiment)
    assert report["p7_old_checkpoint_load"] == {
        "strict_existing_keys": True, "missing_keys": missing,
        "unexpected_keys": [], "weights_only": True}
    assert not model.scene_encoder.cross_cell_goal_residual.output.weight.count_nonzero()
    broken = dict(old); broken.pop("old.bias")
    with pytest.raises(ValueError, match="miss exactly"):
        trainer._load_initial_model_state(Model(), {"model": broken}, experiment)


def test_real_p0_shaped_state_load_and_paired_initialization_are_exact():
    from models.motiondrive_v2.model import MotionDriveV2
    trainer.seed_all(0)
    legacy = MotionDriveV2(MotionDriveV2Config(
        backbone_arch="resnet50", motion_input_mode="low_feature", goal_on=False,
        state_on=False, plan_output_scale=(10., 5.), cross_cell_goal_mode="disabled"))
    old_state = {key: value.clone() for key, value in legacy.state_dict().items()}
    payload = {"model": old_state, "manifest": {
        "model_config": dataclasses.asdict(legacy.config)}}
    del legacy
    initialized = p7.prepare_initialization(payload, 0, "zero")
    assert initialized["initial_model_state_sha256"] == initialized[
        "paired_other_mode_initial_model_state_sha256"]
    assert initialized["branch_parameter_count"] == 17824
    assert initialized["branch_optimizer_group"] == "head"
    assert initialized["fresh_optimizer_step"] == 0
    assert initialized["expected_missing_state_keys"]
    assert all(key.startswith("scene_encoder.cross_cell_goal_residual.")
               for key in initialized["expected_missing_state_keys"])
    # Includes both learned tensors and fixed metric geometry buffers.
    assert any(key.endswith("output.weight") for key in initialized["expected_missing_state_keys"])
    assert any(key.endswith("source_xy") for key in initialized["expected_missing_state_keys"])
    assert any(key.endswith("sigma_m") for key in initialized["expected_missing_state_keys"])


def test_branch_parameters_are_trainable_and_in_existing_nonbackbone_group():
    encoder = SharedSceneEncoder(config("real"))
    names = ["scene_encoder." + name for name, _ in encoder.named_parameters()
             if name.startswith("cross_cell_goal_residual.")]
    assert names and all(not name.startswith("backbone_fpn.") for name in names)
    assert all(parameter.requires_grad for name, parameter in encoder.named_parameters()
               if name.startswith("cross_cell_goal_residual."))


@pytest.mark.parametrize("mode", ["disabled", "zero", "real"])
def test_checkpoint_config_restore_roundtrip_for_every_mode(mode):
    cfg = config(mode)
    args = types.SimpleNamespace(
        resume="/last.pth", motion_input_mode=None, plan_output_scale=None,
        cross_cell_goal_mode=None, bn_policy="adaptive", time_input="raw",
        goal_on=0, state_on=0, arch="resnet34", phase="pretrain",
        alpha_occ=.2, alpha_lane=.2, alpha_motion=.2, uncertainty=1,
        lr=1e-4, backbone_lr=1e-5, weight_decay=.01, warmup=200,
        precision="bf16", seed=0, microbatch=0,
        cuda_memory_limit_mib=0, cuda_min_free_mib=0)
    saved_args = {key: getattr(args, key) for key in (
        "alpha_occ", "alpha_lane", "alpha_motion", "uncertainty", "lr",
        "backbone_lr", "weight_decay", "warmup", "precision", "seed")}
    saved_args.update(phase="joint", bn_policy="fixed", time_input="nominal")
    saved = {"model_config": dataclasses.asdict(cfg), "arguments": saved_args,
             "time_input": "nominal", "time_input_policy": {"mode": "nominal"}}
    restored = trainer.restore_run_configuration(args, saved, set())
    assert restored["cross_cell_goal_mode"] == mode and args.cross_cell_goal_mode == mode


def test_last_only_schedule_has_one_terminal_eval_no_periodic_best_save():
    args = types.SimpleNamespace(steps=6000, eval_every=6000, save_every=6000)
    experiment = protocol()
    assert sum(trainer._training_schedule_actions(step, args, experiment)[0]
               for step in range(1, 6001)) == 1
    assert trainer._training_schedule_actions(6000, args, experiment) == (True, False)
    source = inspect.getsource(trainer.run_training)
    assert "detailed_records=experiment is not None" in source
    assert 'atomic_json(run_dir / "final_eval.json"' in source


def test_source_manifest_requires_external_sha_exact_closure_and_local_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(p7, "ROOT", tmp_path)
    files = {}
    for index, name in enumerate(sorted(p7.SOURCE_FILES)):
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}\n")
        files[name] = p7.file_sha(path)
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "git_sha": "a" * 40,
                                    "file_sha256": files}))
    expected = p7.file_sha(manifest)
    assert p7.validate_source_manifest(manifest, expected)["file_sha256"] == files
    (tmp_path / sorted(p7.SOURCE_FILES)[0]).write_text("mutated\n")
    with pytest.raises(ValueError, match="runtime source"):
        p7.validate_source_manifest(manifest, expected)


def test_build_experiment_differs_only_arm_and_branch_mode():
    p0 = {"model_state_sha256": "b" * 64}
    initialized = {"initial_model_state_sha256": "a" * 64,
                   "expected_missing_state_keys": [
                       "scene_encoder.cross_cell_goal_residual.output.weight"]}
    data = {"train_rows": 54810, "train_rows_sha256": p7.EXPECTED_TRAIN_ROWS_SHA256}
    data.update(tune_rows=1998, tune_rows_sha256=p7.EXPECTED_TUNE_ROWS_SHA256)
    zero = p7.build_experiment("control_zero_slot", p0, initialized, data, {"x": "y"})
    real = p7.build_experiment("goal_real_slot", p0, initialized, data, {"x": "y"})
    assert {key for key in zero if zero[key] != real[key]} == {"arm", "branch"}
    assert {key for key in zero["branch"] if zero["branch"][key] != real["branch"][key]} == {"mode"}


def test_p0_artifact_identities_are_hard_pinned_per_base():
    assert p7.P0_ARTIFACTS == {
        0: {"checkpoint_sha256": "8e91c30947a040f267f01f0642db3f2725900545deeb6d4087c795b5eb398a0c",
            "sidecar_sha256": "9e5ba7ce89c0c1a9e4fe49476ff72c63b451f2658766e74e84c7336a4591fd98",
            "model_state_sha256": "f69e1c52ccd4b9908130ffe02170c0f08bff21712c634fee064517009a6c2c7d"},
        1: {"checkpoint_sha256": "6067ba9cc543e6e1be837d85883696bc7ad12c3da9b970edba3008e30eff83c5",
            "sidecar_sha256": "385ea53269e96d73c7d66f4196bc2e1205b0c6b7e9c7db537f983bc7aa37c421",
            "model_state_sha256": "d595f0788fbb5ae31ea2eaa2f3344ac300028756dcfe8f313ec1dbb761673e0d"},
    }
    assert p7.EXPECTED_CALIBRATION_SHA256 == "8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961"


def test_runtime_namespace_preflight_enforces_fixed_resources_without_cuda():
    args = types.SimpleNamespace(gpu=0, workers=4, cuda_memory_limit_mib=12000,
                                 cuda_min_free_mib=8192, preflight_only=True,
                                 expected_physical_gpu_uuid=next(iter(p7.ALLOWED_PHYSICAL_GPU_UUIDS)))
    receipt = p7.validate_runtime_namespace(args)
    assert receipt["gpu_used"] is False and receipt["logical_gpu"] == 0
    args.workers = 3
    with pytest.raises(ValueError, match="safety contract"): p7.validate_runtime_namespace(args)


@pytest.mark.parametrize("prefixed", [False, True])
def test_runtime_namespace_canonicalizes_only_optional_gpu_prefix(monkeypatch, prefixed):
    expected = "GPU-4b804d68-fd61-af14-393a-573c533d5006"
    args = types.SimpleNamespace(gpu=0, workers=4, cuda_memory_limit_mib=12000,
                                 cuda_min_free_mib=8192, preflight_only=False,
                                 expected_physical_gpu_uuid=expected)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", expected)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    raw = expected if prefixed else expected.removeprefix("GPU-")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: types.SimpleNamespace(uuid=raw))
    receipt = p7.validate_runtime_namespace(args)
    assert receipt["actual_physical_gpu_uuid"] == expected
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: types.SimpleNamespace(uuid="different"))
    with pytest.raises(ValueError, match="UUID mismatch"):
        p7.validate_runtime_namespace(args)


def test_help_does_not_construct_model_or_initialize_cuda(monkeypatch):
    with pytest.raises(SystemExit) as exit_info:
        p7.arguments(["--help"])
    assert exit_info.value.code == 0 and not torch.cuda.is_initialized()
