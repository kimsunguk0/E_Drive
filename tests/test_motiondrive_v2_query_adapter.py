"""CPU structural/strict-IO tests; synthetic tiny backbone is not an accuracy test."""
import copy
import hashlib
import inspect
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import models.motiondrive_v2.model as legacy_module
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from models.motiondrive_v2_query_adapter import (
    ADAPTER_STATE_KEYS, ARCHITECTURE, ImageStateQueryPlanner,
    MotionDriveV2QueryAdapter, migrate_legacy_checkpoint, model_from_query_payload,
)
from scripts.motiondrive_v2_training import tensor_state_sha256


class TinyBackbone(nn.Module):
    def __init__(self, channels, arch):
        super().__init__()
        self.image_conv = nn.Conv2d(3, channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, images):
        feature = F.gelu(self.norm(self.image_conv(images)))
        h, w = images.shape[-2:]
        levels = tuple(F.adaptive_avg_pool2d(feature, (max(1, h // stride), max(1, w // stride)))
                       for stride in (4, 8))
        return levels, F.adaptive_avg_pool2d(feature, (max(1, h // 16), max(1, w // 16)))


@pytest.fixture(autouse=True)
def cpu_fixture(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(legacy_module, "ResNet34FPN128", TinyBackbone)
    yield
    torch.set_num_threads(old_threads)


def config(**changes):
    fields = dict(channels=16, backbone_arch="resnet34", grid_size=(4, 3),
                  x_range=(1., 9.), y_range=(-2., 2.), heights=(0., 1.),
                  motion_grid=(3, 4), correlation_channels=4, scene_attention_channels=4,
                  scene_chunk_size=5, planner_heads=4, planner_layers=1,
                  motion_input_mode="low_feature", plan_output_scale=(10., 5.))
    fields.update(changes)
    return MotionDriveV2Config(**fields)


def inputs():
    generator = torch.Generator().manual_seed(19)
    projection = torch.tensor([[24., 10., 0., 0.], [16., 0., -10., 0.],
                               [1., 0., 0., 0.], [0., 0., 0., 1.]])
    return dict(images=torch.randn(1, 6, 3, 32, 48, generator=generator),
                history_images=torch.randn(1, 4, 3, 16, 24, generator=generator),
                lidar2img=projection[None, None].repeat(1, 6, 1, 1),
                history_transforms=torch.eye(4)[None, None].repeat(1, 4, 1, 1),
                time_offsets=torch.tensor([[.1, .2, .5, 1.]]),
                goal_xy=torch.tensor([[15., 3.]]))


def payload(model):
    metadata = {"architecture": ARCHITECTURE, "query_adapter_on": model.query_adapter_on,
                "model_config": model.config.to_dict()}
    return {**metadata, "model": copy.deepcopy(model.state_dict()),
            "manifest": copy.deepcopy(metadata), "step": 1000}


@pytest.fixture
def legacy_file(tmp_path):
    model = MotionDriveV2(config()).eval()
    path = tmp_path / "last.pth"
    torch.save({"model": model.state_dict(), "step": 3000,
                "manifest": {"model_config": model.config.to_dict()}}, path)
    return model, path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_adapter_size_only_four_new_keys_and_zero_output():
    planner = ImageStateQueryPlanner(MotionDriveV2Config())
    assert sum(p.numel() for p in planner.query_adapter.parameters()) == 35840
    assert {f"planner.{key}" for key in planner.state_dict() if key.startswith("query_adapter.")} == ADAPTER_STATE_KEYS
    assert torch.count_nonzero(planner.query_adapter[0].weight) > 0
    assert torch.count_nonzero(planner.query_adapter[2].weight) == 0
    assert torch.count_nonzero(planner.query_adapter[2].bias) == 0
    assert all(m.p == 0. for m in planner.modules() if isinstance(m, nn.Dropout))


def test_strict_migration_preserves_legacy_and_both_flags_initial_physical_prediction(legacy_file):
    legacy, path, digest = legacy_file
    rng_before = torch.random.get_rng_state().clone()
    off, off_report = migrate_legacy_checkpoint(path, digest, adapter_seed=7, adapter_on=False)
    on, on_report = migrate_legacy_checkpoint(path, digest, adapter_seed=7, adapter_on=True)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert set(off.state_dict()) - set(legacy.state_dict()) == ADAPTER_STATE_KEYS
    assert all(torch.equal(off.state_dict()[k], v) for k, v in legacy.state_dict().items())
    assert tensor_state_sha256(off.state_dict()) == tensor_state_sha256(on.state_dict())
    assert off_report["initial_model_state_sha256"] == on_report["initial_model_state_sha256"]
    assert off_report["legacy_tensors_bitwise_preserved"] and off_report["final_adapter_projection_exact_zero"]
    assert off_report["source_checkpoint_sha256"] == digest
    assert not off_report["initial_prediction_parity_verified"]
    assert not off_report["run_or_os_exit_lineage_verified"]
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected, a, b = legacy(**inputs()), off(**inputs()), on(**inputs())
    for key in expected:
        assert torch.equal(expected[key], a[key]), key
        assert torch.equal(a[key], b[key]), key
    assert a["plan_abs"].dtype == torch.float32
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert not off.training and not on.training
    assert sum(p.numel() for p in off.parameters()) == sum(p.numel() for p in on.parameters())
    json.dumps(off_report, allow_nan=False)


def test_adapter_seed_only_changes_declared_tensors_and_preserves_global_rng(legacy_file):
    _, path, digest = legacy_file
    a, _ = migrate_legacy_checkpoint(path, digest, adapter_seed=1)
    b, _ = migrate_legacy_checkpoint(path, digest, adapter_seed=2)
    changed = {key for key in a.state_dict() if not torch.equal(a.state_dict()[key], b.state_dict()[key])}
    assert changed == {"planner.query_adapter.0.weight", "planner.query_adapter.0.bias"}


def test_complete_payload_strict_restore_and_rng_preservation(legacy_file):
    _, path, digest = legacy_file
    original, _ = migrate_legacy_checkpoint(path, digest, adapter_on=True)
    saved = payload(original)
    before = torch.random.get_rng_state().clone()
    restored = model_from_query_payload(saved)
    assert torch.equal(before, torch.random.get_rng_state())
    assert restored.query_adapter_on is True and restored.architecture == ARCHITECTURE
    assert tensor_state_sha256(original.state_dict()) == tensor_state_sha256(restored.state_dict())
    with torch.no_grad():
        assert torch.equal(original(**inputs())["plan_abs"], restored(**inputs())["plan_abs"])


@pytest.mark.parametrize("fault", ["architecture", "flag_missing", "flag_int", "config_missing", "config_extra",
                                    "manifest_flag", "manifest_config", "manifest_architecture", "state_missing",
                                    "state_extra", "state_dtype", "state_shape", "state_nan", "state_off"])
def test_payload_rejects_metadata_and_state_drift(fault):
    saved = payload(MotionDriveV2QueryAdapter(config()))
    name = "planner.query_adapter.0.weight"
    if fault == "architecture": saved["architecture"] = "legacy"
    elif fault == "flag_missing": saved.pop("query_adapter_on")
    elif fault == "flag_int": saved["query_adapter_on"] = 0
    elif fault == "config_missing": saved["model_config"].pop("motion_input_mode")
    elif fault == "config_extra": saved["model_config"]["query_adapter_on"] = False
    elif fault == "manifest_flag": saved["manifest"]["query_adapter_on"] = True
    elif fault == "manifest_config": saved["manifest"]["model_config"]["goal_on"] = False
    elif fault == "manifest_architecture": saved["manifest"].pop("architecture")
    elif fault == "state_missing": saved["model"].pop(name)
    elif fault == "state_extra": saved["model"]["unknown"] = torch.zeros(1)
    elif fault == "state_dtype": saved["model"][name] = saved["model"][name].double()
    elif fault == "state_shape": saved["model"][name] = saved["model"][name][:1]
    elif fault == "state_nan": saved["model"][name][0, 0] = float("nan")
    else: saved["model_config"]["state_on"] = False
    with pytest.raises((ValueError, RuntimeError)):
        model_from_query_payload(saved)


@pytest.mark.parametrize("fault", ["sha", "seed_bool", "flag_int", "config_missing", "state_extra", "state_dtype"])
def test_legacy_migration_rejects_invalid_source(legacy_file, fault):
    _, path, digest = legacy_file
    args = dict(expected_sha256=digest)
    if fault == "sha": args["expected_sha256"] = "0" * 64
    elif fault == "seed_bool": args["adapter_seed"] = True
    elif fault == "flag_int": args["adapter_on"] = 1
    else:
        saved = torch.load(path, weights_only=True)
        if fault == "config_missing": saved["manifest"]["model_config"].pop("plan_output_scale")
        elif fault == "state_extra": saved["model"]["unknown"] = torch.zeros(1)
        else: saved["model"]["planner.xy_head.3.weight"] = saved["model"]["planner.xy_head.3.weight"].double()
        torch.save(saved, path)
        args["expected_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises((ValueError, RuntimeError)):
        migrate_legacy_checkpoint(path, **args)


def test_adapter_counterfactual_information_not_compute_and_legacy_memory_retained():
    planner = ImageStateQueryPlanner(config()).eval()
    with torch.no_grad():
        planner.query_adapter[2].weight.normal_(std=.05)
    queries, status = torch.randn(2, 6, 16), torch.randn(2, 22)
    calls, adapter_inputs, memory_shapes = [], [], []
    handles = [planner.query_adapter.register_forward_pre_hook(lambda _m, args: adapter_inputs.append(args[0].detach().clone())),
               planner.decoder.register_forward_pre_hook(lambda _m, args: memory_shapes.append(args[1].shape))]
    for module in planner.query_adapter:
        handles.append(module.register_forward_hook(lambda _m, _args, out: calls.append(tuple(out.shape))))
    a = planner.adapt_queries(queries, status)
    b = planner.adapt_queries(queries, status + 20)
    assert torch.equal(a, b) and not adapter_inputs[-1][..., -22:].any()
    control_calls = list(calls)
    calls.clear()
    planner.query_adapter_on = True
    a = planner.adapt_queries(queries, status)
    b = planner.adapt_queries(queries, status + 20)
    assert not torch.equal(a, b) and calls == control_calls
    assert torch.equal(adapter_inputs[-1][..., -22:], (status + 20)[:, None].expand(-1, 6, -1))
    seen_state = []
    handles.append(planner.state_projection.register_forward_pre_hook(lambda _m, args: seen_state.append(args[0].detach().clone())))
    for flag in (False, True):
        planner.query_adapter_on = flag
        planner(torch.randn(2, 12, 16), torch.randn(2, 12, 16), torch.ones(2, 6), torch.ones(2, 4, 4))
    assert memory_shapes == [torch.Size([2, 25, 16])] * 2
    assert torch.equal(seen_state[0], seen_state[1]) and seen_state[0].abs().sum() > 0
    for handle in handles: handle.remove()


def test_frozen_trunk_feature_api_planner_gradient_and_fp32_heads():
    model = MotionDriveV2QueryAdapter(config(), query_adapter_on=True).eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("planner."))
    before = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("planner.")}
    seen = []
    handles = [module.register_forward_pre_hook(lambda _m, args: seen.append(args[0].dtype))
               for module in (model.planner.query_adapter[0], model.planner.query_adapter[2], model.planner.xy_head)]
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        parts = model.forward_parts(**inputs())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        plan = model.plan_from_features(parts["scene_features"], parts["motion_features"], parts["state_hat"], parts["history_hat"])
    assert plan.dtype == torch.float32 and set(seen) == {torch.float32}
    plan.square().mean().backward()
    assert model.planner.query_adapter[2].weight.grad.abs().sum() > 0
    assert model.planner.xy_head[-1].weight.grad.abs().sum() > 0
    assert model.planner.query_adapter[0].weight.grad.abs().sum() == 0  # Expected on zero-W2 first update.
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("planner."))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.plan_from_features(parts["scene_features"], parts["motion_features"], parts["state_hat"], parts["history_hat"]).square().mean().backward()
    assert model.planner.query_adapter[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    for handle in handles: handle.remove()


def test_no_provided_status_goal_or_pose_into_query_and_raw_branch_invariance():
    model = MotionDriveV2QueryAdapter(config(), query_adapter_on=True).eval()
    for function in (model.planner.forward, model.planner.adapt_queries, model.plan_from_features):
        names = set(inspect.signature(function).parameters)
        assert not names.intersection({"goal_xy", "goal", "pose", "history_transforms", "provided_status", "gt_state", "gt_history"})
    with pytest.raises(TypeError):
        model(**inputs(), provided_status=torch.ones(1, 6))
    with pytest.raises(TypeError):
        model.plan_from_features(None, None, None, None, goal_xy=torch.ones(1, 2))
    original = inputs()
    changed = {**original, "goal_xy": original["goal_xy"] + 10,
               "history_transforms": original["history_transforms"].clone()}
    changed["history_transforms"][..., 0, 3] += 1
    with torch.no_grad():
        a, b = model(**original), model(**changed)
    for key in ("motion_features", "state_hat", "history_hat"):
        assert torch.equal(a[key], b[key]), key
    assert not torch.equal(a["scene_features"], b["scene_features"])


@pytest.mark.parametrize("bad", [None, 0, 1, "true"])
def test_flag_is_explicit_boolean(bad):
    with pytest.raises(ValueError):
        MotionDriveV2QueryAdapter(config(), query_adapter_on=bad)


def test_exactly_four_history_and_state_token_on_required():
    for changes in ({"n_history": 3}, {"state_on": False}):
        with pytest.raises(ValueError):
            MotionDriveV2QueryAdapter(config(**changes))
