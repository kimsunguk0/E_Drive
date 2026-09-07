"""CPU-sized structural tests, not evidence of challenge compliance/accuracy."""
import inspect

import pytest
import torch

from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from models.motiondrive_v2.motion_encoder import local_correlation
from models.motiondrive_v2.scene_encoder import (
    grid_centers, masked_softmax, pixel_to_normalized_grid, project_scene_points,
)


@pytest.fixture(autouse=True)
def small_cpu_thread_count():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def tiny_config(**kwargs):
    values = dict(channels=16, backbone_arch="resnet34", grid_size=(6, 4),
                  x_range=(1, 9), y_range=(-2, 2), heights=(0.0, 1.0),
                  motion_grid=(3, 4), correlation_channels=8,
                  scene_attention_channels=8, scene_chunk_size=7,
                  planner_heads=4, planner_layers=1)
    values.update(kwargs)
    return MotionDriveV2Config(**values)


def tiny_inputs(batch=1):
    torch.manual_seed(19)
    projection = torch.tensor([[48., 20., 0., 0.], [32., 0., -20., 0.],
                               [1., 0., 0., 0.], [0., 0., 0., 1.]])
    return dict(images=torch.randn(batch, 6, 3, 64, 96),
                history_images=torch.randn(batch, 4, 3, 32, 48),
                lidar2img=projection[None, None].repeat(batch, 6, 1, 1),
                history_transforms=torch.eye(4)[None, None].repeat(batch, 4, 1, 1),
                time_offsets=torch.tensor([[.1, .2, .5, 1.]]).repeat(batch, 1),
                goal_xy=torch.tensor([[15., 3.]]).repeat(batch, 1))


def test_full_forward_shapes_fp32_heads_and_backward():
    model = MotionDriveV2(tiny_config()).eval()
    seen = {}
    hooks = []
    for name, module in (("xy", model.planner.xy_head),
                         ("history", model.motion_encoder.history_head),
                         ("state", model.motion_encoder.state_head)):
        hooks.append(module.register_forward_pre_hook(
            lambda _, args, name=name: seen.__setitem__(name, args[0].dtype)))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(**tiny_inputs())
    assert out["plan_abs"].shape == (1, 6, 2)
    assert out["occ_logits"].shape == out["lane_logits"].shape == (1, 1, 6, 4)
    assert out["scene_features"].shape == (1, 24, 16)
    assert out["motion_features"].shape == (1, 12, 16)
    assert out["history_hat"].shape == (1, 4, 4)
    assert out["state_hat"].shape == (1, 6)
    assert out["state_logvar"].shape == (1, 5)
    assert all(torch.isfinite(v).all() for v in out.values())
    assert out["plan_abs"].dtype == out["state_hat"].dtype == torch.float32
    assert set(seen.values()) == {torch.float32}
    loss = out["plan_abs"].square().mean() + out["occ_logits"].square().mean() + out["lane_logits"].square().mean()
    loss = loss + out["history_hat"].square().mean() + out["state_hat"].square().mean()
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name
    assert model.backbone_fpn.stem[0].weight.grad.abs().sum() > 0
    assert model.scene_encoder.query_context[0].weight.grad.abs().sum() > 0
    for hook in hooks:
        hook.remove()


def test_goal_scope_and_fixed_feature_planner():
    model = MotionDriveV2(tiny_config()).eval()
    inputs = tiny_inputs()
    with torch.no_grad():
        a = model(**inputs)
        b = model(**{**inputs, "goal_xy": inputs["goal_xy"] + torch.tensor([[20., -10.]])})
        for key in ("motion_features", "history_hat", "state_hat"):
            assert torch.equal(a[key], b[key]), key
        assert not torch.equal(a["scene_features"], b["scene_features"])
        again = model.plan_from_features(a["scene_features"], a["motion_features"],
                                         a["state_hat"], a["history_hat"])
        assert torch.equal(a["plan_abs"], again)
        model.config.goal_on = False
        offa = model(**inputs)
        offb = model(**{**inputs, "goal_xy": inputs["goal_xy"] * -10})
        assert torch.equal(offa["plan_abs"], offb["plan_abs"])
    assert "goal" not in str(inspect.signature(model.planner.forward))
    assert "pose" not in str(inspect.signature(model.motion_encoder.forward))


def test_pose_changes_scene_not_raw_motion():
    model = MotionDriveV2(tiny_config()).eval()
    inp = tiny_inputs()
    altered = inp["history_transforms"].clone()
    angle = torch.tensor(.15)
    altered[..., 0, 0] = altered[..., 2, 2] = angle.cos()
    altered[..., 0, 2], altered[..., 2, 0] = angle.sin(), -angle.sin()
    altered[..., 2, 3] = .3
    with torch.no_grad():
        a = model.forward_parts(**inp)
        b = model.forward_parts(**{**inp, "history_transforms": altered})
    assert not torch.equal(a["scene_features"], b["scene_features"])
    assert torch.equal(a["motion_features"], b["motion_features"])
    assert torch.equal(a["state_hat"], b["state_hat"])


def test_no_visible_samples_and_masked_softmax_are_finite():
    scores = torch.randn(2, 3, requires_grad=True)
    result = masked_softmax(scores, torch.zeros_like(scores, dtype=torch.bool))
    assert torch.equal(result, torch.zeros_like(result))
    result.sum().backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores.grad))
    model = MotionDriveV2(tiny_config()).eval()
    inp = tiny_inputs()
    inp["lidar2img"] = torch.zeros_like(inp["lidar2img"])
    with torch.no_grad():
        out = model(**inp)
    assert not out["scene_visible"].any()
    assert all(torch.isfinite(v).all() for v in out.values())


def test_state_mask_is_real_and_parameter_count_invariant():
    model = MotionDriveV2(tiny_config(state_on=False)).eval()
    n_off = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        out = model(**tiny_inputs())
        altered = model.plan_from_features(out["scene_features"], out["motion_features"],
                                           out["state_hat"] + 100, out["history_hat"] - 100)
        assert torch.equal(out["plan_abs"], altered)
    other = MotionDriveV2(tiny_config(state_on=True, goal_on=False))
    assert n_off == sum(p.numel() for p in other.parameters())


def test_grid_axes_pixel_center_and_full_se3_projection():
    centers = grid_centers((2, 4), (-10, 70), (-32, 32))
    assert centers[0, 0].tolist() == [10., -24.]
    assert centers[-1, -1].tolist() == [50., 24.]
    uv = torch.tensor([[0., 0.], [95., 63.], [47.5, 31.5]])
    norm = pixel_to_normalized_grid(uv, (64, 96))
    recovered = (norm + 1) * torch.tensor([96., 64.]) / 2 - .5
    torch.testing.assert_close(recovered, uv)
    assert torch.equal(norm[-1], torch.zeros(2))
    points = torch.tensor([[[4., 0., 0., 1.]]])
    matrix = tiny_inputs()["lidar2img"][:, :1]
    grid, valid = project_scene_points(points, matrix, (64, 96))
    shifted = torch.eye(4)
    shifted[2, 3] = 1.
    grid_z, _ = project_scene_points(points, matrix @ shifted, (64, 96))
    assert valid.all()
    assert not torch.equal(grid, grid_z)


def test_raw_cost_volume_preserves_translation_bin():
    # Unique channels at each position avoid ambiguous repeated texture.
    current = torch.eye(25).reshape(25, 5, 5)[None]
    past = torch.zeros_like(current)
    past[..., :, 1:] = current[..., :, :-1]
    corr = local_correlation(current, past, 1)
    # dy=0, dx=+1 => flattened local 3x3 index 5.
    assert corr[0, :, 2, 2].argmax().item() == 5


def test_public_backbone_mapping_does_not_load_compact_fpn(tmp_path):
    model = MotionDriveV2(tiny_config()).eval()
    public = {}
    for key, value in model.backbone_fpn.state_dict().items():
        if key.startswith("stem.0."):
            target = "backbone.conv1." + key[len("stem.0."):]
        elif key.startswith("stem.1."):
            target = "backbone.bn1." + key[len("stem.1."):]
        elif key.startswith(("layer1.", "layer2.", "layer3.", "layer4.")):
            target = "backbone." + key
        else:
            continue
        public[target] = value.clone()
    # A same-shaped malicious/unintended FPN field must still be ignored.
    fpn_before = model.backbone_fpn.lat2.weight.detach().clone()
    public["backbone_fpn.lat2.weight"] = torch.zeros_like(fpn_before)
    path = tmp_path / "public_resnet.pth"
    torch.save({"state_dict": public}, path)
    report = model.load_pretrained_backbone(path)
    assert report["injected"] > 100
    assert report["nonhead_missing"] == []
    assert report["unexpected"] == []
    assert not report["fpn_initialized_from_checkpoint"]
    assert torch.equal(fpn_before, model.backbone_fpn.lat2.weight)


def test_unit_reparameterization_preserves_initial_physical_plan_and_legacy_keys():
    model = MotionDriveV2(tiny_config()).eval()
    keys_before = set(model.state_dict())
    param_count = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        # Test preservation at driving-scale metres, not only random tiny output.
        model.planner.xy_head[-1].weight.mul_(100)
        model.planner.xy_head[-1].bias.mul_(100)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            before = model(**tiny_inputs())
        report = model.reparameterize_plan_output_scale((20., 5.), preserve_function=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            after = model(**tiny_inputs())
    assert after["plan_abs"].dtype == torch.float32
    assert (after["plan_abs"] - before["plan_abs"]).abs().max() < 1e-4
    assert report["old_scale"] == [1., 1.]
    assert report["new_scale"] == [20., 5.]
    assert not report["optimizer_state_transformed"]
    assert set(model.state_dict()) == keys_before
    assert sum(p.numel() for p in model.parameters()) == param_count
    # Old state dicts have no output-scale keys and remain strict-load compatible.
    legacy = MotionDriveV2(tiny_config())
    legacy.load_state_dict(model.state_dict(), strict=True)
    # Correct metadata must still be saved/restored with the checkpoint.
    restored = MotionDriveV2(model.config.to_dict()).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(restored(**tiny_inputs())["plan_abs"], after["plan_abs"])


@pytest.mark.parametrize("bad", [(0, 1), (-1, 1), (1,), (1, 2, 3),
                                  (float("nan"), 1), (1, float("inf")), 2., "12"])
def test_unit_scale_validation_happens_before_modifying_weights(bad):
    with pytest.raises(ValueError):
        tiny_config(plan_output_scale=bad)
    model = MotionDriveV2(tiny_config())
    before = model.planner.xy_head[-1].weight.detach().clone()
    with pytest.raises(ValueError):
        model.reparameterize_plan_output_scale(bad)
    assert torch.equal(before, model.planner.xy_head[-1].weight)
    assert model.config.plan_output_scale == (1., 1.)


def test_units_without_preservation_only_change_fixed_head_scale():
    model = MotionDriveV2(tiny_config()).eval()
    with torch.no_grad():
        before = model(**tiny_inputs())["plan_abs"]
        old_weight = model.planner.xy_head[-1].weight.clone()
        model.reparameterize_plan_output_scale((3., 2.), preserve_function=False)
        after = model(**tiny_inputs())["plan_abs"]
    torch.testing.assert_close(after, before * torch.tensor([3., 2.]))
    assert torch.equal(old_weight, model.planner.xy_head[-1].weight)


def test_high_low_motion_modes_have_identical_backbone_inputs_bn_and_scene():
    high = MotionDriveV2(tiny_config(motion_input_mode="high_feature")).train()
    low = MotionDriveV2(tiny_config(motion_input_mode="low_feature")).train()
    low.load_state_dict(high.state_dict(), strict=True)
    assert set(high.state_dict()) == set(low.state_dict())
    calls = {"high": [], "low": []}
    hooks = [high.backbone_fpn.register_forward_pre_hook(
                 lambda _, args: calls["high"].append(args[0].detach().clone())),
             low.backbone_fpn.register_forward_pre_hook(
                 lambda _, args: calls["low"].append(args[0].detach().clone()))]
    inp = tiny_inputs()
    with torch.no_grad():
        a, b = high(**inp), low(**inp)
    for hook in hooks:
        hook.remove()
    assert [tuple(x.shape) for x in calls["high"]] == [(6, 3, 64, 96), (5, 3, 32, 48)]
    assert len(calls["high"]) == len(calls["low"]) == 2
    assert all(torch.equal(x, y) for x, y in zip(calls["high"], calls["low"]))
    expected_small = torch.nn.functional.interpolate(inp["images"][:, 0], size=(32, 48),
                                                    mode="bilinear", align_corners=False,
                                                    antialias=True)
    assert torch.equal(calls["high"][1][:1], expected_small)
    assert torch.equal(calls["high"][1][1:], inp["history_images"].flatten(0, 1))
    # Match scene history BN by using five frames in BOTH experimental arms.
    assert torch.equal(a["scene_features"], b["scene_features"])
    assert not torch.equal(a["motion_features"], b["motion_features"])
    for (na, ba), (nb, bb) in zip(high.backbone_fpn.named_buffers(), low.backbone_fpn.named_buffers()):
        assert na == nb
        assert torch.equal(ba, bb), na
    assert sum(p.numel() for p in high.parameters()) == sum(p.numel() for p in low.parameters())


@pytest.mark.parametrize("mode", ["high_feature", "low_feature"])
def test_new_motion_modes_do_not_read_goal_or_pose_and_keep_fp32_heads(mode):
    model = MotionDriveV2(tiny_config(motion_input_mode=mode)).eval()
    inp = tiny_inputs()
    pose = inp["history_transforms"].clone()
    pose[..., 2, 3] = .4
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        a = model(**inp)
        b = model(**{**inp, "goal_xy": inp["goal_xy"] * -4, "history_transforms": pose})
    for key in ("motion_features", "state_hat", "history_hat"):
        assert torch.equal(a[key], b[key]), key
    assert a["plan_abs"].dtype == a["state_hat"].dtype == a["history_hat"].dtype == torch.float32
    assert not torch.equal(a["scene_features"], b["scene_features"])


def test_legacy_motion_mode_preserves_original_backbone_batch_shapes():
    model = MotionDriveV2(tiny_config()).eval()
    shapes = []
    hook = model.backbone_fpn.register_forward_pre_hook(lambda _, args: shapes.append(tuple(args[0].shape)))
    with torch.no_grad():
        model(**tiny_inputs())
    hook.remove()
    assert shapes == [(6, 3, 64, 96), (4, 3, 32, 48)]
    with pytest.raises(ValueError):
        tiny_config(motion_input_mode="unknown")
