"""CPU mock-model diagnostics only; no driving dataset, accelerator, or training."""
import dataclasses
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_motiondrive_v2_loss_gradients as probe
from models.motiondrive_v2 import MotionDriveV2Config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class TinyModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or MotionDriveV2Config()
        self.backbone_fpn = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Tanh())
        self.scene_encoder = nn.ModuleDict({"shared": nn.Linear(4, 4),
                                           "occ_head": nn.Linear(4, 4), "lane_head": nn.Linear(4, 4)})
        self.motion_encoder = nn.ModuleDict({"shared": nn.Linear(4, 4),
            "history_head": nn.Linear(4, 16), "state_head": nn.Linear(4, 6),
            "history_uncertainty_head": nn.Linear(4, 16), "state_uncertainty_head": nn.Linear(4, 5)})
        self.planner = nn.ModuleDict({"xy_head": nn.Linear(4, 12)})
        self.forward_calls = 0
        self.mutate_buffer = False
        self.input_keys = None

    def forward(self, **inputs):
        self.forward_calls += 1
        self.input_keys = set(inputs)
        assert not self.training
        assert not any(m.training for m in self.modules() if isinstance(m, nn.BatchNorm1d))
        images = inputs["images"]
        batch = len(images)
        shared = self.backbone_fpn(images.flatten(1)[:, :4])
        scene = self.scene_encoder["shared"](shared)
        motion = self.motion_encoder["shared"](shared)
        occ = self.scene_encoder["occ_head"](scene).reshape(batch, 1, 2, 2).float()
        lane = self.scene_encoder["lane_head"](scene).reshape(batch, 1, 2, 2).float()
        with torch.autocast(device_type=images.device.type, enabled=False):
            history = self.motion_encoder["history_head"](motion.float()).reshape(batch, 4, 4)
            state = self.motion_encoder["state_head"](motion.float())
            plan = self.planner["xy_head"]((scene + motion).float()).reshape(batch, 6, 2)
            history_lv = self.motion_encoder["history_uncertainty_head"](motion.float()).reshape(batch, 4, 4).clamp(-8, 8)
            state_lv = self.motion_encoder["state_uncertainty_head"](motion.float()).clamp(-8, 8)
        if self.mutate_buffer:
            self.backbone_fpn[1].num_batches_tracked.add_(1)
        return {"plan_abs": plan, "occ_logits": occ, "lane_logits": lane, "history_hat": history,
                "state_hat": state, "history_logvar": history_lv, "state_logvar": state_lv}


def batch_fixture(size=3):
    generator = torch.Generator().manual_seed(44)
    return {"images": torch.randn(size, 6, 3, 2, 2, generator=generator),
            "history_images": torch.randn(size, 4, 3, 2, 2, generator=generator),
            "lidar2img": torch.eye(4).repeat(size, 6, 1, 1),
            "history_transforms": torch.eye(4).repeat(size, 4, 1, 1),
            "time_offsets": torch.tensor([.1, .2, .5, 1.]).repeat(size, 1),
            "goal_xy": torch.ones(size, 2), "gt_plan": torch.ones(size, 6, 2),
            "plan_valid": torch.ones(size, 6, dtype=torch.bool),
            "occ_target": torch.ones(size, 1, 2, 2), "occ_valid": torch.ones(size, 1, 2, 2, dtype=torch.bool),
            "lane_target": torch.zeros(size, 1, 2, 2), "lane_valid": torch.ones(size, 1, 2, 2, dtype=torch.bool),
            "history_target": torch.zeros(size, 4, 4), "history_valid": torch.ones(size, 4, dtype=torch.bool),
            "state_target": torch.zeros(size, 6), "state_valid": torch.ones(size, 6, dtype=torch.bool),
            "row": torch.arange(size), "frame": torch.arange(size) + 30,
            "scenario": [f"scene_{i}" for i in range(size)]}


def save_checkpoint(path, model):
    torch.save({"step": 500, "model": model.state_dict(), "manifest": {
        "model_config": dataclasses.asdict(model.config),
        "loss_weights": dataclasses.asdict(probe.LossWeights()), "arguments": {"bn_policy": "fixed"}}}, path)


def test_single_forward_components_and_bitwise_immutability(monkeypatch):
    torch.manual_seed(5)
    model = TinyModel().train()
    # Existing gradients also belong to the caller and must remain untouched.
    first = next(model.parameters())
    first.grad = torch.ones_like(first)
    before = probe.fingerprints(model)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *_a, **_k: pytest.fail("No optimizer may be constructed"))
    result = probe.probe_batch(model, batch_fixture(), probe.LossWeights(), "cpu")
    assert model.forward_calls == result["forward_count"] == 1
    assert model.input_keys == set(probe.MODEL_INPUTS)
    assert model.training  # prior mode restored after temporary eval
    assert result["bn_training_modules_during_probe"] == 0
    assert before == probe.fingerprints(model)
    assert result["immutability"]["bitwise_unchanged"]
    assert set(result["gradient_norms"]) == set(probe.TERMS)
    assert result["gradient_norms"]["plan"]["all_parameters"]["raw_l2_norm"] > 0
    assert result["gradient_norms"]["motion"]["planner.xy_head"]["raw_l2_norm"] == 0
    assert result["shared_gradient_pairs"]["plan__vs__motion"]["all_shared_parameters"]["connected_parameter_numel"] > 0
    assert all(dtype == "torch.float32" for values in result["coordinate_state_head_input_dtypes"].values() for dtype in values)
    for name, coefficient in result["coefficients_in_total"].items():
        row = result["gradient_norms"][name]["all_parameters"]
        assert row["weighted_contribution_l2_norm"] == pytest.approx(coefficient * row["raw_l2_norm"])


def test_shared_cosine_ignores_private_parameters_and_applies_loss_weights():
    named = [("backbone_fpn.weight", nn.Parameter(torch.ones(2))),
             ("planner.weight", nn.Parameter(torch.ones(2)))]
    a = [torch.tensor([1., 0.]), torch.tensor([100., 100.])]
    b = [torch.tensor([-2., 0.]), None]
    result = probe.shared_gradient_statistics(named, a, b, 1., .2)["all_shared_parameters"]
    assert result["connected_parameter_numel"] == 2
    assert result["raw_dot"] == -2 and result["raw_cosine"] == -1
    assert result["weighted_dot"] == pytest.approx(-.4)
    assert result["weighted_b_norm"] == pytest.approx(.4)
    zero = probe.shared_gradient_statistics(named, a, b, 1., 0.)["all_shared_parameters"]
    assert zero["weighted_cosine"] is None


def test_logvar_clamp_rates_use_only_valid_labels():
    values = torch.tensor([[[-8., -6., 0., 6.], [8., 8., 8., 8.]]])
    result = probe.logvar_statistics(values, torch.tensor([[True, False]]))["all_valid"]
    assert result["n_valid"] == 4
    assert result["loss_clamp_below_minus6_fraction"] == .25
    assert result["at_or_beyond_loss_lower_fraction"] == .5
    assert result["at_or_beyond_loss_upper_fraction"] == .25
    assert result["loss_clamp_above_plus6_fraction"] == 0
    assert result["at_or_beyond_model_plus8_fraction"] == 0
    assert probe.logvar_statistics(values, torch.zeros(1, 2, dtype=torch.bool))["all_valid"] == {"n_valid": 0}


@pytest.mark.parametrize("key", ["plan_valid", "history_valid", "state_valid"])
def test_fixture_requires_explicit_validity_masks(tmp_path, key):
    batch = batch_fixture()
    del batch[key]
    path = tmp_path / "missing_validity.pt"
    torch.save({"batch": batch, "metadata": {}}, path)
    with pytest.raises(ValueError, match=key):
        probe.load_fixture(path)


def test_planner_submodules_are_reported_separately():
    for child in ("xy_head", "decoder", "state_projection", "scene_position"):
        assert probe.module_group(f"planner.{child}.weight") == f"planner.{child}"
    assert probe.module_group("planner.waypoint_queries") == "planner.other"


def test_buffer_mutation_is_detected():
    model = TinyModel()
    model.mutate_buffer = True
    with pytest.raises(RuntimeError, match="modified weights, buffers"):
        probe.probe_batch(model, batch_fixture(), probe.LossWeights(), "cpu")


def test_missing_uncertainty_cannot_silently_switch_to_smooth_l1():
    class MissingUncertainty(TinyModel):
        def forward(self, **inputs):
            result = super().forward(**inputs)
            result.pop("state_logvar")
            return result
    with pytest.raises(ValueError, match="silently change the loss"):
        probe.probe_batch(MissingUncertainty(), batch_fixture(), probe.LossWeights(), "cpu")


def test_checkpoint_requires_full_config_and_strict_state(tmp_path):
    path = tmp_path / "checkpoint.pth"
    model = TinyModel(MotionDriveV2Config(goal_on=False, state_on=True, plan_output_scale=(10., 5.), motion_input_mode="low_feature"))
    save_checkpoint(path, model)
    loaded, _, info = probe.load_checkpoint(path, "cpu", TinyModel)
    assert not loaded.config.goal_on and loaded.config.state_on
    assert loaded.config.plan_output_scale == (10., 5.)
    assert loaded.config.motion_input_mode == "low_feature"
    assert info["configuration_overrides"] == {}
    assert probe.fingerprints(loaded) == probe.fingerprints(model)
    saved = torch.load(path, weights_only=True)
    saved["manifest"]["model_config"].pop("state_on")
    torch.save(saved, path)
    with pytest.raises(ValueError, match="complete, exact"):
        probe.load_checkpoint(path, "cpu", TinyModel)
    save_checkpoint(path, model)
    saved = torch.load(path, weights_only=True)
    saved["model"].pop(next(iter(saved["model"])))
    torch.save(saved, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        probe.load_checkpoint(path, "cpu", TinyModel)


def test_fixture_and_run_preserve_files_and_separate_small_batches(tmp_path):
    ckpt, fixture = tmp_path / "ckpt.pth", tmp_path / "fixture.pt"
    save_checkpoint(ckpt, TinyModel())
    torch.save({"batch": batch_fixture(3), "metadata": {"split": "train", "selection": "fixed diagnostic"}}, fixture)
    before = [probe.sha256(path) for path in (ckpt, fixture)]
    result = probe.run_probe(ckpt, fixture, device="cpu", batch_size=2, max_samples=3, model_factory=TinyModel)
    assert result["probed_samples"] == 3
    assert [r["batch_size"] for r in result["batches"]] == [2, 1]
    assert [r["fixture_slice"] for r in result["batches"]] == [[0, 2], [2, 3]]
    assert result["batches"][1]["sample_identity"]["row"] == [2]
    assert result["input_files_unchanged"]
    assert result["optimizer_steps"] == result["bn_adaptive_passes"] == 0
    assert before == [probe.sha256(path) for path in (ckpt, fixture)]
    assert result["fixture_metadata"]["split"] == "train"


def test_fixture_requires_labels_and_batch_consistency(tmp_path):
    fixture = tmp_path / "bad.pt"
    batch = batch_fixture()
    batch.pop("gt_plan")
    torch.save({"batch": batch}, fixture)
    with pytest.raises(ValueError, match="gt_plan"):
        probe.load_fixture(fixture)
    batch = batch_fixture()
    batch["state_target"] = torch.zeros(1, 6)
    torch.save({"batch": batch}, fixture)
    with pytest.raises(ValueError, match="dimension mismatch"):
        probe.load_fixture(fixture)


@pytest.mark.parametrize("device,batch_size,max_samples", [("cuda:4", 4, 8), ("cuda", 4, 8),
                                                          ("cpu", 9, 8), ("cpu", 4, 9)])
def test_out_of_scope_execution_is_rejected_without_loading(device, batch_size, max_samples):
    with pytest.raises(ValueError):
        probe.run_probe("does-not-exist", "does-not-exist", device=device,
                        batch_size=batch_size, max_samples=max_samples)
