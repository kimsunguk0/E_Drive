import sys
import json
import signal
from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from motiondrive_v2_training import (LossWeights, MODEL_INPUTS, balanced_raster_bce,
                                     compute_loss, model_inputs, set_training_mode,
                                     tensor_state_sha256, time_input_policy, weighted_d3)


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


def _labeled_time_batch():
    batch = {key: torch.ones(2, 1) for key in MODEL_INPUTS}
    batch["time_offsets"] = torch.tensor([[.098, .195, .490, .985], [.101, .204, .508, 1.018]],
                                           dtype=torch.float64)
    batch.update(gt_plan=torch.arange(24).reshape(2, 6, 2).float(),
                 history_target=torch.arange(32).reshape(2, 4, 4).float(),
                 history_valid=torch.ones(2, 4, 4, dtype=torch.bool),
                 state_target=torch.arange(12).reshape(2, 6).float(),
                 state_valid=torch.ones(2, 6, dtype=torch.bool),
                 scenario=["train_a", "train_b"])
    return batch


def test_raw_time_default_preserves_every_tensor_identity_and_rng():
    batch = _labeled_time_batch()
    rng = torch.get_rng_state().clone()
    implicit, explicit = model_inputs(batch), model_inputs(batch, time_input="raw")
    assert set(implicit) == set(explicit) == set(MODEL_INPUTS)
    assert all(implicit[k] is batch[k] and explicit[k] is batch[k] for k in MODEL_INPUTS)
    assert torch.equal(rng, torch.get_rng_state())
    assert batch["time_offsets"].dtype == torch.float64


def test_nominal_changes_only_forward_time_offsets_not_raw_or_supervision():
    batch = _labeled_time_batch()
    before = {k: v.clone() if isinstance(v, torch.Tensor) else list(v) for k, v in batch.items()}
    rng = torch.get_rng_state().clone()
    forward = model_inputs(batch, time_input="nominal")
    expected = torch.tensor([[.1, .2, .5, 1.], [.1, .2, .5, 1.]], dtype=torch.float32)
    assert torch.equal(forward["time_offsets"], expected)
    assert forward["time_offsets"].dtype == torch.float32
    assert forward["time_offsets"].device == batch["time_offsets"].device
    assert forward["time_offsets"] is not batch["time_offsets"]
    assert all(forward[k] is batch[k] for k in MODEL_INPUTS if k != "time_offsets")
    assert set(forward) == set(MODEL_INPUTS)
    for key, value in batch.items():
        assert torch.equal(value, before[key]) if isinstance(value, torch.Tensor) else value == before[key]
    assert torch.equal(rng, torch.get_rng_state())
    assert time_input_policy("nominal")["supervision_and_dataset_modified"] is False


@pytest.mark.parametrize("bad", [torch.zeros(2, 3), torch.zeros(1, 4), torch.zeros(2, 4, 1), None])
def test_nominal_rejects_wrong_time_shape(bad):
    batch = _labeled_time_batch()
    batch["time_offsets"] = bad
    with pytest.raises(ValueError, match="four historical offsets"):
        model_inputs(batch, time_input="nominal")


def test_unknown_time_policy_rejected():
    with pytest.raises(ValueError, match="raw or nominal"):
        model_inputs(_labeled_time_batch(), time_input="guess")
    with pytest.raises(ValueError, match="raw or nominal"):
        time_input_policy("guess")


def _restore_time_fixture(saved_mode, *, resume=False):
    args = SimpleNamespace(resume="last.pth" if resume else None,
                           goal_on=1, state_on=1, arch="resnet50", phase="joint", time_input="raw")
    saved_args = {"phase": "joint", "alpha_occ": .2, "alpha_lane": .2, "alpha_motion": .2,
                  "uncertainty": 1, "lr": 1e-4, "backbone_lr": 1e-5, "weight_decay": .01,
                  "warmup": 200, "precision": "bf16", "seed": 0}
    if saved_mode is not None:
        saved_args["time_input"] = saved_mode
    saved = {"model_config": {"goal_on": True, "state_on": True, "backbone_arch": "resnet50"},
             "arguments": saved_args}
    return args, saved


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("saved_mode", [None, "raw", "nominal"])
def test_eval_resume_restores_saved_time_and_legacy_defaults_raw(resume, saved_mode):
    from train_motiondrive_v2 import restore_run_configuration
    args, saved = _restore_time_fixture(saved_mode, resume=resume)
    args.time_input = "nominal" if saved_mode is None else "raw"
    restore_run_configuration(args, saved, set())
    assert args.time_input == (saved_mode or "raw")


@pytest.mark.parametrize("saved_mode,requested", [("raw", "nominal"), ("nominal", "raw"), (None, "nominal")])
def test_explicit_time_policy_conflict_rejected(saved_mode, requested):
    from train_motiondrive_v2 import restore_run_configuration
    args, saved = _restore_time_fixture(saved_mode)
    args.time_input = requested
    with pytest.raises(ValueError, match="incompatible explicit --time-input"):
        restore_run_configuration(args, saved, {"--time-input"})
    args.time_input = saved_mode or "raw"
    restore_run_configuration(args, saved, {"--time-input"})


@pytest.mark.parametrize("extra", [{"time_input": "raw"}, {"time_input_policy": {"mode": "raw"}},
                                   {"time_input_policy": None}])
def test_malformed_or_contradictory_checkpoint_time_declaration_rejected(extra):
    from train_motiondrive_v2 import restore_run_configuration
    args, saved = _restore_time_fixture("nominal")
    saved.update(extra)
    with pytest.raises(ValueError, match="Checkpoint time_input"):
        restore_run_configuration(args, saved, set())


def test_time_cli_default_explicit_and_no_abbreviation_bypass():
    from train_motiondrive_v2 import arguments
    required = ["--split-manifest", "split.json", "--supervision-root", "sup", "--run-dir", "run"]
    assert arguments(required).time_input == "raw"
    assert arguments(required + ["--time-input=nominal"]).time_input == "nominal"
    with pytest.raises(SystemExit):
        arguments(required + ["--time-input", "guess"])
    with pytest.raises(SystemExit):
        arguments(required + ["--time-in", "nominal"])


@pytest.mark.parametrize("time_mode", ["raw", "nominal"])
@pytest.mark.parametrize("microbatch", [0, 1, 2])
def test_cpu_mock_all_forward_paths_use_same_policy_and_loss_gets_original_gt(tmp_path, monkeypatch, time_mode, microbatch):
    """Two-step scalar mock only: no driving data, real V2 network, or GPU run."""
    import motiondrive_v2_data as data_module
    import models.motiondrive_v2 as model_module
    import train_motiondrive_v2 as trainer

    raw_times = torch.tensor([.097, .194, .487, .978], dtype=torch.float64)
    nominal_times = torch.tensor([.1, .2, .5, 1.], dtype=torch.float32)
    calls, losses = [], []

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, **kwargs):
            self.rows = np.arange(2)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {
                "images": torch.full((6, 3, 4, 4), float(index)),
                "history_images": torch.full((4, 3, 2, 2), float(index)),
                "lidar2img": torch.eye(4).repeat(6, 1, 1),
                "history_transforms": torch.eye(4).repeat(4, 1, 1),
                "time_offsets": raw_times.clone(), "goal_xy": torch.zeros(2),
                "gt_plan": torch.full((6, 2), index / 10.), "plan_valid": torch.ones(6, dtype=torch.bool),
                "history_target": torch.full((4, 4), index / 10.),
                "history_valid": torch.ones(4, 4, dtype=torch.bool),
                "state_target": torch.full((6,), index / 10.), "state_valid": torch.ones(6, dtype=torch.bool),
                "occ_target": torch.zeros(1, 2, 2), "occ_valid": torch.ones(1, 2, 2, dtype=torch.bool),
                "lane_target": torch.zeros(1, 2, 2), "lane_valid": torch.ones(1, 2, 2, dtype=torch.bool),
                "row": index, "frame": 30 + index, "scenario": f"scene{index}", "session_id": "session",
            }

    class FakeModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.bias = torch.nn.Parameter(torch.tensor(.2))

        def reparameterize_plan_output_scale(self, scale, preserve_function=True):
            assert tuple(scale) == self.config.plan_output_scale == (1., 1.)
            assert preserve_function

        def forward(self, **inputs):
            assert set(inputs) == set(MODEL_INPUTS)
            times = inputs["time_offsets"]
            calls.append({"training": self.training, "time_offsets": times.clone()})
            b = len(inputs["images"])
            value = self.bias + times.float().mean(0).sum() / 100.
            return {"plan_abs": value.expand(b, 6, 2), "history_hat": value.expand(b, 4, 4),
                    "state_hat": value.expand(b, 6), "occ_logits": value.expand(b, 1, 2, 2),
                    "lane_logits": value.expand(b, 1, 2, 2)}

    real_compute_loss = trainer.compute_loss

    def checked_loss(outputs, batch, weights, **kwargs):
        assert torch.equal(batch["time_offsets"], raw_times.expand(len(batch["row"]), -1))
        for row, gt, history, state in zip(batch["row"], batch["gt_plan"], batch["history_target"], batch["state_target"]):
            assert torch.equal(gt, torch.full_like(gt, float(row) / 10.))
            assert torch.equal(history, torch.full_like(history, float(row) / 10.))
            assert torch.equal(state, torch.full_like(state, float(row) / 10.))
        assert all(batch[key].all() for key in ("plan_valid", "history_valid", "state_valid", "occ_valid", "lane_valid"))
        losses.append(True)
        return real_compute_loss(outputs, batch, weights, **kwargs)

    monkeypatch.setattr(data_module, "MotionDriveDataset", FakeDataset)
    monkeypatch.setattr(model_module, "MotionDriveV2", FakeModel)
    monkeypatch.setattr(trainer, "compute_loss", checked_loss)
    monkeypatch.setattr(trainer, "source_sha", lambda: "cpu-mock-test")
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trainer, "ACTIVE_RUN_DIR", None)
    # The production trainer installs graceful handlers. Tests must not change pytest's handlers.
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    split = tmp_path / "split.json"
    split.write_text("{}")
    supervision = tmp_path / "supervision"
    supervision.mkdir()
    (supervision / "supervision_manifest.json").write_text("{}")
    required = ["--cpu", "--allow-unpretrained", "--split-manifest", str(split),
                "--supervision-root", str(supervision), "--steps", "2", "--batch", "2",
                "--eval-batch", "2", "--workers", "0", "--eval-every", "1", "--save-every", "2",
                "--log-every", "1", "--bn-policy", "fixed", "--microbatch", str(microbatch)]

    def run(name, extra):
        target = tmp_path / name
        monkeypatch.setattr(sys, "argv", ["train_motiondrive_v2.py", *required, "--run-dir", str(target), *extra])
        trainer.main()
        return target

    def check_calls(mode, expected_training):
        expected = nominal_times if mode == "nominal" else raw_times
        if microbatch == 1:
            expected_training = [value for is_training in expected_training
                                 for value in ([True, True] if is_training else [False])]
        assert [call["training"] for call in calls] == expected_training
        for call in calls:
            assert call["time_offsets"].dtype == expected.dtype
            assert torch.equal(call["time_offsets"], expected.expand(len(call["time_offsets"]), -1))

    try:
        # Raw omits the new flag entirely; this exercises the backward-compatible default.
        run_dir = run("initial", [] if time_mode == "raw" else ["--time-input", time_mode])
        check_calls(time_mode, [False, True, False, True, False])
        assert len(losses) == (4 if microbatch == 1 else 2)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["time_input"] == manifest["arguments"]["time_input"] == time_mode
        assert manifest["time_input_policy"]["supervision_and_dataset_modified"] is False
        assert manifest["microbatch_policy"]["microbatch"] == microbatch
        assert json.loads((run_dir / "initial_eval.json").read_text())["time_input"] == time_mode
        assert json.loads((run_dir / "latest_eval.json").read_text())["time_input"] == time_mode
        checkpoint = run_dir / "last.pth"

        calls.clear()
        eval_dir = run("eval", ["--init", str(checkpoint), "--eval-only"])
        check_calls(time_mode, [False])
        assert json.loads((eval_dir / "evaluation.json").read_text())["report"]["time_input"] == time_mode

        calls.clear()
        other_mode = "nominal" if time_mode == "raw" else "raw"
        with pytest.raises(ValueError, match="incompatible explicit --time-input"):
            run("bad_eval", ["--init", str(checkpoint), "--eval-only", "--time-input", other_mode])
        assert not calls

        # Fresh --init is a new experiment; changing the input policy is intentional and allowed.
        init_dir = run("new_policy", ["--init", str(checkpoint), "--time-input", other_mode])
        check_calls(other_mode, [False, True, False, True, False])
        assert json.loads((init_dir / "manifest.json").read_text())["time_input"] == other_mode

        calls.clear()
        resumed = run("resumed", ["--resume", str(checkpoint), "--steps", "3"])
        check_calls(time_mode, [True, False])
        assert json.loads((resumed / "manifest.json").read_text())["time_input"] == time_mode
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
