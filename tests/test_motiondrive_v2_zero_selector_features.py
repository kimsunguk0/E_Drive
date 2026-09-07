"""CPU contract tests only; these do not create P5-Z model/cache results."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts import cache_motiondrive_v2_zero_selector_features as cache


class FakePlanner(nn.Module):
    def __init__(self):
        super().__init__()
        self.xy_head = nn.Linear(cache.HIDDEN_CHANNELS, 2)


class FakeModel(nn.Module):
    def __init__(self, twice=False, fail=False):
        super().__init__()
        self.planner = FakePlanner()
        self.register_buffer("counter", torch.tensor(3.))
        self.twice, self.fail = twice, fail

    def forward(self, images, history_images, lidar2img, history_transforms,
                time_offsets, goal_xy):
        b = images.shape[0]
        decoded = images[:, 0, 0, :6, :cache.HIDDEN_CHANNELS].float()
        plan = self.planner.xy_head(decoded) * decoded.new_tensor([10., 5.])
        if self.twice:
            self.planner.xy_head(decoded)
        if self.fail:
            raise RuntimeError("synthetic forward failure")
        return {"plan_abs": plan.float(), "state_hat": torch.zeros(b, 6),
                "history_hat": torch.zeros(b, 4, 4), "other": decoded.sum(2)}


def fake_batch(n=2):
    images = torch.arange(n * 6 * 3 * 6 * cache.HIDDEN_CHANNELS, dtype=torch.float32).reshape(
        n, 6, 3, 6, cache.HIDDEN_CHANNELS) / 10000
    return {"images": images, "history_images": torch.zeros(n, 4, 3, 2, 2),
            "lidar2img": torch.eye(4).repeat(n, 6, 1, 1),
            "history_transforms": torch.eye(4).repeat(n, 4, 1, 1),
            "time_offsets": torch.ones(n, 4), "goal_xy": torch.zeros(n, 2),
            "gt_plan": torch.zeros(n, 6, 2), "plan_valid": torch.ones(n, 6, dtype=torch.bool),
            "state_target": torch.zeros(n, 6), "state_valid": torch.ones(n, 6, dtype=torch.bool),
            "row": torch.arange(10, 10 + n), "frame": torch.arange(30, 30 + n),
            "scenario": [f"scene{i}" for i in range(n)],
            "session_id": [f"session{i}" for i in range(n)]}


def training_manifest(seed=0):
    return {"status": "running", "git_sha": cache.P4_GIT_SHA,
            "arguments": {"phase": "joint", "steps": 6000, "seed": seed,
                          "precision": "bf16", "time_input": "nominal", "bn_policy": "fixed"},
            "model_config": {"backbone_arch": "resnet50", "goal_on": True,
                             "state_on": True, "motion_input_mode": "low_feature",
                             "n_history": 4, "plan_output_scale": [10., 5.]},
            "split_sha256": cache.SPLIT_SHA256,
            "supervision_manifest_sha256": cache.SUPERVISION_SHA256,
            "data_counts": {"train": 54810, "eval": 1998},
            "train_rows_sha256": "a" * 64, "eval_rows_sha256": "b" * 64,
            "time_input": "nominal", "time_input_policy": {"mode": "nominal"},
            "loss_weights": {"plan": 1.}}


def test_feature_contract_is_exact_fp32_and_whitelisted():
    decoded = torch.randn(3, 6, 128)
    state = torch.randn(3, 6)
    history = torch.randn(3, 4, 4)
    value = cache.compose_selector_feature(decoded, state, history)
    assert value.shape == (3, 790) and value.dtype == torch.float32
    assert torch.equal(value[:, :768], decoded.flatten(1))
    assert torch.equal(value[:, 768:774], state)
    assert torch.equal(value[:, 774:], history.flatten(1))
    assert [row["name"] for row in cache.FEATURE_COMPONENTS] == [
        "planner_decoded_fp32", "predicted_state_hat_fp32", "predicted_history_hat_fp32"]


@pytest.mark.parametrize("which", ["decoded_shape", "state_shape", "history_shape", "dtype", "finite"])
def test_feature_contract_rejects_wrong_component(which):
    decoded, state, history = torch.zeros(2, 6, 128), torch.zeros(2, 6), torch.zeros(2, 4, 4)
    if which == "decoded_shape": decoded = decoded[:, :5]
    if which == "state_shape": state = state[:, :5]
    if which == "history_shape": history = history[:, :3]
    if which == "dtype": decoded = decoded.double()
    if which == "finite": history[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        cache.compose_selector_feature(decoded, state, history)


def test_hook_observes_exact_xy_head_input_once_and_is_removed():
    model, batch = FakeModel(), fake_batch()
    original_hooks = len(model.planner.xy_head._forward_pre_hooks)
    output, feature = cache.forward_with_decoded(
        model, cache.model_inputs(batch, "nominal"), cache._autocast(torch.device("cpu")))
    assert feature.shape == (2, 790)
    assert output["plan_abs"].shape == (2, 6, 2)
    assert len(model.planner.xy_head._forward_pre_hooks) == original_hooks


@pytest.mark.parametrize("twice,fail,error", [(True, False, ValueError), (False, True, RuntimeError)])
def test_hook_is_removed_on_wrong_call_count_or_forward_failure(twice, fail, error):
    model, batch = FakeModel(twice=twice, fail=fail), fake_batch()
    with pytest.raises(error):
        cache.forward_with_decoded(model, cache.model_inputs(batch), cache._autocast(torch.device("cpu")))
    assert not model.planner.xy_head._forward_pre_hooks


def test_exact_output_parity_rejects_any_tensor_change():
    model, batch = FakeModel(), fake_batch()
    with torch.inference_mode():
        output = model(**cache.model_inputs(batch))
    assert cache.exact_output_equal(output, copy.deepcopy(output))
    changed = copy.deepcopy(output)
    changed["plan_abs"][0, 0, 0] += 1e-3
    with pytest.raises(ValueError, match="plan_abs"):
        cache.exact_output_equal(output, changed)


def test_candidate_order_costs_and_strict_tie_move():
    gt = torch.zeros(3, 6, 2)
    plan = torch.zeros_like(gt)
    plan[1, :, 0] = 1.
    gt[2, :, 0] = 1.
    candidates, costs, labels = cache.selector_targets(plan, gt)
    assert torch.equal(candidates[:, cache.ZERO], torch.zeros_like(plan))
    assert torch.equal(candidates[:, cache.MOVE], plan)
    assert labels.tolist() == [cache.MOVE, cache.ZERO, cache.MOVE]
    assert torch.equal(costs[:, 0], cache.weighted_d3(torch.zeros_like(gt), gt))


def test_gt_descriptive_buckets_are_not_features():
    gt = torch.zeros(3, 6, 2)
    gt[1, -1, 0] = .21
    state = torch.zeros(3, 6)
    state[2, 0] = .2
    valid = torch.ones(3, 6, dtype=torch.bool)
    assert cache.diagnostic_bucket(gt, state, valid) == ["steady", "depart", "nonstop"]


def test_cache_split_one_hooked_forward_per_row_and_first_batch_parity():
    model, batch = FakeModel(), fake_batch()
    state_sha = cache.tensor_state_sha256(model.state_dict())
    payload, ids, buckets = cache.cache_split(model, [batch], torch.device("cpu"), state_sha=state_sha)
    assert payload["features"].shape == (2, 790)
    assert payload["candidate_plans"].shape == (2, 2, 6, 2)
    assert ids[0] == {"row": 10, "scenario": "scene0", "session": "session0", "frame": 30}
    assert buckets == ["steady", "steady"]
    assert cache.tensor_state_sha256(model.state_dict()) == state_sha


def test_cache_split_rejects_any_invalid_gt_point():
    model, batch = FakeModel(), fake_batch()
    batch["plan_valid"][0, 5] = False
    with pytest.raises(ValueError, match="valid"):
        cache.cache_split(model, [batch], torch.device("cpu"),
                          state_sha=cache.tensor_state_sha256(model.state_dict()))


def test_tune_mismatch_reports_plan_and_gt_separately_without_relaxing(capsys):
    batch = fake_batch(1)
    plan = torch.ones(1, 6, 2)
    gt = torch.zeros(1, 6, 2)
    reference = [{"row": 10, "scenario": "scene0", "session": "session0", "frame": 30,
                  "pred_abs_xy": torch.zeros(6, 2).tolist(), "gt_abs_xy": gt[0].tolist(),
                  "d3": 0.}]
    with pytest.raises(ValueError, match="bitwise"):
        cache.compare_tune_batch(reference, 0, batch, plan, gt)
    event = json.loads(capsys.readouterr().out)
    assert event["event"] == "tune_reference_bitwise_mismatch"
    assert not event["plan"]["bitwise_equal"] and event["plan"]["max_abs"] == 1.
    assert event["ground_truth"]["bitwise_equal"]


def test_main_seeds_like_evaluator_before_device_and_model_creation():
    source = Path(cache.__file__).read_text()
    seed = source.index("torch.manual_seed(args.base_seed)")
    device = source.index("device = torch.device(args.device)", seed)
    model = source.index("model, training_manifest = load_frozen_model", device)
    assert seed < device < model


def test_runtime_receipt_records_exact_interpreter_torch_and_cuda_runtime():
    receipt = cache.runtime_receipt(torch.device("cpu"))
    assert receipt["python_executable"]
    assert receipt["torch_version"] == str(torch.__version__)
    assert receipt["torch_cuda_runtime"] == torch.version.cuda
    assert receipt["device"] == "cpu" and not receipt["cuda_initialized"]


def test_loader_reuses_canonical_evaluator_and_preserves_requires_grad(tmp_path, monkeypatch):
    checkpoint, sidecar = tmp_path / "last.pth", tmp_path / "manifest.json"
    checkpoint.write_bytes(b"checkpoint")
    sidecar.write_text("{}")
    checkpoint_sha, sidecar_sha = "c" * 64, "d" * 64
    model = FakeModel()
    model.audit_load_metadata = {"checkpoint_sha256": checkpoint_sha}
    constructed = []
    monkeypatch.setattr(cache, "file_sha", lambda path: checkpoint_sha
                        if Path(path) == checkpoint else sidecar_sha)
    monkeypatch.setattr(cache, "read_pinned_json", lambda path, expected: {})
    monkeypatch.setattr(cache.torch, "load", lambda *args, **kwargs: {"model": {}})
    monkeypatch.setattr(cache, "validate_checkpoint_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr(cache, "construct_model",
                        lambda args: constructed.append(args) or model)
    loaded, _ = cache.load_frozen_model(checkpoint, checkpoint_sha, sidecar, sidecar_sha,
                                        0, torch.device("cpu"))
    assert loaded is model and constructed[0].checkpoint == str(checkpoint)
    assert all(parameter.requires_grad for parameter in loaded.parameters())
    source = Path(cache.__file__).read_text()
    assert ".requires_grad_(False)" not in source
    assert "planning_model_inputs(batch, time_input=\"nominal\")" in source


@pytest.mark.parametrize("mutation", ["step", "seed", "phase", "config", "git", "data", "sidecar"])
def test_checkpoint_contract_fails_closed(mutation):
    manifest = training_manifest()
    payload = {"step": 6000, "manifest": manifest, "model": {"x": torch.ones(1)}}
    sidecar = copy.deepcopy(manifest) | {"status": "completed", "step": 6000}
    if mutation == "step": payload["step"] = 5999
    if mutation == "seed": manifest["arguments"]["seed"] = 1
    if mutation == "phase": manifest["arguments"]["phase"] = "pretrain"
    if mutation == "config": manifest["model_config"]["goal_on"] = False
    if mutation == "git": manifest["git_sha"] = "0" * 40
    if mutation == "data": manifest["data_counts"]["train"] -= 1
    if mutation == "sidecar": sidecar["step"] = 5999
    with pytest.raises(ValueError):
        cache.validate_checkpoint_payload(payload, sidecar, base_seed=0,
            checkpoint_sha=cache.P4_CHECKPOINT_SHA256[0],
            run_manifest_sha=cache.P4_RUN_MANIFEST_SHA256[0])


def test_checkpoint_contract_accepts_embedded_running_completed_sidecar():
    manifest = training_manifest()
    payload = {"step": 6000, "manifest": manifest, "model": {"x": torch.ones(1)}}
    sidecar = copy.deepcopy(manifest) | {"status": "completed", "step": 6000}
    assert cache.validate_checkpoint_payload(payload, sidecar, base_seed=0,
        checkpoint_sha=cache.P4_CHECKPOINT_SHA256[0],
        run_manifest_sha=cache.P4_RUN_MANIFEST_SHA256[0]) is manifest


def test_source_manifest_requires_exact_22_files_and_separates_git(tmp_path, monkeypatch):
    files = {}
    for name in cache.RUNTIME_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        files[name] = cache.file_sha(path)
    contract = tmp_path / "models/motiondrive_v2_input_contract.py"
    contract.parent.mkdir(parents=True, exist_ok=True)
    if not contract.exists():
        contract.write_text("input contract")
    monkeypatch.setattr(cache, "RUNTIME_SHA256", files)
    monkeypatch.setattr(cache, "RUNTIME_FILES", frozenset(files))
    monkeypatch.setattr(cache, "INPUT_CONTRACT_SHA256", cache.file_sha(contract))
    manifest = {"source_git_sha": "9" * 40, "runtime_origin_git_sha": cache.P4_GIT_SHA,
                "files": files}
    assert cache.validate_source_manifest(manifest, tmp_path) == files
    bad = copy.deepcopy(manifest)
    bad["files"].pop(next(iter(bad["files"])))
    with pytest.raises(ValueError, match="22-file"):
        cache.validate_source_manifest(bad, tmp_path)
    bad_hash = copy.deepcopy(manifest)
    bad_hash["files"][next(iter(bad_hash["files"]))] = "0" * 64
    with pytest.raises(ValueError, match="22-file"):
        cache.validate_source_manifest(bad_hash, tmp_path)


def test_arguments_fix_train8_tune4_and_final_is_unavailable():
    common = ["--checkpoint", "c", "--expected-checkpoint-sha256", "a" * 64,
              "--run-manifest", "m", "--expected-run-manifest-sha256", "b" * 64,
              "--base-seed", "0", "--data-root", "d", "--split-manifest", "s",
              "--supervision-root", "u", "--source-manifest", "x",
              "--expected-source-manifest-sha256", "c" * 64, "--out", "o"]
    assert cache.arguments(common + ["--split", "train"]).batch == 8
    tune = common + ["--split", "tune", "--tune-report", "r",
                     "--expected-tune-report-sha256", "d" * 64]
    assert cache.arguments(tune).batch == 4
    assert cache.arguments(tune + ["--pilot-samples", "8"]).pilot_samples == 8
    with pytest.raises(SystemExit):
        cache.arguments(common + ["--split", "final_val"])
    with pytest.raises(ValueError):
        cache.arguments(tune + ["--batch", "8"])
    with pytest.raises(SystemExit):
        cache.arguments(tune + ["--pilot-samples", "7"])


def test_atomic_writer_refuses_overwrite(tmp_path):
    path = tmp_path / "report.json"
    cache.atomic_new_json(path, {"ok": True})
    with pytest.raises(ValueError, match="overwrite"):
        cache.atomic_new_json(path, {"ok": False})
    assert json.loads(path.read_text()) == {"ok": True}


def test_no_cuda_used_by_cpu_contract_tests():
    assert not torch.cuda.is_initialized()
