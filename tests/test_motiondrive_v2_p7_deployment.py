import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts import smoke_motiondrive_v2_p7_deployment as p7


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_gpu_uuid_allows_prefix_form_but_rejects_other_card():
    expected = p7.EXPECTED_GPU_UUID
    assert p7.canonical_gpu_uuid(expected) == p7.canonical_gpu_uuid(expected[4:])
    assert p7.require_expected_gpu({"gpu_uuid": expected[4:]}, expected)["expected_gpu_uuid"] == expected
    with pytest.raises(ValueError, match="Observed device"):
        p7.require_expected_gpu(
            {"gpu_uuid": "GPU-11111111-1111-1111-1111-111111111111"}, expected)
    with pytest.raises(ValueError, match="CLI expected GPU"):
        p7.require_expected_gpu({"gpu_uuid": expected},
                                "GPU-11111111-1111-1111-1111-111111111111")


def test_validation_source_manifest_requires_exact_runtime_closure(monkeypatch, tmp_path):
    root = tmp_path / "source"
    files = {}
    for index, name in enumerate(sorted(p7.VALIDATION_SOURCE_FILES)):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source-{index}".encode())
        files[name] = sha(path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "git_sha": "a" * 40,
                                    "file_sha256": files}))
    monkeypatch.setattr(p7, "ROOT", root)
    parsed, before = p7.validate_validation_source_manifest(manifest, sha(manifest))
    assert parsed["file_sha256"] == before == files
    bad = json.loads(manifest.read_text())
    bad["file_sha256"].pop(next(iter(bad["file_sha256"])))
    manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="exact runtime set"):
        p7.validate_validation_source_manifest(manifest, sha(manifest))


class FakeBranch(nn.Module):
    COSINE_SCALE = 8.0

    def __init__(self):
        super().__init__()
        self.query_image = nn.Identity()
        self.key_image = nn.Identity()
        self.value_image = nn.Identity()
        self.position = nn.Identity()
        self.output = nn.Linear(128, 128, bias=False)
        nn.init.zeros_(self.output.weight)
        self.register_buffer("sigma_m", torch.tensor([10.0, 32.0 / 3.0]))

    def forward(self, scene, goal, visible):
        self.query_image(scene)
        self.key_image(scene[:, :192])
        self.value_image(scene[:, :192])
        self.position(goal)
        self.position(goal)
        return scene + self.output(scene)


class FakeScene(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_cell_goal_residual = FakeBranch()


class FakeModel(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.scene_encoder = FakeScene()
        self.config = SimpleNamespace(cross_cell_goal_mode=mode, goal_on=True, state_on=True,
                                      backbone_arch="resnet50", motion_input_mode="low_feature",
                                      plan_output_scale=(10.0, 5.0))
        self.anchor = nn.Parameter(torch.ones(1))


def test_assemble_p7_binds_direct_p0_sidecar_and_paired_state(monkeypatch):
    args = SimpleNamespace(
        base_seed=0, arm="control_zero_slot", init="p0.pth",
        expected_init_sha256=p7.P0[0]["file_sha256"], run_manifest="p0.json",
        expected_run_manifest_sha256=p7.P0[0]["manifest_sha256"],
        training_source_manifest="source.json",
        expected_training_source_manifest_sha256="a" * 64,
        expected_initial_model_state_sha256="e" * 64)
    monkeypatch.setattr(p7, "validate_training_source_manifest", lambda path, digest: {
        "sha256": digest, "git_sha": "b" * 40,
        "file_sha256": {name: "c" * 64 for name in p7.TRAINING_SOURCE_FILES}})
    monkeypatch.setattr(p7, "validate_p0", lambda *values: (
        {"model": {}}, {"model_state_sha256": p7.P0[0]["model_state_sha256"],
                        "checkpoint_sha256": p7.P0[0]["file_sha256"],
                        "sidecar_sha256": p7.P0[0]["manifest_sha256"]}))
    monkeypatch.setattr(p7, "_build_initialized_model",
                        lambda payload, seed, mode: (FakeModel(mode), ["branch.key"], "e" * 64))
    model, receipt = p7.assemble_p7(args)
    assert model.config.cross_cell_goal_mode == "zero"
    assert receipt["p0"]["sidecar_sha256"] == p7.P0[0]["manifest_sha256"]
    assert receipt["initial_model_state_sha256"] == receipt["paired_other_mode_initial_model_state_sha256"]
    args.expected_run_manifest_sha256 = "f" * 64
    with pytest.raises(ValueError, match="file/sidecar"):
        p7.assemble_p7(args)


@pytest.mark.parametrize("arm", tuple(p7.ARM_TO_MODE))
def test_branch_probe_observes_fp32_modules_without_kernel_claim(monkeypatch, arm):
    model = FakeModel(p7.ARM_TO_MODE[arm])
    goal = torch.tensor([[7.0, -3.0]], dtype=torch.float32)
    inputs = {"goal_xy": goal}

    def fake_observed(model_arg, inputs_arg, contract, *, device, precision):
        query = torch.ones(1, 2, 3)
        key = torch.ones(1, 4, 3)
        value = torch.ones(1, 4, 5)
        score = torch.einsum("bqd,bkd->bqk", query, key)
        attention = p7.p7_residual.masked_softmax(
            score, torch.ones_like(score, dtype=torch.bool), dim=-1)
        torch.einsum("bqk,bkd->bqd", attention, value)
        scene = torch.ones(1, 3072, 128)
        visible = torch.ones(1, 3072, dtype=torch.bool)
        branch_goal = goal if arm == "goal_real_slot" else torch.zeros_like(goal)
        model_arg.scene_encoder.cross_cell_goal_residual(scene, branch_goal, visible)
        return torch.zeros(6, 2), {"image_encodings": 11,
                                  "backbone_input_shapes": p7.EXPECTED_ENCODINGS}

    monkeypatch.setattr(p7, "observed_forward", fake_observed)
    _, evidence = p7.branch_execution_probe(model, inputs, {}, arm=arm, device="cpu")
    assert evidence["module_call_count"] == 1
    assert {name: len(rows) for name, rows in evidence["component_module_calls"].items()} == {
        "query_image": 1, "key_image": 1, "value_image": 1, "position": 2, "output": 1}
    assert evidence["fp32_projection_dtype_observed"] is True
    assert evidence["fp32_contraction_dtype_observed"] is True
    assert evidence["fp32_score_softmax_dtype_observed"] is True
    assert evidence["instrumentation_removed_before_timing"] is True
    assert evidence["cuda_kernel_identity_observed"] is False
    assert evidence["call"]["goal_matches_arm_input"] is True


def test_branch_probe_restores_function_instrumentation_on_failure(monkeypatch):
    model = FakeModel("zero")
    original_einsum = torch.einsum
    original_softmax = p7.p7_residual.masked_softmax
    monkeypatch.setattr(
        p7, "observed_forward",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("probe failure")))
    with pytest.raises(RuntimeError, match="probe failure"):
        p7.branch_execution_probe(
            model, {"goal_xy": torch.zeros(1, 2)}, {},
            arm="control_zero_slot", device="cpu")
    assert torch.einsum is original_einsum
    assert p7.p7_residual.masked_softmax is original_softmax


@pytest.mark.parametrize("median,p95,passed", [(99.9, 99.99, True), (100.0, 99.0, False),
                                                (99.0, 100.0, False)])
def test_latency_gate_is_strict_for_both_statistics(median, p95, passed):
    assert p7.latency_gate({"median_ms": float(median), "p95_ms": float(p95)})["pass"] is passed


def test_cli_freezes_timing_and_requires_direct_p0_inputs():
    argv = []
    for name in ("init", "expected-init-sha256", "run-manifest",
                 "expected-run-manifest-sha256", "expected-initial-model-state-sha256",
                 "training-source-manifest", "expected-training-source-manifest-sha256",
                 "validation-source-manifest", "expected-validation-source-manifest-sha256",
                 "fixture-root", "reference", "calibration", "geometry-contract", "output",
                 "expected-physical-gpu-uuid"):
        argv += ["--" + name, "x"]
    argv += ["--arm", "goal_real_slot", "--base-seed", "1"]
    args = p7.arguments(argv)
    assert (args.warmup, args.repeats, args.device, args.precision) == (20, 50, "cuda:0", "bf16")
    assert args.arm == "goal_real_slot" and args.base_seed == 1


def test_main_preserves_failed_execution_as_new_json(monkeypatch, tmp_path):
    output = tmp_path / "failed.json"
    args = SimpleNamespace(output=str(output), expected_init_sha256="a" * 64,
                           expected_run_manifest_sha256="b" * 64,
                           expected_initial_model_state_sha256="c" * 64,
                           expected_training_source_manifest_sha256="d" * 64,
                           expected_validation_source_manifest_sha256="e" * 64)
    monkeypatch.setattr(p7, "arguments", lambda argv=None: args)
    monkeypatch.setattr(p7, "run", lambda value: (_ for _ in ()).throw(ValueError("expected failure")))
    assert p7.main([]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["error_type"] == "ValueError"
    assert report["timing_gate_pass"] is False and report["final_holdout_accessed"] is False
    with pytest.raises(FileExistsError):
        p7.main([])
