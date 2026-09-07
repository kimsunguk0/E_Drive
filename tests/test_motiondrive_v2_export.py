"""CPU-only inference packaging tests; never exports a real training checkpoint."""
import dataclasses
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_motiondrive_v2 import construct_model
from export_motiondrive_v2_inference import (export_checkpoint, file_sha256,
                                            prepare_bundle, validate_complete_config)
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from models.motiondrive_v2_input_contract import input_contract
from motiondrive_v2_training import time_input_policy
import export_motiondrive_v2_inference as exporter


@pytest.fixture(autouse=True)
def small_cpu_thread_count():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def tiny_config(**extra):
    values = dict(channels=16, backbone_arch="resnet34", grid_size=(6, 4),
                  x_range=(1, 9), y_range=(-2, 2), heights=(0., 1.),
                  motion_grid=(3, 4), correlation_channels=8,
                  scene_attention_channels=8, scene_chunk_size=7,
                  planner_heads=4, planner_layers=1)
    values.update(extra)
    return MotionDriveV2Config(**values)


def tiny_inputs():
    generator = torch.Generator().manual_seed(19)
    matrix = torch.tensor([[48., 20., 0., 0.], [32., 0., -20., 0.],
                           [1., 0., 0., 0.], [0., 0., 0., 1.]])
    return dict(images=torch.randn(1, 6, 3, 64, 96, generator=generator),
                history_images=torch.randn(1, 4, 3, 32, 48, generator=generator),
                lidar2img=matrix[None, None].repeat(1, 6, 1, 1),
                history_transforms=torch.eye(4)[None, None].repeat(1, 4, 1, 1),
                time_offsets=torch.tensor([[.1, .2, .5, 1.]]), goal_xy=torch.tensor([[15., 3.]]))


def payload(config, state=None):
    return {"model": {} if state is None else state,
            "step": 250, "epoch": 2, "optimizer": {"exp_avg": torch.ones(300000)},
            "rng": {"numpy": np.random.get_state(), "torch": torch.get_rng_state()},
            "manifest": {"model_config": dataclasses.asdict(config),
                         "split_sha256": "b" * 64, "git_sha": "a" * 40,
                         "supervision_manifest_sha256": "c" * 64,
                         "train_rows_sha256": "d" * 64,
                         "arguments": {"split_manifest": "grouped_split_rawtime.json", "seed": 0},
                         "load_report": {"common_checkpoint_sha256": "e" * 64},
                         "torch": torch.__version__}}


def prepare(data):
    return prepare_bundle(data, source_path="/trusted/checkpoint.pth",
                          source_sha256="f" * 64, source_bytes=123)


@pytest.mark.parametrize("mode,goal_on,state_on,scale", [
    ("low_feature", True, True, (10., 5.)),
    ("high_feature", False, False, (10., 5.)),
    ("legacy", False, True, (1., 1.)),
])
def test_export_reload_preserves_complete_config_and_bitwise_outputs(tmp_path, mode, goal_on, state_on, scale):
    cfg = tiny_config(motion_input_mode=mode, goal_on=goal_on, state_on=state_on,
                      plan_output_scale=scale)
    model = MotionDriveV2(cfg).eval()
    source, target = tmp_path / "training.pth", tmp_path / "inference.pth"
    torch.save(payload(cfg, model.state_dict()), source)
    source_hash = file_sha256(source)
    rng_before = torch.get_rng_state().clone()
    result = export_checkpoint(source, target)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert file_sha256(source) == source_hash
    # No unsafe NumPy/Python RNG object survives in the output bundle.
    bundle = torch.load(target, map_location="cpu", weights_only=True)
    assert "input_contract" not in bundle and "deployment_provenance" not in bundle
    assert "optimizer" not in bundle and "rng" not in bundle
    assert result["output_bytes"] < result["source_checkpoint"]["bytes"]
    assert result["source_checkpoint"]["sha256"] == source_hash
    assert bundle["source_checkpoint"]["step"] == bundle["step"] == 250
    assert bundle["epoch"] == 2
    for key in ("split_sha256", "supervision_manifest_sha256", "train_rows_sha256", "load_report"):
        assert bundle["manifest"][key] == payload(cfg)["manifest"][key]
    for name, tensor in bundle["model"].items():
        assert tensor.device.type == "cpu" and not tensor.requires_grad
        assert torch.equal(tensor, model.state_dict()[name])
        assert tensor.dtype == model.state_dict()[name].dtype
    restored = construct_model(SimpleNamespace(checkpoint=str(target), config_json=None,
                                               goal_on=None, state_on=None, device="cpu"))
    assert restored.config.to_dict() == cfg.to_dict()
    assert restored.audit_load_metadata["explicit_overrides"] == {}
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        before, after = model(**tiny_inputs()), restored(**tiny_inputs())
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)


@pytest.mark.parametrize("missing", ["model_config", "motion_input_mode", "plan_output_scale",
                                      "goal_on", "state_on", "scene_chunk_size"])
def test_no_missing_config_field_is_filled_from_defaults(missing):
    data = payload(tiny_config())
    if missing == "model_config":
        del data["manifest"][missing]
    else:
        del data["manifest"]["model_config"][missing]
    with pytest.raises(ValueError, match="model_config"):
        prepare(data)


@pytest.mark.parametrize("key,value", [("goal_on", 0), ("state_on", "false"),
                                        ("channels", True), ("planner_heads", 0),
                                        ("motion_input_mode", "wrong"),
                                        ("plan_output_scale", (0., 5.)),
                                        ("backbone_arch", "unknown"),
                                        ("x_range", (10., 0.)),
                                        ("heights", (float("nan"),)),
                                        ("motion_grid", (3., 4))])
def test_invalid_config_values_rejected(key, value):
    config = tiny_config().to_dict()
    config[key] = value
    with pytest.raises(ValueError):
        validate_complete_config(config)


def test_unknown_config_fields_rejected():
    config = tiny_config().to_dict()
    config["silently_ignored_setting"] = True
    with pytest.raises(ValueError, match="unknown"):
        validate_complete_config(config)


@pytest.mark.parametrize("kind", ["missing_step", "negative_step", "boolean_step", "missing_split", "bad_split"])
def test_lineage_is_required(kind):
    data = payload(tiny_config())
    if kind == "missing_step":
        del data["step"]
    elif kind == "negative_step":
        data["step"] = -1
    elif kind == "boolean_step":
        data["step"] = True
    elif kind == "missing_split":
        del data["manifest"]["split_sha256"]
    else:
        data["manifest"]["split_sha256"] = "unknown"
    with pytest.raises(ValueError, match="step|split_sha256"):
        prepare(data)


def test_incompatible_state_fails_strictly_before_output(tmp_path):
    source, target = tmp_path / "source.pth", tmp_path / "output.pth"
    torch.save(payload(tiny_config(), {"not_a_model_key": torch.zeros(1)}), source)
    with pytest.raises(RuntimeError, match="state_dict"):
        export_checkpoint(source, target)
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_existing_output_and_input_overwrite_are_rejected_before_loading(tmp_path):
    source, target = tmp_path / "source.pth", tmp_path / "output.pth"
    source.write_bytes(b"not even a pickle")
    target.write_bytes(b"keep existing output")
    with pytest.raises(FileExistsError):
        export_checkpoint(source, target)
    assert target.read_bytes() == b"keep existing output"
    with pytest.raises(ValueError, match="overwrite"):
        export_checkpoint(source, source)
    assert source.read_bytes() == b"not even a pickle"


def test_symlink_alias_and_dangling_target_are_rejected(tmp_path):
    source, alias = tmp_path / "source.pth", tmp_path / "alias.pth"
    source.write_bytes(b"keep")
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="overwrite"):
        export_checkpoint(source, alias)
    dangling = tmp_path / "dangling.pth"
    dangling.symlink_to(tmp_path / "missing.pth")
    with pytest.raises(FileExistsError):
        export_checkpoint(source, dangling)
    assert alias.is_symlink() and dangling.is_symlink()


def test_publish_race_never_overwrites_someone_elses_target(tmp_path, monkeypatch):
    import export_motiondrive_v2_inference as exporter
    source, target = tmp_path / "source.pth", tmp_path / "output.pth"
    torch.save({"placeholder": True}, source)
    # Test filesystem publication independently of costly model validation.
    monkeypatch.setattr(exporter, "prepare_bundle", lambda *a, **kw: {"model": {"x": torch.ones(1)}})
    real_link = exporter.os.link
    def race_link(src, dst):
        target.write_bytes(b"owned by concurrent process")
        return real_link(src, dst)
    monkeypatch.setattr(exporter.os, "link", race_link)
    with pytest.raises(FileExistsError):
        exporter.export_checkpoint(source, target)
    assert target.read_bytes() == b"owned by concurrent process"
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.fixture
def deployment_fixture(tmp_path, monkeypatch):
    """Only synthetic files; production immutable artifact SHA pins are not changed."""
    root = tmp_path / "mock_repository"
    run, supervision = root / "work_dirs/mock_c1t1", root / "data/mock_geometry_v2"
    run.mkdir(parents=True)
    supervision.mkdir(parents=True)
    split, initializer = root / "mock_split.json", root / "mock_initializer.pth"
    split.write_text('{"splits":{"train":["synthetic_train"]}}')
    initializer.write_bytes(b"synthetic initializer; never unpickled")
    canonical = supervision / "calibration.npz"
    canonical.write_bytes(b"synthetic canonical; checked as bytes only")
    canonical_sha, split_sha = file_sha256(canonical), file_sha256(split)
    supervision_manifest = supervision / "supervision_manifest.json"
    supervision_manifest.write_text(json.dumps({
        "schema_version": 2, "geometry_edition": "cache_meta_rear_wide_v2",
        "canonical_calibration_sha256": canonical_sha,
        "split_manifest_sha256": split_sha,
        "calibration_sha256": "b" * 64}))  # PKL hash is NOT the canonical hash.
    sup_sha = file_sha256(supervision_manifest)
    monkeypatch.setattr(exporter, "C1_CANONICAL_SHA256", canonical_sha)
    monkeypatch.setattr(exporter, "C1_SUPERVISION_SHA256", sup_sha)
    monkeypatch.setattr(exporter, "RAWTIME_SPLIT_SHA256", split_sha)
    checkpoint = payload(tiny_config())
    checkpoint["manifest"].update({
        "arguments": {"data_root": str(root), "run_dir": "work_dirs/mock_c1t1",
                      "supervision_root": "data/mock_geometry_v2", "split_manifest": "mock_split.json",
                      "init": "mock_initializer.pth", "resume": None, "pretrained": None,
                      "time_input": "nominal", "steps": 3000, "seed": 0},
        "time_input": "nominal", "time_input_policy": time_input_policy("nominal"),
        "split_sha256": split_sha, "supervision_manifest_sha256": sup_sha,
        "load_report": {"common_checkpoint_sha256": file_sha256(initializer)},
        "initial_model_state_sha256": "a" * 64, "eval_rows_sha256": "e" * 64,
        "data_counts": {"train": 8, "eval": 8}, "bn_training": {"policy": "fixed"},
        "loss_weights": {"plan": 1.}, "pid": 12345, "status": "running",
        "pretrained_sha256": None,
    })
    source = run / "best.pth"
    source.write_bytes(b"synthetic checkpoint; validation-only fixture")
    sidecar = run / "manifest.json"

    def sync_sidecar():
        final = copy.deepcopy(checkpoint["manifest"])
        final.update(status="completed", step=3000)
        sidecar.write_text(json.dumps(final))

    sync_sidecar()
    return SimpleNamespace(root=root, run=run, supervision=supervision, split=split,
                           initializer=initializer, canonical=canonical, sidecar=sidecar,
                           supervision_manifest=supervision_manifest, source=source,
                           checkpoint=checkpoint, sync_sidecar=sync_sidecar)


def verify_fixture(fixture, **overrides):
    kwargs = dict(source_path=fixture.source, source_sha256=file_sha256(fixture.source),
                  expected_checkpoint_sha256=file_sha256(fixture.source),
                  deployment_contract=exporter.DEPLOYMENT_CONTRACT)
    kwargs.update(overrides)
    return exporter.verify_deployment_lineage(fixture.checkpoint, **kwargs)


def test_strict_lineage_is_explicit_and_checks_original_run_files(deployment_fixture):
    fixture = deployment_fixture
    report = verify_fixture(fixture)
    assert report["status"] == "input_contract_lineage_verified"
    assert report["checkpoint_step"] == 250 and report["completed_run_step"] == 3000
    assert report["selected_checkpoint_sha256"] == file_sha256(fixture.source)
    assert report["geometry_edition"] == "geometry_v2" and report["time_input"] == "nominal"
    assert report["supervision_manifest_sha256"] == file_sha256(fixture.supervision_manifest)
    assert report["canonical_calibration_sha256"] == file_sha256(fixture.canonical)
    assert report["evidence"]["canonical_calibration"]["sha256"] == file_sha256(fixture.canonical)
    assert report["accuracy_latency_compliance_certified"] is False
    assert report["os_process_exit_verified"] is False
    assert fixture.checkpoint["manifest"]["status"] == "running"
    exporter.reverify_deployment_evidence(report)


@pytest.mark.parametrize("mode,expected", [(None, "a" * 64), ("wrong", "a" * 64),
                                         ("geometry-v2-nominal", None),
                                         ("geometry-v2-nominal", "not-sha"),
                                         ("geometry-v2-nominal", "a" * 64)])
def test_deployment_requires_explicit_mode_and_selected_sha(deployment_fixture, mode, expected):
    with pytest.raises(ValueError, match="contract|SHA256"):
        verify_fixture(deployment_fixture, deployment_contract=mode, expected_checkpoint_sha256=expected)


@pytest.mark.parametrize("which", ["args_raw", "manifest_raw", "missing", "policy_raw", "policy_mutated"])
def test_raw_or_inconsistent_time_is_never_relabelled(deployment_fixture, which):
    manifest = deployment_fixture.checkpoint["manifest"]
    if which == "args_raw":
        manifest["arguments"]["time_input"] = "raw"
    elif which == "manifest_raw":
        manifest["time_input"] = "raw"
    elif which == "missing":
        del manifest["time_input"]
    elif which == "policy_raw":
        manifest["time_input_policy"] = time_input_policy("raw")
    else:
        manifest["time_input_policy"]["nominal_history_seconds"] = [0., .2, .5, 1.]
    deployment_fixture.sync_sidecar()
    with pytest.raises(ValueError, match="nominal"):
        verify_fixture(deployment_fixture)


@pytest.mark.parametrize("kind", ["old_supervision", "wrong_split", "history_count", "step_zero",
                                  "relative_root", "source_relocated"])
def test_old_geometry_or_unbound_checkpoint_fails_closed(deployment_fixture, kind):
    fixture, manifest = deployment_fixture, deployment_fixture.checkpoint["manifest"]
    if kind == "old_supervision":
        manifest["supervision_manifest_sha256"] = "0" * 64
    elif kind == "wrong_split":
        manifest["split_sha256"] = "0" * 64
    elif kind == "history_count":
        manifest["model_config"]["n_history"] = 3
    elif kind == "step_zero":
        fixture.checkpoint["step"] = 0
    elif kind == "relative_root":
        manifest["arguments"]["data_root"] = "."
    else:
        moved = fixture.root / "best.pth"
        moved.write_bytes(fixture.source.read_bytes())
        fixture.source = moved
    fixture.sync_sidecar()
    with pytest.raises(ValueError):
        verify_fixture(fixture)


@pytest.mark.parametrize("field", ["status", "step", "arguments", "model_config", "load_report",
                                   "initial_model_state_sha256", "pid", "train_rows_sha256"])
def test_external_run_manifest_must_match_checkpoint(deployment_fixture, field):
    fixture = deployment_fixture
    final = json.loads(fixture.sidecar.read_text())
    if field == "status":
        final[field] = "running"
    elif field == "step":
        final[field] = 2999
    else:
        del final[field]
    fixture.sidecar.write_text(json.dumps(final))
    with pytest.raises(ValueError, match="completed|lineage mismatch"):
        verify_fixture(fixture)


@pytest.mark.parametrize("artifact", ["source", "supervision_manifest", "canonical", "split", "initializer"])
def test_real_bytes_must_match_the_recorded_sha(deployment_fixture, artifact):
    fixture = deployment_fixture
    original = file_sha256(fixture.source)
    path = getattr(fixture, artifact)
    path.write_bytes(path.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="SHA256"):
        verify_fixture(fixture, source_sha256=original, expected_checkpoint_sha256=original)


@pytest.mark.parametrize("field,value", [("schema_version", 1), ("geometry_edition", "old"),
                                        ("canonical_calibration_sha256", "0" * 64),
                                        ("split_manifest_sha256", "0" * 64)])
def test_geometry_declaration_must_also_match_even_with_valid_file_hash(
        deployment_fixture, monkeypatch, field, value):
    fixture = deployment_fixture
    contract = json.loads(fixture.supervision_manifest.read_text())
    contract[field] = value
    fixture.supervision_manifest.write_text(json.dumps(contract))
    changed_sha = file_sha256(fixture.supervision_manifest)
    monkeypatch.setattr(exporter, "C1_SUPERVISION_SHA256", changed_sha)
    fixture.checkpoint["manifest"]["supervision_manifest_sha256"] = changed_sha
    fixture.sync_sidecar()
    with pytest.raises(ValueError, match="geometry/split declaration"):
        verify_fixture(fixture)


def test_initializer_is_verified_and_no_fallback_is_invented(deployment_fixture):
    fixture = deployment_fixture
    del fixture.checkpoint["manifest"]["load_report"]["common_checkpoint_sha256"]
    fixture.sync_sidecar()
    with pytest.raises(ValueError, match="initialization lineage"):
        verify_fixture(fixture)


def test_pretrained_lineage_checked_when_present(deployment_fixture):
    fixture = deployment_fixture
    pretrained = fixture.root / "pretrained.pth"
    pretrained.write_bytes(b"mock pretraining")
    fixture.checkpoint["manifest"]["arguments"]["pretrained"] = "pretrained.pth"
    fixture.checkpoint["manifest"]["pretrained_sha256"] = file_sha256(pretrained)
    fixture.sync_sidecar()
    assert "pretrained_backbone" in verify_fixture(fixture)["evidence"]
    pretrained.write_bytes(b"changed")
    with pytest.raises(ValueError, match="pretrained_backbone"):
        verify_fixture(fixture)


def test_existing_conflicting_contract_is_not_overwritten(deployment_fixture):
    fixture = deployment_fixture
    fixture.checkpoint["input_contract"] = {**input_contract(), "time_input": "raw"}
    with pytest.raises(ValueError, match="conflicts"):
        verify_fixture(fixture)


def test_publish_time_recheck_detects_changed_sidecar(deployment_fixture):
    fixture = deployment_fixture
    report = verify_fixture(fixture)
    fixture.sidecar.write_bytes(fixture.sidecar.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed before publication"):
        exporter.reverify_deployment_evidence(report)


def test_selected_sha_does_not_silently_enable_deployment_mode():
    with pytest.raises(ValueError, match="explicit deployment contract"):
        prepare_bundle(payload(tiny_config()), source_path="mock", source_sha256="f" * 64,
                       source_bytes=1, expected_checkpoint_sha256="f" * 64)


def test_strict_synthetic_bundle_has_safe_contract_and_loads_with_generic_loader(deployment_fixture):
    fixture = deployment_fixture
    model = MotionDriveV2(tiny_config()).eval()
    fixture.checkpoint["model"] = model.state_dict()
    torch.save(fixture.checkpoint, fixture.source)
    before = file_sha256(fixture.source)
    target = fixture.root / "mock_inference.pth"
    result = export_checkpoint(fixture.source, target, deployment_contract=exporter.DEPLOYMENT_CONTRACT,
                               expected_checkpoint_sha256=before)
    saved = torch.load(target, map_location="cpu", weights_only=True)
    assert saved["input_contract"] == input_contract() == result["input_contract"]
    assert saved["deployment_provenance"] == result["deployment_provenance"]
    assert saved["deployment_provenance"]["selected_checkpoint_sha256"] == before
    assert file_sha256(fixture.source) == before
    assert "optimizer" not in saved and "rng" not in saved
    restored = construct_model(SimpleNamespace(checkpoint=str(target), config_json=None,
                                               goal_on=None, state_on=None, device="cpu"))
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())


def test_strict_nonfinite_tensor_is_rejected_before_model_construction(deployment_fixture):
    fixture = deployment_fixture
    fixture.checkpoint["model"] = {"bad": torch.tensor([float("nan")])}
    with pytest.raises(ValueError, match="nonfinite"):
        prepare_bundle(fixture.checkpoint, source_path=fixture.source,
                       source_sha256=file_sha256(fixture.source), source_bytes=fixture.source.stat().st_size,
                       deployment_contract=exporter.DEPLOYMENT_CONTRACT,
                       expected_checkpoint_sha256=file_sha256(fixture.source))


@pytest.mark.parametrize("package_import", [False, True])
def test_export_import_has_no_opencv_dependency(package_import):
    code = """
import importlib.abc, sys
class NoCV2(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cv2' or fullname.startswith('cv2.'):
            raise RuntimeError('OpenCV must not be imported during CPU export')
sys.meta_path.insert(0, NoCV2())
if sys.argv[1] == 'True':
    import scripts.export_motiondrive_v2_inference as exporter
else:
    sys.path.insert(0, 'scripts')
    import export_motiondrive_v2_inference as exporter
assert 'models.motiondrive_v2_inputs' not in sys.modules
assert 'cv2' not in sys.modules
from scripts.motiondrive_v2_training import time_input_policy
try:
    exporter.verify_deployment_lineage(
        {'manifest': {'arguments': {'time_input': 'nominal'}, 'time_input': 'nominal',
                      'time_input_policy': time_input_policy('nominal')}},
        source_path='unused', source_sha256='a' * 64,
        expected_checkpoint_sha256='a' * 64, deployment_contract='geometry-v2-nominal')
except ValueError as error:
    assert 'C1 supervision' in str(error)
else:
    raise AssertionError('Invalid lineage must not pass')
"""
    result = subprocess.run([sys.executable, "-c", code, str(package_import)], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_strict_publish_aborts_if_evidence_changes_during_serialization(deployment_fixture, monkeypatch):
    fixture = deployment_fixture
    torch.save({"synthetic": True}, fixture.source)
    proof = verify_fixture(fixture)
    monkeypatch.setattr(exporter, "prepare_bundle", lambda *a, **kw: {
        "model": {"x": torch.zeros(1)}, "deployment_provenance": proof})
    original_save = torch.save
    def mutation_during_save(*args, **kwargs):
        original_save(*args, **kwargs)
        fixture.sidecar.write_bytes(fixture.sidecar.read_bytes() + b"\n")
    monkeypatch.setattr(torch, "save", mutation_during_save)
    target = fixture.root / "should_not_be_published.pth"
    with pytest.raises(ValueError, match="changed before publication"):
        export_checkpoint(fixture.source, target, deployment_contract=exporter.DEPLOYMENT_CONTRACT,
                           expected_checkpoint_sha256=file_sha256(fixture.source))
    assert not target.exists()
    assert not list(fixture.root.glob(".*.tmp"))
