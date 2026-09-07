"""CPU-only inference packaging tests; never exports a real training checkpoint."""
import dataclasses
from pathlib import Path
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
