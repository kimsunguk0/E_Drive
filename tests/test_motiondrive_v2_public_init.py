"""CPU-only initializer tests using a synthetic 318-entry public backbone fixture.

The real model's loader/config/planner are used; the heavy R50 convolution trunk
is replaced in tests only. This is not evidence of real public weights or quality.
"""
import argparse
import copy
import hashlib
import json

import pytest
import torch
from torch import nn

import models.motiondrive_v2.model as model_module
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from scripts import initialize_motiondrive_v2_public as public_init
from scripts.motiondrive_v2_training import tensor_state_sha256
from scripts.train_motiondrive_v2 import initialization_configuration


class Tiny318EntryBackbone(nn.Module):
    def __init__(self, channels, arch):
        super().__init__()
        assert arch == "resnet50"
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 1, bias=False), nn.BatchNorm2d(channels))
        self.layer1 = nn.Module()
        for i in range(312):
            self.layer1.register_buffer(f"synthetic_{i:03d}", torch.zeros(1))
        self.layer2 = self.layer3 = self.layer4 = nn.Identity()
        self.lat2 = nn.Conv2d(channels, channels, 1)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(model_module, "ResNet34FPN128", Tiny318EntryBackbone)
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def public_files(tmp_path, monkeypatch):
    cfg = MotionDriveV2Config(backbone_arch="resnet50", goal_on=False, state_on=False,
                            motion_input_mode="low_feature", plan_output_scale=(10., 5.))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        fresh = MotionDriveV2(cfg).eval()
    public_state = {}
    for name, value in fresh.backbone_fpn.state_dict().items():
        if name.startswith("lat"):
            continue
        name = "conv1." + name[len("stem.0."):] if name.startswith("stem.0.") else name
        name = "bn1." + name[len("stem.1."):] if name.startswith("stem.1.") else name
        public_state["backbone." + name] = torch.full_like(value, 2)
    assert len(public_state) == 318
    # The real loader must refuse to inject a public FPN entry into compact FPN.
    public_state["backbone.lat2.weight"] = torch.full_like(fresh.backbone_fpn.lat2.weight, 999.)
    public_path, split_path = tmp_path / "public.pth", tmp_path / "split.json"
    torch.save({"state_dict": public_state}, public_path)
    split_path.write_text('{"fixture": "only hashed, no dataset parsing"}')
    monkeypatch.setattr(public_init, "PUBLIC_SHA256", public_init.sha256(public_path))
    monkeypatch.setattr(public_init, "SPLIT_SHA256", public_init.sha256(split_path))
    return fresh, public_path, split_path


def test_seed_reproducibility_full_config_and_zero_etri_updates(public_files):
    fresh, path, split = public_files
    before_rng = torch.random.get_rng_state().clone()
    a = public_init.prepare_public_payload(path, split)
    b = public_init.prepare_public_payload(path, split)
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert tensor_state_sha256(a["model"]) == tensor_state_sha256(b["model"])
    manifest = a["manifest"]
    assert manifest["seed"] == manifest["initialization_seed"] == 0
    assert manifest["etri_optimizer_steps"] == manifest["etri_optimizer_updates"] == manifest["etri_samples_seen"] == 0
    assert a["step"] == a["epoch"] == 0 and a["optimizer"] == {}
    assert manifest["pretrained_sha256"] == manifest["public_checkpoint_sha256"] == public_init.sha256(path)
    assert manifest["model_config"] == fresh.config.to_dict()
    assert manifest["model_config"]["plan_output_scale"] == (10., 5.)
    assert manifest["arguments"]["bn_policy"] == manifest["bn_training"]["policy"] == "fixed"
    assert manifest["load_report"]["backbone"]["injected"] == 318
    assert not manifest["labels_read"] and not manifest["image_data_read"]
    assert not torch.cuda.is_initialized()


def test_fresh_fpn_heads_preserved_public_backbone_loaded_without_output_reparameterization(public_files):
    fresh, path, split = public_files
    saved = public_init.prepare_public_payload(path, split)
    for name, value in saved["model"].items():
        if public_init.backbone_key(name):
            assert torch.equal(value, torch.full_like(value, 2)), name
        else:
            assert torch.equal(value, fresh.state_dict()[name]), name
    restored = MotionDriveV2(MotionDriveV2Config(**saved["manifest"]["model_config"])).eval()
    restored.load_state_dict(saved["model"], strict=True)
    # Planner fixed-feature predictions remain exactly the seed-zero constructor
    # output. Full image predictions SHOULD change after public backbone loading.
    generator = torch.Generator().manual_seed(31)
    features = [torch.randn(1, 3072, 128, generator=generator),
                torch.randn(1, 192, 128, generator=generator),
                torch.randn(1, 6, generator=generator), torch.randn(1, 4, 4, generator=generator)]
    with torch.no_grad():
        assert torch.equal(fresh.plan_from_features(*features), restored.plan_from_features(*features))


def test_existing_trainer_init_configuration_and_strict_state_contract(public_files):
    _, path, split = public_files
    saved = public_init.prepare_public_payload(path, split)
    # Existing --init inherits all complete config fields and only changes G/S.
    config = initialization_configuration(saved["manifest"], goal_on=0, state_on=0, explicit_arch="resnet50")
    restored = MotionDriveV2(MotionDriveV2Config(**config))
    restored.load_state_dict(saved["model"], strict=True)
    assert tensor_state_sha256(restored.state_dict()) == saved["manifest"]["initial_model_state_sha256"]
    assert saved["manifest"]["split_sha256"] == public_init.sha256(split)
    assert saved["manifest"]["compatible_usage"].endswith("NOT --resume")


@pytest.mark.parametrize("fault", ["missing", "shape", "wrong_public_sha", "wrong_split_sha"])
def test_incomplete_or_unpinned_public_source_rejected_before_output(public_files, tmp_path, monkeypatch, fault):
    _, path, split = public_files
    if fault in ("missing", "shape"):
        raw = torch.load(path, weights_only=True)
        key = "backbone.conv1.weight"
        if fault == "missing": raw["state_dict"].pop(key)
        else: raw["state_dict"][key] = raw["state_dict"][key][:1]
        torch.save(raw, path)
        monkeypatch.setattr(public_init, "PUBLIC_SHA256", public_init.sha256(path))
    elif fault == "wrong_public_sha": monkeypatch.setattr(public_init, "PUBLIC_SHA256", "0" * 64)
    else: monkeypatch.setattr(public_init, "SPLIT_SHA256", "0" * 64)
    output, report = tmp_path / "init.pth", tmp_path / "report.json"
    with pytest.raises(ValueError):
        public_init.initialize_public_checkpoint(path, split, output, report)
    assert not output.exists() and not report.exists()


def test_exclusive_publication_safe_roundtrip_and_inputs_preserved(public_files, tmp_path):
    _, path, split = public_files
    before = (path.read_bytes(), split.read_bytes())
    output, report = tmp_path / "init.pth", tmp_path / "report.json"
    result = public_init.initialize_public_checkpoint(path, split, output, report)
    assert json.loads(report.read_text())["checkpoint_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    saved = torch.load(output, map_location="cpu", weights_only=True)
    assert saved["step"] == 0 and saved["optimizer"] == {}
    assert tensor_state_sha256(saved["model"]) == result["initial_model_state_sha256"]
    assert result["strict_full_model_load_verified"] and result["safe_cpu_roundtrip_verified"]
    assert all(value.device.type == "cpu" for value in saved["model"].values())
    assert before == (path.read_bytes(), split.read_bytes())
    previous = (output.read_bytes(), report.read_bytes())
    with pytest.raises(ValueError, match="existing"):
        public_init.initialize_public_checkpoint(path, split, output, report)
    assert previous == (output.read_bytes(), report.read_bytes())


def test_existing_report_or_dangling_symlink_blocks_checkpoint(public_files, tmp_path):
    _, path, split = public_files
    output, report = tmp_path / "init.pth", tmp_path / "report.json"
    report.symlink_to(tmp_path / "not-created.json")
    with pytest.raises(ValueError, match="existing"):
        public_init.initialize_public_checkpoint(path, split, output, report)
    assert not output.exists() and report.is_symlink()


@pytest.mark.parametrize("env", [None, "0", "GPU-abc"])
def test_explicit_cpu_namespace_required_before_io(monkeypatch, env):
    if env is None: monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else: monkeypatch.setenv("CUDA_VISIBLE_DEVICES", env)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        public_init.prepare_public_payload("does-not-exist", "does-not-exist")


def test_initialized_cuda_rejected_without_using_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="must not be initialized"):
        public_init.prepare_public_payload("does-not-exist", "does-not-exist")


def test_source_change_before_publication_fails_closed(public_files, tmp_path, monkeypatch):
    _, path, split = public_files
    original_snapshot = public_init.source_snapshot
    calls = []
    def changing_snapshot(*args):
        result = original_snapshot(*args)
        calls.append(1)
        if len(calls) >= 3:
            result = copy.deepcopy(result)
            result["file_sha256"]["models/motiondrive_v2/model.py"] = "0" * 64
        return result
    monkeypatch.setattr(public_init, "source_snapshot", changing_snapshot)
    output, report = tmp_path / "init.pth", tmp_path / "report.json"
    with pytest.raises(ValueError, match="source files changed"):
        public_init.initialize_public_checkpoint(path, split, output, report)
    assert not output.exists() and not report.exists()
