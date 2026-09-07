import argparse
import contextlib
import dataclasses
import hashlib

import pytest
import torch

from models.motiondrive_v2 import MotionDriveV2Config
from models.motiondrive_v2_input_contract import input_contract
from scripts import smoke_motiondrive_v2_p4_deployment as p4
from scripts.export_motiondrive_v2_inference import (
    C1_CANONICAL_SHA256, C1_SUPERVISION_SHA256, RAWTIME_SPLIT_SHA256,
)
from scripts.motiondrive_v2_training import time_input_policy


CHECKPOINT_SHA = "a" * 64
SIDECAR_SHA = "b" * 64
TRAINING_GIT = "1" * 40


@pytest.fixture(autouse=True)
def cpu_only_unit_test():
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def complete_p4_config():
    config = dataclasses.asdict(MotionDriveV2Config())
    config.update(backbone_arch="resnet50", goal_on=True, state_on=True,
                  motion_input_mode="low_feature", plan_output_scale=[10., 5.],
                  n_history=4)
    return config


def valid_bundle():
    evidence = {role: {"sha256": digest} for role, digest in (
        ("source_checkpoint", CHECKPOINT_SHA),
        ("completed_run_manifest", SIDECAR_SHA),
        ("canonical_calibration", C1_CANONICAL_SHA256),
        ("supervision_manifest", C1_SUPERVISION_SHA256),
        ("split_manifest", RAWTIME_SPLIT_SHA256),
    )}
    return {
        "format": "motiondrive_v2_inference", "format_version": 1,
        "input_contract": input_contract(), "step": 6000,
        "source_checkpoint": {"sha256": CHECKPOINT_SHA, "step": 6000,
                              "training_git_sha": TRAINING_GIT},
        "manifest": {
            "git_sha": TRAINING_GIT, "time_input": "nominal",
            "arguments": {"time_input": "nominal", "steps": 6000,
                          "phase": "joint"},
            "time_input_policy": time_input_policy("nominal"),
            "supervision_manifest_sha256": C1_SUPERVISION_SHA256,
            "split_sha256": RAWTIME_SPLIT_SHA256,
            "model_config": complete_p4_config(),
            "loss_weights": {"plan": 1.0},
        },
        "deployment_provenance": {
            "mode": "geometry-v2-nominal",
            "status": "input_contract_lineage_verified",
            "selected_checkpoint_sha256": CHECKPOINT_SHA,
            "checkpoint_step": 6000, "completed_run_step": 6000,
            "geometry_edition": "geometry_v2", "time_input": "nominal",
            "canonical_calibration_sha256": C1_CANONICAL_SHA256,
            "supervision_manifest_sha256": C1_SUPERVISION_SHA256,
            "split_sha256": RAWTIME_SPLIT_SHA256,
            "training_git_sha": TRAINING_GIT,
            "run_dir": "/arbitrary/non-canary/p4_fresh_joint_s0",
            "evidence": evidence,
        },
    }


def validate(bundle):
    return p4.validate_p4_bundle(
        bundle, expected_checkpoint_sha256=CHECKPOINT_SHA,
        expected_run_manifest_sha256=SIDECAR_SHA)


def test_exact_p4_bundle_accepts_noncanary_run_and_does_not_bind_evaluation_git():
    receipt = validate(valid_bundle())
    assert receipt["checkpoint_step"] == 6000
    assert receipt["original_run_dir"].endswith("p4_fresh_joint_s0")
    assert receipt["training_source_git_sha"] == TRAINING_GIT
    assert receipt["evaluation_source_git_must_equal_training_source"] is False
    assert receipt["completed_run_manifest_sha256"] == SIDECAR_SHA


@pytest.mark.parametrize("field", ["bundle", "source", "checkpoint", "completed", "arguments"])
def test_every_step_receipt_must_be_exactly_last6000(field):
    bundle = valid_bundle()
    if field == "bundle":
        bundle["step"] = 2
    elif field == "source":
        bundle["source_checkpoint"]["step"] = 2
    elif field == "checkpoint":
        bundle["deployment_provenance"]["checkpoint_step"] = 2
    elif field == "completed":
        bundle["deployment_provenance"]["completed_run_step"] = 6001
    else:
        bundle["manifest"]["arguments"]["steps"] = 2
    with pytest.raises(ValueError):
        validate(bundle)


@pytest.mark.parametrize("change", ["phase", "plan_loss", "backbone", "goal", "state",
                                     "motion", "scale", "history", "incomplete"])
def test_wrong_phase_or_complete_model_config_fails_closed(change):
    bundle = valid_bundle()
    if change == "phase":
        bundle["manifest"]["arguments"]["phase"] = "pretrain"
    elif change == "plan_loss":
        bundle["manifest"]["loss_weights"]["plan"] = 0
    elif change == "backbone":
        bundle["manifest"]["model_config"]["backbone_arch"] = "resnet34"
    elif change == "goal":
        bundle["manifest"]["model_config"]["goal_on"] = False
    elif change == "state":
        bundle["manifest"]["model_config"]["state_on"] = False
    elif change == "motion":
        bundle["manifest"]["model_config"]["motion_input_mode"] = "high_feature"
    elif change == "scale":
        bundle["manifest"]["model_config"]["plan_output_scale"] = [1., 1.]
    elif change == "history":
        bundle["manifest"]["model_config"]["n_history"] = 3
    else:
        del bundle["manifest"]["model_config"]["channels"]
    with pytest.raises(ValueError):
        validate(bundle)


@pytest.mark.parametrize("change", ["selected", "source", "source_evidence",
                                     "missing_sidecar", "sidecar", "checkpoint_receipt",
                                     "sidecar_receipt"])
def test_checkpoint_and_sidecar_sha_lineage_is_externally_pinned(change):
    bundle = valid_bundle()
    if change == "selected":
        bundle["deployment_provenance"]["selected_checkpoint_sha256"] = "c" * 64
    elif change == "source":
        bundle["source_checkpoint"]["sha256"] = "c" * 64
    elif change == "source_evidence":
        bundle["deployment_provenance"]["evidence"]["source_checkpoint"]["sha256"] = "c" * 64
    elif change == "missing_sidecar":
        del bundle["deployment_provenance"]["evidence"]["completed_run_manifest"]
    elif change == "sidecar":
        bundle["deployment_provenance"]["evidence"]["completed_run_manifest"]["sha256"] = "c" * 64
    elif change == "checkpoint_receipt":
        with pytest.raises(ValueError, match="external receipt"):
            p4.validate_p4_bundle(bundle, expected_checkpoint_sha256="c" * 64,
                                   expected_run_manifest_sha256=SIDECAR_SHA)
        return
    else:
        with pytest.raises(ValueError, match="external receipt"):
            p4.validate_p4_bundle(bundle, expected_checkpoint_sha256=CHECKPOINT_SHA,
                                   expected_run_manifest_sha256="c" * 64)
        return
    with pytest.raises(ValueError):
        validate(bundle)


@pytest.mark.parametrize("location", ["manifest", "source", "provenance"])
def test_all_training_git_receipts_must_match(location):
    bundle = valid_bundle()
    if location == "manifest":
        bundle["manifest"]["git_sha"] = "2" * 40
    elif location == "source":
        bundle["source_checkpoint"]["training_git_sha"] = "2" * 40
    else:
        bundle["deployment_provenance"]["training_git_sha"] = "2" * 40
    with pytest.raises(ValueError, match="Git|git"):
        validate(bundle)


@pytest.mark.parametrize("change", ["raw_time", "geometry", "supervision", "split"])
def test_wrong_c1_nominal_contract_fails_closed(change):
    bundle = valid_bundle()
    provenance = bundle["deployment_provenance"]
    if change == "raw_time":
        bundle["manifest"]["arguments"]["time_input"] = "raw"
    elif change == "geometry":
        provenance["geometry_edition"] = "geometry_v1"
    elif change == "supervision":
        provenance["supervision_manifest_sha256"] = "c" * 64
    else:
        provenance["split_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        validate(bundle)


def test_source_manifest_accepts_a_different_valid_evaluation_git(tmp_path):
    (tmp_path / "runtime.py").write_text("tracked runtime\n")
    digest = hashlib.sha256((tmp_path / "runtime.py").read_bytes()).hexdigest()
    before, evaluation_git = p4.validate_source_manifest(
        {"source_git_sha": "9" * 40, "files": {"runtime.py": digest}}, root=tmp_path)
    assert before == {"runtime.py": digest}
    assert evaluation_git != TRAINING_GIT


@pytest.mark.parametrize("change", ["content", "git", "missing"])
def test_source_manifest_fails_closed(change, tmp_path):
    path = tmp_path / "runtime.py"
    path.write_text("tracked runtime\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"source_git_sha": "9" * 40, "files": {"runtime.py": digest}}
    if change == "content":
        path.write_text("changed\n")
    elif change == "git":
        manifest["source_git_sha"] = "not-a-git-sha"
    else:
        manifest["files"] = {}
    with pytest.raises(ValueError):
        p4.validate_source_manifest(manifest, root=tmp_path)


def test_bundle_sha_checked_before_tensor_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "bundle.pth"
    path.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs:
                        pytest.fail("must reject the SHA before deserialization"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        p4.read_pinned_bundle(path, "c" * 64)


def test_bundle_hash_is_checked_again_after_safe_load(tmp_path):
    path = tmp_path / "bundle.pth"
    torch.save({"tensor": torch.arange(3)}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    restored = p4.read_pinned_bundle(path, digest)
    assert torch.equal(restored["tensor"], torch.arange(3))


def test_exactly_eight_input_parities_are_required():
    p4.require_parity([{"all_pass": True} for _ in range(8)])
    with pytest.raises(ValueError):
        p4.require_parity([{"all_pass": True} for _ in range(7)])
    rows = [{"all_pass": True} for _ in range(8)]
    rows[3]["all_pass"] = False
    with pytest.raises(ValueError):
        p4.require_parity(rows)


def test_aba_requires_same_a_after_intervening_b():
    first = torch.zeros(6, 2, dtype=torch.float32)
    middle = torch.ones(6, 2, dtype=torch.float32)
    report = p4.validate_aba(first, middle, first.clone())
    assert report["a_repeat"]["bitwise_equal"]
    assert report["a_vs_b_bitwise_different"]
    changed = first.clone()
    changed[0, 0] = 2e-5
    with pytest.raises(ValueError, match="stateless"):
        p4.validate_aba(first, middle, changed)


@pytest.mark.parametrize("value", [0, -1, "0", "bad", None])
def test_warmup_and_repeats_parser_rejects_nonpositive_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        p4.positive_int(value)


def test_timing_rejects_cpu_and_bad_batch_before_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs:
                        pytest.fail("CUDA must not be touched"))
    with pytest.raises(ValueError, match="explicit CUDA"):
        p4.timed_full_forward(object(), {}, device="cpu", warmup=1, repeats=1)
    with pytest.raises(ValueError, match="six raw-clip"):
        p4.timed_full_forward(object(), {}, device="cuda:0", warmup=1, repeats=1)


class TimedToy:
    def __init__(self, log):
        self.log = log
        self.calls = 0

    def __call__(self, **inputs):
        self.calls += 1
        self.log.append("forward")
        return {"plan_abs": torch.zeros(1, 6, 2),
                "history_hat": torch.zeros(1, 4, 4),
                "state_hat": torch.zeros(1, 6)}


def test_cuda_event_core_orders_warmup_sync_and_complete_forward(monkeypatch):
    log = []

    class Event:
        created = 0

        def __init__(self, **kwargs):
            self.name = "start" if Event.created == 0 else "end"
            Event.created += 1
            log.append("new_" + self.name)

        def record(self):
            log.append("record_" + self.name)

        def elapsed_time(self, other):
            assert self.name == "start" and other.name == "end"
            log.append("elapsed")
            return 7.0

    monkeypatch.setattr(torch, "autocast", lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: log.append("sync"))
    model = TimedToy(log)
    output, cuda_ms, wall_ms = p4.measure_full_forward_samples(
        model, {name: torch.zeros(1) for name in p4.adapter.INPUT_KEYS},
        device=torch.device("cuda:0"), warmup=2, repeats=2)
    assert model.calls == 4 and output["plan_abs"].shape == (1, 6, 2)
    assert cuda_ms == [7., 7.] and len(wall_ms) == 2
    assert log == ["forward", "forward", "sync", "new_start", "new_end",
                   "sync", "record_start", "forward", "record_end", "sync", "elapsed",
                   "sync", "record_start", "forward", "record_end", "sync", "elapsed"]


def test_report_code_marks_real_measurement_separate_from_old_p1_result():
    source = p4.Path(p4.__file__).read_text()
    assert '"not_the_old_p1_32ms_measurement": True' in source
    assert '"p4_measurement_completed": True' in source
    assert '"status": "completed_pass"' in source
    assert "32.2306" not in source
