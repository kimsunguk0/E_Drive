import copy
import json
import hashlib
import dataclasses

import pytest
import torch

from models.motiondrive_v2_input_contract import input_contract
from models.motiondrive_v2_serving import (
    SHAPES, absolute_plan_to_list, forward_clip, validate_clip_inputs,
    validate_bundle_contract, load_deployment_bundle, explicit_device,
)


@pytest.fixture
def clip():
    inputs = {name: torch.zeros(shape) for name, shape in SHAPES.items()}
    inputs["time_offsets"] = torch.tensor([[.1, .2, .5, 1.]])
    return inputs


class ToyCompleteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("xy", torch.arange(12).reshape(6, 2).float())
        self.calls = 0
        self.received = None

    def forward(self, **inputs):
        self.calls += 1
        self.received = set(inputs)
        # Fixed-offset image dependence makes A/B/A interleaving observable.
        return {"plan_abs": self.xy[None] + inputs["images"][0, 0, 0, 0, 0]}

    def forward_parts(self, *args, **kwargs):
        raise AssertionError("The serving path must call the complete forward")


def test_one_complete_forward_and_no_cumsum(clip):
    model = ToyCompleteModel().eval()
    result = forward_clip(model, clip, input_contract())
    assert model.calls == 1 and model.received == set(SHAPES)
    assert torch.equal(result, model.xy)
    encoded = json.dumps(absolute_plan_to_list(result), allow_nan=False)
    assert torch.equal(torch.tensor(json.loads(encoded)), model.xy)
    assert not torch.equal(result, model.xy.cumsum(0))


def test_interleaved_clips_do_not_reuse_predictions_or_features(clip):
    model = ToyCompleteModel().eval()
    first = forward_clip(model, clip, input_contract())
    other = dict(clip)
    other["images"] = clip["images"].clone()
    other["images"][0, 0, 0, 0, 0] = 2.
    middle = forward_clip(model, other, input_contract())
    final = forward_clip(model, clip, input_contract())
    assert model.calls == 3
    assert torch.equal(first, final) and not torch.equal(first, middle)
    assert clip["images"].count_nonzero() == 0


@pytest.mark.parametrize("extra", ["gt_plan", "state_target", "cached_features", "command", "metadata"])
def test_extra_inputs_are_rejected_before_forward(clip, extra):
    model = ToyCompleteModel().eval()
    clip[extra] = torch.zeros(1)
    with pytest.raises(ValueError, match="Exactly"):
        forward_clip(model, clip, input_contract())
    assert model.calls == 0


@pytest.mark.parametrize("change", ["missing", "shape", "dtype", "nan", "requires_grad", "raw_time"])
def test_invalid_input_is_rejected(clip, change):
    if change == "missing":
        del clip["goal_xy"]
    elif change == "shape":
        clip["goal_xy"] = torch.zeros(2, 2)
    elif change == "dtype":
        clip["goal_xy"] = clip["goal_xy"].half()
    elif change == "nan":
        clip["goal_xy"][0, 0] = float("nan")
    elif change == "requires_grad":
        clip["goal_xy"].requires_grad_(True)
    else:
        clip["time_offsets"][0, 0] = .100001
    with pytest.raises(ValueError):
        validate_clip_inputs(clip, input_contract())


def test_old_input_contract_rejected(clip):
    contract = copy.deepcopy(input_contract())
    contract["geometry_edition"] = "old"
    with pytest.raises(ValueError, match="geometry_edition"):
        validate_clip_inputs(clip, contract)


def test_training_mode_rejected_without_silently_mutating_model(clip):
    model = ToyCompleteModel()
    with pytest.raises(ValueError, match="evaluation mode"):
        forward_clip(model, clip, input_contract())
    assert model.training and model.calls == 0


@pytest.mark.parametrize("precision", ["fp16", "auto", ""])
def test_unsupported_precision_rejected(clip, precision):
    model = ToyCompleteModel().eval()
    with pytest.raises(ValueError, match="precision|fp32"):
        forward_clip(model, clip, input_contract(), precision=precision)
    assert model.calls == 0


@pytest.mark.parametrize("bad_output", ["shape", "dtype", "nan", "missing"])
def test_bad_output_does_not_get_silently_repaired(clip, bad_output):
    model = ToyCompleteModel().eval()
    value = model.xy[None].clone()
    if bad_output == "shape":
        value = value[:, :5]
    elif bad_output == "dtype":
        value = value.half()
    elif bad_output == "nan":
        value[0, 0, 0] = float("nan")
    model.forward = lambda **kwargs: {} if bad_output == "missing" else {"plan_abs": value}
    with pytest.raises(ValueError):
        forward_clip(model, clip, input_contract())


def test_bf16_autocast_still_requires_fp32_final_coordinates(clip):
    model = ToyCompleteModel().eval()
    result = forward_clip(model, clip, input_contract(), precision="bf16")
    assert result.dtype == torch.float32 and model.calls == 1


@pytest.mark.parametrize("value", [torch.zeros(1, 6, 2), torch.zeros(6, 2).half(), torch.full((6, 2), float("nan"))])
def test_serialization_rejects_invalid_coordinates(value):
    with pytest.raises(ValueError):
        absolute_plan_to_list(value)


@pytest.fixture
def packaged_contract():
    from scripts.export_motiondrive_v2_inference import (
        C1_CANONICAL_SHA256, C1_SUPERVISION_SHA256, RAWTIME_SPLIT_SHA256,
    )
    from scripts.motiondrive_v2_training import time_input_policy
    selected = "a" * 64
    return {"format": "motiondrive_v2_inference", "format_version": 1,
            "input_contract": input_contract(), "step": 3000,
            "source_checkpoint": {"sha256": selected, "step": 3000},
            "manifest": {"time_input": "nominal", "arguments": {"time_input": "nominal", "steps": 3000},
                         "time_input_policy": time_input_policy("nominal"),
                         "supervision_manifest_sha256": C1_SUPERVISION_SHA256,
                         "split_sha256": RAWTIME_SPLIT_SHA256, "model_config": {"n_history": 4}},
            "deployment_provenance": {"mode": "geometry-v2-nominal", "status": "input_contract_lineage_verified",
                "selected_checkpoint_sha256": selected, "checkpoint_step": 3000,
                "geometry_edition": "geometry_v2", "time_input": "nominal", "completed_run_step": 3000,
                "canonical_calibration_sha256": C1_CANONICAL_SHA256,
                "supervision_manifest_sha256": C1_SUPERVISION_SHA256, "split_sha256": RAWTIME_SPLIT_SHA256,
                "evidence": {role: {"sha256": sha} for role, sha in (
                    ("source_checkpoint", selected), ("canonical_calibration", C1_CANONICAL_SHA256),
                    ("supervision_manifest", C1_SUPERVISION_SHA256), ("split_manifest", RAWTIME_SPLIT_SHA256))}}}


def test_packaged_contract_does_not_open_original_remote_files(packaged_contract):
    validate_bundle_contract(packaged_contract)


@pytest.mark.parametrize("change", ["generic", "raw", "geometry", "source", "step", "missing_evidence", "crop"])
def test_unverified_or_inconsistent_bundle_is_not_deployment_ready(packaged_contract, change):
    bundle = packaged_contract
    if change == "generic":
        del bundle["deployment_provenance"]
    elif change == "raw":
        bundle["manifest"]["arguments"]["time_input"] = "raw"
    elif change == "geometry":
        bundle["deployment_provenance"]["evidence"]["canonical_calibration"]["sha256"] = "b" * 64
    elif change == "source":
        bundle["source_checkpoint"]["sha256"] = "b" * 64
    elif change == "step":
        bundle["step"] = 2000
    elif change == "missing_evidence":
        del bundle["deployment_provenance"]["evidence"]["split_manifest"]
    else:
        bundle["input_contract"]["crop"] = "top crop for every camera"
    with pytest.raises(ValueError):
        validate_bundle_contract(bundle)


def test_file_sha_is_required_before_unpickling(tmp_path, monkeypatch):
    path = tmp_path / "bundle.pth"
    path.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Must reject SHA before loading"))
    with pytest.raises(ValueError, match="SHA256"):
        load_deployment_bundle(path, expected_bundle_sha256="b" * 64)


def test_generic_checkpoint_is_rejected_even_with_correct_file_sha(tmp_path):
    path = tmp_path / "generic.pth"
    torch.save({"format": "motiondrive_v2_inference", "format_version": 1}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="geometry_v2"):
        load_deployment_bundle(path, expected_bundle_sha256=digest)


def test_verified_bundle_loader_preserves_state_and_sets_eval(packaged_contract, tmp_path, monkeypatch):
    import models.motiondrive_v2 as model_package
    reference = ToyCompleteModel()
    packaged_contract["manifest"]["model_config"] = dataclasses.asdict(model_package.MotionDriveV2Config())
    packaged_contract["model"] = reference.state_dict()
    path = tmp_path / "verified_mock.pth"
    torch.save(packaged_contract, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(model_package, "MotionDriveV2", lambda config: ToyCompleteModel())
    before_rng = torch.get_rng_state().clone()
    restored, contract = load_deployment_bundle(path, expected_bundle_sha256=digest)
    assert not restored.training and contract == input_contract()
    assert torch.equal(restored.xy, reference.xy)
    assert torch.equal(before_rng, torch.get_rng_state())


def test_loader_does_not_silently_cast_model_weights(packaged_contract, tmp_path, monkeypatch):
    import models.motiondrive_v2 as model_package
    packaged_contract["manifest"]["model_config"] = dataclasses.asdict(model_package.MotionDriveV2Config())
    packaged_contract["model"] = {"xy": torch.arange(12).reshape(6, 2).half()}
    path = tmp_path / "wrong_dtype.pth"
    torch.save(packaged_contract, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(model_package, "MotionDriveV2", lambda config: ToyCompleteModel())
    with pytest.raises(ValueError, match="silently cast"):
        load_deployment_bundle(path, expected_bundle_sha256=digest)


@pytest.mark.parametrize("field,value", [("geometry_edition", "old"), ("time_input", "raw"),
    ("canonical_calibration_sha256", "0" * 64), ("supervision_manifest_sha256", "0" * 64),
    ("split_sha256", "0" * 64), ("completed_run_step", 0), ("completed_run_step", 4000)])
def test_duplicate_provenance_cannot_contradict_manifest_or_evidence(packaged_contract, field, value):
    packaged_contract["deployment_provenance"][field] = value
    with pytest.raises(ValueError):
        validate_bundle_contract(packaged_contract)


@pytest.mark.parametrize("device", ["cuda", "cpu:0", "mps"])
def test_ambiguous_or_unsupported_device_rejected_without_cuda_initialization(device):
    with pytest.raises(ValueError, match="explicit CUDA"):
        explicit_device(device)
    assert not torch.cuda.is_initialized()


def test_explicit_device_parsing_does_not_touch_hardware():
    assert explicit_device("cpu") == torch.device("cpu")
    assert explicit_device("cuda:1") == torch.device("cuda:1")
    assert not torch.cuda.is_initialized()
