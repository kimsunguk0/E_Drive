"""CPU tests for the fresh-canary auxiliary update audit."""
import argparse
import copy
import hashlib

import pytest
import torch
from torch import nn

from scripts import audit_motiondrive_v2_fresh_aux_updates as audit


class TinyOptimizerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_fpn = nn.Module()
        self.backbone_fpn.stem = nn.Linear(2, 2)
        self.backbone_fpn.lat2 = nn.Linear(2, 2)
        self.scene_encoder = nn.Module()
        self.scene_encoder.occ_head = nn.Linear(2, 1)
        self.scene_encoder.lane_head = nn.Linear(2, 1)
        self.motion_encoder = nn.Module()
        self.motion_encoder.history_head = nn.Linear(2, 1)
        self.motion_encoder.state_head = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 6))
        self.motion_encoder.history_uncertainty_head = nn.Linear(2, 1)
        self.motion_encoder.state_uncertainty_head = nn.Linear(2, 1)


def trainer_optimizer(model):
    backbone, other = [], []
    for name, parameter in model.named_parameters():
        is_backbone = name.startswith("backbone_fpn.") and not name.startswith(
            ("backbone_fpn.lat", "backbone_fpn.out"))
        (backbone if is_backbone else other).append(parameter)
    return torch.optim.AdamW([{"params": backbone, "lr": 1e-2},
                              {"params": other, "lr": 1e-2}], weight_decay=.1)


def states(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_optimizer_ids_follow_original_two_group_named_parameter_order():
    model = TinyOptimizerModel()
    optimizer = trainer_optimizer(model)
    mapping, groups = audit.optimizer_id_to_name(model, optimizer.state_dict())
    assert groups[0]["role"] == "backbone" and groups[1]["role"] == "other"
    assert [mapping[value] for value in optimizer.state_dict()["param_groups"][0]["params"]] == [
        "backbone_fpn.stem.weight", "backbone_fpn.stem.bias"]
    assert [mapping[value] for value in optimizer.state_dict()["param_groups"][1]["params"]][:2] == [
        "backbone_fpn.lat2.weight", "backbone_fpn.lat2.bias"]
    assert set(mapping.values()) == {name for name, _ in model.named_parameters()}


def test_changed_by_weight_decay_only_is_not_reported_as_gradient_update():
    model = TinyOptimizerModel()
    optimizer = trainer_optimizer(model)
    initial = states(model)
    for name, parameter in model.named_parameters():
        parameter.grad = (torch.ones_like(parameter) if name.startswith("scene_encoder.occ_head.")
                          else torch.zeros_like(parameter))
    optimizer.step()
    last = states(model)
    saved = optimizer.state_dict()
    mapping, _ = audit.optimizer_id_to_name(model, saved)
    occ = audit.parameter_update_report(initial, last, saved, mapping,
                                        audit.PRIMARY_AUXILIARIES["occupancy"])
    lane = audit.parameter_update_report(initial, last, saved, mapping,
                                         audit.PRIMARY_AUXILIARIES["lane"])
    assert occ["actual_delta_count"] and occ["finite_nonzero_exp_avg_count"]
    assert occ["actual_gradient_update_observed"]
    assert lane["actual_delta_count"]  # AdamW changed it despite a zero gradient.
    assert lane["finite_nonzero_exp_avg_count"] == 0
    assert not lane["actual_gradient_update_observed"]


def test_stop_row_finds_only_output_index_five_moment():
    model = TinyOptimizerModel()
    optimizer = trainer_optimizer(model)
    initial = states(model)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    model.motion_encoder.state_head[2].weight.grad[5].fill_(2.)
    model.motion_encoder.state_head[2].bias.grad[5] = 2.
    optimizer.step()
    saved, last = optimizer.state_dict(), states(model)
    mapping, _ = audit.optimizer_id_to_name(model, saved)
    result = audit.stop_row_report(initial, last, saved, mapping)
    assert result["actual_delta_count"] == 2
    assert result["finite_nonzero_exp_avg_count"] == 2
    assert result["actual_gradient_update_observed"]
    assert all(row["output_index"] == 5 for row in result["rows"])


def test_mapping_rejects_group_reordering_or_unknown_state_id():
    model = TinyOptimizerModel()
    saved = trainer_optimizer(model).state_dict()
    wrong = copy.deepcopy(saved)
    wrong["param_groups"][0]["params"].append(wrong["param_groups"][1]["params"].pop())
    with pytest.raises(ValueError, match="length/order"):
        audit.optimizer_id_to_name(model, wrong)
    unknown = copy.deepcopy(saved)
    unknown["state"][99999] = {"exp_avg": torch.zeros(1)}
    with pytest.raises(ValueError, match="unmapped"):
        audit.optimizer_id_to_name(model, unknown)


def test_mapping_rejects_internal_serialized_id_swap():
    model = TinyOptimizerModel()
    saved = trainer_optimizer(model).state_dict()
    saved["param_groups"][1]["params"][:2] = reversed(saved["param_groups"][1]["params"][:2])
    with pytest.raises(ValueError, match="serialized id order"):
        audit.optimizer_id_to_name(model, saved)


def test_gradient_update_requires_delta_and_moment_on_same_parameter():
    model = TinyOptimizerModel()
    optimizer = trainer_optimizer(model).state_dict()
    mapping, _ = audit.optimizer_id_to_name(model, optimizer)
    initial, last = states(model), states(model)
    weight = "scene_encoder.occ_head.weight"
    bias = "scene_encoder.occ_head.bias"
    last[weight] = last[weight] * .99  # A weight-decay-only-looking delta.
    name_to_id = {name: key for key, name in mapping.items()}
    optimizer["state"] = {
        name_to_id[weight]: {"exp_avg": torch.zeros_like(last[weight])},
        name_to_id[bias]: {"exp_avg": torch.ones_like(last[bias])},
    }
    result = audit.parameter_update_report(initial, last, optimizer, mapping,
                                           audit.PRIMARY_AUXILIARIES["occupancy"])
    assert result["actual_delta_count"] == 1
    assert result["finite_nonzero_exp_avg_count"] == 1
    assert result["changed_with_nonzero_exp_avg_parameters"] == []
    assert not result["actual_gradient_update_observed"]


def test_input_hash_and_existing_output_are_read_only(tmp_path):
    source = tmp_path / "input.pth"
    source.write_bytes(b"fixed checkpoint bytes")
    assert audit.sha256(source) == hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "audit.json"
    output.write_text("preserve")
    with pytest.raises(ValueError, match="existing output"):
        audit.new_output(output)
    assert output.read_text() == "preserve"


def test_explicit_pretrain_2000_stage_contract_is_accepted():
    result = audit.validate_expected_stage({"phase": "pretrain", "steps": 2000},
                                           2000, 2000, "pretrain", 2000)
    assert result == {"expected": {"phase": "pretrain", "steps": 2000},
                      "actual": {"manifest_phase": "pretrain", "manifest_argument_steps": 2000,
                                 "last_payload_step": 2000, "sidecar_step": 2000}}


def test_default_canary_stage_contract_remains_joint_two_steps():
    result = audit.validate_expected_stage({"phase": "joint", "steps": 2}, 2, 2, "joint", 2)
    assert result["expected"] == {"phase": "joint", "steps": 2}


def producer_consumer_fixture():
    initial = {"git_sha": "producer-git", "initial_model_state_sha256": "tensor-sha",
               "source": {"git_sha": "producer-git", "file_sha256": {"model.py": "a" * 64}}}
    last = {"git_sha": "trainer-git"}
    sidecar = {"git_sha": "trainer-git", "load_report": {"common_checkpoint_sha256": "file-sha"},
               "initial_model_state_sha256": "tensor-sha"}
    return initial, last, sidecar


def test_different_initializer_producer_and_trainer_git_is_valid_lineage():
    initial, last, sidecar = producer_consumer_fixture()
    result = audit.validate_producer_consumer_lineage(
        initial, last, sidecar, "file-sha", "tensor-sha")
    assert result["initializer_producer_git_sha"] == "producer-git"
    assert result["trainer_git_sha"] == "trainer-git"
    assert result["producer_and_trainer_git_may_differ"] is True


@pytest.mark.parametrize("fault", ["file_sha", "tensor_sha", "trainer_git"])
def test_producer_consumer_lineage_rejects_wrong_consumption_or_trainer_git(fault):
    initial, last, sidecar = producer_consumer_fixture()
    if fault == "file_sha":
        sidecar["load_report"]["common_checkpoint_sha256"] = "other-file"
    elif fault == "tensor_sha":
        sidecar["initial_model_state_sha256"] = "other-tensor"
    else:
        last["git_sha"] = "other-trainer"
    with pytest.raises(ValueError):
        audit.validate_producer_consumer_lineage(
            initial, last, sidecar, "file-sha", "tensor-sha")


@pytest.mark.parametrize("arguments,last_step,sidecar_step", [
    ({"phase": "joint", "steps": 2000}, 2000, 2000),
    ({"phase": "pretrain", "steps": 1999}, 2000, 2000),
    ({"phase": "pretrain", "steps": 2000}, 1999, 2000),
    ({"phase": "pretrain", "steps": 2000}, 2000, 1999),
])
def test_explicit_pretrain_stage_rejects_phase_or_step_mismatch(arguments, last_step, sidecar_step):
    with pytest.raises(ValueError, match="explicit expected stage"):
        audit.validate_expected_stage(arguments, last_step, sidecar_step, "pretrain", 2000)


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_cli_expected_steps_requires_positive_integer(value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        audit.positive_int(value)


@pytest.mark.parametrize("value", [None, "0", "GPU-example"])
def test_explicit_empty_cuda_namespace_required(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        audit.assert_cpu_only()
