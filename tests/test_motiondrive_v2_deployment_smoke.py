import copy
import os

import pytest
import torch

from scripts import smoke_motiondrive_v2_deployment as smoke


def test_plan_comparison_absolute_and_fixed_tolerance():
    plan = torch.arange(12).reshape(6, 2).float()
    assert smoke.compare_plans(plan, plan)["bitwise_equal"]
    assert not smoke.compare_plans(plan, plan.cumsum(0))["pass"]
    difference = torch.zeros(6, 2)
    difference[0, 0] = 2e-5
    assert not smoke.compare_plans(difference, torch.zeros(6, 2))["pass"]
    assert smoke.PLAN_ATOL == 1e-5


@pytest.mark.parametrize("value", [torch.zeros(1, 6, 2), torch.zeros(6, 2).half(),
                                    torch.full((6, 2), float("nan"))])
def test_bad_plan_cannot_pass(value):
    with pytest.raises(ValueError):
        smoke.compare_plans(value, torch.zeros(6, 2))


class Toy(torch.nn.Module):
    def __init__(self, wrong_encodings=False):
        super().__init__()
        self.backbone_fpn = torch.nn.Identity()
        self.wrong_encodings = wrong_encodings

    def forward(self, **inputs):
        self.backbone_fpn(torch.empty(6, 3, 432, 768, device="meta"))
        self.backbone_fpn(torch.empty(4 if self.wrong_encodings else 5, 3, 216, 384, device="meta"))
        return {"plan_abs": torch.arange(12).reshape(1, 6, 2).float(),
            "state_hat": torch.zeros(1, 6), "history_hat": torch.zeros(1, 4, 4),
            "scene_features": torch.zeros(1, 3072, 128), "motion_features": torch.zeros(1, 192, 128),
            "occ_logits": torch.zeros(1, 1, 64, 48), "lane_logits": torch.zeros(1, 1, 64, 48)}


def direct_forward(model, inputs, contract, **kwargs):
    return model(**inputs)["plan_abs"][0]


def test_complete_forward_hooks_and_pure_serialization(monkeypatch):
    monkeypatch.setattr(smoke, "forward_clip", direct_forward)
    model = Toy()
    inputs = {key: torch.zeros(1) for key in smoke.adapter.INPUT_KEYS}
    plan, report = smoke.observed_forward(model, inputs, {}, device="cpu", precision="fp32")
    assert report["full_forward_count"] == 1
    assert report["image_encodings"] == 11
    assert report["absolute_json_roundtrip_bitwise"]
    assert plan[-1].tolist() == [10., 11.]
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert not model.backbone_fpn._forward_pre_hooks


def test_reject_missing_current_low_resolution_encoding(monkeypatch):
    monkeypatch.setattr(smoke, "forward_clip", direct_forward)
    model = Toy(wrong_encodings=True)
    with pytest.raises(RuntimeError, match="6\\+5"):
        smoke.observed_forward(model, {key: torch.zeros(1) for key in smoke.adapter.INPUT_KEYS},
                               {}, device="cpu", precision="fp32")
    assert not model._forward_hooks


def test_no_labels_into_full_forward(monkeypatch):
    monkeypatch.setattr(smoke, "forward_clip", direct_forward)
    inputs = {key: torch.zeros(1) for key in smoke.adapter.INPUT_KEYS}
    inputs["state_target"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="non-whitelisted"):
        smoke.observed_forward(Toy(), inputs, {}, device="cpu", precision="fp32")


def test_gpu_guard_accepts_only_own_pid_after_initial_idle(monkeypatch):
    def query(command, **kwargs):
        return ("0, GPU-test, 130, 23986, 24576\n" if "--query-gpu=" in command[1]
                else f"GPU-test, {os.getpid()}, python\n")
    monkeypatch.setattr(smoke.subprocess, "check_output", query)
    assert not smoke.gpu_snapshot()["foreign_processes"]
    with pytest.raises(RuntimeError, match="uncontended"):
        smoke.gpu_snapshot(require_idle=True)


def test_gpu_guard_rejects_foreign_process_without_signals(monkeypatch):
    def query(command, **kwargs):
        return ("0, GPU-test, 130, 23986, 24576\n" if "--query-gpu=" in command[1]
                else f"GPU-test, {os.getpid() + 999999}, foreign\n")
    monkeypatch.setattr(smoke.subprocess, "check_output", query)
    with pytest.raises(RuntimeError, match="no process will be stopped"):
        smoke.gpu_snapshot()
