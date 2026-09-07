"""CPU contract tests only; no P7 optimizer update is executed here."""
import json
from pathlib import Path
import types

import pytest
import torch

from scripts import smoke_motiondrive_v2_p7_training as smoke


def test_three_update_schedule_is_prefix_of_fixed_joint6000_recipe():
    assert smoke.UPDATES == 3 and smoke.LOGICAL_BATCH == 16 and smoke.MICROBATCH == 2
    assert [smoke.lr_factor(step) for step in (1, 2, 3)] == pytest.approx(
        [1 / 200, 2 / 200, 3 / 200])
    for bad in (0, 4, 1.0):
        with pytest.raises(ValueError): smoke.lr_factor(bad)


def tiny_model():
    class Branch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query_image = torch.nn.Linear(2, 2, bias=False)
            self.key_image = torch.nn.Linear(2, 2, bias=False)
            self.value_image = torch.nn.Linear(2, 2, bias=False)
            self.output = torch.nn.Linear(2, 2, bias=False)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone_fpn = torch.nn.Module()
            self.backbone_fpn.core = torch.nn.Linear(2, 2)
            self.backbone_fpn.lat = torch.nn.Linear(2, 2)
            self.scene_encoder = torch.nn.Module()
            self.scene_encoder.cross_cell_goal_residual = Branch()
            self.planner = torch.nn.Linear(2, 2)
    return Model()


def test_optimizer_is_fresh_two_group_recipe_and_branch_is_nonbackbone():
    model = tiny_model()
    optimizer, names = smoke.optimizer_for(model)
    assert not optimizer.state
    assert [group["base_lr"] for group in optimizer.param_groups] == [1e-5, 1e-4]
    assert all(name.startswith("scene_encoder.cross_cell_goal_residual.") for name in names)
    backbone_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    head_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    for name, parameter in model.named_parameters():
        if name.startswith("scene_encoder.cross_cell_goal_residual."):
            assert id(parameter) in head_ids and id(parameter) not in backbone_ids


def test_gradient_gate_requires_output_first_then_qkv():
    branch = tiny_model().scene_encoder.cross_cell_goal_residual
    for parameter in branch.parameters(): parameter.grad = torch.zeros_like(parameter)
    branch.output.weight.grad.fill_(1)
    first = smoke.gradient_receipt(branch, 1)
    assert first["output"]["nonzero"] and not first["query"]["nonzero"]
    for parameter in branch.parameters(): parameter.grad.fill_(1)
    later = smoke.gradient_receipt(branch, 2)
    assert all(later[name]["nonzero"] for name in ("output", "query", "key", "value"))
    branch.key_image.weight.grad.zero_()
    with pytest.raises(ValueError, match="Q/K/V"): smoke.gradient_receipt(branch, 3)


def report(arm):
    return {"status": "completed_three_update_canary_not_training_result",
            "arm": arm, "base_seed": 0, "initial_model_state_sha256": "a" * 64,
            "sample_order_sha256": "b" * 64, "sample_rows": list(range(48))}


def test_pair_receipt_requires_same_base_initial_state_and_data_order():
    control, goal = report("control_zero_slot"), report("goal_real_slot")
    assert smoke.compare_arm_receipts(control, goal) == {
        "same_base_initial_state_exact": True, "same_data_order_exact": True,
        "updates_each": 3, "scientific_result": False}
    goal["sample_rows"][-1] = 99
    with pytest.raises(ValueError, match="sample_rows"):
        smoke.compare_arm_receipts(control, goal)


def test_atomic_report_is_new_only_and_json_roundtrips(tmp_path):
    path = tmp_path / "canary.json"
    payload = {"status": "completed_three_update_canary_not_training_result",
               "scientific_result": False, "finite": True}
    smoke.atomic_new_json(path, payload)
    assert json.loads(path.read_text()) == payload
    with pytest.raises(ValueError, match="overwrite"):
        smoke.atomic_new_json(path, payload)


def test_cli_isolated_output_and_fixed_resource_defaults():
    args = smoke.arguments([
        "--arm", "control_zero_slot", "--base-seed", "0", "--init", "/p0",
        "--expected-init-sha256", "a" * 64, "--run-manifest", "/sidecar",
        "--expected-run-manifest-sha256", "b" * 64, "--source-manifest", "/source",
        "--expected-source-manifest-sha256", "c" * 64,
        "--expected-smoke-source-sha256", "d" * 64, "--data-root", "/data",
        "--split-manifest", "/split", "--supervision-root", "/sup",
        "--output-dir", "/new/p7_goal_routing_canary_b0_c",
        "--expected-physical-gpu-uuid", "GPU-4b804d68-fd61-af14-393a-573c533d5006"])
    assert (args.gpu, args.workers, args.cuda_memory_limit_mib,
            args.cuda_min_free_mib, args.preflight_only) == (0, 4, 12000, 8192, False)
    assert "canary" in Path(args.output_dir).name


def test_probe_source_has_no_evaluation_or_checkpoint_save_path():
    source = Path(smoke.__file__).read_text()
    assert "trainer.evaluate(" not in source
    assert "torch.save(" not in source
    assert '"tune_evaluations": 0' in source
    assert '"checkpoint_written": False' in source
    assert "eligible_as_training_initializer" in source


def test_help_is_cpu_import_only():
    with pytest.raises(SystemExit) as result:
        smoke.arguments(["--help"])
    assert result.value.code == 0 and not torch.cuda.is_initialized()
