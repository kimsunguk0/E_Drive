import contextlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_motiondrive_v2_pv_screen as run
from models.motiondrive_v2.pv_planner_head import (
    FactorizedPVHead, K_COUNT, P_COUNT, V_COUNT, install_factorized_pv_head,
)


@pytest.fixture(scope="module")
def bank_tensors():
    path_id = torch.cat([
        torch.tensor([-1], dtype=torch.int32),
        torch.arange(P_COUNT, dtype=torch.int32).repeat_interleave(V_COUNT - 1),
    ])
    velocity_id = torch.cat([
        torch.tensor([0], dtype=torch.int32),
        torch.arange(1, V_COUNT, dtype=torch.int32).repeat(P_COUNT),
    ])
    bank = torch.zeros(K_COUNT, 6, 2, dtype=torch.float32)
    bank[1:, 0, 0] = torch.arange(1, K_COUNT, dtype=torch.float32) / K_COUNT
    generator = torch.Generator().manual_seed(7)
    path = torch.randn(P_COUNT, 30, generator=generator)
    velocity = torch.randn(V_COUNT, 11, generator=generator)
    return bank, path, velocity, path_id, velocity_id


def make_head(bank_tensors, residual=False):
    return FactorizedPVHead(*bank_tensors, residual=residual)


def test_fixed_full_cartesian_path_major_contract(bank_tensors):
    head = make_head(bank_tensors)
    assert K_COUNT == 65025
    assert torch.equal(head.path_id[1:128], torch.zeros(127, dtype=torch.long))
    assert torch.equal(head.velocity_id[1:128], torch.arange(1, 128))
    pair = head.path_id[1:] * 127 + head.velocity_id[1:] - 1
    assert torch.equal(pair, torch.arange(512 * 127))


def test_equal_but_wrong_id_shapes_are_rejected(bank_tensors):
    bank, path, velocity, path_id, velocity_id = bank_tensors
    with pytest.raises(ValueError, match="flattened P/V ids"):
        FactorizedPVHead(bank, path, velocity, path_id[:-1], velocity_id[:-1], residual=False)


def test_pv_and_residual_have_identical_scorer_rng_and_step_zero_output(bank_tensors):
    torch.manual_seed(41)
    before = torch.get_rng_state().clone()
    pv = make_head(bank_tensors, residual=False)
    after_pv = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    hybrid = make_head(bank_tensors, residual=True)
    after_hybrid = torch.get_rng_state().clone()
    pv_state = pv.state_dict()
    hybrid_state = hybrid.state_dict()
    shared = [name for name in pv_state if name in hybrid_state]
    assert shared and all(torch.equal(pv_state[name], hybrid_state[name]) for name in shared)
    assert not torch.equal(after_pv, after_hybrid)  # residual parameters consume RNG internally

    decoded = torch.randn(2, 6, 128)
    a, b = pv(decoded), hybrid(decoded)
    assert torch.equal(a["pv_logits"], b["pv_logits"])
    assert torch.equal(a["pv_selected_id"], b["pv_selected_id"])
    assert torch.equal(a["pv_post_plan"], b["pv_post_plan"])
    assert torch.count_nonzero(b["pv_candidates"] - b["pv_base_candidates"]) == 0


def test_make_model_restores_cpu_rng_after_optional_head(monkeypatch, bank_tensors):
    class Base(nn.Module):
        def __init__(self, _config):
            super().__init__()
            self.planner = nn.Module()
            self.planner.xy_head = nn.Linear(128, 2)

        def forward(self, *args, **kwargs):
            raise NotImplementedError

    monkeypatch.setattr("models.motiondrive_v2.model.MotionDriveV2", Base)
    monkeypatch.setattr(run, "install_shared_status_query", lambda model: model)
    arrays = {"bank_xy": bank_tensors[0].new_zeros((K_COUNT, 10, 2)).numpy(),
              "path_id": bank_tensors[3].numpy(), "velocity_id": bank_tensors[4].numpy()}
    bank = {"path_descriptor": bank_tensors[1], "velocity_descriptor": bank_tensors[2]}
    torch.manual_seed(91)
    _ = run.make_model(object(), "direct", arrays, bank)
    direct_rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    _ = run.make_model(object(), "pv_residual", arrays, bank)
    assert torch.equal(torch.get_rng_state(), direct_rng)


def test_stable_first_argmax_and_exact_bank_row(bank_tensors):
    head = make_head(bank_tensors)
    for parameter in head.context.parameters():
        nn.init.zeros_(parameter)
    result = head(torch.randn(3, 6, 128))
    assert result["pv_selected_id"].tolist() == [0, 0, 0]
    assert torch.equal(result["pv_post_plan"], torch.zeros(3, 6, 2))


def test_residual_is_bounded_completed_before_selection_and_has_expected_gradients(bank_tensors):
    torch.manual_seed(3)
    head = make_head(bank_tensors, residual=True)
    decoded = torch.randn(1, 6, 128, requires_grad=True)
    first = head(decoded)
    first["pv_candidates"][:, 1:5].sum().backward()
    assert head.residual_candidate.weight.grad is not None
    assert torch.isfinite(head.residual_candidate.weight.grad).all()
    assert torch.count_nonzero(head.residual_candidate.weight.grad)
    assert (head.residual_context.weight.grad is None
            or not torch.count_nonzero(head.residual_context.weight.grad))
    with torch.no_grad():
        head.residual_candidate.weight.add_(.01 * head.residual_candidate.weight.grad)
    head.zero_grad(set_to_none=True)
    changed = head(decoded.detach())
    delta = changed["pv_candidates"] - changed["pv_base_candidates"]
    assert float(delta.detach().abs().max()) <= .5
    chosen = changed["pv_selected_id"]
    assert torch.equal(changed["pv_pre_plan"], changed["pv_base_candidates"][chosen])
    assert torch.equal(changed["pv_post_plan"], changed["pv_candidates"][0, chosen])
    changed["pv_post_plan"].sum().backward()
    assert torch.count_nonzero(head.residual_context.weight.grad)


def test_candidate_loss_replaces_direct_plan_term_and_keeps_other_losses():
    marker = torch.tensor(4.)

    def original(outputs, batch, weights, **kwargs):
        assert weights.plan == 0
        return marker, {"plan_d3": torch.tensor(99.), "aux": marker, "total": marker}

    outputs, batch = tiny_loss_case()
    weights = type("Weights", (), {})()
    # dataclasses.replace is intentionally required by the production wrapper.
    import dataclasses
    @dataclasses.dataclass
    class W:
        plan: float = 1.
    expected, _ = run.pv_plan_objective(outputs, batch)
    total, parts = run.candidate_compute_loss(original, outputs, batch, W())
    assert torch.equal(total, marker + expected)
    assert torch.equal(parts["aux"], marker)
    assert torch.equal(parts["plan_d3"], torch.tensor(99.))


def tiny_loss_case(batch_size=2):
    base = torch.zeros(K_COUNT, 6, 2)
    base[1, :, 0] = 1.
    candidates = base[None].expand(batch_size, -1, -1, -1).clone()
    logits = torch.zeros(batch_size, K_COUNT, requires_grad=True)
    target = torch.zeros(batch_size, 6, 2)
    valid = torch.ones(batch_size, 6, dtype=torch.bool)
    return {"pv_logits": logits, "pv_base_candidates": base,
            "pv_candidates": candidates}, {"gt_plan": target, "plan_valid": valid}


def test_expected_d3_soft_ce_fixed_base_labels_and_invalid_rows():
    outputs, batch = tiny_loss_case(3)
    batch["plan_valid"][1, 5] = False
    batch["gt_plan"][1] = torch.nan
    first, parts = run.pv_plan_objective(outputs, batch, {"plan_complete": torch.tensor(2)})
    altered = dict(outputs)
    altered["pv_candidates"] = outputs["pv_candidates"].clone()
    altered["pv_candidates"][:, 2, :, 1] += .2
    second, changed = run.pv_plan_objective(altered, batch, {"plan_complete": torch.tensor(2)})
    assert torch.isfinite(first) and torch.isfinite(second)
    assert torch.equal(parts["pv_soft_ce"], changed["pv_soft_ce"])
    assert not torch.equal(parts["pv_expected_d3"], changed["pv_expected_d3"])


def test_full_batch_normalizer_matches_uneven_microbatches_and_logits_gradient():
    full_outputs, full_batch = tiny_loss_case(5)
    full_batch["plan_valid"][1, -1] = False
    full_batch["plan_valid"][4, -1] = False
    full_batch["gt_plan"][[1, 4]] = torch.nan
    denominator = {"plan_complete": torch.tensor(3)}
    full, _ = run.pv_plan_objective(full_outputs, full_batch, denominator)
    full.backward()
    full_grad = full_outputs["pv_logits"].grad.clone()

    split_logits = full_outputs["pv_logits"].detach().clone().requires_grad_()
    total = torch.zeros(())
    for start, end in ((0, 3), (3, 5)):
        outputs = {"pv_logits": split_logits[start:end],
                   "pv_base_candidates": full_outputs["pv_base_candidates"],
                   "pv_candidates": full_outputs["pv_candidates"][start:end]}
        batch = {name: value[start:end] for name, value in full_batch.items()}
        value, _ = run.pv_plan_objective(outputs, batch, denominator)
        total = total + value
    total.backward()
    assert torch.allclose(total, full, atol=2e-6, rtol=2e-6)
    assert torch.allclose(split_logits.grad, full_grad, atol=2e-6, rtol=2e-6)
    assert not torch.count_nonzero(split_logits.grad[[1, 4]])


def test_tensor_memory_envelope_is_bounded():
    f32 = 4
    microbatch = 2
    joint_keys = K_COUNT * 32 * f32
    completed_candidates = microbatch * K_COUNT * 6 * 2 * f32
    coordinate_difference = completed_candidates
    waypoint_distance = microbatch * K_COUNT * 6 * f32
    logits = microbatch * K_COUNT * f32
    assert joint_keys == 8_323_200
    assert completed_candidates == 6_242_400
    assert joint_keys + 2 * completed_candidates + waypoint_distance + logits < 32 << 20


class ToyParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.planner = nn.Module()
        self.planner.xy_head = nn.Linear(128, 2)

    def forward(self, decoded, fail=False):
        if fail:
            self.planner.xy_head(decoded)
            raise RuntimeError("parent failure")
        plan = self.planner.xy_head(decoded)
        return {"plan_abs": plan, "state_hat": decoded.mean((1, 2))[:, None]}


def test_install_preserves_parent_schema_inputs_and_clears_capture(bank_tensors):
    model = ToyParent()
    head = make_head(bank_tensors)
    install_factorized_pv_head(model, head)
    decoded = torch.randn(1, 6, 128)
    result = model(decoded)
    assert result["plan_abs"].shape == (1, 6, 2)
    assert "state_hat" in result and result["pv_logits"].shape == (1, K_COUNT)
    assert model._factorized_pv_capture == []
    with pytest.raises(RuntimeError, match="parent failure"):
        model(decoded, fail=True)
    assert model._factorized_pv_capture == []
    assert list(head.forward.__signature__.parameters) == ["decoded"] if hasattr(
        head.forward, "__signature__") else True


def test_terminal_capture_uses_same_evaluation_pass_and_record_rows(monkeypatch):
    import train_motiondrive_v2 as trainer
    import evaluate_motiondrive_v2_planning as planning_eval

    class CaptureModel:
        training = False

    model = CaptureModel()
    calls = {"evaluate": 0}

    def fake_inputs(batch, time_input="raw", nominal_history_seconds=(.1, .2, .5, 1.)):
        return {"provided_status5": batch["provided_status5"]}

    def fake_evaluate(received, *_args, **_kwargs):
        calls["evaluate"] += 1
        batch = {"row": torch.tensor([7, 9]), "provided_status5": torch.zeros(2, 5)}
        trainer.model_inputs(batch)
        received._pv_eval_capture.append({
            "pv_selected_id": torch.tensor([2, 3]),
            "pv_pre_plan": torch.zeros(2, 6, 2),
            "pv_post_plan": torch.ones(2, 6, 2),
            "pv_entropy": torch.tensor([.2, .3]),
        })
        return {"official_d3": 1.}, [{"row": 7, "d3": .4}, {"row": 9, "d3": .5}]

    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)
    monkeypatch.setattr(trainer, "model_inputs", fake_inputs)
    monkeypatch.setattr(planning_eval, "planning_model_inputs",
                        lambda batch, time_input="raw": {"provided_status5": batch["provided_status5"]})
    entered = (trainer.evaluate, trainer.model_inputs, planning_eval.planning_model_inputs)
    arrays = overlay = object()
    bank = {"oracle_rows": torch.tensor([7, 9]),
            "oracle_selected_id": torch.tensor([1, 4]),
            "oracle_d3": torch.tensor([.1, .2])}
    with run.patched_runtime("pv", arrays, bank, overlay, "parent", [], "selector", None):
        _report, records = trainer.evaluate(model)
        assert [row["pv_selected_id"] for row in records] == [2, 3]
        assert [row["pv_oracle_selected_id"] for row in records] == [1, 4]
        assert [row["pv_oracle_d3"] for row in records] == pytest.approx([.1, .2])
        assert calls["evaluate"] == 1
    assert (trainer.evaluate, trainer.model_inputs, planning_eval.planning_model_inputs) == entered


def test_patched_runtime_restores_bindings_on_exception(monkeypatch):
    import train_motiondrive_v2 as trainer
    import models.motiondrive_v2 as model_api
    import motiondrive_v2_data as data_api
    before = (model_api.MotionDriveV2, data_api.MotionDriveDataset, trainer.compute_loss,
              trainer.evaluate)
    with pytest.raises(RuntimeError, match="boom"):
        with run.patched_runtime("direct", object(), object(), object(), "x", [], None, None):
            raise RuntimeError("boom")
    assert (model_api.MotionDriveV2, data_api.MotionDriveDataset, trainer.compute_loss,
            trainer.evaluate) == before


def test_arm_names_and_interpretation_contract():
    assert run.ARMS == ("direct", "pv", "pv_residual")
    assert set(run.GPU_ASSIGNMENTS) == set(run.ARMS)
    assert "provided_causal_5d" in Path(run.__file__).read_text()
    assert "single-seed system/trainability screen" in Path(run.__file__).read_text()
