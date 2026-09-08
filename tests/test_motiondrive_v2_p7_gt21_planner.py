import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import diagnose_motiondrive_v2_p7_gt21_planner as diagnostic


def tensors(batch=3):
    predicted_state = torch.randn(batch, 6)
    predicted_history = torch.randn(batch, 4, 4)
    gt_state = torch.randn(batch, 6)
    gt_history = torch.randn(batch, 4, 4)
    return predicted_state, predicted_history, gt_state, gt_history


def test_gt21_replaces_exact_fields_and_preserves_raw_stop_logit():
    pred_s, pred_h, gt_s, gt_h = tensors()
    state, history = diagnostic.gt21_planner_inputs(
        pred_s, pred_h, gt_s, gt_h, torch.ones_like(gt_s, dtype=torch.bool),
        torch.ones_like(gt_h, dtype=torch.bool))
    assert torch.equal(state[:, :5], gt_s[:, :5])
    assert torch.equal(state[:, 5], pred_s[:, 5])
    assert torch.equal(history, gt_h)
    assert not torch.equal(state[:, 5], gt_s[:, 5])


def test_gt21_rejects_invalid_injected_coordinate():
    pred_s, pred_h, gt_s, gt_h = tensors()
    valid = torch.ones_like(gt_s, dtype=torch.bool); valid[0, 2] = False
    with pytest.raises(ValueError, match="five injected"):
        diagnostic.gt21_planner_inputs(pred_s, pred_h, gt_s, gt_h, valid,
                                       torch.ones_like(gt_h, dtype=torch.bool))


class RecordingModel:
    def __init__(self):
        self.calls = []
    def plan_from_features(self, scene, motion, state, history):
        self.calls.append((scene, motion, state.clone(), history.clone()))
        return state[:, None, :2].expand(-1, 6, -1).clone()


def test_paired_planner_reuses_scene_motion_and_does_not_mutate_baseline():
    pred_s, pred_h, gt_s, gt_h = tensors()
    parts = {"scene_features": torch.randn(3, 8, 5), "motion_features": torch.randn(3, 4, 5),
             "state_hat": pred_s.clone(), "history_hat": pred_h.clone()}
    before = copy.deepcopy(parts)
    model = RecordingModel()
    baseline, changed, _, _ = diagnostic.paired_planner_forward(model, parts, gt_s, gt_h)
    assert len(model.calls) == 2
    assert model.calls[0][0] is model.calls[1][0] is parts["scene_features"]
    assert model.calls[0][1] is model.calls[1][1] is parts["motion_features"]
    assert torch.equal(baseline[:, 0], pred_s[:, :2])
    assert torch.equal(changed[:, 0], gt_s[:, :2])
    assert all(torch.equal(parts[key], before[key]) for key in parts)


def test_batch_join_requires_exact_rows_and_labels():
    raw = {"row": torch.tensor([7, 8]), "state_target": torch.zeros(2, 6),
           "history_target": torch.zeros(2, 4, 4), "gt_plan": torch.zeros(2, 6, 2),
           "state_valid": torch.ones(2, 6, dtype=torch.bool),
           "history_valid": torch.ones(2, 4, 4, dtype=torch.bool),
           "frame": torch.tensor([30, 31]), "scenario": ["a", "a"],
           "session_id": ["s", "s"]}
    labels = {"rows": torch.tensor([7, 8]), "state_target": raw["state_target"],
              "history_target": raw["history_target"], "gt_plan": raw["gt_plan"],
              "state_valid": raw["state_valid"], "history_valid": raw["history_valid"],
              "frame": raw["frame"], "scenario": list(raw["scenario"]),
              "session": list(raw["session_id"])}
    predictions = {"rows": torch.tensor([7, 8])}
    assert diagnostic.validate_batch_join(raw, labels, predictions, 0) == slice(0, 2)
    predictions["rows"] = torch.tensor([7, 9])
    with pytest.raises(ValueError, match="row join"):
        diagnostic.validate_batch_join(raw, labels, predictions, 0)


@pytest.mark.parametrize("field", ["scenario", "session", "frame", "state_valid", "history_valid"])
def test_batch_join_rejects_identity_and_mask_tamper(field):
    raw = {"row": torch.tensor([7]), "state_target": torch.zeros(1, 6),
           "history_target": torch.zeros(1, 4, 4), "gt_plan": torch.zeros(1, 6, 2),
           "state_valid": torch.ones(1, 6, dtype=torch.bool),
           "history_valid": torch.ones(1, 4, 4, dtype=torch.bool),
           "frame": torch.tensor([30]), "scenario": ["a"], "session_id": ["s"]}
    labels = {"rows": torch.tensor([7]), "state_target": raw["state_target"],
              "history_target": raw["history_target"], "gt_plan": raw["gt_plan"],
              "state_valid": raw["state_valid"].clone(),
              "history_valid": raw["history_valid"].clone(), "frame": raw["frame"].clone(),
              "scenario": ["a"], "session": ["s"]}
    if field in ("scenario", "session"):
        labels[field][0] = "tampered"
    elif field == "frame":
        labels[field][0] += 1
    else:
        labels[field].reshape(-1)[0] = False
    with pytest.raises(ValueError, match="pinned GT"):
        diagnostic.validate_batch_join(raw, labels, {"rows": torch.tensor([7])}, 0)


def test_paired_session_summary_is_fixed_and_frame_weighted():
    sessions = np.asarray([f"s{i}" for i in range(11) for _ in range(i + 1)])
    baseline = np.zeros(len(sessions), dtype=np.float32)
    gt21 = np.arange(len(sessions), dtype=np.float32) / 100
    result = diagnostic.paired_session_summary(baseline, gt21, sessions)
    assert set(result["per_session"]) == set(sessions)
    totals = sum(item["gt21_minus_baseline_d3_sum"] for item in result["per_session"].values())
    assert totals == pytest.approx(float(np.asarray(gt21, np.float64).sum()))
    assert result["joint_two_seed_bootstrap_deferred_until_both_saved_outputs"] is True


def test_cli_is_tune_only_and_fixed_batch_worker_device():
    args = diagnostic.arguments([
        "--data-root", "d", "--split-manifest", "s", "--supervision-root", "u",
        "--base-seed", "0", "--checkpoint", "c", "--run-manifest", "m",
        "--p7-source-manifest", "p", "--runtime-source-manifest", "r",
        "--tune-report", "t", "--gt-tune", "g", "--pred-tune", "q", "--out", "o"])
    assert (args.batch, args.workers, args.device) == (4, 4, "cuda:0")
    assert "final" not in vars(args)


def test_exact_artifact_pins_and_interpretation_are_closed():
    assert diagnostic.PROBE_SOURCE_SHA256 == "1ddd1fcf2c47e04283d0ee2e789fd255c4d5c7d537c09bc2f944d53884800852"
    assert set(diagnostic.PRED_TUNE) == {0, 1}
    for item in (diagnostic.GT_TUNE, *diagnostic.PRED_TUNE.values()):
        assert set(item) == {"artifact_sha256", "manifest_sha256"}
        assert all(len(value) == 64 for value in item.values())
    source = open(diagnostic.__file__).read()
    assert '"script_sha256": probe.sha256(__file__)' in source
    assert '"executed_source_sha256": executed_source' in source
    assert '"predicted_raw_stop_logit_preserved": True' in source
    assert '"flat_or_worse_is_inconclusive_due_to_joint_input_distribution_shift": True' in source
