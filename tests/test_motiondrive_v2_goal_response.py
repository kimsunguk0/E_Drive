import copy

import pytest
import torch

from models.motiondrive_v2_input_contract import INPUT_KEYS
from scripts import analyze_motiondrive_v2_goal_response as goal


@pytest.fixture(autouse=True)
def cpu_only_unit_test():
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def goals():
    return torch.tensor([[10., 0.], [9., 0.], [12., 0.], [10., 2.],
                         [15., -1.], [20., 3.], [7., -2.], [25., 1.]],
                        dtype=torch.float32)


def tiny_inputs(index):
    return {"images": torch.full((1, 6, 3, 2, 3), float(index)),
            "history_images": torch.full((1, 4, 3, 1, 2), float(index + 10)),
            "lidar2img": torch.full((1, 6, 4, 4), float(index + 20)),
            "history_transforms": torch.full((1, 4, 4, 4), float(index + 30)),
            "time_offsets": torch.tensor([[.1, .2, .5, 1.]], dtype=torch.float32),
            "goal_xy": goals()[index:index + 1].clone()}


def tracked_outputs(value=0.):
    return {"scene_features": torch.full((1, 3, 4), value),
            "occ_logits": torch.full((1, 1, 2, 2), value),
            "lane_logits": torch.full((1, 1, 2, 2), value),
            "plan_abs": torch.full((1, 6, 2), value),
            "motion_features": torch.full((1, 2, 4), value),
            "history_hat": torch.full((1, 4, 4), value),
            "state_hat": torch.full((1, 6), value)}


def test_condition_grid_is_exactly_96_unique_train8_entries():
    plan = goal.condition_plan()
    assert len(plan) == 96 and len(set(plan)) == 96
    assert {row[0] for row in plan} == set(goal.CONDITIONS)
    assert {row[2] for row in plan} == set(goal.GOAL_ROLES)
    with pytest.raises(ValueError, match="train8"):
        goal.condition_plan(4)


def test_donor_rule_is_deterministic_distinct_and_reports_no_receiver_feasibility():
    ids = [f"clip_{index}" for index in range(8)]
    first = goal.select_goal_donors(goals(), ids)
    second = goal.select_goal_donors(goals().clone(), list(ids))
    assert first == second
    for receiver, row in enumerate(first):
        indices = [donor["index"] for donor in row["donors"]]
        assert len(indices) == len(set(indices)) == 2 and receiver not in indices
        assert row["receiver_road_feasibility_guaranteed"] is False
        assert all(donor["combined_score_m"] == pytest.approx(
            donor["norm_delta_m"] + donor["forward_x_delta_m"])
                   for donor in row["donors"])


def test_donor_ties_use_clip_id_then_index():
    same = torch.zeros(8, 2, dtype=torch.float32)
    ids = ["z", "h", "g", "f", "e", "d", "c", "b"]
    selected = goal.select_goal_donors(same, ids)
    assert [row["clip_id"] for row in selected[0]["donors"]] == ["b", "c"]


@pytest.mark.parametrize("change", ["shape", "nonfinite", "dtype", "duplicate_id"])
def test_bad_goal_selection_input_is_rejected(change):
    value, ids = goals(), [f"clip_{index}" for index in range(8)]
    if change == "shape":
        value = value[:7]
    elif change == "nonfinite":
        value[0, 0] = float("nan")
    elif change == "dtype":
        value = value.double()
    else:
        ids[1] = ids[0]
    with pytest.raises(ValueError):
        goal.select_goal_donors(value, ids)


def test_goal_variants_preserve_observed_values_and_make_explicit_zero():
    values = goals()
    selection = goal.select_goal_donors(values, [str(i) for i in range(8)])[0]
    variants = goal.goal_variants(values, selection, 0)
    assert set(variants) == set(goal.GOAL_ROLES)
    assert torch.equal(variants["own"], values[0:1])
    assert torch.equal(variants["donor_1"], values[selection["donors"][0]["index"]:
                                                        selection["donors"][0]["index"] + 1])
    assert torch.count_nonzero(variants["zero"]) == 0


@pytest.mark.parametrize("condition", goal.CONDITIONS)
def test_condition_inputs_change_only_declared_fields_without_mutation(condition):
    inputs = [tiny_inputs(index) for index in range(8)]
    before = copy.deepcopy(inputs)
    new_goal = torch.tensor([[17., -3.]], dtype=torch.float32)
    result, info = goal.make_condition_inputs(inputs, 2, condition, new_goal)
    assert set(result) == set(INPUT_KEYS) and torch.equal(result["goal_xy"], new_goal)
    assert all(torch.equal(inputs[i][key], before[i][key]) for i in range(8)
               for key in INPUT_KEYS)
    for name in ("lidar2img", "history_transforms", "time_offsets"):
        assert result[name] is inputs[2][name]
    if condition == "normal":
        assert result["images"] is inputs[2]["images"]
        assert info["image_source_index"] == 2 and not info["negative_control"]
    elif condition == "normalized_zero":
        assert torch.count_nonzero(result["images"]) == 0
        assert torch.count_nonzero(result["history_images"]) == 0
        assert info["image_source_index"] is None and info["negative_control"]
    else:
        assert result["images"] is inputs[3]["images"]
        assert result["history_images"] is inputs[3]["history_images"]
        assert info["image_source_index"] == 3 and info["negative_control"]


def test_tensor_delta_reports_scene_cells_and_plan_waypoints_without_a_gate():
    before = tracked_outputs()
    after = tracked_outputs()
    after["scene_features"][0, 0, 0] = 3.
    after["plan_abs"][0, 2] = torch.tensor([3., 4.])
    scene = goal.tensor_delta(after["scene_features"], before["scene_features"],
                              "scene_features")
    plan = goal.tensor_delta(after["plan_abs"], before["plan_abs"], "plan_abs")
    assert scene["cell_l2_max"] == 3.
    assert plan["waypoint_l2"][2] == 5. and plan["waypoint_l2_max"] == 5.
    assert not plan["bitwise_equal"]


def test_goal_response_allows_zero_or_nonzero_descriptive_outputs_but_gates_invariants():
    baseline = tracked_outputs()
    same = goal.analyze_goal_variant(tracked_outputs(), baseline)
    assert same["goal_free_all_bitwise"]
    changed = tracked_outputs()
    changed["scene_features"] += 1
    changed["occ_logits"] += 2
    changed["lane_logits"] -= 3
    changed["plan_abs"] += 4
    report = goal.analyze_goal_variant(changed, baseline)
    assert report["goal_free_all_bitwise"]
    assert report["goal_response"]["plan_abs"]["max_abs"] == 4.
    changed["state_hat"][0, 0] = 1.
    with pytest.raises(ValueError, match="goal-free"):
        goal.analyze_goal_variant(changed, baseline)


def test_repeat_requires_every_tracked_output_bitwise_equal():
    first = tracked_outputs()
    assert goal.validate_repeat(first, copy.deepcopy(first))["all_tracked_outputs_bitwise"]
    repeated = copy.deepcopy(first)
    repeated["motion_features"][0, 0, 0] = 1e-8
    with pytest.raises(ValueError, match="repeat"):
        goal.validate_repeat(first, repeated)


def test_summary_separates_three_controls_and_excludes_own_goal():
    records = []
    baseline = tracked_outputs()
    for condition, receiver, role in goal.condition_plan():
        current = tracked_outputs()
        if role != "own":
            current["scene_features"] += receiver + 1
            current["occ_logits"] += 1
            current["lane_logits"] += 2
            current["plan_abs"] += 3
        records.append({"condition": condition, "receiver_index": receiver,
                        "goal_role": role,
                        "analysis": goal.analyze_goal_variant(current, baseline)})
    summary = goal.summarize_records(records)
    assert set(summary) == set(goal.CONDITIONS)
    assert all(summary[condition][name]["n"] == 24 for condition in goal.CONDITIONS
               for name in goal.GOAL_RESPONSE_OUTPUTS)


def test_script_text_keeps_actual_result_pending_until_gpu_execution():
    source = goal.Path(goal.__file__).read_text()
    assert "no pass threshold" in source
    assert "not_compliance_certificate" in source
    assert "not_accuracy_evaluation" in source
    assert "actual_gpu_diagnostic_completed\": True" in source
    assert "96" not in source or "The 96 unique requested conditions" in source
