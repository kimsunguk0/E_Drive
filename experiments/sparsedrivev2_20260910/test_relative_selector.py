"""CPU contracts for fixed-candidate score-only relative selection."""
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn

SPEC = importlib.util.spec_from_file_location("relative_selector", Path(__file__).with_name("relative_selector.py"))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
make_candidate_features = module.make_candidate_features
RelativeScoreHead = module.RelativeScoreHead
CandidateRelativeSelector = module.CandidateRelativeSelector


def fixture():
    times = torch.arange(1, 7).float() * .5
    xy = torch.zeros(2, 3, 6, 2)
    xy[..., 0] = torch.tensor([2., 4., 6.])[None, :, None] * times
    xy[1, ..., 1] = times
    scores = torch.tensor([[3., 2., 1.], [1., 3., 2.]])
    ids = torch.tensor([[31, 41, 59], [17, 23, 71]])
    selected = scores.argmax(-1)
    batch = torch.arange(2)
    return {"candidate_xy": xy, "scores": scores, "candidate_ids": ids,
            "candidate_valid": torch.ones(2, 3, dtype=torch.bool),
            "trajectory": xy[batch, selected], "selected_candidate_id": ids[batch, selected],
            "coarse": [{"path_ids": torch.tensor([2, 5])}],
            "imitation_scores": scores.clone()}, torch.zeros(2, 8)


class FrozenFixture(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor(1.))
        self.output = output
        self.received = None
        self.grad_enabled = None

    def forward(self, **inputs):
        self.received = inputs
        self.grad_enabled = torch.is_grad_enabled()
        return self.output


class RelativeSelectorContracts(unittest.TestCase):
    def test_feature_order_physical_units_and_no_future_state(self):
        output, status = fixture()
        status[:, :4] = float("nan")  # Future or other slots must never be read.
        status[:, 4:] = torch.tensor([1., -.5, .6, -.9])
        # Row 0 candidate 0 has interval velocities x=[2,4,6,8,10,12], y=0.
        output["candidate_xy"][0, 0, :, 0] = torch.tensor([1., 3., 6., 10., 15., 21.])
        goal = torch.tensor([[50., -25.], [10., 20.]])
        features = make_candidate_features(output, status, goal)
        expected = torch.tensor([v for x in [1., 3., 5., 7., 9., 11.] for v in [x, .5]]
                                + [v for _ in range(5) for v in [4./3., 0.]]
                                + [.05, -.025, .2, -.3, 1., -.5, -.58, .5, 1., 1.])
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(tuple(features.shape), (2, 3, 32))
        torch.testing.assert_close(features[0, 0], expected)
        self.assertEqual(len(module.FEATURE_NAMES), 32)
        no_goal = make_candidate_features(output, status)
        self.assertTrue(torch.equal(no_goal[..., 26:30], torch.zeros(2, 3, 4)))
        self.assertTrue(torch.equal(no_goal[..., 31], torch.zeros(2, 3)))

    def test_zero_init_preserves_every_original_fp32_field(self):
        output, status = fixture()
        wrapper = CandidateRelativeSelector(FrozenFixture(output))
        result = wrapper.rescore(output, status, torch.ones(2, 2))
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(result[key], value), key)
            else:
                self.assertIs(result[key], value)
        self.assertIs(result["candidate_xy"], output["candidate_xy"])
        self.assertIs(result["candidate_ids"], output["candidate_ids"])
        self.assertEqual(int(torch.count_nonzero(result["relative_score_residual"])), 0)
        self.assertGreater(int(torch.count_nonzero(wrapper.relative_head.mlp[0].weight)), 0)

    def test_final_layer_learns_and_hidden_layers_are_not_dead(self):
        torch.manual_seed(21)
        head = RelativeScoreHead()
        features = torch.randn(3, 5, 32)
        target = torch.randn(3, 5)
        (head(features) - target).square().mean().backward()
        self.assertGreater(float(head.mlp[-1].weight.grad.abs().sum()), 0.)
        self.assertEqual(float(head.mlp[0].weight.grad.abs().sum()), 0.)
        # This is the expected first-step zero-final-layer gradient gate. Once
        # final weights are nonzero, ordinary hidden initialization propagates.
        head.zero_grad(set_to_none=True)
        with torch.no_grad():
            head.mlp[-1].weight.fill_(.1)
        (head(features) - target).square().mean().backward()
        self.assertGreater(float(head.mlp[0].weight.grad.abs().sum()), 0.)

    def test_head_conditions_cannot_change_candidate_ids_or_coordinates(self):
        output, status = fixture()
        wrapper = CandidateRelativeSelector(FrozenFixture(output))
        changed_status = status.clone()
        changed_status[:, 4:8] = 3.
        before = make_candidate_features(output, status)
        after = make_candidate_features(output, changed_status, torch.full((2, 2), 10.))
        self.assertFalse(torch.equal(before, after))
        for state, goal in [(status, None), (changed_status, torch.full((2, 2), 10.))]:
            result = wrapper.rescore(output, state, goal)
            self.assertIs(result["candidate_xy"], output["candidate_xy"])
            self.assertIs(result["candidate_ids"], output["candidate_ids"])

    def test_nonzero_residual_reselects_an_existing_row(self):
        output, status = fixture()
        wrapper = CandidateRelativeSelector(FrozenFixture(output))
        with torch.no_grad():
            for parameter in wrapper.relative_head.parameters():
                parameter.zero_()
            wrapper.relative_head.mlp[0].weight[0, 0] = 1.
            wrapper.relative_head.mlp[2].weight[0, 0] = 1.
            wrapper.relative_head.mlp[4].weight[0, 0] = 10.
        result = wrapper.rescore(output, status)
        self.assertFalse(torch.equal(result["selected_candidate_id"], output["selected_candidate_id"]))
        self.assertTrue(torch.equal(result["selected_candidate_id"], output["candidate_ids"][:, 2]))
        self.assertTrue(torch.equal(result["trajectory"], output["candidate_xy"][:, 2]))

    def test_cpu_autocast_keeps_features_head_and_scores_fp32(self):
        output, status = fixture()
        output["scores"] = output["scores"].bfloat16()
        wrapper = CandidateRelativeSelector(FrozenFixture(output))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            features = make_candidate_features(output, status)
            residual = wrapper.relative_head(features.bfloat16())
            result = wrapper.rescore(output, status)
        for tensor in (features, residual, result["scores"], result["relative_score_residual"]):
            self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.equal(result["scores"], output["scores"].float()))

    def test_goal_forward_policy_and_frozen_base_mode(self):
        output, status = fixture()
        goal = torch.ones(2, 2)
        for mode in ("none", "selection"):
            base = FrozenFixture(output)
            wrapper = CandidateRelativeSelector(base, base_goal_mode=mode)
            wrapper.train()
            self.assertTrue(wrapper.relative_head.training)
            self.assertFalse(base.training)
            self.assertFalse(base.parameter.requires_grad)
            wrapper(images=torch.zeros(2, 1), lidar2img=None, status=status, goal_xy=goal)
            self.assertFalse(base.grad_enabled)
            self.assertEqual("goal_xy" in base.received, mode == "selection")
            if mode == "selection":
                self.assertIs(base.received["goal_xy"], goal)
            self.assertIs(base.received["status"], status)

    def test_invalid_candidates_do_not_enter_features_mean_or_selection(self):
        output, status = fixture()
        output["candidate_valid"][:, 2] = False
        output["candidate_xy"][:, 2] = float("nan")
        output["scores"][:, 2] = 1000.
        features = make_candidate_features(output, status)
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.equal(features[:, 2], torch.zeros(2, 32)))
        torch.testing.assert_close(features[:, :2, 30], torch.tensor([[.5, -.5], [-1., 1.]]))
        wrapper = CandidateRelativeSelector(FrozenFixture(output))
        result = wrapper.rescore(output, status)
        self.assertTrue(torch.equal(result["selected_candidate_id"], torch.tensor([31, 23])))
        output["candidate_valid"].fill_(False)
        with self.assertRaisesRegex(ValueError, "valid candidate per row"):
            make_candidate_features(output, status)

    def test_standalone_head_state_dict_roundtrip_and_feature_shape(self):
        first = RelativeScoreHead()
        with torch.no_grad():
            first.mlp[-1].weight.fill_(.2)
        saved = {"relative_head": first.state_dict(), "manifest": {"features": module.FEATURE_VERSION}}
        second = RelativeScoreHead()
        second.load_state_dict(saved["relative_head"], strict=True)
        features = torch.randn(2, 3, 32)
        self.assertTrue(torch.equal(first(features), second(features)))
        with self.assertRaisesRegex(ValueError, "features must be"):
            second(torch.zeros(2, 31))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
