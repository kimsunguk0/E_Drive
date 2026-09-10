"""Independent checks for the decision cost and candidate validity contract."""
import importlib.util
from pathlib import Path
import sys

import torch

SOURCE = Path(__file__).resolve().parents[1] / "experiments/sparsedrivev2_20260910"
sys.path.insert(0, str(SOURCE))
from train_cached_selector import scoring_loss


def test_conditional_expected_cost_ranking_differs_from_modal_target():
    # Three possible outcomes share the same observed inputs, with probabilities
    # .40/.35/.25. Actions at 0/5/6 have expected absolute costs 3.25/2.25/2.75.
    actions = torch.tensor([0., 5., 6.], dtype=torch.float64)
    outcomes = torch.tensor([0.]*8 + [5.]*7 + [6.]*5, dtype=torch.float64)
    costs = (outcomes[:, None] - actions).abs()
    valid = torch.ones_like(costs, dtype=torch.bool)
    modal_target = (-costs/.1).softmax(-1).mean(0)
    assert modal_target.argmax().item() == 0
    optimum = (-(costs-costs.mean(-1, keepdim=True))/.1).mean(0).requires_grad_()
    loss = scoring_loss(optimum.expand(len(outcomes), -1), costs, valid, "centered_d3", .1)
    loss.backward()
    assert optimum.argmax().item() == 1
    assert torch.allclose(costs.mean(0), torch.tensor([3.25, 2.25, 2.75], dtype=torch.float64))
    assert optimum.grad.abs().max().item() < 1e-10


def test_common_offsets_do_not_change_the_cost_regression():
    scores = torch.tensor([[.3, -.2, .8], [1., 2., -2.]], dtype=torch.float64)
    costs = torch.tensor([[.1, .3, .2], [.4, .1, .9]], dtype=torch.float64)
    valid = torch.ones_like(scores, dtype=torch.bool)
    original = scoring_loss(scores, costs, valid, "centered_d3", .1)
    shifted = scoring_loss(scores+17, costs+torch.tensor([[3.], [8.]]), valid, "centered_d3", .1)
    assert torch.allclose(original, shifted, atol=1e-10, rtol=0)


def test_invalid_candidate_has_zero_gradient_even_with_extreme_logit():
    for objective in ("soft_ce", "centered_d3"):
        score = torch.tensor([[-20000., -20001., 1e8]], requires_grad=True)
        cost = torch.tensor([[.1, .2, .0]])
        valid = torch.tensor([[True, True, False]])
        loss = scoring_loss(score, cost, valid, objective, .1)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(score.grad).all()
        assert score.grad[0, 2].item() == 0
