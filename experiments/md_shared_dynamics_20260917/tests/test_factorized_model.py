from __future__ import annotations

import torch

from factorized_model import (SharedStatusFeatureConditioner,
                              compose_factorized_plan)


def _straight(batch=2):
    x = torch.arange(1, 7, dtype=torch.float32)[None, :, None].repeat(batch, 1, 1)
    return torch.cat([x, torch.zeros_like(x)], -1)


def test_zero_coefficients_preserve_forward_values():
    plan = _straight().requires_grad_(True)
    final, info = compose_factorized_plan(plan, torch.zeros(2, 1, requires_grad=True))
    torch.testing.assert_close(final, plan, atol=1e-7, rtol=0)
    torch.testing.assert_close(info["progress_segment_length"],
                               info["base_segment_length"], atol=0, rtol=0)


def test_progress_gradient_does_not_take_direct_length_path():
    plan = _straight(1).requires_grad_(True)
    coefficient = torch.zeros(1, 1, requires_grad=True)
    final, _ = compose_factorized_plan(plan, coefficient)
    final[..., 0].sum().backward()
    assert coefficient.grad is not None and coefficient.grad.abs().sum() > 0
    # A straight proposal's x scale cannot reduce longitudinal loss directly;
    # it must go through the dedicated progress coefficient.
    torch.testing.assert_close(plan.grad[..., 0], torch.zeros_like(plan.grad[..., 0]),
                               atol=1e-6, rtol=0)


def test_two_coefficient_progress_definition():
    plan = _straight(1)
    coefficient = torch.tensor([[0.2, -0.1]])
    final, info = compose_factorized_plan(plan, coefficient)
    midpoint = torch.tensor([.25, .75, 1.25, 1.75, 2.25, 2.75])
    expected = 1.0 + .5 * (.2 - .1 * midpoint)
    torch.testing.assert_close(info["progress_segment_length"][0], expected)
    torch.testing.assert_close(final[0, :, 0], expected.cumsum(0))


def test_stationary_fallback_can_depart_and_never_reverses():
    stopped = torch.zeros(1, 6, 2)
    forward, _ = compose_factorized_plan(stopped, torch.tensor([[0.5]]))
    assert torch.all(forward[0, :, 0] > 0) and torch.count_nonzero(forward[..., 1]) == 0
    reverse, info = compose_factorized_plan(stopped, torch.tensor([[-1.0]]))
    assert torch.count_nonzero(reverse) == 0
    assert torch.all(info["progress_segment_length"] >= 0)


def test_shared_conditioner_is_exact_identity_at_initialization():
    module = SharedStatusFeatureConditioner(128)
    for status in (torch.zeros(3, 5), torch.randn(3, 5)):
        gates = module(status)
        assert len(gates) == 3
        for gate in gates:
            torch.testing.assert_close(gate, torch.ones_like(gate), atol=0, rtol=0)
