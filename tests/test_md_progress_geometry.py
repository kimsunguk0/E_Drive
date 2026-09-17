import torch

from experiments.md_progress_residual_20260917.progress_residual import (
    apply_progress_correction,
    stable_predicted_directions,
)


def test_zero_is_exact_and_keeps_base_gradient():
    base = torch.tensor([[[1., .1], [2., .2], [3., .3], [4., .4], [5., .5], [6., .6]]],
                        requires_grad=True)
    final, _ = apply_progress_correction(base, torch.zeros(1, 1))
    assert torch.equal(final, base)
    final.sum().backward()
    assert torch.equal(base.grad, torch.ones_like(base))


def test_straight_scalar_and_nonnegative_stop():
    base = torch.tensor([[[1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 0.], [6., 0.]]])
    faster, _ = apply_progress_correction(base, torch.tensor([[.2]]))
    assert torch.allclose(faster[0, :, 0], torch.arange(1, 7.) + .1 * torch.arange(1, 7.))
    stopped, info = apply_progress_correction(base, torch.tensor([[-3.]]))
    assert (info["base_segment_length"] + info["progress_delta_length"] >= 0).all()
    assert torch.equal(stopped, torch.zeros_like(stopped))


def test_all_zero_can_start_forward():
    base = torch.zeros(1, 6, 2)
    final, _ = apply_progress_correction(base, torch.tensor([[1.]]))
    expected = torch.stack([torch.arange(.5, 3.1, .5), torch.zeros(6)], -1)[None]
    assert torch.equal(final, expected)


def test_curve_and_mirror_geometry_are_equivariant():
    base = torch.tensor([[[1., .1], [2., .4], [2.8, .9], [3.5, 1.6],
                          [4.0, 2.4], [4.3, 3.3]]])
    coefficient = torch.tensor([[.2, -.05]])
    output, _ = apply_progress_correction(base, coefficient)
    mirrored = base.clone(); mirrored[..., 1] *= -1
    mirrored_output, _ = apply_progress_correction(mirrored, coefficient)
    expected = output.clone(); expected[..., 1] *= -1
    assert torch.allclose(mirrored_output, expected, atol=1e-7, rtol=0)


def test_short_segments_use_predicted_neighbour_only():
    base = torch.tensor([[[0., 0.], [0., 0.], [1., 1.], [2., 2.], [2., 2.], [3., 3.]]])
    lengths, directions = stable_predicted_directions(base)
    unit = torch.tensor([2 ** -.5, 2 ** -.5])
    assert torch.allclose(directions[0], unit.expand(6, 2), atol=1e-6)
    assert (lengths[0, [0, 1, 4]] == 0).all()
