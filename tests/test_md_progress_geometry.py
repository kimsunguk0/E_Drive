import torch

from experiments.md_progress_residual_20260917.progress_residual import (
    CausalStatusDataset,
    PROVIDED_STATUS_KEY,
    SharedCausalStatusQuery,
    apply_progress_correction,
    stable_predicted_directions,
)
from experiments.md_progress_residual_20260917.train_residual import (
    wrap_progress_denoise_loss,
    wrap_side_auxiliary_loss,
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


def test_shared_status_query_is_exact_zero_init_then_status_sensitive():
    fusion = SharedCausalStatusQuery(32)
    query = torch.randn(2, 7, 32)
    status = torch.randn(2, 5)
    initial = fusion(query, status)
    assert torch.equal(initial, query)
    with torch.no_grad():
        fusion.status_mlp[-1].weight[0, 0] = 1.
        fusion.status_mlp[0].weight[0].fill_(.25)
        fusion.status_mlp[0].bias[0] = .5
    changed = fusion(query, status)
    zeroed = fusion(query, torch.zeros_like(status))
    assert not torch.equal(changed, zeroed)


def test_causal_status_dataset_exposes_a_separate_valid_input():
    class Base(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return {"state_target": torch.arange(6, dtype=torch.float32),
                    "state_valid": torch.ones(6, dtype=torch.bool)}

        def set_epoch(self, epoch):
            self.epoch = epoch

    item = CausalStatusDataset(Base())[0]
    assert torch.equal(item[PROVIDED_STATUS_KEY], item["state_target"][:5])
    assert item[PROVIDED_STATUS_KEY].data_ptr() != item["state_target"].data_ptr()


def test_side_auxiliary_loss_reaches_state_and_history_predictions():
    def base_loss(outputs, batch, weights, *, normalizers=None, stop_class_weights=None):
        zero = outputs["plan_abs"].sum() * 0
        return zero, {"total": zero}

    state = torch.zeros(2, 6, requires_grad=True)
    history = torch.zeros(2, 3, 4, requires_grad=True)
    outputs = {
        "plan_abs": torch.zeros(2, 6, 2, requires_grad=True),
        "side_state_aux_hat": state,
        "side_history_aux_hat": history,
    }
    batch = {
        "state_target": torch.ones(2, 6),
        "state_valid": torch.ones(2, 6, dtype=torch.bool),
        "history_target": torch.ones(2, 4, 4),
        "history_valid": torch.ones(2, 4, 4, dtype=torch.bool),
    }
    loss, parts = wrap_side_auxiliary_loss(base_loss, .5)(outputs, batch, None)
    loss.backward()
    assert parts["side_motion_aux"] > 0
    assert state.grad is not None and torch.isfinite(state.grad).all() and state.grad.abs().sum() > 0
    assert history.grad is not None and torch.isfinite(history.grad).all() and history.grad.abs().sum() > 0


def test_progress_denoise_loss_reaches_reconstructed_plan_only():
    def base_loss(outputs, batch, weights, *, normalizers=None, stop_class_weights=None):
        zero = outputs["plan_abs"].sum() * 0
        return zero, {"total": zero}

    denoised = torch.ones(2, 6, 2, requires_grad=True)
    base = torch.zeros(2, 6, 2, requires_grad=True)
    outputs = {"plan_abs": base, "plan_base_abs": base,
               "progress_denoise_plan": denoised}
    batch = {"plan_valid": torch.ones(2, 6, dtype=torch.bool)}
    loss, parts = wrap_progress_denoise_loss(base_loss, .5)(outputs, batch, None)
    loss.backward()
    assert parts["progress_denoise"] > 0
    assert denoised.grad is not None and torch.isfinite(denoised.grad).all()
    assert base.grad is not None and torch.equal(base.grad, torch.zeros_like(base))
