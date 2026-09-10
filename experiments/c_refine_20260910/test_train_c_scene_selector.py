import copy

import torch

from c_scene_selector import SceneResidualSelector
from train_c_scene_selector import cost_loss


def sample():
    torch.manual_seed(123)
    return dict(candidate_xy=torch.randn(2, 7, 6, 2),
                candidate_ids=torch.arange(14).view(2, 7),
                candidate_valid=torch.tensor([[1, 1, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool),
                scores=torch.randn(2, 7), candidate_tokens=torch.randn(2, 7, 256))


def test_invalid_cost_and_score_do_not_change_valid_gradients():
    x = torch.randn(2, 7, requires_grad=True)
    valid = sample()['candidate_valid']
    costs = torch.rand(2, 7)
    loss, _, _ = cost_loss(x, costs, valid, .1)
    gradient, = torch.autograd.grad(loss, x)
    perturbed = x.detach().clone().masked_fill(~valid, float('nan')).requires_grad_()
    bad_costs = costs.masked_fill(~valid, float('nan'))
    other, _, _ = cost_loss(perturbed, bad_costs, valid, .1)
    other_grad, = torch.autograd.grad(other, perturbed)
    torch.testing.assert_close(loss, other, rtol=0, atol=0)
    torch.testing.assert_close(gradient, other_grad, rtol=0, atol=0)
    assert torch.count_nonzero(gradient[~valid]) == 0


def test_matched_heads_start_same_and_labels_enter_only_loss():
    out = sample()
    torch.manual_seed(0)
    real = SceneResidualSelector(mode='real')
    torch.manual_seed(0)
    zero = SceneResidualSelector(mode='zero')
    for key, value in real.state_dict().items():
        assert torch.equal(value, zero.state_dict()[key])
    goal = torch.randn(2, 2)
    rp, zp = real(out, goal), zero(out, goal)
    assert torch.equal(rp['scores'], zp['scores'])
    assert torch.equal(rp['scores'], out['scores'])
    costs = torch.rand(2, 7)
    optimizer = torch.optim.AdamW(real.parameters(), lr=.001)
    before = copy.deepcopy(real.state_dict())
    loss, _, _ = cost_loss(rp['scores'], costs, out['candidate_valid'], .1)
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(before[k], v) for k, v in real.state_dict().items())
    # Frozen upstream cached fields are never edited by fitting a head.
    assert torch.equal(rp['candidate_xy'], out['candidate_xy'])
    assert torch.equal(rp['candidate_ids'], out['candidate_ids'])

