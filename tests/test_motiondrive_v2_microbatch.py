"""Full-label denominators preserve loss/gradient sums, not model RNG or BN."""
import copy

import pytest
import torch
import torch.nn.functional as F

from scripts.motiondrive_v2_training import (
    HISTORY_SCALE, STATE_SCALE, LossWeights, build_loss_normalizers, compute_loss, to_device, weighted_d3,
)


def fixture(n=7, empty=False, uncertainty=True):
    g = torch.Generator().manual_seed(318)
    shapes = {"plan_abs": (n, 6, 2), "occ_logits": (n, 1, 3, 4), "lane_logits": (n, 1, 3, 4),
              "history_hat": (n, 4, 4), "state_hat": (n, 6)}
    if uncertainty:
        shapes.update(history_logvar=(n, 4, 4), state_logvar=(n, 5))
    out = {k: torch.randn(v, generator=g).requires_grad_() for k, v in shapes.items()}
    batch = {"gt_plan": torch.randn(n, 6, 2, generator=g), "plan_valid": torch.ones(n, 6, dtype=torch.bool),
             "occ_target": torch.zeros(n, 1, 3, 4), "occ_valid": torch.rand(n, 1, 3, 4, generator=g) > .2,
             "lane_target": torch.ones(n, 1, 3, 4), "lane_valid": torch.rand(n, 1, 3, 4, generator=g) > .5,
             "history_target": torch.randn(n, 4, 4, generator=g),
             "history_valid": torch.rand(n, 4, 1, generator=g) > .4,
             "state_target": torch.randn(n, 6, generator=g), "state_valid": torch.rand(n, 6, generator=g) > .4}
    batch["state_target"][:, 5] = torch.arange(n) % 2
    batch["plan_valid"][::2, 2] = False
    batch["occ_target"][:2] = 1.  # A foreground-only microbatch, then background-only chunks.
    batch["occ_valid"][-1] = False
    batch["lane_valid"][2:4] = False  # An entirely empty chunk, globally foreground exists.
    if empty:
        for name in ("plan_valid", "occ_valid", "lane_valid", "history_valid", "state_valid"):
            batch[name].zero_()
    # Invalid labels may contain NaNs: existing masked losses must still sanitize.
    batch["gt_plan"][~batch["plan_valid"].all(-1)] = float("nan")
    for name in ("occ", "lane", "history", "state"):
        mask = batch[name + "_valid"].expand_as(batch[name + "_target"])
        batch[name + "_target"][~mask] = float("nan")
    return out, batch


def legacy_reference(out, batch, w):
    """Frozen pre-microbatch arithmetic, independent of modified loss helpers."""
    def masklike(mask, value):
        if mask is None: return torch.ones_like(value, dtype=torch.bool)
        while mask.ndim < value.ndim: mask = mask.unsqueeze(-1)
        return mask.bool().expand_as(value)
    def mean(value, mask):
        valid = masklike(mask, value)
        return torch.where(valid, value, torch.zeros_like(value)).sum() / valid.sum().clamp_min(1)
    def raster(logits, target, valid):
        mask = valid.bool()
        safe = torch.where(mask, target.float(), torch.zeros_like(target.float()))
        bce = F.binary_cross_entropy_with_logits(logits.float(), safe, reduction="none")
        pos, neg = mask & (safe >= .5), mask & (safe < .5)
        pp, np = (pos.sum() > 0).float(), (neg.sum() > 0).float()
        return (mean(bce, pos) * pp + mean(bce, neg) * np) / (pp + np).clamp_min(1)
    def regression(pred, target, valid, scale, lv):
        pred = pred.float(); mask = masklike(valid, pred)
        error = (pred - torch.where(mask, target.float(), torch.zeros_like(pred))) / pred.new_tensor(scale)
        if lv is None: loss = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none")
        else:
            lv = lv.float().clamp(-6, 6)
            loss = .5 * (error.square() * (-lv).exp() + lv)
        return mean(loss, mask)
    pred = out["plan_abs"].float()
    complete = batch.get("plan_valid", torch.ones_like(pred[..., 0], dtype=torch.bool)).bool().all(-1)
    d3 = weighted_d3(pred, torch.where(complete[:, None, None], batch["gt_plan"].float(), torch.zeros_like(pred)))
    plan = mean(d3, complete)
    occ = raster(out["occ_logits"], batch["occ_target"], batch["occ_valid"])
    lane = raster(out["lane_logits"], batch["lane_target"], batch["lane_valid"])
    history = regression(out["history_hat"], batch["history_target"], batch.get("history_valid"),
                         HISTORY_SCALE, out.get("history_logvar") if w.uncertainty else None)
    valid = batch.get("state_valid", torch.ones_like(batch["state_target"], dtype=torch.bool))
    state = regression(out["state_hat"][..., :5], batch["state_target"][..., :5], valid[..., :5],
                       STATE_SCALE, out.get("state_logvar") if w.uncertainty else None)
    stop_target = torch.where(valid[..., 5].bool(), batch["state_target"][..., 5].float(), torch.zeros_like(batch["state_target"][..., 5].float()))
    stop = mean(F.binary_cross_entropy_with_logits(out["state_hat"][..., 5].float(), stop_target, reduction="none"), valid[..., 5])
    motion = history + state + .2 * stop
    loss = w.plan * plan + w.occupancy * occ + w.lane * lane + w.motion * motion
    return loss, {"total": loss, "plan_d3": plan, "occ_bce": occ, "lane_bce": lane,
                  "history": history, "state": state, "stop_bce": stop, "motion": motion}


@pytest.mark.parametrize("uncertainty", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_default_is_bitwise_identical_to_frozen_legacy(uncertainty, empty):
    out, batch = fixture(empty=empty, uncertainty=uncertainty)
    w = LossWeights(uncertainty=uncertainty)
    expected, ep = legacy_reference(out, batch, w)
    actual, ap = compute_loss(out, batch, w)
    explicit_none, nparts = compute_loss(out, batch, w, normalizers=None)
    assert torch.equal(actual, expected) and torch.equal(actual, explicit_none)
    assert all(torch.equal(ap[k], ep[k]) and torch.equal(ap[k], nparts[k]) for k in ap)
    old_grad = torch.autograd.grad(expected, tuple(out.values()), retain_graph=True)
    new_grad = torch.autograd.grad(actual, tuple(out.values()))
    assert all(torch.equal(a, b) for a, b in zip(old_grad, new_grad))


@pytest.mark.parametrize("chunk", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("uncertainty", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_chunk_contributions_and_gradients_sum_to_full_effective_batch(chunk, uncertainty, empty):
    outputs, batch = fixture(empty=empty, uncertainty=uncertainty)
    w = LossWeights(plan=.7, occupancy=.3, lane=.4, motion=.9, uncertainty=uncertainty)
    reference, reference_parts = compute_loss(outputs, batch, w)
    reference.backward()
    expected_grads = {k: v.grad.clone() for k, v in outputs.items()}
    sliced_outputs = {k: v.detach().clone().requires_grad_() for k, v in outputs.items()}
    normalizers = to_device(build_loss_normalizers(batch), torch.device("cpu"))
    sums = {k: torch.zeros_like(v) for k, v in reference_parts.items()}
    for start in range(0, 7, chunk):
        selection = slice(start, min(7, start + chunk))
        out = {k: v[selection] for k, v in sliced_outputs.items()}
        labels = {k: v[selection] for k, v in batch.items()}
        loss, parts = compute_loss(out, labels, w, normalizers=normalizers)
        loss.backward()  # No retain_graph; no extra mean by chunk/sample count.
        for k, v in parts.items(): sums[k] += v.detach()
    for k in sums:
        torch.testing.assert_close(sums[k], reference_parts[k], rtol=2e-6, atol=2e-7)
    for k in sliced_outputs:
        torch.testing.assert_close(sliced_outputs[k].grad, expected_grads[k], rtol=2e-6, atol=2e-7)
    if empty:
        assert all(v.item() == 0 for v in sums.values())
        assert all(v.grad.count_nonzero().item() == 0 for v in sliced_outputs.values())


def test_global_counts_expand_history_masks_and_balance_globally_present_classes():
    _, batch = fixture()
    prior = {k: v.clone() for k, v in batch.items()}
    normals = build_loss_normalizers(batch)
    assert normals["history_valid"].item() == batch["history_valid"].sum().item() * 4
    assert normals["state_valid"].item() == batch["state_valid"][:, :5].sum().item()
    assert normals["stop_valid"].item() == batch["state_valid"][:, 5].sum().item()
    assert normals["plan_complete"].item() == batch["plan_valid"].all(-1).sum().item()
    assert normals["occ_classes"].item() == 2 and normals["lane_classes"].item() == 1
    assert normals["lane_negative"].item() == 0
    assert all(v.dtype == torch.int64 and v.ndim == 0 and v.device.type == "cpu" and not v.requires_grad for v in normals.values())
    assert all(torch.allclose(batch[k], v, equal_nan=True) for k, v in prior.items())


def test_missing_optional_masks_count_all_target_elements():
    _, batch = fixture()
    for k in ("plan_valid", "history_valid", "state_valid"): del batch[k]
    norms = build_loss_normalizers(batch)
    assert [norms[k].item() for k in ("plan_complete", "history_valid", "state_valid", "stop_valid")] == [7, 112, 35, 7]


@pytest.mark.parametrize("bad", ["key", "negative", "float", "vector"])
def test_invalid_normalizers_are_rejected(bad):
    out, batch = fixture()
    norms = build_loss_normalizers(batch)
    if bad == "key": del norms["stop_valid"]
    elif bad == "negative": norms["stop_valid"] = torch.tensor(-1)
    elif bad == "float": norms["stop_valid"] = norms["stop_valid"].float()
    else: norms["stop_valid"] = norms["stop_valid"].reshape(1)
    with pytest.raises(ValueError): compute_loss(out, batch, LossWeights(), normalizers=norms)


def test_builder_rejects_grad_labels_and_bad_mask_shapes():
    _, batch = fixture()
    batch["gt_plan"].requires_grad_()
    with pytest.raises(ValueError, match="detached CPU"): build_loss_normalizers(batch)
    batch["gt_plan"] = batch["gt_plan"].detach()
    batch["plan_valid"] = batch["plan_valid"][:, :5]
    with pytest.raises(ValueError, match="plan_valid"): build_loss_normalizers(batch)


@pytest.mark.parametrize("chunk", [2, 4])
@pytest.mark.parametrize("uncertainty", [False, True])
def test_effective16_shared_model_parameter_gradients_match(chunk, uncertainty):
    templates, batch = fixture(n=16, uncertainty=uncertainty)
    shapes = {k: value.shape[1:] for k, value in templates.items()}
    lengths = {k: value[0].numel() for k, value in templates.items()}
    torch.manual_seed(131)
    full_model = torch.nn.Linear(5, sum(lengths.values()))
    micro_model = copy.deepcopy(full_model)
    inputs = torch.randn(16, 5)
    def forward(model, x):
        packed = model(x)
        chunks = packed.split(list(lengths.values()), dim=-1)
        return {key: value.reshape(len(x), *shapes[key]) for key, value in zip(lengths, chunks)}
    weights = LossWeights(uncertainty=uncertainty)
    loss, _ = compute_loss(forward(full_model, inputs), batch, weights)
    loss.backward()
    norms = build_loss_normalizers(batch)
    total = 0.
    for start in range(0, 16, chunk):
        sl = slice(start, start + chunk)
        contribution, _ = compute_loss(forward(micro_model, inputs[sl]),
                                        {k: v[sl] for k, v in batch.items()}, weights, normalizers=norms)
        contribution.backward()
        total += contribution.detach()
    torch.testing.assert_close(total, loss, rtol=2e-6, atol=2e-7)
    for reference, actual in zip(full_model.parameters(), micro_model.parameters()):
        torch.testing.assert_close(actual.grad, reference.grad, rtol=2e-6, atol=2e-7)
