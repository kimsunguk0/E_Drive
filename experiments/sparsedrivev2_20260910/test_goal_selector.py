"""CPU-only behavioral tests for the optional goal-selection wrapper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .goal_selector import GoalConditionedSelector
    from .public_model import PublicSparseDriveV2
except ImportError:
    from goal_selector import GoalConditionedSelector
    from public_model import PublicSparseDriveV2


def assert_outputs_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), "Zero initialization changed a base output"
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_outputs_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_outputs_equal(x, y)
    else:
        assert a == b


def assert_fixed_rows(model, out):
    bank = model._trajectory_head.traj_vocab.flatten(0, 1)
    expected = bank[out["candidate_ids"], :6, :2]
    assert torch.equal(expected.contiguous().view(torch.uint8), out["candidate_xy"].contiguous().view(torch.uint8))
    chosen = bank[out["selected_candidate_id"], :6, :2]
    assert torch.equal(chosen.contiguous().view(torch.uint8), out["trajectory"].contiguous().view(torch.uint8))


def run(checkpoint, bank_path):
    torch.set_num_threads(4)
    torch.manual_seed(19)
    source, coverage = PublicSparseDriveV2.from_public_checkpoint(checkpoint, bank_path=bank_path, backend="grid")
    h = source._trajectory_head
    bank = {"path_vocab": h.path_vocab[:4], "vel_vocab": h.vel_vocab[:4],
            "traj_vocab": h.traj_vocab[:4, :4], "traj_mask": h.traj_mask[:4, :4]}
    # Keep all rows first, so the candidate tensor itself must be byte-identical
    # when the goal changes. Later enable top-k to verify pruned row identities.
    base = PublicSparseDriveV2(bank, backend="grid", path_filter=(4,4), velocity_filter=(4,4))
    state = {k:v for k,v in source.state_dict().items() if k not in {"_trajectory_head."+x for x in bank}}
    base.load_state_dict(state, strict=False)
    base.eval()
    before = {k: (id(p), p._version) for k, p in base.named_parameters()}
    images = torch.randn(2,3,3,64,128)
    projection = torch.zeros(2,3,4,4)
    projection[:,:,0,0], projection[:,:,0,1] = 64, -96
    projection[:,:,1,0], projection[:,:,1,2], projection[:,:,1,3] = 32, -96, 144
    projection[:,:,2,0], projection[:,:,3,3] = 1, 1
    status = torch.randn(2,8)
    status[:,:4] = 0
    inputs = dict(images=images,lidar2img=projection,status=status)
    goal_a = torch.tensor([[50.,20.],[70.,-15.]])
    goal_b = torch.tensor([[40.,-25.],[30.,25.]])
    with torch.no_grad():
        baseline = base(**inputs)
    wrapper = GoalConditionedSelector(base).eval()
    assert wrapper.base is base
    assert before == {k:(id(p),p._version) for k,p in base.named_parameters()}
    assert wrapper.goal_projection.weight.numel() == 512
    assert torch.count_nonzero(wrapper.goal_projection.weight) == 0
    with torch.no_grad():
        for goal in (None, goal_a, goal_b):
            out = wrapper(**inputs,goal_xy=goal)
            assert_outputs_equal(baseline,out)
            assert_fixed_rows(wrapper,out)
    loss_out = wrapper(**inputs,goal_xy=goal_a)
    loss = loss_out["scores"].square().mean()
    for coarse in loss_out["coarse"]:
        loss = loss + .01*coarse["path_scores"].square().mean() + .01*coarse["velocity_scores"].square().mean()
    loss.backward()
    grad = wrapper.goal_projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    gradient_norm = float(grad.norm())
    with torch.no_grad():
        wrapper.goal_projection.weight.normal_(0,.03)
        out_a = wrapper(**inputs,goal_xy=goal_a)
        out_b = wrapper(**inputs,goal_xy=goal_b)
    assert not torch.equal(out_a["scores"],out_b["scores"])
    assert torch.equal(out_a["candidate_xy"].contiguous().view(torch.uint8),out_b["candidate_xy"].contiguous().view(torch.uint8))
    for out in (out_a,out_b):
        assert_fixed_rows(wrapper,out)
    base.path_filter, base.velocity_filter = (4,2), (4,2)
    with torch.no_grad():
        assert_fixed_rows(wrapper,wrapper(**inputs,goal_xy=goal_a))
        assert_fixed_rows(wrapper,wrapper(**inputs,goal_xy=goal_b))
    hook_count = len(base._status_encoding._forward_hooks)

    def explode(module, args):
        raise ValueError("deliberate exception after status hook")

    trouble = base._backbone.register_forward_pre_hook(explode)
    try:
        try:
            wrapper(**inputs,goal_xy=goal_a)
            raise AssertionError("Expected base exception")
        except ValueError as exc:
            assert "deliberate exception" in str(exc)
    finally:
        trouble.remove()
    assert len(base._status_encoding._forward_hooks) == hook_count
    with torch.no_grad():
        wrapper(**inputs,goal_xy=goal_a)

    def nested(module, args):
        wrapper(**inputs,goal_xy=goal_a)

    trouble = base._status_encoding.register_forward_pre_hook(nested)
    try:
        try:
            wrapper(**inputs,goal_xy=goal_a)
            raise AssertionError("Expected reentrant rejection")
        except RuntimeError as exc:
            assert "reentrant" in str(exc)
    finally:
        trouble.remove()
    assert len(base._status_encoding._forward_hooks) == hook_count
    with torch.no_grad():
        assert_fixed_rows(wrapper,wrapper(**inputs,goal_xy=goal_a))
    # No omitted-goal effect even after the goal projection has changed.
    with torch.no_grad():
        assert_outputs_equal(base(**inputs),wrapper(**inputs))
    return {"passed":True,"device":"cpu","new_parameters":512,
            "public_base_reused_tensors":coverage["reused_tensor_count"],
            "all_original_parameter_objects_and_versions_preserved":True,
            "zero_init_all_fp32_outputs_equal":True,
            "goal_projection_gradient_norm":gradient_norm,
            "nonzero_projection_goal_changes_scores":True,
            "goal_changes_preserve_candidate_bytes_with_all_rows_retained":True,
            "goal_changes_preserve_global_bank_row_bytes_after_pruning":True,
            "exception_hook_cleanup":True,"reentrant_rejected_and_cleaned":True,
            "omitted_goal_equals_base_after_learning":True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/sparsedrivev2_20260910/public/sparsedrive_navsimv1_92p2.ckpt")
    p.add_argument("--bank", default="cache/sparsedrivev2_20260910/bank/p1024_v256_native100m_v8.npz")
    p.add_argument("--output", default="reports/sparsedrivev2_20260910/public_init/goal_selector_cpu.json")
    args = p.parse_args()
    report = run(args.checkpoint,args.bank)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
