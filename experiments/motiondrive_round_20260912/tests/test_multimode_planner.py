import sys, torch
sys.path.insert(0, "/NHNHOME/data/sukim/adcl")
sys.path.insert(0, "/NHNHOME/data/sukim/adcl/scripts")
sys.path.insert(0, "/NHNHOME/data/sukim/adcl/experiments/motiondrive_round_20260912")
from models.motiondrive_v2 import MotionDriveV2Config
from models.motiondrive_v2.planner import DirectTrajectoryPlanner
from multimode_continuous_planner import (MultiModePlanner, block_diagonal_mask,
                                          multimode_losses)

cfg = MotionDriveV2Config()
torch.manual_seed(0)
parent = DirectTrajectoryPlanner(cfg).eval()
torch.manual_seed(0)
m8 = MultiModePlanner(cfg, modes=8).eval()
missing, unexpected = m8.load_state_dict(parent.state_dict(), strict=False)
print("loaded parent into M8: unexpected=%s, new tensors=%s"
      % (list(unexpected), sorted({k.split('.')[0] for k in missing})))

b = 3
scene = torch.randn(b, 3072, 128); motion = torch.randn(b, 192, 128)
state = torch.randn(b, 6); history = torch.randn(b, 4, 4)
with torch.no_grad():
    ref = parent(scene, motion, state, history)
    _ = m8(scene, motion, state, history)
    y = m8.last["candidates"]
print("mode 0 reproduces the parent: max |delta| %.3e" % float((y[:, 0] - ref).abs().max()))
torch.testing.assert_close(y[:, 0], ref, rtol=1e-5, atol=1e-6)

torch.manual_seed(0)
single = MultiModePlanner(cfg, modes=1).eval()
single.load_state_dict(parent.state_dict(), strict=False)
with torch.no_grad():
    _ = single(scene, motion, state, history)
torch.testing.assert_close(single.last["candidates"][:, 0], ref, rtol=1e-5, atol=1e-6)
print("M=1 matches the parent function as well")

mask = block_diagonal_mask(8, 6, torch.device("cpu"))
assert mask.shape == (48, 48) and not mask[:6, :6].any() and mask[:6, 6:12].all()
assert (~mask).sum() == 8 * 36
print("block-diagonal mask: within-mode open, across-mode closed, %d open pairs" % int((~mask).sum()))

gt = torch.randn(b, 6, 2); complete = torch.ones(b, dtype=torch.bool)
_ = m8(scene, motion, state, history)
loss, parts = multimode_losses(m8, gt, complete)
print("losses:", {k: round(float(v), 5) for k, v in parts.items()})
assert float(parts["oracle_d3"]) <= float(parts["selected_d3"]) + 1e-6
loss.backward()
grads = {n: float(p.grad.abs().sum()) for n, p in m8.named_parameters() if p.grad is not None}
print("mode_embedding grad %.4e, scorer grad %.4e, xy_head grad %.4e"
      % (grads["mode_embedding"], sum(v for k, v in grads.items() if "candidate_scorer" in k),
         sum(v for k, v in grads.items() if "xy_head" in k)))
# The scorer output layer is zero-initialised, so at step 0 only it carries
# gradient; the layers before it unblock once that projection moves.
assert grads["mode_embedding"] > 0
assert grads["candidate_scorer.score.4.weight"] > 0 and grads["candidate_scorer.score.0.weight"] == 0
print("\nM8 CHECKS OK")
