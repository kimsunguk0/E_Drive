"""Phase 0.3: cost, fit and objective checks before the real arms."""
import sys, time, numpy as np, torch
sys.path.insert(0, "experiments/sdv2_f4_20260911")
from train_f4_coverage_head import (CoverageScorer, Cache, build_batch,
                                    velocity_descriptor, d3_of, WEIGHTS)

dev = torch.device("cuda")
with np.load("/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910/"
             "cache/sparsedrivev2_20260910/bank/p1024_v1024_native100m_progress6.npz",
             allow_pickle=False) as f:
    bank = torch.from_numpy(np.array(f["traj_xyz8"])[..., :6, :2]).float().to(dev)
    mask = torch.from_numpy(np.array(f["traj_mask"])[..., :6]).bool().all(-1).to(dev)
w = torch.tensor(WEIGHTS, device=dev)
desc = velocity_descriptor(bank)

# 1. Is the speed descriptor really path-independent?
arc = torch.linalg.vector_norm(torch.diff(bank, dim=2,
        prepend=torch.zeros_like(bank[:, :, :1])), dim=-1).cumsum(-1)[..., -1]   # [P,V]
ref = arc[0]
rel = (arc - ref[None]).abs().median() / ref.median()
print("descriptor: total arclength median relative spread across paths %.4f" % float(rel))

cache = Cache("cache/sdv2_f4_20260911/b8k_tune", dev)
print("tune rows", cache.n)

for arm, stage1, vids in (("n64", 64, None), ("w1024", None, torch.arange(1024, device=dev))):
    head = CoverageScorer(visual=True).to(dev)
    idx = np.arange(0, 8)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    inputs, cost, valid = build_batch(cache, idx, bank, mask, vids, desc, w, stage1)
    scores = head(**inputs).flatten(1)
    target = torch.softmax(-cost.flatten(1) / 0.1, -1)
    loss = -(target * torch.log_softmax(scores, -1)).sum(-1).mean()
    loss.backward()
    torch.cuda.synchronize()
    print("%-7s candidates %6d  step %5.0f ms  peak %5.0f MiB  loss %.4f  oracle %.6f"
          % (arm, scores.shape[1], 1000 * (time.time() - t0),
             torch.cuda.max_memory_allocated() / 2**20, float(loss),
             float(torch.where(valid.flatten(1), cost.flatten(1),
                               torch.full_like(cost.flatten(1), 1e9)).min(1).values.mean())))

# 2. Does accumulating microbatches equal one dense batch, in loss and gradient?
head = CoverageScorer(visual=True).to(dev)
vids = torch.arange(1024, device=dev)
def run(indices, accum):
    head.zero_grad(set_to_none=True)
    total = 0.
    for part in np.array_split(indices, accum):
        inputs, cost, _ = build_batch(cache, part, bank, mask, vids, desc, w, None)
        s = head(**inputs).flatten(1)
        t = torch.softmax(-cost.flatten(1) / 0.1, -1)
        l = -(t * torch.log_softmax(s, -1)).sum(-1).sum() / len(indices)
        l.backward(); total += float(l)
    return total, torch.cat([p.grad.flatten() for p in head.parameters() if p.grad is not None])
idx = np.arange(0, 8)
l1, g1 = run(idx, 1)
l2, g2 = run(idx, 4)
print("dense vs 4-way accumulation: loss delta %.3e, grad max delta %.3e"
      % (abs(l1 - l2), float((g1 - g2).abs().max())))

# 3. Can the head fit a small fixed set at all?
head = CoverageScorer(visual=True).to(dev)
opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
idx = np.arange(0, 64)
for step in range(60):
    opt.zero_grad(set_to_none=True)
    for part in np.array_split(idx, 8):
        inputs, cost, _ = build_batch(cache, part, bank, mask, vids, desc, w, None)
        s = head(**inputs).flatten(1)
        t = torch.softmax(-cost.flatten(1) / 0.1, -1)
        (-(t * torch.log_softmax(s, -1)).sum(-1).sum() / len(idx)).backward()
    opt.step()
picked = []
with torch.inference_mode():
    for part in np.array_split(idx, 8):
        inputs, cost, valid = build_batch(cache, part, bank, mask, vids, desc, w, None)
        s = head(**inputs).flatten(1)
        c = cost.flatten(1)
        picked.append(c[torch.arange(len(part), device=dev), s.argmax(1)].cpu().numpy())
print("60-step overfit on 64 rows: mean D3 %.6f (oracle over the same set follows)"
      % float(np.concatenate(picked).mean()))
