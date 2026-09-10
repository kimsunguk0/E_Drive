"""Parameter-free and one-parameter selection rules over the model's own candidates.

Nothing here is trained. The network produces the retained P20 x V64 candidates
and their scores; goal and status are used only to pick one of them.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "experiments/pv_selector_20260910")
from pv_status import load_status8

CODEX = Path("/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910")
CACHE = CODEX / "cache/c_refine_20260910/c_p20v64_tune_full_v1/tune"
W = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0

ids = np.load(CACHE / "candidate_ids.npy", allow_pickle=False)
valid = np.load(CACHE / "candidate_valid.npy", allow_pickle=False)
scores = np.load(CACHE / "scores.npy", allow_pickle=False).astype(np.float64)
goal = np.load(CACHE / "goal_xy.npy", allow_pickle=False).astype(np.float64)
d3 = np.load(CACHE / "d3.npy", allow_pickle=False).astype(np.float64)
rows = np.load(CACHE / "rows.npy", allow_pickle=False)
sess = np.load(CACHE / "session_index.npy", allow_pickle=False)
status8, _ = load_status8(CACHE, "tune", rows)
v0 = status8[:, 4:6].astype(np.float64)

with np.load(CODEX / "cache/sparsedrivev2_20260910/bank/p1024_v1024_native100m_progress6.npz",
             allow_pickle=False) as f:
    key = next(k for k in ("traj_xyz8", "traj_xy8", "traj_vocab") if k in f)
    bank = np.array(f[key][..., :6, :2], dtype=np.float64)
V = bank.shape[1]
flat = bank.reshape(-1, 6, 2)

xy = flat[ids]                                     # [N,K,6,2]
BIG = 1e9
first_vel = xy[:, :, 0, :] * 2.0                   # [N,K,2] first 0.5 s interval velocity
speed_gap = np.linalg.norm(first_vel - v0[:, None, :], axis=-1)
end = xy[:, :, -1, :]                              # 3 s endpoint
gdir = goal / np.maximum(np.linalg.norm(goal, axis=1, keepdims=True), 1e-6)
# cross-track of the 3 s endpoint about the ray toward the provided goal
cross = np.abs(end[..., 0] * gdir[:, None, 1] - end[..., 1] * gdir[:, None, 0])

def score_of(sel):
    return d3[np.arange(len(d3)), sel].mean()

def pick(cost):
    return np.where(valid, cost, BIG).argmin(1)

print("n=%d candidates=%d" % ids.shape)
print("%-46s %s" % ("rule", "tune D3"))
print("%-46s %.6f" % ("R0  network base score only (no goal/status)", score_of(pick(-scores))))
print("%-46s %.6f" % ("R1  velocity by |v0| only", score_of(pick(speed_gap))))
print("%-46s %.6f" % ("R2  path by goal cross-track only", score_of(pick(cross))))
best = None
for lam in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0):
    s = score_of(pick(speed_gap + lam * cross))
    if best is None or s < best[1]:
        best = (lam, s)
    print("%-46s %.6f" % ("R3  |v0| + %.2f * cross-track" % lam, s))
print("%-46s %.6f  (lambda=%.2f)" % ("R3  best", best[1], best[0]))
for lam in (0.2, 0.5, 1.0):
    for mu in (0.05, 0.2, 0.5):
        s = score_of(pick(speed_gap + lam * cross - mu * scores))
        print("%-46s %.6f" % ("R4  |v0| + %.2f*cross - %.2f*score" % (lam, mu), s))
print("%-46s %.6f" % ("oracle over the same candidates", np.where(valid, d3, BIG).min(1).mean()))
print("%-46s %.6f" % ("learned 37,697-parameter MLP head", 0.138274))
