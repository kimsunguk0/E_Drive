import numpy as np
from pathlib import Path
z = np.load("/tmp/pm97/data/etri/ego_cache.npz", allow_pickle=False)
cache = Path("/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910/"
             "cache/c_refine_20260910/c_p20v64_tune_full_v1/tune")
rows = np.load(cache / "rows.npy", allow_pickle=False)
cmd = z["vad_cmd"][rows].astype(int)
goal = z["goal"][rows]
fut = z["fut"][rows]
speed = z["speed"][rows]
u, c = np.unique(cmd, return_counts=True)
print("vad_cmd on tune rows:", dict(zip(u.tolist(), c.tolist())), " (n=%d)" % len(rows))
print("%-6s %6s %10s %10s %10s %10s" % ("cmd", "n", "goal_y", "fut_y@3s", "goal_x", "speed"))
for v in u:
    m = cmd == v
    print("%-6d %6d %10.3f %10.3f %10.3f %10.3f"
          % (v, m.sum(), goal[m, 1].mean(), fut[m, -1, 1].mean(), goal[m, 0].mean(), speed[m].mean()))
# how much does cmd add beyond goal for predicting lateral outcome?
y = fut[:, -1, 1]
X = np.column_stack([np.ones(len(rows)), goal[:, 0], goal[:, 1]])
b, *_ = np.linalg.lstsq(X, y, rcond=None)
r_goal = y - X @ b
Xc = np.column_stack([X] + [(cmd == v).astype(float) for v in u[1:]])
bc, *_ = np.linalg.lstsq(Xc, y, rcond=None)
r_both = y - Xc @ bc
print("lateral y@3s residual std: goal only %.4f  goal+cmd %.4f  (reduction %.2f%%)"
      % (r_goal.std(), r_both.std(), 100 * (1 - r_both.std() / r_goal.std())))
