import numpy as np, json
from pathlib import Path
R = Path("work_dirs/pv_selector_20260910")
a = np.load(R / "pv_status_zero_s0_v1/eval_004000.npz", allow_pickle=False)
b = np.load(R / "pv_status_real_s0_v1/eval_004000.npz", allow_pickle=False)
print("keys:", a.files)
rows_a, rows_b = a["rows"], b["rows"]
assert np.array_equal(rows_a, rows_b)
cache = Path("/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910/"
             "cache/c_refine_20260910/c_p20v64_tune_full_v1/tune")
sess = np.load(cache / "session_index.npy", allow_pickle=False)
crows = np.load(cache / "rows.npy", allow_pickle=False)
order = {int(r): i for i, r in enumerate(crows)}
sidx = np.array([sess[order[int(r)]] for r in rows_a])
da, db = a["d3"], b["d3"]
print("n=%d sessions=%d  zero=%.6f real=%.6f" % (len(da), len(np.unique(sidx)), da.mean(), db.mean()))
diff = db - da
groups = [diff[sidx == s] for s in np.unique(sidx)]
rng = np.random.default_rng(0)
draws = np.array([np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean()
                  for _ in range(20000)])
lo, hi = np.percentile(draws, [2.5, 97.5])
print("paired delta %+.6f  sd %.6f  95%% CI [%+.6f, %+.6f]  P(>0)=%.4f"
      % (diff.mean(), draws.std(), lo, hi, (draws > 0).mean()))
per = np.array([db[sidx == s].mean() - da[sidx == s].mean() for s in np.unique(sidx)])
print("sessions improved: %d/%d   worst %+.4f  best %+.4f" % ((per < 0).sum(), len(per), per.max(), per.min()))
print("real per-session D3:", " ".join("%.3f" % db[sidx == s].mean() for s in np.unique(sidx)))
