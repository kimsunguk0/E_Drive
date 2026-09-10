import numpy as np, json
from pathlib import Path
R = Path("reports/pv_selector_20260910/image_ablation_v1")
a = np.load(R / "cache_original_004000.npz", allow_pickle=False)
b = np.load(R / "cache_different_scene_images_004000.npz", allow_pickle=False)
assert np.array_equal(a["rows"], b["rows"])
cache = Path("/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910/"
             "cache/c_refine_20260910/c_p20v64_tune_full_v1/tune")
sess = np.load(cache / "session_index.npy", allow_pickle=False)
crows = np.load(cache / "rows.npy", allow_pickle=False)
pos = {int(r): i for i, r in enumerate(crows)}
sidx = np.array([sess[pos[int(r)]] for r in a["rows"]])
u = np.unique(sidx)

def boot(x, y, label):
    diff = y - x
    groups = [diff[sidx == s] for s in u]
    rng = np.random.default_rng(0)
    draws = np.array([np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean()
                      for _ in range(20000)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    per = np.array([y[sidx == s].mean() - x[sidx == s].mean() for s in u])
    print("%-22s %.6f -> %.6f  delta %+.6f sd %.6f CI [%+.6f,%+.6f] P(<0)=%.4f  worse %d/%d"
          % (label, x.mean(), y.mean(), diff.mean(), draws.std(), lo, hi,
             (draws < 0).mean(), (per > 0).sum(), len(per)))

print("rows %d sessions %d" % (len(sidx), len(u)))
boot(a["d3"], b["d3"], "head D3")
boot(a["shortlist_oracle"], b["shortlist_oracle"], "shortlist oracle")
boot(a["base_d3"], b["base_d3"], "C base scorer D3")
changed = float(np.mean(a["candidate_id"] != b["candidate_id"]))
print("selected bank row changed on %.2f%% of rows" % (100 * changed))
