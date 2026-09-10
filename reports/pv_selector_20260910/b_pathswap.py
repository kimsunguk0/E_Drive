"""Does substituting images destroy arm B's retained paths, not just its velocities?"""
import json
from pathlib import Path
import numpy as np

CODEX = Path("/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910")
WT = Path("/NHNHOME/data/sukim/adcl/experiment_worktrees/pvselector_20260910")
W = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0

with np.load(CODEX / "cache/sparsedrivev2_20260910/bank/p1024_v1024_native100m_progress6.npz",
             allow_pickle=False) as f:
    key = next(k for k in ("traj_xyz8", "traj_xy8", "traj_vocab") if k in f)
    bank = np.array(f[key][..., :6, :2], dtype=np.float64)      # [1024,1024,6,2]
P, V = bank.shape[0], bank.shape[1]
flat = bank.reshape(P * V, 6, 2)

def report(name, directory):
    d = Path(directory)
    manifest = json.loads((d / "manifest.json").read_text())
    ids = np.load(d / "candidate_ids.npy", allow_pickle=False)
    valid = np.load(d / "candidate_valid.npy", allow_pickle=False)
    gt = np.load(d / "gt.npy", allow_pickle=False).astype(np.float64)
    d3 = np.load(d / "d3.npy", allow_pickle=False).astype(np.float64)
    kept = np.where(valid, d3, np.inf).min(1)
    paths = np.unique(ids[0] // V).size
    # oracle over the same retained paths crossed with every velocity
    allv = np.empty(len(gt))
    for i in range(len(gt)):
        p = np.unique(ids[i] // V)
        cand = flat[(p[:, None] * V + np.arange(V)[None, :]).ravel()]
        err = np.sqrt(((cand - gt[i][None]) ** 2).sum(-1)) @ W
        allv[i] = err.min()
    print("%-26s base_d3 %.6f  P20xV64 oracle %.6f  P20xallV1024 oracle %.6f  paths/row %d"
          % (name, manifest["mean_old_final_selection_d3"], kept.mean(), allv.mean(), paths))
    return allv.mean(), kept.mean()

a = report("B original", WT / "cache/pv_selector_20260910/b_p20v64_tune_orig_v1/tune")
b = report("B different-scene imgs", WT / "cache/pv_selector_20260910/b_p20v64_tune_imgswap_v1/tune")
print("\nallV1024 oracle  %.6f -> %.6f   delta %+.6f" % (a[0], b[0], b[0] - a[0]))
print("V64 oracle       %.6f -> %.6f   delta %+.6f" % (a[1], b[1], b[1] - a[1]))
