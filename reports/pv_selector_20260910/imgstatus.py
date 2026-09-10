"""Score the frozen status head with image-inferred status instead of provided status."""
import json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "experiments/pv_selector_20260910")
from train_pv_selector import Cache, evaluate
from pv_selector import SceneResidualSelector
from pv_status import load_status8

CODEX = Path("/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910")
WT = Path("/NHNHOME/data/sukim/adcl/experiment_worktrees/pvselector_20260910")
CACHE = CODEX / "cache/c_refine_20260910/c_p20v64_tune_full_v1/tune"
HEAD = WT / "work_dirs/pv_selector_20260910/pv_status_real_s0_v1/step_004000.pth"

with np.load(CODEX / "cache/sparsedrivev2_20260910/bank/p1024_v1024_native100m_progress6.npz",
             allow_pickle=False) as f:
    key = next(k for k in ("traj_xyz8", "traj_xy8", "traj_vocab") if k in f)
    bank = torch.from_numpy(np.array(f[key][..., :6, :2].reshape(-1, 6, 2))).float().cuda()

head = SceneResidualSelector(mode="real").cuda()
head.load_state_dict(torch.load(HEAD, map_location="cpu", weights_only=False)["head"], strict=True)
head.eval()

out = WT / "reports/pv_selector_20260910/image_status_v1"
out.mkdir(parents=True, exist_ok=True)
cache = Cache(CACHE, bank, torch.device("cuda"), preload=False)
rows = cache.rows_cpu()
provided, _ = load_status8(CACHE, "tune", rows)

variants = {"provided_causal": provided}
for seed in ("b0", "b1"):
    z = np.load(CODEX / ("reports/sparsedrivev2_status_20260910/empirical/empirical_p7_%s_status.npz" % seed),
                allow_pickle=False)
    assert np.array_equal(z["rows"], rows), seed
    variants["p7_image_" + seed] = z["status8"].astype(np.float32)
variants["zeros"] = np.zeros_like(provided)

summary = {}
for name, s8 in variants.items():
    cache.status8 = torch.from_numpy(s8).cuda()
    report = evaluate(head, cache, out, 4000, 64, prefix="status_" + name)
    err = np.abs(s8[:, 4] - provided[:, 4]).mean()
    summary[name] = {"official_d3": report["official_d3"], "vx_mae_vs_provided": float(err),
                     "selection_regret": report["selection_regret"]}
    print("%-18s D3 %.6f   regret %.6f   vx MAE vs provided %.4f"
          % (name, report["official_d3"], report["selection_regret"], err), flush=True)
(out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
