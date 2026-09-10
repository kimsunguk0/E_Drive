"""Score the original and different-scene-image caches with the same frozen head."""
import json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "experiments/pv_selector_20260910")
from train_pv_selector import Cache, evaluate
from pv_selector import SceneResidualSelector

CODEX = Path("/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910")
WT = Path("/NHNHOME/data/sukim/adcl/experiment_worktrees/pvselector_20260910")
BANK = CODEX / "cache/sparsedrivev2_20260910/bank/p1024_v1024_native100m_progress6.npz"
HEAD = WT / "work_dirs/pv_selector_20260910/pv_status_real_s0_v1/step_004000.pth"
CACHES = {"original": CODEX / "cache/c_refine_20260910/c_p20v64_tune_full_v1/tune",
          "different_scene_images": WT / "cache/pv_selector_20260910/c_p20v64_tune_imgswap_v1/tune"}

with np.load(BANK, allow_pickle=False) as f:
    key = next(k for k in ("traj_xyz8", "traj_xy8", "traj_vocab") if k in f)
    bank = torch.from_numpy(np.array(f[key][..., :6, :2].reshape(-1, 6, 2))).float().cuda()

state = torch.load(HEAD, map_location="cpu", weights_only=False)
head = SceneResidualSelector(mode="real").cuda()
head.load_state_dict(state["model"] if "model" in state else state, strict=True)
head.eval()

out = Path("reports/pv_selector_20260910/image_ablation_v1")
out.mkdir(parents=True, exist_ok=True)
summary = {}
for name, directory in CACHES.items():
    cache = Cache(directory, bank, torch.device("cuda"), preload=False)
    cache.attach_status8("tune")
    report = evaluate(head, cache, out, 4000, 64, prefix="cache_" + name)
    manifest = cache.manifest
    summary[name] = {
        "head_d3": report["official_d3"],
        "shortlist_oracle_d3": report["shortlist_oracle_d3"],
        "selection_regret": report["selection_regret"],
        "base_d3": report["base_d3"],
        "cache_mean_base_d3": manifest.get("mean_old_final_selection_d3"),
        "cache_mean_oracle": manifest.get("mean_shortlist_oracle"),
        "n": report["n"],
    }
    print(name, json.dumps(summary[name]), flush=True)

a, b = summary["original"], summary["different_scene_images"]
summary["delta_substituted_minus_original"] = {
    "head_d3": b["head_d3"] - a["head_d3"],
    "shortlist_oracle_d3": b["shortlist_oracle_d3"] - a["shortlist_oracle_d3"],
    "base_d3": b["base_d3"] - a["base_d3"],
}
(out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print("SUMMARY " + json.dumps(summary["delta_substituted_minus_original"]))
