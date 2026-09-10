"""Cache the direct model's prediction and planner latents for the selector track.

Sharded by --rank/--world-size. Per row we keep: the six predicted waypoints,
the six planner decoder latents the coordinate head reads (the visual evidence
the selector is allowed to use), and the row id. Ego status is NOT cached and
must never become a selector feature; the trunk's own A2 route is unchanged.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
R = Path("/NHNHOME/data/sukim/adcl"); WT = R/"experiment_worktrees/controlflow_20260909"
sys.path.insert(0, str(WT)); sys.path.insert(0, str(WT/"scripts"))
import train_motiondrive_v2 as trainer
from motiondrive_v2_data import MotionDriveDataset
from motiondrive_v2_shared_status_data import SharedStatusDataset, load_status_overlay
from models.motiondrive_v2 import MotionDriveV2Config
from models.motiondrive_v2.model import MotionDriveV2
from models.motiondrive_v2.shared_status_query import install_shared_status_query
from torch.utils.data import DataLoader, Subset
ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True); ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--world-size", type=int, required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
CK = R/"work_dirs/motiondrive_v2/controlflow_b0_direct_last4000/last.pth"
pay = torch.load(CK, map_location="cpu", weights_only=False)
cfg = MotionDriveV2Config(**pay["manifest"]["model_config"]); dev = torch.device("cuda:0")
model = install_shared_status_query(MotionDriveV2(cfg)).to(dev).eval()
model.load_state_dict(pay["model"], strict=True)
lat = {}
model.planner.xy_head.register_forward_pre_hook(
    lambda m, inp: lat.__setitem__("z", inp[0].detach()))
orig = trainer.model_inputs
trainer.model_inputs = lambda b, **k: {**orig(b, **k), "provided_status5": b["provided_status5"]}
_, ov = load_status_overlay(str(R/"data/etri/motiondrive_v2_shared_status_a1_20260908_ops"),
    "8852f1e7ce80895b09ac400e8332d99d9feb6918a8a18e0893738300a3021699")
stride = 1 if a.split == "train" else 5
base = MotionDriveDataset(data_root="/tmp/pm97",
    split_manifest=str(R/"data/etri/motiondrive_v2/grouped_split_rawtime.json"), split=a.split,
    supervision_root=str(R/"data/etri/motiondrive_v2/train_tune_geometry_v2"), min_frame=30,
    frame_stride=stride, max_samples=0, augment=False, seed=0,
    history_contract=cfg.history_contract, history_overlay_root=None,
    expected_history_overlay_sha256=None)
ds = SharedStatusDataset(base, a.split, ov, "provided_causal_5d")
idx = list(range(a.rank, len(ds), a.world_size))
dl = DataLoader(Subset(ds, idx), batch_size=8, shuffle=False, num_workers=5, pin_memory=True)
P, Z, Rw, GX = [], [], [], []
t0 = time.time()
with torch.inference_mode():
    for i, raw in enumerate(dl):
        b = trainer.to_device(raw, dev)
        with trainer.autocast(dev, "bf16"):
            out = model(**trainer.model_inputs(b, time_input="nominal",
                        nominal_history_seconds=cfg.nominal_history_seconds))
        P.append(out["plan_abs"].float().cpu().numpy().astype(np.float32))
        Z.append(lat.pop("z").float().cpu().numpy().astype(np.float16))
        Rw.append(raw["row"].numpy()); GX.append(raw["goal_xy"].numpy().astype(np.float32))
        if i % 200 == 0:
            print("  rank%d %d/%d %.0fs" % (a.rank, i*8, len(idx), time.time()-t0), flush=True)
np.savez(a.out, pred=np.concatenate(P), latent=np.concatenate(Z),
         row=np.concatenate(Rw), goal=np.concatenate(GX))
print("rank%d wrote %s  n=%d  %.0fs" % (a.rank, a.out, len(np.concatenate(Rw)), time.time()-t0), flush=True)
