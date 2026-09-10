"""Cache BEV features sampled ALONG each candidate trajectory.

The previous selector only saw a single row-level latent, so it had no visual
information that distinguished one candidate from another. Here each candidate's
ten waypoints are used to bilinearly sample the shared BEV raster, giving the
scorer per-candidate visual evidence - the analogue of the visual logit that
worked in PHASE5EFG. Ego status is never sampled or stored.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
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
Wn = np.array([11, 11, 5, 5, 2, 2], dtype=np.float32)/36.
bank = torch.tensor(np.load(R/"reports/p7_train203_bank_feasibility_20260908_ops/bank/bank.npy"))
CK = R/"work_dirs/motiondrive_v2/controlflow_b0_direct_last4000/last.pth"
pay = torch.load(CK, map_location="cpu", weights_only=False)
cfg = MotionDriveV2Config(**pay["manifest"]["model_config"]); dev = torch.device("cuda:0")
GX, GY = cfg.grid_size; XR, YR = cfg.x_range, cfg.y_range
model = install_shared_status_query(MotionDriveV2(cfg)).to(dev).eval()
model.load_state_dict(pay["model"], strict=True)
cap = {}
model.scene_encoder.register_forward_hook(
    lambda m, i, o: cap.__setitem__("bev", o["scene_features"].detach()))
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
bank_d = bank.to(dev); Wt = torch.tensor(Wn, device=dev)
PA, SA, RA, GA, TA = [], [], [], [], []
t0 = time.time()
with torch.inference_mode():
    for i, raw in enumerate(dl):
        b = trainer.to_device(raw, dev)
        with trainer.autocast(dev, "bf16"):
            out = model(**trainer.model_inputs(b, time_input="nominal",
                        nominal_history_seconds=cfg.nominal_history_seconds))
        P = out["plan_abs"].float()                                   # (B,6,2)
        bev = cap.pop("bev").float().reshape(len(P), GX, GY, -1).permute(0, 3, 1, 2)  # (B,C,64,48)
        dd = (torch.linalg.norm(P[:, None] - bank_d[None, :, :6, :], dim=3)*Wt).sum(2)
        top = dd.topk(12, dim=1, largest=False).indices                # (B,12)
        cand = bank_d[top]                                             # (B,12,10,2)
        step = P[:, 5] - P[:, 4]
        tail = P[:, 5][:, None] + step[:, None]*torch.arange(1, 5, device=dev)[None, :, None]
        own = torch.cat([P, tail], 1)[:, None]                         # (B,1,10,2)
        cand = torch.cat([own, cand], 1)                               # (B,13,10,2)
        xn = (cand[..., 0] - XR[0])/(XR[1] - XR[0])*2 - 1              # -> H axis (64)
        yn = (cand[..., 1] - YR[0])/(YR[1] - YR[0])*2 - 1              # -> W axis (48)
        grid = torch.stack([yn, xn], -1)                               # (B,13,10,2) grid_sample order
        samp = F.grid_sample(bev, grid, mode="bilinear", align_corners=False,
                             padding_mode="border")                    # (B,C,13,10)
        PA.append(P.cpu().numpy().astype(np.float32))
        SA.append(samp.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16))  # (B,13,10,C)
        TA.append(top.cpu().numpy().astype(np.int16))
        RA.append(raw["row"].numpy()); GA.append(raw["goal_xy"].numpy().astype(np.float32))
        if i % 200 == 0: print("  rank%d %d/%d %.0fs" % (a.rank, i*8, len(idx), time.time()-t0), flush=True)
np.savez(a.out, pred=np.concatenate(PA), path=np.concatenate(SA), top=np.concatenate(TA),
         row=np.concatenate(RA), goal=np.concatenate(GA))
print("rank%d wrote %s n=%d %.0fs" % (a.rank, a.out, len(np.concatenate(RA)), time.time()-t0), flush=True)
