"""Can two OCCUPANCY rasters from frames 0.5 s apart be aligned metrically?

Consecutive tune rows are 5 frames = 0.5 s apart (frame_stride 5). Run the frozen
model over tune, cache the BEV scene map, then correlate consecutive same-scene
pairs over a metric search window. BEV cells are 1.25 m (x) / 1.333 m (y), and
ego displacement at 0.5 s averages 5.27 m = 4.2 x-cells, so the shift is
resolvable in principle. Compares the measured shift against the true
displacement from ego_pose. No training, no code change to the model.
"""
import sys, shutil
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
WT = Path("/NHNHOME/data/sukim/adcl/experiment_worktrees/controlflow_20260909")
sys.path.insert(0, str(WT)); sys.path.insert(0, str(WT / "scripts"))
import train_motiondrive_v2 as trainer
from motiondrive_v2_data import MotionDriveDataset
from motiondrive_v2_shared_status_data import SharedStatusDataset, load_status_overlay
from models.motiondrive_v2 import MotionDriveV2Config
from models.motiondrive_v2.model import MotionDriveV2
from models.motiondrive_v2.shared_status_query import install_shared_status_query
from torch.utils.data import DataLoader

SRC = Path("/NHNHOME/data/sukim/adcl/work_dirs/motiondrive_v2/controlflow_b0_direct_last4000/last.pth")
TMP = Path("/tmp/bev_direct.pth"); shutil.copy2(SRC, TMP)
payload = torch.load(TMP, map_location="cpu", weights_only=False)
config = MotionDriveV2Config(**payload["manifest"]["model_config"])
GX, GY = config.grid_size; XR, YR = config.x_range, config.y_range
CX = (XR[1]-XR[0])/GX; CY = (YR[1]-YR[0])/GY
print("BEV %dx%d  cell %.3f x %.3f m  (step %s)" % (GX, GY, CX, CY, payload.get("step")), flush=True)
dev = torch.device("cuda:0")
model = install_shared_status_query(MotionDriveV2(config)).to(dev).eval()
model.load_state_dict(payload["model"], strict=True)
cap = {}
WHICH = "occ_logits"   # sparse, object-aligned - unlike the 128-d scene features
model.scene_encoder.register_forward_hook(
    lambda m, a, o: cap.__setitem__("bev", o[WHICH].detach().sigmoid()))
orig = trainer.model_inputs
trainer.model_inputs = lambda b, **k: {**orig(b, **k), "provided_status5": b["provided_status5"]}
_, overlay = load_status_overlay("/NHNHOME/data/sukim/adcl/data/etri/motiondrive_v2_shared_status_a1_20260908_ops",
    "8852f1e7ce80895b09ac400e8332d99d9feb6918a8a18e0893738300a3021699")
base = MotionDriveDataset(data_root="/tmp/pm97",
    split_manifest="/NHNHOME/data/sukim/adcl/data/etri/motiondrive_v2/grouped_split_rawtime.json", split="tune",
    supervision_root="/NHNHOME/data/sukim/adcl/data/etri/motiondrive_v2/train_tune_geometry_v2",
    min_frame=30, frame_stride=5, max_samples=0, augment=False, seed=0,
    history_contract=config.history_contract, history_overlay_root=None, expected_history_overlay_sha256=None)
dl = DataLoader(SharedStatusDataset(base, "tune", overlay, "provided_causal_5d"),
                batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
BEV = []; ROWS = []
with torch.inference_mode():
    for i, raw in enumerate(dl):
        b = trainer.to_device(raw, dev)
        with trainer.autocast(dev, "bf16"):
            model(**trainer.model_inputs(b, time_input="nominal",
                  nominal_history_seconds=config.nominal_history_seconds))
        v = cap.pop("bev").float()
        BEV.append(v.reshape(len(v), 1, GX, GY).half().cpu())
        ROWS.append(raw["row"].numpy())
        if i % 60 == 0: print("  %d/%d" % (i*8, len(base)), flush=True)
BEV = torch.cat(BEV); ROWS = np.concatenate(ROWS)
print("cached BEV", tuple(BEV.shape), flush=True)
ego = np.load("/tmp/pm97/data/etri/ego_cache.npz")
sc = ego["scen_idx"][ROWS]; his = ego["his"][ROWS]
pairs = [i for i in range(1, len(ROWS)) if sc[i] == sc[i-1] and ROWS[i]-ROWS[i-1] == 5]
print("usable consecutive 0.5 s pairs:", len(pairs), flush=True)
RX, RY = 14, 6
shifts = [(dx, dy) for dx in range(-RX, RX+1) for dy in range(-RY, RY+1)]
meas = np.zeros((len(pairs), 2)); peak = np.zeros(len(pairs))
CH = 96
for s in range(0, len(pairs), CH):
    idx = pairs[s:s+CH]
    cur = BEV[idx].to(dev).float(); prv = BEV[[i-1 for i in idx]].to(dev).float()
    cur = cur - cur.mean(dim=(2,3), keepdim=True); prv = prv - prv.mean(dim=(2,3), keepdim=True)
    scores = []
    for dx, dy in shifts:
        r = torch.roll(prv, shifts=(dx, dy), dims=(2, 3))
        m = torch.zeros(1, 1, GX, GY, device=dev)
        xs = slice(max(dx, 0), GX+min(dx, 0)); ys = slice(max(dy, 0), GY+min(dy, 0))
        m[:, :, xs, ys] = 1.0
        scores.append(((cur*r).sum(1, keepdim=True)*m).sum((1, 2, 3))/m.sum().clamp_min(1))
    S = torch.stack(scores, 1)
    w = (S/0.05).softmax(1)
    off = torch.tensor(shifts, device=dev, dtype=torch.float32)
    meas[s:s+CH] = (w[:, :, None]*off[None]).sum(1).cpu().numpy()
    peak[s:s+CH] = S.max(1).values.cpu().numpy()
    del cur, prv, scores, S
gt = -his[pairs][:, 25]                      # displacement over the last 0.5 s, current ego frame
mx = -meas[:, 0]*CX; my = -meas[:, 1]*CY     # BEV content shifts backward by the ego displacement
gm = np.linalg.norm(gt, axis=1); mm = np.hypot(mx, my)
print("\n=== BEV correlation vs true 0.5 s displacement ===")
print("  true  displacement: mean %.2f m  std %.2f" % (gm.mean(), gm.std()))
print("  measured (BEV)    : mean %.2f m  std %.2f" % (mm.mean(), mm.std()))
print("  corr(measured, true) = %.4f" % np.corrcoef(mm, gm)[0, 1])
print("  displacement MAE  = %.3f m   -> implied speed MAE %.3f m/s" % (
    np.abs(mm-gm).mean(), np.abs(mm-gm).mean()/0.5))
a, b = np.polyfit(mm, gm, 1)
res = gm-(a*mm+b)
print("  after a global linear calibration (scale %.3f, bias %+.3f):" % (a, b))
print("    displacement MAE %.3f m  -> implied speed MAE %.3f m/s" % (
    np.abs(res).mean(), np.abs(res).mean()/0.5))
print("\n  reference: current v0 MAE 0.264 | need <=0.254 to tie direct, <=0.162 for D3 0.20")
