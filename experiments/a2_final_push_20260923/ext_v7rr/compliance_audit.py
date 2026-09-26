#!/usr/bin/env python3
"""Compliance audit of the submitted model (EXT-FULL-v7) and its held-out twin.

Empirical checks, each mapped to an organiser rule in OPEN_ISSUE.md / the
2026-09-23 open-chat answers. Nothing here changes a model or a submission.
Run with EXT_K=5 EXT_LAT=3 EXT_SPREAD=0.06 EXT_LAT_SPREAD=0.04 from ext_v6/.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_ext as te
import infer_full
from build_ext_submission import load
from ext_model import select_by_goal

ROOT = te.ROOT
FULL = ROOT / 'work_dirs/a2_ext_select_20260923/EXT-FULL-v7/ckpt_step6344.pth'
TWIN = ROOT / 'work_dirs/a2_ext_select_20260923/EXT-TWIN-v7/ckpt_step5230.pth'
W = np.array([11, 11, 5, 5, 2, 2]) / 36.0
dev = torch.device('cuda:0')
out = {}

def fwd(model, x):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        return model(**x)

# ---------------- test clips, submitted model ----------------
infer_full.configure()
model, _ = load(str(FULL), dev)
clips = sorted(p for p in Path('/tmp/etri_test').iterdir() if p.is_dir())[:64]
T1 = T2c = T2s = T3 = 0; n = 0; goal_shift_cand = []; status_shift = []
for c in clips:
    prep = infer_full.prepare_clip(c)
    x = {k: v.to(dev) for k, v in prep.inputs.items()}
    o = fwd(model, x); modes = o['plan_modes'].float(); pick = o['selected_mode']
    # T1: output == selected candidate's first six points, bitwise; selection == pure argmin
    T1 += int(torch.equal(o['plan_abs'], o['plan_modes'][torch.arange(1, device=dev), pick][:, :6]))
    T1 += int(torch.equal(pick, select_by_goal(o['plan_modes'], x['goal_xy'])))
    # T2: goal counterfactual (+5 m forward): candidates move only through the scene query; selection moves
    xg = dict(x, goal_xy=x['goal_xy'] + torch.tensor([[5.0, 0.0]], device=dev))
    og = fwd(model, xg)
    goal_shift_cand.append(float((og['plan_modes'].float() - modes)[:, :, :6].norm(dim=-1).mean()))
    T2s += int(not torch.equal(og['selected_mode'], pick))
    # T3: status counterfactual (zero status): indirect effect size
    xs = dict(x, provided_status5=torch.zeros_like(x['provided_status5']))
    os_ = fwd(model, xs)
    status_shift.append(float((os_['plan_abs'].float() - o['plan_abs'].float()).norm(dim=-1).mean()))
    n += 1
out['T1_output_is_selected_candidate_unchanged_and_argmin'] = f'{T1}/{2*n}'
out['T2_goal+5m: mean candidate shift (m, via scene query)'] = float(np.mean(goal_shift_cand))
out['T2_goal+5m: clips whose selected index changed'] = f'{T2s}/{n}'
out['T3_status_zeroed: mean output shift (m)'] = float(np.mean(status_shift))

# T5: gradient path image -> output (one network, gradient connected)
prep = infer_full.prepare_clip(clips[0]); x = {k: v.to(dev) for k, v in prep.inputs.items()}
for k in ('images', 'history_images', 'motion_current', 'motion_history'):
    x[k] = x[k].clone().requires_grad_(True)
model.zero_grad(set_to_none=True)
with torch.autocast('cuda', dtype=torch.bfloat16):
    o = model(**x)
o['plan_abs'].float().sum().backward()
out['T5_grad_norm d(output)/d(input)'] = {k: float(x[k].grad.float().norm()) for k in ('images', 'history_images', 'motion_current', 'motion_history')}

# T6: timing, one full forward (B200 BF16; organisers time on RTX 4090)
x = {k: v.to(dev) for k, v in prep.inputs.items()}
for _ in range(20): fwd(model, x)
torch.cuda.synchronize(); ts = []
for _ in range(100):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record(); fwd(model, x); e.record(); e.synchronize(); ts.append(s.elapsed_time(e))
out['T6_B200_bf16_forward_ms_median'] = float(np.median(ts))
del model; torch.cuda.empty_cache()

# ---------------- held-out V0, twin model: image contribution ----------------
twin, _ = load(str(TWIN), dev)
raw_train, raw_tune = te.base.nominal.raw_datasets(False, 1)
ds = te.base.wrapped_eval(raw_tune, False)
idx = list(range(0, len(ds), 4))
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=8, shuffle=False, num_workers=6)
res = {'normal': [], 'images_gray': [], 'status_zero': [], 'goal_zero': []}; gts = []
for b in loader:
    x = te.inputs(b, dev); gts.append(b['gt_plan'].float())
    res['normal'].append(fwd(twin, x)['plan_abs'].float().cpu())
    xi = dict(x); [xi.__setitem__(k, torch.zeros_like(x[k])) for k in ('images', 'history_images', 'motion_current', 'motion_history')]
    res['images_gray'].append(fwd(twin, xi)['plan_abs'].float().cpu())
    res['status_zero'].append(fwd(twin, dict(x, provided_status5=torch.zeros_like(x['provided_status5'])))['plan_abs'].float().cpu())
    res['goal_zero'].append(fwd(twin, dict(x, goal_xy=torch.zeros_like(x['goal_xy'])))['plan_abs'].float().cpu())
gt = torch.cat(gts).numpy()
out['T4_V0_twin_PREFIX (n=%d)' % len(gt)] = {k: float((np.linalg.norm(torch.cat(v).numpy() - gt, axis=-1) @ W).mean()) for k, v in res.items()}

# T7: near-stationary share (goal < 5 m) in train / V0 / test inputs
e = np.load('/tmp/pm97/data/etri/ego_cache.npz', allow_pickle=True)
full_rows = np.asarray(te.base.nominal.raw_datasets(True, 1)[0].rows)
tg = []
import pyarrow.parquet as pq
for c in sorted(Path('/tmp/etri_test').iterdir()):
    t = pq.read_table(c / 'ego_pose.parquet', columns=['frame', 'x', 'y']).to_pandas(); r = t[t.frame == 50].iloc[0]; tg.append((r.x, r.y))
share = lambda g: float((np.linalg.norm(g, axis=-1) < 5).mean())
out['T7_goal<5m share'] = dict(train_FULL=share(e['goal'][full_rows]), V0=share(e['goal'][np.asarray(raw_tune.rows)]), test=share(np.array(tg)))
print(json.dumps(out, indent=1, ensure_ascii=False))
