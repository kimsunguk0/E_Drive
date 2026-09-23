"""After the can_bus twin: accuracy vs v7 and compliance characterisation (analysis only)."""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path('/NHNHOME/data/sukim/adcl/experiments/a2_final_push_20260923/ext_canbus')
sys.path.insert(0, str(HERE))
import train_ext as te
from build_ext_submission import load
W = np.array([11, 11, 5, 5, 2, 2]) / 36.0
e = np.load('/tmp/pm97/data/etri/ego_cache.npz', allow_pickle=True)
R = te.ROOT / 'work_dirs/a2_ext_select_20260923'
def split(npz):
    z = np.load(npz); gt, rows = z['gt'], z['rows']
    sel = np.linalg.norm(z['modes'][np.arange(len(rows)), z['picks'], :6] - gt, axis=-1) @ W
    s = np.linalg.norm(e['goal'][rows], axis=-1) < 5
    return dict(V0=round(float(sel.mean()), 5), stat=round(float(sel[s].mean()), 5), moving=round(float(sel[~s].mean()), 5),
                mix29=round(float(0.29 * sel[s].mean() + 0.71 * sel[~s].mean()), 5))
out = {'v7 (A2 query-only)': split(R / 'EXT-TWIN-v7/v0_step5230.npz'), 'cb (VAD can_bus)': split(R / 'EXT-TWIN-cb/v0_step5230.npz')}
dev = torch.device('cuda:0')
model, _ = load(str(R / 'EXT-TWIN-cb/ckpt_step5230.pth'), dev)
raw_train, raw_tune = te.base.nominal.raw_datasets(False, 1)
ds = te.base.wrapped_eval(raw_tune, False)
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, list(range(0, len(ds), 4))), batch_size=8, num_workers=6)
res = {'normal': [], 'images_gray': [], 'status_other_row': []}; gts = []; inv = []
with torch.no_grad():
    for b in loader:
        x = te.inputs(b, dev); gts.append(b['gt_plan'].float())
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(**x); res['normal'].append(o['plan_abs'].float().cpu())
            xi = dict(x); [xi.__setitem__(k, torch.zeros_like(x[k])) for k in ('images', 'history_images', 'motion_current', 'motion_history')]
            res['images_gray'].append(model(**xi)['plan_abs'].float().cpu())
            xs = dict(x, provided_status5=x['provided_status5'].flip(0))
            os_ = model(**xs); res['status_other_row'].append(os_['plan_abs'].float().cpu())
            inv.append(max(float((o[k].float() - os_[k].float()).abs().max()) for k in ('motion_features', 'motion_pair_features', 'state_hat', 'history_hat')))
gt = torch.cat(gts).numpy()
out['cb ablation PREFIX (V0/4)'] = {k: round(float((np.linalg.norm(torch.cat(v).numpy() - gt, axis=-1) @ W).mean()), 4) for k, v in res.items()}
out['cb motion/state change under status swap (must be 0)'] = max(inv)
print(json.dumps(out, indent=1, ensure_ascii=False))
