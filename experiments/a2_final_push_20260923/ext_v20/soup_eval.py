"""Held-out V0 of a uniform weight soup of ExtModel siblings (same parent, same
deterministic mode initialisation, different data order), plus each member.
Usage: soup_eval.py CKPT_A CKPT_B [...]   (EXT_* env must match the members)
With --save OUT.pth the soup is written as a checkpoint the packager can load."""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_ext as te
from build_ext_submission import load

ap = argparse.ArgumentParser(); ap.add_argument('ckpts', nargs='+'); ap.add_argument('--save'); ap.add_argument('--full', action='store_true')
a = ap.parse_args()
dev = torch.device('cuda:0')
model, cp = load(a.ckpts[0], dev)
states = [torch.load(p, map_location='cpu', weights_only=False)['model'] for p in a.ckpts]
soup = {k: (torch.stack([s[k].double() for s in states]).mean(0).to(v.dtype) if v.is_floating_point() else v)
        for k, v in states[0].items()}
diff = {k for k, v in states[0].items() if not all(torch.equal(v, s[k]) for s in states[1:])}
print('tensors that differ between members:', len(diff), 'all in planner:', all(k.startswith('planner.') for k in diff))
model.load_state_dict(soup, strict=True)
if a.save:
    torch.save(dict(model=model.state_dict(), step=cp['step'], manifest=dict(cp['manifest'], soup_of=a.ckpts)), a.save)
    print('saved', a.save)
if a.full:
    sys.exit(0)
raw_train, raw_tune = te.base.nominal.raw_datasets(False, 1)
val = torch.utils.data.DataLoader(te.base.wrapped_eval(raw_tune, False), batch_size=8, shuffle=False, num_workers=6, pin_memory=True)
recs = json.loads((te.ROOT / 'work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1/final_eval.json').read_text())['records']
sbr = {int(r['row']): r['session'] for r in recs}
pv = np.load(te.PARENT_UPRIGHT_VIEW, allow_pickle=True)
res, _ = te.evaluate(model, val, dev, sbr, pv['rows'], np.linalg.norm(pv['up'] - pv['gt'], axis=-1) @ te.W6N)
print(json.dumps({k: res[k] for k in ('PREFIX', 'oracle', 'delta', 'ci', 'sessions_improved', 'ext_5s_err')}))
