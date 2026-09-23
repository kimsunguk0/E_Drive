"""Held-out check: vad_cmd filtering of the image candidates, then goal-endpoint
argmin. Both are organiser-allowed selection signals; coordinates unchanged."""
import numpy as np, json
from pathlib import Path
W = np.array([11, 11, 5, 5, 2, 2]) / 36.0
e = np.load('/tmp/pm97/data/etri/ego_cache.npz', allow_pickle=True)
R = Path('/NHNHOME/data/sukim/adcl/work_dirs/a2_ext_select_20260923')
rec = json.loads(Path('/NHNHOME/data/sukim/adcl/work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1/final_eval.json').read_text())['records']
r2s = {int(r['row']): r['session'] for r in rec}
def cls(y):
    return np.where(y <= -2, 0, np.where(y >= 2, 1, 2))
for tag in ('EXT-TWIN-s1', 'EXT-TWIN-v7', 'EXT-TWIN-v6', 'EXT-TWIN-v4'):
    f = sorted((R / tag).glob('v0_step*.npz'), key=lambda p: int(p.stem.split('step')[1]))[-1]
    d = np.load(f); m, gt, rows = d['modes'], d['gt'], d['rows']
    goal = e['goal'][rows]; cmd = e['vad_cmd'][rows].astype(int); sess = np.array([r2s[int(r)] for r in rows])
    dist = np.linalg.norm(m[:, :, -1] - goal[:, None], axis=-1)
    base_pick = dist.argmin(1)
    ok = cls(m[:, :, 5, 1]) == cmd[:, None]
    anyok = ok.any(1)
    filt = np.where(ok, dist, np.inf); pick = np.where(anyok, filt.argmin(1), base_pick)
    sc = lambda p: np.linalg.norm(m[np.arange(len(rows)), p, :6] - gt, axis=-1) @ W
    b, c = sc(base_pick), sc(pick)
    imp = sum((c - b)[sess == s].mean() < 0 for s in set(sess))
    changed = (pick != base_pick)
    print(f'{tag:12s} {f.stem}: goal-only {b.mean():.6f}  cmd-filter+goal {c.mean():.6f}  ({c.mean()-b.mean():+.6f}, sessions {imp}/11)  '
          f'picks changed {changed.sum()} rows, rows with no matching candidate {(~anyok).sum()}')
    for k, name in enumerate(('right', 'left', 'straight')):
        s = cmd == k
        print(f'    {name:8s} n={s.sum():4d}  goal-only {b[s].mean():.4f} -> filtered {c[s].mean():.4f}')
