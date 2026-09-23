"""Stationary/moving split for every finished EXT twin (analysis only)."""
import numpy as np
from pathlib import Path
W = np.array([11, 11, 5, 5, 2, 2]) / 36.0
e = np.load('/tmp/pm97/data/etri/ego_cache.npz', allow_pickle=True)
R = Path('/NHNHOME/data/sukim/adcl/work_dirs/a2_ext_select_20260923')
base = None; out = []
for d in sorted(R.glob('EXT-TWIN*')):
    f = d / 'v0_step5230.npz'
    if not f.exists():
        fs = sorted(d.glob('v0_step*.npz'), key=lambda p: int(p.stem.split('step')[1]))
        if not fs or '10460' not in fs[-1].stem: continue
        f = fs[-1]
    z = np.load(f); m, gt, rows = z['modes'], z['gt'], z['rows']
    goal = e['goal'][rows]; s = np.linalg.norm(goal, axis=-1) < 5
    sel = np.linalg.norm(m[np.arange(len(rows)), z['picks'], :6] - gt, axis=-1) @ W
    out.append((d.name, m.shape[1], sel.mean(), sel[s].mean(), sel[~s].mean(), 0.29 * sel[s].mean() + 0.71 * sel[~s].mean()))
out.sort(key=lambda r: r[5])
print(f'{"run":<24s}{"K":>4s}{"V0":>8s}{"stat":>8s}{"moving":>8s}{"29%mix":>8s}')
for r in out: print(f'{r[0]:<24s}{r[1]:4d}{r[2]:8.4f}{r[3]:8.4f}{r[4]:8.4f}{r[5]:8.4f}')
