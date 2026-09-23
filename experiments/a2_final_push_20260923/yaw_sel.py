"""Held-out check: goal-pose selection = endpoint distance + lambda * heading gap
at 5 s, over the twin's image candidates (coordinates unchanged). LOSO for lambda."""
import numpy as np, json
from pathlib import Path
W = np.array([11, 11, 5, 5, 2, 2]) / 36.0
e = np.load('/tmp/pm97/data/etri/ego_cache.npz', allow_pickle=True)
rec = json.loads(Path('/NHNHOME/data/sukim/adcl/work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1/final_eval.json').read_text())['records']
r2s = {int(r['row']): r['session'] for r in rec}
for tag in ('EXT-TWIN-v10', 'EXT-TWIN-v7', 'EXT-TWIN-v12'):
    d = np.load(f'/NHNHOME/data/sukim/adcl/work_dirs/a2_ext_select_20260923/{tag}/v0_step5230.npz')
    m, gt, rows = d['modes'], d['gt'], d['rows']; n = len(rows)
    goal, gyaw = e['goal'][rows], e['goal_yaw'][rows]; sess = np.array([r2s[int(r)] for r in rows])
    seg = m[:, :, -1] - m[:, :, -2]; hyaw = np.arctan2(seg[..., 1], seg[..., 0])
    moving = np.linalg.norm(seg, axis=-1) > 0.2
    dpos = np.linalg.norm(m[:, :, -1] - goal[:, None], axis=-1)
    dyaw = np.abs(np.angle(np.exp(1j * (hyaw - gyaw[:, None])))) * moving
    per = np.linalg.norm(m[:, :, :6] - gt[:, None], axis=-1) @ W
    lams = [0, 1, 2, 3, 5, 8, 12, 20]
    sc = {l: per[np.arange(n), (dpos + l * dyaw).argmin(1)] for l in lams}
    print(tag, ' '.join(f'l={l}:{sc[l].mean():.5f}' for l in lams))
    out = np.zeros(n); chosen = []
    for s in set(sess):
        tr = sess != s
        best = min(lams, key=lambda l: sc[l][tr].mean()); chosen.append(best); out[~tr] = sc[best][~tr]
    imp = sum((out - sc[0])[sess == s].mean() < 0 for s in set(sess))
    print(f'   LOSO lambda {sorted(set(chosen))}: {out.mean():.6f} vs position-only {sc[0].mean():.6f} ({out.mean()-sc[0].mean():+.6f}, {imp}/11)')
