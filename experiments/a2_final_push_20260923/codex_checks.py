"""Codex-suggested checks on saved held-out candidates (analysis only).
1) oracle within the goal-nearest top-k; 2) anisotropic goal distance (fixed small family);
3) stationary vs moving comparison K=35 vs K=6."""
import numpy as np, json
from pathlib import Path
W = np.array([11, 11, 5, 5, 2, 2]) / 36.0
e = np.load('/tmp/pm97/data/etri/ego_cache.npz', allow_pickle=True)
R = Path('/NHNHOME/data/sukim/adcl/work_dirs/a2_ext_select_20260923')
rec = json.loads(Path('/NHNHOME/data/sukim/adcl/work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1/final_eval.json').read_text())['records']
r2s = {int(r['row']): r['session'] for r in rec}
runs = {}
for tag in ('EXT-TWIN-s1', 'EXT-TWIN-v10', 'EXT-TWIN-v10s2', 'EXT-TWIN-v21', 'EXT-TWIN-v20'):
    fs = sorted((R / tag).glob('v0_step*.npz'), key=lambda p: int(p.stem.split('step')[1]))
    if fs and fs[-1].stem.endswith('5230'): runs[tag] = np.load(fs[-1])
for tag, d in runs.items():
    m, gt, rows = d['modes'], d['gt'], d['rows']; n = len(rows); goal = e['goal'][rows]
    sess = np.array([r2s[int(r)] for r in rows])
    per = np.linalg.norm(m[:, :, :6] - gt[:, None], axis=-1) @ W
    dist = np.linalg.norm(m[:, :, -1] - goal[:, None], axis=-1); order = np.argsort(dist, 1)
    sel = per[np.arange(n), order[:, 0]]
    stat = (np.linalg.norm(goal, axis=-1) < 5)                      # near-zero 5 s displacement
    print(f'== {tag} (K={m.shape[1]}): selected {sel.mean():.4f}  oracle {per.min(1).mean():.4f}')
    print('   oracle within goal-nearest top-k:', '  '.join(f'k={k}:{np.take_along_axis(per, order[:, :k], 1).min(1).mean():.4f}' for k in (1, 2, 3, 5, 10) if k <= m.shape[1]))
    for lam in (0.5, 1, 2, 4):
        dd = (m[:, :, -1, 0] - goal[:, None, 0]) ** 2 + lam * (m[:, :, -1, 1] - goal[:, None, 1]) ** 2
        s2 = per[np.arange(n), dd.argmin(1)]
        imp = sum((s2 - sel)[sess == q].mean() < 0 for q in set(sess))
        print(f'   anisotropic lambda={lam}: {s2.mean():.4f} ({s2.mean()-sel.mean():+.5f}, better in {imp}/11 sessions)')
    print(f'   goal<5m (near-stationary) n={stat.sum()}: selected {sel[stat].mean():.4f} oracle {per[stat].min(1).mean():.4f} | moving n={(~stat).sum()}: selected {sel[~stat].mean():.4f}')
    runs[tag] = dict(sel=sel, stat=stat)
if 'EXT-TWIN-s1' in runs and 'EXT-TWIN-v10' in runs:
    a, b = runs['EXT-TWIN-v10'], runs['EXT-TWIN-s1']; s = a['stat']
    ds, dm = a['sel'][s].mean() - b['sel'][s].mean(), a['sel'][~s].mean() - b['sel'][~s].mean()
    print(f'\nK=35(v10) minus K=6(s1): stationary {ds:+.4f}  moving {dm:+.4f}  | at V0 mix {s.mean():.3f}: {s.mean()*ds+(1-s.mean())*dm:+.4f}  | sensitivity at 29% stationary: {0.29*ds+0.71*dm:+.4f}')
