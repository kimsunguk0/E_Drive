"""Analysis: goal-only selection rule variants for near-stationary goals.
For |goal| < r, restrict the argmin to a fixed mode subset (lateral centre, or
slowest longitudinal rows). Coordinates unchanged. Report V0 mix and 29% mix,
plus leave-one-session-out choice of the rule to avoid tuning on V0."""
import numpy as np, json
from pathlib import Path
W = np.array([11, 11, 5, 5, 2, 2]) / 36.0
e = np.load('/tmp/pm97/data/etri/ego_cache.npz', allow_pickle=True)
R = Path('/NHNHOME/data/sukim/adcl/work_dirs/a2_ext_select_20260923')
rec = json.loads(Path('/NHNHOME/data/sukim/adcl/work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1/final_eval.json').read_text())['records']
r2s = {int(r['row']): r['session'] for r in rec}
for tag, KL, KT in (('EXT-TWIN-v10', 7, 5), ('EXT-TWIN-v7', 5, 3), ('EXT-TWIN-v10s2', 7, 5), ('EXT-TWIN-v15' if (R/'EXT-TWIN-v15').exists() else 'EXT-TWIN_LENW-v15', 7, 5)):
    z = np.load(R / tag / 'v0_step5230.npz'); m, gt, rows = z['modes'], z['gt'], z['rows']; n = len(rows)
    goal = e['goal'][rows]; sess = np.array([r2s[int(r)] for r in rows]); gd = np.linalg.norm(goal, axis=-1)
    per = np.linalg.norm(m[:, :, :6] - gt[:, None], axis=-1) @ W
    dist = np.linalg.norm(m[:, :, -1] - goal[:, None], axis=-1)
    lon = np.arange(KL * KT) // KT; lat = np.arange(KL * KT) % KT
    subsets = {'all': np.ones(KL * KT, bool), 'lat-centre': lat == KT // 2,
               'lat-centre3': np.abs(lat - KT // 2) <= 1, 'lon-slow-half': lon <= KL // 2}
    base = per[np.arange(n), dist.argmin(1)]; st = gd < 5
    print(f'== {tag}: base V0 {base.mean():.4f}  stat {base[st].mean():.4f}  moving {base[~st].mean():.4f}  29%mix {0.29*base[st].mean()+0.71*base[~st].mean():.4f}')
    for r in (3, 5, 8):
        for name, sub in subsets.items():
            if name == 'all': continue
            dd = np.where((gd[:, None] < r) & ~sub[None], np.inf, dist)
            s = per[np.arange(n), dd.argmin(1)]
            imp = sum((s - base)[sess == q].mean() < 0 for q in set(sess))
            print(f'   r<{r}m {name:<14s}: V0 {s.mean():.4f} ({s.mean()-base.mean():+.4f}, {imp}/11)  stat {s[st].mean():.4f}  29%mix {0.29*s[st].mean()+0.71*s[~st].mean():.4f}')
