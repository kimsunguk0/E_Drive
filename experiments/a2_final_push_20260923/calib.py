import numpy as np
from pathlib import Path
V = Path('/NHNHOME/data/sukim/adcl/reports/a2_final_push_20260923/views')
W = np.array([11, 11, 5, 5, 2, 2]) / 36.0
def load(t):
    d = np.load(V / f'{t}.npz', allow_pickle=True)
    return d['gt'], (d['up'] + d['mir']) / 2, d['sessions']
def score(p, gt): return (np.linalg.norm(p - gt, axis=-1) @ W)
def fit_scale(p, gt, grid):
    best = None
    for sx in grid:
        for sy in grid:
            q = p.copy(); q[..., 0] *= sx; q[..., 1] *= sy
            s = score(q, gt).mean()
            if best is None or s < best[0]: best = (s, sx, sy)
    return best
grid = np.round(np.arange(0.95, 1.051, 0.0025), 4)
for tag in ['L34', 'W34', 'F22', 'SOUP_F22_G22']:
    gt, p, sess = load(tag)
    base = score(p, gt)
    uniq = sorted(set(sess))
    # leave-one-session-out
    out = np.zeros_like(base)
    fits = []
    for s in uniq:
        tr = sess != s; te = ~tr
        _, sx, sy = fit_scale(p[tr], gt[tr], grid)
        fits.append((sx, sy))
        q = p[te].copy(); q[..., 0] *= sx; q[..., 1] *= sy
        out[te] = score(q, gt[te])
    full = fit_scale(p, gt, grid)
    imp = sum(out[sess == s].mean() < base[sess == s].mean() for s in uniq)
    print(f'{tag:14s} base {base.mean():.6f} LOSO {out.mean():.6f} d {out.mean()-base.mean():+.6f} {imp}/11 fullfit sx={full[1]} sy={full[2]} ({full[0]:.6f}) loso-fits x[{min(f[0] for f in fits)},{max(f[0] for f in fits)}] y[{min(f[1] for f in fits)},{max(f[1] for f in fits)}]')
# transfer: fit on L34 apply to others
gt, p, _ = load('L34'); _, sx, sy = fit_scale(p, gt, grid)
for tag in ['W34', 'F22', 'SOUP_F22_G22']:
    g2, p2, _ = load(tag); q = p2.copy(); q[..., 0] *= sx; q[..., 1] *= sy
    print('transfer L34->', tag, f'{score(p2,g2).mean():.6f} -> {score(q,g2).mean():.6f}')
# bias diagnostics: mean signed error per horizon along x
gt, p, _ = load('L34')
print('mean x err per horizon', np.round((p - gt)[..., 0].mean(0), 4), 'mean |gt x|', np.round(gt[..., 0].mean(0), 3))
