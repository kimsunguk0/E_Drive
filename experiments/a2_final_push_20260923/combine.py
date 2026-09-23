import numpy as np, sys, itertools
from pathlib import Path
V = Path('/NHNHOME/data/sukim/adcl/reports/a2_final_push_20260923/views')
W = np.array([11, 11, 5, 5, 2, 2]) / 36.0
tags = sys.argv[1].split(',')
base_spec = sys.argv[2]            # e.g. L34:up
data = {t: np.load(V / f'{t}.npz', allow_pickle=True) for t in tags}
first = data[tags[0]]
rows, sess, gt = first['rows'], first['sessions'], first['gt']
for t in tags:
    assert np.array_equal(data[t]['rows'], rows)

def views(spec):
    out = []
    for part in spec.split('+'):
        t, v = part.split(':')
        if v in ('up', 'both'): out.append(data[t]['up'])
        if v in ('mir', 'both'): out.append(data[t]['mir'])
    return np.mean(out, 0), len(out)

def d3(p): return np.linalg.norm(p - gt, axis=-1) @ W
uniq = sorted(set(sess.tolist())); idx = {s: np.flatnonzero(sess == s) for s in uniq}
rng = np.random.default_rng(0)
picks = [np.concatenate([idx[uniq[j]] for j in r]) for r in rng.integers(0, len(uniq), (20000, len(uniq)))]
bp, _ = views(base_spec); bd = d3(bp)
print('baseline %-24s %.6f' % (base_spec, bd.mean()))
for spec in sys.argv[3:]:
    p, n = views(spec); dd = d3(p); diff = dd - bd
    m = np.array([diff[k].mean() for k in picks]); lo, hi = np.percentile(m, [2.5, 97.5])
    imp = sum(1 for s in uniq if diff[idx[s]].mean() < 0)
    print('%-34s fwd=%d  %.6f  %+.6f  CI[%+.6f,%+.6f] %2d/11%s'
          % (spec, n, dd.mean(), diff.mean(), lo, hi, imp, '' if lo <= 0 <= hi else '  **'))
