"""Compare EXT v7 re-runs with the originals: held-out twin V0 (overall,
stationary, moving), FULL in-fit curve, and test predictions vs submission."""
import json, zipfile, numpy as np
from pathlib import Path
R = Path('/NHNHOME/data/sukim/adcl')
W = np.array([11, 11, 5, 5, 2, 2]) / 36.0
e = np.load('/tmp/pm97/data/etri/ego_cache.npz', allow_pickle=True)
def curve(p):
    return [(d['step'], round(d['PREFIX'], 6)) for d in (json.loads(l) for l in open(p) if l.startswith("{")) if d.get("kind") == "eval"]
def split(npz):
    z = np.load(npz); gt, rows = z['gt'], z['rows']
    sel = np.linalg.norm(z['modes'][np.arange(len(rows)), z['picks'], :6] - gt, axis=-1) @ W
    s = np.linalg.norm(e['goal'][rows], axis=-1) < 5
    return dict(V0=round(float(sel.mean()), 6), stat=round(float(sel[s].mean()), 6), moving=round(float(sel[~s].mean()), 6),
                mix29=round(float(0.29 * sel[s].mean() + 0.71 * sel[~s].mean()), 6))
E = R / 'work_dirs/a2_ext_select_20260923'
print('twin original', curve(R / 'reports/a2_ext_select_20260923/twin_v7.log'))
print('twin re-run  ', curve(R / 'reports/a2_repro_20260923/twin_v7r.log'))
print('twin original split', split(E / 'EXT-TWIN-v7/v0_step5230.npz'))
print('twin re-run   split', split(E / 'EXT-TWIN-v7r/v0_step5230.npz'))
print('full original (in-fit)', curve(R / 'reports/a2_ext_select_20260923/full_v7.log'))
print('full re-run   (in-fit)', curve(R / 'reports/a2_repro_20260923/full_v7r.log'))
sub = json.loads(zipfile.ZipFile(R / 'reports/a2_final_push_20260923/sub_EXT-FULL-v7/package/submission.zip').read('submission.json'))
new = json.loads((R / 'reports/a2_repro_20260923/sub_EXT-FULL-v7r/predictions.json').read_text())
k = sorted(x for x in sub if x != '__flops__')
a, b = np.array([sub[x] for x in k]), np.array([new[x] for x in k])
d = np.linalg.norm(a - b, axis=-1) @ W
print('test predictions re-run vs submitted: weighted mean %.4f m, median %.4f, p90 %.4f, identical clips %d/%d'
      % (d.mean(), np.median(d), np.quantile(d, .9), int((d == 0).sum()), len(k)))
