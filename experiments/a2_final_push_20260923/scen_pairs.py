"""Held-out V0: per-situation effect of the temporal / other-agent experiments
against their matched controls (analysis only)."""
import json, numpy as np
from pathlib import Path
R = Path('/NHNHOME/data/sukim/adcl/work_dirs')
W = np.array([11, 11, 5, 5, 2, 2]) / 36.0
e = np.load('/tmp/pm97/data/etri/ego_cache.npz', allow_pickle=True)

def load(path):
    d = json.loads(Path(path).read_text())
    recs = d['records'] if 'records' in d else d.get('detailed_records') or d['report']['records']
    rows = np.array([int(r['row']) for r in recs])
    pred = np.array([r['pred_abs_xy'] for r in recs], float); gt = np.array([r['gt_abs_xy'] for r in recs], float)
    sess = np.array([r.get('session', '') for r in recs])
    o = np.argsort(rows); return rows[o], pred[o], gt[o], sess[o]

def last(d, pat='predictions_step'):
    fs = sorted(Path(d).glob(pat + '*.json'), key=lambda p: int(''.join(c for c in p.stem.split('step')[-1] if c.isdigit())))
    return fs[-1]

def cats(rows, gt):
    seg = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(gt), 1, 2)), gt], 1), axis=1), axis=-1) / .5
    v0 = e['speed'][rows]; vend = seg[:, -1]; a3 = (seg[:, -1] - seg[:, 0]) / 2.5
    dl = gt[:, -1] - gt[:, -2]; head = np.degrees(np.arctan2(dl[:, 1], dl[:, 0])) * (seg[:, -1] > 0.5)
    meta = e['meta'][rows]; L = list(e['meta_labels'])
    rules = [('정지 유지', (v0 < 0.5) & (vend < 0.5)), ('출발', (v0 < 1.5) & (vend - v0 > 1.5)),
             ('정지하러 감속', (v0 > 2.0) & (vend < 1.0)), ('좌회전', head >= 15), ('우회전', head <= -15),
             ('차선변경', np.isin(meta, [L.index('LANE_CHANGE_L'), L.index('LANE_CHANGE_R')])),
             ('감속', a3 < -0.5), ('가속', a3 > 0.5), ('완만한 변화', np.abs(a3) > 0.2), ('정속 직진', np.ones(len(rows), bool))]
    c = np.full(len(rows), '', dtype=object)
    for n, m in rules: c[(c == '') & m] = n
    return c, [n for n, _ in rules]

pairs = [
    ('AGENT (주변차량 미래궤적 감독)', last(R / 'a2_capacity_dynamics_20260922/C-AGENT-s1'), last(R / 'a2_capacity_dynamics_20260922/C-CTRL-s1')),
    ('H6-NEAR (과거 6시점, 가까움)', last(R / 'a2_h6_20260923/H6-NEAR-s1'), last(R / 'a2_long_motion_20260922/L-TRAIN3106-s1')),
    ('H6-LONG (과거 6시점, 3초까지)', last(R / 'a2_h6_20260923/H6-LONG-s1'), last(R / 'a2_long_motion_20260922/L-TRAIN3106-s1')),
    ('SIDE-SCENE (측방 과거영상)', R / 'md_a2_scene_extensions_20260918/A2-SIDE-SCENE-NOM-s1/final_eval.json', R / 'md_a2_nominal_mh4_20260918/A2-BASE-NOM-s1/final_eval.json'),
]
for name, fa, fc in pairs:
    try:
        ra, pa, ga, sa = load(fa); rc, pc, gc, sc_ = load(fc)
    except Exception as ex:
        print(name, 'load failed', fa.name, fc.name, repr(ex)[:120]); continue
    assert np.array_equal(ra, rc) and np.abs(ga - gc).max() < 1e-3
    da, dc = np.linalg.norm(pa - ga, axis=-1) @ W, np.linalg.norm(pc - gc, axis=-1) @ W
    c, order = cats(ra, ga)
    print(f'\n== {name}: {fa.parent.name}/{fa.name} vs {fc.parent.name}/{fc.name}  전체 {da.mean():.4f} vs {dc.mean():.4f} ({(da.mean()/dc.mean()-1)*100:+.1f}%)')
    for n in order:
        s = c == n
        if s.sum() < 5: continue
        sess_better = sum((da[s & (sa == q)].mean() < dc[s & (sa == q)].mean()) for q in set(sa[s]))
        print(f'   {n:<10s} n={s.sum():4d}  실험 {da[s].mean():.4f}  대조 {dc[s].mean():.4f}  {(da[s].mean()/dc[s].mean()-1)*100:+6.1f}%  (개선 세션 {sess_better}/{len(set(sa[s]))})')
