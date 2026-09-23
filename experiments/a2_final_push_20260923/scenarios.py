"""Held-out V0 error by driving situation (GT-defined bins, analysis only)."""
import numpy as np
W = np.array([11, 11, 5, 5, 2, 2]) / 36.0
e = np.load('/tmp/pm97/data/etri/ego_cache.npz', allow_pickle=True)
v = np.load('/NHNHOME/data/sukim/adcl/work_dirs/a2_ext_select_20260923/EXT-TWIN-v10/v0_step5230.npz')
p = np.load('/NHNHOME/data/sukim/adcl/reports/a2_final_push_20260923/views/L34.npz')
rows, gt = v['rows'], v['gt']; m = v['modes']
ext = m[np.arange(len(rows)), v['picks'], :6]
per_m = np.linalg.norm(m[:, :, :6] - gt[:, None], axis=-1) @ W
sc = lambda x: np.linalg.norm(x - gt, axis=-1) @ W
par, ex, orc = sc(p['up']), sc(ext), per_m.min(1)
seg = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(gt), 1, 2)), gt], 1), axis=1), axis=-1) / .5
v0 = e['speed'][rows]; vend = seg[:, -1]; a3 = (seg[:, -1] - seg[:, 0]) / 2.5
d_last = gt[:, -1] - gt[:, -2]; head = np.degrees(np.arctan2(d_last[:, 1], d_last[:, 0])) * (seg[:, -1] > 0.5)
meta = e['meta'][rows]; labels = list(e['meta_labels'])
cat = np.full(len(rows), '', dtype=object)
rules = [
    ('정지 유지', (v0 < 0.5) & (vend < 0.5)),
    ('출발 (정지→주행)', (v0 < 1.5) & (vend - v0 > 1.5)),
    ('정지하러 감속', (v0 > 2.0) & (vend < 1.0)),
    ('좌회전', head >= 15), ('우회전', head <= -15),
    ('차선변경', np.isin(meta, [labels.index('LANE_CHANGE_L'), labels.index('LANE_CHANGE_R')])),
    ('감속 (a<-0.5)', a3 < -0.5), ('가속 (a>0.5)', a3 > 0.5),
    ('완만한 속도변화 (0.2<|a|<=0.5)', np.abs(a3) > 0.2),
    ('정속 직진 (|a|<=0.2)', np.ones(len(rows), bool)),
]
for name, mask in rules:
    sel = (cat == '') & mask; cat[sel] = name
tot = ex.sum()
print(f'{"상황":<30s}{"n":>5s}{"부모":>8s}{"EXT":>8s}{"oracle":>8s}{"EXT개선":>9s}{"EXT오차몫":>9s}')
for name, _ in rules:
    s = cat == name
    if not s.any(): continue
    print(f'{name:<30s}{s.sum():5d}{par[s].mean():8.3f}{ex[s].mean():8.3f}{orc[s].mean():8.3f}{(ex[s].mean()/par[s].mean()-1)*100:8.1f}%{ex[s].sum()/tot*100:8.1f}%')
print(f'{"전체":<30s}{len(rows):5d}{par.mean():8.3f}{ex.mean():8.3f}{orc.mean():8.3f}{(ex.mean()/par.mean()-1)*100:8.1f}%{100:8.1f}%')
