#!/bin/bash
# After the L-FULL6 reproduction completes: EXT-v7 stage 2 on the reproduced trunk,
# package, and compare with the submitted model. Runs detached on the server.
cd /NHNHOME/data/sukim/adcl
E=experiments/a2_final_push_20260923; R=reports/a2_repro_20260923
until [ -f $R/lfull6/completion_L-FULL6.json ] || [ -f $R/lfull6/failure_L-FULL6.json ]; do sleep 120; done
[ -f $R/lfull6/failure_L-FULL6.json ] && { echo "trunk reproduction failed" > $R/chain_status.txt; exit 1; }
mkdir -p $E/ext_v7rr && cp $E/ext_v6/*.py $E/ext_v7rr/
python3 - <<PY
p="$E/ext_v7rr/train_ext.py"; s=open(p).read()
s=s.replace("           'full_lenw':", "           'full_repro': 'work_dirs/a2_repro_20260923/L-FULL6-s1/ckpt_step38070.pth',\n           'full_lenw':",1)
open(p,"w").write(s)
PY
V7="EXT_K=5 EXT_LAT=3 EXT_SPREAD=0.06 EXT_LAT_SPREAD=0.04 EXT_ALL_W=0 EXT_EXT_W=0.3"
env $V7 CUDA_VISIBLE_DEVICES=3 ~/cv2env/bin/python $E/ext_v7rr/train_ext.py --arm full_repro --gpu 3 --tag v7rr > $R/full_v7rr.log 2>&1 || { echo "EXT stage failed" > $R/chain_status.txt; exit 1; }
env EXT_K=5 EXT_LAT=3 EXT_SPREAD=0.06 EXT_LAT_SPREAD=0.04 bash $E/pack_ext.sh work_dirs/a2_ext_select_20260923/EXT-FULL_REPRO-v7rr/ckpt_step6344.pth $R/sub_EXT-FULL-v7rr EXT-FULL-v7rr $E/ext_v6 > $R/pack_v7rr.log 2>&1
~/cv2env/bin/python - > $R/chain_compare.txt 2>&1 <<PY
import json, zipfile, numpy as np
from pathlib import Path
R = Path('/NHNHOME/data/sukim/adcl'); W = np.array([11,11,5,5,2,2])/36.
o = json.loads((R/'reports/a2_long_motion_20260922/curve_L-FULL6.json').read_text())
n = json.loads((R/'reports/a2_repro_20260923/lfull6/curve_L-FULL6.json').read_text())
print('L-FULL6 in-fit curve original', [(r['step'], round(r['PREFIX'],6)) for r in o])
print('L-FULL6 in-fit curve repro   ', [(r['step'], round(r['PREFIX'],6)) for r in n])
ev = lambda p: [(d['step'], round(d['PREFIX'],6)) for d in (json.loads(l) for l in open(p) if l.startswith('{')) if d.get('kind')=='eval']
print('EXT full in-fit original', ev(R/'reports/a2_ext_select_20260923/full_v7.log'))
print('EXT full in-fit repro   ', ev(R/'reports/a2_repro_20260923/full_v7rr.log'))
sub = json.loads(zipfile.ZipFile(R/'reports/a2_final_push_20260923/sub_EXT-FULL-v7/package/submission.zip').read('submission.json'))
new = json.loads((R/'reports/a2_repro_20260923/sub_EXT-FULL-v7rr/predictions.json').read_text())
k = sorted(x for x in sub if x != '__flops__'); a = np.array([sub[x] for x in k]); b = np.array([new[x] for x in k])
d = np.linalg.norm(a-b, axis=-1) @ W
print('test predictions, full two-stage repro vs submitted: weighted mean %.4f m, median %.4f, p90 %.4f' % (d.mean(), np.median(d), np.quantile(d,.9)))
PY
echo done > $R/chain_status.txt
