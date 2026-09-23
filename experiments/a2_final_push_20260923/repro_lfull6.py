#!/usr/bin/env python3
"""Reproduce L-FULL6 (FULL parent 0.133685 -> +38,070 updates) with the exact
train_long recipe into separate directories; the original run is untouched.

usage: CUDA_VISIBLE_DEVICES=g repro_lfull6.py {smoke|verify|main}
Only source difference vs the original launch (commit ac62e26): the input
shape guard in WaypointTemporalRead (== 4 time grids -> >= 4); computation is
identical for the four-grid H4 graph."""
import os, sys
from pathlib import Path
ROOT = Path('/NHNHOME/data/sukim/adcl')
for p in (ROOT, ROOT / 'scripts', ROOT / 'experiments/a2_progress_h4_20260920',
          ROOT / 'experiments/md_r0_reset_20260914', ROOT / 'experiments/a2_long_motion_20260922'):
    sys.path.insert(0, str(p))
import train_long as base
NEW_R = ROOT / 'reports/a2_repro_20260923/lfull6'
NEW_W = ROOT / 'work_dirs/a2_repro_20260923'
(NEW_R / 'smoke').mkdir(parents=True, exist_ok=True)
mode = sys.argv[1]; gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
base.REPORT, base.RUNS = NEW_R, NEW_W
base.SPECS['L-FULL6']['gpu'] = gpu
if mode == 'verify':
    import verify_long as v
    v.REPORT, v.RUNS = NEW_R, NEW_W
    sys.argv = ['verify_long.py', '--arm', 'L-FULL6']; v.main()
else:
    sys.argv = ['train_long.py', '--arm', 'L-FULL6', '--gpu', str(gpu)] + (['--smoke'] if mode == 'smoke' else [])
    base.main()
