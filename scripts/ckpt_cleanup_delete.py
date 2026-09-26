#!/usr/bin/env python3
"""Execute the approved cleanup: TIER1 + TIER2 minus the 14 notable finals.
Guards: nothing in KEEP/notable/other is touched; every KEEP file exists before
and after; aborts if any training process is running. Logs every deletion."""
import json, os, subprocess, sys
from pathlib import Path
ROOT = Path('/NHNHOME/data/sukim/adcl')
out = ROOT / 'reports/cleanup_20260926'
plan = json.loads((out / 'plan.json').read_text())
NOTABLE = ['A3-DIRECT', 'L-FULL6-LENW-s1/', 'P-NEARFULL-H', 'F-FLOW-s1', 'F-CTRL-s1',
           'L-TRAIN3106-LENW-s1/', 'EXT-FULL-v21', 'EXT-FULL-v22', 'EXT-FULL-v24']
notable = {f for f in plan['tier2'] if any(n in f for n in NOTABLE)}
keep = set(plan['keep']) | notable | set(plan['other_not_in_lineage'])
delete = [f for f in plan['tier1'] + plan['tier2'] if f not in keep]
assert not (set(delete) & keep), 'delete list overlaps keep'
assert len(notable) == 14, len(notable)
missing = [f for f in plan['keep'] if not Path(f).exists()]
assert not missing, missing
ps = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout
busy = [l for l in ps.splitlines() if any(t in l for t in ('train_ext.py', 'train_long.py', 'repro_lfull6.py', 'train_lenw', 'train_flow', 'train_h6'))
        and 'grep' not in l]
assert not busy, busy
log, freed, n = [], 0, 0
for f in delete:
    p = Path(f)
    if not p.exists(): continue
    s = p.stat().st_size
    p.unlink(); freed += s; n += 1; log.append(f'{s}\t{f}')
(out / 'deleted.txt').write_text('\n'.join(log) + '\n')
still = [f for f in plan['keep'] + sorted(notable) if not Path(f).exists()]
assert not still, still
summary = dict(deleted_files=n, freed_gib=round(freed / 2**30, 2), kept_lineage=len(plan['keep']),
               kept_notable=len(notable), kept_downloads=len(plan['other_not_in_lineage']))
(out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
print(json.dumps(summary))
