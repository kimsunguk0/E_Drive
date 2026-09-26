#!/usr/bin/env python3
"""Checkpoint cleanup PLAN (read-only; deletes nothing).

KEEP  = closure of every root below over recorded parent/initializer paths
        (checkpoint manifests + run experiment.json/manifest.json/protocol).
TIER1 = safe: smoke runs, and intermediate step checkpoints (not the last step
        of their run) that are not in KEEP.
TIER2 = final checkpoints of runs outside KEEP (finished experiments whose
        results are already recorded in reports/).
Writes reports/cleanup_20260926/plan.json and prints a summary.
"""
import json, re, sys
from pathlib import Path
import torch

ROOT = Path('/NHNHOME/data/sukim/adcl')
W = ROOT / 'work_dirs'
ROOTS = [
    # the five scored submissions
    'work_dirs/md_r0_reset_20260914/MR-NATIVE-s1/ckpt_step20554.pth',
    'work_dirs/a2_progress_full_20260921/A2-H4-PROGRESS-FULL-s1/ckpt_step24931.pth',
    'work_dirs/a2_ext_select_20260923/EXT-FULL-s1/ckpt_step6344.pth',
    'work_dirs/a2_ext_select_20260923/EXT-FULL-v7/ckpt_step6344.pth',
    # held-out twins that justified the final choice, the alternate package, the can_bus lead
    'work_dirs/a2_ext_select_20260923/EXT-TWIN-v7/ckpt_step5230.pth',
    'work_dirs/a2_ext_select_20260923/EXT-TWIN-s1/ckpt_step5230.pth',
    'work_dirs/a2_ext_select_20260923/EXT-TWIN-v10/ckpt_step5230.pth',
    'work_dirs/a2_ext_select_20260923/EXT-FULL-v10/ckpt_step6344.pth',
    'work_dirs/a2_ext_select_20260923/EXT-TWIN-cb/ckpt_step5230.pth',
] + sys.argv[1:]
PTH = re.compile(r'(/NHNHOME/[^\s"\']+?\.pth|(?:work_dirs|ckpt|checkpoints)/[^\s"\']+?\.pth)')

def strings(obj, out):
    if isinstance(obj, dict):
        for v in obj.values(): strings(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj: strings(v, out)
    elif isinstance(obj, str):
        out.extend(PTH.findall(obj))
    return out

def resolve(s):
    p = Path(s) if s.startswith('/') else ROOT / s
    return p.resolve() if p.exists() else None

def parents(ck):
    found = []
    try:
        cp = torch.load(ck, map_location='cpu', weights_only=False)
        strings({k: v for k, v in cp.items() if k not in ('model', 'optimizer', 'rng', 'training_aux')}, found)
    except Exception as e:
        print('  (manifest unreadable)', ck, repr(e)[:80])
    for side in ('experiment.json', 'manifest.json'):
        f = ck.parent / side
        if f.exists():
            try: strings(json.loads(f.read_text()), found)
            except Exception: pass
    out = set()
    for s in found:
        p = resolve(s)
        if p and p != ck.resolve(): out.add(p)
    return out

keep, todo = set(), [resolve(r) for r in ROOTS]
missing = [r for r, p in zip(ROOTS, todo) if p is None]
todo = [p for p in todo if p]
while todo:
    ck = todo.pop()
    if ck in keep: continue
    keep.add(ck)
    for p in parents(ck):
        if p not in keep: todo.append(p)

allp = [p.resolve() for d in (W, ROOT / 'ckpt', ROOT / 'checkpoints') if d.exists() for p in d.rglob('*.pth')]
step = lambda p: int(m.group(1)) if (m := re.search(r'ckpt_step(\d+)\.pth$', p.name)) else None
last_in_dir = {}
for p in allp:
    s = step(p)
    if s is not None: last_in_dir[p.parent] = max(last_in_dir.get(p.parent, -1), s)
tier1, tier2, other = [], [], []
for p in allp:
    if p in keep: continue
    smoke = 'smoke' in str(p.relative_to(ROOT)).lower()
    s = step(p)
    if smoke or (s is not None and s < last_in_dir[p.parent]):
        tier1.append(p)
    elif str(p).startswith(str(W)):
        tier2.append(p)
    else:
        other.append(p)   # ckpt/ and checkpoints/ not in any lineage: report only
size = lambda L: sum(x.stat().st_size for x in L) / 2**30
out = ROOT / 'reports/cleanup_20260926'; out.mkdir(parents=True, exist_ok=True)
plan = dict(roots=ROOTS, missing_roots=missing,
            keep=sorted(map(str, keep)), tier1=sorted(map(str, tier1)), tier2=sorted(map(str, tier2)),
            other_not_in_lineage=sorted(map(str, other)),
            sizes_gib=dict(keep=size(list(keep)), tier1=size(tier1), tier2=size(tier2), other=size(other)))
(out / 'plan.json').write_text(json.dumps(plan, indent=1) + '\n')
print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in plan['sizes_gib'].items()}))
print('counts', dict(keep=len(keep), tier1=len(tier1), tier2=len(tier2), other=len(other)), 'missing roots', missing)
print('KEEP lineage:'); [print('  ', p.relative_to(ROOT)) for p in sorted(keep)]
