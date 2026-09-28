#!/usr/bin/env python3
"""Container start-up: rebuild the original repository layout from the clean release.

The training scripts pin source-file hashes and read inputs from their original
repository-relative paths. This script recreates those paths inside the
container (source files copied byte-identical, weights and caches symlinked),
so the delivered code runs unmodified. Nothing outside the container changes.
"""
import json, os, shutil, subprocess
from pathlib import Path

REL = Path('/release')
ROOT = Path('/NHNHOME/data/sukim/adcl')
PM97 = Path('/tmp/pm97')
layout = json.loads((REL / 'layout.json').read_text())
for clean, orig in layout['code'].items():
    dst = ROOT / orig; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(REL / clean, dst)
for group in ('checkpoints', 'data'):
    for clean, orig in layout[group].items():
        dst = ROOT / orig; dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink(): dst.unlink()
        if clean.endswith('.json'): shutil.copy2(REL / clean, dst)
        else: dst.symlink_to(REL / clean)
for clean, orig in layout['pm97'].items():
    dst = PM97 / orig; dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists(): dst.symlink_to(REL / clean)
# training outputs go to the mounted /out directory
out = Path('/out')
if out.is_dir():
    for orig, name in (('work_dirs/release_stage1', 'stage1'), ('reports/a2_progress_full_20260921', 'stage1_report'),
                       ('work_dirs/a2_repro_20260923', 'stage2'), ('reports/a2_repro_20260923', 'stage2_report'),
                       ('work_dirs/a2_ext_select_20260923/EXT-FULL-repro', 'stage3'),
                       ('reports/a2_ext_select_20260923', 'stage3_report')):
        (out / name).mkdir(parents=True, exist_ok=True)
        dst = ROOT / orig; dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists(): dst.symlink_to(out / name)
# the training scripts record `git rev-parse HEAD`
if not (ROOT / '.git').exists():
    run = lambda *a: subprocess.run(a, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    run('git', 'init', '-q'); run('git', 'add', '-A')
    run('git', '-c', 'user.name=release', '-c', 'user.email=release@local', 'commit', '-q', '-m', 'release layout')
print('layout ready:', ROOT)
