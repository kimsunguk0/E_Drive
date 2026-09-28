#!/usr/bin/env python3
"""Run on the original B200 checkout: reference predictions for validation."""
import json, shutil, sys, zipfile
from pathlib import Path
import numpy as np, torch
ROOT = Path('/NHNHOME/data/sukim/adcl'); REL = ROOT / 'release/etri_e2e_final'
sys.path.insert(0, str(ROOT / 'experiments/a2_final_push_20260923/ext_v6'))
from build_ext_submission import load, predict
import infer_full
infer_full.configure()
model, _ = load(str(ROOT / 'work_dirs/a2_ext_select_20260923/EXT-FULL-v7/ckpt_step6344.pth'), torch.device('cuda:0'))
V = REL / 'validation'; (V / 'test_subset').mkdir(parents=True, exist_ok=True)
fx = sorted(p for p in (ROOT / 'data/etri/motiondrive_v2/deploy_fixture_train8').iterdir() if p.is_dir())
pred, mode = {}, {}
for c in fx:
    p, m = predict(model, infer_full.prepare_clip(c)); pred[c.name] = p.tolist(); mode[c.name] = m
(V / 'reference_fixtures_B200_bf16.json').write_text(json.dumps(dict(pred=pred, mode=mode, device=torch.cuda.get_device_name(0))) + '\n')
sub = json.loads(zipfile.ZipFile(ROOT / 'reports/a2_final_push_20260923/sub_EXT-FULL-v7/package/submission.zip').read('submission.json'))
modes = {}
for s in (ROOT / 'reports/a2_final_push_20260923/sub_EXT-FULL-v7').glob('shard_*.json'):
    modes.update(json.loads(s.read_text())['mode'])
clips = sorted(p for p in Path('/tmp/etri_test').iterdir() if p.is_dir())[:20]
for c in clips:
    shutil.copytree(c, V / 'test_subset' / c.name, dirs_exist_ok=True)
(V / 'reference_test_submitted.json').write_text(json.dumps(dict(
    pred={c.name: sub[c.name] for c in clips}, mode={c.name: modes[c.name] for c in clips},
    source='the submitted EXT-FULL-v7 submission.zip (server 0.117067)')) + '\n')
print('fixtures', len(fx), 'test clips', len(clips))
