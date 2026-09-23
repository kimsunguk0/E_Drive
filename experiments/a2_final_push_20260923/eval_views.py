#!/usr/bin/env python3
"""Evaluate one model (or a uniform weight average of several) on held-out V0,
upright and mirrored, and cache per-row predictions so any combination --
TTA, checkpoint ensembles, soups -- can be scored later without re-running.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/NHNHOME/data/sukim/adcl')
for p in (ROOT, ROOT / 'scripts', ROOT / 'experiments/a2_progress_h4_20260920',
          ROOT / 'experiments/a2_long_motion_20260922', ROOT / 'experiments/a2_progress_full_20260921',
          ROOT / 'experiments/md_r0_reset_20260914'):
    sys.path.insert(0, str(p))

import train_progress as dev
from h4_data import H4StatusDataset
from progress_model import H4ProgressModel, PROGRESS
from models.motiondrive_v2 import MotionDriveV2Config
from motiondrive_v2_flip_augment import flip_item
from motiondrive_v2_data import MotionDriveDataset

mr, trainer = dev.mr, dev.trainer
SPLIT = ROOT / 'data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json'
SUP = ROOT / 'data/etri/motiondrive_v2/r0reset_tplus_geometry_v2'
SESSION_SOURCE = ROOT / 'work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1/final_eval.json'
OUT = ROOT / 'reports/a2_final_push_20260923/views'


class Flipped(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
        self.flip = mr.wrap_flip_item(flip_item)

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        if name in {'base', 'flip'}:
            raise AttributeError(name)
        return getattr(self.base, name)

    def __getitem__(self, i):
        return self.flip(self.base[i], 768, 384)


def build_model(paths):
    states = []
    config = None
    for p in paths:
        cp = torch.load(p, map_location='cpu', weights_only=False)
        config = config or cp['manifest']['model_config']
        states.append(cp['model'])
    avg = {}
    buffer_mismatch = []
    for k, v in states[0].items():
        if v.is_floating_point():
            stacked = torch.stack([s[k].double() for s in states])
            avg[k] = stacked.mean(0).to(v.dtype)
            if len(states) > 1 and ('running_' in k) and not all(torch.equal(states[0][k], s[k]) for s in states[1:]):
                buffer_mismatch.append(k)
        else:
            if not all(torch.equal(v, s[k]) for s in states[1:]):
                buffer_mismatch.append(k)
            avg[k] = v.clone()
    model = H4ProgressModel(MotionDriveV2Config(**config), arm=PROGRESS)
    mr.rebuild_correlation_fuse(model, 4)
    model.load_state_dict(avg, strict=True)
    return model, buffer_mismatch


def run(model, dataset, device):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)
    plans, gts, rows = [], [], []
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for raw in loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in raw.items()}
            x = mr.model_inputs_with_canvas(trainer.model_inputs, batch, time_input='nominal')
            x['provided_status5'] = batch['provided_status5']
            plans.append(model(**x)['plan_abs'].float().cpu())
            gts.append(raw['gt_plan'].float())
            rows.extend(int(v) for v in raw['row'])
    return torch.cat(plans).numpy(), torch.cat(gts).numpy(), np.asarray(rows)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--ckpts', required=True, help='comma-separated; >1 means uniform weight average')
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    paths = [ROOT / p for p in args.ckpts.split(',')]
    device = torch.device('cuda:0')
    model, mismatch = build_model(paths)
    model.to(device).eval()
    raw = MotionDriveDataset(data_root='/tmp/pm97', split_manifest=str(SPLIT), split='tune',
                             supervision_root=str(SUP), min_frame=30, frame_stride=5, augment=False,
                             seed=0, history_contract='control')
    base = H4StatusDataset(mr.MotionCanvasDataset(raw, 'native'))
    t0 = time.time()
    up, gt, rows = run(model, base, device)
    mir, gt2, rows2 = run(model, Flipped(base), device)
    mir[..., 1] = -mir[..., 1]
    gt2[..., 1] = -gt2[..., 1]
    assert np.array_equal(rows, rows2) and np.abs(gt2 - gt).max() == 0.0
    records = json.loads(SESSION_SOURCE.read_text())['records']
    r2s = {int(r['row']): r['session'] for r in records}
    sessions = np.asarray([r2s[int(r)] for r in rows])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f'{args.tag}.npz', rows=rows, sessions=sessions, gt=gt, up=up, mir=mir)
    W = np.array([11, 11, 5, 5, 2, 2]) / 36.0
    d3 = lambda p: float((np.linalg.norm(p - gt, axis=-1) @ W).mean())
    print(json.dumps(dict(tag=args.tag, n_ckpts=len(paths), upright=d3(up), mirrored=d3(mir),
                          tta=d3(0.5 * (up + mir)), buffer_mismatch=mismatch[:5],
                          seconds=round(time.time() - t0, 1))))


if __name__ == '__main__':
    main()
