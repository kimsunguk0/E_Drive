#!/usr/bin/env python3
"""Upright-vs-mirrored disagreement on the official test clips.

The test clips have no labels, so TTA cannot be scored there. What CAN be
measured is how far the two predictions sit apart, which is the quantity
averaging acts on. If the test disagreement matches V0's, the mechanism TTA
exploits is present at the same strength and the measured V0 gain is more
likely to carry; if it is much smaller, expect less.

The deployment inputs are not dataset items, so the flip is reimplemented for
them -- and then checked against flip_item on a real V0 item, key by key, so the
two cannot silently disagree about a sign.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/NHNHOME/data/sukim/adcl')
for rel in ('', 'scripts', 'experiments/md_r0_reset_20260914',
            'experiments/a2_progress_h4_20260920', 'experiments/a2_progress_full_20260921'):
    sys.path.insert(0, str(ROOT / rel))

import infer_full
from motiondrive_v2_flip_augment import flip_item, CAM_MIRROR, _S4

CKPT = ROOT / 'work_dirs/a2_progress_full_20260921/A2-H4-PROGRESS-FULL-s1/ckpt_step24931.pth'
CLIPS = Path('/tmp/etri_test')
OUT = ROOT / 'reports/a2_tta_20260922'
WIDTH_FULL = 768


def flip_inputs(inputs):
    """Mirror a prepared deployment batch. Same conventions as flip_item."""
    o = dict(inputs)
    o['images'] = torch.flip(inputs['images'], dims=[-1])[:, CAM_MIRROR]
    o['history_images'] = torch.flip(inputs['history_images'], dims=[-1])
    for key in ('motion_current', 'motion_history'):
        if key in inputs:
            o[key] = torch.flip(inputs[key], dims=[-1])
    L = inputs['lidar2img'][0]
    S = _S4(L)
    Fx = torch.eye(3, dtype=L.dtype)
    Fx[0, 0] = -1.0
    Fx[0, 2] = float(WIDTH_FULL - 1)
    new = L.clone()
    for i in range(len(L)):
        m = L[i].clone()
        m[:3, :] = Fx @ (L[i][:3, :] @ S)
        new[CAM_MIRROR[i]] = m
    o['lidar2img'] = new[None]
    T = inputs['history_transforms'][0]
    o['history_transforms'] = torch.stack([S @ T[k] @ S for k in range(len(T))])[None]
    g = inputs['goal_xy'].clone()
    g[..., 1] = -g[..., 1]
    o['goal_xy'] = g
    p = inputs['provided_status5'].clone()
    p[..., 1] = -p[..., 1]
    p[..., 3] = -p[..., 3]
    p[..., 4] = -p[..., 4]
    o['provided_status5'] = p
    return o


def verify_against_flip_item():
    """flip_inputs must agree with flip_item on every key they share."""
    import matching_resolution as mr
    from motiondrive_v2_data import MotionDriveDataset
    from h4_data import H4StatusDataset
    raw = MotionDriveDataset(
        data_root='/tmp/pm97',
        split_manifest=str(ROOT / 'data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json'),
        split='tune', supervision_root=str(ROOT / 'data/etri/motiondrive_v2/r0reset_tplus_geometry_v2'),
        min_frame=30, frame_stride=5, augment=False, seed=0, history_contract='control')
    item = H4StatusDataset(mr.MotionCanvasDataset(raw, 'native'))[0]
    reference = mr.wrap_flip_item(flip_item)(item, 768, 384)
    batched = {k: v[None] for k, v in item.items() if torch.is_tensor(v)}
    batched['motion_current'] = batched.pop(mr.MOTION_CURRENT_KEY)
    batched['motion_history'] = batched.pop(mr.MOTION_HISTORY_KEY)
    mine = flip_inputs(batched)
    checked = {}
    pairs = [('images', 'images'), ('history_images', 'history_images'),
             ('lidar2img', 'lidar2img'), ('history_transforms', 'history_transforms'),
             ('goal_xy', 'goal_xy'), ('provided_status5', 'provided_status5'),
             ('motion_current', mr.MOTION_CURRENT_KEY),
             ('motion_history', mr.MOTION_HISTORY_KEY)]
    for mine_key, ref_key in pairs:
        a = mine[mine_key][0].float()
        b = reference[ref_key].float()
        checked[mine_key] = float((a - b).abs().max())
    return checked


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--clips', type=int, default=300)
    args = ap.parse_args()

    agreement = verify_against_flip_item()
    worst = max(agreement.values())
    print('flip_inputs vs flip_item, max abs diff per key:')
    for k, v in agreement.items():
        print('   %-22s %.3e' % (k, v))
    if worst > 1e-5:
        raise SystemExit('flip conventions disagree; refusing to measure')

    model, payload = infer_full.load_model(CKPT, require_full=True)
    clips = sorted(p for p in CLIPS.iterdir() if p.is_dir())[:args.clips]
    gaps = []
    for i, clip in enumerate(clips):
        prepared = infer_full.prepare_clip(clip)
        upright = infer_full.predict(model, prepared)
        flipped = type(prepared)(flip_inputs(prepared.inputs), prepared.metadata)
        mirrored = infer_full.predict(model, flipped)
        mirrored[:, 1] = -mirrored[:, 1]
        gaps.append(np.linalg.norm(upright - mirrored, axis=-1))
        if (i + 1) % 50 == 0:
            print('  %d/%d clips' % (i + 1, len(clips)), flush=True)
    gaps = np.asarray(gaps)

    v0 = json.loads((OUT / 'flip_tta_full_infit.json').read_text())
    payload_out = {
        'schema_version': 1,
        'checkpoint': str(CKPT),
        'clips': len(clips),
        'flip_convention_check': agreement,
        'test_disagreement_m': {
            'mean': float(gaps.mean()), 'p50': float(np.percentile(gaps, 50)),
            'p95': float(np.percentile(gaps, 95)),
            'per_waypoint_mean': [float(x) for x in gaps.mean(0)],
        },
        'v0_disagreement_m': v0['upright_vs_mirrored_disagreement_m'],
        'reading': ('if test disagreement matches V0, the variance TTA cancels is '
                    'present at the same strength on the scored set; this is not a '
                    'score and no gain is claimed from it'),
    }
    (OUT / 'test_disagreement.json').write_text(json.dumps(payload_out, indent=1, sort_keys=True) + '\n')
    print()
    print('test  mean %.4f m  p50 %.4f  p95 %.4f' % (
        gaps.mean(), np.percentile(gaps, 50), np.percentile(gaps, 95)))
    print('V0    mean %.4f m  p50 %.4f  p95 %.4f' % (
        v0['upright_vs_mirrored_disagreement_m']['mean'],
        v0['upright_vs_mirrored_disagreement_m']['p50'],
        v0['upright_vs_mirrored_disagreement_m']['p95']))
    print('test per waypoint:', [round(float(x), 4) for x in gaps.mean(0)])
    print('V0   per waypoint:', [round(x, 4) for x in
                                 v0['upright_vs_mirrored_disagreement_m']['per_waypoint_mean']])


if __name__ == '__main__':
    main()
