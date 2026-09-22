#!/usr/bin/env python3
"""Flip test-time augmentation on held-out V0, no training.

Runs the fixed checkpoint twice per row -- upright, and on the mirrored input
with the prediction mirrored back -- and scores the average. Costs a second
forward (26 ms -> ~52 ms against a 100 ms penalty threshold, 730 -> 1460 GFLOPs
against a 7053 cutoff), so it is affordable if it helps.

It may well not. motiondrive_v2_flip_augment says outright that flipping is
"applied only in training - the labels carry a real right-hand bias, so
inference stays upright". Averaging with a mirror pushes the prediction toward
left-right symmetry, and if the driving distribution is genuinely asymmetric
that is a bias, not a variance reduction. This measures which one it is.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/NHNHOME/data/sukim/adcl')
for p in (ROOT, ROOT / 'scripts', ROOT / 'experiments/a2_progress_h4_20260920',
          ROOT / 'experiments/a2_long_motion_20260922', ROOT / 'experiments/a2_progress_full_20260921'):
    sys.path.insert(0, str(p))

import train_progress as dev
from h4_data import H4StatusDataset
from progress_model import H4ProgressModel, PROGRESS
from models.motiondrive_v2 import MotionDriveV2Config
from motiondrive_v2_flip_augment import flip_item
from motiondrive_v2_data import MotionDriveDataset
from motiondrive_v2_training import weighted_d3

mr, trainer = dev.mr, dev.trainer
DEV_PARENT = ROOT / 'work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1/ckpt_step20554.pth'
FULL_PARENT = (ROOT / 'work_dirs/a2_progress_full_20260921/'
               'A2-H4-PROGRESS-FULL-s1/ckpt_step24931.pth')
SPLIT = ROOT / 'data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json'
SUP = ROOT / 'data/etri/motiondrive_v2/r0reset_tplus_geometry_v2'
OUT = ROOT / 'reports/a2_tta_20260922'
WIDTH_FULL, WIDTH_HIST = 768, 384


class Flipped(torch.utils.data.Dataset):
    """Same rows, mirrored inputs. GT is mirrored too but never used for scoring."""

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
        return self.flip(self.base[i], WIDTH_FULL, WIDTH_HIST)


def build_eval(full):
    raw = MotionDriveDataset(
        data_root='/tmp/pm97', split_manifest=str(SPLIT), split='tune',
        supervision_root=str(SUP), min_frame=30, frame_stride=5, augment=False,
        seed=0, history_contract='control')
    canvas = mr.MotionCanvasDataset(raw, 'native')
    if full:
        sys.path.insert(0, str(ROOT / 'experiments/a2_progress_full_20260921'))
        from full_data import FullH4StatusDataset
        return FullH4StatusDataset(canvas)
    return H4StatusDataset(canvas)


def run(model, dataset, device):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4,
                        pin_memory=True, collate_fn=torch.utils.data.default_collate)
    plans, gts, sessions = [], [], []
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for raw in loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in raw.items()}
            x = mr.model_inputs_with_canvas(trainer.model_inputs, batch, time_input='nominal')
            x['provided_status5'] = batch['provided_status5']
            out = model(**x)['plan_abs'].float().cpu()
            plans.append(out)
            gts.append(raw['gt_plan'].float())
            sessions.extend(int(v) for v in raw['row'])
    return torch.cat(plans), torch.cat(gts), sessions


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--full', action='store_true',
                    help='evaluate the FULL checkpoint; V0 is then IN-FIT')
    ap.add_argument('--tag', default='dev')
    args = ap.parse_args()
    PARENT = FULL_PARENT if args.full else DEV_PARENT

    device = torch.device('cuda:0')
    cp = torch.load(PARENT, map_location='cpu', weights_only=False)
    model = H4ProgressModel(MotionDriveV2Config(**cp['manifest']['model_config']), arm=PROGRESS)
    mr.rebuild_correlation_fuse(model, 4)
    model.load_state_dict(cp['model'], strict=True)
    model.to(device).eval()

    base = build_eval(args.full)
    t0 = time.time()
    upright, gt, sessions = run(model, base, device)
    t1 = time.time()
    mirrored, gt_flipped, _ = run(model, Flipped(base), device)
    t2 = time.time()

    # Undo the mirror on the prediction: the flip negates y everywhere.
    unmirrored = mirrored.clone()
    unmirrored[..., 1] = -unmirrored[..., 1]
    # Sanity: the flipped dataset's GT, un-mirrored, must equal the original GT.
    check = gt_flipped.clone()
    check[..., 1] = -check[..., 1]
    gt_roundtrip = float((check - gt).abs().max())

    averaged = 0.5 * (upright + unmirrored)
    scores = {name: weighted_d3(v, gt).numpy()
              for name, v in (('upright', upright), ('mirrored_only', unmirrored),
                              ('averaged', averaged))}

    # How far apart the two predictions are. This needs no labels, so the same
    # quantity is measurable on the official test clips, where the score is not.
    disagreement = torch.linalg.vector_norm(upright - unmirrored, dim=-1)

    # `sessions` currently holds row ids; join them to real session names through
    # the run's own final_eval.json so the bootstrap resamples sessions, not rows.
    records = json.loads((PARENT.parent / 'final_eval.json').read_text())['records']
    row_to_session = {int(r['row']): r['session'] for r in records if 'row' in r}
    if len(row_to_session) < len(sessions):
        order = sorted(range(len(records)),
                       key=lambda i: (records[i]['session'], records[i]['scenario'],
                                      int(records[i]['frame'])))
        raise SystemExit('final_eval.json lacks row ids; cannot join sessions safely')
    sessions = np.asarray([row_to_session[r] for r in sessions])
    unique = sorted(set(sessions.tolist()))
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(unique), size=(20000, len(unique)))
    picks = [np.concatenate([index[unique[j]] for j in row]) for row in draws]

    def compare(a, b):
        diff = scores[b] - scores[a]
        means = np.asarray([diff[p].mean() for p in picks])
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        return {'delta': float(diff.mean()), 'ci95': [lo, hi],
                'ci_includes_zero': bool(lo <= 0 <= hi),
                'sessions_improved': sum(1 for s in unique if diff[index[s]].mean() < 0),
                'sessions': len(unique)}

    payload = {
        'schema_version': 1,
        'checkpoint': str(PARENT),
        'eval': ('V0, tune 37 scenes, stride 5, augment off. '
                 + ('IN-FIT: this FULL checkpoint trained on these very rows, so the '
                    'PREFIX values are not generalization estimates and the TTA delta '
                    'measured here need not match the server.'
                    if args.full else
                    'HELD OUT: the DEV checkpoint never saw these rows.')),
        'checkpoint_is_full': bool(args.full),
        'upright_vs_mirrored_disagreement_m': {
            'mean': float(disagreement.mean()),
            'p50': float(disagreement.median()),
            'p95': float(np.percentile(disagreement.numpy(), 95)),
            'per_waypoint_mean': [float(x) for x in disagreement.mean(0)],
            'what_it_is': ('L2 between the upright prediction and the mirrored one '
                           'mapped back; it is how much disagreement averaging has to '
                           'cancel, and it is label-free'),
        },
        'rows': int(len(gt)),
        'gt_roundtrip_max_abs_error_m': gt_roundtrip,
        'PREFIX': {k: float(v.mean()) for k, v in scores.items()},
        'averaged_minus_upright': compare('upright', 'averaged'),
        'mirrored_minus_upright': compare('upright', 'mirrored_only'),
        'seconds': {'upright': t1 - t0, 'mirrored': t2 - t1},
        'cost_if_adopted': 'two forwards per clip instead of one',
        'documented_counterargument': (
            'motiondrive_v2_flip_augment states flipping is training-only because '
            'the labels carry a real right-hand bias; averaging with a mirror can '
            'therefore bias toward symmetry rather than reduce variance'),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ('flip_tta_%s.json' % args.tag)).write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    print(json.dumps(payload, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
