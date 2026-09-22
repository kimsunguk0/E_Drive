#!/usr/bin/env python3
"""H6: six motion timepoints instead of four. Control is L-TRAIN3106.

Same parent, same 34,326 updates, same schedule, same train310 data. The only
change is how many past frames the motion correlation sees, so L-TRAIN3106-s1
(terminal 0.143554) is the matched control without running anything extra.

NEAR adds -0.3 and -0.4 inside the existing reach; LONG adds -2.0 and -2.5
beyond it. The scene branch, the provided status and the history/state targets
all stay at the original four offsets.
"""
from pathlib import Path
import argparse, contextlib, json, os, sys
import numpy as np, torch

ROOT = Path('/NHNHOME/data/sukim/adcl')
HERE = Path(__file__).resolve().parent
for p in (ROOT, ROOT / 'scripts', ROOT / 'experiments/a2_progress_h4_20260920',
          ROOT / 'experiments/a2_temporal_read_20260920',
          ROOT / 'experiments/a2_long_motion_20260922', str(HERE)):
    sys.path.insert(0, str(p))

import train_long as base
import h6_data
from h4_data import H4StatusDataset
from models.motiondrive_v2.motion_encoder import MotionEncoder

REPORT = ROOT / 'reports/a2_h6_20260923'
RUNS = ROOT / 'work_dirs/a2_h6_20260923'
base.RUNS = RUNS


@contextlib.contextmanager
def h6_runtime(arm):
    """Six motion canvases in, four supervised history offsets out."""
    # TemporalReadMotionEncoder overrides forward, so patching the base class
    # would be shadowed; the concrete class in use is the one to patch.
    from temporal_model import TemporalReadMotionEncoder
    saved_forward = h6_data.patch_motion_encoder(
        TemporalReadMotionEncoder, h6_data.H6[arm]['seconds'],
        h6_data.supervised_indices(arm))
    saved_dataset = base.legacy.CausalStatusDataset

    # The hook receives a MotionCanvasDataset that already carries four frames.
    def dataset(canvas):
        return H4StatusDataset(h6_data.H6CanvasDataset(canvas, arm))

    base.legacy.CausalStatusDataset = dataset
    try:
        yield
    finally:
        TemporalReadMotionEncoder.forward = saved_forward
        base.legacy.CausalStatusDataset = saved_dataset


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--arm', choices=sorted(h6_data.H6), required=True)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
    torch.set_num_threads(4)

    spec = h6_data.H6[args.arm]
    base.SPECS[args.arm] = dict(base.SPECS['L-TRAIN3106'], gpu=args.gpu)
    raw_train, raw_tune = base.nominal.raw_datasets(False, 1)
    run = RUNS / (args.arm + '-s1' + ('-smoke' if args.smoke else ''))
    assert not run.exists() or not any(run.iterdir()), 'Refuse overwrite: ' + str(run)

    d, cp = base.declaration('L-TRAIN3106', args.gpu, args.smoke, raw_train, raw_tune)
    d['variant'] = args.arm
    d['h6'] = {
        'motion_frame_offsets': list(spec['offsets']),
        'motion_nominal_seconds': list(spec['seconds']),
        'added_offsets': list(spec['extra']),
        'question': spec['question'],
        'scene_branch_offsets': list(h6_data.BASE_OFFSETS),
        'supervised_history_offsets': list(h6_data.BASE_OFFSETS),
        'supervised_positions': h6_data.supervised_indices(args.arm),
        'why_supervision_unchanged': ('history_target and history_transforms are (N,4,...) '
                                      'on disk; the head therefore still predicts the four '
                                      'offsets that have labels while the correlation sees six'),
        'control': 'L-TRAIN3106-s1, identical except it sees four motion timepoints',
        'parameters_added': 0,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    out = REPORT / ('smoke' if args.smoke else '')
    out.mkdir(parents=True, exist_ok=True)
    protocol = out / f'protocol_{args.arm}_s1.json'
    assert not protocol.exists(), protocol
    base.atomic(protocol, d)
    print('PLAN ' + json.dumps(dict(arm=args.arm, gpu=args.gpu, updates=d['stage_updates'],
                                    offsets=list(spec['offsets']))), flush=True)
    # h6_runtime enters LAST: base.runtime rebinds the dataset hook too, so an
    # earlier override is silently replaced by the four-frame one.
    with base.configure('L-TRAIN3106', d), \
         base.runtime('L-TRAIN3106', d, run, protocol, raw_train, raw_tune, cp), \
         h6_runtime(args.arm):
        argv = base.legacy.trainer_argv(1, run, False)
        argv[argv.index('--warmup') + 1] = '100'
        argv[argv.index('--log-every') + 1] = '1' if args.smoke else '10'
        base.trainer.run_training(argv, experiment=d)
    print('DONE ' + args.arm, flush=True)


if __name__ == '__main__':
    main()
