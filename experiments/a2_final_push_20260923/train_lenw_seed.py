#!/usr/bin/env python3
"""LENW sibling with a different trainer seed, as a weight-soup partner.

Same parent, schedule, data and metric-aligned length auxiliary as the seed-1
LENW run of the same arm; only the trainer seed (sample order, augmentation
draws) changes. The held-out twin (--arm L-TRAIN3106) decides whether a
two-seed soup helps; the FULL arm (--arm L-FULL6) is the submission partner.

Evidence this is aimed at: soup of two equally good siblings from one parent
(F-CTRL + F-FLOW) beat both members on held-out V0 by -0.00075 (8/11), while
soups of one trajectory's checkpoints (SWA) and of unequal siblings did not.
"""
from pathlib import Path
import argparse, contextlib, json, os, sys
import torch

ROOT = Path('/NHNHOME/data/sukim/adcl')
HERE = Path(__file__).resolve().parent
for p in (ROOT, ROOT / 'scripts', ROOT / 'experiments/a2_progress_h4_20260920',
          ROOT / 'experiments/md_r0_reset_20260914', str(HERE),
          str(ROOT / 'experiments/a2_long_motion_20260922')):
    sys.path.insert(0, str(p))

import train_long as base
from length_auxiliary import wrap_compute_loss_metric_weighted, METRIC_INTERVAL_WEIGHTS

TARGETS = {
    'L-TRAIN3106': dict(arm='L-TRAIN3106-LENW', full=False,
                        report='reports/a2_lenw_20260922', runs='work_dirs/a2_lenw_20260922'),
    'L-FULL6': dict(arm='L-FULL6-LENW', full=True,
                    report='reports/a2_lenw_full_20260923', runs='work_dirs/a2_lenw_full_20260923'),
}


@contextlib.contextmanager
def metric_weighted_loss(original):
    trainer = base.trainer
    saved = trainer.compute_loss
    trainer.compute_loss = wrap_compute_loss_metric_weighted(original, 0.25)
    try:
        yield
    finally:
        trainer.compute_loss = saved


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--arm', choices=sorted(TARGETS), required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    assert args.seed != 1, 'seed 1 is the existing run'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
    t = TARGETS[args.arm]
    arm = t['arm']
    report = ROOT / t['report']
    base.RUNS = ROOT / t['runs']
    torch.set_num_threads(4)

    pristine = base.trainer.compute_loss
    raw_train, raw_tune = base.nominal.raw_datasets(t['full'], 1)
    tag = f'{arm}-s{args.seed}'
    run = base.RUNS / (tag + ('-smoke' if args.smoke else ''))
    assert not run.exists() or not any(run.iterdir()), 'Refuse overwrite: ' + str(run)

    d, cp = base.declaration(args.arm, args.gpu, args.smoke, raw_train, raw_tune)
    d['variant'] = arm
    d['trainer_seed'] = args.seed
    d['question'] = (f'seed-{args.seed} sibling of {arm}-s1 (same parent, schedule, data, '
                     'metric-aligned length auxiliary) as a weight-soup partner')
    d['length_auxiliary'] = {
        'lambda': 0.25,
        'interval_weights': [round(float(x), 6) for x in METRIC_INTERVAL_WEIGHTS],
        'parameters_added': 0,
    }
    out = report / ('smoke' if args.smoke else '')
    out.mkdir(parents=True, exist_ok=True)
    protocol = out / f'protocol_{arm}_s{args.seed}.json'
    assert not protocol.exists(), protocol
    base.atomic(protocol, d)
    print('PLAN ' + json.dumps(dict(arm=arm, seed=args.seed, gpu=args.gpu,
                                    updates=d['stage_updates'], train_rows=len(raw_train),
                                    evaluation_is_in_fit=t['full'])), flush=True)
    # train_long.runtime hard-codes seed 1 into the shared-dynamics runtime check;
    # route the declared seed through so the check compares like with like.
    patched = base.legacy.patched_runtime
    base.legacy.patched_runtime = lambda a, _seed, *rest: patched(a, args.seed, *rest)
    with base.configure(args.arm, d), \
         base.runtime(args.arm, d, run, protocol, raw_train, raw_tune, cp), \
         metric_weighted_loss(pristine):
        argv = base.legacy.trainer_argv(1, run, False)
        argv[argv.index('--seed') + 1] = str(args.seed)
        argv[argv.index('--warmup') + 1] = '100'
        argv[argv.index('--log-every') + 1] = '1' if args.smoke else '10'
        base.trainer.run_training(argv, experiment=d)
    print('DONE ' + tag, flush=True)


if __name__ == '__main__':
    main()
