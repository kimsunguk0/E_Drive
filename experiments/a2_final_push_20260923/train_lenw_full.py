#!/usr/bin/env python3
"""FULL-data twin of L-TRAIN3106-LENW: L-FULL6 with metric-aligned LEN.

Same parent as L-FULL6 (the 0.133685 official checkpoint), same 38,070
updates, same 376-scene union; only the interval weighting changes. V0 is in-fit
here, so this arm is a submission candidate and a soup partner for L-FULL6, not
a measurement.

Metric-aligned interval weighting in the length auxiliary. One changed line.

The plan is a cumulative sum, so interval k's length error moves every position
from k onward. Interval k's true share of the official metric is the tail sum of
the position weights [11,11,5,5,2,2]/36 -- 1.000, 0.694, 0.389, 0.250, 0.111,
0.056 -- so interval 1 matters 18x more than interval 6. The existing auxiliary
weights all six equally. This arm weights them by that tail sum, rescaled to
mean 1 so lambda keeps its magnitude and only the distribution changes.

Nothing else moves: same parent, same 34,326 updates, same schedule, same data.
The control is not a new run -- L-TRAIN3106-s1 IS the uniform-weight arm under
exactly this configuration, so the pair is matched by construction.
"""
from pathlib import Path
import argparse, contextlib, json, os, sys
import numpy as np, torch

ROOT = Path('/NHNHOME/data/sukim/adcl')
HERE = Path(__file__).resolve().parent
for p in (ROOT, ROOT / 'scripts', ROOT / 'experiments/a2_progress_h4_20260920',
          ROOT / 'experiments/md_r0_reset_20260914', str(HERE), str(ROOT / 'experiments/a2_long_motion_20260922')):
    sys.path.insert(0, str(p))

import train_long as base
from length_auxiliary import wrap_compute_loss_metric_weighted, METRIC_INTERVAL_WEIGHTS

ARM = 'L-FULL6-LENW'
REPORT = ROOT / 'reports/a2_lenw_full_20260923'
RUNS = ROOT / 'work_dirs/a2_lenw_full_20260923'

base.SPECS[ARM] = dict(base.SPECS['L-FULL6'])
base.SPECS[ARM]['gpu'] = None          # set from the command line
base.RUNS = RUNS


@contextlib.contextmanager
def metric_weighted_loss(original):
    """Replace the uniform auxiliary with the metric-aligned one, after wiring."""
    trainer = base.trainer
    saved = trainer.compute_loss
    trainer.compute_loss = wrap_compute_loss_metric_weighted(original, 0.25)
    try:
        yield
    finally:
        trainer.compute_loss = saved


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
    base.SPECS[ARM]['gpu'] = args.gpu
    torch.set_num_threads(4)

    pristine = base.trainer.compute_loss      # before any wrapping
    raw_train, raw_tune = base.nominal.raw_datasets(True, 1)
    run = RUNS / (ARM + '-s1' + ('-smoke' if args.smoke else ''))
    assert not run.exists() or not any(run.iterdir()), 'Refuse overwrite: ' + str(run)

    d, cp = base.declaration('L-FULL6', args.gpu, args.smoke, raw_train, raw_tune)
    # `arm` is validated against the configured arm downstream, so it stays
    # L-TRAIN3106. The variant is identified by this field and by the run and
    # report directories, not by renaming the arm out from under the validator.
    d['variant'] = ARM
    d['question'] = ('same parent, schedule and data as L-TRAIN3106; the length '
                     'auxiliary weights intervals by their true share of the metric '
                     'instead of uniformly')
    d['length_auxiliary'] = {
        'lambda': 0.25,
        'interval_weights': [round(float(x), 6) for x in METRIC_INTERVAL_WEIGHTS],
        'derivation': ('tail sums of the position weights [11,11,5,5,2,2]/36, because '
                       'the output is a cumsum, rescaled to mean 1'),
        'control': 'L-FULL6-s1, identical except the weights are uniform (in-fit V0)',
        'parameters_added': 0,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    out = REPORT / ('smoke' if args.smoke else '')
    out.mkdir(parents=True, exist_ok=True)
    protocol = out / f'protocol_{ARM}_s1.json'
    assert not protocol.exists(), protocol
    base.atomic(protocol, d)
    print('PLAN ' + json.dumps(dict(arm=ARM, gpu=args.gpu, updates=d['stage_updates'],
                                    train_rows=len(raw_train), eval_in_fit=False,
                                    weights=[round(float(x), 3) for x in METRIC_INTERVAL_WEIGHTS])),
          flush=True)
    with base.configure('L-FULL6', d), \
         base.runtime('L-FULL6', d, run, protocol, raw_train, raw_tune, cp), \
         metric_weighted_loss(pristine):
        argv = base.legacy.trainer_argv(1, run, False)
        argv[argv.index('--warmup') + 1] = '100'
        argv[argv.index('--log-every') + 1] = '1' if args.smoke else '10'
        base.trainer.run_training(argv, experiment=d)
    print('DONE ' + ARM, flush=True)


if __name__ == '__main__':
    main()
