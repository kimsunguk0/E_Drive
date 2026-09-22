#!/usr/bin/env python3
"""Metric-aligned interval weighting on the train339 lineage.

F-CTRL is the matched control by construction: same P-NEARFULL-H parent, same
22,884 updates, same train339 rows, same schedule, same lambda 0.25, no teacher.
The only difference is that the length auxiliary weights intervals by their true
share of the metric instead of uniformly.

On train310 the same change measured -0.000790 with a paired CI of
[-0.001192, -0.000456] and 9 of 11 sessions, the only structural or supervision
change since 09-21 whose CI cleared zero. This asks whether it survives on the
better parent.

flow_common.py's hash is pinned by calibration.json, so it is not edited; the
swap is a runtime rebind of the name objective() looks up.
"""
from pathlib import Path
import argparse, json, os, shutil, sys

ROOT = Path('/NHNHOME/data/sukim/adcl')
for rel in ('', 'scripts', 'experiments/md_r0_reset_20260914',
            'experiments/a2_long_motion_20260922'):
    sys.path.insert(0, str(ROOT / rel))

import torch
import flow_common
import train_flow as tf
from length_auxiliary import length_loss_metric_weighted, METRIC_INTERVAL_WEIGHTS

ARM = 'F-LENW'
DONOR = 'F-CTRL'


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--gpu', type=int, required=True)
    args_cli = ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args_cli.gpu)

    # Reuse the donor's passing gate artifacts: identical code path, identical
    # sample stream, and the source hashes they pin are unchanged.
    smoke = flow_common.FLOW_REPORT / 'smoke'
    if not (smoke / ARM).exists():
        shutil.copytree(smoke / DONOR, smoke / ARM)
    flow_common.FLOW_ARMS[ARM] = args_cli.gpu
    tf.FLOW_ARMS[ARM] = args_cli.gpu

    # The swap. objective() resolves length_loss as a module global at call time.
    flow_common.length_loss = length_loss_metric_weighted

    torch.set_num_threads(4)
    flow_common.trainer.seed_all(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

    model, cp = tf.get_model()
    head = tf.FlowReadout().cuda()
    capture = tf.Capture(model)
    train, val, probe, idx = tf.datasets()
    d = tf.protocol(ARM, args_cli.gpu, model, train, val)
    d['length_auxiliary'] = {
        'lambda': 0.25,
        'interval_weights': [round(float(x), 6) for x in METRIC_INTERVAL_WEIGHTS],
        'derivation': 'tail sums of [11,11,5,5,2,2]/36, rescaled to mean 1',
        'control': '%s-s1, identical except the weights are uniform' % DONOR,
        'gate_artifacts_reused_from': DONOR,
        'parameters_added': 0,
    }
    print('PLAN ' + json.dumps(dict(arm=ARM, gpu=args_cli.gpu, updates=flow_common.UPDATES,
                                    control=DONOR)), flush=True)

    class A:
        arm = ARM
        gpu = args_cli.gpu
        mode = 'train'

    try:
        tf.train_main(A, model, cp, head, capture, None, train, val, probe, idx, d)
    finally:
        capture.remove()
    print('DONE ' + ARM, flush=True)


if __name__ == '__main__':
    main()
