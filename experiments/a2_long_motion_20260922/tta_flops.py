#!/usr/bin/env python3
"""__flops__ for the two-forward TTA graph, counted the organisers' way.

TTA runs the model twice per clip, so the submitted __flops__ must reflect both
forwards. The organisers' tools/measure_flops.py uses FlopCounterMode over the
scored inference; counting one forward and shipping a two-forward model would
understate it.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.utils.module_tracker as _mt
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path('/NHNHOME/data/sukim/adcl')
for rel in ('', 'scripts', 'experiments/md_r0_reset_20260914',
            'experiments/a2_progress_h4_20260920', 'experiments/a2_progress_full_20260921',
            'experiments/a2_long_motion_20260922'):
    sys.path.insert(0, str(ROOT / rel))


class _NoHandle:
    def remove(self):
        pass


_mt.register_multi_grad_hook = lambda *a, **k: _NoHandle()

import infer_full
from test_disagreement import flip_inputs

CUTOFF_G = 7053.0


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--clip', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    model, payload = infer_full.load_model(Path(args.checkpoint))
    prepared = infer_full.prepare_clip(Path(args.clip))
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in prepared.inputs.items()}
    mirrored = {k: v.to(device) for k, v in flip_inputs(prepared.inputs).items()}

    with torch.no_grad():
        model(**inputs)

    def count(feed):
        c = FlopCounterMode(display=False, depth=3)
        with torch.no_grad(), c:
            model(**feed)
        return int(sum(c.get_flop_counts().get('Global', {}).values()))

    one = count(inputs)
    two = one + count(mirrored)
    report = {
        'schema_version': 1,
        'counter': 'torch.utils.flop_counter.FlopCounterMode, Global sum',
        'single_forward_flops': one,
        'tta_two_forward_flops': two,
        'gflops': {'single': one / 1e9, 'tta': two / 1e9},
        'cutoff_gflops': CUTOFF_G,
        'tta_headroom_x': CUTOFF_G / (two / 1e9),
        'tta_passes_cutoff': bool(two / 1e9 <= CUTOFF_G),
        'what_is_submitted': ('the TTA figure, because the scored inference runs both '
                              'forwards; reporting the single-forward count for a '
                              'two-forward model would understate it'),
        'checkpoint': str(Path(args.checkpoint).resolve()),
    }
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: report[k] for k in
                      ('single_forward_flops', 'tta_two_forward_flops', 'gflops',
                       'tta_headroom_x', 'tta_passes_cutoff')}, indent=1))


if __name__ == '__main__':
    main()
