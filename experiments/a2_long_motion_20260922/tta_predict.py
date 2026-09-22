#!/usr/bin/env python3
"""TTA inference on raw test clips: upright plus mirrored, averaged.

The mirror is the same one verified against flip_item key by key at exactly 0.0,
and the prediction is mapped back by negating y before averaging.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path('/NHNHOME/data/sukim/adcl')
for rel in ('', 'scripts', 'experiments/md_r0_reset_20260914',
            'experiments/a2_progress_h4_20260920', 'experiments/a2_progress_full_20260921',
            'experiments/a2_long_motion_20260922'):
    sys.path.insert(0, str(ROOT / rel))

import infer_full
from test_disagreement import flip_inputs


def predict_tta(model, prepared, precision='bf16'):
    upright = infer_full.predict(model, prepared, precision)
    mirrored_inputs = flip_inputs(prepared.inputs)
    flipped = type(prepared)(mirrored_inputs, prepared.metadata)
    mirrored = infer_full.predict(model, flipped, precision)
    mirrored = mirrored.copy()
    mirrored[:, 1] = -mirrored[:, 1]
    averaged = 0.5 * (upright + mirrored)
    assert averaged.shape == (6, 2) and np.isfinite(averaged).all()
    return averaged, upright, mirrored


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--clips', type=int, default=5)
    args = ap.parse_args()
    model, _ = infer_full.load_model(Path(args.checkpoint))
    clips = sorted(p for p in Path('/tmp/etri_test').iterdir() if p.is_dir())[:args.clips]
    gaps = []
    for clip in clips:
        prepared = infer_full.prepare_clip(clip)
        avg, up, mir = predict_tta(model, prepared)
        gaps.append(float(np.linalg.norm(up - mir, axis=-1).mean()))
        print('  %s  endpoint upright %.3f  averaged %.3f  gap %.4f m'
              % (clip.name, np.linalg.norm(up[-1]), np.linalg.norm(avg[-1]), gaps[-1]))
    print(json.dumps({'clips': len(clips), 'mean_disagreement_m': float(np.mean(gaps)),
                      'all_finite': True}))
