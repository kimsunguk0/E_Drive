#!/usr/bin/env python3
"""Final-push submission builder: one checkpoint (or a uniform weight soup of
siblings) with Flip TTA over the raw test clips.

Sharded so the 1,125 clips can run on several GPUs; `--merge` joins the shards
into one predictions file in the format package_submission.py expects
(clip token -> 6x2 absolute XY). Upright and mirrored views are kept next to it
so any later decision (e.g. calibration) can be re-derived without re-running.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/NHNHOME/data/sukim/adcl')
for rel in ('', 'scripts', 'experiments/md_r0_reset_20260914',
            'experiments/a2_progress_h4_20260920', 'experiments/a2_progress_full_20260921',
            'experiments/a2_long_motion_20260922'):
    sys.path.insert(0, str(ROOT / rel))

import infer_full
from tta_predict import predict_tta


def sha(path):
    return infer_full.sha(path)


def load(paths):
    model, payload = infer_full.load_model(Path(paths[0]))
    if len(paths) == 1:
        return model, payload
    states = [torch.load(p, map_location='cpu', weights_only=False)['model'] for p in paths]
    avg = {}
    for k, v in states[0].items():
        if v.is_floating_point():
            avg[k] = torch.stack([s[k].double() for s in states]).mean(0).to(v.dtype)
        else:
            assert all(torch.equal(s[k], v) for s in states), k
            avg[k] = v
    model.load_state_dict(avg, strict=True)
    return model.eval(), payload


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--ckpts', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--clips-root', default='/tmp/etri_test')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = args.ckpts.split(',')
    clips = sorted(p for p in Path(args.clips_root).iterdir() if p.is_dir())
    assert len(clips) == 1125, len(clips)

    if args.merge:
        shards = sorted(out.glob('shard_*.json'))
        up, mir, avg = {}, {}, {}
        for s in shards:
            d = json.loads(s.read_text())
            assert d['ckpts'] == paths, (d['ckpts'], paths)
            up.update(d['upright']); mir.update(d['mirrored']); avg.update(d['tta'])
        assert sorted(avg) == [c.name for c in clips], 'missing clips after merge'
        (out / 'predictions_tta.json').write_text(json.dumps(avg) + '\n')
        (out / 'predictions_upright.json').write_text(json.dumps(up) + '\n')
        (out / 'predictions_mirrored.json').write_text(json.dumps(mir) + '\n')
        a = np.array([avg[k] for k in sorted(avg)]); u = np.array([up[k] for k in sorted(up)])
        print(json.dumps({'clips': len(avg), 'finite': bool(np.isfinite(a).all()),
                          'mean_up_vs_tta_m': float(np.linalg.norm(u - a, axis=-1).mean())}))
        return

    i, n = map(int, args.shard.split('/'))
    infer_full.configure()
    model, payload = load(paths)
    mine = clips[i::n]
    up, mir, avg = {}, {}, {}
    t0 = time.time()
    for c in mine:
        prepared = infer_full.prepare_clip(c)
        a, u, m = predict_tta(model, prepared)
        avg[c.name] = a.tolist(); up[c.name] = u.tolist(); mir[c.name] = m.tolist()
    rec = dict(ckpts=paths, ckpt_sha256=[sha(p) for p in paths], step=payload['step'],
               arm=payload['manifest']['experimental_protocol']['arm'], shard=args.shard,
               n=len(mine), seconds=time.time() - t0, upright=up, mirrored=mir, tta=avg)
    (out / f'shard_{i}of{n}.json').write_text(json.dumps(rec) + '\n')
    print(json.dumps({k: rec[k] for k in ('shard', 'n', 'seconds', 'step', 'arm')}), flush=True)


if __name__ == '__main__':
    main()
