#!/usr/bin/env python3
"""Raw-test-clip inference for ExtModel: ONE forward per clip. The network emits
K complete 5 s candidates; the goal point picks the one whose 5 s endpoint is
nearest; its first six points are written unchanged (format conversion only).

  --shard i/n         run a slice of the 1,125 clips on this GPU
  --merge             join shards into predictions.json
  --flops CLIP        FlopCounterMode count of one full forward (organisers' method)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/NHNHOME/data/sukim/adcl')
HERE = Path(__file__).resolve().parent
for rel in ('', 'scripts', 'experiments/md_r0_reset_20260914', 'experiments/a2_progress_h4_20260920',
            'experiments/a2_progress_full_20260921', 'experiments/a2_long_motion_20260922'):
    sys.path.insert(0, str(ROOT / rel))
sys.path.insert(0, str(HERE))

import infer_full
from ext_model import ExtModel
from models.motiondrive_v2 import MotionDriveV2Config
import matching_resolution as mr


def load(ckpt, device):
    cp = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = ExtModel(MotionDriveV2Config(**cp['manifest']['model_config']))
    mr.rebuild_correlation_fuse(model, 4)
    model.load_state_dict(cp['model'], strict=True)
    return model.to(device).eval(), cp


def predict(model, prepared):
    device = next(model.parameters()).device
    x = {k: v.to(device) for k, v in prepared.inputs.items()}
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        out = model(**x)
    plan = out['plan_abs'].float().cpu().numpy()[0]
    assert plan.shape == (6, 2) and np.isfinite(plan).all()
    return plan, int(out['selected_mode'][0])


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--clips-root', default='/tmp/etri_test')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--flops')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in Path(args.clips_root).iterdir() if p.is_dir())
    assert len(clips) == 1125
    if args.merge:
        pred, modes = {}, {}
        for s in sorted(out.glob('shard_*.json')):
            d = json.loads(s.read_text()); assert d['ckpt'] == args.ckpt
            pred.update(d['pred']); modes.update(d['mode'])
        assert sorted(pred) == [c.name for c in clips]
        (out / 'predictions.json').write_text(json.dumps(pred) + '\n')
        print(json.dumps({'clips': len(pred), 'mode_hist': np.bincount(list(modes.values())).tolist()}))
        return
    infer_full.configure()
    model, cp = load(args.ckpt, torch.device('cuda:0'))
    if args.flops:
        import torch.utils.module_tracker as mt
        class _H:
            def remove(self): pass
        mt.register_multi_grad_hook = lambda *a, **k: _H()
        from torch.utils.flop_counter import FlopCounterMode
        prepared = infer_full.prepare_clip(Path(args.flops))
        x = {k: v.cuda() for k, v in prepared.inputs.items()}
        with torch.no_grad():
            model(**x)
        c = FlopCounterMode(display=False, depth=3)
        with torch.no_grad(), c:
            model(**x)
        flops = int(sum(c.get_flop_counts().get('Global', {}).values()))
        from motiondrive_v2_training import tensor_state_sha256
        rep = dict(flops=flops, gflops=flops / 1e9, passes_cutoff=flops / 1e9 <= 7053.0, cutoff_gflops=7053.0,
                   counter='torch.utils.flop_counter.FlopCounterMode Global sum',
                   scope='one full forward per clip: every encoder, the K-candidate planner and the goal selection',
                   checkpoint=str(Path(args.ckpt).resolve()), model_state_sha256=tensor_state_sha256(model.state_dict()))
        (out / 'flops_report.json').write_text(json.dumps(rep, indent=1) + '\n')
        print(json.dumps(rep))
        return
    i, n = map(int, args.shard.split('/'))
    pred, mode = {}, {}
    t0 = time.time()
    for c in clips[i::n]:
        p, m = predict(model, infer_full.prepare_clip(c))
        pred[c.name] = p.tolist(); mode[c.name] = m
    (out / f'shard_{i}of{n}.json').write_text(json.dumps(dict(ckpt=args.ckpt, pred=pred, mode=mode)) + '\n')
    print(json.dumps(dict(shard=args.shard, n=len(pred), seconds=time.time() - t0)))


if __name__ == '__main__':
    main()
