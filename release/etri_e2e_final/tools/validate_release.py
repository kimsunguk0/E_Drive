#!/usr/bin/env python3
"""Release validation inside the container (run by scripts/validate.sh).

1. Correctness: predictions on 8 train fixtures and 20 official test clips are
   compared with references made on the original B200 (fixtures) and with the
   actually submitted predictions (test clips).
2. Resource: FlopCounterMode total of one forward, and GPU forward latency
   (median of 100 after 20 warm-up; preprocessing excluded, as the organisers time
   model forward only).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

CODE = Path('/NHNHOME/data/sukim/adcl')
sys.path.insert(0, str(CODE / 'experiments/a2_final_push_20260923/ext_v6'))
from build_ext_submission import load, predict   # noqa: E402  (sets up the rest of sys.path)
import infer_full                                   # noqa: E402

W = np.array([11, 11, 5, 5, 2, 2]) / 36.0


def compare(model, clips, ref, ref_mode=None):
    rows = []
    for c in clips:
        p, m = predict(model, infer_full.prepare_clip(c))
        r = np.array(ref[c.name])
        rows.append(dict(clip=c.name, max_abs_m=float(np.abs(p - r).max()),
                         weighted_l2_m=float(np.linalg.norm(p - r, axis=-1) @ W),
                         same_mode=None if ref_mode is None else bool(m == ref_mode[c.name])))
    d = np.array([r['max_abs_m'] for r in rows])
    return dict(n=len(rows), max_abs_m_median=float(np.median(d)), max_abs_m_max=float(d.max()),
                weighted_l2_mean_m=float(np.mean([r['weighted_l2_m'] for r in rows])),
                same_selected_candidate=None if ref_mode is None else f"{sum(r['same_mode'] for r in rows)}/{len(rows)}",
                rows=rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    infer_full.configure()
    dev = torch.device('cuda:0')
    model, cp = load(str(CODE / a.ckpt), dev)
    V = Path('/release/validation')
    fx = sorted(p for p in (CODE / 'data/etri/motiondrive_v2/deploy_fixture_train8').iterdir() if p.is_dir())
    ts = sorted(p for p in (V / 'test_subset').iterdir() if p.is_dir())
    ref_fx = json.loads((V / 'reference_fixtures_B200_bf16.json').read_text())
    ref_ts = json.loads((V / 'reference_test_submitted.json').read_text())
    out = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
               checkpoint=a.ckpt,
               fixtures_vs_B200=compare(model, fx, ref_fx['pred'], ref_fx['mode']),
               test_vs_submitted=compare(model, ts, ref_ts['pred'], ref_ts['mode']))
    # FLOPs of one full forward (organisers' counter)
    import torch.utils.module_tracker as mt
    class _H:
        def remove(self): pass
    mt.register_multi_grad_hook = lambda *x, **k: _H()
    from torch.utils.flop_counter import FlopCounterMode
    x = {k: v.to(dev) for k, v in infer_full.prepare_clip(ts[0]).inputs.items()}
    with torch.no_grad(): model(**x)
    fc = FlopCounterMode(display=False, depth=3)
    with torch.no_grad(), fc: model(**x)
    out['flops'] = int(sum(fc.get_flop_counts().get('Global', {}).values()))
    # latency: one full forward, BF16 autocast as in inference
    def fwd():
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16): model(**x)
    for _ in range(20): fwd()
    torch.cuda.synchronize(); ts_ms = []
    for _ in range(100):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fwd(); e.record(); e.synchronize(); ts_ms.append(s.elapsed_time(e))
    med = float(np.median(ts_ms))
    out['latency_ms'] = dict(median=med, p95=float(np.quantile(ts_ms, .95)),
                             time_penalty_factor=1 + max(0., med - 100) / 200)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    s = {k: v for k, v in out.items() if k not in ('fixtures_vs_B200', 'test_vs_submitted')}
    s['fixtures'] = {k: v for k, v in out['fixtures_vs_B200'].items() if k != 'rows'}
    s['test'] = {k: v for k, v in out['test_vs_submitted'].items() if k != 'rows'}
    print(json.dumps(s, indent=1))


if __name__ == '__main__':
    main()
