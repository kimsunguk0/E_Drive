"""Frozen DEV diagnostics only; no optimizer, checkpoint overwrite or submission."""
from pathlib import Path
import argparse
import datetime
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = Path('/NHNHOME/data/sukim/adcl')
REPORT = ROOT / 'reports/a2_post_native_review_20260922'
WORK = ROOT / 'work_dirs/a2_post_native_review_20260922'
NATIVE = ROOT / 'work_dirs/a2_native_detail_20260921'
spec = importlib.util.spec_from_file_location('review_stats', ROOT / 'experiments/a2_splitread_lanegeom_20260921/collect_next.py')
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def write(path, value):
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def source():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def components():
    paths = {a: NATIVE / f'{a}-s1/predictions_step10277.json' for a in ('M-LOW', 'M-NATIVE', 'S-LOW', 'S-NATIVE')}
    paths['DIRECT'] = ROOT / 'work_dirs/a2_progress_h4_20260920/A2-H4-DIRECT-s1/final_eval.json'
    paths['SPLITREAD'] = ROOT / 'work_dirs/a2_splitread_lanegeom_20260921/P-SPLITREAD-s1/predictions_step6852.json'
    _, keys, gt, _, buckets, sessions = stats.read(paths['M-NATIVE'])
    arrays = {}
    for name, path in paths.items():
        _, k, g, p, b, s = stats.read(path)
        assert k == keys and np.array_equal(gt, g) and np.array_equal(buckets, b) and np.array_equal(sessions, s)
        dp = np.diff(np.concatenate((np.zeros((len(p), 1, 2)), p), 1), axis=1)
        length = np.linalg.norm(dp, axis=-1)
        assert np.all(length > 0), (name, 'unexpected degenerate prediction; do not change the historical convention silently')
        arrays[name] = p, length, dp / length[..., None]
    dg = np.diff(np.concatenate((np.zeros((len(gt), 1, 2)), gt), 1), axis=1)
    lg = np.linalg.norm(dg, axis=-1)
    ug = dg / np.maximum(lg[..., None], 1e-12)
    out = {'source_commit': source(), 'scope': 'CPU diagnostics on saved DEV predictions; not a trained/deployable model; GT replacements are not achievable oracles',
           'sources': {n: {'path': str(p), 'sha256': stats.sha(p)} for n, p in paths.items()}, 'models': {}}
    for name, (p, length, unit) in arrays.items():
        variants = {'original': p,
                    'predicted_length_OLD_DIRECT_direction': np.cumsum(length[..., None] * arrays['DIRECT'][2], 1),
                    'GT_length_predicted_direction': np.cumsum(lg[..., None] * unit, 1),
                    'predicted_length_GT_direction_on_valid': np.cumsum(length[..., None] * np.where((lg > .05)[..., None], ug, unit), 1)}
        _, base_d = stats.stats(gt, p, buckets)
        result = {}
        for label, plan in variants.items():
            summary, d = stats.stats(gt, plan, buckets)
            result[label] = {'metrics': summary, 'delta_vs_own': stats.compare(d - base_d, sessions)}
        out['models'][name] = result
    write(REPORT / 'COMPONENT_RECOMPOSITION.json', out)
    print(json.dumps({n: {v: r['metrics']['PREFIX'] for v, r in rr.items()} for n, rr in out['models'].items()}), flush=True)


def ablate(arm):
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(ROOT / 'experiments/a2_native_detail_20260921'))
    from verify_native import datasets, inputs
    from native_model import load_export
    from motiondrive_v2_training import tensor_state_sha256, to_device
    expected = {'M-NATIVE': '8bc43364c05c899a563438f34538845ed0c33a969ab7e0cb6786e407ed9cbadf',
                'S-NATIVE': '67c7bf3b6f073c4a561a0f1e112fa54ab57875514d66857bf93ed9e75c8546d4'}
    gpu = {'M-NATIVE': '0', 'S-NATIVE': '1'}[arm]
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == gpu
    checkpoint = NATIVE / f'{arm}-s1/ckpt_step10277.pth'
    assert stats.sha(checkpoint) == expected[arm]
    assert not (REPORT / f'{arm}_FROZEN_BRANCH.json').exists()
    write(REPORT / f'{arm}_protocol.json', {'arm': arm, 'source_commit': source(),
          'script_sha256': stats.sha(Path(__file__)), 'checkpoint': str(checkpoint), 'checkpoint_sha256': expected[arm],
          'scope': 'same batch ON versus zeroed native_projection output, frozen checkpoint; 1998 DEV rows',
          'optimizer_updates': 0, 'GPU': gpu, 'interpretation': 'post-training dependency intervention, not a separately trained control'})
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    model, payload = load_export(checkpoint, 'cuda')
    del payload
    model.eval().requires_grad_(False)
    before = tensor_state_sha256(model.state_dict())
    projection = model.native_projection.weight.detach().clone()
    _, data = datasets(arm)
    oldpath = checkpoint.parent / 'predictions_step10277.json'
    _, keys, gt, old, buckets, sessions = stats.read(oldpath)
    predictions = {'ON': [], 'OFF': []}
    rows, trace = [], {}
    start = time.monotonic()
    try:
        with torch.inference_mode():
            for index, raw in enumerate(DataLoader(data, batch_size=8, num_workers=8, pin_memory=True, shuffle=False)):
                x = inputs(to_device(raw, torch.device('cuda')))
                internals = {}
                for label in ('ON', 'OFF'):
                    if label == 'ON':
                        model.native_projection.weight.copy_(projection)
                    else:
                        model.native_projection.weight.zero_()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        out = model(**x)
                    predictions[label].append(out['plan_abs'].float().cpu().numpy())
                    if index == 0:
                        internals[label] = {k: out[k].float().cpu() for k in ('scene_features', 'motion_features', 'state_hat', 'history_hat')}
                    del out
                if index == 0:
                    trace = {k: float((internals['ON'][k] - internals['OFF'][k]).abs().max()) for k in internals['ON']}
                rows.extend(int(r) for r in raw['row'])
                if (index + 1) % 50 == 0:
                    print(json.dumps({'arm': arm, 'rows': len(rows), 'seconds': time.monotonic() - start}), flush=True)
    finally:
        with torch.no_grad():
            model.native_projection.weight.copy_(projection)
    assert before == tensor_state_sha256(model.state_dict())
    assert rows == [k[0] for k in keys]
    predictions = {k: np.concatenate(v).astype(np.float64) for k, v in predictions.items()}
    max_reproduction = float(np.abs(predictions['ON'] - old).max())
    assert max_reproduction < 1e-3, max_reproduction
    result = {'arm': arm, 'source_commit': source(), 'checkpoint_sha256': expected[arm], 'n': len(rows),
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'optimizer_updates': 0,
              'parameters_and_buffers_restored_exactly': True, 'seconds': time.monotonic() - start,
              'projection_weight_norm': float(projection.float().norm()),
              'ON_max_abs_difference_vs_saved_m': max_reproduction, 'reproduction_tolerance_m': 1e-3,
              'first_batch_internal_max_abs_differences': trace, 'metrics': {},
              'limitation': 'OFF is a frozen inference intervention; does not establish what retraining without the branch would achieve'}
    ds = {}
    for label, p in predictions.items():
        assert np.isfinite(p).all()
        result['metrics'][label], ds[label] = stats.stats(gt, p, buckets)
    result['OFF_minus_ON'] = stats.compare(ds['OFF'] - ds['ON'], sessions)
    result['ON_OFF_plan_PREFIX_distance'] = float((np.linalg.norm(predictions['ON'] - predictions['OFF'], axis=-1) @ stats.W).mean())
    WORK.mkdir(parents=True, exist_ok=True)
    artifact = WORK / f'{arm}_frozen_predictions.npz'
    assert not artifact.exists()
    np.savez_compressed(artifact, row=rows, gt=gt, session=sessions, bucket=buckets, **predictions)
    result['predictions'] = {'path': str(artifact), 'sha256': stats.sha(artifact)}
    write(REPORT / f'{arm}_FROZEN_BRANCH.json', result)
    print(json.dumps({'arm': arm, 'PREFIX': {k: v['PREFIX'] for k, v in result['metrics'].items()},
                      'plan_distance': result['ON_OFF_plan_PREFIX_distance'], 'trace': trace}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('components', 'ablate'), required=True)
    parser.add_argument('--arm', choices=('M-NATIVE', 'S-NATIVE'))
    args = parser.parse_args()
    components() if args.mode == 'components' else ablate(args.arm)
