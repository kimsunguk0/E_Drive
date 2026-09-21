"""Read-only DEV diagnostics. No optimizer, checkpoint mutation, or submission."""
from pathlib import Path
import datetime
import importlib.util
import json
import os
import subprocess
import sys
import time
import types

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path('/NHNHOME/data/sukim/adcl')
HERE = ROOT / 'experiments/a2_progress_fourarm_20260921'
sys.path.insert(0, str(HERE))
from verify import datasets, inputs
from arm_model import load_export
from motiondrive_v2_training import tensor_state_sha256, to_device

spec = importlib.util.spec_from_file_location('fourarm_stats', HERE / 'collect.py')
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
REPORT = ROOT / 'reports/a2_next_direction_20260921'
WORK = ROOT / 'work_dirs/a2_next_direction_20260921'
CKPT = ROOT / 'work_dirs/a2_progress_fourarm_20260921/P-CTRL-s1/ckpt_step3426.pth'


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def composition_review():
    _, keys, gt, parent, buckets, sessions = stats.read(stats.PARENT)
    paths = {'PARENT': stats.PARENT,
             'DIRECT': ROOT / 'work_dirs/a2_progress_h4_20260920/A2-H4-DIRECT-s1/final_eval.json'}
    paths.update({a: ROOT / f'work_dirs/a2_progress_fourarm_20260921/{a}-s1/predictions_step3426.json'
                  for a in ('P-CTRL', 'P-VECTOR', 'P-FINE')})
    arrays = {}
    for name, path in paths.items():
        _, k, g, p, b, s = stats.read(path)
        assert k == keys and np.array_equal(g, gt) and np.array_equal(s, sessions)
        dp = np.diff(np.concatenate((np.zeros((len(p), 1, 2)), p), 1), axis=1)
        length = np.linalg.norm(dp, axis=-1)
        assert np.all(length > 0)
        arrays[name] = (p, length, dp / length[..., None])
    dg = np.diff(np.concatenate((np.zeros((len(gt), 1, 2)), gt), 1), axis=1)
    lg = np.linalg.norm(dg, axis=-1)
    ug = dg / np.maximum(lg[..., None], 1e-12)
    out = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'scope': 'saved DEV predictions; no trained new model; no submission; no GT-optimized oracle',
           'GT_direction_mask': 'replace heading only where GT interval length > .05m',
           'sources': {k: {'path': str(v), 'sha256': stats.sha(v)} for k, v in paths.items()},
           'models': {}}
    for name, (p, length, unit) in arrays.items():
        variants = {'original': p,
                    'predicted_length_DIRECT_direction': np.cumsum(length[..., None] * arrays['DIRECT'][2], 1),
                    'GT_length_predicted_direction': np.cumsum(lg[..., None] * unit, 1),
                    'predicted_length_GT_direction_on_valid': np.cumsum(
                        length[..., None] * np.where((lg > .05)[..., None], ug, unit), 1)}
        base, base_d = stats.stats(gt, p, buckets)
        result = {}
        for label, plan in variants.items():
            summary, d = stats.stats(gt, plan, buckets)
            result[label] = {'PREFIX': summary['PREFIX'],
                             'nonstop': summary['groups']['nonstop']['PREFIX'],
                             'L2_1s': summary['L2_1s'], 'L2_2s': summary['L2_2s'], 'L2_3s': summary['L2_3s'],
                             'delta_vs_own_original': stats.compare(d - base_d, sessions)}
        out['models'][name] = result
    write_json(REPORT / 'COMPONENT_RECOMPOSITION.json', out)
    print(json.dumps({'composition': {k: {n: r['PREFIX'] for n, r in v.items()}
                                    for k, v in out['models'].items()}}), flush=True)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    REPORT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    assert not (REPORT / 'FROZEN_FP32_SAMPLING.json').exists()
    composition_review()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    model, payload = load_export(CKPT, 'cuda')
    model.eval().requires_grad_(False)
    before = tensor_state_sha256(model.state_dict())
    _, data = datasets()
    old_path = CKPT.parent / 'predictions_step3426.json'
    _, keys, gt, old, buckets, sessions = stats.read(old_path)
    original = model.scene_encoder._sample
    trace = {}

    def baseline_sample(self, feature, grid):
        out = original(feature, grid)
        if not trace:
            valid = (grid.abs() <= 1).all(-1)
            error = (grid.to(feature.dtype).float() - grid.float()).abs()
            xy_scale = error.new_tensor([feature.shape[-1] / 2, feature.shape[-2] / 2])
            pixel_error = (error * xy_scale)[valid]
            trace.update(feature_dtype=str(feature.dtype), input_grid_dtype=str(grid.dtype),
                         original_sample_output_dtype=str(out.dtype),
                         first_feature_hw=list(feature.shape[-2:]),
                         valid_grid_feature_pixel_rounding_mean=float(pixel_error.mean()),
                         valid_grid_feature_pixel_rounding_max=float(pixel_error.max()))
        return out

    def fp32_sample(self, feature, grid):
        b, v, c, h, w = feature.shape
        q, nh = grid.shape[2:4]
        with torch.autocast(device_type='cuda', enabled=False):
            out = F.grid_sample(feature.flatten(0, 1).float(),
                                grid.reshape(b * v, q, nh, 2).float(),
                                mode='bilinear', padding_mode='zeros', align_corners=False)
        # Preserve the baseline output dtype: change coordinate and sampling precision only.
        dtype = getattr(torch, trace['original_sample_output_dtype'].split('.')[-1])
        return out.to(dtype).reshape(b, v, -1, q, nh).permute(0, 3, 1, 4, 2)

    predictions = {'baseline': [], 'fp32_sampling': []}
    rows = []
    start = time.monotonic()
    with torch.inference_mode():
        for index, raw in enumerate(DataLoader(data, batch_size=8, num_workers=8, pin_memory=True, shuffle=False)):
            batch = to_device(raw, torch.device('cuda'))
            x = inputs(batch)
            outputs = {}
            for name, fn in [('baseline', baseline_sample), ('fp32_sampling', fp32_sample)]:
                model.scene_encoder._sample = types.MethodType(fn, model.scene_encoder)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(**x)
                predictions[name].append(out['plan_abs'].cpu().float().numpy())
                if index == 0:
                    outputs[name] = {k: out[k].cpu().float() for k in ('motion_features', 'state_hat', 'history_hat')}
                del out
            if index == 0:
                trace['motion_state_history_max_abs_delta'] = {
                    k: float((outputs['baseline'][k] - outputs['fp32_sampling'][k]).abs().max())
                    for k in outputs['baseline']}
                assert max(trace['motion_state_history_max_abs_delta'].values()) == 0
            rows.extend(int(r) for r in raw['row'])
            if (index + 1) % 50 == 0:
                print(json.dumps({'rows': len(rows), 'total': len(data), 'seconds': time.monotonic()-start}), flush=True)
    model.scene_encoder._sample = original
    assert before == tensor_state_sha256(model.state_dict())
    assert rows == [k[0] for k in keys]
    predictions = {k: np.concatenate(v).astype(np.float64) for k, v in predictions.items()}
    baseline_max = float(np.abs(predictions['baseline'] - old).max())
    # Compare interventions against their SAME-PROCESS baseline. The first run's
    # overly strict saved-run 1e-5m check found a 0.0001125336m difference; retain
    # that fact rather than calling saved-run reproduction exact. The existing
    # raw cross-process/device contract is 1mm; report both max and mean impact.
    assert baseline_max < 1e-3, baseline_max
    result = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'frozen DEV intervention; no training; not a deployment candidate',
              'checkpoint': str(CKPT), 'checkpoint_sha256': stats.sha(CKPT),
              'baseline_max_abs_delta_vs_saved_m': baseline_max,
              'saved_run_reproduction_is_exact': baseline_max == 0,
              'saved_run_reproduction_tolerance_m': 1e-3,
              'first_attempt_strict_1e_minus_5_m_check_max_delta_m': 0.0001125335693359375,
              'baseline_vs_saved_PREFIX_distance_m': float((np.linalg.norm(predictions['baseline']-old,axis=-1) @ stats.W).mean()),
              'parameter_buffer_unchanged': True, 'optimizer_updates': 0,
              'n': len(rows), 'trace': trace, 'seconds': time.monotonic()-start, 'evaluations': {}}
    ds = {}
    for k, p in predictions.items():
        assert np.isfinite(p).all()
        result['evaluations'][k], ds[k] = stats.stats(gt, p, buckets)
    result['paired_difference'] = stats.compare(ds['fp32_sampling'] - ds['baseline'], sessions)
    result['plan_change_PREFIX_m'] = float((np.linalg.norm(predictions['fp32_sampling']-predictions['baseline'],axis=-1) @ stats.W).mean())
    np.savez_compressed(WORK / 'frozen_fp32_sampling_predictions.npz', row=rows, gt=gt,
                        session=sessions, bucket=buckets, **predictions)
    write_json(REPORT / 'FROZEN_FP32_SAMPLING.json', result)
    print(json.dumps({'status': 'complete', 'baseline': result['evaluations']['baseline']['PREFIX'],
                      'fp32_sampling': result['evaluations']['fp32_sampling']['PREFIX'],
                      'delta': result['paired_difference'], 'trace': trace}), flush=True)


if __name__ == '__main__':
    main()
