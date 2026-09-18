"""Frozen-checkpoint diagnostics of precision and SIDE's unweighted scene mean.

No training, output fitting, GT correction, or changes to production source.
"""
from pathlib import Path
import argparse
import datetime
import inspect
import json
import os
import sys
import textwrap
import time
import types
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path('/NHNHOME/data/sukim/adcl')
sys.path.insert(0, str(ROOT / 'experiments/md_a2_scene_extensions_20260918'))
from preflight import nominal, legacy, mr, inputs, to_device, A2NominalModel, NominalStatusDataset
from scene_extensions import SceneExtensionModel, SIDE
from side_data import SideSceneDataset
from models.motiondrive_v2 import MotionDriveV2Config
from models.motiondrive_v2 import scene_encoder as scene_module

REPORT = ROOT / 'reports/md_a2_bottleneck_review_20260918'
WEIGHTS = np.asarray([11, 11, 5, 5, 2, 2], np.float64) / 36


def front_only_base_function():
    source = textwrap.dedent(inspect.getsource(scene_module.SharedSceneEncoder.forward))
    old = 'base = (value * valid[..., None]).sum(2) / valid.sum(2).clamp_min(1)[..., None]'
    assert source.count(old) == 1
    indent = source[:source.index(old)].split('\n')[-1]
    assert not indent.strip()
    new = ('source_in_level = torch.arange(valid.shape[-1], device=valid.device) % (valid.shape[-1] // len(projected_levels))\n'
           + indent + 'base_valid = valid & (source_in_level < 10 * len(self.config.heights))[None, None]\n'
           + indent + 'base = (value * base_valid[..., None]).sum(2) / base_valid.sum(2).clamp_min(1)[..., None]')
    source = source.replace(old, new).replace('def forward(', 'def front_only_base_forward(', 1)
    namespace = dict(vars(scene_module))
    exec(source, namespace)
    return namespace['front_only_base_forward'], source


def sample_fp32(self, feature, grid):
    b, v, c, h, w = feature.shape
    q, nh = grid.shape[2:4]
    with torch.autocast(device_type=feature.device.type, enabled=False):
        sampled = F.grid_sample(feature.flatten(0, 1).float(), grid.reshape(b*v, q, nh, 2).float(),
                                mode='bilinear', padding_mode='zeros', align_corners=False)
    # Preserve the observed original output dtype; only sampler input precision changes.
    sampled = sampled.to(self._diagnostic_original_sample_dtype)
    return sampled.reshape(b, v, -1, q, nh).permute(0, 3, 1, 4, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('precision', 'side', 'fp32'), required=True)
    parser.add_argument('--gpu', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    # Preserve the original trainer's cuDNN policy for AMP replay. The explicit
    # full-FP32 diagnostic disables TF32 so it actually tests FP32 arithmetic.
    torch.backends.cudnn.allow_tf32 = args.kind != 'fp32'
    arm = SIDE if args.kind == 'side' else 'A2-BASE-NOM'
    group = 'md_a2_scene_extensions_20260918' if args.kind == 'side' else 'md_a2_nominal_mh4_20260918'
    run = ROOT / 'work_dirs' / group / f'{arm}-s1'
    manifest = json.loads((run / 'manifest.json').read_text())
    cfg = MotionDriveV2Config(**manifest['model_config'])
    model = (SceneExtensionModel(cfg, arm=arm) if args.kind == 'side' else A2NominalModel(cfg, arm=arm))
    mr.rebuild_correlation_fuse(model, 4)
    checkpoint = torch.load(run / 'ckpt_step20554.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model.eval().cuda().requires_grad_(False)
    _, tune = nominal.raw_datasets(False, 1)
    data = mr.MotionCanvasDataset(tune, 'native')
    if args.kind == 'side': data = SideSceneDataset(data)
    data = NominalStatusDataset(data)
    loader = DataLoader(data, batch_size=8, num_workers=8, pin_memory=True, shuffle=False)
    stored = json.loads((run / 'final_eval.json').read_text())['records']
    reference = np.asarray([r['pred_abs_xy'] for r in stored], np.float64)
    expected_rows = np.asarray([r['row'] for r in stored])
    modes = {'precision': ('BF16', 'BF16_FP32_SAMPLER', 'FP16'),
             'side': ('SIDE_original', 'SIDE_no_extra_images', 'SIDE_original_sources_in_base'),
             'fp32': ('FP32',)}[args.kind]
    predictions = {m: [] for m in modes}
    elapsed = {m: 0.0 for m in modes}
    ground, row_ids = [], []
    encoder = model.scene_encoder
    original_forward, original_sample = encoder.forward, encoder._sample
    side_forward, side_source = front_only_base_function()
    observed = {}
    def sample_observer(self, feature, grid):
        result = original_sample(feature, grid)
        self._diagnostic_original_sample_dtype = result.dtype
        if not observed:
            valid = (grid.abs() <= 1).all(-1)
            delta = (grid.to(feature.dtype).float() - grid.float()).abs()
            pixels = delta * grid.new_tensor([384., 216.])
            selected = pixels[valid].float().cpu().numpy()
            observed.update(feature_dtype=str(feature.dtype), geometry_dtype=str(grid.dtype),
                original_sample_output_dtype=str(result.dtype), sample_count=len(selected),
                coordinate_rounding_abs_pixels_mean=selected.mean(0).tolist(),
                coordinate_rounding_abs_pixels_p95=np.percentile(selected, 95, axis=0).tolist(),
                coordinate_rounding_abs_pixels_max=selected.max(0).tolist())
        return result
    start = time.monotonic()
    with torch.inference_mode():
        for index, raw in enumerate(loader):
            batch = to_device(raw, torch.device('cuda:0'))
            x = inputs(batch, side=args.kind == 'side')
            ground.append(raw['gt_plan'].numpy())
            row_ids.append(raw['row'].numpy())
            for mode in modes:
                encoder.forward = original_forward
                encoder._sample = types.MethodType(sample_observer, encoder)
                if args.kind == 'side':
                    model.side_scene_enabled = mode != 'SIDE_no_extra_images'
                    if mode == 'SIDE_original_sources_in_base':
                        encoder.forward = types.MethodType(side_forward, encoder)
                if mode == 'BF16_FP32_SAMPLER': encoder._sample = types.MethodType(sample_fp32, encoder)
                torch.cuda.synchronize()
                then = time.monotonic()
                with torch.autocast('cuda', dtype=torch.float16 if mode == 'FP16' else torch.bfloat16,
                                    enabled=mode != 'FP32'):
                    out = model(**x)
                assert torch.isfinite(out['plan_abs']).all(), mode
                predictions[mode].append(out['plan_abs'].float().cpu().numpy())
                torch.cuda.synchronize()
                elapsed[mode] += time.monotonic() - then
                del out
            if (index + 1) % 50 == 0: print(json.dumps({'kind': args.kind, 'rows': min((index+1)*8, len(data)), 'elapsed_s': time.monotonic()-start}), flush=True)
    rows = np.concatenate(row_ids)
    gt = np.concatenate(ground).astype(np.float64)
    assert np.array_equal(rows, expected_rows)
    assert np.array_equal(gt, np.asarray([r['gt_abs_xy'] for r in stored]))
    bucket = np.asarray([r['bucket'] for r in stored])
    summary = dict(kind=args.kind, gpu=args.gpu, checkpoint=str(run / 'ckpt_step20554.pth'),
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), training_performed=False,
        model_parameters_unchanged=True, production_source_modified=False, reference_PREFIX=float((np.linalg.norm(reference-gt,axis=-1)@WEIGHTS).mean()),
        first_sampler_observation=observed, modes={}, n=len(rows), total_seconds=time.monotonic()-start,
        warning='Frozen-weight interventions; out-of-training-distribution changes do not establish the performance of retraining.')
    arrays = {'row': rows, 'gt': gt, 'reference': reference}
    for mode in modes:
        pred = np.concatenate(predictions[mode]).astype(np.float64)
        error = np.linalg.norm(pred-gt,axis=-1); score=error@WEIGHTS
        arrays[mode] = pred
        parity = float(np.abs(pred-reference).max())
        if mode in ('BF16','SIDE_original'):
            assert parity <= 5e-4 and abs(score.mean()-summary['reference_PREFIX']) < 1e-6, (mode, parity)
        summary['modes'][mode] = dict(PREFIX=float(score.mean()),
            delta_vs_reference=float(score.mean()-summary['reference_PREFIX']),
            first2s_PREFIX_contribution=float((error[:,:4]*WEIGHTS[None,:4]).sum(1).mean()),
            nonstop_PREFIX=float(score[bucket=='nonstop'].mean()),
            max_abs_plan_change_vs_reference=parity, forward_seconds=elapsed[mode])
    REPORT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REPORT / f'{args.kind}_predictions.npz', **arrays)
    (REPORT / f'{args.kind}.json').write_text(json.dumps(summary,indent=2)+'\n')
    if args.kind=='side': (REPORT / 'side_base_intervention_source.txt').write_text(side_source)
    print('RESULT '+json.dumps(summary),flush=True)


if __name__ == '__main__': main()
