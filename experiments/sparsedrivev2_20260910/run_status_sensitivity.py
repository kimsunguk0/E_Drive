"""Frozen SDV2 coarse/fine state interventions; no fitting or output refinement.

Runs the recorded checkpoint source, changes only caller-supplied status tensors,
and reuses a batch's identical image features through a temporary forward hook.
Every output remains an exact row of the checkpoint's fixed trajectory bank.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_checkpoint import (inspect_checkpoint, recorded_runtime,
    load_verified_model, build_dataset, verify_bank_output, rows_sha, WEIGHTS)
from evaluate_relative_selector import compare_reference
from relative_artifact import load_relative_artifact, file_sha, require
from relative_selector import CandidateRelativeSelector
from status_perturbations import build_conditions, perturb_status, protocol as perturbation_protocol


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def nominal_timestamp(name):
    date, clock = str(name).rsplit('_', 1)[-1].split('-')
    # These are local collection clock strings; only differences within a
    # session are used. Explicit UTC avoids host timezone changing the offsets.
    return dt.datetime.strptime(date + clock.zfill(6), '%Y%m%d%H%M%S').replace(
        tzinfo=dt.timezone.utc).timestamp()


def dataset_metadata(dataset):
    scenarios = dataset.scenarios[dataset.scen_idx[dataset.rows]]
    sessions = np.asarray([dataset.manifest['scene_to_session'][s] for s in scenarios])
    times = np.asarray([nominal_timestamp(s) - nominal_timestamp(g) + float(f) / 10.
        for s, g, f in zip(scenarios, sessions, dataset.frames[dataset.rows])])
    require(np.isfinite(times).all() and (times >= 0).all(), 'Invalid nominal session times')
    return {'rows': dataset.rows.copy(), 'scenario': scenarios, 'session': sessions,
            'nominal_time_s': times}


def serializable_condition(condition):
    if dataclasses.is_dataclass(condition):
        return dataclasses.asdict(condition)
    return dict(condition)


def condition_name(condition):
    return serializable_condition(condition)['name']


def check_same_outputs(left, right):
    for key in ('candidate_ids', 'candidate_xy', 'candidate_valid', 'scores',
                'selected_candidate_id', 'trajectory'):
        require(torch.equal(left[key], right[key]), f'Cached-backbone parity failed: {key}')


def score_interventions(selector, original_output, changed_output, original_status,
                        changed_status, goal):
    """Keep the other branch fixed; both uses the SAME error in both branches."""
    return {
        'fine': selector.rescore(original_output, changed_status, goal),
        'coarse': selector.rescore(changed_output, original_status, goal),
        'both': selector.rescore(changed_output, changed_status, goal),
    }


def gather_metrics(output, gt, original):
    # Oracle/GT exist only AFTER model selection. No costs enter either head.
    candidates = output['candidate_xy'].double()
    point_cost = torch.linalg.vector_norm(candidates - gt.double()[:, None], dim=-1)
    weights = torch.as_tensor(WEIGHTS, device=gt.device, dtype=torch.float64)
    costs = (point_cost * weights).sum(-1).masked_fill(~output['candidate_valid'], torch.inf)
    oracle, oracle_index = costs.min(-1)
    pred = output['trajectory'].float()
    point_l2 = torch.linalg.vector_norm(pred.double() - gt.double(), dim=-1)
    ids = output['candidate_ids']
    ref_ids = original['candidate_ids']
    baseline_winner_present = (ids == original['selected_candidate_id'][:, None]).any(-1)
    # Group IDs form a unique path x velocity product, so matches count set overlap.
    retained = (ids[:, :, None] == ref_ids[:, None, :]).any(1).sum(-1)
    return {'pred': pred, 'candidate_id': output['selected_candidate_id'],
            'point_l2': point_l2, 'error_xy': pred.double() - gt.double(),
            'd3': (point_l2 * weights).sum(-1), 'shortlist_oracle': oracle,
            'oracle_candidate_id': ids.gather(1, oracle_index[:, None]).squeeze(1),
            'baseline_winner_present': baseline_winner_present,
            'baseline_candidates_retained': retained}


def append_chunks(chunks, values):
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        chunks.setdefault(key, []).append(value)


def summarize(arrays, baseline, metadata):
    result = {'n': len(arrays['d3']), 'd3': float(arrays['d3'].mean()),
        'delta_d3': float((arrays['d3'] - baseline['d3']).mean()),
        'shortlist_oracle': float(arrays['shortlist_oracle'].mean()),
        'delta_shortlist_oracle': float((arrays['shortlist_oracle'] - baseline['shortlist_oracle']).mean()),
        'selection_regret': float((arrays['d3'] - arrays['shortlist_oracle']).mean()),
        'three_second_l2': float(arrays['point_l2'][:, -1].mean()),
        'point_l2': arrays['point_l2'].mean(0).tolist(),
        'changed_winner_fraction': float(np.mean(arrays['candidate_id'] != baseline['candidate_id'])),
        'baseline_winner_present_fraction': float(arrays['baseline_winner_present'].mean()),
        'baseline_candidates_retained_mean': float(arrays['baseline_candidates_retained'].mean()),
        'oracle_over_015_fraction': float(np.mean(arrays['shortlist_oracle'] > .15)),
        'sessions': {}}
    for session in np.unique(metadata['session']):
        mask = metadata['session'] == session
        result['sessions'][str(session)] = {'n': int(mask.sum()),
            'd3': float(arrays['d3'][mask].mean()),
            'delta_d3': float((arrays['d3'][mask] - baseline['d3'][mask]).mean()),
            'oracle': float(arrays['shortlist_oracle'][mask].mean())}
    return result


@torch.inference_mode()
def evaluate(selector, dataset, runtime, perturbations, directory, batch_size, workers, metadata):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    chunks = {'baseline': {}}
    for name in perturbations:
        for mode in ('coarse', 'fine', 'both'):
            chunks[f'{name}__{mode}'] = {}
    completed_rows = 0
    begin = time.time()
    cache_parity_batches = 0
    for batch_index, batch in enumerate(loader):
        n = len(batch['row'])
        indices = slice(completed_rows, completed_rows + n)
        require(np.array_equal(batch['row'].numpy(), dataset.rows[indices]), 'Batch row order changed')
        x = {k: v.to('cuda:0', non_blocking=True) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
        inputs = runtime.data.model_inputs(x, goal_selection=True)
        goal = inputs.pop('goal_xy')
        require(set(inputs) == {'images', 'lidar2img', 'image_hw', 'status'}, 'Unexpected model input')
        cached = []
        handle = selector.base._backbone.register_forward_hook(lambda _m, _i, out: cached.append(out))
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                original_base = selector.base(**inputs)
                original = selector.rescore(original_base, inputs['status'], goal)
        finally:
            handle.remove()
        require(len(cached) == 1, 'Expected exactly one image feature extraction')
        verify_bank_output(selector, original, runtime)
        if batch_index == 0:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                normal_wrapper = selector(**inputs, goal_xy=goal)
            check_same_outputs(original, normal_wrapper)
        append_chunks(chunks['baseline'], {'rows': batch['row'], **gather_metrics(original, x['gt_plan'], original)})
        image_object = inputs['images']

        def reused_backbone(images):
            require(images is image_object, 'Cached features requested for a different image tensor')
            return cached[0]

        with patch.object(selector.base._backbone, 'forward', side_effect=reused_backbone):
            # First and final batch certify reuse does not change numerical output.
            if batch_index == 0 or completed_rows + n == len(dataset):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    replay = selector.base(**inputs)
                check_same_outputs(original_base, replay)
                cache_parity_batches += 1
            for name, changed_status_all in perturbations.items():
                changed_status = torch.from_numpy(changed_status_all[indices]).to('cuda:0')
                changed_inputs = {**inputs, 'status': changed_status}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    changed = selector.base(**changed_inputs)
                    outputs = score_interventions(selector, original_base, changed,
                                                  inputs['status'], changed_status, goal)
                verify_bank_output(selector, changed, runtime)
                for mode, output in outputs.items():
                    verify_bank_output(selector, output, runtime)
                    append_chunks(chunks[f'{name}__{mode}'], {'rows': batch['row'],
                        **gather_metrics(output, x['gt_plan'], original)})
        completed_rows += n
        if batch_index % 25 == 0 or completed_rows == len(dataset):
            progress = {'rows': completed_rows, 'total': len(dataset),
                'elapsed_seconds': time.time() - begin, 'conditions': len(perturbations)}
            write_json(directory / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    arrays = {name: {key: np.concatenate(values) for key, values in fields.items()}
              for name, fields in chunks.items()}
    baseline = arrays['baseline']
    summaries = {}
    for name, values in arrays.items():
        require(np.array_equal(values['rows'], dataset.rows), 'Output population changed')
        require(np.isfinite(values['d3']).all(), 'Nonfinite metric')
        np.savez_compressed(directory / f'{name}.npz', **values)
        summaries[name] = summarize(values, baseline, metadata)
        if name.endswith('__fine'):
            require(np.array_equal(values['shortlist_oracle'], baseline['shortlist_oracle']),
                    'Fine-only intervention changed the shortlist oracle')
            require((values['baseline_candidates_retained'] == 200).all(), 'Fine-only shortlist changed')
    return baseline, summaries, cache_parity_batches


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--head', required=True)
    p.add_argument('--head-sha256', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--limit', type=int, default=0, help='Canary only; first fixed batches, never headline metrics')
    p.add_argument('--conditions', help='Comma separated exact names, optional')
    p.add_argument('--empirical-status', help='Optional immutable NPZ: rows and status8, one measured replacement condition')
    a = p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('0', '1', '4'), 'Use one allocated GPU')
    require(0 <= a.shard < a.shards and a.batch > 0 and a.workers >= 0, 'Invalid execution arguments')
    directory = Path(a.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.time()
    try:
        head, cache, head_receipt = load_relative_artifact(a.head, expected_sha256=a.head_sha256)
        plan = inspect_checkpoint(cache['checkpoint']['path'], bank_path=cache['bank']['path'], population='tune')
        require(plan.goal_mode == 'none', 'This diagnostic fixes goal to final relative scoring only')
        with recorded_runtime(plan) as runtime:
            base, coverage = load_verified_model(plan, runtime)
            selector = CandidateRelativeSelector(base, base_goal_mode='none', freeze_base=True)
            selector.relative_head.load_state_dict(head.state_dict(), strict=True)
            dataset_plan = copy.copy(plan)
            dataset_plan.goal_mode = 'selection'
            dataset = build_dataset(dataset_plan, runtime)
            full_metadata = dataset_metadata(dataset)
            conditions = list(build_conditions())
            conditions = [c for c in conditions if condition_name(c) not in ('baseline', 'identity', 'none')]
            if a.conditions:
                names = a.conditions.split(',')
                require(set(names) <= {condition_name(c) for c in conditions}, 'Unknown condition')
                conditions = [c for c in conditions if condition_name(c) in names]
            perturbations = {condition_name(c): perturb_status(dataset.status,
                full_metadata['session'], full_metadata['nominal_time_s'], c, seed=20260910)
                for c in conditions}
            definitions = {condition_name(c): serializable_condition(c) for c in conditions}
            if a.empirical_status:
                with np.load(a.empirical_status, allow_pickle=False) as z:
                    require(np.array_equal(z['rows'], dataset.rows), 'Empirical status rows differ')
                    require(z['status8'].shape == dataset.status.shape and np.isfinite(z['status8']).all(),
                            'Invalid empirical status')
                    perturbations['empirical_p7_status'] = z['status8'].astype(np.float32)
                    require(np.isfinite(perturbations['empirical_p7_status']).all() and
                            (perturbations['empirical_p7_status'][:, :4] == 0).all(),
                            'Empirical status conversion is nonfinite or has nonzero commands')
                definitions['empirical_p7_status'] = {'path': str(Path(a.empirical_status).resolve()),
                    'sha256': file_sha(a.empirical_status), 'kind': 'measured_replacement'}
            names = list(perturbations)[a.shard::a.shards]
            require(bool(names), 'No conditions selected for this shard')
            perturbations = {name: perturbations[name] for name in names}
            definitions = {name: definitions[name] for name in names}
            if a.limit:
                dataset.rows = dataset.rows[:a.limit]
                dataset.status = dataset.status[:a.limit]
                perturbations = {k: v[:a.limit] for k, v in perturbations.items()}
            metadata = {k: v[:len(dataset)] for k, v in full_metadata.items()}
            np.savez_compressed(directory / 'inputs.npz', **metadata,
                original_status8=dataset.status, **{f'status__{k}': v for k, v in perturbations.items()})
            receipt = {'schema': 'sdv2_status_sensitivity_v1', 'arguments': vars(a),
                'base': plan.receipt, 'head': head_receipt, 'definitions': definitions,
                'rows_sha256': rows_sha(dataset.rows), 'n': len(dataset),
                'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'torch': str(torch.__version__),
                'numpy': str(np.__version__),
                'precision': 'bf16_base_fp32_head_float64_metrics', 'fit_performed': False,
                'goal_mode': 'final_selection_only', 'canary': bool(a.limit),
                'perturbation_protocol': perturbation_protocol(),
                'intervention_modes': {
                    'coarse': 'Original base status input: affects coarse shortlist, downstream latent features and base scores; not a pure pruning intervention',
                    'fine': 'Relative CE head status only; original base output and shortlist fixed',
                    'both': 'Same perturbed state at base input and relative CE head'},
                'time_contract': 'scene clock minus session origin plus frame/10; nominal simulation time',
                'interpretation': 'Frozen input intervention; not a retrained student performance bound',
                'dataset': dataset.provenance(),
                'source_sha256': {name: file_sha(Path(__file__).with_name(name)) for name in
                    ('run_status_sensitivity.py', 'status_perturbations.py', 'relative_selector.py',
                     'relative_artifact.py', 'evaluate_checkpoint.py')}}
            write_json(directory / 'protocol.json', receipt)
            selector.cuda().eval()
            torch.cuda.reset_peak_memory_stats()
            baseline, summaries, parity_batches = evaluate(selector, dataset, runtime, perturbations,
                directory, a.batch, a.workers, metadata)
            if not a.limit:
                reference = Path(a.head).parent / f"eval_{head_receipt['relative_step']:06d}.npz"
                receipt['baseline_reference'] = compare_reference(baseline, reference,
                    allow_batch_numerical_differences=a.batch != 8)
            error_statistics = {}
            for name, changed in perturbations.items():
                error = changed[:, 4:8].astype(np.float64) - dataset.status[:, 4:8]
                error_statistics[name] = {'bias': error.mean(0).tolist(),
                    'mae': np.abs(error).mean(0).tolist(), 'rmse': np.sqrt((error**2).mean(0)).tolist(),
                    'p95_absolute': np.quantile(np.abs(error), .95, axis=0).tolist()}
            result = {**receipt, 'status': 'completed', 'summaries': summaries,
                'error_statistics_vx_vy_ax_ay': error_statistics,
                'backbone_reuse_exact_parity_batches': parity_batches,
                'elapsed_seconds': time.time() - started,
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
            write_json(directory / 'result.json', result)
            print(json.dumps({'status': 'completed', 'output': str(directory),
                'n': len(dataset), 'conditions': len(perturbations), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    except BaseException:
        write_json(directory / 'failure.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
