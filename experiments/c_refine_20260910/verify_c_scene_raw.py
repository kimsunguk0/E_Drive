"""B1 raw/cache parity for a new scene head or C continuation, without GT input.

New payload inspection/loading is delegated to evaluate_c_scene_selector.py.
Only the original C payload is passed to the legacy temporal strict loader.
The old raw adapter/verifier are reused unchanged, by pinned source hashes.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np
import torch

OLD_EVALUATOR_SHA = '9007a4517adaf663d402535bdd56dd5d8340ad675db0f3a3ef44d8d867529242'
RAW_HELPER_SHA = '09a9e6e476143d51c4dd4f6f20c77760176c9144022b40d842a095c52e1598bb'
RAW_ADAPTER_SHA = 'cc0339425ab2bc43553b4b16ae4c0811f248f96a370417afb0740b4c47425f20'
PIXEL_SHA = '33b64b3fe87a46e6b28587282e5615fbb69bb745d5ee32afd0e5d20ae56e3a29'
FIXTURE_SHA = '9a141770df94f83ef461069430322bf1d86ba2deb12d647b9c88e236e59f98b0'
C_SHA = 'b9dcc56af7c2c4c3cfc80d844fe862a2d8f730d7b8d7a813ee997d33abfec2ff'
LIVE_EVALUATOR_SHA = '647bb3653b96ad1c9735c8df8ec78f9c1b3e50f9a96af5dc96087d30f61de048'
OUTPUT_FIELDS = ('trajectory', 'selected_candidate_id', 'candidate_ids', 'candidate_xy',
    'candidate_valid', 'candidate_tokens', 'scores', 'old_final_scores', 'scene_score_residual',
    'base_scores', 'aux_state', 'aux_occ', 'aux_lane')
INPUT_FIELDS = {'images', 'history_images', 'lidar2img', 'image_hw', 'time_offsets', 'perception_status'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temp.replace(path)


def module_from(path, prefix):
    name = prefix+'_'+uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve())
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked_sources(a):
    raw = Path(a.raw_helper_dir)
    sources = {'verify_c_scene_raw.py': (Path(__file__), sha(__file__)),
        'evaluate_c_scene_selector.py': (Path(a.live_evaluator_source), LIVE_EVALUATOR_SHA),
        'evaluate_temporal_checkpoint.py': (Path(a.original_evaluator_source), OLD_EVALUATOR_SHA),
        'verify_temporal_raw_inference.py': (raw/'verify_temporal_raw_inference.py', RAW_HELPER_SHA),
        'temporal_deployment.py': (raw/'temporal_deployment.py', RAW_ADAPTER_SHA)}
    for name, (path, expected) in sources.items():
        require(expected is not None and sha(path) == expected, 'Source pin differs or is not frozen: '+name)
    return sources


def inspect_composition(a, live, ev):
    """Avoid sending a head/continuation payload into the legacy C loader."""
    scene = live.inspect_scene_checkpoint(a.checkpoint, ev)
    original_path = a.original_c_checkpoint or scene['config']['original_c_checkpoint']
    require(Path(original_path).resolve() != Path(a.checkpoint).resolve(), 'New head cannot be its own original C')
    plan = ev.inspect_checkpoint(original_path, worktree=a.worktree, base=a.base)
    require(plan.receipt['checkpoint_sha256'] == C_SHA, 'Original C hash differs')
    require(plan.manifest['arguments']['common_status'] is True
            and plan.manifest['arguments']['history_mode'] == 'real', 'Expected the original C arm')
    require(scene['manifest']['bank_sha256'] == plan.receipt['bank_sha256'], 'Composed model bank differs')
    return scene, plan


def checked_prepared(prepared):
    require(set(prepared.inputs) == INPUT_FIELDS, 'Raw model input whitelist differs')
    require(set(prepared.selector_inputs) == {'goal_xy'}, 'Raw goal must remain a separate selector input')
    require(prepared.metadata['pixel_dependency_sha256'] == PIXEL_SHA, 'Pixel dependency source differs')
    require(prepared.metadata['labels_read'] is False and prepared.metadata['aux_cache_read'] is False
            and prepared.metadata['status_cache_read'] is False, 'Raw adapter read a training cache')
    require(prepared.metadata['planner_status_input_present'] is False
            and prepared.metadata['goal_only_in_selector_inputs'] is True, 'Raw planner/goal input bypass')
    require(all(isinstance(x, torch.Tensor) and x.shape[0] == 1
                for x in [*prepared.inputs.values(), *prepared.selector_inputs.values()]), 'Raw verify requires B1')
    return {**prepared.inputs, **prepared.selector_inputs}


@contextmanager
def new_head_boundary(head):
    """Check direct state slots of the new head, independent of common tokens."""
    require(set(inspect.signature(head.forward).parameters) == {'output', 'goal_xy'},
            'New scene head signature permits unexpected direct inputs')
    counts = {'new_score_head_calls': 0}
    def check(_module, args):
        features = args[0]
        require(features.ndim == 3 and features.shape[-1] == 96, 'Unexpected new scene feature layout')
        require(bool((features[..., 22:26] == 0).all()), 'Direct state entered new scene head feature slots')
        counts['new_score_head_calls'] += 1
    handle = head.score_head.register_forward_pre_hook(check)
    try:
        yield counts
    finally:
        handle.remove()


def save_output_pair(output_dir, raw, cached, raw_helper, row):
    """Persist both actual outputs before recording any parity failure."""
    raw_fields = {key: raw[key] for key in OUTPUT_FIELDS}
    cache_fields = {key: cached[key] for key in OUTPUT_FIELDS}
    arrays = {'rows': np.asarray([row], np.int64)}
    for name, values in (('raw', raw_fields), ('cache', cache_fields)):
        for key, tensor in values.items():
            value = tensor.detach().cpu()
            arrays[name+'_'+key] = value.float().numpy() if value.dtype == torch.bfloat16 else value.numpy()
    np.savez_compressed(Path(output_dir)/'outputs.npz', **arrays)
    return raw_helper.tensor_comparison(raw_fields, cache_fields)


def arguments():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--checkpoint', required=True, help='New scene head or continuation; never original C')
    p.add_argument('--original-c-checkpoint')
    p.add_argument('--live-evaluator-source', required=True)
    p.add_argument('--original-evaluator-source', required=True)
    p.add_argument('--raw-helper-dir', required=True)
    p.add_argument('--fixture', required=True)
    p.add_argument('--worktree', required=True)
    p.add_argument('--base', default='/NHNHOME/data/sukim/adcl')
    p.add_argument('--output', required=True)
    p.add_argument('--gpu', type=int, choices=(0, 1, 4))
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--camera-workers', type=int, choices=(1, 3), default=3)
    p.add_argument('--opencv-threads', type=int, default=1)
    return p.parse_args()


@torch.inference_mode()
def main():
    a = arguments()
    root = Path(a.output).resolve()
    require(not root.exists(), 'Use a new immutable verification output')
    require(a.audit_only or a.gpu is not None, 'GPU execution requires an explicitly assigned device')
    require(not a.audit_only or os.environ.get('CUDA_VISIBLE_DEVICES', '') == '', 'CPU audit must hide GPUs')
    sources = checked_sources(a)
    require(sha(Path(a.fixture)/'fixture_receipt.json') == FIXTURE_SHA, 'Use the fixed tune14730 raw fixture')
    import cv2
    require(a.opencv_threads >= 1, 'Invalid OpenCV thread count')
    cv2.setNumThreads(a.opencv_threads)
    require(cv2.getNumThreads() == a.opencv_threads, 'OpenCV threads were not applied')
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(Path(a.worktree).resolve()))
    root.mkdir(parents=True); captured = root/'verification_source'; captured.mkdir()
    for name, (path, _) in sources.items():
        shutil.copy2(path, captured/name)
    started = time.monotonic()
    try:
        live = module_from(captured/'evaluate_c_scene_selector.py', '_c_raw_live')
        ev = module_from(captured/'evaluate_temporal_checkpoint.py', '_c_raw_original')
        helper = module_from(captured/'verify_temporal_raw_inference.py', '_c_raw_helpers')
        adapter_module = module_from(captured/'temporal_deployment.py', '_c_raw_adapter')
        fixture = helper.validate_fixture(a.fixture)
        require(fixture['row'] == 14730, 'Only the preselected tune14730 row is supported')
        scene, plan = inspect_composition(a, live, ev)
        require(fixture['split_sha256'] == ev.SPLIT_SHA and fixture['ego_sha256'] == ev.EGO_SHA,
                'Raw fixture split/ego identity differs')
        gpu = None if a.audit_only else ev.check_gpu(a.gpu)
        receipt = {'schema': 'c_scene_raw_B1_v1', 'status': 'incomplete', 'row': fixture['row'],
            'arguments': vars(a), 'new_checkpoint_sha256': scene['sha256'],
            'new_manifest_sha256': scene['manifest_sha256'], 'new_checkpoint_step': int(scene['payload']['step']),
            'new_checkpoint_schema': scene['manifest']['schema'], 'config': scene['config'],
            'original_c': plan.receipt, 'gpu': gpu, 'batch_size': 1,
            'raw_fixture_receipt_sha256': FIXTURE_SHA, 'source_sha256': {k: v[1] for k, v in sources.items()},
            'GT_passed_to_model': False, 'GT_metrics_computed': False,
            'cache_reference_dataset_materializes_outer_label_arrays': True,
            'reference_label_values_used_in_forward_or_comparison': False,
            'precision': 'bf16_original_C_fp32_old_relative_and_new_scene_head',
            'comparison_scope': 'same row and same B1 model calls from raw vs cached image loader, not the B8 token cache'}
        with ev.isolated_runtime(plan.source) as runtime:
            model, original, head = live.load_verified_models(scene, plan, runtime, ev)
            if 'continuation_buffers_verified' in scene:
                receipt['continuation_buffers_verified'] = scene['continuation_buffers_verified']
            if 'frozen_head_parent_verified' in scene:
                receipt['frozen_head_parent_verified'] = scene['frozen_head_parent_verified']
            dataset, provenance = ev.build_dataset(plan, runtime)
            batch = helper.cache_sample(dataset, plan.rows, fixture['row'])
            require(batch['scenario'] == [fixture['scene']] and batch['session'] == [fixture['session']]
                    and int(batch['frame'][0]) == fixture['frame'], 'Fixture row/scene/frame/session mismatch')
            cached_cpu = ev.input_tensors(batch, runtime, True, 'cpu')
            with adapter_module.TemporalRawInputAdapter('real', True, 'selection', a.camera_workers) as adapter:
                prepared = adapter.prepare_clip(a.fixture)
                raw_cpu = checked_prepared(prepared)
                parity = helper.tensor_comparison(raw_cpu, cached_cpu)
                receipt.update(input_parity=parity, all_inputs_bitwise_equal=all(v['bitwise_equal'] for v in parity.values()),
                    raw_adapter=prepared.metadata, dataset_provenance=provenance)
                if a.audit_only:
                    receipt.update(CPU_audit_only=True, model_forward_performed=False)
                else:
                    model.to('cuda:0').eval()
                    raw_gpu = {k: v.to('cuda:0') for k, v in raw_cpu.items()}
                    cached_gpu = {k: v.to('cuda:0') for k, v in cached_cpu.items()}
                    with ev.route_monitor(original, True) as old_routes, new_head_boundary(head) as new_routes:
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            raw_out = model(**raw_gpu)
                            cache_out = model(**cached_gpu)
                    ev.verify_rows(original, raw_out); ev.verify_rows(original, cache_out)
                    comparison = save_output_pair(root, raw_out, cache_out, helper, fixture['row'])
                    with new_head_boundary(head) as goal_head_routes:
                        goal_audit = live.audit_goal_routes(model, original, raw_gpu, ev)
                    original_audit = ev.sample_route_audit(original, raw_gpu, True)
                    receipt.update(output_parity=comparison,
                        all_outputs_bitwise_equal=all(v['bitwise_equal'] for v in comparison.values()),
                        selected_trajectory_bitwise_equal=comparison['trajectory']['bitwise_equal'],
                        selected_id_equal=comparison['selected_candidate_id']['bitwise_equal'],
                        candidate_tokens_bitwise_equal=comparison['candidate_tokens']['bitwise_equal'],
                        scores_bitwise_equal=comparison['scores']['bitwise_equal'],
                        original_route_counts=old_routes, new_head_route_counts=new_routes,
                        goal_audit=goal_audit, goal_audit_new_head_counts=goal_head_routes,
                        original_status_and_goal_audit=original_audit,
                        outputs_sha256=sha(root/'outputs.npz'), model_forward_performed=True)
                    if a.profile:
                        profile = ev.profile_model(model, raw_gpu, 10, 50)
                        profile['scope'] = 'full original C/continuation + token capture/alignment checks + new final head; GPU-resident B1'
                        profile['excluded'] = 'I/O/raw decode/H2D/GT/outer verifier route hooks; capture checks are included'
                        receipt['model_only_latency'] = profile
                        receipt['preprocessing_latency'] = helper.preprocessing_profile(adapter, a.fixture, 2, 10)
            receipt['runtime_environment'] = helper.runtime_environment(plan.base)
        require(sha(a.checkpoint) == scene['sha256'], 'New checkpoint changed during verification')
        receipt.update(status='completed', elapsed_seconds=time.monotonic()-started, rtx4090_measured=False)
        atomic_json(root/'receipt.json', receipt)
        print(json.dumps({k: receipt.get(k) for k in ('status', 'all_inputs_bitwise_equal',
            'all_outputs_bitwise_equal', 'selected_id_equal', 'candidate_tokens_bitwise_equal')}), flush=True)
    except BaseException as exc:
        atomic_json(root/'failure.json', {'type': type(exc).__name__, 'message': str(exc),
            'elapsed_seconds': time.monotonic()-started, 'source_sha256': sha(__file__)})
        raise


if __name__ == '__main__':
    main()
