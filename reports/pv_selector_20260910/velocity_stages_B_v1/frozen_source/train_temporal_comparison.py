"""Matched public-init temporal image / common-perception comparison.

Goal is used only AFTER the base has returned completed bank candidates.
Raw causal status is never forwarded to the base planner or the relative head.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import PlanDataset, d3, file_sha, rows_sha, TIME_WEIGHTS
from losses import selection_loss
from public_model import PublicSparseDriveV2
from relative_selector import CandidateRelativeSelector
from temporal_data import TemporalPlanDataset, model_inputs as temporal_model_inputs
from temporal_model import TemporalPerceptionModel
from train import seed_all, fixed_bn_train, verify_initialization, verify_output, save_checkpoint, atomic_json

ROOT = Path(__file__).resolve().parents[2]


class FinalGoalTemporalSelector(CandidateRelativeSelector):
    def __init__(self, base):
        super().__init__(base, base_goal_mode='none', freeze_base=False)

    def forward(self, images, lidar2img, image_hw, history_images, time_offsets,
                perception_status=None, goal_xy=None):
        inputs = dict(images=images, lidar2img=lidar2img, image_hw=image_hw,
                      history_images=history_images, time_offsets=time_offsets)
        if perception_status is not None:
            inputs['perception_status'] = perception_status
        output = self.base(**inputs)
        # These constants preserve the existing relative-feature implementation;
        # no measured or image-predicted state enters these score features.
        zeros = torch.zeros((images.shape[0], 8), device=images.device, dtype=torch.float32)
        return self.rescore(output, zeros, goal_xy)


def transfer(batch):
    return {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def selected_inputs(batch, common_status):
    inputs = temporal_model_inputs(batch, common_status=common_status)
    require = {'images', 'lidar2img', 'image_hw', 'history_images', 'time_offsets'}
    if common_status:
        require.add('perception_status')
    if set(inputs) != require:
        raise RuntimeError('Unexpected temporal inference fields')
    # This wrapper applies goal exclusively in rescore(), after base completion.
    return {**inputs, 'goal_xy': batch['goal_xy']}


def verify_bank(model, output):
    verify_output(output)
    bank = model._trajectory_head.traj_vocab.flatten(0, 1)
    if not torch.equal(output['candidate_xy'], bank[output['candidate_ids'], :6, :2]):
        raise RuntimeError('Candidate coordinates left the immutable bank')
    if not torch.equal(output['trajectory'], bank[output['selected_candidate_id'], :6, :2]):
        raise RuntimeError('Selected trajectory differs from bank row')


@torch.inference_mode()
def verify_initial_contract(model, public, dataset, args):
    model.eval()
    batch = next(iter(DataLoader(dataset, batch_size=min(args.eval_batch, len(dataset)),
                                 shuffle=False, num_workers=0)))
    x = transfer(batch)
    statuses = []
    handle = public._status_encoding.register_forward_pre_hook(
        lambda _m, inputs: statuses.append(inputs[0].detach().clone()))
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            reference = public(images=x['images'], lidar2img=x['lidar2img'], image_hw=x['image_hw'],
                               status=x['images'].new_zeros(len(batch['row']), 8))
            actual = model(**selected_inputs(x, args.common_status))
    finally:
        handle.remove()
    for key in ('scores', 'candidate_ids', 'candidate_xy', 'candidate_valid', 'trajectory', 'selected_candidate_id'):
        if not torch.equal(reference[key], actual[key]):
            raise RuntimeError(f'Initial full-GPU public status0 parity failed: {key}')
    if len(statuses) != 2 or any(torch.count_nonzero(s).item() for s in statuses):
        raise RuntimeError('Original planner received a nonzero status')
    verify_bank(model, actual)
    return {'rows_sha256': rows_sha(batch['row'].numpy()), 'n': len(batch['row']),
            'public_zero_status_output_exact': True, 'planner_status_zero_observed': True,
            'precision': 'bf16_base_fp32_relative_head'}


def masked_bce(logits, targets, mask, pos_weight):
    mask = mask.bool()
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError('Perception auxiliary shape/mask mismatch')
    if not bool(mask.any()):
        return logits.float().sum() * 0.
    loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction='none',
        pos_weight=logits.new_tensor(float(pos_weight), dtype=torch.float32))
    return loss[mask].mean()


def all_losses(model, output, batch, args):
    if 'aux_visible' in output:
        for task in ('occ', 'lane'):
            if bool((batch[f'{task}_valid'].bool() & ~output['aux_visible'].bool()).any()):
                raise RuntimeError('Supervision includes cells invisible to the perception head')
    losses = selection_loss(output, batch['gt_plan'], model._trajectory_head.path_vocab[..., :2],
                            model._trajectory_head.vel_vocab, temperature=args.temperature)
    occ = masked_bce(output['aux_occ'], batch['occ_target'], batch['occ_valid'], args.occ_pos_weight)
    lane = masked_bce(output['aux_lane'], batch['lane_target'], batch['lane_valid'], args.lane_pos_weight)
    scale = output['aux_state'].new_tensor([20., 5., 3., 3.])
    if not bool(batch['state_valid'].all()):
        raise ValueError('Expected the validated all-finite causal-state supervision contract')
    state = F.smooth_l1_loss(output['aux_state'].float() / scale, batch['state_target'].float() / scale)
    planning = losses['loss']
    losses.update(planning_loss=planning, occ_loss=occ, lane_loss=lane, state_loss=state,
                  occ_valid_pixels=batch['occ_valid'].sum(), lane_valid_pixels=batch['lane_valid'].sum(),
                  loss=planning + args.perception_weight * (occ + lane) + args.state_weight * state)
    return losses


@torch.inference_mode()
def evaluate(model, dataset, args, directory, step):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.eval_batch, shuffle=False, num_workers=args.workers,
                        pin_memory=True)
    chunks = {}
    intersections = {'occ': 0, 'lane': 0}
    unions = {'occ': 0, 'lane': 0}
    for batch in loader:
        x = transfer(batch)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(**selected_inputs(x, args.common_status))
        verify_bank(model, out)
        cost = d3(out['candidate_xy'], x['gt_plan'][:, None].expand_as(out['candidate_xy']))
        delta = out['trajectory'].float() - x['gt_plan'].float()
        values = {'rows': batch['row'], 'pred': out['trajectory'].float(),
            'candidate_id': out['selected_candidate_id'], 'd3': d3(out['trajectory'], x['gt_plan']),
            'shortlist_oracle': cost.masked_fill(~out['candidate_valid'], torch.inf).amin(-1),
            'point_l2': torch.linalg.vector_norm(delta, dim=-1), 'error_xy': delta,
            'state_pred': out['aux_state'].float(), 'state_target': x['state_target'],
            'session': np.asarray(batch['session']), 'scenario': np.asarray(batch['scenario'])}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            chunks.setdefault(key, []).append(value)
        for task in ('occ', 'lane'):
            mask = x[f'{task}_valid'].bool()
            pred = out[f'aux_{task}'] > 0
            target = x[f'{task}_target'].bool()
            intersections[task] += int((pred & target & mask).sum())
            unions[task] += int(((pred | target) & mask).sum())
    arrays = {k: np.concatenate(v) for k, v in chunks.items()}
    if not np.array_equal(arrays['rows'], dataset.rows):
        raise RuntimeError('Evaluation row population/order changed')
    result = {'step': step, 'n': len(dataset), 'official_d3': float(arrays['d3'].mean()),
        'shortlist_oracle_d3': float(arrays['shortlist_oracle'].mean()),
        'selection_regret': float((arrays['d3'] - arrays['shortlist_oracle']).mean()),
        'three_second_l2': float(arrays['point_l2'][:, -1].mean()),
        'point_l2': arrays['point_l2'].mean(0).tolist(),
        'state_mae_vx_vy_ax_ay': np.abs(arrays['state_pred'] - arrays['state_target']).mean(0).tolist(),
        'occ_iou': intersections['occ'] / max(unions['occ'], 1),
        'lane_iou': intersections['lane'] / max(unions['lane'], 1),
        'session_d3': {s: float(arrays['d3'][arrays['session'] == s].mean())
                       for s in np.unique(arrays['session'])}}
    np.savez_compressed(directory / f'eval_{step:06d}.npz', **arrays)
    atomic_json(directory / f'eval_{step:06d}.json', result)
    print(json.dumps({'evaluation': result}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--base', required=True)
    p.add_argument('--split-manifest', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--bank', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--history-mode', choices=('repeat', 'real'), required=True)
    p.add_argument('--common-status', action='store_true')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--eval-batch', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone-lr', type=float, default=1e-5)
    p.add_argument('--new-lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--temperature', type=float, default=.1)
    p.add_argument('--perception-weight', type=float, default=.25)
    p.add_argument('--state-weight', type=float, default=.1)
    p.add_argument('--occ-pos-weight', type=float, default=4.)
    p.add_argument('--lane-pos-weight', type=float, default=8.)
    p.add_argument('--eval-limit', type=int, default=0, help='Canary only')
    p.add_argument('--checkpoint-every', type=int, default=500)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '1', '4'):
        raise RuntimeError('Use exactly one allocated GPU')
    if min(a.steps, a.batch, a.eval_batch, a.checkpoint_every) < 1:
        raise ValueError('Invalid step/batch/checkpoint interval')
    directory = Path(a.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        seed_all(a.seed)
        torch.set_num_threads(4)
        common = dict(base=a.base, split_manifest=a.split_manifest,
                      goal_mode='selection', status_mode='zero')
        train = TemporalPlanDataset(PlanDataset(**common, split='train', augment=True, seed=a.seed),
                                    history_mode=a.history_mode, auxiliary=True)
        val = TemporalPlanDataset(PlanDataset(**common, split='tune', augment=False, limit=a.eval_limit),
                                  history_mode=a.history_mode, auxiliary=True)
        train_meta, val_meta = train.provenance(), val.provenance()
        if set(train_meta['sessions']) & set(val_meta['sessions']):
            raise RuntimeError('Training/validation sessions overlap')
        initialization = verify_initialization(a.checkpoint, a.bank, train, val)
        public, coverage = PublicSparseDriveV2.from_public_checkpoint(a.checkpoint,
            bank_path=a.bank, backend='native', score_mode='imitation')
        public_before = {k: v.detach().clone() for k, v in public.state_dict().items()}
        temporal = TemporalPerceptionModel(public)
        model = FinalGoalTemporalSelector(temporal)
        if any(not torch.equal(v, public.state_dict()[k]) for k, v in public_before.items()):
            raise RuntimeError('New model constructor changed public tensors')
        del public_before
        # Device conversion precedes collecting Parameter objects for AdamW.
        model.cuda()
        initial_contract = verify_initial_contract(model, public, val, a)
        public_ids = {id(v) for v in public.parameters()}
        backbone_ids = {id(v) for v in public._backbone.parameters()}
        groups = {'backbone': [], 'public_head': [], 'new': []}
        for parameter in model.parameters():
            if parameter.requires_grad:
                key = 'backbone' if id(parameter) in backbone_ids else 'public_head' if id(parameter) in public_ids else 'new'
                groups[key].append(parameter)
        if any(not group for group in groups.values()):
            raise RuntimeError('Missing optimizer parameter group')
        lrs = [a.backbone_lr, a.lr, a.new_lr]
        optimizer = torch.optim.AdamW([{'params': groups[k], 'lr': lr}
            for k, lr in zip(groups, lrs)], weight_decay=a.weight_decay)
        runtime_files = ['train_temporal_comparison.py', 'temporal_model.py', 'temporal_data.py',
                         'public_model.py', 'relative_selector.py', 'data.py', 'losses.py', 'train.py',
                         'goal_selector.py']
        sources = {}
        for name in runtime_files:
            source = Path(__file__).with_name(name)
            target = directory / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            sources[name] = file_sha(source)
        op_dir = ROOT / 'third_party/SparseDriveV2/navsim/agents/sparsedrive/ops/src'
        for source in op_dir.glob('*'):
            if source.is_file():
                target = directory / 'source' / 'native_ops' / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                sources[f'native_ops/{source.name}'] = file_sha(source)
        manifest = {'schema': 'sdv2_temporal_comparison_v1', 'arguments': vars(a),
            'physical_gpu': int(os.environ['CUDA_VISIBLE_DEVICES']), 'pid': os.getpid(),
            'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_sha256': sources, 'torch': str(torch.__version__), 'numpy': str(np.__version__),
            'public_checkpoint_sha256': file_sha(a.checkpoint), 'bank_sha256': file_sha(a.bank),
            'public_load_coverage': coverage, 'initialization_audit': initialization,
            'public_tensors_unchanged_by_adapter_construction': True,
            'initial_full_gpu_contract': initial_contract,
            'train': train_meta, 'validation': val_meta,
            'parameter_counts': {k: sum(v.numel() for v in values) for k, values in groups.items()},
            'planner_status': 'constant zero, no raw or predicted status',
            'relative_head_status': 'constant zero, no raw or predicted status',
            'goal_route': 'only relative scoring of completed original bank candidates',
            'state_aux_route': 'image-only prediction used solely in training loss/diagnostics',
            'perception_status_route': 'common temporal image attention query only' if a.common_status else 'absent',
            'comparison': 'fixed terminal; repeated tune exploratory; no best-checkpoint selection',
            'class_weights': 'predeclared constants, not fitted using tune',
            'batchnorm': 'fixed running statistics; trainable affine',
            'training_rng_seed': a.seed + 1}
        atomic_json(directory / 'manifest.json', manifest)
        seed_all(a.seed + 1)
        loader = DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                            pin_memory=True, drop_last=False)
        iterator = iter(loader)
        epoch = 0
        initial = evaluate(model, val, a, directory, 0)
        for step in range(1, a.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                train.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            factor = min(step / max(a.warmup, 1), 1.)
            if step > a.warmup:
                factor *= .5 * (1 + math.cos(math.pi * (step - a.warmup) / max(a.steps - a.warmup, 1)))
            for group, lr in zip(optimizer.param_groups, lrs):
                group['lr'] = factor * lr
            fixed_bn_train(model)
            x = transfer(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(**selected_inputs(x, a.common_status))
            verify_bank(model, out)
            with torch.autocast('cuda', enabled=False):
                losses = all_losses(model, out, x, a)
            if not torch.isfinite(losses['loss']):
                raise RuntimeError('Nonfinite training loss')
            losses['loss'].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 10 == 0:
                record = {'step': step, 'epoch': epoch, 'seconds': time.time() - started,
                    'grad_norm': float(grad_norm), 'batch_rows_sha256': rows_sha(batch['row'].numpy()),
                    'learning_rates': [g['lr'] for g in optimizer.param_groups],
                    **{k: float(v.detach()) for k, v in losses.items()}}
                with (directory / 'train.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                print(json.dumps(record), flush=True)
            if step % a.checkpoint_every == 0 or step == a.steps:
                save_checkpoint(directory / 'last.pth', {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'step': step, 'epoch': epoch, 'manifest': manifest})
                atomic_json(directory / 'progress.json', {'status': 'running', 'step': step,
                    'elapsed_seconds': time.time() - started})
        terminal = evaluate(model, val, a, directory, a.steps)
        save_checkpoint(directory / 'last.pth', {'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'step': a.steps, 'epoch': epoch,
            'manifest': manifest, 'result': terminal})
        atomic_json(directory / 'result.json', {'status': 'completed', 'initial': initial,
            'terminal': terminal, 'steps': a.steps, 'elapsed_seconds': time.time() - started,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})
    except BaseException:
        atomic_json(directory / 'failure.json', {'status': 'failed',
            'traceback': traceback.format_exc(), 'elapsed_seconds': time.time() - started})
        raise


if __name__ == '__main__':
    main()
