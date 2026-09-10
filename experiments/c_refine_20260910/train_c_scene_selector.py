"""Matched frozen-C real-token/zero-token final selector experiment.

The inference head receives only completed bank candidates, cached model scores,
candidate scene tokens, and the provided final-selection goal. GT stays in the
trainer/evaluator. Cache extraction and head fitting use the unchanged split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from c_scene_selector import SceneResidualSelector


WEIGHTS = [11/36, 11/36, 5/36, 5/36, 2/36, 2/36]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def tensor_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class Cache:
    """Explicit inference inputs are constructed separately from GT/cost arrays."""
    def __init__(self, directory, bank, device, preload=True):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        if self.manifest.get('status') != 'completed':
            raise ValueError('Only a completed immutable cache may be consumed')
        self.arrays = {}
        for key in ('rows', 'candidate_ids', 'candidate_valid', 'scores', 'token',
                    'goal_xy', 'd3', 'gt', 'scene_index', 'session_index'):
            path = self.directory / (key + '.npy')
            recorded = self.manifest['files'][key]
            if recorded['path'] != path.name or sha(path) != recorded['sha256']:
                raise ValueError('Cache array checksum/path mismatch: ' + key)
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if list(array.shape) != recorded['shape'] or str(array.dtype) != recorded['dtype']:
                raise ValueError('Cache array shape/dtype mismatch: ' + key)
            self.arrays[key] = (torch.from_numpy(np.array(array)).to(device) if preload
                                else array)
            print(json.dumps({'cache_loaded': str(self.directory), 'field': key,
                              'shape': list(array.shape), 'dtype': str(array.dtype)}), flush=True)
        self.bank = bank
        self.device = device
        self.preload = preload
        self.n, self.k = self.arrays['candidate_ids'].shape
        if len(self.arrays['rows']) != self.n:
            raise ValueError('Cache population mismatch')
        if self.arrays['token'].shape != (self.n, self.k, 256):
            raise ValueError('Unexpected scene token shape')

    def field(self, key, index):
        a = self.arrays[key]
        if self.preload:
            return a[index]
        if torch.is_tensor(index):
            index = index.cpu().numpy()
        return torch.from_numpy(np.array(a[index])).to(self.device)

    def inputs(self, index):
        ids = self.field('candidate_ids', index).long()
        return dict(candidate_ids=ids, candidate_xy=self.bank[ids],
                    candidate_valid=self.field('candidate_valid', index).bool(),
                    scores=self.field('scores', index).float(),
                    candidate_tokens=self.field('token', index)), self.field('goal_xy', index)

    def rows_cpu(self):
        a = self.arrays['rows']
        return a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def cost_loss(scores, costs, valid, temperature, expected_weight=0.):
    if scores.shape != costs.shape or valid.shape != scores.shape:
        raise ValueError('Cost/score/mask mismatch')
    if temperature <= 0 or not bool(valid.any(-1).all()):
        raise ValueError('Invalid target temperature or empty candidate set')
    if not bool(torch.isfinite(scores[valid]).all() & torch.isfinite(costs[valid]).all()):
        raise ValueError('Nonfinite valid score/cost')
    logits = scores.float().masked_fill(~valid, -1e4)
    target = torch.softmax((-costs.float() / temperature).masked_fill(~valid, -1e4), -1).detach()
    ce = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
    expected = (torch.softmax(logits, -1) * costs.float().masked_fill(~valid, 0.)).sum(-1).mean()
    return ce + expected_weight * expected, ce, expected


@torch.inference_mode()
def evaluate(head, cache, directory, step, batch=64, limit=0, prefix='eval'):
    head.eval()
    chunks = {k: [] for k in ('d3', 'shortlist_oracle', 'base_d3', 'candidate_id', 'pred', 'point_l2')}
    n = min(cache.n, limit) if limit else cache.n
    for start in range(0, n, batch):
        index = torch.arange(start, min(start + batch, n), device=cache.device)
        inp, goal = cache.inputs(index)
        out = head(inp, goal_xy=goal)
        costs = cache.field('d3', index)
        valid = inp['candidate_valid']
        selected = out['scores'].masked_fill(~valid, -torch.inf).argmax(-1)
        base_selected = inp['scores'].masked_fill(~valid, -torch.inf).argmax(-1)
        bi = torch.arange(len(index), device=cache.device)
        if step == 0 and (not torch.equal(out['scores'], inp['scores'])
                          or not torch.equal(out['selected_candidate_id'], inp['candidate_ids'][bi, base_selected])
                          or not torch.equal(out['trajectory'], inp['candidate_xy'][bi, base_selected])):
            raise RuntimeError('Zero initialized head changed scores, selected IDs, or trajectory')
        pred = out['trajectory'].float()
        if not torch.equal(pred, cache.bank[out['selected_candidate_id']]):
            raise RuntimeError('Final selection changed a bank trajectory')
        gt = cache.field('gt', index)
        point = torch.linalg.vector_norm(pred - gt, dim=-1)
        direct = (point * point.new_tensor(WEIGHTS)).sum(-1)
        metric = costs[bi, selected]
        if not torch.allclose(direct, metric, atol=2e-6, rtol=2e-6):
            raise RuntimeError('Cached D3 disagrees with independent selected trajectory metric')
        vals = dict(d3=metric, shortlist_oracle=costs.masked_fill(~valid, torch.inf).amin(-1),
                    base_d3=costs[bi, base_selected], candidate_id=out['selected_candidate_id'],
                    pred=pred, point_l2=point)
        for k, value in vals.items():
            chunks[k].append(value.cpu().numpy())
    arrays = {k: np.concatenate(v) for k, v in chunks.items()}
    arrays['rows'] = cache.rows_cpu()[:n]
    s = cache.arrays['session_index']
    arrays['session_index'] = (s.cpu().numpy() if torch.is_tensor(s) else np.asarray(s))[:n]
    result = dict(step=step, n=n, official_d3=float(arrays['d3'].mean()),
                  shortlist_oracle_d3=float(arrays['shortlist_oracle'].mean()),
                  selection_regret=float((arrays['d3'] - arrays['shortlist_oracle']).mean()),
                  base_d3=float(arrays['base_d3'].mean()),
                  point_l2=arrays['point_l2'].mean(0).tolist(),
                  session_d3={str(int(s)): float(arrays['d3'][arrays['session_index'] == s].mean())
                              for s in np.unique(arrays['session_index'])})
    np.savez_compressed(directory / f'{prefix}_{step:06d}.npz', **arrays)
    atomic_json(directory / f'{prefix}_{step:06d}.json', result)
    print(json.dumps({prefix: result}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--train-cache', required=True)
    p.add_argument('--tune-cache', required=True)
    p.add_argument('--bank', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--mode', choices=('real', 'zero'), required=True)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--eval-batch', type=int, default=64)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--temperature', type=float, default=.1)
    p.add_argument('--expected-weight', type=float, default=0.)
    p.add_argument('--stream-cpu', action='store_true')
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '1', '4'):
        raise RuntimeError('Only an allocated individual GPU is allowed')
    if min(a.steps, a.batch, a.eval_batch, a.eval_every) < 1:
        raise ValueError('Invalid steps/batch')
    directory = Path(a.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        torch.set_num_threads(4)
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        with np.load(a.bank, allow_pickle=False) as f:
            # Public bank files use traj_vocab (support descriptive legacy key).
            key = next((k for k in ('traj_xyz8', 'traj_xy8', 'traj_vocab') if k in f), None)
            if key is None:
                raise ValueError(f'Bank trajectory key missing: {f.files}')
            arr = f[key]
            bank = torch.from_numpy(np.array(arr[..., :6, :2].reshape(-1, 6, 2))).float().cuda()
        train = Cache(a.train_cache, bank, torch.device('cuda'), not a.stream_cpu)
        tune = Cache(a.tune_cache, bank, torch.device('cuda'), not a.stream_cpu)
        if train.k != tune.k or set(train.rows_cpu()) & set(tune.rows_cpu()):
            raise RuntimeError('Cache candidate layout differs or train/tune rows overlap')
        if (train.n, tune.n) != (54810, 1998):
            raise RuntimeError('Only complete original TRAIN203/TUNE37 caches are allowed')
        for key in ('checkpoint_sha256', 'bank_sha256', 'counts', 'batch_size', 'precision'):
            if train.manifest[key] != tune.manifest[key]:
                raise RuntimeError('Train/tune extraction contract differs: ' + key)
        if set(train.manifest['sessions']) & set(tune.manifest['sessions']):
            raise RuntimeError('Train/tune sessions overlap')
        bank_sha = sha(a.bank)
        for cache, expected_rows in ((train, '75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854'),
                                     (tune, '1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88')):
            if cache.manifest['bank_sha256'] != bank_sha:
                raise RuntimeError('Cache and inference bank have different provenance')
            if hashlib.sha256(cache.rows_cpu().astype(np.int64).tobytes()).hexdigest() != expected_rows:
                raise RuntimeError('Cache row population/order differs from original split')
        head = SceneResidualSelector(mode=a.mode).cuda()
        optimizer = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        sources = {}
        for name in ('train_c_scene_selector.py', 'c_scene_selector.py'):
            source = Path(__file__).with_name(name)
            target = directory / 'source' / name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
            sources[name] = sha(source)
        manifest = dict(schema='c_scene_selector_frozen_v1', arguments=vars(a),
                        physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']), pid=os.getpid(),
                        source_sha256=sources, bank_sha256=sha(a.bank),
                        original_c_checkpoint=train.manifest['checkpoint']['checkpoint'],
                        original_c_checkpoint_sha256=train.manifest['checkpoint_sha256'],
                        path_filter=train.manifest['counts']['path_filter'],
                        velocity_filter=train.manifest['counts']['velocity_filter'],
                        train_cache_manifest=train.manifest, tune_cache_manifest=tune.manifest,
                        train_cache_manifest_sha256=sha(Path(a.train_cache) / 'manifest.json'),
                        tune_cache_manifest_sha256=sha(Path(a.tune_cache) / 'manifest.json'),
                        initial_head_sha256=tensor_hash(head.state_dict()),
                        parameters=sum(x.numel() for x in head.parameters()),
                        torch=str(torch.__version__), numpy=str(np.__version__),
                        base_frozen=True, raw_status_input=False, goal='final completed-candidate selection only',
                        augmentation='fixed no-augmentation cached C features',
                        validation='repeated exploratory TUNE; fixed terminal and predeclared intermediate checkpoints',
                        rng_seed=a.seed + 1)
        atomic_json(directory / 'manifest.json', manifest)
        initial = evaluate(head, tune, directory, 0, a.eval_batch)
        if abs(initial['official_d3'] - initial['base_d3']) > 1e-7:
            raise RuntimeError('Zero initialized selector did not preserve C score decisions')
        rng = torch.Generator(device='cpu').manual_seed(a.seed + 1)
        permutation = torch.randperm(train.n, generator=rng)
        cursor, epoch, seen = 0, 0, 0
        for step in range(1, a.steps + 1):
            if cursor >= train.n:
                permutation = torch.randperm(train.n, generator=rng)
                cursor, epoch = 0, epoch + 1
            index_cpu = permutation[cursor:cursor + a.batch]
            cursor += len(index_cpu)
            seen += len(index_cpu)
            index = index_cpu.cuda()
            factor = min(step / max(a.warmup, 1), 1.)
            if step > a.warmup:
                factor *= .5 * (1 + math.cos(math.pi * (step - a.warmup) / max(a.steps - a.warmup, 1)))
            for group in optimizer.param_groups:
                group['lr'] = a.lr * factor
            head.train()
            inp, goal = train.inputs(index)
            optimizer.zero_grad(set_to_none=True)
            out = head(inp, goal_xy=goal)
            # Labels first enter only after inference output has been produced.
            costs = train.field('d3', index)
            loss, ce, expected = cost_loss(out['scores'], costs, inp['candidate_valid'], a.temperature, a.expected_weight)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 50 == 0:
                selected = out['scores'].masked_fill(~inp['candidate_valid'], -torch.inf).argmax(-1)
                value = costs[torch.arange(len(index), device='cuda'), selected].mean()
                rec = dict(step=step, epoch=epoch, seen=seen, exposure=seen/train.n,
                           loss=float(loss), ce=float(ce), expected_d3=float(expected), selected_d3=float(value),
                           grad_norm=float(norm), lr=a.lr*factor, seconds=time.time()-started,
                           batch_rows_sha256=hashlib.sha256(train.field('rows', index).cpu().numpy().tobytes()).hexdigest())
                with (directory / 'train.jsonl').open('a') as f:
                    f.write(json.dumps(rec, allow_nan=False) + '\n')
                print(json.dumps(rec), flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                terminal = evaluate(head, tune, directory, step, a.eval_batch)
                torch.save(dict(head=head.state_dict(), optimizer=optimizer.state_dict(), step=step,
                                epoch=epoch, seen=seen, manifest=manifest, result=terminal),
                           directory / f'step_{step:06d}.pth')
                atomic_json(directory / 'progress.json', dict(step=step, result=terminal, seconds=time.time()-started))
        train_audit = evaluate(head, train, directory, a.steps, a.eval_batch, prefix='train_eval')
        atomic_json(directory / 'result.json', dict(status='completed', initial=initial, terminal=terminal,
                    train=train_audit, steps=a.steps, seen=seen, exposure=seen/train.n,
                    elapsed_seconds=time.time()-started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
    except BaseException:
        atomic_json(directory / 'failure.json', dict(status='failed', traceback=traceback.format_exc(),
                                                   elapsed_seconds=time.time()-started))
        raise


if __name__ == '__main__':
    main()
