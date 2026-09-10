"""Immutable train/tune candidate-token caches from strictly loaded terminal C.

No live training source is edited. GT is materialized by the outer dataset but
never included in the forward input whitelist; D3 is computed after forward.
Each split is independent .npy memmaps plus a completed, hashed manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

EVALUATOR_SHA = '9007a4517adaf663d402535bdd56dd5d8340ad675db0f3a3ef44d8d867529242'
C_CHECKPOINT_SHA = 'b9dcc56af7c2c4c3cfc80d844fe862a2d8f730d7b8d7a813ee997d33abfec2ff'
TOKEN_SOURCE_SHA = 'be8d9eef8e1e9f087852c1a81d0f5f4437e618a146d02d5037de60181b360ae5'
WEIGHTS = np.asarray([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.
SCHEMA = 'frozen_c_candidate_token_cache_v1'
D3_ATOL, D3_RTOL = 2e-6, 2e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def row_sha(rows):
    return hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_population(plan, runtime, ev, split, limit=0):
    """Construct the actual requested split; never mutate a tune dataset's rows."""
    require(split in ('train', 'tune'), 'Only original primary train/tune are supported')
    recorded = plan.manifest['train' if split == 'train' else 'validation']
    expected_sha = ev.TRAIN_SHA if split == 'train' else ev.TUNE_SHA
    expected_count = 54810 if split == 'train' else 1998
    base = runtime.data.PlanDataset(base=plan.base, split_manifest=plan.split,
        ego_cache=plan.ego, split=split, stride=1 if split == 'train' else 5,
        augment=False, seed=0, status_mode='zero', goal_mode='selection')
    require(len(base.rows) == expected_count and row_sha(base.rows) == expected_sha,
            'Original population/order changed')
    require(np.all(np.diff(base.rows) > 0), 'Population order must be strictly ascending')
    actual = base.provenance()
    for key in ('split', 'rows', 'rows_sha256', 'allowed_rows', 'allowed_rows_sha256',
                'split_sha256', 'ego_cache_sha256', 'calibration_sha256', 'camera_order',
                'image_wh', 'normalization', 'status_mode', 'goal_mode', 'metric_weights',
                'scenes', 'sessions'):
        require(actual[key] == recorded[key], 'Dataset contract changed: ' + key)
    source = recorded['temporal_data']
    dataset = runtime.temporal_data.TemporalPlanDataset(base, history_mode='real', auxiliary=False,
        causal_status_root=Path(source['causal_status']['path']).parent)
    current = dataset.provenance()['temporal_data']
    for key in ('source_sha256', 'base_source_sha256', 'history_mode', 'frame_lags',
                'time_offsets', 'pose_alignment_used', 'causal_status'):
        require(current[key] == source[key], 'Temporal contract changed: ' + key)
    require(not base.augment and base.epoch == 0 and np.count_nonzero(base.status) == 0,
            'Cache must be nonaugmented, epoch0 and planner status0')
    names = base.scenarios[base.scen_idx[base.rows]].astype(str)
    sessions = np.asarray([base.manifest['scene_to_session'][s] for s in names])
    identities = dict(rows=np.asarray(base.rows, np.int64), frame=base.frames[base.rows].astype(np.int64),
                      scene=names, session=sessions)
    provenance = dataset.provenance()
    provenance.update(cache_augment=False, cache_epoch=0, cache_dataset_seed=0,
                      auxiliary_targets_loaded=False, full_population_rows_sha256=expected_sha)
    if limit:
        require(0 < limit <= expected_count, 'Invalid canary row limit')
        outer = torch.utils.data.Subset(dataset, list(range(limit)))
        outer.rows = base.rows[:limit].copy()
        dataset = outer
        identities = {k: v[:limit] for k, v in identities.items()}
    if _DONOR["path"] is not None:
        if limit:
            raise ValueError("Donor substitution runs on the full population only")
        dataset = _wrap_donor_images(dataset, identities["rows"])
        provenance.update(image_substitution={
            "mode": "fixed different-scene donor images",
            "replaced_keys": ["images", "history_images"],
            "unchanged": ["lidar2img", "image_hw", "time_offsets", "causal status",
                          "goal", "row identity", "GT"],
            "mapping_path": str(Path(_DONOR["path"]).resolve()),
            "mapping_sha256": _DONOR["sha256"]})
    return dataset, provenance, identities


_DONOR = {"path": None, "sha256": None}


class DonorImageDataset(torch.utils.data.Dataset):
    """Recipient row with another scene's images; everything else untouched.

    Only ``images`` and ``history_images`` are taken from the donor. The
    recipient keeps its own calibration (``lidar2img``), ``image_hw``,
    ``time_offsets``, causal status, goal, row identity and GT, matching the
    primary model's different-scene substitution control.
    """

    def __init__(self, base, donor_index, rows):
        self.base = base
        self.donor_index = np.asarray(donor_index, dtype=np.int64)
        self.rows = rows
        if len(self.donor_index) != len(base):
            raise ValueError("Donor index must cover every dataset position")

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        if name in ("base", "donor_index", "rows"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def __getitem__(self, index):
        item = dict(self.base[index])
        donor = self.base[int(self.donor_index[index])]
        item["images"] = donor["images"]
        item["history_images"] = donor["history_images"]
        return item


def _wrap_donor_images(dataset, rows):
    mapping = json.loads(Path(_DONOR["path"]).read_text())
    rows = np.asarray(rows, dtype=np.int64)
    position = {int(row): i for i, row in enumerate(rows)}
    if len(mapping) != len(rows):
        raise ValueError("Donor mapping must cover exactly the evaluated rows")
    if {int(r["recipient_row"]) for r in mapping} != set(position):
        raise ValueError("Donor mapping recipient rows differ from the dataset")
    donor_index = np.empty(len(rows), dtype=np.int64)
    for record in mapping:
        recipient, image_row = int(record["recipient_row"]), int(record["image_row"])
        if image_row not in position:
            raise ValueError("Donor image row is outside the evaluated population")
        if record["recipient_scene"] == record["image_scene"]:
            raise ValueError("Donor must come from a different scene")
        donor_index[position[recipient]] = position[image_row]
    if bool((donor_index == np.arange(len(rows))).any()):
        raise ValueError("A row would donate its own images")
    return DonorImageDataset(dataset, donor_index, rows)


def exact_token_dtype(token):
    """Use FP16 only if casting preserves every float32 byte (including -0)."""
    value = np.asarray(token, dtype=np.float32)
    require(np.isfinite(value).all(), 'Nonfinite candidate token')
    if np.any(np.abs(value) > np.finfo(np.float16).max):
        return np.dtype('float32')
    restored = value.astype(np.float16).astype(np.float32)
    exact = np.array_equal(value.view(np.uint8), restored.view(np.uint8))
    return np.dtype('float16' if exact else 'float32')


def label_costs(candidate_xy, gt):
    """CPU labels-only stage, called strictly after token/score forward extraction."""
    xy, target = np.asarray(candidate_xy, np.float32), np.asarray(gt, np.float32)
    require(xy.ndim == 4 and xy.shape[-2:] == (6, 2), 'Expected [B,K,6,2] candidates')
    require(target.shape == (len(xy), 6, 2), 'Expected [B,6,2] outer target')
    require(np.isfinite(xy).all() and np.isfinite(target).all(), 'Nonfinite trajectory label data')
    delta = xy - target[:, None]
    cost = (np.sqrt(np.sum(delta * delta, axis=-1)) * WEIGHTS.astype(np.float32)).sum(-1)
    # Independent higher-precision arithmetic bounds storage/metric rounding.
    high = np.linalg.norm(xy.astype(np.float64) - target[:, None].astype(np.float64), axis=-1) @ WEIGHTS
    require(np.allclose(cost, high, atol=D3_ATOL, rtol=D3_RTOL), 'Float32 D3 exceeded declared tolerance')
    return cost.astype(np.float32), float(np.max(np.abs(cost - high)))


class CacheWriter:
    """Append in exact row order; incomplete files are never a completed cache."""
    def __init__(self, directory, identities, k, token_dtype, common_manifest):
        self.root = Path(directory)
        self.root.mkdir(parents=False, exist_ok=False)
        self.ids = identities
        self.rows = np.asarray(identities['rows'], np.int64)
        require(self.rows.ndim == 1 and len(self.rows) > 0 and np.all(np.diff(self.rows) > 0),
                'Cache rows must be unique ascending nonempty integers')
        self.n, self.k, self.written = len(self.rows), int(k), 0
        self.meta = dict(common_manifest)
        self.token_dtype = np.dtype(token_dtype)
        require(self.token_dtype in (np.dtype('float16'), np.dtype('float32')), 'Unsupported token dtype')
        self.scenes = sorted(set(identities['scene'].tolist()))
        self.sessions = sorted(set(identities['session'].tolist()))
        self.scene_map = {v: i for i, v in enumerate(self.scenes)}
        self.session_map = {v: i for i, v in enumerate(self.sessions)}
        shapes = {'rows': ((self.n,), 'int64'), 'frame': ((self.n,), 'int64'),
                  'scene_index': ((self.n,), 'int32'), 'session_index': ((self.n,), 'int32'),
                  'candidate_ids': ((self.n, self.k), 'int64'),
                  'candidate_valid': ((self.n, self.k), 'bool'), 'scores': ((self.n, self.k), 'float32'),
                  'token': ((self.n, self.k, 256), self.token_dtype),
                  'goal_xy': ((self.n, 2), 'float32'), 'd3': ((self.n, self.k), 'float32'),
                  'gt': ((self.n, 6, 2), 'float32')}
        self.arrays = {key: np.lib.format.open_memmap(self.root / (key + '.npy'), mode='w+',
                       dtype=dtype, shape=shape) for key, (shape, dtype) in shapes.items()}
        self.promoted = False
        self.progress()

    def progress(self):
        for value in self.arrays.values():
            value.flush()
        atomic_json(self.root / 'progress.json', {'status': 'incomplete', 'rows_written': self.written,
            'expected_rows': self.n, 'rows_sha256': row_sha(self.rows), 'token_dtype': str(self.token_dtype)})

    def promote_token(self):
        """Lossless upgrade if a later batch is not FP16-representable."""
        require(self.token_dtype == np.dtype('float16'), 'Only FP16 can be promoted')
        old = self.arrays['token']
        upgraded = np.lib.format.open_memmap(self.root / 'token.float32.tmp.npy', mode='w+',
                        dtype='float32', shape=old.shape)
        for start in range(0, self.written, 128):
            upgraded[start:min(start+128, self.written)] = old[start:min(start+128, self.written)]
        upgraded.flush()
        old.flush()
        del self.arrays['token']
        del old, upgraded
        (self.root / 'token.float32.tmp.npy').replace(self.root / 'token.npy')
        self.arrays['token'] = np.load(self.root / 'token.npy', mmap_mode='r+', allow_pickle=False)
        self.token_dtype, self.promoted = np.dtype('float32'), True

    def append(self, batch):
        expected_fields = {'rows', 'candidate_ids', 'candidate_valid', 'scores', 'token', 'goal_xy', 'd3', 'gt'}
        require(set(batch) == expected_fields, 'Unexpected cache fields; model data and labels must stay explicit')
        count = len(batch['rows'])
        stop = self.written + count
        require(count > 0 and stop <= self.n and np.array_equal(batch['rows'], self.rows[self.written:stop]),
                'Cache row identity/order mismatch')
        for key, value in batch.items():
            require(np.asarray(value).shape == (count, *self.arrays[key].shape[1:]), 'Cache shape mismatch: ' + key)
            if np.asarray(value).dtype.kind == 'f':
                require(np.isfinite(value).all(), 'Nonfinite cache values: ' + key)
        require(np.asarray(batch['candidate_ids']).dtype.kind in 'iu', 'Candidate IDs must be integers')
        require(np.asarray(batch['candidate_valid']).dtype == np.bool_ and batch['candidate_valid'].any(-1).all(),
                'Invalid candidate masks')
        needed = exact_token_dtype(batch['token'])
        if self.token_dtype == np.dtype('float16') and needed == np.dtype('float32'):
            self.promote_token()
        for key, value in batch.items():
            self.arrays[key][self.written:stop] = value
        self.arrays['frame'][self.written:stop] = self.ids['frame'][self.written:stop]
        self.arrays['scene_index'][self.written:stop] = [self.scene_map[x] for x in self.ids['scene'][self.written:stop]]
        self.arrays['session_index'][self.written:stop] = [self.session_map[x] for x in self.ids['session'][self.written:stop]]
        self.written = stop

    def finish(self, extra):
        require(self.written == self.n, 'Cannot publish an incomplete cache')
        self.progress()
        files = {key: {'path': key + '.npy', 'shape': list(value.shape), 'dtype': str(value.dtype),
                       'sha256': sha(self.root / (key + '.npy'))} for key, value in self.arrays.items()}
        result = {**self.meta, **extra, 'schema': SCHEMA, 'status': 'completed', 'rows': self.n,
                  'rows_sha256': row_sha(self.rows), 'candidate_count': self.k, 'files': files,
                  'scenes': self.scenes, 'sessions': self.sessions,
                  'scene_names_sha256': hashlib.sha256(json.dumps(self.scenes).encode()).hexdigest(),
                  'session_names_sha256': hashlib.sha256(json.dumps(self.sessions).encode()).hexdigest(),
                  'token_storage_dtype': str(self.token_dtype), 'token_float32_promotion': self.promoted,
                  'token_storage_roundtrip_exact': True,
                  'candidate_xy_storage': 'reconstruct bank traj_vocab.flatten(0,1)[candidate_ids,:6,:2]',
                  'file_order': 'global original row ID ascending; candidate order exactly frozen model output'}
        atomic_json(self.root / 'manifest.json', result)
        atomic_json(self.root / 'progress.json', {'status': 'completed', 'rows_written': self.n,
                    'manifest_sha256': sha(self.root / 'manifest.json')})
        return result


def configure_counts(model, paths, velocities):
    public = model.base.base
    original = {'path_filter': list(public.path_filter), 'velocity_filter': list(public.velocity_filter)}
    require(original == {'path_filter': [128, 20], 'velocity_filter': [64, 10]}, 'Unexpected loaded base counts')
    paths, velocities = tuple(paths), tuple(velocities)
    h = public._trajectory_head
    require(len(paths) == len(velocities) == 2 and 0 < paths[1] <= paths[0] <= len(h.path_vocab)
            and 0 < velocities[1] <= velocities[0] <= len(h.vel_vocab), 'Invalid path/velocity counts')
    before = {key: (value.data_ptr(), value._version, tuple(value.shape), value.dtype)
              for key, value in model.state_dict().items()}
    # Only selection cardinalities change; no tensor or scoring weight mutation.
    public.path_filter, public.velocity_filter = paths, velocities
    after = {key: (value.data_ptr(), value._version, tuple(value.shape), value.dtype)
             for key, value in model.state_dict().items()}
    require(before == after, 'Counts changed a model tensor/storage/version')
    return {'original': original, 'path_filter': list(paths), 'velocity_filter': list(velocities),
            'candidate_count': paths[-1]*velocities[-1], 'learned_weights_changed': False,
            'fixed_bank_changed': False, 'candidate_attention_context_changes_if_counts_changed': True}


def feature_arrays(output, inputs, rows):
    """This function does not accept GT or a dataset batch."""
    required = ('candidate_ids', 'candidate_valid', 'scores', 'candidate_tokens', 'candidate_xy')
    require(all(key in output for key in required), 'Token wrapper output incomplete')
    token = output['candidate_tokens']
    require(token.ndim == 3 and token.shape[-1] == 256, 'Token capture shape mismatch')
    arr = lambda value: value.detach().float().cpu().numpy()
    features = {'rows': np.asarray(rows, np.int64),
        'candidate_ids': output['candidate_ids'].detach().long().cpu().numpy(),
        'candidate_valid': output['candidate_valid'].detach().bool().cpu().numpy(),
        'scores': arr(output['scores']), 'token': arr(token), 'goal_xy': arr(inputs['goal_xy'])}
    return features, arr(output['candidate_xy']), str(token.dtype)


@torch.inference_mode()
def cache_split(a, model, capture, dataset, provenance, identities, runtime, ev, root, common):
    loader = torch.utils.data.DataLoader(dataset, batch_size=a.batch, shuffle=False, drop_last=False,
        num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    writer, d3_max, token_dtypes, started = None, 0., set(), time.monotonic()
    chosen_sum = oracle_sum = 0.
    with ev.route_monitor(model, True) as counts:
        for index, batch in enumerate(loader):
            inputs = ev.input_tensors(batch, runtime, True, 'cuda:0')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = capture(**inputs)
            ev.verify_rows(model, output)
            features, xy, token_dtype = feature_arrays(output, inputs, batch['row'].numpy())
            # GT is first accessed here, strictly after the frozen forward.
            target = batch['gt_plan'].numpy().astype(np.float32)
            costs, gap = label_costs(xy, target)
            d3_max = max(d3_max, gap)
            token_dtypes.add(token_dtype)
            if writer is None:
                writer = CacheWriter(root, identities, xy.shape[1], exact_token_dtype(features['token']),
                    {**common, 'split': provenance['split'], 'dataset_provenance': provenance})
            writer.append({**features, 'd3': costs, 'gt': target})
            valid = features['candidate_valid']
            selected = np.where(valid, features['scores'], -np.inf).argmax(-1)
            chosen_sum += float(costs[np.arange(len(costs)), selected].astype(np.float64).sum())
            oracle_sum += float(np.where(valid, costs, np.inf).min(-1).astype(np.float64).sum())
            if index == 0 or (index + 1) % 200 == 0:
                writer.progress()
                print(json.dumps({'split': provenance['split'], 'batch': index + 1,
                    'rows_written': writer.written, 'rows_total': writer.n,
                    'elapsed_seconds': time.monotonic()-started, 'token_dtype': str(writer.token_dtype),
                    'candidate_count': writer.k, 'mean_current_selection_d3': chosen_sum/writer.written,
                    'mean_shortlist_oracle': oracle_sum/writer.written}), flush=True)
            del output, inputs, features, xy
    require(writer is not None, 'Empty cache loader')
    require(all(value == len(loader) for value in counts.values()), 'Inference route monitor counts mismatch')
    return writer.finish({'elapsed_seconds': time.monotonic()-started,
        'observed_route_counts': counts, 'token_compute_dtypes': sorted(token_dtypes),
        'd3_float32_max_absolute_difference_from_float64': d3_max,
        'mean_old_final_selection_d3': chosen_sum/writer.n, 'mean_shortlist_oracle': oracle_sum/writer.n})


def snapshot_sources(output, a):
    source = output / 'cache_source'
    source.mkdir()
    files = {'cache_c_candidates.py': Path(__file__), 'evaluate_temporal_checkpoint.py': Path(a.evaluator_source),
             'c_scene_selector.py': Path(a.token_source)}
    for name, path in files.items():
        shutil.copy2(path, source / name)
    return {name: sha(path) for name, path in files.items()}


def arguments():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--worktree', required=True)
    p.add_argument('--base', default='/NHNHOME/data/sukim/adcl')
    p.add_argument('--evaluator-source', default=str(Path(__file__).with_name('evaluate_temporal_checkpoint.py')))
    p.add_argument('--token-source', default=str(Path(__file__).with_name('c_scene_selector.py')))
    p.add_argument('--gpu', type=int, choices=(0, 1, 4))
    p.add_argument('--image-donor-mapping', default=None,
                   help='different-scene image substitution control')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--split', choices=('train', 'tune', 'both'), default='both')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--path-filter', type=int, nargs=2, default=[128, 20])
    p.add_argument('--velocity-filter', type=int, nargs=2, default=[64, 10])
    return p.parse_args()


def _install_donor(path):
    if path is None:
        return
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    _DONOR.update(path=path, sha256=digest.hexdigest())


def main():
    a = arguments()
    _install_donor(a.image_donor_mapping)
    output = Path(a.output).resolve()
    require(not output.exists(), 'Output is immutable; use a new directory, never resume silently')
    require(a.batch == 8 and a.workers >= 0 and a.limit >= 0, 'Frozen cache requires batch8 and valid limits')
    require(sha(a.evaluator_source) == EVALUATOR_SHA, 'Strict evaluator source changed')
    require(sha(a.token_source) == TOKEN_SOURCE_SHA, 'Frozen token wrapper source changed')
    require(a.audit_only or a.gpu in (0, 1, 4), 'Cache task runs on an allocated GPU')
    require(not a.audit_only or os.environ.get('CUDA_VISIBLE_DEVICES', '') == '', 'CPU audit must hide GPUs')
    torch.set_num_threads(4)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(Path(a.worktree).resolve()))
    ev = module_from(a.evaluator_source, '_c_cache_strict_evaluator')
    gpu = None if a.audit_only else ev.check_gpu(a.gpu)
    plan = ev.inspect_checkpoint(a.checkpoint, worktree=a.worktree, base=a.base)
    require(plan.receipt['checkpoint_sha256'] == C_CHECKPOINT_SHA, 'Frozen C checkpoint identity mismatch')
    require(plan.manifest['arguments']['common_status'] is True
            and plan.manifest['arguments']['history_mode'] == 'real', 'Only frozen terminal C is supported')
    output.mkdir(parents=True)
    started = time.monotonic()
    try:
        sources = snapshot_sources(output, a)
        token_module = module_from(output / 'cache_source/c_scene_selector.py', '_c_cache_token_module')
        with ev.isolated_runtime(plan.source) as runtime:
            model = ev.strict_load(plan, runtime).eval()
            counts = configure_counts(model, a.path_filter, a.velocity_filter)
            common = {'schema': SCHEMA, 'checkpoint': plan.receipt,
                'checkpoint_sha256': plan.receipt['checkpoint_sha256'], 'bank_path': str(plan.bank),
                'bank_sha256': plan.receipt['bank_sha256'], 'split_sha256': ev.SPLIT_SHA,
                'counts': counts, 'source_sha256': sources, 'batch_size': a.batch,
                'precision': 'bf16_base_fp32_old_relative_head', 'seed': 0,
                'order': 'ascending global row; shuffle=False; drop_last=False; no augmentation',
                'rng': {'python': 0, 'numpy': 0, 'torch': 0, 'cudnn_benchmark': False,
                        'matmul_tf32': False, 'deterministic_algorithms_required': False},
                'tokens': 'last decoder traj_mlp input, after fine DFA and trajectory attention, before scalar imitation',
                'scores': 'old complete C final goal-relative scores, before new scene head',
                'd3_weights': WEIGHTS.tolist(), 'd3_float64_check_tolerance': {'atol': D3_ATOL, 'rtol': D3_RTOL},
                'GT_passed_to_model': False, 'GT_access_in_cache_loop': 'postforward labels-only label_costs',
                'outer_dataset_materializes_labels': True, 'planner_and_relative_status': 'constant zero',
                'common_status_route': 'past-only causal4 to common perception only',
                'goal_route': 'original final selector only; separate goal_xy for offline final scoring',
                'gpu': gpu, 'canary': bool(a.limit), 'python': sys.version, 'torch': torch.__version__,
                'numpy': np.__version__, 'pid': os.getpid()}
            atomic_json(output / 'manifest.json', {**common, 'status': 'incomplete'})
            names = ('train', 'tune') if a.split == 'both' else (a.split,)
            if not a.audit_only:
                model = model.to('cuda:0')
                capture = token_module.CandidateTokenCapture(model, detach_tokens=True, freeze_base=True).eval()
            results = {}
            for split in names:
                dataset, provenance, identities = build_population(plan, runtime, ev, split, a.limit)
                if a.audit_only:
                    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=min(2, len(dataset)), num_workers=0)))
                    inputs = ev.input_tensors(batch, runtime, True, 'cpu')
                    require('gt_plan' not in inputs and 'status' not in inputs and 'perception_status' in inputs,
                            'CPU input whitelist violated')
                    results[split] = {'full_population': provenance, 'rows': len(identities['rows']),
                        'rows_sha256': row_sha(identities['rows']), 'sample_rows': batch['row'].tolist(),
                        'input_shapes': {k: list(v.shape) for k, v in inputs.items()}, 'forward_performed': False}
                else:
                    result = cache_split(a, model, capture, dataset, provenance, identities, runtime, ev,
                                         output / split, common)
                    results[split] = {'manifest': split + '/manifest.json',
                        'manifest_sha256': sha(output / split / 'manifest.json'), 'rows': result['rows'],
                        'rows_sha256': result['rows_sha256'], 'mean_old_final_selection_d3': result['mean_old_final_selection_d3'],
                        'mean_shortlist_oracle': result['mean_shortlist_oracle']}
                del dataset
            final = {**common, 'status': 'completed', 'mode': 'CPU_audit_only' if a.audit_only else 'GPU_candidate_cache',
                     'splits': results, 'elapsed_seconds': time.monotonic()-started}
            atomic_json(output / 'manifest.json', final)
            print(json.dumps({'status': 'completed', 'mode': final['mode'], 'output': str(output),
                'manifest_sha256': sha(output / 'manifest.json')}), flush=True)
    except BaseException as exc:
        atomic_json(output / 'failure.json', {'type': type(exc).__name__, 'message': str(exc),
            'elapsed_seconds': time.monotonic()-started, 'source_sha256': sha(__file__)})
        raise


if __name__ == '__main__':
    main()
