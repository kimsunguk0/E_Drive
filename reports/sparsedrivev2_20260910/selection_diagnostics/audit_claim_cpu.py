"""Independent input-only cache rebuild and metadata-only confirmation boundary audit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path.cwd()
BASE = Path('/NHNHOME/data/sukim/adcl')
SPLIT = ROOT / 'reports/sparsedrivev2_20260910/split_audit'
CACHE = ROOT / 'cache/sparsedrivev2_20260910/features_primary_none2000_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


torch.set_num_threads(4)
started = time.time()
source = ROOT / 'experiments/sparsedrivev2_20260910/relative_selector.py'
spec = importlib.util.spec_from_file_location('relative_selector_audit', source)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
manifest = json.loads((CACHE / 'manifest.json').read_text())
assert sha(source) == manifest['feature_source_sha256'] == '7fa75415a40691f555cca27ab9fe88331725177f19887c01a0e887121e2ac59f'
assert sha(CACHE / 'manifest.json') == 'e9455b8e6577ff2a9d488dfc6780129df5b336d247adeeaabded246180b9e298'
output = {'schema': 'independent_013075_scientific_audit_v1', 'source_sha256': sha(__file__),
          'feature_source_sha256': sha(source), 'cache_manifest_sha256': sha(CACHE / 'manifest.json'),
          'gpu_used': False, 'held_predictions_or_labels_read': False, 'primary_feature_rebuild': {}}
for partition in ['train', 'tune']:
    keys = ['candidate_xy', 'base_logits', 'valid', 'status8', 'goal_xy', 'features', 'rows', 'scenario', 'session']
    arrays = {}
    for key in keys:
        path = CACHE / partition / (key + '.npy')
        assert sha(path) == manifest['artifacts'][partition + '/' + key + '.npy']['sha256']
        arrays[key] = np.load(path, allow_pickle=False, mmap_mode='r')
    maximum = np.zeros(32, np.float64)
    for start in range(0, len(arrays['rows']), 128):
        sl = slice(start, start + 128)
        data = {k: torch.from_numpy(np.array(arrays[k][sl], copy=True))
                for k in ['candidate_xy', 'base_logits', 'valid', 'status8', 'goal_xy']}
        rebuilt = helper.make_candidate_features({'candidate_xy': data['candidate_xy'],
                    'scores': data['base_logits'], 'candidate_valid': data['valid']},
                    data['status8'], data['goal_xy']).numpy()
        delta = np.abs(rebuilt.astype(np.float64) - arrays['features'][sl].astype(np.float64))
        maximum = np.maximum(maximum, delta.max((0, 1)))
    assert maximum.max() < 2e-5, (partition, maximum)
    output['primary_feature_rebuild'][partition] = {'rows': len(arrays['rows']),
            'candidates': arrays['features'].shape[1], 'max_abs_difference_by_feature': maximum.tolist(),
            'max_abs_difference': float(maximum.max()), 'tolerance': 2e-5,
            'source_inputs': ['candidate_xy', 'base_logits', 'valid', 'status8', 'provided_goal_xy'],
            'gt_or_cost_files_opened': False, 'artifact_hashes_rechecked': True,
            'scenes': len(set(arrays['scenario'].tolist())), 'sessions': len(set(arrays['session'].tolist()))}

confirmation = json.loads((SPLIT / 'confirmation12_manifest.json').read_text())
prior = json.loads((SPLIT / 'audit.json').read_text())
groups = {k: confirmation['splits'][m] for k, m in
          [('train171', 'train'), ('tune37', 'tune'), ('held32', 'confirmation12')]}
support = {}
for scene in sorted(set.union(*(set(v) for v in groups.values()))):
    path = BASE / 'data/etri/meta_train' / scene / 'meta/timestamps.parquet'
    expected = prior['timestamp_source_receipts'][scene]['sha256']
    assert sha(path) == expected
    table = pq.read_table(path, columns=['frame_id', 'timestamp'], use_threads=False)
    frames = np.asarray(table['frame_id'].to_pylist(), np.int64)
    seconds = np.asarray(table['timestamp'].to_pylist(), np.float64) / 1000.
    order = np.argsort(frames); frames = frames[order]; seconds = seconds[order]
    assert np.array_equal(frames, np.arange(-50, 350)) and (np.diff(seconds) > 0).all()
    support[scene] = {'raw_start': float(seconds[0]), 'raw_end': float(seconds[-1]),
                      'used_start': float(seconds[frames == 0][0]), 'used_end': float(seconds[-1]),
                      'sha256': expected}
output['confirmation_metadata'] = {'manifest_sha256': sha(SPLIT / 'confirmation12_manifest.json'),
        'prior_audit_sha256': sha(SPLIT / 'audit.json'), 'timestamp_files_reread': len(support),
        'raw_available_frames': [-50, 349], 'used_support_frames': [0, 349], 'groups': {}, 'pairs': {}}
for label, scenes in groups.items():
    ids = {confirmation['scene_to_session'][s] for s in scenes}
    output['confirmation_metadata']['groups'][label] = {'scenes': len(scenes), 'sessions': sorted(ids)}
for left, right in [('train171', 'held32'), ('tune37', 'held32'), ('train171', 'tune37')]:
    assert not set(groups[left]) & set(groups[right])
    assert not set(output['confirmation_metadata']['groups'][left]['sessions']) & set(output['confirmation_metadata']['groups'][right]['sessions'])
    pair = {'scene_overlap': 0, 'session_overlap': 0}
    for kind in ['used', 'raw']:
        gaps = [(max(support[a][kind+'_start'], support[b][kind+'_start']) -
                 min(support[a][kind+'_end'], support[b][kind+'_end']), a, b)
                for a in groups[left] for b in groups[right]]
        minimum = min(gaps)
        assert minimum[0] > 0
        pair[kind + '_support'] = {'minimum_gap_seconds': minimum[0], 'nearest_scenes': list(minimum[1:]),
                                  'overlapping_pairs': sum(g[0] <= 0 for g in gaps)}
    output['confirmation_metadata']['pairs'][left + '_vs_' + right] = pair
output['confirmation_metadata']['timestamp_source_sha256'] = {s: r['sha256'] for s, r in support.items()}

live = ROOT / 'reports/sparsedrivev2_20260910/public_init/relative_soft_ce_live_tune1998'
live_receipt = json.loads((live / 'result.json').read_text())
assert live_receipt['n'] == 1998 and live_receipt['relative']['relative_step'] == 2000
source_checks = {}
for filename, expected in live_receipt['source_sha256'].items():
    current = sha(ROOT / 'experiments/sparsedrivev2_20260910' / filename)
    source_checks[filename] = {'receipt_sha256': expected, 'current_sha256': current,
                              'current_matches_receipt': current == expected}
assert source_checks['relative_selector.py']['current_matches_receipt']
terminal = ROOT / 'work_dirs/sparsedrivev2_20260910/relative2000_soft_ce_s0_v1/eval_002000.npz'
assert sha(terminal) == live_receipt['cache_live_parity']['reference_sha256']
with np.load(live / 'predictions.npz', allow_pickle=False) as actual, np.load(terminal, allow_pickle=False) as expected:
    parity = {k: np.array_equal(actual[k], expected[k]) for k in ['rows', 'pred', 'candidate_id', 'd3', 'shortlist_oracle', 'point_l2', 'error_xy']}
    assert all(parity.values())
output['live_primary_parity'] = {'live_result_sha256': sha(live / 'result.json'),
                                'live_predictions_sha256': sha(live / 'predictions.npz'),
                                'terminal_predictions_sha256': sha(terminal), 'exact_arrays': parity,
                                'd3': live_receipt['official_d3'], 'receipt_sources_rehashed': source_checks,
                                'source_note': 'Evaluator/artifact currently contain a later confirmation guard. Their original receipt hashes differ; feature/head source is unchanged. Do not present later source bytes as the old executed source.'}
output['passed'] = True
output['elapsed_seconds'] = time.time() - started
target = ROOT / 'reports/sparsedrivev2_20260910/selection_diagnostics/independent_013075_claim_audit.json'
with target.open('x') as stream:
    json.dump(output, stream, indent=2, allow_nan=False); stream.write('\n')
print(json.dumps({k: v for k, v in output.items() if k != 'confirmation_metadata'}, indent=2))
print(json.dumps(output['confirmation_metadata']['pairs'], indent=2))
