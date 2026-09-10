"""CPU-only validation of a completed C cache against an independent diagnostic.

Reads frozen cache/NPZ arrays only. Does not load a model or run inference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bytes_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a.shape == b.shape and a.dtype == b.dtype and
            np.array_equal(np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8)))


def audit(directory, reference):
    root = Path(directory).resolve()
    m = json.loads((root/'manifest.json').read_text())
    assert m['status'] == 'completed' and m['split'] == 'tune' and m['rows'] == 1998
    assert m['rows_sha256'] == '1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88'
    arrays = {}
    for key, field in m['files'].items():
        path = root/field['path']
        assert path.parent == root and sha(path) == field['sha256'], key + ' file hash mismatch'
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        assert list(value.shape) == field['shape'] and str(value.dtype) == field['dtype']
        arrays[key] = value
    assert hashlib.sha256(np.asarray(arrays['rows'], '<i8').tobytes()).hexdigest() == m['rows_sha256']
    assert sha(m['bank_path']) == m['bank_sha256']
    with np.load(m['bank_path'], allow_pickle=False) as bank_npz:
        bank = bank_npz['traj_xy8'][..., :6, :2].reshape(-1, 6, 2)
    ids, valid, scores = arrays['candidate_ids'], arrays['candidate_valid'], arrays['scores']
    assert np.all((ids >= 0) & (ids < len(bank))) and valid.any(-1).all()
    assert np.isfinite(scores).all()
    selection = np.where(valid, scores, -np.inf).argmax(-1)
    selected_ids = ids[np.arange(len(ids)), selection]
    pred = bank[selected_ids]
    weights = np.asarray([11, 11, 5, 5, 2, 2], np.float64)/36.
    direct = (np.linalg.norm(pred.astype(np.float64)-arrays['gt'].astype(np.float64), axis=-1)*weights).sum(-1)
    cached = arrays['d3'][np.arange(len(ids)), selection]
    max_all_cost_error = 0.
    for start in range(0, len(ids), 32):
        stop = min(start+32, len(ids))
        assert np.isfinite(arrays['token'][start:stop]).all()
        exact = (np.linalg.norm(bank[ids[start:stop]].astype(np.float64)-
            arrays['gt'][start:stop,None].astype(np.float64), axis=-1)*weights).sum(-1)
        actual = arrays['d3'][start:stop]
        assert np.allclose(actual, exact, atol=2e-6, rtol=2e-6)
        max_all_cost_error = max(max_all_cost_error, float(np.abs(actual-exact).max()))
    with np.load(reference, allow_pickle=False) as z:
        result = {
            'rows_byte_equal': bytes_equal(arrays['rows'], z['rows']),
            'all_candidate_ids_byte_equal': bytes_equal(ids, z['candidate_ids']),
            'all_candidate_valid_byte_equal': bytes_equal(valid, z['candidate_valid']),
            'all_old_scores_byte_equal': bytes_equal(scores, z['candidate_scores']),
            'old_score_max_absolute_difference': float(np.abs(scores-z['candidate_scores']).max()),
            'selected_ids_byte_equal': bytes_equal(selected_ids, z['candidate_id']),
            'selected_predictions_byte_equal': bytes_equal(pred, z['pred']),
            'direct_float64_d3_byte_equal': bytes_equal(direct, z['d3']),
            'cached_float32_d3_allclose_reference': bool(np.allclose(cached, z['d3'], atol=2e-6, rtol=2e-6)),
            'cached_float32_d3_max_absolute_difference': float(np.abs(cached-z['d3']).max()),
            'oracle_max_absolute_difference': float(np.abs(np.where(valid, arrays['d3'], np.inf).min(-1)-z['final_oracle']).max()),
            'session_identity_equal': bool(np.array_equal(np.asarray(m['sessions'])[arrays['session_index']], z['session'])),
            'scene_identity_equal': bool(np.array_equal(np.asarray(m['scenes'])[arrays['scene_index']], z['scenario']))}
    result.update(status='completed', cpu_only=True, model_loaded=False, inference_performed=False,
        cache_manifest=str(root/'manifest.json'), cache_manifest_sha256=sha(root/'manifest.json'),
        independent_diagnostic=str(Path(reference).resolve()), independent_diagnostic_sha256=sha(reference),
        source_sha256=sha(__file__), all_cache_file_hashes_shapes_dtypes_verified=True,
        all_token_values_finite=True, all_candidate_costs_match_float64_with_declared_tolerance=True,
        all_candidate_d3_max_absolute_rounding_error=max_all_cost_error,
        mean_cached_d3=float(cached.astype(np.float64).mean()), mean_direct_d3=float(direct.mean()),
        mean_cached_oracle=float(np.where(valid, arrays['d3'], np.inf).min(-1).astype(np.float64).mean()))
    return result


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--cache', required=True); p.add_argument('--reference', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    output = Path(a.output)
    assert not output.exists(), 'Refusing overwrite'
    result = audit(a.cache, a.reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
