"""Join the frozen causal-status overlay onto a completed candidate cache.

C already routes this same past-only causal (vx, vy, ax, ay) into its shared
perception query.  This module exposes it to the *final selection* among bank
candidates that are already complete, which the organiser rulings list as an
allowed use next to common-feature improvement: "과거 정보를 직접적으로 planner
입력으로 또는 단순 임베딩 형태로 사용하는 것을 금지하며, 여러 task의 공통 특징을
향상하는 등의 간접적 활용은 허용" and "모델의 여러 출력 중 선택에만 활용하는
것은 허용".  Nothing here reaches candidate generation and no future value is
read; the contract check below refuses any overlay that claims otherwise.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

OVERLAY_ROOT = Path('/NHNHOME/data/sukim/adcl/data/etri/motiondrive_v2_shared_status_a1_20260908_ops')
EXPECTED_SPLIT_SHA = 'f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936'
EXPECTED_EGO_SHA = 'd35a69bb229b2d4736aff7dddefd3522ddbd346e112283e2ce58ce3986a023bd'


def _sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _rows_sha(rows):
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()


def load_status8(cache_directory, split, rows, overlay_root=OVERLAY_ROOT):
    """Return [N,8] float32 with causal vx,vy,ax,ay in slots 4:8, and provenance.

    Slots 0:4 stay zero because the original 32-feature helper reads only 4:8.
    The same contract checks as temporal_data._load_causal_status are applied,
    plus a frame-identity check against the cache's own frame array.
    """
    root = Path(overlay_root)
    manifest_path = root / 'overlay_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    contract = manifest['causal_contract']
    if (manifest['split_manifest_sha256'] != EXPECTED_SPLIT_SHA
            or manifest['ego_cache_sha256'] != EXPECTED_EGO_SHA
            or contract['future_values_used']
            or contract['fields'][:4] != ['vx', 'vy', 'ax', 'ay']
            or contract['frames'] != list(range(-10, 1))
            or not np.array_equal(np.asarray(contract['seconds']), np.arange(-10, 1) / 10.)):
        raise ValueError('Past-only nominal causal status contract mismatch')

    spec = manifest['artifacts'][split]
    artifact = root / spec['file']
    if artifact.parent.resolve() != root.resolve() or _sha(artifact) != spec['sha256']:
        raise ValueError('Causal status artifact path/hash mismatch')
    with np.load(artifact, allow_pickle=False) as bundle:
        source_rows = bundle['row']
        source_frames = bundle['frame']
        source_status = bundle['status5']
    if (source_rows.ndim != 1 or source_rows.dtype.kind not in 'iu'
            or len(np.unique(source_rows)) != len(source_rows)):
        raise ValueError('Causal status rows must be unique integers')
    if _rows_sha(source_rows) != spec['rows_sha256'] or source_status.shape != (len(source_rows), 5):
        raise ValueError('Causal status population mismatch')

    rows = np.asarray(rows, dtype=np.int64)
    lookup = {int(row): position for position, row in enumerate(source_rows)}
    try:
        positions = np.asarray([lookup[int(row)] for row in rows])
    except KeyError as exc:
        raise ValueError('Requested row absent from causal overlay') from exc

    cache_directory = Path(cache_directory)
    cache_manifest = json.loads((cache_directory / 'manifest.json').read_text())
    frame_path = cache_directory / 'frame.npy'
    if _sha(frame_path) != cache_manifest['files']['frame']['sha256']:
        raise ValueError('Cache frame array checksum mismatch')
    frames = np.load(frame_path, allow_pickle=False)
    if len(frames) != len(rows) or not np.array_equal(source_frames[positions], frames):
        raise ValueError('Causal state row/frame mismatch against the cache')

    state4 = source_status[positions, :4].astype(np.float32, copy=True)
    if not np.isfinite(state4).all():
        raise ValueError('Nonfinite causal state')
    status8 = np.zeros((len(rows), 8), dtype=np.float32)
    status8[:, 4:8] = state4
    provenance = {
        'manifest_sha256': _sha(manifest_path), 'path': str(artifact),
        'sha256': _sha(artifact), 'rows_sha256': _rows_sha(rows),
        'split': split, 'fields': ['vx', 'vy', 'ax', 'ay'], 'slots': '4:8',
        'definition': 'exact 11 causal pose frames -10..0, nominal 10Hz quadratic fit, current ego XY',
        'units': ['m/s', 'm/s', 'm/s^2', 'm/s^2'], 'future_values_used': False,
        'use': 'final selection among completed bank candidates only',
    }
    return status8, provenance
