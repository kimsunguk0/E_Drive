"""Strict relative-head artifact verification, without training-data imports.

Initialization reads the selected head and a small provenance JSON. It never
reads feature/label caches. Runtime only consumes live candidate/status/goal
features through the independently pinned relative selector implementation.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch

try:
    from .relative_selector import RelativeScoreHead, FEATURE_NAMES, FEATURE_VERSION
except ImportError:
    from relative_selector import RelativeScoreHead, FEATURE_NAMES, FEATURE_VERSION

RELATIVE_SOURCE_SHA256='7fa75415a40691f555cca27ab9fe88331725177f19887c01a0e887121e2ac59f'
TRAINER_SOURCE_SHA256='8177d10846d229c0adef97db094a0482278aee3d1fd7aaf524f739f1f104a577'
TUNE_SHA256='1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88'


def require(ok,message):
    if not ok:raise ValueError(message)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def load_relative_artifact(path,*,expected_sha256,cache_manifest_path=None,
                           base_checkpoint_sha256=None,bank_sha256=None,
                           base_goal_mode=None):
    """Return strict CPU FP32 head, verified cache metadata and receipt.

    ``cache_manifest_path`` can name a bundled copy of provenance JSON. The
    original cache directory is otherwise used only to locate that JSON.
    No `.npy`/GT/features file is opened by this verifier.
    """
    path=Path(path)
    with path.open('rb') as f:
        h=hashlib.sha256()
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
        actual=h.hexdigest()
        require(actual==expected_sha256,'Relative checkpoint SHA256 mismatch')
        f.seek(0);payload=torch.load(f,map_location='cpu',weights_only=True)
    require(isinstance(payload,dict) and {'relative_head','manifest','step','result'}<=set(payload),
            'Expected trained relative-head payload')
    require(isinstance(payload['step'],int) and payload['step']>0 and
            payload['result']['step']==payload['step'],'Relative head/result step mismatch')
    manifest=payload['manifest']
    require(manifest['torch']==str(torch.__version__),'Use exact relative-head PyTorch runtime')
    require(manifest['source_sha256']['relative_selector.py']==RELATIVE_SOURCE_SHA256,
            'Unrecognized relative feature/head implementation')
    require(file_sha(Path(__file__).with_name('relative_selector.py'))==RELATIVE_SOURCE_SHA256,
            'Runtime relative selector source changed')
    require(manifest['source_sha256']['train_cached_selector.py']==TRAINER_SOURCE_SHA256,
            'Unrecognized relative-head trainer')
    require(manifest['arguments']['objective'] in ('soft_ce','centered_d3'), 'Unknown head objective')
    cache_path=Path(cache_manifest_path) if cache_manifest_path is not None else Path(manifest['cache_path'])/'manifest.json'
    cache_bytes=cache_path.read_bytes()
    cache_sha=hashlib.sha256(cache_bytes).hexdigest()
    require(cache_sha==manifest['cache_manifest_sha256'],'Relative cache provenance JSON changed')
    cache=json.loads(cache_bytes)
    require(cache==manifest['cache_manifest'],'Embedded/external cache provenance differ')
    require(cache['schema']=='sparsedrivev2_selection_cache_v1' and cache['status']=='completed',
            'A completed fixed-candidate cache is required')
    require(cache['feature_source_sha256']==RELATIVE_SOURCE_SHA256 and
            cache['feature_version']==FEATURE_VERSION and cache['feature_names']==list(FEATURE_NAMES),
            'Relative feature schema changed')
    require(cache['feature_dim']==32 and cache['candidate_count']==200,'Unsupported candidate feature dimensions')
    require(cache['base_goal_mode'] in ('none','selection') and cache['goal_feature_mode']=='selection',
            'Relative features require provided goal selection independently of base goal mode')
    require(cache['precision']=='bf16' and cache['held_population_evaluated'] is False,
            'Unsupported feature-cache precision or population history')
    require(cache['tune_rows_sha256']==TUNE_SHA256,'Relative head cache changed original tune population')
    require(cache['checkpoint']['sha256']==cache['provenance']['checkpoint_sha256'] and
            cache['bank']['sha256']==cache['provenance']['bank_sha256'], 'Cache ancestry references disagree')
    if base_checkpoint_sha256 is not None:
        require(cache['checkpoint']['sha256']==base_checkpoint_sha256,'Relative head belongs to another base checkpoint')
    if bank_sha256 is not None:
        require(cache['bank']['sha256']==bank_sha256,'Relative head belongs to another fixed bank')
    if base_goal_mode is not None:
        require(cache['base_goal_mode']==base_goal_mode,'Relative/base goal modes disagree')
    head=RelativeScoreHead().float()
    require(all(t.dtype==torch.float32 for t in payload['relative_head'].values()),'Relative head must be stored FP32')
    head.load_state_dict(payload['relative_head'],strict=True)
    require(all(torch.isfinite(p).all() for p in head.parameters()),'Nonfinite relative head')
    require(sum(p.numel() for p in head.parameters())==12545,'Relative head parameter count changed')
    head.eval()
    receipt={'relative_checkpoint_sha256':actual,'relative_step':payload['step'],
        'relative_objective':manifest['arguments']['objective'],'relative_source_sha256':RELATIVE_SOURCE_SHA256,
        'relative_trainer_sha256':TRAINER_SOURCE_SHA256,'cache_manifest_sha256':cache_sha,
        'cache_manifest_path':str(cache_path.resolve()),'base_checkpoint_sha256':cache['checkpoint']['sha256'],
        'bank_sha256':cache['bank']['sha256'],'base_goal_mode':cache['base_goal_mode'],
        'feature_goal_mode':'selection','strict_relative_head_load':True,
        'relative_parameters':12545,'runtime_feature_or_label_files_opened':False}
    return head,cache,receipt
