from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import cache_c_candidates as cc


def identities():
    return {'rows': np.asarray([4, 8, 10]), 'frame': np.asarray([30, 31, 35]),
            'scene': np.asarray(['s0', 's0', 's1']), 'session': np.asarray(['g0', 'g0', 'g1'])}


def batch(rows, value=1.):
    n, k = len(rows), 2
    return dict(rows=np.asarray(rows, np.int64), candidate_ids=np.tile([0, 1], (n, 1)),
                candidate_valid=np.ones((n, k), bool), scores=np.zeros((n, k), np.float32),
                token=np.full((n, k, 256), value, np.float32), goal_xy=np.zeros((n, 2), np.float32),
                d3=np.zeros((n, k), np.float32), gt=np.zeros((n, 6, 2), np.float32))


def test_exact_storage_preserves_bf16_and_rejects_loss_or_overflow():
    bf = torch.tensor([0., -0., 1.03125, -8.5, 2**-10]).bfloat16().float().numpy()
    assert cc.exact_token_dtype(bf) == np.dtype('float16')
    assert np.array_equal(bf.view(np.uint8), bf.astype(np.float16).astype(np.float32).view(np.uint8))
    assert cc.exact_token_dtype(np.asarray([1.0001], np.float32)) == np.dtype('float32')
    assert cc.exact_token_dtype(np.asarray([1e8], np.float32)) == np.dtype('float32')
    assert cc.exact_token_dtype(np.asarray([2**-30], np.float32)) == np.dtype('float32')
    with pytest.raises(ValueError, match='Nonfinite'):
        cc.exact_token_dtype(np.asarray([np.nan], np.float32))


def test_memmap_promotion_retains_prior_bytes_and_manifest_hashes(tmp_path):
    writer = cc.CacheWriter(tmp_path/'cache', identities(), 2, 'float16', {'bank_path': '/fixed/bank.npz'})
    first = batch([4, 8], -0.)
    writer.append(first)
    assert writer.token_dtype == np.dtype('float16')
    last = batch([10], 1.0001)
    writer.append(last)
    result = writer.finish({'split': 'train'})
    assert result['status'] == 'completed' and result['token_float32_promotion']
    restored = np.load(tmp_path/'cache/token.npy', mmap_mode='r')
    original = np.concatenate([first['token'], last['token']])
    assert np.array_equal(restored.view(np.uint8), original.view(np.uint8))
    for key, spec in result['files'].items():
        assert cc.sha(tmp_path/'cache'/spec['path']) == spec['sha256']
    assert np.load(tmp_path/'cache/scene_index.npy').tolist() == [0, 0, 1]
    assert np.load(tmp_path/'cache/session_index.npy').tolist() == [0, 0, 1]


def test_no_overwrite_no_incomplete_publication_or_row_reorder(tmp_path):
    root = tmp_path/'cache'
    writer = cc.CacheWriter(root, identities(), 2, 'float16', {})
    with pytest.raises(ValueError, match='identity/order'):
        writer.append(batch([8, 4]))
    assert writer.written == 0
    with pytest.raises(ValueError, match='incomplete'):
        writer.finish({})
    assert not (root/'manifest.json').exists()
    with pytest.raises(FileExistsError):
        cc.CacheWriter(root, identities(), 2, 'float16', {})
    malformed = batch([4]); malformed['candidate_valid'][:] = False
    with pytest.raises(ValueError, match='masks'):
        writer.append(malformed)


def test_label_metric_constructive_budget_and_random_rounding():
    gt = np.zeros((1, 6, 2), np.float32)
    xy = np.zeros((1, 2, 6, 2), np.float32)
    xy[0, 0, :, 0] = [.1, .1, .2, .2, .3, .3]
    xy[0, 1, :, 1] = 2.
    d3, _ = cc.label_costs(xy, gt)
    assert d3[0, 0] == pytest.approx(.15, abs=1e-7)
    assert d3[0, 1] == pytest.approx(2., abs=1e-7)
    rng = np.random.default_rng(0)
    xy = rng.normal(0, 40, (8, 640, 6, 2)).astype(np.float32)
    gt = rng.normal(0, 15, (8, 6, 2)).astype(np.float32)
    result, delta = cc.label_costs(xy, gt)
    assert result.shape == (8, 640) and np.isfinite(result).all() and delta < 1e-4


def test_forward_feature_extraction_has_no_label_access():
    class InputOnly:
        def __getitem__(self, key):
            assert key == 'goal_xy', 'Unexpected label or state access'
            return torch.ones(1, 2)
    out = dict(candidate_ids=torch.tensor([[2, 3]]), candidate_valid=torch.ones(1, 2, dtype=torch.bool),
               scores=torch.tensor([[.1, .2]]), candidate_tokens=torch.ones(1, 2, 256, dtype=torch.bfloat16),
               candidate_xy=torch.zeros(1, 2, 6, 2), gt_plan=object(), state_target=object())
    features, xy, dtype = cc.feature_arrays(out, InputOnly(), [42])
    assert set(features) == {'rows', 'candidate_ids', 'candidate_valid', 'scores', 'token', 'goal_xy'}
    assert 'gt_plan' not in features and dtype == 'torch.bfloat16' and xy.shape == (1, 2, 6, 2)


def test_counts_only_changes_selection_attributes():
    public = torch.nn.Module()
    public.path_filter, public.velocity_filter = (128, 20), (64, 10)
    public._trajectory_head = SimpleNamespace(path_vocab=torch.empty(1024), vel_vocab=torch.empty(1024))
    public.register_parameter('weight', torch.nn.Parameter(torch.randn(2)))
    temporal = torch.nn.Module(); temporal.base = public
    model = torch.nn.Module(); model.base = temporal
    original = public.weight.detach().clone()
    info = cc.configure_counts(model, [128, 20], [64, 32])
    assert torch.equal(original, public.weight)
    assert public.velocity_filter == (64, 32) and info['candidate_count'] == 640
