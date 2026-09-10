from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

import verify_c_scene_raw as v
from c_scene_selector import SceneResidualSelector


def test_new_checkpoint_is_never_given_to_legacy_loader():
    calls = []
    config = {'original_c_checkpoint': '/fixed/original_c.pth'}
    scene = {'config': config, 'manifest': {'bank_sha256': 'bank'}}
    live = SimpleNamespace(inspect_scene_checkpoint=lambda path, ev: calls.append(('new', path)) or scene)
    plan = SimpleNamespace(receipt={'checkpoint_sha256': v.C_SHA, 'bank_sha256': 'bank'},
        manifest={'arguments': {'common_status': True, 'history_mode': 'real'}})
    ev = SimpleNamespace(inspect_checkpoint=lambda path, **kw: calls.append(('legacy', path)) or plan)
    args = SimpleNamespace(checkpoint='/new/head.pth', original_c_checkpoint=None, worktree='/work', base='/base')
    assert v.inspect_composition(args, live, ev) == (scene, plan)
    assert calls == [('new', '/new/head.pth'), ('legacy', '/fixed/original_c.pth')]
    args.original_c_checkpoint = args.checkpoint
    with pytest.raises(ValueError, match='its own original'):
        v.inspect_composition(args, live, ev)


def prepared():
    return SimpleNamespace(inputs={key: torch.zeros(1, 2) for key in v.INPUT_FIELDS},
        selector_inputs={'goal_xy': torch.zeros(1, 2)}, metadata={
            'pixel_dependency_sha256': v.PIXEL_SHA, 'labels_read': False, 'aux_cache_read': False,
            'status_cache_read': False, 'planner_status_input_present': False, 'goal_only_in_selector_inputs': True})


def test_raw_boundary_rejects_goal_state_or_label_bypass():
    p = prepared()
    assert set(v.checked_prepared(p)) == v.INPUT_FIELDS | {'goal_xy'}
    for forbidden in ('gt_plan', 'status', 'goal_xy'):
        p = prepared(); p.inputs[forbidden] = torch.zeros(1, 8)
        with pytest.raises(ValueError, match='whitelist'):
            v.checked_prepared(p)
    p = prepared(); p.metadata['status_cache_read'] = True
    with pytest.raises(ValueError, match='training cache'):
        v.checked_prepared(p)


def test_new_head_hook_rejects_direct_state_slots_and_removes_hook():
    head = SceneResidualSelector()
    with v.new_head_boundary(head) as counts:
        head.score_head(torch.zeros(1, 2, 96))
        bad = torch.zeros(1, 2, 96); bad[..., 24] = 1.
        with pytest.raises(ValueError, match='Direct state'):
            head.score_head(bad)
    assert counts == {'new_score_head_calls': 1}
    assert not head.score_head._forward_pre_hooks
    # An exception inside the guarded block still removes the hook.
    with pytest.raises(RuntimeError):
        with v.new_head_boundary(head):
            raise RuntimeError('deliberate')
    assert not head.score_head._forward_pre_hooks


def test_actual_outputs_saved_even_when_byte_parity_fails(tmp_path):
    helper = v.module_from('/home/a/adcl_status_20260910/verify_temporal_raw_inference.py', '_test_raw_helper')
    outputs = {key: torch.zeros(1, 2) for key in v.OUTPUT_FIELDS}
    second = {key: value.clone() for key, value in outputs.items()}
    second['candidate_tokens'][0, 0] = 1.
    report = v.save_output_pair(tmp_path, outputs, second, helper, 14730)
    assert not report['candidate_tokens']['bitwise_equal']
    assert report['scores']['bitwise_equal']
    with np.load(tmp_path/'outputs.npz', allow_pickle=False) as z:
        assert z['rows'].tolist() == [14730]
        assert z['cache_candidate_tokens'][0, 0] == 1.
        assert z['raw_candidate_tokens'][0, 0] == 0.


def test_all_dependency_sources_are_pinned_and_mutation_rejected(tmp_path):
    import shutil
    for name in ('verify_temporal_raw_inference.py', 'temporal_deployment.py'):
        shutil.copy2(Path('/home/a/adcl_status_20260910')/name, tmp_path/name)
    a = SimpleNamespace(raw_helper_dir=tmp_path,
        live_evaluator_source=Path(__file__).with_name('evaluate_c_scene_selector.py'),
        original_evaluator_source='/home/a/adcl_status_20260910/evaluate_temporal_checkpoint.py')
    assert len(v.checked_sources(a)) == 5
    with (tmp_path/'temporal_deployment.py').open('a') as f:
        f.write('\n# mutation\n')
    with pytest.raises(ValueError, match='Source pin'):
        v.checked_sources(a)
