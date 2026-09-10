"""CPU tests for the new capture/head only; no datasets, checkpoints or GPU."""
from __future__ import annotations

import copy
import importlib.util
import inspect
import os
from pathlib import Path
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from c_scene_selector import CandidateTokenCapture, SceneResidualSelector, build_features32


SOURCE = Path(__file__).resolve().parent.parent / "adcl_status_20260910/source/sparsedrivev2_20260910"


def read_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        bank = torch.arange(2*3*8*3, dtype=torch.float32).reshape(2,3,8,3) / 50.
        self.register_buffer("traj_vocab", bank)
        mask = torch.ones(2,3,8)
        mask[0, 0, -3] = 0
        self.register_buffer("traj_mask", mask)
        self.decoder = nn.Module()
        layer = nn.Module()
        layer.traj_mlp = nn.Sequential(nn.Linear(256, 16), nn.ReLU(), nn.Linear(16, 1))
        self.decoder.layers = nn.ModuleList([layer])
        self.embedding = nn.Embedding(6, 256)


class TinyPublic(nn.Module):
    def __init__(self):
        super().__init__()
        self._trajectory_head = TinyHead()
        self.fail = False
        self.twice = False
        self.reorder = False
        self.coordinate_corrupt = False
        self.last_tokens = None

    def forward(self, images):
        b = len(images)
        h = self._trajectory_head
        p = torch.tensor([1,0], device=images.device).expand(b,-1)
        v = torch.tensor([2,0], device=images.device).expand(b,-1)
        ids = (p[:,:,None]*3+v[:,None,:]).flatten(1,2)
        tokens = h.embedding(ids)+images[:,None,:]
        self.last_tokens = tokens
        scores = h.decoder.layers[-1].traj_mlp(tokens).squeeze(-1)
        if self.twice:
            h.decoder.layers[-1].traj_mlp(tokens)
        if self.fail:
            raise RuntimeError("deliberate base exception")
        valid = h.traj_mask[p[:,:,None],v[:,None,:],:6].bool().all(-1).flatten(1,2)
        scores = scores.masked_fill(~valid,-1e4)
        xy = h.traj_vocab[p[:,:,None],v[:,None,:],:6,:2].flatten(1,2)
        if self.reorder:
            ids, xy, scores, valid = [x.flip(1) for x in (ids,xy,scores,valid)]
        if self.coordinate_corrupt:
            xy = xy+.01
        winner = scores.argmax(-1)
        batch = torch.arange(b)
        return dict(scores=scores,candidate_xy=xy,candidate_ids=ids,candidate_valid=valid,
            trajectory=xy[batch,winner],selected_candidate_id=ids[batch,winner],
            path_ids=p,velocity_ids=v)


class TinyTemporal(nn.Module):
    def __init__(self, public):
        super().__init__()
        self.base = public

    @property
    def _trajectory_head(self):
        return self.base._trajectory_head

    def forward(self, images, perception_status=None):
        state = images.mean(-1)
        conditioned = images if perception_status is None else images + perception_status.mean(-1, keepdim=True)
        out = self.base(conditioned)
        return {**out,"aux_state":state}


class TinyFinalGoal(nn.Module):
    def __init__(self, public):
        super().__init__()
        self.base = TinyTemporal(public)

    @property
    def _trajectory_head(self):
        return self.base._trajectory_head

    def forward(self, images, perception_status=None, goal_xy=None):
        out = self.base(images,perception_status)
        if goal_xy is None:
            return out
        scores = out["scores"] - .05*(out["candidate_xy"][:,:,-1]-goal_xy[:,None]).square().sum(-1)
        winner=scores.masked_fill(~out['candidate_valid'],-torch.inf).argmax(-1)
        batch=torch.arange(len(images))
        return {**out,"scores":scores,"trajectory":out['candidate_xy'][batch,winner],
                "selected_candidate_id":out['candidate_ids'][batch,winner]}


def activate(head):
    with torch.no_grad():
        nn.init.normal_(head.score_head[-1].weight, std=.3)


class SceneSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)
        self.public=TinyPublic()
        self.original=TinyFinalGoal(self.public).eval()
        self.images=torch.randn(2,256)
        self.goal=torch.tensor([[4.,0.],[7.,1.]])
        self.capture=CandidateTokenCapture(self.original)

    def output(self):
        return self.capture(images=self.images,goal_xy=self.goal)

    def test_capture_preserves_old_fields_exactly_and_matches_token_order(self):
        original=self.original(images=self.images,goal_xy=self.goal)
        expected_tokens=self.public.last_tokens.detach().clone()
        before={k:v.clone() for k,v in self.original.state_dict().items()}
        out=self.output()
        for key in original:
            self.assertTrue(torch.equal(original[key],out[key]),key)
        self.assertTrue(torch.equal(out['candidate_tokens'],expected_tokens))
        self.assertEqual(out['candidate_tokens'].shape,(2,4,256))
        self.assertTrue(torch.equal(out['candidate_ids'][0],torch.tensor([5,3,2,0])))
        for key,value in before.items():
            self.assertTrue(torch.equal(value,self.original.state_dict()[key]),key)
        self.assertFalse(out['candidate_tokens'].requires_grad)
        self.assertTrue(all(not p.requires_grad for p in self.original.parameters()))
        self.capture.train()
        self.assertFalse(self.original.training)

    def test_goal_cannot_change_upstream_tokens_ids_coordinates_or_aux(self):
        a=self.output()
        b=self.capture(images=self.images,goal_xy=self.goal+torch.tensor([10.,-5.]))
        for key in ('candidate_tokens','candidate_xy','candidate_ids','candidate_valid','aux_state','path_ids','velocity_ids'):
            self.assertTrue(torch.equal(a[key],b[key]),key)
        self.assertFalse(torch.equal(a['scores'],b['scores']))
        head=SceneResidualSelector()
        activate(head)
        c,d=head(a,self.goal),head(a,self.goal+torch.tensor([10.,-5.]))
        for key in ('candidate_tokens','candidate_xy','candidate_ids','candidate_valid','aux_state'):
            self.assertIs(c[key],a[key])
            self.assertIs(d[key],a[key])

    def test_hook_cleanup_on_exception_duplicate_and_alignment_failures(self):
        mlp=self.public._trajectory_head.decoder.layers[-1].traj_mlp
        unrelated=mlp.register_forward_pre_hook(lambda *_: None)
        original_hooks=len(mlp._forward_pre_hooks)
        for attr,message in (('fail','deliberate'),('twice','more than once'),
                             ('reorder','token order'),('coordinate_corrupt','immutable bank')):
            setattr(self.public,attr,True)
            with self.assertRaisesRegex((ValueError,RuntimeError),message):
                self.output()
            self.assertEqual(len(mlp._forward_pre_hooks),original_hooks)
            setattr(self.public,attr,False)
            self.output()
        unrelated.remove()

    def test_reentrancy_guard_and_deepcopy(self):
        self.capture._capture_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError,'concurrent/reentrant'):
                self.output()
        finally:
            self.capture._capture_lock.release()
        other=copy.deepcopy(self.capture)
        self.assertTrue(torch.equal(other(images=self.images,goal_xy=self.goal)['candidate_tokens'],
                                    self.output()['candidate_tokens']))

    def test_optional_token_gradients_and_detach(self):
        for detach in (False,True):
            original=TinyFinalGoal(TinyPublic())
            wrapped=CandidateTokenCapture(original,detach_tokens=detach,freeze_base=False)
            out=wrapped(images=self.images,goal_xy=self.goal)
            self.assertEqual(out['candidate_tokens'].requires_grad,not detach)
            loss=out['scores'][out['candidate_valid']].sum()
            if not detach:
                loss=loss+out['candidate_tokens'].square().mean()
            loss.backward()
            grad=original.base.base._trajectory_head.embedding.weight.grad
            self.assertGreater(float(grad.abs().sum()),0.)

    def test_real_zero_same_initial_state_and_zero_head_exact_noop(self):
        torch.manual_seed(31); real=SceneResidualSelector(mode='real')
        torch.manual_seed(31); zero=SceneResidualSelector(mode='zero')
        self.assertEqual(sum(p.numel() for p in real.parameters()),37697)
        for key,value in real.state_dict().items():
            self.assertTrue(torch.equal(value,zero.state_dict()[key]),key)
        out=self.output()
        for head in (real,zero):
            result=head(out,self.goal)
            for key in out:
                self.assertTrue(torch.equal(out[key],result[key]),key)
            self.assertTrue(torch.equal(result['scene_score_residual'],torch.zeros_like(out['scores'])))
            self.assertIs(result['old_final_scores'],out['scores'])

    def test_real_uses_tokens_zero_is_token_invariant_after_activation(self):
        real=SceneResidualSelector(mode='real'); activate(real)
        zero=SceneResidualSelector(mode='zero'); zero.load_state_dict(real.state_dict(),strict=True)
        a=self.output()
        b={**a,'candidate_tokens':a['candidate_tokens'].flip(-1)*2.}
        self.assertFalse(torch.equal(real(a,self.goal)['scores'],real(b,self.goal)['scores']))
        self.assertTrue(torch.equal(zero(a,self.goal)['scores'],zero(b,self.goal)['scores']))

    def test_invalid_tokens_masked_before_arithmetic_and_no_raw_state_inputs(self):
        a=self.output()
        invalid=~a['candidate_valid']
        dirty_tokens=a['candidate_tokens'].clone()
        dirty_tokens[invalid]=torch.nan
        dirty_xy=a['candidate_xy'].clone()
        dirty_xy[invalid]=torch.nan
        b={**a,'candidate_tokens':dirty_tokens,'candidate_xy':dirty_xy}
        head=SceneResidualSelector();activate(head)
        first,second=head(a,self.goal),head(b,self.goal)
        self.assertTrue(torch.equal(first['scores'],second['scores']))
        self.assertTrue(torch.equal(first['trajectory'],second['trajectory']))
        self.assertTrue(torch.equal(first['selected_candidate_id'],second['selected_candidate_id']))
        self.assertTrue(torch.equal(second['scene_score_residual'][invalid],torch.zeros_like(second['scene_score_residual'][invalid])))
        for key in ('status','state_target','gt_plan'):
            with self.assertRaises(TypeError):
                head(a,self.goal,**{key:torch.zeros(2,8)})
        self.assertEqual(set(inspect.signature(head.forward).parameters),{'output','goal_xy'})

    def test_initial_and_activated_gradients_are_finite(self):
        original=TinyFinalGoal(TinyPublic())
        capture=CandidateTokenCapture(original,detach_tokens=False,freeze_base=False)
        head=SceneResidualSelector()
        for active in (False,True):
            original.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
            if active:
                activate(head)
            out=head(capture(images=self.images,goal_xy=self.goal),self.goal)
            loss=out['scores'][out['candidate_valid']].square().mean()
            loss.backward()
            self.assertGreater(float(head.score_head[-1].weight.grad.abs().sum()),0.)
            self.assertGreater(float(original.base.base._trajectory_head.embedding.weight.grad.abs().sum()),0.)
            if active:
                self.assertGreater(float(head.token_projection[1].weight.grad.abs().sum()),0.)
            for module in (original,head):
                self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in module.parameters()))

    def test_exact_feature_parity_with_original_uuid_isolated_helper(self):
        source=Path(os.environ.get('ORIGINAL_RELATIVE_SELECTOR',SOURCE/'relative_selector.py'))
        self.assertTrue(source.is_file(),'Provide original frozen helper path for parity test')
        old=read_module('isolated_relative_for_scene_test',source)
        a=self.output()
        # Invalid coordinates may be NaN; both helpers must mask before math.
        xy=a['candidate_xy'].clone();xy[~a['candidate_valid']]=torch.nan
        a={**a,'candidate_xy':xy}
        for goal in (self.goal,None):
            for autocast in (False,True):
                with torch.autocast('cpu',dtype=torch.bfloat16,enabled=autocast):
                    expected=old.make_candidate_features(a,torch.zeros(2,8),goal)
                    actual=build_features32(a,goal)
                self.assertTrue(torch.equal(expected,actual))
                self.assertEqual(expected.numpy().tobytes(),actual.numpy().tobytes())

    def test_actual_public_decoder_reference_capture_and_order(self):
        path=Path(os.environ.get('PUBLIC_MODEL_PATH',SOURCE/'public_model.py'))
        self.assertTrue(path.is_file(),'Provide original frozen public model source')
        public=read_module('isolated_public_for_scene_test',path)
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__();self.conv=nn.Conv2d(3,256,1)
            def forward(self,images):
                b,c=images.shape[:2]
                x=self.conv(images.flatten(0,1))
                return [F.adaptive_avg_pool2d(x,(h,w)).reshape(b,c,256,h,w) for h,w in ((8,16),(4,8),(2,4),(1,2))]
        bank={'path_vocab':torch.zeros(2,50,3),'vel_vocab':torch.ones(3,8),
              'traj_vocab':torch.zeros(2,3,8,3),'traj_mask':torch.ones(2,3,8)}
        bank['path_vocab'][:,:,0]=torch.arange(1,51)[None]/10.
        bank['path_vocab'][1,:,1]=.1
        for p in range(2):
            for v in range(3):
                bank['traj_vocab'][p,v,:,0]=torch.arange(1,9)*(v+1)*.15
                bank['traj_vocab'][p,v,:,1]=p*.1
        with mock.patch.object(public,'SparseBackbone',Backbone):
            base=public.PublicSparseDriveV2(bank,backend='reference',mask_invalid_candidates=True).eval()
        images=torch.randn(1,3,3,32,64)
        projection=torch.tensor([[32.,-16.,0.,320.],[16.,0.,-8.,160.],[1.,0.,0.,10.],[0.,0.,0.,1.]])[None,None].repeat(1,3,1,1)
        kwargs=dict(images=images,lidar2img=projection,image_hw=torch.tensor([[32.,64.]]),status=torch.zeros(1,8))
        with torch.no_grad():
            expected=base(**kwargs)
        out=CandidateTokenCapture(base)(**kwargs)
        for key in ('candidate_xy','scores','candidate_ids','trajectory','selected_candidate_id'):
            self.assertTrue(torch.equal(expected[key],out[key]),key)
        self.assertEqual(out['candidate_tokens'].shape,(1,6,256))
        direct=base._trajectory_head.decoder.layers[-1].traj_mlp(out['candidate_tokens']).squeeze(-1)
        self.assertTrue(torch.equal(direct,out['imitation_scores']))
        self.assertTrue(torch.equal(SceneResidualSelector()(out,torch.zeros(1,2))['trajectory'],out['trajectory']))


if __name__=='__main__':
    unittest.main(verbosity=2)
