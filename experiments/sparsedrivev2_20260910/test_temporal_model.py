"""CPU contracts for temporal perception, including the real public decoder.

No datasets/labels/checkpoints or GPU are required. The public decoder integration
uses its exact source and a tiny synthetic image backbone/bank to test the hook
boundary without installing timm or reading large weights.
"""
from __future__ import annotations
import copy
import importlib.util
import inspect
import os
from pathlib import Path
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from torch import nn
from torch.nn import functional as F

from temporal_model import TemporalPerceptionModel, ImageTemporalFusion, SpatialPerceptionAux


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 256, 1)
        self.bn = nn.BatchNorm2d(256)

    def forward(self, images):
        b, cams, _, h, w = images.shape
        feature = self.bn(self.conv(images.flatten(0, 1)))
        return [F.adaptive_avg_pool2d(feature, (max(h//s,1), max(w//s,1))).reshape(
            b, cams, 256, max(h//s,1), max(w//s,1)) for s in (4, 8, 16, 32)]


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        bank = torch.arange(4*6*2).float().reshape(4, 6, 2)/40.
        self.register_buffer("traj_vocab", bank)
        self.score = nn.Linear(256, 4)


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self._backbone = TinyBackbone()
        self._status_encoding = nn.Linear(8, 256)
        self._trajectory_head = TinyHead()
        self.seen_status = None
        self.fail = False
        self.twice = False

    def forward(self, images, lidar2img, image_hw, status):
        self.seen_status = status.detach().clone()
        levels = self._backbone(images)
        if self.twice:
            self._backbone(images)
        if self.fail:
            raise RuntimeError("synthetic base failure")
        feature = levels[2].mean((1, 3, 4)) + self._status_encoding(status)
        scores = self._trajectory_head.score(feature)
        xy = self._trajectory_head.traj_vocab[None].expand(len(images), -1, -1, -1)
        ids = torch.arange(4).expand(len(images), -1)
        winner = scores.argmax(1)
        return dict(trajectory=xy[torch.arange(len(images)), winner], scores=scores,
                    candidate_xy=xy, candidate_ids=ids, candidate_valid=torch.ones_like(ids, dtype=torch.bool),
                    selected_candidate_id=winner)


def inputs(batch=2):
    torch.manual_seed(14)
    images = torch.randn(batch, 3, 3, 32, 64)
    matrix = torch.tensor([[32.,-16.,0.,320.], [16.,0.,-8.,160.],
                           [1.,0.,0.,10.], [0.,0.,0.,1.]])
    return dict(images=images, lidar2img=matrix[None,None].repeat(batch,3,1,1),
                image_hw=torch.tensor([32.,64.])[None].repeat(batch,1),
                history_images=torch.randn(batch,2,3,32,64),
                time_offsets=torch.tensor([.1,.5])[None].repeat(batch,1))


def unzero(model):
    with torch.no_grad():
        nn.init.normal_(model.temporal.output.weight, std=.03)
        nn.init.normal_(model.temporal.status_condition[-1].weight, std=.15)


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.base = TinyBase().eval()
        self.before = {k:v.clone() for k,v in self.base.state_dict().items()}
        self.model = TemporalPerceptionModel(self.base).eval()
        self.x = inputs()

    def test_zero_init_exact_base_parity_and_parameter_identity(self):
        base_ids = {id(p) for p in self.base.parameters()}
        self.assertTrue(base_ids.issubset({id(p) for p in self.model.parameters()}))
        for name, value in self.before.items():
            self.assertTrue(torch.equal(value, self.model.base.state_dict()[name]), name)
        with torch.no_grad():
            original = self.base(**{k:self.x[k] for k in ('images','lidar2img','image_hw')}, status=torch.zeros(2,8))
            wrapped = self.model(**self.x, perception_status=torch.tensor([[8.,.1,.2,.3],[21.,-.1,.0,-.2]]))
        for name in original:
            self.assertTrue(torch.equal(original[name], wrapped[name]), name)
        self.assertEqual(wrapped['aux_occ'].shape, (2,1,64,48))
        self.assertEqual(wrapped['aux_lane'].shape, (2,1,64,48))
        self.assertEqual(wrapped['aux_state'].shape, (2,4))
        self.assertIs(self.model._backbone, self.base._backbone)
        self.assertIs(self.model._trajectory_head, self.base._trajectory_head)

    def test_status_only_perception_and_state_aux_independent(self):
        unzero(self.model)
        s = torch.tensor([[8.,.1,.2,.3],[21.,-.1,.0,-.2]], requires_grad=True)
        a = self.model(**self.x, perception_status=s)
        b = self.model(**self.x, perception_status=s+torch.tensor([5.,1.,1.,1.]))
        self.assertTrue(torch.equal(a['aux_state'], b['aux_state']))
        self.assertFalse(torch.equal(a['scores'], b['scores']))
        self.assertFalse(torch.equal(a['aux_occ'], b['aux_occ']))
        self.assertTrue(torch.equal(a['candidate_xy'], b['candidate_xy']))
        self.assertTrue(torch.equal(self.base.seen_status, torch.zeros(2,8)))
        self.assertIsNone(torch.autograd.grad(a['aux_state'].sum(), s, allow_unused=True, retain_graph=True)[0])
        grad = torch.autograd.grad(a['scores'].square().mean(), s)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.)
        for out in (a,b):
            expected = out['candidate_xy'][torch.arange(2),out['selected_candidate_id']]
            self.assertTrue(torch.equal(expected, out['trajectory']))

    def test_image_history_affects_scores_and_aux_after_activation(self):
        unzero(self.model)
        a = self.model(**self.x)
        y = {**self.x, 'history_images': self.x['history_images'].flip(0)}
        b = self.model(**y)
        self.assertFalse(torch.equal(a['scores'], b['scores']))
        self.assertFalse(torch.equal(a['aux_state'], b['aux_state']))
        self.assertTrue(torch.equal(a['candidate_xy'], b['candidate_xy']))

    def test_initial_plan_and_aux_gradients_are_finite(self):
        self.model.train()
        self.assertFalse(self.base._backbone.bn.training)
        out = self.model(**self.x)
        loss = out['scores'].square().mean() + out['aux_occ'].square().mean() + out['aux_lane'].square().mean() + out['aux_state'].square().mean()
        loss.backward()
        self.assertGreater(float(self.model.temporal.output.weight.grad.abs().sum()),0.)
        self.assertGreater(float(self.model.temporal.attention.in_proj_weight.grad.abs().sum()),0.)
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_bfloat16_autocast_zero_parity_and_backward(self):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            original = self.base(**{k:self.x[k] for k in ('images','lidar2img','image_hw')}, status=torch.zeros(2,8))
            wrapped = self.model(**self.x, perception_status=torch.ones(2,4))
        for key in ('scores','trajectory','candidate_ids'):
            self.assertTrue(torch.equal(original[key],wrapped[key]),key)
        loss = wrapped['scores'].float().square().mean()+wrapped['aux_state'].square().mean()+wrapped['aux_occ'].square().mean()
        loss.backward()
        self.assertGreater(float(self.model.temporal.output.weight.grad.abs().sum()),0.)
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(),name)

    def test_exception_cleanup_and_recovery(self):
        hook_count = len(self.base._backbone._forward_hooks)
        self.base.fail = True
        with self.assertRaisesRegex(RuntimeError, 'synthetic base failure'):
            self.model(**self.x)
        self.assertEqual(len(self.base._backbone._forward_hooks), hook_count)
        self.base.fail = False
        self.model(**self.x)
        self.assertEqual(len(self.base._backbone._forward_hooks), hook_count)

    def test_reentrant_and_multiple_backbone_guard(self):
        self.model._forward_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'concurrent/reentrant'):
                self.model(**self.x)
        finally:
            self.model._forward_lock.release()
        self.base.twice = True
        with self.assertRaisesRegex(RuntimeError, 'more than once'):
            self.model(**self.x)
        self.assertEqual(len(self.base._backbone._forward_hooks), 0)
        self.base.twice = False
        self.model(**self.x)

    def test_forbidden_inputs_and_validation(self):
        signature = inspect.signature(self.model.forward)
        self.assertNotIn('goal_xy', signature.parameters)
        self.assertNotIn('status', signature.parameters)
        self.assertNotIn('state_target', signature.parameters)
        with self.assertRaises(TypeError):
            self.model(**self.x, state_target=torch.zeros(2,4))
        with self.assertRaisesRegex(ValueError, 'perception_status'):
            self.model(**self.x, perception_status=torch.ones(2,8))
        with self.assertRaisesRegex(ValueError, 'nominal'):
            self.model(**{**self.x, 'time_offsets':torch.tensor([[.1,1.],[.1,1.]])})
        with self.assertRaisesRegex(ValueError, 'perception_status'):
            self.model(**self.x, perception_status=torch.full((2,4),float('nan')))
        self.assertEqual(len(self.base._backbone._forward_hooks), 0)

    def test_grid_centers_and_visibility_independent_projection(self):
        head = self.model.perception
        self.assertTrue(torch.allclose(head.points[0,0],torch.tensor([-9.375,-31.333333,0.,1.]),atol=2e-6))
        self.assertTrue(torch.allclose(head.points[-1,1],torch.tensor([69.375,31.333333,1.,1.]),atol=2e-6))
        grid, visible = head.projection_grid(self.x['lidar2img'],self.x['image_hw'])
        raw = (self.x['lidar2img'][:,:,None,None] @ head.points[None,None,...,None]).squeeze(-1)
        u, v = raw[...,0]/raw[...,2], raw[...,1]/raw[...,2]
        expected = (raw[...,2]>.05)&(u>=0)&(u<64)&(v>=0)&(v<32)
        self.assertTrue(torch.equal(expected,visible))
        self.assertTrue(torch.isfinite(grid).all())

    def test_deepcopy_state_roundtrip_and_initial_arm_equality(self):
        other = copy.deepcopy(self.model)
        other.load_state_dict(self.model.state_dict(),strict=True)
        with torch.no_grad():
            a = self.model(**{**self.x,'history_images':self.x['images'][:,1:2].expand(-1,2,-1,-1,-1)})
            b = other(**self.x,perception_status=torch.ones(2,4))
        for key in ('trajectory','scores','candidate_xy','candidate_ids','aux_occ','aux_lane'):
            self.assertTrue(torch.equal(a[key],b[key]),key)
        # State aux intentionally sees real vs repeat history even at initialization.

    def test_actual_public_decoder_zero_init_parity(self):
        path = Path(os.environ.get('PUBLIC_MODEL_PATH', str(Path(__file__).parent/'source/sparsedrivev2_20260910/public_model.py')))
        if not path.is_file():
            try:
                from experiments.sparsedrivev2_20260910 import public_model as public
            except ImportError:
                self.skipTest('Pinned public source unavailable; set PUBLIC_MODEL_PATH')
        else:
            spec = importlib.util.spec_from_file_location('temporal_test_public',path)
            public = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(public)
        bank = {'path_vocab':torch.zeros(2,50,3),'vel_vocab':torch.ones(2,8),
                'traj_vocab':torch.zeros(2,2,8,3),'traj_mask':torch.ones(2,2,8)}
        bank['path_vocab'][:,:,0] = torch.arange(1,51)[None]/10.
        bank['path_vocab'][1,:,1] = .1
        for p in range(2):
            for v in range(2):
                bank['traj_vocab'][p,v,:,0] = torch.arange(1,9)*(v+1)*.15
                bank['traj_vocab'][p,v,:,1] = p*.1
        with mock.patch.object(public,'SparseBackbone',TinyBackbone):
            base = public.PublicSparseDriveV2(bank,backend='reference',mask_invalid_candidates=True).eval()
        model = TemporalPerceptionModel(base).eval()
        x = inputs(1)
        with torch.no_grad():
            before = base(**{k:x[k] for k in ('images','lidar2img','image_hw')},status=torch.zeros(1,8))
            after = model(**x,perception_status=torch.tensor([[12.,.1,.2,.3]]))
        for key in ('trajectory','candidate_xy','scores','candidate_ids','selected_candidate_id','imitation_scores','navsim_scores'):
            self.assertTrue(torch.equal(before[key],after[key]),key)
        for sa,sb in zip(before['coarse'],after['coarse']):
            for key in sa:
                self.assertTrue(torch.equal(sa[key],sb[key]),key)


if __name__ == '__main__':
    unittest.main(verbosity=2)
