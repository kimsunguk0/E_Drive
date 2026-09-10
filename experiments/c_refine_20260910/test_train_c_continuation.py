"""CPU contract tests; no competition data, weights or GPU required."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from c_scene_selector import SceneResidualSelector
from train_c_continuation import (CSceneModel,C_SHA,SCENE_SHA,EXPECTED_COUNTS,
    checkpoint_payload,optimizer_groups,lr_factor,validate_head_manifest,configure_counts)
from test_c_scene_selector import TinyPublic,TinyFinalGoal,activate


class TinyCompatibleC(TinyFinalGoal):
    def __init__(self):
        public=TinyPublic()
        public._backbone=nn.Linear(256,256)
        public.path_filter=(128,20);public.velocity_filter=(64,10)
        super().__init__(public)
        self.relative_head=nn.Linear(1,1)

    def forward(self,images,lidar2img,image_hw,history_images,time_offsets,
                perception_status=None,goal_xy=None):
        return super().forward(images,perception_status,goal_xy)


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(93)
        self.c=TinyCompatibleC().eval()
        self.head=SceneResidualSelector(mode='real')
        self.model=CSceneModel(self.c,self.head).eval()
        self.inputs=dict(images=torch.randn(2,256),lidar2img=torch.eye(4)[None,None].expand(2,3,-1,-1),
            image_hw=torch.tensor([[256,512]]).expand(2,-1),history_images=torch.zeros(2,2,3,4,4),
            time_offsets=torch.tensor([[.1,.5]]).expand(2,-1),perception_status=torch.randn(2,4),
            goal_xy=torch.tensor([[5.,0.],[8.,1.]]))

    def test_zero_head_preserves_all_selected_outputs(self):
        old=self.c(**self.inputs);new=self.model(**self.inputs)
        for key in ('scores','candidate_ids','candidate_xy','trajectory','selected_candidate_id','aux_state'):
            self.assertTrue(torch.equal(old[key],new[key]),key)
        self.assertTrue(new['candidate_tokens'].requires_grad)

    def test_final_goal_cannot_modify_candidate_coordinates_or_tokens(self):
        activate(self.head)
        old=self.model(**self.inputs)
        new=self.model(**{**self.inputs,'goal_xy':self.inputs['goal_xy']+10})
        for key in ('candidate_ids','candidate_xy','candidate_tokens','aux_state'):
            self.assertTrue(torch.equal(old[key],new[key]),key)
        bank=self.model._trajectory_head.traj_vocab.flatten(0,1)
        self.assertTrue(torch.equal(new['trajectory'],bank[new['selected_candidate_id'],:6,:2]))

    def test_real_scene_residual_backpropagates_into_original_tokens(self):
        activate(self.head)
        images=self.inputs['images'].clone().requires_grad_(True)
        out=self.model(**{**self.inputs,'images':images})
        out['scene_score_residual'][out['candidate_valid']].sum().backward()
        self.assertGreater(images.grad.abs().sum().item(),0)
        self.assertGreater(self.c._trajectory_head.embedding.weight.grad.abs().sum().item(),0)

    def test_gt_and_direct_status_args_rejected(self):
        for key,value in (('gt_plan',torch.zeros(2,6,2)),('status',torch.ones(2,8))):
            with self.assertRaises(TypeError):self.model(**{**self.inputs,key:value})
        with self.assertRaises(RuntimeError):self.model(**{**self.inputs,'perception_status':None})

    def test_counts_only_and_original_keys_remain(self):
        before={k:v.clone() for k,v in self.c.state_dict().items()}
        configure_counts(self.c)
        self.assertEqual(self.c.base.base.velocity_filter,(64,64))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in self.c.state_dict().items()))
        groups=optimizer_groups(self.model,[5e-6,5e-5,2e-4,5e-4])
        opt=torch.optim.AdamW(groups)
        self.assertEqual(len(opt.state),0)
        payload=checkpoint_payload(self.model,opt,1,0,16,{}, {})
        self.assertEqual(set(payload['model_c']),set(before))
        self.assertFalse(any(k.startswith('capture.') for k in payload['model_c']))
        restored=TinyCompatibleC();restored.load_state_dict(payload['model_c'],strict=True)
        restored_head=SceneResidualSelector();restored_head.load_state_dict(payload['head'],strict=True)

    def test_groups_cover_each_parameter_once_with_expected_lrs(self):
        lrs=[5e-6,5e-5,2e-4,5e-4]
        groups=optimizer_groups(self.model,lrs)
        self.assertEqual([g['lr'] for g in groups],lrs)
        flat=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(flat),len(set(flat)))
        self.assertEqual(set(flat),{id(p) for p in self.model.parameters()})
        self.assertEqual({id(p) for p in groups[-1]['params']},{id(p) for p in self.head.parameters()})

    def test_schedule_warmup_and_zero_terminal(self):
        self.assertAlmostEqual(lr_factor(1,4000,100),.01)
        self.assertEqual(lr_factor(100,4000,100),1.)
        self.assertEqual(lr_factor(4000,4000,100),0.)
        self.assertGreater(lr_factor(3999,4000,100),0.)

    def manifest_fixture(self):
        plan=SimpleNamespace(receipt={'bank_sha256':'bank'},manifest={'train':{'rows_sha256':'train'},'validation':{'rows_sha256':'tune'}})
        def cache(split):
            return dict(status='completed',checkpoint_sha256=C_SHA,bank_sha256='bank',counts=copy.deepcopy(EXPECTED_COUNTS),
                rows_sha256=split,source_sha256={'c_scene_selector.py':SCENE_SHA},batch_size=8,sessions=[split])
        m=dict(schema='c_scene_selector_frozen_v1',arguments={'mode':'real','steps':4000},
               source_sha256={'c_scene_selector.py':SCENE_SHA},bank_sha256='bank',base_frozen=True,raw_status_input=False,
               train_cache_manifest=cache('train'),tune_cache_manifest=cache('tune'))
        return m,plan

    def test_manifest_accepts_explicit_trained_step_and_rejects_wrong_inputs(self):
        m,plan=self.manifest_fixture();validate_head_manifest(m,plan,500)
        for field,value in (('bank_sha256','wrong'),('raw_status_input',True)):
            bad=copy.deepcopy(m);bad[field]=value
            with self.assertRaises(RuntimeError):validate_head_manifest(bad,plan,500)
        for key,value in (('rows_sha256','wrong'),('checkpoint_sha256','wrong'),('counts',dict(EXPECTED_COUNTS,velocity_filter=[64,32]))):
            bad=copy.deepcopy(m);bad['train_cache_manifest'][key]=value
            with self.assertRaises(RuntimeError):validate_head_manifest(bad,plan,500)
        control=copy.deepcopy(m);control['arguments']['mode']='zero'
        validate_head_manifest(control,plan,500)
        bad=copy.deepcopy(m);bad['arguments']['mode']='unsupported'
        with self.assertRaises(RuntimeError):validate_head_manifest(bad,plan,500)


if __name__=='__main__':unittest.main(verbosity=2)
