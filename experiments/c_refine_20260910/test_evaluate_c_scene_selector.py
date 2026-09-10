"""CPU provenance/buffer/cache comparison tests; no large checkpoints or GPU."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
import shutil

import numpy as np
import torch
from torch import nn

import evaluate_c_scene_selector as e
from c_scene_selector import CandidateTokenCapture, SceneResidualSelector
from test_c_scene_selector import TinyFinalGoal, TinyPublic


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_counts_frozen_and_continued_and_reject_wrong_parent(self):
        cache=dict(counts=dict(path_filter=[128,20],velocity_filter=[64,64]),
            checkpoint_sha256=e.C_CHECKPOINT_SHA,bank_sha256='bank',source_sha256={'x':'y'},
            checkpoint={'checkpoint':'/old/C.pth'})
        frozen=dict(schema='c_scene_selector_frozen_v1',train_cache_manifest=cache,
            tune_cache_manifest=copy.deepcopy(cache),arguments={'mode':'real'})
        self.assertEqual(e.counts_and_mode(frozen)['original_c_checkpoint'],'/old/C.pth')
        continued=dict(schema='c_scene_continuation_v1',original_c_checkpoint_sha256=e.C_CHECKPOINT_SHA,
            original_c_checkpoint={'path':'/old/C.pth','sha256':e.C_CHECKPOINT_SHA},
            path_filter=[128,20],velocity_filter=[64,64],head_mode='real')
        self.assertEqual(e.counts_and_mode(continued),e.counts_and_mode(frozen))
        frozen['tune_cache_manifest']['checkpoint_sha256']='wrong'
        with self.assertRaises(ValueError):
            e.counts_and_mode(frozen)

    def test_model_c_strict_load_preserves_all_buffers(self):
        model=nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4))
        def digest(tensor):
            return tensor.detach().numpy().tobytes()
        ev=types.SimpleNamespace(tensor_sha=digest)
        state={key:value.clone() for key,value in model.state_dict().items()}
        state['0.weight']+=.1
        e.apply_continuation(model,state,ev)
        self.assertTrue(torch.equal(model[0].weight,state['0.weight']))
        bad={key:value.clone() for key,value in state.items()}
        bad['1.running_mean']+=1.
        with self.assertRaisesRegex(ValueError,'fixed original buffer'):
            e.apply_continuation(model,bad,ev)
        with self.assertRaises(RuntimeError):
            e.apply_continuation(model,{'bad_key':torch.zeros(1)},ev)

    def test_continuation_checks_parent_checkpoint_and_source_receipts(self):
        ev=types.SimpleNamespace(load_internal_payload=lambda p:torch.load(p,map_location='cpu',weights_only=True),
            canonical=lambda m:json.dumps(m,sort_keys=True,separators=(',',':')))
        cache=dict(counts=dict(path_filter=[128,20],velocity_filter=[64,64]),
            checkpoint_sha256=e.C_CHECKPOINT_SHA,bank_sha256='bank',source_sha256={'x':'y'},
            checkpoint={'checkpoint':'/old/C.pth'})
        common=dict(arguments={'steps':4000,'mode':'real'},bank_sha256='bank',
            source_sha256={'c_scene_selector.py':e.SCENE_SOURCE_SHA})
        with tempfile.TemporaryDirectory() as temp:
            def save(name,manifest,continued=False):
                root=Path(temp)/name;(root/'source').mkdir(parents=True)
                shutil.copyfile(Path(__file__).with_name('c_scene_selector.py'),root/'source'/'c_scene_selector.py')
                (root/'manifest.json').write_text(json.dumps(manifest))
                payload=dict(head={'weight':torch.tensor([1.])},manifest=manifest,step=4000)
                if continued:
                    payload['model_c']={}
                path=root/'last.pth';torch.save(payload,path);return path
            frozen={**common,'schema':'c_scene_selector_frozen_v1','train_cache_manifest':cache,'tune_cache_manifest':cache}
            parent=save('parent',frozen)
            cont={**common,'schema':'c_scene_continuation_v1',
                'original_c_checkpoint':'/old/C.pth','original_c_checkpoint_sha256':e.C_CHECKPOINT_SHA,
                'path_filter':[128,20],'velocity_filter':[64,64],'head_mode':'real',
                'frozen_head_checkpoint':dict(path=str(parent),sha256=e.sha(parent),step=4000)}
            child=save('child',cont,True)
            plan=e.inspect_scene_checkpoint(child,ev)
            self.assertEqual(plan['frozen_head_parent_verified']['sha256'],e.sha(parent))
            with parent.open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'parent identity'):
                e.inspect_scene_checkpoint(child,ev)

    def test_cache_comparison_checks_all_tokens_and_same_batch_head_scores(self):
        torch.manual_seed(1)
        original=TinyFinalGoal(TinyPublic()).eval()
        capture=CandidateTokenCapture(original)
        goal=torch.tensor([[5.,0.],[6.,1.]])
        base=capture(images=torch.randn(2,256),goal_xy=goal)
        head=SceneResidualSelector().eval()
        with torch.no_grad():
            nn.init.normal_(head.score_head[-1].weight,std=.1)
            out=head(base,goal)
        rows=np.asarray([10,20],np.int64)
        values=dict(rows=rows,candidate_ids=base['candidate_ids'].numpy(),
            candidate_valid=base['candidate_valid'].numpy(),scores=base['scores'].numpy(),
            token=base['candidate_tokens'].float().numpy(),goal_xy=goal.numpy())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);files={}
            for key,value in values.items():
                np.save(root/(key+'.npy'),value)
                files[key]=dict(path=key+'.npy',sha256=e.sha(root/(key+'.npy')),shape=list(value.shape),dtype=str(value.dtype))
            m=dict(status='completed',canary=False,split='tune',checkpoint_sha256=e.C_CHECKPOINT_SHA,
                bank_sha256='bank',counts=dict(path_filter=[128,20],velocity_filter=[64,64]),files=files)
            (root/'manifest.json').write_text(json.dumps(m))
            plan=types.SimpleNamespace(rows=rows,receipt={'bank_sha256':'bank'})
            scene={'manifest':{'schema':'c_scene_selector_frozen_v1'},'config':m['counts']}
            ref=e.CacheReference(root,plan,scene,None)
            bank=original._trajectory_head.traj_vocab.flatten(0,1)[:,:6,:2]
            ref.compare_batch({'row':torch.from_numpy(rows)},out,goal,head,bank,0)
            corrupt={**out,'candidate_tokens':out['candidate_tokens'].clone()}
            corrupt['candidate_tokens'][0,0,0]+=1.
            with self.assertRaisesRegex(ValueError,'token'):
                ref.compare_batch({'row':torch.from_numpy(rows)},corrupt,goal,head,bank,0)
            with (root/'token.npy').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                e.CacheReference(root,plan,scene,None)


if __name__=='__main__':
    unittest.main(verbosity=2)
