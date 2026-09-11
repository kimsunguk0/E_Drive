"""Frozen temporal C velocity-pruning diagnosis; GT is outer-metric only.

This changes an in-process integer search policy only. It never trains or changes
checkpoint tensors, fixed-bank coordinates, direct-zero status or final-only goal.
Accuracy uses original tune1998, B8/BF16. A changed count is an inference-policy
experiment, not a claim of parity with the original 20x10 candidate policy.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

HELPERS = {
    'evaluate_temporal_checkpoint.py': '77de9c59cad973bbec06baa63322f894a6fff8bb82a9629aecfd5724470144f0',
    'diagnose_temporal_shortlist.py': 'df8f69320773e11dd37a939214371b57ce20a046c3685136187e6b3a880a4d9b',
}
FILTERS = ((64,10),(64,32),(64,64),(128,32),(128,64))
WEIGHTS = np.asarray([11,11,5,5,2,2],np.float64)/36


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''): h.update(block)
    return h.hexdigest()


def helpers(directory):
    result=[]
    for index,(name,expected) in enumerate(HELPERS.items()):
        path=Path(directory)/name
        require(sha(path)==expected,f'Pinned helper changed: {path}')
        spec=importlib.util.spec_from_file_location(f'_c_velocity_helper_{index}',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result.append(module)
    return result


def tensor_stamp(model):
    return {name:(value.data_ptr(),value._version,tuple(value.shape),str(value.dtype))
            for name,value in model.state_dict(keep_vars=True).items()}


@contextmanager
def configured_velocity_filter(model, velocity_filter):
    """Set only public.velocity_filter, restore on exit, reject tensor mutation."""
    requested=tuple(velocity_filter)
    require(requested in FILTERS,'Unapproved velocity count policy')
    public=model.base.base
    require(tuple(public.path_filter)==(128,20),'Path policy changed')
    original=tuple(public.velocity_filter)
    require(original==(64,10),'Model did not start with frozen 64 -> 10 policy')
    before=tensor_stamp(model)
    public.velocity_filter=requested
    try:
        yield public
    finally:
        public.velocity_filter=original
        require(tensor_stamp(model)==before,'Parameters/buffers changed storage, version, shape or dtype')


def stage_axes(out, bank_shape, velocity_filter, oracle):
    """Validate actual retained IDs, including stage1 -> final containment."""
    p_total,v_total=bank_shape
    p,v=oracle.cartesian_axes(out['candidate_ids'],p_total,v_total,20,velocity_filter[1])
    get=lambda x:x.detach().cpu().numpy().astype(np.int64)
    require(np.array_equal(p,get(out['path_ids'])),'Cartesian paths differ from returned path IDs')
    require(np.array_equal(v,get(out['velocity_ids'])),'Cartesian velocities differ from returned velocity IDs')
    require(len(out['coarse'])==2,'Expected exactly two pruning stages')
    initial=get(out['coarse'][0]['velocity_ids'])
    stage1=get(out['coarse'][1]['velocity_ids'])
    require(initial.shape==(len(p),v_total) and np.array_equal(initial,np.broadcast_to(np.arange(v_total),initial.shape)),
            'Stage0 did not score the full velocity bank')
    require(stage1.shape==(len(p),velocity_filter[0]),'Actual stage1 count differs from configured count')
    for before,after in zip(stage1,v):
        require(len(np.unique(before))==len(before) and np.all((before>=0)&(before<v_total)),
                'Stage1 velocity IDs are invalid or duplicated')
        require(np.isin(after,before).all(),'Final velocities were not retained at stage1')
    return p,stage1,v


@torch.inference_mode()
def stage_oracles(bank, mask, paths, stage1, final, target, oracle, chunk):
    all_v=np.broadcast_to(np.arange(bank.shape[1]),(len(target),bank.shape[1])).copy()
    result={}
    for name,axis in (('final',final),('stage1',stage1),('all_v',all_v)):
        values=oracle.oracle_for_axes(bank,mask,paths,axis,target,chunk)
        for field,value in zip(('oracle','oracle_bank_id','valid_count'),values):
            result[f'{name}_{field}']=value.cpu().numpy()
    require(np.isfinite(result['final_oracle']).all(),'Final candidates have no valid row')
    require(np.all(result['all_v_oracle']<=result['stage1_oracle']+1e-12) and
            np.all(result['stage1_oracle']<=result['final_oracle']+1e-12),'Nested oracle ordering failed')
    require(np.all(result['all_v_valid_count']>=result['stage1_valid_count']) and
            np.all(result['stage1_valid_count']>=result['final_valid_count']),'Nested validity count failed')
    result['stage1_prune_loss']=result['stage1_oracle']-result['all_v_oracle']
    result['final_prune_loss']=result['final_oracle']-result['stage1_oracle']
    return result


@torch.inference_mode()
def profile_model(model, inputs, iterations):
    report={'scope':'B200 CUDA model forward only; GPU-resident input; no dataloader, raw decode, H2D, GT, hooks, output copy',
            'not_RTX4090_latency':True,'iterations':iterations,'warmup':5}
    for batch in (8,1):
        selected={key:value[:batch] for key,value in inputs.items()}
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for _ in range(5): model(**selected)
            torch.cuda.synchronize()
            times=[]
            for _ in range(iterations):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();out=model(**selected);end.record();end.synchronize()
                times.append(start.elapsed_time(end))
        report[f'batch_{batch}']={'p50_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
            'mean_ms':float(np.mean(times)),'max_allocated_bytes':torch.cuda.max_memory_allocated(),
            'max_reserved_bytes':torch.cuda.max_memory_reserved(),'candidate_count':int(out['candidate_ids'].shape[1])}
    return report


@torch.inference_mode()
def evaluate(model,dataset,runtime,helper,oracle,args):
    loader=torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=False,num_workers=args.workers,pin_memory=True)
    chunks={};first_inputs=None
    with helper.route_monitor(model,True) as route_counts:
        for batch_index,batch in enumerate(loader):
            inputs=helper.input_tensors(batch,runtime,True,'cuda:0')
            if first_inputs is None: first_inputs=inputs
            with torch.autocast('cuda',dtype=torch.bfloat16): out=model(**inputs)
            helper.verify_rows(model,out)
            h=model._trajectory_head
            require(h.traj_vocab.shape[:2]==(1024,1024),'Expected pinned 1024x1024 bank')
            p,v1,v2=stage_axes(out,h.traj_vocab.shape[:2],args.velocity_filter,oracle)
            # Labels enter only here, after all candidate IDs/scores/coordinates exist.
            target=batch['gt_plan'].to('cuda:0',non_blocking=True)
            values=stage_oracles(h.traj_vocab,h.traj_mask,p,v1,v2,target,oracle,args.chunk)
            pred=out['trajectory'].float().cpu().numpy()
            gt=batch['gt_plan'].float().numpy().astype(np.float64)
            point=np.linalg.norm(pred.astype(np.float64)-gt,axis=-1)
            scores=out['scores'].float().cpu().numpy()
            valid=out['candidate_valid'].cpu().numpy().astype(bool)
            base_index=out['base_scores'].masked_fill(~out['candidate_valid'].bool(),-torch.inf).argmax(-1)
            base_pred=out['candidate_xy'][torch.arange(len(target),device=target.device),base_index].float().cpu().numpy()
            with torch.autocast('cuda',enabled=False):
                direct=(torch.linalg.vector_norm(out['candidate_xy'].double()-target[:,None].double(),dim=-1)
                        *torch.as_tensor(WEIGHTS,device=target.device)).sum(-1)
                direct=direct.masked_fill(~out['candidate_valid'].bool(),torch.inf).amin(-1).cpu().numpy()
            require(np.array_equal(direct,values['final_oracle']),'Actual candidate oracle differs from Cartesian bank oracle')
            values.update(rows=batch['row'].numpy(),pred=pred,candidate_id=out['selected_candidate_id'].cpu().numpy(),
                point_l2=point,d3=(point*WEIGHTS).sum(-1),session=np.asarray(batch['session']),scenario=np.asarray(batch['scenario']),
                retained_path_ids=p,stage1_velocity_ids=v1,final_velocity_ids=v2,
                candidate_ids=out['candidate_ids'].cpu().numpy(),candidate_scores=scores,candidate_valid=valid,
                base_pred=base_pred,base_d3=(np.linalg.norm(base_pred.astype(np.float64)-gt,axis=-1)*WEIGHTS).sum(-1))
            for key,value in values.items(): chunks.setdefault(key,[]).append(value)
            if (batch_index+1)%25==0:
                print(json.dumps({'rows':sum(map(len,chunks['rows'])),'filter':args.velocity_filter,
                    'partial_d3':float(np.concatenate(chunks['d3']).mean()),
                    'partial_oracles':{k:float(np.concatenate(chunks[k+'_oracle']).mean()) for k in ('final','stage1','all_v')}}),flush=True)
    arrays={key:np.concatenate(value) for key,value in chunks.items()}
    require(np.array_equal(arrays['rows'],dataset.rows),'Tune row order changed')
    profile=profile_model(model,first_inputs,args.profile_iterations) if args.profile_iterations else None
    return arrays,route_counts,profile


def summarize(arrays,oracle):
    keys=('d3','base_d3','final_oracle','stage1_oracle','all_v_oracle','stage1_prune_loss','final_prune_loss')
    return {**{key:oracle.distribution(arrays[key]) for key in keys},
        'session':{str(s):{key:float(arrays[key][arrays['session']==s].mean()) for key in keys}
                   for s in np.unique(arrays['session'])},
        'oracle_is_privileged_GT_containment_not_selector_performance':True,
        'same_retained_final_P20_for_all_three_oracles':True}


def self_test(helper_dir):
    import unittest
    from types import SimpleNamespace
    _,oracle=helpers(helper_dir)
    class Tests(unittest.TestCase):
        def test_nested_masked_oracles(self):
            bank=torch.zeros(2,4,8,3)
            bank[:,1,:,0]=1;bank[:,2,:,0]=2;bank[:,3,:,0]=3
            mask=torch.ones(2,4,8);mask[:,0,0]=0
            values=stage_oracles(bank,mask,np.array([[0,1]]),np.array([[2,3]]),np.array([[3]]),torch.zeros(1,6,2),oracle,1)
            self.assertAlmostEqual(values['final_oracle'][0],3.)
            self.assertAlmostEqual(values['stage1_oracle'][0],2.)
            self.assertAlmostEqual(values['all_v_oracle'][0],1.)
            self.assertEqual(values['all_v_valid_count'][0],6)
        def test_missing_stage_retention_rejected(self):
            p=torch.arange(20)[None];v=torch.tensor([[0,1]])
            out={'candidate_ids':(p[:,:,None]*4+v[:,None]).flatten(1,2),'path_ids':p,'velocity_ids':v,
                 'coarse':[{'velocity_ids':torch.arange(4)[None]},{'velocity_ids':torch.tensor([[1,2,3]])}]}
            with self.assertRaises(RuntimeError):stage_axes(out,(20,4),(3,2),oracle)
            out['coarse'][1]['velocity_ids']=torch.tensor([[2,0,1]])
            axes=stage_axes(out,(20,4),(3,2),oracle)
            self.assertEqual(axes[1].tolist(),[[2,0,1]])
        def test_configuration_restore_and_tensor_identity(self):
            public=torch.nn.Linear(2,2);public.path_filter=(128,20);public.velocity_filter=(64,10)
            model=torch.nn.Module();model.base=torch.nn.Module();model.base.base=public
            before={k:v.clone() for k,v in model.state_dict().items()}
            with configured_velocity_filter(model,(128,64)):
                self.assertEqual(public.velocity_filter,(128,64))
            self.assertEqual(public.velocity_filter,(64,10))
            self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
            with self.assertRaises(RuntimeError):
                with configured_velocity_filter(model,(64,32)):
                    with torch.no_grad():public.weight.add_(1)
            self.assertEqual(public.velocity_filter,(64,10))
        def test_configuration_rejects_unplanned_counts(self):
            public=SimpleNamespace(path_filter=(128,20),velocity_filter=(64,10))
            model=SimpleNamespace(base=SimpleNamespace(base=public))
            with self.assertRaises(RuntimeError):
                with configured_velocity_filter(model,(32,10)):pass
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    require(result.wasSuccessful(),'CPU tests failed')


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--helper-dir',required=True);p.add_argument('--checkpoint');p.add_argument('--output')
    p.add_argument('--gpu',type=int,choices=(0,));p.add_argument('--worktree');p.add_argument('--base')
    p.add_argument('--velocity-filter',type=int,nargs=2,default=(64,10))
    p.add_argument('--baseline-diagnostic',help='Pinned completed baseline diagnostic NPZ for actual path comparison')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--chunk',type=int,default=2048)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--profile-iterations',type=int,default=30)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args();a.velocity_filter=tuple(a.velocity_filter)
    if a.self_test:self_test(a.helper_dir);return
    require(a.checkpoint and a.output,'Checkpoint/output required')
    require(a.velocity_filter in FILTERS and a.workers>=0 and a.chunk>0 and 0<=a.limit<=1998 and a.profile_iterations>=0,'Invalid arguments')
    helper,oracle=helpers(a.helper_dir);gpu=helper.check_gpu(a.gpu)
    plan=helper.inspect_checkpoint(a.checkpoint,worktree=a.worktree,base=a.base,allow_canary=bool(a.limit))
    # Same decomposition for the B arm, whose perception carries no status.
    require(plan.manifest['arguments']['history_mode']=='real','Expected a real-history temporal arm')
    arm_common_status=plan.manifest['arguments']['common_status']
    require(isinstance(arm_common_status,bool),'common_status must be an explicit bool')
    print(json.dumps({'arm_common_status':arm_common_status}),flush=True)
    require(plan.manifest['arguments']['eval_batch']==8 and str(torch.__version__)==plan.manifest['torch'],'B8/training runtime contract changed')
    reference=plan.checkpoint.parent/f"eval_{plan.payload['step']:06d}.npz"
    require(reference.is_file(),'Terminal training reference missing')
    output=Path(a.output).resolve();require(not output.exists(),'Output already exists');output.mkdir(parents=True)
    shutil.copy2(__file__,output/Path(__file__).name)
    for name in HELPERS:shutil.copy2(Path(a.helper_dir)/name,output/name)
    shutil.copytree(plan.source,output/'frozen_source')
    receipt={**plan.receipt,'script_sha256':sha(__file__),'helper_sha256':HELPERS,'gpu':gpu,'batch_size':8,
        'precision':'bf16_base_fp32_relative_head','oracle_arithmetic':'FP64 after complete model forward',
        'velocity_filter':a.velocity_filter,'path_filter':[128,20],'arguments':vars(a),'training_reference_sha256':sha(reference),
        'count_policy_only':True,'changed_count_is_inference_distribution_shift':a.velocity_filter!=(64,10),
        'gt_to_model':False,'coordinate_mutation':False,'all_P_all_V_oracle':False}
    started=time.time();torch.set_num_threads(4)
    try:
        with helper.isolated_runtime(plan.source) as runtime:
            model=helper.strict_load(plan,runtime).cuda().eval();receipt.update(plan.receipt)
            dataset,provenance=helper.build_dataset(plan,runtime,a.limit)
            with configured_velocity_filter(model,a.velocity_filter):
                arrays,counts,profile=evaluate(model,dataset,runtime,helper,oracle,a)
        np.savez_compressed(output/'diagnostics.npz',**arrays)
        parity=oracle.reference_parity(arrays,reference)
        result={'status':'completed','n':len(arrays['rows']),'velocity_filter':a.velocity_filter,
            'summary':summarize(arrays,oracle),'profile':profile,'original_B8_reference':parity,
            'original_B8_byte_parity_required':a.velocity_filter==(64,10),'elapsed_seconds':time.time()-started}
        if a.baseline_diagnostic:
            with np.load(a.baseline_diagnostic,allow_pickle=False) as old:
                require(np.array_equal(arrays['rows'],old['rows']),'Baseline diagnostic rows differ')
                result['baseline_comparison']={'path':str(Path(a.baseline_diagnostic).resolve()),'sha256':sha(a.baseline_diagnostic),
                    'same_retained_path_ids_bytes':oracle.byte_equal(arrays['retained_path_ids'],old['retained_path_ids']),
                    'd3_delta':float((arrays['d3']-old['d3']).mean()),
                    'oracle_delta':float((arrays['final_oracle']-old['final_oracle']).mean()),
                    'changed_selected_id_count':int(np.count_nonzero(arrays['candidate_id']!=old['candidate_id']))}
        passed=a.velocity_filter!=(64,10) or parity['passed']
        if not passed:result['status']='parity_failed'
        receipt.update(dataset_provenance=provenance,route_checks=counts,tensor_identity_and_version_unchanged=True,
            diagnostics_sha256=sha(output/'diagnostics.npz'),elapsed_seconds=time.time()-started)
        helper.json_write(output/'receipt.json',receipt);helper.json_write(output/'result.json',result)
        print(json.dumps(result,allow_nan=False),flush=True)
        require(passed,'Baseline B8 differs from original; actual outputs preserved without adjustments')
    except BaseException:
        receipt.update(status='failed',traceback=traceback.format_exc(),elapsed_seconds=time.time()-started)
        helper.json_write(output/'failure.json',receipt)
        raise


if __name__=='__main__':main()
