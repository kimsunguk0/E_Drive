"""GT-only shortlist containment diagnosis of terminal temporal C, B8 fixed.

The model returns unchanged completed bank candidates before any GT is used.
Only three oracles are computed: its P20 x V10, P20 x all V, and all P x V10.
These privileged candidate-coverage metrics are NOT deployed selector results.
No all-P x all-V oracle or inference-coordinate modification is performed.

CPU: python diagnose_temporal_shortlist.py --self-test
GPU, only after assignment/release: CUDA_VISIBLE_DEVICES=4 python ... --gpu 4
 --checkpoint work_dirs/.../temporal_c_common_s0_v1/last.pth --output NEW_DIR
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

HELPER_SHA='9007a4517adaf663d402535bdd56dd5d8340ad675db0f3a3ef44d8d867529242'
BATCH_SIZE=8
WEIGHTS=(11/36,11/36,5/36,5/36,2/36,2/36)


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def load_helper():
    import importlib.util
    path=Path(__file__).with_name('evaluate_temporal_checkpoint.py')
    require(sha(path)==HELPER_SHA,'Strict evaluator helper revision changed')
    spec=importlib.util.spec_from_file_location('_shortlist_verified_evaluator',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cartesian_axes(candidate_ids,bank_paths,bank_velocities,expected_paths=None,expected_velocities=None):
    """Recover exact Cartesian axes solely from returned completed row IDs.

    Axis ordering follows first occurrence in the output, but the validator does
    not assume flattened product order. Duplicates or missing combinations fail.
    """
    ids=candidate_ids.detach().cpu().numpy() if isinstance(candidate_ids,torch.Tensor) else np.asarray(candidate_ids)
    require(ids.ndim==2 and ids.dtype.kind in 'iu','Candidate IDs must be integer [B,K]')
    require(ids.shape[0]>0 and ids.shape[1]>0,'Empty candidate set')
    require(np.all((ids>=0)&(ids<bank_paths*bank_velocities)),'Candidate ID outside full bank')
    path_rows=[]; velocity_rows=[]
    for row in ids:
        require(len(np.unique(row))==len(row),'Duplicate completed candidate IDs')
        p_value,p_first=np.unique(row//bank_velocities,return_index=True)
        v_value,v_first=np.unique(row%bank_velocities,return_index=True)
        paths=p_value[np.argsort(p_first)]
        velocities=v_value[np.argsort(v_first)]
        product=(paths[:,None]*bank_velocities+velocities[None]).reshape(-1)
        require(len(product)==len(row) and np.array_equal(np.sort(product),np.sort(row)),
                'Returned candidates are not the complete Cartesian product')
        if expected_paths is not None:
            require(len(paths)==expected_paths,'Unexpected retained path count')
        if expected_velocities is not None:
            require(len(velocities)==expected_velocities,'Unexpected retained velocity count')
        path_rows.append(paths.astype(np.int64)); velocity_rows.append(velocities.astype(np.int64))
    require(len({len(x) for x in path_rows})==1 and len({len(x) for x in velocity_rows})==1,
            'Ragged retained axes within batch')
    return np.stack(path_rows),np.stack(velocity_rows)


@torch.inference_mode()
def oracle_for_axes(bank,mask,paths,velocities,gt,chunk=2048):
    """Outer-metric FP64 oracle over exact stored coordinates and first6 mask.

    GT never enters any model or scorer. Ties in oracle IDs choose the smallest
    global bank row ID, independently of chunk size and output-axis ordering.
    """
    require(bank.ndim==4 and bank.shape[-2]>=6 and bank.shape[-1]>=2,'Malformed trajectory bank')
    p_total,v_total=bank.shape[:2]
    require(mask.shape[:2]==(p_total,v_total) and mask.ndim==3 and mask.shape[-1]>=6,'Malformed bank validity mask')
    require(gt.ndim==3 and gt.shape[-2:]==(6,2) and bool(torch.isfinite(gt).all()),'GT must be finite [B,6,2]')
    require(isinstance(chunk,int) and chunk>0,'Oracle chunk must be positive')
    device=bank.device
    paths=torch.as_tensor(paths,device=device,dtype=torch.long)
    velocities=torch.as_tensor(velocities,device=device,dtype=torch.long)
    b=len(gt)
    require(paths.ndim==velocities.ndim==2 and len(paths)==len(velocities)==b,'Axis batch mismatch')
    require(paths.shape[1]>0 and velocities.shape[1]>0,'Empty expanded axis')
    require(bool(((paths>=0)&(paths<p_total)).all()) and bool(((velocities>=0)&(velocities<v_total)).all()),
            'Expanded axis outside bank')
    ids=(paths[:,:,None]*v_total+velocities[:,None,:]).flatten(1,2)
    flattened=bank.flatten(0,1)
    complete=mask[...,:6].bool().all(-1).flatten()
    best=torch.full((b,),torch.inf,device=device,dtype=torch.float64)
    sentinel=p_total*v_total
    best_id=torch.full((b,),sentinel,device=device,dtype=torch.long)
    valid_count=torch.zeros(b,device=device,dtype=torch.long)
    with torch.autocast(device_type=device.type,enabled=False):
        target=gt.detach().to(device=device,dtype=torch.float64)
        weights=torch.tensor(WEIGHTS,device=device,dtype=torch.float64)
        for start in range(0,ids.shape[1],chunk):
            index=ids[:,start:start+chunk]
            valid=complete[index]
            coordinates=flattened[index,:,:2][...,:6,:].to(torch.float64)
            costs=(torch.linalg.vector_norm(coordinates-target[:,None],dim=-1)*weights).sum(-1)
            costs=costs.masked_fill(~valid,torch.inf)
            value=costs.amin(-1)
            tied=(costs==value[:,None])&valid
            row_id=index.masked_fill(~tied,sentinel).amin(-1)
            replace=(value<best)|((value==best)&(row_id<best_id))
            best=torch.where(replace,value,best)
            best_id=torch.where(replace,row_id,best_id)
            valid_count+=valid.sum(-1)
    best_id=best_id.masked_fill(valid_count==0,-1)
    return best,best_id,valid_count


@torch.inference_mode()
def three_oracles(bank,mask,candidate_ids,gt,chunk=2048,expected_paths=None,expected_velocities=None):
    p_total,v_total=bank.shape[:2]
    paths,velocities=cartesian_axes(candidate_ids,p_total,v_total,expected_paths,expected_velocities)
    p=torch.as_tensor(paths,device=bank.device,dtype=torch.long)
    v=torch.as_tensor(velocities,device=bank.device,dtype=torch.long)
    all_p=torch.arange(p_total,device=bank.device)[None].expand(len(gt),-1)
    all_v=torch.arange(v_total,device=bank.device)[None].expand(len(gt),-1)
    current=oracle_for_axes(bank,mask,p,v,gt,chunk)
    expand_v=oracle_for_axes(bank,mask,p,all_v,gt,chunk)
    expand_p=oracle_for_axes(bank,mask,all_p,v,gt,chunk)
    require(bool(torch.isfinite(current[0]).all()),'Current completed shortlist has no valid candidate')
    tolerance=1e-12
    require(bool((expand_v[0]<=current[0]+tolerance).all()),'All-V oracle exceeds nested current oracle')
    require(bool((expand_p[0]<=current[0]+tolerance).all()),'All-P oracle exceeds nested current oracle')
    require(bool((expand_v[2]>=current[2]).all()) and bool((expand_p[2]>=current[2]).all()),
            'Expanded valid set is not a superset')
    result={'retained_path_ids':paths,'retained_velocity_ids':velocities}
    for name,values in (('current',current),('all_v',expand_v),('all_p',expand_p)):
        for field,value in zip(('oracle','oracle_bank_id','valid_count'),values):
            result[f'{name}_{field}']=value.cpu().numpy()
    result['velocity_expansion_gain']=result['current_oracle']-result['all_v_oracle']
    result['path_expansion_gain']=result['current_oracle']-result['all_p_oracle']
    return result


def distribution(value):
    value=np.asarray(value,np.float64)
    require(value.ndim==1 and np.isfinite(value).all(),'Invalid metric distribution')
    quantiles=(0,10,25,50,75,90,95,99,100)
    return {'mean':float(value.mean()),'std':float(value.std()),
            'percentiles':{str(q):float(np.percentile(value,q)) for q in quantiles}}


def byte_equal(left,right):
    return left.dtype==right.dtype and left.shape==right.shape and left.tobytes()==right.tobytes()


def reference_parity(arrays,path):
    with np.load(path,allow_pickle=False) as source:
        n=len(arrays['rows'])
        report={'path':str(Path(path).resolve()),'sha256':sha(path),'keys':{},'passed':True}
        for key in ('rows','pred','candidate_id'):
            require(key in source,f'Training reference missing {key}')
            actual,expected=arrays[key],source[key][:n]
            same=byte_equal(actual,expected)
            report['keys'][key]={'byte_equal':same,'actual_dtype':str(actual.dtype),
                'expected_dtype':str(expected.dtype),'actual_shape':list(actual.shape),'expected_shape':list(expected.shape)}
            report['passed'] &= same
        if arrays['candidate_id'].shape==source['candidate_id'][:n].shape:
            report['changed_candidate_id_count']=int(np.count_nonzero(arrays['candidate_id']!=source['candidate_id'][:n]))
        if arrays['pred'].shape==source['pred'][:n].shape:
            report['max_absolute_xy_delta']=float(np.max(np.abs(arrays['pred'].astype(np.float64)-source['pred'][:n])))
    return report


def summarize(arrays):
    keys=('current_oracle','all_v_oracle','all_p_oracle','velocity_expansion_gain','path_expansion_gain','d3')
    summary={key:distribution(arrays[key]) for key in keys}
    summary['session']={s:{key:float(arrays[key][arrays['session']==s].mean()) for key in keys}
                        for s in np.unique(arrays['session'])}
    for axis in ('velocity','path'):
        value=arrays[axis+'_expansion_gain']
        summary[axis+'_expansion_fraction_gain_gt_001']=float(np.mean(value>.01))
        summary[axis+'_expansion_fraction_gain_gt_005']=float(np.mean(value>.05))
    summary['interpretation']='Expanded oracles use GT to choose stored rows; not learned selector accuracy or an additive decomposition.'
    summary['paired_set_note']='All-V keeps original image-selected paths; All-P keeps original image-selected velocities. Neither uses full all-P x all-V.'
    return summary


@torch.inference_mode()
def run_diagnostic(model,dataset,runtime,helper,chunk,workers):
    loader=torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=workers,pin_memory=True)
    chunks={}
    with helper.route_monitor(model,True) as route_counts:
        for batch_index,batch in enumerate(loader):
            # Explicit input whitelist before model forward; gt_plan remains outside.
            inputs=helper.input_tensors(batch,runtime,True,'cuda:0')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model(**inputs)
            helper.verify_rows(model,out)
            # Labels become metric operands only after the complete frozen forward.
            target=batch['gt_plan'].to('cuda:0',non_blocking=True)
            head=model._trajectory_head
            require(head.traj_vocab.shape[:2]==(1024,1024),'This C diagnostic expects the pinned 1024x1024 bank')
            values=three_oracles(head.traj_vocab,head.traj_mask,out['candidate_ids'],target,chunk,20,10)
            pred=out['trajectory'].float().cpu().numpy()
            gt=batch['gt_plan'].float().numpy()
            delta=pred.astype(np.float64)-gt.astype(np.float64)
            values.update(rows=batch['row'].numpy(),pred=pred,candidate_id=out['selected_candidate_id'].cpu().numpy(),
                d3=(np.linalg.norm(delta,axis=-1)*np.asarray(WEIGHTS)).sum(-1),
                session=np.asarray(batch['session']),scenario=np.asarray(batch['scenario']))
            # Independent direct shortlist calculation checks Cartesian reconstruction.
            with torch.autocast('cuda',enabled=False):
                direct=(torch.linalg.vector_norm(out['candidate_xy'].double()-target[:,None].double(),dim=-1)
                        *torch.tensor(WEIGHTS,device=target.device,dtype=torch.float64)).sum(-1)
                direct=direct.masked_fill(~out['candidate_valid'].bool(),torch.inf).amin(-1).cpu().numpy()
            require(np.array_equal(direct,values['current_oracle']),'Reconstructed/current-output oracle mismatch')
            for key,value in values.items():
                chunks.setdefault(key,[]).append(value)
            if (batch_index+1)%25==0:
                print(json.dumps({'diagnosed_rows':sum(len(x) for x in chunks['rows'])}),flush=True)
    arrays={key:np.concatenate(value) for key,value in chunks.items()}
    require(np.array_equal(arrays['rows'],dataset.rows),'Diagnostic row order changed')
    return arrays,route_counts


def self_test():
    import tempfile
    import unittest
    class OracleTests(unittest.TestCase):
        def fixture(self):
            bank=torch.zeros(3,4,8,3)
            for p in range(3):
                for v in range(4):
                    bank[p,v,:,0]=v
                    bank[p,v,:,1]=p
            return bank,torch.ones(3,4,8),torch.zeros(1,6,2)
        def test_cartesian_from_permuted_ids(self):
            p,v=cartesian_axes(np.asarray([[10,1,9,2]]),3,4,2,2)
            self.assertEqual(p.tolist(),[[2,0]])
            self.assertEqual(v.tolist(),[[2,1]])
        def test_missing_duplicate_outside_rejected(self):
            for ids in ([[0,1,4]],[[0,0]],[[12]],[[0.,1.]]):
                with self.assertRaises(RuntimeError): cartesian_axes(np.asarray(ids),3,4)
        def test_velocity_and_path_expansion_isolated(self):
            bank,mask,gt=self.fixture()
            result=three_oracles(bank,mask,torch.tensor([[10]]),gt,chunk=2)
            self.assertAlmostEqual(result['current_oracle'][0],np.sqrt(8),places=12)
            self.assertAlmostEqual(result['all_v_oracle'][0],2.,places=12)
            self.assertAlmostEqual(result['all_p_oracle'][0],2.,places=12)
            self.assertEqual(result['all_v_oracle_bank_id'].tolist(),[8])
            self.assertEqual(result['all_p_oracle_bank_id'].tolist(),[2])
        def test_first_six_mask_and_ignored_later_mask(self):
            bank,mask,gt=self.fixture()
            mask[2,0,0]=0
            mask[2,1,6:]=0
            result=three_oracles(bank,mask,torch.tensor([[10]]),gt,chunk=1)
            self.assertAlmostEqual(result['all_v_oracle'][0],np.sqrt(5),places=12)
            self.assertEqual(result['all_v_valid_count'].tolist(),[3])
            self.assertEqual(result['all_v_oracle_bank_id'].tolist(),[9])
        def test_chunk_ties_stable_and_no_mutation(self):
            bank,mask,gt=self.fixture(); bank.zero_()
            before=bank.clone(); p=torch.tensor([[2,0]]); v=torch.tensor([[3,1]])
            one=oracle_for_axes(bank,mask,p,v,gt,1)
            all_at_once=oracle_for_axes(bank,mask,p,v,gt,100)
            for a,b in zip(one,all_at_once): self.assertTrue(torch.equal(a,b))
            self.assertEqual(one[1].tolist(),[1])
            self.assertTrue(torch.equal(bank,before))
        def test_no_valid_rows_and_nested_requirement(self):
            bank,mask,gt=self.fixture(); mask.zero_()
            value,ids,count=oracle_for_axes(bank,mask,torch.tensor([[0]]),torch.tensor([[0]]),gt)
            self.assertTrue(torch.isinf(value).all()); self.assertEqual(ids.tolist(),[-1]); self.assertEqual(count.tolist(),[0])
            with self.assertRaisesRegex(RuntimeError,'no valid'):
                three_oracles(bank,mask,torch.tensor([[0]]),gt)
        def test_multiple_rows_and_nested_oracles(self):
            bank,mask,gt=self.fixture(); gt=gt.repeat(2,1,1); gt[1,:,0]=3.;gt[1,:,1]=2.
            result=three_oracles(bank,mask,torch.tensor([[0,1,4,5],[5,6,9,10]]),gt,chunk=3,expected_paths=2,expected_velocities=2)
            self.assertTrue(np.all(result['all_v_oracle']<=result['current_oracle']))
            self.assertTrue(np.all(result['all_p_oracle']<=result['current_oracle']))
            self.assertTrue(np.array_equal(result['current_valid_count'],[4,4]))
        def test_reference_requires_bytes_and_preserves_difference(self):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'ref.npz'; pred=np.zeros((1,6,2),np.float32)
                original={'rows':np.asarray([1],np.int64),'pred':pred,'candidate_id':np.asarray([2],np.int64)}
                np.savez(path,**original)
                self.assertTrue(reference_parity(original,path)['passed'])
                modified={**original,'pred':pred.copy()}; modified['pred'][0,0,0]=-0.
                self.assertFalse(reference_parity(modified,path)['passed'])
                modified={**original,'candidate_id':np.asarray([3],np.int64)}
                report=reference_parity(modified,path)
                self.assertFalse(report['passed']);self.assertEqual(report['changed_candidate_id_count'],1)
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OracleTests))
    require(result.wasSuccessful(),'CPU oracle tests failed')


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--checkpoint');p.add_argument('--output');p.add_argument('--gpu',type=int,choices=(0,1,4))
    p.add_argument('--worktree');p.add_argument('--bank');p.add_argument('--public-checkpoint');p.add_argument('--base')
    p.add_argument('--limit',type=int,default=0);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--chunk',type=int,default=2048);p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test: self_test();return
    require(a.checkpoint and a.output,'--checkpoint and --output required')
    require(a.workers>=0 and 0<=a.limit<=1998 and a.chunk>0,'Invalid run options')
    helper=load_helper();gpu=helper.check_gpu(a.gpu)
    plan=helper.inspect_checkpoint(a.checkpoint,worktree=a.worktree,bank=a.bank,public_checkpoint=a.public_checkpoint,
        base=a.base,allow_canary=bool(a.limit))
    require(plan.manifest['arguments']['common_status'] is True and plan.manifest['arguments']['history_mode']=='real',
            'This predeclared diagnostic is for temporal C')
    require(plan.manifest['arguments']['eval_batch']==BATCH_SIZE,'Training reference was not B8')
    require(str(torch.__version__)==plan.manifest['torch'],'Training/runtime torch versions differ')
    reference=plan.checkpoint.parent/f"eval_{plan.payload['step']:06d}.npz"
    require(reference.is_file(),'Terminal training prediction reference is required')
    reference_hash=sha(reference)
    output=Path(a.output).resolve();require(not output.exists(),'Output directory already exists');output.mkdir(parents=True)
    shutil.copy2(__file__,output/Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('evaluate_temporal_checkpoint.py'),output/'evaluate_temporal_checkpoint.py')
    shutil.copytree(plan.source,output/'frozen_source')
    receipt={**plan.receipt,'script_sha256':sha(__file__),'helper_sha256':HELPER_SHA,'gpu':gpu,
        'batch_size':BATCH_SIZE,'precision':'bf16_base_fp32_relative_head','oracle_arithmetic':'FP64 after frozen model forward',
        'oracle_chunk':a.chunk,'canary_limit':a.limit,'training_reference_sha256_before':reference_hash,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'gt_use':'outer oracle metrics only; excluded by model input whitelist',
        'all_p_all_v_oracle_computed':False,'inference_coordinates_modified':False}
    started=time.time();torch.set_num_threads(4)
    try:
        with helper.isolated_runtime(plan.source) as runtime:
            model=helper.strict_load(plan,runtime).cuda().eval();receipt.update(plan.receipt)
            dataset,provenance=helper.build_dataset(plan,runtime,a.limit)
            arrays,counts=run_diagnostic(model,dataset,runtime,helper,a.chunk,a.workers)
        # Save every actual prediction and oracle before any reference parity failure.
        np.savez_compressed(output/'diagnostics.npz',**arrays)
        result={'status':'evaluated_pending_reference_check','n':len(arrays['rows']),'batch_size':BATCH_SIZE,
            'oracle_shapes':{'current':[20,10],'all_v':[20,1024],'all_p':[1024,10]},
            'summary':summarize(arrays),'elapsed_seconds':time.time()-started,
            'privileged_gt_diagnostic_not_selector_performance':True}
        helper.json_write(output/'result.json',result)
        require(sha(reference)==reference_hash,'Training reference changed during diagnostic')
        parity=reference_parity(arrays,reference)
        result.update(status='completed' if parity['passed'] else 'parity_failed',training_reference_parity=parity)
        receipt.update(dataset_provenance=provenance,route_checks=counts,
            diagnostics_sha256=sha(output/'diagnostics.npz'),training_reference_parity=parity,
            all_candidate_rows_and_nested_oracles_verified=True,elapsed_seconds=time.time()-started)
        helper.json_write(output/'receipt.json',receipt);helper.json_write(output/'result.json',result)
        print(json.dumps(result,allow_nan=False),flush=True)
        require(parity['passed'],'B8 frozen reload differs from original B8 predictions; actual arrays preserved without adjustment')
    except BaseException:
        receipt.update(status='failed',traceback=traceback.format_exc(),elapsed_seconds=time.time()-started)
        helper.json_write(output/'failure.json',receipt)
        raise


if __name__=='__main__':
    main()
