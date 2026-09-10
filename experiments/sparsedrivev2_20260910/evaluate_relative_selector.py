"""Verified live composition with an explicit frozen-decision held guard.

CPU --audit-only strictly verifies head/base/bank/source ancestry without loading
label arrays or performing inference. Full evaluation must receive an explicitly
allocated GPU and compares every predicted bank row against cached head results.
"""
from __future__ import annotations
import argparse,copy,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .relative_artifact import load_relative_artifact,file_sha,require,require_confirmation_binding
    from .relative_selector import CandidateRelativeSelector
    from .evaluate_checkpoint import inspect_checkpoint,recorded_runtime,load_verified_model,build_dataset,verify_bank_output,TUNE_SHA,rows_sha,WEIGHTS
except ImportError:
    from relative_artifact import load_relative_artifact,file_sha,require,require_confirmation_binding
    from relative_selector import CandidateRelativeSelector
    from evaluate_checkpoint import inspect_checkpoint,recorded_runtime,load_verified_model,build_dataset,verify_bank_output,TUNE_SHA,rows_sha,WEIGHTS


def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def paired_base_prediction(model,output):
    """Select the already-computed base winner from the same immutable shortlist."""
    valid=output['candidate_valid']
    base_index=output['base_scores'].masked_fill(~valid,-torch.inf).argmax(-1)
    row=torch.arange(len(base_index),device=base_index.device)
    prediction=output['candidate_xy'][row,base_index]
    ids=output['candidate_ids'][row,base_index]
    bank=model._trajectory_head.traj_vocab.flatten(0,1)
    require(torch.equal(prediction,bank[ids,:6,:2]),'Paired base prediction is not its bank row')
    return prediction,ids


def compare_reference(arrays,path,tolerance=1e-5,allow_batch_numerical_differences=False):
    result={}
    with np.load(path,allow_pickle=False) as ref:
        require(rows_sha(ref['rows'])==TUNE_SHA and len(ref['rows'])==1998,'Reference must contain exact original tune population')
        require(np.array_equal(arrays['rows'],ref['rows']),'Reference row order changed')
        for key in ('rows','pred','candidate_id'):
            exact=bool(np.array_equal(arrays[key],ref[key]))
            if not allow_batch_numerical_differences:
                require(exact,f'Live/cache exact bank-row parity failed: {key}')
            result[key]='bitwise exact' if exact else 'different; explicit batch-numerical comparison'
        for key in ('d3','shortlist_oracle','point_l2','error_xy'):
            if key in ref:
                error=float(np.max(np.abs(arrays[key].astype(np.float64)-ref[key].astype(np.float64))))
                if not allow_batch_numerical_differences:require(error<=tolerance,f'Live/cache metric tolerance failed: {key} {error}')
                result[key]={'max_abs_error':error,'tolerance':tolerance}
        numerical={'changed_candidate_ids':int(np.count_nonzero(arrays['candidate_id']!=ref['candidate_id'])),
            'changed_prediction_rows':int(np.count_nonzero(np.any(arrays['pred']!=ref['pred'],axis=(1,2)))),
            'prediction_max_abs_metres':float(np.max(np.abs(arrays['pred'].astype(np.float64)-ref['pred'].astype(np.float64)))),
            'd3_mean_delta':float(arrays['d3'].mean()-ref['d3'].mean()),'reference_d3':float(ref['d3'].mean())}
    return {'reference_sha256':file_sha(path),'checks':result,'batch_numerical_comparison':numerical,
            'exact_xy_ids':numerical['changed_candidate_ids']==0 and numerical['changed_prediction_rows']==0}


@torch.inference_mode()
def evaluate(model,dataset,runtime,batch_size,workers,expected_rows):
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=True)
    chunks={k:[] for k in ('rows','pred','candidate_id','gt_xy','shortlist_oracle','base_pred','base_candidate_id')}
    for batch in loader:
        x={k:v.to('cuda:0',non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        # GT remains in the outer evaluation batch, never in this selected mapping.
        inputs=runtime.data.model_inputs(x,goal_selection=True)
        require(set(inputs)=={'images','lidar2img','image_hw','status','goal_xy'},'Unexpected live selector inputs')
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**inputs)
        verify_bank_output(model,out,runtime)
        # The frozen base result is already present in this same forward. This
        # paired diagnostic never changes head selection or opens another model.
        base_pred,base_ids=paired_base_prediction(model,out)
        costs=runtime.data.d3(out['candidate_xy'],x['gt_plan'][:,None].expand_as(out['candidate_xy']))
        values={'rows':batch['row'],'pred':out['trajectory'].float(),
                'candidate_id':out['selected_candidate_id'],'gt_xy':batch['gt_plan'],
                'shortlist_oracle':costs.masked_fill(~out['candidate_valid'],torch.inf).amin(-1),
                'base_pred':base_pred.float(),'base_candidate_id':base_ids}
        for key,value in values.items():chunks[key].append(value.detach().cpu().numpy())
    arrays={key:np.concatenate(value) for key,value in chunks.items()}
    require(np.array_equal(arrays['rows'],dataset.rows) and np.array_equal(arrays['rows'],expected_rows),
            'Live selector changed the approved evaluation population/order')
    gt=arrays.pop('gt_xy').astype(np.float64)
    arrays['error_xy']=arrays['pred'].astype(np.float64)-gt
    arrays['point_l2']=np.linalg.norm(arrays['error_xy'],axis=-1)
    arrays['d3']=(arrays['point_l2']*WEIGHTS).sum(-1)
    arrays['base_error_xy']=arrays['base_pred'].astype(np.float64)-gt
    arrays['base_point_l2']=np.linalg.norm(arrays['base_error_xy'],axis=-1)
    arrays['base_d3']=(arrays['base_point_l2']*WEIGHTS).sum(-1)
    return arrays


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--head',required=True);p.add_argument('--head-sha256',required=True)
    p.add_argument('--cache-manifest');p.add_argument('--reference')
    p.add_argument('--population',choices=('tune','confirmation12'),default='tune')
    p.add_argument('--train-rows');p.add_argument('--eval-rows');p.add_argument('--approved-candidate')
    p.add_argument('--output',required=True);p.add_argument('--audit-only',action='store_true')
    p.add_argument('--batch',type=int,default=8);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--allow-batch-numerical-differences',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    if not a.audit_only:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1','4'),'Expose one explicitly assigned GPU')
    directory=Path(a.output);directory.mkdir(parents=True,exist_ok=False)
    started=time.time()
    try:
        head,cache,receipt=load_relative_artifact(a.head,expected_sha256=a.head_sha256,cache_manifest_path=a.cache_manifest)
        confirmation_binding=None
        if a.population=='confirmation12':
            require(a.train_rows is not None and a.eval_rows is not None and a.approved_candidate is not None,
                    'Confirmation requires explicit frozen train/held rows and root candidate receipt')
            require(a.reference is None,'Held evaluation does not read a cached tune reference')
            confirmation_binding=require_confirmation_binding(json.loads(Path(a.approved_candidate).read_text()),receipt,cache)
            require(a.batch==confirmation_binding['evaluation_batch_size'],
                    'Held batch size differs from frozen numerical protocol')
        else:
            require(a.approved_candidate is None,'A held decision receipt is not used for tune evaluation')
        require(file_sha(cache['checkpoint']['path'])==cache['checkpoint']['sha256'],'Cached base checkpoint changed')
        require(file_sha(cache['bank']['path'])==cache['bank']['sha256'],'Cached bank changed')
        plan=inspect_checkpoint(cache['checkpoint']['path'],bank_path=cache['bank']['path'],population=a.population,
                train_rows_path=a.train_rows,eval_rows_path=a.eval_rows,approval_path=a.approved_candidate)
        require(plan.receipt['checkpoint_sha256']==receipt['base_checkpoint_sha256'] and
                plan.receipt['bank_sha256']==receipt['bank_sha256'],'Live/base provenance mismatch')
        require(plan.goal_mode==receipt['base_goal_mode'],'Live/cache base goal mode mismatch')
        with recorded_runtime(plan) as runtime:
            base,coverage=load_verified_model(plan,runtime)
            model=CandidateRelativeSelector(base,base_goal_mode=plan.goal_mode,freeze_base=True)
            model.relative_head.load_state_dict(head.state_dict(),strict=True)
            result={'schema':'sparsedrivev2_live_relative_evaluation_v1','status':'completed',
                    'audit_only':a.audit_only,'base':plan.receipt,'relative':receipt,
                    'strict_base_and_head_load':True,'goal_feature_mode':'selection',
                    'population':a.population,'held_evaluation_enabled':a.population=='confirmation12',
                    'confirmation_binding':confirmation_binding,'evaluation_batch_size':a.batch,
                    'evaluation_precision':'bf16_base_fp32_head',
                    'same_forward_base_comparison':True,'same_candidate_set':True,
                    'base_selection':'argmax of original base_scores over the same valid 200 bank candidates'}
            if not a.audit_only:
                # Copy only the dataset goal-mode flag. The model retains the
                # frozen base goal mode and uses provided goal for its head.
                dataset_plan=copy.copy(plan);dataset_plan.goal_mode='selection'
                dataset=build_dataset(dataset_plan,runtime)
                model.cuda().eval();torch.cuda.reset_peak_memory_stats()
                arrays=evaluate(model,dataset,runtime,a.batch,a.workers,plan.eval_rows)
                # Preserve actual predictions even if a later strict reference
                # check rejects batch-dependent BF16 ranking differences.
                np.savez_compressed(directory/'predictions.npz',**arrays)
                if a.population=='tune':
                    reference=Path(a.reference) if a.reference else Path(a.head).parent/f"eval_{receipt['relative_step']:06d}.npz"
                    result['cache_live_parity']=compare_reference(arrays,reference,
                        allow_batch_numerical_differences=a.allow_batch_numerical_differences)
                result.update(n=len(arrays['rows']),official_d3=float(arrays['d3'].mean()),
                    same_forward_base_d3=float(arrays['base_d3'].mean()),
                    paired_head_minus_base_d3=float((arrays['d3']-arrays['base_d3']).mean()),
                    shortlist_oracle_d3=float(arrays['shortlist_oracle'].mean()),
                    selection_regret=float((arrays['d3']-arrays['shortlist_oracle']).mean()),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),fixed_bank_rows_exact=True)
            result['elapsed_seconds']=time.time()-started
            result['source_sha256']={name:file_sha(Path(__file__).with_name(name)) for name in
                ('evaluate_relative_selector.py','relative_artifact.py','relative_selector.py')}
            write_json(directory/'result.json',result)
            print(json.dumps(result,indent=2),flush=True)
    except BaseException:
        write_json(directory/'failure.json',{'status':'failed','traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
