"""Verified live composition on original tune only; no held population enabled.

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
    from .relative_artifact import load_relative_artifact,file_sha,require
    from .relative_selector import CandidateRelativeSelector
    from .evaluate_checkpoint import inspect_checkpoint,recorded_runtime,load_verified_model,build_dataset,verify_bank_output,TUNE_SHA,rows_sha,WEIGHTS
except ImportError:
    from relative_artifact import load_relative_artifact,file_sha,require
    from relative_selector import CandidateRelativeSelector
    from evaluate_checkpoint import inspect_checkpoint,recorded_runtime,load_verified_model,build_dataset,verify_bank_output,TUNE_SHA,rows_sha,WEIGHTS


def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def compare_reference(arrays,path,tolerance=1e-5):
    result={}
    with np.load(path,allow_pickle=False) as ref:
        require(rows_sha(ref['rows'])==TUNE_SHA and len(ref['rows'])==1998,'Reference must contain exact original tune population')
        for key in ('rows','pred','candidate_id'):
            require(np.array_equal(arrays[key],ref[key]),f'Live/cache exact bank-row parity failed: {key}')
            result[key]='bitwise exact'
        for key in ('d3','shortlist_oracle','point_l2','error_xy'):
            if key in ref:
                error=float(np.max(np.abs(arrays[key].astype(np.float64)-ref[key].astype(np.float64))))
                require(error<=tolerance,f'Live/cache metric tolerance failed: {key} {error}')
                result[key]={'max_abs_error':error,'tolerance':tolerance}
    return {'reference_sha256':file_sha(path),'checks':result}


@torch.inference_mode()
def evaluate(model,dataset,runtime,batch_size,workers):
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=True)
    chunks={k:[] for k in ('rows','pred','candidate_id','gt_xy','shortlist_oracle')}
    for batch in loader:
        x={k:v.to('cuda:0',non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        # GT remains in the outer evaluation batch, never in this selected mapping.
        inputs=runtime.data.model_inputs(x,goal_selection=True)
        require(set(inputs)=={'images','lidar2img','image_hw','status','goal_xy'},'Unexpected live selector inputs')
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**inputs)
        verify_bank_output(model,out,runtime)
        costs=runtime.data.d3(out['candidate_xy'],x['gt_plan'][:,None].expand_as(out['candidate_xy']))
        values={'rows':batch['row'],'pred':out['trajectory'].float(),
                'candidate_id':out['selected_candidate_id'],'gt_xy':batch['gt_plan'],
                'shortlist_oracle':costs.masked_fill(~out['candidate_valid'],torch.inf).amin(-1)}
        for key,value in values.items():chunks[key].append(value.detach().cpu().numpy())
    arrays={key:np.concatenate(value) for key,value in chunks.items()}
    require(np.array_equal(arrays['rows'],dataset.rows) and rows_sha(arrays['rows'])==TUNE_SHA,
            'Live selector changed tune population/order')
    arrays['error_xy']=arrays['pred'].astype(np.float64)-arrays.pop('gt_xy').astype(np.float64)
    arrays['point_l2']=np.linalg.norm(arrays['error_xy'],axis=-1)
    arrays['d3']=(arrays['point_l2']*WEIGHTS).sum(-1)
    return arrays


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--head',required=True);p.add_argument('--head-sha256',required=True)
    p.add_argument('--cache-manifest');p.add_argument('--reference')
    p.add_argument('--output',required=True);p.add_argument('--audit-only',action='store_true')
    p.add_argument('--batch',type=int,default=8);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(4)
    if not a.audit_only:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1','4'),'Expose one explicitly assigned GPU')
    directory=Path(a.output);directory.mkdir(parents=True,exist_ok=False)
    started=time.time()
    try:
        head,cache,receipt=load_relative_artifact(a.head,expected_sha256=a.head_sha256,cache_manifest_path=a.cache_manifest)
        require(file_sha(cache['checkpoint']['path'])==cache['checkpoint']['sha256'],'Cached base checkpoint changed')
        require(file_sha(cache['bank']['path'])==cache['bank']['sha256'],'Cached bank changed')
        plan=inspect_checkpoint(cache['checkpoint']['path'],bank_path=cache['bank']['path'],population='tune')
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
                    'population':'original_tune1998','held_evaluation_enabled':False}
            if not a.audit_only:
                # Copy only the dataset goal-mode flag. The model retains the
                # frozen base goal mode and uses provided goal for its head.
                dataset_plan=copy.copy(plan);dataset_plan.goal_mode='selection'
                dataset=build_dataset(dataset_plan,runtime)
                model.cuda().eval();torch.cuda.reset_peak_memory_stats()
                arrays=evaluate(model,dataset,runtime,a.batch,a.workers)
                reference=Path(a.reference) if a.reference else Path(a.head).parent/f"eval_{receipt['relative_step']:06d}.npz"
                result['cache_live_parity']=compare_reference(arrays,reference)
                result.update(n=len(arrays['rows']),official_d3=float(arrays['d3'].mean()),
                    shortlist_oracle_d3=float(arrays['shortlist_oracle'].mean()),
                    selection_regret=float((arrays['d3']-arrays['shortlist_oracle']).mean()),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),fixed_bank_rows_exact=True)
                np.savez_compressed(directory/'predictions.npz',**arrays)
            result['elapsed_seconds']=time.time()-started
            result['source_sha256']={name:file_sha(Path(__file__).with_name(name)) for name in
                ('evaluate_relative_selector.py','relative_artifact.py','relative_selector.py')}
            write_json(directory/'result.json',result)
            print(json.dumps(result,indent=2),flush=True)
    except BaseException:
        write_json(directory/'failure.json',{'status':'failed','traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
