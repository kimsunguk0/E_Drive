"""Strict live evaluation of frozen-C scene heads and C continuations.

CPU inspection is available with --audit-only. GPU work requires explicit
--gpu and its matching CUDA_VISIBLE_DEVICES. Only original tune1998 is supported.
For frozen heads, --cache-reference TUNE_DIR compares B8 live features and head
outputs against the immutable cache using the same head and batch size. A cached
trainer's differently batched saved predictions are reported, not assumed exact.

The original runtime is UUID-isolated. Load original C first; continuation
model_c is then applied with strict original keys and immutable-buffer checks.
The new head lives outside that original object, preserving route_monitor layout.
The captured representation combines image DFA, path/velocity candidate
embeddings, and indirect common-status conditioning. A real-versus-zero gain
measures the value of this additional representation, not pure image causality.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import torch
from torch import nn

EVALUATOR_SHA = '9007a4517adaf663d402535bdd56dd5d8340ad675db0f3a3ef44d8d867529242'
C_CHECKPOINT_SHA = 'b9dcc56af7c2c4c3cfc80d844fe862a2d8f730d7b8d7a813ee997d33abfec2ff'
SCENE_SOURCE_SHA = '57dbe094546243329ead690ff70b18cb54553a012397b3917a67868f14bceb47'
WEIGHTS = np.asarray([11,11,5,5,2,2], np.float64)/36.


def require(value, message):
    if not value:
        raise ValueError(message)


def equal_bits(actual, reference):
    """Exact representation equality, including signed zero, after explicit casts."""
    return (actual.dtype==reference.dtype and actual.shape==reference.shape and
            torch.equal(actual.contiguous().view(torch.uint8),reference.contiguous().view(torch.uint8)))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def module_from(path, prefix):
    name=prefix+'_'+uuid.uuid4().hex
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def counts_and_mode(manifest):
    """Accept the frozen trainer's existing nested receipts or explicit continuation fields."""
    schema=manifest['schema']
    require(schema in ('c_scene_selector_frozen_v1','pv_selector_status_v1','c_scene_continuation_v1'),'Unknown refinement schema')
    if schema in ('c_scene_selector_frozen_v1','pv_selector_status_v1'):
        train,tune=manifest['train_cache_manifest'],manifest['tune_cache_manifest']
        for key in ('counts','checkpoint_sha256','bank_sha256','source_sha256'):
            require(train[key]==tune[key],f'Train/tune cache provenance differs: {key}')
        require(train['checkpoint_sha256']==C_CHECKPOINT_SHA,'Cache parent is not frozen C')
        counts=train['counts']; mode=manifest['arguments']['mode']
        public_c=train['checkpoint']['checkpoint']
        p,v=counts['path_filter'],counts['velocity_filter']
    else:
        require(manifest['original_c_checkpoint_sha256']==C_CHECKPOINT_SHA,'Continuation parent changed')
        p,v=manifest['path_filter'],manifest['velocity_filter']
        mode=manifest['head_mode'];public_c=manifest['original_c_checkpoint']
        if isinstance(public_c,dict):
            require(public_c.get('sha256',public_c.get('sha'))==C_CHECKPOINT_SHA,'Continuation original C path receipt changed')
            public_c=public_c['path']
    require(mode in ('real','zero'),'Invalid scene-head mode')
    # 'status' routes the provided causal vx,vy,ax,ay into the 32 selection
    # features of already-completed bank candidates; absent means the old zeros.
    status=manifest.get('arguments',{}).get('status','zero')
    require(status in ('real','zero'),'Invalid selection status mode')
    require(list(p)==[128,20] and list(v)==[64,64],'Only the frozen selected V64 protocol is supported')
    return dict(path_filter=list(p),velocity_filter=list(v),mode=mode,status=status,
                original_c_checkpoint=public_c)


def inspect_scene_checkpoint(path, ev):
    checkpoint=Path(path).resolve(); digest=sha(checkpoint)
    payload=ev.load_internal_payload(checkpoint)
    require(sha(checkpoint)==digest,'Refinement checkpoint changed while loading')
    require(isinstance(payload,dict) and {'head','manifest','step'}.issubset(payload),'Malformed refinement checkpoint')
    payload.pop('optimizer',None)
    m=payload['manifest'];config=counts_and_mode(m)
    require(('model_c' in payload)==(m['schema']=='c_scene_continuation_v1'),'Payload model_c/schema mismatch')
    disk=checkpoint.parent/'manifest.json'
    require(ev.canonical(json.loads(disk.read_text()))==ev.canonical(m),'Refinement disk/payload manifest mismatch')
    source=checkpoint.parent/'source'
    require(m['source_sha256'].get('pv_selector.py')==SCENE_SOURCE_SHA,'Unsupported scene head implementation')
    for name,digest_source in m['source_sha256'].items():
        rel=Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe refinement source path')
        require(sha(source/rel)==digest_source,f'Refinement source changed: {name}')
    arguments=m.get('arguments',{})
    require(0<int(payload['step'])<=int(arguments['steps']),'Checkpoint step outside recorded budget')
    result=dict(path=checkpoint,payload=payload,manifest=m,config=config,source=source,
                sha256=sha(checkpoint),manifest_sha256=sha(disk))
    if m['schema']=='c_scene_continuation_v1':
        parent=m['frozen_head_checkpoint']; parent_path=Path(parent['path']).resolve()
        require(parent_path!=checkpoint and sha(parent_path)==parent['sha256'],'Frozen head parent identity changed')
        parent_payload=ev.load_internal_payload(parent_path)
        require(parent_payload['manifest']['schema']=='c_scene_selector_frozen_v1','Continuation parent must be a frozen head')
        del parent_payload
        parent_plan=inspect_scene_checkpoint(parent_path,ev)
        require(parent_plan['payload']['step']==parent['step'],'Frozen head parent step differs')
        require(parent_plan['config']==config,'Frozen head parent mode, candidate filters, or original C differ')
        require(parent_plan['manifest']['bank_sha256']==m['bank_sha256'],'Frozen head parent bank differs')
        result['frozen_head_parent_verified']=dict(path=str(parent_path),sha256=parent_plan['sha256'],step=parent['step'],
            manifest_sha256=parent_plan['manifest_sha256'])
    return result


def apply_continuation(model, state, ev):
    """Preserve every original buffer, including bank, BN, and geometry constants."""
    expected={name:ev.tensor_sha(value) for name,value in model.named_buffers()}
    model.load_state_dict(state,strict=True)
    for name,value in model.state_dict().items():
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()),f'Nonfinite model_c tensor: {name}')
    for name,value in model.named_buffers():
        require(ev.tensor_sha(value)==expected[name],f'Continuation changed fixed original buffer: {name}')
    return expected


class FullSceneModel(nn.Module):
    def __init__(self, original_c, scene_head, scene_module, status_mode='zero'):
        super().__init__()
        self.capture=scene_module.CandidateTokenCapture(original_c,detach_tokens=True,freeze_base=True)
        self.scene_head=scene_head
        if status_mode not in ('real','zero'):
            raise ValueError('status_mode must be real or zero')
        self.status_mode=status_mode

    @staticmethod
    def status8_from(inputs, status_mode):
        """Provided causal vx,vy,ax,ay for final selection only; None when zero.

        The same tensor already conditions the shared perception query, so this
        reads it from the live batch rather than from any offline artifact.
        """
        if status_mode!='real':
            return None
        state=inputs['perception_status']
        if state.shape[-1:]!=(4,) or not bool(torch.isfinite(state).all()):
            raise ValueError('perception_status must be finite [...,4]')
        status8=state.new_zeros((state.shape[0],8))
        status8[:,4:8]=state
        return status8

    @property
    def _trajectory_head(self):
        return self.capture._trajectory_head

    @property
    def original_c(self):
        return self.capture.base

    def forward(self, **inputs):
        output=self.capture(**inputs)
        return self.scene_head(output,goal_xy=inputs['goal_xy'],
                               status=self.status8_from(inputs,self.status_mode))


def load_verified_models(scene,plan,runtime,ev):
    """CPU-only reusable loader; caller keeps ev.isolated_runtime alive.

    Returns (FullSceneModel, original_C, scene_head). Raw-fixture evaluators can
    reuse the full model and apply the original route_monitor to original_C.
    """
    mod=module_from(scene['source']/'pv_selector.py','_scene_head_recorded')
    original=ev.strict_load(plan,runtime)
    if 'model_c' in scene['payload']:
        scene['continuation_buffers_verified']=apply_continuation(original,scene['payload']['model_c'],ev)
    original.base.base.path_filter=tuple(scene['config']['path_filter'])
    original.base.base.velocity_filter=tuple(scene['config']['velocity_filter'])
    head=mod.SceneResidualSelector(mode=scene['config']['mode'])
    head.load_state_dict(scene['payload']['head'],strict=True)
    require(all(bool(torch.isfinite(v).all()) for v in head.state_dict().values()),'Nonfinite head weights')
    model=FullSceneModel(original,head,mod,scene['config']['status']).eval()
    return model,original,head


class CacheReference:
    def __init__(self,path,plan,scene,ev):
        self.root=Path(path).resolve();self.manifest=json.loads((self.root/'manifest.json').read_text())
        m=self.manifest
        require(scene['manifest']['schema'] in ('c_scene_selector_frozen_v1','pv_selector_status_v1'),'A frozen base cache cannot audit a continued model_c')
        require(m['status']=='completed' and not m.get('canary',False) and m['split']=='tune','Need a full completed tune cache')
        require(m['checkpoint_sha256']==C_CHECKPOINT_SHA and m['bank_sha256']==plan.receipt['bank_sha256'],'Cache model/bank differs')
        require(m['counts']['path_filter']==scene['config']['path_filter'] and
                m['counts']['velocity_filter']==scene['config']['velocity_filter'],'Cache candidate count configuration differs')
        self.artifacts={}
        # Writer records specs by field name, with an explicit path for each file.
        specs=m['files']
        self.arrays={}
        for key in ('rows','candidate_ids','candidate_valid','scores','token','goal_xy'):
            name=key+'.npy'; path=self.root/name
            require(key in specs and specs[key]['path']==name and sha(path)==specs[key]['sha256'],f'Cache artifact changed: {name}')
            self.artifacts[name]=specs[key]['sha256']
            self.arrays[key]=np.load(path,mmap_mode='r',allow_pickle=False)
            require(list(self.arrays[key].shape)==specs[key]['shape'] and str(self.arrays[key].dtype)==specs[key]['dtype'],
                    f'Cache artifact shape/dtype differs: {name}')
        require(np.array_equal(self.arrays['rows'],plan.rows),'Cache original tune row order differs')
        self.receipt=dict(path=str(self.root),manifest_sha256=sha(self.root/'manifest.json'),
                          compared_fields=list(self.arrays),artifacts=self.artifacts)

    def compare_batch(self, batch, output, goal, head, bank, start, status=None):
        n=len(batch['row']);end=start+n;device=goal.device
        require(np.array_equal(self.arrays['rows'][start:end],batch['row'].numpy()),'Live/cache row mismatch')
        refs={key:torch.from_numpy(np.array(value[start:end])).to(device) for key,value in self.arrays.items() if key!='rows'}
        for actual,key in ((output['candidate_ids'],'candidate_ids'),(output['candidate_valid'],'candidate_valid'),
                           (output['old_final_scores'],'scores'),(output['candidate_tokens'].float(),'token'),(goal,'goal_xy')):
            reference=refs[key].float() if actual.is_floating_point() else refs[key]
            require(equal_bits(actual,reference),f'Live/cache exact comparison failed: {key}')
        offline=dict(candidate_ids=refs['candidate_ids'],candidate_xy=bank[refs['candidate_ids']],
            candidate_valid=refs['candidate_valid'],scores=refs['scores'],candidate_tokens=refs['token'])
        repeated=head(offline,goal_xy=refs['goal_xy'],status=status)
        for key in ('scores','trajectory','selected_candidate_id'):
            require(equal_bits(output[key],repeated[key]),f'Same-batch cached/live scene head differs: {key}')


@torch.inference_mode()
def evaluate_live(model, original, head, dataset, runtime, ev, output_dir, batch_size, workers, cache=None):
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=True)
    chunks={};start=0
    with ev.route_monitor(original,True) as counts:
        for batch in loader:
            inputs=ev.input_tensors(batch,runtime,True,'cuda:0')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model(**inputs)
            ev.verify_rows(original,out)
            if cache:
                bank=original._trajectory_head.traj_vocab.flatten(0,1)[:,:6,:2]
                cache.compare_batch(batch,out,inputs['goal_xy'],head,bank,start,
                                    FullSceneModel.status8_from(inputs,model.status_mode))
            # Labels enter only after all base and selector forward computations.
            gt=batch['gt_plan'].numpy().astype(np.float64)
            pred=out['trajectory'].float().cpu().numpy()
            xy=out['candidate_xy'].float().cpu().numpy().astype(np.float64)
            valid=out['candidate_valid'].cpu().numpy()
            error=pred.astype(np.float64)-gt
            point=np.linalg.norm(error,axis=-1)
            cost=np.linalg.norm(xy-gt[:,None],axis=-1) @ WEIGHTS
            original_index=out['old_final_scores'].masked_fill(~out['candidate_valid'],-torch.inf).argmax(-1).cpu().numpy()
            values=dict(rows=batch['row'].numpy(),pred=pred,candidate_id=out['selected_candidate_id'].cpu().numpy(),
                error_xy=error,point_l2=point,d3=point @ WEIGHTS,shortlist_oracle=np.where(valid,cost,np.inf).min(-1),
                base_d3=cost[np.arange(len(gt)),original_index],session=np.asarray(batch['session']),scenario=np.asarray(batch['scenario']))
            for key,value in values.items():
                chunks.setdefault(key,[]).append(value)
            start+=len(gt)
            if start%128==0 or start==len(dataset):
                print(json.dumps({'evaluation_rows':start,'total':len(dataset),'batch_size':batch_size}),flush=True)
    arrays={key:np.concatenate(values) for key,values in chunks.items()}
    require(np.array_equal(arrays['rows'],dataset.rows),'Live evaluation row order differs')
    require(all(v==len(loader) for v in counts.values()),'Original route monitor call count mismatch')
    result=dict(n=len(arrays['rows']),batch_size=batch_size,official_d3=float(arrays['d3'].mean()),
        shortlist_oracle_d3=float(arrays['shortlist_oracle'].mean()),
        selection_regret=float((arrays['d3']-arrays['shortlist_oracle']).mean()),base_d3=float(arrays['base_d3'].mean()),
        point_l2=arrays['point_l2'].mean(0).tolist(),three_second_l2=float(arrays['point_l2'][:,-1].mean()),
        route_counts=counts,GT_passed_to_model=False,
        session_d3={str(s):float(arrays['d3'][arrays['session']==s].mean()) for s in np.unique(arrays['session'])})
    np.savez_compressed(output_dir/'predictions.npz',**arrays)
    atomic_json(output_dir/'result.json',result)
    return result,arrays


@torch.inference_mode()
def audit_goal_routes(model,original,inputs,ev):
    with ev.route_monitor(original,True) as counts:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            first=model(**inputs)
            other=model(**{**inputs,'goal_xy':inputs['goal_xy']+inputs['goal_xy'].new_tensor([10.,-3.])})
        for key in ('candidate_ids','candidate_xy','candidate_valid','candidate_tokens','base_scores','aux_occ','aux_lane','aux_state'):
            require(torch.equal(first[key],other[key]),f'Goal changed a non-final field: {key}')
        ev.verify_rows(original,first);ev.verify_rows(original,other)
    return dict(goal_only_after_complete_candidates=True,candidate_token_goal_invariant=True,
        original_route_counts=counts,new_head_signature='output, goal_xy, provided causal status for completed-candidate selection only')


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--checkpoint',required=True,help='Frozen scene head or continuation checkpoint')
    p.add_argument('--original-c-checkpoint')
    p.add_argument('--evaluator-source',required=True)
    p.add_argument('--worktree',required=True)
    p.add_argument('--base',default='/NHNHOME/data/sukim/adcl')
    p.add_argument('--output',required=True)
    p.add_argument('--gpu',type=int,choices=(0,1,4))
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--cache-reference')
    p.add_argument('--reference-predictions')
    p.add_argument('--profile',action='store_true')
    args=p.parse_args()
    require(sha(args.evaluator_source)==EVALUATOR_SHA,'Strict original evaluator source differs')
    require(args.audit_only or args.gpu is not None,'Explicit GPU is required for live evaluation')
    require(not args.audit_only or not os.environ.get('CUDA_VISIBLE_DEVICES',''),'Hide GPUs for CPU audit')
    require(args.batch in (1,8) and args.workers>=0,'Evaluate the declared B1 or B8 protocols')
    require(not args.cache_reference or args.batch==8,'Cache parity must use original batch8')
    outdir=Path(args.output).resolve();require(not outdir.exists(),'Use a new immutable output directory')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,str(Path(args.worktree).resolve()))
    ev=module_from(Path(args.evaluator_source).resolve(),'_scene_original_evaluator')
    scene=inspect_scene_checkpoint(args.checkpoint,ev)
    original_path=args.original_c_checkpoint or scene['config']['original_c_checkpoint']
    plan=ev.inspect_checkpoint(original_path,worktree=args.worktree,base=args.base)
    require(plan.receipt['checkpoint_sha256']==C_CHECKPOINT_SHA,'Original C checkpoint identity mismatch')
    require(str(torch.__version__)==plan.manifest['torch'],'Original C runtime torch version changed')
    require(plan.manifest['arguments']['common_status'] is True,'Only common-status C is supported')
    require(scene['manifest']['bank_sha256']==plan.receipt['bank_sha256'],'Refinement bank differs from original C')
    gpu=None if args.audit_only else ev.check_gpu(args.gpu)
    outdir.mkdir(parents=True)
    started=time.monotonic()
    receipt=dict(schema='c_scene_live_evaluation_v1',arguments=vars(args),checkpoint_sha256=scene['sha256'],
        scene_manifest_sha256=scene['manifest_sha256'],config=scene['config'],original_c=plan.receipt,
        evaluator_sha256=sha(__file__),scene_source_sha256=SCENE_SOURCE_SHA,gpu=gpu,status='incomplete',
        interpretation='real versus zero tests additional candidate representation containing image DFA, path/velocity embeddings, and indirect common-status conditioning; it does not isolate pure image causality')
    if 'frozen_head_parent_verified' in scene:
        receipt['frozen_head_parent_verified']=scene['frozen_head_parent_verified']
    try:
        with ev.isolated_runtime(plan.source) as runtime:
            model,original,head=load_verified_models(scene,plan,runtime,ev)
            if 'continuation_buffers_verified' in scene:
                receipt['continuation_buffers_verified']=scene['continuation_buffers_verified']
            dataset,provenance=ev.build_dataset(plan,runtime,args.limit)
            receipt['dataset_provenance']=provenance
            if args.audit_only:
                receipt.update(status='completed',CPU_audit_only=True,forward_performed=False)
            else:
                model.cuda();head.eval();original.eval()
                cache=CacheReference(args.cache_reference,plan,scene,ev) if args.cache_reference else None
                result,arrays=evaluate_live(model,original,head,dataset,runtime,ev,outdir,args.batch,args.workers,cache)
                receipt['result']=result
                receipt['cache_parity']=None if cache is None else {**cache.receipt,'exact_all_consumed_rows':True}
                first=next(iter(torch.utils.data.DataLoader(dataset,batch_size=1,num_workers=0)))
                inputs=ev.input_tensors(first,runtime,True,'cuda:0')
                receipt['goal_route_audit']=audit_goal_routes(model,original,inputs,ev)
                if args.reference_predictions:
                    receipt['saved_prediction_comparison']=ev.compare_reference(arrays,args.reference_predictions,
                        training_batch_size=scene['manifest']['arguments'].get('eval_batch',8))
                    receipt['saved_prediction_comparison']['metric_note']='Actual live B%d predictions and FP64 metrics versus saved predictions; differing base/head batch sizes can change floating-point scores and selected IDs' % args.batch
                if args.profile:
                    profile=ev.profile_model(model,inputs)
                    profile['scope']='full original C + scoped candidate token capture/alignment checks + new final scene head'
                    profile['excluded']='raw/cache I/O, input assembly, H2D, GT metrics, outer evaluator route hooks; capture checks ARE included'
                    receipt['profile']=profile
                receipt.update(status='completed',predictions_sha256=sha(outdir/'predictions.npz'))
        require(sha(args.checkpoint)==scene['sha256'],'Refinement checkpoint changed during evaluation')
        receipt['elapsed_seconds']=time.monotonic()-started
        atomic_json(outdir/'receipt.json',receipt)
        print(json.dumps({'status':'completed','output':str(outdir),'result':receipt.get('result')},allow_nan=False),flush=True)
    except BaseException as exc:
        atomic_json(outdir/'failure.json',dict(type=type(exc).__name__,message=str(exc),elapsed_seconds=time.monotonic()-started))
        raise


if __name__=='__main__':
    main()
