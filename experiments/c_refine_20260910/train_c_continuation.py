"""Weights continuation of verified temporal C plus a trained final scene head.

Prepared for a separately launched canary/full run. This never resumes AdamW's
terminal zero-LR state: a new optimizer/schedule is explicit. Original C tensor
keys and new scene-head keys are saved separately, with frozen source ancestry.
Only original TRAIN203/TUNE37 are supported; TUNE is repeated exploratory data.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from c_scene_selector import CandidateTokenCapture, SceneResidualSelector

SCHEMA='c_scene_continuation_v1'
EVALUATOR_SHA='9007a4517adaf663d402535bdd56dd5d8340ad675db0f3a3ef44d8d867529242'
SCENE_SHA='be8d9eef8e1e9f087852c1a81d0f5f4437e618a146d02d5037de60181b360ae5'
C_SHA='b9dcc56af7c2c4c3cfc80d844fe862a2d8f730d7b8d7a813ee997d33abfec2ff'
EXPECTED_COUNTS={'path_filter':[128,20],'velocity_filter':[64,64],'candidate_count':1280}
INPUT_KEYS={'images','lidar2img','image_hw','history_images','time_offsets','perception_status','goal_xy'}


def require(condition,message):
    if not condition:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def load_evaluator(path):
    require(sha(path)==EVALUATOR_SHA,'Strict C evaluator source changed')
    spec=importlib.util.spec_from_file_location('_continuation_verified_c_evaluator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class CSceneModel(nn.Module):
    """Original C -> complete fixed candidates/tokens -> trained final scene head."""
    def __init__(self,c_model,head):
        super().__init__()
        self.capture=CandidateTokenCapture(c_model,freeze_base=False,detach_tokens=False)
        self.head=head

    @property
    def model_c(self):return self.capture.base

    @property
    def _trajectory_head(self):return self.model_c._trajectory_head

    @property
    def _backbone(self):return self.model_c.base.base._backbone

    def forward(self,images,lidar2img,image_hw,history_images,time_offsets,
                perception_status=None,goal_xy=None):
        require(perception_status is not None,'Temporal C needs its existing common-perception condition')
        inputs=dict(images=images,lidar2img=lidar2img,image_hw=image_hw,
            history_images=history_images,time_offsets=time_offsets,
            perception_status=perception_status,goal_xy=goal_xy)
        output=self.capture(**inputs)
        # Goal is passed only after the complete bank candidate set exists.
        return self.head(output,goal_xy=goal_xy)


def validate_head_manifest(manifest,plan,step):
    require(manifest['schema']=='c_scene_selector_frozen_v1','Expected a trained frozen-C scene-head checkpoint')
    require(manifest['arguments']['mode'] in ('real','zero'),'Unsupported scene-head token mode')
    require(0<int(step)<=int(manifest['arguments']['steps']),'Invalid trained head step')
    require(manifest['source_sha256']['c_scene_selector.py']==SCENE_SHA,'Head implementation revision changed')
    require(manifest['bank_sha256']==plan.receipt['bank_sha256'],'Head/continuation bank mismatch')
    require(manifest['base_frozen'] is True and manifest['raw_status_input'] is False,'Head training input contract changed')
    for split,expected_rows in (('train',plan.manifest['train']),('tune',plan.manifest['validation'])):
        cache=manifest[f'{split}_cache_manifest']
        require(cache['status']=='completed','Scene head used an incomplete cache')
        require(cache['checkpoint_sha256']==C_SHA and cache['bank_sha256']==plan.receipt['bank_sha256'],
                'Scene head used different C/bank ancestry')
        require(all(cache['counts'][key]==value for key,value in EXPECTED_COUNTS.items()),'Head candidate count policy differs')
        require(cache['rows_sha256']==expected_rows['rows_sha256'],'Head cache population changed')
        require(cache['source_sha256']['c_scene_selector.py']==SCENE_SHA,'Head cache token source changed')
        require(cache['batch_size']==8,'Head cache was not produced with fixed B8')
    require(not set(manifest['train_cache_manifest']['sessions']) & set(manifest['tune_cache_manifest']['sessions']),
            'Head cache training/tune sessions overlap')


def load_trained_head(path,plan,helper):
    path=Path(path).resolve();digest=sha(path)
    payload=helper.load_internal_payload(path)
    require(sha(path)==digest,'Head checkpoint changed while loading')
    require({'head','step','manifest','result'}.issubset(payload),'Incomplete trained-head checkpoint')
    payload.pop('optimizer',None)
    manifest=payload['manifest']
    validate_head_manifest(manifest,plan,payload['step'])
    require(canonical(json.loads((path.parent/'manifest.json').read_text()))==canonical(manifest),
            'Head checkpoint/disk manifest mismatch')
    root=(path.parent/'source').resolve()
    for name,expected in manifest['source_sha256'].items():
        source=(root/name).resolve()
        require(source.is_relative_to(root) and sha(source)==expected,'Head source snapshot mismatch: '+name)
    require(payload['result']['step']==payload['step'],'Head checkpoint/result step mismatch')
    head=SceneResidualSelector(mode=manifest['arguments']['mode'])
    head.load_state_dict(payload['head'],strict=True)
    require(all(bool(torch.isfinite(v).all()) for v in head.state_dict().values()),'Nonfinite trained head tensor')
    return head,{'path':str(path),'sha256':digest,'step':int(payload['step']),
                 'manifest_sha256':sha(path.parent/'manifest.json'),'manifest':manifest,
                 'terminal':payload['step']==manifest['arguments']['steps'],'source_root':str(root)}


def configure_counts(c_model):
    public=c_model.base.base
    require(tuple(public.path_filter)==(128,20) and tuple(public.velocity_filter)==(64,10),
            'Original C count policy changed before continuation')
    before={k:(v.data_ptr(),v._version) for k,v in c_model.state_dict().items()}
    public.velocity_filter=(64,64)
    require(before=={k:(v.data_ptr(),v._version) for k,v in c_model.state_dict().items()},
            'Count configuration changed a tensor')


def build_datasets(plan,runtime,seed,eval_limit=0):
    require(seed==plan.manifest['arguments']['seed'],'Continuation must preserve original dataset seed')
    common=dict(base=plan.base,split_manifest=plan.split,ego_cache=plan.ego,
                goal_mode='selection',status_mode='zero',seed=seed)
    datasets=[];metadata={}
    for split,key,stride in (('train','train',1),('tune','validation',5)):
        record=plan.manifest[key]
        temporal_record=record['temporal_data']
        base=runtime.data.PlanDataset(**common,split=split,stride=stride,augment=split=='train')
        value=runtime.temporal_data.TemporalPlanDataset(base,history_mode='real',auxiliary=True,
            causal_status_root=Path(temporal_record['causal_status']['path']).parent,
            supervision_root=temporal_record['auxiliary']['directory'])
        actual=value.provenance()
        require(canonical(actual)==canonical(record),f'Original {split} data/auxiliary provenance changed')
        datasets.append(value);metadata[key]=actual
    train,tune=datasets
    require(len(train)==54810 and len(tune)==1998,'Original train/tune population changed')
    require(not np.intersect1d(train.rows,tune.rows).size,'Training and tune rows overlap')
    require(not set(metadata['train']['sessions']) & set(metadata['validation']['sessions']),'Session overlap')
    if eval_limit:
        keep=np.arange(min(eval_limit,len(tune)))
        subset=Subset(tune,keep.tolist());subset.rows=tune.rows[keep];tune=subset
    return train,tune,metadata


def optimizer_groups(model,lrs):
    public=model.model_c.base.base
    public_ids={id(p) for p in public.parameters()}
    backbone_ids={id(p) for p in public._backbone.parameters()}
    scene_ids={id(p) for p in model.head.parameters()}
    groups={key:[] for key in ('backbone','public_head','common_and_old_relative','scene_head')}
    for parameter in model.parameters():
        if not parameter.requires_grad:continue
        key=('scene_head' if id(parameter) in scene_ids else 'backbone' if id(parameter) in backbone_ids
             else 'public_head' if id(parameter) in public_ids else 'common_and_old_relative')
        groups[key].append(parameter)
    ids=[id(v) for group in groups.values() for v in group]
    require(len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad},
            'Optimizer groups overlap or omit learned parameters')
    require(all(groups.values()),'An optimizer group is empty')
    return [{'name':key,'params':values,'lr':lr} for (key,values),lr in zip(groups.items(),lrs)]


def lr_factor(step,steps,warmup):
    require(1<=step<=steps and warmup>=0,'Invalid schedule step')
    factor=min(step/max(warmup,1),1.)
    if step>warmup:factor*=.5*(1+math.cos(math.pi*(step-warmup)/max(steps-warmup,1)))
    return factor


def fixed_buffers(model,helper,expected):
    state=model.model_c.state_dict()
    actual={name:helper.tensor_sha(state[name]) for name in expected}
    require(actual==expected,'Continuation changed a fixed bank/geometry buffer')
    return actual


def checkpoint_payload(model,optimizer,step,epoch,seen,manifest,result):
    return {'schema':SCHEMA,'model_c':model.model_c.state_dict(),'head':model.head.state_dict(),
            'optimizer':optimizer.state_dict(),'step':step,'epoch':epoch,'seen':seen,
            'manifest':manifest,'result':result}


def snapshot_sources(directory,plan,head_ref,evaluator_source):
    files={'train_c_continuation.py':Path(__file__),
           'c_scene_selector.py':Path(__file__).with_name('c_scene_selector.py'),
           'evaluate_temporal_checkpoint.py':Path(evaluator_source)}
    destination=directory/'source';destination.mkdir()
    for name,path in files.items():shutil.copy2(path,destination/name)
    shutil.copytree(plan.source,directory/'original_c_source')
    shutil.copytree(head_ref['source_root'],directory/'frozen_head_source')
    return {name:sha(path) for name,path in files.items()}


def validate_training_args(args):
    require(min(args.steps,args.batch,args.eval_batch,args.eval_every)>0,'Invalid training dimensions')
    require(args.workers>=0 and args.warmup>=0 and args.eval_limit>=0 and args.training_rng_seed>=0,'Invalid runtime options')
    require(args.eval_batch==8,'Continuation diagnostics remain B8; deployment B1 is a separate evaluation')
    require(args.common_status is True,'Continuation must retain C common-perception conditioning')
    require((args.temperature,args.perception_weight,args.state_weight,args.occ_pos_weight,args.lane_pos_weight)
            ==(.1,.25,.1,4.,8.),'Predeclared original planning/perception/state loss weights changed')


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--checkpoint',required=True,help='Original terminal C, preserved unchanged')
    p.add_argument('--head-checkpoint',required=True,help='Explicit trained frozen-C head, real or zero token')
    p.add_argument('--run-dir',required=True);p.add_argument('--worktree');p.add_argument('--base')
    p.add_argument('--evaluator-source',required=True);p.add_argument('--gpu',type=int,choices=(0,1,4))
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--steps',type=int,default=4000);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--eval-batch',type=int,default=8);p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--training-rng-seed',type=int,default=2,help='New shuffle/dropout seed; no original cursor/RNG resume')
    p.add_argument('--backbone-lr',type=float,default=5e-6);p.add_argument('--public-head-lr',type=float,default=5e-5)
    p.add_argument('--common-lr',type=float,default=2e-4);p.add_argument('--scene-head-lr',type=float,default=5e-4)
    p.add_argument('--weight-decay',type=float,default=.01);p.add_argument('--warmup',type=int,default=100)
    p.add_argument('--eval-limit',type=int,default=0,help='Canary only; full original data provenance still verified')
    p.set_defaults(common_status=True,history_mode='real',temperature=.1,perception_weight=.25,
                   state_weight=.1,occ_pos_weight=4.,lane_pos_weight=8.)
    a=p.parse_args();validate_training_args(a)
    require(sha(Path(__file__).with_name('c_scene_selector.py'))==SCENE_SHA,'Scene capture/head source changed')
    helper=load_evaluator(a.evaluator_source)
    if a.audit_only:require(os.environ.get('CUDA_VISIBLE_DEVICES','')=='','CPU audit must hide all GPUs')
    else:helper.check_gpu(a.gpu)
    directory=Path(a.run_dir).resolve();require(not directory.exists(),'Run directory is immutable; use a new path')
    directory.mkdir(parents=True);started=time.time();torch.set_num_threads(4)
    try:
        plan=helper.inspect_checkpoint(a.checkpoint,worktree=a.worktree,base=a.base)
        require(plan.receipt['checkpoint_sha256']==C_SHA,'Continuation original C checkpoint pin mismatch')
        require(str(torch.__version__)==plan.manifest['torch'],'Original C runtime torch version changed')
        with helper.isolated_runtime(plan.source) as runtime:
            trainer=runtime.train_temporal_comparison
            runtime.train.seed_all(a.seed)
            c_model=helper.strict_load(plan,runtime)
            original_keys=list(c_model.state_dict())
            head,head_ref=load_trained_head(a.head_checkpoint,plan,helper)
            configure_counts(c_model)
            model=CSceneModel(c_model,head)
            require(list(model.model_c.state_dict())==original_keys,'C checkpoint tensor topology changed')
            require(all(torch.equal(value,plan.payload['model'][name]) for name,value in model.model_c.state_dict().items()),
                    'Wrapper/head construction modified original trained C tensors')
            train,tune,data_meta=build_datasets(plan,runtime,a.seed,a.eval_limit)
            source=snapshot_sources(directory,plan,head_ref,a.evaluator_source)
            if not a.audit_only:model.cuda()
            lrs=(a.backbone_lr,a.public_head_lr,a.common_lr,a.scene_head_lr)
            groups=optimizer_groups(model,lrs)
            optimizer=torch.optim.AdamW(groups,weight_decay=a.weight_decay)
            expected_fixed=plan.receipt['fixed_buffer_sha256']
            fixed_buffers(model,helper,expected_fixed)
            manifest={'schema':SCHEMA,'arguments':vars(a),'torch':str(torch.__version__),'numpy':str(np.__version__),
                'original_c_checkpoint':str(plan.checkpoint),'original_c_checkpoint_sha256':C_SHA,
                'original_c_receipt':plan.receipt,'original_c_training_steps':2000,
                'original_c_source_sha256':plan.manifest['source_sha256'],
                'frozen_head_checkpoint':head_ref,'head_mode':head.mode,'source_sha256':source,
                'bank':str(plan.bank),'bank_sha256':plan.receipt['bank_sha256'],
                **EXPECTED_COUNTS,'train':data_meta['train'],'validation':data_meta['validation'],
                'fixed_buffer_sha256':expected_fixed,'original_c_tensor_keys':original_keys,
                'optimizer':'new AdamW; weights continuation, no optimizer state/RNG resume',
                'optimizer_reason':'Original C terminal schedule had LR=0; scene head is a new jointly optimized module',
                'parameter_counts':{g['name']:sum(v.numel() for v in g['params']) for g in groups},
                'learning_rates':dict(zip((g['name'] for g in groups),lrs)),
                'batchnorm':'fixed running statistics, trainable affine','training_rng_seed':a.training_rng_seed,
                'data_order':'new independent DataLoader generator with training_rng_seed; no saved original cursor/RNG available',
                'loss':'unchanged original coarse/fine soft CE + .25*(occ BCE pos4 + lane BCE pos8) + .1 state SmoothL1',
                'fine_scores':'new final scene residual + old final scores; gradients through old C and captured tokens',
                'planner_status':'constant zero','old_relative_status':'constant zero','new_head_raw_status':False,
                'perception_status':'past-only causal4, existing common image attention query only',
                'goal':'old/new final scoring of completed bank candidates only','GT_to_model':False,
                'population':'original TRAIN203 54810 / TUNE37 1998; previously exposed through original C and head training',
                'validation':'repeated exploratory TUNE; every500 plus terminal; no untouched-test claim',
                'canary':bool(a.eval_limit or a.steps!=4000 or a.batch!=16),'audit_only':a.audit_only,
                'physical_gpu':None if a.audit_only else a.gpu,'pid':os.getpid()}
            helper.json_write(directory/'manifest.json',manifest)
            if a.audit_only:
                helper.json_write(directory/'result.json',{'status':'cpu_audit_passed','strict_load':True,
                    'data_provenance_exact':True,'fixed_buffers_unchanged':True,'elapsed_seconds':time.time()-started})
                return
            runtime.train.seed_all(a.training_rng_seed)
            generator=torch.Generator().manual_seed(a.training_rng_seed)
            loader=DataLoader(train,batch_size=a.batch,shuffle=True,num_workers=a.workers,pin_memory=True,
                              drop_last=False,generator=generator)
            iterator=iter(loader);epoch=0;seen=0
            with helper.route_monitor(c_model,True) as route_counts:
                initial=trainer.evaluate(model,tune,a,directory,0)
                for step in range(1,a.steps+1):
                    try:batch=next(iterator)
                    except StopIteration:
                        epoch+=1;train.set_epoch(epoch);iterator=iter(loader);batch=next(iterator)
                    seen+=len(batch['row'])
                    factor=lr_factor(step,a.steps,a.warmup)
                    for group,lr in zip(optimizer.param_groups,lrs):group['lr']=factor*lr
                    runtime.train.fixed_bn_train(model)
                    x=trainer.transfer(batch);inputs=trainer.selected_inputs(x,True)
                    require(set(inputs)==INPUT_KEYS,'GT/metadata or a status bypass entered model inputs')
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**inputs)
                    trainer.verify_bank(model,out)
                    require(out['candidate_tokens'].requires_grad,'Captured scene tokens were detached during continuation')
                    with torch.autocast('cuda',enabled=False):losses=trainer.all_losses(model,out,x,a)
                    require(bool(torch.isfinite(losses['loss'])),'Nonfinite continuation loss')
                    losses['loss'].backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                    optimizer.step()
                    if step==1 or step%10==0:
                        record={'step':step,'epoch':epoch,'seen':seen,'exposure':seen/len(train),
                            'seconds':time.time()-started,'grad_norm':float(norm),
                            'learning_rates':[g['lr'] for g in optimizer.param_groups],
                            'batch_rows_sha256':runtime.data.rows_sha(batch['row'].numpy()),
                            **{key:float(value.detach()) for key,value in losses.items()}}
                        with (directory/'train.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
                        print(json.dumps(record,allow_nan=False),flush=True)
                    if step%a.eval_every==0 or step==a.steps:
                        terminal=trainer.evaluate(model,tune,a,directory,step)
                        fixed_buffers(model,helper,expected_fixed)
                        payload=checkpoint_payload(model,optimizer,step,epoch,seen,manifest,terminal)
                        runtime.train.save_checkpoint(directory/f'step_{step:06d}.pth',payload)
                        runtime.train.save_checkpoint(directory/'last.pth',payload)
                        helper.json_write(directory/'progress.json',{'step':step,'result':terminal,'seconds':time.time()-started})
            helper.json_write(directory/'result.json',{'status':'completed','initial':initial,'terminal':terminal,
                'steps':a.steps,'seen':seen,'exposure':seen/len(train),'route_checks':route_counts,
                'fixed_buffers_unchanged':True,'elapsed_seconds':time.time()-started,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
    except BaseException:
        helper.json_write(directory/'failure.json',{'status':'failed','traceback':traceback.format_exc(),
                                                  'elapsed_seconds':time.time()-started})
        raise


if __name__=='__main__':main()
