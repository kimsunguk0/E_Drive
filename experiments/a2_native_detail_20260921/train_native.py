"""Four matched DEV continuations; native RGB detail is the within-family variable."""
from pathlib import Path
import argparse,contextlib,hashlib,json,os,subprocess,sys
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
for p in (ROOT,ROOT/'scripts',ROOT/'experiments/a2_progress_h4_20260920',HERE):sys.path.insert(0,str(p))
import train_progress as reference
nominal,legacy,trainer,mr=reference.nominal,reference.legacy,reference.trainer,reference.mr
from h4_data import H4StatusDataset,sha
from motiondrive_v2_training import tensor_state_sha256
from length_auxiliary import wrap_compute_loss
import motiondrive_v2_flip_augment as flip_api
from native_model import NativeDetailModel,ARMS,arm_config,load_parent,PREFIX
from native_data import NativeDataset,wrap_flip,KEY,CACHE

REPORT=ROOT/'reports/a2_native_detail_20260921';RUNS=ROOT/'work_dirs/a2_native_detail_20260921'
PARENT_RUN=ROOT/'work_dirs/a2_splitread_lanegeom_20260921/P-SPLITREAD-s1'
PARENT=PARENT_RUN/'ckpt_step6852.pth'
PARENT_SHA='8252ae12063b0e684899ae22cfac2f3a29fedc0ea06df41849bab3d47c2767f6'
UPDATES=10277;EVERY=3426;MICRO=8

def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)

@contextlib.contextmanager
def configure(arm,steps=UPDATES):
    keys=('INITIALIZER','HEAD_LR','BACKBONE_LR','CUDA_MEMORY_LIMIT_MIB','WORKERS','EVAL_BATCH')
    saved={k:getattr(legacy,k) for k in keys}
    class Factory:
        VALID_ARMS={a:0 for a in ARMS}
        def __new__(cls,config,*,arm):return NativeDetailModel(config,execution_config=arm_config(arm))
    with nominal.configure('A2-BASE-NOM',MICRO):
        legacy.SharedDynamicsMotionDriveV2=Factory
        legacy.CausalStatusDataset=lambda base:NativeDataset(H4StatusDataset(base),arm)
        legacy.INITIALIZER=PARENT;legacy.REPORT_DIR=REPORT
        legacy.HEAD_LR=1e-5;legacy.BACKBONE_LR=1e-6
        legacy.UPDATES=steps;legacy.EVAL_EVERY=EVERY if steps==UPDATES else steps
        legacy.CUDA_MEMORY_LIMIT_MIB=147456;legacy.WORKERS=8;legacy.EVAL_BATCH=8
        try:yield
        finally:
            for k,v in saved.items():setattr(legacy,k,v)

def declaration(arm,gpu,run,steps):
    if sha(PARENT)!=PARENT_SHA:raise ValueError('Parent checksum changed')
    pm=json.loads((PARENT_RUN/'manifest.json').read_text())
    if pm['status']!='completed' or pm['step']!=6852 or pm['nonfinite_count']:raise ValueError('Bad parent')
    d=json.loads((PARENT_RUN/'experiment.json').read_text())
    if d['full_fit']['enabled']:raise ValueError('FULL parent forbidden')
    cp=torch.load(PARENT,map_location='cpu',weights_only=False)
    for k in ('geometry','initial_load','extension_policy','full_transfer_policy','expected_shared_base_initial_sha256'):
        d.pop(k,None)
    d.update(name='a2_native_detail_20260921',arm=arm,physical_gpu=gpu,run_dir=str(run),smoke=steps!=UPDATES,
        native_execution_config=arm_config(arm),next_execution_config=arm_config(arm)['parent_next_execution_config'],
        execution_config=arm_config(arm)['parent_next_execution_config']['base_execution_config'],
        parent_updates=6852,stage_updates=steps,training_aux_in_state=False,
        parent_model_config=pm['model_config'],expected_initial_model_state_sha256=None,
        initializer={'path':str(PARENT),'checkpoint_sha256':PARENT_SHA,'model_state_sha256':tensor_state_sha256(cp['model']),
            'source':'completed DEV SplitRead; weights and buffers only'},
        expected_optimizer_groups=[{'name':'backbone','base_lr':1e-6},{'name':'head','base_lr':1e-5}],
        optimizer_native_group={'base_lr':5e-5,'all_new_parameters':True},
        question='Native RGB1152 detail versus same-canvas 768-down/up control, independently in motion or shared scene',
        control={'arm':arm[0]+'-LOW','parent_PREFIX':.14739373370951964,'primary':'native terminal minus family LOW terminal AND frozen parent'},
        comparison_contract={'same_parent_tensors':True,'same_family_new_parameters':True,'same_rows_baseline_pixels_jitter_flip':True,
            'same_updates_optimizer_schedule':True,'detail_pixels_intentionally_different':True,'FULL_leakage':False},
        automatic_followups=False,automatic_followup_training=False,automatic_FULL=False,
        interpretation='New high-detail branch starts at zero contribution. Within-family contrast isolates input detail at equal shape and graph.',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip())
    d['recipe'].update(updates=steps,warmup=100,head_lr=1e-5,backbone_lr=1e-6,new_detail_lr=5e-5,
        microbatch=MICRO,eval_batch=8,workers=8,eval_every=EVERY if steps==UPDATES else steps,
        eval_steps=[0,3426,6852,10277] if steps==UPDATES else [0,steps],
        schedule='fresh cosine over fixed stage horizon',optimizer='fresh AdamW',additional_train_exposure=steps*16/83700,
        motion_canvas='baseline768 retained; additional1152 in M arms',cuda_memory_limit_mib=147456)
    d['evaluations']={'primary':steps,'scheduled_steps':d['recipe']['eval_steps']}
    d['native_cache']={'path':str(CACHE),'manifest_sha256':sha(CACHE/'manifest.json'),'plan_sha256':sha(CACHE/'plan.json')}
    for p in HERE.glob('*.py'):d['source'][str(p.relative_to(ROOT))]=sha(p)
    d['source']={p:sha(ROOT/p) for p in d['source']}
    return d

def argv(run):
    a=legacy.trainer_argv(1,run,False);a[a.index('--warmup')+1]='100';a[a.index('--log-every')+1]='1';return a

def input_adapter(old):
    def inputs(batch,**kw):
        x=old(batch,**kw);x['high_images']=batch[KEY];return x
    return inputs

@contextlib.contextmanager
def runtime(arm,d,run,protocol):
    common_loss=trainer.compute_loss
    raw_probe,_=nominal.raw_datasets(False,1)
    probe=NativeDataset(H4StatusDataset(mr.MotionCanvasDataset(raw_probe,'native')),arm)
    probe_idx=np.random.default_rng(20260921).choice(len(probe),256,replace=False).tolist()
    probe_loader=DataLoader(Subset(probe,probe_idx),batch_size=8,num_workers=4)
    stages=[3426,6852,10277] if d['stage_updates']==UPDATES else [d['stage_updates']];ei=[0]
    with legacy.patched_runtime(arm,1,d,run,False):
        old={k:getattr(trainer,k) for k in ('_load_initial_model_state','_validate_experimental_runtime','model_inputs','evaluate','atomic_json','atomic_checkpoint','compute_loss')}
        old_flip=flip_api.flip_item;old_adam=torch.optim.AdamW;old_clip=torch.nn.utils.clip_grad_norm_
        refs=[];row_sha=hashlib.sha256();base_sha=hashlib.sha256();high_sha=hashlib.sha256();calls=[0];step=[0];groups=[];grad=[]
        def initialize(model,common,experiment=None):
            if tensor_state_sha256(common['model'])!=d['initializer']['model_state_sha256']:raise ValueError('Wrong parent state')
            if json.loads(json.dumps(model.config.to_dict()))!=d['parent_model_config']:raise ValueError('Base config changed')
            extras=load_parent(model,common['model']);refs.append(model)
            d['expected_initial_model_state_sha256']=tensor_state_sha256(model.state_dict())
            d['initial_load']={'parent_tensors_equal':True,'new_keys':extras,'optimizer_moments_restored':False,
                'parent_step':6852,'stage_step':0,'initial_state_sha256':d['expected_initial_model_state_sha256']}
            atomic(protocol,d);return {'native_initial_load':d['initial_load']}
        def validate(args,decl):
            assert args.warmup==100 and args.init==str(PARENT) and not args.resume
            args.warmup=200
            try:old['_validate_experimental_runtime'](args,decl)
            finally:args.warmup=100
        def enrich(m):
            m.update(execution_config=d['execution_config'],next_execution_config=d['next_execution_config'],
                native_execution_config=arm_config(arm),training_aux_in_state=False,parent_updates=6852,stage_updates=d['stage_updates'],
                optimizer_effective_groups=groups,
                stream_audit={'microcalls':calls[0],'rows_sha256':row_sha.hexdigest(),'baseline_augmentation_sha256':base_sha.hexdigest(),
                    'detail_pixels_sha256':high_sha.hexdigest(),'detail_pixels_expected_to_differ':True})
        def write_json(path,payload):
            path=Path(path)
            if path.name=='manifest.json':enrich(payload)
            old['atomic_json'](path,payload)
            if path.name=='final_eval.json' and payload.get('records'):
                s=payload['report']['step'];old['atomic_json'](run/f'predictions_step{s}.json',payload)
                old['atomic_json'](run/f'diagnostics_step{s}.json',nominal.diagnostics(payload['records']))
        def checkpoint(path,payload):enrich(payload['manifest']);old['atomic_checkpoint'](path,payload)
        adapt=input_adapter(old['model_inputs'])
        def inputs(batch,**kw):
            x=adapt(batch,**kw)
            if refs and refs[0].training:
                row_sha.update(batch['row'].detach().cpu().numpy().astype('<i8').tobytes())
                for key in ('images','history_images','motion_current','motion_history','provided_status5'):
                    v=x[key].detach()
                    if v.ndim>=4:v=v[...,::37,::41]
                    base_sha.update(v.float().cpu().contiguous().numpy().tobytes())
                high_sha.update(x['high_images'][...,::37,::41].detach().float().cpu().contiguous().numpy().tobytes());calls[0]+=1
            return x
        def evaluate(model,loader,device,precision,**kw):
            result=old['evaluate'](model,loader,device,precision,**kw)
            pr,rr=old['evaluate'](model,probe_loader,device,precision,time_input='nominal',detailed_records=True,
                nominal_history_seconds=model.config.nominal_history_seconds)
            s=stages[ei[0]];ei[0]+=1;atomic(run/f'train_probe_step{s}.json',{'report':pr,'records':rr,'indices':probe_idx})
            return result
        class DetailAdamW(old_adam):
            def __init__(self,params,*a,**kw):
                gs=[dict(g,params=list(g['params'])) for g in params]
                ps=[p for n,p in refs[0].named_parameters() if n.startswith(PREFIX)];ids={id(p) for p in ps}
                for g in gs:g['params']=[p for p in g['params'] if id(p) not in ids]
                gs.append({'params':ps,'lr':5e-5,'base_lr':5e-5})
                got=[id(p) for g in gs for p in g['params']]
                assert len(got)==len(set(got)) and set(got)=={id(p) for p in refs[0].parameters()}
                super().__init__(gs,*a,**kw)
                groups.extend({'name':n,'base_lr':g['base_lr'],'parameters':sum(p.numel() for p in g['params'])}
                              for n,g in zip(('backbone','head','native_detail'),self.param_groups))
                assert [g['base_lr'] for g in self.param_groups]==[1e-6,1e-5,5e-5]
                atomic(run/'optimizer_groups.json',groups)
            def step(self,*a,**kw):
                result=super().step(*a,**kw);step[0]+=1;return result
        def clip(ps,*a,**kw):
            ps=list(ps);s=step[0]+1
            if s<=5 or s in (100,500,3426,6852,10277):
                watch=('native_projection.weight','native_motion.projections.0.weight','native_motion.correlation_fuse.0.weight',
                       'scene_encoder.key_proj.0.weight','backbone_fpn.layer1.0.conv1.weight')
                grad.append({'step':s,'unclipped_grad_norms':{n:float(p.grad.norm()) if p.grad is not None else None
                    for n,p in refs[0].named_parameters() if n in watch}});atomic(run/'gradient_audit.json',grad)
            return old_clip(ps,*a,**kw)
        trainer._load_initial_model_state=initialize;trainer._validate_experimental_runtime=validate
        trainer.atomic_json=write_json;trainer.atomic_checkpoint=checkpoint;trainer.model_inputs=inputs;trainer.evaluate=evaluate
        trainer.compute_loss=wrap_compute_loss(common_loss,.25)
        flip_api.flip_item=wrap_flip(old_flip,arm);torch.optim.AdamW=DetailAdamW;torch.nn.utils.clip_grad_norm_=clip
        try:yield
        finally:
            flip_api.flip_item=old_flip;torch.optim.AdamW=old_adam;torch.nn.utils.clip_grad_norm_=old_clip
            for k,v in old.items():setattr(trainer,k,v)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--gpu',type=int,choices=range(4),required=True)
    ap.add_argument('--run-dir',required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise ValueError('Wrong physical GPU')
    torch.set_num_threads(4);steps=5 if args.smoke else UPDATES;run=Path(args.run_dir).resolve()
    if run.exists() and any(run.iterdir()):raise ValueError('Refuse run overwrite')
    protocol=REPORT/f"protocol_{args.arm}{'_smoke' if args.smoke else ''}.json"
    if protocol.exists():raise ValueError('Refuse protocol overwrite')
    with configure(args.arm,steps):
        d=declaration(args.arm,args.gpu,run,steps);atomic(protocol,d)
        if not args.smoke:
            checks=json.loads((REPORT/'smoke_and_reload.json').read_text());assert checks['status']=='passed'
            for p,h in checks['source'].items():
                if sha(ROOT/p)!=h:raise ValueError('Source changed after preflight: '+p)
        try:
            with runtime(args.arm,d,run,protocol):trainer.run_training(argv(run),experiment=d)
        except BaseException as exc:
            p=run/'manifest.json'
            if p.exists():m=json.loads(p.read_text());m.update(status='failed',error_type=type(exc).__name__,error=str(exc));atomic(p,m)
            raise
        atomic(run/'experiment.json',d)

if __name__=='__main__':main()
