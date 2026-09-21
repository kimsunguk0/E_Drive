"""Matched, weights-only DEV continuation. No FULL tensors enter this launcher."""
from pathlib import Path
import argparse, contextlib, copy, hashlib, json, os, subprocess, sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
for p in (ROOT,ROOT/'scripts',ROOT/'experiments/a2_progress_h4_20260920',HERE):sys.path.insert(0,str(p))
import train_progress as parent_code
from arm_model import FourArmModel, ARMS, arm_config, load_dev_parent
from motiondrive_v2_training import tensor_state_sha256
from h4_data import H4StatusDataset,sha
nominal,legacy,trainer,mr=parent_code.nominal,parent_code.legacy,parent_code.trainer,parent_code.mr
sys.path.insert(0,str(ROOT/'experiments/a2_pro_review_20260921/reference'))
from interval_vector_auxiliary import wrap_compute_loss as vector_wrapper,vector_loss
from length_auxiliary import wrap_compute_loss as length_wrapper,length_loss as interval_length_loss

REPORT=ROOT/'reports/a2_progress_fourarm_20260921'
RUNS=ROOT/'work_dirs/a2_progress_fourarm_20260921'
PARENT_RUN=ROOT/'work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1'
PARENT=PARENT_RUN/'ckpt_step20554.pth'
PARENT_SHA='18128b1af6624333a00baa493a0601459804e8dfb660bf4dd7dbf60f381f2958'
UPDATES=3426
EVERY=1142

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

@contextlib.contextmanager
def configure(arm,steps=UPDATES):
    saved={k:getattr(legacy,k) for k in ('INITIALIZER','HEAD_LR','BACKBONE_LR')}
    class Factory:
        VALID_ARMS={a:0 for a in ARMS}
        def __new__(cls,config,*,arm):
            return FourArmModel(config,execution_config=arm_config(arm))
    with nominal.configure('A2-BASE-NOM',8):
        legacy.SharedDynamicsMotionDriveV2=Factory
        legacy.CausalStatusDataset=H4StatusDataset
        legacy.REPORT_DIR=REPORT
        legacy.INITIALIZER=PARENT
        legacy.HEAD_LR=1e-5;legacy.BACKBONE_LR=1e-6
        legacy.UPDATES=steps;legacy.EVAL_EVERY=EVERY if steps==UPDATES else steps
        try:yield
        finally:
            for k,v in saved.items():setattr(legacy,k,v)

def declaration(arm,gpu,run,steps):
    if sha(PARENT)!=PARENT_SHA:raise ValueError('DEV parent bytes changed')
    pm=json.loads((PARENT_RUN/'manifest.json').read_text())
    if pm['status']!='completed' or pm['step']!=20554 or pm['nonfinite_count']!=0:
        raise ValueError('Incomplete DEV parent')
    d=json.loads((PARENT_RUN/'experiment.json').read_text())
    if d['full_fit']['enabled']:raise ValueError('FULL is forbidden as DEV parent')
    cp=torch.load(PARENT,map_location='cpu',weights_only=False)
    d.update(name='a2_progress_fourarm_20260921',arm=arm,physical_gpu=gpu,run_dir=str(run),
        execution_config=arm_config(arm),parent_updates=20554,stage_updates=steps,
        full_lineage_used=False,smoke=steps!=UPDATES,
        parent_model_config=pm['model_config'],fresh_optimizer=True,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip(),
        initializer={'path':str(PARENT),'checkpoint_sha256':PARENT_SHA,
                     'model_state_sha256':tensor_state_sha256(cp['model']),
                     'source':'complete DEV H4-PROGRESS terminal; weights/buffers only'},
        expected_initial_model_state_sha256=None,
        expected_optimizer_groups=[{'name':'backbone','base_lr':1e-6},{'name':'head','base_lr':1e-5}],
        control={'arm':'P-CTRL','parent_PREFIX':.15117885989032362,
                 'primary':'terminal arm minus terminal CTRL AND frozen DEV parent'},
        question='One matched stage-2 screen of vector auxiliary, fine temporal read, or shared native history',
        comparison_contract={'same_parent_tensors':True,'same_sample_and_augmentation_stream':True,
            'same_new_updates':True,'same_input_status':True,'no_FULL_leakage':True},
        automatic_followup_training=False,
        full_transfer_policy={'requires_completed_DEV_decision':True,'one_selected_arm_only':True,
            'parent':'preserved submitted FULL terminal copy','updates':4156,'warmup':100,
            'checkpoint_selection':'terminal; old V0 in-fit','upload':False})
    d['recipe'].update(updates=steps,warmup=100,head_lr=1e-5,backbone_lr=1e-6,
        eval_every=EVERY if steps==UPDATES else steps,
        eval_steps=[0,1142,2284,3426] if steps==UPDATES else [0,steps],
        optimizer='fresh AdamW, no parent moment/cursor/RNG restore',
        schedule='new stage-local cosine horizon; no automatic extension',
        interval_auxiliary=d['execution_config']['interval_auxiliary'],auxiliary_lambda=.25,
        interval_length_auxiliary_lambda=0 if arm=='P-VECTOR' else .25,
        interval_vector_auxiliary_lambda=.25 if arm=='P-VECTOR' else 0,
        initial_eval='new same-checkpoint DEV evaluation before optimization; stored separately',
        additional_train_exposure=steps*16/83700)
    d['planner']['fine_read']=arm=='P-FINE'
    d['scene_extension']['history_feature_source']=d['execution_config']['scene_history_feature_source']
    for p in [*HERE.glob('*.py'),ROOT/'experiments/a2_pro_review_20260921/reference/interval_vector_auxiliary.py']:
        d['source'][str(p.relative_to(ROOT))]=sha(p)
    d['source']={p:sha(ROOT/p) for p in d['source']}
    d.pop('initial_load',None)
    return d

def argv(run):
    a=legacy.trainer_argv(1,run,False)
    a[a.index('--warmup')+1]='100'
    # Full rolling sample digest at terminal as well as every other update.
    a[a.index('--log-every')+1]='1'
    return a

@contextlib.contextmanager
def runtime(arm,d,run,protocol):
    original_common=trainer.compute_loss
    # Build before legacy patches the dataset factory: the probe must be an
    # unaugmented train subset, never the augmented training-factory branch.
    from torch.utils.data import DataLoader,Subset
    raw_probe,_=nominal.raw_datasets(False,1)
    probe=H4StatusDataset(mr.MotionCanvasDataset(raw_probe,'native'))
    probe_idx=np.random.default_rng(20260921).choice(len(probe),256,replace=False).tolist()
    probe_loader=DataLoader(Subset(probe,probe_idx),batch_size=8,num_workers=4)
    eval_steps=[1142,2284,3426] if d['stage_updates']==UPDATES else [d['stage_updates']]
    eval_index=[0]
    with legacy.patched_runtime(arm,1,d,run,False):
        old={n:getattr(trainer,n) for n in ('_load_initial_model_state','_validate_experimental_runtime',
             'atomic_json','atomic_checkpoint','model_inputs','evaluate','compute_loss')}
        row_digest=hashlib.sha256();augmentation_digest=hashlib.sha256();microcalls=[0]
        model_ref=[];gradient_rows=[]
        def initialize(model,common,experiment=None):
            if tensor_state_sha256(common['model'])!=d['initializer']['model_state_sha256']:
                raise ValueError('Parent tensor hash mismatch')
            if json.loads(json.dumps(model.config.to_dict()))!=d['parent_model_config']:
                raise ValueError('Parent geometry differs')
            extras=load_dev_parent(model,common['model'])
            d['expected_initial_model_state_sha256']=tensor_state_sha256(model.state_dict())
            d['initial_load']={'all_parent_tensors_equal':True,'extra_keys':extras,
                'measured_model_state_sha256':d['expected_initial_model_state_sha256'],
                'parent_step':20554,'stage_step':0,'optimizer_moments_restored':False}
            model_ref.append(model);atomic(protocol,d)
            return {'fourarm_initial_load':d['initial_load']}
        def validate(args,value):
            assert args.warmup==100 and args.init==str(PARENT) and not args.resume
            args.warmup=200
            try:old['_validate_experimental_runtime'](args,value)
            finally:args.warmup=100
        def enrich_manifest(payload):
            payload.update(execution_config=arm_config(arm),parent_updates=20554,
                           stage_updates=d['stage_updates'])
            payload['stream_audit']={'microcalls':microcalls[0],'rows_sha256':row_digest.hexdigest(),
                'augmentation_fingerprint_sha256':augmentation_digest.hexdigest(),
                'fingerprint':'ordered rows + sampled normalized current/history/motion pixels + status; deterministic row/epoch augmentation'}
        def write_json(path,payload):
            path=Path(path)
            if path.name=='manifest.json':enrich_manifest(payload)
            old['atomic_json'](path,payload)
            if path.name=='final_eval.json' and payload.get('records'):
                step=payload['report']['step']
                old['atomic_json'](run/f'predictions_step{step}.json',payload)
                old['atomic_json'](run/f'diagnostics_step{step}.json',nominal.diagnostics(payload['records']))
        def save_checkpoint(path,payload):
            enrich_manifest(payload['manifest'])
            old['atomic_checkpoint'](path,payload)
        def inputs(batch,**kwargs):
            x=old['model_inputs'](batch,**kwargs)
            if model_ref and model_ref[0].training:
                row_digest.update(batch['row'].detach().cpu().numpy().astype('<i8').tobytes())
                for k in ('images','history_images','motion_current','motion_history','provided_status5'):
                    v=x[k].detach()
                    if v.ndim>=4:v=v[...,::37,::41]
                    augmentation_digest.update(v.float().cpu().contiguous().numpy().tobytes())
                microcalls[0]+=1
            return x
        selected=(vector_wrapper if arm=='P-VECTOR' else length_wrapper)(original_common,.25)
        def loss(outputs,batch,weights,**kwargs):
            total,parts=selected(outputs,batch,weights,**kwargs)
            if ('plan_interval_vector' in parts)!=(arm=='P-VECTOR'):
                raise ValueError('Wrong auxiliary installed')
            if ('plan_interval_length' in parts)==(arm=='P-VECTOR'):
                raise ValueError('LENGTH/VECTOR were stacked or omitted')
            with torch.no_grad():
                parts['monitor_vector']=vector_loss(outputs,batch,kwargs.get('normalizers'))
                parts['monitor_length']=interval_length_loss(outputs,batch,kwargs.get('normalizers'))
            return total,parts
        def evaluate(model,loader,device,precision,**kwargs):
            result=old['evaluate'](model,loader,device,precision,**kwargs)
            # Identical fixed train probe, no augmentation, no adaptation.
            pr,records=old['evaluate'](model,probe_loader,device,precision,time_input='nominal',
                detailed_records=True,nominal_history_seconds=model.config.nominal_history_seconds)
            step=eval_steps[eval_index[0]];eval_index[0]+=1
            atomic(run/f'train_probe_step{step}.json',{'report':pr,'records':records,'indices':probe_idx})
            return result
        trainer._load_initial_model_state=initialize
        trainer._validate_experimental_runtime=validate
        trainer.atomic_json=write_json;trainer.atomic_checkpoint=save_checkpoint
        trainer.model_inputs=inputs;trainer.compute_loss=loss;trainer.evaluate=evaluate
        original_clip=torch.nn.utils.clip_grad_norm_
        def clip(params,*a,**kw):
            params=list(params)
            step=microcalls[0]//2
            if model_ref and (step<=5 or step in (100,1142,2284,3426)):
                watched=('planner.fine_read.attention.in_proj_weight','planner.fine_read.attention.out_proj.weight',
                    'motion_encoder.correlation_fuse.0.weight','backbone_fpn.layer1.0.conv1.weight')
                gradient_rows.append({'step':step,'unclipped_grad_norms':{n:float(p.grad.norm()) if p.grad is not None else None
                    for n,p in model_ref[0].named_parameters() if n in watched}})
                atomic(run/'gradient_audit.json',gradient_rows)
            return original_clip(params,*a,**kw)
        torch.nn.utils.clip_grad_norm_=clip
        try:yield
        finally:
            torch.nn.utils.clip_grad_norm_=original_clip
            for n,v in old.items():setattr(trainer,n,v)

def main():
    ap=argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--arm',choices=ARMS,required=True)
    ap.add_argument('--gpu',type=int,choices=range(4),required=True)
    ap.add_argument('--run-dir',required=True)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise ValueError('Wrong GPU visibility')
    torch.set_num_threads(4);steps=5 if args.smoke else UPDATES
    run=Path(args.run_dir).resolve()
    if run.exists() and any(run.iterdir()):raise ValueError('Refuse run overwrite')
    protocol=REPORT/f"protocol_{args.arm}{'_smoke' if args.smoke else ''}.json"
    if protocol.exists():raise ValueError('Refuse protocol overwrite')
    with configure(args.arm,steps):
        d=declaration(args.arm,args.gpu,run,steps)
        atomic(protocol,d)
        if not args.smoke:
            checked=json.loads((REPORT/'smoke_and_reload.json').read_text())
            if checked['status']!='passed':raise ValueError('Production smoke/reload incomplete')
            for p,digest in checked['source'].items():
                if sha(ROOT/p)!=digest:raise ValueError('Source changed after smoke: '+p)
        try:
            with runtime(args.arm,d,run,protocol):trainer.run_training(argv(run),experiment=d)
        except BaseException as exc:
            path=run/'manifest.json'
            if path.exists():
                m=json.loads(path.read_text());m.update(status='failed',error_type=type(exc).__name__,error=str(exc));atomic(path,m)
            raise
    atomic(run/'experiment.json',d)

if __name__=='__main__':main()
