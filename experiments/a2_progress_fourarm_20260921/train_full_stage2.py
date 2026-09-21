"""Transfer one completed DEV-selected recipe to a copy of the submitted FULL terminal."""
from pathlib import Path
import argparse,contextlib,copy,json,math,os,subprocess,sys
import torch
from train_fourarm import (ROOT,HERE,REPORT,ARMS,nominal,legacy,trainer,mr,atomic,
    sha,arm_config,FourArmModel,load_dev_parent,tensor_state_sha256,vector_wrapper,length_wrapper)
sys.path.insert(0,str(ROOT/'experiments/a2_progress_full_20260921'))
from full_data import FullH4StatusDataset
from nominal_data import make_union

PARENT_RUN=ROOT/'work_dirs/a2_progress_full_20260921/A2-H4-PROGRESS-FULL-s1'
PARENT=PARENT_RUN/'ckpt_step24931.pth'
PARENT_SHA='dae99f86f29e92296b1d036db3f7933d41bf9a5c3dd3af5153416a8eaa676b14'
OUT=ROOT/'reports/a2_progress_fourarm_full_20260921'
RUNS=ROOT/'work_dirs/a2_progress_fourarm_full_20260921'
UPDATES=4156

def declaration(arm,gpu,run,smoke):
    decision=json.loads((REPORT/'decision.json').read_text())
    if decision['status']!='DEV_complete' or decision['candidate_for_FULL']!=arm:
        raise ValueError('No completed DEV selection for this single arm')
    if sha(PARENT)!=PARENT_SHA:raise ValueError('Preserved official FULL bytes changed')
    selected=json.loads((ROOT/f'work_dirs/a2_progress_fourarm_20260921/{arm}-s1/experiment.json').read_text())
    if selected['full_fit']['enabled']:raise ValueError('Selection must come from independent DEV lineage')
    full_manifest=json.loads((PARENT_RUN/'manifest.json').read_text())
    assert full_manifest['status']=='completed' and full_manifest['step']==24931
    cp=torch.load(PARENT,map_location='cpu',weights_only=False)
    d=copy.deepcopy(json.loads((PARENT_RUN/'experiment.json').read_text()))
    train,tune=nominal.raw_datasets(True,1)
    assert len(train)==101520 and len(tune)==1998 and math.ceil(3426*len(train)/83700)==UPDATES
    d.update(name='a2_progress_fourarm_full_20260921',arm=arm,execution_config=arm_config(arm),
        physical_gpu=gpu,run_dir=str(run),smoke=smoke,parent_updates=24931,
        stage_updates=5 if smoke else UPDATES,parent_model_config=full_manifest['model_config'],
        initializer={'path':str(PARENT),'checkpoint_sha256':PARENT_SHA,
            'model_state_sha256':tensor_state_sha256(cp['model']),
            'source':'preserved submitted FULL terminal; no DEV checkpoint loaded'},
        expected_initial_model_state_sha256=None,
        expected_optimizer_groups=[{'name':'backbone','base_lr':1e-6},{'name':'head','base_lr':1e-5}],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip(),
        dev_selection={'decision':str(REPORT/'decision.json'),'sha256':sha(REPORT/'decision.json'),
            'arm':arm,'terminal_PREFIX':decision['primary']['PREFIX'],'DEV_tensor_imported':False},
        control={'role':'FULL final fit continuation, not unseen validation'},
        full_fit={'enabled':True,'rows':101520,'unique_scenes':376,'evaluation_is_in_fit':True,
            'not_terminal_continuation':False,'checkpoint_selection':'fixed stage terminal4156',
            'schedule_transfer':'ceil(3426*101520/83700)','DEV_weights_teacher_features_imported':False,
            'FULL_artifacts_must_not_return_to_DEV':True},
        evaluations={'scheduled_steps':[5] if smoke else [4156],'role':'in-fit diagnostic only',
            'checkpoint_selection_from_evaluation':False},
        automatic_followups={'raw_parity':True,'test_inference':True,'submission_package':True,
            'additional_training':False,'official_upload_automated':False})
    d['recipe'].update(updates=5 if smoke else UPDATES,eval_every=5 if smoke else UPDATES,
        warmup=100,head_lr=1e-5,backbone_lr=1e-6,optimizer='fresh AdamW',
        schedule='new stage-local cosine horizon',interval_auxiliary=arm_config(arm)['interval_auxiliary'],
        interval_length_auxiliary_lambda=0 if arm=='P-VECTOR' else .25,
        interval_vector_auxiliary_lambda=.25 if arm=='P-VECTOR' else 0,
        checkpoint_selection='fixed terminal',additional_train_exposure=4156*16/101520)
    d.pop('initial_load',None)
    for p in (HERE/'arm_model.py',HERE/'fine_motion.py',Path(__file__),HERE/'infer_fourarm.py'):
        d['source'][str(p.relative_to(ROOT))]=sha(p)
    d['source']={p:sha(ROOT/p) for p in d['source']}
    return d

@contextlib.contextmanager
def configure(arm,smoke):
    import motiondrive_v2_data as data_api
    original=data_api.MotionDriveDataset
    class Factory:
        VALID_ARMS={a:0 for a in ARMS}
        def __new__(cls,config,*,arm):return FourArmModel(config,execution_config=arm_config(arm))
    saved={k:getattr(legacy,k) for k in ('INITIALIZER','HEAD_LR','BACKBONE_LR')}
    with nominal.configure('A2-FULL-NOM',8):
        legacy.SharedDynamicsMotionDriveV2=Factory;legacy.CausalStatusDataset=FullH4StatusDataset
        legacy.INITIALIZER=PARENT;legacy.REPORT_DIR=OUT
        legacy.UPDATES=5 if smoke else UPDATES;legacy.EVAL_EVERY=legacy.UPDATES
        legacy.HEAD_LR=1e-5;legacy.BACKBONE_LR=1e-6
        def dataset(**kwargs):return make_union(original,kwargs) if kwargs.get('split')=='train' else original(**kwargs)
        data_api.MotionDriveDataset=dataset
        try:yield
        finally:
            data_api.MotionDriveDataset=original
            for k,v in saved.items():setattr(legacy,k,v)

@contextlib.contextmanager
def runtime(arm,d,run,protocol):
    common_loss=trainer.compute_loss
    with legacy.patched_runtime(arm,1,d,run,False):
        names=('_load_initial_model_state','_validate_experimental_runtime','atomic_checkpoint','atomic_json','compute_loss')
        old={n:getattr(trainer,n) for n in names}
        def initialize(model,common,experiment=None):
            assert tensor_state_sha256(common['model'])==d['initializer']['model_state_sha256']
            assert json.loads(json.dumps(model.config.to_dict()))==d['parent_model_config']
            extras=load_dev_parent(model,common['model'])
            d['expected_initial_model_state_sha256']=tensor_state_sha256(model.state_dict())
            d['initial_load']={'strict_all_FULL_parent_tensors':True,'new_keys':extras,
                'parent_step':24931,'stage_step':0,'DEV_weights_imported':False,'fresh_optimizer':True}
            atomic(protocol,d);return {'FULL_stage2_initial_load':d['initial_load']}
        def validate(args,value):
            assert args.warmup==100 and args.init==str(PARENT) and not args.resume
            args.warmup=200
            try:old['_validate_experimental_runtime'](args,value)
            finally:args.warmup=100
        def enrich(m):m.update(execution_config=arm_config(arm),parent_updates=24931,stage_updates=d['stage_updates'])
        def checkpoint(path,payload):
            enrich(payload['manifest']);old['atomic_checkpoint'](path,payload)
        def write_json(path,payload):
            if Path(path).name=='manifest.json':enrich(payload)
            old['atomic_json'](path,payload)
            if Path(path).name=='final_eval.json' and payload.get('records'):
                old['atomic_json'](run/f"predictions_step{payload['report']['step']}.json",payload)
        selected=(vector_wrapper if arm=='P-VECTOR' else length_wrapper)(common_loss,.25)
        def loss(*args,**kw):
            value,parts=selected(*args,**kw)
            assert ('plan_interval_vector' in parts)==(arm=='P-VECTOR')
            assert ('plan_interval_length' in parts)!=(arm=='P-VECTOR')
            return value,parts
        trainer._load_initial_model_state=initialize;trainer._validate_experimental_runtime=validate
        trainer.atomic_checkpoint=checkpoint;trainer.atomic_json=write_json;trainer.compute_loss=loss
        try:yield
        finally:
            for n,v in old.items():setattr(trainer,n,v)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True)
    ap.add_argument('--gpu',type=int,choices=range(4),required=True);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.gpu)
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    suffix='-smoke' if args.smoke else ''
    run=RUNS/f'{args.arm}-FULL-STAGE2-s1{suffix}'
    if run.exists() and any(run.iterdir()):raise ValueError('Refuse run overwrite')
    protocol=OUT/f'protocol_{args.arm}{suffix}.json'
    if protocol.exists():raise ValueError('Refuse protocol overwrite')
    d=declaration(args.arm,args.gpu,run,args.smoke);atomic(protocol,d)
    if not args.smoke:
        receipt=json.loads((OUT/'smoke_and_export.json').read_text())
        assert receipt['status']=='passed' and receipt['arm']==args.arm
        for p,h in receipt['source'].items():assert sha(ROOT/p)==h
    try:
        with configure(args.arm,args.smoke),runtime(args.arm,d,run,protocol):
            a=legacy.trainer_argv(1,run,False);a[a.index('--warmup')+1]='100';a[a.index('--log-every')+1]='10'
            trainer.run_training(a,experiment=d)
    except BaseException as exc:
        p=run/'manifest.json'
        if p.exists():
            m=json.loads(p.read_text());m.update(status='failed',error_type=type(exc).__name__,error=str(exc));atomic(p,m)
        raise
    atomic(run/'experiment.json',d)
if __name__=='__main__':main()
