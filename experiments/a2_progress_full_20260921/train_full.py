"""Transfer the selected PROGRESS recipe from its public initializer to FULL."""
from pathlib import Path
import argparse,contextlib,copy,json,math,os,subprocess,sys
import torch
ROOT=Path('/NHNHOME/data/sukim/adcl');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/a2_progress_h4_20260920'))
import train_progress as dev
sys.path.insert(0,str(HERE))
from full_data import FullH4StatusDataset,CACHE,sha
from progress_model import H4ProgressModel,PROGRESS,load_initializer
from motiondrive_v2_training import tensor_state_sha256
from nominal_data import make_union

ARM='A2-H4-PROGRESS-FULL';REPORT=ROOT/'reports/a2_progress_full_20260921'
RUNS=ROOT/'work_dirs/a2_progress_full_20260921';UPDATES=24931;EVERY=6345
base,nominal,legacy,trainer,mr=dev.base,dev.nominal,dev.legacy,dev.trainer,dev.mr
DEV_PROTOCOL=dev.REPORT/'protocol_A2-H4-PROGRESS_s1.json'

class FullModel(H4ProgressModel):
    VALID_ARMS={**H4ProgressModel.VALID_ARMS,ARM:0}
    def __init__(self,config,*,arm=ARM):
        assert arm==ARM
        super().__init__(config,arm=PROGRESS)

def make_protocol(gpu,smoke):
    value=copy.deepcopy(json.loads(DEV_PROTOCOL.read_text()))
    for path,checksum in value['source'].items():
        assert sha(ROOT/path)==checksum,('Selected DEV source changed',path)
    record=base.ensure_fresh_initializer();assert record==value['initializer']
    train,tune=nominal.raw_datasets(True,1)
    assert len(train)==101520 and len(set(train.scene_names[train.rows]))==376
    assert UPDATES==math.ceil(20554*len(train)/83700)
    value.update(name='a2_progress_full_20260921',arm=ARM,physical_gpu=gpu,smoke=smoke,
        question='Final fit of selected H4-PROGRESS recipe on all unique official train scenes',
        train_data={'rows':len(train),'rows_sha256':legacy.rows_sha(train.rows),'scenes':376},
        tune_data={'rows':len(tune),'rows_sha256':legacy.rows_sha(tune.rows),'scenes':37,'role':'in-fit diagnostic only'},
        full_fit={'enabled':True,'logical_splits':['train','tune','val'],'rows':101520,'unique_scenes':376,
            'evaluation_is_in_fit':True,'checkpoint_selection':'fixed terminal step24931',
            'schedule_transfer':'ceil(20554*101520/83700)','not_terminal_continuation':True,
            'DEV_or_FULL_weights_teacher_features_imported':False,'FULL_artifacts_must_not_return_to_DEV':True},
        dev_selection={'protocol':str(DEV_PROTOCOL),'protocol_sha256':sha(DEV_PROTOCOL),
            'PREFIX':.15117885989032362,'step':20554,'single_model':True,
            'limitations':'nonstop regression vs matched DIRECT retained in terminal review; user explicitly requested FULL'},
        comparison_contract={'same_graph_initializer_loss_BN_precision_augmentation_as_selected_DEV':True,
            'same_train_rows_or_sample_stream_as_DEV':False,'same_exposure_count_rounded':True},
        control={'role':'final fit, no unseen local validation or new matched architecture comparison'},
        evaluations={'primary':2 if smoke else UPDATES,'scheduled_steps':[2] if smoke else [6345,12690,19035,24931],
            'role':'in-fit diagnostic','checkpoint_selection_from_evaluation':False},
        nominal_input={'manifest':str(CACHE/'manifest.json'),'sha256':sha(CACHE/'manifest.json'),
            'producer_sha256':value['nominal_input']['producer_sha256'],'relative_fit_frames':[-10,-5,-2,-1,0],
            'allowed_splits':['train','tune','val'],'input_field':'provided_status5',
            'state_history_supervision':'original real-time definition retained','all_fit_times_have_consumed_RGB':True},
        automatic_followups={'raw_parity':True,'test_inference':True,'submission_package':True,
            'additional_training':False,'official_upload_automated':False},
        user_request='2026-09-21: strongest current candidate FULL for one submission')
    value.pop('initial_load',None);value.pop('control_source_checks',None)
    value['recipe'].update(updates=2 if smoke else UPDATES,eval_every=2 if smoke else EVERY,
        eval_role='in-fit diagnostic only; not generalization performance')
    value['expected_initial_model_state_sha256']='75eec301bdc2abb533e0f0fa4ffa73c5b2e4340b7119a84f11b2d1b2212ed200'
    for path in (Path(__file__),HERE/'full_data.py',HERE/'build_full_cache.py'):
        value['source'][str(path.relative_to(ROOT))]=sha(path)
    value['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    return value

@contextlib.contextmanager
def configure():
    import motiondrive_v2_data as data_api
    original=data_api.MotionDriveDataset
    with dev.configure():
        legacy.SharedDynamicsMotionDriveV2=FullModel;legacy.CausalStatusDataset=FullH4StatusDataset
        legacy.UPDATES=UPDATES;legacy.EVAL_EVERY=EVERY;legacy.TRAIN_ROWS=101520;legacy.REPORT_DIR=REPORT
        def dataset(**kwargs):return make_union(original,kwargs) if kwargs.get('split')=='train' else original(**kwargs)
        data_api.MotionDriveDataset=dataset
        try:yield
        finally:data_api.MotionDriveDataset=original

@contextlib.contextmanager
def runtime(declared,run,smoke,protocol):
    with legacy.patched_runtime(ARM,1,declared,run,smoke):
        old_loader,old_json=trainer._load_initial_model_state,trainer.atomic_json
        def load_initial(model,common,experiment=None):
            assert tensor_state_sha256(common['model'])==declared['initializer']['model_state_sha256']
            extras=load_initializer(model,common['model'])
            state=tensor_state_sha256(model.state_dict())
            assert state==declared['expected_initial_model_state_sha256']
            declared['initial_load']={'strict_assembled_state':True,'model_state_sha256':state,
                'same_as_selected_DEV_initial_state':True,'DEV_terminal_loaded':False,'extra_keys':extras}
            protocol.write_text(json.dumps(declared,indent=2)+'\n');return {'FULL_public_initial_load':declared['initial_load']}
        def atomic_json(path,payload):
            old_json(path,payload)
            if Path(path).name=='final_eval.json' and payload.get('records'):
                step=payload['report']['step'];old_json(run/f'predictions_step{step}.json',payload)
                old_json(run/f'diagnostics_step{step}.json',nominal.diagnostics(payload['records']))
        trainer._load_initial_model_state,trainer.atomic_json=load_initial,atomic_json
        try:yield
        finally:trainer._load_initial_model_state,trainer.atomic_json=old_loader,old_json

def main():
    ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--gpu',type=int,choices=range(4),required=True)
    ap.add_argument('--run-dir',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.gpu)
    torch.set_num_threads(4);run=Path(args.run_dir).resolve();REPORT.mkdir(parents=True,exist_ok=True)
    assert not run.exists() or not any(run.iterdir())
    declared=make_protocol(args.gpu,args.smoke)
    if args.dry_run:print(json.dumps(declared,indent=2));return
    protocol=REPORT/f"protocol_{ARM}_s1{'_smoke' if args.smoke else ''}.json";assert not protocol.exists()
    protocol.write_text(json.dumps(declared,indent=2)+'\n');print('PLAN '+json.dumps(declared),flush=True)
    try:
        with configure(),runtime(declared,run,args.smoke,protocol):
            trainer.run_training(legacy.trainer_argv(1,run,args.smoke),experiment=declared)
    except BaseException as exc:
        p=run/'manifest.json'
        if p.exists():
            m=json.loads(p.read_text());m.update(status='failed',error_type=type(exc).__name__,error=str(exc));p.write_text(json.dumps(m,indent=2)+'\n')
        raise
    (run/'experiment.json').write_text(json.dumps(declared,indent=2)+'\n')
if __name__=='__main__':main()
