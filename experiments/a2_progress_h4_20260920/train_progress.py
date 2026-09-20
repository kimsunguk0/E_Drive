"""Matched full-budget H4-status direct-XY versus interval length/heading."""
from pathlib import Path
import argparse,contextlib,json,os,subprocess,sys
import torch

ROOT=Path('/NHNHOME/data/sukim/adcl')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/a2_temporal_read_20260920'))
import train_temporal as reference
base=reference.base
nominal,trainer,legacy,mr=reference.nominal,reference.trainer,reference.legacy,reference.mr
from motiondrive_v2_training import tensor_state_sha256
sys.path.insert(0,str(HERE))
from h4_data import H4StatusDataset,CACHE,sha
from progress_model import H4ProgressModel,ARMS,DIRECT,PROGRESS,TAG,load_initializer,LENGTH_SCALE,INITIAL_LENGTH,LENGTH_BIAS

REPORT=ROOT/'reports/a2_progress_h4_20260920'
RUNS=ROOT/'work_dirs/a2_progress_h4_20260920'
UPDATES=20554
EVERY=3426

@contextlib.contextmanager
def configure():
    with base.configure(base.FRESH):
        legacy.SharedDynamicsMotionDriveV2=H4ProgressModel
        legacy.CausalStatusDataset=H4StatusDataset
        legacy.REPORT_DIR=REPORT
        yield

def make_protocol(arm,gpu,smoke):
    # Reuse the proven data/recipe checks, then explicitly replace the changed
    # input policy and the comparison. Old nominal inputs are NOT this control.
    value=reference.make_protocol(gpu,smoke)
    manifest=json.loads((CACHE/'manifest.json').read_text())
    value.update(name='a2_progress_h4_20260920',arm=arm,
        question='With H4-covered scene status and identical visual temporal memory, does explicit per-interval progress/direction improve planning?',
        nominal_input={'manifest':str(CACHE/'manifest.json'),'sha256':sha(CACHE/'manifest.json'),
            'producer_sha256':manifest['producer_sha256'],'relative_fit_frames':manifest['relative_fit_frames'],
            'allowed_splits':['train','tune'],'input_field':'provided_status5',
            'state_history_supervision':'original real-time definition retained',
            'pose_fit':'quadratic on exactly five H4/current poses; full rank, condition <100, RMS <0.25m',
            'all_fit_times_have_consumed_RGB':True},
        control={'run_dir':str(RUNS/f'{DIRECT}-s1'),'reuse':False,
            'same_trainable_initial_tensors':True,'same_initial_outputs':False,
            'same_rows_status_recipe_sample_stream_budget':True,
            'primary':'20554-step PROGRESS minus 20554-step DIRECT',
            'old_frozen_candidates':'context only; different trained status policy or training budget'},
        planner={'factorized':arm==PROGRESS,'candidate_bank_or_selector':False,
            'temporal_read_before_output':True,'output_shape':[6,2],
            'output_semantics':'absolute XY; no serving cumsum',
            'composition':('cumsum(5*softplus(raw_length+bias) * [cos(raw_heading), sin(raw_heading)])'
                           if arm==PROGRESS else 'independent absolute XY with existing [10,5] scaling'),
            'raw_neural_outputs':12,'extra_trainable_parameters':0,
            'positive_length_unbounded_heading':arm==PROGRESS,
            'length_scale_m':LENGTH_SCALE if arm==PROGRESS else None,
            'length_bias':LENGTH_BIAS if arm==PROGRESS else None,
            'length_at_zero_raw_m':INITIAL_LENGTH if arm==PROGRESS else None,
            'exact_zero_guaranteed':False,'heading_clamp':False,
            'new_GT_target_or_loss':False,'state_extrapolation_or_base_residual':False})
    value['information_route']['source']='Exactly five RGB-consumed poses [-10,-5,-2,-1,0], nominal frame differences * 0.1s'
    value['information_route']['progress_head_inputs']=[]
    value['information_route']['output_head_inputs']='six decoded visual scene/motion queries; existing predicted state/history memory retained'
    value['temporal_read']['query']='six decoded output queries (waypoints for DIRECT; intervals for PROGRESS)'
    value['temporal_read']['output']='FP32 feature residual before the neural two-channel output head'
    value['scene_extension']['new_progress_residual_or_auxiliary_loss']=False
    value['comparison_contract']['same_initializer_as_FRESH_control']=True
    value['comparison_contract']['same_input_policy_as_older_TEMPORAL']=False
    value['expected_initial_model_state_sha256']=None
    value['recipe'].update(schedule='cosine to terminal',optimizer='fresh AdamW in both arms',
        fixed_bn=True,alpha_occ=.2,alpha_lane=.2,alpha_motion=.2,uncertainty=True)
    value['interpretation']='Output parameterization, physical units and its optimization behavior are the tested package; shared initial tensors do not imply identical initial outputs.'
    for path in (Path(__file__),HERE/'h4_status.py',HERE/'h4_data.py',HERE/'progress_model.py',
                 ROOT/'experiments/a2_temporal_read_20260920/temporal_model.py'):
        value['source'][str(path.relative_to(ROOT))]=sha(path)
    value['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    return value

@contextlib.contextmanager
def runtime(arm,declared,run,smoke,protocol):
    with legacy.patched_runtime(arm,1,declared,run,smoke):
        old_loader,old_json=trainer._load_initial_model_state,trainer.atomic_json
        def load_initial(model,common,experiment=None):
            expected=declared['initializer']['model_state_sha256']
            if tensor_state_sha256(common['model'])!=expected:raise ValueError('Public FRESH initializer differs')
            extras=load_initializer(model,common['model'])
            measured=tensor_state_sha256(model.state_dict())
            checked=json.loads((REPORT/'preflight.json').read_text())
            if checked['status']!='passed' or measured!=checked['initial_states'][arm]:
                raise ValueError('Initial state differs from tested graph')
            if checked['source_sha256']['progress_model.py']!=sha(HERE/'progress_model.py'):
                raise ValueError('Model source differs from preflight')
            declared['expected_initial_model_state_sha256']=measured
            declared['initial_load']={'strict_assembled_state':True,'all_public_parent_tensors_identical':True,
                'model_state_sha256':measured,'new_keys':extras,'prior_ETRI_updates':0,
                'progress_units_buffer':model.planner.progress_units.tolist() if arm==PROGRESS else None}
            protocol.write_text(json.dumps(declared,indent=2)+'\n')
            return {'h4_progress_initial_load':declared['initial_load']}
        def atomic_json(path,payload):
            old_json(path,payload)
            if Path(path).name=='final_eval.json' and payload.get('records'):
                step=payload['report']['step']
                old_json(run/f'predictions_step{step}.json',payload)
                old_json(run/f'diagnostics_step{step}.json',nominal.diagnostics(payload['records']))
        trainer._load_initial_model_state,trainer.atomic_json=load_initial,atomic_json
        try:yield
        finally:trainer._load_initial_model_state,trainer.atomic_json=old_loader,old_json

def main():
    ap=argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--arm',choices=ARMS,required=True)
    ap.add_argument('--gpu',type=int,choices=range(4),required=True)
    ap.add_argument('--run-dir',required=True)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise ValueError('Expose only recorded GPU')
    torch.set_num_threads(4);REPORT.mkdir(parents=True,exist_ok=True)
    run=Path(args.run_dir).resolve()
    if run.exists() and any(run.iterdir()):raise ValueError('Refuse existing run')
    with configure():
        declared=make_protocol(args.arm,args.gpu,args.smoke)
        if args.dry_run:print(json.dumps(declared,indent=2));return
        protocol=REPORT/f"protocol_{args.arm}_s1{'_smoke' if args.smoke else ''}.json"
        if protocol.exists():raise ValueError('Refuse existing protocol')
        protocol.write_text(json.dumps(declared,indent=2)+'\n')
        print('PLAN '+json.dumps(declared),flush=True)
        try:
            with runtime(args.arm,declared,run,args.smoke,protocol):
                trainer.run_training(legacy.trainer_argv(1,run,args.smoke),experiment=declared)
        except BaseException as exc:
            path=run/'manifest.json'
            if path.exists():
                value=json.loads(path.read_text());value.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                path.write_text(json.dumps(value,indent=2)+'\n')
            raise
        (run/'experiment.json').write_text(json.dumps(declared,indent=2)+'\n')

if __name__=='__main__':main()
