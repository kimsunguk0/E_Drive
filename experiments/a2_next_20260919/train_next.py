"""G0/G1 matched continuation and independent learned-sampling training."""
import argparse
import contextlib
import json
import os
import shutil
from common import *

class ExperimentModel:
    VALID_ARMS = {a:0 for a in (*G_ARMS,S_ARM)}
    def __new__(cls,config,*,arm):
        if arm in G_ARMS:return SceneExtensionModel(config,arm=QREFINE)
        if arm==S_ARM:
            from learned_sample import LearnedSampleModel
            return LearnedSampleModel(config)
        raise ValueError(arm)

@contextlib.contextmanager
def configure(arm):
    names=['INITIALIZER','HEAD_LR','BACKBONE_LR']
    saved={k:getattr(legacy,k) for k in names}
    with nominal.configure('A2-BASE-NOM',8):
        legacy.SharedDynamicsMotionDriveV2=ExperimentModel
        legacy.REPORT_DIR=REPORT
        if arm in G_ARMS:
            legacy.INITIALIZER=PARENT;legacy.HEAD_LR=1e-5;legacy.BACKBONE_LR=1e-6
            legacy.UPDATES=3426;legacy.EVAL_EVERY=1142
        try:yield
        finally:
            for k,v in saved.items():setattr(legacy,k,v)

def argv_for(arm,run,smoke):
    args=legacy.trainer_argv(1,run,smoke)
    if arm in G_ARMS:args[args.index('--warmup')+1]='100'
    return args

def declaration(arm,gpu,run,smoke):
    train,tune=nominal.raw_datasets(False,1)
    d=nominal.build_experiment(arm,1,train,tune,smoke,gpu)
    d.update(name='a2_next_20260919',arm=arm,run_dir=str(run),
      question='paired joint-loss balance continuation' if arm in G_ARMS else 'image-dependent shared-scene sampling positions',
      aux_bundle_multiplier=.25 if arm=='A2-G1' else 1.,
      parent_graph='A2-QREFINE-NOM' if arm in G_ARMS else 'A2-BASE-NOM',
      fresh_optimizer=True,full_lineage_used=False)
    d['recipe']['warmup']=100 if arm in G_ARMS else 200
    d['information_route'].update(motion_state_history='unconditioned image branch',
      offset_predictor='unconditioned sampled image values + fixed source metadata' if arm==S_ARM else None,
      raw_status_direct_planner_input=False)
    if arm in G_ARMS:
        d['initializer']['source']='complete DEV QREFINE terminal model state, strict load; no optimizer state'
        d['expected_shared_base_initial_sha256']=d['initializer']['model_state_sha256']
        d['control']={'parent_run':str(PARENT.parent),'parent_PREFIX':.16425176970362365,
            'matched_control':'A2-G0','comparison':'G1-G0; each vs identical parent',
            'sample_stream':'fresh seed1 epoch0 in both arms; not a claim of resuming the parent cursor'}
        d['scene_attention'].update(initialization='strict trained QREFINE weights',query_refinement=True)
        d['comparison_contract']['same_initializer_as_registered_mr']=False
        d['comparison_contract']['same_full_update_budget']=False
        d['recipe']['eval_steps']=[0,1142,2284,3426]
    else:
        d['control']={'run':str(BASE_CONTROL),'PREFIX':.1655106489655904,
            'reuse_requires':'same shared initial tensor SHA and logged sample-order SHA, recipe and rows'}
        d['scene_attention'].update(sampling_grid_camera_time_metadata='same sources/geometry; one bounded learned offset per source',
          initialization='zero offset; parent scene function preserved',query_refinement=False)
        d['sampling']={'radius_feature_cells':2.,'hidden':64,'status_goal_query_input':False,
          'initial_invalid_sources_remain_invalid':True,'key_value_same_position':True}
    sources=[HERE/'common.py',Path(__file__)]
    if arm==S_ARM:sources.append(HERE/'learned_sample.py')
    else:sources.append(ROOT/'experiments/md_a2_scene_extensions_20260918/scene_extensions.py')
    for p in sources:d['source'][str(p.relative_to(ROOT))]=sha(p)
    return d

@contextlib.contextmanager
def runtime(arm,d,run,smoke,protocol):
    with legacy.patched_runtime(arm,1,d,run,smoke):
        old_loader=trainer._load_initial_model_state
        old_validator=trainer._validate_experimental_runtime
        old_loss=trainer.compute_loss;old_json=trainer.atomic_json
        def load_initial(model,common,experiment=None):
            if arm in G_ARMS:
                expected=d['initializer']['model_state_sha256']
                assert tensor_state_sha256(common['model'])==expected
                model.load_state_dict(common['model'],strict=True)
                assert tensor_state_sha256(model.state_dict())==expected
                d['expected_initial_model_state_sha256']=expected
                d['initial_load']={'strict_full_parent':True,'measured_model_state_sha256':expected}
                result={'initial_load':d['initial_load']}
            else:
                result=old_loader(model,common,experiment)
                from learned_sample import OFFSET_PREFIX
                shared={k:v for k,v in model.state_dict().items() if not k.startswith(OFFSET_PREFIX)}
                measured=tensor_state_sha256(shared)
                assert measured==nominal.BASE_INITIAL_SHA
                d['initial_load']['shared_base_state_sha256']=measured
                model.scene_encoder.offset_predictor.assert_zero()
            protocol.write_text(json.dumps(d,indent=2)+'\n')
            return result
        def validate(args,value):
            if arm not in G_ARMS:return old_validator(args,value)
            warmup=args.warmup
            assert warmup==100 and args.init==str(PARENT)
            # Reuse every other exact recipe check; the stage-2 warmup is deliberate.
            args.warmup=200
            try:old_validator(args,value)
            finally:args.warmup=warmup
        def loss(output,batch,weights,**kwargs):
            total,parts=old_loss(output,batch,weights,**kwargs)
            aux=weights.occupancy*parts['occ_bce']+weights.lane*parts['lane_bce']+weights.motion*parts['motion']
            if arm=='A2-G1':
                total=weights.plan*parts['plan_d3']+.25*parts['plan_interval_length']+.25*aux
            parts=dict(parts,total=total,weighted_auxiliary=aux*d['aux_bundle_multiplier'])
            return total,parts
        def atomic_json(path,payload):
            old_json(path,payload)
            if Path(path).name=='final_eval.json' and payload.get('records'):
                step=payload['report']['step']
                old_json(run/f'diagnostics_step{step}.json',nominal.diagnostics(payload['records']))
                old_json(run/f'predictions_step{step}.json',payload)
        trainer._load_initial_model_state=load_initial
        trainer._validate_experimental_runtime=validate
        trainer.compute_loss=loss
        trainer.atomic_json=atomic_json
        try:yield
        finally:
            trainer._load_initial_model_state=old_loader
            trainer._validate_experimental_runtime=old_validator
            trainer.compute_loss=old_loss
            trainer.atomic_json=old_json

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--arm',choices=(*G_ARMS,S_ARM),required=True)
    p.add_argument('--gpu',type=int,choices=range(4),required=True)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.gpu)
    torch.set_num_threads(4);run=Path(a.run_dir).resolve()
    assert not run.exists() or not any(run.iterdir()),'Never overwrite a run'
    REPORT.mkdir(parents=True,exist_ok=True)
    with configure(a.arm):
        d=declaration(a.arm,a.gpu,run,a.smoke)
        protocol=REPORT/f"protocol_{a.arm}_s1{'_smoke' if a.smoke else ''}.json"
        print('PLAN '+json.dumps(d),flush=True)
        if a.dry_run:return
        assert not protocol.exists(),'Protocol name already used'
        protocol.write_text(json.dumps(d,indent=2)+'\n')
        with runtime(a.arm,d,run,a.smoke,protocol):
            trainer.run_training(argv_for(a.arm,run,a.smoke),experiment=d)
        (run/'experiment.json').write_text(json.dumps(d,indent=2)+'\n')
        if a.arm in G_ARMS:
            shutil.copy2(REPORT/'parent_initial_eval.json',run/'initial_eval.json')

if __name__=='__main__':main()
