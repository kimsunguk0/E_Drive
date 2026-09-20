"""One authorized low-LR continuation of the complete DEV FRESH terminal."""
from pathlib import Path
import argparse
import contextlib
import copy
import json
import os
import shutil
import subprocess
import sys
import torch

ROOT=Path('/NHNHOME/data/sukim/adcl')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/a2_motion_fresh_20260919'))
import train_experiment as prior
from nominal_data import sha
from motiondrive_v2_training import tensor_state_sha256

nominal,legacy,trainer=prior.nominal,prior.legacy,prior.trainer
ARM='A2-FRESH-CONT'
REPORT=ROOT/'reports/a2_fresh_continue_20260920'
RUNS=ROOT/'work_dirs/a2_fresh_continue_20260920'
PARENT_RUN=prior.RUNS/'A2-FRESH-NUIM-s1'
PARENT=PARENT_RUN/'ckpt_step20554.pth'
PARENT_SHA='f5ed023c23733fcb83ba0c85fe9b5e3c8f97a91361f8e28b98ec9da2c2e9dfce'
UPDATES=6852
EVERY=1142


class Factory:
    VALID_ARMS={ARM:0}
    def __new__(cls,config,*,arm):
        if arm!=ARM:raise ValueError(arm)
        return prior.ExperimentModel(config,arm=prior.FRESH)


@contextlib.contextmanager
def configure():
    saved={k:getattr(legacy,k) for k in ('INITIALIZER','HEAD_LR','BACKBONE_LR')}
    with nominal.configure('A2-BASE-NOM',8):
        legacy.SharedDynamicsMotionDriveV2=Factory
        legacy.REPORT_DIR=REPORT
        legacy.INITIALIZER=PARENT
        legacy.HEAD_LR=1e-5
        legacy.BACKBONE_LR=1e-6
        legacy.UPDATES=UPDATES
        legacy.EVAL_EVERY=EVERY
        try:yield
        finally:
            for k,v in saved.items():setattr(legacy,k,v)


def declaration(gpu,run,smoke):
    if sha(PARENT)!=PARENT_SHA:raise ValueError('Pinned FRESH terminal checksum changed')
    m=json.loads((PARENT_RUN/'manifest.json').read_text())
    assert m['status']=='completed' and m['step']==20554 and m['nonfinite_count']==0
    parent_recipe=json.loads((PARENT_RUN/'experiment.json').read_text())
    assert parent_recipe['full_fit']['enabled'] is False
    train,tune=nominal.raw_datasets(False,1)
    d=nominal.build_experiment(ARM,1,train,tune,smoke,gpu)
    d.update(name='a2_fresh_continue_20260920',run_dir=str(run),
        question='Does one lower-LR continuation improve the freshly initialized A2, including its stop/departure regressions?',
        fresh_optimizer=True,full_lineage_used=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        parent_model_config=m['model_config'],
        parent_updates=20554,terminal_total_updates=20556 if smoke else 27406,
        primary_comparison='fixed terminal minus fixed FRESH parent; repeated DEV, not hidden-test estimate',
        checkpoint_selection='terminal primary; planned intermediate best marked separately; parent preserved',
        sample_stream='fresh seed1 epoch0; no claim of resuming parent data cursor or optimizer moments',
        architecture_changes=False,loss_changes=False,automatic_followup_training_FULL_submission=False)
    d['initializer']['source']='complete DEV FRESH terminal, all tensors strict loaded; no optimizer/RNG state import'
    d['expected_initial_model_state_sha256']=d['initializer']['model_state_sha256']
    d['expected_shared_base_initial_sha256']=d['initializer']['model_state_sha256']
    d['scene_attention']=copy.deepcopy(parent_recipe['scene_attention'])
    d['scene_extension']=copy.deepcopy(parent_recipe['scene_extension'])
    d['recipe'].update(warmup=100,schedule='new cosine decay over continuation updates',
        eval_steps=[0,2] if smoke else [0,1142,2284,3426,4568,5710,6852],
        initial_eval='reuse full V0 parent predictions; identical model hash and input policy',
        optimizer='fresh AdamW; all parameters trainable; no frozen phase')
    d['comparison_contract'].update(same_initializer_as_registered_mr=False,
        same_full_update_budget=False,same_seed_and_sample_order_across_arms=False,
        same_graph_input_loss_as_parent=True)
    for p in (Path(__file__),ROOT/'experiments/a2_motion_fresh_20260919/motion_model.py',
              ROOT/'experiments/md_a2_scene_extensions_20260918/scene_extensions.py',
              ROOT/'scripts/train_motiondrive_v2.py'):
        d['source'][str(p.relative_to(ROOT))]=sha(p)
    return d


def argv(run,smoke):
    v=legacy.trainer_argv(1,run,smoke)
    v[v.index('--warmup')+1]='100'
    return v


@contextlib.contextmanager
def runtime(d,run,smoke,protocol):
    with legacy.patched_runtime(ARM,1,d,run,smoke):
        old_loader=trainer._load_initial_model_state
        old_validator=trainer._validate_experimental_runtime
        old_json=trainer.atomic_json
        def initialize(model,common,experiment=None):
            expected=d['initializer']['model_state_sha256']
            assert tensor_state_sha256(common['model'])==expected
            assert json.loads(json.dumps(model.config.to_dict()))==d['parent_model_config']
            model.load_state_dict(common['model'],strict=True)
            assert tensor_state_sha256(model.state_dict())==expected
            d['initial_load']={'strict_complete_parent':True,'correlation_fuse_reinitialized':False,
                'measured_model_state_sha256':expected,'all_parameters_trainable':all(p.requires_grad for p in model.parameters())}
            protocol.write_text(json.dumps(d,indent=2)+'\n')
            return {'initial_load':d['initial_load']}
        def validate(args,value):
            assert args.warmup==100 and args.init==str(PARENT) and not args.resume
            # Keep the legacy exact checks; only the declared warmup differs.
            args.warmup=200
            try:old_validator(args,value)
            finally:args.warmup=100
        def write_json(path,payload):
            old_json(path,payload)
            if Path(path).name=='final_eval.json' and payload.get('records'):
                step=payload['report']['step']
                old_json(run/f'predictions_step{step}.json',payload)
                old_json(run/f'diagnostics_step{step}.json',nominal.diagnostics(payload['records']))
        trainer._load_initial_model_state=initialize
        trainer._validate_experimental_runtime=validate
        trainer.atomic_json=write_json
        try:yield
        finally:
            trainer._load_initial_model_state=old_loader
            trainer._validate_experimental_runtime=old_validator
            trainer.atomic_json=old_json


def main():
    ap=argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--gpu',type=int,choices=range(4),required=True)
    ap.add_argument('--run-dir',required=True)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.gpu)
    torch.set_num_threads(4)
    run=Path(args.run_dir).resolve()
    assert not run.exists() or not any(run.iterdir()),'Never overwrite an existing run'
    REPORT.mkdir(parents=True,exist_ok=True)
    protocol=REPORT/f"protocol_{ARM}_s1{'_smoke' if args.smoke else ''}.json"
    with configure():
        d=declaration(args.gpu,run,args.smoke)
        if args.dry_run:
            print(json.dumps(d,indent=2));return
        assert not protocol.exists(),'Protocol already exists; inspect prior run'
        protocol.write_text(json.dumps(d,indent=2)+'\n')
        print('PLAN '+json.dumps(d),flush=True)
        try:
            with runtime(d,run,args.smoke,protocol):
                trainer.run_training(argv(run,args.smoke),experiment=d)
        except BaseException as exc:
            p=run/'manifest.json'
            if p.exists():
                failed=json.loads(p.read_text())
                failed.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                p.write_text(json.dumps(failed,indent=2)+'\n')
            raise
        shutil.copy2(PARENT_RUN/'final_eval.json',run/'parent_eval.json')
        (run/'experiment.json').write_text(json.dumps(d,indent=2)+'\n')


if __name__=='__main__':main()
