"""One matched full-budget FRESH + waypoint temporal read; no other new arm."""
from pathlib import Path
import argparse
import contextlib
import copy
import json
import os
import subprocess
import sys

import torch

ROOT = Path('/NHNHOME/data/sukim/adcl')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/a2_motion_fresh_20260919'))
import train_experiment as base
from motiondrive_v2_training import tensor_state_sha256
from nominal_data import sha
sys.path.insert(0, str(HERE))
from temporal_model import TemporalReadModel, ARM, NEW_PREFIX, load_fresh_parent

REPORT = ROOT / 'reports/a2_temporal_read_20260920'
RUNS = ROOT / 'work_dirs/a2_temporal_read_20260920'
CONTROL = base.RUNS / 'A2-FRESH-NUIM-s1'
CONTROL_PROTOCOL = base.REPORT / 'protocol_A2-FRESH-NUIM_s1.json'
UPDATES = 20554
EVERY = 3426
nominal, trainer, legacy, mr = base.nominal, base.trainer, base.legacy, base.mr


@contextlib.contextmanager
def configure():
    with base.configure(base.FRESH):
        legacy.SharedDynamicsMotionDriveV2 = TemporalReadModel
        legacy.REPORT_DIR = REPORT
        yield


def make_protocol(gpu, smoke):
    parent = json.loads(CONTROL_PROTOCOL.read_text())
    train, tune = nominal.raw_datasets(False, 1)
    declared = nominal.build_experiment(ARM, 1, train, tune, smoke, gpu)
    # Check source/data/recipe against the existing completed full-budget control.
    for name in ('split_manifest', 'supervision', 'train_data', 'tune_data', 'nominal_input'):
        if declared[name] != parent[name]:
            raise ValueError('Control contract changed: ' + name)
    expected_recipe = copy.deepcopy(parent['recipe'])
    if smoke:
        expected_recipe.update(updates=2, eval_every=2)
    if declared['recipe'] != expected_recipe:
        raise ValueError('Control recipe changed')
    source_checks = {}
    for name, checksum in parent['source'].items():
        measured = sha(ROOT / name)
        if measured != checksum:
            raise ValueError('Control source changed: ' + name)
        source_checks[name] = measured
    initializer = base.ensure_fresh_initializer()
    if initializer != parent['initializer']:
        raise ValueError('Control public initializer changed')
    declared.update(name='a2_temporal_read_20260920',
        question='Do decoded waypoint queries benefit from direct unpooled visual temporal memory?',
        initializer=initializer, scene_extension=copy.deepcopy(parent['scene_extension']),
        scene_attention=copy.deepcopy(parent['scene_attention']),
        expected_shared_base_initial_sha256=initializer['model_state_sha256'],
        control={'run_dir': str(CONTROL), 'protocol': str(CONTROL_PROTOCOL),
            'protocol_sha256': sha(CONTROL_PROTOCOL), 'terminal_PREFIX': 0.1587072591966789,
            'updates': UPDATES, 'reuse': True,
            'same_base_initial_tensors_recipe_rows_augmentation': True,
            'sample_stream_verification': 'same dedicated generator and deterministic augmentation; compare logged row SHA',
            'historical_ETRI_training_exposure_equal': True,
            'terminal_vs_terminal_primary': True},
        temporal_read={'memory_shape': [4, 192, 128], 'memory_tokens': 768,
            'source': 'same unaligned front H4 pair_tokens before time pooling, with original time/position embeddings',
            'query': 'existing six decoded planner waypoint features', 'heads': 4, 'channels': 128,
            'query_memory_layernorm': True, 'dropout': 0.0,
            'output': 'FP32 feature residual before existing XY head',
            'zero_init': 'attention.out_proj weight and bias only',
            'pooled_motion_state_history_path_retained': True,
            'same_RGB_frames_resolution': True,
            'new_raw_status_goal_pose_inputs': False,
            'planner_to_motion_state_feedback': False,
            'new_loss_or_teacher': False,
            'constructor_rng_seed': 2026092001, 'constructor_rng_isolated': True},
        evaluations={'primary': UPDATES, 'scheduled_steps': [*range(EVERY, UPDATES, EVERY), UPDATES],
            'selected_intermediate_is_independent_validation': False},
        control_source_checks=source_checks,
        automatic_followups={'continuation': False, 'weight_average': False, 'FULL': False, 'submission': False})
    declared['comparison_contract']['same_initializer_as_registered_mr'] = False
    declared['comparison_contract']['same_initializer_as_FRESH_control'] = True
    declared['information_route']['new_temporal_memory'] = 'unaligned RGB and nominal elapsed time only'
    declared['information_route']['new_query_inherits_scene_indirect_conditions'] = True
    declared['planner']['temporal_read_before_xy'] = True
    for path in (Path(__file__), HERE/'temporal_model.py', ROOT/'models/motiondrive_v2/motion_encoder.py',
                 ROOT/'models/motiondrive_v2/planner.py', ROOT/'scripts/train_motiondrive_v2.py',
                 ROOT/'scripts/motiondrive_v2_training.py', ROOT/'scripts/motiondrive_v2_flip_augment.py'):
        declared['source'][str(path.relative_to(ROOT))] = sha(path)
    declared['source'].update(source_checks)
    declared['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    return declared


@contextlib.contextmanager
def runtime(declared, run, smoke, protocol):
    with legacy.patched_runtime(ARM, 1, declared, run, smoke):
        old_loader, old_json = trainer._load_initial_model_state, trainer.atomic_json

        def load_initial(model, common, experiment=None):
            expected = declared['initializer']['model_state_sha256']
            if tensor_state_sha256(common['model']) != expected:
                raise ValueError('Public FRESH initial state changed')
            extras = load_fresh_parent(model, common['model'])
            shared = {k:v for k,v in model.state_dict().items() if not k.startswith(NEW_PREFIX)}
            if tensor_state_sha256(shared) != expected:
                raise ValueError('A control tensor was modified')
            output = model.planner.temporal_read.attention.out_proj
            if output.weight.count_nonzero() or output.bias.count_nonzero():
                raise ValueError('New output must initially be zero')
            measured = tensor_state_sha256(model.state_dict())
            preflight = json.loads((REPORT/'preflight.json').read_text())
            if measured != preflight['new_initial_state_sha256']:
                raise ValueError('Initial state differs from checked model')
            declared['expected_initial_model_state_sha256'] = measured
            declared['initial_load'] = {'strict_assembled_state': True,
                'all_parent_tensors_identical': True, 'shared_initial_state_sha256': expected,
                'model_state_sha256': measured, 'new_keys': extras,
                'public_trunk_only': True, 'prior_ETRI_training_updates': 0}
            protocol.write_text(json.dumps(declared,indent=2)+'\n')
            return {'temporal_read_initial_load': declared['initial_load']}

        def atomic_json(path, payload):
            old_json(path, payload)
            if Path(path).name == 'final_eval.json' and payload.get('records'):
                step = payload['report']['step']
                old_json(run/f'predictions_step{step}.json', payload)
                old_json(run/f'diagnostics_step{step}.json', nominal.diagnostics(payload['records']))

        trainer._load_initial_model_state, trainer.atomic_json = load_initial, atomic_json
        try:
            yield
        finally:
            trainer._load_initial_model_state, trainer.atomic_json = old_loader, old_json


def main():
    ap=argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--gpu', type=int, choices=range(4), required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise ValueError('Expose exactly the recorded physical GPU')
    torch.set_num_threads(4)
    REPORT.mkdir(parents=True,exist_ok=True)
    run=Path(args.run_dir).resolve()
    if run.exists() and any(run.iterdir()):
        raise ValueError('Never overwrite an existing run')
    with configure():
        declared=make_protocol(args.gpu,args.smoke)
        if args.dry_run:
            print(json.dumps(declared,indent=2));return
        protocol=REPORT/f"protocol_{ARM}_s1{'_smoke' if args.smoke else ''}.json"
        if protocol.exists():
            raise ValueError('Protocol already exists')
        protocol.write_text(json.dumps(declared,indent=2)+'\n')
        print('PLAN '+json.dumps(declared),flush=True)
        try:
            with runtime(declared,run,args.smoke,protocol):
                trainer.run_training(legacy.trainer_argv(1,run,args.smoke),experiment=declared)
        except BaseException as exc:
            p=run/'manifest.json'
            if p.exists():
                value=json.loads(p.read_text())
                value.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                p.write_text(json.dumps(value,indent=2)+'\n')
            raise
        (run/'experiment.json').write_text(json.dumps(declared,indent=2)+'\n')


if __name__ == '__main__':
    main()
