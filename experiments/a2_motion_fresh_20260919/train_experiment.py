"""Two independent, preregistered A2 experiments. Reuse the existing trainer."""
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
REPORT = ROOT / 'reports/a2_motion_fresh_20260919'
RUNS = ROOT / 'work_dirs/a2_motion_fresh_20260919'
for p in (ROOT, ROOT/'scripts', ROOT/'experiments/md_a2_nominal_mh4_20260918',
          ROOT/'experiments/md_a2_scene_extensions_20260918', HERE):
    sys.path.insert(0, str(p))
import train_nominal as nominal
import train_motiondrive_v2 as trainer
from nominal_data import sha
from motiondrive_v2_training import tensor_state_sha256
from models.motiondrive_v2 import MotionDriveV2Config
from scene_extensions import SceneExtensionModel, QREFINE
from motion_model import ExperimentModel, FINE, FRESH, ARMS, FINE_PREFIX

legacy = nominal.legacy
mr = legacy.mr
OLD_INIT = legacy.INITIALIZER
PUBLIC = ROOT / 'ckpt/backbones/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
PUBLIC_SHA = '4096396018c0cf59fbe0eb1afe6e269f4676b34460bed5eedde5d7680d58bb4e'
CONTROL = ROOT / 'work_dirs/md_a2_scene_extensions_20260918/A2-QREFINE-NOM-s1'
CONTROL_PROTOCOL = ROOT / 'reports/md_a2_scene_extensions_20260918/protocol_A2-QREFINE-NOM_s1.json'
CONTROL_INITIAL_SHA = '17e2f7488642a8e0acedb162999981ae5fd3ee430d32ee704e0f6a272d5e3133'
FRESH_INIT = RUNS / 'initializers/public_nuimages_qrefine_s1.pth'


def config_from_control():
    # Architecture metadata only; no DEV terminal tensor, BN statistic or cache.
    value = json.loads((CONTROL/'manifest.json').read_text())['model_config']
    return MotionDriveV2Config(**value)


def ensure_fresh_initializer():
    if sha(PUBLIC) != PUBLIC_SHA:
        raise ValueError('Public backbone checksum changed')
    manifest_path = REPORT/'fresh_initializer.json'
    if FRESH_INIT.exists():
        record = json.loads(manifest_path.read_text())
        if sha(FRESH_INIT) != record['checkpoint_sha256']:
            raise ValueError('Existing fresh initializer hash differs')
        return record
    trainer.seed_all(1)
    model = ExperimentModel(config_from_control(), arm=FRESH)
    mr.rebuild_correlation_fuse(model, 4)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    loaded = model.load_pretrained_backbone(PUBLIC)
    if loaded['nonhead_missing'] or loaded['unexpected']:
        raise ValueError(loaded)
    state = model.state_dict()
    trunk_prefix = tuple('backbone_fpn.'+x for x in ('stem.', 'layer1.', 'layer2.', 'layer3.', 'layer4.'))
    nontrunk = [k for k in state if not k.startswith(trunk_prefix)]
    if not all(torch.equal(state[k], before[k]) for k in nontrunk):
        raise ValueError('Public trunk loading modified a random FPN/scene/motion/planner tensor')
    payload = {'model': state, 'step': 0, 'epoch': 0, 'manifest': {
        'model_config': model.config.to_dict(), 'split_sha256': sha(legacy.SPLIT),
        'arguments': {'arch':'resnet50','time_input':'nominal','bn_policy':'fixed'},
        'time_input': 'nominal', 'status':'public_only_initializer',
        'public_pretrained_sha256':PUBLIC_SHA, 'ETRI_optimizer_updates':0}}
    FRESH_INIT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, FRESH_INIT)
    record = {'path':str(FRESH_INIT), 'checkpoint_sha256':sha(FRESH_INIT),
        'model_state_sha256':tensor_state_sha256(state), 'source':'public nuImages ResNet50 trunk + random nontrunk',
        'public_path':str(PUBLIC), 'public_sha256':PUBLIC_SHA, 'public_load':loaded,
        'ETRI_weight_BN_teacher_feature_imported':False, 'ETRI_optimizer_updates':0,
        'config_only_source':str(CONTROL/'manifest.json'),
        'nontrunk_random_preserved':True, 'nontrunk_tensor_count':len(nontrunk),
        'nontrunk_state_sha256':tensor_state_sha256({k:state[k] for k in nontrunk}),
        'fresh_parts':['compact FPN','scene/QREFINE','motion','state/history heads','planner','A2 query'],
        'random_seed':1}
    REPORT.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(record, indent=2)+'\n')
    return record


@contextlib.contextmanager
def configure(arm):
    previous_init = legacy.INITIALIZER
    with nominal.configure('A2-BASE-NOM', microbatch=8):
        legacy.SharedDynamicsMotionDriveV2 = ExperimentModel
        legacy.REPORT_DIR = REPORT
        legacy.INITIALIZER = FRESH_INIT if arm == FRESH else OLD_INIT
        try:
            yield
        finally:
            legacy.INITIALIZER = previous_init


def make_protocol(arm, gpu, smoke):
    train, tune = nominal.raw_datasets(False, 1)
    declared = nominal.build_experiment(arm, 1, train, tune, smoke, gpu)
    parent = json.loads(CONTROL_PROTOCOL.read_text())
    declared['name'] = 'a2_motion_fresh_20260919'
    declared['question'] = ('Does image-only stride16-to-stride4 matching improve A2?'
        if arm == FINE else 'Does A2 trained without prior ETRI weight inheritance improve?')
    declared['scene_extension'] = copy.deepcopy(parent['scene_extension'])
    declared['scene_attention'] = copy.deepcopy(parent['scene_attention'])
    declared['control'] = {'run_dir':str(CONTROL), 'terminal_PREFIX':.16425176973102093,
        'initial_state_sha256':CONTROL_INITIAL_SHA, 'updates':20554,
        'same_recipe_rows_sample_stream':True,
        'same_ETRI_initializer':arm == FINE,
        'comparison':'fine matching only' if arm == FINE else 'initialization recipe only; same new-update budget',
        'historical_ETRI_training_exposure_equal':arm == FINE}
    declared['motion_extension'] = {'enabled':arm == FINE,
        'source':'existing native front H4 backbone pass, stride4 C1 and stride16 FPN',
        'coarse_radius':4, 'coarse_softmax_temperature':.07, 'fine_radius':2,
        'fine_descriptor_channels':32, 'coarse_flow_units':'stride16 cells; x4 to stride4 cells',
        'warp':'history(x + visually inferred flow(x) + local offset)',
        'grid_dtype':'float32', 'align_corners':False,
        'output':'zero-initialized neural feature residual before motion spatial pooling',
        'new_raw_pose_goal_status_inputs':False, 'new_GT_or_flow_loss':False,
        'same_RGB_frames_and_resolutions':True}
    declared['comparison_contract']['same_initializer_as_registered_mr'] = arm == FINE
    declared['expected_shared_base_initial_sha256'] = CONTROL_INITIAL_SHA if arm == FINE else None
    if arm == FRESH:
        record = json.loads((REPORT/'fresh_initializer.json').read_text())
        declared['initializer'] = record
        declared['expected_initial_model_state_sha256'] = record['model_state_sha256']
        declared['fresh_training_limit'] = ('20554 new joint updates is a matched-budget comparison, '
            'not proof of convergence or the performance ceiling of a random nontrunk.')
    for path in (Path(__file__), HERE/'motion_model.py'):
        declared['source'][str(path.relative_to(ROOT))] = sha(path)
    declared['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    return declared


@contextlib.contextmanager
def runtime(arm, declared, run, smoke, protocol):
    import train_motiondrive_v2 as tr
    with legacy.patched_runtime(arm, 1, declared, run, smoke):
        old_loader, old_json = tr._load_initial_model_state, tr.atomic_json
        def load_initial(model, common, experiment=None):
            if arm == FINE:
                result = old_loader(model, common, experiment)
                retained = {k:v for k,v in model.state_dict().items() if not k.startswith(FINE_PREFIX)}
                if tensor_state_sha256(retained) != CONTROL_INITIAL_SHA:
                    raise ValueError('Fine arm changed existing QREFINE initializer tensors')
                if model.motion_encoder.fine_match.output.weight.count_nonzero():
                    raise ValueError('Fine output must initially be zero')
                declared['initial_load']['shared_QREFINE_state_sha256'] = CONTROL_INITIAL_SHA
            else:
                if tensor_state_sha256(common['model']) != declared['initializer']['model_state_sha256']:
                    raise ValueError('Fresh initializer differs')
                model.load_state_dict(common['model'], strict=True)
                measured = tensor_state_sha256(model.state_dict())
                if measured != declared['initializer']['model_state_sha256']:
                    raise ValueError('Fresh strict load changed tensors')
                declared['expected_initial_model_state_sha256'] = measured
                declared['initial_load'] = {'strict':True, 'model_state_sha256':measured,
                    'ETRI_checkpoint_tensors_loaded':False, 'public_trunk_only':True}
                result = {'fresh_public_initial_load':declared['initial_load']}
            protocol.write_text(json.dumps(declared,indent=2)+'\n')
            return result
        def atomic_json(path, payload):
            old_json(path,payload)
            if Path(path).name == 'final_eval.json' and payload.get('records'):
                step = payload['report']['step']
                old_json(run/f'predictions_step{step}.json',payload)
                old_json(run/f'diagnostics_step{step}.json',nominal.diagnostics(payload['records']))
        tr._load_initial_model_state, tr.atomic_json = load_initial, atomic_json
        try:
            yield
        finally:
            tr._load_initial_model_state, tr.atomic_json = old_loader, old_json


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--arm',choices=ARMS,required=True)
    ap.add_argument('--gpu',type=int,choices=range(4),required=True)
    ap.add_argument('--run-dir',required=True)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise ValueError('Expose exactly the declared GPU')
    torch.set_num_threads(4)
    REPORT.mkdir(parents=True, exist_ok=True)
    if args.arm == FRESH:
        ensure_fresh_initializer()
    run = Path(args.run_dir).resolve()
    if run.exists() and any(run.iterdir()):
        raise ValueError('Existing run directory must not be overwritten')
    with configure(args.arm):
        declared = make_protocol(args.arm,args.gpu,args.smoke)
        if args.dry_run:
            print(json.dumps(declared,indent=2)); return
        protocol = REPORT/f"protocol_{args.arm}_s1{'_smoke' if args.smoke else ''}.json"
        protocol.write_text(json.dumps(declared,indent=2)+'\n')
        print('PLAN '+json.dumps(declared),flush=True)
        try:
            with runtime(args.arm,declared,run,args.smoke,protocol):
                trainer.run_training(legacy.trainer_argv(1,run,args.smoke),experiment=declared)
        except BaseException as exc:
            manifest_path = run/'manifest.json'
            if manifest_path.exists():
                manifest=json.loads(manifest_path.read_text())
                manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
            raise
        (run/'experiment.json').write_text(json.dumps(declared,indent=2)+'\n')


if __name__ == '__main__':
    main()
