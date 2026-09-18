"""Matched-budget SIDE-SCENE / QREFINE arms, reusing the completed BASE recipe."""
from pathlib import Path
import argparse
import contextlib
import json
import os
import sys
import torch

ROOT = Path('/NHNHOME/data/sukim/adcl')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/md_a2_nominal_mh4_20260918'))
import train_nominal as nominal
sys.path.insert(0, str(HERE))
from scene_extensions import SceneExtensionModel, ARMS, SIDE, QREFINE, SIDE_CAMERA_IDS, SIDE_FRAME_OFFSETS, SIDE_POSE_INDICES
from side_data import SideSceneDataset, SIDE_KEY, wrap_side_flip
from nominal_data import NominalStatusDataset, sha

REPORT = ROOT / 'reports/md_a2_scene_extensions_20260918'
legacy = nominal.legacy


@contextlib.contextmanager
def configure(arm):
    with nominal.configure('A2-BASE-NOM', microbatch=8):
        legacy.SharedDynamicsMotionDriveV2 = SceneExtensionModel
        legacy.CausalStatusDataset = lambda base: NominalStatusDataset(
            SideSceneDataset(base) if arm == SIDE else base, allowed_splits=('train', 'tune'))
        legacy.REPORT_DIR = REPORT
        yield


@contextlib.contextmanager
def runtime(arm, declared, run, smoke, protocol):
    import train_motiondrive_v2 as trainer
    import evaluate_motiondrive_v2_planning as evaluation
    import motiondrive_v2_flip_augment as flip_api
    with nominal.runtime(arm, 1, declared, run, smoke, protocol):
        old_train, old_eval = trainer.model_inputs, evaluation.planning_model_inputs
        old_flip, old_loader = flip_api.flip_item, trainer._load_initial_model_state
        def add_side(original, batch, **kwargs):
            inputs = original(batch, **kwargs)
            if arm == SIDE:
                inputs[SIDE_KEY] = batch[SIDE_KEY]
            return inputs
        def loader(model, common, experiment=None):
            result = old_loader(model, common, experiment)
            if arm == QREFINE:
                model.scene_encoder.evidence_attention.assert_zero_output()
            return result
        trainer.model_inputs = lambda batch, **kw: add_side(old_train, batch, **kw)
        evaluation.planning_model_inputs = lambda batch, **kw: add_side(old_eval, batch, **kw)
        trainer._load_initial_model_state = loader
        flip_api.flip_item = wrap_side_flip(old_flip)
        try:
            yield
        finally:
            trainer.model_inputs, evaluation.planning_model_inputs = old_train, old_eval
            trainer._load_initial_model_state, flip_api.flip_item = old_loader, old_flip


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--gpu', type=int, choices=range(4), required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise ValueError('Expose only the declared physical GPU')
    torch.set_num_threads(4)
    run = Path(args.run_dir).resolve()
    if run.exists() and any(run.iterdir()):
        raise ValueError('Refusing to overwrite an existing run')
    REPORT.mkdir(parents=True, exist_ok=True)
    with configure(args.arm):
        train, tune = nominal.raw_datasets(False, 1)
        declared = nominal.build_experiment(args.arm, 1, train, tune, args.smoke, args.gpu)
        declared['name'] = 'md_a2_scene_extensions_20260918'
        declared['question'] = ('Do four additional historical side images improve shared perception?'
            if args.arm == SIDE else 'Does an image-dependent second scene query improve the same observations?')
        declared['control'] = {'arm': 'A2-BASE-NOM', 'terminal_PREFIX': 0.1655106489655904,
            'run_dir': str(ROOT / 'work_dirs/md_a2_nominal_mh4_20260918/A2-BASE-NOM-s1'),
            'same_initial_shared_tensors': True, 'same_sample_order_and_budget': True,
            'prior_A2_nominal_eval_PREFIX': 0.1644553140831734}
        declared['scene_extension'] = {
            'side_scene': args.arm == SIDE,
            'side_camera_ids': list(SIDE_CAMERA_IDS) if args.arm == SIDE else [],
            'side_frame_offsets': list(SIDE_FRAME_OFFSETS) if args.arm == SIDE else [],
            'side_pose_indices_in_CONTROL': list(SIDE_POSE_INDICES) if args.arm == SIDE else [],
            'side_image_hw': [216, 384] if args.arm == SIDE else None,
            'shared_consumers': ['occupancy', 'lane', 'planning'],
            'motion_observations': 'unchanged native front CONTROL H4',
            'query_refinement': args.arm == QREFINE,
            'refine_query_inputs': ['original query', 'first image-value attention read'] if args.arm == QREFINE else [],
            'refine_value': 'second weighted image-value read only; no query residual or bias' if args.arm == QREFINE else None,
            'refine_output_init': 'zero output projection; query update is nonzero initialized' if args.arm == QREFINE else None,
            'command_or_new_status_value_route': False,
            'new_progress_residual_or_auxiliary_loss': False}
        declared['scene_attention']['heads'] = 1
        declared['scene_attention']['initialization'] = ('BASE function exactly preserved by zero residual output'
            if args.arm == QREFINE else 'BASE weights; added observations change attention from the first update')
        declared['scene_attention']['sampling_grid_camera_time_metadata'] = (
            'same grid/calibration; append FL/FR -.1/-.5s sources' if args.arm == SIDE else 'unchanged')
        for path in [Path(__file__), HERE / 'scene_extensions.py', HERE / 'side_data.py']:
            declared['source'][str(path.relative_to(ROOT))] = sha(path)
        protocol = REPORT / f"protocol_{args.arm}_s1{'_smoke' if args.smoke else ''}.json"
        print('PLAN ' + json.dumps(declared), flush=True)
        if args.dry_run:
            return
        protocol.write_text(json.dumps(declared, indent=2) + '\n')
        import train_motiondrive_v2 as trainer
        with runtime(args.arm, declared, run, args.smoke, protocol):
            trainer.run_training(legacy.trainer_argv(1, run, args.smoke), experiment=declared)
        (run / 'experiment.json').write_text(json.dumps(declared, indent=2) + '\n')


if __name__ == '__main__':
    main()
