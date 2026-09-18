"""Fresh 20,554-update command comparison against the completed BASE-NOM."""
import argparse
import contextlib
import json
import os
import subprocess
from command_common import *
from command_data import CommandDataset, KEY, LABELS, wrap_command_flip
from command_model import CommandA2Model, COMMAND_PREFIX

@contextlib.contextmanager
def configure():
    with nominal.configure('A2-BASE-NOM', 8):
        legacy.SharedDynamicsMotionDriveV2 = CommandA2Model
        legacy.CausalStatusDataset = lambda base: CommandDataset(NominalStatusDataset(base))
        legacy.REPORT_DIR = REPORT
        yield

def declaration(gpu, run, smoke):
    train, tune = nominal.raw_datasets(False, 1)
    d = nominal.build_experiment(ARM, 1, train, tune, smoke, gpu)
    command_manifest = CACHE/'manifest.json'
    m = json.loads(command_manifest.read_text())
    assert m['split_manifest_sha256'] == sha(legacy.SPLIT)
    d.update(name='a2_command_20260919', run_dir=str(run),
        question='Does supplied semantic command improve shared-scene visual evidence selection?',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        fresh_optimizer=True, full_lineage_used=False,
        command_input=dict(manifest=str(command_manifest), sha256=sha(command_manifest),
            labels=list(LABELS), producer='current raw command; exact frame/timestamp join',
            vad_cmd_used=False, zero_init=True, new_parameters=192),
        control=dict(run=str(CONTROL), PREFIX=.1655106489655904,
            comparison='fixed terminal vs fixed terminal; same existing tensors, data, schedule and sample stream',
            current_best_terminal_QREFINE_PREFIX=.16425176970362365))
    d['information_route'].update(command='shared scene query only; image key/value unchanged',
        raw_command_direct_planner_input=False, motion_state_history='unconditioned image path',
        shared_consumers=['occupancy','lane','planning'])
    for p in HERE.glob('*.py'):
        d['source'][str(p.relative_to(ROOT))] = sha(p)
    return d

@contextlib.contextmanager
def runtime(d, run, smoke, protocol):
    import motiondrive_v2_flip_augment as flip_api
    import evaluate_motiondrive_v2_planning as eval_api
    with legacy.patched_runtime(ARM, 1, d, run, smoke):
        loader, write_json = trainer._load_initial_model_state, trainer.atomic_json
        train_inputs, eval_inputs = trainer.model_inputs, eval_api.planning_model_inputs
        flip = flip_api.flip_item
        def add_command(original, batch, **kwargs):
            x = original(batch, **kwargs)
            x[KEY] = batch[KEY]
            return x
        def load_initial(model, common, experiment=None):
            result = loader(model, common, experiment)
            shared = {k:v for k,v in model.state_dict().items() if not k.startswith(COMMAND_PREFIX)}
            assert tensor_state_sha256(shared) == nominal.BASE_INITIAL_SHA
            assert not bool(model.shared_command_query.weight.count_nonzero())
            assert all(p.requires_grad for p in model.parameters())
            d['initial_load']['shared_base_state_sha256'] = nominal.BASE_INITIAL_SHA
            d['initial_load']['command_exact_zero'] = True
            protocol.write_text(json.dumps(d, indent=2)+'\n')
            return result
        def atomic_json(path, payload):
            write_json(path, payload)
            if Path(path).name == 'final_eval.json' and payload.get('records'):
                step = payload['report']['step']
                write_json(run/f'diagnostics_step{step}.json', nominal.diagnostics(payload['records']))
                write_json(run/f'predictions_step{step}.json', payload)
        trainer._load_initial_model_state = load_initial
        trainer.atomic_json = atomic_json
        trainer.model_inputs = lambda batch, **kw: add_command(train_inputs, batch, **kw)
        eval_api.planning_model_inputs = lambda batch, **kw: add_command(eval_inputs, batch, **kw)
        flip_api.flip_item = wrap_command_flip(flip)
        try:
            yield
        finally:
            trainer._load_initial_model_state = loader
            trainer.atomic_json = write_json
            trainer.model_inputs = train_inputs
            eval_api.planning_model_inputs = eval_inputs
            flip_api.flip_item = flip

def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--gpu', type=int, choices=range(4), required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(a.gpu)
    torch.set_num_threads(4)
    run = Path(a.run_dir).resolve()
    assert not run.exists() or not any(run.iterdir()), 'Never overwrite a run'
    REPORT.mkdir(parents=True, exist_ok=True)
    with configure():
        d = declaration(a.gpu, run, a.smoke)
        protocol = REPORT/f"protocol_{ARM}_s1{'_smoke' if a.smoke else ''}.json"
        print('PLAN '+json.dumps(d), flush=True)
        if a.dry_run:
            return
        assert not protocol.exists(), 'Protocol already used'
        protocol.write_text(json.dumps(d, indent=2)+'\n')
        with runtime(d, run, a.smoke, protocol):
            trainer.run_training(legacy.trainer_argv(1, run, a.smoke), experiment=d)
        (run/'experiment.json').write_text(json.dumps(d, indent=2)+'\n')

if __name__ == '__main__':
    main()
