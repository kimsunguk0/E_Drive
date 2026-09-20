"""Verify full-data union and exact selected initial state after real updates."""
import json
from train_full import ROOT,REPORT,RUNS,ARM,dev
from full_data import sha
def main():
    run=RUNS/f'{ARM}-s1-smoke';m=json.loads((run/'manifest.json').read_text());p=json.loads((run/'experiment.json').read_text())
    expected=json.loads((dev.REPORT/'preflight.json').read_text())['initial_states'][dev.PROGRESS]
    assert m['status']=='completed' and m['step']==2 and m['nonfinite_count']==0
    assert m['initial_model_state_sha256']==expected
    assert p['train_data']['rows']==101520 and p['train_data']['scenes']==376
    assert p['full_fit']['evaluation_is_in_fit'] and not p['initial_load']['DEV_terminal_loaded']
    assert m['bn_training']['policy']=='fixed'
    old=json.loads((ROOT/'work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1/manifest.json').read_text())
    for key in ('model_config','loss_weights','initial_parameter_count','bn_training'):
        assert m[key]==old[key],key
    same=('batch','microbatch','eval_batch','lr','backbone_lr','warmup','weight_decay','grad_clip',
        'alpha_occ','alpha_lane','alpha_motion','uncertainty','precision','time_input','seed','history_contract')
    assert all(m['arguments'][k]==old['arguments'][k] for k in same)
    row={'status':'passed','run':str(run),'actual_updates':2,'nonfinite_count':0,'training_rows':101520,'unique_scenes':376,
        'same_initial_state_as_selected_DEV':True,'initial_state_sha256':expected,
        'same_graph_loss_optimizer_recipe':True,'evaluation_is_in_fit':True,
        'manifest_sha256':sha(run/'manifest.json'),'DEV_weights_loaded':False,
        'DEV_preflight_sha256':sha(dev.REPORT/'preflight.json')}
    dest=REPORT/'smoke_summary.json';assert not dest.exists();dest.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row,indent=2))
if __name__=='__main__':main()
