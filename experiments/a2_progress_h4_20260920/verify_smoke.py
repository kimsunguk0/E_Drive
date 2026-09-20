"""Check the two actual training paths before launching the full comparison."""
import json
from pathlib import Path
from train_progress import REPORT,RUNS,ARMS,DIRECT,PROGRESS,sha

def main():
    manifests={};protocols={};streams={};records={}
    checked=json.loads((REPORT/'preflight.json').read_text())
    for arm in ARMS:
        run=RUNS/f'{arm}-s1-smoke';m=json.loads((run/'manifest.json').read_text())
        p=json.loads((run/'experiment.json').read_text())
        assert m['status']=='completed' and m['step']==2 and m['nonfinite_count']==0
        assert m['initial_model_state_sha256']==checked['initial_states'][arm]
        assert m['bn_training']['policy']=='fixed' and m['arguments']['bn_policy']=='fixed'
        logs=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
        streams[arm]={r['step']:r['sample_order_sha256'] for r in logs if 'sample_order_sha256' in r}
        assert set(streams[arm])=={1,2}
        eval_json=json.loads((run/'predictions_step2.json').read_text())
        assert eval_json['report']['n']==1998
        manifests[arm]=m;protocols[arm]=p
        records[arm]={'run':str(run),'status':m['status'],'step':m['step'],
            'nonfinite_count':m['nonfinite_count'],'initial_state_sha256':m['initial_model_state_sha256'],
            'elapsed_seconds':m['elapsed_seconds'],'manifest_sha256':sha(run/'manifest.json'),
            'sample_hashes':streams[arm],'smoke_PREFIX_not_model_quality':eval_json['report']['official_d3']}
    assert streams[DIRECT]==streams[PROGRESS]
    for key in ('loss_weights','model_config','train_rows_sha256','eval_rows_sha256','sample_order_policy','microbatch_policy','data_counts'):
        assert manifests[DIRECT][key]==manifests[PROGRESS][key],key
    a=manifests[DIRECT]['arguments'].copy();b=manifests[PROGRESS]['arguments'].copy()
    a.pop('run_dir');b.pop('run_dir');assert a==b
    for key in ('recipe','nominal_input','initializer','source'):
        assert protocols[DIRECT][key]==protocols[PROGRESS][key],key
    result={'status':'passed','arms':records,'same_all_logged_sample_hashes':True,
        'same_data_arguments_loss_recipe_input_policy':True,'preflight_sha256':sha(REPORT/'preflight.json'),
        'purpose':'Implementation smoke only, no architecture selection from two updates',
        'initial_trainable_tensors_identical':checked['all_common_initial_tensors_identical'],
        'initial_plans_intentionally_different':True}
    dest=REPORT/'smoke_summary.json';assert not dest.exists()
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
