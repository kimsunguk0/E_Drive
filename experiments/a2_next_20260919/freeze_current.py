"""Record completed work and the exact two independent comparison lineages."""
import datetime
import json
import subprocess
from common import *

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    model,ckpt=load_parent();m=ckpt['manifest']
    current={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'source_base_commit':head,'checkpoint':str(PARENT),'checkpoint_sha256':sha(PARENT),
      'model_state_sha256':tensor_state_sha256(model.state_dict()),'seed':1,'step':20554,
      'graph':'A2-QREFINE-NOM','PREFIX':.16425176970362365,
      'selection':'numerically lowest completed query-only A2 DEV; improvement over older A2 is not established',
      'status':'nominal causal producer, original supervision targets retained',
      'split_sha256':m['split_sha256'],'train_rows_sha256':m['train_rows_sha256'],
      'eval_rows_sha256':m['eval_rows_sha256'],'data_counts':m['data_counts'],
      'initializer':m['experimental_protocol']['initializer'],
      'FULL_lineage_used':False,'sources':{str(p.relative_to(ROOT)):sha(p) for p in [
         ROOT/'models/motiondrive_v2/scene_encoder.py',ROOT/'models/motiondrive_v2/shared_status_query.py',
         ROOT/'experiments/md_a2_scene_extensions_20260918/scene_extensions.py']}}
    (REPORT/'CURRENT_BASE.json').write_text(json.dumps(current,indent=2)+'\n')
    (REPORT/'INPUT_POLICY.json').write_text(json.dumps(m['experimental_protocol']['nominal_input'],indent=2)+'\n')
    arms={}
    for name,folder in [('BASE',BASE_CONTROL),('MH4',BASE_CONTROL.with_name('A2-MH4-NOM-s1')),
      ('QREFINE',PARENT.parent),('SIDE',PARENT.parent.with_name('A2-SIDE-SCENE-NOM-s1')),
      ('A2_FULL',BASE_CONTROL.with_name('A2-FULL-NOM-s1'))]:
        manifest=json.loads((folder/'manifest.json').read_text())
        report=json.loads((folder/'final_eval.json').read_text())['report']
        arms[name]={'status':manifest['status'],'step':manifest['step'],'PREFIX':report['official_d3'],
                     'run':str(folder),'evaluation_role':'in-fit' if name=='A2_FULL' else 'DEV'}
    (REPORT/'completed_work.json').write_text(json.dumps(arms,indent=2)+'\n')
    (REPORT/'COMPLETED_WORK_CHECK.md').write_text(
      '# Latest work check / 2026-09-19\n\n'+
      'Source base: '+head+'; tracked tree clean at entry. New files are isolated under a2_next_20260919.\n\n'+
      '|Arm|Status|Step|PREFIX|Role|\n|---|---|---:|---:|---|\n'+
      ''.join(f"|{k}|{v['status']}|{v['step']}|{v['PREFIX']:.12f}|{v['evaluation_role']}|\n" for k,v in arms.items())+
      '\nMR FULL is already submitted: server 0.18596892793122946. A2 FULL is completed, not server-scored.\n'+
      'No completed or running G0/G1 or learned-offset scene experiment was found in the current records.\n'+
      'Existing scene uses fixed calibration/pose projections, zero learned offsets. QREFINE changes reads, not locations.\n'+
      'G uses complete QREFINE terminal weights. Sampling independently uses the BASE upstream initializer and completed BASE control.\n'+
      'No FULL weights/teacher/cache enter DEV. No FULL restart. Physical GPUs 0–3 were idle on inspection; 4–7 are excluded.\n'+
      'Prior gradient diagnostic: 32 rows/8 batches on BASE. This user-specified extension: 32 effective batches/512 rows on QREFINE, separate shared parameter groups.\n')
    print(json.dumps(current),flush=True)

if __name__=='__main__':main()
