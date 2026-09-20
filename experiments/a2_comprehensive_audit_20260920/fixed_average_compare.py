"""Fixed equal averages only; no coefficient search or deployment export."""
from pathlib import Path
import hashlib,json,numpy as np
ROOT=Path('/NHNHOME/data/sukim/adcl');REPORT=ROOT/'reports/a2_comprehensive_audit_20260920'

def main():
    suffixes=['md_a2_scene_extensions_20260918/A2-QREFINE-NOM-s1/final_eval.json','a2_fresh_continue_20260920/A2-FRESH-CONT-s1/predictions_step5710.json','a2_temporal_read_20260920/A2-TEMPORAL-READ-s1/final_eval.json']
    records=[json.loads((ROOT/'work_dirs'/p).read_text())['records'] for p in suffixes]
    gt=np.array([r['gt_abs_xy'] for r in records[0]]);rows=[r['row'] for r in records[0]]
    for rr in records:assert rows==[r['row'] for r in rr] and np.array_equal(gt,np.array([r['gt_abs_xy'] for r in rr]))
    p=np.array([[r['pred_abs_xy'] for r in rr] for rr in records]);sessions=np.array([r['session'] for r in records[0]]);w=np.array([11,11,5,5,2,2])/36
    delta=np.linalg.norm(p.mean(0)-gt,axis=-1)@w-np.linalg.norm((p[0]+p[1])/2-gt,axis=-1)@w
    names=sorted(set(sessions));sums=np.array([delta[sessions==s].sum() for s in names]);counts=np.array([(sessions==s).sum() for s in names]);draw=np.random.default_rng(0).integers(0,len(names),(20000,len(names)));boot=sums[draw].sum(1)/counts[draw].sum(1)
    out={'three_equal_minus_QREFINE_CONT':float(delta.mean()),'session_CI95':np.quantile(boot,[.025,.975]).tolist(),'improving_sessions':int((sums<0).sum()),'GT_rowwise_best_of_three_PREFIX':float(np.min(np.linalg.norm(p-gt[None],axis=-1)@w,axis=0).mean()),'warning':'GT selection is not deployable. Fixed equal average is posthoc repeated-DEV analysis, not raw packaged candidate.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (REPORT/'fixed_average_comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
