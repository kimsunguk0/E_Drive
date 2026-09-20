"""Fixed equal-output average after the completed continuation; no fitting."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np

ROOT=Path('/NHNHOME/data/sukim/adcl')
REPORT=ROOT/'reports/a2_fresh_continue_20260920'
RUN=ROOT/'work_dirs/a2_fresh_continue_20260920/A2-FRESH-CONT-s1'
W=np.array([11,11,5,5,2,2],np.float64)/36


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest=json.loads((RUN/'manifest.json').read_text())
    assert manifest['status']=='completed' and manifest['step']==6852
    qpath=ROOT/'work_dirs/md_a2_scene_extensions_20260918/A2-QREFINE-NOM-s1/final_eval.json'
    paths={'parent':ROOT/'work_dirs/a2_motion_fresh_20260919/A2-FRESH-NUIM-s1/final_eval.json',
           'step5710':RUN/'predictions_step5710.json','step6852':RUN/'predictions_step6852.json'}
    q=json.loads(qpath.read_text())['records']
    gt=np.array([x['gt_abs_xy'] for x in q]);qp=np.array([x['pred_abs_xy'] for x in q])
    buckets=np.array([x['bucket'] for x in q]);sessions=np.array([x['session'] for x in q])
    groups=sorted(set(buckets));unique=sorted(set(sessions))
    indices=[np.flatnonzero(sessions==s) for s in unique]
    counts=np.array([len(i) for i in indices])
    draws=np.random.default_rng(0).integers(0,len(indices),(20000,len(indices)))
    results={};scores={}
    for name,path in paths.items():
        a=json.loads(path.read_text())['records']
        assert [(x['row'],x['session']) for x in q]==[(x['row'],x['session']) for x in a]
        assert np.array_equal(gt,np.array([x['gt_abs_xy'] for x in a]))
        pred=(qp+np.array([x['pred_abs_xy'] for x in a]))/2
        error=np.linalg.norm(pred-gt,axis=-1);score=error@W;scores[name]=score
        results[name]=dict(PREFIX=float(score.mean()),L2_1s=float(error[:,:2].mean()),
            L2_2s=float(error[:,:4].mean()),L2_3s=float(error.mean()),
            groups={g:dict(n=int((buckets==g).sum()),PREFIX=float(score[buckets==g].mean()),
                contribution=float(score[buckets==g].sum()/len(score))) for g in groups},
            fresh_prediction_source=str(path),fresh_predictions_sha256=sha(path))
    for name in ['step5710','step6852']:
        delta=scores[name]-scores['parent'];sums=np.array([delta[i].sum() for i in indices])
        boot=sums[draws].sum(1)/counts[draws].sum(1)
        results[name]['versus_original_equal_average']=dict(PREFIX_delta=float(delta.mean()),
            session_CI95=np.quantile(boot,[.025,.975]).tolist(),sessions_improved=int((sums<0).sum()))
    payload=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method='fixed 0.5*QREFINE + 0.5*FRESH; unchanged weights; saved DEV predictions only',
        source_sha256=sha(Path(__file__)),QREFINE_source=str(qpath),QREFINE_sha256=sha(qpath),models=results,
        new_forward_or_training=False,coefficient_sweep=False,raw_deployment_verified=False,
        limitations=['5710 was selected on the same reused V0; not independent validation.',
            'No FULL or server score, no latency measurement.',
            'Session bootstrap conditional on 11 observed sessions.'])
    dest=REPORT/'fixed_average_after_continuation.json'
    if dest.exists():
        previous=json.loads(dest.read_text())
        assert previous['models']==results,'Do not replace inconsistent prior measurements'
    dest.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v['PREFIX'] for k,v in results.items()}))


if __name__=='__main__':main()
