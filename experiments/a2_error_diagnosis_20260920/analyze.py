"""Saved-prediction diagnostics; GT is used only for labels and measurements."""
from pathlib import Path
import datetime
import hashlib
import json
import sys
import numpy as np

ROOT=Path('/NHNHOME/data/sukim/adcl')
sys.path.insert(0,str(ROOT/'experiments/md_a2_scene_extensions_20260918'))
from terminal_review import metrics

REPORT=ROOT/'reports/a2_error_diagnosis_20260920'
FRESH=ROOT/'work_dirs/a2_motion_fresh_20260919/A2-FRESH-NUIM-s1'
CONTROL=ROOT/'work_dirs/md_a2_scene_extensions_20260918/A2-QREFINE-NOM-s1'
W=np.array([11,11,5,5,2,2],np.float64)/36


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def segments(p):
    return np.diff(np.concatenate([np.zeros((len(p),1,2)),p],1),axis=1)


def from_records(records):
    names={'row':'row','session':'session','gt':'gt_abs_xy','pred':'pred_abs_xy',
           'gt_state':'gt_state','gt_state_valid':'gt_state_valid','state':'pred_state','bucket':'bucket'}
    return {k:np.array([r[v] for r in records]) for k,v in names.items()}


def subset_stats(score,mask,total):
    return dict(n=int(mask.sum()),PREFIX=float(score[mask].mean()) if mask.any() else None,
                contribution=float(score[mask].sum()/total))


def diagnose(a,pred):
    gt=a['gt'].astype(np.float64)
    dp,dg=segments(pred),segments(gt)
    lp,lg=np.linalg.norm(dp,axis=-1),np.linalg.norm(dg,axis=-1)
    mask=lg>.05
    stats,score=metrics(pred,gt,a['bucket'],mask)
    tangent=dg/np.maximum(lg[...,None],1e-8)
    err=pred-gt
    signed=(err*tangent).sum(-1)
    stats['longitudinal_signed_by_point_m']=[float(signed[mask[:,k],k].mean()) if mask[:,k].any() else None for k in range(6)]
    stats['interval_progress_error_signed_mps']=(2*(lp-lg)).mean(0).tolist()
    stats['interval_progress_error_abs_mps']=abs(2*(lp-lg)).mean(0).tolist()
    moving=a['bucket']=='nonstop'
    future_speed=2*lg[:,:4]
    speedrange=np.ptp(future_speed,axis=1)
    delta=future_speed[:,-1]-future_speed[:,0]
    # Fixed diagnostic bins, not deployment selectors or true current acceleration.
    cats=np.where(speedrange<=.5,'range_le_0.5mps',np.where(delta>.5,'speed_increase_gt0.5mps',np.where(delta<-.5,'speed_decrease_lt_minus0.5mps','nonmonotonic_other')))
    stats['nonstop_future_first2s_speed_bins']={g:subset_stats(score,moving&(cats==g),len(score)) for g in sorted(set(cats[moving]))}
    state=a['state'].astype(np.float64)
    prob=1/(1+np.exp(-np.clip(state[:,5],-60,60)))
    stats['current_stop_readout']={}
    for b in sorted(set(a['bucket'])):
        m=a['bucket']==b
        stats['current_stop_readout'][b]=dict(n=int(m.sum()),prob_mean=float(prob[m].mean()),
            prob_median=float(np.median(prob[m])),prob_ge_0p5_count=int((prob[m]>=.5).sum()),
            prob_ge_0p9_count=int((prob[m]>=.9).sum()),
            predicted_vx_mean=float(state[m,0].mean()),predicted_vx_abs_mean=float(abs(state[m,0]).mean()),
            predicted_displacement_by_point_m=np.linalg.norm(pred[m],axis=-1).mean(0).tolist(),
            GT_displacement_by_point_m=np.linalg.norm(gt[m],axis=-1).mean(0).tolist(),
            pred_signed_xy_mean=pred[m].mean(0).tolist())
    # Controlled replacement of one geometric component. These are GT-assisted
    # diagnostic reconstructions, not PREFIX optima or deployable predictions.
    up=dp/np.maximum(lp[...,None],1e-8)
    for k in range(6):
        invalid=lp[:,k]<1e-8
        up[invalid,k]=up[invalid,k-1] if k else np.array([1.,0.])
    ug=dg/np.maximum(lg[...,None],1e-8)
    ug=np.where((lg<1e-8)[...,None],up,ug)
    rep={'GT_lengths_predicted_directions':np.cumsum(lg[...,None]*up,1),
         'predicted_lengths_GT_directions':np.cumsum(lp[...,None]*ug,1)}
    stats['GT_assisted_component_replacement']={k:metrics(v,gt,a['bucket'],mask)[0] for k,v in rep.items()}
    stats['sessions']={s:subset_stats(score,a['session']==s,len(score)) for s in sorted(set(a['session']))}
    return stats


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    dest=REPORT/'analysis.json'
    assert not dest.exists()
    sources={}
    models={}
    terminal={}
    for name,path in [('QREFINE',CONTROL/'final_eval.json'),('FRESH',FRESH/'final_eval.json')]:
        v=json.loads(path.read_text())
        a=from_records(v['records']);terminal[name]=a
        models[name]=diagnose(a,a['pred'])
        models[name]['auxiliary']=v['report']
        sources[str(path)]=digest(path)
    for k in ('row','session','gt','bucket','gt_state'):
        assert np.array_equal(terminal['QREFINE'][k],terminal['FRESH'][k])
    mixed=dict(terminal['FRESH'])
    mixed['pred']=(terminal['QREFINE']['pred']+terminal['FRESH']['pred'])/2
    models['fixed_equal_output_average']=diagnose(mixed,mixed['pred'])
    # There is no ensemble state head; remove diagnostics misleadingly borrowing one.
    del models['fixed_equal_output_average']['current_stop_readout']
    curve=[]
    for step in (3426,6852,10278,13704,17130,20554):
        path=FRESH/f'predictions_step{step}.json'
        v=json.loads(path.read_text());a=from_records(v['records'])
        d=diagnose(a,a['pred'])
        curve.append(dict(step=step,**{k:d[k] for k in ('PREFIX','groups','per_waypoint_L2',
            'weighted_longitudinal_abs_m','weighted_lateral_abs_m','current_stop_readout')},
            occ_iou=v['report']['occ_iou'],lane_iou=v['report']['lane_iou'],
            state_mae=v['report']['state_mae_vx_vy_ax_ay_yawrate']))
        sources[str(path)]=digest(path)
    probes={}
    for name in ('QREFINE','FRESH'):
        path=REPORT/f'probe_{name}.json'
        v=json.loads(path.read_text());probes[name]=v
        sources[str(path)]=digest(path)
        for scope in ('V0','train_probe'):
            path=Path(v[scope]['arrays_path'])
            assert digest(path)==v[scope]['arrays_sha256']
            with np.load(path,allow_pickle=False) as z:
                a={k:z[k] for k in z.files}
            v[scope]['baseline_detail']=diagnose(a,a['ON'])
    with np.load(REPORT/'QREFINE_train_probe.npz',allow_pickle=False) as q,np.load(REPORT/'FRESH_train_probe.npz',allow_pickle=False) as f:
        assert np.array_equal(q['row'],f['row']) and np.array_equal(q['gt'],f['gt'])
    result=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources_sha256=sources,
        script_sha256=digest(Path(__file__)),models=models,fresh_learning_curve=curve,frozen_probes=probes,
        limitations=['Repeated DEV, not hidden-test score.','Stop readout describes current state, not future departure.',
            'GT component replacement is descriptive; not PREFIX-minimizing or deployable.',
            'Future-progress bins use GT for diagnosis only; not runtime gating.',
            'Frozen OFF probes are OOD interventions, not retrained removal ablations.',
            'Longitudinal/lateral projections are not additive PREFIX contributions.'])
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(dest),models={k:dict(PREFIX=v['PREFIX'],groups=v['groups'],
        future_bins=v['nonstop_future_first2s_speed_bins'],component_replacement={n:x['PREFIX'] for n,x in v['GT_assisted_component_replacement'].items()}) for k,v in models.items()})))


if __name__=='__main__':
    main()
