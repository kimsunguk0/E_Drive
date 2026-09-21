"""CPU-only analysis of existing held-out DEV predictions after official result.

No model inference, training, test-target fitting, or submission generation.
Fixed comparisons were chosen from the matched H4 DIRECT/PROGRESS terminal
pair and the immediately preceding PROGRESS checkpoint; no ratio sweep.
"""
from pathlib import Path
import datetime,hashlib,json
import numpy as np

ROOT=Path('/NHNHOME/data/sukim/adcl')
REPORT=ROOT/'reports/a2_after_submission_20260921'
W=np.array([11,11,5,5,2,2],np.float64)/36

SERVER={'score':0.1336848279459137,'metrics':{
    'L2_1s':0.07277257524810496,'L2_2s':0.12524497423827546,
    'L2_3s':0.20303693435136064,'L2_avg':0.1336848279459137,
    'flops':730044861120,'cutoff':True,'elapsed_ms':303,
    'runtime':{'pull_mode':True,'num_inflight':1,'want':31,'pull_interval_sec':10}},
    'provenance':{'source':'User supplied official FULL scoring response in this conversation',
        'reported_rank':4,'rank_source':'user report, leaderboard not independently queried',
        'submission_id':None,'official_submission_timestamp':None,
        'run':'A2-H4-PROGRESS-FULL-s1','terminal_step':24931,
        'checkpoint_sha256':'dae99f86f29e92296b1d036db3f7933d41bf9a5c3dd3af5153416a8eaa676b14',
        'submission_zip_sha256':'3981ee5541458d62d1b3cb94cfe9f67ab619672e892879eec92f8e08e8f0c1cc',
        'elapsed_ms_interpretation':'server harness time; not RTX4090 model-forward latency'}}

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def intervals(p):return np.diff(np.concatenate([np.zeros((len(p),1,2)),p],1),axis=1)
def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    runroot=ROOT/'work_dirs/a2_progress_h4_20260920'
    paths={'DIRECT':runroot/'A2-H4-DIRECT-s1/predictions_step20554.json',
           'PROGRESS':runroot/'A2-H4-PROGRESS-s1/predictions_step20554.json',
           'PROGRESS_17130':runroot/'A2-H4-PROGRESS-s1/predictions_step17130.json'}
    data={};truth=None;provenance={}
    for name,p in paths.items():
        obj=json.loads(p.read_text());r=obj['records']
        ids=[(a['row'],a['session'],a['scenario'],a['frame']) for a in r]
        pred=np.array([a['pred_abs_xy'] for a in r],np.float64)
        gt=np.array([a['gt_abs_xy'] for a in r],np.float64)
        if truth is None:
            truth=gt;identity=ids;groups=np.array([a['bucket'] for a in r]);sessions=np.array([a['session'] for a in r])
        assert ids==identity and np.array_equal(gt,truth) and pred.shape==(1998,6,2) and np.isfinite(pred).all()
        data[name]=pred;provenance[name]={'path':str(p.relative_to(ROOT)),'sha256':digest(p),'checkpoint_step':obj['report']['step']}
    def stats(pred):
        e=np.linalg.norm(pred-truth,axis=-1)
        return {name:{'n':int(mask.sum()),'PREFIX':float((e[mask]@W).mean()),
            'per_point_L2':e[mask].mean(0).tolist(),
            'first2s_total_contribution':float((e[mask,:4]@W[:4]).sum()/len(e)),
            'last_two_total_contribution':float((e[mask,4:]@W[4:]).sum()/len(e))}
            for name,mask in [('all',np.ones(len(e),bool)),*[(n,groups==n) for n in ('nonstop','depart','steady')]]}
    gd=intervals(truth);gl=np.linalg.norm(gd,axis=-1);gu=gd/np.maximum(gl[...,None],1e-12)
    component={}
    for name,p in data.items():
        pd=intervals(p);pl=np.linalg.norm(pd,axis=-1);pu=pd/np.maximum(pl[...,None],1e-12)
        mask=groups=='nonstop';er=p-truth;long=(er*gu).sum(-1);lat=er[...,1]*gu[...,0]-er[...,0]*gu[...,1]
        component[name]={'nonstop_longitudinal_signed_mean':long[mask].mean(0).tolist(),
            'nonstop_lateral_signed_mean':lat[mask].mean(0).tolist(),
            'nonstop_longitudinal_MAE':np.abs(long[mask]).mean(0).tolist(),
            'nonstop_lateral_MAE':np.abs(lat[mask]).mean(0).tolist(),
            'nonstop_interval_length_signed_mean':(pl-gl)[mask].mean(0).tolist(),
            'nonstop_interval_length_MAE':np.abs(pl-gl)[mask].mean(0).tolist(),
            'GT_length_substitution_not_optimized_oracle':stats((gl[...,None]*pu).cumsum(1))['all']['PREFIX'],
            'GT_direction_substitution_not_optimized_oracle':stats((pl[...,None]*gu).cumsum(1))['all']['PREFIX'],
            'GT_substitution_note':'Diagnostic only; neither deployable nor a PREFIX-minimizing oracle. No GT input to fixed prediction combinations below.'}
    dd=intervals(data['DIRECT']);dl=np.linalg.norm(dd,axis=-1);du=dd/np.maximum(dl[...,None],1e-12)
    pd=intervals(data['PROGRESS']);pl=np.linalg.norm(pd,axis=-1);pu=pd/np.maximum(pl[...,None],1e-12)
    combinations={
        'FIXED_HALF_DIRECT_PROGRESS':(data['DIRECT']+data['PROGRESS'])/2,
        'FIXED_HALF_PROGRESS_TAIL_SNAPSHOTS':(data['PROGRESS_17130']+data['PROGRESS'])/2,
        'PROGRESS_LENGTH_DIRECT_DIRECTION':(pl[...,None]*du).cumsum(1),
        'DIRECT_LENGTH_PROGRESS_DIRECTION':(dl[...,None]*pu).cumsum(1)}
    base=np.linalg.norm(data['PROGRESS']-truth,axis=-1)@W;us=sorted(set(sessions));cnt=np.array([(sessions==s).sum() for s in us])
    rng=np.random.default_rng(20260921);indices=rng.integers(0,len(us),(10000,len(us)))
    comparison={}
    for name,p in combinations.items():
        d=np.linalg.norm(p-truth,axis=-1)@W-base;sums=np.array([d[sessions==s].sum() for s in us]);boot=sums[indices].sum(1)/cnt[indices].sum(1)
        comparison[name]={'metrics':stats(p),'delta_vs_PROGRESS':float(d.mean()),
            'session_paired_CI95':np.quantile(boot,[.025,.975]).tolist(),'sessions_improved':int((sums<0).sum()),
            'session_deltas':{s:float(sums[i]/cnt[i]) for i,s in enumerate(us)},
            'interpretation':'Saved-prediction diagnostic, not a newly trained model or approved submission transformation; no GT-based per-row selection.'}
    m=SERVER['metrics'];blocks=[m['L2_1s'],2*m['L2_2s']-m['L2_1s'],3*m['L2_3s']-2*m['L2_2s']]
    contributions=(np.array(blocks)*np.array([22,10,4])/36).tolist()
    server_analysis={'block_points_seconds':[[.5,1.],[1.5,2.],[2.5,3.]],'block_mean_L2':blocks,
        'score_contributions':contributions,'first2s_score_share':sum(contributions[:2])/m['L2_avg'],
        'previous_MR_FULL_score':.18596892793122946,'relative_improvement_vs_MR_FULL':1-m['L2_avg']/.18596892793122946,
        'target':.12,'additional_absolute_reduction_required':m['L2_avg']-.12,'additional_relative_reduction_required':1-.12/m['L2_avg'],
        'scope':'Official aggregate metric decomposition only. Server per-row errors/GT are unavailable; DEV group and causal findings are not asserted for server.'}
    result={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':1998,'sessions':11,
        'source_predictions':provenance,'baseline':{k:stats(v) for k,v in data.items()},'component_diagnostics':component,
        'fixed_comparisons':comparison,'server_analysis':server_analysis,
        'bootstrap':'10000 conditional paired cluster draws from the same reused 11 DEV sessions; seed 20260921. Not independent validation or seed uncertainty.',
        'new_GPU_inference':False,'new_training':False,'new_official_submission':False,
        'zero_direction_segments':{'DIRECT':int((dl<1e-12).sum()),'PROGRESS':int((pl<1e-12).sum())}}
    (REPORT/'DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n')
    target=ROOT/'reports/a2_progress_full_20260921/SERVER_RESULT_20260921.json'
    if target.exists():assert json.loads(target.read_text())==SERVER,'Do not replace a different official record'
    else:target.write_text(json.dumps(SERVER,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'server':SERVER['score'],'reported_rank':4,'server_analysis':server_analysis,
        'comparisons':{k:{'PREFIX':v['metrics']['all']['PREFIX'],'delta':v['delta_vs_PROGRESS'],'CI':v['session_paired_CI95'],'sessions_improved':v['sessions_improved']} for k,v in comparison.items()}},indent=2))
if __name__=='__main__':main()
