"""Same-row residual audit. GT-assisted reconstructions are diagnosis only."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path('/NHNHOME/data/sukim/adcl');REPORT=ROOT/'reports/a2_comprehensive_audit_20260920'
spec=importlib.util.spec_from_file_location('old_error_analysis',ROOT/'experiments/a2_error_diagnosis_20260920/analyze.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
W=np.array([11,11,5,5,2,2])/36
paths={'QREFINE':'md_a2_scene_extensions_20260918/A2-QREFINE-NOM-s1/final_eval.json','FRESH':'a2_motion_fresh_20260919/A2-FRESH-NUIM-s1/final_eval.json','CONT_SELECTED':'a2_fresh_continue_20260920/A2-FRESH-CONT-s1/predictions_step5710.json','CONT_TERMINAL':'a2_fresh_continue_20260920/A2-FRESH-CONT-s1/final_eval.json','TEMPORAL':'a2_temporal_read_20260920/A2-TEMPORAL-READ-s1/final_eval.json'}

def main():
    datasets={};sources={}
    for name,suffix in paths.items():
        path=ROOT/'work_dirs'/suffix;v=json.loads(path.read_text());datasets[name]=old.from_records(v['records']);sources[str(path)]=old.digest(path)
    ref=datasets['CONT_SELECTED'];gt=ref['gt'];N=len(gt)
    for a in datasets.values():
        for k in ('row','session','gt','bucket','gt_state'):assert np.array_equal(ref[k],a[k]),k
    combinations={'QREFINE_CONT':('QREFINE','CONT_SELECTED'),'QREFINE_TEMPORAL':('QREFINE','TEMPORAL'),'CONT_TEMPORAL':('CONT_SELECTED','TEMPORAL'),'THREE_EQUAL':('QREFINE','CONT_SELECTED','TEMPORAL')}
    for name,parts in combinations.items():
        a=dict(ref);a['pred']=np.mean([datasets[k]['pred'] for k in parts],axis=0);datasets[name]=a
    dg=old.segments(gt);lg=np.linalg.norm(dg,axis=-1);speed=lg*2
    # Curvature bins use GT displacement heading change; label diagnostics only.
    headings=np.unwrap(np.arctan2(dg[...,1],dg[...,0]),axis=1)
    turn=np.rad2deg(abs(headings[:,-1]-headings[:,0]));moving=ref['bucket']=='nonstop'
    stable=np.ptp(speed[:,:4],axis=1)<=.5
    all_scores={};details={}
    for name,a in datasets.items():
        p=a['pred'];e=np.linalg.norm(p-gt,axis=-1);score=e@W;all_scores[name]=score
        d=old.diagnose(a,p)
        if name in combinations:del d['current_stop_readout']
        else:
            src=json.loads((ROOT/'work_dirs'/paths[name]).read_text());d['auxiliary']=src.get('report')
        d['point_PREFIX_contributions']=(e.mean(0)*W).tolist()
        d['first2s_fraction']=float((e[:,:4]@W[:4]).mean()/score.mean())
        d['speed_bins']={}
        for label,mask in [('lt2',speed[:,:4].mean(1)<2),('2to5',(speed[:,:4].mean(1)>=2)&(speed[:,:4].mean(1)<5)),('5to10',(speed[:,:4].mean(1)>=5)&(speed[:,:4].mean(1)<10)),('ge10',speed[:,:4].mean(1)>=10)]:d['speed_bins'][label]=old.subset_stats(score,moving&mask,N)
        d['heading_change_bins']={label:old.subset_stats(score,moving&mask,N) for label,mask in [('lt5deg',turn<5),('5to15deg',(turn>=5)&(turn<15)),('ge15deg',turn>=15)]}
        d['stable_nonstop_first2s_contribution']=float((e[moving&stable,:4]@W[:4]).sum()/N)
        d['worst_fraction_contribution']={str(f):float(np.sort(score)[-int(np.ceil(N*f)):].sum()/N) for f in (.01,.05,.1,.2)}
        dp=old.segments(p);speederr=(np.linalg.norm(dp,axis=-1)-lg)*2
        common=speederr[:,:4].mean(1);varying=speederr[:,:4]-common[:,None]
        d['nonstop_first2s_interval_speed_error']={'common_signed_mean_mps':float(common[moving].mean()),'common_abs_mean_mps':float(abs(common[moving]).mean()),'varying_RMS_mps':float(np.sqrt(np.mean(varying[moving]**2))),'constant_component_MSE_fraction':float(np.mean(common[moving]**2)/np.mean(speederr[moving,:4]**2)),'common_abs_quantiles':np.quantile(abs(common[moving]),[.5,.9,.99]).tolist()}
        # Do errors cancel across samples? This does not fit a deployable correction.
        d['signed_xy_bias_by_point_m']=(p-gt).mean(0).tolist()
        d['session_common_speed_error']={s:float(common[moving&(ref['session']==s)].mean()) if (moving&(ref['session']==s)).any() else None for s in sorted(set(ref['session']))}
        d['target_budget_0p12']={'needed_absolute_reduction':float(score.mean()-.12),'needed_relative_reduction':float(1-.12/score.mean()),'perfect_depart_and_steady_floor':float(score[moving].sum()/N),'late_only_relative_reduction_needed':float((score.mean()-.12)/(e[:,4:]@W[4:]).mean()),'first2s_only_relative_reduction_needed':float((score.mean()-.12)/(e[:,:4]@W[:4]).mean())}
        details[name]=d
    names=list(paths);scores=np.stack([all_scores[n] for n in names],axis=1)
    errors=np.stack([datasets[n]['pred']-gt for n in names])
    pair={}
    for i,a in enumerate(names):
        for j,b in enumerate(names[:i]):
            pair[a+'__'+b]={'row_PREFIX_error_correlation':float(np.corrcoef(scores[:,i],scores[:,j])[0,1]),'weighted_error_vector_cosine':float(np.sum(errors[i]*errors[j]*W[None,:,None])/np.sqrt(np.sum(errors[i]**2*W[None,:,None])*np.sum(errors[j]**2*W[None,:,None]))),'prediction_PREFIX_distance':float((np.linalg.norm(errors[i]-errors[j],axis=-1)@W).mean())}
    oracle_idx=np.argmin(scores,axis=1);oracle=np.stack([datasets[n]['pred'] for n in names],axis=1)[np.arange(N),oracle_idx]
    oracle_score=np.min(scores,axis=1)
    cache=np.load('/tmp/pm97/data/etri/ego_cache.npz',allow_pickle=False);scenes=cache['scenarios'][cache['scen_idx'][ref['row']]]
    hard=all_scores['QREFINE_CONT'];rank=np.argsort(-hard)[:30]
    hardest=[dict(row=int(ref['row'][i]),scene=str(scenes[i]),session=str(ref['session'][i]),frame=int(cache['frame'][ref['row'][i]]),bucket=str(ref['bucket'][i]),PREFIX=float(hard[i]),all_model_best=float(oracle_score[i]),future_mean_speed_mps=float(speed[i,:4].mean()),first2s_speed_range_mps=float(np.ptp(speed[i,:4])),heading_change_deg=float(turn[i])) for i in rank]
    scene_table={s:old.subset_stats(hard,scenes==s,N) for s in sorted(set(scenes))}
    scene_table=dict(sorted(scene_table.items(),key=lambda x:-x[1]['contribution']))
    out=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources_sha256=sources,script_sha256=old.digest(Path(__file__)),models=details,pairwise=pair,GT_rowwise_best_of_existing_singles={'PREFIX':float(oracle_score.mean()),'chosen_counts':{n:int((oracle_idx==i).sum()) for i,n in enumerate(names)},'not_deployable':True},worst_rows=hardest,reference_scene_contributions=scene_table,limitations=['Repeated V0; fixed additional averages are diagnostic and not raw deployment validated.','No coefficient optimization, new training, or parameter averaging.','GT component replacements and best-of-model selector are not deployable and not learning guarantees.','Longitudinal/lateral projections are not additive PREFIX components.','Speed and heading bins are GT-based diagnostic labels, not actual current state or controls.'])
    dest=REPORT/'saved_prediction_analysis.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({n:{'PREFIX':d['PREFIX'],'groups':d['groups'],'first2s_fraction':d['first2s_fraction'],'stable':d['nonstop_future_first2s_speed_bins'],'progress':d['nonstop_first2s_interval_speed_error'],'GT_replacement':{k:v['PREFIX'] for k,v in d['GT_assisted_component_replacement'].items()}} for n,d in details.items()}))

if __name__=='__main__':main()
