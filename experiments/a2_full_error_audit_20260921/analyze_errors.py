"""Explicitly separated FULL in-fit, held-out DEV, and train U-turn statistics."""
from pathlib import Path
import json,hashlib,csv,datetime,subprocess
import numpy as np
ROOT=Path('/NHNHOME/data/sukim/adcl');OUT=ROOT/'reports/a2_full_error_audit_20260921'
W=np.array([11,11,5,5,2,2],np.float64)/36
LABELS=('LANE_KEEP','TURN_LEFT','TURN_RIGHT','LANE_CHANGE_L','LANE_CHANGE_R','U_TURN')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def interval(p):return np.diff(np.concatenate([np.zeros((len(p),1,2)),p],1),axis=1)
def analyze(records,command):
    p=np.array([r['pred_abs_xy'] for r in records],np.float64);g=np.array([r['gt_abs_xy'] for r in records],np.float64)
    session=np.array([r['session'] for r in records]);bucket=np.array([r['bucket'] for r in records]);n=len(p)
    d=interval(g);length=np.linalg.norm(d,axis=-1);unit=d/np.maximum(length[...,None],1e-12);valid=length>.05
    err=p-g;e=np.linalg.norm(err,axis=-1);score=e@W
    longitudinal=(err*unit).sum(-1);lateral=unit[...,0]*err[...,1]-unit[...,1]*err[...,0]
    vw=valid*W;total_lat=float((abs(lateral)*vw).sum());total_long=float((abs(longitudinal)*vw).sum())
    speed=length*2;pred_delta=interval(p);pred_len=np.linalg.norm(pred_delta,axis=-1)
    angle=np.unwrap(np.arctan2(d[...,1],d[...,0]),axis=1);pred_angle=np.arctan2(pred_delta[...,1],pred_delta[...,0])
    heading_error=np.arctan2(np.sin(pred_angle-angle),np.cos(pred_angle-angle))
    max_angle=np.max(np.abs(angle),axis=1)*180/np.pi
    # Diagnostic geometry, distinct from supplied navigation semantics. Includes
    # current heading 0; thresholds fixed before scores were computed.
    moving=valid.all(1)&(length.sum(1)>=3.)
    geometry=np.full(n,'low_motion_or_partial_stop',dtype='<U40')
    geometry[moving&(max_angle<5)]='near_straight_lt5deg'
    geometry[moving&(max_angle>=5)&(max_angle<15)]='gentle_bend_5to15deg'
    geometry[moving&(max_angle>=15)&(angle[:,-1]>0)]='left_bend_ge15deg'
    geometry[moving&(max_angle>=15)&(angle[:,-1]<=0)]='right_bend_ge15deg'
    geometry[moving&(abs(angle[:,-1])*180/np.pi>=135)]='turnback_ge135deg'
    def stats(mask):
        count=int(mask.sum())
        if not count:return {'n':0,'sessions':0,'PREFIX':None,'score_contribution':0.,'lateral_abs_total_share':0.}
        v=valid[mask];weight=vw[mask];den=weight.sum();ll=longitudinal[mask];tt=lateral[mask];ee=e[mask]
        mean_valid=lambda a:[float(a[:,k][v[:,k]].mean()) if v[:,k].any() else None for k in range(6)]
        # Exact descriptive allocation: L2 = long^2/L2 + lat^2/L2 at valid
        # tangents. Not a causal decomposition or error-removal oracle.
        shares=np.maximum(ee,1e-12)
        long_alloc=float(((ll**2/shares)*weight).sum()/n)
        lat_alloc=float(((tt**2/shares)*weight).sum()/n)
        invalid_alloc=float((ee*(~v)*W).sum()/n)
        contribution=float(score[mask].sum()/n)
        assert abs(long_alloc+lat_alloc+invalid_alloc-contribution)<1e-9
        return {'n':count,'sessions':len(set(session[mask])),'PREFIX':float(score[mask].mean()),
            'score_contribution':contribution,'score_share':contribution/float(score.mean()),
            'L2_prefix':[float(ee[:,:k].mean()) for k in (2,4,6)],'point_L2':ee.mean(0).tolist(),
            'first2s_score_contribution':float((ee[:,:4]@W[:4]).sum()/n),'last_two_score_contribution':float((ee[:,4:]@W[4:]).sum()/n),
            'longitudinal_MAE':float((abs(ll)*weight).sum()/den) if den else None,
            'lateral_MAE':float((abs(tt)*weight).sum()/den) if den else None,
            'longitudinal_signed_mean':float((ll*weight).sum()/den) if den else None,
            'lateral_signed_mean':float((tt*weight).sum()/den) if den else None,
            'lateral_abs_total_share':float((abs(tt)*weight).sum()/total_lat) if total_lat else 0.,
            'longitudinal_abs_total_share':float((abs(ll)*weight).sum()/total_long) if total_long else 0.,
            'lateral_point_MAE':mean_valid(abs(tt)),'longitudinal_point_MAE':mean_valid(abs(ll)),
            'heading_point_MAE_deg':mean_valid(abs(heading_error[mask])*180/np.pi),
            'valid_tangent_points':int(v.sum()),'valid_tangent_fraction':float(v.mean()),
            'PREFIX_allocation_by_squared_projection':{'longitudinal':long_alloc,'lateral':lat_alloc,'invalid_tangent':invalid_alloc},
            'progress_speed_error_first2_common_MAE':float(abs((2*(pred_len-length))[mask,:4].mean(1)).mean()),
            'progress_speed_error_first2_residual_RMS':float(np.sqrt(np.mean((2*(pred_len-length)[mask,:4]-2*(pred_len-length)[mask,:4].mean(1,keepdims=True))**2)))}
    speed0=speed[:,0];stable=np.ptp(speed[:,:4],axis=1)<=.5
    masks={
        'groups':{k:bucket==k for k in sorted(set(bucket))},
        'semantic':{k:command==i for i,k in enumerate(LABELS)},
        'geometry':{k:geometry==k for k in ('near_straight_lt5deg','gentle_bend_5to15deg','left_bend_ge15deg','right_bend_ge15deg','turnback_ge135deg','low_motion_or_partial_stop')},
        'future_first2_speed_change':{'nonstop_range_le_0p5mps':(bucket=='nonstop')&stable,'nonstop_range_gt_0p5mps':(bucket=='nonstop')&~stable,'stop_or_depart':bucket!='nonstop'},
        'first_interval_speed_bins':{f'{lo}_to_{hi}_mps':(speed0>=lo)&(speed0<hi) for lo,hi in ((0,.5),(.5,2),(2,5),(5,10),(10,15),(15,float('inf')))},
        'sessions':{s:session==s for s in sorted(set(session))}}
    result={'all':stats(np.ones(n,bool)),**{k:{label:stats(mask) for label,mask in group.items()} for k,group in masks.items()}}
    result['tail_rows']={str(q):{'rows':int(np.ceil(n*q)),'share_of_total_score':float(np.sort(score)[-int(np.ceil(n*q)):].sum()/score.sum())} for q in (.01,.05,.10)}
    result['curvature_and_command_crosstab']={label:{geo:int(((command==i)&(geometry==geo)).sum()) for geo in sorted(set(geometry))} for i,label in enumerate(LABELS)}
    result['turn_direction_diagnostic']={}
    for i,label in [(1,'TURN_LEFT'),(2,'TURN_RIGHT')]:
        mask=(command==i)&valid[:,-1]&(max_angle>=15)
        if mask.any():
            sign=np.sign(angle[mask,-1]);signed=sign*heading_error[mask,-1]*180/np.pi
            result['turn_direction_diagnostic'][label]={'n':int(mask.sum()),'mean_predicted_rotation_error_deg':float(signed.mean()),
                'fraction_less_rotation_than_GT':float((signed<0).mean()),'note':'Terminal segment angle relative to current ego heading; negative is less rotation. A correlated diagnostic, not cause identification.'}
    rows=[]
    for i,r in enumerate(records):
        rows.append({'row':r['row'],'session':r['session'],'scenario':r['scenario'],'frame':r['frame'],'bucket':r['bucket'],
            'command':LABELS[int(command[i])],'geometry':str(geometry[i]),'PREFIX':float(score[i]),
            'first_interval_GT_progress_mps':float(speed0[i]),'GT_max_abs_segment_heading_deg':float(max_angle[i]),
            'lateral_weighted_abs':float((abs(lateral[i])*vw[i]).sum()),'longitudinal_weighted_abs':float((abs(longitudinal[i])*vw[i]).sum()),
            'endpoint_L2':float(e[i,-1]),'endpoint_lateral_signed':float(lateral[i,-1]) if valid[i,-1] else None,
            'endpoint_longitudinal_signed':float(longitudinal[i,-1]) if valid[i,-1] else None})
    return result,rows

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    files={'FULL_infit_V0':ROOT/'work_dirs/a2_progress_full_20260921/A2-H4-PROGRESS-FULL-s1/final_eval.json',
        'DEV_PROGRESS_heldout':ROOT/'work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1/final_eval.json',
        'DEV_DIRECT_heldout':ROOT/'work_dirs/a2_progress_h4_20260920/A2-H4-DIRECT-s1/final_eval.json'}
    if (OUT/'UTURN_train_probe.json').exists():files['FULL_infit_UTURN_train375']=OUT/'UTURN_train_probe.json'
    path=ROOT/'data/etri/motiondrive_v2/a2_command_20260919/provided_commands.npz'
    with np.load(path,allow_pickle=False) as z:cr=z['row'];cv=z['command']
    result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':{},'source_files':{},
        'definitions':{'official_weights':W.tolist(),'tangent_mask':'GT interval chord length >0.05 m, same across models',
            'projections':'signed longitudinal dot(error,GT unit tangent); signed lateral cross(GT unit tangent,error), positive left of GT segment',
            'MAE_normalizer':'sum of PREFIX weights across valid tangent points in the specified group',
            'lateral_share':'share of total PREFIX-weighted absolute lateral error; not a share of the official L2 score',
            'allocation':'Exact descriptive L2 split long^2/L2, lat^2/L2, invalid tangent; not causal contribution or separately achievable gains',
            'command':'provided current semantic command, not GT-derived turn class; LANE_KEEP includes road bends',
            'geometry':'moving requires all six GT segment lengths >0.05m and total length >=3m. Max absolute segment angle relative to current ego forward <5deg straight; [5,15) gentle; >=15 left/right by terminal angle. Terminal abs>=135 turnback. Other rows low-motion/partial-stop. These are geometric labels, not command semantics.',
            'FULL_scope':'V0 was included in FULL training. Neither FULL in-fit V0 nor the U-turn train probe is unseen/test accuracy.',
            'server_scope':'Only official aggregate L2 values are available. No server per-row/turn/longitudinal/lateral error can be computed.'}}
    reference=None
    for name,f in files.items():
        obj=json.loads(f.read_text());r=obj['records'];ids=np.array([x['row'] for x in r]);pos=np.searchsorted(cr,ids);assert np.array_equal(cr[pos],ids)
        if len(r)==1998:
            identity=[(x['row'],x['gt_abs_xy']) for x in r]
            if reference is None:reference=identity
            else:assert identity==reference
        stats,rows=analyze(r,cv[pos]);assert abs(stats['all']['PREFIX']-obj['report']['official_d3'])<1e-6
        result['models'][name]=stats;result['source_files'][name]={'path':str(f.relative_to(ROOT)),'sha256':digest(f),'rows':len(r),'step':obj['report']['step']}
        with (OUT/(name+'_rows.csv')).open('w') as h:
            writer=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    command_manifest=json.loads((path.parent/'manifest.json').read_text())
    result['command_coverage']={'DEV':command_manifest['splits']['tune']['counts'],'train310':command_manifest['splits']['train']['counts'],
        'test_input_only_counts':command_manifest['test_input_check']['counts'],'test_counts_not_used_for_reweighting':True}
    result['analysis_script_sha256']=digest(Path(__file__))
    result['inspected_source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    run=files['FULL_infit_V0'].parent
    manifest=json.loads((run/'manifest.json').read_text())
    result['FULL_training']={k:manifest[k] for k in ['data_counts','step','status','nonfinite_count','elapsed_seconds','initial_model_state_sha256','train_rows_sha256','eval_rows_sha256','split_sha256','git_sha']}
    result['FULL_training']['scope']='Learning curve is in-fit V0; cannot establish unseen benefit of continuing FULL.'
    result['FULL_training']['learning_curve']=sorted([
        {k:v[k] for k in ['step','official_d3','state_mae_vx_vy_ax_ay_yawrate','occ_iou','lane_iou']}
        for v in [json.loads(p.read_text()) for p in run.glob('eval_step*.json')]],key=lambda d:d['step'])
    server=json.loads((ROOT/'reports/a2_progress_full_20260921/SERVER_RESULT_20260921.json').read_text())
    result['server_user_supplied']=server
    a,b,c=[server['metrics'][f'L2_{s}s'] for s in (1,2,3)]
    block_sums=np.array([2*a,4*b-2*a,6*c-4*b])
    result['server_block_decomposition']={'pair_point_mean_L2':(block_sums/2).tolist(),
        'PREFIX_contribution':(block_sums*np.array([11,5,2])/36).tolist(),
        'first2s_score_share':float((block_sums*np.array([11,5,2])/36)[:2].sum()/server['score']),
        'scope':'Exact from prefix aggregates; cannot recover individual points or direction/command errors.'}
    (OUT/'STATISTICS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (OUT/'GROUP_SUMMARY.csv').open('w') as h:
        fields=['model','group_type','group','n','sessions','PREFIX','score_contribution','longitudinal_MAE','lateral_MAE','lateral_abs_total_share']
        w=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader()
        for model,m in result['models'].items():
            for kind in ('groups','semantic','geometry','future_first2_speed_change','first_interval_speed_bins'):
                for group,v in m[kind].items():w.writerow({'model':model,'group_type':kind,'group':group,**v})
    print(json.dumps({k:{'all':m['all'],'semantic':m['semantic'],'geometry':m['geometry']} for k,m in result['models'].items() if k.startswith('FULL')},indent=2))
if __name__=='__main__':main()
