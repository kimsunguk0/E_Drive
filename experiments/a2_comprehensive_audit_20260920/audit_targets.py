"""Recompute DEV cached labels from original metadata; no model update."""
from pathlib import Path
import datetime,hashlib,json,sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
ROOT=Path('/NHNHOME/data/sukim/adcl');REPORT=ROOT/'reports/a2_comprehensive_audit_20260920'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from motiondrive_v2_data import full_pose_matrices,motion_targets
sys.path.insert(0,str(ROOT/'experiments/md_a2_deploy_status_20260918'))
from nominal_status import fit_status5

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quant(a):return {str(q):float(x) for q,x in zip([0,.01,.5,.99,1],np.quantile(a,[0,.01,.5,.99,1]))}

def main():
    split_path=ROOT/'data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json'
    split=json.loads(split_path.read_text());cache=np.load('/tmp/pm97/data/etri/ego_cache.npz',allow_pickle=False)
    nominal=np.load(ROOT/'data/etri/motiondrive_v2/a2_nominal_status_20260918/tune.npz',allow_pickle=False)
    status_lookup={int(r):s for r,s in zip(nominal['row'],nominal['status5'])}
    rec=json.loads((ROOT/'work_dirs/a2_fresh_continue_20260920/A2-FRESH-CONT-s1/predictions_step5710.json').read_text())['records'];record_lookup={r['row']:r for r in rec}
    tr=set(split['splits']['train']);va=set(split['splits']['tune'])
    sessions={k:set(split['scene_to_session'][s] for s in split['splits'][k]) for k in ('train','tune','val')}
    assert not tr&va and not sessions['train']&sessions['tune']
    result={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'split_sha256':digest(split_path),'cache_sha256':digest('/tmp/pm97/data/etri/ego_cache.npz'),'train_tune_scene_overlap':0,'train_tune_session_overlap':0,'sessions':{k:len(v) for k,v in sessions.items()},'future_offsets_frames':[5,10,15,20,25,30],'goal_offset_frames':50,'source_files_sha256':{},'scenes':{}}
    dt=[];future_dt=[];errors=[];goals=[];state_diff=[];status_diff=[];vx_vs_future=[];counts={'train':0,'tune':0};verified_v0=[]
    for si,s0 in enumerate(cache['scenarios']):
        s=str(s0)
        if s not in tr|va:continue
        folder=Path('/tmp/pm97/data/etri/meta_train')/s
        ts_path=folder/'meta/timestamps.parquet';pose_path=folder/'annotation/ego_pose.parquet'
        ts=pd.read_parquet(ts_path);ep=pd.read_parquet(pose_path)
        joined=ts[['frame_id','timestamp']].merge(ep,on='timestamp',how='left',validate='one_to_one').sort_values('frame_id')
        fr=joined['frame_id'].to_numpy(np.int64);tm=joined['timestamp'].to_numpy(np.float64)
        xyz=joined[['x','y','z']].to_numpy(np.float64);rpy=joined[['roll','pitch','yaw']].to_numpy(np.float64)
        assert np.isfinite(xyz).all() and np.isfinite(rpy).all() and len(set(fr))==len(fr) and np.all(np.diff(tm)>0)
        # Verify the old implicit frame+50 convention against explicit frame lookup.
        assert np.array_equal(fr,np.arange(len(fr))-50)
        lookup={int(f):i for i,f in enumerate(fr)}
        rows=np.flatnonzero((cache['scen_idx']==si)&(cache['frame']>=30))
        part='train' if s in tr else 'tune'
        if part=='tune':rows=rows[cache['frame'][rows]%5==0]
        frames=cache['frame'][rows].astype(np.int64);now=np.array([lookup[int(f)] for f in frames]);fidx=np.array([[lookup[int(f+k)] for k in (5,10,15,20,25,30)] for f in frames]);gidx=np.array([lookup[int(f+50)] for f in frames])
        rot=Rotation.from_euler('xyz',rpy[now]).as_matrix()
        future=np.einsum('nkj,nji->nki',xyz[fidx]-xyz[now,None],rot)[...,:2]
        goal=np.einsum('nj,nji->ni',xyz[gidx]-xyz[now],rot)[...,:2]
        er=abs(future-cache['fut'][rows]);ge=abs(goal-cache['goal'][rows]);errors.append(er.reshape(-1));goals.append(ge.reshape(-1))
        assert np.array_equal(future.astype(np.float32),cache['fut'][rows]) and np.array_equal(goal.astype(np.float32),cache['goal'][rows])
        actual_dt=(tm[fidx]-tm[now,None])/1000.;future_dt.append(actual_dt-np.arange(1,7)[None]*.5);dt.append(np.diff(tm)/1000.)
        counts[part]+=len(rows)
        result['source_files_sha256'][str(ts_path)]=digest(ts_path);result['source_files_sha256'][str(pose_path)]=digest(pose_path)
        result['scenes'][s]={'split':part,'rows':len(rows),'source_frames':len(fr),'max_future_cache_float_error_m':float(er.max()),'max_goal_cache_float_error_m':float(ge.max()),'max_frame_interval_deviation_s':float(abs(np.diff(tm)/1000.-.1).max())}
        if part=='tune':
            poses=full_pose_matrices(xyz,rpy);real=(tm-tm[0])/1000.;nom=(fr-fr[0])*.1
            for row,frame,i in zip(rows,frames,now):
                r=record_lookup[int(row)];assert np.array_equal(np.array(r['gt_abs_xy']),cache['fut'][row])
                target=motion_targets(poses,real,int(i),[lookup[int(frame-k)] for k in (1,2,5,10)])
                state_diff.append(abs(target['state_target']-np.array(r['gt_state'])))
                st,_=fit_status5(poses,nom,int(i));status_diff.append(abs(st-status_lookup[int(row)]))
                vx_vs_future.append([float(target['state_target'][0]),float(cache['fut'][row,0,0]*2),float(np.linalg.norm(cache['fut'][row,0])*2)])
                verified_v0.append(int(row))
    assert counts=={'train':83700,'tune':1998} and set(verified_v0)==set(record_lookup)
    status_diff=np.array(status_diff);state_diff=np.array(state_diff);assert status_diff.max()<2e-5 and state_diff.max()<2e-5
    t=np.concatenate(future_dt);v=np.array(vx_vs_future)
    result.update(rows=counts,cache_recomputed_bitwise_after_float32=True,max_unrounded_future_error_m=float(np.concatenate(errors).max()),max_unrounded_goal_error_m=float(np.concatenate(goals).max()),frame_interval_s=quant(np.concatenate(dt)),future_timestamp_minus_nominal_s={str((i+1)*.5):quant(t[:,i]) for i in range(6)},state_target_replay_max_abs_by_channel=state_diff.max(0).tolist(),nominal_status_replay_max_abs_by_channel=status_diff.max(0).tolist(),current_vx_vs_future_first_interval={'signed_mean_difference_mps':float((v[:,0]-v[:,1]).mean()),'MAE_difference_mps':float(abs(v[:,0]-v[:,1]).mean()),'warning':'Different target times; difference is not label noise or model error.'},limitations=['Reproduction verifies cache indexing and coordinate transforms, not independent GPS accuracy.','Metadata frame timestamps do not independently establish sensor exposure synchronization.','No test poses or hidden future labels used.'])
    dest=REPORT/'target_and_time_audit.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('scenes','source_files_sha256')}))

if __name__=='__main__':main()
