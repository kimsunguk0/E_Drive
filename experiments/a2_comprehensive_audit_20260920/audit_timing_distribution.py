"""Locate observed timestamp variation; do not retime official labels."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path('/NHNHOME/data/sukim/adcl');REPORT=ROOT/'reports/a2_comprehensive_audit_20260920'

def main():
    prior=json.loads((REPORT/'target_and_time_audit.json').read_text());split=json.loads((ROOT/'data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json').read_text());train=set(split['splits']['train']);tune=set(split['splits']['tune'])
    with np.load('/tmp/pm97/data/etri/ego_cache.npz',allow_pickle=False) as z:
        cache={k:z[k] for k in ('scenarios','scen_idx','frame')}
    sources={str(s):int(i) for i,s in enumerate(cache['scenarios']) if str(s) in train|tune}
    def process(item):
        scene,si=item;part='train' if scene in train else 'tune'
        path=Path('/tmp/pm97/data/etri/meta_train')/scene/'meta/timestamps.parquet'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==prior['source_files_sha256'][str(path)]
        ts=pd.read_parquet(path).sort_values('frame_id');frames=ts['frame_id'].to_numpy();times=ts['timestamp'].to_numpy(np.float64);lookup={int(f):i for i,f in enumerate(frames)}
        rows=np.flatnonzero((cache['scen_idx']==si)&(cache['frame']>=30))
        if part=='tune':rows=rows[cache['frame'][rows]%5==0]
        fr=cache['frame'][rows];now=np.array([lookup[int(f)] for f in fr]);future=np.array([[lookup[int(f+k)] for k in (5,10,15,20,25,30)] for f in fr])
        delta=(times[future]-times[now,None])/1000.-np.arange(1,7)[None]*.5
        med=float(np.median(np.diff(times)/1000.))
        return scene,part,rows,delta,med
    items=list(ThreadPoolExecutor(max_workers=8).map(process,sources.items()))
    result={'policy':'Official frame-index targets retained. Nominal input and real-time auxiliary targets are separate definitions.','scenes':{},'splits':{}}
    for scene,part,rows,delta,med in items:
        result['scenes'][scene]={'split':part,'n':len(rows),'median_frame_dt_s':med,'mean_future_dt_error_s':delta.mean(0).tolist(),'max_abs_future_dt_error_s':abs(delta).max(0).tolist()}
    for part in ('train','tune'):
        rr=np.concatenate([rows for _,p,rows,_,_ in items if p==part]);dd=np.concatenate([d for _,p,_,d,_ in items if p==part]);m=np.max(abs(dd),axis=1)
        result['splits'][part]={'n':len(rr),'mean_abs_future_dt_error_s':abs(dd).mean(0).tolist(),'future_3s_signed_quantiles':np.quantile(dd[:,-1],[0,.01,.5,.99,1]).tolist(),'rows_any_horizon_above_1ms':int((m>.001).sum()),'rows_any_horizon_above_10ms':int((m>.01).sum()),'rows_any_horizon_above_50ms':int((m>.05).sum())}
        if part=='tune':
            data=json.loads((ROOT/'work_dirs/a2_fresh_continue_20260920/A2-FRESH-CONT-s1/predictions_step5710.json').read_text())['records'];lookup={r['row']:r for r in data};score=np.array([np.linalg.norm(np.array(lookup[int(r)]['pred_abs_xy'])-np.array(lookup[int(r)]['gt_abs_xy']),axis=-1)@ (np.array([11,11,5,5,2,2])/36) for r in rr])
            result['splits'][part]['rows_above_10ms_error_contribution']=float(score[m>.01].sum()/len(rr))
    result['persistent_slow_scenes']={s:v for s,v in result['scenes'].items() if v['median_frame_dt_s']>.102}
    (REPORT/'timing_distribution.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='scenes'}))

if __name__=='__main__':main()
