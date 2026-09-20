"""Build new train/tune input cache, never overwrite old status or GT caches."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime,json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from h4_status import FRAME_OFFSETS,status_from_five_poses
from h4_data import ROOT,CACHE,PRODUCER,sha

def main():
    if CACHE.exists():raise FileExistsError(CACHE)
    split_path=ROOT/'data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json';split=json.loads(split_path.read_text())
    cache_path=Path('/tmp/pm97/data/etri/ego_cache.npz')
    with np.load(cache_path,allow_pickle=False) as z:cache={k:z[k] for k in ('scenarios','scen_idx','frame')}
    allowed={s:part for part in ('train','tune') for s in split['splits'][part]}
    def process(item):
        si,s=item;s=str(s);folder=Path('/tmp/pm97/data/etri/meta_train')/s
        paths=[folder/'meta/timestamps.parquet',folder/'annotation/ego_pose.parquet']
        df=pd.read_parquet(paths[0])[['frame_id','timestamp']].merge(pd.read_parquet(paths[1]),on='timestamp',how='left',validate='one_to_one').sort_values('frame_id')
        fr=df['frame_id'].to_numpy(np.int64);lookup={int(f):i for i,f in enumerate(fr)}
        xyz=df[['x','y','z']].to_numpy(np.float64);rpy=df[['roll','pitch','yaw']].to_numpy(np.float64)
        poses=np.broadcast_to(np.eye(4),(len(fr),4,4)).copy();poses[:,:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();poses[:,:3,3]=xyz
        rows=np.flatnonzero((cache['scen_idx']==si)&(cache['frame']>=30))
        # Tune stores every eligible frame, evaluator still chooses stride5.
        indices=np.array([[lookup[int(f+off)] for off in FRAME_OFFSETS] for f in cache['frame'][rows]])
        values,rms=status_from_five_poses(poses[indices])
        return allowed[s],s,rows,values,float(rms.max()),{str(p):sha(p) for p in paths}
    with ThreadPoolExecutor(max_workers=8) as pool:items=list(pool.map(process,[(i,s) for i,s in enumerate(cache['scenarios']) if str(s) in allowed]))
    CACHE.mkdir(parents=True)
    manifest={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'producer_sha256':sha(PRODUCER),'producer':str(PRODUCER),'split_manifest_sha256':sha(split_path),'ego_cache_sha256':sha(cache_path),'relative_fit_frames':list(FRAME_OFFSETS),'time_policy':'frame differences * 0.1 seconds','supervision_unchanged':True,'DEV_only':True,'splits':{},'sources_sha256':{k:v for item in items for k,v in item[5].items()},'fit_rms_max_m':max(i[4] for i in items)}
    for part,n in [('train',83700),('tune',9990)]:
        rows=np.concatenate([i[2] for i in items if i[0]==part]);values=np.concatenate([i[3] for i in items if i[0]==part]);order=np.argsort(rows);rows=rows[order];values=values[order]
        assert len(rows)==n and len(set(rows))==n and np.isfinite(values).all()
        p=CACHE/f'{part}.npz';np.savez_compressed(p,row=rows,status5=values)
        with np.load(ROOT/f'data/etri/motiondrive_v2/a2_nominal_status_20260918/{part}.npz',allow_pickle=False) as z:
            assert np.array_equal(rows,z['row']);diff=values.astype(np.float64)-z['status5']
        manifest['splits'][part]={'path':str(p),'sha256':sha(p),'rows':n,'old_input_difference_signed_mean':diff.mean(0).tolist(),'old_input_difference_MAE':abs(diff).mean(0).tolist(),'old_input_difference_abs_p99':np.quantile(abs(diff),.99,axis=0).tolist()}
    (CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=ROOT/'reports/a2_progress_h4_20260920';report.mkdir(parents=True,exist_ok=True)
    (report/'input_policy.json').write_text(json.dumps({k:v for k,v in manifest.items() if k!='sources_sha256'},indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='sources_sha256'}))

if __name__=='__main__':main()
