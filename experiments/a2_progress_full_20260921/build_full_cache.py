"""Same fixed five-pose producer, all 376 official train scenes, separate cache."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime,json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from full_data import ROOT,CACHE,sha,h4_status

def main():
    assert not CACHE.exists(),'Do not overwrite a preserved input cache'
    split_path=ROOT/'data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json'
    split=json.loads(split_path.read_text());allowed={s:part for part in ('train','tune','val') for s in split['splits'][part]}
    assert len(allowed)==376
    cache_path=Path('/tmp/pm97/data/etri/ego_cache.npz')
    with np.load(cache_path,allow_pickle=False) as z:cache={k:z[k] for k in ('scenarios','scen_idx','frame')}
    def process(item):
        si,s=item;s=str(s);folder=Path('/tmp/pm97/data/etri/meta_train')/s
        sources=[folder/'meta/timestamps.parquet',folder/'annotation/ego_pose.parquet']
        df=pd.read_parquet(sources[0])[['frame_id','timestamp']].merge(pd.read_parquet(sources[1]),on='timestamp',how='left',validate='one_to_one').sort_values('frame_id')
        frames=df['frame_id'].to_numpy(np.int64);lookup={int(f):i for i,f in enumerate(frames)}
        poses=np.broadcast_to(np.eye(4),(len(frames),4,4)).copy()
        poses[:,:3,:3]=Rotation.from_euler('xyz',df[['roll','pitch','yaw']].to_numpy(np.float64)).as_matrix()
        poses[:,:3,3]=df[['x','y','z']].to_numpy(np.float64)
        rows=np.flatnonzero((cache['scen_idx']==si)&(cache['frame']>=30))
        ids=np.array([[lookup[int(f+off)] for off in h4_status.FRAME_OFFSETS] for f in cache['frame'][rows]])
        status,rms=h4_status.status_from_five_poses(poses[ids])
        return allowed[s],rows,status,float(rms.max()),{str(p):sha(p) for p in sources}
    with ThreadPoolExecutor(max_workers=8) as pool:
        parts=list(pool.map(process,[(i,s) for i,s in enumerate(cache['scenarios']) if str(s) in allowed]))
    assert len(parts)==376
    CACHE.mkdir(parents=True)
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'producer':h4_status.__file__,'producer_sha256':sha(h4_status.__file__),
        'split_manifest_sha256':sha(split_path),'ego_cache_sha256':sha(cache_path),
        'supervision_unchanged':True,'FULL_only':True,'DEV_wrapper_cannot_read_this_cache':True,
        'relative_fit_frames':list(h4_status.FRAME_OFFSETS),'time_policy':'frame differences *0.1 seconds',
        'scenes':376,'rows':101520,'splits':{},'sources_sha256':{k:v for i in parts for k,v in i[4].items()},
        'fit_rms_max_m':max(i[3] for i in parts),'DEV_overlap_exact':{}}
    for split,n in [('train',83700),('tune',9990),('val',7830)]:
        rows=np.concatenate([i[1] for i in parts if i[0]==split]);status=np.concatenate([i[2] for i in parts if i[0]==split])
        order=np.argsort(rows);rows=rows[order];status=status[order]
        assert len(rows)==n and len(np.unique(rows))==n and np.isfinite(status).all()
        if split!='val':
            old=ROOT/f'data/etri/motiondrive_v2/a2_h4_status_20260920/{split}.npz'
            with np.load(old,allow_pickle=False) as z:
                assert np.array_equal(rows,z['row']) and np.array_equal(status,z['status5'])
            manifest['DEV_overlap_exact'][split]=True
        p=CACHE/f'{split}.npz';np.savez_compressed(p,row=rows,status5=status)
        manifest['splits'][split]={'path':str(p),'rows':n,'sha256':sha(p)}
    (CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=ROOT/'reports/a2_progress_full_20260921';report.mkdir(parents=True,exist_ok=True)
    brief={k:v for k,v in manifest.items() if k!='sources_sha256'}
    (report/'input_policy.json').write_text(json.dumps(brief,indent=2)+'\n');print(json.dumps(brief))
if __name__=='__main__':main()
