"""Train-only continuous map targets; precomputed before augmentation."""
from pathlib import Path
import argparse, hashlib, json, multiprocessing as mp, os, sys, time
from functools import lru_cache
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
for p in (ROOT,ROOT/'scripts',ROOT/'experiments/a2_design_preflight_20260921',ROOT/'experiments/a2_progress_fourarm_20260921'):
    sys.path.insert(0,str(p))
from design import geometry_targets,GEOMETRY_POLICY
from build_scene_supervision_v2 import read_scene_metadata,transform_points
from motiondrive_v2_data import grid_centers

CACHE=ROOT/'data/etri/motiondrive_v2/a2_lane_geometry_20260921_v3'
REPORT=ROOT/'reports/a2_splitread_lanegeom_20260921'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def build_scene(task):
    scene,rows,frames,supervision=task
    started=time.monotonic();directory=CACHE/scene
    directory.mkdir(exist_ok=False)
    metadata=json.loads((Path(supervision)/(scene+'.json')).read_text())
    paths=[Path(p) for p in metadata['sources'] if p.endswith('/annotation/map.parquet')]
    source,=paths
    allframes,_,poses,_,lines,schema=read_scene_metadata(source.parent.parent)
    lookup={int(f):i for i,f in enumerate(allframes)}
    with np.load(Path(supervision)/(scene+'.npz'),allow_pickle=False) as z:
        oldrows=z['row'];oldframes=z['frame'];oldvalid=z['lane_valid']
    oldlookup={int(r):i for i,r in enumerate(oldrows)}
    values=np.lib.format.open_memmap(directory/'values.npy',mode='w+',dtype=np.float32,shape=(len(rows),4,64,48))
    valid=np.lib.format.open_memmap(directory/'valid.npy',mode='w+',dtype=np.bool_,shape=(len(rows),2,64,48))
    xy=grid_centers();counts=np.zeros(2,np.int64)
    for i,(row,frame) in enumerate(zip(rows,frames)):
        loc=oldlookup[row];assert int(oldframes[loc])==frame
        w2e=np.linalg.inv(poses[lookup[frame]])
        local=[transform_points(line,w2e) for line in lines]
        result=geometry_targets(local,xy,oldvalid[loc,0])
        values[i,:2]=result['offset'].transpose(2,0,1);values[i,2:]=result['axis'].transpose(2,0,1)
        valid[i,0]=result['offset_valid'];valid[i,1]=result['axis_valid']
        counts+=valid[i].sum((1,2))
    values.flush();valid.flush();del values,valid
    np.save(directory/'rows.npy',np.asarray(rows,dtype=np.int64),allow_pickle=False)
    record={'scene':scene,'rows':len(rows),'frames':frames,'split':'train','schema':schema,
        'source_map_sha256':sha(source),'supervision_sha256':sha(Path(supervision)/(scene+'.npz')),
        'valid_cells':counts.tolist(),'seconds':time.monotonic()-started,
        'files':{n:sha(directory/n) for n in ('rows.npy','values.npy','valid.npy')}}
    atomic(directory/'manifest.json',record);return record

class GeometryDataset(torch.utils.data.Dataset):
    def __init__(self,base):
        self.base=base
        self.enabled=base.split=='train'
        if self.enabled:
            m=json.loads((CACHE/'manifest.json').read_text())
            if m['status']!='complete' or m['policy']!=GEOMETRY_POLICY or m['split_sha256']!=base.split_sha:
                raise ValueError('Geometry cache contract mismatch')
            if m['rows_sha256']!=hashlib.sha256(np.asarray(base.rows,dtype='<i8').tobytes()).hexdigest():
                raise ValueError('Geometry rows differ from training')
    def __getattr__(self,name):
        if name=='base':raise AttributeError(name)
        return getattr(self.base,name)
    def __len__(self):return len(self.base)
    def set_epoch(self,epoch):return self.base.set_epoch(epoch)
    @lru_cache(maxsize=16)
    def _scene(self,scene):
        p=CACHE/scene
        return tuple(np.load(p/n,mmap_mode='r',allow_pickle=False) for n in ('rows.npy','values.npy','valid.npy'))
    def __getitem__(self,index):
        item=self.base[index]
        if not self.enabled:return item
        rows,values,valid=self._scene(item['scenario']);row=int(item['row'])
        pos=int(np.searchsorted(rows,row))
        if pos>=len(rows) or rows[pos]!=row:raise ValueError('Missing geometry train row')
        item['geo_offset']=torch.from_numpy(values[pos,:2].copy())
        item['geo_axis']=torch.from_numpy(values[pos,2:].copy())
        item['geo_offset_valid']=torch.from_numpy(valid[pos,0].copy())
        item['geo_axis_valid']=torch.from_numpy(valid[pos,1].copy())
        return item

def wrap_geometry_flip(original):
    def flip(item,wf,wh):
        out=original(item,wf,wh)
        for key in ('offset','axis'):
            name='geo_'+key
            if name not in item:continue
            out[name]=torch.flip(item[name],[-1]);out[name][1]*=-1
            out[name+'_valid']=torch.flip(item[name+'_valid'],[-1])
        return out
    return flip

def geometry_batch(batch):
    return {k:batch['geo_'+k] for k in ('offset','axis','offset_valid','axis_valid')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=20);args=p.parse_args()
    from train_fourarm import nominal
    tr,_=nominal.raw_datasets(False,1)
    if CACHE.exists():raise FileExistsError('Refuse cache overwrite: '+str(CACHE))
    CACHE.mkdir(parents=True)
    tasks=[]
    for scene in sorted(set(tr.scene_names[tr.rows])):
        rows=tr.rows[tr.scene_names[tr.rows]==scene].tolist()
        tasks.append((str(scene),rows,[int(tr.arr['frame'][r]) for r in rows],str(tr.supervision_root)))
    start=time.monotonic();records=[]
    status={'status':'building','rows':len(tr),'scenes':len(tasks),'policy':GEOMETRY_POLICY,
        'source_sha256':sha(HERE/'geometry_data.py'),'target_source_sha256':sha(ROOT/'experiments/a2_design_preflight_20260921/design.py'),
        'split_sha256':tr.split_sha,'rows_sha256':hashlib.sha256(np.asarray(tr.rows,dtype='<i8').tobytes()).hexdigest(),
        'train_only':True,'future_ego_trajectory_used':False,'workers':args.workers}
    atomic(CACHE/'manifest.json',status)
    with mp.get_context('fork').Pool(args.workers) as pool:
        for record in pool.imap_unordered(build_scene,tasks):
            records.append(record);status.update(completed_scenes=len(records),elapsed_seconds=time.monotonic()-start)
            atomic(REPORT/'runtime/geometry_cache_progress.json',status)
            print(json.dumps({k:status[k] for k in ('completed_scenes','scenes','elapsed_seconds')}),flush=True)
    status.update(status='complete',artifacts=sorted(records,key=lambda x:x['scene']))
    atomic(CACHE/'manifest.json',status)
    atomic(REPORT/'geometry_cache.json',{'path':str(CACHE),'manifest_sha256':sha(CACHE/'manifest.json'),
        'rows':len(tr),'scenes':len(tasks),'elapsed_seconds':time.monotonic()-start,'policy':GEOMETRY_POLICY})

if __name__=='__main__':main()
