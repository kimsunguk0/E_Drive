"""Current-ego agent future displacement targets, strictly train310 and loss-only."""
from pathlib import Path
from functools import lru_cache
import argparse,hashlib,json,multiprocessing as mp,sys,time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'scripts',ROOT/'experiments/a2_native_detail_20260921'):sys.path.insert(0,str(p))
from build_scene_supervision_v2 import read_scene_metadata,transform_points,camera_visible
from motiondrive_v2_data import GRID_EXTENT

CACHE=ROOT/'data/etri/motiondrive_v2/a2_agent_future_20260922'
REPORT=ROOT/'reports/a2_capacity_dynamics_20260922'
MAX_AGENTS=128
OFFSETS=(5,10,15,20,25,30)
POLICY=dict(schema=1,max_agents=MAX_AGENTS,offset_frames=list(OFFSETS),target='world future minus world current, rotated into current ego XY',
    validity='current and each future num_points>0, same obj_id and class, center inside shared grid and visible camera',
    classes=['Car','Pedestrian','Cyclist'],dtype='float32 after float64 transforms',target_scale_m=10.,
    loss='SmoothL1 beta0.1 on normalized displacement; valid agent-time-XY full-effective-batch mean',
    targets_in_model_forward=False,no_truncation=True,train_only=True)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def make_targets(records,all_records,frame,w2e,visible):
    centers=np.zeros((MAX_AGENTS,2),np.float32);delta=np.zeros((MAX_AGENTS,6,2),np.float32)
    valid=np.zeros((MAX_AGENTS,6),bool);ids=np.full(MAX_AGENTS,-1,np.int64)
    current=[]
    for obj in records:
        if obj['class'] not in POLICY['classes'] or not np.isfinite(obj.get('num_points',0)) or obj.get('num_points',0)<=0:continue
        xyz=np.asarray([obj['x[m]'],obj['y[m]'],obj['z[m]']],np.float64)
        if not np.isfinite(xyz).all():continue
        loc=transform_points(xyz[None],w2e)[0]
        x,y=loc[:2]
        if not(-10<=x<70 and -32<=y<32):continue
        ix=int((x+10)*64/80);iy=int((y+32)*48/64)
        if not visible[ix,iy]:continue
        current.append((obj,xyz,loc))
    current.sort(key=lambda v:int(v[0]['obj_id']))
    if len(current)>MAX_AGENTS:raise ValueError('Refuse object truncation; enlarge fixed storage before training')
    if len({int(o['obj_id']) for o,_,_ in current})!=len(current):raise ValueError('Duplicate current object ID')
    for ai,(obj,xyz,loc) in enumerate(current):
        centers[ai]=loc[:2];ids[ai]=int(obj['obj_id'])
        for k,off in enumerate(OFFSETS):
            nxt=all_records.get(frame+off,{}).get(ids[ai])
            if nxt is None or nxt['class']!=obj['class'] or not np.isfinite(nxt.get('num_points',0)) or nxt.get('num_points',0)<=0:continue
            pos=np.asarray([nxt['x[m]'],nxt['y[m]'],nxt['z[m]']],np.float64)
            if not np.isfinite(pos).all():continue
            delta[ai,k]=((pos-xyz)@w2e[:3,:3].T)[:2];valid[ai,k]=True
    return centers,delta,valid,ids,len(current)

def build_scene(task):
    scene,rows,frames,supervision,l2i=task
    meta=json.loads((Path(supervision)/(scene+'.json')).read_text())
    source=next(Path(p) for p in meta['sources'] if p.endswith('/annotation/object.parquet'))
    allframes,_,poses,groups,_,_=read_scene_metadata(source.parent.parent)
    lookup={int(f):i for i,f in enumerate(allframes)};allrecords={}
    for f,recs in groups.items():
        rr=[r for r in recs if r['class'] in POLICY['classes']]
        ids=[int(r['obj_id']) for r in rr]
        if len(set(ids))!=len(ids):raise ValueError(f'Duplicate object IDs {scene}/{f}')
        allrecords[f]=dict(zip(ids,rr))
    visible=camera_visible(l2i);values=[make_targets(groups.get(f,[]),allrecords,f,np.linalg.inv(poses[lookup[f]]),visible) for f in frames]
    centers=np.stack([v[0] for v in values]);delta=np.stack([v[1] for v in values]);valid=np.stack([v[2] for v in values])
    ids=np.stack([v[3] for v in values]);counts=np.asarray([v[4] for v in values],np.int64)
    target=CACHE/(scene+'.npz');assert not target.exists()
    np.savez_compressed(target,rows=np.asarray(rows,np.int64),frames=frames,centers=centers,delta=delta,valid=valid,obj_id=ids)
    speed=np.linalg.norm(delta,axis=-1)/np.asarray(OFFSETS)[None,None,:]*10
    rec=dict(scene=scene,rows=len(rows),valid_points=valid.sum(0).sum(0).tolist(),max_agents=int(counts.max()),
        zero_valid_rows=int((~valid.any((1,2))).sum()),moving_valid_points=int(((speed>0.5)&valid).sum()),
        files={str(p):sha(p) for p in (source,source.parent/'ego_pose.parquet',source.parent.parent/'meta/timestamps.parquet')},
        cache_sha256=sha(target))
    return rec

class AgentDataset(torch.utils.data.Dataset):
    def __init__(self,base):
        self.base=base;self.enabled=base.split=='train'
        if self.enabled:
            manifest=json.loads((CACHE/'manifest.json').read_text())
            assert manifest['status']=='complete' and manifest['policy']==POLICY
            assert manifest['split_sha256']==base.split_sha
            assert manifest['rows_sha256']==hashlib.sha256(np.asarray(base.rows,dtype='<i8').tobytes()).hexdigest()
    def __getattr__(self,n):
        if n=='base':raise AttributeError(n)
        return getattr(self.base,n)
    def __len__(self):return len(self.base)
    def set_epoch(self,e):self.base.set_epoch(e)
    @lru_cache(maxsize=12)
    def scene(self,name):
        with np.load(CACHE/(name+'.npz'),allow_pickle=False) as z:return {k:z[k] for k in ('rows','frames','centers','delta','valid')}
    def __getitem__(self,i):
        item=self.base[i]
        if not self.enabled:return item
        z=self.scene(item['scenario']);r=int(item['row']);loc=int(np.searchsorted(z['rows'],r))
        assert loc<len(z['rows']) and int(z['rows'][loc])==r and int(z['frames'][loc])==int(item['frame'])
        for k in ('centers','delta','valid'):item['agent_'+k]=torch.from_numpy(z[k][loc].copy())
        return item

def wrap_flip(old):
    def flip(item,wf,wh):
        out=old(item,wf,wh)
        for k in ('agent_centers','agent_delta'):
            if k in item:out[k]=item[k].clone();out[k][...,1]*=-1
        return out
    return flip

def normalizers(batch):return {'agent_count':batch['agent_valid'].bool().sum().double()*2}

def agent_loss(field,batch,norm):
    with torch.autocast(device_type=field.device.type,enabled=False):
        centers=batch['agent_centers'].float()
        grid=torch.stack([2*(centers[...,1]+32)/64-1,2*(centers[...,0]+10)/80-1],-1)
        pred=F.grid_sample(field.float(),grid[:,:,None],align_corners=False,padding_mode='zeros').squeeze(-1)
        pred=pred.transpose(1,2).reshape(len(field),-1,6,2)
        valid=batch['agent_valid'].bool()
        target=torch.where(valid[...,None],batch['agent_delta'].float()/10.,torch.zeros_like(batch['agent_delta'],dtype=torch.float32))
        assert torch.isfinite(target).all()
        terms=F.smooth_l1_loss(pred,target,reduction='none',beta=.1)*valid[...,None]
        return terms.sum()/norm['agent_count'].to(field.device).clamp_min(1).float()

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);args=p.parse_args()
    from train_native import nominal
    tr,va=nominal.raw_datasets(False,1)
    assert len(tr)==83700 and len(va)==1998
    assert set(tr.scene_names[tr.rows]).isdisjoint(set(va.scene_names[va.rows]))
    if CACHE.exists():raise FileExistsError(CACHE)
    CACHE.mkdir(parents=True)
    tasks=[]
    for scene in sorted(set(tr.scene_names[tr.rows])):
        rows=tr.rows[tr.scene_names[tr.rows]==scene].tolist()
        tasks.append((str(scene),rows,[int(tr.arr['frame'][r]) for r in rows],str(tr.supervision_root),tr.lidar2img))
    status=dict(status='building',policy=POLICY,rows=len(tr),scenes=len(tasks),split_sha256=tr.split_sha,
        rows_sha256=hashlib.sha256(np.asarray(tr.rows,dtype='<i8').tobytes()).hexdigest(),source_sha256=sha(Path(__file__)))
    atomic(CACHE/'manifest.json',status);records=[];start=time.monotonic()
    with mp.get_context('fork').Pool(args.workers) as pool:
        for r in pool.imap_unordered(build_scene,tasks):
            records.append(r)
            if len(records)%20==0:
                progress={'scenes_done':len(records),'seconds':time.monotonic()-start}
                atomic(REPORT/'runtime/agent_cache_progress.json',progress);print(json.dumps(progress),flush=True)
    status.update(status='complete',artifacts=sorted(records,key=lambda r:r['scene']),seconds=time.monotonic()-start)
    atomic(CACHE/'manifest.json',status)
    atomic(REPORT/'agent_cache.json',dict(path=str(CACHE),manifest_sha256=sha(CACHE/'manifest.json'),rows=len(tr),scenes=len(tasks),
        max_agents=max(r['max_agents'] for r in records),zero_valid_rows=sum(r['zero_valid_rows'] for r in records),
        valid_points=np.asarray([r['valid_points'] for r in records]).sum(0).tolist(),policy=POLICY,seconds=status['seconds']))
    print('AGENT CACHE COMPLETE',flush=True)

if __name__=='__main__':main()
