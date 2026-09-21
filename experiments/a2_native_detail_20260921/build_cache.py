"""Bake only DEV train/current+H4 and V0 RGB, from raw JPEGs, at native 1152."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, os, sys, tarfile, time
import numpy as np

ROOT=Path('/NHNHOME/data/sukim/adcl')
HERE=Path(__file__).resolve().parent
REPORT=ROOT/'reports/a2_native_detail_20260921'
CACHE=ROOT/'data/etri/motiondrive_v2/native_detail_1152_20260921'
for p in (ROOT,ROOT/'scripts',ROOT/'experiments/md_a2_nominal_mh4_20260918'):
    sys.path.insert(0,str(p))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)

def plan():
    import train_nominal as nominal
    from motiondrive_v2_data import CAMERA_ORDER
    train,tune=nominal.raw_datasets(False,1)
    needed={};sources={}
    for role,ds in [('train',train),('tune',tune)]:
        sources[role]={'n':len(ds),'rows_sha256':hashlib.sha256(np.asarray(ds.rows,dtype='<i8').tobytes()).hexdigest()}
        for row in ds.rows:
            scene=str(ds.scene_names[row]);frame=int(ds.arr['frame'][row])
            cams=needed.setdefault(scene,{cam:set() for cam in CAMERA_ORDER})
            for cam in CAMERA_ORDER:cams[cam].add(frame)
            cams['camera_front'].update(frame-int(o) for o in ds.history_offsets)
    result={'schema':1,'output_wh':[1152,648],'jpeg_quality':95,'resize':'cv2.INTER_AREA',
            'undistort_crop':'same existing etri_build_cache_b200 geometry',
            'split_manifest_sha256':train.split_sha,'splits':sources,
            'scenes':{s:{c:sorted(f) for c,f in cams.items()} for s,cams in sorted(needed.items())}}
    assert sources['train']['n']==83700 and sources['tune']['n']==1998
    assert len(result['scenes'])==347
    return result

def worker(task):
    import cv2
    import pyarrow.parquet as pq
    from etri_build_cache_b200 import new_intrinsic,build_maps,crop_of
    cv2.setNumThreads(1)
    scene,frames=task;beg=time.monotonic();out=CACHE/scene;out.mkdir(parents=True,exist_ok=True)
    need={c:set(v) for c,v in frames.items()};expected=sum(map(len,need.values()))
    calpath=ROOT/'data/etri/meta_train'/scene/'calibration/calibration.parquet'
    raw=ROOT/'train'/(scene+'.tar')
    if not raw.exists() or not calpath.exists():raise FileNotFoundError((raw,calpath))
    complete=out/'complete.json'
    identity={'frames':frames,'calibration_sha256':sha(calpath),'raw_tar_bytes':raw.stat().st_size,
              'raw_tar_mtime_ns':raw.stat().st_mtime_ns,'producer_sha256':sha(__file__)}
    if complete.exists():
        d=json.loads(complete.read_text())
        if d['identity']!=identity:raise ValueError('Existing cache identity differs: '+scene)
        return d
    cal=pq.read_table(calpath).to_pydict();maps={};meta={}
    for i,c in enumerate(cal['camera_name']):
        if c not in need:continue
        K=np.asarray(cal['K'][i],np.float64).reshape(3,3);dist=np.asarray(cal['distortion'][i],np.float64)
        size=(int(cal['image_width'][i]),int(cal['image_height'][i]));fish=bool(cal['is_fisheye'][i])
        nk=new_intrinsic(K,dist,size,fish);maps[c]=build_maps(K,dist,nk,size,fish)
        x,y=crop_of(c,*size);shift=np.eye(3);shift[0,2]=-x;shift[1,2]=-y
        meta[c]={'raw_wh':list(size),'crop_xy':[x,y],'K1152':(np.diag([.6,.6,1])@shift@nk).tolist()}
        (out/c).mkdir(exist_ok=True)
    if set(maps)!=set(need):raise ValueError('Camera coverage mismatch')
    seen=set();hashes={};nbytes=0
    with tarfile.open(raw,'r|') as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith('.jpg'):continue
            bits=m.name.split('/');c=bits[-2];filename=bits[-1]
            if c not in need:continue
            try:frame=int(Path(filename).stem)
            except ValueError:continue
            if frame not in need[c]:continue
            key=(c,frame)
            if key in seen:raise ValueError('Duplicate raw RGB')
            buf=tf.extractfile(m).read();img=cv2.imdecode(np.frombuffer(buf,np.uint8),cv2.IMREAD_COLOR)
            if img is None or tuple(img.shape[1::-1])!=tuple(meta[c]['raw_wh']):raise ValueError('Bad raw RGB')
            und=cv2.remap(img,*maps[c],cv2.INTER_LINEAR);x,y=meta[c]['crop_xy']
            crop=und[y:y+1080,x:x+1920]
            if crop.shape[:2]!=(1080,1920):raise ValueError('Bad crop')
            scaled=cv2.resize(crop,(1152,648),interpolation=cv2.INTER_AREA)
            ok,enc=cv2.imencode('.jpg',scaled,[cv2.IMWRITE_JPEG_QUALITY,95])
            if not ok:raise ValueError('JPEG encoding failed')
            target=out/c/f'{frame:08d}.jpg';temp=target.with_suffix('.jpg.part');data=enc.tobytes()
            temp.write_bytes(data);temp.replace(target)
            hashes[f'{c}/{frame:08d}.jpg']={'raw':hashlib.sha256(buf).hexdigest(),'cache':hashlib.sha256(data).hexdigest()}
            nbytes+=len(data);seen.add(key)
    if len(seen)!=expected:raise ValueError((scene,len(seen),expected))
    atomic(out/'image_hashes.json',hashes)
    d={'scene':scene,'identity':identity,'camera_geometry':meta,'images':expected,'bytes':nbytes,
       'seconds':time.monotonic()-beg,'image_hash_manifest_sha256':sha(out/'image_hashes.json')}
    atomic(complete,d);return d

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=32);args=ap.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    p=plan();atomic(REPORT/'cache_plan.json',p);atomic(CACHE/'plan.json',p)
    start=time.monotonic();done=[];tasks=list(p['scenes'].items())
    # Prepare representative train/V0 fixtures early, without changing training order.
    priority=['20260210-142336','20260210-100044']
    tasks.sort(key=lambda x:(x[0] not in priority,x[0]))
    atomic(REPORT/'runtime/cache_status.json',{'status':'running','pid':os.getpid(),'scenes':len(tasks),'completed':0})
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for item in pool.map(worker,tasks,chunksize=1):
            done.append(item);elapsed=time.monotonic()-start
            status={'status':'running','pid':os.getpid(),'completed':len(done),'scenes':len(tasks),'seconds':elapsed,
                    'images':sum(d['images'] for d in done),'bytes':sum(d['bytes'] for d in done),
                    'ETA_seconds':elapsed/len(done)*(len(tasks)-len(done))}
            atomic(REPORT/'runtime/cache_status.json',status)
            print(json.dumps(status),flush=True)
    manifest={'status':'completed','plan_sha256':sha(CACHE/'plan.json'),'producer_sha256':sha(__file__),
              'seconds':time.monotonic()-start,'images':sum(d['images'] for d in done),
              'bytes':sum(d['bytes'] for d in done),'scenes':{d['scene']:d for d in done}}
    atomic(CACHE/'manifest.json',manifest)
    atomic(REPORT/'cache_summary.json',{k:v for k,v in manifest.items() if k!='scenes'}|{'cache':str(CACHE),'manifest_sha256':sha(CACHE/'manifest.json'),'n_scenes':len(done)})
    atomic(REPORT/'runtime/cache_status.json',{'status':'completed','seconds':manifest['seconds'],'scenes':len(done)})

if __name__=='__main__':main()
