"""CPU raw/cache parity and strict trained-entry smoke, using sampled train/tune.

This test reads training metadata/cache references; deployment.py itself does not.
No actual hidden test clips are read and no checkpoint is trained or modified.
"""
from __future__ import annotations
import argparse,json,tempfile,time,tarfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .data import PlanDataset
from .deployment import RawInputAdapter,FrozenBankDriver,CAMERAS,MEAN,STD,file_sha256
from models.motiondrive_v2_inputs import build_camera_geometry

BASE=Path('/NHNHOME/data/sukim/adcl')
WT=BASE/'experiment_worktrees/sparsedrivev2_20260910'
META=Path('/tmp/pm97/data/etri/meta_train')
RAW=BASE/'train'


def raw_records(scene,frame):
    root=META/scene
    cal=pq.read_table(root/'calibration/calibration.parquet').to_pylist()
    ts=pd.read_parquet(root/'meta/timestamps.parquet')
    ep=pd.read_parquet(root/'annotation/ego_pose.parquet')
    joined=ts[['timestamp','frame_id']].merge(ep,on='timestamp',how='left',validate='one_to_one').set_index('frame_id')
    poses=[]
    for offset in list(range(-30,1))+[50]:
        row=joined.loc[frame+offset]
        record={k:float(row[k]) for k in ('x','y','z','roll','pitch','yaw')}
        record['frame']=offset
        if offset==50:record.update(roll=float('nan'),pitch=float('nan'),yaw=float('nan'))
        poses.append(record)
    return cal,poses


def without_jpeg_roundtrip(raw_bytes,g):
    raw=cv2.imdecode(np.frombuffer(raw_bytes,np.uint8),cv2.IMREAD_COLOR)
    image=cv2.remap(raw,g.map1,g.map2,cv2.INTER_LINEAR)
    x,y=g.crop_xy
    image=image[y:y+1080,x:x+1920]
    image=cv2.resize(image,(768,432),interpolation=cv2.INTER_AREA)
    image=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).resize((512,256),Image.Resampling.BILINEAR)
    value=(np.asarray(image,np.float32)/255.-MEAN)/STD
    return torch.from_numpy(value.transpose(2,0,1).copy())


def pixels(a,b):
    diff=(a-b).abs().numpy()
    intensity=diff*STD[:,None,None]*255.
    mse=float(np.mean(intensity**2))
    return {'normalized_max_abs':float(diff.max()),'normalized_mean_abs':float(diff.mean()),
            'rgb_255_mean_abs':float(intensity.mean()),'rgb_255_max_abs':float(intensity.max()),
            'psnr_db':None if mse==0 else float(10*np.log10(255**2/mse)),
            'tensor_exact':bool(torch.equal(a,b))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='reports/sparsedrivev2_20260910/public_init/deployment_cpu.json')
    p.add_argument('--sample-per-split',type=int,default=4)
    p.add_argument('--checkpoint',default='work_dirs/sparsedrivev2_20260910/pilot500_causal_selection_s0_v1/last.pth')
    p.add_argument('--checkpoint-sha',default='0e92346d2f3b063f11326a28305b5513fe1f348d5d3250484cd4969330cf2b2e')
    p.add_argument('--bank',default='cache/sparsedrivev2_20260910/bank/p1024_v256_native100m_v8.npz')
    a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(4)
    adapter=RawInputAdapter('causal_selection','selection')
    report={'device':'cpu','opencv':cv2.__version__,'pillow':Image.__version__,
            'samples':[],'source_sha256':{'deployment.py':file_sha256(Path(__file__).with_name('deployment.py')),
                'test_deployment_cpu.py':file_sha256(__file__)}}
    first=None
    for split in ('train','tune'):
        data=PlanDataset(str(BASE),str(WT/'reports/sparsedrivev2_20260910/split_audit/primary_manifest.json'),split,
                         status_mode='causal_selection',goal_mode='selection',limit=a.sample_per_split)
        for index in range(len(data)):
            expected=data[index];scene=expected['scenario'];frame=expected['frame']
            cal,poses=raw_records(scene,frame)
            with tarfile.open(RAW/f'{scene}.tar','r:') as archive:
                image_bytes={camera:archive.extractfile(f'{scene}/{camera}/{frame:08d}.jpg').read() for camera in CAMERAS}
            calls=[]
            def loader(camera,offset):
                assert camera in CAMERAS and offset==0
                calls.append((camera,offset))
                return image_bytes[camera]
            started=time.perf_counter();prepared=adapter.prepare_records(cal,poses,loader)
            seconds=time.perf_counter()-started
            assert calls==[(c,0) for c in CAMERAS]
            status_error=float((prepared.inputs['status'][0]-expected['status']).abs().max())
            goal_error=float((prepared.inputs['goal_xy'][0]-expected['goal_xy']).abs().max())
            projection_error=float((prepared.inputs['lidar2img'][0]-expected['lidar2img']).abs().max())
            assert status_error==0 and goal_error<2e-5 and projection_error<1e-4
            geometry={g.name:g for g in build_camera_geometry(cal)}
            image_checks={}
            for i,camera in enumerate(CAMERAS):
                direct=without_jpeg_roundtrip(image_bytes[camera],geometry[camera])
                cache=BASE/'cache/etri_768'/scene/camera/f'{frame:08d}.jpg'
                image_checks[camera]={
                    'cache_equivalent_vs_archived':pixels(prepared.inputs['images'][0,i],expected['images'][i]),
                    'omit_q95_vs_archived':pixels(direct,expected['images'][i]),
                    'reencoded_jpeg_sha_matches_archive':prepared.metadata['camera_geometry'][camera]['reconstructed_cache_jpeg_sha256']==file_sha256(cache)}
            report['samples'].append({'split':split,'row':int(expected['row']),'scene':scene,'frame':frame,
                'status_max_abs':status_error,'goal_max_abs':goal_error,'projection_max_abs':projection_error,
                'raw_image_reads':len(calls),'prepare_seconds':seconds,'images':image_checks})
            if first is None:first=(cal,poses,image_bytes,prepared,expected)
    cal,poses,image_bytes,prepared,expected=first
    # Interleave a changed goal/pose table: no stale image features or status.
    repeat=adapter.prepare_records(cal,poses,lambda c,f:image_bytes[c])
    assert all(torch.equal(repeat.inputs[k],prepared.inputs[k]) for k in prepared.inputs)
    no_goal=RawInputAdapter('causal_selection','none')
    altered=[dict(r) for r in poses]
    altered[-1].update(x=float('nan'),y=float('nan'),z=float('nan'))
    nong=no_goal.prepare_records(cal,altered,lambda c,f:image_bytes[c])
    assert 'goal_xy' not in nong.inputs and torch.equal(nong.inputs['status'],prepared.inputs['status'])
    # Official-shaped disk interface; copy only three requested raw JPEGs.
    with tempfile.TemporaryDirectory(prefix='raw_clip_cpu_') as directory:
        directory=Path(directory)
        pq.write_table(pa.Table.from_pylist(cal),directory/'calibration.parquet')
        pq.write_table(pa.Table.from_pylist(poses),directory/'ego_pose.parquet')
        for camera,value in image_bytes.items():
            (directory/camera).mkdir();(directory/camera/'frame_0.jpg').write_bytes(value)
        on_disk=adapter.prepare_clip(directory)
        assert all(torch.equal(on_disk.inputs[k],prepared.inputs[k]) for k in prepared.inputs)
        driver=FrozenBankDriver.from_training_checkpoint(a.checkpoint,expected_sha256=a.checkpoint_sha,
                        bank_path=a.bank,device='cpu',precision='fp32',backend='grid')
        started=time.perf_counter();prediction=driver.predict_clip(directory,return_details=True)
        seconds=time.perf_counter()-started
        assert prediction['trajectory'].shape==(6,2) and np.isfinite(prediction['trajectory']).all()
        bank=driver.model._trajectory_head.traj_vocab.flatten(0,1)
        assert np.array_equal(prediction['trajectory'],bank[prediction['candidate_id'],:6,:2].numpy())
        report['trained_entry']={'passed':True,'checkpoint_sha256':a.checkpoint_sha,
            'candidate_id':prediction['candidate_id'],'trajectory_shape':[6,2],
            'raw_to_model_seconds_cpu':seconds,'fixed_bank_row_exact':True,
            'strict_load':driver.provenance,'raw_metadata_only':True}
    report.update(passed=True,raw_test_shaped_disk_interface_exact=True,
                  repeated_input_exact=True,disabled_goal_not_read=True,
                  no_gpu_used=True,hidden_test_clips_read=False)
    Path(a.output).write_text(json.dumps(report,indent=2,allow_nan=False))
    slim={k:v for k,v in report.items() if k!='samples'}
    print(json.dumps(slim,indent=2))
    for r in report['samples']:
        print(json.dumps({'split':r['split'],'row':r['row'],'status':r['status_max_abs'],'goal':r['goal_max_abs'],
            'projection':r['projection_max_abs'],'image_mae':[v['cache_equivalent_vs_archived']['rgb_255_mean_abs'] for v in r['images'].values()],
            'no_jpeg_mae':[v['omit_q95_vs_archived']['rgb_255_mean_abs'] for v in r['images'].values()]}))


if __name__=='__main__':main()
