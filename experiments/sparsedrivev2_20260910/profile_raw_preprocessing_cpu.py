"""Bounded CPU-only sequential/three-worker raw preprocessing parity/profile.

Inputs are eight train/tune fixtures with JPEG bytes already in memory. No global
OpenCV thread setter is called: pin OPENCV_FOR_THREADS_NUM before process start
when testing a desired per-process OpenCV configuration.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from .deployment import CAMERAS, RawInputAdapter, file_sha256
from .test_deployment_cpu import BASE, WT, RAW, PlanDataset, raw_records


def same(a,b):
    assert set(a.inputs)==set(b.inputs)
    for name in a.inputs:
        assert torch.equal(a.inputs[name],b.inputs[name]), name
    for camera in CAMERAS:
        assert a.metadata['camera_geometry'][camera] == b.metadata['camera_geometry'][camera]


def stats(wall,cpu):
    wall,cpu=np.array(wall)*1000,np.array(cpu)*1000
    return {'iterations':len(wall),'wall_ms_mean':float(wall.mean()),
        'wall_ms_p50':float(np.percentile(wall,50)), 'wall_ms_p95':float(np.percentile(wall,95)),
        'wall_ms_min':float(wall.min()),'wall_ms_max':float(wall.max()),
        'process_cpu_ms_mean':float(cpu.mean()),'mean_cpu_cores_used':float(cpu.sum()/wall.sum()),
        'wall_ms_all':wall.tolist()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--iterations',type=int,default=32)
    parser.add_argument('--torch-threads',type=int,default=4)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert args.iterations>=30 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(args.torch_threads)
    fixtures=[]
    for split in ('train','tune'):
        ds=PlanDataset(str(BASE),str(WT/'reports/sparsedrivev2_20260910/split_audit/primary_manifest.json'),
                       split,status_mode='causal_selection',goal_mode='selection',limit=4)
        for i in range(4):
            expected=ds[i];scene,frame=expected['scenario'],expected['frame']
            cal,poses=raw_records(scene,frame)
            with tarfile.open(RAW/f'{scene}.tar','r:') as archive:
                raw={c:archive.extractfile(f'{scene}/{c}/{frame:08d}.jpg').read() for c in CAMERAS}
            fixtures.append((cal,poses,raw,expected,split))
    report={'scope':'CPU in-memory raw JPEG -> model tensors; no parquet/tar I/O or model/GPU',
        'loadavg_start':os.getloadavg(),'opencv_threads':cv2.getNumThreads(),
        'torch_threads':torch.get_num_threads(),'affinity_cpus':len(os.sched_getaffinity(0)),
        'opencv_version':cv2.__version__,
        'thread_environment':{k:os.environ.get(k) for k in ('OPENCV_FOR_THREADS_NUM','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        'source_sha256':{p:file_sha256(Path(__file__).with_name(p)) for p in ('deployment.py','profile_raw_preprocessing_cpu.py')},
        'parity':[],'timing':{}}
    with RawInputAdapter('causal_selection','selection',1) as sequential, RawInputAdapter('causal_selection','selection',3) as parallel:
        reference=[]
        for cal,poses,raw,expected,split in fixtures:
            loader=lambda c,f:raw[c]
            one=sequential.prepare_records(cal,poses,loader)
            three=parallel.prepare_records(cal,poses,loader)
            same(one,three)
            assert torch.equal(one.inputs['images'][0],expected['images'])
            assert torch.equal(one.inputs['status'][0],expected['status'])
            assert torch.equal(one.inputs['goal_xy'][0],expected['goal_xy'])
            assert torch.equal(one.inputs['lidar2img'][0],expected['lidar2img'])
            reference.append(one)
            report['parity'].append({'split':split,'row':int(expected['row']),'all_float32_inputs_exact':True,
                                    'camera_provenance_exact':True})
        # Exercise concurrent calls against a single adapter. Inputs and remap
        # references are request-local; workers never mutate calibration state.
        with ThreadPoolExecutor(max_workers=2) as callers:
            jobs=[callers.submit(parallel.prepare_records,cal,poses,lambda c,f,r=raw:r[c])
                  for cal,poses,raw,_,_ in fixtures]
            for output,expected in zip(jobs,reference):same(output.result(),expected)
        report['concurrent_requests_exact']=True
        # Interleave 1/3-worker order to reduce bias from changing shared host load.
        measurements={(workers,kind):([],[]) for workers in (1,3) for kind in ('warm','cold')}
        for i in range(args.iterations):
            cal,poses,raw,_,_=fixtures[i%len(fixtures)]
            for workers in ((1,3) if i%2==0 else (3,1)):
                for kind in ('warm','cold'):
                    cpu0=time.process_time();wall0=time.perf_counter()
                    adapter=(sequential if workers==1 else parallel) if kind=='warm' else RawInputAdapter('causal_selection','selection',workers)
                    actual=adapter.prepare_records(cal,poses,lambda c,f:raw[c])
                    wall=time.perf_counter()-wall0;cpu=time.process_time()-cpu0
                    if kind=='cold':adapter.close()
                    same(actual,reference[i%len(fixtures)])
                    measurements[(workers,kind)][0].append(wall)
                    measurements[(workers,kind)][1].append(cpu)
        for (workers,kind),values in measurements.items():
            report['timing'][f'workers{workers}_{kind}']=stats(*values)
    report.update(passed=True,no_gpu_used=not torch.cuda.is_initialized(),
                  loadavg_end=os.getloadavg(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  cold_boundary='includes adapter/pool and calibration-map construction; cleanup excluded',
                  warm_boundary='reuses calibration maps and camera pool; includes status, pixels, normalization, validation')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('parity','timing')},indent=2))
    print(json.dumps({k:{x:y for x,y in v.items() if x!='wall_ms_all'} for k,v in report['timing'].items()},indent=2))


if __name__=='__main__':main()
