"""Bounded B=1 native model FLOP and raw-preprocessing timing audit on allocated GPU1."""
from __future__ import annotations
import argparse,json,os,tarfile,time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
import torch.utils.module_tracker as module_tracker

from .deployment import FrozenBankDriver,CAMERAS,file_sha256
from .test_deployment_cpu import BASE,WT,RAW,PlanDataset,raw_records
from .dfa_flops_supplement import build_report


class NoHandle:
    def remove(self):pass


def stats(values):
    x=np.asarray(values)*1000
    return {'n':len(x),'mean_ms':float(x.mean()),'p50_ms':float(np.percentile(x,50)),
            'p95_ms':float(np.percentile(x,95)),'min_ms':float(x.min()),'max_ms':float(x.max())}


def sync():torch.cuda.synchronize()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=50)
    p.add_argument('--physical-gpu',type=int,choices=(0,1,4),default=1)
    p.add_argument('--relative-head');p.add_argument('--relative-head-sha256')
    a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.physical_gpu) and a.samples>=50
    torch.set_num_threads(4)
    checkpoint=Path(a.checkpoint)
    manifest=json.loads((checkpoint.parent/'manifest.json').read_text())
    sha=file_sha256(checkpoint)
    report={'device':torch.cuda.get_device_name(0),'physical_gpu':a.physical_gpu,'checkpoint_sha256':sha,
            'opencv_threads':cv2.getNumThreads(),'torch_threads':torch.get_num_threads(),
            'scope':'B200 only, no RTX4090 inference claim; batch1',
            'loadavg_start':os.getloadavg(),'samples':[]}
    driver=FrozenBankDriver.from_training_checkpoint(checkpoint,expected_sha256=sha,
            bank_path=manifest['arguments']['bank'],device='cuda:0',precision='bf16',backend='native',
            relative_head_path=a.relative_head,relative_head_sha256=a.relative_head_sha256)
    data=PlanDataset(str(BASE),str(WT/'reports/sparsedrivev2_20260910/split_audit/primary_manifest.json'),
            'tune',status_mode=driver.adapter.status_mode,goal_mode=driver.adapter.goal_mode,limit=a.samples)
    fixtures=[]
    for i in range(len(data)):
        sample=data[i];scene,frame=sample['scenario'],sample['frame']
        cal,poses=raw_records(scene,frame)
        with tarfile.open(RAW/f'{scene}.tar','r:') as archive:
            raw={c:archive.extractfile(f'{scene}/{c}/{frame:08d}.jpg').read() for c in CAMERAS}
        prepared=driver.adapter.prepare_records(cal,poses,lambda c,f:raw[c])
        assert torch.equal(prepared.inputs['images'][0],sample['images'])
        assert torch.equal(prepared.inputs['status'][0],sample['status'])
        assert torch.equal(prepared.inputs['lidar2img'][0],sample['lidar2img'])
        fixtures.append((cal,poses,raw,prepared))
        report['samples'].append({'row':int(sample['row']),'scene':scene,'frame':frame,'raw_cache_inputs_exact':True})
    # Extension compilation and 10 model warmups are excluded from steady-state timing.
    for i in range(10):driver.predict_prepared(fixtures[i][3])
    sync();torch.cuda.reset_peak_memory_stats()
    stages={k:[] for k in ('raw_prepare','h2d','model_only','bank_check_and_d2h','full_warm_raw')}
    ids=[]
    with torch.inference_mode():
        for cal,poses,raw,prepared in fixtures:
            begin=time.perf_counter()
            raw_prepared=driver.adapter.prepare_records(cal,poses,lambda c,f:raw[c])
            stages['raw_prepare'].append(time.perf_counter()-begin)
            start=time.perf_counter();values={k:v.to('cuda:0') for k,v in raw_prepared.inputs.items()};sync()
            stages['h2d'].append(time.perf_counter()-start)
            start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):out=driver.model(**values)
            sync();stages['model_only'].append(time.perf_counter()-start)
            start=time.perf_counter()
            bank=driver.model._trajectory_head.traj_vocab.flatten(0,1)
            assert torch.equal(out['candidate_xy'],bank[out['candidate_ids'],:6,:2])
            assert torch.equal(out['trajectory'],bank[out['selected_candidate_id'],:6,:2])
            prediction=out['trajectory'][0].float().cpu().numpy()
            ids.append(int(out['selected_candidate_id'].item()))
            assert prediction.shape==(6,2)
            sync();stages['bank_check_and_d2h'].append(time.perf_counter()-start)
            stages['full_warm_raw'].append(time.perf_counter()-begin)
    report['timing']={k:stats(v) for k,v in stages.items()}
    report['timing_boundary']='in-memory raw JPEG bytes -> calibrated normalized tensors -> H2D -> native bf16 model -> all-candidate/final bank equality checks -> six XY CPU; excludes parquet/tar/file I/O and cold map/extension/model initialization'
    report['fixed_bank_rows_exact']=True;report['selected_ids']=ids
    report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    report['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
    # Follow official FlopCounterMode and no-op multi-grad-hook compatibility workaround.
    old_hook=module_tracker.register_multi_grad_hook
    module_tracker.register_multi_grad_hook=lambda *args,**kwargs:NoHandle()
    try:
        values={k:v.to('cuda:0') for k,v in fixtures[0][3].inputs.items()}
        counter=FlopCounterMode(display=False,depth=3)
        with torch.inference_mode(),counter,torch.autocast('cuda',dtype=torch.bfloat16):
            measured=driver.model(**values)
        sync()
        glob=counter.get_flop_counts().get('Global',{})
        raw_count=sum(glob.values())
        supplement=build_report()
        report['flops']={'raw_torch_counter':int(raw_count),'raw_torch_gflops':raw_count/1e9,
            'operators':{str(k):int(v) for k,v in glob.items()},
            'native_dfa_all_camera_upper':supplement['totals']['native_kernel_flops'],
            'counter_plus_native_upper':raw_count+supplement['totals']['native_kernel_flops'],
            'counter_plus_native_upper_gflops':(raw_count+supplement['totals']['native_kernel_flops'])/1e9,
            'convention':'MAC/FMA=2; native scalar arithmetic21 per valid camera-point/level/channel',
            'limitation':'native contribution is conservative all-camera bound; standard unregistered elementwise/reduction ops and any unregistered attention dispatch are not silently treated as exact total',
            'projection_matmul_extra_added':False,'core_interpolation_subset_extra_added':False,
            'no_submission_json_written':True}
    except Exception as error:
        report['flops']={'error':repr(error)}
    finally:module_tracker.register_multi_grad_hook=old_hook
    report['source_sha256']={name:file_sha256(Path(__file__).with_name(name)) for name in ('deployment.py','public_model.py','profile_deployment_gpu.py','dfa_flops_supplement.py')}
    report.update(passed=True,provenance=driver.provenance,loadavg_end=os.getloadavg())
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples','selected_ids')},indent=2),flush=True)
    driver.adapter.close()


if __name__=='__main__':main()
