"""Portable A2 FULL raw inference / complete forward timing. No teacher imports."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(os.environ.get('ADCL_CODE_ROOT',Path(__file__).resolve().parents[2]))
for p in (ROOT,ROOT/'scripts',ROOT/'experiments/md_r0_reset_20260914',
          ROOT/'experiments/md_a2_nominal_mh4_20260918',ROOT/'experiments/md_shared_dynamics_20260917',
          ROOT/'experiments/md_a2_deploy_status_20260918'):
    sys.path.insert(0,str(p))
from a2_model import A2NominalModel
from nominal_status import status5_from_clip_records
from models.motiondrive_v2 import MotionDriveV2Config
from models import motiondrive_v2_inputs as adapter
import matching_resolution as mr
import mr_deploy


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def raw_inputs(clip):
    import pyarrow.parquet as pq
    calibration=pq.read_table(clip/'calibration.parquet',columns=list(adapter.CALIBRATION_COLUMNS)).to_pylist()
    poses=pq.read_table(clip/'ego_pose.parquet',columns=list(adapter.POSE_COLUMNS)).to_pylist()
    prepared=mr_deploy.prepare_mr_clip_from_records(calibration,poses,
        lambda camera,frame:(clip/camera/f'frame_{frame}.jpg').read_bytes(),detail='native')
    status,_=status5_from_clip_records(poses)
    prepared.inputs['provided_status5']=torch.from_numpy(status)[None]
    return prepared.inputs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--clips-root');p.add_argument('--input-tensor')
    p.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    p.add_argument('--timing',action='store_true');p.add_argument('--expected-device')
    p.add_argument('--limit',type=int,default=0);p.add_argument('--warmup',type=int,default=30)
    p.add_argument('--repeats',type=int,default=200);p.add_argument('--flops',type=int)
    a=p.parse_args();assert bool(a.clips_root)!=bool(a.input_tensor)
    torch.set_num_threads(4);torch.manual_seed(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    name=torch.cuda.get_device_name(0)
    if a.expected_device:assert a.expected_device in name,(a.expected_device,name)
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=A2NominalModel(MotionDriveV2Config(**payload['manifest']['model_config']),arm='A2-FULL-NOM')
    mr.rebuild_correlation_fuse(model,4);model.load_state_dict(payload['model'],strict=True)
    model.cuda().eval()
    sources=[Path(a.input_tensor)] if a.input_tensor else sorted(x for x in Path(a.clips_root).iterdir() if x.is_dir())
    if a.limit:sources=sources[:a.limit]
    outputs={};costs=[]
    def forward(inp):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.precision=='bf16'):
            return model(**inp)['plan_abs'].float()
    for source in sources:
        cpu=torch.load(source,map_location='cpu',weights_only=True) if a.input_tensor else raw_inputs(source)
        inp={k:v.cuda() for k,v in cpu.items()}
        plan=forward(inp).cpu().numpy()[0]
        assert plan.shape==(6,2) and np.isfinite(plan).all()
        outputs[source.name]=plan.tolist()
        if a.timing:
            for _ in range(a.warmup):forward(inp)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            values=[]
            for _ in range(a.repeats):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();forward(inp);end.record();end.synchronize();values.append(start.elapsed_time(end))
            costs.append({'input':source.name,'median_ms':float(np.median(values)),
                'p95_ms':float(np.percentile(values,95)),'min_ms':min(values),'max_ms':max(values),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'warmup':a.warmup,'repeats':a.repeats})
    result={'checkpoint_sha256':sha(a.checkpoint),'device':name,'torch':torch.__version__,'cuda':torch.version.cuda,
        'precision':a.precision,'teacher':False,'projector':False,'forwards_per_clip':1,
        'scope':'entire forward including all current/history backbone calls; raw preprocessing excluded',
        'timing':costs,'predictions':outputs}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if a.flops is not None:
        assert not a.timing and not a.input_tensor
        outputs['__flops__']=a.flops
        out.write_text(json.dumps(outputs,allow_nan=False))
        out.with_suffix('.runtime.json').write_text(json.dumps(result,indent=2)+'\n')
    else:out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['device','checkpoint_sha256','timing']}),flush=True)

if __name__=='__main__':main()
