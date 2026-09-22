"""Portable raw-clip H4-PROGRESS inference. No training loader or GT required."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(os.environ.get('ADCL_CODE_ROOT',Path(__file__).resolve().parents[2]))
for rel in ('','scripts','experiments/md_r0_reset_20260914','experiments/md_shared_dynamics_20260917',
            'experiments/md_a2_scene_extensions_20260918','experiments/a2_motion_fresh_20260919',
            'experiments/a2_temporal_read_20260920','experiments/a2_progress_h4_20260920'):
    sys.path.insert(0,str(ROOT/rel))
from progress_model import H4ProgressModel,PROGRESS
import h4_status
from models.motiondrive_v2 import MotionDriveV2Config
from models import motiondrive_v2_inputs as adapter
import matching_resolution as mr
import mr_deploy

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def prepare_clip(clip):
    import pyarrow.parquet as pq
    clip=Path(clip)
    calibration=pq.read_table(clip/'calibration.parquet',columns=list(adapter.CALIBRATION_COLUMNS)).to_pylist()
    poses=pq.read_table(clip/'ego_pose.parquet',columns=list(adapter.POSE_COLUMNS)).to_pylist()
    calls=[]
    def image(camera,frame):
        calls.append((camera,int(frame)));return (clip/camera/f'frame_{frame}.jpg').read_bytes()
    prepared=mr_deploy.prepare_mr_clip_from_records(calibration,poses,image,detail='native')
    status,diagnostic=h4_status.status_from_clip_records(poses)
    frames=sorted(set(f for _,f in calls));assert set(diagnostic['fit_relative_frames'])<=set(frames)
    prepared.inputs['provided_status5']=torch.from_numpy(status)[None]
    prepared.metadata['provided_status']={**diagnostic,'producer':'h4_status.status_from_clip_records',
        'route':'shared scene query only','consumed_RGB_frames':frames,'image_getter_calls':calls}
    return prepared

def load_model(checkpoint,require_full=False,device=None):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    protocol=payload['manifest']['experimental_protocol']
    # Continuations of the H4-PROGRESS graph keep the same deployed architecture:
    # they differ in weights, data and schedule only. The flow arms train an extra
    # readout head, but it is never written to the checkpoint -- asserted below --
    # so the loaded graph is identical for all of them.
    assert protocol['arm'] in ('A2-H4-PROGRESS','A2-H4-PROGRESS-FULL','L-FULL6',
                               'L-TRAIN3106','F-CTRL','F-FLOW','F-LENW',
                               # the flow arms inherit the parent protocol
                               # verbatim, so their checkpoints report the
                               # parent's arm name rather than their own
                               'P-NEARFULL-H'), protocol['arm']
    assert not any('flow_readout' in k for k in payload['model']), (
        'a training-only flow head reached the checkpoint')
    if require_full:
        assert protocol['arm']=='A2-H4-PROGRESS-FULL' and payload['step']==24931 and not protocol['smoke']
    assert protocol['nominal_input']['producer_sha256']==sha(h4_status.__file__)
    model=H4ProgressModel(MotionDriveV2Config(**payload['manifest']['model_config']),arm=PROGRESS)
    mr.rebuild_correlation_fuse(model,4);model.load_state_dict(payload['model'],strict=True)
    assert model.planner.progress_units.shape==(3,) and model.progress_head is None
    return model.to(device or torch.device('cuda:0')).eval(),payload

def predict(model,prepared,precision='bf16'):
    device=next(model.parameters()).device
    inputs={k:v.to(device) for k,v in prepared.inputs.items()}
    with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=precision=='bf16'):
        plan=model(**inputs)['plan_abs'].float().cpu().numpy()[0]
    assert plan.shape==(6,2) and np.isfinite(plan).all()
    return plan

def configure():
    torch.set_num_threads(4);torch.manual_seed(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--clips-root',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--precision',choices=('bf16','fp32'),default='bf16')
    ap.add_argument('--require-full',action='store_true');ap.add_argument('--expected-device');ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--cpu',action='store_true',help='Portable CPU correctness check, not target GPU timing')
    ap.add_argument('--timing',action='store_true');ap.add_argument('--warmup',type=int,default=30);ap.add_argument('--repeats',type=int,default=200)
    args=ap.parse_args();configure();name='CPU' if args.cpu else torch.cuda.get_device_name(0)
    assert not (args.cpu and args.timing),'GPU timing requires a GPU'
    if args.expected_device:assert args.expected_device in name,(args.expected_device,name)
    model,payload=load_model(args.checkpoint,args.require_full,torch.device('cpu' if args.cpu else 'cuda:0'))
    clips=sorted(p for p in Path(args.clips_root).iterdir() if p.is_dir())
    if args.limit:clips=clips[:args.limit]
    assert clips
    predictions={};timing=[];frames={};input_sha={}
    for clip in clips:
        prepared=prepare_clip(clip);predictions[clip.name]=predict(model,prepared,args.precision).tolist()
        frames[clip.name]=prepared.metadata['provided_status']
        input_sha[clip.name]={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in prepared.inputs.items()}
        if args.timing:
            inp={k:v.cuda() for k,v in prepared.inputs.items()}
            def forward():
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.precision=='bf16'):model(**inp)
            for _ in range(args.warmup):forward()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(args.repeats):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();forward();end.record();end.synchronize();times.append(start.elapsed_time(end))
            timing.append({'clip':clip.name,'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
                'warmup':args.warmup,'repeats':args.repeats,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
    result={'checkpoint_sha256':sha(args.checkpoint),'arm':payload['manifest']['experimental_protocol']['arm'],
        'step':payload['step'],'device':name,'torch':torch.__version__,'cuda':torch.version.cuda,
        'precision':args.precision,'output':'absolute XY6x2, no serving cumsum','forwards_per_clip':1,
        'predictions':predictions,'input_frame_checks':frames,'input_tensor_sha256':input_sha,'timing':timing,
        'timing_scope':'whole model forward including all current/history image encoders; preprocessing excluded',
        'provided_status_route':'common scene query only','official_upload_performed':False,
        'imported_source_files':{module.__name__:str(Path(module.__file__).resolve()) for module in
            (sys.modules['progress_model'],sys.modules['temporal_model'],sys.modules['motion_model'],
             sys.modules['scene_extensions'],sys.modules['factorized_model'],mr,mr_deploy,h4_status)}}
    out=Path(args.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('device','arm','step','checkpoint_sha256','timing')}),flush=True)
if __name__=='__main__':main()
