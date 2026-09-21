"""Inference-only full-forward cost of the concrete five-update student exports."""
from pathlib import Path
import gc,hashlib,json,os,sys
import numpy as np
import torch

ROOT=Path('/app/code')
for rel in ('experiments/a2_progress_fourarm_20260921','experiments/a2_splitread_lanegeom_20260921'):
    sys.path.insert(0,str(ROOT/rel))
from infer_fourarm import prepare_clip,configure
from next_model import load_export

configure();torch.set_num_threads(4)
results=[];clips=sorted(p for p in Path('/clips').iterdir() if p.is_dir())[:2]
for arm in ('P-CTRL-NEXT','P-SPLITREAD','P-LANE-GEOM'):
    path=Path('/weights')/(arm+'_student.pth')
    model,payload=load_export(path,'cuda')
    assert not hasattr(model,'lane_geometry_head')
    timing=[];flops=None
    for clip in clips:
        prepared=prepare_clip(clip);inputs={k:v.cuda() for k,v in prepared.inputs.items()}
        if flops is None:
            from torch.utils.flop_counter import FlopCounterMode
            import torch.utils.module_tracker as tracker
            class Handle:
                def remove(self):pass
            old=tracker.register_multi_grad_hook;tracker.register_multi_grad_hook=lambda *a,**kw:Handle()
            try:
                with torch.no_grad(),FlopCounterMode(display=False) as counter:model(**inputs)
                flops=int(sum(counter.get_flop_counts()['Global'].values()))
            finally:tracker.register_multi_grad_hook=old
        milliseconds=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for _ in range(30):model(**inputs)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            for _ in range(200):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();output=model(**inputs);b.record();b.synchronize();milliseconds.append(a.elapsed_time(b))
        assert output['plan_abs'].shape==(1,6,2) and torch.isfinite(output['plan_abs']).all()
        timing.append({'clip':clip.name,'median_ms':float(np.median(milliseconds)),
            'p95_ms':float(np.quantile(milliseconds,.95)),'warmup':30,'repeats':200,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
        del inputs,output
    results.append({'arm':arm,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'weights_role':'five-update smoke; not a trained candidate score','flops':flops,'timing':timing,
        'student_without_geometry_head':True,'device':torch.cuda.get_device_name(0),'torch':torch.__version__})
    del model,payload;gc.collect();torch.cuda.empty_cache()
out={'status':'passed','scope':'entire B1 student forward, BF16 image path and FP32 planner; file/preprocessing/transfer excluded',
     'counter':'torch.utils.flop_counter.FlopCounterMode, full FP32 graph Global sum','models':results}
Path('/out/RTX4090_cost.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
