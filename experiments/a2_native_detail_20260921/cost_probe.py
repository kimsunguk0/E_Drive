"""Whole native-detail student graph on two saved, real DEV B1 inputs."""
from pathlib import Path
import gc,hashlib,json,sys
import numpy as np
import torch

ROOT=Path('/app/code')
sys.path.insert(0,str(ROOT/'experiments/a2_native_detail_20260921'))
from native_model import load_export

torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
results=[]
for arm in ('M-NATIVE','S-NATIVE'):
    path=Path('/weights')/f'{arm}_student.pth'
    model,payload=load_export(path,'cuda')
    batch=torch.load(Path('/weights')/f'{arm}_inputs.pth',map_location='cpu',weights_only=True)
    timing=[];flops=None
    for index in (0,1):
        inputs={k:v[index:index+1].cuda() for k,v in batch.items()}
        if flops is None:
            from torch.utils.flop_counter import FlopCounterMode
            import torch.utils.module_tracker as tracker
            class Handle:
                def remove(self):pass
            old=tracker.register_multi_grad_hook
            tracker.register_multi_grad_hook=lambda *a,**kw:Handle()
            try:
                with torch.no_grad(),FlopCounterMode(display=False) as counter:
                    model(**inputs)
                flops=int(sum(counter.get_flop_counts()['Global'].values()))
            finally:tracker.register_multi_grad_hook=old
        ms=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for _ in range(30):model(**inputs)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            for _ in range(200):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();output=model(**inputs);b.record();b.synchronize()
                ms.append(a.elapsed_time(b))
        assert output['plan_abs'].shape==(1,6,2) and torch.isfinite(output['plan_abs']).all()
        timing.append({'train_row':(0,30000)[index],'median_ms':float(np.median(ms)),
            'p95_ms':float(np.quantile(ms,.95)),'warmup':30,'repeats':200,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
        del inputs,output
    results.append({'arm':arm,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'weights_role':'five-update smoke; cost check, not a candidate score',
        'flops':flops,'timing':timing,'device':torch.cuda.get_device_name(0),'torch':torch.__version__})
    del model,payload,batch;gc.collect();torch.cuda.empty_cache()
out={'status':'passed','scope':'entire B1 student forward including baseline AND additional detail branch; file/preprocessing/transfer excluded',
    'counter':'torch.utils.flop_counter.FlopCounterMode, FP32 Global sum',
    'precision':'BF16 image path, FP32 planner','two_real_DEV_inputs':True,'models':results,
    'LOW_cost':'Identical tensor shapes and graph within each pair; LOW preprocessing is outside model forward.',
    'raw_submission_adapter_verified':False}
Path('/out/RTX4090_cost.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
