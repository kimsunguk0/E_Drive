"""Measure effective-batch16 step cost; LR zero, train inputs, no retained model."""
import gc,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
ROOT=Path("/NHNHOME/data/sukim/adcl");HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from preflight import make_initial,inputs
from train_nominal import raw_datasets,REPORT
from nominal_data import NominalStatusDataset
import train_shared_dynamics as legacy
import matching_resolution as mr
import train_motiondrive_v2 as trainer
from motiondrive_v2_training import LossWeights,build_loss_normalizers,to_device,set_training_mode,tensor_state_sha256
from length_auxiliary import wrap_compute_loss
def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="3"
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    train,_=raw_datasets(False,1)
    dataset=NominalStatusDataset(mr.MotionCanvasDataset(train,"native"))
    raw=next(iter(DataLoader(Subset(dataset,list(range(16))),batch_size=16,num_workers=2,pin_memory=True)))
    normalizers=to_device(build_loss_normalizers(raw),torch.device("cuda:0"))
    common=torch.load(legacy.INITIALIZER,map_location="cpu",weights_only=False)
    weights=LossWeights(plan=1.,occupancy=.2,lane=.2,motion=.2,uncertainty=True)
    loss_fn=wrap_compute_loss(trainer.compute_loss,.25)
    result={"physical_gpu":3,"logical_batch":16,"optimizer_lr":0.,"precision":"bf16","bn_policy":"fixed",
       "data":"first16 train310 rows; decode outside timing; host-to-device included",
       "candidate_weights_retained":False,"cases":{}}
    for arm in ("A2-BASE-NOM","A2-MH4-NOM"):
        for micro in (8,16):
            model=make_initial(arm,common).cuda();set_training_mode(model,"fixed")
            initial=tensor_state_sha256(model.state_dict())
            optimizer=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=.01)
            times=[];losses=[];torch.cuda.reset_peak_memory_stats()
            for step in range(7):
                torch.cuda.synchronize();t=time.perf_counter();optimizer.zero_grad(set_to_none=True);total=0.
                for lo in range(0,16,micro):
                    batch=to_device(trainer.slice_batch(raw,lo,lo+micro),torch.device("cuda:0"))
                    with torch.autocast("cuda",dtype=torch.bfloat16):out=model(**inputs(batch))
                    loss,_=loss_fn(out,batch,weights,normalizers=normalizers)
                    assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
                    del batch,out,loss
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                assert torch.isfinite(norm);optimizer.step();torch.cuda.synchronize()
                if step>=2:times.append(time.perf_counter()-t);losses.append(total)
            assert tensor_state_sha256(model.state_dict())==initial
            item={"median_seconds":float(np.median(times)),"step_seconds":times,
              "peak_allocated_GiB":torch.cuda.max_memory_allocated()/1024**3,"loss_mean":float(np.mean(losses))}
            result["cases"][arm+"_mb"+str(micro)]=item;print("CASE "+json.dumps({arm+"_mb"+str(micro):item}),flush=True)
            del model,optimizer;gc.collect();torch.cuda.empty_cache()
    result["selected_microbatch"]=(16 if all(
       result["cases"][arm+"_mb16"]["median_seconds"]<=.9*result["cases"][arm+"_mb8"]["median_seconds"]
       and result["cases"][arm+"_mb16"]["peak_allocated_GiB"]<150.
       for arm in ("A2-BASE-NOM","A2-MH4-NOM")) else 8)
    result["selection_rule"]="16 only if >=10% faster on both arms and peak allocation <150GiB; otherwise retain8"
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/"throughput_benchmark.json").write_text(json.dumps(result,indent=2)+"\n")
    print("RESULT "+json.dumps(result),flush=True)
if __name__=="__main__":main()
