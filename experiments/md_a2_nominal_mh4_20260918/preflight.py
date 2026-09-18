"""Actual-input checks before launching matched A2-NOM training."""
from __future__ import annotations
import dataclasses,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
ROOT=Path("/NHNHOME/data/sukim/adcl");HERE=Path(__file__).resolve().parent
for p in (ROOT,ROOT/"scripts",ROOT/"experiments/md_shared_dynamics_20260917",HERE):sys.path.insert(0,str(p))
from a2_model import A2NominalModel,MultiHeadEvidence,MH4_PREFIX
from nominal_data import NominalStatusDataset,load_cache,CACHE
from train_nominal import raw_datasets,BASE_INITIAL_SHA,diagnostics
import train_shared_dynamics as legacy
import matching_resolution as mr
import train_motiondrive_v2 as trainer
from models.motiondrive_v2 import MotionDriveV2Config
from motiondrive_v2_training import tensor_state_sha256,model_inputs,to_device
from motiondrive_v2_flip_augment import flip_item
OUT=ROOT/"reports/md_a2_nominal_mh4_20260918"
def make_initial(arm,common):
    trainer.seed_all(1)
    cfg=MotionDriveV2Config(**trainer.initialization_configuration(common["manifest"],
      goal_on=1,state_on=1,explicit_arch="resnet50",cross_cell_goal_mode="zero",history_contract="control"))
    cfg.motion_input_mode="low_feature"
    model=A2NominalModel(cfg,arm=arm);mr.rebuild_correlation_fuse(model,4)
    state={k:v for k,v in common["model"].items() if not k.startswith("motion_encoder.correlation_fuse.0.")}
    missing=model.load_state_dict(state,strict=False)
    assert not missing.unexpected_keys
    shared={k:v for k,v in model.state_dict().items() if not k.startswith(MH4_PREFIX)}
    assert tensor_state_sha256(shared)==BASE_INITIAL_SHA
    if arm=="A2-MH4-NOM":model.scene_encoder.evidence_attention.assert_identity()
    return model

def inputs(batch):
    x=mr.model_inputs_with_canvas(model_inputs,batch,time_input="nominal")
    x["provided_status5"]=batch["provided_status5"]
    return x
def maxdiff(a,b):return float((a.float()-b.float()).abs().max())
def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="3":raise ValueError("Preflight uses physical GPU3 only")
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    result={"status":"running","physical_gpu":3,"training_performed":False}
    _,raw=raw_datasets(False,1)
    base_dataset=mr.MotionCanvasDataset(raw,"native")
    nominal=NominalStatusDataset(base_dataset)
    with np.load(ROOT/"reports/md_a2_deploy_status_20260918/paired_status_eval.npz",allow_pickle=False) as z:
        np.testing.assert_array_equal(raw.rows,z["row"])
        np.testing.assert_array_equal(nominal.nominal_status,z["nominal_status"])
    result["all1998_nominal_inputs_equal_P0"]=True
    item=nominal[0];original=base_dataset[0]
    for key in ("state_target","history_target","gt_plan"):
        assert torch.equal(item[key],original[key])
    assert item["provided_status5"].data_ptr()!=item["state_target"].data_ptr()
    flipped=mr.wrap_flip_item(flip_item)(item,768,384)
    sign=torch.tensor([1.,-1.,1.,-1.,-1.])
    assert torch.equal(flipped["provided_status5"],item["provided_status5"]*sign)
    assert torch.equal(flipped["state_target"][:5],item["state_target"][:5]*sign)
    result["supervision_unchanged_and_separate_flip"]=True
    full,_=raw_datasets(True,1)
    full_nominal=NominalStatusDataset(mr.MotionCanvasDataset(full,"native"),allowed_splits=("train","tune","val"))
    result["data"]={"DEV_train_rows":83700,"DEV_eval_rows":len(raw),"FULL_rows":len(full_nominal),
        "FULL_scenes":len(set(full.scene_names[full.rows])),"DEV_cache_splits":list(nominal.allowed_nominal_splits)}
    common=torch.load(legacy.INITIALIZER,map_location="cpu",weights_only=False)
    base=make_initial("A2-BASE-NOM",common).eval().cuda()
    mh4=make_initial("A2-MH4-NOM",common).eval().cuda()
    del common
    result["shared_initial_sha256"]=BASE_INITIAL_SHA
    result["parameter_counts"]={"BASE":sum(p.numel() for p in base.parameters()),"MH4":sum(p.numel() for p in mh4.parameters())}
    loader=DataLoader(Subset(nominal,list(range(8))),batch_size=8,shuffle=False,num_workers=2)
    batch=to_device(next(iter(loader)),torch.device("cuda:0"));x=inputs(batch)
    result["fresh_initial_output_differences"]={}
    for label,use_amp in (("fp32",False),("bf16",True)):
        # Full-input parity on the same eight examples, including shared heads.
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=use_amp):
            a=base(**x);b=mh4(**x)
        differences={k:maxdiff(a[k],b[k]) for k in ("scene_features","occ_logits","lane_logits","plan_abs","motion_features","state_hat","history_hat")}
        assert differences["plan_abs"]<=1e-4 and max(differences.values())<=5e-4,differences
        result["fresh_initial_output_differences"][label]=differences
        del a,b
    print("INITIAL "+json.dumps(result),flush=True)
    # Actual final planning loss must reach all new query/key/value read transforms.
    tiny={k:v[:2] for k,v in x.items()};mh4.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        out=mh4(**tiny)
    w=torch.tensor([11,11,5,5,2,2],device="cuda",dtype=torch.float32)/36
    loss=(torch.linalg.vector_norm(out["plan_abs"]-batch["gt_plan"][:2].float(),dim=-1)*w).sum(-1).mean()
    loss.backward()
    module=mh4.scene_encoder.evidence_attention
    grads={}
    for name in ("query_matrix","key_matrix","output_matrix"):
        grad=getattr(module,name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
        grads[name]=grad.flatten(1).norm(dim=1).detach().cpu().tolist()
    qgrad=module.query_matrix.grad.flatten(1)
    assert all(not torch.equal(qgrad[0],qgrad[h]) for h in range(1,4))
    result["actual_planning_loss_gradient_norms"]=grads
    del out,loss;mh4.zero_grad(set_to_none=True)
    # Load trained A2 into both graphs; preserve only the three identity MH4 tensors.
    terminal=torch.load(ROOT/"work_dirs/md_shared_dynamics_20260917/A2-DIRECT-s1/ckpt_step20554.pth",
                        map_location="cpu",weights_only=False)
    assert dataclasses.asdict(base.config)==terminal["manifest"]["model_config"]
    base.load_state_dict(terminal["model"],strict=True)
    missing=mh4.load_state_dict(terminal["model"],strict=False)
    assert sorted(missing.missing_keys)==[MH4_PREFIX+"key_matrix",MH4_PREFIX+"output_matrix",MH4_PREFIX+"query_matrix"]
    assert not missing.unexpected_keys
    module.assert_identity();del terminal
    real=dict(x,provided_status5=batch["state_target"][:,:5])
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        old=base(**real);new=mh4(**real)
        changed=mh4(**dict(real,provided_status5=real["provided_status5"].flip(0)))
    stored=json.loads((ROOT/"work_dirs/md_shared_dynamics_20260917/A2-DIRECT-s1/final_eval.json").read_text())["records"]
    reference=torch.tensor([r["pred_abs_xy"] for r in stored[:8]],device="cuda")
    result["registered_A2_prediction_replay_max_abs"]=maxdiff(old["plan_abs"],reference)
    assert result["registered_A2_prediction_replay_max_abs"]==0.
    result["trained_A2_to_MH4_initial_plan_max_abs"]=maxdiff(old["plan_abs"],new["plan_abs"])
    assert result["trained_A2_to_MH4_initial_plan_max_abs"]<=1e-4
    result["MH4_status_swap_invariance"]={k:maxdiff(new[k],changed[k]) for k in ("motion_features","state_hat","history_hat")}
    assert max(result["MH4_status_swap_invariance"].values())==0.
    # Partial/all-invalid sources through the integrated aggregation module.
    q=torch.randn(2,4,32,device="cuda",requires_grad=True)
    k=torch.randn(2,4,6,32,device="cuda",requires_grad=True)
    v=torch.randn(2,4,6,128,device="cuda",requires_grad=True)
    valid=torch.ones(2,4,6,device="cuda",dtype=torch.bool);valid[:,0]=False;valid[:,1,:3]=False
    evidence=module(q,k,v,valid);assert torch.equal(evidence[:,0],torch.zeros_like(evidence[:,0]))
    evidence.sum().backward()
    assert torch.equal(q.grad[:,0],torch.zeros_like(q.grad[:,0])) and torch.equal(v.grad[:,0],torch.zeros_like(v.grad[:,0]))
    result["all_invalid_evidence_and_input_gradients_zero"]=True
    result.update(status="passed",elapsed_s=time.monotonic()-start,
       peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
       scope="Actual input/model parity, supervision/input separation and planning gradients; not MH4 trained performance")
    (OUT/"preflight.json").write_text(json.dumps(result,indent=2)+"\n")
    print("RESULT "+json.dumps(result),flush=True)
if __name__=="__main__":main()
