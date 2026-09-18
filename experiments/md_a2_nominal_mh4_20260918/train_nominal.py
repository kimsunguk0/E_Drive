"""Matched A2 NOM FULL/BASE/MH4 recipes using the proven joint trainer."""
from __future__ import annotations
import argparse,contextlib,hashlib,json,math,os,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path("/NHNHOME/data/sukim/adcl");HERE=Path(__file__).resolve().parent
for p in (ROOT,ROOT/"scripts",ROOT/"experiments/md_shared_dynamics_20260917"):
    sys.path.insert(0,str(p))
import train_shared_dynamics as legacy
sys.path.insert(0,str(HERE))
from a2_model import A2NominalModel,ARMS,MH4_PREFIX
from nominal_data import NominalStatusDataset,make_union,CACHE,sha
from motiondrive_v2_training import tensor_state_sha256
REPORT=ROOT/"reports/md_a2_nominal_mh4_20260918"
BASE_INITIAL_SHA="116f67a476b5ab519ec4384d6e15d5f5e36e83d1c89812f733b056ba5874593f"

def raw_datasets(full,seed):
    import motiondrive_v2_data as data_api
    common=dict(data_root="/tmp/pm97",split_manifest=str(legacy.SPLIT),
                supervision_root=str(legacy.SUPERVISION),min_frame=30,max_samples=0,
                seed=seed,history_contract="control")
    kwargs=dict(split="train",frame_stride=1,augment=False,**common)
    train=(make_union(data_api.MotionDriveDataset,kwargs) if full else data_api.MotionDriveDataset(**kwargs))
    tune=data_api.MotionDriveDataset(split="tune",frame_stride=5,augment=False,**common)
    if len(train)!=(101520 if full else 83700) or len(tune)!=1998:raise ValueError("Wrong rows")
    return train,tune

def diagnostics(records):
    p=np.asarray([r["pred_abs_xy"] for r in records],np.float64)
    g=np.asarray([r["gt_abs_xy"] for r in records],np.float64)
    w=np.asarray([11,11,5,5,2,2],np.float64)/36
    d=np.linalg.norm(p-g,axis=-1);score=d@w
    dp=np.diff(np.concatenate([np.zeros((len(p),1,2)),p],1),axis=1)
    dg=np.diff(np.concatenate([np.zeros((len(g),1,2)),g],1),axis=1)
    lp=np.linalg.norm(dp,axis=-1);lg=np.linalg.norm(dg,axis=-1)
    dv=2*(lp-lg);tangent=dg/np.maximum(lg[...,None],1e-8);valid=lg>.05
    err=p-g;longitudinal=np.abs((err*tangent).sum(-1))
    lateral=np.abs(err[...,0]*tangent[...,1]-err[...,1]*tangent[...,0])
    bucket=np.asarray([r.get("bucket",r.get("stop_bucket")) for r in records])
    groups={}
    for name,mask in [("all",np.ones(len(p),bool))]+[(b,bucket==b) for b in sorted(set(bucket))]:
        a=dv[mask,:4];b=a.mean(1,keepdims=True);remaining=a-b
        groups[name]={"n":int(mask.sum()),"PREFIX":float(score[mask].mean()),
            "overall_PREFIX_contribution":float(score[mask].sum()/len(p)),
            "first2s_common_progress_error_MAE_mps":float(abs(b).mean()),
            "first2s_time_varying_progress_error_RMS_mps":float(np.sqrt(np.square(remaining).mean()))}
    vw=valid*w[None];den=vw.sum()
    return {"n":len(p),"official_weights":w.tolist(),"first2s_PREFIX_contribution":float((d[:,:4]*w[None,:4]).sum(1).mean()),
       "groups":groups,"tangent_mask":"GT segment chord length >0.05m; identical for compared models",
       "valid_tangent_fraction":float(valid.mean()),
       "weighted_longitudinal_abs_m":float((longitudinal*vw).sum()/den),
       "weighted_lateral_abs_m":float((lateral*vw).sum()/den),
       "warning":"Projected errors are not additive D3 components; interval speeds are not measured v0"}

@contextlib.contextmanager
def configure(arm,microbatch=8):
    full=arm=="A2-FULL-NOM"
    keys=["SharedDynamicsMotionDriveV2","CausalStatusDataset","UPDATES","EVAL_EVERY","TRAIN_ROWS","MICROBATCH","REPORT_DIR"]
    saved={k:getattr(legacy,k) for k in keys}
    legacy.SharedDynamicsMotionDriveV2=A2NominalModel
    allowed=("train","tune","val") if full else ("train","tune")
    legacy.CausalStatusDataset=lambda base:NominalStatusDataset(base,allowed_splits=allowed)
    legacy.UPDATES=24931 if full else 20554
    legacy.EVAL_EVERY=6345 if full else 3426
    legacy.TRAIN_ROWS=101520 if full else 83700
    legacy.MICROBATCH=microbatch;legacy.REPORT_DIR=REPORT
    try:yield
    finally:
        for k,v in saved.items():setattr(legacy,k,v)

def build_experiment(arm,seed,train,tune,smoke,gpu):
    full=arm=="A2-FULL-NOM";value=legacy.build_experiment(arm,seed,train,tune,smoke)
    manifest=json.loads((CACHE/"manifest.json").read_text())
    if manifest["split_manifest_sha256"]!=sha(legacy.SPLIT) or manifest["supervision_manifest_sha256"]!=sha(legacy.SUPERVISION/"supervision_manifest.json"):
        raise ValueError("Nominal cache provenance mismatch")
    value.update(name="md_a2_nominal_mh4_20260918",physical_gpu=gpu,
        question="matched nominal-input query-only A2: single vs four scene evidence reads",
        nominal_input={"manifest":str(CACHE/"manifest.json"),"sha256":sha(CACHE/"manifest.json"),
          "producer_sha256":manifest["producer_sha256"],"allowed_splits":["train","tune","val"] if full else ["train","tune"],
          "input_field":"provided_status5","state_history_supervision":"original real-time definition retained"},
        scene_attention={"heads":4 if arm=="A2-MH4-NOM" else 1,"qk_dimensions_per_head":32,
          "value_dimensions_per_head":32 if arm=="A2-MH4-NOM" else 128,"output_dimensions":128,
          "sampling_grid_camera_time_metadata":"unchanged","query_as_value_residual":False,
          "initialization":"identity head transforms and image-value channel slices" if arm=="A2-MH4-NOM" else "registered A2"},
        full_fit={"enabled":full,"evaluation_is_in_fit":full,"dev_weights_or_features_imported":False,
                  "checkpoint_selection":"fixed terminal; no best-overlapping-metric selection"},
        expected_shared_base_initial_sha256=BASE_INITIAL_SHA)
    value["information_route"]["source"]="causal pose fit with frame-difference * 0.1s; same deployment producer"
    value["information_route"]["shared_consumers"]=["occupancy","lane","planning"]
    value["information_route"]["motion_state_history"]="unconditioned image path"
    value["comparison_contract"]["same_train310_tune37_split"]=not full
    value["comparison_contract"]["same_full_update_budget"]=not full and not smoke
    value["recipe"]["eval_role"]="in-fit diagnostic" if full else "DEV V0, excluded from fitting"
    value["source"]={str(p.relative_to(ROOT)):sha(p) for p in [
       Path(__file__),HERE/"a2_model.py",HERE/"nominal_data.py",HERE/"build_nominal_inputs.py",
       ROOT/"models/motiondrive_v2/scene_encoder.py",ROOT/"experiments/md_shared_dynamics_20260917/train_shared_dynamics.py",
       ROOT/"experiments/md_a2_deploy_status_20260918/nominal_status.py"]}
    return value

@contextlib.contextmanager
def runtime(arm,seed,declared,run_dir,smoke,protocol_path):
    import motiondrive_v2_data as data_api
    import train_motiondrive_v2 as trainer
    original_dataset=data_api.MotionDriveDataset
    if arm=="A2-FULL-NOM":
        def full_factory(**kwargs):
            return make_union(original_dataset,kwargs) if kwargs.get("split")=="train" else original_dataset(**kwargs)
        data_api.MotionDriveDataset=full_factory
    try:
        with legacy.patched_runtime(arm,seed,declared,run_dir,smoke):
            loader=trainer._load_initial_model_state;write_json=trainer.atomic_json
            def load_initial(model,common,experiment=None):
                report=loader(model,common,experiment)
                shared={k:v for k,v in model.state_dict().items() if not k.startswith(MH4_PREFIX)}
                measured=tensor_state_sha256(shared)
                if measured!=BASE_INITIAL_SHA:raise ValueError("Shared initial tensors differ from the successful A2 recipe")
                if arm=="A2-MH4-NOM":model.scene_encoder.evidence_attention.assert_identity()
                declared["initial_load"]["shared_base_state_sha256"]=measured
                declared["initial_load"]["all_parameters_trainable"]=all(p.requires_grad for p in model.parameters())
                protocol_path.write_text(json.dumps(declared,indent=2)+"\n")
                return report
            def atomic_json(path,payload):
                write_json(path,payload)
                if Path(path).name=="final_eval.json" and payload.get("records"):
                    extra=diagnostics(payload["records"]);extra.update(step=payload["report"]["step"],arm=arm,
                         eval_role=declared["recipe"]["eval_role"])
                    write_json(run_dir/f"diagnostics_step{extra['step']}.json",extra)
            trainer._load_initial_model_state=load_initial;trainer.atomic_json=atomic_json
            try:yield
            finally:trainer._load_initial_model_state=loader;trainer.atomic_json=write_json
    finally:data_api.MotionDriveDataset=original_dataset

def main():
    ap=argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--arm",choices=ARMS,required=True);ap.add_argument("--gpu",type=int,choices=range(4),required=True)
    ap.add_argument("--seed",type=int,choices=(1,),default=1);ap.add_argument("--run-dir",required=True)
    ap.add_argument("--smoke",action="store_true");ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--microbatch",type=int,choices=(8,16),default=8)
    args=ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=str(args.gpu):
        raise ValueError("Expose exactly the declared physical GPU")
    torch.set_num_threads(4)
    run=Path(args.run_dir).resolve()
    with configure(args.arm,args.microbatch):
        train,tune=raw_datasets(args.arm=="A2-FULL-NOM",args.seed)
        declared=build_experiment(args.arm,args.seed,train,tune,args.smoke,args.gpu)
        print("PLAN "+json.dumps({"arm":args.arm,"gpu":args.gpu,"train_rows":len(train),"eval_rows":len(tune),
          "run_dir":str(run),"recipe":declared["recipe"],"nominal_input":declared["nominal_input"]}),flush=True)
        if args.dry_run:return
        REPORT.mkdir(parents=True,exist_ok=True)
        protocol=REPORT/f"protocol_{args.arm}_s{args.seed}{'_smoke' if args.smoke else ''}_mb{args.microbatch}.json"
        protocol.write_text(json.dumps(declared,indent=2)+"\n")
        import train_motiondrive_v2 as trainer
        with runtime(args.arm,args.seed,declared,run,args.smoke,protocol):
            trainer.run_training(legacy.trainer_argv(args.seed,run,args.smoke),experiment=declared)
        (run/"experiment.json").write_text(json.dumps(declared,indent=2)+"\n")
if __name__=="__main__":main()
