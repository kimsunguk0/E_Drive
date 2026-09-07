#!/usr/bin/env python3
"""P4 fresh public baseline: isolated GPU4/5 parent and same-PID trainer bridge.

Reuses the tested own-child supervisor, NOT the old physical0--3 launcher.
The original trainer/model are unchanged. LAST checkpoints advance stages;
canary is a plumbing/finite-backward check, not an accuracy result.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_motiondrive_v2_query_trial import supervise,require,sha256,valid_sha256
from scripts.launch_motiondrive_v2_trials import inspect_repository

SCRIPT="scripts/run_motiondrive_v2_fresh_trial.py"
TRAINER="scripts/train_motiondrive_v2.py"
INITIALIZER="scripts/initialize_motiondrive_v2_public.py"
PUBLIC_INIT="work_dirs/motiondrive_v2/p4_public_init_s0.pth"
PUBLIC_SHA="4096396018c0cf59fbe0eb1afe6e269f4676b34460bed5eedde5d7680d58bb4e"
DATA_FILES={
 "data/etri/motiondrive_v2/grouped_split_rawtime.json":"f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936",
 "data/etri/motiondrive_v2/train_tune_geometry_v2/supervision_manifest.json":"ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93",
 "data/etri/motiondrive_v2/train_tune_geometry_v2/calibration.npz":"8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961"}
STAGES={
 "canary":{"phase":"joint","steps":2,"eval_every":2,"save_every":2,"warmup":2,"train_n":16,"eval_n":8},
 "pretrain":{"phase":"pretrain","steps":2000,"eval_every":250,"save_every":250,"warmup":200,"train_n":54810,"eval_n":1998},
 "joint":{"phase":"joint","steps":6000,"eval_every":250,"save_every":250,"warmup":200,"train_n":54810,"eval_n":1998}}

def snapshot(expected):
    state=inspect_repository(ROOT)
    require(state["actual_git_sha"]==expected and not state["tracked_dirty"],"Pinned clean P4 source required")
    from scripts.evaluate_motiondrive_v2_planning import source_manifest
    base=source_manifest()
    require(base["git_sha"]==expected and not base["tracked_changes"],"Original runtime source mismatch")
    files=dict(base["file_sha256"])
    for name in (SCRIPT,INITIALIZER,"scripts/run_motiondrive_v2_query_trial.py",
                 "scripts/train_motiondrive_v2_query_adapter.py","models/motiondrive_v2_query_adapter.py",
                 "scripts/evaluate_motiondrive_v2_shared.py","scripts/supervise_motiondrive_v2_job.py",
                 "scripts/launch_motiondrive_v2_trials.py"):
        files[name]=sha256(ROOT/name)
    return {"git_sha":expected,"file_sha256":files}

def validate_sources_and_data(req):
    require(snapshot(req["expected_commit"])==req["sources"],"Source changed during P4 run")
    for name,digest in req["data_sha256"].items():
        require(sha256(ROOT/name)==digest,"C1 data changed")
    require(sha256(req["initializer"])==req["initializer_sha256"],"Initial checkpoint changed")

def initializer_path(stage,seed):
    return ROOT/(f"work_dirs/motiondrive_v2/p4_fresh_pretrain_s{seed}/last.pth" if stage=="joint" else PUBLIC_INIT)

def verify_initial_payload(payload,stage):
    require(isinstance(payload,dict),"Trainer checkpoint mapping required")
    m=payload.get("manifest",{})
    c=m.get("model_config",{})
    require(m.get("split_sha256")==next(iter(DATA_FILES.values())),"Initializer split mismatch")
    require(c.get("backbone_arch")=="resnet50" and c.get("motion_input_mode")=="low_feature"
            and list(c.get("plan_output_scale",[]))==[10.,5.] and c.get("goal_on") is False
            and c.get("state_on") is False,"G0S0 low_feature public-stage initializer required")
    if stage in ("canary","pretrain"):
        require(payload.get("step")==0 and m.get("etri_optimizer_steps")==0 and m.get("seed")==0
                and m.get("public_checkpoint_sha256")==PUBLIC_SHA and m.get("pretrained_sha256")==PUBLIC_SHA,
                "Fresh public initializer must contain no ETRI optimizer updates")
        require(not payload.get("optimizer",{}),"Fresh initializer may not contain optimizer moments")
    else:
        require(payload.get("step")==2000 and m.get("arguments",{}).get("phase")=="pretrain"
                and m.get("arguments",{}).get("time_input")=="nominal"
                and m.get("supervision_manifest_sha256")==DATA_FILES[
                    "data/etri/motiondrive_v2/train_tune_geometry_v2/supervision_manifest.json"],
                "Joint must start from the own corrected-geometry pretrain LAST2000")

def prerequisite(stage,seed,init_sha,sources):
    if stage=="canary":
        return None
    name="p4_fresh_canary_s0" if stage=="pretrain" else f"p4_fresh_pretrain_s{seed}"
    path=ROOT/f"logs/motiondrive_v2/{name}.supervisor.json"
    s=json.loads(path.read_text())
    require(s.get("actual_returncode")==0 and s.get("supervisor_exit_code")==0
            and s.get("outcome")=="completed_cleanly" and s.get("pressure_event") is None,
            "Prerequisite needs verified clean OS exit")
    require(all(type(s.get(k)) is int and s[k]>0 and not Path(f"/proc/{s[k]}").exists()
                for k in ("parent_pid","child_pid")),"Prerequisite processes must be absent")
    req=s.get("request",{})
    require(req.get("stage")==("canary" if stage=="pretrain" else "pretrain"),"Wrong prerequisite stage")
    prior_seed=0 if stage=="pretrain" else seed
    prior_run=ROOT/"work_dirs/motiondrive_v2"/name
    require(req.get("seed")==prior_seed and req.get("run_dir")==str(prior_run)
            and req.get("record")==str(path) and req.get("manifest_path")==str(prior_run/"manifest.json")
            and req.get("initializer")==str(ROOT/PUBLIC_INIT),"Prerequisite seed/run/initializer identity mismatch")
    if stage=="pretrain":
        require(req.get("initializer_sha256")==init_sha,"Canary tested a different public initializer")
    else:
        require(s.get("completion_evidence",{}).get("last_sha256")==init_sha,"Joint initializer is not audited LAST2000")
    require(req.get("sources",{}).get("file_sha256")==sources["file_sha256"],
            "Runtime changed since prerequisite; repeat the canary rather than assuming parity")
    evidence=s.get("completion_evidence",{})
    for name,key in (("manifest_path","manifest_sha256"),("child_receipt","receipt_sha256")):
        require(sha256(req[name])==evidence[key],"Prerequisite evidence changed")
    previous_manifest=json.loads(Path(req["manifest_path"]).read_text())
    require(previous_manifest.get("pid")==s["child_pid"] and previous_manifest.get("status")=="completed"
            and previous_manifest.get("step")== (2 if stage=="pretrain" else 2000),"Prerequisite manifest/PID/step mismatch")
    prior_last=prior_run/"last.pth"
    require(prior_last.is_file() and not prior_last.is_symlink(),"Prerequisite LAST missing")
    prior_last_sha=sha256(prior_last)
    audited=evidence.get("checkpoint_validation",{})
    require(evidence.get("last_sha256")==prior_last_sha
            and audited.get("last_sha256")==prior_last_sha
            and audited.get("full_model_strict_cpu_load") is True
            and audited.get("optimizer_tensors_finite") is True
            and audited.get("optimizer_steps")==previous_manifest["step"]
            and audited.get("model_forward_performed") is False,
            "Prerequisite LAST lacks matching strict CPU validation")
    return {"path":str(path),"sha256":sha256(path),"actual_returncode":0}

def trainer_arguments(stage,seed,run,init):
    c=STAGES[stage]
    args=[str(ROOT/TRAINER),"--data-root",str(ROOT),"--split-manifest",str(ROOT/next(iter(DATA_FILES))),
        "--supervision-root",str(ROOT/"data/etri/motiondrive_v2/train_tune_geometry_v2"),
        "--run-dir",str(run),"--init",str(init),"--phase",c["phase"],"--gpu","0",
        "--seed",str(seed),"--goal-on",str(int(c["phase"]=="joint")),"--state-on",str(int(c["phase"]=="joint")),
        "--arch","resnet50","--motion-input-mode","low_feature","--time-input","nominal",
        "--precision","bf16","--bn-policy","fixed","--batch","16","--microbatch","2",
        "--eval-batch","4","--workers","4","--train-stride","1","--eval-stride","5","--eval-split","tune",
        "--steps",str(c["steps"]),"--eval-every",str(c["eval_every"]),"--save-every",str(c["save_every"]),
        "--warmup",str(c["warmup"]),"--lr","0.0001","--backbone-lr","0.00001",
        "--weight-decay","0.01","--grad-clip","5","--log-every","10",
        "--alpha-occ","0.2","--alpha-lane","0.2","--alpha-motion","0.2","--uncertainty","1",
        "--cuda-memory-limit-mib","12000","--cuda-min-free-mib","8192"]
    if stage=="canary":
        args.extend(["--max-train-samples","16","--max-eval-samples","8"])
    return args

def prepare(stage,seed,gpu,expected,init_sha,child=False):
    require(stage in STAGES and type(seed) is int and seed in (0,1),"Fixed stage/seed required")
    require(stage!="canary" or seed==0,"Only one plumbing canary, seed0")
    require(type(gpu) is int and gpu in (4,5),"Only authorized physical GPU4/5")
    require(valid_sha256(init_sha),"Explicit initializer SHA256 required")
    sources=snapshot(expected)
    for name,digest in DATA_FILES.items():
        require(sha256(ROOT/name)==digest,"C1 data provenance mismatch")
    init=initializer_path(stage,seed)
    require(init.is_file() and not init.is_symlink() and sha256(init)==init_sha,"Initializer changed/missing")
    name=f"p4_fresh_{stage}_s{seed}"
    run=ROOT/"work_dirs/motiondrive_v2"/name
    record=ROOT/"logs/motiondrive_v2"/(name+".supervisor.json")
    log=ROOT/"logs/motiondrive_v2"/(name+".log")
    receipt=ROOT/"logs/motiondrive_v2"/(name+".child.json")
    for path in ((run,receipt) if child else (run,record,log,receipt)):
        require(not os.path.lexists(path) and path.parent.is_dir() and path.resolve().is_relative_to(ROOT),
                "Existing or escaping P4 output is forbidden")
    prior=prerequisite(stage,seed,init_sha,sources)
    command=[sys.executable,str(ROOT/SCRIPT),"--child","--stage",stage,"--seed",str(seed),
        "--physical-gpu",str(gpu),"--expected-git-sha",expected,"--expected-init-sha256",init_sha]
    return {"root":str(ROOT),"stage":stage,"seed":seed,"physical_gpu":gpu,"expected_commit":expected,
        "sources":sources,"data_sha256":dict(DATA_FILES),"initializer":str(init),"initializer_sha256":init_sha,
        "run_dir":str(run),"manifest_path":str(run/"manifest.json"),"record":str(record),"log":str(log),
        "child_receipt":str(receipt),"command":command,"trainer_arguments":trainer_arguments(stage,seed,run,init),
        "prerequisite":prior,"policy":"fresh practical baseline, not matched geometry-history causal ablation"}

def child_run(req):
    import torch
    from scripts.train_motiondrive_v2_query_adapter import validate_cuda_namespace
    from scripts.motiondrive_v2_training import tensor_state_sha256
    import time
    parent=None
    for _ in range(50):
        parent=json.loads(Path(req["record"]).read_text())
        if parent.get("child_pid")==os.getpid():
            break
        time.sleep(.1)
    require(parent.get("child_pid")==os.getpid() and parent.get("parent_pid")==os.getppid()
            and parent.get("status")=="running", "Child must belong to its actual live parent")
    recorded=parent["request"]
    require(all(recorded.get(k)==v for k,v in req.items()),"Child request differs from owned parent request")
    namespace=validate_cuda_namespace(torch.device("cuda:0"))
    require(namespace["physical_gpu"]==req["physical_gpu"] and namespace["observed_uuid"]==recorded["gpu_uuid"],
            "Wrong actual CUDA device")
    payload=torch.load(req["initializer"],map_location="cpu",weights_only=False)
    verify_initial_payload(payload,req["stage"])
    initial_tensor_sha=tensor_state_sha256(payload["model"])
    del payload
    receipt={"pid":os.getpid(),"parent_pid":os.getppid(),"device_mapping":namespace,
        "source":req["sources"],"initial_checkpoint_sha256":req["initializer_sha256"],
        "initial_model_state_sha256":initial_tensor_sha,"trainer_arguments":req["trainer_arguments"],
        "logical_gpu":0,"optimizer_resume":False,"full_forward_history_cost_required":True}
    with Path(req["child_receipt"]).open("x") as f:
        json.dump(receipt,f,indent=2,allow_nan=False)
        f.write("\n")
    validate_sources_and_data(req)
    sys.argv=req["trainer_arguments"]
    runpy.run_path(str(ROOT/TRAINER),run_name="__main__")
    # CUDA native teardown can still fail AFTER this; only the parent owns OS status.
    validate_sources_and_data(req)

def validate_last_checkpoint(req,manifest,last):
    """Read actual LAST tensors on CPU after OS exit, not just terminal metadata."""
    import torch
    from models.motiondrive_v2 import MotionDriveV2
    from scripts.export_motiondrive_v2_inference import validate_complete_config
    from scripts.initialize_motiondrive_v2_public import strict_cpu_state
    from scripts.motiondrive_v2_training import tensor_state_sha256
    def same(a,b):
        return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
    before=sha256(last)
    initial_before=sha256(req["initializer"])
    require(initial_before==req["initializer_sha256"],"Initializer changed before CPU validation")
    saved=torch.load(last,map_location="cpu",weights_only=False)
    require(saved.get("step")==STAGES[req["stage"]]["steps"],"LAST optimizer step mismatch")
    embedded=saved.get("manifest",{})
    # The unchanged original trainer saves LAST before updating terminal status.
    require(embedded.get("status") in ("running","completed"),"Unexpected LAST embedded status")
    terminal_fields={"status","step","best_metric","cuda_memory","elapsed_seconds","nonfinite_count"}
    embedded_lineage={key:value for key,value in embedded.items() if key not in terminal_fields}
    sidecar_lineage={key:value for key,value in manifest.items() if key not in terminal_fields}
    require(same(embedded_lineage,sidecar_lineage),"LAST embedded manifest lineage mismatch")
    model=MotionDriveV2(validate_complete_config(manifest["model_config"])).cpu()
    strict_cpu_state(model,saved.get("model"))
    initial=torch.load(req["initializer"],map_location="cpu",weights_only=False)
    bn_names=[name for name in saved["model"] if name.endswith(("running_mean","running_var","num_batches_tracked"))]
    require(bool(bn_names),"Expected fixed BN buffers")
    bn={name:saved["model"][name] for name in bn_names}
    original_bn={name:initial["model"][name] for name in bn_names}
    require(tensor_state_sha256(bn)==tensor_state_sha256(original_bn),"Fixed BN running buffers changed")
    optimizer_payload=saved.get("optimizer")
    require(isinstance(optimizer_payload,dict),"LAST optimizer mapping missing")
    optimizer=optimizer_payload.get("state",{})
    require(bool(optimizer),"LAST optimizer state missing")
    for values in optimizer.values():
        require(isinstance(values,dict) and float(values.get("step",-1))==STAGES[req["stage"]]["steps"]
                and all(not isinstance(t,torch.Tensor) or bool(torch.isfinite(t).all()) for t in values.values()),
                "Nonfinite optimizer state or wrong update count")
    require(sha256(last)==before and sha256(req["initializer"])==initial_before,
            "Checkpoint changed during CPU validation")
    return {"last_sha256":before,"full_model_strict_cpu_load":True,"optimizer_tensors_finite":True,
        "optimizer_steps":saved["step"],"fixed_bn_state_sha256":tensor_state_sha256(bn),
        "model_state_sha256":tensor_state_sha256(saved["model"]),"model_forward_performed":False}

def validate_completion(req,pid):
    validate_sources_and_data(req)
    path=Path(req["manifest_path"])
    require(path.is_file() and not path.is_symlink(),"Completed trainer manifest required")
    m=json.loads(path.read_text())
    c=STAGES[req["stage"]]
    require(m.get("pid")==pid and m.get("status")=="completed" and m.get("step")==c["steps"]
            and m.get("nonfinite_count")==0 and m.get("git_sha")==req["expected_commit"],"Trainer did not complete fixed P4 steps")
    a=m.get("arguments",{})
    expected={"phase":c["phase"],"seed":req["seed"],"gpu":0,"cpu":False,"batch":16,"microbatch":2,
        "eval_batch":4,"bn_policy":"fixed","time_input":"nominal","precision":"bf16","init":req["initializer"],
        "run_dir":req["run_dir"],"steps":c["steps"],"cuda_memory_limit_mib":12000,"cuda_min_free_mib":8192,
        "resume":None,"pretrained":None,"allow_unpretrained":False,"eval_split":"tune"}
    require(all(a.get(k)==v for k,v in expected.items()),"Actual P4 trainer arguments differ")
    require(m.get("data_counts")=={"train":c["train_n"],"eval":c["eval_n"]},"P4 sample counts differ")
    cfg=m.get("model_config",{})
    require(cfg.get("backbone_arch")=="resnet50" and cfg.get("motion_input_mode")=="low_feature"
        and cfg.get("goal_on") is (c["phase"]=="joint") and cfg.get("state_on") is (c["phase"]=="joint")
        and list(cfg.get("plan_output_scale",[]))==[10.,5.],"P4 complete model configuration mismatch")
    receipt_path=Path(req["child_receipt"])
    r=json.loads(receipt_path.read_text())
    require(r.get("pid")==pid and r.get("parent_pid")==os.getpid()
        and r.get("initial_checkpoint_sha256")==req["initializer_sha256"] and r.get("source")==req["sources"]
        and r.get("initial_model_state_sha256")==m.get("initial_model_state_sha256"),"Initial tensor/PID/source lineage differs")
    d=r.get("device_mapping",{})
    require(d.get("physical_gpu")==req["physical_gpu"] and d.get("logical_device")=="cuda:0"
        and d.get("observed_uuid")==req["gpu_uuid"] and d.get("cuda_visible_devices")==req["gpu_uuid"],"Actual CUDA namespace mismatch")
    require(m.get("load_report",{}).get("common_checkpoint_sha256")==req["initializer_sha256"],"Trainer loaded wrong initializer")
    require(m.get("supervision_manifest_sha256")==DATA_FILES[
        "data/etri/motiondrive_v2/train_tune_geometry_v2/supervision_manifest.json"],"Actual supervision edition differs")
    last=Path(req["run_dir"])/"last.pth"
    require(last.is_file() and not last.is_symlink(),"LAST checkpoint missing")
    checkpoint_evidence=validate_last_checkpoint(req,m,last)
    validate_sources_and_data(req)
    if req["prerequisite"]:
        require(sha256(req["prerequisite"]["path"])==req["prerequisite"]["sha256"],"Prerequisite record changed")
    return {"manifest_sha256":sha256(path),"last_sha256":sha256(last),"receipt_sha256":sha256(receipt_path),
            "checkpoint_validation":checkpoint_evidence,
            "initial_model_state_sha256":m["initial_model_state_sha256"],"steps":c["steps"],
            "data_counts":m["data_counts"],"is_accuracy_claim":False}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--stage",choices=tuple(STAGES),required=True)
    p.add_argument("--seed",type=int,choices=(0,1),required=True)
    p.add_argument("--physical-gpu",type=int,choices=(4,5),required=True)
    p.add_argument("--expected-git-sha",required=True)
    p.add_argument("--expected-init-sha256",required=True)
    p.add_argument("--child",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    req=prepare(args.stage,args.seed,args.physical_gpu,args.expected_git_sha,args.expected_init_sha256,args.child)
    if args.child:
        child_run(req)
        return 0
    result=supervise(req,validator=validate_completion)
    print(json.dumps({"outcome":result["outcome"],"actual_returncode":result["actual_returncode"],"record":req["record"]}))
    return result["supervisor_exit_code"]

if __name__=="__main__":
    raise SystemExit(main())
