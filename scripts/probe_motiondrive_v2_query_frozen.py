#!/usr/bin/env python3
"""Establish frozen P3 initializer using legacy/control/state full tune parity."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from models.motiondrive_v2_query_adapter import migrate_legacy_checkpoint
from scripts.motiondrive_v2_data import MotionDriveDataset
from scripts.motiondrive_v2_training import model_inputs,to_device,weighted_d3,tensor_state_sha256
from scripts.train_motiondrive_v2 import configure_cuda_memory,check_cuda_headroom,cuda_memory_snapshot,sha256
from scripts.train_motiondrive_v2_query_adapter import (validate_data_paths,validate_initial_payload,
    validate_cuda_namespace,configure_numerics,configure_planner_training,frozen_state_sha)
from scripts.run_motiondrive_v2_query_trial import INIT,INIT_SHA,source_snapshot,require
from scripts.probe_motiondrive_v2_query_initial import OUTPUTS,exact_tensor_equal

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--expected-git-sha",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    require(not os.path.lexists(args.out),"Existing report forbidden")
    out=Path(args.out).resolve()
    require(out.is_relative_to((ROOT/"reports").resolve()) and out.parent.is_dir(),"New reports path required")
    source=source_snapshot(ROOT,args.expected_git_sha)
    own_sha=sha256(__file__)
    a=argparse.Namespace(split_manifest=str(ROOT/"data/etri/motiondrive_v2/grouped_split_rawtime.json"),
        supervision_root=str(ROOT/"data/etri/motiondrive_v2/train_tune_geometry_v2"),
        init=str(ROOT/INIT),expected_init_sha256=INIT_SHA)
    data=validate_data_paths(a)
    payload=torch.load(a.init,map_location="cpu",weights_only=False)
    validate_initial_payload(payload)
    device=torch.device("cuda:0")
    namespace=validate_cuda_namespace(device)
    memory=configure_cuda_memory(device,12000,8192)
    numerics=configure_numerics()
    legacy=MotionDriveV2(MotionDriveV2Config(**payload["manifest"]["model_config"]))
    legacy.load_state_dict(payload["model"],strict=True)
    for name,param in legacy.named_parameters():
        param.requires_grad_(name.startswith("planner."))
    legacy.to(device).eval()
    query,migration=migrate_legacy_checkpoint(a.init,INIT_SHA,adapter_seed=0,adapter_on=False,device="cpu")
    configure_planner_training(query)
    query.to(device).eval()
    frozen_sha=frozen_state_sha(query)
    initial_sha=tensor_state_sha256(query.state_dict())
    dataset=MotionDriveDataset(data_root=str(ROOT),split_manifest=a.split_manifest,
        supervision_root=a.supervision_root,split="tune",min_frame=30,frame_stride=5,augment=False,seed=0)
    require(len(dataset)==1998,"Expected tune1998")
    loader=DataLoader(dataset,batch_size=4,shuffle=False,num_workers=4,pin_memory=True)
    failed={}
    for arm in ("control","state"):
        path=ROOT/f"work_dirs/motiondrive_v2/p3_query_{arm}_s0/eval_step0000.json"
        failed[arm]={"path":str(path),"sha256":sha256(path),"records":json.loads(path.read_text())["records"]}
        require(len(failed[arm]["records"])==1998,"Original failed step-zero records required")
    records=[]
    with torch.inference_mode():
        for batch_no,raw in enumerate(loader):
            check_cuda_headroom(device,8192)
            b=to_device(raw,device)
            inputs=model_inputs(b,time_input="nominal")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                reference=legacy(**inputs)
                query.query_adapter_on=False
                control=query(**inputs)
                query.query_adapter_on=True
                state=query(**inputs)
            for name in OUTPUTS:
                require(torch.isfinite(reference[name]).all().item(),"Nonfinite reference")
                require(exact_tensor_equal(reference[name],control[name]) and exact_tensor_equal(control[name],state[name]),
                    f"Frozen initializer output differs: batch{batch_no}/{name}")
            scores=weighted_d3(reference["plan_abs"],b["gt_plan"]).cpu().tolist()
            for i,score in enumerate(scores):
                row={"scenario":raw["scenario"][i],"session":raw["session_id"][i],
                    "frame":int(raw["frame"][i]),"d3":float(score),"proxy":float(raw["proxy_weight"][i])}
                for arm in failed:
                    require(row==failed[arm]["records"][len(records)],"Failed-arm row-level initializer mismatch")
                records.append(row)
            if batch_no%100==0:
                print(json.dumps({"batches":batch_no+1,"frames":len(records),"all_output_bytes_equal":True}),flush=True)
    require(len(records)==1998 and len({r["session"] for r in records})==11,"Final row/session count mismatch")
    require(frozen_state_sha(query)==frozen_sha and tensor_state_sha256(query.state_dict())==initial_sha,
        "Probe mutated model tensors")
    require(validate_data_paths(a)==data and source_snapshot(ROOT,args.expected_git_sha)==source
        and sha256(__file__)==own_sha and all(sha256(f["path"])==f["sha256"] for f in failed.values()),"Inputs changed")
    report={"status":"frozen_legacy_control_state_full_tune_bitwise_pass","pid":os.getpid(),
        "source":source,"probe_source_sha256":own_sha,"data":data,"device_mapping":namespace,
        "memory_policy":memory,"memory":cuda_memory_snapshot(device),"numerics":numerics,
        "initial_model_sha":initial_sha,"frozen_sha256":frozen_sha,"migration":migration,
        "official_d3":float(np.mean([r["d3"] for r in records])),"n":1998,"n_sessions":11,
        "full_forward_count":1500,"optimizer_steps":0,"outputs_bitwise_equal":list(OUTPUTS),
        "comparison":"exact dtype/shape/contiguous bytes, all batches/all frames",
        "original_failed_evaluations":{k:{a:b for a,b in f.items() if a!="records"} for k,f in failed.items()},
        "records":records,"final_val_accessed":False}
    with out.open("x") as f:
        json.dump(report,f,ensure_ascii=False,allow_nan=False,indent=2)
        f.write("\n")
    print(json.dumps({"status":report["status"],"official_d3":report["official_d3"],"report":str(out)}))
if __name__=="__main__":
    main()
