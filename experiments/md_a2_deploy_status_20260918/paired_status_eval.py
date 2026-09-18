"""Paired V0 evaluation: fixed A2 checkpoint, only provided status changes."""
from __future__ import annotations
import collections,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset,DataLoader

ROOT=Path("/NHNHOME/data/sukim/adcl")
for directory in (ROOT,ROOT/"scripts",ROOT/"experiments/md_r0_reset_20260914",
                  ROOT/"experiments/md_shared_dynamics_20260917"):
    sys.path.insert(0,str(directory))
from factorized_model import SharedDynamicsMotionDriveV2
from models.motiondrive_v2 import MotionDriveV2Config
from train_shared_dynamics import CausalStatusDataset,raw_datasets
from motiondrive_v2_data import full_pose_matrices
from motiondrive_v2_training import model_inputs,to_device
import matching_resolution as mr
from nominal_status import fit_status5,status5_from_clip_records

OUTPUT=ROOT/"reports/md_a2_deploy_status_20260918"
RUN=ROOT/"work_dirs/md_shared_dynamics_20260917/A2-DIRECT-s1"
SUP=ROOT/"data/etri/motiondrive_v2/r0reset_tplus_geometry_v2"
WEIGHTS=np.array([11,11,5,5,2,2],np.float64)/36

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b""):h.update(part)
    return h.hexdigest()

def prepare_status(records):
    real=np.zeros((len(records),5),np.float32);nominal=real.copy()
    counts=np.zeros((len(records),2),np.int64);groups=collections.defaultdict(list)
    for index,rec in enumerate(records):groups[rec["scenario"]].append(index)
    for scene,indices in groups.items():
        sources=json.loads((SUP/(scene+".json")).read_text())["sources"]
        ts=pd.read_parquet(next(p for p in sources if p.endswith("/meta/timestamps.parquet")))
        ep=pd.read_parquet(next(p for p in sources if p.endswith("/annotation/ego_pose.parquet")))
        joined=ts[["timestamp","frame_id"]].merge(ep,on="timestamp",how="left",validate="one_to_one").sort_values("frame_id")
        frames=joined.frame_id.to_numpy(np.int64);timestamps=joined.timestamp.to_numpy(np.float64)
        times=(timestamps-timestamps[0])/1000.
        xyz=joined[["x","y","z"]].to_numpy(np.float64);rpy=joined[["roll","pitch","yaw"]].to_numpy(np.float64)
        poses=full_pose_matrices(xyz,rpy);lookup={int(f):i for i,f in enumerate(frames)}
        for index in indices:
            frame=int(records[index]["frame"]);now=lookup[frame]
            selected=np.flatnonzero((frames>=frame-30)&(frames<=frame))
            real[index],info=fit_status5(poses[selected],times[selected],len(selected)-1)
            clip=[dict(frame=int(frames[k]-frame),**dict(zip(("x","y","z"),xyz[k])),**dict(zip(("roll","pitch","yaw"),rpy[k]))) for k in selected]
            nominal[index],info_nom=status5_from_clip_records(clip)
            counts[index]=[info["fit_count"],info_nom["fit_count"]]
    expected=np.asarray([r["gt_state"][:5] for r in records],np.float32)
    np.testing.assert_array_equal(real,expected)
    return real,nominal,counts

class WithNominal(Dataset):
    def __init__(self,base,nominal):self.base,self.nominal=base,nominal
    def __len__(self):return len(self.base)
    def __getitem__(self,index):
        item=self.base[index]
        item["nominal_status5"]=torch.from_numpy(self.nominal[index].copy())
        return item

def prefix(pred,target):
    distance=np.linalg.norm(pred-target,axis=-1)
    return distance@WEIGHTS

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    records=json.loads((RUN/"final_eval.json").read_text())["records"]
    real,nominal,counts=prepare_status(records)
    test_root=Path("/tmp/etri_test")
    clips=sorted(p for p in test_root.iterdir() if p.is_dir() and (p/"ego_pose.parquet").is_file())
    schemas=collections.Counter();patterns=collections.Counter();max_goal_change=0.
    for clip in clips:
        patterns[tuple(sorted(p.name for p in clip.iterdir() if p.is_file()))]+=1
        table=pq.read_table(clip/"ego_pose.parquet")
        schemas[tuple(table.column_names)]+=1
        rows=table.to_pylist()
        first,_=status5_from_clip_records(rows)
        poisoned=[dict(row) for row in rows]
        for row in poisoned:
            if row["frame"]>0:
                for name in ("x","y","z","roll","pitch","yaw"):row[name]=float("nan")
        second,_=status5_from_clip_records(poisoned)
        max_goal_change=max(max_goal_change,float(np.max(abs(first-second))))
    delta=(nominal-real).astype(np.float64)
    status_summary={"order":["vx","vy","ax","ay","yaw_rate"],
      "nominal_minus_real_signed_mean":delta.mean(0).tolist(),
      "mae":abs(delta).mean(0).tolist(),"p95_abs":np.percentile(abs(delta),95,axis=0).tolist(),
      "max_abs":abs(delta).max(0).tolist(),
      "real_replay_equals_cached_state_target":True,
      "fit_count_pairs":{str(tuple(k)):int(n) for k,n in zip(*np.unique(counts,axis=0,return_counts=True))},
      "test_clips":len(clips),"test_pose_schemas":[{"columns":k,"n":v} for k,v in schemas.items()],
      "test_top_level_files":[{"files":k,"n":v} for k,v in patterns.items()],
      "future_goal_poison_status_max_change":max_goal_change}
    print("STATUS "+json.dumps(status_summary),flush=True)
    torch.set_num_threads(4);torch.manual_seed(20260918)
    checkpoint=RUN/"ckpt_step20554.pth"
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model=SharedDynamicsMotionDriveV2(MotionDriveV2Config(**payload["manifest"]["model_config"]),arm="A2-DIRECT")
    model.load_state_dict(payload["model"],strict=True);model.eval().cuda()
    del payload
    _,raw=raw_datasets(1)
    assert raw.rows.tolist()==[int(r["row"]) for r in records]
    dataset=WithNominal(CausalStatusDataset(mr.MotionCanvasDataset(raw,"native")),nominal)
    loader=DataLoader(dataset,batch_size=8,shuffle=False,num_workers=4,pin_memory=True,prefetch_factor=2)
    plans={"real":[],"nominal":[]};targets=[];invariance={k:0. for k in ("state_hat","history_hat","motion_features")}
    with torch.inference_mode():
        for index,batch in enumerate(loader):
            batch=to_device(batch,torch.device("cuda:0"))
            inputs=mr.model_inputs_with_canvas(model_inputs,batch,time_input="nominal")
            outputs={}
            for condition,key in (("real","provided_status5"),("nominal","nominal_status5")):
                current=dict(inputs,provided_status5=batch[key])
                with torch.autocast("cuda",dtype=torch.bfloat16):outputs[condition]=model(**current)
                plans[condition].append(outputs[condition]["plan_abs"].float().cpu().numpy())
            for key in invariance:
                invariance[key]=max(invariance[key],float((outputs["real"][key].float()-outputs["nominal"][key].float()).abs().max()))
            targets.append(batch["gt_plan"].float().cpu().numpy())
            if index%25==0:
                print(json.dumps({"batches":index+1,"elapsed_s":time.monotonic()-started}),flush=True)
    plans={k:np.concatenate(v).astype(np.float64) for k,v in plans.items()}
    target=np.concatenate(targets).astype(np.float64)
    np.testing.assert_array_equal(target,np.asarray([r["gt_abs_xy"] for r in records],np.float64))
    stored=np.asarray([r["pred_abs_xy"] for r in records],np.float64)
    scores={k:prefix(v,target) for k,v in plans.items()}
    plan_change=prefix(plans["nominal"],plans["real"])
    score_delta=scores["nominal"]-scores["real"]
    assert np.max(abs(score_delta)-plan_change)<1e-10
    buckets=np.array([r["bucket"] for r in records])
    result={"schema_version":1,"kind":"fixed-checkpoint paired status input evaluation",
      "training_performed":False,"physical_gpu":0,"rows":len(records),
      "checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint),
      "evaluation_reference_sha256":sha(RUN/"final_eval.json"),
      "producer_sha256":sha(Path(__file__).with_name("nominal_status.py")),
      "script_sha256":sha(__file__),"status":status_summary,
      "supervision_policy":"original state/history targets retained; only provided_status5 changes",
      "other_model_inputs":"identical; time_offsets remains nominal for both conditions",
      "original_plan_max_abs_difference_from_saved":float(abs(plans["real"]-stored).max()),
      "motion_state_history_max_abs_difference":invariance,
      "conditions":{},
      "nominal_minus_real_PREFIX":float(score_delta.mean()),
      "nominal_vs_real_plan_PREFIX":float(plan_change.mean()),
      "plan_change_per_row_p50_p95_max":np.percentile(plan_change,[50,95,100]).tolist(),
      "per_row_PREFIX_delta_p0_p5_p50_p95_p100":np.percentile(score_delta,[0,5,50,95,100]).tolist(),
      "triangle_bound_max_violation":float(np.max(abs(score_delta)-plan_change)),
      "elapsed_s":time.monotonic()-started,
      "limitations":["The paired eval uses cached images/geometry with a deployable status producer; it is not a full raw-image submission adapter parity test.",
                     "This evaluates a real-timestamp-trained checkpoint under nominal inputs; it does not measure nominal-input retraining or MH4."]}
    for condition,pred in plans.items():
        d=np.linalg.norm(pred-target,axis=-1)
        result["conditions"][condition]={"PREFIX":float(scores[condition].mean()),
          "L2_1s":float(d[:,:2].mean()),"L2_2s":float(d[:,:4].mean()),"L2_3s":float(d.mean()),
          "buckets":{b:{"n":int((buckets==b).sum()),"PREFIX":float(scores[condition][buckets==b].mean())} for b in sorted(set(buckets))}}
    (OUTPUT/"paired_status_eval.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(OUTPUT/"paired_status_eval.npz",row=raw.rows,real_status=real,nominal_status=nominal,
       gt=target,plan_real=plans["real"],plan_nominal=plans["nominal"],delta_prefix=score_delta,
       plan_change_prefix=plan_change,fit_counts=counts)
    print("RESULT "+json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":main()
