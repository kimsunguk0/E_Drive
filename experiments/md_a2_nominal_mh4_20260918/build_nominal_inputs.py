"""Freeze per-split causal nominal-status inputs, independently of supervision."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path("/NHNHOME/data/sukim/adcl")
for p in (ROOT,ROOT/"scripts",ROOT/"experiments/md_a2_deploy_status_20260918"):
    sys.path.insert(0,str(p))
from nominal_status import fit_status5
from motiondrive_v2_data import full_pose_matrices
SPLIT=ROOT/"data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json"
SUP=ROOT/"data/etri/motiondrive_v2/r0reset_tplus_geometry_v2"
EGO=Path("/tmp/pm97/data/etri/ego_cache.npz")
OUT=ROOT/"data/etri/motiondrive_v2/a2_nominal_status_20260918"
PRODUCER=ROOT/"experiments/md_a2_deploy_status_20260918/nominal_status.py"
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def scene_status(task):
    scene,rows,requested_frames=task
    sources=json.loads((SUP/(scene+".json")).read_text())["sources"]
    tp=next(p for p in sources if p.endswith("/meta/timestamps.parquet"))
    pp=next(p for p in sources if p.endswith("/annotation/ego_pose.parquet"))
    ts=pd.read_parquet(tp);ep=pd.read_parquet(pp)
    joined=ts[["timestamp","frame_id"]].merge(ep,on="timestamp",how="left",validate="one_to_one").sort_values("frame_id")
    frames=joined.frame_id.to_numpy(np.int64)
    poses=full_pose_matrices(joined[["x","y","z"]].to_numpy(),joined[["roll","pitch","yaw"]].to_numpy())
    statuses=[];counts=[]
    for f in requested_frames:
        ids=np.flatnonzero((frames>=f-30)&(frames<=f))
        if not len(ids) or frames[ids[-1]]!=f:raise ValueError("Missing current pose")
        # Absolute timestamps are used ONLY to join frame IDs to poses above.
        # Fit time uses actual frame-number differences and no future sample.
        nominal_times=(frames[ids]-f).astype(np.float64)*.1
        status,info=fit_status5(poses[ids],nominal_times,len(ids)-1)
        statuses.append(status);counts.append(info["fit_count"])
    return rows,np.stack(statuses),{"scene":scene,"rows":len(rows),"fit_count_min":min(counts),"fit_count_max":max(counts),
        "pose_source":str(pp),"pose_sha256":sha(pp),"frame_mapping_source":str(tp),"frame_mapping_sha256":sha(tp)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=8);args=ap.parse_args()
    if (OUT/"manifest.json").exists():raise RuntimeError("Frozen cache already exists; validate/reuse it instead")
    start=time.monotonic()
    manifest=json.loads(SPLIT.read_text())
    with np.load(EGO,allow_pickle=False) as z:
        scenarios=z["scenarios"].astype(str);scene_names=scenarios[z["scen_idx"]]
        frames=z["frame"].copy()
    splits={name:np.flatnonzero(np.isin(scene_names,manifest["splits"][name])&(frames>=30)) for name in ["train","tune","val"]}
    expected={"train":83700,"tune":9990,"val":7830}
    if {k:len(v) for k,v in splits.items()}!=expected:raise ValueError("Unexpected split rows")
    all_rows=np.concatenate(list(splits.values()))
    if len(np.unique(all_rows))!=101520:raise ValueError("Split overlap or missing rows")
    scenes=sorted(set(scene_names[all_rows]));tasks=[]
    for scene in scenes:
        rows=all_rows[scene_names[all_rows]==scene];tasks.append((scene,rows,frames[rows]))
    if len(scenes)!=376:raise ValueError("Unexpected unique scene count")
    table=np.full((len(frames),5),np.nan,np.float32);source_index=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for count,(rows,values,info) in enumerate(pool.map(scene_status,tasks),1):
            table[rows]=values;source_index.append(info)
            if count%40==0 or count==len(tasks):
                print(json.dumps({"scenes_done":count,"total_scenes":len(tasks),"elapsed_s":time.monotonic()-start}),flush=True)
    if not np.isfinite(table[all_rows]).all():raise ValueError("Missing/nonfinite nominal input")
    OUT.mkdir(parents=True,exist_ok=True)
    report={"schema_version":1,"kind":"deterministic raw-pose INPUT cache, not learned features or supervision",
        "status_policy":"frame_difference_times_0.1_seconds","fields":["vx","vy","ax","ay","yaw_rate"],
        "split_manifest_sha256":sha(SPLIT),"supervision_manifest_sha256":sha(SUP/"supervision_manifest.json"),
        "ego_cache_sha256":sha(EGO),"producer":str(PRODUCER),"producer_sha256":sha(PRODUCER),
        "builder_sha256":sha(__file__),"supervision_unchanged":True,"splits":{},"source_index":source_index,
        "future_policy":"Each row selects frame-30..frame; fit uses last <=1.001 nominal seconds",
        "timestamp_policy":"Timestamp parquet is used only to join frame IDs to original poses, never as fit time",
        "DEV_policy":"DEV reads train/tune input caches only; FULL additionally reads val. No fitted statistics or model weights are shared."}
    for name,rows in splits.items():
        path=OUT/(name+".npz")
        if path.exists():raise RuntimeError("Cache file already exists")
        np.savez_compressed(path,row=rows,frame=frames[rows],status5=table[rows])
        report["splits"][name]={"rows":len(rows),"rows_sha256":hashlib.sha256(np.asarray(rows,dtype="<i8").tobytes()).hexdigest(),
             "path":str(path),"sha256":sha(path),"scenes":len(set(scene_names[rows]))}
    report["elapsed_s"]=time.monotonic()-start
    tmp=OUT/"manifest.json.tmp";tmp.write_text(json.dumps(report,indent=2)+"\n");os.replace(tmp,OUT/"manifest.json")
    print("COMPLETE "+json.dumps({k:v for k,v in report.items() if k!="source_index"}),flush=True)
if __name__=="__main__":main()
