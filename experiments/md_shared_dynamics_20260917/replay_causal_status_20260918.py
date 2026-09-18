
import json, pathlib, sys
import numpy as np, pandas as pd
r=pathlib.Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0,str(r))
sys.path.insert(0,str(r/"scripts"))
from motiondrive_v2_data import full_pose_matrices,motion_targets
sup=r/"data/etri/motiondrive_v2/r0reset_tplus_geometry_v2"
records=json.loads((r/"work_dirs/md_shared_dynamics_20260917/A3-DIRECT-s1/final_eval.json").read_text())["records"]
groups={}
for rec in records: groups.setdefault(rec["scenario"],[]).append(rec)
diffs=[]; future_diffs=[]; counted=0; max_case=None
for scene,recs in groups.items():
 report=json.loads((sup/(scene+".json")).read_text())
 paths=list(report["sources"])
 time_path=next(p for p in paths if p.endswith("/meta/timestamps.parquet"))
 pose_path=next(p for p in paths if p.endswith("/annotation/ego_pose.parquet"))
 ts=pd.read_parquet(time_path); ep=pd.read_parquet(pose_path)
 joined=ts[["timestamp","frame_id"]].merge(ep,on="timestamp",how="left",validate="one_to_one").sort_values("frame_id")
 frames=joined["frame_id"].to_numpy(np.int64)
 timestamps=joined["timestamp"].to_numpy(np.float64)
 times=(timestamps-timestamps[0])/1000.
 poses=full_pose_matrices(joined[["x","y","z"]].to_numpy(),joined[["roll","pitch","yaw"]].to_numpy())
 look={int(f):i for i,f in enumerate(frames)}
 with np.load(sup/(scene+".npz"),allow_pickle=False) as z:
  stored={int(f):(z["state_target"][i,:5].copy(),z["state_valid"][i,:5].copy()) for i,f in enumerate(z["frame"])}
 for rec in recs:
  f=int(rec["frame"]); now=look[f]; past=[look[f-x] for x in [1,2,5,10]]
  # Replay using only the 31 poses permitted within the clip; all future poses absent.
  start=look[f-30]
  output=motion_targets(poses[start:now+1],times[start:now+1],now-start,np.asarray(past)-start)
  expected,valid=stored[f]
  difference=np.abs(output["state_target"][:5]-expected)
  if max_case is None or difference.max()>max_case["max_abs"]:
   max_case={"scene":scene,"frame":f,"max_abs":float(difference.max())}
  assert output["state_valid"][:5].all() and valid.all()
  diffs.append(difference);counted+=1
result={"rows":counted,"scenes":len(groups),"source":"original ego_pose and timestamps parquet files from supervision source manifests",
"input_window":"frame-30 through frame inclusive, all future poses omitted",
"max_abs_difference_by_vx_vy_ax_ay_yaw":np.stack(diffs).max(0).tolist(),"max_case":max_case,
"all_causal_recomputed_status_valid":True,"compared_to":"state_target[:5] used as provided_status5 by A2/A3"}
out=r/"reports/md_shared_dynamics_20260917/causal_status_replay_20260918.json"
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
