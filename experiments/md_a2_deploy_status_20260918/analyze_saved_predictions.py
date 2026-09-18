
"""Read-only diagnosis of saved A2/MR predictions; no model inference or fitting."""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path("/NHNHOME/data/sukim/adcl")
WEIGHTS=np.array([11,11,5,5,2,2],dtype=np.float64)/36
def analyze(relative):
 path=ROOT/relative
 raw=path.read_bytes()
 records=json.loads(raw)["records"]
 pred=np.array([x["pred_abs_xy"] for x in records],dtype=np.float64)
 gt=np.array([x["gt_abs_xy"] for x in records],dtype=np.float64)
 bucket=np.array([x["bucket"] for x in records])
 def segment_speed(x):
  delta=np.diff(np.concatenate([np.zeros((len(x),1,2)),x],axis=1),axis=1)
  return np.linalg.norm(delta,axis=-1)/.5
 vp,vg=segment_speed(pred),segment_speed(gt)
 dv=vp-vg
 distance=np.linalg.norm(pred-gt,axis=-1)
 score=distance@WEIGHTS
 normal=bucket=="nonstop"
 speed_range=np.ptp(vg,axis=1)
 net_change=vg[:,-1]-vg[:,0]
 def group(mask):
  x=dv[mask]
  z={}
  for k,label in [(4,"first2s"),(6,"full3s")]:
   a=x[:,:k];common=a.mean(1,keepdims=True);residual=a-common
   total=float(np.square(a).sum())
   z[label]={"common_component_energy_fraction":float(k*np.square(common).sum()/total),
     "common_component_abs_mean_mps":float(np.abs(common).mean()),
     "common_component_signed_mean_mps":float(common.mean()),
     "time_varying_component_rms_mps":float(np.sqrt(np.square(residual).mean())),
     "identity_error":float(total-k*np.square(common).sum()-np.square(residual).sum())}
  return {"n":int(mask.sum()),"prefix":float(score[mask].mean()),
    "contribution_to_all_prefix":float(score[mask].sum()/len(records)),
    "fraction_of_all_prefix":float(score[mask].sum()/score.sum()),
    "interval_speed_error_mean_mps":x.mean(0).tolist(),
    "interval_speed_error_mae_mps":np.abs(x).mean(0).tolist(),
    "gt_net_interval_speed_change_mean_mps":float(net_change[mask].mean()),
    "pred_net_interval_speed_change_mean_mps":float((vp[mask,-1]-vp[mask,0]).mean()),
    "decomposition":z}
 contributions=distance.mean(0)*WEIGHTS
 masks={"all":np.ones(len(records),bool),"nonstop":normal,
   "nonstop_range_le_0p5":normal&(speed_range<=.5),
   "nonstop_range_gt_0p5":normal&(speed_range>.5),
   "nonstop_range_le_0p3_sensitivity":normal&(speed_range<=.3),
   "nonstop_net_accel_ge_0p5":normal&(net_change>=.5),
   "nonstop_net_decel_le_neg0p5":normal&(net_change<=-.5)}
 return {"source":str(path),"source_sha256":hashlib.sha256(raw).hexdigest(),"rows":len(records),
   "point_l2_mean":distance.mean(0).tolist(),"point_weighted_contributions":contributions.tolist(),
   "first2s_fraction_of_prefix":float(contributions[:4].sum()/contributions.sum()),
   "groups":{name:group(mask) for name,mask in masks.items()}}

output={"schema_version":1,"purpose":"Distinguish persistent progress-error patterns from a claim of acceleration/braking timing causality",
 "definitions":{"interval_speed":"Chord length between successive predicted/GT waypoints divided by 0.5 s; not instantaneous vehicle status",
  "nonstop":"Existing bucket label from saved evaluator",
  "nearconstant":"Within nonstop, max-min of the six GT interval speeds <=0.5 m/s; 0.3 m/s sensitivity also reported",
  "common_component":"For each row and chosen horizon, arithmetic mean of predicted-minus-GT interval speed",
  "energy_identity":"sum(delta_v**2) = horizon_intervals * sum(common**2) + sum((delta_v-common)**2)",
  "energy_warning":"Unweighted interval-speed squared-error decomposition, not a decomposition of official D3",
  "accel_decel_groups":"Net change between last and first GT interval-average speed >=0.5 or <=-0.5 m/s; no onset-time label",
  "causality_warning":"Persistent speed error can arise from initial progress bias, acceleration magnitude, time shift, and other causes; these summaries do not identify the cause",
  "scope":"Saved DEV V0 predictions only; no server A2 evaluation, no model execution, no new training"},
 "models":{"A2":analyze("work_dirs/md_shared_dynamics_20260917/A2-DIRECT-s1/final_eval.json"),
           "MR":analyze("work_dirs/md_r0_reset_20260914/MR-NATIVE-s1/final_eval.json")}}
print(json.dumps(output,indent=2,ensure_ascii=False))
