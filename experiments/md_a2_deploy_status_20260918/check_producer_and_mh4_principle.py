
import json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
root=Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0,str(root));sys.path.insert(0,str(root/"experiments/md_a2_deploy_status_20260918"))
from nominal_status import status5_from_clip_records
from models.motiondrive_v2.scene_encoder import masked_softmax
torch.set_num_threads(2)
rows=[]
for f in range(-30,1):
 t=f*.1
 rows.append(dict(frame=f,x=7*t+.5*.4*t*t,y=.2*t+.5*(-.1)*t*t,z=0.,roll=0.,pitch=0.,yaw=.03*t))
s,diag=status5_from_clip_records(rows)
expected=np.array([7,.2,.4,-.1,.03],np.float32)
np.testing.assert_allclose(s,expected,atol=1e-6,rtol=0)
flipped=[dict(row,y=-row["y"],roll=-row["roll"],yaw=-row["yaw"]) for row in rows]
sf,_=status5_from_clip_records(flipped)
np.testing.assert_allclose(sf,s*np.array([1,-1,1,-1,-1]),atol=1e-6,rtol=0)
missing=[row for row in rows if row["frame"]!=-5]
try:
 status5_from_clip_records(missing);gap_rejected=False
except ValueError:gap_rejected=True
assert gap_rejected
# A different row ordinal must not disguise the 0.2s gap.
poison=rows+[dict(frame=50,x=float("nan"),y=float("nan"),z=float("nan"),roll=float("nan"),pitch=float("nan"),yaw=float("nan"))]
sp,_=status5_from_clip_records(poison)
np.testing.assert_array_equal(sp,s)
out={"producer_checks":{"known_quadratic_fit_max_error":float(abs(s-expected).max()),"flip_max_error":float(abs(sf-s*np.array([1,-1,1,-1,-1])).max()),"missing_frame_fit_gap_rejected":gap_rejected,"future_goal_nan_ignored":True},"pooling_checks":{}}
for dtype in [torch.float32,torch.bfloat16]:
 torch.manual_seed(18)
 q=torch.randn(2,7,32).to(dtype);k=torch.randn(2,7,60,32).to(dtype);v=torch.randn(2,7,60,128).to(dtype)
 valid=torch.rand(2,7,60)>.25;valid[:,0,:]=False
 base_weight=masked_softmax((q.float()[:,:,None]*k.float()).sum(-1)/(32**.5),valid)
 base=(base_weight[:,:,:,None]*v.float()).sum(2)
 qm=torch.nn.Parameter(torch.eye(32).repeat(4,1,1));km=torch.nn.Parameter(torch.eye(32).repeat(4,1,1))
 identity_out=torch.nn.Parameter(torch.eye(128))
 pieces=[];weights=[]
 for h in range(4):
  qh=torch.nn.functional.linear(q.float(),qm[h]);kh=torch.nn.functional.linear(k.float(),km[h])
  wh=masked_softmax((qh[:,:,None]*kh).sum(-1)/(32**.5),valid)
  pieces.append((wh[:,:,:,None]*v.float()[:,:,:,h*32:(h+1)*32]).sum(2));weights.append(wh)
 final=torch.nn.functional.linear(torch.cat(pieces,-1),identity_out)
 diff=float((final-base).abs().max().detach())
 assert diff<1e-6 and torch.isfinite(final).all() and torch.equal(final[:,0],torch.zeros_like(final[:,0]))
 loss=(final*torch.randn_like(final)).sum();loss.backward()
 gradients=qm.grad.flatten(1)
 distinct=bool(all(not torch.equal(gradients[0],gradients[h]) for h in range(1,4)))
 assert distinct and torch.isfinite(qm.grad).all() and torch.isfinite(km.grad).all()
 out["pooling_checks"][str(dtype)]={"max_abs_initial_evidence_difference":diff,"all_invalid_evidence_exact_zero":True,"q_gradient_norms":gradients.norm(dim=1).tolist(),"head_q_gradients_distinct":distinct}
out["scope"]="Synthetic tensors with the current production masked_softmax; not an integrated MH4 model or training/latency result"
path=root/"reports/md_a2_deploy_status_20260918/producer_and_mh4_principle_checks.json"
path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
