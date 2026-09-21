"""CPU synthetic checks, not trained-model performance validation."""
import json, math
from pathlib import Path
import torch
from interval_vector_auxiliary import interval_displacements,vector_loss,wrap_compute_loss

torch.set_num_threads(2)
HERE=Path(__file__).resolve().parent

def old_length(p,g):
    return (torch.linalg.vector_norm(interval_displacements(p),dim=-1)-torch.linalg.vector_norm(interval_displacements(g),dim=-1)).abs().mean()

def lv(p,g,mask=None,norm=None):
    batch={'gt_plan':g}
    if mask is not None:batch['plan_valid']=mask
    return vector_loss({'plan_abs':p},batch,norm)

results={}
gen=torch.Generator().manual_seed(61)
p=torch.cumsum(torch.randn(4,6,2,generator=gen),1).requires_grad_(); g=torch.zeros_like(p)
a=old_length(p,g); b=lv(p,g)
ga=torch.autograd.grad(a,p,retain_graph=True)[0];gb=torch.autograd.grad(b,p)[0]
results['stationary_target']={'loss_difference':float((a-b).abs()),'gradient_max_difference':float((ga-gb).abs().max())}
assert torch.allclose(a,b) and torch.allclose(ga,gb)

angle=torch.randn(3,6,generator=gen);u=torch.stack([angle.cos(),angle.sin()],-1)
g=torch.cumsum(u*3,1);p=torch.cumsum(u*4,1)
results['aligned_directions']={'old':float(old_length(p,g)),'new':float(lv(p,g))}
assert torch.allclose(old_length(p,g),lv(p,g),atol=1e-6)

p=torch.cumsum(torch.tensor([[[3.,4.]]]).expand(2,6,2),1);g=p.clone();g[...,1]*=-1
results['mirror_discrimination']={'old':float(old_length(p,g)),'new':float(lv(p,g))}
assert float(old_length(p,g))==0 and float(lv(p,g))>0
assert torch.allclose(lv(p,g),lv(p*torch.tensor([1.,-1.]),g*torch.tensor([1.,-1.])))
assert float(lv(p,p))==0

cases=[]
for bs in (16,15):
 for all_invalid in (False,True):
  p=torch.cumsum(torch.randn(bs,6,2,generator=gen),1);g=torch.cumsum(torch.randn(bs,6,2,generator=gen),1)
  valid=torch.ones(bs,6,dtype=torch.bool);valid[::3,-1]=False
  if all_invalid:valid[:]=False
  g[~valid.all(-1)]=float('nan')
  pp=p.clone().requires_grad_();loss=lv(pp,g,valid);grad=torch.autograd.grad(loss,pp)[0]
  for mb in (1,2,8,16):
   x=p.clone().requires_grad_(); total=x.new_zeros(())
   for i in range(0,bs,mb):
    total=total+lv(x[i:i+mb],g[i:i+mb],valid[i:i+mb],{'plan_complete':valid.all(-1).sum()})
   gg=torch.autograd.grad(total,x)[0]
   le=float(abs(total-loss)); ge=float((gg-grad).abs().max())
   assert torch.allclose(total,loss,atol=1e-6,rtol=1e-6)
   assert torch.allclose(gg,grad,atol=1e-7,rtol=1e-6)
   assert torch.isfinite(gg).all()
   cases.append({'batch':bs,'microbatch':mb,'all_invalid':all_invalid,'loss_diff':le,'grad_max_diff':ge})
results['microbatch_cases']=cases

def original(outputs,batch,weights,**kwargs):
 loss=outputs['plan_abs'].square().mean();return loss,{'total':loss,'plan_d3':loss}
out={'plan_abs':p};bat={'gt_plan':g,'plan_valid':valid}
a,ap=original(out,bat,None);b,bp=wrap_compute_loss(original,0)(out,bat,None)
results['lambda0_identity']=torch.equal(a,b)
assert results['lambda0_identity']

# Equal-length angle error: the existing length loss gives no angular signal.
theta=torch.full((1,6),.02,requires_grad=True)
ell=torch.full((1,6),5.)
p=torch.cumsum(ell[...,None]*torch.stack([theta.cos(),theta.sin()],-1),1)
g=torch.stack([torch.arange(1,7)*5.,torch.zeros(6)],-1)[None]
old=old_length(p,g);new=lv(p,g)
go=torch.autograd.grad(.25*old,theta,retain_graph=True)[0]
gn=torch.autograd.grad(.25*new,theta,retain_graph=True)[0]
w=torch.tensor([11,11,5,5,2,2])/36
prefix=(torch.linalg.vector_norm(p-g,dim=-1)*w).mean(0).sum()
gp=torch.autograd.grad(prefix,theta)[0]
results['equal_length_angle_error']={'theta_rad':.02,'interval_length_m':5.,'old_length_loss':float(old),'vector_loss':float(new),'old_aux_theta_grad':go.tolist(),'new_aux_theta_grad':gn.tolist(),'prefix_theta_grad':gp.tolist()}
assert gn.abs().min()>.1
results['all_pass']=True
(HERE/'vector_auxiliary_tests.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
