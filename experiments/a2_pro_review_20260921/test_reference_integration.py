"""CPU checks of the proposed reference wrapper with actual common losses."""
from pathlib import Path
import argparse,sys,json,hashlib
import torch
ROOT=Path('/NHNHOME/data/sukim/adcl');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'experiments/md_r0_reset_20260914'),str(HERE/'reference')]
from motiondrive_v2_training import compute_loss,LossWeights,build_loss_normalizers
from length_auxiliary import wrap_compute_loss as wrap_length,length_loss
from interval_vector_auxiliary import wrap_compute_loss as wrap_vector,vector_loss

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(61);b=15
    batch={'gt_plan':torch.randn(b,6,2).cumsum(1),'plan_valid':torch.ones(b,6,dtype=torch.bool),
        'history_target':torch.randn(b,4,4),'history_valid':torch.ones(b,4,4,dtype=torch.bool),
        'state_target':torch.randn(b,6),'state_valid':torch.ones(b,6,dtype=torch.bool)}
    batch['plan_valid'][::3,-1]=False;batch['gt_plan'][~batch['plan_valid'].all(-1)]=float('nan')
    batch['state_target'][:,5]=(torch.arange(b)%2).float()
    for p in ('occ','lane'):
        batch[p+'_target']=(torch.rand(b,1,4,4)>.5).float();batch[p+'_valid']=torch.rand(b,1,4,4)>.2
    raw={'plan_abs':torch.randn(b,6,2).cumsum(1),'history_hat':torch.randn(b,4,4),'history_logvar':torch.randn(b,4,4),
        'state_hat':torch.randn(b,6),'state_logvar':torch.randn(b,5),'occ_logits':torch.randn(b,1,4,4),'lane_logits':torch.randn(b,1,4,4)}
    weights=LossWeights();norm=build_loss_normalizers(batch);out={k:v.clone().requires_grad_() for k,v in raw.items()}
    common,parts=compute_loss(out,batch,weights,normalizers=norm)
    old,old_parts=wrap_length(compute_loss,.25)(out,batch,weights,normalizers=norm)
    new,new_parts=wrap_vector(compute_loss,.25)(out,batch,weights,normalizers=norm)
    assert 'plan_interval_length' not in new_parts and 'plan_interval_vector' not in old_parts
    for key in parts:
        if key!='total':assert torch.equal(old_parts[key],new_parts[key])
    assert torch.allclose(new-old,.25*(vector_loss(out,batch,norm)-length_loss(out,batch,norm)),atol=1e-6)
    zero,zero_parts=wrap_vector(compute_loss,0)(out,batch,weights,normalizers=norm);assert torch.equal(zero,common)
    guarded=[]
    for wrapper in (wrap_length(compute_loss,.25),wrap_vector(compute_loss,.25)):
        try:wrap_vector(wrapper,.25)(out,batch,weights,normalizers=norm)
        except ValueError:guarded.append(True)
        else:raise AssertionError('Auxiliary wrapper nesting was accepted')
    new.backward();grad={k:v.grad.clone() for k,v in out.items()};checks=[]
    for mb in (1,2,8,15):
        out_mb={k:v.clone().requires_grad_() for k,v in raw.items()};total=torch.zeros(())
        for start in range(0,b,mb):
            oo={k:v[start:start+mb] for k,v in out_mb.items()};bb={k:v[start:start+mb] for k,v in batch.items()}
            loss,_=wrap_vector(compute_loss,.25)(oo,bb,weights,normalizers=norm);total=total+loss
        total.backward();gap=max(float((out_mb[k].grad-grad[k]).abs().max()) for k in out_mb)
        assert torch.allclose(total,new.detach(),atol=2e-6,rtol=1e-6) and gap<2e-7
        assert all(torch.isfinite(v.grad).all() for v in out_mb.values())
        checks.append({'microbatch':mb,'loss_gap':float((total.detach()-new.detach()).abs()),'max_output_gradient_gap':gap})
    result={'passed':True,'scope':'CPU synthetic outputs with actual uncertainty/common loss; no model fitting',
        'coefficient':.25,'batch':b,'complete_rows':int(norm['plan_complete']),'length_aux_removed':True,
        'all_common_loss_parts_identical':True,'lambda_zero_bitwise_common_loss':True,'both_wrapper_nesting_cases_rejected':all(guarded),
        'microbatch':checks,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256((HERE/'reference/interval_vector_auxiliary.py').read_bytes()).hexdigest()}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
