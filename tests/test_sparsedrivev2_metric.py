"""Reference checks for unit conversion and fixed-candidate loss gradients."""
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments/sparsedrivev2_20260910"))
from data import d3
from losses import coarse_costs, selection_loss


def test_constant_velocity_error_matches_official_cumulative_ade():
    times=torch.arange(1,7)*.5
    pred=torch.stack((times*.1,torch.zeros(6)),dim=-1)[None]
    assert abs(d3(pred,torch.zeros_like(pred)).item()-.125)<1e-7
    e=pred.norm(dim=-1)
    explicit=(e[:,:2].mean(-1)+e[:,:4].mean(-1)+e.mean(-1))/3
    torch.testing.assert_close(d3(pred,torch.zeros_like(pred)),explicit)


def test_coarse_costs_known_path_and_speed():
    path=torch.stack((torch.arange(1,51)*2.,torch.zeros(50)),dim=-1)[None]
    vel=torch.tensor([[2.]*8,[3.]*8])
    gt=torch.stack((torch.arange(1,7)*1.,torch.zeros(6)),dim=-1)[None]
    pc,vc=coarse_costs(path,vel,gt)
    torch.testing.assert_close(pc,torch.zeros(1,1))
    torch.testing.assert_close(vc,torch.tensor([[0.,1.25]]))


def test_fixed_bank_ranking_gradient_ignores_invalid_candidates():
    path=torch.stack((torch.arange(1,51)*2.,torch.zeros(50)),dim=-1)[None]
    vel=torch.tensor([[2.]*8,[3.]*8])
    gt=torch.stack((torch.arange(1,7)*1.,torch.zeros(6)),dim=-1)[None]
    scores=torch.tensor([[0.,0.,-1e4]],requires_grad=True)
    ps=torch.zeros(1,1,requires_grad=True)
    vs=torch.zeros(1,2,requires_grad=True)
    candidate=torch.stack((gt[0],gt[0]*1.5,gt[0]*10))[None]
    out={"trajectory":candidate[:,0],"candidate_xy":candidate,"scores":scores,
         "candidate_valid":torch.tensor([[True,True,False]]),
         "coarse":[{"path_scores":ps,"velocity_scores":vs,
                    "path_ids":torch.tensor([[0]]),"velocity_ids":torch.tensor([[0,1]])}]}
    result=selection_loss(out,gt,path,vel)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert scores.grad[0,0]<0 and scores.grad[0,1]>0 and scores.grad[0,2]==0
    assert vs.grad[0,0]<0 and vs.grad[0,1]>0
    assert result["shortlist_oracle"]==0
