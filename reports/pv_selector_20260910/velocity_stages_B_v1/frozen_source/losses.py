"""D3-aligned supervision for fixed-bank coarse/fine selection.

Ground truth is read only here, after the complete model forward. Coarse path
labels compare each path at GT progress; coarse speed labels compare progress
along the common path. Targets use only the six scored ETRI future points.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from data import d3, TIME_WEIGHTS


@torch.no_grad()
def coarse_costs(path_xy, velocity, gt):
    gt = gt.float()
    path_xy = path_xy.float()
    velocity = velocity.float()[..., :6]
    delta = torch.diff(gt, dim=1, prepend=torch.zeros_like(gt[:, :1]))
    progress = torch.linalg.vector_norm(delta, dim=-1).cumsum(-1)
    velocity_progress = (velocity * .5).cumsum(-1)
    velocity_cost = ((progress[:, None, :] - velocity_progress[None]).abs()
                     * progress.new_tensor(TIME_WEIGHTS)).sum(-1)
    p, s, _ = path_xy.shape
    path = torch.cat((path_xy.new_zeros(p, 1, 2), path_xy), dim=1)
    arc = torch.linalg.vector_norm(torch.diff(path, dim=1), dim=-1).cumsum(-1)
    arc = torch.cat((arc.new_zeros(p, 1), arc), dim=1)
    b = len(gt)
    query = progress[None].expand(p, b, 6).reshape(p, b * 6).contiguous()
    upper = torch.searchsorted(arc.contiguous(), query, right=True).clamp(1, s)
    lower = upper - 1
    lo = torch.gather(arc, 1, lower)
    hi = torch.gather(arc, 1, upper)
    fraction = ((query - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    lo_xy = torch.gather(path, 1, lower[..., None].expand(-1, -1, 2))
    hi_xy = torch.gather(path, 1, upper[..., None].expand(-1, -1, 2))
    reconstructed = (lo_xy + fraction[..., None] * (hi_xy - lo_xy)).reshape(p, b, 6, 2).permute(1, 0, 2, 3)
    path_cost = d3(reconstructed, gt[:, None].expand_as(reconstructed))
    valid = (progress[:, -1, None] <= arc[None, :, -1] + 1e-4)
    path_cost = path_cost.masked_fill(~valid, 1e4)
    return path_cost, velocity_cost


def _gather(cost, ids):
    if ids.ndim == 1:
        ids = ids[None].expand(len(cost), -1)
    return torch.gather(cost, 1, ids.long())


def soft_cost_ce(logits, costs, temperature, valid=None):
    if logits.shape != costs.shape or temperature <= 0:
        raise ValueError("Scoring target shape/temperature mismatch")
    if valid is None:
        valid = torch.isfinite(logits) & torch.isfinite(costs)
    if not valid.any(-1).all():
        raise ValueError("At least one complete candidate is required for each sample")
    clean_logits = logits.float().masked_fill(~valid, -1e4)
    target_logits = (-costs.float() / temperature).masked_fill(~valid, -1e4)
    target = F.softmax(target_logits, dim=-1).detach()
    return -(target * F.log_softmax(clean_logits, dim=-1)).sum(-1).mean()


def selection_loss(output, gt, path_xy, velocity, *, temperature=.1):
    path_cost, velocity_cost = coarse_costs(path_xy, velocity, gt)
    path_losses, velocity_losses = [], []
    for stage in output["coarse"]:
        path_losses.append(soft_cost_ce(stage["path_scores"], _gather(path_cost, stage["path_ids"]), temperature))
        velocity_losses.append(soft_cost_ce(stage["velocity_scores"], _gather(velocity_cost, stage["velocity_ids"]), temperature))
    candidate = output["candidate_xy"]
    costs = d3(candidate, gt[:, None].expand_as(candidate)).detach()
    valid = output.get("candidate_valid", torch.isfinite(output["scores"]))
    if valid.ndim == 3:
        valid = valid.all(-1)
    valid = valid.bool() & torch.isfinite(output["scores"])
    fine = soft_cost_ce(output["scores"], costs, temperature, valid)
    path = torch.stack(path_losses).mean()
    vel = torch.stack(velocity_losses).mean()
    loss = fine + .5 * (path + vel)
    oracle = costs.masked_fill(~valid, float("inf")).amin(-1).mean()
    return {"loss": loss, "fine_ce": fine, "path_ce": path, "velocity_ce": vel,
            "shortlist_oracle": oracle, "selected_d3": d3(output["trajectory"], gt).mean()}
