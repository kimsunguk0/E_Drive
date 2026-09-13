"""Multi-mode continuous planner: M completed trajectories, one chosen.

The parent's time queries, decoder, memory construction and XY head are reused
unchanged. Each mode adds a learned embedding to the six time queries, the 48
queries run in one decoder pass with self-attention confined to each mode's own
six future steps, and cross-attention to the shared memory is common to all
modes. There is no bank, no velocity rescaling and no goal-based correction:
every candidate is regressed directly.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.motiondrive_v2.planner import DirectTrajectoryPlanner

TIME_WEIGHTS = (11 / 36, 11 / 36, 5 / 36, 5 / 36, 2 / 36, 2 / 36)
MODE_INIT_STD = 0.02


def block_diagonal_mask(modes, steps, device):
    """Allow attention inside a mode only. This is not a causal mask: the six
    future steps of one mode still attend to each other as in the parent."""
    index = torch.arange(modes * steps, device=device) // steps
    return index[:, None] != index[None, :]


class CandidateScorer(nn.Module):
    """Score a candidate from its own hidden states and its own geometry."""

    def __init__(self, channels=128, steps=6):
        super().__init__()
        self.steps = steps
        self.hidden = nn.Sequential(nn.LayerNorm(channels * steps),
                                    nn.Linear(channels * steps, 128), nn.GELU())
        self.score = nn.Sequential(nn.Linear(128 + 34, 128), nn.GELU(),
                                   nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    @staticmethod
    def geometry(xy):
        """12 positions, 12 interval velocities, 10 interval accelerations."""
        position = xy / xy.new_tensor([80.0, 64.0])
        velocity = torch.diff(xy, dim=-2, prepend=torch.zeros_like(xy[..., :1, :])) * 2.0
        acceleration = torch.diff(velocity, dim=-2) * 2.0 / 3.0
        return torch.cat([position.flatten(-2),
                          (velocity / xy.new_tensor([20.0, 5.0])).flatten(-2),
                          (acceleration / 3.0).flatten(-2)], -1)

    def forward(self, hidden, xy):
        b, m = hidden.shape[:2]
        features = torch.cat([self.hidden(hidden.reshape(b, m, -1)), self.geometry(xy)], -1)
        if not torch.isfinite(features).all():
            raise ValueError("Non-finite candidate scorer features")
        return self.score(features).squeeze(-1)


class MultiModePlanner(DirectTrajectoryPlanner):
    """Parent planner emitting `modes` complete trajectories plus a base score."""

    def __init__(self, config, modes=8):
        super().__init__(config)
        self.modes = int(modes)
        # Mode 0 is exactly the parent query, so the first candidate reproduces it.
        embedding = torch.randn(self.modes, config.channels,
                                generator=torch.Generator().manual_seed(0)) * MODE_INIT_STD
        embedding[0].zero_()
        self.mode_embedding = nn.Parameter(embedding)
        self.candidate_scorer = CandidateScorer(config.channels, len(self.waypoint_queries))
        self.last = None

    def forward(self, scene_features, motion_features, predicted_state, predicted_history):
        b = scene_features.shape[0]
        steps = len(self.waypoint_queries)
        with torch.autocast(device_type=scene_features.device.type, enabled=False):
            scene = scene_features.float() + self.scene_position(self.scene_xy.float())[None]
            motion = motion_features.float() + self.motion_type.float()
            state = predicted_state.float() / self.state_scale
            history = (predicted_history.float() / self.history_scale).flatten(1)
            status = torch.cat([state, history], -1)
            if not self.config.state_on:
                status = torch.zeros_like(status)
            state_token = self.state_projection(status)[:, None]
            memory = torch.cat([scene, motion, state_token], 1)
            queries = (self.waypoint_queries.float()[None, None]
                       + self.mode_embedding.float()[None, :, None]).expand(b, -1, -1, -1)
            decoded = self.decoder(queries.reshape(b, self.modes * steps, -1), memory,
                                   tgt_mask=block_diagonal_mask(self.modes, steps,
                                                                queries.device))
            hidden = decoded.reshape(b, self.modes, steps, -1)
            candidates = (self.xy_head(hidden.float())
                          * hidden.new_tensor(self.config.plan_output_scale))
            if not torch.isfinite(candidates).all():
                raise ValueError("Non-finite candidate trajectories")
            # The scorer must not become a shortcut into the regression path.
            scores = self.candidate_scorer(hidden.detach(), candidates.detach())
            selected = scores.argmax(-1)
            chosen = candidates[torch.arange(b, device=candidates.device), selected]
            self.last = {"candidates": candidates, "hidden": hidden, "scores": scores,
                         "selected": selected}
            return chosen


def multimode_losses(planner, gt_plan, complete, temperature=0.1,
                     winner_weight=0.95, score_weight=0.1):
    """Relaxed winner-take-all regression plus the base scorer's own CE."""
    last = planner.last
    if last is None:
        raise RuntimeError("planner produced no candidates")
    candidates = last["candidates"].float()
    b, m = candidates.shape[:2]
    target = torch.where(complete[:, None, None], gt_plan.float(),
                         torch.zeros_like(gt_plan.float()))[:, None]
    distance = torch.linalg.vector_norm(candidates - target, dim=-1)
    cost = (distance * distance.new_tensor(TIME_WEIGHTS)).sum(-1)      # [B,M]
    rows = complete.float()
    denominator = rows.sum().clamp_min(1.0)
    regression = ((winner_weight * cost.min(-1).values
                   + (1.0 - winner_weight) * cost.mean(-1)) * rows).sum() / denominator
    probability = F.softmax(-cost.detach() / temperature, dim=-1)
    log_probability = F.log_softmax(last["scores"].float(), dim=-1)
    score_ce = (-(probability * log_probability).sum(-1) * rows).sum() / denominator
    return regression + score_weight * score_ce, {
        "multi_regression": regression, "base_score_ce": score_ce,
        "oracle_d3": (cost.min(-1).values * rows).sum() / denominator,
        "selected_d3": (cost[torch.arange(b, device=cost.device),
                             last["selected"]] * rows).sum() / denominator,
        "candidate_spread": _pairwise_spread(candidates, rows, denominator),
    }


def _pairwise_spread(candidates, rows, denominator):
    """Weighted mean distance between distinct candidates; 0 means collapse."""
    m = candidates.shape[1]
    step = torch.linalg.vector_norm(candidates[:, :, None] - candidates[:, None, :], dim=-1)
    weighted = (step * step.new_tensor(TIME_WEIGHTS)).sum(-1)          # [B,M,M]
    off_diagonal = ~torch.eye(m, dtype=torch.bool, device=weighted.device)
    per_row = weighted[:, off_diagonal].mean(-1)
    return (per_row * rows).sum() / denominator
