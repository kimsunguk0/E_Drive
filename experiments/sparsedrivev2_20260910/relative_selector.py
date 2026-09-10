"""Score completed bank candidates using explicit relative kinematic features.

This experimental adapter never constructs or refines trajectory coordinates.
Its goal input affects only its new final score residual. An explicitly chosen
``base_goal_mode='selection'`` also preserves the base's existing goal selection.
The caller controls RNG state and checkpoint provenance; construction uses normal
PyTorch initialization and does not seed or restore global RNG state.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext

import torch
from torch import nn


FEATURE_VERSION = "candidate_relative_32_v1"
FEATURE_NAMES = (
    tuple(f"relative_velocity_t{t}_{axis}" for t in range(1, 7) for axis in "xy")
    + tuple(f"interval_acceleration_t{t}_{axis}_div3" for t in range(2, 7) for axis in "xy")
    + ("status_vx_div20", "status_vy_div20", "status_ax_div3", "status_ay_div3",
       "goal_x_div50", "goal_y_div50", "endpoint_minus_goal_x_div50",
       "endpoint_minus_goal_y_div50", "centered_base_score", "goal_present")
)


def _candidate_contract(output: Mapping):
    xy, scores = output["candidate_xy"], output["scores"]
    if not isinstance(xy, torch.Tensor) or xy.ndim != 4 or xy.shape[-2:] != (6, 2):
        raise ValueError("candidate_xy must be a tensor [B,K,6,2]")
    if not isinstance(scores, torch.Tensor) or scores.shape != xy.shape[:2]:
        raise ValueError("scores must be a tensor [B,K]")
    if xy.device != scores.device or xy.shape[0] == 0 or xy.shape[1] == 0:
        raise ValueError("candidate tensors must share a device and have nonempty B,K")
    valid = output.get("candidate_valid")
    valid = (torch.ones_like(scores, dtype=torch.bool) if valid is None
             else torch.as_tensor(valid, device=xy.device, dtype=torch.bool))
    if valid.shape != scores.shape or not valid.any(-1).all():
        raise ValueError("candidate_valid must be [B,K] with a valid candidate per row")
    if not torch.isfinite(scores[valid]).all() or not torch.isfinite(xy[valid]).all():
        raise ValueError("Valid candidate coordinates and scores must be finite")
    return xy, scores, valid


def make_candidate_features(output: Mapping, status: torch.Tensor,
                            goal_xy: torch.Tensor | None = None) -> torch.Tensor:
    """Return FP32 [B,K,32] features, without accepting targets or future labels.

    ``output`` requires candidate_xy [B,K,6,2], scores [B,K], and optionally
    candidate_valid [B,K]. XY is cumulative current-ego metres at .5s spacing.
    Status is [B,8]; only causal vx,vy,ax,ay in slots 4:8 are used.
    Goal is provided current-ego XY metres [B,2], or None. Feature order:
      0:12  six interval XY velocities minus causal vx/vy (m/s)
      12:22 five interval XY accelerations / 3 (m/s^2)
      22:26 causal vx/vy / 20, causal ax/ay / 3
      26:30 goal / 50, (3s candidate endpoint - goal) / 50
      30    base score minus the mean over valid candidates in its row
      31    goal present (1 or 0)
    With no goal, all 26:30 and 31 entries are zero. Invalid candidates have
    entirely zero features and never enter the score-centering denominator.
    """
    xy, scores, valid = _candidate_contract(output)
    b, k = scores.shape
    status = torch.as_tensor(status, device=xy.device, dtype=torch.float32)
    if status.shape != (b, 8) or not torch.isfinite(status[:, 4:8]).all():
        raise ValueError("status must be [B,8] with finite causal slots 4:8")
    if goal_xy is not None:
        goal_xy = torch.as_tensor(goal_xy, device=xy.device, dtype=torch.float32)
        if goal_xy.shape != (b, 2) or not torch.isfinite(goal_xy).all():
            raise ValueError("goal_xy must be finite [B,2] current-ego XY metres")
    with torch.autocast(device_type=xy.device.type, enabled=False):
        # Mask before arithmetic, so even nonfinite invalid rows cannot leak NaNs.
        xy32 = torch.where(valid[..., None, None], xy.float(), 0.)
        vel = torch.diff(xy32, dim=2, prepend=torch.zeros_like(xy32[:, :, :1])) * 2.
        relative_vel = (vel - status[:, None, None, 4:6]).reshape(b, k, 12)
        accel = (torch.diff(vel, dim=2) * (2. / 3.)).reshape(b, k, 10)
        state = torch.cat((status[:, 4:6] / 20., status[:, 6:8] / 3.), dim=-1)
        state = state[:, None].expand(b, k, 4)
        if goal_xy is None:
            goal_features = torch.zeros((b, k, 4), device=xy.device, dtype=torch.float32)
            present = torch.zeros((b, k, 1), device=xy.device, dtype=torch.float32)
        else:
            goal_features = torch.cat((goal_xy[:, None].expand(b, k, 2) / 50.,
                                       (xy32[:, :, -1] - goal_xy[:, None]) / 50.), -1)
            present = torch.ones((b, k, 1), device=xy.device, dtype=torch.float32)
        safe_scores = torch.where(valid, scores.float(), 0.)
        mean = safe_scores.sum(-1, keepdim=True) / valid.sum(-1, keepdim=True)
        centered = safe_scores - mean
        features = torch.cat((relative_vel, accel, state, goal_features,
                              centered[..., None], present), dim=-1)
        return torch.where(valid[..., None], features, 0.)


class RelativeScoreHead(nn.Module):
    """Standalone FP32 32 -> 128 -> 64 -> 1 residual scorer.

    Only the last linear layer is zero initialized. The first update can train
    that layer; hidden layers start receiving gradients after its weights move.
    Parameters must remain FP32, including when the base uses BF16 autocast.
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(32, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != 32:
            raise ValueError("features must be [B,K,32]")
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise TypeError("RelativeScoreHead parameters must remain float32")
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.mlp(features.float()).squeeze(-1)


class CandidateRelativeSelector(nn.Module):
    """Wrap a frozen selector and choose one of its completed candidate rows.

    ``rescore`` preserves its input candidate tensors and IDs under any status
    or goal change. ``forward`` first invokes the base with its original status;
    the base may itself choose a different shortlist for a different status.
    base_goal_mode='none' never forwards this wrapper's goal into the base;
    'selection' forwards goal_xy to an already goal-capable base explicitly.

    Scores are float32 base scores plus the residual, with fixed coefficient 1.
    Invalid scores retain their original sentinel, but argmax always masks them.
    Consumers must also respect candidate_valid. All current 100m banks are valid.
    The zero head preserves all original FP32 fields and decisions when the
    base already selects the first valid score argmax (the public model contract).
    """
    def __init__(self, base: nn.Module, *, base_goal_mode: str = "none",
                 freeze_base: bool = True):
        super().__init__()
        if base_goal_mode not in ("none", "selection"):
            raise ValueError("base_goal_mode must be 'none' or 'selection'")
        self.base = base
        self.base_goal_mode = base_goal_mode
        self.freeze_base = bool(freeze_base)
        self.relative_head = RelativeScoreHead()
        reference = next(base.parameters(), None)
        if reference is None:
            reference = next(base.buffers(), None)
        if reference is not None:
            self.relative_head.to(device=reference.device)
        if self.freeze_base:
            self.base.requires_grad_(False)
            self.base.eval()

    @property
    def _trajectory_head(self):
        return self.base._trajectory_head

    @property
    def _backbone(self):
        return self.base._backbone

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base:
            self.base.eval()
        return self

    def score_residual(self, features: torch.Tensor) -> torch.Tensor:
        return self.relative_head(features)

    def rescore(self, output: Mapping, status: torch.Tensor,
                goal_xy: torch.Tensor | None = None) -> dict:
        xy, base_scores, valid = _candidate_contract(output)
        ids = output["candidate_ids"]
        if ids.shape != base_scores.shape or ids.device != xy.device:
            raise ValueError("candidate_ids must be [B,K] on the candidate device")
        features = make_candidate_features(output, status, goal_xy)
        residual = self.score_residual(features)
        residual = torch.where(valid, residual, 0.)
        with torch.autocast(device_type=xy.device.type, enabled=False):
            scores = base_scores.float() + residual.float()
            selected = scores.masked_fill(~valid, float("-inf")).argmax(-1)
        batch = torch.arange(xy.shape[0], device=xy.device)
        result = dict(output)
        result.update(scores=scores, trajectory=xy[batch, selected],
                      selected_candidate_id=ids[batch, selected],
                      relative_score_residual=residual, base_scores=base_scores)
        return result

    def forward(self, images, lidar2img, image_hw=None, status=None, goal_xy=None):
        if status is None:
            raise ValueError("Explicit causal status [B,8] is required")
        inputs = dict(images=images, lidar2img=lidar2img, image_hw=image_hw, status=status)
        if self.base_goal_mode == "selection":
            inputs["goal_xy"] = goal_xy
        with torch.no_grad() if self.freeze_base else nullcontext():
            output = self.base(**inputs)
        return self.rescore(output, status, goal_xy)
