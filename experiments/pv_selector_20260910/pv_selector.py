"""Completed-candidate token capture and a matched real/zero-token selector.

Load the original C model with its own isolated, verified runtime first. Then:

    capture = CandidateTokenCapture(c_model, freeze_base=True)
    output = capture(**original_c_inputs)
    selector = SceneResidualSelector(mode="real")  # or "zero", same state shape
    result = selector(output, goal_xy)

The capture owns its base exclusively and installs a hook for one forward only.
It reads the last public traj_mlp input after candidate DFA and self-attention.
The original C goal route, scores, IDs, coordinates and auxiliary outputs remain
unchanged. In C these tokens include indirect common-perception status effects;
they are image-conditioned candidate tokens, not pure image-only state estimates.

The new selector has no raw/predicted status input and never constructs candidate
coordinates. Its 32 features reproduce the old helper with status=zeros. No
top-level import from the UUID-isolated original runtime is needed. Zero and real
arms must be initialized from the same state_dict by the caller, with mode recorded
separately; mode is not a learned tensor. Goal is used after candidates are complete.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
import threading

import torch
from torch import nn


FEATURE_VERSION = "candidate_relative_32_v1_optional_causal_status"
TOKEN_VERSION = "public_final_traj_mlp_input_256_v1"
SELECTOR_VERSION = "candidate_scene_residual_real_zero_status_v1"


def _candidate_contract(output):
    if not isinstance(output, Mapping):
        raise TypeError("Expected a completed-candidate output mapping")
    xy, scores = output["candidate_xy"], output["scores"]
    if not isinstance(xy, torch.Tensor) or xy.ndim != 4 or xy.shape[-2:] != (6, 2):
        raise ValueError("candidate_xy must be [B,K,6,2]")
    if not isinstance(scores, torch.Tensor) or scores.shape != xy.shape[:2]:
        raise ValueError("scores must be [B,K]")
    if xy.device != scores.device or not xy.is_floating_point() or not scores.is_floating_point():
        raise ValueError("Candidate coordinates/scores must be floating on one device")
    if min(scores.shape) < 1:
        raise ValueError("Empty candidate population")
    valid = output.get("candidate_valid")
    valid = torch.ones_like(scores, dtype=torch.bool) if valid is None else valid
    if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or valid.device != scores.device or valid.shape != scores.shape
            or not bool(valid.any(-1).all())):
        raise ValueError("candidate_valid must be bool [B,K] with a valid candidate per row")
    if not bool(torch.isfinite(xy[valid]).all()) or not bool(torch.isfinite(scores[valid]).all()):
        raise ValueError("Valid coordinates/scores must be finite")
    return xy, scores, valid


def _candidate_ids(output, scores):
    ids = output["candidate_ids"]
    if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64
            or ids.shape != scores.shape or ids.device != scores.device
            or bool((ids < 0).any())):
        raise ValueError("candidate_ids must be nonnegative int64 [B,K]")
    return ids


def build_features32(output, goal_xy=None, status=None):
    """Exact old make_candidate_features(output, status[B,8], goal) in FP32.

    Features are candidate interval velocities relative to causal vx/vy (12),
    interval accelerations/3 (10), the four causal state slots, goal/50 and
    endpoint-minus-goal/50 (4), centered original score (1), and goal-present
    (1). With ``status=None`` the four state slots stay zero and the velocities
    stay absolute, reproducing the earlier zero-status behaviour bit for bit.
    Only slots 4:8 (causal vx, vy, ax, ay) are read. Labels are never accepted.
    Invalid rows are zero before arithmetic and again in the final output.
    """
    xy, scores, valid = _candidate_contract(output)
    b, k = scores.shape
    if goal_xy is not None:
        goal_xy = torch.as_tensor(goal_xy, device=xy.device, dtype=torch.float32)
        if goal_xy.shape != (b, 2) or not bool(torch.isfinite(goal_xy).all()):
            raise ValueError("goal_xy must be finite [B,2] current-ego metres")
    with torch.autocast(device_type=xy.device.type, enabled=False):
        safe_xy = torch.where(valid[..., None, None], xy.float(), 0.)
        vel = torch.diff(safe_xy, dim=2, prepend=torch.zeros_like(safe_xy[:, :, :1])) * 2.
        # Keep the explicit subtraction in the original helper, including zeros.
        if status is None:
            status8 = torch.zeros((b, 8), device=xy.device, dtype=torch.float32)
        else:
            status8 = torch.as_tensor(status, device=xy.device, dtype=torch.float32)
            if status8.shape != (b, 8) or not bool(torch.isfinite(status8[:, 4:8]).all()):
                raise ValueError("status must be [B,8] with finite causal slots 4:8")
        relative_vel = (vel - status8[:, None, None, 4:6]).reshape(b, k, 12)
        acceleration = (torch.diff(vel, dim=2) * (2. / 3.)).reshape(b, k, 10)
        state = torch.cat((status8[:, 4:6] / 20., status8[:, 6:8] / 3.), -1)
        state = state[:, None].expand(b, k, 4)
        if goal_xy is None:
            goal = torch.zeros((b, k, 4), device=xy.device, dtype=torch.float32)
            present = torch.zeros((b, k, 1), device=xy.device, dtype=torch.float32)
        else:
            goal = torch.cat((goal_xy[:, None].expand(b, k, 2) / 50.,
                              (safe_xy[:, :, -1] - goal_xy[:, None]) / 50.), -1)
            present = torch.ones((b, k, 1), device=xy.device, dtype=torch.float32)
        safe_scores = torch.where(valid, scores.float(), 0.)
        mean = safe_scores.sum(-1, keepdim=True) / valid.sum(-1, keepdim=True)
        features = torch.cat((relative_vel, acceleration, state, goal,
                              (safe_scores - mean)[..., None], present), -1)
        return torch.where(valid[..., None], features, 0.)


class CandidateTokenCapture(nn.Module):
    """Wrap an already loaded original C model; scoped hook, immutable rows.

    With freeze_base=True, base parameters are frozen and every forward is under
    no_grad in eval mode. With freeze_base=False and detach_tokens=False, both
    ordinary base outputs and captured tokens retain their autograd graphs.
    Never call the owned base concurrently outside this wrapper.
    """
    def __init__(self, model, *, detach_tokens=True, freeze_base=True):
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an already loaded original C module")
        head = getattr(model, "_trajectory_head", None)
        if head is None or not isinstance(getattr(head, "traj_vocab", None), torch.Tensor):
            raise TypeError("Original model must expose its public _trajectory_head")
        layers = getattr(getattr(head, "decoder", None), "layers", None)
        if layers is None or len(layers) < 1 or not isinstance(getattr(layers[-1], "traj_mlp", None), nn.Module):
            raise TypeError("Expected final public decoder traj_mlp")
        if head.traj_vocab.ndim != 4 or head.traj_vocab.shape[-2:] != (8, 3):
            raise ValueError("Expected precomposed bank [P,V,8,3]")
        self.base = model
        self.detach_tokens = bool(detach_tokens)
        self.freeze_base = bool(freeze_base)
        self._capture_lock = threading.Lock()
        if self.freeze_base:
            self.base.requires_grad_(False)
            self.base.eval()

    @property
    def _trajectory_head(self):
        return self.base._trajectory_head

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_base:
            self.base.eval()
        return self

    def __getstate__(self):
        state = super().__getstate__().copy()
        state.pop("_capture_lock", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._capture_lock = threading.Lock()

    def _verify_alignment(self, output, tokens):
        xy, scores, valid = _candidate_contract(output)
        ids = _candidate_ids(output, scores)
        head = self._trajectory_head
        bank = head.traj_vocab
        path_ids, velocity_ids = output["path_ids"], output["velocity_ids"]
        for label, value, upper in (("path", path_ids, bank.shape[0]),
                                    ("velocity", velocity_ids, bank.shape[1])):
            if (not isinstance(value, torch.Tensor) or value.ndim != 2
                    or value.shape[0] != len(xy) or value.dtype != torch.int64
                    or value.device != xy.device or bool((value < 0).any()) or bool((value >= upper).any())):
                raise ValueError(f"Malformed final {label} IDs")
        expected_ids = (path_ids[:, :, None] * bank.shape[1] + velocity_ids[:, None, :]).flatten(1, 2)
        if not torch.equal(ids, expected_ids):
            raise ValueError("Candidate order is not the final path-major/velocity-minor token order")
        if (tokens.shape != (*ids.shape, 256) or tokens.device != xy.device
                or not tokens.is_floating_point() or not bool(torch.isfinite(tokens).all())):
            raise ValueError("Captured final trajectory tokens must be finite [B,K,256]")
        flat_bank = bank.flatten(0, 1)
        if not torch.equal(xy, flat_bank[ids, :6, :2]):
            raise ValueError("Completed candidate coordinates differ from immutable bank rows")
        expected_valid = head.traj_mask[path_ids[:, :, None], velocity_ids[:, None, :], :6].bool().all(-1).flatten(1, 2)
        if not torch.equal(valid, expected_valid):
            raise ValueError("Candidate validity differs from the fixed bank")
        winner = scores.masked_fill(~valid, -torch.inf).argmax(-1)
        batch = torch.arange(len(xy), device=xy.device)
        if (not torch.equal(output["selected_candidate_id"], ids[batch, winner])
                or not torch.equal(output["trajectory"], xy[batch, winner])):
            raise ValueError("Original C selected output does not match its completed scores/IDs")

    def forward(self, **original_inputs):
        if not self._capture_lock.acquire(blocking=False):
            raise RuntimeError("CandidateTokenCapture forbids concurrent/reentrant forward")
        hook = None
        captured = []
        try:
            def read_token(_module, args):
                if captured:
                    raise RuntimeError("Final traj_mlp ran more than once in one forward")
                if len(args) != 1 or not isinstance(args[0], torch.Tensor):
                    raise ValueError("Expected a single trajectory token tensor before traj_mlp")
                captured.append(args[0].detach() if self.detach_tokens else args[0])
            hook = self._trajectory_head.decoder.layers[-1].traj_mlp.register_forward_pre_hook(read_token)
            with torch.no_grad() if self.freeze_base else nullcontext():
                output = self.base(**original_inputs)
            if len(captured) != 1:
                raise RuntimeError("Original model did not execute exactly one final traj_mlp")
            if "candidate_tokens" in output:
                raise ValueError("Original output already contains candidate_tokens")
            self._verify_alignment(output, captured[0])
            return {**output, "candidate_tokens": captured[0]}
        finally:
            if hook is not None:
                hook.remove()
            self._capture_lock.release()


class SceneResidualSelector(nn.Module):
    """Matched real/zero-token, FP32 residual scorer over completed candidates.

    A valid output mapping needs only candidate_xy, candidate_ids, scores,
    candidate_tokens and optionally candidate_valid. Every existing output field
    is retained unless it is one of scores/trajectory/selected_candidate_id.
    Input tokens can be BF16, FP16, or FP32; feature/head arithmetic is FP32.
    """
    def __init__(self, *, mode="real", lead_dim=0):
        super().__init__()
        if mode not in ("real", "zero"):
            raise ValueError("mode must be real or zero")
        self.mode = mode
        # Per-row scene context (lead vehicle) broadcast over candidates.
        self.lead_dim = int(lead_dim)
        self.token_projection = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 64), nn.GELU())
        self.score_head = nn.Sequential(nn.Linear(96 + self.lead_dim, 128), nn.ReLU(),
                                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def forward(self, output, goal_xy=None, status=None, lead=None):
        xy, old_scores, valid = _candidate_contract(output)
        ids = _candidate_ids(output, old_scores)
        tokens = output["candidate_tokens"]
        if (not isinstance(tokens, torch.Tensor) or tokens.shape != (*ids.shape, 256)
                or tokens.device != xy.device or not tokens.is_floating_point()
                or not bool(torch.isfinite(tokens[valid]).all())):
            raise ValueError("candidate_tokens must have finite valid [B,K,256] values")
        if any(parameter.dtype != torch.float32 or parameter.device != xy.device for parameter in self.parameters()):
            raise TypeError("SceneResidualSelector parameters must be FP32 on the candidate device")
        features32 = build_features32(output, goal_xy, status)
        with torch.autocast(device_type=xy.device.type, enabled=False):
            safe_tokens = torch.where(valid[..., None], tokens.float(), 0.)
            if self.mode == "zero":
                safe_tokens = torch.zeros_like(safe_tokens)
            scene = self.token_projection(safe_tokens)
            parts = [features32, scene]
            if self.lead_dim:
                if lead is None or lead.shape != (features32.shape[0], self.lead_dim):
                    raise ValueError("lead must be [B,lead_dim] when lead_dim is set")
                parts.append(lead.float()[:, None].expand(-1, features32.shape[1], -1))
            elif lead is not None:
                raise ValueError("lead given but the head was built without lead_dim")
            residual = self.score_head(torch.cat(parts, -1)).squeeze(-1)
            residual = torch.where(valid, residual, 0.)
            scores = old_scores.float() + residual
            winner = scores.masked_fill(~valid, -torch.inf).argmax(-1)
        batch = torch.arange(len(xy), device=xy.device)
        result = dict(output)
        result.update(scores=scores, trajectory=xy[batch, winner], selected_candidate_id=ids[batch, winner],
                      scene_score_residual=residual, old_final_scores=old_scores)
        return result
