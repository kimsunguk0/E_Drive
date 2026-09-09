"""Opt-in factorized path/velocity planning head.

This experimental head consumes the existing planner decoder features.  It has
no raw goal, provided-status, pose, command, label, or image input of its own.
"""
from __future__ import annotations

import math
import types
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


P_COUNT = 512
V_COUNT = 128
NONZERO_V_COUNT = V_COUNT - 1
K_COUNT = 1 + P_COUNT * NONZERO_V_COUNT
WAYPOINTS = 6
CHANNELS = 128
KEY_CHANNELS = 32
RESIDUAL_RANK = 4
RESIDUAL_LIMIT_METRES = .5


class FactorizedPVHead(nn.Module):
    """Score all P×V rows with compact keys and optionally complete residuals."""

    def __init__(self, bank_prefix6: Tensor, path_descriptor: Tensor,
                 velocity_descriptor: Tensor, path_id: Tensor, velocity_id: Tensor,
                 *, residual: bool):
        super().__init__()
        if bank_prefix6.shape != (K_COUNT, WAYPOINTS, 2):
            raise ValueError("bank prefix must be [65025,6,2]")
        if path_descriptor.shape != (P_COUNT, 30):
            raise ValueError("path descriptor must be [512,30]")
        if velocity_descriptor.shape != (V_COUNT, 11):
            raise ValueError("velocity descriptor must be [128,11]")
        if path_id.shape != (K_COUNT,) or velocity_id.shape != (K_COUNT,):
            raise ValueError("flattened P/V ids must be [65025]")
        if (path_id.dtype not in (torch.int32, torch.int64)
                or velocity_id.dtype not in (torch.int32, torch.int64)):
            raise ValueError("P/V ids must be integral")
        if (int(path_id[0]) != -1 or int(velocity_id[0]) != 0
                or torch.count_nonzero(bank_prefix6[0])):
            raise ValueError("candidate row0 must be the dedicated exact-zero row")
        p = path_id[1:].cpu().long()
        v = velocity_id[1:].cpu().long()
        pair_index = p * NONZERO_V_COUNT + (v - 1)
        if (p.min() != 0 or p.max() != P_COUNT - 1 or v.min() != 1
                or v.max() != V_COUNT - 1
                or not torch.equal(pair_index.sort().values, torch.arange(P_COUNT * NONZERO_V_COUNT))):
            raise ValueError("candidate ids must contain every P×nonzero-V pair exactly once")
        for name, value in (("bank", bank_prefix6), ("path", path_descriptor),
                            ("velocity", velocity_descriptor)):
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} tensor must be finite float32")
        self.residual_enabled = bool(residual)
        self.register_buffer("bank_prefix6", bank_prefix6.detach().clone(), persistent=True)
        self.register_buffer("path_descriptor", path_descriptor.detach().clone(), persistent=True)
        self.register_buffer("velocity_descriptor", velocity_descriptor.detach().clone(), persistent=True)
        self.register_buffer("path_id", path_id.detach().clone().long(), persistent=True)
        self.register_buffer("velocity_id", velocity_id.detach().clone().long(), persistent=True)
        self.context = nn.Sequential(nn.LayerNorm(WAYPOINTS * CHANNELS),
                                     nn.Linear(WAYPOINTS * CHANNELS, CHANNELS), nn.GELU(),
                                     nn.Linear(CHANNELS, KEY_CHANNELS))
        self.path_key = nn.Sequential(nn.LayerNorm(30), nn.Linear(30, 64), nn.GELU(),
                                      nn.Linear(64, KEY_CHANNELS))
        self.velocity_key = nn.Sequential(nn.LayerNorm(11), nn.Linear(11, 32), nn.GELU(),
                                          nn.Linear(32, KEY_CHANNELS))
        self.stop_key = nn.Parameter(torch.randn(1, KEY_CHANNELS) * .02)
        # 20*sigmoid(0)=10; the positive cosine scale is bounded by 20
        # (and the resulting cosine logits lie approximately in [-20,20]).
        self.logit_scale_raw = nn.Parameter(torch.zeros(()))
        if self.residual_enabled:
            self.residual_context = nn.Linear(KEY_CHANNELS, RESIDUAL_RANK * WAYPOINTS * 2)
            self.residual_candidate = nn.Linear(KEY_CHANNELS, RESIDUAL_RANK, bias=True)
            nn.init.zeros_(self.residual_candidate.weight)
            nn.init.zeros_(self.residual_candidate.bias)

    def joint_keys(self) -> Tensor:
        with torch.autocast(device_type=self.path_descriptor.device.type, enabled=False):
            path = F.normalize(self.path_key(self.path_descriptor.float()), dim=-1, eps=1e-6)
            velocity = F.normalize(self.velocity_key(self.velocity_descriptor[1:].float()),
                                   dim=-1, eps=1e-6)
            combined = path[:, None] + velocity[None] + path[:, None] * velocity[None]
            combined = F.normalize(combined, dim=-1, eps=1e-6)
            combined = combined[self.path_id[1:], self.velocity_id[1:] - 1]
            stop = F.normalize(self.stop_key.float(), dim=-1, eps=1e-6)
            return torch.cat([stop, combined], 0)

    def forward(self, decoded: Tensor) -> Mapping[str, Tensor]:
        if decoded.shape[-2:] != (WAYPOINTS, CHANNELS) or decoded.ndim != 3:
            raise ValueError("planner decoder feature must be [B,6,128]")
        with torch.autocast(device_type=decoded.device.type, enabled=False):
            z = F.normalize(self.context(decoded.float().flatten(1)), dim=-1, eps=1e-6)
            keys = self.joint_keys()
            scale = 20. * self.logit_scale_raw.float().sigmoid()
            logits = scale * torch.einsum("bd,kd->bk", z, keys)
            base = self.bank_prefix6.float()
            if self.residual_enabled:
                context_factor = self.residual_context(z).reshape(
                    len(z), RESIDUAL_RANK, WAYPOINTS * 2)
                candidate_factor = self.residual_candidate(keys)
                raw = torch.einsum("brd,kr->bkd", context_factor, candidate_factor)
                residual = RESIDUAL_LIMIT_METRES * raw.tanh()
                candidates = base[None] + residual.reshape(len(z), K_COUNT, WAYPOINTS, 2)
            else:
                candidates = base[None].expand(len(z), -1, -1, -1)
            selected = logits.argmax(-1)
            rows = torch.arange(len(z), device=z.device)
            post = candidates[rows, selected]
            pre = base[selected]
        return {"pv_logits": logits, "pv_base_candidates": base,
                "pv_candidates": candidates, "pv_selected_id": selected,
                "pv_pre_plan": pre, "pv_post_plan": post,
                "pv_logit_scale": scale}


def install_factorized_pv_head(model: nn.Module, head: FactorizedPVHead) -> nn.Module:
    """Install an exception-safe decoder capture and replace only final plan output."""
    if hasattr(model, "factorized_pv_head"):
        raise ValueError("factorized P/V head is already installed")
    xy_head = getattr(getattr(model, "planner", None), "xy_head", None)
    if not isinstance(xy_head, nn.Module):
        raise ValueError("parent planner xy_head is missing")
    model.add_module("factorized_pv_head", head)
    captured: list[Tensor] = []

    def capture(_module, args):
        if captured:
            raise RuntimeError("planner decoder capture is reentrant or stale")
        if len(args) != 1:
            raise RuntimeError("unexpected xy_head input")
        captured.append(args[0])

    handle = xy_head.register_forward_pre_hook(capture)
    original_forward = model.forward

    def experimental_forward(self, *args, **kwargs):
        if captured:
            raise RuntimeError("factorized P/V capture was not cleared")
        try:
            parent = original_forward(*args, **kwargs)
            if len(captured) != 1:
                raise RuntimeError("parent planner must expose exactly one decoder feature")
            pv = head(captured[0])
            evidence = getattr(self, "_pv_eval_capture", None)
            if evidence is not None:
                if not isinstance(evidence, list) or self.training:
                    raise RuntimeError("P/V evaluation capture requires eval mode and a list")
                probability = pv["pv_logits"].float().softmax(-1)
                entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
                evidence.append({name: pv[name].detach() for name in
                                 ("pv_selected_id", "pv_pre_plan", "pv_post_plan")} |
                                {"pv_entropy": entropy.detach()})
            # Keep the parent output keys plus opt-in training/evidence tensors.
            return {**parent, **pv, "plan_abs": pv["pv_post_plan"]}
        finally:
            captured.clear()

    model.forward = types.MethodType(experimental_forward, model)
    object.__setattr__(model, "_factorized_pv_hook_handle", handle)
    object.__setattr__(model, "_factorized_pv_capture", captured)
    return model


def pv_head_state(model: nn.Module) -> dict[str, Tensor]:
    prefix = "factorized_pv_head."
    state = {name: value for name, value in model.state_dict().items()
             if name.startswith(prefix)}
    if not state:
        raise ValueError("factorized P/V head state is absent")
    return state
