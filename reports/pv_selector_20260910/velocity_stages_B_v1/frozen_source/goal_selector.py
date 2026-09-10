"""Optional goal conditioning for SparseDriveV2 fixed-bank selection.

Modified ETRI experiment adapter, 2026-09-10. The running public_model.py is
unchanged. A zero-initialized projection of current-ego goal XY / 50 metres is
added to its status embedding; all candidate coordinates remain base bank rows.
This implements a selection-only hypothesis, not an assertion of official approval.
"""
from __future__ import annotations

import threading

import torch
from torch import nn

try:
    from .public_model import PublicSparseDriveV2
except ImportError:
    from public_model import PublicSparseDriveV2


class GoalConditionedSelector(nn.Module):
    """Wrap an existing public selector without changing its tensors or forward.

    ``goal_xy`` is [B,2] in the same current-ego metre frame as the fixed bank.
    Omitting it calls the base unmodified. Its projection has 512 new trainable
    elements, initialized to zero, and no bias. The existing 8D causal status
    input is passed through unchanged.

    The temporary hook is removed in a finally block, including on exceptions.
    Concurrent or reentrant calls through the same wrapper are rejected. Calls
    from other threads directly into the base do not receive this call's goal.
    Do not nest/share multiple goal wrappers around the same base during a call.

    Save/load state_dict(), whose keys are base.<original key> plus
    goal_projection.weight; full-module pickling of the thread lock is unsupported.
    """
    def __init__(self, base: PublicSparseDriveV2, goal_scale: float = 50.0):
        super().__init__()
        if not isinstance(base._status_encoding, nn.Linear):
            raise TypeError("Expected the public linear status encoder")
        if base._status_encoding.in_features != 8 or base._status_encoding.out_features != 256:
            raise ValueError("Expected the unchanged public 8-to-256 status encoder")
        if not 0 < float(goal_scale) < float("inf"):
            raise ValueError("goal_scale must be finite and positive")
        self.base = base
        self.goal_scale = float(goal_scale)
        self.goal_projection = nn.Linear(2, 256, bias=False)
        self.goal_projection.to(device=base._status_encoding.weight.device,
                                dtype=base._status_encoding.weight.dtype)
        nn.init.zeros_(self.goal_projection.weight)
        self._forward_lock = threading.Lock()

    @classmethod
    def from_public_checkpoint(cls, checkpoint, bank_path=None, *, goal_scale=50.0, **base_kwargs):
        base, coverage = PublicSparseDriveV2.from_public_checkpoint(
            checkpoint, bank_path=bank_path, **base_kwargs)
        wrapper = cls(base, goal_scale=goal_scale)
        report = dict(coverage)
        report["goal_adapter"] = {"new_parameter_elements": 512, "initialization": "zeros, no bias",
                                  "normalization_metres": float(goal_scale),
                                  "injection": "add to base status embedding; fixed bank coordinates unchanged",
                                  "state_dict_base_prefix": "base."}
        return wrapper, report

    @property
    def _trajectory_head(self):
        """Read access for external loss code and fixed-bank identity checks."""
        return self.base._trajectory_head

    @property
    def _backbone(self):
        return self.base._backbone

    def forward(self, images, lidar2img, image_hw=None, status=None, goal_xy=None):
        goal = None
        if goal_xy is not None:
            goal = torch.as_tensor(goal_xy, device=images.device)
            if goal.shape != (images.shape[0], 2):
                raise ValueError("goal_xy must be [B,2] current-ego XY metres")
            if not torch.is_floating_point(goal):
                goal = goal.float()
            if not torch.isfinite(goal).all():
                raise ValueError("goal_xy must be finite")
        if not self._forward_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent or reentrant GoalConditionedSelector.forward is unsupported")
        handle = None
        try:
            if goal is not None:
                owner_thread = threading.get_ident()

                def add_goal(module, inputs, output):
                    if threading.get_ident() != owner_thread:
                        return output
                    normalized = goal.to(device=output.device, dtype=self.goal_projection.weight.dtype) / self.goal_scale
                    delta = self.goal_projection(normalized).to(dtype=output.dtype)
                    return output + delta

                handle = self.base._status_encoding.register_forward_hook(add_goal)
            return self.base(images=images, lidar2img=lidar2img, image_hw=image_hw, status=status)
        finally:
            if handle is not None:
                handle.remove()
            self._forward_lock.release()


GoalSelector = GoalConditionedSelector
