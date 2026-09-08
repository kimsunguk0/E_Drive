"""Small late residual used only by the frozen-body P9 diagnostic."""
from __future__ import annotations

import torch
from torch import nn


STATE_SCALE = (10., 5., 3., 3., .5, 1.)
HISTORY_SCALE = (10., 5., 1., 1.)


class LateStateResidual(nn.Module):
    """Shared 151->256->128->2 residual over six frozen decoder features."""

    def __init__(self):
        super().__init__()
        self.image_norm = nn.LayerNorm(128)
        self.mlp = nn.Sequential(nn.Linear(151, 256), nn.GELU(),
                                 nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("waypoint_time_over_3", torch.arange(1, 7, dtype=torch.float32) / 6.)
        self.register_buffer("compact_scale", torch.tensor(
            STATE_SCALE + HISTORY_SCALE * 4, dtype=torch.float32))

    def forward(self, decoder_features, compact22):
        if decoder_features.ndim != 3 or decoder_features.shape[1:] != (6, 128):
            raise ValueError("decoder_features must be [B,6,128]")
        if compact22.shape != (len(decoder_features), 22):
            raise ValueError("compact22 must be [B,22]")
        if not torch.isfinite(decoder_features).all() or not torch.isfinite(compact22).all():
            raise ValueError("P9 inputs must be finite")
        # This complete diagnostic branch is FP32 even under the parent's BF16
        # encoder autocast. The returned residual is in physical metres.
        with torch.autocast(device_type=decoder_features.device.type, enabled=False):
            image = self.image_norm(decoder_features.float())
            compact = compact22.float() / self.compact_scale.float()
            compact = compact[:, None].expand(-1, 6, -1)
            time = self.waypoint_time_over_3.float()[None, :, None].expand(len(image), -1, -1)
            return self.mlp(torch.cat((image, compact, time), -1))


def compact22(predicted_state, predicted_history, *, arm, gt_state=None, gt_history=None):
    """Return fixed A/B/C inputs; C preserves predicted raw stop logit."""
    if predicted_state.ndim != 2 or predicted_state.shape[1] != 6:
        raise ValueError("predicted_state must be [B,6]")
    if predicted_history.shape != (len(predicted_state), 4, 4):
        raise ValueError("predicted_history must be [B,4,4]")
    predicted = torch.cat((predicted_state, predicted_history.flatten(1)), -1)
    if arm == "no_compact":
        return torch.zeros_like(predicted)
    if arm == "predicted22":
        return predicted
    if arm != "gt21":
        raise ValueError("arm must be no_compact, predicted22, or gt21")
    if gt_state is None or gt_state.shape != predicted_state.shape:
        raise ValueError("gt21 requires GT state [B,6]")
    if gt_history is None or gt_history.shape != predicted_history.shape:
        raise ValueError("gt21 requires GT history [B,4,4]")
    state = predicted_state.clone()
    state[:, :5] = gt_state[:, :5].to(device=state.device, dtype=state.dtype)
    result = torch.cat((state, gt_history.to(device=state.device, dtype=state.dtype).flatten(1)), -1)
    if not torch.equal(result[:, 5], predicted_state[:, 5]):
        raise ValueError("gt21 changed predicted raw stop logit")
    return result


class LateResidualP7Adapter(nn.Module):
    """Legal raw-image A/B adapter; privileged GT21 is deliberately unavailable."""

    def __init__(self, parent: nn.Module, arm: str):
        super().__init__()
        if arm not in ("no_compact", "predicted22"):
            raise ValueError("Deployment adapter supports only legal no_compact/predicted22 arms")
        self.parent = parent
        self.parent.requires_grad_(False)
        self.parent.eval()
        self.arm = arm
        self.residual = LateStateResidual()

    def train(self, mode: bool = True):
        """Train only the new residual while the inherited P7-C body stays frozen."""
        super().train(mode)
        self.parent.eval()
        return self

    def forward(self, images, history_images, lidar2img, history_transforms,
                time_offsets, goal_xy):
        parts = self.parent.forward_parts(images, history_images, lidar2img,
                                          history_transforms, time_offsets, goal_xy)
        captured = []
        handle = self.parent.planner.xy_head.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0]))
        try:
            base_plan = self.parent.plan_from_features(
                parts["scene_features"], parts["motion_features"],
                parts["state_hat"], parts["history_hat"])
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape[1:] != (6, 128):
            raise RuntimeError("Parent xy_head input capture failed")
        status = compact22(parts["state_hat"], parts["history_hat"], arm=self.arm)
        corrected = base_plan + self.residual(captured[0], status)
        return {**parts, "plan_abs": corrected}
