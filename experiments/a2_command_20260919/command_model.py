"""Semantic navigation condition in the shared scene query, never a value/token."""
import torch
from torch import nn
from torch.nn import functional as F
from command_common import ARM, A2NominalModel
from command_data import LABELS

COMMAND_PREFIX = 'shared_command_query.'

class SharedCommandQuery(nn.Module):
    def __init__(self):
        super().__init__()
        # 192 new parameters, no random initialization / no RNG-stream changes.
        self.weight = nn.Parameter(torch.zeros(32, len(LABELS)))
    def forward(self, query, command):
        if query.ndim != 3 or query.shape[-1] != 32 or command.shape != (len(query), len(LABELS)):
            raise ValueError('Expected query [B,N,32] and provided command [B,6]')
        if command.device != query.device or not command.is_floating_point():
            raise ValueError('Command device/dtype mismatch')
        if not bool(((command == 0) | (command == 1)).all()) or not bool((command.sum(-1) == 1).all()):
            raise ValueError('Provided command must be finite and one-hot')
        with torch.autocast(device_type=query.device.type, enabled=False):
            result = query.float() + F.linear(command.float(), self.weight.float())[:, None, :]
        return result.to(query.dtype)

class CommandA2Model(A2NominalModel):
    VALID_ARMS = {**A2NominalModel.VALID_ARMS, ARM: 0}
    def __init__(self, config, *, arm=ARM):
        if arm != ARM:
            raise ValueError(arm)
        super().__init__(config, arm='A2-BASE-NOM')
        self.shared_command_query = SharedCommandQuery()
        self._command_context = None
        def condition(_module, _args, output):
            if self._command_context is None:
                raise RuntimeError('Current command context is missing')
            return self.shared_command_query(output, self._command_context)
        handle = self.scene_encoder.query_context.register_forward_hook(condition)
        object.__setattr__(self, '_command_hook_handle', handle)
    def forward(self, images, history_images, lidar2img, history_transforms,
                time_offsets, goal_xy, motion_current=None, motion_history=None,
                provided_status5=None, provided_command=None):
        if provided_command is None or self._command_context is not None:
            raise ValueError('Missing command or reentrant command forward')
        self._command_context = provided_command
        try:
            return super().forward(images, history_images, lidar2img, history_transforms,
                time_offsets, goal_xy, motion_current, motion_history, provided_status5)
        finally:
            self._command_context = None
