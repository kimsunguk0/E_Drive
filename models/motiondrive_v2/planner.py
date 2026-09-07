"""Goal-free planner signature; reads shared continuous visual representations."""
from __future__ import annotations

import torch
from torch import nn

from .scene_encoder import grid_centers


class DirectTrajectoryPlanner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config.channels
        self.waypoint_queries = nn.Parameter(torch.randn(6, c) * 0.02)
        self.scene_position = nn.Linear(2, c, bias=False)
        self.motion_type = nn.Parameter(torch.randn(1, 1, c) * 0.02)
        self.state_projection = nn.Sequential(nn.Linear(6 + config.n_history * 4, c),
                                              nn.GELU(), nn.Linear(c, c))
        self.register_buffer("scene_xy", grid_centers(config.grid_size, config.x_range,
                                                     config.y_range).reshape(-1, 2)
                             / torch.tensor([80.0, 64.0]))
        self.register_buffer("state_scale", torch.tensor([10.0, 5.0, 3.0, 3.0, 0.5, 1.0]))
        self.register_buffer("history_scale", torch.tensor([10.0, 5.0, 1.0, 1.0]))
        decoder_layer = nn.TransformerDecoderLayer(c, config.planner_heads, c * 4,
                                                   dropout=0.0, batch_first=True,
                                                   activation="gelu", norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.planner_layers)
        self.xy_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.GELU(), nn.Linear(c, 2))

    def forward(self, scene_features, motion_features, predicted_state, predicted_history):
        b = scene_features.shape[0]
        # Planner and coordinate head are intentionally FP32 in both train and
        # deployment. Backbone AMP precision does not leak into final projection.
        with torch.autocast(device_type=scene_features.device.type, enabled=False):
            scene = scene_features.float() + self.scene_position(self.scene_xy.float())[None]
            motion = motion_features.float() + self.motion_type.float()
            state = predicted_state.float() / self.state_scale
            history = (predicted_history.float() / self.history_scale).flatten(1)
            status = torch.cat([state, history], -1)
            if not self.config.state_on:
                status = torch.zeros_like(status)
            # Identical parameter counts/operations for S=ON and S=OFF.
            state_token = self.state_projection(status)[:, None]
            memory = torch.cat([scene, motion, state_token], 1)
            queries = self.waypoint_queries.float()[None].expand(b, -1, -1)
            decoded = self.decoder(queries, memory)
            return self.xy_head(decoded.float())
