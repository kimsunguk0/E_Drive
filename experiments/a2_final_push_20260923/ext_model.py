"""H4-PROGRESS with K image-generated 5 s candidates and goal-endpoint selection.

Compliance shape (OPEN_ISSUE.md + 2026-09-23 open-chat answers):
* One network: images -> shared scene/motion -> planner -> K complete candidate
  trajectories, gradient connected end to end. No separate selector network.
* The planner is unchanged in what it reads (scene, motion, image-predicted
  state/history). Goal and provided status still enter only the shared scene
  attention query, exactly as in the parent.
* Each candidate is a 10-point trajectory to 5.0 s. The goal point (+50 frame)
  is used only to pick the candidate whose 5.0 s endpoint is closest to it --
  "goal point만을 활용하여 이미 생성된 궤적의 좌표를 변형하지 않고 선택".
  No learned module sees the goal for selection; no status/history is used for
  selection; the chosen candidate's first six points are returned unchanged.
"""
from __future__ import annotations
import torch
from torch import nn

from progress_model import H4ProgressModel, ProgressHeadingPlanner, PROGRESS, compose_progress

K = 6
EXT = 4                       # 3.5, 4.0, 4.5, 5.0 s
SPREAD = 0.03                 # initial per-mode progress spread at 5.0 s (raw length units)


class ModeExtPlanner(ProgressHeadingPlanner):
    def __init__(self, config):
        super().__init__(config)
        c = config.channels
        # Anchor-like initial diversity: mode k progresses slightly slower/faster,
        # growing with horizon. Learned from training data, image-conditioned.
        ramp = torch.arange(1, 7 + EXT, dtype=torch.float32)[None] / (6 + EXT)
        self.mode_length_bias = nn.Parameter(torch.linspace(-SPREAD, SPREAD, K)[:, None] * ramp)
        self.mode_embed = nn.Parameter(torch.zeros(K, 1, c))
        self.ext_queries = nn.Parameter(torch.zeros(EXT, c))
        # The 5 s extension starts as a straight continuation of interval 6 and
        # learns a residual from its own decoded tokens (zero-initialised).
        self.ext_delta = nn.Linear(c, 2)
        nn.init.zeros_(self.ext_delta.weight); nn.init.zeros_(self.ext_delta.bias)

    def init_ext_from_parent(self):
        with torch.no_grad():
            self.ext_queries.copy_(self.waypoint_queries[-1:].expand(EXT, -1))

    def forward(self, scene_features, motion_features, predicted_state, predicted_history,
                motion_pair_features):
        b = scene_features.shape[0]
        with torch.autocast(device_type=scene_features.device.type, enabled=False):
            scene = scene_features.float() + self.scene_position(self.scene_xy.float())[None]
            motion = motion_features.float() + self.motion_type.float()
            state = predicted_state.float() / self.state_scale
            history = (predicted_history.float() / self.history_scale).flatten(1)
            numeric = torch.cat([state, history], -1)
            if not self.config.state_on:
                numeric = torch.zeros_like(numeric)
            memory = torch.cat([scene, motion, self.state_projection(numeric)[:, None]], 1)
            m = memory.shape[1]
            memory = memory[:, None].expand(b, K, m, memory.shape[-1]).reshape(b * K, m, -1)
            pairs = motion_pair_features.float()
            pairs = pairs[:, None].expand(b, K, *pairs.shape[1:]).reshape(b * K, *pairs.shape[1:])
            q6 = (self.waypoint_queries.float()[None] + self.mode_embed.float())      # K,6,c
            qe = (self.ext_queries.float()[None] + self.mode_embed.float())           # K,4,c
            q6 = q6[None].expand(b, -1, -1, -1).reshape(b * K, 6, -1)
            qe = qe[None].expand(b, -1, -1, -1).reshape(b * K, EXT, -1)
            # The six metric waypoints decode exactly as in the parent (own
            # self-attention set); the 5 s extension decodes as a separate set so
            # it cannot perturb the first six.
            d6 = self.decoder(q6, memory)
            d6 = d6 + self.temporal_read(d6, pairs)
            de = self.decoder(qe, memory)
            raw6 = self.xy_head(d6.float())                                            # bK,6,2
            raw = torch.cat([raw6, raw6[:, 5:6].expand(-1, EXT, -1) + self.ext_delta(de.float())], 1)
            bias = self.mode_length_bias.float()[None].expand(b, -1, -1).reshape(b * K, 6 + EXT)
            raw = torch.stack([raw[..., 0] + bias, raw[..., 1]], -1)
            plan = compose_progress(raw, self.progress_units)                          # bK,10,2
            return plan.reshape(b, K, 6 + EXT, 2)


def select_by_goal(modes, goal_xy):
    """Index the candidate whose 5.0 s endpoint is nearest the goal point."""
    with torch.autocast(device_type=modes.device.type, enabled=False):
        dist = torch.linalg.norm(modes[:, :, -1].float() - goal_xy.float()[:, None], dim=-1)
        pick = dist.argmin(1)
    return pick


class ExtModel(H4ProgressModel):
    def __init__(self, config, *, arm=PROGRESS):
        super().__init__(config, arm=arm)
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(2026092301)
            planner = ModeExtPlanner(config)
        state = planner.state_dict()
        state.update(self.planner.state_dict())
        planner.load_state_dict(state, strict=True)
        planner.init_ext_from_parent()
        self.planner = planner

    def forward(self, **inputs):
        out = super().forward(**inputs)
        modes = out['plan_abs']
        pick = select_by_goal(modes, inputs['goal_xy'])
        chosen = modes[torch.arange(modes.shape[0], device=modes.device), pick]
        return {**out, 'plan_modes': modes, 'selected_mode': pick, 'plan_abs': chosen[:, :6]}


NEW_KEYS = ('planner.mode_length_bias', 'planner.mode_embed', 'planner.ext_queries',
            'planner.ext_delta.weight', 'planner.ext_delta.bias')
