"""The audit must detect actual bypasses, not compare artifacts with themselves."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_motiondrive_v2 import run_audit, synthetic_batch, validate_batch
from benchmark_motiondrive_v2 import summarize_ms


class TinyModel(nn.Module):
    def __init__(self, goal_on=True, leak_goal_motion=False, cache_goal=False, detach_scene=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75))
        self.goal_on = goal_on
        self.leak_goal_motion = leak_goal_motion
        self.cache_goal = cache_goal
        self.detach_scene = detach_scene
        self.goal_cache = 0.

    def plan_from_features(self, scene_features, motion_features, predicted_state, predicted_history):
        scene = scene_features.detach() if self.detach_scene else scene_features
        x = scene.mean((1, 2)) + motion_features.mean((1, 2))
        if self.cache_goal:
            x = x + self.goal_cache
        return x[:, None, None].repeat(1, 6, 2).float()

    def forward(self, images, history_images, lidar2img, history_transforms, time_offsets, goal_xy):
        b = len(images)
        scene = images.mean((1, 2, 3, 4))[:, None, None].repeat(1, 3, 4) * self.scale
        scene = scene + (goal_xy.mean(1)[:, None, None] * .001 if self.goal_on else 0.)
        mot = (history_images.mean((2, 3, 4)) * time_offsets).sum(1)
        if self.leak_goal_motion:
            mot = mot + goal_xy.mean(1)
        motion = mot[:, None, None].repeat(1, 4, 4) * self.scale
        state = motion.mean((1, 2))[:, None].repeat(1, 6).float()
        history = motion.mean((1, 2))[:, None, None].repeat(1, 4, 4).float()
        self.goal_cache = goal_xy.mean(1)
        return {"plan_abs": self.plan_from_features(scene, motion, state, history),
                "scene_features": scene, "motion_features": motion,
                "state_hat": state, "history_hat": history,
                "occ_logits": scene.mean(-1)[:, None, :, None],
                "lane_logits": scene.mean(-1)[:, None, :, None] * 2}


@pytest.fixture
def batch():
    return synthetic_batch(image_hw=(8, 12), history_hw=(4, 6))


def test_audit_allows_goal_conditioned_scene_with_goal_blind_motion(batch):
    report = run_audit(TinyModel(), batch, precision="fp32")
    assert report["all_executed_checks_pass"], report["checks"]
    assert not report["measurements"]["goal_changes"]["plan_abs"]["bitwise_equal"]


def test_audit_goal_off_requires_whole_forward_invariance(batch):
    report = run_audit(TinyModel(goal_on=False), batch, precision="fp32", goal_on=False)
    assert report["checks"]["goal_off_whole_forward_bitwise"]


def test_audit_detects_goal_leak_into_motion(batch):
    report = run_audit(TinyModel(leak_goal_motion=True), batch, precision="fp32")
    assert not report["checks"]["goal_counterfactual_motion_features_bitwise"]


def test_audit_detects_hidden_goal_cache_in_planner(batch):
    report = run_audit(TinyModel(cache_goal=True), batch, precision="fp32")
    assert not report["checks"]["fixed_features_planner_goal_cache_invariant"]


def test_audit_detects_nominal_but_disconnected_scene_feature(batch):
    report = run_audit(TinyModel(detach_scene=True), batch, precision="fp32")
    assert not report["checks"]["plan_abs_uses_scene_features"]


def test_output_float_cast_cannot_hide_bf16_final_head_input(batch):
    class CastOnlyModel(TinyModel):
        def __init__(self):
            super().__init__()
            self.planner = nn.Module()
            self.planner.xy_head = nn.Identity()

        def plan_from_features(self, scene_features, motion_features, predicted_state, predicted_history):
            x = scene_features.mean((1, 2)) + motion_features.mean((1, 2))
            rounded = self.planner.xy_head(x.bfloat16())
            return rounded.float()[:, None, None].repeat(1, 6, 2)

    report = run_audit(CastOnlyModel(), batch, precision="fp32")
    assert report["checks"]["plan_abs_fp32"]
    assert not report["checks"]["planner.xy_head_input_fp32"]


def test_batch_rejects_wrong_history_camera_shape(batch):
    batch["history_images"] = batch["history_images"].unsqueeze(2)
    with pytest.raises(ValueError, match="history_images"):
        validate_batch(batch)


def test_benchmark_summary_preserves_tail_latency():
    summary = summarize_ms([10., 10., 10., 100.])
    assert summary["median_ms"] == 10.
    assert summary["p99_ms"] > 95.
    with pytest.raises(ValueError):
        summarize_ms([])
