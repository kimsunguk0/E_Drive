import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_motiondrive_v2_bn_motion_pair import (CONDITIONS, EXPECTED_FRAMES, INIT_SUFFIX,
                                                   analyze_bn_pair, bn_buffer_comparison, validate_bn_pair)


def fake_reports():
    scenes = [f"scene{i:02d}" for i in range(37)]
    mapping = {scene: f"session{i % 11}" for i, scene in enumerate(scenes)}
    mapping.update(train0="train_session", val0="val_session")
    manifest = {"splits": {"train": ["train0"], "tune": scenes, "val": ["val0"], "historical_val": ["val0"]},
                "scene_to_session": mapping}
    args = dict(seed=0, phase="pretrain", batch=16, steps=1000, warmup=100, lr=1e-4, backbone_lr=1e-5,
                weight_decay=.01, precision="bf16", uncertainty=1, alpha_occ=.2, alpha_lane=.2, alpha_motion=.2,
                grad_clip=5, goal_on=0, state_on=0, motion_input_mode="low_feature", plan_output_scale=[1., 1.],
                eval_split="tune", train_stride=1, eval_stride=5, max_train_samples=0, max_eval_samples=0,
                init=INIT_SUFFIX, resume=None, train_scenes=None, eval_scenes=None, eval_only=False)
    config = dict(motion_input_mode="low_feature", goal_on=False, state_on=False, plan_output_scale=[1., 1.])
    reports = []
    for policy, normal, perturb in (("adaptive", 1., .01), ("fixed", .8, .2)):
        conditions = {}
        for condition in CONDITIONS:
            value = normal + (0 if condition == "normal" else perturb)
            conditions[condition] = {"n": 370, "n_scenes": 37, "n_sessions": 11,
                "per_scene": {scene: {"n": 10, "frames": EXPECTED_FRAMES, "session_id": mapping[scene],
                                      "vx_mae_m_s": value, "history_position_mae_mean4_m": value / 2} for scene in scenes}}
        checkpoint = {"path": f"{policy}/last.pth", "step": 1000, "sha256": policy,
                      "model_config": dict(config), "conditions": conditions,
                      "run_provenance": {"git_sha": "same_source", "initial_model_state_sha256": "same_state",
                          "initial_parameter_count": 1, "train_rows_sha256": "tr", "eval_rows_sha256": "ev",
                          "sample_order_sha256_at_checkpoint": "order", "loss_weights": {"plan": 0},
                          "load_report": {"common_checkpoint_sha256": "same_init"}, "model_config": dict(config),
                          "arguments": dict(args, bn_policy=policy, gpu=2 if policy == "adaptive" else 3, run_dir=policy)}}
        reports.append({"split_manifest_sha256": "split", "supervision_manifest_sha256": "sup",
                        "protocol": {"split": "tune", "n": 370, "frames_per_scene": EXPECTED_FRAMES,
                                     "expected_step": 1000, "checkpoints": ["last.pth"], "donor_mapping": {"a": "b"}},
                        "checkpoint_results": {"last": checkpoint}})
    return reports, manifest


def test_bn_pair_uses_same_low_feature_and_separates_temporal_effect():
    (adaptive, fixed), manifest = fake_reports()
    result = analyze_bn_pair(adaptive, fixed, manifest, "split", repeats=100)
    normal = result["normal_accuracy"]["vx_mae_m_s"]["primary_rawtime_session"]
    temporal = result["temporal_controls"]["vx_mae_m_s"]["reverse_history"]
    assert normal["mean_difference"] == pytest.approx(-.2)
    assert normal["n_clusters"] == 11
    assert temporal["fixed_advantage"]["primary_rawtime_session"]["mean_difference"] == pytest.approx(.2)
    assert temporal["difference_in_advantages_fixed_minus_adaptive"]["primary_rawtime_session"]["mean_difference"] == pytest.approx(.19)


@pytest.mark.parametrize("change", ["mode", "policy", "step", "best", "order", "lr", "extra_arg", "session"])
def test_bn_pair_rejects_additional_changes_or_mixed_evaluation(change):
    (adaptive, fixed), manifest = fake_reports()
    checkpoint = fixed["checkpoint_results"]["last"]
    provenance = checkpoint["run_provenance"]
    if change == "mode":checkpoint["model_config"]["motion_input_mode"] = "high_feature"
    elif change == "policy":provenance["arguments"]["bn_policy"] = "adaptive"
    elif change == "step":checkpoint["step"] = 750
    elif change == "best":checkpoint["path"] = "fixed/best.pth"
    elif change == "order":provenance["sample_order_sha256_at_checkpoint"] = "different"
    elif change == "lr":provenance["arguments"]["lr"] = 2e-4
    elif change == "extra_arg":provenance["arguments"]["unknown_training_change"] = True
    elif change == "session":checkpoint["conditions"]["normal"]["per_scene"]["scene00"]["session_id"] = "wrong"
    with pytest.raises(ValueError):
        validate_bn_pair(adaptive, fixed, manifest, "split")


def test_bn_buffers_and_affine_are_reported_separately():
    initial = {"backbone.bn.running_mean": torch.zeros(2), "backbone.bn.running_var": torch.ones(2),
               "backbone.bn.num_batches_tracked": torch.tensor(0), "backbone.bn.weight": torch.ones(2),
               "backbone.bn.bias": torch.zeros(2)}
    fixed = copy.deepcopy(initial)
    fixed["backbone.bn.weight"] += .1
    report = bn_buffer_comparison(initial, fixed)
    assert report["전체버퍼동일"]
    assert report["buffer_changed_count"] == 0
    assert report["affine_changed_count"] == 1
    adaptive = copy.deepcopy(fixed)
    adaptive["backbone.bn.running_mean"] += 1
    adaptive["backbone.bn.num_batches_tracked"] += 1
    report = bn_buffer_comparison(initial, adaptive)
    assert not report["전체버퍼동일"]
    assert report["buffer_changed_count"] == 2
    assert report["initial_buffer_sha256"] != report["last_buffer_sha256"]


def test_bn_buffer_audit_requires_matching_state_keys():
    with pytest.raises(ValueError, match="key"):
        bn_buffer_comparison({"bn.running_mean": torch.zeros(2)}, {})
    with pytest.raises(ValueError, match="buffer"):
        bn_buffer_comparison({"weight": torch.ones(2)}, {"weight": torch.ones(2)})
