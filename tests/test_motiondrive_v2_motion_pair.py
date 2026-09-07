import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_motiondrive_v2_motion_pair import (CONDITIONS, EXPECTED_FRAMES, analyze_pair,
                                                paired_cluster_bootstrap, validate_pair)


def fake_reports():
    scenes = [f"scene{i:02d}" for i in range(37)]
    mapping = {s: f"session{i % 11}" for i, s in enumerate(scenes)}
    mapping["train0"] = "train_session"
    mapping["val0"] = "val_session"
    manifest = {"splits": {"train": ["train0"], "tune": scenes, "val": ["val0"], "historical_val": ["val0"]},
                "scene_to_session": mapping}
    reports = []
    arguments = dict(seed=0, phase="pretrain", batch=16, steps=1000, warmup=100, lr=.0001, backbone_lr=.00001,
                     weight_decay=.01, precision="bf16", uncertainty=1, alpha_occ=.2, alpha_lane=.2, alpha_motion=.2, grad_clip=5)
    for mode, normal, perturb in (("high_feature", 1., .01), ("low_feature", .8, .2)):
        conditions = {}
        for condition in CONDITIONS:
            value = normal + (0 if condition == "normal" else perturb)
            conditions[condition] = {"per_scene": {s: {"n": 10, "frames": EXPECTED_FRAMES,
                "session_id": mapping[s], "vx_mae_m_s": value, "history_position_mae_mean4_m": value / 2} for s in scenes}}
        checkpoint = {"path": f"{mode}/last.pth", "step": 1000, "sha256": mode,
                      "model_config": {"motion_input_mode": mode, "channels": 128}, "conditions": conditions,
                      "run_provenance": {"initial_model_state_sha256": "same", "initial_parameter_count": 1,
                          "train_rows_sha256": "tr", "eval_rows_sha256": "ev", "sample_order_sha256_at_checkpoint": "order",
                          "loss_weights": {"plan": 0}, "arguments": arguments}}
        reports.append({"split_manifest_sha256": "split", "supervision_manifest_sha256": "sup",
                        "protocol": {"split": "tune", "n": 370, "frames_per_scene": EXPECTED_FRAMES,
                                     "expected_step": 1000, "donor_mapping": {"a": "b"}},
                        "checkpoint_results": {"last": checkpoint}})
    return reports, manifest


def test_cluster_bootstrap_preserves_counts_and_pairing():
    r = paired_cluster_bootstrap([1., 3., 10.], [1, 3, 10], ["a", "a", "b"], repeats=1000, seed=1)
    assert r["mean_difference"] == pytest.approx(110 / 14)
    assert r["n_clusters"] == 2
    assert r["ci95_percentile"] == pytest.approx([2.5, 10.])
    same = paired_cluster_bootstrap(np.full(37, -.2), np.full(37, 10), [str(i % 11) for i in range(37)], repeats=100)
    assert same["ci95_percentile"] == pytest.approx([-.2, -.2])


def test_normal_improvement_and_temporal_advantage_are_separate():
    (high, low), manifest = fake_reports()
    r = analyze_pair(high, low, manifest, "split", repeats=100)
    normal = r["normal_accuracy"]["vx_mae_m_s"]["primary_rawtime_session"]
    temporal = r["temporal_controls"]["vx_mae_m_s"]["repeat_current"]
    assert normal["mean_difference"] == pytest.approx(-.2)
    assert normal["n_clusters"] == 11
    assert temporal["low_feature_advantage"]["primary_rawtime_session"]["mean_difference"] == pytest.approx(.2)
    assert temporal["difference_in_advantages_low_minus_high"]["primary_rawtime_session"]["mean_difference"] == pytest.approx(.19)


def test_best_last_mixture_or_wrong_step_rejected():
    (high, low), manifest = fake_reports()
    low["checkpoint_results"]["last"]["path"] = "low/best.pth"
    with pytest.raises(ValueError, match="LAST step1000"):
        validate_pair(high, low, manifest, "split")
    low["checkpoint_results"]["last"]["path"] = "low/last.pth"
    low["checkpoint_results"]["last"]["step"] = 750
    with pytest.raises(ValueError, match="LAST step1000"):
        validate_pair(high, low, manifest, "split")


def test_session_or_sample_order_mismatch_rejected():
    (high, low), manifest = fake_reports()
    low["checkpoint_results"]["last"]["run_provenance"]["sample_order_sha256_at_checkpoint"] = "different"
    with pytest.raises(ValueError, match="공정성"):
        validate_pair(high, low, manifest, "split")
    low["checkpoint_results"]["last"]["run_provenance"]["sample_order_sha256_at_checkpoint"] = "order"
    low["checkpoint_results"]["last"]["conditions"]["normal"]["per_scene"]["scene00"]["session_id"] = "wrong"
    with pytest.raises(ValueError, match="세션"):
        validate_pair(high, low, manifest, "split")
