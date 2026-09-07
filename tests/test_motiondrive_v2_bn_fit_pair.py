import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_motiondrive_v2_bn_fit_pair import (checkpoint_policy_integrity, compare_bn_fit_reports,
                                               validate_pair_manifests)


def probe(errors, step=500):
    out = {}
    for split, n in (("train", 16), ("tune", 12)):
        gt = np.zeros((n, 6, 2))
        pred = gt.copy(); pred[..., 0] = np.asarray(errors)
        out[split] = {"predictions": pred.tolist(), "targets": gt.tolist(),
                      "records": [{"scenario": split, "frame": i} for i in range(n)]}
    return {"checkpoints": {"last": {"step": step, **out}}}


def manifests():
    a = {"status": "completed", "step": 500, "git_sha": "git", "initial_model_state_sha256": "state",
         "initial_parameter_count": 10, "model_config": {"plan_output_scale": [10, 5]},
         "split_sha256": "split", "train_rows_sha256": "train", "eval_rows_sha256": "eval",
         "supervision_manifest_sha256": "sup", "loss_weights": {}, "data_counts": {"train": 16, "eval": 12},
         "arguments": {"resume": None, "bn_policy": "adaptive"},
         "bn_training": {"policy": "adaptive", "affine_and_backbone_weights_trainable": True},
         "load_report": {"common_checkpoint_sha256": "source"}}
    b = copy.deepcopy(a)
    b["arguments"]["bn_policy"] = b["bn_training"]["policy"] = "fixed"
    logs = [{"kind": "train", "step": 500, "sample_order_sha256": "rows"}]
    return a, b, logs


def test_fixed_only_fit_pass_and_no_scale1_assumption():
    report = compare_bn_fit_reports(probe([.2] * 6), probe([.1] * 6))
    assert report["screening_decision"] == "FIXED_ONLY_PASSES_VERIFY_TRANSFER_TO_LARGER_TRAINING"
    a, b, logs = manifests()
    assert validate_pair_manifests(a, b, logs, logs)["only_bn_policy_differs"]


def test_last500_and_first2_harm_veto():
    with pytest.raises(ValueError, match="LAST500"):
        compare_bn_fit_reports(probe([.1] * 6, 400), probe([.1] * 6))
    report = compare_bn_fit_reports(probe([.01] * 4 + [2, 2]), probe([.04] * 4 + [.1, .1]))
    assert report["screening_decision"] == "DO_NOT_AUTO_ADOPT_FIXED_EARLY_HARM"


def test_both_pass_prefers_existing_policy_and_tune_is_not_selected():
    report = compare_bn_fit_reports(probe([.1] * 6), probe([.08] * 6))
    assert report["screening_decision"].startswith("BOTH_PASS_PREFER_EXISTING_ADAPTIVE")


def test_initial_state_and_order_hash_mismatch_are_rejected():
    a, b, logs = manifests()
    b["initial_model_state_sha256"] = "bad"
    with pytest.raises(ValueError, match="lineage"):
        validate_pair_manifests(a, b, logs, logs)
    b["initial_model_state_sha256"] = "state"
    with pytest.raises(ValueError, match="trace"):
        validate_pair_manifests(a, b, logs, [{"kind": "train", "step": 500, "sample_order_sha256": "bad"}])


def test_fixed_stats_unchanged_while_weights_update():
    initial = {"backbone_fpn.stem.1.running_mean": torch.zeros(2),
               "backbone_fpn.stem.1.running_var": torch.ones(2),
               "backbone_fpn.stem.1.num_batches_tracked": torch.tensor(10),
               "backbone_fpn.stem.1.weight": torch.ones(2)}
    adaptive, fixed = copy.deepcopy(initial), copy.deepcopy(initial)
    adaptive["backbone_fpn.stem.1.num_batches_tracked"] += 4
    adaptive["backbone_fpn.stem.1.weight"] += .1
    fixed["backbone_fpn.stem.1.weight"] += .2
    report = checkpoint_policy_integrity(initial, initial, adaptive, fixed)
    assert report["fixed_bn_running_buffers_unchanged"]
    assert report["representative_trainable_weight_changes"]["fixed"]["backbone_fpn.stem.1.weight"]["changed"]
    fixed["backbone_fpn.stem.1.running_mean"] += .1
    with pytest.raises(ValueError, match="Fixed BN"):
        checkpoint_policy_integrity(initial, initial, adaptive, fixed)
