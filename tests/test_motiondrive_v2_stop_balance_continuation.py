import copy
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts import motiondrive_v2_training as training
from scripts import run_motiondrive_v2_stop_balance_continuation as p6
from scripts import train_motiondrive_v2 as trainer


def loss_fixture(n=8):
    generator = torch.Generator().manual_seed(19)
    outputs = {
        "plan_abs": torch.randn(n, 6, 2, generator=generator, requires_grad=True),
        "occ_logits": torch.randn(n, 1, 2, 3, generator=generator, requires_grad=True),
        "lane_logits": torch.randn(n, 1, 2, 3, generator=generator, requires_grad=True),
        "history_hat": torch.randn(n, 4, 4, generator=generator, requires_grad=True),
        "history_logvar": torch.randn(n, 4, 4, generator=generator, requires_grad=True),
        "state_hat": torch.randn(n, 6, generator=generator, requires_grad=True),
        "state_logvar": torch.randn(n, 5, generator=generator, requires_grad=True),
    }
    batch = {
        "gt_plan": torch.randn(n, 6, 2, generator=generator),
        "plan_valid": torch.ones(n, 6, dtype=torch.bool),
        "occ_target": (torch.rand(n, 1, 2, 3, generator=generator) > .7).float(),
        "occ_valid": torch.ones(n, 1, 2, 3, dtype=torch.bool),
        "lane_target": (torch.rand(n, 1, 2, 3, generator=generator) > .7).float(),
        "lane_valid": torch.ones(n, 1, 2, 3, dtype=torch.bool),
        "history_target": torch.randn(n, 4, 4, generator=generator),
        "history_valid": torch.ones(n, 4, 4, dtype=torch.bool),
        "state_target": torch.randn(n, 6, generator=generator),
        "state_valid": torch.ones(n, 6, dtype=torch.bool),
    }
    batch["state_target"][:, 5] = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1][:n])
    return outputs, batch


def test_fixed_weight_formula_and_order():
    negative, positive = training.global_binary_class_weights(90, 10)
    assert negative == pytest.approx(100 / 180)
    assert positive == pytest.approx(100 / 20)
    with pytest.raises(ValueError):
        training.global_binary_class_weights(10, 0)


def test_default_stop_bce_remains_exact_unweighted_expression():
    outputs, batch = loss_fixture()
    loss, parts = training.compute_loss(outputs, batch, training.LossWeights())
    expected = F.binary_cross_entropy_with_logits(
        outputs["state_hat"][:, 5].float(), batch["state_target"][:, 5].float())
    assert torch.equal(parts["stop_bce"], expected)
    explicit, explicit_parts = training.compute_loss(
        outputs, batch, training.LossWeights(), stop_class_weights=None)
    assert torch.equal(loss, explicit)
    assert all(torch.equal(parts[k], explicit_parts[k]) for k in parts)


def test_treatment_weights_unreduced_bce_then_existing_valid_denominator():
    outputs, batch = loss_fixture()
    weights = training.global_binary_class_weights(6, 2)
    _, parts = training.compute_loss(outputs, batch, training.LossWeights(),
                                     stop_class_weights=weights)
    raw = F.binary_cross_entropy_with_logits(outputs["state_hat"][:, 5].float(),
                                              batch["state_target"][:, 5].float(), reduction="none")
    expected = (raw * torch.where(batch["state_target"][:, 5] >= .5,
                                  raw.new_tensor(weights[1]), raw.new_tensor(weights[0]))).sum() / 8
    assert torch.equal(parts["stop_bce"], expected)
    torch.testing.assert_close(parts["motion"], parts["history"] + parts["state"] + .2 * parts["stop_bce"])


def test_positive_absent_microbatch_is_finite_and_sums_to_full_logical_batch():
    outputs, batch = loss_fixture()
    fixed = training.global_binary_class_weights(6, 2)
    full, full_parts = training.compute_loss(outputs, batch, training.LossWeights(),
                                             stop_class_weights=fixed)
    normalizers = training.build_loss_normalizers(batch)
    sums = {key: torch.zeros_like(value) for key, value in full_parts.items()}
    for sl in (slice(0, 4), slice(4, 6), slice(6, 8)):
        out = {k: v[sl] for k, v in outputs.items()}
        labels = {k: v[sl] for k, v in batch.items()}
        loss, parts = training.compute_loss(out, labels, training.LossWeights(),
                                            normalizers=normalizers, stop_class_weights=fixed)
        assert torch.isfinite(loss)
        for key, value in parts.items():
            sums[key] += value
    torch.testing.assert_close(sums["stop_bce"], full_parts["stop_bce"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(sums["total"], full, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("bad", [(1,), [1., 2.], (1, 2), (0., 1.), (float("nan"), 1.)])
def test_invalid_stop_weight_contract_fails(bad):
    outputs, batch = loss_fixture()
    with pytest.raises(ValueError):
        training.compute_loss(outputs, batch, training.LossWeights(), stop_class_weights=bad)


class FakeDataset:
    split = "train"
    rows = np.arange(p6.EXPECTED_TRAIN_ROWS, dtype=np.int64)
    scene_names = np.asarray(["a"] * 27000 + ["b"] * (p6.EXPECTED_TRAIN_ROWS - 27000))

    def __init__(self):
        self.getitem_calls = 0

    def __getitem__(self, index):
        self.getitem_calls += 1
        raise AssertionError("image path must not be used")

    def _supervision(self, scene):
        rows = np.flatnonzero(self.scene_names == scene)
        target = np.zeros((len(rows), 6), dtype=np.float32)
        valid = np.ones((len(rows), 6), dtype=bool)
        target[::10, 5] = 1
        valid[::37, 5] = False
        return {"row": rows, "state_target": target, "state_valid": valid}


def test_train_count_is_label_only_global_and_binary():
    dataset = FakeDataset()
    result = p6.count_train_stop_labels(dataset)
    expected_valid = p6.EXPECTED_TRAIN_ROWS - len(np.arange(0, p6.EXPECTED_TRAIN_ROWS, 37))
    expected_positive = len(set(range(0, p6.EXPECTED_TRAIN_ROWS, 10)) -
                            set(range(0, p6.EXPECTED_TRAIN_ROWS, 37)))
    assert result["valid"] == expected_valid
    assert result["positive"] == expected_positive
    assert result["negative"] == expected_valid - expected_positive
    assert result["image_getitem_calls"] == dataset.getitem_calls == 0
    assert result["split"] == "train" and result["final_validation_accessed"] is False
    digest = __import__("hashlib").sha256()
    for scene in sorted(set(dataset.scene_names)):
        source = dataset._supervision(scene)
        scene_digest = __import__("hashlib").sha256()
        scene_digest.update(scene.encode() + b"\0")
        scene_digest.update(np.asarray(source["row"], dtype="<i8").tobytes())
        scene_digest.update(np.asarray(source["state_target"][:, 5], dtype="<f4").tobytes())
        scene_digest.update(np.asarray(source["state_valid"][:, 5], dtype=np.uint8).tobytes())
        digest.update(scene.encode() + b"\0" + scene_digest.hexdigest().encode() + b"\n")
    assert result["labels_sha256"] == digest.hexdigest()


@pytest.mark.parametrize("change", ["split", "short", "duplicate", "nonbinary", "missing"])
def test_train_count_rejects_wrong_scope_or_labels(change):
    dataset = FakeDataset()
    if change == "split":
        dataset.split = "tune"
    elif change == "short":
        dataset.rows = dataset.rows[:-1]
    elif change == "duplicate":
        dataset.rows = dataset.rows.copy(); dataset.rows[-1] = 0
    elif change in ("nonbinary", "missing"):
        original = dataset._supervision
        def altered(scene):
            value = original(scene)
            if scene == "a":
                if change == "nonbinary": value["state_target"][1, 5] = .4
                else: value["row"] = value["row"][1:]; value["state_target"] = value["state_target"][1:]; value["state_valid"] = value["state_valid"][1:]
            return value
        dataset._supervision = altered
    with pytest.raises(ValueError):
        p6.count_train_stop_labels(dataset)


def protocol(arm="balanced_global_train"):
    return {"schema_version": 1, "name": "p6_global_stop_class_balance", "arm": arm,
            "last_only_final_eval": True,
            "stop_class_weights": None if arm == "control_unweighted" else [.625, 2.5],
            "expected_initial_model_state_sha256": "a" * 64,
            "expected_optimizer_groups": [{"name": "backbone", "base_lr": 5e-6},
                                          {"name": "head", "base_lr": 5e-5}],
            "train_label_counts": {"rows": 100, "rows_sha256": "f" * 64,
                                   "labels_sha256": "e" * 64,
                                   "valid": 100, "negative": 80, "positive": 20},
            "fresh_optimizer_step_zero": True,
            "all_model_parameters_joint_trainable": True,
            "stop_target_definition": "state_valid[:,5] AND state_target[:,5]>=0.5",
            "balanced_output_is_calibrated_posterior": False,
            "raw_stop_logit_is_planner_input": True,
            "source": {"driver": "b" * 64}}


def test_changed_configuration_manifest_is_complete_and_control_is_none():
    assert trainer._validate_experimental_protocol(protocol()) == (.625, 2.5)
    assert trainer._validate_experimental_protocol(protocol("control_unweighted")) is None
    broken = protocol(); del broken["source"]
    with pytest.raises(ValueError, match="complete changed-configuration"):
        trainer._validate_experimental_protocol(broken)


@pytest.mark.parametrize("change", ["arm_weight", "formula", "lr", "count", "sha", "last_only"])
def test_experimental_protocol_fails_closed(change):
    value = protocol()
    if change == "arm_weight": value["arm"] = "control_unweighted"
    elif change == "formula": value["stop_class_weights"][1] += .01
    elif change == "lr": value["expected_optimizer_groups"][1]["base_lr"] = 1e-4
    elif change == "count": value["train_label_counts"]["valid"] = 99
    elif change == "sha": value["expected_initial_model_state_sha256"] = "a" * 63
    else: value["last_only_final_eval"] = False
    with pytest.raises(ValueError):
        trainer._validate_experimental_protocol(value)


def test_driver_argv_is_weights_only_fresh_step0_to1000_and_fixed_recipe(tmp_path):
    args = types.SimpleNamespace(data_root="/data", split_manifest="/split.json",
        supervision_root="/sup", run_dir=str(tmp_path / "run"), gpu=0, base_seed=1,
        workers=4, cuda_memory_limit_mib=12000, cuda_min_free_mib=8192)
    argv = p6.trainer_argv(args, {"checkpoint_path": "/immutable/p4/last.pth"})
    pairs = dict(zip(argv[::2], argv[1::2]))
    assert pairs["--init"] == "/immutable/p4/last.pth"
    assert "--resume" not in argv
    assert pairs["--steps"] == "1000" and pairs["--warmup"] == "100"
    assert pairs["--lr"] == "0.00005" and pairs["--backbone-lr"] == "0.000005"
    assert pairs["--batch"] == "16" and pairs["--microbatch"] == "2"
    assert pairs["--bn-policy"] == "fixed" and pairs["--time-input"] == "nominal"
    assert pairs["--eval-every"] == pairs["--save-every"] == "1000"
    assert pairs["--seed"] == "1" and pairs["--precision"] == "bf16"


def test_experimental_schedule_has_only_one_final_eval_and_last_checkpoint():
    args = types.SimpleNamespace(steps=1000, eval_every=1000, save_every=1000)
    actions = [trainer._training_schedule_actions(step, args, protocol())
               for step in range(1, 1001)]
    assert sum(evaluate_now for evaluate_now, _ in actions) == 1
    assert actions[-1] == (True, False)
    assert not any(periodic for _, periodic in actions)
    assert trainer._training_schedule_actions(1000, args, None) == (True, True)


def test_fixed_runtime_recipe_and_no_resume():
    args = trainer.arguments([
        "--data-root", "/data", "--split-manifest", "/split", "--supervision-root", "/sup",
        "--run-dir", "/run", "--phase", "joint", "--goal-on", "1", "--state-on", "1",
        "--gpu", "0", "--seed", "0", "--steps", "1000", "--batch", "16",
        "--microbatch", "2", "--eval-batch", "4", "--workers", "4", "--eval-every", "1000",
        "--save-every", "1000", "--lr", "0.00005", "--backbone-lr", "0.000005",
        "--weight-decay", "0.01", "--warmup", "100", "--grad-clip", "5",
        "--alpha-occ", "0.2", "--alpha-lane", "0.2", "--alpha-motion", "0.2",
        "--uncertainty", "1", "--precision", "bf16", "--time-input", "nominal",
        "--bn-policy", "fixed", "--init", "/p4/last.pth", "--arch", "resnet50",
        "--motion-input-mode", "low_feature", "--train-stride", "1", "--eval-stride", "5",
        "--max-train-samples", "0", "--max-eval-samples", "0", "--eval-split", "tune"])
    trainer._validate_experimental_runtime_args(args)
    resumed = copy.deepcopy(args); resumed.resume = "/old/last.pth"
    with pytest.raises(ValueError, match="weights-only init"):
        trainer._validate_experimental_runtime_args(resumed)


def test_build_experiment_changes_only_fixed_stop_weights():
    counts = {"rows": 100, "rows_sha256": "a" * 64, "labels_sha256": "b" * 64,
              "valid": 100, "negative": 80, "positive": 20,
              "target_definition": "state_valid[:,5] AND state_target[:,5]>=0.5",
              "weights_negative_positive": [.625, 2.5]}
    init = {"initial_model_state_sha256": "c" * 64}
    control = p6.build_experiment("control_unweighted", counts, init, {"x": "d" * 64})
    balanced = p6.build_experiment("balanced_global_train", counts, init, {"x": "d" * 64})
    assert control.keys() == balanced.keys()
    assert {k for k in control if control[k] != balanced[k]} == {"arm", "stop_class_weights"}
    assert control["stop_class_weights"] is None
    assert balanced["stop_class_weights"] == [.625, 2.5]


def test_immutable_p4_forward_and_label_sources_match_known_launch_receipt():
    assert p6.validate_unchanged_model_data_sources() == p6.UNCHANGED_MODEL_DATA_SHA256


def test_experiment_only_detailed_eval_records_same_forward_row_gt_prediction_and_bucket():
    class Model(torch.nn.Module):
        def forward(self, **inputs):
            n = len(inputs["images"])
            plan = torch.zeros(n, 6, 2)
            plan[1, :, 0] = .3
            return {"plan_abs": plan, "state_hat": torch.zeros(n, 6),
                    "history_hat": torch.zeros(n, 4, 4),
                    "occ_logits": torch.zeros(n, 1, 1, 1),
                    "lane_logits": torch.zeros(n, 1, 1, 1)}
    batch = {key: torch.zeros(2, 1) for key in training.MODEL_INPUTS}
    batch.update(images=torch.zeros(2, 1), history_images=torch.zeros(2, 1),
                 lidar2img=torch.zeros(2, 1), history_transforms=torch.zeros(2, 1),
                 time_offsets=torch.zeros(2, 4), goal_xy=torch.zeros(2, 2),
                 gt_plan=torch.zeros(2, 6, 2), plan_valid=torch.ones(2, 6, dtype=torch.bool),
                 state_target=torch.tensor([[0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 0.]]),
                 state_valid=torch.ones(2, 6, dtype=torch.bool),
                 history_target=torch.zeros(2, 4, 4), history_valid=torch.ones(2, 4, 4, dtype=torch.bool),
                 occ_target=torch.zeros(2, 1, 1, 1), occ_valid=torch.ones(2, 1, 1, 1, dtype=torch.bool),
                 lane_target=torch.zeros(2, 1, 1, 1), lane_valid=torch.ones(2, 1, 1, 1, dtype=torch.bool),
                 scenario=["a", "b"], session_id=["046", "x"], frame=torch.tensor([30, 35]),
                 row=torch.tensor([11, 12]), proxy_weight=torch.ones(2))
    report, plain = trainer.evaluate(Model(), [batch], torch.device("cpu"), "fp32")
    detailed_report, detailed = trainer.evaluate(Model(), [batch], torch.device("cpu"), "fp32",
                                                  detailed_records=True)
    assert report == detailed_report
    assert set(plain[0]) == {"scenario", "session", "frame", "d3", "proxy"}
    assert detailed[0]["row"] == 11 and detailed[0]["bucket"] == "steady"
    assert detailed[1]["bucket"] == "nonstop"
    assert detailed[0]["gt_abs_xy"] == [[0., 0.]] * 6
    assert np.asarray(detailed[1]["pred_abs_xy"]) == pytest.approx(np.asarray([[.3, 0.]] * 6))
    assert detailed[0]["pred_state"] == [0.] * 6
    assert detailed[0]["gt_state_valid"] == [True] * 6
    for record in detailed:
        expected = training.weighted_d3(torch.tensor(record["pred_abs_xy"])[None],
                                        torch.tensor(record["gt_abs_xy"])[None]).item()
        assert record["d3"] == expected


def install_tiny_model(monkeypatch):
    module = types.ModuleType("models.motiondrive_v2")
    class Config:
        def __init__(self, **kwargs): self.kwargs = kwargs
    class Model(torch.nn.Module):
        def __init__(self, config): super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(2))
    module.MotionDriveV2Config, module.MotionDriveV2 = Config, Model
    monkeypatch.setitem(sys.modules, "models.motiondrive_v2", module)
    return Model(Config()).state_dict()


def test_p4_checkpoint_is_weights_only_strict_load_and_step6000(tmp_path, monkeypatch):
    model = install_tiny_model(monkeypatch)
    config = {"backbone_arch": "resnet50", "goal_on": True, "state_on": True,
              "motion_input_mode": "low_feature", "plan_output_scale": [10., 5.]}
    args = {"phase": "joint", "steps": 6000, "goal_on": 1, "state_on": 1,
            "motion_input_mode": "low_feature", "bn_policy": "fixed", "time_input": "nominal",
            "precision": "bf16", "batch": 16, "microbatch": 2, "seed": 0,
            "train_stride": 1, "eval_stride": 5, "max_train_samples": 0, "max_eval_samples": 0}
    load = {"common_checkpoint_sha256": "e" * 64}
    manifest = {"git_sha": p6.P4_TRAINING_GIT_SHA, "arguments": args, "model_config": config,
                "load_report": load, "split_sha256": p6.EXPECTED_SPLIT_SHA256,
                "loss_weights": {"plan": 1.}}
    checkpoint = tmp_path / "last.pth"
    torch.save({"model": model, "optimizer": {"state": {}}, "step": 6000, "manifest": manifest}, checkpoint)
    sidecar = tmp_path / "manifest.json"
    sidecar.write_text(json.dumps({"git_sha": p6.P4_TRAINING_GIT_SHA, "status": "completed",
                                   "step": 6000, "load_report": load}))
    result = p6.validate_p4_initialization(checkpoint, p6.file_sha(checkpoint), sidecar,
        p6.file_sha(sidecar), p6.EXPECTED_SPLIT_SHA256, 0)
    assert result["step"] == 6000 and result["strict_load_missing"] == []
    assert result["initial_model_state_sha256"] == training.tensor_state_sha256(model)


def test_checkpoint_missing_model_key_fails_strict(tmp_path, monkeypatch):
    model = install_tiny_model(monkeypatch)
    checkpoint = tmp_path / "last.pth"
    manifest = {"git_sha": p6.P4_TRAINING_GIT_SHA, "arguments": {}, "model_config": {},
                "load_report": {}, "split_sha256": p6.EXPECTED_SPLIT_SHA256,
                "loss_weights": {"plan": 1.}}
    torch.save({"model": {}, "optimizer": {}, "step": 6000, "manifest": manifest}, checkpoint)
    sidecar = tmp_path / "manifest.json"; sidecar.write_text("{}")
    with pytest.raises(ValueError):
        p6.validate_p4_initialization(checkpoint, p6.file_sha(checkpoint), sidecar,
            p6.file_sha(sidecar), p6.EXPECTED_SPLIT_SHA256, 0)
