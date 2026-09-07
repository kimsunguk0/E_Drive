"""CPU-only checks for reporting and image-only counterfactual boundaries."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_motiondrive_v2_planning as evaluator
from motiondrive_v2_training import MODEL_INPUTS, weighted_d3
from train_motiondrive_v2 import evaluate as train_evaluate


class FakeDataset(Dataset):
    def __init__(self):
        self.rows = np.arange(6)
        self.scene_names = np.array(["a", "a", "b", "b", "c", "c"])
        self.arr = {"frame": np.array([30, 35, 30, 35, 30, 35])}
        self.manifest = {"splits": {"tune": ["a", "b", "c"]}}
        self.image_root = Path("/synthetic/only")
        self.cache_sha = "c" * 64

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = int(self.rows[index])
        history = torch.arange(4.).reshape(4, 1, 1, 1).expand(4, 3, 2, 3).clone() + row
        sample = dict(images=torch.full((6, 3, 4, 6), row + 1.), history_images=history,
                      lidar2img=torch.eye(4).repeat(6, 1, 1) * (row + 1),
                      history_transforms=torch.eye(4).repeat(4, 1, 1) * (row + 2),
                      time_offsets=torch.tensor([.1, .2, .5, 1.]), goal_xy=torch.tensor([30., row]),
                      gt_plan=torch.arange(12.).reshape(6, 2), plan_valid=torch.ones(6, dtype=torch.bool),
                      scenario=str(self.scene_names[row]), session_id="first" if row < 4 else "second",
                      frame=int(self.arr["frame"][row]), row=row,
                      state_target=torch.tensor([1., 0., [-1., 0., 1.][row % 3], 0., 0., 0.]),
                      state_valid=torch.ones(6, dtype=torch.bool),
                      history_target=torch.zeros(4, 4), history_valid=torch.ones(4, 4, dtype=torch.bool),
                      occ_target=torch.zeros(1, 2, 3), lane_target=torch.zeros(1, 2, 3),
                      occ_valid=torch.ones(1, 2, 3, dtype=torch.bool), lane_valid=torch.ones(1, 2, 3, dtype=torch.bool),
                      proxy_weight=torch.tensor(float(row + 1)), supplied_status=torch.full((6,), 1e8))
        return sample


class ImageOnlyModel(nn.Module):
    def __init__(self, before_forward=None):
        super().__init__()
        self.before_forward = before_forward
        self.seen = []

    def forward(self, **inputs):
        assert set(inputs) == set(MODEL_INPUTS)
        self.seen.append(set(inputs))
        if self.before_forward:
            self.before_forward()
        b = len(inputs["images"])
        value = inputs["images"].mean((1, 2, 3, 4))
        plan = torch.arange(12.).reshape(1, 6, 2).repeat(b, 1, 1) + value[:, None, None]
        return dict(plan_abs=plan, state_hat=torch.zeros(b, 6), history_hat=torch.zeros(b, 4, 4),
                    occ_logits=torch.zeros(b, 1, 2, 3), lane_logits=torch.zeros(b, 1, 2, 3))


def test_official_metric_is_exact_trainer_weighted_d3_and_cumulative_ade():
    gt = torch.arange(36.).reshape(3, 6, 2)
    pred = gt + torch.randn_like(gt)
    got = evaluator.per_frame_errors(pred, gt)
    assert torch.equal(got["d3"], weighted_d3(pred, gt))
    torch.testing.assert_close(got["d3"], got["cumulative_ade_1_2_3s"].mean(1))
    error_basis = torch.zeros(6, 6, 2)
    error_basis[torch.arange(6), torch.arange(6), 0] = 1.
    torch.testing.assert_close(evaluator.per_frame_errors(error_basis, torch.zeros_like(error_basis))["d3"],
                               torch.tensor([11., 11., 5., 5., 2., 2.]) / 36)
    # Applying cumsum to already cumulative GT would break this zero-error case.
    assert torch.equal(evaluator.per_frame_errors(gt, gt)["d3"], torch.zeros(3))


def test_normal_matches_train_evaluation_including_session_mean_without_proxy_weights():
    data = FakeDataset()
    loader = DataLoader(data, batch_size=2, shuffle=False)
    model = ImageOnlyModel()
    reference, rows = train_evaluate(model, loader, torch.device("cpu"), "bf16")
    actual = evaluator.evaluate_planning(model, loader, torch.device("cpu"), "bf16")
    for key in ("official_d3", "session_mean_d3", "n", "n_sessions", "session_d3"):
        assert actual["summary"][key] == reference[key]
    assert actual["summary"]["official_d3"] != reference["proxy_d3"]
    assert [r["d3"] for r in actual["records"]] == [r["d3"] for r in rows]
    assert sum(v["n"] for v in actual["buckets"].values()) == len(data)
    assert actual["buckets"]["stop"]["n"] == 0
    assert actual["buckets"]["stop"]["official_d3"] is None
    record = actual["records"][0]
    assert set(record) >= {"scenario", "session", "frame", "row", "pred_abs_xy", "gt_abs_xy",
                           "point_l2", "cumulative_ade_1_2_3s", "longitudinal_error", "lateral_error"}
    assert record["longitudinal_error"] == record["lateral_error"] == [1.] * 6
    assert len(record["point_l2"]) == 6 and len(record["cumulative_ade_1_2_3s"]) == 3
    assert model.seen and all(v == set(MODEL_INPUTS) for v in model.seen)


@pytest.mark.parametrize("speed,ax,expected", [(0., 1., "stop"), (.199, -1., "stop"),
                                              (.2, .5, "accel"), (1., -.5, "decel"),
                                              (1., .499, "cruise"), (1., -.499, "cruise")])
def test_fixed_state_bucket_thresholds_and_priority(speed, ax, expected):
    assert evaluator.state_bucket([speed, 0., ax, 0., 0., 0.], [True] * 6) == expected


def test_invalid_state_is_reported_as_unknown_not_silently_discarded():
    assert evaluator.state_bucket([1., 0., np.nan, 0., 0., 0.], [True] * 6) == "unknown"
    assert evaluator.state_bucket([1., 0., 1., 0., 0., 0.], [False, True, True, True, True, True]) == "unknown"
    # Stationary classification does not need an acceleration label.
    assert evaluator.state_bucket([0., 0., np.nan, 0., 0., 0.], [True, True, False, True, True, True]) == "stop"


def test_permutation_is_fixed_by_sorted_split_not_receiver_order_and_has_no_self_donor():
    a = evaluator.scene_derangement(["a", "b", "c"], 17)
    b = evaluator.scene_derangement(["c", "b", "a"], 17)
    assert a == b and set(a) == set(a.values())
    assert all(k != v for k, v in a.items())
    with pytest.raises(ValueError):
        evaluator.scene_derangement(["a"], 0)


def test_donor_resolution_is_same_frame_and_fails_before_inference_if_missing():
    data = FakeDataset()
    mapping = evaluator.scene_derangement(data.manifest["splits"]["tune"], 0)
    indices = evaluator.donor_indices(data, data, mapping)
    for i, j in enumerate(indices):
        assert data[i]["frame"] == data[j]["frame"]
        assert data[i]["scenario"] != data[j]["scenario"]
    broken = FakeDataset()
    broken.rows = np.array([0, 2, 4])
    with pytest.raises(ValueError, match="same frame"):
        evaluator.donor_indices(data, broken, mapping)
    with pytest.raises(ValueError, match="self donor"):
        evaluator.donor_indices(data, data, {s: s for s in "abc"})


@pytest.mark.parametrize("condition", ["normal", "image_shuffle", "image_mismatch", "repeat_current", "reverse_history"])
def test_counterfactuals_change_images_only_keep_receiver_gt_goal_pose_time(condition):
    data = FakeDataset()
    mapping = evaluator.scene_derangement(data.manifest["splits"]["tune"], 0)
    indices = evaluator.donor_indices(data, data, mapping)
    wrapped = evaluator.ImageCounterfactualDataset(data, condition, data, indices)
    original, modified = data[0], wrapped[0]
    for name, value in original.items():
        if name in ("images", "history_images"):
            continue
        assert torch.equal(modified[name], value) if torch.is_tensor(value) else modified[name] == value
    if condition in ("image_shuffle", "image_mismatch"):
        donor = data[indices[0]]
        assert torch.equal(modified["images"], donor["images"])
        assert torch.equal(modified["history_images"], donor["history_images"])
        assert modified["donor_scenario"] == donor["scenario"]
    elif condition == "repeat_current":
        assert torch.equal(modified["images"], original["images"])
        assert all(torch.equal(frame, modified["history_images"][0]) for frame in modified["history_images"])
        assert modified["history_images"].unique().item() == original["images"][0].unique().item()
    elif condition == "reverse_history":
        assert torch.equal(modified["history_images"], original["history_images"].flip(0))
    else:
        assert torch.equal(modified["images"], original["images"])
        assert torch.equal(modified["history_images"], original["history_images"])


def test_invalid_or_nonfinite_plans_and_non_fp32_prediction_are_rejected():
    class BadModel(ImageOnlyModel):
        def forward(self, **inputs):
            out = super().forward(**inputs)
            out["plan_abs"] = out["plan_abs"].bfloat16()
            return out
    with pytest.raises(ValueError, match="FP32"):
        evaluator.evaluate_planning(BadModel(), DataLoader(FakeDataset(), batch_size=2), torch.device("cpu"))
    with pytest.raises(FloatingPointError):
        evaluator.per_frame_errors(torch.full((1, 6, 2), float("nan")), torch.zeros(1, 6, 2))
    class InvalidGT(FakeDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["plan_valid"][0] = False
            return sample
    model = ImageOnlyModel()
    with pytest.raises(ValueError, match="Invalid six-point"):
        evaluator.evaluate_planning(model, DataLoader(InvalidGT(), batch_size=2), torch.device("cpu"))
    assert model.seen == []


def test_cli_is_tune_only_default_normal_and_has_no_config_override_switch():
    argv = ["--checkpoint", "trusted.pth", "--split-manifest", "split.json",
            "--supervision-root", "supervision", "--out", "result.json"]
    args = evaluator.arguments(argv)
    assert args.split == "tune" and args.conditions == ["normal"] and args.device == "cpu"
    for option in (["--split", "val"], ["--split", "historical_val"], ["--goal-on", "1"],
                   ["--config-json", "override.json"], ["--device", "cuda:6"]):
        with pytest.raises(SystemExit):
            evaluator.arguments(argv + option)
    with pytest.raises(ValueError):
        evaluator.require_tune("val")
    with pytest.raises(ValueError):
        evaluator.normalize_conditions(["image_shuffle", "image_mismatch"])


def test_json_publication_refuses_existing_and_dangling_output(tmp_path):
    path = tmp_path / "report.json"
    evaluator.write_new_json(path, {"metric": 1.})
    with pytest.raises(FileExistsError):
        evaluator.write_new_json(path, {"metric": 0.})
    assert json.loads(path.read_text()) == {"metric": 1.}
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing.json")
    with pytest.raises(FileExistsError):
        evaluator.write_new_json(dangling, {})
    assert dangling.is_symlink()


def test_publication_race_does_not_replace_existing_report(tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    real_link = evaluator.os.link
    def racing_link(src, dst):
        target.write_text("belongs to another process")
        return real_link(src, dst)
    monkeypatch.setattr(evaluator.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        evaluator.write_new_json(target, {"new": True})
    assert target.read_text() == "belongs to another process"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_main_preregisters_before_any_forward_and_never_overrides_checkpoint(tmp_path, monkeypatch):
    split = tmp_path / "split.json"
    split.write_text("{}")
    for name in ["supervision_manifest.json", "calibration.npz", *[f"{s}{suffix}" for s in "abc" for suffix in (".npz", ".json")]]:
        (tmp_path / name).write_bytes(b"synthetic test fixture")
    report = tmp_path / "result.json"
    protocol = report.with_suffix(".protocol.json")
    checkpoint = tmp_path / "trusted.pth"
    checkpoint.write_bytes(b"synthetic checkpoint metadata")
    monkeypatch.setattr(evaluator, "MotionDriveDataset", lambda **kwargs: FakeDataset())
    monkeypatch.setattr(evaluator, "checkpoint_manifest", lambda *a: {
        "checkpoint_sha256": "f" * 64, "checkpoint_file_identity": evaluator.file_identity(checkpoint.stat())})
    monkeypatch.setattr(evaluator, "source_manifest", lambda: {"synthetic_test": True})
    def load(args):
        assert args.config_json is args.goal_on is args.state_on is None
        def before_forward():
            frozen = json.loads(protocol.read_text())
            assert frozen["status"] == "preregistered_before_any_forward"
            assert frozen["donor_scene_map"] is not None
            assert not report.exists()
        model = ImageOnlyModel(before_forward)
        model.audit_load_metadata = {"explicit_overrides": {}, "checkpoint_sha256": "f" * 64}
        return model
    monkeypatch.setattr(evaluator, "construct_model", load)
    argv = ["--checkpoint", str(checkpoint), "--split-manifest", str(split), "--supervision-root", str(tmp_path),
            "--out", str(report), "--conditions", "normal", "image_shuffle", "repeat_current", "reverse_history",
            "--workers", "0", "--batch", "2", "--device", "cpu"]
    assert evaluator.main(argv) == 0
    result = json.loads(report.read_text())
    assert set(result["conditions"]) == set(evaluator.CONDITIONS)
    assert not result["selection_performed"] and not result["final_val_accessed"]
    assert result["precision"] == "fp32" and result["precision_requested"] == "bf16"
    assert result["protocol"]["data"]["receiver_rows_sha256"] == evaluator.rows_sha256(np.arange(6))
    assert result["conditions"]["image_shuffle"]["records"][0]["donor"]["frame"] == 30
    with pytest.raises(FileExistsError):
        evaluator.main(argv)


def test_checkpoint_header_requires_complete_config_and_matching_split(tmp_path):
    from models.motiondrive_v2 import MotionDriveV2Config
    path = tmp_path / "checkpoint.pth"
    config = MotionDriveV2Config().to_dict()
    torch.save({"manifest": {"model_config": config, "split_sha256": "a" * 64},
                "step": 3, "rng": {"numpy": np.random.get_state()}}, path)
    header = evaluator.checkpoint_manifest(path, "a" * 64)
    assert header["checkpoint_step"] == 3
    with pytest.raises(ValueError, match="lineage"):
        evaluator.checkpoint_manifest(path, "b" * 64)
    del config["plan_output_scale"]
    torch.save({"manifest": {"model_config": config, "split_sha256": "a" * 64}}, path)
    with pytest.raises(ValueError, match="missing"):
        evaluator.checkpoint_manifest(path, "a" * 64)
