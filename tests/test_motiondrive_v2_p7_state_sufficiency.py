import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_motiondrive_v2_p7_state_sufficiency as probe


def labels(n=4):
    state = torch.zeros(n, 6)
    state[:, :5] = torch.arange(n * 5, dtype=torch.float32).reshape(n, 5) / 10
    state[:, 5] = torch.tensor(([0., 1.] * ((n + 1) // 2))[:n])
    return {"state_target": state, "history_target": torch.arange(n * 16, dtype=torch.float32).reshape(n, 4, 4),
            "goal_xy": torch.tensor([[10., -2.]]).repeat(n, 1), "gt_plan": torch.zeros(n, 6, 2)}


def predictions(source):
    state = source["state_target"].clone()
    state[:, 5] = torch.tensor([-2., 2.] * ((len(state) + 1) // 2))[:len(state)]
    return {"pred_state": state, "pred_history": source["history_target"].clone(),
            "pred_plan": source["gt_plan"].clone()}


def test_fixed_contract_and_step_count():
    assert len(probe.FEATURE_NAMES) == 24
    assert probe.FEATURE_NAMES[5] == "stop_probability"
    assert probe.N_STEPS == 60 * 54 == 3240
    assert tuple(probe.NOMINAL_SECONDS) == (.1, .2, .5, 1.)


def test_canonical24_uses_binary_gt_sigmoid_prediction_and_same_goal():
    source = labels()
    gt = probe.canonical_raw24(source)
    pred = probe.canonical_raw24(source, predictions(source))
    assert torch.equal(gt[:, 5], source["state_target"][:, 5])
    assert torch.allclose(pred[:, 5], torch.sigmoid(predictions(source)["pred_state"][:, 5]))
    assert torch.equal(gt[:, -2:], pred[:, -2:])
    assert torch.equal(gt[:, 6:22], pred[:, 6:22])


def test_gt_stop_must_be_binary():
    source = labels(); source["state_target"][0, 5] = .2
    with pytest.raises(ValueError, match="binary"):
        probe.canonical_raw24(source)


def test_one_gt_train_normalizer_is_shared_and_tune_independent():
    raw = probe.canonical_raw24(labels())
    normalizer = probe.fit_shared_normalizer(raw)
    pred = probe.canonical_raw24(labels(), predictions(labels()))
    before = probe.apply_normalizer(pred, normalizer)
    changed_tune = pred + 1000
    assert torch.equal(before, probe.apply_normalizer(pred, normalizer))
    assert not torch.equal(before, probe.apply_normalizer(changed_tune, normalizer))
    assert tuple(normalizer["scale"].tolist()) == tuple(probe.FEATURE_SCALE.tolist())


def test_normalizer_rejects_wrong_shape():
    with pytest.raises(ValueError, match="canonical GT"):
        probe.fit_shared_normalizer(torch.zeros(2, 23))


def full_label_payload(n=4):
    base = labels(n)
    return {"rows": torch.arange(n, dtype=torch.int64), "state_target": base["state_target"],
            "state_valid": torch.ones(n, 6, dtype=torch.bool),
            "history_target": base["history_target"],
            "history_valid": torch.ones(n, 4, 4, dtype=torch.bool),
            "goal_xy": base["goal_xy"], "gt_plan": base["gt_plan"],
            "plan_valid": torch.ones(n, 6, dtype=torch.bool),
            "scenario": ["scene"] * n, "session": ["session"] * n,
            "frame": torch.arange(30, 30 + n, dtype=torch.int64)}


def test_label_payload_strict_shape_dtype_rows_and_validity():
    payload = full_label_payload()
    digest = probe.rows_sha256(payload["rows"].numpy())
    assert probe.validate_label_payload(payload, 4, digest)
    bad = dict(payload); bad["history_valid"] = bad["history_valid"].clone(); bad["history_valid"][0, 0, 0] = False
    with pytest.raises(ValueError, match="all 22"):
        probe.validate_label_payload(bad, 4, digest)


def test_prediction_payload_rejects_row_mismatch_and_extra_key():
    source = predictions(labels())
    payload = {"rows": torch.arange(4, dtype=torch.int64), **source}
    digest = probe.rows_sha256(payload["rows"].numpy())
    assert probe.validate_prediction_payload(payload, 4, digest, 0, "train")
    with pytest.raises(ValueError, match="row order"):
        probe.validate_prediction_payload(payload, 4, "0" * 64, 0, "train")
    with pytest.raises(ValueError, match="keys"):
        probe.validate_prediction_payload({**payload, "gt": torch.zeros(1)}, 4, digest, 0, "train")


def test_tune_reference_requires_exact_pinned_hash(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"report": {}, "records": []}))
    with pytest.raises(ValueError, match="SHA"):
        probe.validate_tune_reference(path, 0)


def test_extract_cli_has_no_final_and_fixed_cuda_device():
    with pytest.raises(SystemExit):
        probe.arguments(["extract", "--help"])
    parser_source = Path(probe.__file__).read_text()
    assert 'choices=("train", "tune")' in parser_source
    assert 'choices=("cuda:0",)' in parser_source
    assert "final" not in probe.ROWS


def test_pair_initialization_and_order_are_identical_without_training_full_inventory(monkeypatch):
    monkeypatch.setattr(probe, "ROWS", {"train": (8, 1, "x"), "tune": (4, 5, "y")})
    monkeypatch.setattr(probe, "N_EPOCHS", 2)
    monkeypatch.setattr(probe, "BATCH_SIZE", 4)
    monkeypatch.setattr(probe, "N_STEPS", 4)
    train_x = torch.randn(8, 24); train_y = torch.randn(8, 6, 2)
    tune_x = torch.randn(4, 24); tune_y = torch.randn(4, 6, 2)
    _, left = probe.train_tiny(train_x, train_y, tune_x, tune_y, seed=0)
    _, right = probe.train_tiny(train_x + .1, train_y, tune_x + .1, tune_y, seed=0)
    assert left["initial_model_state_sha256"] == right["initial_model_state_sha256"]
    assert left["sample_order_sha256"] == right["sample_order_sha256"]
    assert left["steps"] == right["steps"] == 4


def test_seed_changes_tiny_initialization():
    torch.manual_seed(0); left = probe.model_state_sha(probe.tiny_model())
    torch.manual_seed(1); right = probe.model_state_sha(probe.tiny_model())
    assert left != right


def test_tune_metric_uses_exactly_two_batches_and_one_pass():
    class Counting(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.calls = 0
        def forward(self, value):
            self.calls += 1
            return torch.zeros(len(value), 12)
    model = Counting()
    d3, rows, horizons = probe.evaluate_tiny(
        model, torch.zeros(1998, 24), torch.zeros(1998, 6, 2))
    assert model.calls == 2 == int(np.ceil(1998 / 1024))
    assert d3 == 0 and torch.equal(rows, torch.zeros(1998))
    assert horizons == {"ade1": 0., "ade2": 0., "ade3": 0.}


def test_terminal_metric_uses_canonical_float64_mean_of_saved_fp32_rows():
    class Fixed(torch.nn.Module):
        def forward(self, value):
            # Deliberately mix large/small FP32 values so FP32 and float64
            # reductions are observably distinct.
            x = torch.tensor([1e5, 1e-2, 3e-2], dtype=torch.float32)[:len(value)]
            return torch.stack((x, torch.zeros_like(x)), 1).repeat(1, 6)
    target = torch.zeros(3, 6, 2)
    d3, rows, _ = probe.evaluate_tiny(Fixed(), torch.zeros(3, 24), target)
    assert d3 == float(np.asarray(rows.numpy(), dtype=np.float64).mean())


def test_json_native_roundtrip_and_strict_unsupported(tmp_path):
    out = tmp_path / "out.json"
    probe.atomic_json(out, {"f": np.float32(.5), "i": np.int64(2),
                            "a": np.array([1, 2]), "tensor": torch.tensor([3., 4.])})
    assert json.loads(out.read_text()) == {"a": [1, 2], "f": .5, "i": 2,
                                           "tensor": [3., 4.]}
    with pytest.raises(TypeError, match="Unsupported"):
        probe.to_native({"x": object()})


def test_atomic_outputs_refuse_overwrite(tmp_path):
    path = tmp_path / "x.json"; probe.atomic_json(path, {"x": 1})
    with pytest.raises(ValueError, match="overwrite"):
        probe.atomic_json(path, {"x": 2})


def test_p7_control_artifact_pins_are_complete():
    assert set(probe.P7_CONTROL) == {0, 1}
    for item in probe.P7_CONTROL.values():
        assert set(item) == {"checkpoint_sha256", "manifest_sha256", "final_eval_sha256", "official_d3"}
        assert all(len(item[key]) == 64 for key in ("checkpoint_sha256", "manifest_sha256", "final_eval_sha256"))


def test_prediction_artifact_manifest_tamper_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROWS", {"train": (4, 1, probe.rows_sha256(np.arange(4))),
                                        "tune": probe.ROWS["tune"]})
    payload = {"rows": torch.arange(4, dtype=torch.int64),
               "pred_state": torch.zeros(4, 6), "pred_history": torch.zeros(4, 4, 4),
               "pred_plan": torch.zeros(4, 6, 2)}
    path = tmp_path / "pred.pt"
    probe.atomic_torch(path, payload)
    manifest = {"schema_version": probe.SCHEMA_VERSION, "status": "completed",
        "kind": "prediction", "split": "train", "rows": 4,
        "rows_sha256": probe.ROWS["train"][2], "artifact_sha256": probe.sha256(path),
        "script_sha256": probe.sha256(probe.__file__), "base_seed": 0,
        "checkpoint_sha256": probe.P7_CONTROL[0]["checkpoint_sha256"],
        "run_manifest_sha256": probe.P7_CONTROL[0]["manifest_sha256"],
        "p7_training_source_manifest_sha256": probe.P7_SOURCE_MANIFEST_SHA256,
        "runtime_source_manifest_sha256": probe.RUNTIME_SOURCE_MANIFEST_SHA256,
        "model_state_sha256_before": "a" * 64, "model_state_sha256_after": "a" * 64,
        "mode": "eval", "context": "torch.inference_mode",
        "precision": "bf16_encoder_fp32_heads", "batch": 8, "workers": 4,
        "augment": False, "time_input": "nominal",
        "nominal_seconds": list(probe.NOMINAL_SECONDS), "full_unmodified_forward": True,
        "motion_only_shortcut": False,
        "tune_terminal_state_plan_gt_bitwise_verified": False,
        "optimizer_created": False, "backward_called": False, "weights_updated": False,
        "final_validation_accessed": False}
    manifest_path = path.with_suffix(".pt.manifest.json")
    probe.atomic_json(manifest_path, manifest)
    loaded, _, _ = probe.load_artifact(path, "prediction", "train")
    assert torch.equal(loaded["rows"], payload["rows"])
    manifest["motion_only_shortcut"] = True
    manifest_path.unlink()
    probe.atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="provenance"):
        probe.load_artifact(path, "prediction", "train")
