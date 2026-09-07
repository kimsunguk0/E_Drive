"""CPU tests of initial-parity byte comparison and pre-forward input gates."""
from pathlib import Path

import pytest
import torch

from scripts import probe_motiondrive_v2_query_initial as probe


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16,
                                  torch.bfloat16, torch.int64, torch.bool])
def test_exact_tensor_equal_identical_values_and_bytes(dtype):
    left = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=dtype)
    result = probe.exact_tensor_equal(left, left.clone())
    assert result is True


@pytest.mark.parametrize("change", ["dtype", "shape", "value", "signed_zero"])
def test_exact_tensor_equal_rejects_nonidentical_representation(change):
    left = torch.tensor([[0., 1.], [2., 3.]], dtype=torch.float32)
    right = left.clone()
    if change == "dtype":
        right = right.double()
        assert torch.equal(left, right)  # Numeric equality alone is insufficient.
    elif change == "shape":
        right = right.flatten()
    elif change == "value":
        right[1, 1] = 4.
    else:
        right[0, 0] = -0.
        assert torch.equal(left, right)
        assert torch.signbit(right[0, 0]) and not torch.signbit(left[0, 0])
    assert probe.exact_tensor_equal(left, right) is False


def test_exact_tensor_equal_supports_strided_logically_identical_values():
    strided = torch.arange(24, dtype=torch.float32).reshape(3, 8)[:, ::2]
    assert not strided.is_contiguous()
    contiguous = strided.contiguous()
    assert probe.exact_tensor_equal(strided, contiguous) is True
    assert probe.exact_tensor_equal(contiguous, strided) is True
    changed = contiguous.clone()
    changed[1, 2] += 1
    assert probe.exact_tensor_equal(strided, changed) is False


def test_probe_data_lineage_gate_runs_before_checkpoint_load_or_cuda(tmp_path, monkeypatch):
    (tmp_path / "reports").mkdir()
    output = tmp_path / "reports/parity.json"
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "source_snapshot", lambda *_: {"git_sha": "a" * 40})
    monkeypatch.setattr(probe, "sha256", lambda _: probe.INIT_SHA)
    initialized = torch.cuda.is_initialized()
    received = []

    class DeliberateDataGateFailure(Exception):
        pass

    def data_gate(args):
        received.append(args)
        raise DeliberateDataGateFailure("data lineage must fail before any forward")

    monkeypatch.setattr(probe, "validate_data_paths", data_gate)
    monkeypatch.setattr(probe.torch, "load", lambda *_a, **_kw: pytest.fail("Must not load checkpoint after failed data gate"))
    monkeypatch.setattr(probe, "validate_cuda_namespace", lambda *_: pytest.fail("Must not access CUDA after failed data gate"))
    monkeypatch.setattr(probe, "configure_cuda_memory", lambda *_: pytest.fail("Must not allocate CUDA after failed data gate"))
    with pytest.raises(DeliberateDataGateFailure):
        probe.main(["--expected-git-sha", "a" * 40, "--out", str(output)])
    assert len(received) == 1
    args = received[0]
    assert Path(args.split_manifest) == tmp_path / "data/etri/motiondrive_v2/grouped_split_rawtime.json"
    assert Path(args.supervision_root) == tmp_path / "data/etri/motiondrive_v2/train_tune_geometry_v2"
    assert Path(args.init) == tmp_path / probe.INIT
    assert args.expected_init_sha256 == probe.INIT_SHA
    assert not output.exists()
    assert torch.cuda.is_initialized() == initialized


def test_probe_model_input_whitelist_nominal_only_and_gt_unchanged():
    batch = {
        "images": torch.zeros(1, 6, 3, 4, 6),
        "history_images": torch.zeros(1, 4, 3, 2, 3),
        "lidar2img": torch.eye(4)[None, None].repeat(1, 6, 1, 1),
        "history_transforms": torch.eye(4)[None, None].repeat(1, 4, 1, 1),
        "time_offsets": torch.tensor([[.1001, .2002, .5003, 1.0004]], dtype=torch.float64),
        "goal_xy": torch.ones(1, 2),
        "gt_plan": torch.randn(1, 6, 2),
        "state_target": torch.randn(1, 6),
        "history_target": torch.randn(1, 4, 4),
        "provided_status": torch.randn(1, 6),
    }
    before = {name: value.clone() for name, value in batch.items()}
    result = probe.model_inputs(batch, time_input="nominal")
    expected_keys = {"images", "history_images", "lidar2img", "history_transforms", "time_offsets", "goal_xy"}
    assert set(result) == expected_keys
    assert all(result[name] is batch[name] for name in expected_keys - {"time_offsets"})
    assert probe.exact_tensor_equal(result["time_offsets"], torch.tensor([[.1, .2, .5, 1.]], dtype=torch.float32))
    assert all(probe.exact_tensor_equal(batch[name], saved) for name, saved in before.items())
    for name in ("gt_plan", "state_target", "history_target", "provided_status"):
        batch[name] = torch.full_like(batch[name], 999.)
    replaced = probe.model_inputs(batch, time_input="nominal")
    assert all(probe.exact_tensor_equal(result[name], replaced[name]) for name in expected_keys)
