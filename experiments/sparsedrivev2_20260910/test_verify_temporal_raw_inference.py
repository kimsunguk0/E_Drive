import json
from pathlib import Path

import numpy as np
import pytest
import torch

import verify_temporal_raw_inference as vr


def test_byte_parity_distinguishes_signed_zero_and_supports_bfloat16():
    report = vr.tensor_comparison({"x": torch.tensor([0.])}, {"x": torch.tensor([-0.])})["x"]
    assert report["numerically_equal"] and not report["bitwise_equal"]
    assert report["max_absolute_finite_difference"] == 0
    value = torch.tensor([.1, .2], dtype=torch.bfloat16)
    result = vr.tensor_comparison({"x": value}, {"x": value.clone()})["x"]
    assert result["bitwise_equal"] and result["raw_sha256"] == result["cache_sha256"]


def test_comparison_preserves_changed_predictions_and_rejects_shape_dtype():
    result = vr.tensor_comparison({"x": torch.zeros(1, 6, 2)}, {"x": torch.ones(1, 6, 2)})["x"]
    assert not result["bitwise_equal"] and result["changed_elements"] == 12
    assert result["max_absolute_finite_difference"] == 1
    with pytest.raises(ValueError, match="shape/dtype"):
        vr.tensor_comparison({"x": torch.zeros(1)}, {"x": torch.zeros(2)})
    with pytest.raises(ValueError, match="shape/dtype"):
        vr.tensor_comparison({"x": torch.zeros(1)}, {"x": torch.zeros(1, dtype=torch.float64)})


def fixture_files():
    names = {"calibration.parquet", "ego_pose.parquet"} | {f"{c}/frame_{f}.jpg" for c, f in vr.REQUESTS}
    files = {name: name.encode() for name in names}
    receipt = {"schema": "temporal_raw_fixture_v1", "files_sha256": {k: vr.digest(v) for k, v in files.items()},
               "future_trajectory_labels_read": False}
    files["fixture_receipt.json"] = json.dumps(receipt).encode()
    return files


def test_fixture_byte_mutation_is_detected_and_overwrite_refused(tmp_path):
    output = tmp_path / "fixture"
    vr.publish_fixture(fixture_files(), output)
    assert vr.validate_fixture(output)["schema"] == "temporal_raw_fixture_v1"
    with pytest.raises(ValueError, match="overwrite"):
        vr.publish_fixture(fixture_files(), output)
    (output / "camera_front/frame_-5.jpg").write_bytes(b"changed raw image")
    with pytest.raises(ValueError, match="content changed"):
        vr.validate_fixture(output)


def test_fixture_rejects_unlisted_target_file_in_receipt(tmp_path):
    output = tmp_path / "fixture"
    vr.publish_fixture(fixture_files(), output)
    receipt_path = output / "fixture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["files_sha256"]["future_labels.npz"] = "a" * 64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="file list"):
        vr.validate_fixture(output)


def test_cache_sample_requires_exact_unique_row():
    dataset = [{"x": torch.tensor([1.])}, {"x": torch.tensor([2.])}]
    selected = vr.cache_sample(dataset, np.asarray([11, 30]), 30)
    assert torch.equal(selected["x"], torch.tensor([[2.]]))
    with pytest.raises(ValueError, match="absent or duplicated"):
        vr.cache_sample(dataset, [11, 30], 40)
    with pytest.raises(ValueError, match="absent or duplicated"):
        vr.cache_sample(dataset, [30, 30], 30)
