import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/audit_motiondrive_v2_rear_crop.py"
SPEC = importlib.util.spec_from_file_location("rear_crop_audit", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pixel_metrics_avoid_uint8_underflow():
    result = MODULE.pixel_metrics(np.array([0], np.uint8), np.array([255], np.uint8))
    assert result["mae_uint8"] == 255
    assert result["rmse_uint8"] == 255
    assert not result["bitwise_equal"]


def test_crop_shift_direction_and_magnitude():
    row = {"euler": [0., 0., 0.], "translation": [0., 0., 0.]}
    nk = np.array([[400., 0., 960.], [0., 400., 678.], [0., 0., 1.]])
    top = MODULE.projected_matrix(row, nk, [0, 0, 1920, 1080])
    bottom = MODULE.projected_matrix(row, nk, [0, 456, 1920, 1080])
    point = np.array([1., 2., 10., 1.])
    uv_top, uv_bottom = top @ point, bottom @ point
    assert uv_bottom[1] / uv_bottom[2] - uv_top[1] / uv_top[2] == pytest.approx(-182.4)
    assert uv_bottom[0] / uv_bottom[2] == uv_top[0] / uv_top[2]


def test_nontrain_scene_rejected_before_any_data_access(tmp_path):
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"splits": {"train": ["allowed"]}}))
    with pytest.raises(ValueError, match="train"):
        MODULE.audit(tmp_path / "absent", "forbidden", 30, manifest)
