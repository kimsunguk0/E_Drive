from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
from PIL import Image
import pytest
import torch

# Local review snapshot only; deployed bundle provides its own models package.
_local_core = Path("/home/a/adcl_analysis_20260910/core")
if _local_core.is_dir():
    sys.path.append(str(_local_core))
import temporal_deployment as td


class DoNotIterate:
    def __iter__(self):
        raise AssertionError("unused past poses were accessed")


@pytest.fixture
def pixels(monkeypatch):
    torch.set_num_threads(1)
    builds = []
    def geometry(records):
        builds.append(records)
        return [SimpleNamespace(name=name, raw_wh=(1920, 1080), crop_xy=(0, 0),
                                cached_intrinsic=np.eye(3), lidar2img=np.eye(4, dtype=np.float32) * records[0]["scale"])
                for name in td.CAMERAS]
    def compatible(raw, g):
        value = raw[0]
        return Image.new("RGB", (768, 432), (value, value + 1, value + 2)), {"raw_jpeg_sha256": str(value)}
    monkeypatch.setattr(td, "build_camera_geometry", geometry)
    monkeypatch.setattr(td, "cache_compatible_image", compatible)
    return builds


def poses():
    result = []
    for frame in [*range(-30, 1), 50]:
        t = frame / 10.
        item = {"frame": frame, "x": 10 * t + .5 * t * t, "y": -.3 * t + .1 * t * t,
                "z": 0., "roll": 0., "pitch": 0., "yaw": 0.}
        if frame < -10:
            item.update(x=float("nan"), yaw=float("nan"))
        if frame == 50:
            item.update(x=30., y=1., roll=float("nan"), pitch=float("nan"), yaw=float("nan"))
        result.append(item)
    return result


def loader(calls, value=20):
    def read(camera, frame):
        calls.append((camera, frame))
        return bytes([value - frame])
    return read


def test_AB_no_pose_or_status_access_and_exact_read_sets(pixels, monkeypatch):
    monkeypatch.setattr(td, "causal_status4_from_records", lambda r: (_ for _ in ()).throw(AssertionError("state called")))
    for mode, count in [("repeat", 3), ("real", 5)]:
        calls = []
        with td.TemporalRawInputAdapter(mode, False, "none", camera_workers=1) as adapter:
            p = adapter.prepare_records([{"scale": 1}], DoNotIterate(), loader(calls))
        assert len(calls) == count
        assert calls[:3] == [(camera, 0) for camera in td.CAMERAS]
        assert calls[3:] == ([] if mode == "repeat" else [("camera_front", -1), ("camera_front", -5)])
        assert set(p.inputs) == td.MODEL_KEYS and p.selector_inputs == {}
        assert p.metadata["pose_frames_used"] == [] and not p.metadata["causal_status_computed"]
        assert p.inputs["history_images"].shape == (1, 2, 3, 256, 512)
        if mode == "repeat":
            assert torch.equal(p.inputs["history_images"][0, 0], p.inputs["images"][0, 1])
        with pytest.raises(ValueError, match="closed"):
            adapter.prepare_records([{"scale": 1}], None, loader([]))


def test_C_state_matches_kinematics_and_goal_is_separate(pixels):
    with td.TemporalRawInputAdapter("real", True, "selection", 1) as adapter:
        p = adapter.prepare_records([{"scale": 1}], poses(), loader([]))
        assert set(p.inputs) == td.MODEL_KEYS | {"perception_status"}
        assert torch.allclose(p.inputs["perception_status"], torch.tensor([[10., -.3, 1., .2]]), atol=1e-6, rtol=0)
        assert torch.equal(p.goal_xy, torch.tensor([[30., 1.]]))
        assert set(p.selector_inputs) == {"goal_xy"}
        assert "goal_xy" not in p.inputs and "status" not in p.inputs
        altered = poses()
        altered[-1]["x"] = 40.
        q = adapter.prepare_records([{"scale": 1}], altered, loader([]))
        assert all(torch.equal(p.inputs[k], q.inputs[k]) for k in p.inputs)
        assert not torch.equal(p.goal_xy, q.goal_xy)


def test_AB_goal_uses_only_current_and_future_XYZ(pixels, monkeypatch):
    records = [r for r in poses() if r["frame"] in (0, 50)]
    monkeypatch.setattr(td, "causal_status4_from_records", lambda r: (_ for _ in ()).throw(AssertionError("state called")))
    with td.TemporalRawInputAdapter("real", False, "selection", 1) as adapter:
        p = adapter.prepare_records([{"scale": 1}], records, loader([]))
    assert p.metadata["pose_frames_used"] == [0, 50]
    assert p.metadata["causal_status_computed"] is False
    assert torch.equal(p.goal_xy, torch.tensor([[30., 1.]]))


def test_C_rejects_missing_duplicate_or_nonfinite_required_pose():
    rows = poses()
    with pytest.raises(ValueError, match="Missing"):
        td.causal_status4_from_records([r for r in rows if r["frame"] != -1])
    with pytest.raises(ValueError, match="Duplicate"):
        td.causal_status4_from_records(rows + [dict(rows[-2])])
    rows[-2]["x"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        td.causal_status4_from_records(rows)


def test_calibration_cache_invalidation_and_no_image_feature_cache(pixels):
    with td.TemporalRawInputAdapter("real", False, "none", 3) as adapter:
        a = adapter.prepare_records([{"scale": 1}], None, loader([], 20))
        b = adapter.prepare_records([{"scale": 1}], None, loader([], 40))
        c = adapter.prepare_records([{"scale": 2}], None, loader([], 20))
        d = adapter.prepare_records([{"scale": 1}], None, loader([], 20))
    assert len(pixels) == 3
    assert not torch.equal(a.inputs["images"], b.inputs["images"])
    assert not torch.equal(a.inputs["lidar2img"], c.inputs["lidar2img"])
    assert all(torch.equal(a.inputs[k], d.inputs[k]) for k in a.inputs)


def test_C_without_goal_never_needs_future_pose(pixels):
    rows = [r for r in poses() if -10 <= r["frame"] <= 0]
    with td.TemporalRawInputAdapter("repeat", True, "none", 1) as adapter:
        p = adapter.prepare_records([{"scale": 1}], rows, loader([]))
    assert p.selector_inputs == {} and p.metadata["pose_frames_used"] == list(range(-10, 1))


def test_clip_reader_skips_pose_file_for_AB_and_filters_C(pixels, monkeypatch, tmp_path):
    reads = []
    package = ModuleType("pyarrow")
    parquet = ModuleType("pyarrow.parquet")
    package.parquet = parquet
    def read_table(path, columns, filters=None):
        reads.append((Path(path).name, columns, filters))
        if Path(path).name == "calibration.parquet":
            result = [{"scale": 1}]
        else:
            assert filters == [("frame", "in", [*range(-10, 1), 50])]
            result = [r for r in poses() if r["frame"] in filters[0][2]]
        return SimpleNamespace(to_pylist=lambda: result)
    parquet.read_table = read_table
    monkeypatch.setitem(sys.modules, "pyarrow", package)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    (tmp_path / "calibration.parquet").write_bytes(b"synthetic calibration bytes")
    for camera in td.CAMERAS:
        (tmp_path / camera).mkdir()
        (tmp_path / camera / "frame_0.jpg").write_bytes(bytes([20]))
    with td.TemporalRawInputAdapter("repeat", False, "none", 1) as adapter:
        result = adapter.prepare_clip(tmp_path)
    assert [r[0] for r in reads] == ["calibration.parquet"]
    assert not result.metadata["pose_file_opened"]
    with td.TemporalRawInputAdapter("repeat", True, "selection", 1) as adapter:
        result = adapter.prepare_clip(tmp_path)
    assert reads[-1][0] == "ego_pose.parquet"
    assert result.metadata["pose_parquet_frames_selected"] == [*range(-10, 1), 50]
