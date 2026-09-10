import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

import temporal_data as td


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakePlanDataset:
    def __init__(self, root):
        self.base = root
        self.rows = np.asarray([30, 31], np.int64)
        self.frames = np.arange(40, dtype=np.int64)
        self.scenarios = np.asarray(["scene_a"])
        self.scen_idx = np.zeros(40, np.int64)
        self.status = np.zeros((2, 8), np.float32)
        self.status_mode = "zero"
        self.split = "train"
        self.image_size = (512, 256)
        self.seed, self.epoch, self.augment = 7, 0, False
        self.split_manifest = root / "split.json"
        self.split_manifest.write_text('{}\n')
        self.ego_cache = root / "ego.source"
        self.ego_cache.write_bytes(b"synthetic row-index source\n")
        self.calibration_path = root / "calibration.source"
        self.calibration_path.write_bytes(b"synthetic calibration\n")
        # Every point is behind all cameras. Thus no auxiliary pixel is visible.
        self.projection = torch.zeros(3, 4, 4)
        self.projection[:, 2, 3] = -1
        self.image_root = root / "images"
        front = self.image_root / "scene_a" / "camera_front"
        front.mkdir(parents=True)
        for frame in [25, 26, 29, 30, 31]:
            Image.new("RGB", (768, 432), (frame, frame * 2, frame * 3)).save(front / f"{frame:08d}.jpg", quality=100)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def provenance(self):
        return {"rows": 2, "status_mode": self.status_mode}

    def __getitem__(self, index):
        row = int(self.rows[index])
        return {"images": torch.full((3, 3, 256, 512), float(row)), "status": torch.zeros(8),
                "lidar2img": self.projection.clone(), "image_hw": torch.tensor([256., 512.]),
                "row": row, "frame": row, "scenario": "scene_a", "session": "session_a",
                "gt_plan": torch.full((6, 2), 99.), "goal_xy": torch.tensor([88., 77.])}


@pytest.fixture
def data(tmp_path):
    torch.set_num_threads(1)
    base = FakePlanDataset(tmp_path)
    status_root = tmp_path / "status"
    status_root.mkdir()
    status = np.asarray([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]], np.float32)
    artifact = status_root / "train.npz"
    np.savez(artifact, row=base.rows, frame=base.rows, status5=status)
    metadata = {"split_manifest_sha256": sha(base.split_manifest), "ego_cache_sha256": sha(base.ego_cache),
                "causal_contract": {"future_values_used": False, "fields": ["vx", "vy", "ax", "ay", "yaw_rate"],
                                    "frames": list(range(-10, 1)), "seconds": (np.arange(-10, 1) / 10.).tolist()},
                "artifacts": {"train": {"file": "train.npz", "sha256": sha(artifact), "rows_sha256": td.rows_sha(base.rows)}}}
    (status_root / "overlay_manifest.json").write_text(json.dumps(metadata))
    aux = tmp_path / "aux"
    aux.mkdir()
    (aux / "supervision_manifest.json").write_text(json.dumps({
        "split_manifest_sha256": sha(base.split_manifest), "ego_cache_sha256": sha(base.ego_cache),
        "canonical_calibration_sha256": sha(base.calibration_path), "grid_shape": [64, 48],
        "grid_extent": [-10., 70., -32., 32.]}))
    np.savez(aux / "scene_a.npz", row=base.rows, frame=base.rows,
             occ_target=np.ones((2, 1, 64, 48), np.float32), lane_target=np.ones((2, 1, 64, 48), np.float32),
             occ_valid=np.ones((2, 1, 64, 48), bool), lane_valid=np.zeros((2, 1, 64, 48), bool))
    (aux / "scene_a.json").write_text(json.dumps({"scene": "scene_a", "split_manifest_sha256": sha(base.split_manifest),
                                                 "ego_cache_sha256": sha(base.ego_cache)}))
    return base, status_root, aux


def wrapped(data, **kwargs):
    base, status, aux = data
    return td.TemporalPlanDataset(base, causal_status_root=status, supervision_root=aux, **kwargs)


def test_repeat_preserves_current_and_uses_independent_tensor(data):
    ds = wrapped(data, history_mode="repeat")
    item = ds[0]
    assert torch.equal(item["images"], data[0][0]["images"])
    assert item["history_images"].shape == (2, 3, 256, 512)
    for history in item["history_images"]:
        assert torch.equal(history, item["images"][1])
    item["history_images"][0].zero_()
    assert item["images"].min() == 30 and item["history_images"][1].min() == 30
    assert torch.equal(item["time_offsets"], torch.tensor([.1, .5]))


def test_real_history_uses_scene_frame_lags_and_missing_frame_fails(data):
    ds = wrapped(data, history_mode="real")
    item = ds[0]
    assert torch.equal(item["images"], data[0][0]["images"])
    assert not torch.equal(item["history_images"][0], item["history_images"][1])
    first = data[0].image_root / "scene_a/camera_front/00000029.jpg"
    with Image.open(first) as image:
        rgb = np.asarray(image)[0, 0].astype(np.float32)
    expected = (rgb / 255. - td.MEAN) / td.STD
    assert torch.allclose(item["history_images"][0, :, 0, 0], torch.from_numpy(expected), atol=0, rtol=0)
    first.unlink()
    with pytest.raises(FileNotFoundError):
        ds[0]


def test_default_boundary_excludes_privileged_and_labels_C_explicit(data):
    item = wrapped(data, history_mode="repeat")[0]
    base_inputs = td.model_inputs(item)
    assert set(base_inputs) == set(td.MODEL_INPUT_KEYS)
    assert not {"status", "causal_status4", "state_target", "gt_plan", "goal_xy", "row", "session"} & set(base_inputs)
    item["state_target"].fill_(999)
    assert torch.equal(item["causal_status4"], torch.tensor([1., 2., 3., 4.]))
    common = td.model_inputs(item, common_status=True)
    assert set(common) == set(td.MODEL_INPUT_KEYS) | {"perception_status"}
    assert common["perception_status"] is item["causal_status4"]
    assert not {"status", "goal_xy", "state_target"} & set(common)
    with pytest.raises(ValueError):
        td.model_inputs(item, common_status="false")


def test_auxiliary_unknown_and_out_of_view_are_never_valid(data):
    ds = wrapped(data, history_mode="repeat", auxiliary=True)
    item = ds[0]
    assert item["occ_target"].shape == (1, 64, 48)
    assert item["occ_target"].all()  # Keep target values even when loss is masked.
    assert not item["occ_valid"].any() and not item["lane_valid"].any()
    assert item["occ_valid"].dtype == torch.bool


def test_causal_source_hash_and_time_contract_fail_closed(data):
    base, status, _ = data
    p = status / "overlay_manifest.json"
    m = json.loads(p.read_text())
    m["causal_contract"]["future_values_used"] = True
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Past-only"):
        wrapped(data)
    m["causal_contract"]["future_values_used"] = False
    p.write_text(json.dumps(m))
    with (status / "train.npz").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="hash"):
        wrapped(data)


def test_reject_raw_planner_status_or_unknown_augmentation(data):
    data[0].status_mode = "causal_selection"
    with pytest.raises(ValueError, match="constant zero"):
        wrapped(data)
    data[0].status_mode = "zero"
    data[0].augment = True
    with pytest.raises(ValueError, match="pinned"):
        wrapped(data)


def test_aux_row_frame_binding_rejects_wrong_source(data):
    ds = wrapped(data, history_mode="repeat", auxiliary=True)
    p = data[2] / "scene_a.npz"
    with np.load(p, allow_pickle=False) as z:
        content = {k: z[k] for k in z.files}
    content["frame"] = content["frame"] + 1
    np.savez(p, **content)
    with pytest.raises(ValueError, match="row/scene/frame"):
        ds[0]


def test_aux_cache_avoids_rehash_and_uses_compact_binary_storage(data, monkeypatch):
    ds = wrapped(data, history_mode="repeat", auxiliary=True)
    monkeypatch.setattr(td, "file_sha", lambda path: (_ for _ in ()).throw(AssertionError("per-row hash")))
    first = ds._aux("scene_a")
    second = ds._aux("scene_a")
    assert first is second
    for key in ("occ_target", "occ_valid", "lane_target", "lane_valid"):
        assert first[key].dtype == np.bool_ and not first[key].flags.writeable
    assert ds[0]["occ_target"].dtype == torch.float32
