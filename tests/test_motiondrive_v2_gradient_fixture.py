import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_motiondrive_v2_gradient_fixture import (check_cpu_tree, ensure_new_pair, fixed_train_selection,
                                                   ordered_samples, publish_json_new, publish_torch_new)


def manifest():
    train = [f"scene{i:03d}" for i in range(203)]
    return {"splits": {"train": train[::-1], "tune": ["tune"], "val": ["val"], "historical_val": ["val"]},
            "scene_to_session": {scene: scene for scene in [*train, "tune", "val"]}}


def test_train_selection_is_name_sorted_and_predeclared():
    selected = fixed_train_selection(manifest())
    assert selected == [(f"scene{i:03d}", frame) for i in range(4) for frame in (30, 180)]
    bad = manifest();bad["splits"]["train"].pop()
    with pytest.raises(ValueError, match="train203"):
        fixed_train_selection(bad)


def test_source_rows_preserved_without_using_other_splits():
    selected = fixed_train_selection(manifest())
    class FakeDataset:
        rows = np.arange(8)
        scene_names = np.asarray([scene for scene, _ in selected[::-1]])
        arr = {"frame": np.asarray([frame for _, frame in selected[::-1]])}
        def __getitem__(self, index):
            return {"scenario": self.scene_names[index], "frame": int(self.arr["frame"][index]), "row": index}
    source = FakeDataset()
    result = ordered_samples(source, selected)
    assert [row["row"] for row in result] == list(range(7, -1, -1))
    assert [(row["scenario"], row["frame"]) for row in result] == selected
    source.rows = np.arange(7)
    with pytest.raises(ValueError, match="train8"):
        ordered_samples(source, selected)


def test_cpu_tree_rejects_gradients_nonfinite_and_unsupported_values():
    check_cpu_tree({"x": torch.ones(2), "mask": torch.ones(2, dtype=torch.bool), "names": ["a", "b"]})
    with pytest.raises(ValueError, match="requires_grad"):
        check_cpu_tree({"x": torch.ones(2, requires_grad=True)})
    with pytest.raises(ValueError, match="비유한"):
        check_cpu_tree({"x": torch.tensor([float("nan")])})
    with pytest.raises(ValueError, match="부적합"):
        check_cpu_tree({"x": np.ones(2)})


def test_exclusive_cpu_torch_roundtrip_and_existing_pair_preserved(tmp_path):
    output, provenance = tmp_path / "fixture.pt", tmp_path / "fixture.provenance.json"
    fixture = {"batch": {"images": torch.arange(8.), "valid": torch.tensor([True, False]), "scenario": ["a", "b"]},
               "metadata": {"split": "train", "batch_slices": [[0, 4], [4, 8]]}}
    ensure_new_pair(output, provenance)
    publish_torch_new(output, fixture)
    actual = torch.load(output, map_location="cpu", weights_only=True)
    assert torch.equal(actual["batch"]["images"], fixture["batch"]["images"])
    assert actual["batch"]["valid"].dtype == torch.bool
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        ensure_new_pair(output, provenance)
    with pytest.raises(FileExistsError):
        publish_torch_new(output, fixture)
    assert output.read_bytes() == before
    publish_json_new(provenance, {"train_only": True})
    with pytest.raises(FileExistsError):
        publish_json_new(provenance, {"train_only": False})
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_provenance_prevents_any_new_tensor_file(tmp_path):
    output, provenance = tmp_path / "fixture.pt", tmp_path / "fixture.json"
    publish_json_new(provenance, {"existing": True})
    with pytest.raises(FileExistsError):
        ensure_new_pair(output, provenance)
    assert not output.exists()
    with pytest.raises(ValueError, match="별도"):
        ensure_new_pair(output, output)
