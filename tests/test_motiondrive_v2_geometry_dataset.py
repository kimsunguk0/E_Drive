from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_motiondrive_v2_geometry_dataset import REAR, compare_samples, selected_samples


def samples():
    original = {"lidar2img": torch.eye(4).repeat(6, 1, 1), "images": torch.zeros(2, 3),
                "valid": torch.tensor([True, False]), "row": 30, "scenario": "train"}
    changed = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in original.items()}
    changed["lidar2img"][REAR, 1, 2] -= 182.4
    return original, changed


def test_only_rear_change_passes():
    result = compare_samples(*samples())
    assert result["checks"]["lidar2img"] == "rear_only_changed"
    assert result["checks"]["images"] == "bitwise_equal"


@pytest.mark.parametrize("field", ["images", "valid", "row", "scenario", "other_camera"])
def test_nonrear_change_fails(field):
    old, new = samples()
    if field == "images":
        new[field][0, 0] = 1
    elif field == "valid":
        new[field][0] = False
    elif field == "row":
        new[field] = 31
    elif field == "scenario":
        new[field] = "different"
    else:
        new["lidar2img"][0, 0, 0] += 1
    with pytest.raises(ValueError):
        compare_samples(old, new)


def test_same_edition_cannot_claim_correction():
    old, _ = samples()
    with pytest.raises(ValueError, match="바뀌지"):
        compare_samples(old, old)


def test_prespecified_train_and_tune_selection_excludes_quarantine():
    manifest = {"splits": {"train": ["tr4", "tr2", "tr1", "tr3", "tr5"], "tune": ["tu2", "tu1"],
                           "val": ["heldout"], "historical_val": ["heldout"]},
                "scene_to_session": {name: name for name in ["tr1", "tr2", "tr3", "tr4", "tr5", "tu1", "tu2", "heldout"]}}
    selected = selected_samples(manifest)
    assert selected[:2] == [("train", "tr1", 30), ("train", "tr1", 180)]
    assert selected[-2:] == [("tune", "tu1", 30), ("tune", "tu1", 295)]
    assert len(selected) == 10
    assert not {"heldout", "tr5", "tu2"} & {scene for _, scene, _ in selected}
