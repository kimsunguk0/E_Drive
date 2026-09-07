import pytest
import torch
from torch.nn import functional as F

from scripts.benchmark_motiondrive_v2_flow_candidate import (
    prepare_pairs, summary_ms, validate_outputs, foreign_pids, sha256,
)


def test_pairs_only_images_and_expected_resize():
    torch.manual_seed(0)
    mean = torch.tensor([.485, .456, .406])[None, None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, None, :, None, None]
    current = torch.rand(1, 6, 3, 24, 32)
    past = torch.rand(1, 4, 3, 12, 16)
    batch = {"images": (current - mean) / std, "history_images": (past - mean) / std,
             "goal_xy": object(), "state_gt": object(), "history_transforms": object()}
    pairs = prepare_pairs(batch)
    expected = 2 * F.interpolate(current[:, 0], size=(12, 16), mode="bilinear",
                                 align_corners=False, antialias=True) - 1
    assert torch.allclose(pairs[1][0], expected, atol=4e-7, rtol=0)
    assert torch.allclose(pairs[4][1], 2 * past[0] - 1, atol=4e-7, rtol=0)
    assert torch.equal(pairs[4][0][0], pairs[1][0][0])
    assert torch.equal(pairs[4][1][0], pairs[1][1][0])
    assert all(x.dtype == torch.float32 and x.is_contiguous() for p in pairs.values() for x in p)


def test_invalid_image_range():
    with pytest.raises(ValueError, match="normalized"):
        prepare_pairs({"images": torch.full((1, 6, 3, 24, 32), 99.),
                       "history_images": torch.zeros(1, 4, 3, 12, 16)})


def test_nonfinite_images():
    with pytest.raises(ValueError, match="Nonfinite"):
        prepare_pairs({"images": torch.full((1, 6, 3, 24, 32), float("nan")),
                       "history_images": torch.zeros(1, 4, 3, 12, 16)})


def test_output_validation_all_updates():
    flow = torch.zeros(1, 2, 216, 384)
    assert validate_outputs([flow] * 4, 1, 4)["all_updates_finite"]
    with pytest.raises(ValueError, match="per update"):
        validate_outputs([flow], 1, 4)
    with pytest.raises(ValueError, match="precision"):
        validate_outputs([flow.half()] * 4, 1, 4)
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_outputs([torch.full_like(flow, float("nan")), flow, flow, flow], 1, 4)


def test_summary_and_occupancy():
    result = summary_ms(list(range(1, 51)))
    assert result["median_ms"] == 25.5 and result["max_ms"] == 50
    assert result["p90_ms"] == pytest.approx(45.1)
    assert foreign_pids([12, 13, 12], 12) == [13]
    assert foreign_pids([12], 12) == []
    with pytest.raises(ValueError):
        summary_ms([1] * 49)


def test_hash(tmp_path):
    path = tmp_path / "empty"
    path.touch()
    assert sha256(path) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
