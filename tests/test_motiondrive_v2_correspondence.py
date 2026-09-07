import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_motiondrive_v2_correspondence import aggregate_records, correlation_summary, shift_pixels


def test_shift_sign_and_non_wrapping_boundaries():
    x = torch.zeros(1, 1, 10, 12)
    x[0, 0, 4, 5] = 1
    y = shift_pixels(x, 2, -1)
    assert y[0, 0, 3, 7] == 1
    assert y.sum() == 1
    assert shift_pixels(x, 20, 0).sum() == 0


def test_cost_peak_displacement_sign_and_exact_identity():
    torch.manual_seed(1)
    x = torch.randn(1, 32, 20, 22)
    same = correlation_summary(x, x, expected_shift=(0, 0))
    assert same["center_cosine"] == pytest.approx(1., abs=1e-6)
    assert same["center_peak_fraction"] == 1.
    for dx, dy in [(1, 0), (-1, 0), (0, 2), (1, -2)]:
        shifted = correlation_summary(x, shift_pixels(x, dx, dy), expected_shift=(dx, dy))
        assert shifted["peak_dx_cells"] == dx
        assert shifted["peak_dy_cells"] == dy
        assert shifted["nearest_bin_fraction"] == 1.
        assert shifted["peak_error_cells"] == 0.


def test_empty_roi_rejected_and_frame_aggregation():
    with pytest.raises(ValueError):
        correlation_summary(torch.ones(1, 4, 8, 8), torch.ones(1, 4, 8, 8))
    records = [{"split": "train", "mode": "m", "representation": "r", "level": "p2", "condition": "c",
                "metrics": {"center_cosine": value}} for value in (.1, .3)]
    r = aggregate_records(records)[0]
    assert r["n_frames"] == 2
    assert r["metrics"]["center_cosine"]["mean"] == pytest.approx(.2)
    assert r["metrics"]["center_cosine"]["std_across_frames"] == pytest.approx(.1)
