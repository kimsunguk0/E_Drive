#!/usr/bin/env python3
"""Pytest gates plus an optional real-calibration CLI smoke for Phase-5B.

Run the portable unit gates with::

    python -m pytest -q tests/test_sparse_scoredrive.py

Run the real 768x432 fixture smoke with the CLI arguments shown in the report.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make direct ``pytest`` collection work without a caller-supplied PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sparse_scoredrive import SparseScoreDrive, project_candidate_points
from scoredrive_api import shortlist_np


def _make_synthetic_bank(path: Path) -> Path:
    """Small-on-disk, canonical-shape bank for portable unit tests."""
    k, t = 1024, 10
    abs5 = np.zeros((k, t, 2), np.float32)
    progress = np.linspace(0.05, 1.0, t, dtype=np.float32)
    abs5[:, :, 0] = np.arange(k, dtype=np.float32)[:, None] / 1024.0 * progress
    inc5 = np.diff(abs5, axis=1, prepend=np.zeros_like(abs5[:, :1]))
    index = np.arange(k, dtype=np.float32)
    distance = np.abs(index[:, None] - index[None, :]) / 1024.0
    np.fill_diagonal(distance, np.inf)
    np.savez(
        path,
        candidate_xy_abs_5s=abs5,
        candidate_xy_inc_5s=inc5,
        candidate_ids=np.arange(k, dtype=np.int64),
        anchor_dist=distance.astype(np.float32),
        nms_tau=np.asarray(0.01, np.float64),
    )
    return path


def test_goal_free_signature_and_stable_shortlist(tmp_path):
    """C1 plus exact parity with the audited NumPy shortlist on ties."""
    bank_path = _make_synthetic_bank(tmp_path / "bank.npz")
    model = SparseScoreDrive(bank_path).eval()
    signature = inspect.signature(model.forward)
    assert tuple(signature.parameters) == ("images", "lidar2img")
    assert not ({"goal", "cmd", "command", "status"} & set(signature.parameters))
    rng = np.random.default_rng(5)
    distance = model.anchor_dist.numpy()
    tau = float(model.nms_tau)
    for probe in (rng.standard_normal((4, 1024)).astype(np.float32),
                  np.zeros((4, 1024), np.float32)):
        probe[:, ::7] = 1.0
        got = model._stable_score3_nms9(torch.from_numpy(probe)).numpy()
        expected = np.stack([shortlist_np(row, distance, tau) for row in probe])
        assert np.array_equal(got, expected)


def test_zero_evidence_complete_candidate_api(tmp_path):
    """Portable C3/C4: zero input, hidden full-K, exact row/prefix outputs."""
    bank_path = _make_synthetic_bank(tmp_path / "bank.npz")
    model = SparseScoreDrive(bank_path).eval()
    images = torch.zeros(1, 6, 3, 64, 96)
    lidar2img = torch.zeros(1, 6, 4, 4)
    with torch.inference_mode():
        output = model(images, lidar2img)
    assert set(output) == {
        "candidate_xy_abs_5s", "candidate_xy_inc_5s",
        "visual_logits", "candidate_ids",
    }
    assert output["candidate_xy_abs_5s"].shape == (1, 12, 10, 2)
    assert output["candidate_xy_inc_5s"].shape == (1, 12, 10, 2)
    assert output["visual_logits"].shape == (1, 12)
    assert output["candidate_ids"].shape == (1, 12)
    assert torch.count_nonzero(output["visual_logits"]) == 0
    assert int(output["candidate_ids"][0, 0]) == 0
    ids = output["candidate_ids"][0]
    assert torch.equal(output["candidate_xy_abs_5s"][0], model.candidate_abs[ids])
    assert torch.equal(output["candidate_xy_inc_5s"][0], model.candidate_inc[ids])
    assert torch.equal(output["candidate_xy_inc_5s"][0, :, :6], model.candidate_inc[ids, :6])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    fixture = np.load(args.fixture, allow_pickle=False)
    images = torch.from_numpy(fixture["images"])[None].to(device)
    lidar2img = torch.from_numpy(fixture["lidar2img"])[None].to(device)
    model = SparseScoreDrive(args.bank).eval().to(device)

    signature = inspect.signature(model.forward)
    assert tuple(signature.parameters) == ("images", "lidar2img")
    assert not ({"goal", "cmd", "command", "status"} & set(signature.parameters))
    with torch.inference_mode():
        output = model(images, lidar2img)
        zero_output = model(torch.zeros_like(images), lidar2img)
        grid, visible = project_candidate_points(
            model.candidate_abs, lidar2img, images.shape[-2:], model.heights)

    # Stable shortlist must mirror the already-audited offline implementation,
    # including exact ties.
    rng = np.random.default_rng(5)
    dist = model.anchor_dist.detach().cpu().numpy()
    tau = float(model.nms_tau)
    for probe in (rng.standard_normal((4, 1024)).astype(np.float32),
                  np.zeros((4, 1024), np.float32)):
        probe[:, ::7] = 1.0
        got = model._stable_score3_nms9(torch.from_numpy(probe).to(device)).cpu().numpy()
        ref = np.stack([shortlist_np(row, dist, tau) for row in probe])
        assert np.array_equal(got, ref)

    expected_keys = {
        "candidate_xy_abs_5s", "candidate_xy_inc_5s",
        "visual_logits", "candidate_ids",
    }
    assert set(output) == expected_keys  # full-K is not externally visible
    assert output["candidate_xy_abs_5s"].shape == (1, 12, 10, 2)
    assert output["candidate_xy_inc_5s"].shape == (1, 12, 10, 2)
    assert output["visual_logits"].shape == (1, 12)
    assert output["candidate_ids"].shape == (1, 12)
    shortlist = output["candidate_ids"][0]
    assert torch.equal(output["candidate_xy_abs_5s"][0], model.candidate_abs[shortlist])
    assert torch.equal(output["candidate_xy_inc_5s"][0], model.candidate_inc[shortlist])
    assert torch.equal(
        output["candidate_xy_inc_5s"][0, :, :6], model.candidate_inc[shortlist, :6])
    # C3: selecting any row is an exact slice, with no coordinate mutation.
    selected_local = 7
    selected = output["candidate_xy_inc_5s"][0, selected_local, :6]
    assert torch.equal(selected, model.candidate_inc[shortlist[selected_local], :6])
    # C4: exact zero visual evidence annihilates every visual logit.  Stable
    # ordering then makes candidate 0 the safe first row.
    assert torch.count_nonzero(zero_output["visual_logits"]) == 0
    assert int(zero_output["candidate_ids"][0, 0]) == 0
    assert torch.isfinite(grid).all()
    visible_any = visible.any(dim=(0, 1, 4)).float().mean().item()
    assert visible_any > 0.5, visible_any
    report = {
        "C1_goal_free_signature": True,
        "C3_exact_row_and_prefix": True,
        "C4_zero_logits_and_candidate0": True,
        "stable_shortlist_matches_offline_random_and_ties": True,
        "output_shapes": {k: list(v.shape) for k, v in output.items()},
        "real_projection_visible_waypoint_fraction": visible_any,
        "real_projection_visible_fraction_all_rays": visible.float().mean().item(),
        "fixture_manifest": str(fixture["manifest"]),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
