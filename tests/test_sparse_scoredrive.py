#!/usr/bin/env python3
"""C1/C3/C4 and real-calibration smoke for the Phase-5B skeleton."""
from __future__ import annotations

import argparse
import inspect
import json

import numpy as np
import torch

from sparse_scoredrive import SparseScoreDrive, project_candidate_points
from scoredrive_api import shortlist_np


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
