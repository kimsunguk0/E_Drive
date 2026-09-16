#!/usr/bin/env python3
"""One forward pass, every shape in the motion branch recorded, nothing guessed.

Answers the reviewer's request: image and feature shapes, how the motion pairs
are built and in what dt order, the shapes immediately before and after the
local correlation together with its radius and where the pooling happens, the
position and dt embedding, and what reaches the planner.

No training, no measurement claim, no architectural judgement here; this is a
shape record for deciding the next observation contrast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models import motiondrive_v2_inputs as adapter
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from models.motiondrive_v2 import motion_encoder as motion_module
import motiondrive_v2_training as mt

OUT = ROOT / "reports/md_exp_diagnosis_20260915/motion_shape_trace.json"
FIXTURE = ROOT / "data/etri/motiondrive_v2/deploy_fixture_train8/fixture_000"


def shape(x):
    return list(x.shape) if torch.is_tensor(x) else None


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--init", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    payload = torch.load(args.init, map_location="cpu", weights_only=False)
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    model = MotionDriveV2(config)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(f"cuda:{args.gpu}")
    model.to(device).eval()

    inputs = {k: v.to(device) for k, v in adapter.prepare_clip_inputs(FIXTURE).inputs.items()}
    trace = {"inputs": {k: shape(v) for k, v in inputs.items()}}

    # correlation is a free function; wrap it to capture its actual operands
    original_corr = motion_module.local_correlation
    corr_records = []

    def traced_corr(current, history, radius):
        result = original_corr(current, history, radius)
        corr_records.append({
            "current_in": shape(current), "history_in": shape(history),
            "radius": int(radius), "bins": (2 * int(radius) + 1) ** 2,
            "out": shape(result),
        })
        return result

    captured = {}

    def capture(name):
        def hook(module, args_in, output):
            entry = {"in": [shape(a) for a in args_in]}
            if isinstance(output, dict):
                entry["out"] = {k: shape(v) for k, v in output.items()}
            elif isinstance(output, (tuple, list)):
                entry["out"] = [shape(o) if torch.is_tensor(o) else
                                [shape(x) for x in o] for o in output]
            else:
                entry["out"] = shape(output)
            captured[name] = entry
        return hook

    handles = [
        model.backbone_fpn.register_forward_hook(capture("backbone_fpn")),
        model.motion_encoder.register_forward_hook(capture("motion_encoder")),
        model.motion_encoder.correlation_fuse.register_forward_hook(capture("correlation_fuse")),
        model.motion_encoder.time_embed.register_forward_hook(capture("time_embed")),
        model.motion_encoder.token_refine.register_forward_hook(capture("token_refine")),
        model.motion_encoder.state_head.register_forward_hook(capture("state_head")),
        model.motion_encoder.history_head.register_forward_hook(capture("history_head")),
        model.scene_encoder.register_forward_hook(capture("scene_encoder")),
        model.planner.register_forward_hook(capture("planner")),
    ]
    motion_module.local_correlation = traced_corr
    try:
        with torch.no_grad():
            out = model(**inputs)
    finally:
        motion_module.local_correlation = original_corr
        for handle in handles:
            handle.remove()

    current_hw = tuple(inputs["images"].shape[-2:])
    history_hw = tuple(inputs["history_images"].shape[-2:])
    corr_hw = tuple(corr_records[0]["current_in"][-2:])
    trace.update({
        "config": {
            "backbone_arch": config.backbone_arch, "channels": config.channels,
            "n_history": config.n_history,
            "history_frame_offsets": list(config.history_frame_offsets),
            "nominal_history_seconds": list(config.nominal_history_seconds),
            "motion_grid": list(config.motion_grid),
            "correlation_radius": config.correlation_radius,
            "correlation_channels": config.correlation_channels,
            "motion_input_mode": config.motion_input_mode,
            "grid_size": list(config.grid_size),
        },
        "modules": captured,
        "local_correlation_calls": corr_records,
        "outputs": {k: shape(v) for k, v in out.items()},
        "derived": {
            "current_image_hw": list(current_hw),
            "history_image_hw": list(history_hw),
            "history_is_half_resolution": [current_hw[0] // history_hw[0],
                                           current_hw[1] // history_hw[1]],
            "correlation_feature_hw": list(corr_hw),
            "stride_on_history_image": [history_hw[0] / corr_hw[0], history_hw[1] / corr_hw[1]],
            "stride_in_current_image_pixels": [current_hw[0] / corr_hw[0],
                                               current_hw[1] / corr_hw[1]],
            "search_halfwidth_in_current_pixels": [
                config.correlation_radius * current_hw[0] / corr_hw[0],
                config.correlation_radius * current_hw[1] / corr_hw[1]],
            "pooling_after_correlation_to": list(config.motion_grid),
            "pooling_note": ("correlation runs at the history feature resolution; the "
                             "adaptive_avg_pool2d to motion_grid happens AFTER correlation_fuse, "
                             "so the matching itself is not done at the motion grid"),
            "state_history_head_pool": "adaptive_avg_pool2d to (3,4) before the heads",
        },
        "pair_construction": {
            "mode": config.motion_input_mode,
            "described": ("the current front image is resized to the history size and stacked "
                          "in front of the four past frames, so one backbone pass covers "
                          "n_history+1 temporal images; the current feature for the motion "
                          "branch is taken from that same stack"),
            "dt_order_seconds": list(config.nominal_history_seconds),
            "dt_is_positive_elapsed_time": True,
        },
        "planner_inputs": {
            "scene_features": shape(out.get("scene_features")),
            "motion_features": shape(out.get("motion_features")),
            "state_hat": shape(out.get("state_hat")),
            "history_hat": shape(out.get("history_hat")),
        },
        "limitations": [
            "one clip, batch 1, inference mode; shapes only, no timing or quality claim",
            "this records what the graph does, it does not say whether the resolution or the "
            "search radius is adequate",
        ],
    })
    OUT.write_text(json.dumps(trace, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "current_image_hw": trace["derived"]["current_image_hw"],
        "history_image_hw": trace["derived"]["history_image_hw"],
        "correlation_feature_hw": trace["derived"]["correlation_feature_hw"],
        "correlation_calls": [{"radius": c["radius"], "bins": c["bins"],
                               "current_in": c["current_in"], "out": c["out"]}
                              for c in corr_records],
        "stride_in_current_image_pixels": trace["derived"]["stride_in_current_image_pixels"],
        "search_halfwidth_in_current_pixels": trace["derived"]["search_halfwidth_in_current_pixels"],
        "pooling_after_correlation_to": trace["derived"]["pooling_after_correlation_to"],
        "planner_inputs": trace["planner_inputs"],
    }, indent=1))


if __name__ == "__main__":
    main()
