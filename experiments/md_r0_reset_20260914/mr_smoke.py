#!/usr/bin/env python3
"""Pre-training checks for the MR arms: shapes, detail, flip coverage, cost."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments/md_r0_reset_20260914"))

import motiondrive_v2_training as mt
import motiondrive_v2_flip_augment as flip_module
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from models.motiondrive_v2 import motion_encoder as motion_module
import matching_resolution as mr

OUT = ROOT / "reports/md_exp_diagnosis_20260915/mr_smoke.json"
SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json"
SUPERVISION = ROOT / "data/etri/motiondrive_v2/r0reset_tplus_geometry_v2"
INIT = ROOT / "work_dirs/md_r0_reset_20260914/r0_init_tplus.pth"


def build(detail, device, augment):
    from motiondrive_v2_data import MotionDriveDataset
    base = MotionDriveDataset(
        data_root="/tmp/pm97", split_manifest=str(SPLIT), split="train",
        supervision_root=str(SUPERVISION), min_frame=30, frame_stride=1,
        augment=augment, seed=0, history_contract="control")
    return mr.MotionCanvasDataset(base, detail)


def main() -> None:
    device = torch.device("cuda:0")
    payload = torch.load(INIT, map_location="cpu", weights_only=False)
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    result = {"schema_version": 1}

    # 1. canvas contract and detail difference
    contracts = {}
    for detail in ("native", "lowdetail"):
        dataset = build(detail, device, augment=False)
        item = dataset[0]
        contracts[detail] = mr.assert_canvas_contract(item, detail)
    contracts["detail_actually_differs"] = (
        contracts["native"]["mean_abs_vertical_gradient"]
        > contracts["lowdetail"]["mean_abs_vertical_gradient"] * 1.15)
    result["canvas_contract"] = contracts

    # 2. flip must reach the canvas tensors
    original_flip = flip_module.flip_item
    wrapped = mr.wrap_flip_item(original_flip)
    dataset = build("native", device, augment=False)
    item = dataset[0]
    flipped = wrapped(item, 768, 384)
    naive = original_flip(item, 768, 384)
    result["flip_coverage"] = {
        "wrapped_flips_canvas": bool(torch.equal(
            flipped[mr.MOTION_CURRENT_KEY], torch.flip(item[mr.MOTION_CURRENT_KEY], dims=[-1]))),
        "unwrapped_would_leave_it_unflipped": bool(torch.equal(
            naive[mr.MOTION_CURRENT_KEY], item[mr.MOTION_CURRENT_KEY])),
        "double_flip_identity": bool(torch.equal(
            wrapped(flipped, 768, 384)[mr.MOTION_CURRENT_KEY], item[mr.MOTION_CURRENT_KEY])),
    }

    # 3. model surgery: shapes, radius, what got reinitialised
    model = MotionDriveV2(config)
    missing = model.load_state_dict(payload["model"], strict=True)
    rebuild = mr.rebuild_correlation_fuse(model, mr.NEW_RADIUS)
    mr.install(model, "native")
    model.to(device).eval()
    result["module_change"] = rebuild

    corr_calls = []
    original_corr = motion_module.local_correlation

    def traced(current, history, radius):
        out = original_corr(current, history, radius)
        corr_calls.append({"current_in": list(current.shape), "radius": int(radius),
                           "bins": (2 * int(radius) + 1) ** 2, "out": list(out.shape)})
        return out

    from torch.utils.data import DataLoader
    loader = DataLoader(build("native", device, augment=False), batch_size=2,
                        shuffle=False, num_workers=2)
    raw = next(iter(loader))
    batch = mt.to_device(raw, device)
    inputs = mr.model_inputs_with_canvas(mt.model_inputs, batch, time_input="nominal",
                                         nominal_history_seconds=config.nominal_history_seconds)
    motion_module.local_correlation = traced
    try:
        with torch.no_grad():
            out = model(**inputs)
    finally:
        motion_module.local_correlation = original_corr
    result["forward"] = {
        "correlation_calls": corr_calls,
        "plan_abs": list(out["plan_abs"].shape),
        "motion_features": list(out["motion_features"].shape),
        "scene_features": list(out["scene_features"].shape),
        "state_hat": list(out["state_hat"].shape),
        "history_hat": list(out["history_hat"].shape),
        "scene_history_input_hw": list(inputs["history_images"].shape[-2:]),
        "motion_canvas_hw": list(inputs["motion_history"].shape[-2:]),
    }
    search = []
    for call in corr_calls:
        fh, fw = call["current_in"][-2:]
        search.append({"feature_hw": [fh, fw],
                       "stride_current_px": [432 / fh, 768 / fw],
                       "search_halfwidth_current_px": [call["radius"] * 432 / fh,
                                                       call["radius"] * 768 / fw]})
    result["search_reach"] = search

    # 4. cost: forward and backward at the training microbatch
    model.train()
    mt.set_training_mode(model, "fixed")
    weights = mt.LossWeights(plan=1., occupancy=.2, lane=.2, motion=.2, uncertainty=True)
    timings = []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(4):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**inputs)
        loss, _ = mt.compute_loss(output, batch, weights)
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
    result["cost_microbatch2"] = {
        "forward_backward_s": {"median": float(sorted(timings)[len(timings) // 2]),
                               "all": timings},
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20,
        "note": "microbatch 2, the training setting; effective batch stays 16",
    }
    result["all_checks_pass"] = bool(
        contracts["detail_actually_differs"]
        and result["flip_coverage"]["wrapped_flips_canvas"]
        and result["flip_coverage"]["unwrapped_would_leave_it_unflipped"]
        and result["flip_coverage"]["double_flip_identity"]
        and len(corr_calls) == 2 and all(c["bins"] == 81 for c in corr_calls))
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
