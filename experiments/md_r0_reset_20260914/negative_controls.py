#!/usr/bin/env python3
"""Negative controls on the leading candidate: does the imagery carry the answer?

Public notice 1, items 3-1 and 3-3, judge a model by where its performance
actually comes from.  This measures that directly on V0: replace the imagery and
see what happens to the trajectory, and perturb the goal to show it is a
reference condition rather than the thing being integrated into coordinates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import motiondrive_v2_training as mt
import train_motiondrive_v2 as trainer
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config

OUT = ROOT / "reports/md_r0_reset_20260914/negative_controls.json"


def conditions(inputs, rng):
    """Each condition returns a new input dict; originals are never mutated."""
    def zero_images(batch):
        out = dict(batch)
        out["images"] = torch.zeros_like(batch["images"])
        out["history_images"] = torch.zeros_like(batch["history_images"])
        return out

    def swapped_images(batch):
        """Swap imagery between the two halves of the batch.

        The loader pairs each row with one about half the split away, so the
        donor is a different scene and session, not the same clip half a second
        later.  Swapping inside a naturally ordered batch is nearly a no-op.
        """
        out = dict(batch)
        n = len(batch["images"])
        if n % 2:
            raise ValueError("cross-scene swap needs an even batch")
        order = torch.arange(n, device=batch["images"].device).view(-1, 2).flip(1).reshape(-1)
        out["images"] = batch["images"][order]
        out["history_images"] = batch["history_images"][order]
        return out

    def goal_zero(batch):
        out = dict(batch)
        out["goal_xy"] = torch.zeros_like(batch["goal_xy"])
        return out

    def goal_perturbed(batch):
        out = dict(batch)
        noise = torch.from_numpy(rng.normal(0., 5., size=tuple(batch["goal_xy"].shape))
                                 ).to(batch["goal_xy"])
        out["goal_xy"] = batch["goal_xy"] + noise
        return out

    return {"normal": lambda b: dict(b), "image_zero": zero_images,
            "image_swap_cross_scene": swapped_images,
            "goal_zero": goal_zero, "goal_perturbed_5m_sigma": goal_perturbed}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--init", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--eval-batch", type=int, default=2,
                        help="fixed at 2: rows are paired across the split for the swap")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    payload = torch.load(args.init, map_location="cpu", weights_only=False)
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    model = MotionDriveV2(config)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(f"cuda:{args.gpu}")
    model.to(device).eval()

    from motiondrive_v2_data import MotionDriveDataset
    from torch.utils.data import DataLoader
    dataset = MotionDriveDataset(
        data_root="/tmp/pm97", split_manifest=args.split_manifest, split="tune",
        supervision_root=args.supervision_root, min_frame=30, frame_stride=5,
        max_samples=args.max_rows, augment=False, seed=0, history_contract="control")
    count = len(dataset)
    half = count // 2
    paired = [index for i in range(half) for index in (i, i + half)]
    if count % 2:
        paired.append(count - 1)          # the odd row is paired with itself
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4,
                        sampler=paired)

    rng = np.random.default_rng(0)
    builders = conditions(None, rng)
    d3 = {name: [] for name in builders}
    plans = {name: [] for name in builders}
    for raw in loader:
        batch = mt.to_device(raw, device)
        base_inputs = mt.model_inputs(batch, time_input="nominal",
                                      nominal_history_seconds=config.nominal_history_seconds)
        gt = batch["gt_plan"].float()
        for name, build in builders.items():
            with torch.no_grad(), trainer.autocast(device, "bf16"):
                out = model(**build(base_inputs))["plan_abs"].float()
            d3[name].extend(mt.weighted_d3(out, gt).cpu().tolist())
            plans[name].append(out.cpu())

    normal = torch.cat(plans["normal"])
    summary = {}
    for name in builders:
        values = np.asarray(d3[name])
        stacked = torch.cat(plans[name])
        difference = (stacked - normal).abs()
        summary[name] = {
            "official_d3": float(values.mean()),
            "d3_ratio_vs_normal": float(values.mean() / np.mean(d3["normal"])),
            "max_abs_xy_diff_vs_normal_m": float(difference.max()),
            "mean_abs_xy_diff_vs_normal_m": float(difference.mean()),
            "rows": int(len(values)),
        }

    result = {
        "schema_version": 1, "label": args.label,
        "checkpoint": str(Path(args.init).resolve()),
        "checkpoint_sha256": trainer.sha256(args.init),
        "split": "tune (V0)", "rows": summary["normal"]["rows"],
        "eval_batch": 2, "row_pairing": "row i paired with row i + n//2 so the swap crosses scenes",
        "precision": "bf16",
        "conditions": summary,
        "reading": {
            "images_carry_the_trajectory": summary["image_zero"]["d3_ratio_vs_normal"],
            "goal_is_a_reference_not_the_source": summary["goal_zero"]["d3_ratio_vs_normal"],
            "note": ("A large image-ablation ratio is evidence the trajectory comes from the "
                     "imagery. The goal rows show how much a goal change moves the output; "
                     "goal enters the shared scene features only, never the planner."),
        },
        "limitations": [
            "Ablations are out-of-distribution inputs; the size of a degradation is not a linear measure of contribution.",
            "This is evidence about where performance comes from, not an approval of the submission by the operators.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    existing[args.label] = result
    OUT.write_text(json.dumps(existing, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"label": args.label, "rows": result["rows"],
                      "conditions": {k: {"d3": round(v["official_d3"], 6),
                                         "ratio": round(v["d3_ratio_vs_normal"], 3)}
                                     for k, v in summary.items()}}, indent=1))


if __name__ == "__main__":
    main()
