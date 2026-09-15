#!/usr/bin/env python3
"""Capture the per-row history head outputs the stored eval records do not carry.

The trainer's detailed records keep plan and state but not history, and analysis
C needs history error on the same rows.  This runs one evaluation pass under the
same conditions and saves history only; the plan and state numbers continue to
come from the stored final_eval.json, which this does not touch.
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


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--init", required=True)
    parser.add_argument("--split", default="tune")
    parser.add_argument("--eval-stride", type=int, default=5)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--scenes-file")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = torch.load(args.init, map_location="cpu", weights_only=False)
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    model = MotionDriveV2(config)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(f"cuda:{args.gpu}")
    model.to(device).eval()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

    from motiondrive_v2_data import MotionDriveDataset
    from torch.utils.data import DataLoader
    scenes = None
    if args.scenes_file:
        scenes = sorted({line.strip() for line in
                         Path(args.scenes_file).read_text().splitlines() if line.strip()})
    dataset = MotionDriveDataset(
        data_root="/tmp/pm97", split_manifest=args.split_manifest, split=args.split,
        supervision_root=args.supervision_root, min_frame=30,
        frame_stride=args.eval_stride, augment=False, seed=0,
        history_contract="control", **({"scenes": scenes} if scenes else {}))
    loader = DataLoader(dataset, batch_size=args.eval_batch, shuffle=False, num_workers=4)

    rows, hat, target, valid, state_hat, plan = [], [], [], [], [], []
    for raw in loader:
        batch = mt.to_device(raw, device)
        with torch.no_grad(), trainer.autocast(device, "bf16"):
            out = model(**mt.model_inputs(batch, time_input="nominal",
                                          nominal_history_seconds=config.nominal_history_seconds))
        rows.extend(int(v) for v in raw["row"])
        hat.append(out["history_hat"].float().cpu().numpy())
        target.append(batch["history_target"].float().cpu().numpy())
        valid.append(batch["history_valid"].bool().cpu().numpy())
        state_hat.append(out["state_hat"].float().cpu().numpy())
        plan.append(out["plan_abs"].float().cpu().numpy())

    np.savez_compressed(
        args.out,
        row=np.asarray(rows, dtype=np.int64),
        history_hat=np.concatenate(hat), history_target=np.concatenate(target),
        history_valid=np.concatenate(valid), state_hat=np.concatenate(state_hat),
        plan_abs=np.concatenate(plan),
        history_frame_offsets=np.asarray(config.history_frame_offsets, dtype=np.int64),
        nominal_history_seconds=np.asarray(config.nominal_history_seconds, dtype=np.float64),
        history_scale=np.asarray(mt.HISTORY_SCALE, dtype=np.float64))
    print(json.dumps({"out": args.out, "rows": len(rows),
                      "history_shape": list(np.concatenate(hat).shape),
                      "offsets": list(config.history_frame_offsets),
                      "nominal_seconds": list(config.nominal_history_seconds)}, indent=1))


if __name__ == "__main__":
    main()
