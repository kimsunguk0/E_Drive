"""Re-evaluate the existing direct model on the unchanged tune population."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from models.motiondrive_v2 import MotionDriveV2
from models.motiondrive_v2.shared_status_query import install_shared_status_query
from motiondrive_v2_data import MotionDriveDataset
from motiondrive_v2_training import model_inputs, weighted_d3
from data import file_sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Baseline evaluation is assigned physical GPU 0 only")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        base = Path(args.base)
        cp = Path(args.checkpoint)
        payload = torch.load(cp, map_location="cpu", weights_only=False)  # user-owned checkpoint
        manifest = payload["manifest"]
        config = manifest["model_config"]
        model = install_shared_status_query(MotionDriveV2(config))
        model.load_state_dict(payload["model"], strict=True)
        del payload
        model.cuda().eval()
        split = base / "data/etri/motiondrive_v2/grouped_split_rawtime.json"
        ds = MotionDriveDataset(str(base), str(split), split="tune", frame_stride=5,
            augment=False, supervision_root=str(base / "data/etri/motiondrive_v2/train_tune_geometry_v2"))
        status_path = base / "data/etri/motiondrive_v2_shared_status_a1_20260908_ops/tune.npz"
        with np.load(status_path, allow_pickle=False) as z:
            if not np.array_equal(z["row"], ds.rows):
                raise ValueError("Baseline status rows differ from tune rows")
            status = torch.from_numpy(z["status5"].copy())
        receipt = {"checkpoint": str(cp), "checkpoint_sha256": file_sha(cp),
                   "split_sha256": file_sha(split), "status_sha256": file_sha(status_path),
                   "rows_sha256": hashlib.sha256(ds.rows.astype("<i8").tobytes()).hexdigest(),
                   "n": len(ds), "source_root": str(ROOT), "physical_gpu": 0,
                   "pid": os.getpid(), "torch": str(torch.__version__), "model_config": config,
                   "time_input": "nominal", "selection": "existing terminal long_s1, fixed before this evaluation"}
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                            pin_memory=True)
        predictions, errors, records = [], [], []
        cursor = 0
        with torch.inference_mode():
            for batch in loader:
                n = len(batch["images"])
                x = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                inputs = model_inputs(x, "nominal")
                inputs["provided_status5"] = status[cursor:cursor+n].cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(**inputs)["plan_abs"]
                if not torch.isfinite(pred).all():
                    raise RuntimeError("Nonfinite direct predictions")
                error = weighted_d3(pred, x["gt_plan"]).cpu().numpy()
                predictions.append(pred.float().cpu().numpy())
                errors.append(error)
                for i in range(n):
                    records.append({"row": int(batch["row"][i]), "scenario": batch["scenario"][i],
                                    "session": batch["session_id"][i], "d3": float(error[i])})
                cursor += n
                if cursor % 200 < n:
                    print(json.dumps({"evaluated": cursor, "d3": float(np.concatenate(errors).mean()),
                                      "seconds": time.time()-started}), flush=True)
        pred = np.concatenate(predictions)
        error = np.concatenate(errors)
        np.savez_compressed(out / "predictions.npz", rows=ds.rows, pred=pred, d3=error)
        sessions = {}
        for row in records:
            sessions.setdefault(row["session"], []).append(row["d3"])
        report = {"status": "completed", "n": len(error), "official_d3": float(error.mean()),
                  "session_d3": {k:float(np.mean(v)) for k,v in sessions.items()},
                  "elapsed_seconds": time.time()-started, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                  "receipt": receipt}
        (out / "records.json").write_text(json.dumps(records) + "\n")
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    except BaseException:
        (out / "failure.json").write_text(json.dumps({"status": "failed", "traceback": traceback.format_exc(),
                                                      "elapsed_seconds": time.time()-started}, indent=2))
        raise


if __name__ == "__main__":
    main()
