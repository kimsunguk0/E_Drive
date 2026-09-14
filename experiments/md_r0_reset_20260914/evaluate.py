#!/usr/bin/env python3
"""P1 evaluator - run the canonical trainer's --eval-only path and report more.

Nothing about the computation is re-implemented here: the model build, config
restore, strict load, dataset and metric all come from
`scripts/train_motiondrive_v2.py`.  This adds three things the trainer's
eval-only path does not emit on its own:

  * detailed per-row predictions (the trainer supports it, the CLI path does not
    request it),
  * ADE1 / ADE2 / ADE3, per-timestep L2 and the unweighted ADE6 alongside the
    project's weighted D3, so §4.4 of the roadmap can be checked,
  * an optional deterministic row subset (`train_probe`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from torch.utils.data import Subset

import train_motiondrive_v2 as trainer
import motiondrive_v2_data as data_api

SPLIT_MANIFEST = ROOT / "data/etri/motiondrive_v2/grouped_split_rawtime.json"
SUPERVISION = ROOT / "data/etri/motiondrive_v2/train_tune_geometry_v2"
D3_WEIGHTS = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0


class _RowSubset(Subset):
    """Subset that keeps the `.rows` attribute the trainer manifests read."""

    def __init__(self, dataset, indices):
        super().__init__(dataset, list(indices))
        self.rows = [dataset.rows[i] for i in self.indices]


def probe_indices(dataset, per_session):
    """Deterministic <=per_session rows per session, evenly spaced by row id."""
    scene_to_session = dataset.manifest["scene_to_session"]
    by_session = {}
    for index, row in enumerate(dataset.rows):
        scene = str(dataset.scene_names[row])
        by_session.setdefault(scene_to_session[scene], []).append((int(row), index))
    chosen = []
    for session in sorted(by_session, key=lambda s: (s is None, s)):
        entries = sorted(by_session[session])
        if len(entries) <= per_session:
            picked = entries
        else:
            positions = np.linspace(0, len(entries) - 1, per_session)
            picked = [entries[int(round(p))] for p in positions]
        chosen.extend(index for _row, index in picked)
    return sorted(set(chosen))


def row_metrics(records):
    pred = np.asarray([r["pred_abs_xy"] for r in records], dtype=np.float64)
    gt = np.asarray([r["gt_abs_xy"] for r in records], dtype=np.float64)
    l2 = np.linalg.norm(pred - gt, axis=-1)                       # [N, 6]
    ade1 = l2[:, :2].mean(axis=1)
    ade2 = l2[:, :4].mean(axis=1)
    ade3 = l2[:, :6].mean(axis=1)
    guide_d3 = (ade1 + ade2 + ade3) / 3.0
    weighted = l2 @ D3_WEIGHTS
    return {
        "rows": int(len(records)),
        "official_d3_weighted": float(weighted.mean()),
        "guide_two_stage_d3": float(guide_d3.mean()),
        "weighted_vs_two_stage_max_row_diff": float(np.abs(weighted - guide_d3).max()),
        "ade1": float(ade1.mean()), "ade2": float(ade2.mean()), "ade3": float(ade3.mean()),
        "ade6_unweighted": float(l2.mean()),
        "point_l2_by_timestep": [float(v) for v in l2.mean(axis=0)],
        "point_l2_p95_by_timestep": [float(v) for v in np.percentile(l2, 95, axis=0)],
        "d3_p50": float(np.percentile(weighted, 50)),
        "d3_p95": float(np.percentile(weighted, 95)),
        "d3_p99": float(np.percentile(weighted, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--init", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--eval-split", choices=["train", "tune", "val"], default="tune")
    parser.add_argument("--eval-stride", type=int, default=5)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--probe-per-session", type=int, default=0,
                        help="keep at most N deterministic rows per session")
    parser.add_argument("--split-manifest", default=str(SPLIT_MANIFEST))
    parser.add_argument("--supervision-root", default=str(SUPERVISION))
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--scenes-file",
                        help="newline-separated scene list; restricts the eval split")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(f"run dir already exists: {run_dir}")

    original_evaluate = trainer.evaluate
    original_dataset = data_api.MotionDriveDataset
    captured = {}

    def detailed_evaluate(model, loader, device, precision, **kwargs):
        kwargs["detailed_records"] = True
        return original_evaluate(model, loader, device, precision, **kwargs)

    def dataset_factory(**kwargs):
        base = original_dataset(**kwargs)
        if kwargs.get("augment"):
            raise ValueError("evaluation datasets must never be augmented")
        if args.probe_per_session > 0:
            subset = _RowSubset(base, probe_indices(base, args.probe_per_session))
            captured["probe_rows"] = len(subset)
            return subset
        return base

    command = [
        "--data-root", "/tmp/pm97",
        "--split-manifest", str(Path(args.split_manifest).resolve()),
        "--supervision-root", str(Path(args.supervision_root).resolve()),
        "--run-dir", str(run_dir),
        "--phase", "joint", "--goal-on", "1", "--state-on", "1",
        "--gpu", str(args.gpu), "--seed", "0", "--steps", "1",
        "--batch", "16", "--microbatch", "2",
        "--eval-batch", str(args.eval_batch),
        "--workers", str(args.workers), "--eval-every", "1", "--save-every", "1",
        "--log-every", "10", "--lr", "0.00005", "--backbone-lr", "0.000005",
        "--weight-decay", "0.01", "--warmup", "100", "--grad-clip", "5",
        "--alpha-occ", "0.2", "--alpha-lane", "0.2", "--alpha-motion", "0.2",
        "--uncertainty", "1", "--precision", args.precision,
        "--time-input", "nominal", "--bn-policy", "fixed",
        "--init", str(Path(args.init).resolve()),
        "--train-stride", "1", "--eval-stride", str(args.eval_stride),
        "--max-train-samples", "0", "--max-eval-samples", "0",
        "--eval-split", args.eval_split, "--eval-only",
        "--cuda-memory-limit-mib", "12000", "--cuda-min-free-mib", "8192",
    ]
    scenes = list(args.scenes or [])
    if args.scenes_file:
        scenes.extend(line.strip() for line in Path(args.scenes_file).read_text().splitlines()
                      if line.strip())
    if scenes:
        command.extend(("--eval-scenes", *sorted(set(scenes))))

    try:
        trainer.evaluate = detailed_evaluate
        data_api.MotionDriveDataset = dataset_factory
        trainer.run_training(command, experiment=None)
    finally:
        trainer.evaluate = original_evaluate
        data_api.MotionDriveDataset = original_dataset

    payload = json.loads((run_dir / "evaluation.json").read_text())
    report, records = payload["report"], payload["records"]
    rows = np.asarray(sorted(int(r["row"]) for r in records), dtype="<i8")
    by_session = {}
    for record in records:
        by_session.setdefault(record["session"], []).append(record["d3"])

    result = {
        "tag": args.tag,
        "checkpoint": str(Path(args.init).resolve()),
        "checkpoint_sha256": trainer.sha256(args.init),
        "eval_split": args.eval_split,
        "eval_stride": args.eval_stride,
        "eval_batch": args.eval_batch,
        "precision": args.precision,
        "time_input": "nominal",
        "augmentation": "off",
        "probe_per_session": args.probe_per_session,
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "split_manifest_sha256": trainer.sha256(args.split_manifest),
        "supervision_root": str(Path(args.supervision_root).resolve()),
        "rows": int(len(records)),
        "rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
        "sessions": len(by_session),
        "scenes": len({r["scenario"] for r in records}),
        "trainer_report": report,
        "metrics": row_metrics(records),
        "session_d3": {k: float(np.mean(v)) for k, v in sorted(by_session.items())},
        "run_dir": str(run_dir),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print("RESULT " + json.dumps({
        "tag": args.tag, "rows": result["rows"],
        "official_d3": result["metrics"]["official_d3_weighted"],
        "trainer_official_d3": report["official_d3"],
        "guide_two_stage_d3": result["metrics"]["guide_two_stage_d3"],
        "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
