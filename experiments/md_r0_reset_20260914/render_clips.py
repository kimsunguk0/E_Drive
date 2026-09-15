#!/usr/bin/env python3
"""Render the 24 preselected review clips: front view, past views, pred vs GT path.

Diagnostic rendering only. The future GT path is drawn because this is an offline
review; it is never an inference input.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path("/NHNHOME/data/sukim/adcl")
OUT = ROOT / "reports/md_exp_diagnosis_20260915/clips"
CACHE = ROOT / "cache/etri_768"
EXP_EVAL = ROOT / "work_dirs/md_r0_reset_20260914/E1-EXP/final_eval.json"
DIAGNOSIS = ROOT / "reports/md_exp_diagnosis_20260915/diagnosis.json"
PAST = (1, 2, 5, 10)


def load_image(scene, frame):
    path = CACHE / scene / "camera_front" / f"{frame:08d}.jpg"
    return Image.open(path) if path.exists() else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = {(r["scenario"], int(r["frame"])): r
               for r in json.loads(EXP_EVAL.read_text())["records"]}
    groups = json.loads(DIAGNOSIS.read_text())["review_clips"]

    index = []
    for group, entries in groups.items():
        if group == "note":
            continue
        for entry in entries:
            key = (entry["scenario"], entry["frame"])
            record = records[key]
            pred = np.asarray(record["pred_abs_xy"], dtype=float)
            gt = np.asarray(record["gt_abs_xy"], dtype=float)
            figure = plt.figure(figsize=(16, 5.2))
            figure.suptitle(
                f"{group} | {entry['scenario']} frame {entry['frame']} | "
                f"regime {entry['regime']} | D3 {entry['d3']:.3f} | "
                f"first-interval chord error {entry['first_chord_error_m']:+.3f} m",
                fontsize=11)

            axis = figure.add_subplot(1, 4, 1)
            current = load_image(entry["scenario"], entry["frame"])
            if current is not None:
                axis.imshow(current)
            axis.set_title("front, t0", fontsize=9)
            axis.axis("off")

            for i, back in enumerate((10, 5)):
                axis = figure.add_subplot(1, 4, 2 + i)
                past = load_image(entry["scenario"], entry["frame"] - back)
                if past is not None:
                    axis.imshow(past)
                axis.set_title(f"front, t-{back / 10:.1f}s", fontsize=9)
                axis.axis("off")

            axis = figure.add_subplot(1, 4, 4)
            zero = np.zeros((1, 2))
            gt_path = np.vstack([zero, gt])
            pred_path = np.vstack([zero, pred])
            axis.plot(gt_path[:, 1], gt_path[:, 0], "o-", label="GT", linewidth=2)
            axis.plot(pred_path[:, 1], pred_path[:, 0], "s--", label="pred", linewidth=2)
            axis.plot(0, 0, "k*", markersize=12, label="ego t0")
            span = max(3.0, np.abs(np.vstack([gt_path, pred_path])).max() * 1.15)
            axis.set_xlim(span, -span)
            axis.set_ylim(-1, span * 1.6)
            axis.set_xlabel("left y (m)")
            axis.set_ylabel("forward x (m)")
            axis.grid(alpha=.3)
            axis.legend(fontsize=8)
            axis.set_title(f"GT {np.linalg.norm(gt[-1]):.1f} m vs pred "
                           f"{np.linalg.norm(pred[-1]):.1f} m", fontsize=9)

            name = f"{group}__{entry['scenario']}_f{entry['frame']}.png"
            figure.tight_layout()
            figure.savefig(OUT / name, dpi=78, bbox_inches="tight")
            plt.close(figure)
            index.append({"group": group, "file": name, **entry,
                          "gt_endpoint_m": float(np.linalg.norm(gt[-1])),
                          "pred_endpoint_m": float(np.linalg.norm(pred[-1])),
                          "gt_total_chord_m": float(np.linalg.norm(
                              np.diff(np.vstack([zero, gt]), axis=0), axis=-1).sum()),
                          "pred_total_chord_m": float(np.linalg.norm(
                              np.diff(np.vstack([zero, pred]), axis=0), axis=-1).sum())})

    (OUT / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"rendered": len(index), "dir": str(OUT)}, indent=1))


if __name__ == "__main__":
    main()
