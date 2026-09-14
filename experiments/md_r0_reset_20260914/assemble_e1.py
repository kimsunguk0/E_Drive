#!/usr/bin/env python3
"""Assemble the E1 curves and result table from the finished evaluations."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/NHNHOME/data/sukim/adcl")
REPORTS = ROOT / "reports/md_r0_reset_20260914"
EVAL = REPORTS / "eval"
WORK = ROOT / "work_dirs/md_r0_reset_20260914"
STEPS = [3426, 6852, 10278, 13704, 17130, 20554]
ARMS = ["E1-T203", "E1-EXP"]


def tune_curve(arm):
    rows = [json.loads(line) for line in
            (WORK / arm / "metrics.jsonl").read_text().splitlines() if line.strip()]
    return {int(r["step"]): r for r in rows if r["kind"] == "eval"}


def main() -> None:
    registry = json.loads((REPORTS / "baseline_registry.json").read_text())
    baseline_probe = json.loads((EVAL / "train_probe.json").read_text())
    baseline_tune = json.loads((EVAL / "tune_legacy.json").read_text())

    curves, table = {}, []
    for arm in ARMS:
        tune = tune_curve(arm)
        probes = {}
        for step in STEPS:
            path = EVAL / f"{arm}_probe_step{step}.json"
            if path.exists():
                payload = json.loads(path.read_text())
                if payload["rows_sha256"] != baseline_probe["rows_sha256"]:
                    raise SystemExit(f"{arm} probe at {step} uses a different row set")
                probes[step] = payload
        curves[arm] = {
            "tune_b4": {str(s): tune[s]["official_d3"] for s in sorted(tune)},
            "train_probe": {str(s): probes[s]["metrics"]["official_d3_weighted"]
                            for s in sorted(probes)},
            "vx_mae": {str(s): tune[s]["state_mae_vx_vy_ax_ay_yawrate"][0] for s in sorted(tune)},
            "occ_iou": {str(s): tune[s]["occ_iou"] for s in sorted(tune)},
            "lane_iou": {str(s): tune[s]["lane_iou"] for s in sorted(tune)},
        }
        best_step = min(sorted(tune), key=lambda s: (tune[s]["official_d3"], s))
        b1 = EVAL / f"{arm}_tuneB1_step20554.json"
        experiment = json.loads((WORK / arm / "experiment.json").read_text())
        table.append({
            "run": arm,
            "train_rows": experiment["train_data"]["rows"],
            "train_scenes": experiment["train_data"]["scenes"],
            "exposures_of_own_train": round(20554 * 16 / experiment["train_data"]["rows"], 3),
            "train_probe_terminal": probes.get(20554, {}).get("metrics", {}).get("official_d3_weighted"),
            "v0_b4_terminal": tune[20554]["official_d3"],
            "v0_b4_best": tune[best_step]["official_d3"],
            "best_step": best_step,
            "best_is_terminal": best_step == 20554,
            "v0_b1_terminal": (json.loads(b1.read_text())["metrics"]["official_d3_weighted"]
                               if b1.exists() else None),
            "vx_mae_terminal": tune[20554]["state_mae_vx_vy_ax_ay_yawrate"][0],
            "occ_iou_terminal": tune[20554]["occ_iou"],
            "lane_iou_terminal": tune[20554]["lane_iou"],
        })

    reference = {
        "run": "R0",
        "train_rows": registry["train_split_id"]["rows"],
        "train_scenes": registry["train_split_id"]["scenes"],
        "exposures_of_own_train": None,
        "train_probe_terminal": baseline_probe["metrics"]["official_d3_weighted"],
        "v0_b4_terminal": baseline_tune["metrics"]["official_d3_weighted"],
        "v0_b4_best": baseline_tune["metrics"]["official_d3_weighted"],
        "best_step": registry["recipe"]["updates"],
        "best_is_terminal": True,
        "v0_b1_terminal": json.loads((EVAL / "tune_B1.json").read_text())["metrics"]["official_d3_weighted"],
        "vx_mae_terminal": registry["historical_auxiliary"]["state_mae_vx_vy_ax_ay_yawrate"][0],
        "occ_iou_terminal": registry["historical_auxiliary"]["occ_iou"],
        "lane_iou_terminal": registry["historical_auxiliary"]["lane_iou"],
    }
    rows = [reference] + table

    with (REPORTS / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reference))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})

    with (REPORTS / "learning_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "step", "v0_b4_official_d3", "train_probe_d3",
                         "vx_mae", "occ_iou", "lane_iou"])
        writer.writerow(["R0", 0, reference["v0_b4_terminal"], reference["train_probe_terminal"],
                         reference["vx_mae_terminal"], reference["occ_iou_terminal"],
                         reference["lane_iou_terminal"]])
        for arm in ARMS:
            for step in STEPS:
                key = str(step)
                writer.writerow([arm, step,
                                 curves[arm]["tune_b4"].get(key, ""),
                                 curves[arm]["train_probe"].get(key, ""),
                                 curves[arm]["vx_mae"].get(key, ""),
                                 curves[arm]["occ_iou"].get(key, ""),
                                 curves[arm]["lane_iou"].get(key, "")])

    (REPORTS / "e1_curves.json").write_text(json.dumps(
        {"reference": reference, "arms": curves,
         "probe_row_set_sha256": baseline_probe["rows_sha256"],
         "probe_rows": baseline_probe["rows"],
         "note": "the train probe uses the SAME 3,456 T0 rows for every run"},
        indent=1, sort_keys=True) + "\n")
    print(json.dumps({"results": rows, "probe_rows_identical": True}, indent=1))


if __name__ == "__main__":
    main()
