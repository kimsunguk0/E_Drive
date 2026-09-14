#!/usr/bin/env python3
"""The single H confirmation run, executed only after the design is locked.

Refuses to run unless --i-have-locked-the-design is passed and no prior H
result exists for the same candidate, because H is meant to be opened once.
Reads the pre-registration and records it alongside the numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
REPORTS = ROOT / "reports/md_r0_reset_20260914"
EVAL = REPORTS / "eval"
NEW_SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json"
SUPERVISION = ROOT / "data/etri/motiondrive_v2/r0reset_tplus_geometry_v2"
PREREGISTRATION = REPORTS / "H_PREREGISTRATION_KO.md"
OUT = REPORTS / "h_confirmation.json"
EVALUATOR = ROOT / "experiments/md_r0_reset_20260914/evaluate.py"
PYTHON = ROOT / "env/venv/bin/python"
RESAMPLES = 20000


def run_eval(name, checkpoint, batch, gpu):
    tag = f"H_{name}_b{batch}"
    out = EVAL / f"{tag}.json"
    if out.exists():
        return json.loads(out.read_text())
    command = [str(PYTHON), str(EVALUATOR), "--init", str(checkpoint), "--tag", tag,
               "--gpu", "0", "--eval-split", "val", "--eval-stride", "1",
               "--eval-batch", str(batch), "--split-manifest", str(NEW_SPLIT),
               "--supervision-root", str(SUPERVISION),
               "--run-dir", str(ROOT / f"work_dirs/md_r0_reset_20260914/evals/{tag}"),
               "--out", str(out)]
    environment = {"CUDA_VISIBLE_DEVICES": str(gpu), "PATH": "/usr/bin:/bin",
                   "HOME": str(Path.home())}
    completed = subprocess.run(command, capture_output=True, text=True, env=environment)
    if completed.returncode != 0:
        raise SystemExit(f"H evaluation failed for {name}:\n{completed.stderr[-4000:]}")
    return json.loads(out.read_text())


def paired(a, b, sessions, rng):
    """b minus a, resampled by H session."""
    unique = sorted(set(sessions.tolist()))
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    diff = b - a
    draws = rng.integers(0, len(unique), size=(RESAMPLES, len(unique)))
    means = np.asarray([diff[np.concatenate([index[unique[j]] for j in row])].mean()
                        for row in draws])
    return {"delta": float(diff.mean()),
            "bootstrap_sd": float(means.std(ddof=1)),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "p_worse": float((means > 0).mean()),
            "sessions_improved": int(sum(1 for s in unique if diff[index[s]].mean() < 0)),
            "sessions": len(unique),
            "per_session_delta": {s: float(diff[index[s]].mean()) for s in unique}}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=CKPT",
                        help="pre-selected candidate; at most two besides R0")
    parser.add_argument("--r0", required=True, help="the R0 checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--i-have-locked-the-design", action="store_true", required=False)
    args = parser.parse_args()

    if not args.i_have_locked_the_design:
        raise SystemExit("H is opened once, after the design is locked. Refusing.")
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; H has been opened. Refusing to re-open.")
    if len(args.candidate) > 2:
        raise SystemExit("the pre-registration allows at most two candidates besides R0")

    models = {"R0": Path(args.r0)}
    for item in args.candidate:
        name, _, path = item.partition("=")
        models[name] = Path(path)

    results, records = {}, {}
    for name, checkpoint in models.items():
        results[name] = {}
        for batch in (1, 4):
            payload = run_eval(name, checkpoint, batch, args.gpu)
            results[name][f"b{batch}"] = {
                "official_d3": payload["metrics"]["official_d3_weighted"],
                "ade6_unweighted": payload["metrics"]["ade6_unweighted"],
                "ade1": payload["metrics"]["ade1"], "ade2": payload["metrics"]["ade2"],
                "ade3": payload["metrics"]["ade3"],
                "point_l2_by_timestep": payload["metrics"]["point_l2_by_timestep"],
                "rows": payload["rows"], "sessions": payload["sessions"],
                "session_d3": payload["session_d3"],
                "vx_mae": payload["trainer_report"]["state_mae_vx_vy_ax_ay_yawrate"][0],
                "rows_sha256": payload["rows_sha256"],
            }
            if batch == 1:
                run_dir = Path(payload["run_dir"])
                rows = json.loads((run_dir / "evaluation.json").read_text())["records"]
                order = sorted(range(len(rows)), key=lambda i: (
                    rows[i]["session"], rows[i]["scenario"], int(rows[i]["frame"])))
                records[name] = (
                    np.asarray([rows[i]["d3"] for i in order], dtype=np.float64),
                    np.asarray([rows[i]["session"] for i in order]))

    reference_sessions = None
    for name, (_, sessions) in records.items():
        if reference_sessions is None:
            reference_sessions = sessions
        elif not np.array_equal(sessions, reference_sessions):
            raise SystemExit(f"{name} is not scored on the same H rows")

    rng = np.random.default_rng(0)
    comparisons = {}
    names = list(models)
    for i, base in enumerate(names):
        for arm in names[i + 1:]:
            comparisons[f"{base} -> {arm}"] = paired(
                records[base][0], records[arm][0], reference_sessions, rng)

    payload = {
        "schema_version": 1,
        "status": "H_OPENED_ONCE",
        "pre_registration": str(PREREGISTRATION),
        "pre_registration_sha256": __import__("hashlib").sha256(
            PREREGISTRATION.read_bytes()).hexdigest(),
        "set": {"name": "H", "description": "ancestor-fit-excluded confirmation set",
                "not_a_blind_set": True,
                "split_manifest": str(NEW_SPLIT), "supervision": str(SUPERVISION)},
        "models": {name: str(path.resolve()) for name, path in models.items()},
        "results": results,
        "comparisons": comparisons,
        "primary_batch": 1,
        "caveats": [
            "H was opened once; no candidate was selected using it.",
            "Absolute H scores are not comparable to V0 scores: different sessions.",
            "This is not a server score and not a ruling on regulatory compliance.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"models": list(models),
                      "h_b1_official_d3": {k: v["b1"]["official_d3"] for k, v in results.items()},
                      "comparisons": {k: {"delta": round(v["delta"], 6),
                                          "ci95": [round(x, 6) for x in v["ci95"]],
                                          "sessions_improved": v["sessions_improved"]}
                                      for k, v in comparisons.items()}}, indent=1))


if __name__ == "__main__":
    main()
