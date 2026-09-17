#!/usr/bin/env python3
"""Package a submission the way the accepted one was packaged, and check it.

Format, taken from the submission that actually scored on 2026-09-01: a single
top-level JSON object whose keys are the 1,125 clip tokens plus the integer
"__flops__", zipped as submission.zip containing exactly one member named
submission.json.

The checks here are the ones whose failure is charged to us: a missing or
malformed clip is scored as [0, 0] and the penalty is the entrant's
responsibility, and a submission without __flops__ cannot be judged against the
7,053 GFLOPs cutoff at all.
"""
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
CUTOFF_GFLOPS = 7053.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--flops-report", required=True)
    parser.add_argument("--clips-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    submission = json.loads(Path(args.submission).read_text())
    flops_report = json.loads(Path(args.flops_report).read_text())
    submission.pop("__flops__", None)
    expected = sorted(p.name for p in Path(args.clips_root).iterdir() if p.is_dir())

    keys = sorted(submission)
    missing = sorted(set(expected) - set(keys))
    extra = sorted(set(keys) - set(expected))
    values = np.asarray([submission[k] for k in keys], dtype=np.float64)

    checks = {
        "clips_expected": len(expected),
        "clips_present": len(keys),
        "missing_clips": missing[:10],
        "n_missing": len(missing),
        "extra_clips": extra[:10],
        "n_extra": len(extra),
        "all_shapes_6x2": bool(values.ndim == 3 and values.shape[1:] == (6, 2)),
        "all_finite": bool(np.isfinite(values).all()),
        "n_nan_or_inf": int((~np.isfinite(values)).sum()),
        "flops_present": True,
        "flops": flops_report["flops"],
        "gflops": flops_report["gflops"],
        "flops_cutoff_gflops": CUTOFF_GFLOPS,
        "flops_passes_cutoff": flops_report["passes_cutoff"],
        "endpoint_m": {"p50": float(np.percentile(np.linalg.norm(values[:, -1], axis=-1), 50)),
                       "max": float(np.linalg.norm(values[:, -1], axis=-1).max())},
        "interval_speed_max_ms": float(
            (np.linalg.norm(np.diff(np.concatenate(
                [np.zeros((len(values), 1, 2)), values], 1), axis=1), axis=-1) / 0.5).max()),
    }
    failures = []
    if checks["n_missing"] or checks["n_extra"]:
        failures.append("clip set does not match the test clips exactly")
    if not checks["all_shapes_6x2"]:
        failures.append("not every entry is 6x2")
    if not checks["all_finite"]:
        failures.append("non-finite values present")
    if not checks["flops_passes_cutoff"]:
        failures.append("__flops__ exceeds the cutoff")
    checks["failures"] = failures

    if failures:
        raise SystemExit("refusing to package: " + "; ".join(failures))

    # __flops__ is added last so the clip entries above were counted without it.
    submission["__flops__"] = int(flops_report["flops"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "submission.json"
    json_path.write_text(json.dumps(submission))
    zip_path = out / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="submission.json")

    manifest = {
        "schema_version": 1,
        "label": args.label,
        "format": {
            "container": "submission.zip containing exactly one member, submission.json",
            "json": "top-level object: 1,125 clip tokens plus the integer __flops__",
            "coordinates": "cumulative absolute XY in metres, current ego frame, "
                           "0.5 s steps to 3.0 s",
            "matches": "the packaging of the submission that scored on 2026-09-01",
        },
        "keys_in_json": len(submission),
        "checks": checks,
        "submission_json_sha256": sha256(json_path),
        "submission_zip_sha256": sha256(zip_path),
        "flops_report": str(Path(args.flops_report).resolve()),
        "checkpoint": flops_report["checkpoint"],
        "model_state_sha256": flops_report["model_state_sha256"],
        "uploaded": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in
                      ("label", "keys_in_json", "submission_zip_sha256")}, indent=1))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
