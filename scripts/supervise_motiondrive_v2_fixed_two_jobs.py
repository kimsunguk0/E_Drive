#!/usr/bin/env python3
"""Durably supervise exactly two frozen MotionDrive commands, without retries.

The caller is responsible for starting this script detached (for example with
``nohup``).  This supervisor never selects a recipe, retries a failed process,
or starts a follow-on stage.  It records a launch receipt immediately and
updates that same receipt after both fixed subprocesses terminate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ALLOWED_JOB_NAMES = {"zero", "provided"}
ALLOWED_GPU_UUIDS = {
    "GPU-5d2254f9-41a7-62dd-2b38-de82459acb24",
    "GPU-041334c0-089c-6ff5-b0b5-59ff445fa015",
}
FIXED_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_receipt(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def gpu_snapshot() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 7, "Unexpected nvidia-smi output")
        if fields[1] in ALLOWED_GPU_UUIDS:
            rows.append(dict(zip(
                ("index", "uuid", "name", "memory_total_mib", "memory_used_mib",
                 "memory_free_mib", "utilization_percent"), fields)))
    require({row["uuid"] for row in rows} == ALLOWED_GPU_UUIDS,
            "Approved physical GPUs are missing")
    return rows


def validate_spec(spec: dict, cwd: Path, receipt: Path) -> tuple[list[dict], dict]:
    require(set(spec) == {"schema_version", "experiment", "source_pins", "jobs"},
            "Spec keyset mismatch")
    require(spec["schema_version"] == 1, "Unsupported spec schema")
    require(isinstance(spec["experiment"], str) and spec["experiment"],
            "Experiment name missing")
    require(isinstance(spec["source_pins"], dict) and spec["source_pins"],
            "Source pins missing")
    verified = {}
    for raw_path, expected in sorted(spec["source_pins"].items()):
        path = Path(raw_path)
        if not path.is_absolute():
            path = cwd / path
        require(isinstance(expected, str) and len(expected) == 64,
                f"Invalid SHA pin: {raw_path}")
        actual = sha256(path)
        require(actual == expected, f"Source pin mismatch: {raw_path}")
        verified[str(path.resolve())] = actual

    jobs = spec["jobs"]
    require(isinstance(jobs, list) and len(jobs) == 2, "Exactly two jobs required")
    require({job.get("name") for job in jobs} == ALLOWED_JOB_NAMES,
            "Jobs must be exactly zero and provided")
    require({job.get("gpu_uuid") for job in jobs} == ALLOWED_GPU_UUIDS,
            "Jobs must use each approved physical GPU exactly once")
    seen_paths = {receipt.resolve()}
    for job in jobs:
        require(set(job) == {"name", "gpu_uuid", "argv", "log", "run_dir", "result_paths"},
                f"Job keyset mismatch: {job.get('name')}")
        argv = job["argv"]
        require(isinstance(argv, list) and len(argv) >= 2
                and argv[0] == "/usr/bin/python"
                and all(isinstance(value, str) and value for value in argv),
                f"Invalid argv: {job['name']}")
        require("--preflight-only" not in argv, "Training argv contains preflight-only")
        run_dir = Path(job["run_dir"])
        log = Path(job["log"])
        if not run_dir.is_absolute():
            run_dir = cwd / run_dir
        if not log.is_absolute():
            log = cwd / log
        require(not run_dir.exists(), f"Fresh run directory already exists: {run_dir}")
        require(not log.exists(), f"Fresh log already exists: {log}")
        require(run_dir.resolve() not in seen_paths and log.resolve() not in seen_paths,
                "Receipt/log/run paths must be distinct")
        seen_paths.update((run_dir.resolve(), log.resolve()))
        require(isinstance(job["result_paths"], list) and job["result_paths"]
                and all(isinstance(value, str) and value for value in job["result_paths"]),
                f"Expected result paths missing: {job['name']}")
        result_paths = []
        for raw_result in job["result_paths"]:
            result = Path(raw_result)
            if not result.is_absolute():
                result = cwd / result
            require(result.resolve() not in seen_paths, "Result paths must be distinct")
            require(not result.exists() and not result.is_symlink(),
                    f"Expected result path must be fresh: {result}")
            seen_paths.add(result.resolve())
            result_paths.append(str(result.resolve()))
        job["run_dir"] = str(run_dir.resolve())
        job["log"] = str(log.resolve())
        job["result_paths"] = result_paths
    return jobs, verified


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    cwd = Path.cwd().resolve()
    spec_path, receipt = Path(args.spec).resolve(), Path(args.receipt).resolve()
    require(spec_path.is_file(), "Spec file missing")
    require(not receipt.exists(), "Receipt must be fresh")
    with spec_path.open(encoding="utf-8") as stream:
        spec = json.load(stream)
    jobs, verified = validate_spec(spec, cwd, receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    snapshots = gpu_snapshot()
    by_uuid = {row["uuid"]: row for row in snapshots}
    for job in jobs:
        require(int(by_uuid[job["gpu_uuid"]]["memory_used_mib"]) == 0,
                f"Approved GPU not empty at launch: {job['gpu_uuid']}")

    payload = {
        "schema_version": 1,
        "experiment": spec["experiment"],
        "status": "launching",
        "supervisor_pid": os.getpid(),
        "supervisor_pgid": os.getpgrp(),
        "cwd": str(cwd),
        "spec": str(spec_path),
        "spec_sha256": sha256(spec_path),
        "supervisor_source": str(Path(__file__).resolve()),
        "supervisor_source_sha256": sha256(Path(__file__).resolve()),
        "source_pins_verified": verified,
        "gpu_snapshot_before": snapshots,
        "started_utc": timestamp(),
        "automatic_retry": False,
        "follow_on_stage": None,
        "jobs": {},
    }
    write_receipt(receipt, payload)
    processes = {}
    streams = {}
    try:
        for job in jobs:
            try:
                log = Path(job["log"])
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open("xb")
                streams[job["name"]] = stream
                env = os.environ.copy()
                env.update(FIXED_ENV)
                env["CUDA_VISIBLE_DEVICES"] = job["gpu_uuid"]
                process = subprocess.Popen(
                    job["argv"], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                processes[job["name"]] = process
                # start_new_session=True makes the child the leader of its own
                # new session/process group, so PGID is exactly its PID.  Avoid
                # racing os.getpgid against a fast child exit.
                payload["jobs"][job["name"]] = {
                    "gpu_uuid": job["gpu_uuid"], "argv": job["argv"],
                    "environment": {**FIXED_ENV, "CUDA_VISIBLE_DEVICES": job["gpu_uuid"]},
                    "run_dir": job["run_dir"], "log": str(log),
                    "result_paths": job["result_paths"], "pid": process.pid,
                    "pgid": process.pid, "started_utc": timestamp(),
                    "status": "running", "return_code": None,
                }
                payload["status"] = "running"
                write_receipt(receipt, payload)
            except Exception as error:
                payload["status"] = "spawn_failed"
                payload["spawn_failure"] = {
                    "job": job["name"], "error_type": type(error).__name__,
                    "error": str(error), "utc": timestamp(),
                }
                payload["jobs"].setdefault(job["name"], {
                    "gpu_uuid": job["gpu_uuid"], "argv": job["argv"],
                    "run_dir": job["run_dir"], "log": job["log"],
                    "result_paths": job["result_paths"], "pid": None, "pgid": None,
                    "status": "spawn_failed", "return_code": None,
                })
                # Roll back only subprocess groups created by this supervisor.
                # No shared/foreign PID can be targeted because every PGID is
                # the PID returned by our own start_new_session Popen call.
                for name, launched in processes.items():
                    was_running = launched.poll() is None
                    if was_running:
                        try:
                            os.killpg(launched.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            launched.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(launched.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            launched.wait()
                    else:
                        # Reap an already-exited child as well.
                        launched.wait()
                    result_files = {path: {
                        "exists": Path(path).is_file() and not Path(path).is_symlink(),
                        "sha256": sha256(Path(path))
                        if Path(path).is_file() and not Path(path).is_symlink() else None,
                    } for path in payload["jobs"][name]["result_paths"]}
                    payload["jobs"][name].update(
                        status=("cleanup_after_peer_spawn_failure" if was_running
                                else "exited_before_peer_spawn_failure"),
                        termination_requested=was_running,
                        return_code=launched.returncode, ended_utc=timestamp(),
                        results_complete=all(value["exists"] for value in result_files.values()),
                        result_files=result_files)
                failed = payload["jobs"][job["name"]]
                failed_results = {path: {
                    "exists": Path(path).is_file() and not Path(path).is_symlink(),
                    "sha256": sha256(Path(path))
                    if Path(path).is_file() and not Path(path).is_symlink() else None,
                } for path in failed["result_paths"]}
                failed.update(results_complete=all(
                    value["exists"] for value in failed_results.values()),
                    result_files=failed_results)
                payload["ended_utc"] = timestamp()
                payload["gpu_snapshot_after"] = gpu_snapshot()
                write_receipt(receipt, payload)
                return 1

        pending = set(processes)
        while pending:
            for name in tuple(pending):
                return_code = processes[name].poll()
                if return_code is not None:
                    pending.remove(name)
                    streams[name].flush()
                    streams[name].close()
                    result_files = {path: {
                        "exists": Path(path).is_file() and not Path(path).is_symlink(),
                        "sha256": sha256(Path(path))
                        if Path(path).is_file() and not Path(path).is_symlink() else None,
                    } for path in payload["jobs"][name]["result_paths"]}
                    results_complete = all(value["exists"] for value in result_files.values())
                    payload["jobs"][name].update(
                        status="completed" if return_code == 0 and results_complete else "failed",
                        process_exit_status="zero" if return_code == 0 else "nonzero",
                        return_code=return_code, ended_utc=timestamp(),
                        results_complete=results_complete, result_files=result_files)
                    write_receipt(receipt, payload)
            if pending:
                time.sleep(5)
    finally:
        for name, stream in streams.items():
            if not stream.closed:
                stream.close()
    payload["status"] = "completed" if all(
        job["status"] == "completed" for job in payload["jobs"].values()) else "failed"
    payload["ended_utc"] = timestamp()
    payload["gpu_snapshot_after"] = gpu_snapshot()
    write_receipt(receipt, payload)
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"supervisor error: {error}", file=sys.stderr)
        raise
