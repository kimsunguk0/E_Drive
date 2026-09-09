#!/usr/bin/env python3
"""Durably supervise exactly three frozen early-precision jobs without retries."""
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


DRIVER = "scripts/run_motiondrive_v2_early_precision.py"
GPU0 = "GPU-5d2254f9-41a7-62dd-2b38-de82459acb24"
GPU1 = "GPU-041334c0-089c-6ff5-b0b5-59ff445fa015"
JOB_GPU = {
    "continuation_control": GPU0,
    "ordered_motion_residual": GPU1,
    "early_delta_aux": GPU0,
}
FIXED_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
CONTROLLED_OPTIONS = {
    "--arm", "--gpu", "--run-dir", "--expected-physical-gpu-uuid",
    "--cuda-memory-limit-mib", "--cuda-min-free-mib",
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
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,"
        "memory.free,utilization.gpu", "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 7, "Unexpected nvidia-smi output")
        if fields[1] in {GPU0, GPU1}:
            rows.append(dict(zip(
                ("index", "uuid", "name", "memory_total_mib", "memory_used_mib",
                 "memory_free_mib", "utilization_percent"), fields)))
    require({row["uuid"] for row in rows} == {GPU0, GPU1},
            "Approved physical GPUs are missing")
    return rows


def exact_options(argv: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for index, token in enumerate(argv):
        if not token.startswith("--"):
            continue
        option = token.partition("=")[0]
        for controlled in CONTROLLED_OPTIONS:
            if controlled.startswith(option) and option != controlled:
                raise RuntimeError(f"Abbreviated controlled option: {option}")
        if option not in CONTROLLED_OPTIONS:
            continue
        require(option not in values, f"Duplicate controlled option: {option}")
        if "=" in token:
            value = token.partition("=")[2]
        else:
            require(index + 1 < len(argv) and not argv[index + 1].startswith("--"),
                    f"Missing controlled option value: {option}")
            value = argv[index + 1]
        require(value, f"Empty controlled option value: {option}")
        values[option] = value
    require(set(values) == CONTROLLED_OPTIONS, "Controlled option set mismatch")
    return values


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
    require(str((cwd / DRIVER).resolve()) in verified,
            "The executed early-precision driver must be source-pinned")

    jobs = spec["jobs"]
    require(isinstance(jobs, list) and len(jobs) == 3, "Exactly three jobs required")
    require({job.get("name") for job in jobs} == set(JOB_GPU),
            "Early-precision job-name set mismatch")
    seen_paths = {receipt.resolve()}
    prepared = []
    for raw_job in jobs:
        require(set(raw_job) == {"name", "gpu_uuid", "argv", "log", "run_dir",
                                 "result_paths"},
                f"Job keyset mismatch: {raw_job.get('name')}")
        job = dict(raw_job)
        name, gpu_uuid, argv = job["name"], job["gpu_uuid"], job["argv"]
        require(gpu_uuid == JOB_GPU[name], f"Fixed GPU mapping mismatch: {name}")
        require(isinstance(argv, list) and len(argv) >= 2
                and argv[:2] == ["/usr/bin/python", DRIVER]
                and all(isinstance(value, str) and value and "\0" not in value
                        for value in argv), f"Invalid argv: {name}")
        require("--preflight-only" not in argv, "Training argv contains preflight-only")
        controlled = exact_options(argv)
        require(controlled["--arm"] == name, f"Arm mismatch: {name}")
        require(controlled["--gpu"] == "0", f"Logical GPU must be zero: {name}")
        require(controlled["--expected-physical-gpu-uuid"] == gpu_uuid,
                f"Physical GPU UUID argv mismatch: {name}")
        require(controlled["--cuda-memory-limit-mib"] == "12000",
                f"Allocator cap mismatch: {name}")
        require(controlled["--cuda-min-free-mib"] == "8192",
                f"Free-memory guard mismatch: {name}")

        run_dir, log = Path(job["run_dir"]), Path(job["log"])
        if not run_dir.is_absolute():
            run_dir = cwd / run_dir
        if not log.is_absolute():
            log = cwd / log
        argv_run = Path(controlled["--run-dir"])
        if not argv_run.is_absolute():
            argv_run = cwd / argv_run
        require(argv_run.resolve() == run_dir.resolve(), f"Run-dir argv mismatch: {name}")
        require(not run_dir.exists() and not run_dir.is_symlink(),
                f"Fresh run directory already exists: {run_dir}")
        require(not log.exists() and not log.is_symlink(), f"Fresh log exists: {log}")
        require(run_dir.resolve() not in seen_paths and log.resolve() not in seen_paths,
                "Receipt/log/run paths must be distinct")
        seen_paths.update((run_dir.resolve(), log.resolve()))
        require(isinstance(job["result_paths"], list) and job["result_paths"],
                f"Expected result paths missing: {name}")
        result_paths = []
        for raw_result in job["result_paths"]:
            require(isinstance(raw_result, str) and raw_result, "Invalid result path")
            result = Path(raw_result)
            if not result.is_absolute():
                result = cwd / result
            require(result.resolve() not in seen_paths, "Result paths must be distinct")
            require(not result.exists() and not result.is_symlink(),
                    f"Expected result path must be fresh: {result}")
            seen_paths.add(result.resolve())
            result_paths.append(str(result.resolve()))
        job.update(run_dir=str(run_dir.resolve()), log=str(log.resolve()),
                   result_paths=result_paths)
        prepared.append(job)
    return prepared, verified


def result_files(job: dict) -> tuple[dict, bool]:
    files = {}
    for raw_path in job["result_paths"]:
        path = Path(raw_path)
        exists = path.is_file() and not path.is_symlink()
        files[raw_path] = {"exists": exists, "sha256": sha256(path) if exists else None}
    return files, all(item["exists"] for item in files.values())


def terminate_owned(processes: dict, payload: dict) -> None:
    for name, process in processes.items():
        was_running = process.poll() is None
        if was_running:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        else:
            process.wait()
        files, complete = result_files(payload["jobs"][name])
        payload["jobs"][name].update(
            status="cleanup_after_peer_spawn_failure" if was_running
            else "exited_before_peer_spawn_failure",
            termination_requested=was_running, return_code=process.returncode,
            ended_utc=timestamp(), results_complete=complete, result_files=files)


def run(spec_path: Path, receipt: Path) -> int:
    cwd = Path.cwd().resolve()
    spec_path, receipt = spec_path.resolve(), receipt.resolve()
    require(spec_path.is_file(), "Spec file missing")
    require(not receipt.exists() and not receipt.is_symlink(), "Receipt must be fresh")
    with spec_path.open(encoding="utf-8") as stream:
        spec = json.load(stream)
    jobs, verified = validate_spec(spec, cwd, receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    snapshots = gpu_snapshot()
    by_uuid = {row["uuid"]: row for row in snapshots}
    for gpu_uuid in {GPU0, GPU1}:
        require(int(by_uuid[gpu_uuid]["memory_used_mib"]) == 0,
                f"Approved GPU not empty at launch: {gpu_uuid}")
        job_count = sum(job["gpu_uuid"] == gpu_uuid for job in jobs)
        required_free = job_count * 12000 + 8192
        require(int(by_uuid[gpu_uuid]["memory_free_mib"]) >= required_free,
                f"Insufficient aggregate guarded free memory: {gpu_uuid}")

    payload = {
        "schema_version": 1, "experiment": spec["experiment"], "status": "launching",
        "supervisor_pid": os.getpid(), "supervisor_pgid": os.getpgrp(),
        "cwd": str(cwd), "spec": str(spec_path), "spec_sha256": sha256(spec_path),
        "supervisor_source": str(Path(__file__).resolve()),
        "supervisor_source_sha256": sha256(Path(__file__).resolve()),
        "source_pins_verified": verified, "gpu_snapshot_before": snapshots,
        "aggregate_memory_gate": {
            GPU0: {"jobs": 2, "allocator_cap_mib_each": 12000,
                   "reserve_mib": 8192, "required_free_mib": 32192},
            GPU1: {"jobs": 1, "allocator_cap_mib_each": 12000,
                   "reserve_mib": 8192, "required_free_mib": 20192}},
        "started_utc": timestamp(), "automatic_retry": False,
        "follow_on_stage": None, "jobs": {},
    }
    write_receipt(receipt, payload)
    processes, streams = {}, {}
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
                payload["jobs"][job["name"]] = {
                    **job, "environment": {**FIXED_ENV, "CUDA_VISIBLE_DEVICES": job["gpu_uuid"]},
                    "pid": process.pid, "pgid": process.pid, "started_utc": timestamp(),
                    "status": "running", "return_code": None,
                }
                payload["status"] = "running"
                write_receipt(receipt, payload)
            except Exception as error:
                payload.update(status="spawn_failed", spawn_failure={
                    "job": job["name"], "error_type": type(error).__name__,
                    "error": str(error), "utc": timestamp()})
                payload["jobs"].setdefault(job["name"], {
                    **job, "pid": None, "pgid": None, "status": "spawn_failed",
                    "return_code": None})
                terminate_owned(processes, payload)
                files, complete = result_files(payload["jobs"][job["name"]])
                payload["jobs"][job["name"]].update(
                    results_complete=complete, result_files=files)
                payload.update(ended_utc=timestamp(), gpu_snapshot_after=gpu_snapshot())
                write_receipt(receipt, payload)
                return 1

        pending = set(processes)
        while pending:
            for name in tuple(pending):
                return_code = processes[name].poll()
                if return_code is None:
                    continue
                pending.remove(name)
                streams[name].flush()
                streams[name].close()
                files, complete = result_files(payload["jobs"][name])
                payload["jobs"][name].update(
                    status="completed" if return_code == 0 and complete else "failed",
                    process_exit_status="zero" if return_code == 0 else "nonzero",
                    return_code=return_code, ended_utc=timestamp(),
                    results_complete=complete, result_files=files)
                write_receipt(receipt, payload)
            if pending:
                time.sleep(5)
    finally:
        for stream in streams.values():
            if not stream.closed:
                stream.close()
    payload["status"] = "completed" if all(
        job["status"] == "completed" for job in payload["jobs"].values()) else "failed"
    payload.update(ended_utc=timestamp(), gpu_snapshot_after=gpu_snapshot())
    write_receipt(receipt, payload)
    return 0 if payload["status"] == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    return run(Path(args.spec), Path(args.receipt))


if __name__ == "__main__":
    raise SystemExit(main())
