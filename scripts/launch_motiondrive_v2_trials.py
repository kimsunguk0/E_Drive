#!/usr/bin/env python3
"""Preflight a JSON trial plan; only --launch creates artifacts or processes.

The trainer is the only allowed child script. Launches use separate process
groups and expose physical GPUs 0--3 by UUID, preserving the requested indices.
This is a launcher, not a recurring training monitor.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone


class LaunchError(ValueError):
    pass


NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
TRAINER = "scripts/train_motiondrive_v2.py"
CONTROLLED = ("--gpu", "--run-dir")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained(path, anchor):
    resolved, boundary = Path(path).resolve(), Path(anchor).resolve()
    if not resolved.is_relative_to(boundary):
        raise LaunchError(f"Path escapes allowed directory: {path}")
    return resolved


def options_from_argv(argv):
    """Keep explicit option values, including nargs lists, without shell parsing."""
    options, current = {}, None
    for token in argv[1:]:
        if token == "--":
            raise LaunchError("The -- terminator would hide launcher-owned arguments")
        if token.startswith("--"):
            option, equal, value = token.partition("=")
            if any(reserved.startswith(option) for reserved in CONTROLLED):
                raise LaunchError(f"Launcher owns {option}; do not include it in argv")
            if option in options:
                raise LaunchError(f"Duplicate option: {option}")
            options[option] = [value] if equal else []
            current = option
        elif current is None:
            raise LaunchError(f"Unexpected positional trainer argument: {token}")
        else:
            options[current].append(token)
    return options


def scalar_option(options, name, required=False):
    if name not in options:
        if required:
            raise LaunchError(f"Plan must explicitly provide {name}")
        return None
    if len(options[name]) != 1 or not options[name][0]:
        raise LaunchError(f"{name} requires exactly one value")
    return options[name][0]


def validate_plan(plan):
    """Validate every job and artifact path without creating anything."""
    if not isinstance(plan, dict):
        raise LaunchError("Plan must be a JSON object")
    allowed = {"experiment_id", "root", "expected_git_sha", "jobs", "description"}
    if set(plan) - allowed:
        raise LaunchError(f"Unknown plan keys: {sorted(set(plan) - allowed)}")
    experiment = plan.get("experiment_id", "")
    if not isinstance(experiment, str) or not NAME.fullmatch(experiment):
        raise LaunchError("Invalid experiment_id")
    root_value = plan.get("root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise LaunchError("root must be an absolute project directory")
    root = Path(root_value).resolve()
    if not root.is_dir() or root == Path(root.anchor):
        raise LaunchError("root must be an existing non-filesystem-root project directory")
    trainer = contained(root / TRAINER, root)
    if not trainer.is_file():
        raise LaunchError(f"Missing allowed trainer: {trainer}")
    run_base = contained(root / "work_dirs/motiondrive_v2", root)
    log_base = contained(root / "logs/motiondrive_v2", root)
    manifest = contained(log_base / f"launch_{experiment}.json", log_base)
    lexical_targets = [root / "logs/motiondrive_v2" / f"launch_{experiment}.json"]
    if not isinstance(plan.get("description", ""), str):
        raise LaunchError("description must be a string")
    expected = plan.get("expected_git_sha")
    if expected is not None and (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{7,40}", expected)):
        raise LaunchError("expected_git_sha must contain 7--40 hexadecimal characters")
    jobs_in = plan.get("jobs")
    if not isinstance(jobs_in, list) or not jobs_in:
        raise LaunchError("jobs must be a nonempty list")
    jobs, names, gpus, hashes = [], set(), set(), {}

    def fingerprint(value, kind):
        path = Path(value)
        path = (path if path.is_absolute() else root / path).resolve()
        if kind == "supervision":
            if not path.is_dir():
                raise LaunchError(f"Missing supervision directory: {path}")
            source = path / "supervision_manifest.json"
        else:
            source = path
        if not source.is_file():
            raise LaunchError(f"Missing {kind} input: {source}")
        if source not in hashes:
            hashes[source] = sha256(source)
        return {"path": str(path), "sha256_source": str(source),
                "sha256": hashes[source], "size_bytes": source.stat().st_size}

    for spec in jobs_in:
        if not isinstance(spec, dict) or set(spec) != {"name", "gpu", "argv"}:
            raise LaunchError("Each job must contain exactly name, gpu and argv")
        name, gpu, argv = spec["name"], spec["gpu"], spec["argv"]
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise LaunchError(f"Invalid job name: {name!r}")
        if name in names:
            raise LaunchError(f"Duplicate job name: {name}")
        if type(gpu) is not int or gpu not in range(4):
            raise LaunchError("Only physical GPUs 0, 1, 2, 3 may be assigned")
        if gpu in gpus:
            raise LaunchError(f"Duplicate GPU assignment: {gpu}")
        if (not isinstance(argv, list) or not argv or argv[0] != TRAINER
                or any(not isinstance(x, str) or not x or "\0" in x for x in argv)):
            raise LaunchError(f"argv must start with exactly {TRAINER} and contain nonempty strings")
        names.add(name)
        gpus.add(gpu)
        run_dir = contained(run_base / name, run_base)
        log_path = contained(log_base / f"{name}.log", log_base)
        lexical_targets.extend((root / "work_dirs/motiondrive_v2" / name,
                                root / "logs/motiondrive_v2" / f"{name}.log"))
        options = options_from_argv(argv)
        lineage = {}
        for flag, kind in (("--init", "checkpoint_init"), ("--resume", "checkpoint_resume"),
                           ("--pretrained", "public_pretrained"),
                           ("--split-manifest", "split"), ("--supervision-root", "supervision")):
            value = scalar_option(options, flag, required=flag in ("--split-manifest", "--supervision-root"))
            if value is not None:
                lineage[kind] = fingerprint(value, "supervision" if kind == "supervision" else kind)
        if "--init" in options and "--resume" in options:
            raise LaunchError("Trainer cannot combine --init with --resume")
        jobs.append({"name": name, "gpu": gpu, "run_dir": str(run_dir), "log_path": str(log_path),
                     "explicit_argv": list(argv), "explicit_options": options, "inputs": lineage,
                     "command": [sys.executable, *argv, "--gpu", str(gpu), "--run-dir", str(run_dir)]})
    # Check the complete target set before the caller can reserve or launch anything.
    for path in [*lexical_targets, manifest, *(Path(j[k]) for j in jobs for k in ("run_dir", "log_path"))]:
        if path.exists() or path.is_symlink():
            raise LaunchError(f"Refusing existing artifact: {path}")
    return {"schema_version": 1, "experiment_id": experiment, "root": str(root),
            "description": plan.get("description", ""), "expected_git_sha": expected,
            "launch_manifest": str(manifest), "trainer_sha256": sha256(trainer), "jobs": jobs}


def inspect_repository(root):
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              text=True, capture_output=True).stdout.strip()
    return {"actual_git_sha": git("rev-parse", "HEAD"),
            "tracked_dirty": git("status", "--porcelain=v1", "--untracked-files=no")}


def inspect_gpus():
    def query(argument):
        output = subprocess.run(["nvidia-smi", argument, "--format=csv,noheader,nounits"],
                                check=True, text=True, capture_output=True).stdout
        return list(csv.reader(io.StringIO(output), skipinitialspace=True))
    devices = {}
    for row in query("--query-gpu=index,uuid,memory.used"):
        if not row:
            continue
        index, uuid, used = row
        devices[int(index)] = {"uuid": uuid.strip(), "memory_used_mib": float(used)}
    processes = []
    for row in query("--query-compute-apps=gpu_uuid,pid,process_name"):
        if not row or row[0].startswith("No running processes"):
            continue
        uuid, pid, name = row
        processes.append({"gpu_uuid": uuid.strip(), "pid": int(pid), "process_name": name.strip()})
    return {"devices": devices, "compute_processes": processes}


def preflight(prepared):
    repository = inspect_repository(prepared["root"])
    if repository["tracked_dirty"]:
        raise LaunchError(f"Tracked source is dirty: {repository['tracked_dirty']}")
    actual = repository["actual_git_sha"]
    expected = prepared["expected_git_sha"]
    if expected and not actual.lower().startswith(expected.lower()):
        raise LaunchError(f"Source SHA mismatch: expected {expected}, got {actual}")
    gpu = inspect_gpus()
    devices, processes = gpu["devices"], gpu["compute_processes"]
    selected = sorted(j["gpu"] for j in prepared["jobs"])
    # CUDA ids become the UUID order, so requested logical id N still maps to
    # physical nvidia-smi GPU N. GPUs 4--7 never enter the child CUDA namespace.
    visible = sorted(index for index in devices if index in range(4))
    if visible != list(range(len(visible))) or not all(index in visible for index in selected):
        raise LaunchError("Cannot preserve requested GPU indices in a GPU0--3-only CUDA namespace")
    known_uuids = {entry["uuid"] for entry in devices.values()}
    if any(proc["gpu_uuid"] not in known_uuids for proc in processes):
        raise LaunchError("Compute-process GPU UUID cannot be mapped safely (possibly MIG)")
    for index in selected:
        entry = devices[index]
        if not (0 <= entry["memory_used_mib"] < 1000):
            raise LaunchError(f"GPU {index} memory is not below 1000 MiB: {entry['memory_used_mib']}")
        if any(proc["gpu_uuid"] == entry["uuid"] for proc in processes):
            raise LaunchError(f"GPU {index} has an active compute process")
    return {**repository, "gpu_preflight": gpu,
            "child_environment_overrides": {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                                             "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                             "CUDA_VISIBLE_DEVICES": ",".join(devices[i]["uuid"] for i in visible)}}


def now():
    return datetime.now(timezone.utc).isoformat()


def replace_manifest(path, record):
    """Atomically replace only the manifest this invocation exclusively reserved."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def terminate_owned(started):
    cleanup = []
    for job, process in started:
        item = {"name": job["name"], "pid": process.pid, "process_group": process.pid}
        try:
            # Popen(start_new_session=True) created exactly this process group.
            os.killpg(process.pid, signal.SIGTERM)
            item["sigterm_sent"] = True
        except ProcessLookupError:
            item["already_exited"] = True
        except OSError as exc:
            item["signal_error"] = str(exc)
        cleanup.append(item)
    deadline = time.monotonic() + 5.
    for item, (_, process) in zip(cleanup, started):
        try:
            item["returncode"] = process.wait(timeout=max(0., deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            item["termination_pending"] = True
    return cleanup


def execute_plan(plan, *, launch=False):
    prepared = validate_plan(plan)
    readiness = preflight(prepared)
    record = {**prepared, **readiness, "status": "dry_run", "created_at": now(), "processes": []}
    if not launch:
        return record
    # Recheck every target/source/GPU before any artifact reservation or child.
    prepared = validate_plan(plan)
    readiness = preflight(prepared)
    record.update(prepared)
    record.update(readiness)
    path = Path(prepared["launch_manifest"])
    handles, started, manifest_owned = [], [], False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            manifest_owned = True
            record["status"] = "reserving"
            json.dump(record, stream, indent=2, ensure_ascii=False)
        # Reserve ALL logs/run directories before the first Popen. Trainer allows
        # a pre-existing empty run directory and writes its own run manifest.
        for job in prepared["jobs"]:
            Path(job["run_dir"]).parent.mkdir(parents=True, exist_ok=True)
            Path(job["run_dir"]).mkdir(exist_ok=False)
            handles.append(Path(job["log_path"]).open("x", encoding="utf-8"))
        env = {**os.environ, **record["child_environment_overrides"]}
        for job, stream in zip(prepared["jobs"], handles):
            process = subprocess.Popen(job["command"], cwd=prepared["root"],
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=env)
            started.append((job, process))
            record["processes"].append({"name": job["name"], "gpu": job["gpu"],
                                        "pid": process.pid, "process_group": process.pid,
                                        "started_at": now(), "log_path": job["log_path"]})
            record["status"] = "launching"
            replace_manifest(path, record)
            if process.poll() is not None:
                raise LaunchError(f"Child exited during launch: {job['name']} ({process.returncode})")
        for job, process in started:
            if process.poll() is not None:
                raise LaunchError(f"Child exited during launch: {job['name']} ({process.returncode})")
        record.update(status="launched", launched_at=now(),
                      notice="Processes spawned; training progress and completion require separate inspection.")
        replace_manifest(path, record)
    except BaseException as exc:
        cleanup = terminate_owned(started)
        record.update(status="launch_failed", failed_at=now(), error=f"{type(exc).__name__}: {exc}",
                      cleanup=cleanup, artifacts_preserved=True)
        if manifest_owned:
            replace_manifest(path, record)
        raise LaunchError(f"Launch failed; only newly started process groups were terminated. {exc}") from exc
    finally:
        for stream in handles:
            stream.close()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--launch", action="store_true", help="Explicitly reserve artifacts and start processes")
    args = parser.parse_args()
    try:
        plan = json.loads(Path(args.plan).read_text())
        result = execute_plan(plan, launch=args.launch)
    except (LaunchError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
