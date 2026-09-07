#!/usr/bin/env python3
"""Persist actual trainer process termination separately from training status.

Usage (launcher creates the surrounding process group/log redirection):
  python scripts/supervise_motiondrive_v2_job.py --root /project \
    --record /project/logs/motiondrive_v2/job.supervisor.json -- \
    scripts/train_motiondrive_v2.py ... --gpu 0 --run-dir /project/work_dirs/motiondrive_v2/job

Only the reviewed trainer is executable. No shell, new child session, deletion,
or interpretation of stdout completion messages is used. By default a child still
alive after its manifest becomes terminal is recorded as teardown_pending and
never automatically killed. Explicit trainer --cuda-min-free-mib > 0 enables a
separate shared-GPU safety policy: pressure or a failed memory query terminates
only our Popen child, escalating to child.kill() after the safety grace period.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

TRAINER = "scripts/train_motiondrive_v2.py"
TERMINAL_TRAINER_STATES = {"completed", "failed", "stopped"}


class SupervisorError(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def contained(path, directory):
    candidate, boundary = Path(path).resolve(), Path(directory).resolve()
    if not candidate.is_relative_to(boundary):
        raise SupervisorError(f"Path escapes allowed directory: {path}")
    return candidate


def validate_command(root, record, argv):
    root = Path(root).resolve()
    if not root.is_dir() or root == Path(root.anchor):
        raise SupervisorError("root must be an existing project directory")
    if (not argv or argv[0] != TRAINER or any(not isinstance(t, str) or not t or "\0" in t for t in argv)):
        raise SupervisorError(f"Only {TRAINER} with nonempty string arguments is allowed")
    script = contained(root / TRAINER, root)
    if not script.is_file():
        raise SupervisorError("Allowed trainer does not exist")
    controlled = {}
    for i, token in enumerate(argv[1:], 1):
        if token == "--":
            raise SupervisorError("Nested option terminator is not allowed")
        if not token.startswith("--"):
            continue
        option, equal, value = token.partition("=")
        for name in ("--gpu", "--run-dir", "--cuda-min-free-mib"):
            if name.startswith(option) and name != option:
                raise SupervisorError("Abbreviated controlled options are not allowed")
        if option not in ("--gpu", "--run-dir", "--cuda-min-free-mib"):
            continue
        if option in controlled:
            raise SupervisorError(f"Duplicate controlled option: {option}")
        if not equal:
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise SupervisorError(f"Missing value for {option}")
            value = argv[i + 1]
        controlled[option] = value
    if not {"--gpu", "--run-dir"}.issubset(controlled):
        raise SupervisorError("Explicit --gpu and --run-dir are required")
    if controlled["--gpu"] not in {"0", "1", "2", "3"}:
        raise SupervisorError("Only GPU0--3 may be assigned")
    minimum = controlled.get("--cuda-min-free-mib", "0")
    if not minimum.isascii() or not minimum.isdecimal():
        raise SupervisorError("--cuda-min-free-mib must be a nonnegative integer; 0 disables the guard")
    run = Path(controlled["--run-dir"])
    run_base = contained(root / "work_dirs/motiondrive_v2", root)
    run = contained(run if run.is_absolute() else root / run, run_base)
    if run == run_base:
        raise SupervisorError("run-dir must name a single run, not its parent")
    record_lexical = Path(record)
    record_lexical = record_lexical if record_lexical.is_absolute() else root / record_lexical
    if record_lexical.exists() or record_lexical.is_symlink():
        raise SupervisorError(f"Refusing existing supervisor record: {record_lexical}")
    log_base = contained(root / "logs/motiondrive_v2", root)
    record_path = contained(record_lexical, log_base)
    if record_path == log_base or record_path.suffix != ".json":
        raise SupervisorError("record must be a new JSON file under logs/motiondrive_v2")
    # Use this interpreter, never a caller-supplied executable or shell command.
    return {"root": root, "record": record_path, "run_dir": run,
            "manifest_path": run / "manifest.json", "command": [sys.executable, *argv],
            "gpu": int(controlled["--gpu"]), "cuda_min_free_mib": int(minimum)}


def query_gpu_free_memory(gpu):
    """Read only the requested physical GPU; no CUDA context or foreign signals."""
    if type(gpu) is not int or gpu not in range(4):
        raise SupervisorError("Memory guard is restricted to physical GPU0--3")
    command = ["nvidia-smi", "-i", str(gpu), "--query-gpu=index,uuid,memory.free",
               "--format=csv,noheader,nounits"]
    output = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5).stdout
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise SupervisorError("Memory query must return exactly one GPU")
    fields = [value.strip() for value in lines[0].split(",")]
    if len(fields) != 3:
        raise SupervisorError("Malformed GPU memory query")
    index, gpu_uuid, free = fields
    if index != str(gpu) or not gpu_uuid.startswith("GPU-"):
        raise SupervisorError("Memory query physical GPU identity mismatch")
    free = float(free)
    if not math.isfinite(free) or free < 0:
        raise SupervisorError("GPU free memory must be finite and nonnegative")
    return {"physical_gpu": gpu, "gpu_uuid": gpu_uuid, "free_mib": free,
            "command": command}


class OwnedRecord:
    """Exclusive reservation, then atomic replacement of only our own record."""
    def __init__(self, path, record):
        self.path = Path(path)
        self.identity = record["supervisor_record_id"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def write(self, record):
        if self.path.is_symlink():
            raise SupervisorError("Supervisor record replaced by a symlink")
        previous = json.loads(self.path.read_text())
        if previous.get("supervisor_record_id") != self.identity:
            raise SupervisorError("Supervisor record ownership changed")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                         prefix=self.path.name + ".", suffix=".tmp", delete=False) as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, self.path)


def trainer_snapshot(path, child_pid):
    try:
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError("Manifest is not an object")
    except FileNotFoundError:
        return {"exists": False, "status": None, "belongs_to_child": False}
    except (OSError, ValueError) as exc:
        return {"exists": True, "status": None, "belongs_to_child": False,
                "read_error": f"{type(exc).__name__}: {exc}"}
    return {"exists": True, "status": raw.get("status"), "pid": raw.get("pid"),
            "step": raw.get("step"), "belongs_to_child": raw.get("pid") == child_pid,
            "manifest_path": str(path)}


def exit_fields(returncode, snapshot):
    """A completed manifest cannot convert a native signal/nonzero exit to success."""
    sig = -returncode if returncode < 0 else None
    try:
        name = signal.Signals(sig).name if sig is not None else None
    except ValueError:
        name = f"SIGNAL_{sig}"
    trainer_complete = snapshot.get("belongs_to_child") and snapshot.get("status") == "completed"
    if sig is not None:
        outcome, supervisor_code = "child_signaled", 128 + sig
    elif returncode:
        outcome, supervisor_code = "child_nonzero_exit", returncode
    elif trainer_complete:
        outcome, supervisor_code = "completed_cleanly", 0
    else:
        outcome, supervisor_code = "zero_exit_without_completed_trainer_manifest", 1
    return {"actual_returncode": int(returncode), "exit_code": returncode if returncode >= 0 else None,
            "termination_signal": sig, "termination_signal_name": name, "outcome": outcome,
            "trainer_reported_completed": bool(trainer_complete), "supervisor_exit_code": supervisor_code}


def supervise_job(root, record_path, argv, *, teardown_grace_seconds=60., poll_seconds=1.,
                  process_factory=None, sleeper=None, monotonic=None, install_signal_handlers=True,
                  gpu_memory_query=None, pressure_poll_seconds=5., safety_grace_seconds=30.):
    if not math.isfinite(teardown_grace_seconds) or teardown_grace_seconds < 0:
        raise SupervisorError("teardown grace must be finite and nonnegative")
    if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 45:
        raise SupervisorError("poll interval must be between 0 and 45 seconds")
    if not math.isfinite(pressure_poll_seconds) or not 0 < pressure_poll_seconds <= 45:
        raise SupervisorError("pressure poll interval must be between 0 and 45 seconds")
    if not math.isfinite(safety_grace_seconds) or safety_grace_seconds < 0:
        raise SupervisorError("safety grace must be finite and nonnegative")
    prepared = validate_command(root, record_path, argv)
    popen = process_factory or subprocess.Popen
    sleep, clock = sleeper or time.sleep, monotonic or time.monotonic
    pressure_enabled = prepared["cuda_min_free_mib"] > 0
    query_memory = gpu_memory_query or query_gpu_free_memory
    record = {"schema_version": 1, "supervisor_record_id": uuid.uuid4().hex,
              "supervisor_pid": os.getpid(), "supervisor_process_group": os.getpgrp(),
              "child_pid": None, "child_start_new_session": False,
              "command": prepared["command"], "run_dir": str(prepared["run_dir"]),
              "trainer_manifest_path": str(prepared["manifest_path"]), "gpu": prepared["gpu"],
              "created_at": now(), "status": "reserved", "actual_returncode": None,
              "termination_signal": None, "trainer_manifest": None,
              "teardown_grace_seconds": teardown_grace_seconds,
              "teardown_pending_observed": False,
              "received_signals": [], "automatic_kill_enabled": pressure_enabled}
    if pressure_enabled:
        record.update(pressure_policy={
            "enabled": True, "source": "trainer --cuda-min-free-mib",
            "physical_gpu": prepared["gpu"], "minimum_free_mib": prepared["cuda_min_free_mib"],
            "poll_seconds": pressure_poll_seconds, "safety_grace_seconds": safety_grace_seconds,
            "trigger": "free_mib < minimum_free_mib OR query failure",
            "signal_scope": "exact Popen child only; never another PID or process group",
            "actions": ["SIGTERM once", "kill once if still alive after safety grace"]},
            pressure_checks={"count": 0, "last": None, "minimum_observed_free_mib": None},
            pressure_event=None)
    owned = OwnedRecord(prepared["record"], record)
    pending, previous_handlers = [], {}
    def received(signum, _frame):
        # The Python handler only queues. The loop signals the exact Popen child;
        # it never expands this request to other processes or a process group.
        pending.append(int(signum))
    if install_signal_handlers:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, received)
    try:
        try:
            child = popen(prepared["command"], cwd=str(prepared["root"]),
                          stdin=subprocess.DEVNULL, stdout=None, stderr=None,
                          start_new_session=False)
        except (OSError, subprocess.SubprocessError) as exc:
            record.update(status="spawn_failed", error=f"{type(exc).__name__}: {exc}",
                          ended_at=now(), supervisor_exit_code=125)
            owned.write(record)
            return record
        record.update(child_pid=child.pid, status="running", child_started_at=now())
        owned.write(record)
        terminal_since, first_terminal_at = None, None
        pressure_since, next_pressure_check = None, -math.inf
        while True:
            rc = child.poll()
            for sig in pending[:]:
                pending.pop(0)
                event = {"signal": sig, "received_at": now(), "child_pid": child.pid,
                         "forwarded_to_exact_child": False}
                if rc is None:
                    try:
                        child.send_signal(sig)
                        event["forwarded_to_exact_child"] = True
                    except ProcessLookupError:
                        event["child_already_exited"] = True
                else:
                    event["child_already_exited"] = True
                record["received_signals"].append(event)
            snapshot = trainer_snapshot(prepared["manifest_path"], child.pid)
            record["trainer_manifest"] = snapshot
            record["last_observed_at"] = now()
            if rc is not None:
                record.update(status="process_exited", ended_at=now(), **exit_fields(rc, snapshot))
                if pressure_enabled and record["pressure_event"] is not None:
                    # Even a graceful rc=0/completed exit after pressure is not a
                    # successful experiment. Preserve the actual native exit above.
                    record["exit_outcome_without_pressure"] = record["outcome"]
                    record.update(outcome="cuda_memory_pressure", supervisor_exit_code=1)
                owned.write(record)
                return record
            if pressure_enabled and pressure_since is None and clock() >= next_pressure_check:
                sample = {"observed_at": now(), "physical_gpu": prepared["gpu"]}
                reason = None
                try:
                    observed = query_memory(prepared["gpu"])
                    if (not isinstance(observed, dict) or observed.get("physical_gpu") != prepared["gpu"]
                            or not isinstance(observed.get("gpu_uuid"), str)
                            or not observed["gpu_uuid"].startswith("GPU-")):
                        raise SupervisorError("Memory query identity/record mismatch")
                    free = float(observed["free_mib"])
                    if not math.isfinite(free) or free < 0:
                        raise SupervisorError("Memory query returned invalid free memory")
                    sample.update(observed, free_mib=free)
                    minimum_seen = record["pressure_checks"]["minimum_observed_free_mib"]
                    record["pressure_checks"]["minimum_observed_free_mib"] = (
                        free if minimum_seen is None else min(minimum_seen, free))
                    if free < prepared["cuda_min_free_mib"]:
                        reason = "below_minimum_free_memory"
                except Exception as exc:
                    # A failed/malformed query cannot silently disable a guard
                    # that was explicitly requested for a shared GPU.
                    sample["query_error"] = f"{type(exc).__name__}: {exc}"
                    reason = "memory_query_failed"
                record["pressure_checks"]["count"] += 1
                record["pressure_checks"]["last"] = sample
                next_pressure_check = clock() + pressure_poll_seconds
                if reason is not None:
                    pressure_since = clock()
                    event = {"reason": reason, "observed_at": now(), "child_pid": child.pid,
                             "query": sample, "sigterm_attempted": True, "sigterm_sent": False,
                             "kill_attempted": False, "kill_sent": False}
                    record["pressure_event"] = event
                    record["status"] = "pressure_terminating"
                    owned.write(record)  # persist ownership/evidence before any signal
                    try:
                        child.send_signal(signal.SIGTERM)
                        event.update(sigterm_sent=True, sigterm_sent_at=now())
                    except ProcessLookupError:
                        event["child_already_exited_before_sigterm"] = True
                    except OSError as exc:
                        event["sigterm_error"] = f"{type(exc).__name__}: {exc}"
                    pressure_since = clock()
                    event["safety_grace_started_at"] = now()
            if pressure_since is not None:
                event = record["pressure_event"]
                elapsed = max(0., clock() - pressure_since)
                event["seconds_since_pressure"] = elapsed
                if elapsed >= safety_grace_seconds and not event["kill_attempted"]:
                    event.update(kill_attempted=True, kill_attempted_at=now())
                    owned.write(record)
                    try:
                        child.kill()
                        event.update(kill_sent=True, kill_sent_at=now())
                    except ProcessLookupError:
                        event["child_already_exited_before_kill"] = True
                    except OSError as exc:
                        event["kill_error"] = f"{type(exc).__name__}: {exc}"
                record["status"] = "pressure_kill_pending" if event["kill_attempted"] else "pressure_terminating"
                owned.write(record)
                sleep(min(poll_seconds, pressure_poll_seconds))
                continue
            terminal = snapshot.get("belongs_to_child") and snapshot.get("status") in TERMINAL_TRAINER_STATES
            if terminal:
                if terminal_since is None:
                    terminal_since, first_terminal_at = clock(), now()
                elapsed = max(0., clock() - terminal_since)
                record["trainer_terminal_first_observed_at"] = first_terminal_at
                record["seconds_alive_after_trainer_terminal"] = elapsed
                record["status"] = "teardown_pending" if elapsed >= teardown_grace_seconds else "running"
                if record["status"] == "teardown_pending":
                    record["teardown_pending_observed"] = True
                    record.setdefault("teardown_pending_first_observed_at", now())
            else:
                terminal_since, first_terminal_at = None, None
                record["status"] = "running"
            # Ordinary trainer teardown never enables automatic killing. Only
            # the explicit pressure policy above may terminate the owned child.
            owned.write(record)
            sleep(min(poll_seconds, pressure_poll_seconds) if pressure_enabled else poll_seconds)
    finally:
        for sig, old in previous_handlers.items():
            signal.signal(sig, old)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--root", required=True)
    p.add_argument("--record", required=True)
    p.add_argument("--teardown-grace-seconds", type=float, default=60.)
    p.add_argument("--poll-seconds", type=float, default=1.)
    p.add_argument("trainer_argv", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.trainer_argv[1:] if args.trainer_argv[:1] == ["--"] else args.trainer_argv
    try:
        result = supervise_job(args.root, args.record, command,
                               teardown_grace_seconds=args.teardown_grace_seconds,
                               poll_seconds=args.poll_seconds)
    except (SupervisorError, OSError) as exc:
        print(json.dumps({"supervisor_error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 125
    return result["supervisor_exit_code"]


if __name__ == "__main__":
    sys.exit(main())
