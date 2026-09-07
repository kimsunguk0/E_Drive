#!/usr/bin/env python3
"""Run one fixed P3 query-adapter screening arm on explicitly allowed GPU4/5.

This parent owns only its exact Popen child. It preserves actual exit status;
a completed trainer manifest never overrides a native failure. No other job,
process group, GPU6/7, checkpoint or existing output is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.supervise_motiondrive_v2_job import OwnedRecord, now, trainer_snapshot
from scripts.evaluate_motiondrive_v2_shared import query_evaluation_memory, canonical_cuda_uuid
from scripts.launch_motiondrive_v2_trials import inspect_repository

SCRIPT = "scripts/run_motiondrive_v2_query_trial.py"
TRAINER = "scripts/train_motiondrive_v2_query_adapter.py"
MODEL = "models/motiondrive_v2_query_adapter.py"
INIT = "work_dirs/motiondrive_v2/p2_c1t1_s0/last.pth"
INIT_SHA = "5cd98d28157abf169f0f2378c3d0ab29d43f4d6382e9c03ee611e7fc5d864c20"
ARCHITECTURE = "motiondrive_v2_image_state_query_v1"
CAP_MIB, RESERVE_MIB = 12000, 8192


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def valid_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def source_snapshot(root, expected_commit):
    state = inspect_repository(root)
    require(state["actual_git_sha"] == expected_commit and not state["tracked_dirty"],
            "P3 requires the explicitly pinned clean source commit")
    from scripts.evaluate_motiondrive_v2_planning import source_manifest
    original = source_manifest()
    require(original["git_sha"] == expected_commit and not original["tracked_changes"], "Evaluator source mismatch")
    files = {**original["file_sha256"]}
    for name in (SCRIPT, TRAINER, MODEL, "scripts/evaluate_motiondrive_v2_shared.py",
                 "scripts/supervise_motiondrive_v2_job.py", "scripts/launch_motiondrive_v2_trials.py"):
        files[name] = sha256(Path(root) / name)
    return {"git_sha": expected_commit, "file_sha256": files}


def prepare(root, arm, physical_gpu, expected_commit, attempt=1):
    root = Path(root).resolve()
    require(root == ROOT and arm in ("control", "state"), "Fixed repository and P3 arm required")
    require(type(physical_gpu) is int and physical_gpu in (4, 5), "Only explicitly authorized physical GPU4/5")
    require(type(attempt) is int and attempt in (1, 2), "Only explicitly identified first/repaired attempts")
    suffix = "" if attempt == 1 else "_r2"
    run = root / f"work_dirs/motiondrive_v2/p3_query_{arm}_s0{suffix}"
    record = root / f"logs/motiondrive_v2/p3_query_{arm}_s0{suffix}.supervisor.json"
    log = root / f"logs/motiondrive_v2/p3_query_{arm}_s0{suffix}.log"
    prior = None
    if attempt == 2:
        prior_path = root / f"logs/motiondrive_v2/p3_query_{arm}_s0.supervisor.json"
        prior_manifest = root / f"work_dirs/motiondrive_v2/p3_query_{arm}_s0/manifest.json"
        old = json.loads(prior_path.read_text())
        old_m = json.loads(prior_manifest.read_text())
        require(old.get("actual_returncode") == 1 and old.get("supervisor_exit_code") == 1
                and old.get("outcome") == "failed" and old.get("pressure_event") is None
                and old_m.get("step") == 0 and old_m.get("status") == "failed"
                and old_m.get("nonfinite_count") == 0 and old_m.get("error_type") == "ValueError"
                and str(old_m.get("error", "")).startswith("초기 full-tune 재현 실패"),
                "r2 requires the preserved step-zero numeric-context failure")
        require(all(type(old.get(k)) is int and old[k] > 0
                    and not Path(f"/proc/{old[k]}").exists() for k in ("parent_pid", "child_pid")),
                "Previous P3 processes must have actually exited")
        prior = {"record": str(prior_path), "record_sha256": sha256(prior_path),
                 "manifest": str(prior_manifest), "manifest_sha256": sha256(prior_manifest),
                 "actual_returncode": 1, "optimizer_steps": 0}
    for p in (run, record, log):
        require(not os.path.lexists(p), f"Refusing existing P3 output: {p}")
        require(p.parent.is_dir() and p.resolve().is_relative_to(root), "Invalid P3 output parent")
    require((root / INIT).resolve().is_relative_to(root), "Initializer escapes repository root")
    require(sha256(root / INIT) == INIT_SHA, "Pinned C1T1 initialization changed")
    # The original trainer actually exited cleanly; do not accept a terminal JSON alone.
    original = json.loads((root / "logs/motiondrive_v2/p2_c1t1_s0.supervisor.json").read_text())
    require(original.get("actual_returncode") == 0 and original.get("supervisor_exit_code") == 0
            and original.get("outcome") == "completed_cleanly" and original.get("pressure_event") is None,
            "C1T1 source training must have a verified clean OS exit")
    require(all(type(original.get(k)) is int and original[k] > 0 for k in ("child_pid", "supervisor_pid")),
            "Original trainer process identities must be positive integer PIDs")
    require(all(not Path(f"/proc/{original[k]}").exists() for k in ("child_pid", "supervisor_pid")),
            "Original trainer processes must be absent")
    return {"root": str(root), "arm": arm, "adapter_on": arm == "state", "physical_gpu": physical_gpu,
            "attempt": attempt, "preserved_prior_failure": prior,
            "run_dir": str(run), "manifest_path": str(run / "manifest.json"), "record": str(record), "log": str(log),
            "sources": source_snapshot(root, expected_commit), "expected_commit": expected_commit,
            "initializer": str(root / INIT), "initializer_sha256": INIT_SHA,
            "command": [sys.executable, str(root / TRAINER), "--run-dir", str(run), "--init", str(root / INIT),
                        "--expected-init-sha256", INIT_SHA, "--adapter-on", str(int(arm == "state"))]}


def check_memory(query, gpu, expected_uuid=None, admission=False):
    d = query(gpu)
    require(type(d.get("physical_gpu")) is int and d["physical_gpu"] == gpu, "Physical memory query mismatch")
    require(isinstance(d.get("gpu_uuid"), str) and d["gpu_uuid"].startswith("GPU-"), "GPU UUID required")
    free = d.get("free_mib")
    require(type(free) in (int, float) and 0 <= free < float("inf"), "Finite nonnegative free MiB required")
    require(expected_uuid is None or d["gpu_uuid"] == expected_uuid, "GPU UUID changed")
    if admission:
        require(free >= CAP_MIB + RESERVE_MIB, "Insufficient memory before P3 launch")
    return d


def idle_gpu(gpu):
    require(type(gpu) is int and gpu in (4, 5), "Idle check is restricted to GPU4/5")
    memory = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=5)
    lines = [line.strip() for line in memory.stdout.splitlines() if line.strip()]
    require(len(lines) == 1 and lines[0].isascii() and lines[0].isdecimal()
            and int(lines[0]) < 1000, "Requested idle GPU must use less than 1000 MiB")
    result = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=5)
    require(not result.stdout.strip(), "Requested idle GPU has another compute process; no job is stopped")


def validate_completion(request, child_pid):
    require(not Path(request["manifest_path"]).is_symlink(), "Trainer manifest must not be a symlink")
    manifest = json.loads(Path(request["manifest_path"]).read_text())
    require(manifest.get("pid") == child_pid and manifest.get("status") == "completed"
            and manifest.get("step") == 1000 and manifest.get("nonfinite_count") == 0,
            "Expected completed LAST1000 trainer manifest")
    require(manifest.get("architecture") == ARCHITECTURE
            and manifest.get("query_adapter_on") is request["adapter_on"], "P3 architecture/arm mismatch")
    require(valid_sha256(manifest.get("frozen_initial_sha256")) and
            manifest["frozen_initial_sha256"] == manifest.get("frozen_final_sha256"), "Frozen trunk changed")
    require(manifest.get("init_checkpoint_sha256") == INIT_SHA == request["initializer_sha256"],
            "Completed trainer initializer provenance mismatch")
    require(manifest.get("git_sha") == request["expected_commit"] and manifest.get("source_unchanged") is True,
            "Completed trainer source provenance mismatch")
    memory = manifest.get("cuda_memory_policy", {})
    require(memory.get("enabled") is True
            and type(memory.get("allocator_limit_mib")) is int and memory["allocator_limit_mib"] == CAP_MIB
            and type(memory.get("min_free_mib")) is int and memory["min_free_mib"] == RESERVE_MIB,
            "Completed trainer CUDA memory policy mismatch")
    device = manifest.get("device_mapping", {})
    require(device.get("physical_gpu") == request["physical_gpu"] and device.get("logical_device") == "cuda:0"
            and device.get("observed_uuid") == request["gpu_uuid"]
            and canonical_cuda_uuid(device.get("observed_uuid_raw")) == request["gpu_uuid"]
            and device.get("cuda_visible_devices") == request["gpu_uuid"], "Actual CUDA identity mismatch")
    require(sha256(request["initializer"]) == request["initializer_sha256"], "Source checkpoint changed")
    require(source_snapshot(request["root"], request["expected_commit"]) == request["sources"], "P3 sources changed")
    prior = request.get("preserved_prior_failure")
    if prior is not None:
        require(sha256(prior["record"]) == prior["record_sha256"]
                and sha256(prior["manifest"]) == prior["manifest_sha256"], "Prior failed attempt was altered")
    last = Path(request["run_dir"]) / "last.pth"
    require(last.is_file() and not last.is_symlink(), "Missing or symlinked LAST checkpoint")
    return {"manifest_sha256": sha256(request["manifest_path"]), "last_sha256": sha256(last),
            "frozen_state_sha256": manifest["frozen_final_sha256"]}


def cleanup_owned_child(child):
    """Keep ownership until this Popen is reaped, even when signaling fails."""
    evidence = {"child_pid": None if child is None else child.pid, "actual_returncode": None,
                "sigterm_attempted": False, "kill_attempted": False, "reaped": False, "errors": []}
    if child is None:
        return evidence

    def error(stage, exc):
        evidence["errors"].append({"stage": stage, "error": f"{type(exc).__name__}: {exc}"})

    try:
        rc = child.poll()  # Popen.poll() itself waitpid/reaps an already exited child.
    except BaseException as exc:
        error("poll", exc)
        rc = None
    if rc is None:
        evidence["sigterm_attempted"] = True
        try:
            child.terminate()
        except BaseException as exc:
            error("sigterm", exc)
        try:
            rc = child.wait(timeout=30)
        except BaseException as exc:
            error("grace_wait", exc)
            evidence["kill_attempted"] = True
            try:
                child.kill()
            except BaseException as kill_error:
                error("kill", kill_error)
            # A sent signal (or ProcessLookupError) is not evidence of reaping.
            # This uses Popen.wait, not the possibly broken injected sleeper.
            while rc is None:
                try:
                    rc = child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    continue
                except BaseException as wait_error:
                    error("reap_wait", wait_error)
                    try:
                        rc = child.poll()
                    except BaseException as poll_error:
                        error("reap_poll", poll_error)
                    if rc is None:
                        time.sleep(.1)
    evidence.update(actual_returncode=rc, reaped=rc is not None)
    return evidence


def supervise(request, *, query=None, check_idle=None, popen=None, sleep=None, clock=None,
              validator=None, install_handlers=True):
    query, check_idle = query or query_evaluation_memory, check_idle or idle_gpu
    popen, sleep, clock = popen or subprocess.Popen, sleep or time.sleep, clock or time.monotonic
    gpu = request["physical_gpu"]
    require(type(gpu) is int and gpu in (4, 5), "P3 is restricted to physical GPU4/5")
    admitted = check_memory(query, gpu, admission=True)
    check_idle(gpu)
    request = {**request, "gpu_uuid": admitted["gpu_uuid"]}
    record = {"schema_version": 1, "supervisor_record_id": uuid.uuid4().hex, "parent_pid": os.getpid(),
              "child_pid": None, "request": request, "status": "reserved", "actual_returncode": None,
              "created_at": now(), "preflight": admitted, "memory_policy": {"allocator_limit_mib": CAP_MIB,
              "reserve_mib": RESERVE_MIB, "poll_seconds": 5, "grace_seconds": 30},
              "pressure_event": None, "minimum_free_mib": None, "received_signals": [],
              "signal_scope": "Only the exact owned Popen child; no foreign PID or process group"}
    owned = OwnedRecord(request["record"], record)
    child, pending, old_handlers = None, [], {}
    stop_time, next_check, killed = None, -float("inf"), False
    try:
        if install_handlers:
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_handlers[sig] = signal.signal(sig, lambda signum, _: pending.append(signum))
        with Path(request["log"]).open("x", encoding="utf-8") as log:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": admitted["gpu_uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                   "MOTIONDRIVE_EXPECTED_GPU_UUID": admitted["gpu_uuid"],
                   "MOTIONDRIVE_EXPECTED_PHYSICAL_GPU": str(gpu), "PYTHONDONTWRITEBYTECODE": "1",
                   "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"}
            child = popen(request["command"], cwd=request["root"], env=env,
                          stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=False)
            record.update(child_pid=child.pid, status="running", started_at=now())
            owned.write(record)
            while True:
                # A signal received at the exit boundary must not disappear
                # merely because the following poll already returns zero.
                new_signals = []
                while pending:
                    new_signals.append(pending.pop(0))
                record["received_signals"].extend(new_signals)
                rc = child.poll()
                if rc is not None:
                    while pending:
                        record["received_signals"].append(pending.pop(0))
                    valid = None
                    if rc == 0 and record["pressure_event"] is None and not record["received_signals"]:
                        try:
                            valid = (validator or validate_completion)(request, child.pid)
                        except Exception as exc:
                            record["validation_error"] = f"{type(exc).__name__}: {exc}"
                    while pending:
                        record["received_signals"].append(pending.pop(0))
                    success = valid is not None and not record["received_signals"]
                    record.update(status="process_exited", actual_returncode=rc,
                                  termination_signal=-rc if rc < 0 else None, ended_at=now(),
                                  outcome="completed_cleanly" if success else "failed",
                                  supervisor_exit_code=0 if success else (128 - rc if rc < 0 else rc or 1),
                                  completion_evidence=valid)
                    owned.write(record)
                    return record
                if new_signals and stop_time is None:
                    owned.write(record)
                    child.send_signal(signal.SIGTERM)
                    stop_time = clock()
                if stop_time is None and clock() >= next_check:
                    try:
                        sample = check_memory(query, gpu, request["gpu_uuid"])
                        seen = record["minimum_free_mib"]
                        record["minimum_free_mib"] = sample["free_mib"] if seen is None else min(seen, sample["free_mib"])
                        require(sample["free_mib"] >= RESERVE_MIB, "Minimum CUDA reserve exhausted")
                        record["last_memory"] = sample
                    except Exception as exc:
                        record["pressure_event"] = {"error": f"{type(exc).__name__}: {exc}", "at": now()}
                        owned.write(record)
                        child.send_signal(signal.SIGTERM)
                        stop_time = clock()
                    next_check = clock() + 5.
                if stop_time is not None and not killed and clock() - stop_time >= 30:
                    killed = True
                    record["kill_attempted_at"] = now()
                    owned.write(record)
                    child.kill()
                record["trainer_manifest"] = trainer_snapshot(request["manifest_path"], child.pid)
                owned.write(record)
                sleep(1.)
    except BaseException as exc:
        # A failed parent must not orphan its GPU child. Wait/reap, not just signal.
        cleanup = cleanup_owned_child(child)
        record.update(status="parent_failed", actual_returncode=cleanup["actual_returncode"],
                      parent_cleanup=cleanup,
                      error=f"{type(exc).__name__}: {exc}", ended_at=now(), supervisor_exit_code=1)
        try:
            owned.write(record)
        except Exception as record_error:
            print(json.dumps({"parent_failed": True, "record_not_overwritten": True,
                              "actual_returncode": record["actual_returncode"], "parent_cleanup": cleanup,
                              "record_error": f"{type(record_error).__name__}: {record_error}"}),
                  file=sys.stderr, flush=True)
        raise
    finally:
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--arm", choices=("control", "state"), required=True)
    p.add_argument("--physical-gpu", type=int, choices=(4, 5), required=True)
    p.add_argument("--expected-git-sha", required=True)
    p.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    args = p.parse_args(argv)
    request = prepare(args.root, args.arm, args.physical_gpu, args.expected_git_sha, args.attempt)
    result = supervise(request)
    print(json.dumps({"outcome": result["outcome"], "actual_returncode": result["actual_returncode"],
                      "record": request["record"]}))
    return result["supervisor_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
