"""No real trainer/GPU subprocesses: mock process exits and signal delivery."""
import json
from pathlib import Path
import signal
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import supervise_motiondrive_v2_job as supervisor


@pytest.fixture
def job(tmp_path):
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / supervisor.TRAINER).write_text("# never executed\n")
    run = root / "work_dirs/motiondrive_v2/test_job"
    return root, root / "logs/motiondrive_v2/test_job.supervisor.json", run


def argv(run):
    return [supervisor.TRAINER, "--phase", "joint", "--gpu", "1", "--run-dir", str(run)]


class FakeChild:
    pid = 12345
    def __init__(self, codes):
        self.codes = iter(codes)
        self.signals = []
        self.kill_calls = 0
    def poll(self):
        return next(self.codes)
    def send_signal(self, sig):
        self.signals.append(sig)
    def kill(self):
        self.kill_calls += 1


def manifest(run, status, pid=12345):
    run.mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(json.dumps({"status": status, "pid": pid, "step": 500}))


@pytest.mark.parametrize("code,outcome,exit_code", [(0, "completed_cleanly", 0),
                                                   (-11, "child_signaled", 139),
                                                   (2, "child_nonzero_exit", 2)])
def test_actual_exit_is_separate_from_completed_manifest(job, code, outcome, exit_code):
    root, record, run = job
    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        manifest(run, "completed")
        return FakeChild([code])
    result = supervisor.supervise_job(root, record, argv(run), process_factory=popen,
                                     install_signal_handlers=False)
    assert result["actual_returncode"] == code
    assert result["trainer_manifest"]["status"] == "completed"
    assert result["outcome"] == outcome
    assert result["supervisor_exit_code"] == exit_code
    assert calls[0][0][0] == sys.executable
    assert calls[0][1]["start_new_session"] is False
    assert calls[0][1]["stdout"] is None and calls[0][1]["stderr"] is None
    if code == -11:
        assert result["termination_signal"] == 11
        assert result["termination_signal_name"] == "SIGSEGV"


def test_no_record_in_run_dir_and_new_record_reserved_exclusively(job):
    root, record, run = job
    def popen(*args, **kwargs):
        assert record.exists()
        assert not run.exists()  # supervisor has not broken trainer's empty-run gate
        return FakeChild([0])
    result = supervisor.supervise_job(root, record, argv(run), process_factory=popen,
                                     install_signal_handlers=False)
    assert result["outcome"] == "zero_exit_without_completed_trainer_manifest"
    with pytest.raises(supervisor.SupervisorError, match="existing"):
        supervisor.supervise_job(root, record, argv(run), process_factory=lambda *_a, **_k: pytest.fail("spawn"))
    with pytest.raises(supervisor.SupervisorError, match="escapes"):
        supervisor.validate_command(root, run / "record.json", argv(run))


def test_completed_hang_records_pending_without_kill_or_early_success(job):
    root, record, run = job
    child = FakeChild([None, None, None, -11])
    tick, snapshots = [0.], []
    def popen(*args, **kwargs):
        manifest(run, "completed")
        return child
    def sleep(seconds):
        snapshots.append(json.loads(record.read_text()))
        tick[0] += seconds
    result = supervisor.supervise_job(root, record, argv(run), process_factory=popen,
        sleeper=sleep, monotonic=lambda: tick[0], poll_seconds=1, teardown_grace_seconds=1,
        install_signal_handlers=False)
    assert any(r["status"] == "teardown_pending" for r in snapshots)
    assert all(r["actual_returncode"] is None for r in snapshots)
    assert child.signals == []
    assert result["actual_returncode"] == -11
    assert result["teardown_pending_observed"]
    assert result["teardown_pending_first_observed_at"]


def test_sigterm_only_targets_exact_owned_child(job, monkeypatch):
    root, record, run = job
    handlers = {}
    monkeypatch.setattr(supervisor.signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler))
    monkeypatch.setattr(supervisor.os, "killpg", lambda *_: pytest.fail("must not signal a group"))
    child = FakeChild([None, None, -15])
    first = [True]
    def sleep(_seconds):
        if first[0]:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            first[0] = False
    result = supervisor.supervise_job(root, record, argv(run), process_factory=lambda *a, **k: child,
                                     sleeper=sleep, install_signal_handlers=True)
    assert child.signals == [signal.SIGTERM]
    assert result["received_signals"][0]["child_pid"] == child.pid
    assert result["received_signals"][0]["forwarded_to_exact_child"]


def test_old_manifest_pid_cannot_claim_child_success(job):
    root, record, run = job
    manifest(run, "completed", pid=98765)
    result = supervisor.supervise_job(root, record, argv(run), process_factory=lambda *a, **k: FakeChild([0]),
                                     install_signal_handlers=False)
    assert not result["trainer_reported_completed"]
    assert result["supervisor_exit_code"] == 1


@pytest.mark.parametrize("replacement", [["evil.py", "--gpu", "1"],
                                         [supervisor.TRAINER, "--gpu", "6"],
                                         [supervisor.TRAINER, "--gp", "1"],
                                         [supervisor.TRAINER, "--gpu", "1", "--gpu=2"]])
def test_command_scope_is_restricted(job, replacement):
    root, record, run = job
    with pytest.raises(supervisor.SupervisorError):
        supervisor.validate_command(root, record, replacement + ["--run-dir", str(run)])


def test_spawn_failure_is_persisted_without_deleting_artifacts(job):
    root, record, run = job
    def fail(*a, **k):
        raise OSError("mock spawn failure")
    result = supervisor.supervise_job(root, record, argv(run), process_factory=fail,
                                     install_signal_handlers=False)
    assert result["status"] == "spawn_failed" and result["child_pid"] is None
    assert record.exists() and not run.exists()


@pytest.mark.parametrize("escaped_directory", ["work_dirs", "logs"])
def test_symlinked_scope_cannot_escape_project(job, escaped_directory, tmp_path):
    root, record, run = job
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / escaped_directory).symlink_to(outside, target_is_directory=True)
    with pytest.raises(supervisor.SupervisorError, match="escapes"):
        supervisor.validate_command(root, record, argv(run))


def memory_sample(gpu, free):
    return {"physical_gpu": gpu, "gpu_uuid": f"GPU-test-{gpu}", "free_mib": free}


@pytest.mark.parametrize("option,expected", [([], 0), (["--cuda-min-free-mib", "0"], 0),
                                            (["--cuda-min-free-mib", "8192"], 8192),
                                            (["--cuda-min-free-mib=8192"], 8192)])
def test_pressure_guard_explicit_opt_in(job, option, expected):
    root, record, run = job
    prepared = supervisor.validate_command(root, record, argv(run) + option)
    assert prepared["cuda_min_free_mib"] == expected


@pytest.mark.parametrize("option", [["--cuda-min-free-mib", "-1"],
                                    ["--cuda-min-free-mib", "1.5"],
                                    ["--cuda-min-free-mib", "nan"],
                                    ["--cuda-min-free-mib"], ["--cuda-min-free-mib="],
                                    ["--cuda-min-free", "8192"],
                                    ["--cuda-min-free-mib", "8192", "--cuda-min-free-mib=4096"]])
def test_pressure_policy_malformed_options_fail_before_spawn(job, option):
    root, record, run = job
    with pytest.raises(supervisor.SupervisorError):
        supervisor.supervise_job(root, record, argv(run) + option,
                                 process_factory=lambda *_a, **_k: pytest.fail("must not spawn"))
    assert not record.exists()


@pytest.mark.parametrize("option", [[], ["--cuda-min-free-mib", "0"]])
def test_disabled_pressure_never_queries_or_automatically_signals(job, option):
    root, record, run = job
    child = FakeChild([None, None, 0])
    manifest(run, "completed")
    result = supervisor.supervise_job(root, record, argv(run) + option,
        process_factory=lambda *_a, **_k: child, sleeper=lambda _: None,
        gpu_memory_query=lambda _: pytest.fail("disabled guard queried GPU"), install_signal_handlers=False)
    assert result["outcome"] == "completed_cleanly"
    assert not result["automatic_kill_enabled"]
    assert "pressure_policy" not in result
    assert child.signals == [] and child.kill_calls == 0


def test_memory_query_only_requests_one_physical_gpu(job, monkeypatch):
    calls = []
    def query(command, **kwargs):
        calls.append((command, kwargs))
        return type("Output", (), {"stdout": "1, GPU-test-1, 8192\n"})()
    monkeypatch.setattr(supervisor.subprocess, "run", query)
    sample = supervisor.query_gpu_free_memory(1)
    assert sample["physical_gpu"] == 1 and sample["free_mib"] == 8192
    assert calls[0][0] == ["nvidia-smi", "-i", "1", "--query-gpu=index,uuid,memory.free",
                           "--format=csv,noheader,nounits"]
    assert calls[0][1]["timeout"] == 5 and calls[0][1]["check"] is True
    with pytest.raises(supervisor.SupervisorError, match="GPU0--3"):
        supervisor.query_gpu_free_memory(6)
    assert len(calls) == 1


@pytest.mark.parametrize("output", ["", "1, GPU-test, 8192\n2, GPU-other, 9000\n",
                                    "2, GPU-test, 8192\n", "1, unknown, 8192\n",
                                    "1, GPU-test, nan\n", "1, GPU-test, -1\n", "bad output\n"])
def test_memory_query_rejects_unusable_evidence(monkeypatch, output):
    monkeypatch.setattr(supervisor.subprocess, "run",
                        lambda *_a, **_k: type("Output", (), {"stdout": output})())
    with pytest.raises((supervisor.SupervisorError, ValueError)):
        supervisor.query_gpu_free_memory(1)


def test_pressure_poll_defaults_to_five_seconds_and_threshold_is_strict(job):
    root, record, run = job
    child = FakeChild([None] * 12 + [0])
    manifest(run, "completed")
    tick, observations = [0.], []
    def sleep(seconds):
        tick[0] += seconds
    def query(gpu):
        observations.append((tick[0], gpu))
        return memory_sample(gpu, 8192)
    result = supervisor.supervise_job(root, record, argv(run) + ["--cuda-min-free-mib", "8192"],
        process_factory=lambda *_a, **_k: child, sleeper=sleep, monotonic=lambda: tick[0],
        gpu_memory_query=query, install_signal_handlers=False)
    assert observations == [(0., 1), (5., 1), (10., 1)]
    assert result["pressure_policy"]["poll_seconds"] == 5.
    assert result["pressure_policy"]["safety_grace_seconds"] == 30.
    assert result["pressure_checks"]["count"] == 3
    assert result["pressure_event"] is None
    assert result["outcome"] == "completed_cleanly"
    assert child.signals == [] and child.kill_calls == 0


def test_pressure_sigterm_once_and_graceful_completed_exit_is_failure(job, monkeypatch):
    root, record, run = job
    child = FakeChild([None, None, 0])
    tick, calls = [0.], []
    manifest(run, "completed")
    monkeypatch.setattr(supervisor.os, "killpg", lambda *_: pytest.fail("foreign/group signal"))
    monkeypatch.setattr(supervisor.os, "kill", lambda *_: pytest.fail("raw PID signal"))
    def query(gpu):
        calls.append(gpu)
        return memory_sample(gpu, 8191)
    def sleep(seconds):
        tick[0] += seconds
    def send(sig):
        evidence = json.loads(record.read_text())["pressure_event"]
        assert evidence["child_pid"] == child.pid and evidence["sigterm_attempted"]
        child.signals.append(sig)
    child.send_signal = send
    result = supervisor.supervise_job(root, record, argv(run) + ["--cuda-min-free-mib", "8192"],
        process_factory=lambda *_a, **_k: child, sleeper=sleep, monotonic=lambda: tick[0],
        gpu_memory_query=query, poll_seconds=5, install_signal_handlers=False)
    assert child.signals == [signal.SIGTERM] and child.kill_calls == 0
    assert calls == [1]  # no repeated queries or signals after the latched event
    assert result["actual_returncode"] == 0 and result["trainer_reported_completed"]
    assert result["outcome"] == "cuda_memory_pressure" and result["supervisor_exit_code"] == 1
    assert result["exit_outcome_without_pressure"] == "completed_cleanly"
    assert result["pressure_event"]["reason"] == "below_minimum_free_memory"
    assert result["pressure_event"]["sigterm_sent"]


@pytest.mark.parametrize("error", [OSError("query unavailable"),
                                   supervisor.subprocess.TimeoutExpired("nvidia-smi", 5),
                                   ValueError("malformed query")])
def test_query_failure_fails_closed_and_only_signals_owned_child(job, error):
    root, record, run = job
    child = FakeChild([None, -15])
    def query(_gpu):
        raise error
    result = supervisor.supervise_job(root, record, argv(run) + ["--cuda-min-free-mib", "8192"],
        process_factory=lambda *_a, **_k: child, sleeper=lambda _: None,
        gpu_memory_query=query, install_signal_handlers=False)
    assert child.signals == [signal.SIGTERM] and child.kill_calls == 0
    assert result["pressure_event"]["reason"] == "memory_query_failed"
    assert type(error).__name__ in result["pressure_checks"]["last"]["query_error"]
    assert result["actual_returncode"] == -15 and result["supervisor_exit_code"] != 0


@pytest.mark.parametrize("process_poll_seconds", [5, 45])
def test_pressure_kills_exact_child_once_only_after_thirty_seconds(job, monkeypatch, process_poll_seconds):
    root, record, run = job
    child = FakeChild([None] * 9 + [-9])
    tick, kill_times, snapshots = [0.], [], []
    manifest(run, "completed")
    monkeypatch.setattr(supervisor.os, "killpg", lambda *_: pytest.fail("foreign/group signal"))
    monkeypatch.setattr(supervisor.os, "kill", lambda *_: pytest.fail("raw PID signal"))
    def sleep(seconds):
        snapshots.append(json.loads(record.read_text()))
        tick[0] += seconds
    def kill():
        evidence = json.loads(record.read_text())["pressure_event"]
        assert evidence["kill_attempted"] and evidence["child_pid"] == child.pid
        kill_times.append(tick[0])
        child.kill_calls += 1
    child.kill = kill
    result = supervisor.supervise_job(root, record, argv(run) + ["--cuda-min-free-mib", "8192"],
        process_factory=lambda *_a, **_k: child, sleeper=sleep, monotonic=lambda: tick[0],
        gpu_memory_query=lambda gpu: memory_sample(gpu, 0), poll_seconds=process_poll_seconds,
        install_signal_handlers=False)
    assert child.signals == [signal.SIGTERM] and child.kill_calls == 1
    assert kill_times == [30.]
    assert snapshots[0]["status"] == "pressure_terminating"
    assert any(row["status"] == "pressure_kill_pending" for row in snapshots)
    assert result["pressure_event"]["kill_sent"]
    assert result["actual_returncode"] == -9 and result["termination_signal_name"] == "SIGKILL"
    assert result["outcome"] == "cuda_memory_pressure"


@pytest.mark.parametrize("sample", [None, {"physical_gpu": 4, "gpu_uuid": "GPU-other", "free_mib": 99999},
                                    {"physical_gpu": 1, "gpu_uuid": "GPU-test", "free_mib": float("nan")}])
def test_injected_invalid_memory_records_fail_closed(job, sample):
    root, record, run = job
    child = FakeChild([None, -15])
    queried = []
    def query(gpu):
        queried.append(gpu)
        return sample
    result = supervisor.supervise_job(root, record, argv(run) + ["--cuda-min-free-mib", "8192"],
        process_factory=lambda *_a, **_k: child, sleeper=lambda _: None,
        gpu_memory_query=query, install_signal_handlers=False)
    assert queried == [1] and child.signals == [signal.SIGTERM]
    assert result["pressure_event"]["reason"] == "memory_query_failed"
    assert result["supervisor_exit_code"] == 1
