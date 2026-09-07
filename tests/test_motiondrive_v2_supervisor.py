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
    def poll(self):
        return next(self.codes)
    def send_signal(self, sig):
        self.signals.append(sig)


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
