import json
from pathlib import Path
import signal
import subprocess

import pytest

from scripts import run_motiondrive_v2_query_trial as runner


@pytest.fixture
def trial_request(tmp_path):
    run = tmp_path / "work/run"
    run.parent.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    return {"root": str(tmp_path), "arm": "control", "adapter_on": False, "physical_gpu": 4,
            "run_dir": str(run), "manifest_path": str(run / "manifest.json"),
            "record": str(logs / "new.json"), "log": str(logs / "new.log"),
            "command": ["mock-python", "mock-training"], "sources": {}, "expected_commit": "b" * 40,
            "initializer": str(tmp_path / "init.pth"), "initializer_sha256": "a" * 64}


class Child:
    pid = 98765
    def __init__(self, codes):
        self.codes = iter(codes)
        self.last = None
        self.signals = []
        self.kills = self.waits = 0
    def poll(self):
        self.last = next(self.codes, self.last)
        return self.last
    def send_signal(self, sig):
        self.signals.append(sig)
    def terminate(self):
        self.send_signal(signal.SIGTERM)
    def kill(self):
        self.kills += 1
        self.last = -9
    def wait(self, timeout=None):
        self.waits += 1
        if self.last is not None:
            # Once wait reaps an exit, later poll cannot return queued None.
            self.codes = iter(())
            return self.last
        if timeout is not None:
            raise subprocess.TimeoutExpired("own-child", timeout)
        return self.last


def memory(gpu, free=180000):
    return {"physical_gpu": gpu, "gpu_uuid": f"GPU-test-{gpu}", "free_mib": free}


def run(trial_request, child, failure=None, validator=None):
    commands, queries, idle, ticks = [], [], [], [0.]
    def query(gpu):
        queries.append(gpu)
        if len(queries) > 1:
            if failure == "query":
                raise OSError("synthetic query failure")
            if failure == "pressure":
                return memory(gpu, 8191)
        return memory(gpu)
    def spawn(command, **kw):
        commands.append((command, kw))
        return child
    def sleep(seconds):
        ticks[0] += seconds
    result = runner.supervise(trial_request, query=query, check_idle=idle.append, popen=spawn,
                              sleep=sleep, clock=lambda: ticks[0], validator=validator or (lambda *_: {}),
                              install_handlers=False)
    return result, commands, queries, idle, ticks[0]


@pytest.mark.parametrize("gpu", [4, 5])
def test_only_selected_gpu_visible_and_actual_exit_recorded(trial_request, gpu):
    trial_request["physical_gpu"] = gpu
    child = Child([None, 0])
    result, commands, queries, idle, _ = run(trial_request, child)
    assert set(queries) == {gpu} and idle == [gpu]
    env = commands[0][1]["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == f"GPU-test-{gpu}"
    assert env["MOTIONDRIVE_EXPECTED_GPU_UUID"] == f"GPU-test-{gpu}"
    assert env["MOTIONDRIVE_EXPECTED_PHYSICAL_GPU"] == str(gpu)
    assert commands[0][1]["start_new_session"] is False
    assert result["actual_returncode"] == 0 and result["outcome"] == "completed_cleanly"
    assert child.signals == [] and child.kills == 0
    assert "env" not in result


@pytest.mark.parametrize("gpu", [0, 1, 2, 3, 6, 7, True, 4.])
def test_non_p3_physical_gpu_rejected(trial_request, gpu):
    trial_request["physical_gpu"] = gpu
    with pytest.raises(ValueError):
        runner.supervise(trial_request, query=lambda *_: pytest.fail("no query"), install_handlers=False)


@pytest.mark.parametrize("free", [20191, float("nan"), float("inf"), -1, True, "180000"])
def test_admission_cannot_spawn_without_headroom(trial_request, free):
    with pytest.raises(ValueError):
        runner.supervise(trial_request, query=lambda gpu: memory(gpu, free),
                          check_idle=lambda *_: pytest.fail("no idle lookup"),
                          popen=lambda *_a, **_k: pytest.fail("no launch"), install_handlers=False)
    assert not Path(trial_request["record"]).exists()


def test_busy_gpu_is_not_stopped_or_launched(trial_request):
    def busy(_):
        raise ValueError("busy")
    with pytest.raises(ValueError, match="busy"):
        runner.supervise(trial_request, query=memory, check_idle=busy,
                         popen=lambda *_a, **_k: pytest.fail("no launch"), install_handlers=False)
    assert not Path(trial_request["record"]).exists()


@pytest.mark.parametrize("rc", [1, -11])
def test_nonzero_exit_cannot_become_success(trial_request, rc):
    child = Child([rc])
    result, *_ = run(trial_request, child, validator=lambda *_: pytest.fail("invalid OS exit"))
    assert result["actual_returncode"] == rc and result["outcome"] == "failed"
    assert result["supervisor_exit_code"] == (139 if rc == -11 else 1)


@pytest.mark.parametrize("failure", ["pressure", "query"])
def test_pressure_or_query_failure_stops_only_own_child_and_rc0_still_fails(trial_request, failure, monkeypatch):
    monkeypatch.setattr(runner.os, "kill", lambda *_: pytest.fail("foreign PID forbidden"))
    monkeypatch.setattr(runner.os, "killpg", lambda *_: pytest.fail("process group forbidden"))
    child = Child([None, 0])
    result, *_ = run(trial_request, child, failure=failure, validator=lambda *_: pytest.fail("pressure cannot pass"))
    assert result["actual_returncode"] == 0 and result["outcome"] == "failed"
    assert result["pressure_event"] is not None and child.signals == [signal.SIGTERM]


def test_pressure_waits_before_killing_owned_child(trial_request):
    child = Child([None] * 33 + [-9])
    result, _, _, _, seconds = run(trial_request, child, failure="pressure")
    assert seconds >= 30 and child.kills == 1 and child.signals == [signal.SIGTERM]
    assert result["actual_returncode"] == -9 and result["supervisor_exit_code"] == 137


def test_invalid_completed_manifest_is_failure(trial_request):
    def invalid(*_):
        raise ValueError("frozen trunk changed")
    result, *_ = run(trial_request, Child([0]), validator=invalid)
    assert result["outcome"] == "failed" and result["actual_returncode"] == 0
    assert "frozen" in result["validation_error"]


def test_parent_failure_reaps_exact_child(trial_request):
    child = Child([None] * 10)
    def bad_sleep(_):
        raise RuntimeError("parent diagnostic exception")
    with pytest.raises(RuntimeError, match="parent"):
        runner.supervise(trial_request, query=memory, check_idle=lambda _: None, popen=lambda *_a, **_kw: child,
                         sleep=bad_sleep, validator=lambda *_: {}, install_handlers=False)
    assert child.signals == [signal.SIGTERM] and child.kills == 1 and child.waits == 2
    saved = json.loads(Path(trial_request["record"]).read_text())
    assert saved["status"] == "parent_failed" and saved["supervisor_exit_code"] == 1
    assert saved["actual_returncode"] == -9 and child.poll() == -9
    assert saved["parent_cleanup"]["reaped"] is True


def test_existing_record_or_log_never_overwritten(trial_request):
    p = Path(trial_request["record"])
    p.write_text("original")
    with pytest.raises(FileExistsError):
        run(trial_request, Child([0]))
    assert p.read_text() == "original"


def test_cli_cannot_select_unapproved_gpus():
    with pytest.raises(SystemExit):
        runner.main(["--arm", "control", "--physical-gpu", "6", "--expected-git-sha", "a" * 40])


@pytest.mark.parametrize("stage", ["terminate", "kill", "wait"])
def test_cleanup_oserror_cannot_skip_exact_child_reap(trial_request, stage):
    class ErrorChild(Child):
        def terminate(self):
            if stage == "terminate":
                raise PermissionError("synthetic terminate error")
            super().terminate()

        def kill(self):
            super().kill()
            if stage == "kill":
                raise ProcessLookupError("exited during kill")

        def wait(self, timeout=None):
            if stage == "wait" and self.waits == 0:
                self.waits += 1
                raise InterruptedError("synthetic wait interruption")
            return super().wait(timeout)

    child = ErrorChild([None] * 20)
    def failed_sleep(_):
        raise RuntimeError("original parent exception")
    with pytest.raises(RuntimeError, match="original parent exception"):
        runner.supervise(trial_request, query=memory, check_idle=lambda _: None,
                         popen=lambda *_a, **_k: child, sleep=failed_sleep, install_handlers=False)
    saved = json.loads(Path(trial_request["record"]).read_text())
    assert saved["actual_returncode"] == -9 and child.poll() == -9
    assert saved["parent_cleanup"]["reaped"] is True
    assert saved["parent_cleanup"]["errors"] and child.kills == 1
    assert saved["supervisor_exit_code"] == 1


def test_record_ownership_change_is_preserved_but_child_reaped(trial_request, capsys):
    child = Child([None] * 20)
    replacement = {"supervisor_record_id": "someone-else", "preserve": True}
    def fail_after_replacement(_):
        Path(trial_request["record"]).write_text(json.dumps(replacement))
        raise RuntimeError("ownership changed")
    with pytest.raises(RuntimeError, match="ownership changed"):
        runner.supervise(trial_request, query=memory, check_idle=lambda _: None,
                         popen=lambda *_a, **_k: child, sleep=fail_after_replacement,
                         install_handlers=False)
    assert json.loads(Path(trial_request["record"]).read_text()) == replacement
    assert child.poll() == -9 and child.waits == 2
    fallback = json.loads(capsys.readouterr().err)
    assert fallback["record_not_overwritten"] and fallback["actual_returncode"] == -9


def test_pending_signal_at_zero_exit_is_not_reported_clean(trial_request, monkeypatch):
    handlers = {}
    def set_handler(sig, handler):
        old = handlers.get(sig)
        handlers[sig] = handler
        return old
    monkeypatch.setattr(runner.signal, "signal", set_handler)
    child = Child([0])
    def spawn(*_a, **_kw):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return child
    saved = runner.supervise(trial_request, query=memory, check_idle=lambda _: None,
                             popen=spawn, validator=lambda *_: pytest.fail("signal cannot pass"))
    assert saved["actual_returncode"] == 0 and saved["outcome"] == "failed"
    assert saved["received_signals"] == [signal.SIGTERM]
    assert child.signals == []  # Already exited: no signal is delivered to its PID.


def test_signal_during_completion_validation_still_records_unsuccessful_parent(trial_request, monkeypatch):
    handlers = {}
    def set_handler(sig, handler):
        old = handlers.get(sig); handlers[sig] = handler
        return old
    monkeypatch.setattr(runner.signal, "signal", set_handler)
    def validate(*_):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return {"valid_before_signal": True}
    saved = runner.supervise(trial_request, query=memory, check_idle=lambda _: None,
                             popen=lambda *_a, **_k: Child([0]), validator=validate)
    assert saved["actual_returncode"] == 0 and saved["supervisor_exit_code"] == 1
    assert saved["outcome"] == "failed" and saved["received_signals"] == [signal.SIGTERM]


def test_handler_registration_failure_restores_handlers_and_records_no_child(trial_request, monkeypatch):
    restored = []
    def register(sig, handler):
        if sig == signal.SIGINT:
            raise ValueError("handler registration failed")
        if handler is None:
            restored.append(sig)
        return None
    monkeypatch.setattr(runner.signal, "signal", register)
    with pytest.raises(ValueError, match="registration failed"):
        runner.supervise(trial_request, query=memory, check_idle=lambda _: None,
                         popen=lambda *_a, **_k: pytest.fail("no child before handler setup"))
    saved = json.loads(Path(trial_request["record"]).read_text())
    assert saved["status"] == "parent_failed" and saved["actual_returncode"] is None
    assert restored == [signal.SIGTERM]


@pytest.mark.parametrize("used,processes,ok", [(0, "", True), (999, "", True),
                                               (1000, "", False), (0, "42\n", False)])
def test_actual_idle_gpu_checks_used_memory_and_compute_pids(monkeypatch, used, processes, ok):
    commands = []
    def query(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=(f"{used}\n" if "--query-gpu=memory.used" in command else processes))
    monkeypatch.setattr(runner.subprocess, "run", query)
    if ok:
        runner.idle_gpu(4)
    else:
        with pytest.raises(ValueError):
            runner.idle_gpu(4)
    assert all(c[c.index("-i") + 1] == "4" for c in commands)


@pytest.fixture
def prepared_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "work_dirs/motiondrive_v2").mkdir(parents=True)
    (tmp_path / "logs/motiondrive_v2").mkdir(parents=True)
    init = tmp_path / runner.INIT
    init.parent.mkdir()
    init.write_bytes(b"fixed test initializer")
    monkeypatch.setattr(runner, "INIT_SHA", runner.sha256(init))
    original = tmp_path / "logs/motiondrive_v2/p2_c1t1_s0.supervisor.json"
    original.write_text(json.dumps({"actual_returncode": 0, "supervisor_exit_code": 0,
        "outcome": "completed_cleanly", "pressure_event": None,
        "child_pid": 2147483646, "supervisor_pid": 2147483645}))
    source = tmp_path / "source_marker"
    source.write_bytes(b"fixed test source")
    commit = "b" * 40
    def snapshot(root, expected):
        runner.require(Path(root) == tmp_path and expected == commit, "pinned source mismatch")
        return {"git_sha": commit, "file_sha256": {"source_marker": runner.sha256(source)}}
    monkeypatch.setattr(runner, "source_snapshot", snapshot)
    return {"root": tmp_path, "init": init, "original": original, "source": source, "commit": commit}


@pytest.mark.parametrize("arm,gpu", [("control", 4), ("state", 5)])
def test_real_prepare_pins_fixed_paths_original_exit_and_initializer(prepared_repository, arm, gpu):
    f = prepared_repository
    req = runner.prepare(f["root"], arm, gpu, f["commit"])
    assert req["adapter_on"] is (arm == "state") and req["physical_gpu"] == gpu
    assert req["initializer_sha256"] == runner.sha256(f["init"])
    assert req["command"][1] == str(f["root"] / runner.TRAINER)
    assert not Path(req["run_dir"]).exists() and not Path(req["record"]).exists()


def test_repaired_attempt_requires_prior_actual_step_zero_failure_and_preserves_it(prepared_repository):
    f=prepared_repository
    old_run=f["root"]/"work_dirs/motiondrive_v2/p3_query_control_s0"
    old_run.mkdir()
    manifest=old_run/"manifest.json"
    manifest.write_text(json.dumps({"step":0,"status":"failed","nonfinite_count":0,
        "error_type":"ValueError","error":"초기 full-tune 재현 실패: numeric context"}))
    record=f["root"]/"logs/motiondrive_v2/p3_query_control_s0.supervisor.json"
    old={"actual_returncode":1,"supervisor_exit_code":1,"outcome":"failed","pressure_event":None,
         "parent_pid":2147483646,"child_pid":2147483645}
    record.write_text(json.dumps(old))
    before=(manifest.read_bytes(),record.read_bytes())
    req=runner.prepare(f["root"],"control",4,f["commit"],attempt=2)
    assert req["run_dir"].endswith("p3_query_control_s0_r2")
    assert req["record"].endswith("p3_query_control_s0_r2.supervisor.json")
    assert req["preserved_prior_failure"]["record_sha256"]==runner.sha256(record)
    assert (manifest.read_bytes(),record.read_bytes())==before
    old["actual_returncode"]=0
    record.write_text(json.dumps(old))
    with pytest.raises(ValueError,match="step-zero"):
        runner.prepare(f["root"],"control",4,f["commit"],attempt=2)


@pytest.mark.parametrize("attempt",[0,3,True,"2"])
def test_invalid_attempt_is_not_a_run_path_escape(prepared_repository,attempt):
    f=prepared_repository
    with pytest.raises(ValueError,match="attempts"):
        runner.prepare(f["root"],"control",4,f["commit"],attempt=attempt)


@pytest.mark.parametrize("mutation", ["init", "exit", "pressure", "pid", "pid_type", "source", "outside_root", "gpu6"])
def test_real_prepare_rejects_wrong_lineage_or_scope(prepared_repository, mutation):
    f = prepared_repository
    root, gpu, commit = f["root"], 4, f["commit"]
    d = json.loads(f["original"].read_text())
    if mutation == "init":
        f["init"].write_bytes(b"changed")
    elif mutation == "exit":
        d["actual_returncode"] = -11
    elif mutation == "pressure":
        d["pressure_event"] = {"error": "pressure"}
    elif mutation == "pid":
        d["child_pid"] = runner.os.getpid()
    elif mutation == "pid_type":
        d["child_pid"] = "2147483646"
    elif mutation == "source":
        commit = "a" * 40
    elif mutation == "outside_root":
        root = root.parent
    elif mutation == "gpu6":
        gpu = 6
    f["original"].write_text(json.dumps(d))
    with pytest.raises(ValueError):
        runner.prepare(root, "control", gpu, commit)


@pytest.mark.parametrize("target", ["run_dir", "record", "log"])
def test_prepare_existing_artifact_refuses_without_writes(prepared_repository, target):
    f = prepared_repository
    req = runner.prepare(f["root"], "control", 4, f["commit"])
    p = Path(req[target]); p.write_text("existing user artifact")
    with pytest.raises(ValueError, match="existing"):
        runner.prepare(f["root"], "control", 4, f["commit"])
    assert p.read_text() == "existing user artifact"


def test_prepare_rejects_dangling_output_symlink(prepared_repository):
    f = prepared_repository
    p = f["root"] / "logs/motiondrive_v2/p3_query_control_s0.log"
    p.symlink_to(f["root"].parent / "nonexistent")
    with pytest.raises(ValueError, match="existing"):
        runner.prepare(f["root"], "control", 4, f["commit"])
    assert p.is_symlink()


@pytest.fixture
def completed_trial(prepared_repository):
    f = prepared_repository
    req = runner.prepare(f["root"], "control", 4, f["commit"])
    req["gpu_uuid"] = "GPU-test-4"
    Path(req["run_dir"]).mkdir()
    (Path(req["run_dir"]) / "last.pth").write_bytes(b"completed test artifact")
    d = {"pid": 98765, "status": "completed", "step": 1000, "nonfinite_count": 0,
         "architecture": runner.ARCHITECTURE, "query_adapter_on": False,
         "frozen_initial_sha256": "c" * 64, "frozen_final_sha256": "c" * 64,
         "init_checkpoint_sha256": runner.INIT_SHA, "git_sha": f["commit"], "source_unchanged": True,
         "cuda_memory_policy": {"enabled": True, "allocator_limit_mib": 12000, "min_free_mib": 8192},
         "device_mapping": {"physical_gpu": 4, "logical_device": "cuda:0", "observed_uuid": "GPU-test-4",
                            "observed_uuid_raw": "test-4", "cuda_visible_devices": "GPU-test-4"}}
    Path(req["manifest_path"]).write_text(json.dumps(d))
    return f, req, d


def test_real_validate_completion_accepts_full_contract(completed_trial):
    _, req, _ = completed_trial
    result = runner.validate_completion(req, 98765)
    assert result["frozen_state_sha256"] == "c" * 64 and len(result["last_sha256"]) == 64


@pytest.mark.parametrize("key,value", [
    ("pid", 42), ("status", "stopped"), ("step", 999), ("nonfinite_count", 1),
    ("architecture", "legacy"), ("query_adapter_on", True),
    ("frozen_initial_sha256", "short"), ("frozen_final_sha256", "d" * 64),
    ("init_checkpoint_sha256", "0" * 64), ("git_sha", "a" * 40), ("source_unchanged", False),
    ("cuda_memory_policy.enabled", False), ("cuda_memory_policy.allocator_limit_mib", 13000),
    ("cuda_memory_policy.min_free_mib", 0), ("device_mapping.physical_gpu", 5),
    ("device_mapping.logical_device", "cuda:1"), ("device_mapping.observed_uuid_raw", "other"),
    ("device_mapping.observed_uuid", "GPU-other"), ("device_mapping.cuda_visible_devices", "GPU-other")])
def test_real_completion_rejects_incomplete_or_wrong_contract(completed_trial, key, value):
    _, req, d = completed_trial
    if "." in key:
        parent, key = key.split(".")
        d[parent][key] = value
    else:
        d[key] = value
    Path(req["manifest_path"]).write_text(json.dumps(d))
    with pytest.raises(ValueError):
        runner.validate_completion(req, 98765)


@pytest.mark.parametrize("change", ["source", "init", "missing_last", "symlink_last", "symlink_manifest"])
def test_completion_rejects_changed_or_replaced_files(completed_trial, change):
    f, req, _ = completed_trial
    last = Path(req["run_dir"]) / "last.pth"
    if change == "source":
        f["source"].write_bytes(b"modified source")
    elif change == "init":
        f["init"].write_bytes(b"modified init")
    elif change == "missing_last":
        last.unlink()
    elif change == "symlink_last":
        last.unlink(); last.symlink_to(f["init"])
    else:
        p = Path(req["manifest_path"]); q = p.with_name("other.json")
        p.rename(q); p.symlink_to(q)
    with pytest.raises(ValueError):
        runner.validate_completion(req, 98765)


def test_source_snapshot_requires_clean_exact_commit_and_hashes_new_sources(tmp_path, monkeypatch):
    from scripts import evaluate_motiondrive_v2_planning as evaluation
    names = (runner.SCRIPT, runner.TRAINER, runner.MODEL, "scripts/evaluate_motiondrive_v2_shared.py",
             "scripts/supervise_motiondrive_v2_job.py", "scripts/launch_motiondrive_v2_trials.py")
    for name in names:
        p = tmp_path / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(name)
    state = {"actual_git_sha": "b" * 40, "tracked_dirty": False}
    monkeypatch.setattr(runner, "inspect_repository", lambda _: state)
    monkeypatch.setattr(evaluation, "source_manifest", lambda: {"git_sha": "b" * 40, "tracked_changes": [], "file_sha256": {}})
    snapshot = runner.source_snapshot(tmp_path, "b" * 40)
    assert set(snapshot["file_sha256"]) == set(names)
    assert snapshot["file_sha256"][runner.SCRIPT] == runner.sha256(tmp_path / runner.SCRIPT)
    state["tracked_dirty"] = True
    with pytest.raises(ValueError):
        runner.source_snapshot(tmp_path, "b" * 40)
    state["tracked_dirty"] = False
    with pytest.raises(ValueError):
        runner.source_snapshot(tmp_path, "a" * 40)
