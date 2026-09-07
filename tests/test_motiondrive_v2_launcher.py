"""CPU-only launcher safety tests; never start a training/GPU process."""
import copy
import json
from pathlib import Path
import signal
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_motiondrive_v2_trials as launcher


@pytest.fixture
def plan(tmp_path):
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/train_motiondrive_v2.py").write_text("# trainer fixture\n")
    (root / "data/supervision").mkdir(parents=True)
    (root / "data/split.json").write_text("{}")
    (root / "data/supervision/supervision_manifest.json").write_text("{}")
    (root / "data/common.pth").write_bytes(b"checkpoint fixture")
    argv = [launcher.TRAINER, "--phase", "joint", "--init", "data/common.pth",
            "--split-manifest", "data/split.json", "--supervision-root", "data/supervision",
            "--goal-on", "0", "--state-on", "0", "--seed", "0"]
    return {"experiment_id": "paired_trial", "root": str(root), "expected_git_sha": "a" * 7,
            "description": "CPU fixture", "jobs": [{"name": "control_s0", "gpu": 0, "argv": argv}]}


@pytest.fixture
def system(monkeypatch):
    gpu = {"devices": {i: {"uuid": f"GPU-{i}", "memory_used_mib": 130.} for i in range(8)},
           "compute_processes": [{"gpu_uuid": "GPU-6", "pid": 999, "process_name": "unrelated"}]}
    monkeypatch.setattr(launcher, "inspect_repository", lambda root: {"actual_git_sha": "a" * 40, "tracked_dirty": ""})
    monkeypatch.setattr(launcher, "inspect_gpus", lambda: gpu)
    return gpu


def test_dry_run_does_not_create_artifacts_or_processes(plan, system, monkeypatch):
    root = Path(plan["root"])
    before = sorted(str(p) for p in root.rglob("*"))
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("dry run spawned a process"))
    result = launcher.execute_plan(plan)
    assert result["status"] == "dry_run"
    assert sorted(str(p) for p in root.rglob("*")) == before
    assert result["child_environment_overrides"]["CUDA_VISIBLE_DEVICES"] == "GPU-0,GPU-1,GPU-2,GPU-3"
    assert result["jobs"][0]["inputs"]["checkpoint_init"]["sha256"]


def test_supervised_plan_preserves_trainer_arguments_and_log_scope(plan, system):
    plan["supervise"] = True
    root = Path(plan["root"])
    (root / launcher.SUPERVISOR).write_text("# supervisor fixture\n")
    result = launcher.execute_plan(plan)
    job = result["jobs"][0]
    assert result["supervise"] and result["supervisor_sha256"]
    assert job["command"][:2] == [sys.executable, launcher.SUPERVISOR]
    separator = job["command"].index("--")
    assert job["command"][separator + 1:] == job["trainer_command"][1:]
    assert Path(job["supervisor_record"]).parent == root / "logs/motiondrive_v2"
    assert not Path(job["run_dir"]).exists()


def test_missing_supervisor_is_rejected_before_launch(plan):
    plan["supervise"] = True
    with pytest.raises(launcher.LaunchError, match="Missing required supervisor"):
        launcher.validate_plan(plan)


def test_existing_supervisor_record_refuses_plan(plan, system):
    plan["supervise"] = True
    root = Path(plan["root"])
    (root / launcher.SUPERVISOR).write_text("# supervisor fixture\n")
    path = root / "logs/motiondrive_v2/control_s0.supervisor.json"
    path.parent.mkdir(parents=True)
    path.write_text("preserve this prior execution record")
    with pytest.raises(launcher.LaunchError, match="existing artifact"):
        launcher.execute_plan(plan, launch=True)
    assert path.read_text() == "preserve this prior execution record"


@pytest.mark.parametrize("value", [1, "true", None])
def test_supervise_flag_requires_boolean(plan, value):
    plan["supervise"] = value
    with pytest.raises(launcher.LaunchError, match="explicit boolean"):
        launcher.validate_plan(plan)


@pytest.mark.parametrize("field,value", [("name", "../escape"), ("name", "/outside"), ("gpu", 6), ("gpu", True)])
def test_rejects_bad_job_scope(plan, field, value):
    plan["jobs"][0][field] = value
    with pytest.raises(launcher.LaunchError):
        launcher.validate_plan(plan)


@pytest.mark.parametrize("parent", ["work_dirs", "logs"])
def test_rejects_symlink_escape(plan, tmp_path, parent):
    outside = tmp_path / "outside"
    outside.mkdir()
    (Path(plan["root"]) / parent).symlink_to(outside, target_is_directory=True)
    with pytest.raises(launcher.LaunchError, match="escapes"):
        launcher.validate_plan(plan)


def test_existing_dangling_run_symlink_is_not_hidden_by_resolution(plan):
    parent = Path(plan["root"]) / "work_dirs/motiondrive_v2"
    parent.mkdir(parents=True)
    (parent / "control_s0").symlink_to(parent / "missing_target", target_is_directory=True)
    with pytest.raises(launcher.LaunchError, match="existing artifact"):
        launcher.validate_plan(plan)


@pytest.mark.parametrize("duplicate", ["gpu", "name"])
def test_rejects_duplicate_gpu_or_name(plan, duplicate):
    other = copy.deepcopy(plan["jobs"][0])
    other.update(name="other_s0", gpu=1)
    other[duplicate] = plan["jobs"][0][duplicate]
    plan["jobs"].append(other)
    with pytest.raises(launcher.LaunchError, match="Duplicate"):
        launcher.validate_plan(plan)


@pytest.mark.parametrize("kind", ["run_dir", "log_path", "launch_manifest"])
def test_existing_artifact_refuses_every_job_before_launch(plan, system, monkeypatch, kind):
    other = copy.deepcopy(plan["jobs"][0])
    other.update(name="second_s0", gpu=1)
    plan["jobs"].append(other)
    prepared = launcher.validate_plan(plan)
    target = Path(prepared[kind] if kind == "launch_manifest" else prepared["jobs"][1][kind])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir() if kind == "run_dir" else target.write_text("preserve me")
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("partial launch"))
    with pytest.raises(launcher.LaunchError, match="existing artifact"):
        launcher.execute_plan(plan, launch=True)
    assert not Path(prepared["jobs"][0]["run_dir"]).exists()


@pytest.mark.parametrize("tokens", [["--gpu", "6"], ["--gpu=6"], ["--gp=6"],
                                   ["--run-dir", "/outside"], ["--run-d=/outside"], ["--"]])
def test_launcher_owned_flags_cannot_be_injected(plan, tokens):
    plan["jobs"][0]["argv"].extend(tokens)
    with pytest.raises(launcher.LaunchError):
        launcher.validate_plan(plan)


@pytest.mark.parametrize("reason", ["memory", "process", "dirty", "sha"])
def test_preflight_refuses_busy_or_changed_source(plan, system, monkeypatch, reason):
    if reason == "memory":
        system["devices"][0]["memory_used_mib"] = 1000.
    elif reason == "process":
        system["compute_processes"].append({"gpu_uuid": "GPU-0", "pid": 123, "process_name": "busy"})
    elif reason == "dirty":
        monkeypatch.setattr(launcher, "inspect_repository", lambda root: {"actual_git_sha": "a" * 40, "tracked_dirty": " M model.py"})
    else:
        plan["expected_git_sha"] = "b" * 7
    with pytest.raises(launcher.LaunchError):
        launcher.execute_plan(plan, launch=True)
    assert not (Path(plan["root"]) / "logs").exists()


def test_partial_failure_signals_only_groups_started_here(plan, system, monkeypatch):
    other = copy.deepcopy(plan["jobs"][0])
    other.update(name="second_s0", gpu=1)
    plan["jobs"].append(other)
    commands, signals = [], []

    class FakeProcess:
        pid = 12345
        returncode = None

        def poll(self):
            return None

        def wait(self, timeout):
            return -15

    def popen(command, **kwargs):
        commands.append((command, kwargs))
        if len(commands) == 2:
            raise OSError("simulated second spawn failure")
        return FakeProcess()

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    with pytest.raises(launcher.LaunchError, match="only newly started"):
        launcher.execute_plan(plan, launch=True)
    assert signals == [(12345, signal.SIGTERM)]
    assert commands[0][1]["start_new_session"] is True
    assert commands[0][1]["stdin"] == launcher.subprocess.DEVNULL
    assert commands[0][1]["env"]["OMP_NUM_THREADS"] == "4"
    assert commands[0][1]["env"]["MKL_NUM_THREADS"] == "4"
    report = json.loads((Path(plan["root"]) / "logs/motiondrive_v2/launch_paired_trial.json").read_text())
    assert report["status"] == "launch_failed"
    assert report["artifacts_preserved"] is True
    assert report["cleanup"][0]["pid"] == 12345
    assert (Path(plan["root"]) / "logs/motiondrive_v2/control_s0.log").exists()


def test_successful_launch_records_exact_child_arguments_and_clean_source(plan, system, monkeypatch):
    calls = []

    class FakeProcess:
        pid = 43210
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(launcher.subprocess, "Popen",
                        lambda command, **kwargs: calls.append((command, kwargs)) or FakeProcess())
    result = launcher.execute_plan(plan, launch=True)
    assert result["status"] == "launched"
    assert result["actual_git_sha"] == "a" * 40
    assert result["tracked_dirty"] == ""
    assert calls[0][0][-4:] == ["--gpu", "0", "--run-dir", result["jobs"][0]["run_dir"]]
    assert result["processes"][0]["process_group"] == 43210
    assert Path(result["launch_manifest"]).is_file()
