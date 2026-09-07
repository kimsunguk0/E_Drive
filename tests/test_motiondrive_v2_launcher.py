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


@pytest.mark.parametrize("sharing", [False, True])
def test_partial_failure_signals_only_groups_started_here(plan, system, monkeypatch, sharing):
    if sharing:
        make_shared(plan, system)
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


def make_shared(plan, system):
    plan["gpu_sharing"] = {"mode": "shared", "allocator_limit_mib": 12000, "reserve_mib": 8192}
    for job in plan["jobs"]:
        job["argv"].extend(["--cuda-memory-limit-mib", "12000", "--cuda-min-free-mib", "8192"])
    for entry in system["devices"].values():
        entry.update(memory_total_mib=131072., memory_free_mib=22332., memory_used_mib=108740.)
    system["compute_processes"].append({"gpu_uuid": "GPU-0", "pid": 77, "process_name": "foreign_job"})


def test_shared_is_explicit_preserves_plan_flags_and_foreign_pid_evidence(plan, system):
    make_shared(plan, system)
    original = copy.deepcopy(plan)
    result = launcher.execute_plan(plan)
    assert plan == original
    assert result["gpu_sharing"] == original["gpu_sharing"]
    assert result["gpu_sharing_explicit"] is True
    assert result["jobs"][0]["explicit_argv"] == original["jobs"][0]["argv"]
    assert result["jobs"][0]["trainer_command"][1:-4] == original["jobs"][0]["argv"]
    evidence = result["gpu_memory_admission"][0]
    assert evidence["required_free_mib"] == 20192
    assert evidence["remaining_above_reserve_mib"] == 2140
    assert evidence["free_after_full_allocator_budget_mib"] == 10332
    assert evidence["existing_compute_processes"] == [{"gpu_uuid": "GPU-0", "pid": 77, "process_name": "foreign_job"}]
    assert not (Path(plan["root"]) / "logs").exists()


def test_default_and_explicit_idle_only_have_same_admission_and_commands(plan, system):
    default = launcher.execute_plan(plan)
    plan["gpu_sharing"] = {"mode": "idle_only"}
    explicit = launcher.execute_plan(plan)
    assert default["gpu_sharing"] == explicit["gpu_sharing"] == {"mode": "idle_only"}
    assert default["gpu_sharing_explicit"] is False
    assert explicit["gpu_sharing_explicit"] is True
    assert default["jobs"] == explicit["jobs"]
    assert default["gpu_memory_admission"] == explicit["gpu_memory_admission"] == []
    assert default["plan_json_sha256"] != explicit["plan_json_sha256"]


@pytest.mark.parametrize("policy", [None, True, "shared", {}, {"mode": "SHARED"},
    {"mode": "idle_only", "reserve_mib": 8192}, {"mode": "shared"},
    {"mode": "shared", "allocator_limit_mib": 12000, "reserve_mib": 8192, "override": True}])
def test_invalid_sharing_schema_refused(plan, policy):
    plan["gpu_sharing"] = policy
    with pytest.raises(launcher.LaunchError, match="gpu_sharing"):
        launcher.validate_plan(plan)


@pytest.mark.parametrize("field", ["allocator_limit_mib", "reserve_mib"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 12000., "12000", float("inf"), None])
def test_shared_budget_requires_positive_json_integer(plan, system, field, value):
    make_shared(plan, system)
    plan["gpu_sharing"][field] = value
    with pytest.raises(launcher.LaunchError, match="positive JSON integer"):
        launcher.validate_plan(plan)


@pytest.mark.parametrize("flag", ["--cuda-memory-limit-mib", "--cuda-min-free-mib"])
@pytest.mark.parametrize("variant", ["missing", "mismatch", "zero", "abbreviation", "duplicate", "two_values", "leading_zero"])
def test_shared_requires_exact_matching_trainer_limits(plan, system, flag, variant):
    make_shared(plan, system)
    argv = plan["jobs"][0]["argv"]
    index = argv.index(flag)
    if variant == "missing":
        del argv[index:index + 2]
    elif variant == "mismatch":
        argv[index + 1] = "1"
    elif variant == "zero":
        argv[index + 1] = "0"
    elif variant == "abbreviation":
        argv[index] = flag[:-1]
    elif variant == "duplicate":
        argv.extend([flag, argv[index + 1]])
    elif variant == "two_values":
        argv.insert(index + 2, "123")
    else:
        argv[index + 1] = "0" + argv[index + 1]
    with pytest.raises(launcher.LaunchError):
        launcher.validate_plan(plan)


def test_shared_equal_syntax_and_supervisor_keep_identical_trainer_argv(plan, system):
    make_shared(plan, system)
    plan["supervise"] = True
    (Path(plan["root"]) / launcher.SUPERVISOR).write_text("# supervisor fixture\n")
    argv = plan["jobs"][0]["argv"]
    argv[-4:] = ["--cuda-memory-limit-mib=12000", "--cuda-min-free-mib=8192"]
    result = launcher.execute_plan(plan)
    job = result["jobs"][0]
    separator = job["command"].index("--")
    assert job["command"][separator + 1:] == job["trainer_command"][1:]
    assert job["explicit_argv"] == argv


@pytest.mark.parametrize("field,value", [
    ("memory_free_mib", None), ("memory_free_mib", float("nan")),
    ("memory_free_mib", float("inf")), ("memory_free_mib", True),
    ("memory_free_mib", -1.), ("memory_total_mib", 0.),
    ("memory_total_mib", float("inf")), ("memory_total_mib", "131072"),
    ("memory_used_mib", -1.), ("memory_used_mib", 131073.),
    ("memory_free_mib", 131073.), ("memory_free_mib", 30000.),
    ("memory_free_mib", 20191.), ("memory_total_mib", 20191.),
])
def test_shared_refuses_unknown_inconsistent_or_insufficient_memory(plan, system, monkeypatch, field, value):
    make_shared(plan, system)
    system["devices"][0][field] = value
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("unsafe process spawn"))
    with pytest.raises(launcher.LaunchError):
        launcher.execute_plan(plan, launch=True)
    assert not (Path(plan["root"]) / "logs").exists()


def test_shared_exact_free_boundary_passes_but_every_selected_gpu_is_checked(plan, system):
    other = copy.deepcopy(plan["jobs"][0])
    other.update(name="second_s0", gpu=1)
    plan["jobs"].append(other)
    make_shared(plan, system)
    system["devices"][0]["memory_free_mib"] = 20192.
    system["devices"][1]["memory_free_mib"] = 20191.
    with pytest.raises(launcher.LaunchError, match="GPU 1 free memory"):
        launcher.execute_plan(plan)
    system["devices"][1]["memory_free_mib"] = 20192.
    result = launcher.execute_plan(plan)
    assert len(result["gpu_memory_admission"]) == 2
    assert all(row["remaining_above_reserve_mib"] == 0 for row in result["gpu_memory_admission"])


def test_shared_still_refuses_nonapproved_gpu_and_unmapped_compute_uuid(plan, system):
    make_shared(plan, system)
    plan["jobs"][0]["gpu"] = 4
    with pytest.raises(launcher.LaunchError, match="Only physical GPUs"):
        launcher.execute_plan(plan)
    plan["jobs"][0]["gpu"] = 0
    system["compute_processes"].append({"gpu_uuid": "UNKNOWN", "pid": 42, "process_name": "unknown"})
    with pytest.raises(launcher.LaunchError, match="UUID cannot be mapped"):
        launcher.execute_plan(plan)


def test_sharing_rechecks_free_memory_before_reservation(plan, system, monkeypatch):
    make_shared(plan, system)
    calls = []

    def inspect():
        snapshot = copy.deepcopy(system)
        calls.append(1)
        if len(calls) == 2:
            snapshot["devices"][0]["memory_free_mib"] = 20191.
        return snapshot

    monkeypatch.setattr(launcher, "inspect_gpus", inspect)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("unsafe process spawn"))
    with pytest.raises(launcher.LaunchError, match="free memory"):
        launcher.execute_plan(plan, launch=True)
    assert len(calls) == 2
    assert not (Path(plan["root"]) / "logs").exists()


def test_inspect_gpus_records_actual_free_total_and_processes(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        output = "0, GPU-0, 108740, 22332, 131072\n" if "--query-gpu=" in command[1] else "GPU-0, 77, foreign_job\n"
        return type("Result", (), {"stdout": output})()

    monkeypatch.setattr(launcher.subprocess, "run", run)
    data = launcher.inspect_gpus()
    assert data["devices"][0]["memory_free_mib"] == 22332.
    assert data["devices"][0]["memory_total_mib"] == 131072.
    assert data["compute_processes"][0]["pid"] == 77
    assert commands[0][1] == "--query-gpu=index,uuid,memory.used,memory.free,memory.total"


def test_shared_successful_spawn_keeps_both_exact_limits_and_lineage(plan, system, monkeypatch):
    make_shared(plan, system)
    commands = []

    class FakeProcess:
        pid = 43210
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(launcher.subprocess, "Popen",
                        lambda command, **kwargs: commands.append(command) or FakeProcess())
    monkeypatch.setattr(launcher.os, "killpg", lambda *a: pytest.fail("successful launch must not signal any process"))
    result = launcher.execute_plan(plan, launch=True)
    assert result["status"] == "launched"
    assert result["gpu_preflight"]["compute_processes"][-1]["pid"] == 77
    command = commands[0]
    assert command[command.index("--cuda-memory-limit-mib") + 1] == "12000"
    assert command[command.index("--cuda-min-free-mib") + 1] == "8192"
    saved = json.loads(Path(result["launch_manifest"]).read_text())
    assert saved["gpu_sharing"] == plan["gpu_sharing"]
    assert saved["plan_json_sha256"] == result["plan_json_sha256"]
    assert saved["gpu_memory_admission"][0]["existing_compute_processes"][0]["pid"] == 77


@pytest.mark.parametrize("reply", ["0, GPU-0, 100\n", "0, GPU-0, N/A, 20000, 131072\n",
                                     "0, GPU-0, 1, 2, 3\n0, GPU-0, 1, 2, 3\n"])
def test_inspect_gpus_rejects_incomplete_unparseable_duplicate_measurements(monkeypatch, reply):
    monkeypatch.setattr(launcher.subprocess, "run",
                        lambda *a, **k: type("Result", (), {"stdout": reply})())
    with pytest.raises(launcher.LaunchError):
        launcher.inspect_gpus()


def test_shared_policy_is_not_a_mutable_alias_and_changes_plan_hash(plan, system):
    make_shared(plan, system)
    prepared = launcher.validate_plan(plan)
    original_hash = prepared["plan_json_sha256"]
    plan["gpu_sharing"]["allocator_limit_mib"] = 12001
    assert prepared["gpu_sharing"]["allocator_limit_mib"] == 12000
    argv = plan["jobs"][0]["argv"]
    argv[argv.index("--cuda-memory-limit-mib") + 1] = "12001"
    changed = launcher.validate_plan(plan)
    assert changed["plan_json_sha256"] != original_hash
