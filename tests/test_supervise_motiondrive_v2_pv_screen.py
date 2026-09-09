from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.supervise_motiondrive_v2_pv_screen as sup


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_spec(root: Path) -> tuple[Path, dict]:
    driver = root / sup.DRIVER
    driver.parent.mkdir(parents=True)
    driver.write_text("# frozen test driver\n")
    jobs = []
    for name, gpu_uuid in sup.JOB_GPU.items():
        run = root / "work_dirs" / name
        jobs.append({
            "name": name,
            "gpu_uuid": gpu_uuid,
            "argv": [
                "/usr/bin/python", sup.DRIVER,
                "--arm", name,
                "--gpu", "0",
                "--run-dir", str(run),
                "--expected-physical-gpu-uuid", gpu_uuid,
                "--cuda-memory-limit-mib", "12000",
                "--cuda-min-free-mib", "8192",
            ],
            "log": str(root / "logs" / f"{name}.log"),
            "run_dir": str(run),
            "result_paths": [str(run / "result.json")],
        })
    spec = {
        "schema_version": 1,
        "experiment": "pv_fixed_three_test",
        "source_pins": {sup.DRIVER: digest(driver)},
        "jobs": jobs,
    }
    path = root / "spec.json"
    path.write_text(json.dumps(spec))
    return path, spec


def snapshots():
    return [
        {"index": str(index), "uuid": uuid, "name": "B200",
         "memory_total_mib": "183359", "memory_used_mib": "0",
         "memory_free_mib": "182632", "utilization_percent": "0"}
        for index, uuid in ((0, sup.GPU0), (1, sup.GPU1), (5, sup.GPU5))
    ]


def test_validate_exact_three_mapping_and_rejects_gpu_drift(tmp_path):
    spec_path, spec = make_spec(tmp_path)
    receipt = tmp_path / "receipt.json"
    jobs, pins = sup.validate_spec(spec, tmp_path, receipt)
    assert {job["name"]: job["gpu_uuid"] for job in jobs} == sup.JOB_GPU
    assert str((tmp_path / sup.DRIVER).resolve()) in pins

    spec["jobs"][0]["gpu_uuid"] = sup.GPU5
    with pytest.raises(RuntimeError, match="mapping mismatch"):
        sup.validate_spec(spec, tmp_path, receipt)
    assert spec_path.is_file()


def test_validate_rejects_stale_result_and_duplicate_option(tmp_path):
    _path, spec = make_spec(tmp_path)
    receipt = tmp_path / "receipt.json"
    stale = tmp_path / "stale.json"
    stale.write_text("{}\n")
    spec["jobs"][0]["result_paths"] = [str(stale)]
    with pytest.raises(RuntimeError, match="must be fresh"):
        sup.validate_spec(spec, tmp_path, receipt)

    spec["jobs"][0]["result_paths"] = [str(tmp_path / "fresh.json")]
    spec["jobs"][0]["argv"] += ["--gpu", "0"]
    with pytest.raises(RuntimeError, match="Duplicate controlled option"):
        sup.validate_spec(spec, tmp_path, receipt)


class ImmediateProcess:
    next_pid = 51000

    def __init__(self, argv, **_kwargs):
        self.argv = argv
        self.pid = ImmediateProcess.next_pid
        ImmediateProcess.next_pid += 1
        self.returncode = 0
        run = Path(argv[argv.index("--run-dir") + 1])
        run.mkdir(parents=True)
        (run / "result.json").write_text("{}\n")

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


def test_run_records_three_clean_exits_and_distinct_gpu_gates(tmp_path, monkeypatch):
    spec_path, _spec = make_spec(tmp_path)
    receipt = tmp_path / "receipt.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sup, "gpu_snapshot", snapshots)
    monkeypatch.setattr(sup.subprocess, "Popen", ImmediateProcess)
    monkeypatch.setattr(sup.time, "sleep", lambda _seconds: None)
    assert sup.run(spec_path, receipt) == 0
    result = json.loads(receipt.read_text())
    assert result["status"] == "completed"
    assert set(result["aggregate_memory_gate"]) == sup.APPROVED_GPUS
    assert all(gate == {"jobs": 1, "allocator_cap_mib_each": 12000,
                        "reserve_mib": 8192, "required_free_mib": 20192}
               for gate in result["aggregate_memory_gate"].values())
    assert all(job["return_code"] == 0 and job["results_complete"]
               for job in result["jobs"].values())


def test_shared_gpu_with_foreign_usage_passes_when_free_memory_is_sufficient(
        tmp_path, monkeypatch):
    spec_path, _spec = make_spec(tmp_path)
    receipt = tmp_path / "receipt.json"
    shared = snapshots()
    for row in shared:
        if row["uuid"] in (sup.GPU0, sup.GPU1):
            row["memory_used_mib"] = "32400"
            row["memory_free_mib"] = "150000"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sup, "gpu_snapshot", lambda: shared)
    monkeypatch.setattr(sup.subprocess, "Popen", ImmediateProcess)
    monkeypatch.setattr(sup.time, "sleep", lambda _seconds: None)
    assert sup.run(spec_path, receipt) == 0
    result = json.loads(receipt.read_text())
    before = {row["uuid"]: row for row in result["gpu_snapshot_before"]}
    assert before[sup.GPU0]["memory_used_mib"] == "32400"
    assert before[sup.GPU1]["memory_free_mib"] == "150000"


def test_shared_gpu_rejects_insufficient_free_memory(tmp_path, monkeypatch):
    spec_path, _spec = make_spec(tmp_path)
    receipt = tmp_path / "receipt.json"
    constrained = snapshots()
    for row in constrained:
        if row["uuid"] == sup.GPU0:
            row["memory_used_mib"] = "32400"
            row["memory_free_mib"] = "20191"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sup, "gpu_snapshot", lambda: constrained)
    with pytest.raises(RuntimeError, match="Insufficient aggregate guarded free memory"):
        sup.run(spec_path, receipt)
    assert not receipt.exists()


class WaitingProcess:
    def __init__(self, argv, **_kwargs):
        self.argv = argv
        self.pid = 52000
        self.returncode = None

    def poll(self):
        return None

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def test_partial_spawn_failure_cleans_only_owned_process(tmp_path, monkeypatch):
    spec_path, _spec = make_spec(tmp_path)
    receipt = tmp_path / "receipt.json"
    calls = {"count": 0}

    def factory(argv, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected spawn failure")
        return WaitingProcess(argv, **kwargs)

    killed = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sup, "gpu_snapshot", snapshots)
    monkeypatch.setattr(sup.subprocess, "Popen", factory)
    monkeypatch.setattr(sup.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    assert sup.run(spec_path, receipt) == 1
    result = json.loads(receipt.read_text())
    assert result["status"] == "spawn_failed"
    assert result["spawn_failure"]["job"] == "pv"
    assert killed == [(52000, sup.signal.SIGTERM)]
    assert result["jobs"]["direct"]["termination_requested"] is True


def test_nonzero_child_exit_cannot_be_success(tmp_path, monkeypatch):
    spec_path, _spec = make_spec(tmp_path)
    receipt = tmp_path / "receipt.json"
    calls = {"count": 0}

    class MixedProcess(ImmediateProcess):
        def __init__(self, argv, **kwargs):
            super().__init__(argv, **kwargs)
            calls["count"] += 1
            if calls["count"] == 3:
                self.returncode = 9

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sup, "gpu_snapshot", snapshots)
    monkeypatch.setattr(sup.subprocess, "Popen", MixedProcess)
    monkeypatch.setattr(sup.time, "sleep", lambda _seconds: None)
    assert sup.run(spec_path, receipt) == 1
    result = json.loads(receipt.read_text())
    assert result["status"] == "failed"
    assert result["jobs"]["pv_residual"]["return_code"] == 9
