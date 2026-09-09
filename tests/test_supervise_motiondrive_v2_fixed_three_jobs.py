from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.supervise_motiondrive_v2_fixed_three_jobs as sup


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
        "experiment": "fixed_three_test",
        "source_pins": {sup.DRIVER: digest(driver)},
        "jobs": jobs,
    }
    path = root / "spec.json"
    path.write_text(json.dumps(spec))
    return path, spec


def snapshots():
    return [
        {"index": "0", "uuid": sup.GPU0, "name": "B200",
         "memory_total_mib": "183359", "memory_used_mib": "0",
         "memory_free_mib": "182632", "utilization_percent": "0"},
        {"index": "1", "uuid": sup.GPU1, "name": "B200",
         "memory_total_mib": "183359", "memory_used_mib": "0",
         "memory_free_mib": "182632", "utilization_percent": "0"},
    ]


def test_validate_exact_three_mapping_and_reject_stale_result(tmp_path, monkeypatch):
    spec_path, spec = make_spec(tmp_path)
    receipt = tmp_path / "receipt.json"
    jobs, pins = sup.validate_spec(spec, tmp_path, receipt)
    assert len(jobs) == 3
    assert {job["name"]: job["gpu_uuid"] for job in jobs} == sup.JOB_GPU
    assert str((tmp_path / sup.DRIVER).resolve()) in pins

    stale = tmp_path / "stale_result.json"
    spec["jobs"][0]["result_paths"] = [str(stale)]
    stale.write_text("stale")
    with pytest.raises(RuntimeError, match="must be fresh"):
        sup.validate_spec(spec, tmp_path, receipt)


def test_validate_rejects_gpu_mapping_and_duplicate_controlled_option(tmp_path):
    _path, spec = make_spec(tmp_path)
    receipt = tmp_path / "receipt.json"
    spec["jobs"][0]["gpu_uuid"] = sup.GPU1
    with pytest.raises(RuntimeError, match="mapping mismatch"):
        sup.validate_spec(spec, tmp_path, receipt)
    spec["jobs"][0]["gpu_uuid"] = sup.JOB_GPU[spec["jobs"][0]["name"]]
    spec["jobs"][0]["argv"] += ["--gpu", "0"]
    with pytest.raises(RuntimeError, match="Duplicate controlled option"):
        sup.validate_spec(spec, tmp_path, receipt)


class ImmediateProcess:
    next_pid = 41000

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


def test_run_records_three_clean_native_exits(tmp_path, monkeypatch):
    spec_path, _spec = make_spec(tmp_path)
    receipt = tmp_path / "receipt.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sup, "gpu_snapshot", snapshots)
    monkeypatch.setattr(sup.subprocess, "Popen", ImmediateProcess)
    monkeypatch.setattr(sup.time, "sleep", lambda _seconds: None)
    assert sup.run(spec_path, receipt) == 0
    result = json.loads(receipt.read_text())
    assert result["status"] == "completed"
    assert set(result["jobs"]) == set(sup.JOB_GPU)
    assert all(job["return_code"] == 0 and job["results_complete"]
               for job in result["jobs"].values())


class WaitingProcess:
    def __init__(self, argv, **_kwargs):
        self.argv = argv
        self.pid = 42000
        self.returncode = None

    def poll(self):
        return None

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def test_second_spawn_failure_cleans_only_owned_child_and_records_failure(
        tmp_path, monkeypatch):
    spec_path, _spec = make_spec(tmp_path)
    receipt = tmp_path / "receipt.json"
    calls = {"count": 0}

    def factory(argv, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected second spawn failure")
        return WaitingProcess(argv, **kwargs)

    killed = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sup, "gpu_snapshot", snapshots)
    monkeypatch.setattr(sup.subprocess, "Popen", factory)
    monkeypatch.setattr(sup.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    assert sup.run(spec_path, receipt) == 1
    result = json.loads(receipt.read_text())
    assert result["status"] == "spawn_failed"
    assert result["spawn_failure"]["job"] == "ordered_motion_residual"
    assert killed == [(42000, sup.signal.SIGTERM)]
    owned = result["jobs"]["continuation_control"]
    assert owned["termination_requested"] is True and owned["return_code"] == 0


def test_native_nonzero_exit_cannot_be_success(tmp_path, monkeypatch):
    spec_path, _spec = make_spec(tmp_path)
    receipt = tmp_path / "receipt.json"
    calls = {"count": 0}

    class MixedProcess(ImmediateProcess):
        def __init__(self, argv, **kwargs):
            super().__init__(argv, **kwargs)
            calls["count"] += 1
            if calls["count"] == 2:
                self.returncode = 7

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sup, "gpu_snapshot", snapshots)
    monkeypatch.setattr(sup.subprocess, "Popen", MixedProcess)
    monkeypatch.setattr(sup.time, "sleep", lambda _seconds: None)
    assert sup.run(spec_path, receipt) == 1
    result = json.loads(receipt.read_text())
    failed = result["jobs"]["ordered_motion_residual"]
    assert failed["return_code"] == 7
    assert failed["process_exit_status"] == "nonzero"
    assert result["status"] == "failed"
