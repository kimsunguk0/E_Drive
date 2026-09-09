from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.supervise_motiondrive_v2_long_training as sup


GPU_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
GPU_B = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_spec(root: Path, gpu_uuids=(GPU_A, GPU_A, GPU_B, GPU_B)) -> tuple[Path, dict]:
    driver = root / sup.DRIVER
    driver.parent.mkdir(parents=True)
    driver.write_text("# frozen test driver\n")
    jobs = []
    for name, gpu_uuid in zip(sup.JOB_NAMES, gpu_uuids, strict=True):
        run = root / "work_dirs" / name
        jobs.append({
            "name": name,
            "gpu_uuid": gpu_uuid,
            "argv": [
                "/usr/bin/python", sup.DRIVER,
                "--arm", name,
                "--seed", sup.ARM_SEEDS[name],
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
        "experiment": "motiondrive_v2_long_training",
        "source_pins": {sup.DRIVER: digest(driver)},
        "jobs": jobs,
    }
    path = root / "spec.json"
    path.write_text(json.dumps(spec))
    return path, spec


def snapshots(gpu_uuids, free_mib="182632", used_mib="0"):
    return [
        {"index": str(index), "uuid": uuid, "name": "B200",
         "memory_total_mib": "183359", "memory_used_mib": used_mib,
         "memory_free_mib": free_mib, "utilization_percent": "0"}
        for index, uuid in enumerate(dict.fromkeys(gpu_uuids))
    ]


def test_validate_exact_four_names_and_spec_supplied_gpu_bindings(tmp_path):
    spec_path, spec = make_spec(tmp_path)
    receipt = tmp_path / "receipt.json"
    jobs, pins = sup.validate_spec(spec, tmp_path, receipt)
    assert tuple(job["name"] for job in jobs) == sup.JOB_NAMES
    assert [job["gpu_uuid"] for job in jobs] == [GPU_A, GPU_A, GPU_B, GPU_B]
    assert str((tmp_path / sup.DRIVER).resolve()) in pins

    spec["jobs"].pop()
    with pytest.raises(RuntimeError, match="Exactly four jobs required"):
        sup.validate_spec(spec, tmp_path, receipt)
    assert spec_path.is_file()


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("arm", "long_s1", "Arm mismatch"),
        ("seed", "1", "Seed mismatch"),
        ("gpu", "1", "Logical GPU must be zero"),
        ("expected-physical-gpu-uuid", GPU_B,
         "Physical GPU UUID argv mismatch"),
    ],
)
def test_validate_rejects_job_name_seed_or_gpu_argv_disagreement(
        tmp_path, field, replacement, message):
    _path, spec = make_spec(tmp_path)
    argv = spec["jobs"][0]["argv"]
    argv[argv.index(f"--{field}") + 1] = replacement
    with pytest.raises(RuntimeError, match=message):
        sup.validate_spec(spec, tmp_path, tmp_path / "receipt.json")


def test_validate_rejects_job_gpu_uuid_disagreement(tmp_path):
    _path, spec = make_spec(tmp_path)
    spec["jobs"][0]["gpu_uuid"] = GPU_B
    with pytest.raises(RuntimeError, match="Physical GPU UUID argv mismatch"):
        sup.validate_spec(spec, tmp_path, tmp_path / "receipt.json")


class ImmediateProcess:
    next_pid = 61000
    launches = []

    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.pid = ImmediateProcess.next_pid
        ImmediateProcess.next_pid += 1
        self.returncode = 0
        ImmediateProcess.launches.append((argv, kwargs))
        run = Path(argv[argv.index("--run-dir") + 1])
        run.mkdir(parents=True)
        (run / "result.json").write_text("{}\n")

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


def test_run_records_process_groups_snapshots_results_and_aggregate_gate(
        tmp_path, monkeypatch):
    assigned = (GPU_A, GPU_A, GPU_B, GPU_B)
    spec_path, _spec = make_spec(tmp_path, assigned)
    receipt = tmp_path / "receipt.json"
    ImmediateProcess.launches = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sup, "gpu_snapshot", lambda requested: snapshots(requested))
    monkeypatch.setattr(sup.subprocess, "Popen", ImmediateProcess)
    monkeypatch.setattr(sup.time, "sleep", lambda _seconds: None)
    assert sup.run(spec_path, receipt) == 0
    result = json.loads(receipt.read_text())
    assert result["status"] == "completed"
    assert result["automatic_retry"] is False
    assert result["aggregate_memory_gate"] == {
        gpu: {"jobs": 2, "allocator_cap_mib_each": 12000,
              "reserve_mib": 8192, "required_free_mib": 32192}
        for gpu in (GPU_A, GPU_B)
    }
    assert result["gpu_snapshot_before"] and result["gpu_snapshot_after"]
    assert result["started_utc"] and result["ended_utc"]
    assert result["supervisor_source_sha256"] == digest(Path(sup.__file__))
    assert all(job["return_code"] == 0 and job["results_complete"]
               and next(iter(job["result_files"].values()))["sha256"]
               for job in result["jobs"].values())
    assert all(kwargs["start_new_session"] is True
               for _argv, kwargs in ImmediateProcess.launches)
    assert [kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            for _argv, kwargs in ImmediateProcess.launches] == list(assigned)


def test_shared_gpu_memory_gate_is_checked_before_any_launch(tmp_path, monkeypatch):
    assigned = (GPU_A, GPU_A, GPU_A, GPU_A)
    spec_path, _spec = make_spec(tmp_path, assigned)
    receipt = tmp_path / "receipt.json"
    constrained = snapshots((GPU_A,), free_mib="56191", used_mib="50000")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sup, "gpu_snapshot", lambda _requested: constrained)
    monkeypatch.setattr(
        sup.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("memory gate must precede launch"))
    with pytest.raises(RuntimeError, match="Insufficient aggregate guarded free memory"):
        sup.run(spec_path, receipt)
    assert not receipt.exists()


def test_shared_gpu_exact_memory_gate_passes(tmp_path, monkeypatch):
    assigned = (GPU_A, GPU_A, GPU_A, GPU_A)
    spec_path, _spec = make_spec(tmp_path, assigned)
    receipt = tmp_path / "receipt.json"
    exact = snapshots((GPU_A,), free_mib="56192", used_mib="50000")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sup, "gpu_snapshot", lambda _requested: exact)
    monkeypatch.setattr(sup.subprocess, "Popen", ImmediateProcess)
    monkeypatch.setattr(sup.time, "sleep", lambda _seconds: None)
    assert sup.run(spec_path, receipt) == 0
    result = json.loads(receipt.read_text())
    assert result["aggregate_memory_gate"] == {
        GPU_A: {"jobs": 4, "allocator_cap_mib_each": 12000,
                "reserve_mib": 8192, "required_free_mib": 56192}}


def test_nonzero_child_exit_is_recorded_as_actual_returncode(tmp_path, monkeypatch):
    spec_path, _spec = make_spec(tmp_path)
    receipt = tmp_path / "receipt.json"
    calls = {"count": 0}

    class MixedProcess(ImmediateProcess):
        def __init__(self, argv, **kwargs):
            super().__init__(argv, **kwargs)
            calls["count"] += 1
            if calls["count"] == 4:
                self.returncode = 9

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sup, "gpu_snapshot", lambda requested: snapshots(requested))
    monkeypatch.setattr(sup.subprocess, "Popen", MixedProcess)
    monkeypatch.setattr(sup.time, "sleep", lambda _seconds: None)
    assert sup.run(spec_path, receipt) == 1
    result = json.loads(receipt.read_text())
    failed = result["jobs"]["holdout_s1"]
    assert result["status"] == "failed"
    assert failed["return_code"] == 9
    assert failed["process_exit_status"] == "nonzero"
