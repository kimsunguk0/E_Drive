import hashlib
import json
from pathlib import Path

import pytest

from scripts import supervise_motiondrive_v2_fixed_two_jobs as supervisor


UUIDS = sorted(supervisor.ALLOWED_GPU_UUIDS)


def spec(tmp_path: Path) -> dict:
    source = tmp_path / "source.py"
    source.write_text("frozen\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    jobs = []
    for name, uuid in zip(("zero", "provided"), UUIDS):
        jobs.append({
            "name": name,
            "gpu_uuid": uuid,
            "argv": ["/usr/bin/python", "frozen_runner.py"],
            "log": str(tmp_path / f"{name}.log"),
            "run_dir": str(tmp_path / f"{name}.run"),
            "result_paths": [str(tmp_path / f"{name}.result")],
        })
    return {"schema_version": 1, "experiment": "test",
            "source_pins": {str(source): source_sha}, "jobs": jobs}


def test_rejects_stale_expected_result(tmp_path):
    payload = spec(tmp_path)
    Path(payload["jobs"][0]["result_paths"][0]).write_text("stale\n")
    with pytest.raises(RuntimeError, match="result path must be fresh"):
        supervisor.validate_spec(payload, tmp_path, tmp_path / "receipt.json")


def test_second_spawn_failure_survives_first_child_exit_race(tmp_path, monkeypatch):
    payload = spec(tmp_path)
    spec_path, receipt = tmp_path / "spec.json", tmp_path / "receipt.json"
    spec_path.write_text(json.dumps(payload))

    class FirstProcess:
        pid = 424242
        returncode = None

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    calls = 0

    def fake_popen(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FirstProcess()
        raise OSError("injected second-spawn failure")

    snapshots = [{"index": str(i), "uuid": uuid, "name": "B200",
                  "memory_total_mib": "183359", "memory_used_mib": "0",
                  "memory_free_mib": "182632", "utilization_percent": "0"}
                 for i, uuid in enumerate(UUIDS)]
    monkeypatch.setattr(supervisor, "gpu_snapshot", lambda: snapshots)
    monkeypatch.setattr(supervisor.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(supervisor.os, "killpg",
                        lambda *args: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(supervisor.sys, "argv",
                        ["supervisor", "--spec", str(spec_path), "--receipt", str(receipt)])

    assert supervisor.main() == 1
    result = json.loads(receipt.read_text())
    assert result["status"] == "spawn_failed"
    assert result["jobs"]["provided"]["status"] == "spawn_failed"
    assert result["jobs"]["zero"]["status"] == "cleanup_after_peer_spawn_failure"
    assert result["jobs"]["zero"]["return_code"] == 0
    assert result["jobs"]["zero"]["results_complete"] is False
