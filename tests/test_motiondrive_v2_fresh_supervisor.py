"""CPU fixture tests for the P4 fresh-stage bridge and owned supervisor gates."""
import copy
import json
import os
from pathlib import Path
import signal
import time

import pytest
import torch

from models.motiondrive_v2 import MotionDriveV2Config
from scripts import run_motiondrive_v2_fresh_trial as fresh
from scripts import initialize_motiondrive_v2_public as public_init
from scripts import train_motiondrive_v2_query_adapter as query_trainer
from scripts.train_motiondrive_v2 import arguments as trainer_arguments


COMMIT = "a" * 40
ABSENT_PARENT, ABSENT_CHILD = 2147400101, 2147400102


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh, "ROOT", tmp_path)
    (tmp_path / "logs/motiondrive_v2").mkdir(parents=True)
    (tmp_path / "work_dirs/motiondrive_v2").mkdir(parents=True)
    data = {}
    for name in fresh.DATA_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic " + name)
        data[name] = fresh.sha256(path)
    monkeypatch.setattr(fresh, "DATA_FILES", data)
    sources = {"git_sha": COMMIT, "file_sha256": {"fixed-source.py": "b" * 64}}
    monkeypatch.setattr(fresh, "snapshot", lambda expected: copy.deepcopy(sources))
    config = MotionDriveV2Config(goal_on=False, state_on=False, motion_input_mode="low_feature", plan_output_scale=(10., 5.))
    payload = {"step": 0, "epoch": 0, "optimizer": {}, "model": {"fixture.weight": torch.ones(1)},
               "manifest": {"model_config": config.to_dict(), "split_sha256": next(iter(data.values())),
                            "seed": 0, "etri_optimizer_steps": 0, "public_checkpoint_sha256": fresh.PUBLIC_SHA,
                            "pretrained_sha256": fresh.PUBLIC_SHA}}
    initial = tmp_path / fresh.PUBLIC_INIT
    torch.save(payload, initial)
    return {"root": tmp_path, "sources": sources, "initial": initial,
            "initial_sha": fresh.sha256(initial), "payload": payload}


def request(tree, stage="canary", seed=0, gpu=4):
    init = fresh.initializer_path(stage, seed)
    return fresh.prepare(stage, seed, gpu, COMMIT, fresh.sha256(init))


def completed_artifacts(req, *, pid=ABSENT_CHILD, parent_pid=None):
    c = fresh.STAGES[req["stage"]]
    args = vars(trainer_arguments(req["trainer_arguments"][1:]))
    cfg = MotionDriveV2Config(goal_on=c["phase"] == "joint", state_on=c["phase"] == "joint",
                            motion_input_mode="low_feature", plan_output_scale=(10., 5.)).to_dict()
    initial_sha = "c" * 64
    manifest = {"pid": pid, "status": "completed", "step": c["steps"], "nonfinite_count": 0,
                "git_sha": COMMIT, "arguments": args, "model_config": cfg,
                "data_counts": {"train": c["train_n"], "eval": c["eval_n"]},
                "initial_model_state_sha256": initial_sha,
                "load_report": {"common_checkpoint_sha256": req["initializer_sha256"]},
                "split_sha256": next(iter(fresh.DATA_FILES.values())),
                "supervision_manifest_sha256": fresh.DATA_FILES[
                    "data/etri/motiondrive_v2/train_tune_geometry_v2/supervision_manifest.json"]}
    mapping = {"physical_gpu": req["physical_gpu"], "logical_device": "cuda:0",
               "observed_uuid": req.get("gpu_uuid", f"GPU-test-{req['physical_gpu']}"),
               "observed_uuid_raw": f"test-{req['physical_gpu']}",
               "cuda_visible_devices": req.get("gpu_uuid", f"GPU-test-{req['physical_gpu']}")}
    receipt = {"pid": pid, "parent_pid": os.getpid() if parent_pid is None else parent_pid,
               "initial_checkpoint_sha256": req["initializer_sha256"], "source": req["sources"],
               "initial_model_state_sha256": initial_sha, "device_mapping": mapping,
               "trainer_arguments": req["trainer_arguments"]}
    write_json(req["manifest_path"], manifest)
    write_json(req["child_receipt"], receipt)
    embedded = copy.deepcopy(manifest)
    embedded["status"] = "running"  # Real old trainer saves LAST before final sidecar.
    last = Path(req["run_dir"]) / "last.pth"
    torch.save({"step": c["steps"], "model": {"fixture.weight": torch.ones(1)},
                "optimizer": {"state": {0: {"step": torch.tensor(float(c["steps"]))}}}, "manifest": embedded}, last)
    return manifest, receipt, last


class TinyCheckpointModel(torch.nn.Module):
    """Small state inventory for exercising the real strict CPU checkpoint gate."""
    def __init__(self, _config):
        super().__init__()
        self.fixture = torch.nn.Linear(1, 1, bias=False)
        self.bn = torch.nn.BatchNorm1d(1)


def strict_checkpoint_artifacts(tree, monkeypatch):
    model = TinyCheckpointModel(None)
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    tree["payload"]["model"] = initial_state
    torch.save(tree["payload"], tree["initial"])
    tree["initial_sha"] = fresh.sha256(tree["initial"])
    req = request(tree)
    req["gpu_uuid"] = "GPU-test-4"
    manifest, _, last = completed_artifacts(req)
    trained_state = {name: value.detach().clone() for name, value in initial_state.items()}
    trained_state["fixture.weight"].add_(1.)
    optimizer = {"state": {0: {"step": torch.tensor(2.), "exp_avg": torch.zeros(1),
                                "exp_avg_sq": torch.ones(1)}}}
    embedded = copy.deepcopy(manifest)
    embedded["status"] = "running"
    torch.save({"step": 2, "model": trained_state, "optimizer": optimizer,
                "manifest": embedded}, last)
    monkeypatch.setattr("models.motiondrive_v2.MotionDriveV2", TinyCheckpointModel)
    return req, manifest, last


def install_prerequisite(tree, stage="canary", seed=0):
    """Filesystem completion evidence, with no process or CUDA creation."""
    req = request(tree, stage, seed)
    req["gpu_uuid"] = "GPU-test-4"
    _, _, last = completed_artifacts(req, parent_pid=ABSENT_PARENT)
    last_sha = fresh.sha256(last)
    evidence = {"manifest_sha256": fresh.sha256(req["manifest_path"]),
                "receipt_sha256": fresh.sha256(req["child_receipt"]), "last_sha256": last_sha,
                "checkpoint_validation": {"last_sha256": last_sha,
                    "full_model_strict_cpu_load": True, "optimizer_tensors_finite": True,
                    "optimizer_steps": fresh.STAGES[stage]["steps"], "model_forward_performed": False}}
    record = {"actual_returncode": 0, "supervisor_exit_code": 0, "outcome": "completed_cleanly",
              "pressure_event": None, "parent_pid": ABSENT_PARENT, "child_pid": ABSENT_CHILD,
              "request": req, "completion_evidence": evidence}
    write_json(req["record"], record)
    return req, record


@pytest.mark.parametrize("gpu", [4, 5])
def test_canary_prepare_strict_paths_single_logical_gpu_and_fixed_memory(tree, gpu):
    req = request(tree, gpu=gpu)
    assert req["physical_gpu"] == gpu and req["stage"] == "canary"
    assert req["initializer_sha256"] == tree["initial_sha"]
    args = trainer_arguments(req["trainer_arguments"][1:])
    assert args.gpu == 0 and args.batch == 16 and args.microbatch == 2 and args.eval_batch == 4
    assert args.steps == 2 and args.phase == "joint"
    assert args.cuda_memory_limit_mib == 12000 and args.cuda_min_free_mib == 8192
    assert args.time_input == "nominal" and args.bn_policy == "fixed"
    assert args.max_train_samples == 16 and args.max_eval_samples == 8
    assert args.resume is None and args.pretrained is None and not args.allow_unpretrained
    assert all(not Path(req[k]).exists() for k in ("record", "log", "run_dir", "child_receipt"))


@pytest.mark.parametrize("stage,seed,gpu", [("wrong",0,4),("canary",1,4),("pretrain",True,4),
                                           ("canary",0,0),("canary",0,6),("canary",0,7),("canary",0,4.)])
def test_unapproved_stage_seed_gpu_rejected(tree, stage, seed, gpu):
    with pytest.raises(ValueError):
        fresh.prepare(stage, seed, gpu, COMMIT, tree["initial_sha"])


@pytest.mark.parametrize("key", ["record", "log", "run_dir", "child_receipt"])
def test_existing_outputs_never_reused(tree, key):
    req = request(tree)
    target = Path(req[key])
    if key == "run_dir": target.mkdir()
    else: target.write_text("preserved")
    with pytest.raises(ValueError, match="Existing"):
        request(tree)
    assert target.is_dir() if key == "run_dir" else target.read_text() == "preserved"


@pytest.mark.parametrize("fault", ["init", "data", "source"])
def test_pinned_initializer_data_and_source_changes_rejected(tree, monkeypatch, fault):
    req = request(tree)
    if fault == "init": tree["initial"].write_bytes(b"changed")
    elif fault == "data": (tree["root"] / next(iter(fresh.DATA_FILES))).write_text("changed")
    else: monkeypatch.setattr(fresh, "snapshot", lambda *_: {"git_sha": "other"})
    with pytest.raises(ValueError):
        fresh.validate_sources_and_data(req)


@pytest.mark.parametrize("seed", [0, 1])
def test_pretrain_prerequisite_actual_exit_source_and_same_public_initialization(tree, seed):
    prior, _ = install_prerequisite(tree)
    req = request(tree, "pretrain", seed, 4 + seed)
    assert req["prerequisite"]["path"] == prior["record"]
    assert req["initializer_sha256"] == tree["initial_sha"]
    args = trainer_arguments(req["trainer_arguments"][1:])
    assert args.phase == "pretrain" and args.steps == 2000 and args.seed == seed
    assert args.max_train_samples == args.max_eval_samples == 0


@pytest.mark.parametrize("fault", ["rc", "pressure", "source", "manifest", "receipt", "init_sha",
                                    "last", "last_attestation"])
def test_prerequisite_failure_or_mutation_cannot_promote_next_stage(tree, fault):
    req, record = install_prerequisite(tree)
    if fault == "rc": record["actual_returncode"] = -11
    elif fault == "pressure": record["pressure_event"] = {"error": "pressure"}
    elif fault == "source": record["request"]["sources"] = {"file_sha256": {"changed": "x"}}
    elif fault == "manifest": Path(req["manifest_path"]).write_text("changed")
    elif fault == "receipt": Path(req["child_receipt"]).write_text("changed")
    elif fault == "init_sha": record["request"]["initializer_sha256"] = "f" * 64
    elif fault == "last": Path(req["run_dir"], "last.pth").write_bytes(b"changed")
    else: record["completion_evidence"]["checkpoint_validation"]["optimizer_steps"] = 1
    write_json(req["record"], record)
    with pytest.raises((ValueError, KeyError)):
        request(tree, "pretrain", 1, 5)


@pytest.mark.parametrize("fault", ["seed", "run_dir", "record", "initializer"])
def test_joint_requires_own_seed_pretrain_not_renamed_other_record(tree, fault):
    install_prerequisite(tree)
    prior, record = install_prerequisite(tree, "pretrain", 1)
    record_path = prior["record"]
    if fault == "seed": record["request"]["seed"] = 0
    elif fault == "run_dir": record["request"]["run_dir"] = str(tree["root"] / "other-run")
    elif fault == "record": record["request"]["record"] = str(tree["root"] / "other-record.json")
    else: record["request"]["initializer"] = str(tree["root"] / "other-init.pth")
    write_json(record_path, record)
    with pytest.raises(ValueError):
        request(tree, "joint", 1, 5)


def test_child_same_pid_bridge_keeps_exact_trainer_argv_and_cpu_safe_fixture(tree, monkeypatch):
    req = request(tree)
    req["gpu_uuid"] = "GPU-test-4"
    write_json(req["record"], {"child_pid": os.getpid(), "parent_pid": os.getppid(),
                              "status": "running", "request": req})
    mapping = {"physical_gpu": 4, "logical_device": "cuda:0", "observed_uuid": "GPU-test-4",
               "observed_uuid_raw": "test-4", "cuda_visible_devices": "GPU-test-4"}
    monkeypatch.setattr(query_trainer, "validate_cuda_namespace", lambda *_: mapping)
    monkeypatch.setattr(fresh.sys, "argv", list(fresh.sys.argv))
    calls = []
    def fake_run_path(path, run_name):
        calls.append((path, run_name, os.getpid(), list(fresh.sys.argv)))
    monkeypatch.setattr(fresh.runpy, "run_path", fake_run_path)
    initialized = torch.cuda.is_initialized()
    fresh.child_run(req)
    assert calls == [(str(tree["root"] / fresh.TRAINER), "__main__", os.getpid(), req["trainer_arguments"])]
    receipt = json.loads(Path(req["child_receipt"]).read_text())
    assert receipt["pid"] == os.getpid() and receipt["parent_pid"] == os.getppid()
    assert receipt["device_mapping"] == mapping and not receipt["optimizer_resume"]
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("fault", ["child_pid", "parent_pid", "status", "request", "device"])
def test_child_wrong_identity_or_device_cannot_enter_old_trainer(tree, monkeypatch, fault):
    req = request(tree)
    recorded = {**req, "gpu_uuid": "GPU-test-4"}
    record = {"child_pid": os.getpid(), "parent_pid": os.getppid(), "status": "running", "request": recorded}
    if fault in ("child_pid", "parent_pid"): record[fault] = -1
    elif fault == "status": record["status"] = "finished"
    elif fault == "request": record["request"]["seed"] = 1
    write_json(req["record"], record)
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    monkeypatch.setattr(query_trainer, "validate_cuda_namespace", lambda *_: {
        "physical_gpu": 5 if fault == "device" else 4, "observed_uuid": "GPU-test-4"})
    monkeypatch.setattr(fresh.runpy, "run_path", lambda *_a, **_k: pytest.fail("Old trainer must not run"))
    with pytest.raises(ValueError):
        fresh.child_run(req)
    assert not Path(req["child_receipt"]).exists()


@pytest.mark.parametrize("fault", [None, "status", "step", "pid", "config", "count", "seed", "cap",
                                   "receipt_pid", "receipt_parent", "receipt_init", "device", "initializer"])
def test_completion_external_manifest_receipt_and_strict_last_gate(tree, monkeypatch, fault):
    req = request(tree)
    req["gpu_uuid"] = "GPU-test-4"
    manifest, receipt, last = completed_artifacts(req)
    strict_calls = []
    def strict_gate(given, sidecar, checkpoint):
        strict_calls.append((given, sidecar, checkpoint))
        assert checkpoint == last
        embedded = torch.load(last, weights_only=True)
        assert embedded["manifest"]["status"] == "running"
        return {"synthetic_strict_gate": "passed"}
    monkeypatch.setattr(fresh, "validate_last_checkpoint", strict_gate)
    if fault == "status": manifest["status"] = "running"
    elif fault == "step": manifest["step"] = 1
    elif fault == "pid": manifest["pid"] += 1
    elif fault == "config": manifest["model_config"]["motion_input_mode"] = "legacy"
    elif fault == "count": manifest["data_counts"]["train"] = 15
    elif fault == "seed": manifest["arguments"]["seed"] = 1
    elif fault == "cap": manifest["arguments"]["cuda_memory_limit_mib"] = 0
    elif fault == "receipt_pid": receipt["pid"] += 1
    elif fault == "receipt_parent": receipt["parent_pid"] = -1
    elif fault == "receipt_init": receipt["initial_model_state_sha256"] = "f" * 64
    elif fault == "device": receipt["device_mapping"]["physical_gpu"] = 5
    elif fault == "initializer": manifest["load_report"]["common_checkpoint_sha256"] = "f" * 64
    write_json(req["manifest_path"], manifest)
    write_json(req["child_receipt"], receipt)
    if fault is None:
        result = fresh.validate_completion(req, ABSENT_CHILD)
        assert result["steps"] == 2 and result["last_sha256"] == fresh.sha256(last)
        assert len(strict_calls) == 1
    else:
        with pytest.raises(ValueError):
            fresh.validate_completion(req, ABSENT_CHILD)


def test_strict_last_failure_cannot_become_completed_cleanly(tree, monkeypatch):
    req = request(tree)
    req["gpu_uuid"] = "GPU-test-4"
    completed_artifacts(req)
    def invalid_last(*_):
        raise ValueError("Strict LAST rejected")
    monkeypatch.setattr(fresh, "validate_last_checkpoint", invalid_last)
    with pytest.raises(ValueError, match="Strict LAST"):
        fresh.validate_completion(req, ABSENT_CHILD)


def test_real_strict_last_cpu_gate_loads_complete_finite_model_and_fixed_bn(tree, monkeypatch):
    req, manifest, last = strict_checkpoint_artifacts(tree, monkeypatch)
    initialized = torch.cuda.is_initialized()
    result = fresh.validate_last_checkpoint(req, manifest, last)
    assert result["full_model_strict_cpu_load"] and result["optimizer_tensors_finite"]
    assert result["optimizer_steps"] == 2 and not result["model_forward_performed"]
    assert result["last_sha256"] == fresh.sha256(last)
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("fault", ["model_key", "model_dtype", "model_shape", "model_nonfinite",
                                    "bn_changed", "optimizer_missing", "optimizer_step",
                                    "optimizer_nonfinite", "lineage", "initializer_before",
                                    "last_during", "initializer_during"])
def test_real_strict_last_cpu_gate_rejects_tensor_optimizer_lineage_and_hash_faults(
        tree, monkeypatch, fault):
    req, manifest, last = strict_checkpoint_artifacts(tree, monkeypatch)
    saved = torch.load(last, map_location="cpu", weights_only=False)
    if fault == "model_key":
        saved["model"].pop("fixture.weight")
    elif fault == "model_dtype":
        saved["model"]["fixture.weight"] = saved["model"]["fixture.weight"].double()
    elif fault == "model_shape":
        saved["model"]["fixture.weight"] = torch.ones(2, 1)
    elif fault == "model_nonfinite":
        saved["model"]["fixture.weight"].fill_(float("nan"))
    elif fault == "bn_changed":
        saved["model"]["bn.running_mean"].add_(1.)
    elif fault == "optimizer_missing":
        saved["optimizer"]["state"] = {}
    elif fault == "optimizer_step":
        saved["optimizer"]["state"][0]["step"] = torch.tensor(1.)
    elif fault == "optimizer_nonfinite":
        saved["optimizer"]["state"][0]["exp_avg"].fill_(float("inf"))
    elif fault == "lineage":
        saved["manifest"]["pid"] += 1
    elif fault == "initializer_before":
        tree["initial"].write_bytes(b"changed")
    if fault not in ("initializer_before", "last_during", "initializer_during"):
        torch.save(saved, last)
    if fault in ("last_during", "initializer_during"):
        original = public_init.strict_cpu_state
        def mutate_during_validation(model, state):
            original(model, state)
            if fault == "last_during":
                last.write_bytes(last.read_bytes() + b"changed")
            else:
                payload = torch.load(tree["initial"], map_location="cpu", weights_only=False)
                payload["manifest"]["changed_during_validation"] = True
                torch.save(payload, tree["initial"])
        monkeypatch.setattr(public_init, "strict_cpu_state", mutate_during_validation)
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        fresh.validate_last_checkpoint(req, manifest, last)


@pytest.mark.parametrize("returncode,pressure", [(0,False),(-11,False),(0,True)])
def test_reused_supervisor_preserves_actual_os_exit_and_pressure(tree, returncode, pressure):
    req = request(tree)
    class Child:
        pid = ABSENT_CHILD
        def __init__(self): self.polls, self.signals = 0, []
        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else returncode
        def send_signal(self, sig): self.signals.append(sig)
    child, queries, calls, tick = Child(), [], [], [0.]
    def memory(gpu):
        queries.append(gpu)
        return {"physical_gpu": gpu, "gpu_uuid": f"GPU-test-{gpu}",
                "free_mib": 8191 if pressure and len(queries) > 1 else 180000}
    def validator(*_):
        calls.append(True)
        return {"fixture": "valid"}
    envs = []
    def launch(_argv, **kw):
        envs.append(kw["env"])
        return child
    result = fresh.supervise(req, query=memory, check_idle=lambda _: None, popen=launch,
        sleep=lambda n: tick.__setitem__(0, tick[0] + n), clock=lambda: tick[0], validator=validator,
        install_handlers=False)
    assert set(queries) == {4} and envs[0]["CUDA_VISIBLE_DEVICES"] == "GPU-test-4"
    assert result["actual_returncode"] == returncode
    assert result["outcome"] == ("completed_cleanly" if returncode == 0 and not pressure else "failed")
    assert bool(calls) is (returncode == 0 and not pressure)
    assert child.signals == ([signal.SIGTERM] if pressure else [])
