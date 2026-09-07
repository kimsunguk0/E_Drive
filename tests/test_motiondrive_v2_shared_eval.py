"""공유 GPU 평가 wrapper CPU 모의 시험. 실제 CUDA/학습/평가는 실행하지 않는다."""
import copy
import json
from pathlib import Path
import signal
import sys

import pytest
import torch

from scripts import evaluate_motiondrive_v2_shared as shared
from scripts import analyze_motiondrive_v2_p2 as p2
from scripts.evaluate_motiondrive_v2_planning import arguments as eval_arguments, motion_prediction_contract


@pytest.fixture
def eval_request(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "reports").mkdir(parents=True)
    (root / "logs/motiondrive_v2").mkdir(parents=True)
    monkeypatch.setattr(shared, "ROOT", root)
    output = root / "reports/p2_c0t0_last3000_motion.json"
    record = root / "logs/motiondrive_v2/p2_c0t0_eval.execution.json"
    return {"schema_version": 1, "root": str(root), "arm": "c0t0", "gpu": 0,
            "output": str(output), "protocol": str(output.with_suffix(".protocol.json")),
            "record": str(record), "child_receipt": str(record.with_suffix(".child.json")),
            "checkpoint_sha256": "a" * 64, "expected_evaluation_git_sha": "b" * 40,
            "training": {"selected_checkpoint_sha256": "a" * 64, "eval_rows_sha256": "c" * 64,
                         "input_file_sha256": {}},
            "sources": {"evaluation": {"git_sha": "b" * 40, "tracked_changes": [], "file_sha256": {shared.EVALUATOR: "d" * 64}}},
            "evaluation_argv": shared.evaluation_argv(root, "c0t0", output),
            "memory_policy": {"allocator_limit_mib": 12000, "reserve_mib": 8192, "preflight_free_mib": 20192,
                              "pressure_poll_seconds": 5., "safety_grace_seconds": 30.},
            "gpu_uuid_mapping": {str(i): f"GPU-synthetic-{i}" for i in range(4)}}


class FakeChild:
    pid = 987654
    def __init__(self, codes):
        self.codes = iter(codes)
        self.last = None
        self.signals, self.kill_calls = [], 0
    def poll(self):
        self.last = next(self.codes, self.last)
        return self.last
    def send_signal(self, sig):
        self.signals.append(sig)
    def kill(self):
        self.kill_calls += 1


def memory(gpu, free=25000):
    return {"physical_gpu": gpu, "gpu_uuid": f"GPU-synthetic-{gpu}", "free_mib": free}


def supervise_mock(eval_request, child, runtime_free=25000, runtime_error=None, result_valid=True):
    tick, calls = [0.], []
    queries = [0]
    preflight_queries = 2 if shared.request_gpu_selection(eval_request)["physical_gpu_explicit"] else 5
    def query(gpu):
        queries[0] += 1
        if queries[0] > preflight_queries:
            if runtime_error is not None:
                raise runtime_error
            return memory(gpu, runtime_free)
        return memory(gpu)
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return child
    def sleep(seconds):
        tick[0] += seconds
    def inspect(*args):
        if not result_valid:
            raise ValueError("합성 결과 검증 실패")
        return {"synthetic_result_verified": True}
    result = shared.supervise(eval_request, process_factory=popen, gpu_query=query, sleeper=sleep,
                              monotonic=lambda: tick[0], result_inspector=inspect, install_signal_handlers=False)
    return result, calls, tick[0]


@pytest.mark.parametrize("arm,gpu,edition,timing", [("c0t0", 0, "train_tune_rawtime", "raw"),
    ("c1t0", 1, "train_tune_geometry_v2", "raw"), ("c0t1", 2, "train_tune_rawtime", "nominal"),
    ("c1t1", 3, "train_tune_geometry_v2", "nominal")])
def test_four_arm_exact_fixed_argv(arm, gpu, edition, timing):
    argv = shared.evaluation_argv("/project", arm, "/project/reports/new.json")
    args = eval_arguments(argv)
    assert args.device == f"cuda:{gpu}" and args.time_input == timing
    assert Path(args.supervision_root).name == edition
    assert args.checkpoint == f"/project/work_dirs/motiondrive_v2/p2_{arm}_s0/last.pth"
    assert (args.batch, args.workers, args.frame_stride, args.max_samples, args.scenes) == (4, 4, 5, 0, None)
    assert args.precision == "bf16" and args.split == "tune"
    assert args.conditions == shared.CONDITIONS and args.include_motion_predictions is True


def opt_in_request(request, physical_gpu, arm="c0t0"):
    result = copy.deepcopy(request)
    result.update(arm=arm, gpu=shared.ARMS[arm][0], **shared.gpu_selection(arm, physical_gpu))
    result["evaluation_argv"] = shared.evaluation_argv(result["root"], arm, result["output"], physical_gpu)
    result["gpu_uuid_mapping"] = {str(physical_gpu): f"GPU-synthetic-{physical_gpu}"}
    return result


@pytest.mark.parametrize("arm,physical", [("c1t1", 4), ("c0t1", 5), ("c0t0", 4), ("c0t0", 5)])
def test_explicit_physical_gpu_queries_only_selected_and_exposes_one_uuid(eval_request, arm, physical):
    request = opt_in_request(eval_request, physical, arm)
    shared.validate_request(request)
    queried, spawned = [], []
    child = FakeChild([None, 0])
    def query(gpu):
        queried.append(gpu)
        return memory(gpu)
    def popen(command, **kwargs):
        spawned.append((command, kwargs))
        return child
    result = shared.supervise(request, gpu_query=query, process_factory=popen, sleeper=lambda _: None,
        result_inspector=lambda *_: {}, install_signal_handlers=False)
    assert queried == [physical, physical, physical]
    assert spawned[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == f"GPU-synthetic-{physical}"
    assert result["training_gpu"] == shared.ARMS[arm][0] and result["physical_gpu"] == physical
    assert result["logical_device"] == "cuda:0" and result["physical_gpu_explicit"] is True
    argv = result["request"]["evaluation_argv"]
    args = eval_arguments(argv)
    assert args.device == "cuda:0" and args.checkpoint.endswith(f"p2_{arm}_s0/last.pth")
    assert args.batch == 4 and args.time_input == shared.ARMS[arm][2]
    assert result["request"]["memory_policy"] == eval_request["memory_policy"]
    assert result["outcome"] == "completed_cleanly"


@pytest.mark.parametrize("physical", [0, 1, 2, 3, 4, 5])
def test_every_explicit_gpu_uses_logical_zero_and_cpu_mock_verifies_uuid(eval_request, monkeypatch, physical):
    request = opt_in_request(eval_request, physical, "c1t1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"GPU-synthetic-{physical}")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    seen = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: seen.append(str(device)))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device:
                        type("Properties", (), {"uuid": f"GPU-synthetic-{physical}"})())
    result = shared.verify_child_device(request, torch.device("cuda:0"))
    assert seen == ["cuda:0"] and result["physical_gpu"] == physical and result["training_gpu"] == 3
    with pytest.raises(ValueError, match="logical"):
        shared.verify_child_device(request, torch.device("cuda:1"))


@pytest.mark.parametrize("physical", [6, 7, -1, True, 4., "4"])
def test_invalid_physical_gpu_rejected_before_query_or_namespace(eval_request, physical):
    with pytest.raises(ValueError):
        shared.gpu_selection("c1t1", physical)
    with pytest.raises(ValueError):
        shared.checked_memory(lambda *_: pytest.fail("금지 GPU 조회"), physical)


@pytest.mark.parametrize("value", ["6", "7", "-1", "4.0"])
def test_cli_forbids_gpu6_7_and_noninteger_physical_ids(value):
    with pytest.raises(SystemExit):
        shared.arguments(["--physical-gpu", value])


def test_child_cli_cannot_mix_physical_gpu_option():
    with pytest.raises(ValueError, match="child"):
        shared.arguments(["--child-record", "/record.json", "--expected-request-sha256", "a" * 64,
                          "--physical-gpu", "4"])


@pytest.mark.parametrize("field,value", [("physical_gpu", 6), ("physical_gpu_explicit", False),
    ("logical_device", "cuda:4"), ("training_gpu", 4), ("cuda_namespace", "legacy_four_uuid")])
def test_optin_request_cannot_silently_relabel_gpu_or_training(eval_request, field, value):
    request = opt_in_request(eval_request, 4, "c1t1")
    request[field] = value
    with pytest.raises(ValueError):
        shared.validate_request(request)


def test_partial_selection_metadata_and_extra_visible_gpu_rejected(eval_request):
    eval_request["physical_gpu"] = 4
    with pytest.raises(ValueError, match="metadata"):
        shared.validate_request(eval_request)
    request = opt_in_request(eval_request, 4)
    request["gpu_uuid_mapping"]["5"] = "GPU-synthetic-5"
    with pytest.raises(ValueError, match="UUID"):
        shared.validate_request(request)


@pytest.mark.parametrize("physical", [4, 5])
def test_new_physical_memory_query_is_exact_and_timed(monkeypatch, physical):
    commands = []
    def run(command, **kwargs):
        commands.append((command, kwargs))
        return type("Result", (), {"stdout": f"{physical}, GPU-synthetic-{physical}, 25000\n"})()
    monkeypatch.setattr(shared.subprocess, "run", run)
    result = shared.query_evaluation_memory(physical)
    assert result["physical_gpu"] == physical and result["free_mib"] == 25000
    assert commands[0][0][:3] == ["nvidia-smi", "-i", str(physical)]
    assert commands[0][1]["timeout"] == 5 and commands[0][1]["check"] is True


@pytest.mark.parametrize("stdout", ["5, GPU-other, 25000\n", "4, GPU-x, nan\n", "4, GPU-x, -1\n", "", "4, GPU-x\n"])
def test_physical4_memory_query_malformed_or_wrong_identity_fails(monkeypatch, stdout):
    monkeypatch.setattr(shared.subprocess, "run", lambda *_a, **_k: type("Result", (), {"stdout": stdout})())
    with pytest.raises(ValueError):
        shared.query_evaluation_memory(4)


@pytest.mark.parametrize("physical", [4, 5])
@pytest.mark.parametrize("failure", ["pressure", "query", "sigsegv"])
def test_optin_preserves_pressure_query_and_nonzero_exit_failures(eval_request, physical, failure):
    request = opt_in_request(eval_request, physical, "c1t1")
    child = FakeChild([None, -11 if failure == "sigsegv" else 0])
    result, _, _ = supervise_mock(request, child,
        runtime_free=8191 if failure == "pressure" else 25000,
        runtime_error=RuntimeError("조회 실패") if failure == "query" else None)
    assert result["wrapper_exit_code"] != 0
    if failure == "sigsegv":
        assert result["actual_returncode"] == -11 and child.signals == []
    else:
        assert result["outcome"] == "memory_pressure_failed" and child.signals == [signal.SIGTERM]


def test_physical_optin_cannot_bypass_original_arm_training_gate(eval_request, monkeypatch):
    observed = []
    def failed_training(root, arm, launch, expected):
        observed.append(arm)
        raise ValueError("원 학습 OS exit -11; 평가 금지")
    monkeypatch.setattr(shared, "verify_training", failed_training)
    with pytest.raises(ValueError, match="OS exit -11"):
        shared.prepare(eval_request["root"], "c1t0", eval_request["output"], eval_request["record"],
                       "launch.json", "a" * 64, "b" * 40, physical_gpu=4)
    assert observed == ["c1t0"]


@pytest.mark.parametrize("change", ["batch", "conditions", "gpu", "cap", "root", "checkpoint_sha", "receipt", "duplicate_uuid"])
def test_request_rejects_uncontrolled_overrides(eval_request, change):
    if change == "batch": eval_request["evaluation_argv"][eval_request["evaluation_argv"].index("--batch") + 1] = "2"
    elif change == "conditions": eval_request["evaluation_argv"].remove("reverse_history")
    elif change == "gpu": eval_request["gpu"] = 4
    elif change == "cap": eval_request["memory_policy"]["allocator_limit_mib"] = 0
    elif change == "root": eval_request["root"] = "/"
    elif change == "checkpoint_sha": eval_request["checkpoint_sha256"] = "bad"
    elif change == "receipt": eval_request["child_receipt"] = eval_request["output"]
    else: eval_request["gpu_uuid_mapping"]["1"] = eval_request["gpu_uuid_mapping"]["0"]
    with pytest.raises(ValueError):
        shared.validate_request(eval_request)


@pytest.mark.parametrize("extra", [["--gpu", "4"], ["--batch", "2"], ["--precision", "fp32"], ["--conditions", "normal"]])
def test_cli_does_not_expose_gpu_batch_precision_or_conditions(extra):
    with pytest.raises(SystemExit):
        shared.arguments(extra)


@pytest.mark.parametrize("free", [float("nan"), float("inf"), -1., "25000", True])
def test_invalid_free_memory_fails_closed(free):
    with pytest.raises(ValueError):
        shared.checked_memory(lambda gpu: memory(gpu, free), 0)


def test_admission_exact_20192_mib_and_no_spawn_before_capacity(eval_request):
    assert shared.checked_memory(lambda gpu: memory(gpu, 20192), 0, admission=True)["free_mib"] == 20192
    with pytest.raises(ValueError, match="20192"):
        shared.supervise(eval_request, gpu_query=lambda gpu: memory(gpu, 20191),
                         process_factory=lambda *_a, **_k: pytest.fail("실행 금지"), install_signal_handlers=False)
    assert not Path(eval_request["record"]).exists()


@pytest.mark.parametrize("code,outcome", [(0, "completed_cleanly"), (2, "child_nonzero_exit"), (-11, "child_signaled")])
def test_os_exit_separate_from_result_and_only_owned_process(eval_request, code, outcome, monkeypatch):
    monkeypatch.setattr(shared.os, "kill", lambda *_a: pytest.fail("임의 PID 신호 금지"))
    monkeypatch.setattr(shared.os, "killpg", lambda *_a: pytest.fail("process group 신호 금지"))
    child = FakeChild([None, code])
    result, calls, _ = supervise_mock(eval_request, child)
    assert result["actual_returncode"] == code and result["outcome"] == outcome
    assert child.signals == [] and child.kill_calls == 0
    assert calls[0][1]["start_new_session"] is False and calls[0][1]["stdin"] is shared.subprocess.DEVNULL
    env = calls[0][1]["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == ",".join(f"GPU-synthetic-{i}" for i in range(4))
    assert calls[0][0][0] == sys.executable
    assert "--child-record" in calls[0][0]
    assert "env" not in result  # 전체 환경/비밀을 기록하지 않는다.


@pytest.mark.parametrize("failure", ["pressure", "query"])
def test_pressure_or_query_failure_only_terminates_owned_child_and_zero_is_not_success(eval_request, failure, monkeypatch):
    monkeypatch.setattr(shared.os, "kill", lambda *_a: pytest.fail("임의 PID 신호 금지"))
    monkeypatch.setattr(shared.os, "killpg", lambda *_a: pytest.fail("process group 신호 금지"))
    child = FakeChild([None, 0])
    result, _, _ = supervise_mock(eval_request, child, runtime_free=8191 if failure == "pressure" else 25000,
                                  runtime_error=RuntimeError("조회 실패") if failure == "query" else None)
    assert child.signals == [signal.SIGTERM] and child.kill_calls == 0
    assert result["actual_returncode"] == 0 and result["outcome"] == "memory_pressure_failed"
    assert result["wrapper_exit_code"] != 0 and result["pressure_event"]["child_pid"] == child.pid


def test_pressure_escalates_after_30_seconds_only_once(eval_request):
    child = FakeChild([None] * 34 + [-9])
    result, _, elapsed = supervise_mock(eval_request, child, runtime_free=0)
    assert child.signals == [signal.SIGTERM] and child.kill_calls == 1
    assert elapsed >= 30. and result["actual_returncode"] == -9
    assert result["pressure_event"]["kill_sent"] is True


def test_zero_exit_without_verified_report_is_not_success(eval_request):
    result, _, _ = supervise_mock(eval_request, FakeChild([0]), result_valid=False)
    assert result["outcome"] == "zero_exit_without_verified_result"
    assert result["wrapper_exit_code"] == 1 and "result_validation_error" in result


def test_existing_execution_record_is_never_overwritten(eval_request):
    record = Path(eval_request["record"])
    record.write_text("원본 보존", encoding="utf-8")
    with pytest.raises(FileExistsError):
        supervise_mock(eval_request, FakeChild([0]))
    assert record.read_text(encoding="utf-8") == "원본 보존"


def test_spawn_failure_has_no_fake_os_exit(eval_request):
    def fail(*args, **kwargs):
        raise OSError("합성 spawn 실패")
    result = shared.supervise(eval_request, process_factory=fail, gpu_query=memory, install_signal_handlers=False)
    assert result["status"] == "spawn_failed" and result["actual_returncode"] is None
    assert result["wrapper_exit_code"] == 125


@pytest.mark.parametrize("hijack", [False, True])
def test_parent_exception_reaps_exact_child_and_preserves_record_ownership(eval_request, monkeypatch, capsys, hijack):
    monkeypatch.setattr(shared.os, "kill", lambda *_a: pytest.fail("임의 PID 신호 금지"))
    monkeypatch.setattr(shared.os, "killpg", lambda *_a: pytest.fail("process group 신호 금지"))
    child = FakeChild([None] * 37 + [-9])
    tick, sleep_calls = [0.], [0]
    hijacked_bytes = []
    def popen(*_a, **_k):
        if hijack:
            path = Path(eval_request["record"])
            record = json.loads(path.read_text())
            record["supervisor_record_id"] = "다른 소유자"
            path.write_text(json.dumps(record))
            hijacked_bytes.append(path.read_bytes())
        return child
    def sleeper(seconds):
        sleep_calls[0] += 1
        if not hijack and sleep_calls[0] == 1:
            raise RuntimeError("합성 parent 루프 오류")
        tick[0] += seconds
    with pytest.raises((ValueError, RuntimeError)):
        shared.supervise(eval_request, process_factory=popen, gpu_query=memory, sleeper=sleeper,
                         monotonic=lambda: tick[0], install_signal_handlers=False)
    assert child.signals == [signal.SIGTERM] and child.kill_calls == 1
    assert child.poll() == -9 and tick[0] >= 30
    if hijack:
        assert Path(eval_request["record"]).read_bytes() == hijacked_bytes[0]
        evidence = json.loads(capsys.readouterr().err)
        assert evidence["record_not_overwritten"] is True
        assert evidence["parent_failure"]["actual_returncode"] == -9
    else:
        record = json.loads(Path(eval_request["record"]).read_text())
        assert record["status"] == "parent_failed" and record["actual_returncode"] == -9
        assert record["parent_failure"]["kill_sent"] is True


def test_cleanup_signal_and_sleeper_errors_do_not_skip_reaping(eval_request, monkeypatch):
    child = FakeChild([None] * 36 + [-9])
    tick, sleeps = [0.], [0]
    def send_signal(sig):
        child.signals.append(sig)
        raise OSError("합성 SIGTERM 오류")
    child.send_signal = send_signal
    def sleeper(seconds):
        sleeps[0] += 1
        if sleeps[0] <= 2:
            raise RuntimeError("합성 sleeper 오류")
        tick[0] += seconds
    monkeypatch.setattr(shared.time, "sleep", lambda seconds: tick.__setitem__(0, tick[0] + seconds))
    with pytest.raises(RuntimeError, match="sleeper 오류"):
        shared.supervise(eval_request, process_factory=lambda *_a, **_k: child, gpu_query=memory,
                         sleeper=sleeper, monotonic=lambda: tick[0], install_signal_handlers=False)
    record = json.loads(Path(eval_request["record"]).read_text())
    failure = record["parent_failure"]
    assert failure["actual_returncode"] == -9 and child.kill_calls == 1
    assert {e["stage"] for e in failure["cleanup_errors"]} == {"sigterm", "sleep"}


def test_cap_and_device_verification_precede_existing_main_without_cuda(eval_request):
    events = []
    initialized = torch.cuda.is_initialized()
    def device(req, dev):
        events.append("device")
        return {"observed_uuid": req["gpu_uuid_mapping"]["0"]}
    def configure(dev, cap, reserve):
        events.append(("cap", str(dev), cap, reserve))
        return {"enabled": True, "allocator_limit_mib": cap, "min_free_mib": reserve}
    def snapshot(dev):
        events.append("snapshot")
        return {}
    def evaluate(argv):
        events.append("main")
        assert argv == eval_request["evaluation_argv"]
        return 0
    result = shared.execute_evaluation(eval_request, device_verifier=device, configure=configure,
                                       snapshot=snapshot, evaluate=evaluate)
    assert events == ["device", ("cap", "cuda:0", 12000, 8192), "snapshot", "main", "snapshot"]
    assert result["allocator_configured_before_main"] is True
    assert torch.cuda.is_initialized() == initialized


def test_failed_cap_prevents_model_evaluation(eval_request):
    def configure(*args):
        raise RuntimeError("free 부족")
    with pytest.raises(RuntimeError, match="free 부족"):
        shared.execute_evaluation(eval_request, device_verifier=lambda *_: {}, configure=configure,
                                  evaluate=lambda *_: pytest.fail("main 실행 금지"), snapshot=lambda *_: {})


def test_wrong_child_mapping_is_rejected_before_cuda_initialization(eval_request, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-foreign")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    initialized = torch.cuda.is_initialized()
    with pytest.raises(ValueError, match="UUID 순서"):
        shared.verify_child_device(eval_request, torch.device("cuda:0"))
    assert torch.cuda.is_initialized() == initialized


def test_new_artifact_symlink_escape_and_existing_targets(eval_request):
    root = Path(eval_request["root"])
    target = root / "reports/existing.json"
    target.write_text("원본", encoding="utf-8")
    with pytest.raises(ValueError, match="덮어쓰지"):
        shared.new_artifact(target, root, "reports", ".json")
    alias = root / "reports/dangling.json"
    alias.symlink_to(root / "reports/missing.json")
    with pytest.raises(ValueError, match="덮어쓰지"):
        shared.new_artifact(alias, root, "reports", ".json")
    with pytest.raises(ValueError, match="루트 밖"):
        shared.new_artifact(root.parent / "escape.json", root, "reports", ".json")


def make_result(eval_request):
    protocol = {"arguments": vars(eval_arguments(eval_request["evaluation_argv"])),
                "checkpoint_sha256": eval_request["checkpoint_sha256"], "checkpoint_step": 3000,
                "source": eval_request["sources"]["evaluation"], "motion_prediction_contract": motion_prediction_contract(),
                "data": {"receiver_count": 1998, "receiver_rows_sha256": eval_request["training"]["eval_rows_sha256"],
                         "split_sha256": p2.SPLIT_SHA,
                         "supervision_sha256": {"supervision_manifest.json": p2.MANIFEST_SHAS[0], "calibration.npz": p2.CALIBRATION_SHAS[0]}}}
    Path(eval_request["protocol"]).write_text(json.dumps(protocol))
    result = {"status": "completed", "selection_performed": False, "final_val_accessed": False,
              "protocol": protocol, "protocol_path": eval_request["protocol"], "protocol_sha256": shared.sha256(eval_request["protocol"]),
              "conditions": {c: {"records": [{"pred_state": [0.] * 6, "pred_history": [[0.] * 4] * 4}] * 1998,
                                 "summary": {"n": 1998}} for c in shared.CONDITIONS}}
    receipt = {"status": "completed", "child_pid": FakeChild.pid, "request_sha256": shared.json_digest(eval_request),
               "main_returncode": 0, "allocator_configured_before_main": True,
               "memory_policy": {"enabled": True, "allocator_limit_mib": 12000, "min_free_mib": 8192},
               "device_mapping": {"observed_uuid": eval_request["gpu_uuid_mapping"]["0"]}}
    Path(eval_request["output"]).write_text(json.dumps(result))
    Path(eval_request["child_receipt"]).write_text(json.dumps(receipt))
    return result, receipt


@pytest.mark.parametrize("change", [None, "receipt_pid", "receipt_cap", "condition", "checkpoint", "rows", "source", "protocol"])
def test_actual_receipt_and_report_contract_before_success(eval_request, monkeypatch, change):
    result, receipt = make_result(eval_request)
    monkeypatch.setattr(shared, "recheck_files", lambda *_: None)
    if change == "receipt_pid": receipt["child_pid"] += 1
    elif change == "receipt_cap": receipt["memory_policy"]["enabled"] = False
    elif change == "condition": result["conditions"].pop("reverse_history")
    elif change == "checkpoint": result["protocol"]["checkpoint_sha256"] = "f" * 64
    elif change == "rows": result["conditions"]["normal"]["records"].pop()
    elif change == "source": result["protocol"]["source"] = {}
    elif change == "protocol": result["protocol_sha256"] = "f" * 64
    Path(eval_request["output"]).write_text(json.dumps(result))
    Path(eval_request["child_receipt"]).write_text(json.dumps(receipt))
    if change is None:
        found = shared.inspect_result(eval_request, FakeChild.pid, shared.json_digest(eval_request))
        assert found["normal_summary"]["n"] == 1998
    else:
        with pytest.raises(ValueError):
            shared.inspect_result(eval_request, FakeChild.pid, shared.json_digest(eval_request))


@pytest.mark.parametrize("status,step,nonfinite,git_sha", [("running", 3000, 0, shared.TRAINING_GIT_SHA),
    ("completed", 2999, 0, shared.TRAINING_GIT_SHA), ("completed", 3000, 1, shared.TRAINING_GIT_SHA),
    ("completed", 3000, 0, "other")])
def test_training_must_finish_last3000_clean_before_any_gpu_or_launch(eval_request, status, step, nonfinite, git_sha):
    root = Path(eval_request["root"])
    run = root / "work_dirs/motiondrive_v2/p2_c0t0_s0"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"status": status, "step": step, "nonfinite_count": nonfinite, "git_sha": git_sha}))
    with pytest.raises(ValueError):
        shared.verify_training(root, "c0t0", root / "unused.json", "a" * 64)


def complete_training_fixture(eval_request, monkeypatch):
    root = Path(eval_request["root"])
    run = root / "work_dirs/motiondrive_v2/p2_c0t0_s0"
    run.mkdir(parents=True)
    files = {
        "split": root / "data/etri/motiondrive_v2/grouped_split_rawtime.json",
        "supervision": root / "data/etri/motiondrive_v2/train_tune_rawtime/supervision_manifest.json",
        "calibration": root / "data/etri/motiondrive_v2/train_tune_rawtime/calibration.npz",
        "initializer": root / "work_dirs/motiondrive_v2/p1_g1s1_s0/last.pth",
        "trainer": root / shared.TRAINER, "supervisor": root / shared.SUPERVISOR,
    }
    for name, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"합성 CPU fixture {name}".encode())
    # 실제 데이터 SHA 대신 임시 fixture의 실제 SHA를 사용하는 독립 CPU 테스트.
    monkeypatch.setattr(p2, "SPLIT_SHA", shared.sha256(files["split"]))
    monkeypatch.setattr(p2, "COMMON_SHA", shared.sha256(files["initializer"]))
    monkeypatch.setattr(p2, "MANIFEST_SHAS", (shared.sha256(files["supervision"]), "d" * 64))
    monkeypatch.setattr(p2, "CALIBRATION_SHAS", (shared.sha256(files["calibration"]), "e" * 64))
    args = {**copy.deepcopy(p2.EXPECTED_ARGS), "gpu": 0, "run_dir": str(run), "data_root": str(root),
            "microbatch": 2, "cuda_memory_limit_mib": 12000, "cuda_min_free_mib": 8192,
            "supervision_root": "data/etri/motiondrive_v2/train_tune_rawtime", "time_input": "raw",
            "init": "work_dirs/motiondrive_v2/p1_g1s1_s0/last.pth", "resume": None}
    manifest = {"status": "completed", "step": 3000, "nonfinite_count": 0, "git_sha": shared.TRAINING_GIT_SHA,
                "arguments": args, "time_input": "raw", "data_counts": {"train": 54810, "eval": 1998},
                "split_sha256": p2.SPLIT_SHA, "supervision_manifest_sha256": p2.MANIFEST_SHAS[0],
                "load_report": {"common_checkpoint_sha256": p2.COMMON_SHA},
                "train_rows_sha256": "f" * 64, "eval_rows_sha256": "a" * 64,
                "initial_model_state_sha256": "b" * 64, "pid": 2147483000}
    (run / "manifest.json").write_text(json.dumps(manifest))
    selected = run / "last.pth"
    torch.save({"step": 3000, "manifest": copy.deepcopy(manifest), "model": {"mock.weight": torch.ones(2)}}, selected)
    supervisor_pid = 2147483001
    supervisor_path = root / "logs/motiondrive_v2/p2_c0t0_s0.supervisor.json"
    command = [sys.executable, shared.TRAINER, "--mock-not-executed"]
    snapshot = {"exists": True, "belongs_to_child": True, "pid": manifest["pid"], "status": "completed",
                "step": 3000, "manifest_path": str(run / "manifest.json")}
    exit_record = {"schema_version": 1, "supervisor_record_id": "CPU-only", "supervisor_pid": supervisor_pid,
                   "child_pid": manifest["pid"], "gpu": 0, "command": command, "run_dir": str(run),
                   "trainer_manifest_path": str(run / "manifest.json"), "trainer_manifest": snapshot,
                   "status": "process_exited", "actual_returncode": 0, "exit_code": 0,
                   "termination_signal": None, "termination_signal_name": None, "outcome": "completed_cleanly",
                   "supervisor_exit_code": 0, "trainer_reported_completed": True, "ended_at": "mock",
                   "pressure_event": None}
    supervisor_path.write_text(json.dumps(exit_record))
    job = {"name": "p2_c0t0_s0", "gpu": 0, "explicit_argv": [shared.TRAINER, "--mock-not-executed"],
           "supervisor_record": str(supervisor_path), "trainer_command": command,
           "inputs": {"checkpoint_init": {"sha256": p2.COMMON_SHA}, "split": {"sha256": p2.SPLIT_SHA},
                      "supervision": {"sha256": p2.MANIFEST_SHAS[0]}}}
    plan_path = root / "configs/motiondrive_v2/p2_geometry_time_shared_r1_s0.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps({"supervise": True, "gpu_sharing": {"mode": "shared", "allocator_limit_mib": 12000,
                                                                       "reserve_mib": 8192},
                                    "jobs": [{"name": job["name"], "gpu": 0, "argv": job["explicit_argv"]}]}))
    launch_path = root / "logs/motiondrive_v2/launch_cpu_mock.json"
    launch = {"status": "launched", "supervise": True, "tracked_dirty": "", "actual_git_sha": shared.TRAINING_GIT_SHA,
              "trainer_sha256": shared.sha256(files["trainer"]), "supervisor_sha256": shared.sha256(files["supervisor"]),
              "jobs": [job], "processes": [{"name": job["name"], "gpu": 0, "pid": supervisor_pid}]}
    launch_path.write_text(json.dumps(launch))
    return root, run, selected, launch_path, supervisor_path, files


@pytest.mark.parametrize("damage", [None, "os_nonzero", "source", "initializer", "last_sha", "nan_tensor"])
def test_actual_cpu_training_artifact_lineage_gate(eval_request, monkeypatch, damage):
    root, run, selected, launch, supervisor, files = complete_training_fixture(eval_request, monkeypatch)
    expected = shared.sha256(selected)
    if damage == "os_nonzero":
        record = json.loads(supervisor.read_text()); record["actual_returncode"] = 1
        supervisor.write_text(json.dumps(record))
    elif damage == "source": files["trainer"].write_bytes(b"changed")
    elif damage == "initializer": files["initializer"].write_bytes(b"changed")
    elif damage == "last_sha": expected = "0" * 64
    elif damage == "nan_tensor":
        saved = torch.load(selected, map_location="cpu", weights_only=False)
        saved["model"]["mock.weight"][0] = float("nan")
        torch.save(saved, selected); expected = shared.sha256(selected)
    if damage is None:
        result = shared.verify_training(root, "c0t0", launch, expected)
        assert result["os_exit"]["actual_returncode"] == 0 and result["selected_checkpoint_sha256"] == expected
        assert result["all_four_training_exits_checked"] is False
        assert result["input_file_sha256"][str(selected)] == shared.sha256(selected)
    else:
        with pytest.raises(ValueError):
            shared.verify_training(root, "c0t0", launch, expected)
