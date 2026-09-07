import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_motiondrive_v2_gs import (ARMS, COMMON_SHA, EXPECTED_FRAMES, GS,
                                      analyze_gs, compare_evaluation_log, joint_session_bootstrap, normalize_evaluation,
                                      normalize_record, load_supervisor_evidence, validate_experiment,
                                      validate_supervisor_record)


def fake_arms():
    scenes = [f"scene{i:02d}" for i in range(37)]
    mapping = {scene: f"session{i % 11}" for i, scene in enumerate(scenes)}
    mapping.update(train0="train_session", val0="val_session")
    split = {"splits": {"train": ["train0"], "tune": scenes, "val": ["val0"], "historical_val": ["val0"]},
             "scene_to_session": mapping}
    arms = {}
    for arm, gs, value in zip(ARMS, GS, (1., .8, .9, .6)):
        goal, state = gs
        args = dict(phase="joint", steps=6000, batch=16, eval_batch=4, workers=4, seed=0,
            precision="bf16", lr=1e-4, backbone_lr=1e-5, weight_decay=.01, grad_clip=5., warmup=200,
            alpha_occ=.2, alpha_lane=.2, alpha_motion=.2, uncertainty=1, bn_policy="fixed",
            motion_input_mode="low_feature", train_stride=1, eval_stride=5, eval_split="tune",
            max_train_samples=0, max_eval_samples=0, eval_every=250, save_every=1000,
            goal_on=int(goal), state_on=int(state), gpu=ARMS.index(arm), run_dir=arm,
            plan_output_scale=[10., 5.], init="work_dirs/motiondrive_v2/p0_motion_bn_fixed_s0/best.pth",
            resume=None, train_scenes=None, eval_scenes=None, eval_only=False)
        manifest = dict(status="completed", step=6000, nonfinite_count=0,
            data_counts={"train": 54810, "eval": 1998}, split_sha256="split", supervision_manifest_sha256="sup",
            arguments=args, model_config=dict(goal_on=goal, state_on=state, motion_input_mode="low_feature", plan_output_scale=[10., 5.]),
            load_report={"common_checkpoint_sha256": COMMON_SHA},
            loss_weights=dict(plan=1., occupancy=.2, lane=.2, motion=.2, uncertainty=True),
            bn_training=dict(policy="fixed", affine_and_backbone_weights_trainable=True),
            git_sha="source", initial_model_state_sha256="initial", initial_parameter_count=123,
            train_rows_sha256="train", eval_rows_sha256="eval")
        records = [{"scenario": scene, "session": mapping[scene], "frame": frame, "d3": value}
                   for scene in scenes for frame in EXPECTED_FRAMES]
        arms[arm] = dict(checkpoint_path=f"{arm}/last.pth", checkpoint_sha256=arm, checkpoint_step=6000,
            split_manifest_sha256="split", supervision_manifest_sha256="sup", run_manifest=manifest,
            sample_order_sha256_at_checkpoint="order", records=records,
            report={"official_d3": value, "n": 1998, "n_sessions": 11},
            best={"selection_definition": "official_d3", "step": 250, "official_d3": value / 2})
    return arms, split


def test_conditional_effects_and_interaction_signs():
    arms, split = fake_arms()
    result = analyze_gs(arms, split, "split", repeats=100)
    effects = result["last6000"]["primary_frame_weighted"]["contrasts"]
    expected = {"G_given_S0": -.2, "G_given_S1": -.3, "S_given_G0": -.1, "S_given_G1": -.2, "interaction_GxS": -.1}
    for key, value in expected.items():
        assert effects[key]["estimate"] == pytest.approx(value)
        assert effects[key]["ci95_percentile"] == pytest.approx([value, value])
    assert result["best_secondary_selection_table"]["g0s0"]["official_d3"] == .5
    assert result["last6000"]["primary_frame_weighted"]["arms"]["g0s0"]["estimate"] == 1.


def test_joint_session_draws_preserve_pairing_and_distinct_estimands():
    base = np.array([0., 10., 10., 10.])
    matrix = np.column_stack([base, base - .2, base - .1, base - .4])
    result = joint_session_bootstrap(matrix, ["small", "large", "large", "large"], repeats=1000, seed=7)
    assert result["primary_frame_weighted"]["arms"]["g0s0"]["estimate"] == 7.5
    assert result["secondary_session_mean"]["arms"]["g0s0"]["estimate"] == 5.
    # 각 세션은 크게 다르지만 paired 효과는 상수다. 독립 팔 재표집이면 이 검사를 통과하지 못한다.
    for estimator in ("primary_frame_weighted", "secondary_session_mean"):
        contrast = result[estimator]["contrasts"]["G_given_S0"]
        assert contrast["ci95_percentile"] == pytest.approx([-.2, -.2])
        assert result[estimator]["contrasts"]["interaction_GxS"]["ci95_percentile"] == pytest.approx([-.1, -.1])
    assert result["resampling"]["shared_draws"]


def test_record_order_does_not_change_frame_pairs():
    arms, split = fake_arms()
    arms["g1s1"]["records"].reverse()
    result = analyze_gs(arms, split, "split", repeats=100)
    assert len(result["per_frame_pairs"]) == 1998
    assert result["last6000"]["primary_frame_weighted"]["contrasts"]["interaction_GxS"]["estimate"] == pytest.approx(-.1)


def test_real_planning_evaluator_envelope_and_metric_alias():
    arms, _ = fake_arms()
    source = arms["g0s0"]
    normal = {"summary": source["report"], "records": source["records"], "buckets": {"stop": {"n": 1}}}
    raw = {"status": "completed", "selection_performed": False, "final_val_accessed": False,
           "precision": "bf16", "protocol_path": "raw.protocol.json", "protocol_sha256": "protocol",
           "model_load": {"explicit_overrides": {}, "checkpoint_sha256": "lastsha"},
           "conditions": {"normal": normal},
           "protocol": {"checkpoint_sha256": "lastsha", "checkpoint_step": 6000,
               "arguments": dict(checkpoint="g0s0/last.pth", split="tune", frame_stride=5, max_samples=0,
                                  scenes=None, batch=4, workers=4, seed=0, precision="bf16", device="cuda:0"),
               "data": {"split_sha256": "split", "receiver_count": 1998, "receiver_rows_sha256": "eval",
                        "supervision_sha256": {"supervision_manifest.json": "sup"}}}}
    normalized = normalize_evaluation(raw)
    assert normalized["records"] == source["records"]
    assert normalized["checkpoint_step"] == 6000
    assert normalized["time_input"] == "raw"  # legacy envelope has no time flag
    assert normalized["planning_diagnostics"]["normal_buckets"] == {"stop": {"n": 1}}
    assert normalize_record({"official_d3": .2})["d3"] == .2
    with pytest.raises(ValueError, match="충돌"):
        normalize_record({"d3": .2, "official_d3": .3})
    raw["protocol"]["arguments"]["max_samples"] = 370
    with pytest.raises(ValueError, match="1998"):
        normalize_evaluation(raw)


@pytest.mark.parametrize("location", ["top", "top_policy", "protocol", "protocol_args", "protocol_policy",
                                      "condition", "conflicting"])
def test_nominal_evaluation_is_rejected_from_every_declared_location(location):
    evidence = {"records": []}
    if location == "top":
        evidence["time_input"] = "nominal"
    elif location == "top_policy":
        evidence["time_input_policy"] = {"mode": "nominal"}
    elif location == "protocol":
        evidence["protocol"] = {"time_input": "nominal"}
    elif location == "protocol_args":
        evidence["protocol"] = {"arguments": {"time_input": "nominal"}}
    elif location == "protocol_policy":
        evidence["protocol"] = {"time_input_policy": {"mode": "nominal"}}
    elif location == "condition":
        evidence["conditions"] = {"normal": {"time_input": "nominal"}}
    else:
        evidence["time_input"] = "raw"
        evidence["protocol"] = {"arguments": {"time_input": "nominal"}}
    with pytest.raises(ValueError, match="time_input=raw"):
        normalize_evaluation(evidence)


def test_legacy_and_explicit_raw_reports_have_identical_primary_gs_results():
    arms, split = fake_arms()
    legacy = analyze_gs(arms, split, "split", repeats=100)
    for arm in arms.values():
        arm["time_input"] = "raw"
    explicit = analyze_gs(arms, split, "split", repeats=100)
    assert explicit == legacy
    assert explicit["protocol"]["time_input"] == "raw"
    assert normalize_evaluation({"records": []})["time_input"] == "raw"
    assert normalize_evaluation({"records": [], "time_input": "raw"})["time_input"] == "raw"
    arms["g1s1"]["time_input"] = "nominal"
    with pytest.raises(ValueError, match="time_input=raw"):
        validate_experiment(arms, split, "split")


def test_only_session_aggregation_rounding_is_distinguished_from_forward_mismatch():
    logged = {"official_d3": .2, "session_mean_d3": .3, "session_d3": {"a": .1, "b": .5}}
    actual = dict(logged, session_mean_d3=.3 + 1e-16)
    result = compare_evaluation_log(actual, logged)
    assert result["exact_fields"]["official_d3"]
    assert result["session_mean_aggregation_difference"] != 0
    with pytest.raises(ValueError, match="정확히"):
        compare_evaluation_log(dict(actual, official_d3=.2 + 1e-16), logged)
    with pytest.raises(ValueError, match="집계 차이"):
        compare_evaluation_log(dict(actual, session_mean_d3=.31), logged)


def test_same_frame_gt_mismatch_rejected():
    arms, split = fake_arms()
    arms["g1s1"]["records"][0]["gt_abs_xy"] = [[0., 0.]] * 6
    with pytest.raises(ValueError, match="GT"):
        validate_experiment(arms, split, "split")


def supervisor_fixture(tmp_path):
    run = tmp_path / "project/work_dirs/motiondrive_v2/p1_g0s0_s0"
    manifest = {"pid": 12345, "status": "completed", "step": 6000, "arguments": {"gpu": 0, "run_dir": str(run)}}
    record = {"schema_version": 1, "supervisor_record_id": "owned", "child_pid": 12345,
              "run_dir": str(run), "trainer_manifest_path": str(run / "manifest.json"), "gpu": 0,
              "command": ["/usr/bin/python", "scripts/train_motiondrive_v2.py", "--gpu", "0", "--run-dir", str(run)],
              "trainer_manifest": {"exists": True, "belongs_to_child": True, "pid": 12345,
                                   "manifest_path": str(run / "manifest.json"), "status": "completed", "step": 6000},
              "status": "process_exited", "actual_returncode": 0, "exit_code": 0, "termination_signal": None,
              "termination_signal_name": None, "outcome": "completed_cleanly", "supervisor_exit_code": 0,
              "trainer_reported_completed": True, "ended_at": "2026-09-07T00:00:00+00:00",
              "received_signals": [], "teardown_pending_observed": False}
    return run, manifest, record


def test_clean_actual_exit_and_pid_absence_required(tmp_path):
    run, manifest, record = supervisor_fixture(tmp_path)
    checks = validate_supervisor_record(record, run, manifest, pid_alive=lambda pid: False)
    assert checks["actual_returncode"] == 0
    assert not checks["trainer_pid_alive_at_analysis"]
    with pytest.raises(ValueError, match="아직 살아"):
        validate_supervisor_record(record, run, manifest, pid_alive=lambda pid: True)


@pytest.mark.parametrize("change", ["pid", "snapshot_pid", "path", "snapshot_path", "command_path", "manifest_arg_path",
                                    "gpu", "pending", "nonzero", "signal", "missing_exit"])
def test_completed_manifest_cannot_hide_supervisor_mismatch_or_failed_exit(tmp_path, change):
    run, manifest, record = supervisor_fixture(tmp_path)
    if change == "pid":record["child_pid"] = 999
    elif change == "snapshot_pid":record["trainer_manifest"]["pid"] = 999
    elif change == "path":record["run_dir"] = str(run.parent / "other")
    elif change == "snapshot_path":record["trainer_manifest"]["manifest_path"] = str(run.parent / "other/manifest.json")
    elif change == "command_path":record["command"][-1] = str(run.parent / "other")
    elif change == "manifest_arg_path":manifest["arguments"]["run_dir"] = str(run.parent / "other")
    elif change == "gpu":record["gpu"] = 1
    elif change == "pending":record["status"] = "teardown_pending";record["actual_returncode"] = None
    elif change == "nonzero":record["actual_returncode"] = 2;record["exit_code"] = 2
    elif change == "signal":record["actual_returncode"] = -11;record["termination_signal"] = 11
    elif change == "missing_exit":record.pop("ended_at")
    with pytest.raises(ValueError):
        validate_supervisor_record(record, run, manifest, pid_alive=lambda pid: False)


def test_supervisor_owned_path_sha_and_raw_exit_record_preserved(tmp_path, monkeypatch):
    run, manifest, record = supervisor_fixture(tmp_path)
    record["teardown_pending_observed"] = True  # 지연 뒤 실제 정상 종료했다면 사건 기록을 보존한다.
    path = run.parents[2] / "logs/motiondrive_v2" / f"{run.name}.supervisor.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record))
    monkeypatch.setattr("analyze_motiondrive_v2_gs.process_is_alive", lambda pid: False)
    evidence = load_supervisor_evidence(run, manifest)
    assert evidence["path"] == str(path)
    assert len(evidence["sha256"]) == 64
    assert evidence["record"]["teardown_pending_observed"] is True
    path.unlink()
    with pytest.raises(ValueError, match="기록이 없습니다"):
        load_supervisor_evidence(run, manifest)


@pytest.mark.parametrize("change", ["best_as_last", "step", "gs", "common_sha", "order", "missing_frame",
                                     "duplicate_frame", "session", "bn", "extra_arg", "nan", "report", "seed"])
def test_unfair_or_unpaired_inputs_fail_closed(change):
    arms, split = fake_arms()
    arm = arms["g1s1"]
    manifest = arm["run_manifest"]
    if change == "best_as_last":arm["checkpoint_path"] = "g1s1/best.pth"
    elif change == "step":arm["checkpoint_step"] = 5000
    elif change == "gs":manifest["model_config"]["state_on"] = False
    elif change == "common_sha":manifest["load_report"]["common_checkpoint_sha256"] = "wrong"
    elif change == "order":arm["sample_order_sha256_at_checkpoint"] = "different"
    elif change == "missing_frame":arm["records"].pop()
    elif change == "duplicate_frame":arm["records"].append(copy.deepcopy(arm["records"][0]))
    elif change == "session":arm["records"][0]["session"] = "wrong"
    elif change == "bn":manifest["arguments"]["bn_policy"] = "adaptive"
    elif change == "extra_arg":manifest["arguments"]["new_loss"] = 1
    elif change == "nan":arm["records"][0]["d3"] = float("nan")
    elif change == "report":arm["report"]["official_d3"] = .5
    elif change == "seed":manifest["arguments"]["seed"] = 1
    with pytest.raises(ValueError):
        validate_experiment(arms, split, "split")
