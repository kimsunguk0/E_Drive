"""Mock P2 results only. No actual experiments, predictions or result files."""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_motiondrive_v2_p2 as p2


def time_policy(mode):
    return {"mode": mode, "nominal_history_seconds": [.1, .2, .5, 1.], "nominal_dtype": "float32",
            "supervision_and_dataset_modified": False}


def summarize(records):
    values = [r["d3"] for r in records]
    session_d3 = {s: float(np.mean([r["d3"] for r in records if r["session"] == s]))
                  for s in sorted({r["session"] for r in records})}
    return {"official_d3": float(np.mean(values)), "n": len(records), "n_sessions": len(session_d3),
            "n_scenes": len({r["scenario"] for r in records}), "session_d3": session_d3,
            "session_mean_d3": float(np.mean(list(session_d3.values())))}


@pytest.fixture(scope="module")
def experiment_template():
    train, tune = [f"train{i:03d}" for i in range(203)], [f"tune{i:03d}" for i in range(37)]
    mapping = {s: f"train_session{i % 72:02d}" for i, s in enumerate(train)}
    mapping.update({s: f"tune_session{i % 11:02d}" for i, s in enumerate(tune)})
    mapping["val000"] = "val_session"
    split = {"splits": {"train": train, "tune": tune, "val": ["val000"], "historical_val": []}, "scene_to_session": mapping}
    config = {"goal_on": True, "state_on": True, "backbone_arch": "resnet50", "channels": 128,
              "motion_input_mode": "low_feature", "plan_output_scale": [10., 5.]}
    signature = {"xy.weight": [[2, 4], "torch.float32"]}
    common = {"model_config": config, "tensor_signature": signature, "model_state_sha256": "common-state-sha"}
    geometry = {"train_tune_scenes": 240, "all_scene_npz_json_bitwise_preserved": True,
                "other_five_calibrations_bitwise_preserved": True, "manifest_sha256": list(p2.MANIFEST_SHAS),
                "calibration_sha256": list(p2.CALIBRATION_SHAS),
                "scene_file_sha256": {s + ext: s + ext + "-sha" for s in train + tune for ext in (".npz", ".json")}}
    row_sha = hashlib.sha256(np.arange(1998, dtype="<i8").tobytes()).hexdigest()
    source = {"git_sha": "evaluation-commit", "tracked_changes": [], "file_sha256": {s: "same-sha" for s in (
        "scripts/evaluate_motiondrive_v2_planning.py", "scripts/motiondrive_v2_data.py", "scripts/motiondrive_v2_training.py",
        "scripts/train_motiondrive_v2.py", "models/motiondrive_v2/model.py")}}
    arms = {}
    for index, (arm, (c, t)) in enumerate(zip(p2.ARMS, p2.FACTORS)):
        timing = "nominal" if t else "raw"
        run = f"/mock/work_dirs/motiondrive_v2/p2_{arm}_s0"
        supervision = "data/etri/motiondrive_v2/" + ("train_tune_geometry_v2" if c else "train_tune_rawtime")
        args = {**copy.deepcopy(p2.EXPECTED_ARGS), "gpu": index, "run_dir": run, "resume": None, "pretrained": None,
                "train_scenes": None, "eval_scenes": None, "eval_only": False, "cpu": False,
                "init": "work_dirs/motiondrive_v2/p1_g1s1_s0/last.pth", "supervision_root": supervision,
                "time_input": timing, "data_root": "/mock", "split_manifest": "same-split"}
        manifest = {"status": "completed", "step": 3000, "nonfinite_count": 0, "data_counts": {"train": 54810, "eval": 1998},
                    "split_sha256": p2.SPLIT_SHA, "supervision_manifest_sha256": p2.MANIFEST_SHAS[c],
                    "time_input": timing, "time_input_policy": time_policy(timing), "arguments": args,
                    "model_config": config, "initial_model_state_sha256": "common-state-sha", "initial_parameter_count": 26409112,
                    "load_report": {"common_checkpoint_sha256": p2.COMMON_SHA},
                    "loss_weights": dict(plan=1., occupancy=.2, lane=.2, motion=.2, uncertainty=True),
                    "bn_training": {"policy": "fixed", "affine_and_backbone_weights_trainable": True},
                    "git_sha": "training-commit", "train_rows_sha256": "train-rows-sha", "eval_rows_sha256": row_sha,
                    "torch": "torch-version", "numpy": "numpy-version", "pid": 800000 + index}
        records = []
        for scene in tune:
            for frame in range(30, 300, 5):
                error = .4 + int(scene[-3:]) * .002 + frame * .0001 + (-.04 * c) + (.015 * t) + (-.01 * c * t)
                records.append({"scenario": scene, "frame": frame, "session": mapping[scene], "row": len(records),
                    "gt_abs_xy": [[0., 0.]] * 6, "pred_abs_xy": [[error, 0.]] * 6, "d3": error,
                    "point_l2": [error] * 6, "cumulative_ade_1_2_3s": [error] * 3,
                    "longitudinal_error": [error] * 6, "lateral_error": [0.] * 6,
                    "gt_state": [1., 0., 0., 0., 0., 0.], "gt_state_valid": [True] * 6, "bucket": "cruise",
                    "weighted_abs_longitudinal_error": error, "weighted_abs_lateral_error": 0.})
        summary = summarize(records)
        normal = {"time_input": timing, "records": records, "summary": summary, "buckets": {"cruise": summary}}
        protocol = {"status": "preregistered_before_any_forward", "schema_version": 1,
                    "time_input": timing, "time_input_policy": time_policy(timing),
                    "model_input_whitelist": list(p2.MODEL_INPUTS), "nominal_waypoint_seconds": [.5, 1., 1.5, 2., 2.5, 3.],
                    "checkpoint_step": 3000, "checkpoint_sha256": f"last-sha-{arm}", "checkpoint_manifest": copy.deepcopy(manifest),
                    "arguments": {"split": "tune", "frame_stride": 5, "max_samples": 0, "scenes": None, "batch": 4,
                        "workers": 4, "seed": 0, "precision": "bf16", "device": "cuda:1", "conditions": ["normal"],
                        "time_input": timing, "supervision_root": supervision, "checkpoint": run + "/last.pth"},
                    "data": {"receiver_count": 1998, "split_sha256": p2.SPLIT_SHA, "receiver_rows_sha256": row_sha,
                        "supervision_sha256": {"supervision_manifest.json": p2.MANIFEST_SHAS[c], "calibration.npz": p2.CALIBRATION_SHAS[c],
                            **{s + ext: geometry["scene_file_sha256"][s + ext] for s in tune for ext in (".npz", ".json")}},
                        "ego_cache_sha256": "same-cache-sha", "image_root": "/mock/cache/etri_768"}, "source": copy.deepcopy(source)}
        evaluation = {"status": "completed", "selection_performed": False, "final_val_accessed": False,
                      "time_input": timing, "precision": "bf16", "protocol": protocol,
                      "model_load": {"explicit_overrides": {}, "checkpoint_sha256": f"last-sha-{arm}"}, "conditions": {"normal": normal}}
        metrics = [{"kind": "initial_eval", "step": 0, "time_input": timing}]
        for step in [1, *range(10, 3001, 10)]:
            k = step - 1
            lr = 5e-5 * min(1., (k + 1) / 100.) * .5 * (1. + math.cos(math.pi * max(0., (k - 100) / 2900.)))
            metrics.append({"kind": "train", "step": step, "epoch": 0, "sample_order_sha256": f"identical-trace-{step}",
                            "lr": lr, "grad_norm": 10., "total": -.5, "plan_d3": .2, "occ_bce": .1, "lane_bce": .1,
                            "history": -1., "state": -1., "stop_bce": .1, "motion": -1.98})
        for step in range(250, 3001, 250):
            value = summary["official_d3"] if step == 3000 else summary["official_d3"] + .01
            metrics.append({"kind": "eval", "step": step, "time_input": timing, **summary,
                            "selection_definition": "official_d3", "selection_metric": value, "official_d3": value})
        arms[arm] = {"run_dir": run, "run_manifest": copy.deepcopy(manifest), "evaluation": evaluation, "metrics": metrics,
                     "initial_tensor_sha256": "common-state-sha", "initial_tensor_signature": signature,
                     "last_tensor_signature": signature, "last_checkpoint_sha256": f"last-sha-{arm}",
                     "os_exit": {"actual_returncode": 0, "trainer_pid_absent": True},
                     "training_source": {"git_sha": "training-commit", "trainer_sha256": "same-trainer-sha", "supervisor_sha256": "same-supervisor-sha"},
                     "best": {"step": 3000, "official_d3": summary["official_d3"], "selection_definition": "official_d3"}}
    return arms, split, common, geometry


@pytest.fixture
def experiment(experiment_template):
    return copy.deepcopy(experiment_template)


def test_mock_p2_accepts_intended_geometry_time_differences_and_separate_source_stages(experiment):
    arms, split, common, geometry = experiment
    result = p2.analyze_p2(arms, split, common, geometry)
    assert result["last3000"]["resampling"]["repeats"] == 10000
    assert result["last3000"]["resampling"]["seed"] == 20260907
    contrasts = result["last3000"]["primary_frame_weighted"]["contrasts"]
    assert contrasts["C_given_T0"]["estimate"] == pytest.approx(-.04)
    assert contrasts["C_given_T1"]["estimate"] == pytest.approx(-.05)
    assert contrasts["T_given_C0"]["estimate"] == pytest.approx(.015)
    assert contrasts["T_given_C1"]["estimate"] == pytest.approx(.005)
    assert contrasts["interaction_CxT"]["estimate"] == pytest.approx(-.01)
    assert not result["c1t1_image_diagnostics_complete"]
    assert result["c1t1_image_diagnostics_missing"] == ["image_shuffle", "repeat_current", "reverse_history"]
    assert result["protocol"]["only_deployable_contract"].startswith("C1/T1")


@pytest.mark.parametrize("change", ["train_time", "missing_train_time", "eval_time", "missing_eval_policy", "condition_time",
    "calibration", "supervision", "initial_tensor", "shape", "train_source", "eval_source", "source_missing", "lr", "steps",
    "bn", "loss", "state_off", "exposure", "row_trace", "gt", "record_order", "gt_file", "nonfinite", "best_step", "os_exit"])
def test_strict_p2_rejects_uncontrolled_or_missing_evidence(experiment, change):
    arms, split, common, geometry = experiment
    arm = arms["c1t1"]
    manifest, evidence = arm["run_manifest"], arm["evaluation"]
    if change == "train_time": manifest["time_input"] = "raw"
    elif change == "missing_train_time": del manifest["arguments"]["time_input"]
    elif change == "eval_time": evidence["time_input"] = "raw"
    elif change == "missing_eval_policy": del evidence["protocol"]["time_input_policy"]
    elif change == "condition_time": evidence["conditions"]["normal"]["time_input"] = "raw"
    elif change == "calibration": evidence["protocol"]["data"]["supervision_sha256"]["calibration.npz"] = p2.CALIBRATION_SHAS[0]
    elif change == "supervision": manifest["supervision_manifest_sha256"] = p2.MANIFEST_SHAS[0]
    elif change == "initial_tensor": arm["initial_tensor_sha256"] = "other-state"
    elif change == "shape": arm["last_tensor_signature"] = {"changed": [[2, 5], "torch.float32"]}
    elif change == "train_source": arm["training_source"]["trainer_sha256"] = "other-code"
    elif change == "eval_source": evidence["protocol"]["source"]["git_sha"] = "other-eval-commit"
    elif change == "source_missing": del evidence["protocol"]["source"]["file_sha256"]["models/motiondrive_v2/model.py"]
    elif change == "lr": manifest["arguments"]["lr"] = 1e-4
    elif change == "steps": manifest["arguments"]["steps"] = 6000
    elif change == "bn": manifest["bn_training"]["policy"] = "adaptive"
    elif change == "loss": manifest["loss_weights"]["motion"] = 0.
    elif change == "state_off": manifest["model_config"]["state_on"] = False
    elif change == "exposure": manifest["data_counts"]["train"] += 1
    elif change == "row_trace": arm["metrics"][5]["sample_order_sha256"] = "other-order"
    elif change == "gt": evidence["conditions"]["normal"]["records"][0]["gt_state"] = [2., 0., 0., 0., 0., 0.]
    elif change == "record_order": evidence["conditions"]["normal"]["records"].reverse()
    elif change == "gt_file": evidence["protocol"]["data"]["supervision_sha256"]["tune000.npz"] = "changed-labels"
    elif change == "nonfinite": arm["metrics"][5]["total"] = float("nan")
    elif change == "best_step": arm["best"]["step"] = 250
    elif change == "os_exit": arm["os_exit"]["actual_returncode"] = -11
    with pytest.raises(ValueError):
        p2.analyze_p2(arms, split, common, geometry)


def test_best_log_choice_is_auxiliary_and_does_not_enter_last_effects(experiment):
    arms, split, common, geometry = experiment
    baseline = p2.analyze_p2(arms, split, common, geometry)["last3000"]
    for index, arm in enumerate(p2.ARMS):
        log = next(row for row in arms[arm]["metrics"] if row.get("kind") == "eval" and row["step"] == 250)
        log["official_d3"] = log["selection_metric"] = .01 * (index + 1)
        arms[arm]["best"] = {"step": 250, "official_d3": log["official_d3"], "selection_definition": "official_d3"}
    changed = p2.analyze_p2(arms, split, common, geometry)
    assert changed["last3000"] == baseline
    assert all(value["step"] == 250 for value in changed["best_secondary"].values())


def test_joint_bootstrap_preserves_pairing_and_frame_vs_session_estimands():
    counts = np.arange(1, 12)
    ids = np.repeat([f"session{i:02d}" for i in range(11)], counts)
    base = np.repeat(np.arange(11) / 10., counts)
    values = np.stack([base + .5, base + .4, base + .7, base + .55], 1)
    first = p2.joint_session_bootstrap(values, ids)
    assert first == p2.joint_session_bootstrap(values, ids)
    primary = first["primary_frame_weighted"]["arms"]["c0t0"]["estimate"]
    secondary = first["secondary_session_equal"]["arms"]["c0t0"]["estimate"]
    assert primary == pytest.approx(values[:, 0].mean())
    assert secondary == pytest.approx(1.)
    assert primary != pytest.approx(secondary)
    interval = first["primary_frame_weighted"]["contrasts"]["C_given_T0"]["ci95_percentile"]
    assert interval == pytest.approx([-.1, -.1], abs=1e-14)
    assert first["primary_frame_weighted"]["contrasts"]["interaction_CxT"]["estimate"] == pytest.approx(-.05)
    with pytest.raises(ValueError, match="eleven"):
        p2.joint_session_bootstrap(values[:-11], ids[:-11])


def test_last_requires_all_steps_and_original_lr_schedule(experiment):
    metrics = experiment[0]["c0t0"]["metrics"]
    trace, last, selected, evaluations = p2.validate_metrics(metrics)
    assert len(trace) == 301 and last["step"] == selected["step"] == 3000 and len(evaluations) == 13
    bad = copy.deepcopy(metrics)
    bad[1]["lr"] *= .5
    with pytest.raises(ValueError, match="Logged LR"):
        p2.validate_metrics(bad)
    with pytest.raises(ValueError, match="log steps"):
        p2.validate_metrics(metrics[1:2] + metrics[3:])


def exit_fixture(tmp_path):
    run = tmp_path / "p2_c0t0_s0"
    command = ["python", "scripts/train_motiondrive_v2.py", "--gpu", "0", "--run-dir", str(run)]
    manifest = {"pid": 123456, "arguments": {"gpu": 0, "run_dir": str(run)}}
    job = {"gpu": 0, "trainer_command": command}
    process = {"pid": 123455}
    snapshot = {"exists": True, "belongs_to_child": True, "pid": 123456, "status": "completed", "step": 3000,
                "manifest_path": str(run / "manifest.json")}
    record = {"schema_version": 1, "supervisor_record_id": "identity", "child_pid": 123456, "supervisor_pid": 123455,
        "gpu": 0, "command": command, "run_dir": str(run), "trainer_manifest_path": str(run / "manifest.json"),
        "trainer_manifest": snapshot, "status": "process_exited", "actual_returncode": 0, "exit_code": 0,
        "termination_signal": None, "termination_signal_name": None, "outcome": "completed_cleanly",
        "supervisor_exit_code": 0, "trainer_reported_completed": True, "ended_at": "mock-ended"}
    return record, manifest, run, job, process


def test_actual_process_exit_zero_is_separate_from_completed_manifest(tmp_path):
    data = exit_fixture(tmp_path)
    assert p2.validate_exit(*data, alive=lambda _: False)["actual_returncode"] == 0
    with pytest.raises(ValueError, match="still exists"):
        p2.validate_exit(*data, alive=lambda _: True)
    data[0]["actual_returncode"] = -11
    data[0]["termination_signal"] = 11
    with pytest.raises(ValueError, match="Actual clean"):
        p2.validate_exit(*data, alive=lambda _: False)


def test_state_signature_checks_finite_and_includes_dtype_shape_bytes():
    signature, digest = p2.state_signature({"weight": torch.ones(2, 4), "counter": torch.tensor(5)})
    assert signature["weight"] == [[2, 4], "torch.float32"]
    assert p2.state_signature({"counter": torch.tensor(5), "weight": torch.ones(2, 4)})[1] == digest
    assert p2.state_signature({"weight": torch.ones(2, 4).bfloat16(), "counter": torch.tensor(5)})[1] != digest
    with pytest.raises(ValueError, match="Nonfinite"):
        p2.state_signature({"bad": torch.tensor([float("nan")])})


@pytest.mark.parametrize("field", ["d3", "point_l2", "cumulative_ade_1_2_3s", "longitudinal_error", "lateral_error",
                                  "weighted_abs_longitudinal_error", "weighted_abs_lateral_error"])
def test_coordinate_metric_guard_fixed_tolerance_no_silent_metric_replacement(experiment_template, field):
    record = copy.deepcopy(experiment_template[0]["c0t0"]["evaluation"]["conditions"]["normal"]["records"][0])
    original = copy.deepcopy(record)
    assert max(p2.validate_record_metrics(record).values()) < 1e-14
    assert record == original
    if isinstance(record[field], list):
        record[field][0] += 5e-6
    else:
        record[field] += 5e-6
    accepted = copy.deepcopy(record)
    assert p2.validate_record_metrics(record)[field] == pytest.approx(5e-6)
    assert record == accepted
    if isinstance(record[field], list):
        record[field][0] += 6e-6
    else:
        record[field] += 6e-6
    with pytest.raises(ValueError, match="fixed atol=1e-5"):
        p2.validate_record_metrics(record)


def test_load_launches_accepts_unchanged_subsets_but_not_changed_treatment(tmp_path):
    plan = json.loads((p2.ROOT / "configs/motiondrive_v2/p2_geometry_time_r1_s0.json").read_text())
    records = []
    for indices in ((1, 2, 3), (0,)):
        jobs, processes = [], []
        for index in indices:
            spec = plan["jobs"][index]
            c = p2.FACTORS[index][0]
            jobs.append({"name": spec["name"], "gpu": index, "explicit_argv": spec["argv"],
                         "inputs": {"checkpoint_init": {"sha256": p2.COMMON_SHA}, "split": {"sha256": p2.SPLIT_SHA},
                                    "supervision": {"sha256": p2.MANIFEST_SHAS[c]}}})
            processes.append({"name": spec["name"], "gpu": index, "pid": 1000 + index})
        record = {"status": "launched", "supervise": True, "tracked_dirty": "", "actual_git_sha": "same-training-sha",
                  "trainer_sha256": "same-trainer-sha", "supervisor_sha256": "same-supervisor-sha", "jobs": jobs, "processes": processes}
        path = tmp_path / f"launch_{len(records)}.json"
        path.write_text(json.dumps(record));records.append(path)
    loaded = p2.load_launches(records, plan)
    assert len(loaded) == 4
    payload = json.loads(records[0].read_text())
    payload["jobs"][0]["explicit_argv"][-1] = "nominal"
    records[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unchanged"):
        p2.load_launches(records, plan)


def test_geometry_edition_audit_checks_actual_scene_bytes_not_only_report_flags(tmp_path, monkeypatch, experiment):
    split = experiment[1]
    roots = [tmp_path / "old", tmp_path / "new"]
    for root in roots: root.mkdir()
    calibrations = [np.tile(np.eye(4, dtype=np.float32), (6, 1, 1)) for _ in range(2)]
    calibrations[1][3, 1, 2] += 2.
    for root, calibration in zip(roots, calibrations):
        np.savez(root / "calibration.npz", lidar2img=calibration)
    calibration_shas = tuple(p2.sha256(root / "calibration.npz") for root in roots)
    for c, root in enumerate(roots):
        (root / "supervision_manifest.json").write_text(json.dumps({"geometry_edition": "cache_meta_rear_wide_v2" if c else "original",
                    "canonical_calibration_sha256": calibration_shas[c]}))
        for scene in split["splits"]["train"] + split["splits"]["tune"]:
            for suffix in (".npz", ".json"):
                (root / (scene + suffix)).write_bytes((scene + suffix + "identical source bytes").encode())
    monkeypatch.setattr(p2, "CALIBRATION_SHAS", calibration_shas)
    monkeypatch.setattr(p2, "MANIFEST_SHAS", tuple(p2.sha256(root / "supervision_manifest.json") for root in roots))
    evidence = p2.audit_geometry(*roots, split)
    assert evidence["train_tune_scenes"] == 240 and len(evidence["scene_file_sha256"]) == 480
    assert not evidence["finalval_test_scene_files_read"]
    (roots[1] / "train000.npz").write_bytes(b"changed GT")
    with pytest.raises(ValueError, match="GT/history"):
        p2.audit_geometry(*roots, split)


def test_load_common_pins_existing_last_file_step_and_tensor_state(tmp_path, monkeypatch):
    path = tmp_path / "last.pth"
    state = {"xy.weight": torch.ones(2, 4)}
    checkpoint = {"step": 6000, "model": state, "manifest": {"model_config": {"goal_on": True, "state_on": True}}}
    torch.save(checkpoint, path)
    monkeypatch.setattr(p2, "COMMON_SHA", p2.sha256(path))
    result = p2.load_common(path)
    assert result["model_state_sha256"] == p2.state_signature(state)[1]
    assert result["tensor_signature"] == {"xy.weight": [[2, 4], "torch.float32"]}
    checkpoint["step"] = 5000
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="wrong SHA"):
        p2.load_common(path)
    monkeypatch.setattr(p2, "COMMON_SHA", p2.sha256(path))
    with pytest.raises(ValueError, match="LAST6000"):
        p2.load_common(path)


@pytest.mark.parametrize("damage", [None, "initial_optimizer", "last_step", "embedded_args", "last_nonfinite", "best_step"])
def test_load_arm_checks_actual_checkpoint_artifacts_before_analysis(tmp_path, monkeypatch, experiment, damage):
    item = experiment[0]["c0t0"]
    run = tmp_path / "p2_c0t0_s0"
    run.mkdir()
    manifest = copy.deepcopy(item["run_manifest"])
    manifest["arguments"]["run_dir"] = str(run)
    state = {"xy.weight": torch.ones(2, 4)}
    manifest["initial_model_state_sha256"] = p2.state_signature(state)[1]
    (run / "manifest.json").write_text(json.dumps(manifest))
    for name, step in (("initial", 0), ("last", 3000), ("best", 3000)):
        checkpoint = {"step": step, "model": copy.deepcopy(state), "manifest": copy.deepcopy(manifest), "optimizer": {"state": {}}}
        if damage == "initial_optimizer" and name == "initial": checkpoint["optimizer"]["state"] = {0: {"step": 1}}
        if damage == "last_step" and name == "last": checkpoint["step"] = 2999
        if damage == "embedded_args" and name == "last": checkpoint["manifest"]["arguments"]["lr"] = 1e-4
        if damage == "last_nonfinite" and name == "last": checkpoint["model"]["xy.weight"][0, 0] = float("nan")
        if damage == "best_step" and name == "best": checkpoint["step"] = 250
        torch.save(checkpoint, run / (name + ".pth"))
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in item["metrics"]) + "\n")
    evidence = copy.deepcopy(item["evaluation"])
    evidence["protocol"]["checkpoint_manifest"] = manifest
    evidence["protocol"]["arguments"]["checkpoint"] = str(run / "last.pth")
    evidence["protocol"]["checkpoint_sha256"] = p2.sha256(run / "last.pth")
    evidence["model_load"]["checkpoint_sha256"] = evidence["protocol"]["checkpoint_sha256"]
    protocol_path = tmp_path / "last.protocol.json"
    protocol_path.write_text(json.dumps(evidence["protocol"]))
    evidence["protocol_path"] = str(protocol_path)
    evidence["protocol_sha256"] = p2.sha256(protocol_path)
    evaluation_path = tmp_path / "last_eval.json"
    evaluation_path.write_text(json.dumps(evidence))
    supervisor_path = tmp_path / "supervisor.json"
    supervisor_path.write_text("{}")
    # Actual exit/PID checks are tested separately; no process queries in this file-loader fixture.
    monkeypatch.setattr(p2, "validate_exit", lambda *args, **kwargs: {"actual_returncode": 0, "trainer_pid_absent": True})
    launch = {"job": {"run_dir": str(run), "supervisor_record": str(supervisor_path)}, "process": {},
              "training_source": item["training_source"]}
    if damage is not None:
        with pytest.raises(ValueError):
            p2.load_arm("c0t0", run, evaluation_path, launch)
    else:
        loaded = p2.load_arm("c0t0", run, evaluation_path, launch)
        assert loaded["initial_tensor_sha256"] == p2.state_signature(state)[1]
        assert loaded["last_checkpoint_sha256"] == p2.sha256(run / "last.pth")
        assert loaded["best"]["step"] == 3000
        assert loaded["best"]["excluded_from_last_factorial_estimates"] is True
