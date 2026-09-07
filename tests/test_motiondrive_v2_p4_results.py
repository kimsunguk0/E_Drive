"""CPU-only schema/statistics tests; these are not P4 model evaluation results."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import analyze_motiondrive_v2_p4_results as results


@pytest.fixture(autouse=True)
def tiny_fixed_tune(monkeypatch):
    monkeypatch.setattr(results, "EXPECTED_N", 4)
    monkeypatch.setattr(results, "EXPECTED_SCENES", 2)
    monkeypatch.setattr(results, "EXPECTED_SESSIONS", 2)
    monkeypatch.setattr(results, "EXPECTED_FRAMES", (30, 35))
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def split_document():
    return {"splits": {"train": ["unused"], "tune": ["scene_a", "scene_b"],
                       "final_val": ["forbidden"]},
            "scene_to_session": {"scene_a": "session_a", "scene_b": "session_b",
                                 "unused": "train_session", "forbidden": "final_session"}}


def record(index, seed):
    scenario = "scene_a" if index < 2 else "scene_b"
    session = "session_a" if index < 2 else "session_b"
    frame = (30, 35)[index % 2]
    gt = np.asarray([[index + .2 * step, -.1 * step] for step in range(1, 7)], np.float64)
    error = np.asarray([[.03 * (seed + 1) * step, -.02 * (index + 1)]
                        for step in range(1, 7)], np.float64)
    pred = gt + error
    point = np.linalg.norm(error, axis=-1)
    d3 = float(point @ results.WEIGHTS)
    state = ([0., 0., 0., 0., 0., 0.] if index < 2
             else [2., 0., 0., 0., 0., 0.])
    return {"row": (100, 101, 200, 201)[index], "scenario": scenario,
            "session": session, "frame": frame, "pred_abs_xy": pred.tolist(),
            "gt_abs_xy": gt.tolist(), "d3": d3, "point_l2": point.tolist(),
            "cumulative_ade_1_2_3s": [float(point[:n].mean()) for n in (2, 4, 6)],
            "longitudinal_error": error[:, 0].tolist(),
            "lateral_error": error[:, 1].tolist(),
            "weighted_abs_longitudinal_error": float(np.abs(error[:, 0]) @ results.WEIGHTS),
            "weighted_abs_lateral_error": float(np.abs(error[:, 1]) @ results.WEIGHTS),
            "gt_state": state, "gt_state_valid": [True] * 6,
            "bucket": "stop" if index < 2 else "cruise"}


def condition(records):
    d3 = [item["d3"] for item in records]
    summary = {"n": len(records), "n_scenes": len({r["scenario"] for r in records}),
               "n_sessions": len({r["session"] for r in records}),
               "official_d3": float(np.mean(d3))}
    buckets = {}
    for name in results.BUCKET_NAMES:
        selected = [item["d3"] for item in records if item["bucket"] == name]
        buckets[name] = {"n": len(selected),
                         "official_d3": float(np.mean(selected)) if selected else None}
    return {"time_input": "nominal", "summary": summary, "buckets": buckets,
            "records": records}


def report(seed, *, p4=True, extra_conditions=False):
    records = [record(index, seed if p4 else 0) for index in range(4)]
    rows_sha = digest(np.asarray([item["row"] for item in records], dtype="<i8").tobytes())
    step = results.P4_STEP if p4 else results.C0_STEP
    checkpoint_sha = digest(f"checkpoint-{seed}-{step}".encode())
    requested = (["normal"] if not extra_conditions else
                 ["normal", "image_shuffle", "repeat_current", "reverse_history"])
    training_manifest = {
        "status": "running", "git_sha": results.P4_GIT_SHA,
        "arguments": {"phase": "joint", "steps": results.P4_STEP, "batch": 16,
                      "microbatch": 2, "eval_batch": 4, "workers": 4, "seed": seed,
                      "precision": "bf16", "time_input": "nominal", "eval_stride": 5,
                      "eval_split": "tune", "max_eval_samples": 0, "bn_policy": "fixed"},
        "model_config": {"backbone_arch": "resnet50", "goal_on": True,
                         "state_on": True, "motion_input_mode": "low_feature",
                         "n_history": 4, "plan_output_scale": [10., 5.]},
        "loss_weights": {"plan": 1.}, "split_sha256": results.RAWTIME_SPLIT_SHA256,
        "supervision_manifest_sha256": results.C1_SUPERVISION_SHA256,
        "data_counts": {"train": 54810, "eval": 4}, "eval_rows_sha256": rows_sha}
    protocol = {
        "status": "preregistered_before_any_forward", "schema_version": 1,
        "arguments": {"split": "tune", "frame_stride": 5, "max_samples": 0,
                      "scenes": None, "batch": 4, "workers": 4, "precision": "bf16",
                      "time_input": "nominal", "seed": seed, "conditions": requested,
                      "device": "cuda:0"},
        "checkpoint_sha256": checkpoint_sha, "checkpoint_step": step,
        "checkpoint_manifest": training_manifest,
        "data": {"split_sha256": results.RAWTIME_SPLIT_SHA256,
                 "receiver_count": 4, "receiver_rows_sha256": rows_sha,
                 "supervision_sha256": {
                     "supervision_manifest.json": results.C1_SUPERVISION_SHA256,
                     "calibration.npz": results.C1_CANONICAL_SHA256}},
        "source": {"git_sha": results.P4_GIT_SHA, "tracked_changes": [],
                   "file_sha256": {"scripts/evaluate_motiondrive_v2_planning.py":
                                       digest(b"fixed evaluator")}},
        "time_input": "nominal", "selection_performed": False,
        "final_val_accessed": False}
    normal = condition(records)
    conditions = {name: copy.deepcopy(normal) for name in requested}
    value = {"status": "completed", "protocol": protocol,
             "protocol_sha256": results.protocol_sha(protocol),
             "model_load": {"checkpoint_sha256": checkpoint_sha},
             "time_input": "nominal", "precision": "bf16",
             "precision_requested": "bf16", "conditions": conditions,
             "selection_performed": False, "final_val_accessed": False}
    return value


def resign(document):
    document["protocol_sha256"] = results.protocol_sha(document["protocol"])


def pair():
    return report(0), report(1), split_document()


def test_valid_pair_recomputes_metrics_and_keeps_seed_raw_results():
    s0, s1, split = pair()
    value = results.analyze(s0, s1, split, repeats=200, bootstrap_seed=7)
    assert len(value["per_frame_p4_results"]) == 4
    assert value["seeds"]["s0"]["all"]["n"] == 4
    assert set(value["seeds"]["s0"]["scenarios"]) == {"scene_a", "scene_b"}
    assert set(value["seeds"]["s0"]["sessions"]) == {"session_a", "session_b"}
    assert set(value["seeds"]["s0"]["buckets"]) == set(results.BUCKET_NAMES)
    assert value["seeds"]["s0"]["buckets"]["unknown"] == {"n": 0, "official_d3": None}
    assert value["two_seed_description"]["n_seeds"] == 2
    assert not value["two_seed_description"]["distribution_established_by_two_seeds"]
    assert not value["two_seed_description"]["best_of_two_selection_performed"]
    assert value["bootstrap"]["unit"].startswith("eleven sessions")
    assert not value["bootstrap"]["frame_iid_bootstrap"]
    assert value["seeds"]["s0"]["session_cluster_absolute_interval"]["clusters"] == 2
    assert "estimate" in value["seeds"]["s0"]["session_cluster_absolute_interval"]
    assert not value["model_forward_performed"] and not value["gpu_used"]
    assert value["lineage"]["seed_evaluation_sources_exactly_equal"]
    assert value["cpu_unit_tests_are_not_model_results"]
    assert not value["training_completion_receipt"]["completed_training_sidecar_verified_here"]
    assert not value["training_completion_receipt"]["supervisor_exit_zero_asserted"]
    json.dumps(value, allow_nan=False)


@pytest.mark.parametrize("field,value", [
    ("status", "running"), ("selection_performed", True), ("final_val_accessed", True),
    ("time_input", "raw"), ("precision", "fp32"), ("precision_requested", "fp32")])
def test_top_level_protocol_contract_rejects_bad_values(field, value):
    s0, s1, split = pair()
    s0[field] = value
    with pytest.raises(ValueError):
        results.analyze(s0, s1, split, repeats=100, bootstrap_seed=0)


@pytest.mark.parametrize("path,value", [
    (("status",), "completed_after_forward"),
    (("arguments", "batch"), 1), (("arguments", "precision"), "fp32"),
    (("arguments", "time_input"), "raw"), (("arguments", "device"), "cpu"),
    (("arguments", "frame_stride"), 1), (("arguments", "scenes"), ["scene_a"]),
    (("checkpoint_step",), 5999), (("data", "receiver_count"), 3),
    (("data", "split_sha256"), "0" * 64), (("source", "git_sha"), "not-a-git-sha"),
    (("source", "tracked_changes"), [" M scripts/x.py"]),
])
def test_nested_p4_protocol_contract_rejects_bad_values(path, value):
    s0, s1, split = pair()
    target = s0["protocol"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    resign(s0)
    with pytest.raises(ValueError):
        results.analyze(s0, s1, split, repeats=100, bootstrap_seed=0)


@pytest.mark.parametrize("section,key,value", [
    ("arguments", "phase", "pretrain"), ("arguments", "steps", 5999),
    ("arguments", "batch", 8), ("arguments", "microbatch", 1),
    ("arguments", "bn_policy", "train"), ("model_config", "backbone_arch", "resnet18"),
    ("model_config", "goal_on", False), ("model_config", "state_on", False),
    ("model_config", "motion_input_mode", "raster"),
    ("model_config", "plan_output_scale", [1., 1.]), ("loss_weights", "plan", 0.),
])
def test_checkpoint_manifest_p4_identity_is_strict(section, key, value):
    s0, s1, split = pair()
    s0["protocol"]["checkpoint_manifest"][section][key] = value
    resign(s0)
    with pytest.raises(ValueError):
        results.analyze(s0, s1, split, repeats=100, bootstrap_seed=0)


def test_protocol_sha_and_checkpoint_sha_relations_are_strict():
    s0, s1, split = pair()
    s0["protocol_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="protocol"):
        results.analyze(s0, s1, split, repeats=100, bootstrap_seed=0)
    s0, s1, split = pair()
    s0["model_load"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint SHA"):
        results.analyze(s0, s1, split, repeats=100, bootstrap_seed=0)


def test_evaluation_git_can_differ_from_training_but_must_match_between_seeds():
    s0, s1, split = pair()
    evaluation_git = "a" * 40
    for document in (s0, s1):
        document["protocol"]["source"]["git_sha"] = evaluation_git
        resign(document)
    value = results.analyze(s0, s1, split, repeats=100, bootstrap_seed=0)
    assert value["lineage"]["training_git_sha"] == results.P4_GIT_SHA
    assert value["lineage"]["evaluation_git_sha"] == evaluation_git
    s1["protocol"]["source"]["file_sha256"]["scripts/evaluate_motiondrive_v2_planning.py"] = digest(b"other")
    resign(s1)
    with pytest.raises(ValueError, match="same evaluator source"):
        results.analyze(s0, s1, split, repeats=100, bootstrap_seed=0)


@pytest.mark.parametrize("change", ["shape", "nan", "wrong_d3", "wrong_axis", "wrong_bucket",
                                     "duplicate_identity", "missing_state"])
def test_per_frame_predictions_and_errors_fail_closed(change):
    s0, s1, split = pair()
    rows = s0["conditions"]["normal"]["records"]
    if change == "shape":
        rows[0]["pred_abs_xy"] = rows[0]["pred_abs_xy"][:5]
    elif change == "nan":
        rows[0]["gt_abs_xy"][0][0] = float("nan")
    elif change == "wrong_d3":
        rows[0]["d3"] += 1
    elif change == "wrong_axis":
        rows[0]["longitudinal_error"][0] += 1
    elif change == "wrong_bucket":
        rows[0]["bucket"] = "cruise"
    elif change == "duplicate_identity":
        rows[1] = copy.deepcopy(rows[0])
    else:
        rows[0].pop("gt_state_valid")
    with pytest.raises((ValueError, KeyError)):
        results.analyze(s0, s1, split, repeats=100, bootstrap_seed=0)


def test_seed_pair_requires_exact_order_and_gt_state_labels():
    s0, s1, split = pair()
    rows = s1["conditions"]["normal"]["records"]
    rows.reverse()
    row_sha = digest(np.asarray([item["row"] for item in rows], dtype="<i8").tobytes())
    s1["protocol"]["data"]["receiver_rows_sha256"] = row_sha
    s1["protocol"]["checkpoint_manifest"]["eval_rows_sha256"] = row_sha
    resign(s1)
    with pytest.raises(ValueError, match="identities/order"):
        results.analyze(s0, s1, split, repeats=100, bootstrap_seed=0)
    s0, s1, split = pair()
    s1["conditions"]["normal"]["records"][2]["gt_state"][0] = 2.25
    with pytest.raises(ValueError, match="GT/mask/bucket"):
        results.analyze(s0, s1, split, repeats=100, bootstrap_seed=0)


def test_summary_and_bucket_receipts_are_recomputed():
    s0, s1, split = pair()
    s0["conditions"]["normal"]["summary"]["official_d3"] += .1
    with pytest.raises(ValueError, match="summary"):
        results.analyze(s0, s1, split, repeats=100, bootstrap_seed=0)
    s0, s1, split = pair()
    s0["conditions"]["normal"]["buckets"]["stop"]["n"] = 3
    with pytest.raises(ValueError, match="bucket"):
        results.analyze(s0, s1, split, repeats=100, bootstrap_seed=0)


def test_bootstrap_is_deterministic_cluster_resampling_with_frame_weighting():
    values = np.asarray([0., 0., 10., 10.])
    sessions = ["short", "long", "long", "long"]
    first = results.absolute_cluster_interval(values, sessions, repeats=500, seed=3)
    second = results.absolute_cluster_interval(values, sessions, repeats=500, seed=3)
    assert first == second and first["estimate"] == pytest.approx(5.)
    assert first["clusters"] == 2 and "frame IID" in first["method"]


def test_historical_c0_allows_original_extra_conditions_but_is_never_causal(monkeypatch):
    s0, s1, split = pair()
    c0 = report(0, p4=False, extra_conditions=True)
    c0_d3 = c0["conditions"]["normal"]["summary"]["official_d3"]
    monkeypatch.setattr(results, "C0_D3", c0_d3)
    value = results.analyze(s0, s1, split, repeats=100, bootstrap_seed=4, c0=c0)
    historical = value["historical_c0"]
    assert historical["all"]["official_d3"] == c0_d3
    assert not historical["causal_or_same_condition_comparison"]
    assert "geometry" in historical["not_matched_for"]
    assert set(historical["buckets"]) == set(results.BUCKET_NAMES)


def test_preserved_supervisor_rc1_is_recorded_and_never_relabelled_exit0():
    s0, s1, split = pair()
    outcome = "post_training_global_head_guard_rc1_preserved"
    value = results.analyze(s0, s1, split, repeats=100, bootstrap_seed=0,
                            training_integrity={"s0": outcome, "s1": outcome})
    receipt = value["training_completion_receipt"]
    assert receipt["integrity_exceptions_reported"] == ["s0", "s1"]
    assert not receipt["supervisor_exit_zero_asserted"]
    assert receipt["operator_reported_supervisor_outcome"] == {"s0": outcome, "s1": outcome}
    with pytest.raises(ValueError, match="integrity outcome"):
        results.analyze(s0, s1, split, repeats=100, bootstrap_seed=0,
                        training_integrity={"s0": "supervisor_exit0", "s1": "rc0"})


@pytest.mark.parametrize("change", ["step", "d3", "gt", "missing_normal"])
def test_historical_c0_identity_and_recorded_reference_are_strict(monkeypatch, change):
    s0, s1, split = pair()
    c0 = report(0, p4=False, extra_conditions=True)
    monkeypatch.setattr(results, "C0_D3",
                        c0["conditions"]["normal"]["summary"]["official_d3"])
    if change == "step":
        c0["protocol"]["checkpoint_step"] = 2999
        resign(c0)
    elif change == "d3":
        monkeypatch.setattr(results, "C0_D3", results.C0_D3 + 1)
    elif change == "gt":
        c0["conditions"]["normal"]["records"][0]["gt_state"][0] = .1
    else:
        c0["conditions"].pop("normal")
    with pytest.raises((ValueError, KeyError)):
        results.analyze(s0, s1, split, repeats=100, bootstrap_seed=0, c0=c0)


def test_read_pinned_json_rejects_wrong_sha_duplicate_nonfinite_and_symlink(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"ok": true}')
    with pytest.raises(ValueError, match="SHA256"):
        results.read_pinned_json(path, "0" * 64)
    for raw in (b'{"a": 1, "a": 2}', b'{"a": NaN}'):
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            results.read_pinned_json(path, digest(raw))
    target = tmp_path / "target.json"
    target.write_text("{}")
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="Ordinary"):
        results.read_pinned_json(path, digest(b"{}"))


def test_cli_pins_inputs_writes_once_and_marks_no_gpu(tmp_path, monkeypatch, capsys):
    split = split_document()
    split_raw = (json.dumps(split, indent=2) + "\n").encode()
    split_sha = digest(split_raw)
    monkeypatch.setattr(results, "RAWTIME_SPLIT_SHA256", split_sha)
    s0, s1 = report(0), report(1)
    paths = {}
    for name, value in (("s0", s0), ("s1", s1), ("split", split)):
        path = tmp_path / f"{name}.json"
        raw = split_raw if name == "split" else (json.dumps(value, indent=2) + "\n").encode()
        path.write_bytes(raw)
        paths[name] = (path, digest(raw))
    out = tmp_path / "analysis.json"
    argv = ["--s0-report", str(paths["s0"][0]), "--expected-s0-sha256", paths["s0"][1],
            "--s1-report", str(paths["s1"][0]), "--expected-s1-sha256", paths["s1"][1],
            "--split-manifest", str(paths["split"][0]), "--bootstrap-repeats", "100",
            "--bootstrap-seed", "5", "--out", str(out)]
    assert results.main(argv) == 0
    value = json.loads(out.read_text())
    assert value["input_sha256"] == {name: item[1] for name, item in paths.items()}
    assert results.valid_sha(value["analysis_script_sha256"])
    assert not value["gpu_used"] and not value["final_val_accessed"]
    assert json.loads(capsys.readouterr().out)["status"] == value["status"]
    previous = out.read_bytes()
    with pytest.raises(ValueError, match="Output"):
        results.main(argv)
    assert out.read_bytes() == previous


def test_cli_requires_c0_path_and_sha_together(tmp_path):
    with pytest.raises(ValueError, match="C0 path"):
        results.main(["--s0-report", "x", "--expected-s0-sha256", "0" * 64,
                      "--s1-report", "y", "--expected-s1-sha256", "0" * 64,
                      "--split-manifest", "z", "--c0-report", "c0", "--out", str(tmp_path / "o")])


def test_source_text_keeps_actual_training_completion_external_and_final_forbidden():
    source = Path(results.__file__).read_text()
    assert "training_completion_receipt" in source
    assert "completed_training_sidecar_verified_here\": False" in source
    assert "final validation never accessed" in source
    assert "best_of_two_selection_performed\": False" in source
