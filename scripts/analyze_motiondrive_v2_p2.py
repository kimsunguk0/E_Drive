#!/usr/bin/env python3
"""P2 C x T: strict LAST3000, paired tune-session analysis (CPU only).

Unlike P1, geometry and time policies MUST differ according to arm. This module
does not import the P1 analyzer, run a model, or reselect a checkpoint. Inputs
are original run/checkpoint/log/supervisor artifacts and nested planning-eval
JSONs. BEST stays in a separate descriptive table; never in LAST contrasts.
Training-source equality and evaluation-source equality are separate gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_grouped_split_v2 import validate_manifest

ARMS = ("c0t0", "c1t0", "c0t1", "c1t1")
FACTORS = ((0, 0), (1, 0), (0, 1), (1, 1))
STEP, REPEATS, SEED = 3000, 10000, 20260907
RECORD_METRIC_ATOL = 1e-5  # Preregistered FP64 coordinate replay vs stored FP32 metrics; never fitted to results.
COMMON_SHA = "e2a0e2dc9f3320f5e9744e3e4ef0b77bf9dffffee33aed1274b610bd6faac80e"
SPLIT_SHA = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
MANIFEST_SHAS = ("073d9d2eb89956cac5c3d78b91622f5e397cfa116b1ed8f9a9f639ec3003a8d0",
                 "ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93")
CALIBRATION_SHAS = ("a6e29e9ecdf596560161b5c68806445e0d9139dded7c7bb403842af5a8a93cbb",
                   "8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961")
CONTRASTS = {"C_given_T0": (-1., 1., 0., 0.), "C_given_T1": (0., 0., -1., 1.),
             "T_given_C0": (-1., 0., 1., 0.), "T_given_C1": (0., -1., 0., 1.),
             "interaction_CxT": (1., -1., -1., 1.)}
MODEL_INPUTS = ["images", "history_images", "lidar2img", "history_transforms", "time_offsets", "goal_xy"]
CONDITIONS = {"normal", "image_shuffle", "repeat_current", "reverse_history"}
EXPECTED_ARGS = dict(phase="joint", steps=3000, batch=16, eval_batch=4, workers=4, seed=0,
    precision="bf16", lr=5e-5, backbone_lr=5e-6, weight_decay=.01, grad_clip=5., warmup=100,
    alpha_occ=.2, alpha_lane=.2, alpha_motion=.2, uncertainty=1, bn_policy="fixed",
    motion_input_mode="low_feature", train_stride=1, eval_stride=5, eval_split="tune",
    max_train_samples=0, max_eval_samples=0, eval_every=250, save_every=500, log_every=10,
    goal_on=1, state_on=1, plan_output_scale=[10., 5.])
EMBEDDED_KEYS = ("git_sha", "arguments", "model_config", "loss_weights", "split_sha256",
    "supervision_manifest_sha256", "initial_model_state_sha256", "initial_parameter_count",
    "train_rows_sha256", "eval_rows_sha256", "load_report", "bn_training", "data_counts",
    "time_input", "time_input_policy")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_form(value):
    return json.loads(json.dumps(value, allow_nan=False))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def state_signature(state):
    import torch
    require(isinstance(state, dict) and bool(state), "Missing model tensor state")
    signature, digest = {}, hashlib.sha256()
    for name, value in sorted(state.items()):
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu", "Checkpoint must be CPU tensors")
        require(bool(torch.isfinite(value).all()), f"Nonfinite checkpoint tensor: {name}")
        signature[name] = [list(value.shape), str(value.dtype)]
        digest.update((name + "\0" + str(value.dtype) + "\0" + str(tuple(value.shape)) + "\0").encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return signature, digest.hexdigest()


def check_split(split):
    validate_manifest(split)
    for name, count, sessions in (("train", 203, 72), ("tune", 37, 11)):
        scenes = split["splits"][name]
        require(len(scenes) == len(set(scenes)) == count, f"Wrong {name} scene count")
        require(len({split["scene_to_session"][s] for s in scenes}) == sessions, f"Wrong {name} session count")


def check_time(node, expected, location, *, policy=False):
    require(isinstance(node, dict) and node.get("time_input") == expected,
            f"Missing/conflicting explicit time_input at {location}; no legacy raw fallback")
    if policy:
        entry = node.get("time_input_policy")
        require(isinstance(entry, dict) and entry.get("mode") == expected, f"Missing/conflicting time policy at {location}")
        require(entry.get("nominal_history_seconds") == [.1, .2, .5, 1.] and entry.get("nominal_dtype") == "float32",
                f"Incorrect nominal tensor contract at {location}")
        require(entry.get("supervision_and_dataset_modified") is False or entry.get("dataset_and_labels_modified") is False,
                f"Missing unchanged-supervision time policy at {location}")


def validate_metrics(rows):
    training = [row for row in rows if row.get("kind") == "train"]
    expected_steps = [1, *range(10, STEP + 1, 10)]
    require([row.get("step") for row in training] == expected_steps, "Missing/duplicate/nonstandard training log steps")
    trace = []
    for row in training:
        require(row.get("epoch") == 0 and bool(row.get("sample_order_sha256")), "P2 exposure/order evidence missing")
        for name in ("total", "plan_d3", "occ_bce", "lane_bce", "history", "state", "stop_bce", "motion", "grad_norm", "lr"):
            require(name in row and np.isfinite(row[name]), f"Nonfinite/missing training metric: {name}")
        require(row["plan_d3"] >= 0 and row["grad_norm"] >= 0, "Invalid nonnegative metric")
        step = row["step"] - 1
        warm = min(1., (step + 1) / 100.)
        progress = max(0., (step - 100) / (STEP - 100))
        expected_lr = 5e-5 * warm * .5 * (1. + np.cos(np.pi * min(1., progress)))
        require(np.isclose(row["lr"], expected_lr, atol=1e-15, rtol=0), "Logged LR differs from warmup100/cosine3000")
        trace.append([row["step"], row["epoch"], row["sample_order_sha256"]])
    evaluations = [row for row in rows if row.get("kind") == "eval"]
    require([row.get("step") for row in evaluations] == list(range(250, STEP + 1, 250)), "Expected all twelve tune evaluations")
    for row in evaluations:
        require(row.get("selection_definition") == "official_d3" and row.get("n") == 1998 and row.get("n_sessions") == 11,
                "BEST must use full tune official D3")
        require(np.isfinite(row["selection_metric"]) and row["selection_metric"] == row.get("official_d3") and row["official_d3"] >= 0,
                "Invalid checkpoint selection metric")
    initial = [row for row in rows if row.get("kind") == "initial_eval"]
    require(len(initial) == 1 and initial[0].get("step") == 0, "Fresh weights-only run must have initial evaluation")
    return trace, evaluations[-1], min(evaluations, key=lambda row: row["selection_metric"]), [*initial, *evaluations]


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def validate_exit(record, manifest, run_dir, launch_job, launch_process, *, alive=pid_alive):
    run = Path(run_dir).resolve()
    child = record.get("child_pid")
    require(type(child) is int and child > 0 and child == manifest.get("pid"), "Supervisor/trainer child PID mismatch")
    require(record.get("schema_version") == 1 and record.get("supervisor_record_id"), "Missing supervisor identity")
    require(record.get("supervisor_pid") == launch_process.get("pid"), "Launch/supervisor PID mismatch")
    require(record.get("gpu") == manifest["arguments"]["gpu"] == launch_job["gpu"], "Supervisor/run GPU mismatch")
    require(record.get("command") == launch_job.get("trainer_command"), "Supervisor trainer command differs from launch")
    require(Path(record.get("run_dir", "")).resolve() == run and Path(manifest["arguments"]["run_dir"]).resolve() == run,
            "Supervisor/run path mismatch")
    expected_manifest = run / "manifest.json"
    require(Path(record.get("trainer_manifest_path", "")).resolve() == expected_manifest, "Supervisor manifest path mismatch")
    snapshot = record.get("trainer_manifest") or {}
    require(snapshot.get("exists") is True and snapshot.get("belongs_to_child") is True and snapshot.get("pid") == child
            and snapshot.get("status") == "completed" and snapshot.get("step") == STEP
            and Path(snapshot.get("manifest_path", "")).resolve() == expected_manifest, "Incomplete supervisor trainer snapshot")
    require(record.get("status") == "process_exited" and type(record.get("actual_returncode")) is int
            and record["actual_returncode"] == 0 and record.get("exit_code") == 0
            and record.get("termination_signal") is None and record.get("termination_signal_name") is None
            and record.get("outcome") == "completed_cleanly" and record.get("supervisor_exit_code") == 0
            and record.get("trainer_reported_completed") is True and record.get("ended_at"),
            "Actual clean OS exit not proven; completed manifest alone is insufficient")
    require(not alive(child), "Trainer PID still exists; do not analyze a running process")
    return {"actual_returncode": 0, "trainer_pid_absent": True, "child_pid": child,
            "supervisor_pid": record["supervisor_pid"], "outcome": "completed_cleanly"}


def normalize_evaluation(evidence, arm, manifest):
    c, t = FACTORS[ARMS.index(arm)]
    timing = "nominal" if t else "raw"
    require(evidence.get("status") == "completed" and evidence.get("selection_performed") is False
            and evidence.get("final_val_accessed") is False, "Incomplete/non-tune planning evaluation")
    protocol = evidence.get("protocol", {})
    args, data = protocol.get("arguments", {}), protocol.get("data", {})
    check_time(evidence, timing, "evaluation")
    check_time(protocol, timing, "protocol", policy=True)
    check_time(args, timing, "protocol.arguments")
    require(protocol.get("status") == "preregistered_before_any_forward" and protocol.get("schema_version") == 1,
            "Missing preregistered planning protocol")
    require(protocol.get("model_input_whitelist") == MODEL_INPUTS, "Model input whitelist differs")
    require(protocol.get("nominal_waypoint_seconds") == [.5, 1., 1.5, 2., 2.5, 3.], "Changed planning target horizon")
    expected = dict(split="tune", frame_stride=5, max_samples=0, scenes=None, batch=4, workers=4, seed=0, precision="bf16")
    require(all(args.get(key) == value for key, value in expected.items()) and args.get("device") in {f"cuda:{i}" for i in range(4)}
            and evidence.get("precision") == "bf16", "Planning evaluation is not full tune1998/batch4/bf16")
    require(data.get("receiver_count") == 1998 and data.get("split_sha256") == SPLIT_SHA, "Evaluation split/count mismatch")
    require(data.get("receiver_rows_sha256") == manifest.get("eval_rows_sha256"), "Eval sample order differs from training eval")
    supervised = data.get("supervision_sha256", {})
    require(supervised.get("supervision_manifest.json") == MANIFEST_SHAS[c]
            and supervised.get("calibration.npz") == CALIBRATION_SHAS[c], "C arm has wrong geometry/manifest edition")
    require(Path(args.get("supervision_root", "")).name == ("train_tune_geometry_v2" if c else "train_tune_rawtime"),
            "Evaluation supervision path disagrees with C arm")
    require(protocol.get("checkpoint_step") == STEP and Path(args.get("checkpoint", "")).name == "last.pth",
            "Primary evaluation must be LAST3000, not BEST")
    load = evidence.get("model_load", {})
    require(load.get("explicit_overrides") == {} and load.get("checkpoint_sha256") == protocol.get("checkpoint_sha256"),
            "Evaluation model override or checkpoint SHA conflict")
    embedded = protocol.get("checkpoint_manifest", {})
    require(all(json_form(embedded.get(key)) == json_form(manifest.get(key)) for key in EMBEDDED_KEYS),
            "Planning protocol checkpoint manifest differs from run")
    conditions = evidence.get("conditions", {})
    require("normal" in conditions and set(conditions) <= CONDITIONS, "Missing normal/unknown image condition")
    require(args.get("conditions") == list(conditions), "Protocol condition order differs from observed evaluation")
    for name, result in conditions.items():
        check_time(result, timing, f"conditions.{name}")
    source = protocol.get("source", {})
    require(source.get("git_sha") and source.get("tracked_changes") == [] and source.get("file_sha256"), "Missing/dirty evaluation SOURCE")
    for name in ("scripts/evaluate_motiondrive_v2_planning.py", "scripts/motiondrive_v2_data.py", "scripts/motiondrive_v2_training.py",
                 "scripts/train_motiondrive_v2.py", "models/motiondrive_v2/model.py"):
        require(bool(source["file_sha256"].get(name)), f"Missing evaluation core source SHA: {name}")
    return {"protocol": protocol, "conditions": conditions, "evaluation_source": source,
            "checkpoint_sha256": protocol["checkpoint_sha256"], "checkpoint_path": args["checkpoint"],
            "data": data, "time_input": timing}


def validate_record_metrics(record):
    """Check stored metrics against coordinates; do NOT replace any stored score."""
    pred, gt = np.asarray(record.get("pred_abs_xy"), np.float64), np.asarray(record.get("gt_abs_xy"), np.float64)
    require(pred.shape == gt.shape == (6, 2) and np.isfinite(pred).all() and np.isfinite(gt).all(), "Invalid absolute planning coordinates")
    error = pred - gt
    distance = np.linalg.norm(error, axis=1)
    weights = np.array([11., 11., 5., 5., 2., 2.]) / 36.
    recomputed = {"point_l2": distance, "d3": distance @ weights,
        "cumulative_ade_1_2_3s": np.array([distance[:n].mean() for n in (2, 4, 6)]),
        "longitudinal_error": error[:, 0], "lateral_error": error[:, 1],
        "weighted_abs_longitudinal_error": np.abs(error[:, 0]) @ weights,
        "weighted_abs_lateral_error": np.abs(error[:, 1]) @ weights}
    differences = {}
    for name, expected in recomputed.items():
        observed = np.asarray(record.get(name), np.float64)
        require(observed.shape == np.shape(expected) and np.isfinite(observed).all(), f"Invalid stored metric: {name}")
        difference = float(np.max(np.abs(observed - expected)))
        require(difference <= RECORD_METRIC_ATOL, f"Stored {name} disagrees with absolute coordinates at fixed atol=1e-5, rtol=0")
        differences[name] = difference
    return differences


def validate_records(result, split):
    expected = {(scene, frame) for scene in split["splits"]["tune"] for frame in range(30, 300, 5)}
    records, labels, keys, values, sessions, rows = result["records"], [], [], [], [], []
    seen = set()
    for record in records:
        require(type(record.get("frame")) is int, "Frame identity must be an integer")
        key = (record["scenario"], record["frame"])
        require(key in expected and key not in seen, "Duplicate or non-tune/stride5 frame")
        seen.add(key)
        require(record["session"] == split["scene_to_session"][key[0]], "Wrong rawtime session identity")
        require(type(record.get("row")) is int, "Missing exact cache row identity")
        require("d3" in record and np.isfinite(record["d3"]) and record["d3"] >= 0, "Invalid per-frame D3")
        if "official_d3" in record:
            require(record["official_d3"] == record["d3"], "Conflicting D3 fields")
        for name, shape in (("gt_abs_xy", (6, 2)), ("pred_abs_xy", (6, 2)), ("point_l2", (6,)),
                            ("cumulative_ade_1_2_3s", (3,))):
            arr = np.asarray(record.get(name), float)
            require(arr.shape == shape and np.isfinite(arr).all(), f"Invalid planning field {name}")
        require(len(record.get("gt_state", [])) == len(record.get("gt_state_valid", [])) == 6, "Missing GT-state/valid identity")
        require(record.get("bucket") in ("stop", "accel", "decel", "cruise", "unknown"), "Missing canonical state bucket")
        for name in ("weighted_abs_longitudinal_error", "weighted_abs_lateral_error"):
            require(np.isfinite(record.get(name, np.nan)) and record[name] >= 0, f"Invalid {name}")
        validate_record_metrics(record)
        labels.append({name: record[name] for name in ("scenario", "frame", "session", "row", "gt_abs_xy", "gt_state", "gt_state_valid", "bucket")})
        keys.append(key);values.append(float(record["d3"]));sessions.append(record["session"]);rows.append(record["row"])
    require(len(records) == 1998 and set(keys) == expected, "Full tune37 x 54 =1998 frames required")
    summary = result["summary"]
    require(summary.get("n") == 1998 and summary.get("n_sessions") == 11 and summary.get("n_scenes") == 37, "Summary counts differ")
    require(np.isclose(np.mean(values), summary.get("official_d3", np.nan), atol=1e-12, rtol=0), "D3 summary/records disagree")
    session_means = {session: float(np.mean([v for v, s in zip(values, sessions) if s == session])) for session in sorted(set(sessions))}
    require(summary.get("session_d3") == session_means, "Session D3 summary/records disagree")
    require(np.isclose(np.mean(list(session_means.values())), summary.get("session_mean_d3", np.nan), atol=1e-12, rtol=0),
            "Session-equal aggregate mismatch")
    row_sha = hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()
    return keys, np.asarray(values), sessions, labels, row_sha


def validate_experiment(arms, split, common, geometry):
    check_split(split)
    require(set(arms) == set(ARMS), "All four C/T arms are required")
    require(geometry.get("train_tune_scenes") == 240 and geometry.get("all_scene_npz_json_bitwise_preserved") is True
            and geometry.get("other_five_calibrations_bitwise_preserved") is True
            and geometry.get("manifest_sha256") == list(MANIFEST_SHAS)
            and geometry.get("calibration_sha256") == list(CALIBRATION_SHAS), "Missing complete geometry/GT preservation audit")
    require(set(geometry.get("scene_file_sha256", {})) == {s + ext for name in ("train", "tune")
            for s in split["splits"][name] for ext in (".npz", ".json")}, "Incomplete train/tune scene-file equality evidence")
    configs, arguments, sources, eval_sources, traces, label_sets, matrices, session_ids = [], [], [], [], [], [], [], []
    keys = None
    for i, (arm, (c, t)) in enumerate(zip(ARMS, FACTORS)):
        data, timing = arms[arm], "nominal" if t else "raw"
        manifest, args = data["run_manifest"], data["run_manifest"]["arguments"]
        require(manifest.get("status") == "completed" and manifest.get("step") == STEP and manifest.get("nonfinite_count") == 0,
                "Run must finish cleanly at LAST3000 with zero nonfinite events")
        require(manifest.get("data_counts") == {"train": 54810, "eval": 1998}, "Changed sample counts")
        require(manifest.get("split_sha256") == SPLIT_SHA and manifest.get("supervision_manifest_sha256") == MANIFEST_SHAS[c],
                "Run split or C-specific supervision SHA mismatch")
        check_time(manifest, timing, "run_manifest", policy=True)
        check_time(args, timing, "training arguments")
        require(all(json_form(args.get(k)) == v for k, v in EXPECTED_ARGS.items()) and args.get("gpu") == i, "P2 schedule/optimizer/GPU mismatch")
        for key in ("resume", "pretrained", "train_scenes", "eval_scenes"):
            require(args.get(key) is None, f"Unexpected run override {key}")
        require(args.get("eval_only") is False and args.get("cpu") is False, "Run was eval-only/CPU")
        require(str(args.get("init", "")).endswith("work_dirs/motiondrive_v2/p1_g1s1_s0/last.pth"), "Wrong common P1 LAST init")
        require(Path(args.get("supervision_root", "")).name == ("train_tune_geometry_v2" if c else "train_tune_rawtime"), "Wrong C training edition path")
        config = json_form(manifest.get("model_config"))
        require(config == common["model_config"] and config.get("goal_on") is True and config.get("state_on") is True,
                "P2 must retain identical P1 G1S1 architecture")
        require(config.get("motion_input_mode") == "low_feature" and config.get("plan_output_scale") == [10., 5.], "Wrong low_feature/XY units")
        require(manifest.get("initial_model_state_sha256") == common["model_state_sha256"] == data.get("initial_tensor_sha256"),
                "Initial tensors do not exactly match common P1 LAST")
        require(data.get("last_tensor_signature") == common["tensor_signature"] == data.get("initial_tensor_signature"), "Model tensor shapes/dtypes changed")
        require(manifest.get("load_report", {}).get("common_checkpoint_sha256") == COMMON_SHA, "Wrong common initialization SHA")
        require(manifest.get("loss_weights") == dict(plan=1., occupancy=.2, lane=.2, motion=.2, uncertainty=True), "Loss differs from P1")
        bn = manifest.get("bn_training", {})
        require(bn.get("policy") == "fixed" and bn.get("affine_and_backbone_weights_trainable") is True, "Wrong fixed-stat/full-weight BN policy")
        require(data.get("os_exit", {}).get("actual_returncode") == 0 and data["os_exit"].get("trainer_pid_absent") is True,
                "Missing actual clean process exit audit")
        trace, logged, selected, eval_logs = validate_metrics(data["metrics"])
        for log in eval_logs:
            check_time(log, timing, "training evaluation log")
        require(data["best"]["step"] == selected["step"] and data["best"]["official_d3"] == selected["official_d3"], "BEST not selected by original tune logs")
        normalized = normalize_evaluation(data["evaluation"], arm, manifest)
        require(normalized["checkpoint_sha256"] == data["last_checkpoint_sha256"], "Independent evaluation LAST SHA mismatch")
        require(Path(normalized["checkpoint_path"]).resolve() == Path(data["run_dir"]).resolve() / "last.pth", "Evaluation points outside this arm LAST")
        normal = normalized["conditions"]["normal"]
        record_keys, values, sessions, labels, row_sha = validate_records(normal, split)
        require(row_sha == manifest.get("eval_rows_sha256"), "Actual record row order mismatch")
        for key in ("official_d3", "n", "n_sessions", "session_d3"):
            require(normal["summary"].get(key) == logged.get(key), f"Independent LAST evaluation/log mismatch: {key}")
        require(np.isclose(normal["summary"]["session_mean_d3"], logged["session_mean_d3"], atol=1e-12, rtol=0), "LAST session aggregate/log mismatch")
        for condition, result in normalized["conditions"].items():
            if condition == "normal":
                continue  # Already checked above, including the coordinate/metric replay.
            ck, _, _, cl, cr = validate_records(result, split)
            require(ck == record_keys and cl == labels and cr == row_sha, f"Image condition changed receiver labels/order: {condition}")
        supervised = normalized["data"]["supervision_sha256"]
        for scene in split["splits"]["tune"]:
            for suffix in (".npz", ".json"):
                name = scene + suffix
                require(supervised.get(name) == geometry["scene_file_sha256"].get(name), "Evaluation scene GT/arrays edition mismatch")
        require(normalized["data"].get("ego_cache_sha256") and normalized["data"].get("image_root"), "Missing image/ego-cache provenance")
        require(manifest.get("git_sha") == data.get("training_source", {}).get("git_sha"), "Run git SHA differs from launch training SOURCE")
        configs.append(config)
        arguments.append({k: v for k, v in json_form(args).items() if k not in {"gpu", "run_dir", "supervision_root", "time_input"}})
        sources.append(data["training_source"]);eval_sources.append(normalized["evaluation_source"])
        traces.append(trace);label_sets.append(labels);matrices.append(values);session_ids.append(sessions)
        if keys is None:
            keys = record_keys
        require(keys == record_keys, "Arm sample orders differ")
    for name, values in (("architecture", configs), ("non-treatment arguments", arguments), ("training SOURCE", sources),
                         ("evaluation SOURCE", eval_sources), ("all logged sample-order hashes", traces), ("GT/rows/buckets", label_sets), ("sessions", session_ids)):
        require(all(value == values[0] for value in values[1:]), f"Cross-arm mismatch: {name}")
    for key in ("initial_parameter_count", "train_rows_sha256", "eval_rows_sha256", "load_report", "bn_training", "torch", "numpy"):
        values = [arms[arm]["run_manifest"].get(key) for arm in ARMS]
        require(values[0] is not None and all(v == values[0] for v in values[1:]), f"Missing/different fairness evidence: {key}")
    for key in ("ego_cache_sha256", "image_root"):
        values = [arms[arm]["evaluation"]["protocol"]["data"].get(key) for arm in ARMS]
        require(all(value == values[0] for value in values[1:]), f"Different evaluation data source: {key}")
    return keys, np.stack(matrices, axis=1), session_ids[0]


def joint_session_bootstrap(matrix, session_ids):
    """One fixed 10000x11 draw matrix shared by all four arms and five effects."""
    values, ids = np.asarray(matrix, np.float64), np.asarray(session_ids, str)
    require(values.ndim == 2 and values.shape == (len(ids), 4) and np.isfinite(values).all() and (values >= 0).all(), "Invalid paired D3 matrix")
    sessions = sorted(set(ids))
    require(len(sessions) == 11, "P2 bootstrap requires exactly eleven rawtime sessions")
    totals = np.array([values[ids == session].sum(0) for session in sessions])
    counts = np.array([(ids == session).sum() for session in sessions])
    means = totals / counts[:, None]
    draws = np.random.default_rng(SEED).integers(0, 11, size=(REPEATS, 11))
    coefficients = np.array(list(CONTRASTS.values()))
    results = {}
    for label, point, sample in (
        ("primary_frame_weighted", values.mean(0), totals[draws].sum(1) / counts[draws].sum(1)[:, None]),
        ("secondary_session_equal", means.mean(0), means[draws].mean(1))):
        def summary(value, samples):
            return {"estimate": float(value), "ci95_percentile": np.quantile(samples, [.025, .975]).tolist()}
        effects, effect_draws = point @ coefficients.T, sample @ coefficients.T
        results[label] = {"arms": {arm: summary(point[i], sample[:, i]) for i, arm in enumerate(ARMS)},
            "contrasts": {name: {**summary(effects[i], effect_draws[:, i]), "coefficients": list(CONTRASTS[name])}
                          for i, name in enumerate(CONTRASTS)}}
    results["resampling"] = {"n_sessions": 11, "n_frames": len(ids), "repeats": REPEATS, "seed": SEED,
        "session_frame_counts": dict(zip(sessions, map(int, counts))), "shared_draws": True,
        "draw_matrix_sha256": hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest()}
    return results


def analyze_p2(arms, split, common, geometry):
    keys, matrix, sessions = validate_experiment(arms, split, common, geometry)
    statistics = joint_session_bootstrap(matrix, sessions)
    diagnostics = {arm: {name: {"summary": value["summary"], "buckets": value["buckets"]}
                        for name, value in arms[arm]["evaluation"]["conditions"].items()} for arm in ARMS}
    return {"status": "LAST3000 P2 CxT analysis completed", "arm_order": list(ARMS),
        "arm_policies": {arm: {"C": c, "T": t, "time_input": "nominal" if t else "raw",
            "supervision_manifest_sha256": MANIFEST_SHAS[c], "calibration_sha256": CALIBRATION_SHAS[c],
            "goal_on": True, "image_inferred_status_on": True} for arm, (c, t) in zip(ARMS, FACTORS)},
        "protocol": {"primary_checkpoint": "LAST3000", "best": "separate descriptive table; excluded from all effects/CI",
            "common_init": "P1 G1S1 LAST6000; weights-only, new AdamW; not corrected-P0 retraining",
            "split_sha256": SPLIT_SHA, "common_checkpoint_sha256": COMMON_SHA,
            "sign": "on-minus-off conditional effects; interaction D11-D10-D01+D00; negative means lower error",
            "metric": "frame mean of official [11,11,5,5,2,2]/36 weighted six-point L2",
            "coordinate_metric_crosscheck": {"float64_recomputed_vs_stored_float32_atol": RECORD_METRIC_ATOL,
                "rtol": 0., "stored_scores_replaced": False,
                "scope": "point L2, D3, cumulative ADE, signed x/y and weighted absolute x/y"},
            "data_exposure": "3000 x batch16 =48000 samples, before first 54810-frame epoch ends; row traces audited",
            "training_and_evaluation_sources_compared_separately": True,
            "only_deployable_contract": "C1/T1, regardless of which arm has lower tune error"},
        "last3000": statistics, "best_secondary": {arm: arms[arm]["best"] for arm in ARMS},
        "last_checkpoint_sha256": {arm: arms[arm]["last_checkpoint_sha256"] for arm in ARMS},
        "initial_tensor_sha256": common["model_state_sha256"],
        "training_source": arms["c0t0"]["training_source"],
        "evaluation_source": arms["c0t0"]["evaluation"]["protocol"]["source"],
        "planning_diagnostics": diagnostics,
        "c1t1_image_diagnostics_complete": CONDITIONS <= set(diagnostics["c1t1"]),
        "c1t1_image_diagnostics_missing": sorted(CONDITIONS - set(diagnostics["c1t1"])),
        "geometry_evidence": geometry,
        "paired_records": [{"scenario": key[0], "frame": key[1], "session": sessions[i],
                            **{arm: float(matrix[i, j]) for j, arm in enumerate(ARMS)}} for i, key in enumerate(keys)],
        "limitations": ["Single training seed and repeatedly used tune11 sessions; not untouched final-val/test evidence.",
                        "Five exploratory contrast intervals are not multiplicity-adjusted significance claims.",
                        "C0 or T0 is an experimental control, never a deployment choice even when its error is lower.",
                        "Common initialization already learned erroneous rear geometry; this measures repair adaptation only.",
                        "Training/evaluation source commits may differ; equality is checked within each four-arm stage.",
                        "No wall-clock efficiency comparison across differently started or shared-GPU runs.",
                        "Same image root is checked; this analysis does not reread/hash the entire training image cache."]}


def audit_geometry(old_root, new_root, split):
    roots = [Path(old_root), Path(new_root)]
    before, contracts = {}, []
    for c, root in enumerate(roots):
        for name, pinned in (("supervision_manifest.json", MANIFEST_SHAS[c]), ("calibration.npz", CALIBRATION_SHAS[c])):
            path = root / name
            before[str(path.resolve())] = sha256(path)
            require(before[str(path.resolve())] == pinned, "Pinned geometry edition file SHA mismatch")
        contracts.append(json.loads((root / "supervision_manifest.json").read_text()))
    require(contracts[1].get("geometry_edition") == "cache_meta_rear_wide_v2" and
            contracts[1].get("canonical_calibration_sha256") == CALIBRATION_SHAS[1], "Wrong geometry_v2 contract")
    arrays = []
    for root in roots:
        with np.load(root / "calibration.npz", allow_pickle=False) as z:
            arrays.append(z["lidar2img"].copy())
    require(all(a.shape == (6, 4, 4) and a.dtype == np.float32 and np.isfinite(a).all() for a in arrays), "Malformed canonical calibration")
    kept = [0, 1, 2, 4, 5]
    require(arrays[0][kept].tobytes() == arrays[1][kept].tobytes() and arrays[0][3].tobytes() != arrays[1][3].tobytes(),
            "Correction must change rear only, preserving other five bitwise")
    scene_hashes = {}
    scenes = sorted(set(split["splits"]["train"] + split["splits"]["tune"]))
    for scene in scenes:
        for suffix in (".npz", ".json"):
            name = scene + suffix
            hashes = [sha256(root / name) for root in roots]
            require(hashes[0] == hashes[1], f"GT/history/scene file changed across C editions: {name}")
            scene_hashes[name] = hashes[0]
            before.update({str((root / name).resolve()): digest for root, digest in zip(roots, hashes)})
    require(before == {path: sha256(path) for path in before}, "Geometry source changed during verification")
    return {"train_tune_scenes": len(scenes), "scene_file_sha256": scene_hashes,
            "all_scene_npz_json_bitwise_preserved": True, "other_five_calibrations_bitwise_preserved": True,
            "manifest_sha256": list(MANIFEST_SHAS), "calibration_sha256": list(CALIBRATION_SHAS),
            "file_sha256": before, "finalval_test_scene_files_read": False}


def load_common(path):
    import torch
    require(Path(path).name == "last.pth" and sha256(path) == COMMON_SHA, "Pinned common P1 LAST file missing or wrong SHA")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(checkpoint.get("step") == 6000, "Common init is not P1 LAST6000")
    signature, digest = state_signature(checkpoint["model"])
    require(sha256(path) == COMMON_SHA, "Common init changed during read")
    return {"model_config": json_form(checkpoint["manifest"]["model_config"]),
            "tensor_signature": signature, "model_state_sha256": digest, "path": str(Path(path).resolve()), "sha256": COMMON_SHA}


def load_launches(paths, plan):
    expected = {job["name"]: job for job in plan["jobs"]}
    require(set(expected) == {f"p2_{arm}_s0" for arm in ARMS} and plan.get("supervise") is True, "Wrong preregistered P2 plan")
    jobs = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        require(payload.get("status") == "launched" and payload.get("supervise") is True
                and payload.get("tracked_dirty") == "" and payload.get("actual_git_sha") and payload.get("trainer_sha256"),
                "Missing clean supervised launch SOURCE")
        for job in payload["jobs"]:
            name = job["name"]
            require(name in expected and name not in jobs and job["explicit_argv"] == expected[name]["argv"]
                    and job["gpu"] == expected[name]["gpu"], "Launch is not an unchanged, nonduplicated plan subset")
            process = [p for p in payload["processes"] if p["name"] == name]
            require(len(process) == 1 and process[0]["gpu"] == job["gpu"], "Launch process identity missing")
            require(job["inputs"]["checkpoint_init"]["sha256"] == COMMON_SHA
                    and job["inputs"]["split"]["sha256"] == SPLIT_SHA, "Launch init/split lineage mismatch")
            c = FACTORS[ARMS.index(name[3:-3])][0]
            require(job["inputs"]["supervision"]["sha256"] == MANIFEST_SHAS[c], "Launch C edition mismatch")
            jobs[name] = {"job": job, "process": process[0], "record_path": str(Path(path).resolve()),
                          "training_source": {"git_sha": payload["actual_git_sha"], "trainer_sha256": payload["trainer_sha256"],
                                              "supervisor_sha256": payload.get("supervisor_sha256")}}
    require(set(jobs) == set(expected), "Missing P2 launch arm")
    return jobs


def load_arm(arm, run_dir, evaluation_path, launch):
    import torch
    run = Path(run_dir).resolve()
    require(run.name == f"p2_{arm}_s0" and run == Path(launch["job"]["run_dir"]).resolve(), "Wrong run directory identity")
    manifest = json.loads((run / "manifest.json").read_text())
    supervisor = json.loads(Path(launch["job"]["supervisor_record"]).read_text())
    exit_check = validate_exit(supervisor, manifest, run, launch["job"], launch["process"])
    evaluation = json.loads(Path(evaluation_path).read_text())
    protocol_path = Path(evaluation["protocol_path"])
    require(sha256(protocol_path) == evaluation["protocol_sha256"] and json.loads(protocol_path.read_text()) == evaluation["protocol"],
            "External preregistered evaluation protocol SHA/content mismatch")
    checkpoints = {}
    for name, expected_step in (("initial", 0), ("last", STEP), ("best", None)):
        path = run / f"{name}.pth"
        digest = sha256(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if expected_step is not None:
            require(saved.get("step") == expected_step, f"Wrong {name} checkpoint step")
        if name == "initial":
            require(saved.get("optimizer", {}).get("state") == {}, "Initial AdamW has inherited optimizer moments")
        require(all(json_form(saved["manifest"].get(key)) == json_form(manifest.get(key)) for key in EMBEDDED_KEYS),
                f"{name} checkpoint/run manifest mismatch")
        signature, tensor_sha = state_signature(saved["model"])
        require(sha256(path) == digest, f"{name} checkpoint changed during read")
        checkpoints[name] = {"path": str(path), "sha256": digest, "step": saved["step"], "signature": signature, "tensor_sha256": tensor_sha}
        del saved
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    _, _, selected, _ = validate_metrics(metrics)
    require(checkpoints["best"]["step"] == selected["step"], "BEST file is not original first minimum tune checkpoint")
    require(checkpoints["best"]["signature"] == checkpoints["last"]["signature"], "BEST/LAST architecture mismatch")
    best = {"path": checkpoints["best"]["path"], "sha256": checkpoints["best"]["sha256"], "step": selected["step"],
            "selection_definition": "official_d3", "source": "original training full-tune evaluation log, not independent BEST replay",
            "official_d3": selected["official_d3"], "session_mean_d3": selected["session_mean_d3"],
            "auxiliary_report": selected, "excluded_from_last_factorial_estimates": True}
    return {"run_dir": str(run), "run_manifest": manifest, "evaluation": evaluation, "metrics": metrics,
            "initial_tensor_sha256": checkpoints["initial"]["tensor_sha256"], "initial_tensor_signature": checkpoints["initial"]["signature"],
            "last_tensor_signature": checkpoints["last"]["signature"], "last_checkpoint_sha256": checkpoints["last"]["sha256"],
            "os_exit": exit_check, "training_source": launch["training_source"], "best": best}


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    for arm in ARMS:
        parser.add_argument("--" + arm, nargs=2, metavar=("RUN_DIR", "LAST_EVALUATION_JSON"), required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--common-init", required=True)
    parser.add_argument("--old-supervision", required=True)
    parser.add_argument("--new-supervision", required=True)
    parser.add_argument("--launch-record", nargs="+", required=True)
    parser.add_argument("--plan", default=str(ROOT / "configs/motiondrive_v2/p2_geometry_time_r1_s0.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    require(not os.path.lexists(output), "Existing P2 report must not be overwritten")
    require(sha256(args.split_manifest) == SPLIT_SHA, "Wrong pinned rawtime split")
    split = json.loads(Path(args.split_manifest).read_text());check_split(split)
    plan = json.loads(Path(args.plan).read_text())
    launches = load_launches(args.launch_record, plan)
    artifacts = [Path(__file__), ROOT / "scripts/build_grouped_split_v2.py", ROOT / "MOTIONDRIVE_V2_P2_REPAIR_PROTOCOL.md",
                 Path(args.plan), Path(args.split_manifest), Path(args.common_init), *map(Path, args.launch_record)]
    for arm in ARMS:
        run, report = map(Path, getattr(args, arm))
        evidence = json.loads(report.read_text())
        artifacts += [run / name for name in ("manifest.json", "metrics.jsonl", "initial.pth", "last.pth", "best.pth")]
        artifacts += [report, Path(evidence["protocol_path"]), Path(launches[f"p2_{arm}_s0"]["job"]["supervisor_record"])]
    before = {str(path.resolve()): sha256(path) for path in artifacts}
    geometry = audit_geometry(args.old_supervision, args.new_supervision, split)
    common = load_common(args.common_init)
    arms = {arm: load_arm(arm, *getattr(args, arm), launches[f"p2_{arm}_s0"]) for arm in ARMS}
    result = analyze_p2(arms, split, common, geometry)
    before.update(geometry["file_sha256"])
    after = {path: sha256(path) for path in before}
    require(before == after, "An input artifact changed during analysis")
    result.update(input_sha256_before=before, input_sha256_after=after,
                  actual_os_exit={arm: arms[arm]["os_exit"] for arm in ARMS}, gpu_used=False, model_forward_performed=False)
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False);stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(output), "sha256": sha256(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
