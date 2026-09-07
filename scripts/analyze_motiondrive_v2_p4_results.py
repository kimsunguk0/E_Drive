#!/usr/bin/env python3
"""CPU-only analysis of two completed P4 LAST6000 tune prediction dumps.

This reads immutable evaluator reports; it never runs a model or opens final
validation.  The optional C0 report is a historical reference with different
training and geometry, not a causal or matched-condition control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from scripts.analyze_motiondrive_v2_query_pair import paired_cluster_bootstrap
from scripts.analyze_motiondrive_v2_temporal_shape import WEIGHTS
from scripts.evaluate_motiondrive_v2_planning import BUCKET_NAMES, state_bucket
from scripts.export_motiondrive_v2_inference import (
    C1_CANONICAL_SHA256, C1_SUPERVISION_SHA256, RAWTIME_SPLIT_SHA256,
)

P4_STEP = 6000
P4_GIT_SHA = "86620b4ffc7e6838b49cf83b5be789eba12d8027"
EXPECTED_N, EXPECTED_SCENES, EXPECTED_SESSIONS = 1998, 37, 11
EXPECTED_FRAMES = tuple(range(30, 300, 5))
C0_D3 = 0.36617518485979633
C0_STEP = 3000
TRAINING_INTEGRITY_OUTCOMES = (
    "not_supplied",
    "supervisor_exit0",
    "post_training_global_head_guard_rc1_preserved",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_pinned_json(path, expected_sha256):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Ordinary input file required: {path}")
    require(valid_sha(expected_sha256), "Explicit lowercase input SHA256 required")
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected_sha256, f"Input SHA256 mismatch: {path}")

    def no_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(raw, parse_constant=no_constant, object_pairs_hook=unique)
    require(isinstance(value, dict), "JSON report must be a mapping")
    return value, raw


def protocol_sha(protocol):
    encoded = json.dumps(protocol, indent=2, allow_nan=False).encode() + b"\n"
    return sha256_bytes(encoded)


def validate_split(split):
    tune = split.get("splits", {}).get("tune")
    mapping = split.get("scene_to_session")
    require(isinstance(tune, list) and len(tune) == EXPECTED_SCENES
            and len(set(tune)) == EXPECTED_SCENES and isinstance(mapping, dict),
            "Expected fixed tune37 split")
    require(all(isinstance(scene, str) and isinstance(mapping.get(scene), str)
                and mapping[scene] for scene in tune), "Tune scene/session mapping missing")
    require(len({mapping[scene] for scene in tune}) == EXPECTED_SESSIONS,
            "Expected exactly eleven tune sessions")
    return tune, mapping


def numeric(value, shape, label):
    array = np.asarray(value)
    require(array.shape == shape and array.dtype.kind in "iuf" and np.isfinite(array).all(),
            f"Finite numeric {label}{shape} required")
    return array.astype(np.float64, copy=False)


def exact_json(a, b):
    return json.dumps(a, sort_keys=True, separators=(",", ":"), allow_nan=False) == \
           json.dumps(b, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_training_manifest(manifest, seed):
    require(isinstance(manifest, dict), "Checkpoint training manifest missing")
    args, config = manifest.get("arguments", {}), manifest.get("model_config", {})
    required_args = {"phase": "joint", "steps": P4_STEP, "batch": 16,
                     "microbatch": 2, "eval_batch": 4,
                     "workers": 4, "seed": seed, "precision": "bf16",
                     "time_input": "nominal", "eval_stride": 5,
                     "eval_split": "tune", "max_eval_samples": 0,
                     "bn_policy": "fixed"}
    require(all(args.get(key) == value for key, value in required_args.items()),
            "Checkpoint is not the requested P4 joint schedule")
    required_config = {"backbone_arch": "resnet50", "goal_on": True,
                       "state_on": True, "motion_input_mode": "low_feature",
                       "n_history": 4}
    require(all(config.get(key) == value for key, value in required_config.items())
            and list(config.get("plan_output_scale", ())) == [10., 5.],
            "Checkpoint is not complete P4 R50/G1S1/low_feature/scale(10,5)")
    require(manifest.get("loss_weights", {}).get("plan") == 1,
            "P4 joint checkpoint must have nonzero unit planning loss")
    require(manifest.get("git_sha") == P4_GIT_SHA
            and manifest.get("split_sha256") == RAWTIME_SPLIT_SHA256
            and manifest.get("supervision_manifest_sha256") == C1_SUPERVISION_SHA256,
            "P4 source/split/C1 training lineage mismatch")
    require(manifest.get("data_counts") == {"train": 54810, "eval": EXPECTED_N},
            "P4 full train/tune counts missing")
    require(manifest.get("eval_rows_sha256") and valid_sha(manifest["eval_rows_sha256"]),
            "P4 checkpoint eval-row receipt missing")
    require(manifest.get("status") in ("running", "completed"),
            "LAST embedded trainer status invalid")
    return args, config


def recompute_record(record, mapping):
    require(isinstance(record, dict) and type(record.get("row")) is int and record["row"] >= 0
            and type(record.get("frame")) is int and record["frame"] >= 0
            and isinstance(record.get("scenario"), str) and record["scenario"]
            and isinstance(record.get("session"), str) and record["session"],
            "Per-frame identity is invalid")
    require(mapping.get(record["scenario"]) == record["session"],
            "Per-frame session mapping differs from split")
    pred = numeric(record.get("pred_abs_xy"), (6, 2), "pred_abs_xy")
    gt = numeric(record.get("gt_abs_xy"), (6, 2), "gt_abs_xy")
    error = pred - gt
    point = np.linalg.norm(error, axis=-1)
    cumulative = np.asarray([point[:n].mean() for n in (2, 4, 6)])
    d3 = float(point @ WEIGHTS)
    expected = {
        "point_l2": point,
        "cumulative_ade_1_2_3s": cumulative,
        "longitudinal_error": error[:, 0],
        "lateral_error": error[:, 1],
    }
    for name, actual in expected.items():
        stored = numeric(record.get(name), actual.shape, name)
        require(float(np.max(np.abs(stored - actual))) <= 1e-5,
                f"Stored per-frame error disagrees with pred/GT: {name}")
    require(type(record.get("d3")) in (int, float) and math.isfinite(record["d3"])
            and abs(float(record["d3"]) - d3) <= 1e-5,
            "Stored D3 disagrees with official reconstruction")
    for name, value in (("weighted_abs_longitudinal_error", np.abs(error[:, 0]) @ WEIGHTS),
                        ("weighted_abs_lateral_error", np.abs(error[:, 1]) @ WEIGHTS)):
        require(type(record.get(name)) in (int, float) and math.isfinite(record[name])
                and abs(float(record[name]) - float(value)) <= 1e-5,
                f"Stored weighted axis error disagrees: {name}")
    state = record.get("gt_state")
    mask = record.get("gt_state_valid")
    require(isinstance(state, list) and len(state) == 6 and isinstance(mask, list)
            and len(mask) == 6 and all(type(value) is bool for value in mask),
            "GT state/mask contract missing")
    restored = np.full(6, np.nan)
    for index, (value, valid) in enumerate(zip(state, mask)):
        if valid:
            require(type(value) in (int, float) and math.isfinite(value), "Valid GT state must be finite")
            restored[index] = value
        else:
            require(value is None, "Invalid GT state must be null")
    require(record.get("bucket") in BUCKET_NAMES
            and record["bucket"] == state_bucket(restored, mask), "Stored state bucket mismatch")
    return {"identity": (record["row"], record["scenario"], record["session"], record["frame"]),
            "pred": pred, "gt": gt, "error": error, "point": point,
            "d3": float(record["d3"]), "recomputed_d3": d3,
            "state": state, "state_valid": mask, "bucket": record["bucket"]}


def aggregate(rows):
    require(bool(rows), "Cannot aggregate an empty group")
    error = np.asarray([row["error"] for row in rows])
    point = np.asarray([row["point"] for row in rows])
    d3 = np.asarray([row["d3"] for row in rows])
    return {"n": len(rows), "official_d3": float(d3.mean()),
            "waypoint_l2_mean_m": point.mean(0).tolist(),
            "cumulative_ade_1_2_3s_mean_m": np.asarray(
                [[p[:n].mean() for n in (2, 4, 6)] for p in point]).mean(0).tolist(),
            "longitudinal_signed_mean_m": error[..., 0].mean(0).tolist(),
            "longitudinal_absolute_mean_m": np.abs(error[..., 0]).mean(0).tolist(),
            "lateral_signed_mean_m": error[..., 1].mean(0).tolist(),
            "lateral_absolute_mean_m": np.abs(error[..., 1]).mean(0).tolist(),
            "weighted_abs_longitudinal_m": float((np.abs(error[..., 0]) @ WEIGHTS).mean()),
            "weighted_abs_lateral_m": float((np.abs(error[..., 1]) @ WEIGHTS).mean())}


def grouped(rows, field):
    keys = sorted({row[field] for row in rows})
    return {str(key): aggregate([row for row in rows if row[field] == key]) for key in keys}


def grouped_buckets(rows):
    result = {}
    for bucket in BUCKET_NAMES:
        selected = [row for row in rows if row["bucket"] == bucket]
        result[bucket] = aggregate(selected) if selected else {"n": 0, "official_d3": None}
    return result


def absolute_cluster_interval(values, sessions, *, repeats, seed):
    result = paired_cluster_bootstrap(values, sessions, repeats=repeats, seed=seed)
    result["estimate"] = result.pop("delta")
    return result


def validate_stored_aggregates(condition, rows):
    summary = condition.get("summary", {})
    computed = aggregate(rows)
    require(summary.get("n") == len(rows)
            and summary.get("n_scenes") == len({row["scenario"] for row in rows})
            and summary.get("n_sessions") == len({row["session"] for row in rows})
            and abs(float(summary.get("official_d3", float("nan"))) - computed["official_d3"]) <= 1e-12,
            "Stored full-tune summary disagrees with records")
    for bucket in BUCKET_NAMES:
        selected = [row for row in rows if row["bucket"] == bucket]
        stored = condition.get("buckets", {}).get(bucket, {})
        require(stored.get("n") == len(selected), f"Stored bucket count mismatch: {bucket}")
        if selected:
            require(abs(float(stored.get("official_d3", float("nan")))
                        - aggregate(selected)["official_d3"]) <= 1e-12,
                    f"Stored bucket D3 mismatch: {bucket}")
        else:
            require(stored.get("official_d3") is None, f"Empty bucket D3 must be null: {bucket}")


def validate_report(document, seed, tune_scenes, mapping, *, p4=True):
    protocol, args = document.get("protocol", {}), document.get("protocol", {}).get("arguments", {})
    require(document.get("status") == "completed"
            and protocol.get("status") == "preregistered_before_any_forward"
            and document.get("protocol_sha256") == protocol_sha(protocol),
            "Completed evaluator report/protocol SHA required")
    require(document.get("selection_performed") is False and document.get("final_val_accessed") is False
            and protocol.get("selection_performed") is False and protocol.get("final_val_accessed") is False,
            "Selection or final validation access is forbidden")
    common = {"split": "tune", "frame_stride": 5, "max_samples": 0,
              "scenes": None, "batch": 4,
              "precision": "bf16", "time_input": "nominal"}
    require(all(args.get(key) == value for key, value in common.items())
            and args.get("seed") == seed
            and isinstance(args.get("device"), str) and args["device"].startswith("cuda:"),
            "Evaluator arguments are not normal full-tune/batch4/BF16/nominal")
    report_conditions = document.get("conditions", {})
    requested_conditions = args.get("conditions")
    require(isinstance(report_conditions, dict) and isinstance(requested_conditions, list)
            and len(requested_conditions) == len(set(requested_conditions))
            and set(requested_conditions) == set(report_conditions)
            and "normal" in report_conditions,
            "Evaluator requested/reported conditions disagree or omit normal")
    if p4:
        require(requested_conditions == ["normal"], "P4 result report must contain normal only")
    require(document.get("time_input") == protocol.get("time_input") == "nominal"
            and document.get("precision") == document.get("precision_requested") == "bf16"
            and report_conditions["normal"].get("time_input") == "nominal",
            "Report precision/time contract mismatch")
    data = protocol.get("data", {})
    require(data.get("split_sha256") == RAWTIME_SPLIT_SHA256
            and data.get("receiver_count") == EXPECTED_N and valid_sha(data.get("receiver_rows_sha256")),
            "Full fixed tune data receipt missing")
    checkpoint_sha = protocol.get("checkpoint_sha256")
    require(valid_sha(checkpoint_sha)
            and checkpoint_sha == document.get("model_load", {}).get("checkpoint_sha256"),
            "Evaluator model-load/checkpoint SHA receipt mismatch")
    if p4:
        require(protocol.get("checkpoint_step") == P4_STEP
                and protocol.get("checkpoint_manifest", {}).get("arguments", {}).get("steps") == P4_STEP,
                "P4 checkpoint/completed schedule step mismatch")
        supervision = data.get("supervision_sha256", {})
        require(supervision.get("supervision_manifest.json") == C1_SUPERVISION_SHA256
                and supervision.get("calibration.npz") == C1_CANONICAL_SHA256,
                "P4 evaluation is not C1 corrected geometry")
        validate_training_manifest(protocol.get("checkpoint_manifest"), seed)
        require(protocol["checkpoint_manifest"]["eval_rows_sha256"] == data["receiver_rows_sha256"],
                "Evaluation rows differ from P4 training tune rows")
        source = protocol.get("source", {})
        source_files = source.get("file_sha256")
        require(isinstance(source.get("git_sha"), str)
                and re.fullmatch(r"[0-9a-f]{40}", source["git_sha"]) is not None
                and source.get("tracked_changes") == []
                and isinstance(source_files, dict) and bool(source_files)
                and "scripts/evaluate_motiondrive_v2_planning.py" in source_files
                and all(isinstance(path, str) and path and valid_sha(value)
                        for path, value in source_files.items()),
                "P4 evaluator must record a clean Git/file-SHA source receipt")
    else:
        require(protocol.get("checkpoint_step") == C0_STEP, "Historical C0 must be LAST3000")
    records = report_conditions["normal"].get("records")
    require(isinstance(records, list) and len(records) == EXPECTED_N, "Expected exactly tune1998 records")
    parsed = []
    for source in records:
        row = recompute_record(source, mapping)
        row.update(scenario=source["scenario"], session=source["session"], frame=source["frame"],
                   source=source)
        parsed.append(row)
    identities = [row["identity"] for row in parsed]
    require(len(set(identities)) == EXPECTED_N, "Duplicate tune identity")
    expected = {(scene, frame) for scene in tune_scenes for frame in EXPECTED_FRAMES}
    require({(row["scenario"], row["frame"]) for row in parsed} == expected,
            "Records are not exact tune37 x frames30:5:295")
    observed_rows_sha = sha256_bytes(np.asarray([row["identity"][0] for row in parsed], dtype="<i8").tobytes())
    require(observed_rows_sha == data["receiver_rows_sha256"], "Record order/row SHA mismatch")
    validate_stored_aggregates(report_conditions["normal"], parsed)
    return parsed


def require_same_labels(reference, other, label):
    require([row["identity"] for row in reference] == [row["identity"] for row in other],
            f"{label} identities/order differ")
    for first, second in zip(reference, other):
        require(exact_json(first["source"]["gt_abs_xy"], second["source"]["gt_abs_xy"])
                and exact_json(first["state"], second["state"])
                and exact_json(first["state_valid"], second["state_valid"])
                and first["bucket"] == second["bucket"], f"{label} GT/mask/bucket differs")


def paired_groups(first, second, field):
    result = {}
    keys = sorted({row[field] for row in first})
    for key in keys:
        use = [index for index, row in enumerate(first) if row[field] == key]
        a, b = np.asarray([first[i]["d3"] for i in use]), np.asarray([second[i]["d3"] for i in use])
        result[str(key)] = {"n": len(use), "s0_official_d3": float(a.mean()),
                            "s1_official_d3": float(b.mean()),
                            "s1_minus_s0_d3": float((b - a).mean())}
    return result


def analyze(s0, s1, split, *, repeats, bootstrap_seed, c0=None,
            training_integrity=None):
    require(type(repeats) is int and repeats >= 100 and type(bootstrap_seed) is int
            and bootstrap_seed >= 0, "Bootstrap repeats/seed invalid")
    training_integrity = ({"s0": "not_supplied", "s1": "not_supplied"}
                          if training_integrity is None else training_integrity)
    require(isinstance(training_integrity, dict)
            and set(training_integrity) == {"s0", "s1"}
            and all(value in TRAINING_INTEGRITY_OUTCOMES
                    for value in training_integrity.values()),
            "Explicit per-seed training integrity outcome is invalid")
    tune, mapping = validate_split(split)
    first = validate_report(s0, 0, tune, mapping)
    second = validate_report(s1, 1, tune, mapping)
    require(exact_json(s0["protocol"]["source"], s1["protocol"]["source"]),
            "P4 seeds must use the exact same evaluator source receipt")
    require_same_labels(first, second, "P4 seed pair")
    sessions = [row["session"] for row in first]
    d0, d1 = np.asarray([row["d3"] for row in first]), np.asarray([row["d3"] for row in second])
    seed_summaries = {
        "s0": {"all": aggregate(first), "scenarios": grouped(first, "scenario"),
               "sessions": grouped(first, "session"), "buckets": grouped_buckets(first),
               "session_cluster_absolute_interval": absolute_cluster_interval(
                   d0, sessions, repeats=repeats, seed=bootstrap_seed)},
        "s1": {"all": aggregate(second), "scenarios": grouped(second, "scenario"),
               "sessions": grouped(second, "session"), "buckets": grouped_buckets(second),
               "session_cluster_absolute_interval": absolute_cluster_interval(
                   d1, sessions, repeats=repeats, seed=bootstrap_seed)},
    }
    means = np.asarray([seed_summaries[key]["all"]["official_d3"] for key in ("s0", "s1")])
    paired = {"s1_minus_s0_session_cluster": paired_cluster_bootstrap(
                  d1 - d0, sessions, repeats=repeats, seed=bootstrap_seed),
              "scenarios": paired_groups(first, second, "scenario"),
              "sessions": paired_groups(first, second, "session"),
              "buckets": paired_groups(first, second, "bucket")}
    historical = None
    if c0 is not None:
        control = validate_report(c0, 0, tune, mapping, p4=False)
        require_same_labels(first, control, "Historical C0/P4")
        c = np.asarray([row["d3"] for row in control])
        require(abs(float(c.mean()) - C0_D3) <= 1e-12, "Historical C0 is not the recorded .366 reference")
        historical = {"all": aggregate(control), "scenarios": grouped(control, "scenario"),
            "sessions": grouped(control, "session"), "buckets": grouped_buckets(control),
            "p4_s0_minus_c0_session_cluster": paired_cluster_bootstrap(
                d0 - c, sessions, repeats=repeats, seed=bootstrap_seed),
            "p4_s1_minus_c0_session_cluster": paired_cluster_bootstrap(
                d1 - c, sessions, repeats=repeats, seed=bootstrap_seed),
            "matched_only_for": "same tune row identities and exact GT labels",
            "not_matched_for": ["training source/schedule", "initialization", "geometry", "checkpoint selection"],
            "causal_or_same_condition_comparison": False}
    per_frame = []
    for index, (a, b) in enumerate(zip(first, second)):
        row = {"row": a["identity"][0], "scenario": a["scenario"], "session": a["session"],
               "frame": a["frame"], "bucket": a["bucket"], "gt_abs_xy": a["source"]["gt_abs_xy"],
               "s0": {key: a["source"][key] for key in ("pred_abs_xy", "d3", "point_l2",
                       "longitudinal_error", "lateral_error")},
               "s1": {key: b["source"][key] for key in ("pred_abs_xy", "d3", "point_l2",
                       "longitudinal_error", "lateral_error")},
               "s1_minus_s0": {"d3": float(d1[index] - d0[index]),
                    "point_l2": (b["point"] - a["point"]).tolist(),
                    "longitudinal_error": (b["error"][:, 0] - a["error"][:, 0]).tolist(),
                    "lateral_error": (b["error"][:, 1] - a["error"][:, 1]).tolist()}}
        if c0 is not None:
            row["historical_c0_d3"] = float(c0["conditions"]["normal"]["records"][index]["d3"])
        per_frame.append(row)
    return {"schema_version": 1, "status": "completed_cpu_analysis_of_actual_reports",
        "scope": "P4 LAST6000 normal full tune1998; final validation never accessed",
        "lineage": {"training_git_sha": P4_GIT_SHA,
            "evaluation_git_sha": s0["protocol"]["source"]["git_sha"],
            "evaluation_source_file_sha256": s0["protocol"]["source"]["file_sha256"],
            "training_and_evaluation_git_may_differ": True,
            "seed_evaluation_sources_exactly_equal": True},
        "seeds": seed_summaries,
        "two_seed_description": {"values": means.tolist(), "mean": float(means.mean()),
            "population_std_ddof0": float(means.std(ddof=0)), "min": float(means.min()),
            "max": float(means.max()), "n_seeds": 2,
            "distribution_established_by_two_seeds": False, "best_of_two_selection_performed": False},
        "paired_seed_comparison": paired, "historical_c0": historical,
        "per_frame_p4_results": per_frame,
        "bootstrap": {"unit": "eleven sessions sampled uniformly with replacement",
            "within_session": "all fixed frames retained; sums/counts preserve the frame-weighted estimator",
            "frame_iid_bootstrap": False, "repeats": repeats, "seed": bootstrap_seed},
        "interpretation": {"two_seed_statistics_are_descriptive": True,
            "leaderboard_or_final_performance_claim": False,
            "selection_optimism_warning": "Do not report the better of two seeds as an unbiased estimate.",
            "c0_is_historical_only": c0 is not None},
        "training_completion_receipt": {
            "completed_evaluator_protocol_verified_here": True,
            "completed_training_sidecar_verified_here": False,
            "required_external_precondition": "Each LAST6000 must come from its immutable completed run sidecar; evaluator JSON embeds the LAST trainer manifest but not that sidecar.",
            "operator_reported_supervisor_outcome": training_integrity,
            "supervisor_exit_zero_asserted": all(
                value == "supervisor_exit0" for value in training_integrity.values()),
            "integrity_exceptions_reported": [seed for seed, value in training_integrity.items()
                                               if value.endswith("rc1_preserved")],
            "operator_report_is_not_validated_or_relabelled_by_this_analyzer": True},
        "model_forward_performed": False, "gpu_used": False, "final_val_accessed": False,
        "cpu_unit_tests_are_not_model_results": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--s0-report", required=True)
    parser.add_argument("--expected-s0-sha256", required=True)
    parser.add_argument("--s1-report", required=True)
    parser.add_argument("--expected-s1-sha256", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--c0-report")
    parser.add_argument("--expected-c0-sha256")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260908)
    parser.add_argument("--s0-training-integrity", choices=TRAINING_INTEGRITY_OUTCOMES,
                        default="not_supplied")
    parser.add_argument("--s1-training-integrity", choices=TRAINING_INTEGRITY_OUTCOMES,
                        default="not_supplied")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    require((args.c0_report is None) == (args.expected_c0_sha256 is None),
            "C0 path and expected SHA must be supplied together")
    inputs = {}
    for name, path, expected in (("s0", args.s0_report, args.expected_s0_sha256),
                                 ("s1", args.s1_report, args.expected_s1_sha256),
                                 ("split", args.split_manifest, RAWTIME_SPLIT_SHA256)):
        inputs[name] = (*read_pinned_json(path, expected), Path(path))
    if args.c0_report:
        inputs["c0"] = (*read_pinned_json(args.c0_report, args.expected_c0_sha256), Path(args.c0_report))
    target = Path(args.out).absolute()
    require(not os.path.lexists(target) and target.parent.is_dir(), "Output must be new in an existing directory")
    result = analyze(inputs["s0"][0], inputs["s1"][0], inputs["split"][0],
                     repeats=args.bootstrap_repeats, bootstrap_seed=args.bootstrap_seed,
                     c0=inputs.get("c0", (None,))[0],
                     training_integrity={"s0": args.s0_training_integrity,
                                         "s1": args.s1_training_integrity})
    before = {name: sha256_bytes(item[1]) for name, item in inputs.items()}
    require(all(item[2].read_bytes() == item[1] for item in inputs.values()), "Input changed during analysis")
    result["input_sha256"] = before
    result["analysis_script_sha256"] = sha256_bytes(Path(__file__).read_bytes())
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "s0_d3": result["seeds"]["s0"]["all"]["official_d3"],
                      "s1_d3": result["seeds"]["s1"]["all"]["official_d3"],
                      "output": str(target), "sha256": sha256_bytes(target.read_bytes())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
