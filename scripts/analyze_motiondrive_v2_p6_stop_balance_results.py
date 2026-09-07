#!/usr/bin/env python3
"""CPU-only, fail-closed analysis of the preregistered P6 four-arm LAST reports.

The script consumes saved predictions only.  It performs no model forward,
training, checkpoint selection, threshold fitting, or final-validation access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_motiondrive_v2_query_pair import paired_cluster_bootstrap


EXPECTED_N = 1998
EXPECTED_SCENES = 37
EXPECTED_SESSIONS = 11
EXPECTED_BUCKETS = {"steady": 99, "depart": 24, "nonstop": 1875}
SESSION_046 = "session_046_20260210-100504"
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 20260908
P4_REFERENCE_SHA256 = {
    0: "cffa9cc8b12f5f9812af7b3fd3631e74597e291b02aecd1cc14d8ec8a1aeba2f",
    1: "0e71bef199931cb1916abdc819ca78cf75d8cd7055439bd084ca975bb610c137",
}
FP32_AUDIT_TOLERANCE = float(8 * np.finfo(np.float32).eps)
OFFICIAL_TIME_WEIGHTS = (11 / 36, 11 / 36, 5 / 36, 5 / 36, 2 / 36, 2 / 36)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def valid_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_pinned_json(path, expected_sha):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Ordinary JSON input required: {path}")
    require(valid_sha(expected_sha), f"Explicit lowercase SHA256 required: {path}")
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected_sha, f"Input SHA256 mismatch: {path}")

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant in {path}: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    require(isinstance(value, dict), f"JSON root must be a mapping: {path}")
    return value, raw


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float32)
    require(result.shape == shape and np.isfinite(result).all(), f"Invalid {name}: {result.shape}")
    return result


def _max_gt_displacement(gt):
    return float(torch.linalg.vector_norm(torch.from_numpy(gt), dim=-1).max())


def _identity(record):
    require(type(record.get("row")) is int and type(record.get("frame")) is int,
            "Record row/frame must be integers")
    require(isinstance(record.get("scenario"), str) and record["scenario"]
            and isinstance(record.get("session"), str) and record["session"],
            "Record scenario/session identity missing")
    return (record["row"], record["scenario"], record["session"], record["frame"])


def _bucket_from_saved_gt(record, gt):
    require(type(record.get("stop_valid")) is bool, "P6 stop_valid must be boolean")
    if not record["stop_valid"]:
        require(record.get("stop_target") is None, "Invalid stop target must be null")
        return "invalid_stop"
    target = record.get("stop_target")
    require(type(target) in (int, float) and math.isfinite(float(target))
            and float(target) in (0., 1.), "Valid stop target must be exact binary")
    if float(target) < .5:
        return "nonstop"
    maximum = _max_gt_displacement(gt)
    return "steady" if maximum <= .2 else "depart"


def _d3_audit(predictions, ground_truth, stored):
    pred = torch.from_numpy(np.stack(predictions).astype(np.float32, copy=False))
    gt = torch.from_numpy(np.stack(ground_truth).astype(np.float32, copy=False))
    distance = torch.linalg.vector_norm(pred - gt, dim=-1)
    recomputed = (distance * distance.new_tensor(OFFICIAL_TIME_WEIGHTS)).sum(-1).numpy().astype(np.float64)
    stored = np.asarray(stored, dtype=np.float64)
    difference = np.abs(recomputed - stored)
    allowed = FP32_AUDIT_TOLERANCE + FP32_AUDIT_TOLERANCE * np.abs(stored)
    require(np.isfinite(stored).all() and np.all(difference <= allowed),
            f"Stored D3 differs from fixed FP32 recomputation: max={float(difference.max())}")
    return recomputed, {
        "rtol": FP32_AUDIT_TOLERANCE,
        "atol": FP32_AUDIT_TOLERANCE,
        "bitwise_unequal": int(np.count_nonzero(recomputed.astype(np.float32)
                                                 != stored.astype(np.float32))),
        "max_abs_difference": float(difference.max(initial=0.)),
    }


def validate_p6_report(payload, label):
    require(set(("report", "records")) <= set(payload), f"{label}: final_eval wrapper incomplete")
    report, source = payload["report"], payload["records"]
    require(isinstance(report, dict) and isinstance(source, list) and len(source) == EXPECTED_N,
            f"{label}: expected one LAST tune{EXPECTED_N} report")
    require(report.get("kind") == "eval" and report.get("step") == 1000
            and report.get("time_input") == "nominal" and report.get("n") == EXPECTED_N
            and report.get("n_sessions") == EXPECTED_SESSIONS,
            f"{label}: LAST1000/tune/time contract mismatch")
    identities, predictions, ground_truth, stored_d3, buckets, logits = [], [], [], [], [], []
    normalized = []
    for item in source:
        require(isinstance(item, dict), f"{label}: malformed record")
        identity = _identity(item)
        pred = _array(item.get("pred_abs_xy"), (6, 2), f"{label} pred_abs_xy")
        gt = _array(item.get("gt_abs_xy"), (6, 2), f"{label} gt_abs_xy")
        d3 = item.get("d3")
        require(type(d3) in (int, float) and math.isfinite(float(d3)) and float(d3) >= 0,
                f"{label}: invalid stored D3")
        bucket = _bucket_from_saved_gt(item, gt)
        require(item.get("bucket") == bucket and item.get("stop_bucket") == bucket,
                f"{label}: stored stop/future bucket mismatch")
        maximum = _max_gt_displacement(gt)
        require(type(item.get("max_gt_displacement_m")) in (int, float)
                and float(item["max_gt_displacement_m"]) == maximum,
                f"{label}: max GT displacement mismatch")
        state = _array(item.get("pred_state"), (6,), f"{label} pred_state")
        gt_state = _array(item.get("gt_state"), (6,), f"{label} gt_state")
        gt_valid = item.get("gt_state_valid")
        require(isinstance(gt_valid, list) and len(gt_valid) == 6
                and all(type(x) is bool for x in gt_valid), f"{label}: GT state mask invalid")
        require(gt_valid[5] == item["stop_valid"], f"{label}: stop validity/state mask mismatch")
        if item["stop_valid"]:
            require(float(gt_state[5]) == float(item["stop_target"]),
                    f"{label}: stop target/GT state mismatch")
        identities.append(identity); predictions.append(pred); ground_truth.append(gt)
        stored_d3.append(float(d3)); buckets.append(bucket); logits.append(float(state[5]))
        normalized.append({"identity": identity, "pred": pred, "gt": gt, "d3": float(d3),
                           "bucket": bucket, "stop_target": item.get("stop_target"),
                           "stop_valid": item["stop_valid"], "stop_logit": float(state[5]),
                           "gt_state": gt_state, "gt_state_valid": tuple(gt_valid)})
    require(len(set(identities)) == EXPECTED_N, f"{label}: duplicate row identities")
    require(len(set(x[1] for x in identities)) == EXPECTED_SCENES,
            f"{label}: expected 37 tune scenes")
    require(len(set(x[2] for x in identities)) == EXPECTED_SESSIONS,
            f"{label}: expected eleven sessions")
    observed_buckets = {name: buckets.count(name) for name in (*EXPECTED_BUCKETS, "invalid_stop")}
    require(observed_buckets == {**EXPECTED_BUCKETS, "invalid_stop": 0},
            f"{label}: preregistered tune stop-bucket counts differ: {observed_buckets}")
    recomputed, audit = _d3_audit(predictions, ground_truth, stored_d3)
    official = report.get("official_d3")
    require(type(official) in (int, float) and math.isfinite(float(official)),
            f"{label}: official D3 missing")
    record_mean = float(np.asarray(stored_d3, dtype=np.float64).mean())
    require(float(official) == record_mean,
            f"{label}: official D3 is not the exact trainer mean of saved rows")
    audit.update(recomputed_mean=float(recomputed.mean()), stored_official_d3=float(official))
    return normalized, audit


def validate_p4_reference(payload, base_seed):
    require(payload.get("status") == "completed" and payload.get("selection_performed") is False
            and payload.get("final_val_accessed") is False and payload.get("time_input") == "nominal"
            and payload.get("precision_requested") == "bf16", "Immutable P4 reference contract mismatch")
    checkpoint = payload.get("protocol", {}).get("checkpoint_manifest", {})
    require(payload.get("protocol", {}).get("checkpoint_step") == 6000
            and checkpoint.get("arguments", {}).get("seed") == base_seed,
            "P4 reference checkpoint/base-seed mismatch")
    condition = payload.get("conditions", {}).get("normal")
    require(isinstance(condition, dict) and isinstance(condition.get("records"), list)
            and len(condition["records"]) == EXPECTED_N, "P4 normal full-tune records missing")
    normalized, predictions, ground_truth, stored_d3 = [], [], [], []
    for item in condition["records"]:
        identity = _identity(item)
        pred = _array(item.get("pred_abs_xy"), (6, 2), "P4 pred_abs_xy")
        gt = _array(item.get("gt_abs_xy"), (6, 2), "P4 gt_abs_xy")
        state = _array(item.get("pred_state"), (6,), "P4 pred_state")
        gt_state = _array(item.get("gt_state"), (6,), "P4 gt_state")
        gt_valid = item.get("gt_state_valid")
        require(isinstance(gt_valid, list) and len(gt_valid) == 6
                and all(type(x) is bool for x in gt_valid), "P4 GT state mask invalid")
        stop_valid = gt_valid[5]
        stop_target = float(gt_state[5]) if stop_valid else None
        derived = {"stop_valid": stop_valid, "stop_target": stop_target}
        bucket = _bucket_from_saved_gt(derived, gt)
        d3 = item.get("d3")
        require(type(d3) in (int, float) and math.isfinite(float(d3)) and float(d3) >= 0,
                "P4 stored D3 invalid")
        normalized.append({"identity": identity, "pred": pred, "gt": gt, "d3": float(d3),
                           "bucket": bucket, "stop_target": stop_target, "stop_valid": stop_valid,
                           "stop_logit": float(state[5]), "gt_state": gt_state,
                           "gt_state_valid": tuple(gt_valid)})
        predictions.append(pred); ground_truth.append(gt); stored_d3.append(float(d3))
    require(len({x["identity"] for x in normalized}) == EXPECTED_N, "P4 identities duplicate")
    require(len({x["identity"][1] for x in normalized}) == EXPECTED_SCENES
            and len({x["identity"][2] for x in normalized}) == EXPECTED_SESSIONS,
            "P4 reference tune scene/session inventory differs")
    buckets = [x["bucket"] for x in normalized]
    require({name: buckets.count(name) for name in (*EXPECTED_BUCKETS, "invalid_stop")}
            == {**EXPECTED_BUCKETS, "invalid_stop": 0}, "P4 GT stop buckets differ")
    _, audit = _d3_audit(predictions, ground_truth, stored_d3)
    summary = condition.get("summary", {})
    require(summary.get("n") == EXPECTED_N
            and abs(float(summary.get("official_d3")) - float(np.mean(stored_d3))) <= 1e-12,
            "P4 reference summary differs from saved rows")
    audit.update(stored_official_d3=float(summary["official_d3"]), base_seed=base_seed)
    return normalized, audit


def require_same_labels(reference, reports):
    reference_id = [x["identity"] for x in reference]
    reference_gt = [x["gt"] for x in reference]
    reference_state = [x["gt_state"] for x in reference]
    reference_valid = [x["gt_state_valid"] for x in reference]
    reference_bucket = [x["bucket"] for x in reference]
    for label, rows in reports.items():
        require([x["identity"] for x in rows] == reference_id, f"{label}: row identity/order mismatch")
        require(all(np.array_equal(a, b) for a, b in zip(reference_gt, [x["gt"] for x in rows])),
                f"{label}: GT plan differs")
        require(all(np.array_equal(a, b) for a, b in zip(reference_state, [x["gt_state"] for x in rows])),
                f"{label}: GT state differs")
        require([x["gt_state_valid"] for x in rows] == reference_valid,
                f"{label}: GT state validity differs")
        require([x["bucket"] for x in rows] == reference_bucket, f"{label}: GT bucket differs")


def _group(values, mask):
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    require(selected.size > 0 and np.isfinite(selected).all(), "Empty/nonfinite metric group")
    return {"n": int(selected.size), "official_d3": float(selected.mean())}


def summarize_variant(rows):
    values = np.asarray([x["d3"] for x in rows], dtype=np.float64)
    buckets = np.asarray([x["bucket"] for x in rows])
    sessions = np.asarray([x["identity"][2] for x in rows])
    return {"all": _group(values, np.ones(len(rows), bool)),
            "buckets": {name: _group(values, buckets == name) for name in EXPECTED_BUCKETS},
            "excluding_session_046": _group(values, sessions != SESSION_046)}


def _paired(left, right, sessions):
    delta = np.asarray(left, np.float64) - np.asarray(right, np.float64)
    result = paired_cluster_bootstrap(delta, sessions, repeats=BOOTSTRAP_REPEATS,
                                      seed=BOOTSTRAP_SEED)
    result.update(estimator="frame-weighted mean D3 difference",
                  frame_iid_bootstrap=False)
    return result


def summarize_comparison(left_rows, right_rows):
    left = np.asarray([x["d3"] for x in left_rows], np.float64)
    right = np.asarray([x["d3"] for x in right_rows], np.float64)
    sessions = np.asarray([x["identity"][2] for x in left_rows])
    buckets = np.asarray([x["bucket"] for x in left_rows])
    result = {"all": _paired(left, right, sessions), "buckets": {},
              "excluding_session_046": _paired(left[sessions != SESSION_046],
                                                 right[sessions != SESSION_046],
                                                 sessions[sessions != SESSION_046])}
    for name in EXPECTED_BUCKETS:
        mask = buckets == name
        result["buckets"][name] = {"n": int(mask.sum()),
                                    "delta_d3": float((left[mask] - right[mask]).mean())}
    return result


def _average_rank_auc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, np.float64)
    require(labels.shape == scores.shape and labels.ndim == 1 and np.isfinite(scores).all(),
            "AUC arrays invalid")
    positives, negatives = int(labels.sum()), int((~labels).sum())
    require(positives > 0 and negatives > 0, "AUC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = .5 * (start + 1 + end)
        start = end
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def stop_logit_summary(rows):
    logits = np.asarray([x["stop_logit"] for x in rows], np.float64)
    labels = np.asarray([bool(x["stop_target"]) for x in rows], bool)
    require(all(x["stop_valid"] for x in rows), "Preregistered tune stop labels must all be valid")

    def stats(mask):
        value = logits[mask]
        return {"n": int(value.size), "mean": float(value.mean()),
                "std_ddof0": float(value.std(ddof=0)),
                "q05_q25_q50_q75_q95": np.quantile(value, [.05, .25, .5, .75, .95]).tolist()}

    return {"raw_logit_only_no_threshold_selected": True,
            "current_stop_roc_auc": _average_rank_auc(labels, logits),
            "current_stop_positive": stats(labels), "current_stop_negative": stats(~labels),
            "by_future_bucket": {name: stats(np.asarray([x["bucket"] == name for x in rows]))
                                 for name in EXPECTED_BUCKETS}}


def analyze(reports, references):
    for base in (0, 1):
        require_same_labels(references[base], {
            f"b{base}_control": reports[(base, "control")],
            f"b{base}_balanced": reports[(base, "balanced")],
            f"b{1-base}_reference": references[1-base],
            f"b{1-base}_control": reports[(1-base, "control")],
            f"b{1-base}_balanced": reports[(1-base, "balanced")],
        })
    variants = {}
    comparisons = {}
    for base in (0, 1):
        control, balanced, original = (reports[(base, "control")],
                                       reports[(base, "balanced")], references[base])
        variants[f"base{base}"] = {
            "original_p4": summarize_variant(original),
            "control": summarize_variant(control), "balanced": summarize_variant(balanced),
            "stop_logit_descriptive": {
                "original_p4": stop_logit_summary(original),
                "control": stop_logit_summary(control), "balanced": stop_logit_summary(balanced)},
        }
        comparisons[f"base{base}"] = {
            "balanced_minus_control": summarize_comparison(balanced, control),
            "balanced_minus_original_p4": summarize_comparison(balanced, original),
            "control_minus_original_p4": summarize_comparison(control, original),
        }
    sessions = [x["identity"][2] for x in references[0]]
    arrays = {name: [np.asarray([x["d3"] for x in rows], np.float64) for rows in pair]
              for name, pair in {
                  "balanced": [reports[(0, "balanced")], reports[(1, "balanced")]],
                  "control": [reports[(0, "control")], reports[(1, "control")]],
                  "original_p4": [references[0], references[1]],
              }.items()}
    means = {name: [float(x.mean()) for x in values] for name, values in arrays.items()}
    mean_balanced_minus_control = .5 * ((arrays["balanced"][0] - arrays["control"][0])
                                        + (arrays["balanced"][1] - arrays["control"][1]))
    mean_balanced_minus_original = .5 * ((arrays["balanced"][0] - arrays["original_p4"][0])
                                         + (arrays["balanced"][1] - arrays["original_p4"][1]))
    shared_bc = paired_cluster_bootstrap(mean_balanced_minus_control, sessions,
                                         repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED)
    shared_bo = paired_cluster_bootstrap(mean_balanced_minus_original, sessions,
                                         repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED)
    shared_masks = {
        **{name: np.asarray([x["bucket"] == name for x in references[0]], bool)
           for name in EXPECTED_BUCKETS},
        "excluding_session_046": np.asarray([value != SESSION_046 for value in sessions], bool),
    }
    shared_groups = {
        name: {"n": int(mask.sum()),
               "balanced_minus_control_d3": float(mean_balanced_minus_control[mask].mean()),
               "balanced_minus_original_p4_d3": float(mean_balanced_minus_original[mask].mean())}
        for name, mask in shared_masks.items()
    }
    per_base_bc = [comparisons[f"base{base}"]["balanced_minus_control"]["all"]["delta"]
                   for base in (0, 1)]
    per_base_bo = [comparisons[f"base{base}"]["balanced_minus_original_p4"]["all"]["delta"]
                   for base in (0, 1)]
    mean_bo = float(np.mean(means["balanced"]) - np.mean(means["original_p4"]))
    checks = {
        "both_bases_balanced_better_than_control": all(value < 0 for value in per_base_bc),
        "both_bases_balanced_better_than_original_p4": all(value < 0 for value in per_base_bo),
        "mean_balanced_minus_original_p4_at_most_minus_0p01": mean_bo <= -.01,
        "shared_session_mean_balanced_minus_control_ci_upper_below_zero": shared_bc["ci95"][1] < 0,
    }
    return {
        "variants": variants, "comparisons": comparisons,
        "two_base_description": {
            name: {"values": values, "mean": float(np.mean(values)),
                   "min": float(np.min(values)), "max": float(np.max(values)), "n_bases": 2}
            for name, values in means.items()},
        "shared_session_bootstrap": {
            "balanced_minus_control": shared_bc,
            "balanced_minus_original_p4": shared_bo,
            "method": "average the two same-row base deltas first; resample the same eleven sessions; never 22 clusters",
            "repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED,
        },
        "shared_two_base_group_description": shared_groups,
        "preregistered_keep_gate": {"checks": checks, "keep": all(checks.values()),
            "mean_balanced_minus_original_p4": mean_bo,
            "meaning": "Exploratory reused-tune KEEP screen only; not final holdout, compliance, deployment, or first-place evidence"},
    }


def write_new_json(path, value):
    path = Path(path)
    require(path.parent.is_dir(), f"Output parent missing: {path.parent}")
    require(not os.path.lexists(path), f"Refusing to overwrite output: {path}")
    def native(item):
        if item is None or type(item) in (bool, int, float, str):
            return item
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        if isinstance(item, np.ndarray):
            return [native(element) for element in item.tolist()]
        if isinstance(item, (list, tuple)):
            return [native(element) for element in item]
        if isinstance(item, dict):
            require(all(type(key) is str for key in item), "JSON output keys must be strings")
            return {key: native(element) for key, element in item.items()}
        raise TypeError(f"Unsupported JSON output type: {type(item).__name__}")

    normalized = native(value)
    serialized = json.dumps(normalized, indent=2, allow_nan=False) + "\n"
    # Exercise the complete serialized representation before publishing it.
    require(json.loads(serialized) == normalized, "JSON output round-trip mismatch")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("b0-control", "b0-balanced", "b1-control", "b1-balanced",
                 "b0-reference", "b1-reference"):
        parser.add_argument(f"--{name}", nargs=2, metavar=("REPORT", "SHA256"), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only analysis requires CUDA_VISIBLE_DEVICES=''")
    require(not torch.cuda.is_initialized(), "CUDA was initialized before CPU-only analysis")
    inputs = {
        (0, "control"): args.b0_control, (0, "balanced"): args.b0_balanced,
        (1, "control"): args.b1_control, (1, "balanced"): args.b1_balanced,
    }
    raw_inputs, reports, audits = {}, {}, {}
    for key, (path, digest) in inputs.items():
        payload, raw = read_pinned_json(path, digest)
        reports[key], audits[f"b{key[0]}_{key[1]}"] = validate_p6_report(
            payload, f"b{key[0]}_{key[1]}")
        raw_inputs[str(Path(path).resolve())] = (digest, raw)
    references = {}
    for base, (path, digest) in enumerate((args.b0_reference, args.b1_reference)):
        require(digest == P4_REFERENCE_SHA256[base], f"P4 base{base} immutable reference SHA mismatch")
        payload, raw = read_pinned_json(path, digest)
        references[base], audits[f"p4_base{base}"] = validate_p4_reference(payload, base)
        raw_inputs[str(Path(path).resolve())] = (digest, raw)
    result = analyze(reports, references)
    for path, (digest, raw) in raw_inputs.items():
        require(Path(path).read_bytes() == raw and sha256_bytes(raw) == digest,
                f"Input changed during analysis: {path}")
    output = {
        "schema_version": 1, "status": "completed_cpu_analysis_of_actual_p6_reports",
        "scope": "P6 four LAST1000 reports and two immutable P4 references; saved arrays only",
        "inputs_sha256_before_after_exact": {path: digest for path, (digest, _) in raw_inputs.items()},
        "numeric_d3_audit": audits, "analysis": result,
        "bootstrap_contract": {"unit": "eleven tune sessions", "repeats": BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED, "frame_iid": False, "seed_bootstrap": False},
        "boundaries": {"model_forward": False, "training": False, "threshold_selection": False,
            "final_validation_accessed": False, "gpu_used": False,
            "actual_performance_was_pending_when_analyzer_was_authored": True,
            "reused_tune_is_exploratory": True},
        "environment": {"python": os.sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_initialized": torch.cuda.is_initialized(),
            "analyzer_sha256": sha256_bytes(Path(__file__).read_bytes())},
    }
    write_new_json(args.output, output)
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "keep": result["preregistered_keep_gate"]["keep"],
                      "gpu_used": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
