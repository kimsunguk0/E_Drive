#!/usr/bin/env python3
"""CPU analysis for preregistered P0 BN Pair C: adaptive vs fixed statistics.

Both arms use scale(10,5), identical initial state, new AdamW, and LAST500.
Normal eval train16 is the fitting gate. Tune12/batch-stat results are diagnostics,
not checkpoint selection or evidence of heldout generalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_motiondrive_v2_tail_pair import recompute_split


def compare_bn_fit_reports(adaptive, fixed, baseline=None):
    result = {}
    for name, report in (("adaptive", adaptive), ("fixed", fixed)):
        last = report["checkpoints"]["last"]
        if last["step"] != 500:
            raise ValueError(f"{name}: use LAST500 only, got {last['step']}")
        result[name] = {split: recompute_split(last[split]) for split in ("train", "tune")}
        if result[name]["train"]["n"] != 16 or result[name]["tune"]["n"] != 12:
            raise ValueError("Pair C requires exactly train16/tune12")
    for split in ("train", "tune"):
        a, b = [p["checkpoints"]["last"][split] for p in (adaptive, fixed)]
        if a["records"] != b["records"] or not np.array_equal(a["targets"], b["targets"]):
            raise ValueError(f"{split}: row order or GT differs")
    a, b = result["adaptive"]["train"], result["fixed"]["train"]
    a_pass = a["d3"] <= .15 and a["l2_by_time"][5] <= 1.
    b_pass = b["d3"] <= .15 and b["l2_by_time"][5] <= 1.
    early_delta = b["first2_official_renormalized_d3"] - a["first2_official_renormalized_d3"]
    if early_delta > .02:
        decision = "DO_NOT_AUTO_ADOPT_FIXED_EARLY_HARM"
    elif a_pass and b_pass:
        decision = "BOTH_PASS_PREFER_EXISTING_ADAPTIVE_UNLESS_FURTHER_EVIDENCE"
    elif b_pass:
        decision = "FIXED_ONLY_PASSES_VERIFY_TRANSFER_TO_LARGER_TRAINING"
    elif a_pass:
        decision = "ADAPTIVE_ONLY_PASSES_KEEP_ADAPTIVE"
    else:
        decision = "NEITHER_PASSES_CONTINUE_P0"
    report = {
        "protocol": "MOTIONDRIVE_V2_P0_BN_PROTOCOL.md / Pair C; LAST500 normal eval",
        "first2_definition": "sum([11,11,5,5] * per-time L2[:4]) / 32",
        "arms": result,
        "gates": {"adaptive_fit_pass": bool(a_pass), "fixed_fit_pass": bool(b_pass),
                  "fit_threshold_d3": .15, "fit_threshold_3s_l2": 1.,
                  "first2_fixed_minus_adaptive": early_delta, "first2_harm_over_0p02": early_delta > .02},
        "deltas_fixed_minus_adaptive": {
            "train_d3": b["d3"] - a["d3"],
            "train_l2_by_time": (np.asarray(b["l2_by_time"]) - np.asarray(a["l2_by_time"])).tolist(),
            "train_first2_normalized_d3": early_delta,
            "tune_d3_diagnostic_only": result["fixed"]["tune"]["d3"] - result["adaptive"]["tune"]["d3"]},
        "screening_decision": decision,
        "interpretation_limits": [
            "This diagnostic compares BN training policies, not frozen vs trainable backbone weights.",
            "Both arms use the same physical output scale(10,5) and all trainable model weights remain trainable.",
            "The 12-frame tune set and training batch-stat evaluation cannot choose the winner or checkpoint.",
            "A one-seed train16 fit improvement does not establish heldout accuracy; transfer to larger training must be checked separately."]}
    if baseline is not None:
        base = baseline["checkpoints"]["last"]["train"]
        target = adaptive["checkpoints"]["last"]["train"]
        if base["records"] != target["records"] or not np.array_equal(base["targets"], target["targets"]):
            raise ValueError("Previous baseline row/GT mismatch")
        report["previous_stage_train"] = recompute_split(base)
        report["adaptive_extension_delta_d3"] = a["d3"] - report["previous_stage_train"]["d3"]
        report["fixed_total_delta_d3"] = b["d3"] - report["previous_stage_train"]["d3"]
    return report


def validate_pair_manifests(adaptive, fixed, adaptive_logs, fixed_logs):
    for expected, m in (("adaptive", adaptive), ("fixed", fixed)):
        if m.get("status") != "completed" or m.get("step") != 500:
            raise ValueError("Both runs must complete exactly 500 steps")
        if m["arguments"].get("resume"):
            raise ValueError("Resumed arm is not permitted by Pair C")
        if m["arguments"].get("bn_policy") != expected or m["bn_training"]["policy"] != expected:
            raise ValueError("BN policy does not match arm")
        if not m["bn_training"].get("affine_and_backbone_weights_trainable"):
            raise ValueError("BN affine and backbone must remain trainable")
        if tuple(m["model_config"].get("plan_output_scale", (1, 1))) != (10, 5):
            raise ValueError("Both Pair C arms require scale(10,5)")
    equal = ("git_sha", "initial_model_state_sha256", "initial_parameter_count", "model_config",
             "split_sha256", "train_rows_sha256", "eval_rows_sha256", "supervision_manifest_sha256",
             "loss_weights", "data_counts")
    for key in equal:
        if key not in adaptive or key not in fixed or adaptive[key] != fixed[key]:
            raise ValueError(f"Initial/data/architecture lineage mismatch: {key}")
    args_equal = ("seed", "batch", "steps", "warmup", "lr", "backbone_lr", "weight_decay", "grad_clip",
                  "phase", "init", "precision", "train_stride", "eval_stride", "train_scenes", "eval_scenes",
                  "max_train_samples", "max_eval_samples")
    for key in args_equal:
        if adaptive["arguments"].get(key) != fixed["arguments"].get(key):
            raise ValueError(f"Non-BN training condition differs: {key}")
    source_sha = adaptive["load_report"].get("common_checkpoint_sha256")
    if not source_sha or source_sha != fixed["load_report"].get("common_checkpoint_sha256"):
        raise ValueError("Initial source checkpoint differs or is undocumented")
    digests = [{r["step"]: r.get("sample_order_sha256") for r in rows if r["kind"] == "train"}
               for rows in (adaptive_logs, fixed_logs)]
    if any(500 not in d or any(v is None for v in d.values()) for d in digests) or digests[0] != digests[1]:
        raise ValueError("Cumulative training-row trace mismatch or missing")
    return {"initial_state_sha256_equal": True, "only_bn_policy_differs": True,
            "all_logged_row_digests_match": True, "logged_steps_checked": len(digests[0]),
            "initial_model_state_sha256": adaptive["initial_model_state_sha256"],
            "final_sample_order_sha256": digests[0][500]}


def checkpoint_policy_integrity(initial_adaptive, initial_fixed, last_adaptive, last_fixed):
    """CPU tensor check: equal initialization, fixed stats, and learned weights."""
    if set(initial_adaptive) != set(initial_fixed):
        raise ValueError("Initial checkpoint keys differ")
    for key in initial_adaptive:
        if not torch.equal(initial_adaptive[key], initial_fixed[key]):
            raise ValueError(f"Initial checkpoint differs: {key}")
    suffixes = ("running_mean", "running_var", "num_batches_tracked")
    bn_keys = [key for key in initial_adaptive if key.startswith("backbone_fpn.") and key.endswith(suffixes)]
    if not bn_keys:
        raise ValueError("No backbone BN running buffers found")
    fixed_changed = [k for k in bn_keys if not torch.equal(initial_fixed[k], last_fixed[k])]
    adaptive_changed = [k for k in bn_keys if not torch.equal(initial_adaptive[k], last_adaptive[k])]
    if fixed_changed:
        raise ValueError(f"Fixed BN running buffers changed: {fixed_changed[:4]}")
    if not adaptive_changed:
        raise ValueError("Adaptive BN running buffers did not update")
    representatives = ("backbone_fpn.stem.0.weight", "backbone_fpn.stem.1.weight",
                       "backbone_fpn.lat2.weight", "planner.xy_head.3.weight")
    weights = {}
    for name, old, new in (("adaptive", initial_adaptive, last_adaptive), ("fixed", initial_fixed, last_fixed)):
        weights[name] = {}
        for key in representatives:
            if key in old and key in new:
                weights[name][key] = {"changed": not torch.equal(old[key], new[key]),
                                     "max_abs_delta": float((old[key].float() - new[key].float()).abs().max())}
    return {"initial_state_tensors_bit_identical": True, "fixed_bn_running_buffers_unchanged": True,
            "bn_running_buffer_count": len(bn_keys), "adaptive_bn_running_buffers_changed": len(adaptive_changed),
            "representative_trainable_weight_changes": weights}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adaptive-probe", required=True)
    p.add_argument("--fixed-probe", required=True)
    p.add_argument("--adaptive-run", required=True)
    p.add_argument("--fixed-run", required=True)
    p.add_argument("--baseline-probe")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    runs = [Path(args.adaptive_run), Path(args.fixed_run)]
    manifests = [read(run / "manifest.json") for run in runs]
    logs = [[json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()] for run in runs]
    report = compare_bn_fit_reports(read(args.adaptive_probe), read(args.fixed_probe),
                                   read(args.baseline_probe) if args.baseline_probe else None)
    report["fairness"] = validate_pair_manifests(*manifests, *logs)
    checkpoints = [[torch.load(run / f"{which}.pth", map_location="cpu", weights_only=False)
                    for which in ("initial", "last")] for run in runs]
    if any(pair[1]["step"] != 500 for pair in checkpoints):
        raise ValueError("Actual checkpoint step differs from LAST500")
    report["checkpoint_policy_integrity"] = checkpoint_policy_integrity(
        checkpoints[0][0]["model"], checkpoints[1][0]["model"],
        checkpoints[0][1]["model"], checkpoints[1][1]["model"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"gates": report["gates"], "screening_decision": report["screening_decision"],
                      "fairness": report["fairness"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
