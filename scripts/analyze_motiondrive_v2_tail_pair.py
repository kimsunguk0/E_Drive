#!/usr/bin/env python3
"""CPU-only, predeclared LAST500 analysis for the P0 XY-unit pair.

Consumes reports from probe_motiondrive_v2_fit.py; does not select checkpoints,
run GPUs, fit parameters, or reinterpret the 12-frame tune set as generalization.
First-2s means the official first-four weights renormalized to sum to one;
plain ADE@2s is also shown, but is not substituted after looking at results.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from probe_motiondrive_v2_fit import WEIGHTS, summary

EXPECTED_STEP = 500
FIRST2_TOLERANCE = .02


def first2_metric(l2_by_time):
    error = np.asarray(l2_by_time, float)
    if error.shape != (6,) or not np.isfinite(error).all():
        raise ValueError("Expected six finite per-time errors")
    return float(error[:4] @ WEIGHTS[:4] / WEIGHTS[:4].sum())


def recompute_split(raw):
    pred, gt = np.asarray(raw["predictions"], float), np.asarray(raw["targets"], float)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[1:] != (6, 2):
        raise ValueError("Expected matching [N,6,2] predictions/targets")
    if not (np.isfinite(pred).all() and np.isfinite(gt).all()):
        raise ValueError("Nonfinite predictions or targets")
    if len(raw["records"]) != len(pred):
        raise ValueError("Record count does not match prediction count")
    result = summary(pred, gt)
    result["first2_official_renormalized_d3"] = first2_metric(result["l2_by_time"])
    result["first2_plain_ade"] = float(np.mean(result["l2_by_time"][:4]))
    result["tail_d3_contribution"] = float(np.asarray(result["l2_by_time"])[4:] @ WEIGHTS[4:])
    return result


def compare_probe_reports(control, scaled, baseline=None):
    """Use fixed LAST500 only and reject mismatched row order or labels."""
    results = {}
    for label, report in (("unit1", control), ("unit10x5", scaled)):
        last = report["checkpoints"]["last"]
        if last["step"] != EXPECTED_STEP:
            raise ValueError(f"{label}: require LAST{EXPECTED_STEP}, got {last['step']}")
        results[label] = {split: recompute_split(last[split]) for split in ("train", "tune")}
        if results[label]["train"]["n"] != 16 or results[label]["tune"]["n"] != 12:
            raise ValueError("Pair A requires exactly train16/tune12")
    for split in ("train", "tune"):
        a = control["checkpoints"]["last"][split]
        b = scaled["checkpoints"]["last"][split]
        if a["records"] != b["records"]:
            raise ValueError(f"{split}: row identity/order mismatch")
        if not np.array_equal(np.asarray(a["targets"]), np.asarray(b["targets"])):
            raise ValueError(f"{split}: ground-truth mismatch")
    a, b = results["unit1"]["train"], results["unit10x5"]["train"]
    ga = a["d3"] <= .15 and a["l2_by_time"][5] <= 1.
    gb = b["d3"] <= .15 and b["l2_by_time"][5] <= 1.
    early_delta = b["first2_official_renormalized_d3"] - a["first2_official_renormalized_d3"]
    early_harm = early_delta > FIRST2_TOLERANCE
    if early_harm:
        decision = "DO_NOT_AUTO_ADOPT_SCALE_EARLY_HARM"
    elif ga and gb:
        decision = "BOTH_PASS_KEEP_CONTROL_UNLESS_ADDITIONAL_BENEFIT_JUSTIFIES_CHANGE"
    elif gb:
        decision = "SCALED_PASSES_CONTROL_FAILS_ELIGIBLE_FOR_FOLLOWUP"
    elif ga:
        decision = "CONTROL_PASSES_KEEP_UNIT1"
    else:
        decision = "NEITHER_PASSES_CONTINUE_P0"
    report = {
        "schema_version": 1,
        "protocol": "MOTIONDRIVE_V2_P0_REPAIR_PROTOCOL.md / Pair A; LAST500, normal eval mode",
        "first2_definition": "sum([11,11,5,5]*per_time_L2[:4])/32; not plain ADE@2s",
        "arms": results,
        "gates": {"unit1_fit_pass": bool(ga), "unit10x5_fit_pass": bool(gb),
                  "fit_threshold_d3": .15, "fit_threshold_l2_3s": 1.,
                  "first2_scaled_minus_control": early_delta,
                  "first2_harm_over_0p02": bool(early_harm)},
        "paired_deltas_scaled_minus_control": {
            "train_d3": b["d3"] - a["d3"],
            "train_l2_by_time": (np.asarray(b["l2_by_time"]) - np.asarray(a["l2_by_time"])).tolist(),
            "train_first2_renormalized": early_delta,
            "tune_d3_diagnostic_only": results["unit10x5"]["tune"]["d3"] - results["unit1"]["tune"]["d3"]},
        "screening_decision": decision,
        "interpretation_limits": [
            "This is a train-fitting and optimizer-unit diagnostic, not evidence of unseen-scenario accuracy.",
            "Tune12 is reported but never used for checkpoint selection or adoption criteria.",
            "Both runs restart AdamW and add 500 steps; improvement in the control is schedule-extension evidence, not a scale benefit.",
            "A single seed does not establish robustness or a winning score."],
    }
    if baseline is not None:
        old = baseline["checkpoints"]["last"]["train"]
        current = control["checkpoints"]["last"]["train"]
        if old["records"] != current["records"] or not np.array_equal(old["targets"], current["targets"]):
            raise ValueError("Baseline train rows/targets do not match the repair pair")
        old_summary = recompute_split(old)
        report["pre_repair_baseline_train"] = old_summary
        report["schedule_extension_control_delta_d3"] = a["d3"] - old_summary["d3"]
        report["scaled_total_delta_d3"] = b["d3"] - old_summary["d3"]
    return report


def validate_run_pair(control_manifest, scaled_manifest, control_logs, scaled_logs):
    """Validate actual launch state plus the digest of every training row."""
    for name, manifest in (("control", control_manifest), ("scaled", scaled_manifest)):
        if manifest.get("status") != "completed" or manifest.get("step") != EXPECTED_STEP:
            raise ValueError(f"{name} must have completed exactly {EXPECTED_STEP} steps")
        if manifest["arguments"].get("resume"):
            raise ValueError("Pair protocol does not permit a resumed arm")
    equal_fields = ("git_sha", "split_sha256", "train_rows_sha256", "eval_rows_sha256",
                    "supervision_manifest_sha256", "loss_weights", "data_counts")
    for key in equal_fields:
        if key not in control_manifest or key not in scaled_manifest:
            raise ValueError(f"Missing pair lineage field {key}")
        if control_manifest[key] != scaled_manifest[key]:
            raise ValueError(f"Pair lineage mismatch: {key}")
    c, s = copy.deepcopy(control_manifest["model_config"]), copy.deepcopy(scaled_manifest["model_config"])
    if tuple(c.pop("plan_output_scale", (1., 1.))) != (1., 1.):
        raise ValueError("Control output scale must be (1,1)")
    if tuple(s.pop("plan_output_scale", (1., 1.))) != (10., 5.):
        raise ValueError("Treatment output scale must be (10,5)")
    if c != s:
        raise ValueError("Architecture differs outside output units")
    arguments = ("seed", "batch", "steps", "warmup", "lr", "backbone_lr", "weight_decay",
                 "grad_clip", "precision", "phase", "init", "train_stride", "eval_stride",
                 "max_train_samples", "max_eval_samples", "train_scenes", "eval_scenes")
    for name in arguments:
        if control_manifest["arguments"].get(name) != scaled_manifest["arguments"].get(name):
            raise ValueError(f"Training conditions differ: {name}")
    if control_manifest["load_report"].get("common_checkpoint_sha256") != scaled_manifest["load_report"].get("common_checkpoint_sha256"):
        raise ValueError("Initial source checkpoint differs")
    def order_at_steps(logs):
        return {r["step"]: r.get("sample_order_sha256") for r in logs if r["kind"] == "train"}
    order_a, order_b = order_at_steps(control_logs), order_at_steps(scaled_logs)
    if EXPECTED_STEP not in order_a or EXPECTED_STEP not in order_b:
        raise ValueError("Missing final rolling training-row digest")
    if any(v is None for v in list(order_a.values()) + list(order_b.values())):
        raise ValueError("Missing rolling training-row digest")
    if order_a != order_b:
        raise ValueError("Training row order differs between the paired runs")
    return {"lineage_and_conditions_match": True, "rolling_order_matches_all_logged_steps": True,
            "logged_steps_checked": len(order_a), "final_sample_order_sha256": order_a[EXPECTED_STEP],
            "train_rows_sha256": control_manifest["train_rows_sha256"],
            "eval_rows_sha256": control_manifest["eval_rows_sha256"]}


def initial_function_check(control_path, scaled_path):
    """CPU strict state comparison plus physical-output equality on a tiny input.

    Comparing every non-output state proves the only parameter changes are the
    permitted two Linear rows. Tiny-input forward checks FP32 rounding as well;
    the real-resolution initial counterfactual remains part of GPU fit audit.
    """
    from models.motiondrive_v2 import MotionDriveV2
    torch.set_num_threads(4)
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False)
                   for path in (control_path, scaled_path)]
    a, b = [checkpoint["model"] for checkpoint in checkpoints]
    cfg_a, cfg_b = [checkpoint["manifest"]["model_config"] for checkpoint in checkpoints]
    scale_a = np.asarray(cfg_a.get("plan_output_scale", (1., 1.)), float)
    scale_b = np.asarray(cfg_b.get("plan_output_scale", (1., 1.)), float)
    if set(a) != set(b):
        raise ValueError("Initial state keys differ")
    head_keys = {"planner.xy_head.3.weight", "planner.xy_head.3.bias"}
    if not head_keys <= set(a):
        raise ValueError("Expected the reviewed final XY Linear keys")
    for key in set(a) - head_keys:
        if not torch.equal(a[key], b[key]):
            raise ValueError(f"Initial non-output parameter/buffer differs: {key}")
    effective = {}
    for key in head_keys:
        shape = (2, 1) if key.endswith("weight") else (2,)
        wa = a[key].double() * torch.tensor(scale_a).reshape(shape)
        wb = b[key].double() * torch.tensor(scale_b).reshape(shape)
        delta = float((wa - wb).abs().max())
        effective[key] = delta
        if delta >= 1e-6:
            raise ValueError(f"Physical output head differs before training: {key}, max {delta}")
    torch.manual_seed(412)
    matrix = torch.tensor([[48., 20., 0., 0.], [32., 0., -20., 0.],
                           [1., 0., 0., 0.], [0., 0., 0., 1.]])
    inp = dict(images=torch.randn(1, 6, 3, 64, 96), history_images=torch.randn(1, 4, 3, 32, 48),
               lidar2img=matrix[None, None].repeat(1, 6, 1, 1),
               history_transforms=torch.eye(4)[None, None].repeat(1, 4, 1, 1),
               time_offsets=torch.tensor([[.1, .2, .5, 1.]]), goal_xy=torch.tensor([[30., 2.]]))
    predictions = []
    with torch.inference_mode():
        for state, cfg in ((a, cfg_a), (b, cfg_b)):
            model = MotionDriveV2(cfg).cpu().eval()
            model.load_state_dict(state, strict=True)
            predictions.append(model(**inp)["plan_abs"])
            del model
    delta = float((predictions[0] - predictions[1]).abs().max())
    if delta >= 1e-4:
        raise ValueError(f"Initial tiny-input physical plan mismatch {delta}m")
    return {"non_output_state_bit_identical": True, "effective_xy_parameter_max_abs_difference": effective,
            "cpu_tiny_input_max_abs_plan_difference_m": delta, "threshold_m": 1e-4,
            "pass": True, "uses_real_validation_images": False}


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_initial_real_check(data):
    """Verify optional whole-train16 initial raw-image predictions, not just D3."""
    a, b = data["unit1"], data["unit10x5"]
    if a["records"] != b["records"] or len(a["records"]) != 16:
        raise ValueError("Initial real check requires identical train16 row order")
    if not np.array_equal(a["targets"], b["targets"]):
        raise ValueError("Initial real check GT mismatch")
    p, q = np.asarray(a["predictions"], float), np.asarray(b["predictions"], float)
    if p.shape != (16, 6, 2) or q.shape != p.shape or not (np.isfinite(p).all() and np.isfinite(q).all()):
        raise ValueError("Initial real check prediction shape/finite violation")
    delta = float(np.abs(p - q).max())
    if delta >= 1e-4:
        raise ValueError(f"Initial real train16 prediction difference {delta}m exceeds protocol")
    return {"pass": True, "n": 16, "max_abs_plan_difference_m": delta,
            "threshold_m": 1e-4, "uses_real_training_images": True,
            "unit1_initial_train_d3": summary(p, np.asarray(a["targets"]))["d3"],
            "unit10x5_initial_train_d3": summary(q, np.asarray(b["targets"]))["d3"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-probe", required=True)
    p.add_argument("--scaled-probe", required=True)
    p.add_argument("--control-run", required=True)
    p.add_argument("--scaled-run", required=True)
    p.add_argument("--baseline-probe")
    p.add_argument("--initial-real-check", help="Optional GPU-generated real train16 initial outputs JSON; CPU validates it")
    p.add_argument("--output", required=True)
    p.add_argument("--skip-initial-cpu-check", action="store_true",
                   help="Report unverified initialization; never silently assume equality")
    args = p.parse_args()
    control_run, scaled_run = Path(args.control_run), Path(args.scaled_run)
    logs = [[json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
            for path in (control_run, scaled_run)]
    fairness = validate_run_pair(read_json(control_run / "manifest.json"),
                                 read_json(scaled_run / "manifest.json"), *logs)
    report = compare_probe_reports(read_json(args.control_probe), read_json(args.scaled_probe),
                                   read_json(args.baseline_probe) if args.baseline_probe else None)
    report["fairness"] = fairness
    report["initial_function_check"] = ({"pass": None, "reason": "Explicitly skipped; not verified"}
        if args.skip_initial_cpu_check else initial_function_check(control_run / "initial.pth", scaled_run / "initial.pth"))
    if args.skip_initial_cpu_check:
        report["screening_decision"] = "INITIALIZATION_UNVERIFIED_" + report["screening_decision"]
    if args.initial_real_check:
        report["initial_real_function_check"] = validate_initial_real_check(read_json(args.initial_real_check))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"gates": report["gates"], "screening_decision": report["screening_decision"],
                      "initial_function_check": report["initial_function_check"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
