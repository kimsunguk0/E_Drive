#!/usr/bin/env python3
"""Privileged same-forward GT21 intervention for the trained P7-C planner.

This diagnostic is never a candidate output.  For each canonical tune row it
computes image-derived scene/motion features once, then calls the unchanged
planner with either its own predicted status or GT continuous state/history.
The model's predicted raw stop logit is preserved in both calls.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import run_motiondrive_v2_p7_state_sufficiency as probe
from motiondrive_v2_training import model_inputs, tensor_state_sha256, to_device, weighted_d3
from train_motiondrive_v2 import autocast as training_autocast


PROBE_SOURCE_SHA256 = "1ddd1fcf2c47e04283d0ee2e789fd255c4d5c7d537c09bc2f944d53884800852"
GT_TUNE = {
    "artifact_sha256": "8548f5df313eac452cc1ba34c894d5f68bcc1d7337d6122fe7012f6a7d485ea2",
    "manifest_sha256": "53f0cf2e532c20dfe1e957204ba409a9095a6cd59aff6494e4d8c5d5dfd5922e",
}
PRED_TUNE = {
    0: {"artifact_sha256": "dc7e547345938a00407f0a739cb188849cf31f372499ad9ec029b39cc74bdad7",
        "manifest_sha256": "c00fa96a393a6edaf924ea22f1910f07250dbc49a280b0e213aca4ab781b289d"},
    1: {"artifact_sha256": "8b5fb837ceb1c75c36d388b891e76f17df574e072103ae92f7158b2ff2d92e6f",
        "manifest_sha256": "ad58f21b979a84c75ea4150edf8d5437ad8d9553e52388c90caf645f8fa2214f"},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)



def load_pinned_tune_artifact(path, kind, seed=None):
    expected = GT_TUNE if kind == "ground_truth" else PRED_TUNE[seed]
    path = Path(path).resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    require(probe.sha256(path) == expected["artifact_sha256"], f"Pinned {kind} artifact SHA mismatch")
    require(probe.sha256(manifest_path) == expected["manifest_sha256"],
            f"Pinned {kind} manifest SHA mismatch")
    payload, manifest, _ = probe.load_artifact(path, kind, "tune")
    if kind == "prediction":
        require(manifest["base_seed"] == seed, "Prediction artifact seed mismatch")
    return payload


def gt21_planner_inputs(predicted_state, predicted_history, gt_state, gt_history,
                        state_valid=None, history_valid=None):
    """Replace exactly 5 continuous state + 16 history values; retain stop logit."""
    require(predicted_state.ndim == 2 and predicted_state.shape[1] == 6,
            "predicted_state must be [B,6]")
    require(predicted_history.shape == gt_history.shape
            and predicted_history.ndim == 3 and predicted_history.shape[1:] == (4, 4),
            "history tensors must be [B,4,4]")
    require(gt_state.shape == predicted_state.shape, "GT state shape mismatch")
    if state_valid is not None:
        require(state_valid.shape == gt_state.shape and bool(state_valid[:, :5].all()),
                "All five injected GT state fields must be valid")
    if history_valid is not None:
        require(history_valid.shape == gt_history.shape and bool(history_valid.all()),
                "All sixteen injected GT history fields must be valid")
    state = predicted_state.clone()
    state[:, :5] = gt_state[:, :5].to(device=state.device, dtype=state.dtype)
    history = gt_history.to(device=predicted_history.device,
                            dtype=predicted_history.dtype).clone()
    require(torch.equal(state[:, 5], predicted_state[:, 5]),
            "GT21 intervention changed predicted raw stop logit")
    return state, history


def paired_planner_forward(model, parts, gt_state, gt_history, state_valid=None,
                           history_valid=None):
    """Two planner calls over the exact same image-derived scene and motion tensors."""
    pred_state = parts["state_hat"]
    pred_history = parts["history_hat"]
    before_state, before_history = pred_state.detach().clone(), pred_history.detach().clone()
    baseline = model.plan_from_features(parts["scene_features"], parts["motion_features"],
                                        pred_state, pred_history)
    state_gt21, history_gt21 = gt21_planner_inputs(
        pred_state, pred_history, gt_state, gt_history, state_valid, history_valid)
    intervention = model.plan_from_features(parts["scene_features"], parts["motion_features"],
                                            state_gt21, history_gt21)
    require(torch.equal(pred_state, before_state) and torch.equal(pred_history, before_history),
            "Planner intervention mutated baseline predictions")
    return baseline, intervention, state_gt21, history_gt21


def validate_batch_join(raw, labels, predictions, offset):
    n = len(raw["row"])
    rows = torch.as_tensor(raw["row"], dtype=torch.int64).cpu()
    sl = slice(offset, offset + n)
    require(torch.equal(rows, labels["rows"][sl]) and torch.equal(rows, predictions["rows"][sl]),
            "Dataset/GT/prediction row join mismatch")
    require(torch.equal(raw["state_target"].float().cpu(), labels["state_target"][sl])
            and torch.equal(raw["history_target"].float().cpu(), labels["history_target"][sl])
            and torch.equal(raw["gt_plan"].float().cpu(), labels["gt_plan"][sl])
            and torch.equal(raw["state_valid"].bool().cpu(), labels["state_valid"][sl])
            and torch.equal(raw["history_valid"].bool().cpu(), labels["history_valid"][sl])
            and torch.equal(torch.as_tensor(raw["frame"], dtype=torch.int64).cpu(),
                            labels["frame"][sl])
            and list(raw["scenario"]) == labels["scenario"][sl]
            and list(raw["session_id"]) == labels["session"][sl],
            "Dataset labels differ from pinned GT artifact")
    return sl


def paired_session_summary(baseline_d3, gt21_d3, sessions):
    baseline = np.asarray(baseline_d3, dtype=np.float64)
    intervention = np.asarray(gt21_d3, dtype=np.float64)
    sessions = np.asarray(sessions)
    require(baseline.shape == intervention.shape == sessions.shape and baseline.ndim == 1,
            "Paired session arrays must be aligned vectors")
    unique = sorted(set(sessions.tolist()))
    require(len(unique) == 11, "Canonical tune diagnostic requires exactly 11 sessions")
    delta = intervention - baseline
    per_session = {}
    for session in unique:
        mask = sessions == session
        per_session[session] = {
            "n": int(mask.sum()), "baseline_d3_sum": float(baseline[mask].sum()),
            "gt21_d3_sum": float(intervention[mask].sum()),
            "gt21_minus_baseline_d3_sum": float(delta[mask].sum()),
            "baseline_d3_mean": float(baseline[mask].mean()),
            "gt21_d3_mean": float(intervention[mask].mean()),
            "gt21_minus_baseline_d3_mean": float(delta[mask].mean()),
        }
    return {"per_session": per_session, "sessions": 11,
            "aggregate_is_frame_weighted": True,
            "joint_two_seed_bootstrap_deferred_until_both_saved_outputs": True}


def run(args):
    require(probe.sha256(probe.__file__) == PROBE_SOURCE_SHA256,
            "Frozen P7-C extraction/validation helper differs")
    require(torch.cuda.is_available() and args.device == "cuda:0",
            "GT21 diagnostic requires one isolated CUDA device")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", None)
            and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
            and torch.cuda.device_count() == 1 and args.batch == 4 and args.workers == 4,
            "GT21 diagnostic requires one isolated GPU, batch4, workers4")
    manifest, runtime_source = probe.validate_p7_inputs(args)
    dataset = probe.validate_common_inputs(args, "tune")
    labels = load_pinned_tune_artifact(args.gt_tune, "ground_truth")
    predictions = load_pinned_tune_artifact(args.pred_tune, "prediction", args.base_seed)
    require(torch.equal(labels["rows"], predictions["rows"]), "Pinned GT/pred rows differ")
    reference = probe.validate_tune_reference(args.tune_report, args.base_seed)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    model = probe.load_p7_control_model(args.checkpoint, manifest, device)
    state_before = tensor_state_sha256(model.state_dict())
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                        pin_memory=True, drop_last=False)
    collected = {key: [] for key in ("rows", "frames", "baseline", "gt21", "gt",
                                      "baseline_d3", "gt21_d3", "raw_stop_logit")}
    scenarios, sessions = [], []
    offset = 0
    for raw in loader:
        sl = validate_batch_join(raw, labels, predictions, offset)
        batch = to_device(raw, device)
        with torch.inference_mode(), training_autocast(device, "bf16"):
            inputs = model_inputs(batch, time_input="nominal",
                                  nominal_history_seconds=probe.NOMINAL_SECONDS)
            parts = model.forward_parts(**inputs)
            baseline, gt21, injected_state, _ = paired_planner_forward(
                model, parts, batch["state_target"], batch["history_target"],
                batch.get("state_valid"), batch.get("history_valid"))
        baseline_cpu, gt21_cpu = baseline.float().cpu(), gt21.float().cpu()
        require(torch.equal(parts["state_hat"].float().cpu(), predictions["pred_state"][sl])
                and torch.equal(parts["history_hat"].float().cpu(), predictions["pred_history"][sl])
                and torch.equal(baseline_cpu, predictions["pred_plan"][sl]),
                "Same-forward baseline differs from pinned P7-C prediction artifact")
        batch_rows = [int(value) for value in raw["row"]]
        for i, row in enumerate(batch_rows):
            item = reference[offset + i]
            require(item["row"] == row and item["scenario"] == raw["scenario"][i]
                    and item["session"] == raw["session_id"][i]
                    and item["frame"] == int(raw["frame"][i])
                    and torch.equal(baseline_cpu[i], torch.tensor(item["pred_abs_xy"], dtype=torch.float32)),
                    "Baseline identity/plan differs from immutable terminal report")
        gt_cpu = batch["gt_plan"].float().cpu()
        collected["rows"].append(torch.as_tensor(batch_rows, dtype=torch.int64))
        collected["frames"].append(torch.as_tensor(raw["frame"], dtype=torch.int64))
        collected["baseline"].append(baseline_cpu); collected["gt21"].append(gt21_cpu)
        collected["gt"].append(gt_cpu)
        collected["baseline_d3"].append(weighted_d3(baseline, batch["gt_plan"]).float().cpu())
        collected["gt21_d3"].append(weighted_d3(gt21, batch["gt_plan"]).float().cpu())
        collected["raw_stop_logit"].append(injected_state[:, 5].float().cpu())
        scenarios.extend(raw["scenario"]); sessions.extend(raw["session_id"])
        offset += len(batch_rows)
    require(offset == probe.ROWS["tune"][0], "Incomplete canonical tune pass")
    payload = {key: torch.cat(value) for key, value in collected.items()}
    payload["scenario"], payload["session"] = scenarios, sessions
    require(torch.equal(payload["rows"], labels["rows"]), "Output row inventory mismatch")
    require(tensor_state_sha256(model.state_dict()) == state_before,
            "Model state changed during privileged diagnostic")
    paired_analysis = paired_session_summary(payload["baseline_d3"].numpy(),
                                              payload["gt21_d3"].numpy(), sessions)
    executed_source = {"scripts/diagnose_motiondrive_v2_p7_gt21_planner.py":
                       probe.sha256(__file__),
                       "scripts/run_motiondrive_v2_p7_state_sufficiency.py": PROBE_SOURCE_SHA256}
    executed_source.update(probe.VALIDATION_FILES)
    executed_source.update({name: runtime_source["file_sha256"][name]
                            for name in sorted(probe.RUNTIME_FILES)})
    out = Path(args.out).resolve()
    probe.atomic_torch(out, payload)
    report = {
        "schema_version": 1, "status": "completed", "diagnostic_only": True,
        "privileged_gt_not_candidate_output": True, "base_seed": args.base_seed,
        "rows": offset, "rows_sha256": probe.rows_sha256(payload["rows"].numpy()),
        "baseline_official_d3": float(np.asarray(payload["baseline_d3"], dtype=np.float64).mean()),
        "gt21_official_d3": float(np.asarray(payload["gt21_d3"], dtype=np.float64).mean()),
        "gt21_minus_baseline_d3": float(np.asarray(
            payload["gt21_d3"] - payload["baseline_d3"], dtype=np.float64).mean()),
        "paired_11_session_analysis": paired_analysis,
        "artifact": str(out), "artifact_sha256": probe.sha256(out),
        "checkpoint_sha256": probe.P7_CONTROL[args.base_seed]["checkpoint_sha256"],
        "run_manifest_sha256": probe.P7_CONTROL[args.base_seed]["manifest_sha256"],
        "terminal_reference_sha256": probe.P7_CONTROL[args.base_seed]["final_eval_sha256"],
        "gt_tune_sha256": GT_TUNE, "pred_tune_sha256": PRED_TUNE[args.base_seed],
        "script_sha256": probe.sha256(__file__),
        "probe_source_sha256": PROBE_SOURCE_SHA256,
        "runtime_source_manifest_sha256": probe.RUNTIME_SOURCE_MANIFEST_SHA256,
        "runtime_source": runtime_source,
        "validation_source_sha256": probe.VALIDATION_FILES,
        "executed_source_sha256": executed_source,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": tensor_state_sha256(model.state_dict()),
        "contract": {"full_image_forward_parts_per_row": 1, "planner_calls_per_row": 2,
                     "baseline_status": "own predicted state6 and history4x4",
                     "intervention": "GT physical state[0:5] and GT history4x4",
                     "predicted_raw_stop_logit_preserved": True,
                     "scene_motion_goal_path_identical": True,
                     "raw_goal_planner_argument": False, "batch": 4, "workers": 4,
                     "precision": "bf16_encoder_fp32_heads"},
        "boundaries": {"no_optimizer_or_backward": True, "weights_unchanged": True,
                       "no_final_validation_access": True,
                       "improvement_supports_planner_ability_to_use_precise_status": True,
                       "flat_or_worse_is_inconclusive_due_to_joint_input_distribution_shift": True,
                       "not_state_token_dilution_proof": True,
                       "not_an_information_theoretic_bound": True},
    }
    report_path = out.with_suffix(out.suffix + ".report.json")
    probe.atomic_json(report_path, report)
    print(json.dumps({"status": "completed", "artifact": str(out),
                      "report": str(report_path), "report_sha256": probe.sha256(report_path)},
                     sort_keys=True))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--base-seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--p7-source-manifest", required=True)
    parser.add_argument("--runtime-source-manifest", required=True)
    parser.add_argument("--tune-report", required=True)
    parser.add_argument("--gt-tune", required=True)
    parser.add_argument("--pred-tune", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    run(arguments(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
