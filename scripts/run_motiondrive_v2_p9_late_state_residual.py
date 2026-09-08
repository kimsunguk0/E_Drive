#!/usr/bin/env python3
"""Extract frozen P7-C decoder features and fit the fixed P9 late residual."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.motiondrive_v2_late_state_residual import LateStateResidual, compact22
from scripts.analyze_motiondrive_v2_query_pair import paired_cluster_bootstrap
from scripts import run_motiondrive_v2_p7_state_sufficiency as probe
from motiondrive_v2_training import TIME_WEIGHTS, model_inputs, tensor_state_sha256, to_device, weighted_d3
from train_motiondrive_v2 import autocast as training_autocast


PROBE_SHA256 = "1ddd1fcf2c47e04283d0ee2e789fd255c4d5c7d537c09bc2f944d53884800852"
MODULE_SHA256 = "8bdf03750e2db93c10a8ac68f7f5128f4c0bd6d8ac9e242fc4bc7f877305689e"
BOOTSTRAP_SOURCE_SHA256 = "03e714e845f7c66a152b5e28449444e61b96d82541550747003fa58e71065651"
TRAINING_HELPER_SHA256 = "6e3d5ecccee0a87cf6a85cd490ee192990fd2ca92f0c307040a93ca2cdc7077c"
ARTIFACTS = {
    "gt_train": ("96eb0d2c9fa895a79e62f85f5c4a626185930790ec778ee6eeef0ce055cb46ec",
                 "0e16b881e77f16a1548d9eb86fe5077a20bba9ad447627870c297f598c075076"),
    "gt_tune": ("8548f5df313eac452cc1ba34c894d5f68bcc1d7337d6122fe7012f6a7d485ea2",
                "53f0cf2e532c20dfe1e957204ba409a9095a6cd59aff6494e4d8c5d5dfd5922e"),
    "pred0_train": ("5b06a432912324312bae97b4a14f796585a86b9532e2c57edc18694948e4699c",
                    "7f2e3dd98a80ea8c6f172e4d7824fad29f4a63ae2c8f01683d4afdf66b218bdc"),
    "pred0_tune": ("dc7e547345938a00407f0a739cb188849cf31f372499ad9ec029b39cc74bdad7",
                   "c00fa96a393a6edaf924ea22f1910f07250dbc49a280b0e213aca4ab781b289d"),
    "pred1_train": ("c5e6efe92e593abb4bc2781fb610660a76853a35ac35139bf35355486b62ac9c",
                    "4244f015229c868bcc25dd604e4f2d289ab9a00b580f2f2cf4615c36a26de920"),
    "pred1_tune": ("8b5fb837ceb1c75c36d388b891e76f17df574e072103ae92f7158b2ff2d92e6f",
                   "ad58f21b979a84c75ea4150edf8d5437ad8d9553e52388c90caf645f8fa2214f"),
}
ARMS = ("no_compact", "predicted22", "gt21")
BATCH_SIZE = 1024
EPOCHS = 60
STEPS = EPOCHS * math.ceil(probe.ROWS["train"][0] / BATCH_SIZE)
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 20_260_908


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_p9_sources():
    expected = {
        ROOT / "models/motiondrive_v2_late_state_residual.py": MODULE_SHA256,
        ROOT / "scripts/analyze_motiondrive_v2_query_pair.py": BOOTSTRAP_SOURCE_SHA256,
        ROOT / "scripts/motiondrive_v2_training.py": TRAINING_HELPER_SHA256,
        Path(probe.__file__).resolve(): PROBE_SHA256,
    }
    require(all(probe.sha256(path) == digest for path, digest in expected.items()),
            "P9 executed source closure differs")
    return {str(path.relative_to(ROOT)): digest for path, digest in expected.items()}


def pinned_artifact(path, name, kind, split, seed=None):
    path = Path(path).resolve()
    artifact_sha, manifest_sha = ARTIFACTS[name]
    require(probe.sha256(path) == artifact_sha, f"{name} artifact SHA mismatch")
    require(probe.sha256(path.with_suffix(path.suffix + ".manifest.json")) == manifest_sha,
            f"{name} manifest SHA mismatch")
    payload, manifest, _ = probe.load_artifact(path, kind, split)
    if seed is not None:
        require(manifest["base_seed"] == seed, f"{name} seed mismatch")
    return payload


def capture_decoder_features(model, parts):
    """Capture the actual tensor passed to the existing xy_head, then reconstruct."""
    captured = []

    def hook(_module, inputs):
        require(len(inputs) == 1 and inputs[0].shape[1:] == (6, 128),
                "Unexpected existing planner-head input")
        captured.append(inputs[0].detach())

    handle = model.planner.xy_head.register_forward_pre_hook(hook)
    try:
        baseline = model.plan_from_features(parts["scene_features"], parts["motion_features"],
                                            parts["state_hat"], parts["history_hat"])
    finally:
        handle.remove()
    require(len(captured) == 1, "Existing output head must execute exactly once")
    decoded = captured[0]
    with torch.autocast(device_type=decoded.device.type, enabled=False):
        reconstructed = (model.planner.xy_head(decoded.float()) * decoded.new_tensor(
            model.config.plan_output_scale))
    require(torch.equal(reconstructed, baseline), "Existing head did not reconstruct baseline plan")
    return decoded, baseline


def validate_batch(raw, labels, predictions, offset):
    n = len(raw["row"]); sl = slice(offset, offset + n)
    rows = torch.as_tensor(raw["row"], dtype=torch.int64).cpu()
    require(torch.equal(rows, labels["rows"][sl]) and torch.equal(rows, predictions["rows"][sl]),
            "P9 row join mismatch")
    checks = (
        torch.equal(raw["state_target"].float().cpu(), labels["state_target"][sl]),
        torch.equal(raw["history_target"].float().cpu(), labels["history_target"][sl]),
        torch.equal(raw["gt_plan"].float().cpu(), labels["gt_plan"][sl]),
        torch.equal(raw["state_valid"].bool().cpu(), labels["state_valid"][sl]),
        torch.equal(raw["history_valid"].bool().cpu(), labels["history_valid"][sl]),
        torch.equal(torch.as_tensor(raw["frame"], dtype=torch.int64).cpu(), labels["frame"][sl]),
        list(raw["scenario"]) == labels["scenario"][sl],
        list(raw["session_id"]) == labels["session"][sl],
    )
    require(all(checks), "P9 dataset differs from frozen GT artifact")
    return sl


def extract_cache(args):
    source_closure = validate_p9_sources()
    require(torch.cuda.is_available() and args.device == "cuda:0"
            and os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", None)
            and "," not in os.environ["CUDA_VISIBLE_DEVICES"] and torch.cuda.device_count() == 1,
            "P9 extraction requires one isolated visible CUDA device")
    require(args.batch == (8 if args.split == "train" else 4) and args.workers == 4,
            "P9 fixed extraction batch/workers mismatch")
    manifest, runtime = probe.validate_p7_inputs(args)
    dataset = probe.validate_common_inputs(args, args.split)
    labels = pinned_artifact(args.gt, f"gt_{args.split}", "ground_truth", args.split)
    predictions = pinned_artifact(args.pred, f"pred{args.base_seed}_{args.split}",
                                  "prediction", args.split, args.base_seed)
    reference = (probe.validate_tune_reference(args.tune_report, args.base_seed)
                 if args.split == "tune" else None)
    require((args.tune_report is None) == (args.split == "train"),
            "Only tune extraction accepts a terminal reference")
    random.seed(args.base_seed); np.random.seed(args.base_seed); torch.manual_seed(args.base_seed)
    torch.cuda.manual_seed_all(args.base_seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device); torch.cuda.set_device(device)
    model = probe.load_p7_control_model(args.checkpoint, manifest, device)
    before = tensor_state_sha256(model.state_dict())
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=4,
                        pin_memory=True, drop_last=False)
    values = {key: [] for key in ("rows", "frame", "decoder_features", "baseline_plan",
                                   "pred_state", "pred_history")}
    scenarios, sessions, offset = [], [], 0
    for raw in loader:
        sl = validate_batch(raw, labels, predictions, offset)
        batch = to_device(raw, device)
        with torch.inference_mode(), training_autocast(device, "bf16"):
            parts = model.forward_parts(**model_inputs(
                batch, time_input="nominal", nominal_history_seconds=probe.NOMINAL_SECONDS))
            decoded, baseline = capture_decoder_features(model, parts)
        cpu = {"decoder_features": decoded.float().cpu(), "baseline_plan": baseline.float().cpu(),
               "pred_state": parts["state_hat"].float().cpu(),
               "pred_history": parts["history_hat"].float().cpu()}
        prediction_keys = {"baseline_plan": "pred_plan", "pred_state": "pred_state",
                           "pred_history": "pred_history"}
        require(all(torch.equal(cpu[key], predictions[prediction_keys[key]][sl])
                    for key in prediction_keys),
                "P9 cache differs from frozen prediction artifact")
        rows = torch.as_tensor(raw["row"], dtype=torch.int64)
        if reference is not None:
            for i, row in enumerate(rows.tolist()):
                item = reference[offset + i]
                require(item["row"] == row and item["scenario"] == raw["scenario"][i]
                        and item["session"] == raw["session_id"][i]
                        and item["frame"] == int(raw["frame"][i])
                        and torch.equal(cpu["baseline_plan"][i],
                                        torch.tensor(item["pred_abs_xy"], dtype=torch.float32)),
                        "P9 baseline differs from terminal P7-C reference")
        values["rows"].append(rows); values["frame"].append(torch.as_tensor(raw["frame"], dtype=torch.int64))
        for key in cpu: values[key].append(cpu[key])
        scenarios.extend(raw["scenario"]); sessions.extend(raw["session_id"])
        offset += len(rows)
    require(offset == probe.ROWS[args.split][0] and tensor_state_sha256(model.state_dict()) == before,
            "P9 extraction incomplete or model state changed")
    payload = {key: torch.cat(items) for key, items in values.items()}
    payload["scenario"], payload["session"] = scenarios, sessions
    validate_cache_payload(payload, args.split)
    out = Path(args.out).resolve(); probe.atomic_torch(out, payload)
    sidecar = {
        "schema_version": 1, "status": "completed", "kind": "p9_decoder_cache",
        "split": args.split, "base_seed": args.base_seed, "rows": offset,
        "rows_sha256": probe.rows_sha256(payload["rows"].numpy()),
        "artifact_sha256": probe.sha256(out), "script_sha256": probe.sha256(__file__),
        "probe_sha256": PROBE_SHA256, "executed_source_closure": source_closure,
        "checkpoint_sha256": probe.P7_CONTROL[args.base_seed]["checkpoint_sha256"],
        "run_manifest_sha256": probe.P7_CONTROL[args.base_seed]["manifest_sha256"],
        "source_manifest_sha256": probe.RUNTIME_SOURCE_MANIFEST_SHA256,
        "runtime_source": runtime, "model_state_before": before, "model_state_after": before,
        "frozen_body": True, "forward_parts_per_row": 1, "existing_planner_calls_per_row": 1,
        "baseline_head_reconstruction_exact": True, "augment": False,
        "precision": "bf16_encoder_fp32_planner", "batch": args.batch, "workers": 4,
        "final_validation_accessed": False, "optimizer_created": False,
    }
    probe.atomic_json(out.with_suffix(out.suffix + ".manifest.json"), sidecar)
    print(json.dumps({"status": "completed", "artifact": str(out),
                      "sha256": sidecar["artifact_sha256"]}, sort_keys=True))


def validate_cache_payload(payload, split):
    n = probe.ROWS[split][0]
    required = {"rows", "frame", "decoder_features", "baseline_plan", "pred_state",
                "pred_history", "scenario", "session"}
    require(set(payload) == required, "P9 cache keys mismatch")
    shapes = {"rows": (n,), "frame": (n,), "decoder_features": (n, 6, 128),
              "baseline_plan": (n, 6, 2), "pred_state": (n, 6),
              "pred_history": (n, 4, 4)}
    require(all(isinstance(payload[k], torch.Tensor) and tuple(payload[k].shape) == shape
                for k, shape in shapes.items()), "P9 cache shape mismatch")
    require(payload["rows"].dtype == payload["frame"].dtype == torch.int64
            and all(payload[k].dtype == torch.float32 and torch.isfinite(payload[k]).all()
                    for k in ("decoder_features", "baseline_plan", "pred_state", "pred_history")),
            "P9 cache dtype/finite mismatch")
    require(probe.rows_sha256(payload["rows"].numpy()) == probe.ROWS[split][2]
            and len(payload["scenario"]) == len(payload["session"]) == n,
            "P9 cache identity mismatch")
    return True


def load_cache(path, split, seed):
    path = Path(path).resolve(); side = path.with_suffix(path.suffix + ".manifest.json")
    manifest = probe.read_json(side)
    required = {"schema_version", "status", "kind", "split", "base_seed", "rows",
                "rows_sha256", "artifact_sha256", "script_sha256", "probe_sha256",
                "executed_source_closure",
                "checkpoint_sha256", "run_manifest_sha256", "source_manifest_sha256",
                "runtime_source", "model_state_before", "model_state_after", "frozen_body",
                "forward_parts_per_row", "existing_planner_calls_per_row",
                "baseline_head_reconstruction_exact", "augment", "precision", "batch",
                "workers", "final_validation_accessed", "optimizer_created"}
    require(set(manifest) == required and manifest["schema_version"] == 1
            and manifest["status"] == "completed" and manifest["kind"] == "p9_decoder_cache"
            and manifest["split"] == split and manifest["base_seed"] == seed
            and manifest["rows"] == probe.ROWS[split][0]
            and manifest["rows_sha256"] == probe.ROWS[split][2]
            and manifest["artifact_sha256"] == probe.sha256(path)
            and manifest["script_sha256"] == probe.sha256(__file__)
            and manifest["probe_sha256"] == PROBE_SHA256
            and manifest["executed_source_closure"] == validate_p9_sources()
            and manifest["checkpoint_sha256"] == probe.P7_CONTROL[seed]["checkpoint_sha256"]
            and manifest["run_manifest_sha256"] == probe.P7_CONTROL[seed]["manifest_sha256"]
            and manifest["source_manifest_sha256"] == probe.RUNTIME_SOURCE_MANIFEST_SHA256
            and manifest["model_state_before"] == manifest["model_state_after"]
            and manifest["frozen_body"] is True and manifest["forward_parts_per_row"] == 1
            and manifest["existing_planner_calls_per_row"] == 1
            and manifest["baseline_head_reconstruction_exact"] is True
            and manifest["augment"] is False and manifest["final_validation_accessed"] is False
            and manifest["optimizer_created"] is False,
            "P9 cache manifest provenance mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_cache_payload(payload, split)
    return payload, {"artifact": str(path), "artifact_sha256": manifest["artifact_sha256"],
                     "manifest": str(side), "manifest_sha256": probe.sha256(side)}


def model_state_sha(model):
    return tensor_state_sha256(model.state_dict())


def evaluate_once(model, features, compact, baseline, target):
    plans, row_d3, calls = [], [], 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), BATCH_SIZE):
            sl = slice(start, start + BATCH_SIZE)
            corrected = baseline[sl].float() + model(features[sl], compact[sl])
            plans.append(corrected.float()); row_d3.append(weighted_d3(corrected, target[sl]).float())
            calls += 1
    plan, d3 = torch.cat(plans), torch.cat(row_d3)
    return plan, d3, calls


def train_residual(features, compact, baseline, target, tune, *, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = LateStateResidual()
    initial = model_state_sha(model)
    with torch.no_grad():
        require(torch.equal(model(features[:2], compact[:2]), torch.zeros(2, 6, 2)),
                "P9 residual must start at exact zero")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    require(not optimizer.state, "P9 optimizer must start fresh")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
    generator = torch.Generator().manual_seed(seed)
    order_sha = hashlib.sha256(); steps = 0
    model.train()
    for _epoch in range(EPOCHS):
        order = torch.randperm(len(features), generator=generator)
        order_sha.update(order.numpy().astype("<i8", copy=False).tobytes())
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start:start + BATCH_SIZE]
            corrected = baseline[index].float() + model(features[index], compact[index])
            loss = weighted_d3(corrected, target[index]).mean()
            require(torch.isfinite(loss), "P9 training loss nonfinite")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            require(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                        for parameter in model.parameters()), "P9 gradient missing/nonfinite")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); scheduler.step(); steps += 1
            require(all(torch.isfinite(parameter).all() for parameter in model.parameters()),
                    "P9 parameter became nonfinite")
    require(steps == STEPS, "P9 optimizer step mismatch")
    tune_plan, tune_d3, calls = evaluate_once(model, *tune)
    require(calls == math.ceil(probe.ROWS["tune"][0] / BATCH_SIZE),
            "P9 tune must use one terminal pass")
    return model, {"initial_model_state_sha256": initial, "sample_order_sha256": order_sha.hexdigest(),
                   "steps": steps, "tune_plan": tune_plan, "tune_d3": tune_d3,
                   "tune_official_d3": float(np.asarray(tune_d3, dtype=np.float64).mean()),
                   "terminal_tune_forward_calls": calls}


def compare_rows(left_by_seed, right_by_seed, sessions, threshold=None):
    require(set(left_by_seed) == set(right_by_seed) == {0, 1}, "P9 comparison requires both seeds")
    sessions = np.asarray(sessions)
    require(len(set(sessions.tolist())) == 11, "P9 comparison requires 11 tune sessions")
    deltas = {seed: np.asarray(left_by_seed[seed], np.float64)
              - np.asarray(right_by_seed[seed], np.float64) for seed in (0, 1)}
    require(all(value.shape == sessions.shape and np.isfinite(value).all() for value in deltas.values()),
            "P9 paired comparison shape/nonfinite mismatch")
    shared = .5 * (deltas[0] + deltas[1])
    bootstrap = paired_cluster_bootstrap(shared, sessions, repeats=BOOTSTRAP_REPEATS,
                                         seed=BOOTSTRAP_SEED)
    per_seed = {f"base{seed}": float(deltas[seed].mean()) for seed in (0, 1)}
    result = {"per_seed_delta_d3": per_seed, "mean_delta_d3": float(shared.mean()),
            "shared_11_session_bootstrap": bootstrap,
            "gate": None}
    if threshold is not None:
        result["gate"] = {"both_seeds_negative": all(value < 0 for value in per_seed.values()),
                          "mean_at_most": threshold,
                          "mean_pass": float(shared.mean()) <= threshold,
                          "ci_upper_below_zero": bootstrap["ci95"][1] < 0}
    return result


def fit(args):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized()
            and torch.get_num_threads() == 1, "P9 fit requires CPU-only threads1")
    source_closure = validate_p9_sources()
    gt_train = pinned_artifact(args.gt_train, "gt_train", "ground_truth", "train")
    gt_tune = pinned_artifact(args.gt_tune, "gt_tune", "ground_truth", "tune")
    out = Path(args.output_dir).resolve(); require(not out.exists(), "P9 output dir must be fresh")
    out.mkdir(parents=True)
    runs, inputs, row_d3 = {}, {}, {}
    for seed in (0, 1):
        train_cache, tr = load_cache(getattr(args, f"cache{seed}_train"), "train", seed)
        tune_cache, tu = load_cache(getattr(args, f"cache{seed}_tune"), "tune", seed)
        require(torch.equal(train_cache["rows"], gt_train["rows"])
                and torch.equal(tune_cache["rows"], gt_tune["rows"]), "P9 cache/GT row mismatch")
        inputs[f"base{seed}_train"], inputs[f"base{seed}_tune"] = tr, tu
        pred_train = compact22(train_cache["pred_state"], train_cache["pred_history"],
                               arm="predicted22")
        pred_tune = compact22(tune_cache["pred_state"], tune_cache["pred_history"],
                              arm="predicted22")
        arm_inputs = {
            "no_compact": (torch.zeros_like(pred_train), torch.zeros_like(pred_tune)),
            "predicted22": (pred_train, pred_tune),
            "gt21": (compact22(train_cache["pred_state"], train_cache["pred_history"], arm="gt21",
                               gt_state=gt_train["state_target"], gt_history=gt_train["history_target"]),
                     compact22(tune_cache["pred_state"], tune_cache["pred_history"], arm="gt21",
                               gt_state=gt_tune["state_target"], gt_history=gt_tune["history_target"])),
        }
        seed_runs, row_d3[seed] = {}, {}
        parent_d3 = weighted_d3(tune_cache["baseline_plan"], gt_tune["gt_plan"]).float()
        row_d3[seed]["parent"] = parent_d3.numpy()
        for arm in ARMS:
            model, result = train_residual(
                train_cache["decoder_features"], arm_inputs[arm][0], train_cache["baseline_plan"],
                gt_train["gt_plan"],
                (tune_cache["decoder_features"], arm_inputs[arm][1],
                 tune_cache["baseline_plan"], gt_tune["gt_plan"]), seed=seed)
            checkpoint = out / f"base{seed}_{arm}_last60.pt"
            probe.atomic_torch(checkpoint, {"model": model.state_dict(), "seed": seed, "arm": arm,
                                            "epoch": EPOCHS, "steps": STEPS})
            corrected_d3 = result.pop("tune_d3")
            row_d3[seed][arm] = corrected_d3.numpy()
            rows = out / f"base{seed}_{arm}_tune_rows.pt"
            probe.atomic_torch(rows, {"rows": gt_tune["rows"], "scenario": gt_tune["scenario"],
                                      "session": gt_tune["session"], "frame": gt_tune["frame"],
                                      "baseline_plan": tune_cache["baseline_plan"],
                                      "corrected_plan": result.pop("tune_plan"),
                                      "gt_plan": gt_tune["gt_plan"], "baseline_d3": parent_d3,
                                      "corrected_d3": corrected_d3})
            result.update(checkpoint=str(checkpoint), checkpoint_sha256=probe.sha256(checkpoint),
                          tune_rows=str(rows), tune_rows_sha256=probe.sha256(rows))
            seed_runs[arm] = result
        require(len({seed_runs[a]["initial_model_state_sha256"] for a in ARMS}) == 1
                and len({seed_runs[a]["sample_order_sha256"] for a in ARMS}) == 1,
                "P9 A/B/C initialization or sample order differs")
        runs[f"base{seed}"] = seed_runs
    sessions = np.asarray(gt_tune["session"])
    a = {seed: row_d3[seed]["no_compact"] for seed in (0, 1)}
    b = {seed: row_d3[seed]["predicted22"] for seed in (0, 1)}
    c = {seed: row_d3[seed]["gt21"] for seed in (0, 1)}
    parent = {seed: row_d3[seed]["parent"] for seed in (0, 1)}
    comparisons = {
        "primary_B_minus_A": compare_rows(b, a, sessions, -.010),
        "secondary_A_minus_parent": compare_rows(a, parent, sessions, -.015),
        "secondary_B_minus_parent": compare_rows(b, parent, sessions, -.015),
        "privileged_C_minus_B_no_gate": compare_rows(c, b, sessions),
    }
    primary = comparisons["primary_B_minus_A"]["gate"]
    versus_parent = comparisons["secondary_B_minus_parent"]["gate"]
    adoption_gate = all((primary["both_seeds_negative"], primary["mean_pass"],
                         primary["ci_upper_below_zero"], versus_parent["both_seeds_negative"],
                         versus_parent["mean_pass"], versus_parent["ci_upper_below_zero"]))
    report = {
        "schema_version": 1, "status": "completed", "diagnostic_only": True,
        "frozen_p7_c_body": True, "arms": list(ARMS), "runs": runs, "inputs": inputs,
        "comparisons": comparisons, "preregistered_B_adoption_gate_pass": adoption_gate,
        "recipe": {"batch": BATCH_SIZE, "epochs": EPOCHS, "steps": STEPS,
                   "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
                   "scheduler": "per-step cosine", "gradient_clip": 5.,
                   "loss": "official D3", "selection": "LAST60 only"},
        "metric": {"plan_valid_required_all_six": True,
                   "time_weights": list(TIME_WEIGHTS),
                   "terminal_rows_aggregated_float64": True,
                   "bootstrap_repeats": BOOTSTRAP_REPEATS, "bootstrap_seed": BOOTSTRAP_SEED,
                   "bootstrap_unit": "shared same eleven sessions; frame-weighted within draw"},
        "script_sha256": probe.sha256(__file__), "probe_sha256": PROBE_SHA256,
        "executed_source_closure": source_closure,
        "boundaries": {"no_raw_goal": True, "no_final_validation": True,
                       "gt21_privileged_non_deployable": True, "no_body_gradient": True,
                       "single_terminal_tune_pass": True, "per_row_outputs_saved": True},
    }
    for receipt in inputs.values():
        require(probe.sha256(receipt["artifact"]) == receipt["artifact_sha256"]
                and probe.sha256(receipt["manifest"]) == receipt["manifest_sha256"],
                "P9 cache changed during fit")
    probe.atomic_json(out / "result.json", report)
    print(json.dumps({"status": "completed", "report": str(out / "result.json"),
                      "sha256": probe.sha256(out / "result.json")}, sort_keys=True))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    for name in ("data-root", "split-manifest", "supervision-root", "checkpoint",
                 "run-manifest", "p7-source-manifest", "runtime-source-manifest",
                 "gt", "pred", "out"):
        extract.add_argument("--" + name, required=True)
    extract.add_argument("--split", choices=("train", "tune"), required=True)
    extract.add_argument("--base-seed", type=int, choices=(0, 1), required=True)
    extract.add_argument("--tune-report")
    extract.add_argument("--batch", type=int, required=True); extract.add_argument("--workers", type=int, default=4)
    extract.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--gt-train", required=True); fit_parser.add_argument("--gt-tune", required=True)
    for seed in (0, 1):
        fit_parser.add_argument(f"--cache{seed}-train", dest=f"cache{seed}_train", required=True)
        fit_parser.add_argument(f"--cache{seed}-tune", dest=f"cache{seed}_tune", required=True)
    fit_parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    extract_cache(args) if args.command == "extract" else fit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
