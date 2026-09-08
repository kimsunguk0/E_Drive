#!/usr/bin/env python3
"""Fit the fixed privileged GT7/GT24/time-contract diagnostic on CPU.

All three arms retain the exact 24D MLP shape.  The two GT7 arms mask
normalized stop and all history coordinates to zero, leaving physical state
(first five fields) and the same raw 5-second goal.  The secondary GT7 arm
replaces only state5 with the frozen A1 nominal-time values before the shared
normalization.  This is diagnostic-only and non-submittable.
"""
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
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_motiondrive_v2_p7_state_sufficiency as base
from motiondrive_v2_shared_status_data import load_status_overlay


BASE_SCRIPT_SHA256 = "1ddd1fcf2c47e04283d0ee2e789fd255c4d5c7d537c09bc2f944d53884800852"
TRAINING_HELPER_SHA256 = "6e3d5ecccee0a87cf6a85cd490ee192990fd2ca92f0c307040a93ca2cdc7077c"
STATUS_HELPER_SHA256 = "fb7dcf68349e5b27b1578bd6525244972ff89e3a22a8a04c169de505ea24095f"
SPLIT_HELPER_SHA256 = "5d3019cf730fe56a27ab7f204ee3f15bc8a4ffaef5df9fbf0635f086c84a49c1"
STATUS_OVERLAY_SHA256 = "8852f1e7ce80895b09ac400e8332d99d9feb6918a8a18e0893738300a3021699"
STATUS_ARTIFACT_SHA256 = {
    "train": "f76ac0ef4cf3f8626869ec63db6376ccfa2e0d135cf161c076d67854578b05f1",
    "tune": "aff4c00cd36c35c03c1a43f9f0689c85bc75008be7e8444829ea9a10b9493c6f",
}
GT_ARTIFACTS = {
    "train": ("96eb0d2c9fa895a79e62f85f5c4a626185930790ec778ee6eeef0ce055cb46ec",
              "0e16b881e77f16a1548d9eb86fe5077a20bba9ad447627870c297f598c075076"),
    "tune": ("8548f5df313eac452cc1ba34c894d5f68bcc1d7337d6122fe7012f6a7d485ea2",
             "53f0cf2e532c20dfe1e957204ba409a9095a6cd59aff6494e4d8c5d5dfd5922e"),
}
FULL24_REFERENCE = {0: 0.0887658248724951, 1: 0.08918161638506648}
MASKED_INDICES = tuple(range(5, 22))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path) -> str:
    return base.sha256(path)


def validate_sources(expected_script_sha256):
    actual = {
        "scripts/run_motiondrive_v2_gt7_vs_gt24.py": sha256(__file__),
        "scripts/run_motiondrive_v2_p7_state_sufficiency.py": sha256(base.__file__),
        "scripts/motiondrive_v2_training.py": sha256(ROOT / "scripts/motiondrive_v2_training.py"),
        "scripts/motiondrive_v2_shared_status_data.py": sha256(
            ROOT / "scripts/motiondrive_v2_shared_status_data.py"),
        "scripts/build_grouped_split_v2.py": sha256(ROOT / "scripts/build_grouped_split_v2.py"),
    }
    require(actual["scripts/run_motiondrive_v2_gt7_vs_gt24.py"] == expected_script_sha256,
            "GT7 wrapper self SHA mismatch")
    require(actual["scripts/run_motiondrive_v2_p7_state_sufficiency.py"] == BASE_SCRIPT_SHA256
            and actual["scripts/motiondrive_v2_training.py"] == TRAINING_HELPER_SHA256
            and actual["scripts/motiondrive_v2_shared_status_data.py"] == STATUS_HELPER_SHA256
            and actual["scripts/build_grouped_split_v2.py"] == SPLIT_HELPER_SHA256,
            "GT7 calculation source closure mismatch")
    return actual


def load_gt(path, split):
    path = Path(path).resolve()
    artifact_sha, manifest_sha = GT_ARTIFACTS[split]
    require(sha256(path) == artifact_sha
            and sha256(path.with_suffix(path.suffix + ".manifest.json")) == manifest_sha,
            f"GT7 {split} artifact identity mismatch")
    payload, manifest, receipt = base.load_artifact(path, "ground_truth", split)
    require(manifest["artifact_sha256"] == artifact_sha,
            f"GT7 {split} producer declaration mismatch")
    return payload, receipt


def mask_to_gt7(normalized24):
    require(isinstance(normalized24, torch.Tensor) and normalized24.ndim == 2
            and normalized24.shape[1] == 24 and normalized24.dtype == torch.float32
            and bool(torch.isfinite(normalized24).all()),
            "GT7 masking requires finite normalized float32 [N,24]")
    result = normalized24.clone()
    result[:, 5:22] = 0
    require(torch.equal(result[:, :5], normalized24[:, :5])
            and torch.equal(result[:, 22:], normalized24[:, 22:])
            and not bool(result[:, 5:22].count_nonzero()),
            "GT7 mask did not preserve exactly state5+goal2")
    return result


def replace_state5_with_nominal(raw24, status5):
    require(isinstance(raw24, torch.Tensor) and raw24.ndim == 2 and raw24.shape[1] == 24
            and raw24.dtype == torch.float32 and isinstance(status5, np.ndarray)
            and status5.shape == (len(raw24), 5) and status5.dtype == np.float32
            and np.isfinite(status5).all(), "A1 nominal replacement input mismatch")
    result = raw24.clone()
    result[:, :5] = torch.from_numpy(status5)
    require(torch.equal(result[:, 5:], raw24[:, 5:]),
            "A1 nominal replacement changed stop/history/goal")
    return result


def evaluate_once(model, features, target):
    predictions, row_d3, distances = [], [], []
    with torch.inference_mode():
        for start in range(0, len(features), base.BATCH_SIZE):
            prediction = model(features[start:start + base.BATCH_SIZE].float()).reshape(-1, 6, 2)
            truth = target[start:start + base.BATCH_SIZE].float()
            predictions.append(prediction)
            row_d3.append(base.weighted_d3(prediction, truth))
            distances.append(torch.linalg.vector_norm(prediction - truth, dim=-1))
    prediction = torch.cat(predictions)
    row_d3 = torch.cat(row_d3)
    distance64 = torch.cat(distances).numpy().astype(np.float64)
    d3_64 = row_d3.numpy().astype(np.float64)
    return prediction, row_d3, {
        "official_d3": float(d3_64.mean()),
        "ade1": float(distance64[:, :2].mean()),
        "ade2": float(distance64[:, :4].mean()),
        "ade3": float(distance64.mean()),
    }


def train_fixed(train_x, train_y, tune_x, tune_y, seed, arm):
    require(train_x.shape == (54810, 24) and tune_x.shape == (1998, 24),
            "GT7 fixed train/tune input inventory mismatch")
    torch.manual_seed(seed)
    model = base.tiny_model().cpu()
    initial_sha = base.model_state_sha(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=base.N_STEPS)
    generator = torch.Generator().manual_seed(seed)
    order_digest = hashlib.sha256()
    step = 0
    model.train()
    for epoch in range(base.N_EPOCHS):
        order = torch.randperm(len(train_x), generator=generator)
        order_digest.update(order.numpy().astype("<i8", copy=False).tobytes())
        last_loss = None
        for start in range(0, len(train_x), base.BATCH_SIZE):
            index = order[start:start + base.BATCH_SIZE]
            prediction = model(train_x[index]).reshape(-1, 6, 2)
            loss = F.smooth_l1_loss(prediction, train_y[index], beta=.1)
            require(bool(torch.isfinite(loss)), "GT7 training loss became nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            last_loss = float(loss.detach())
        print(json.dumps({"status": "epoch_completed", "seed": seed, "arm": arm,
                          "epoch": epoch + 1, "steps": step, "last_loss": last_loss},
                         sort_keys=True), flush=True)
    require(step == 3240 == base.N_STEPS, "GT7 fixed optimizer-update count mismatch")
    require(all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()),
            "GT7 final model parameters are nonfinite")
    model.eval()
    _, _, train_metrics = evaluate_once(model, train_x, train_y)
    tune_prediction, tune_row_d3, tune_metrics = evaluate_once(model, tune_x, tune_y)
    return model, {
        "initial_model_state_sha256": initial_sha,
        "final_model_state_sha256": base.model_state_sha(model),
        "sample_order_sha256": order_digest.hexdigest(), "steps": step,
        "train_d3_in_sample": train_metrics["official_d3"],
        "tune": tune_metrics, "tune_prediction": tune_prediction,
        "tune_row_d3": tune_row_d3,
    }


def fit(args):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "GT7 fitting is CPU-only and requires empty CUDA visibility")
    sources = validate_sources(args.expected_script_sha256)
    output = Path(args.output_dir).resolve()
    require(not output.exists(), "refusing to overwrite GT7 output directory")
    gt_train, train_receipt = load_gt(args.gt_train, "train")
    gt_tune, tune_receipt = load_gt(args.gt_tune, "tune")
    overlay_root = Path(args.status_overlay_root).resolve()
    overlay_manifest, overlay = load_status_overlay(overlay_root, STATUS_OVERLAY_SHA256)
    require(all(sha256(overlay_root / f"{split}.npz") == STATUS_ARTIFACT_SHA256[split]
                and np.array_equal(overlay[split]["row"], gt["rows"].numpy())
                and np.array_equal(overlay[split]["frame"], gt["frame"].numpy())
                for split, gt in (("train", gt_train), ("tune", gt_tune))),
            "A1 overlay artifact or GT row/frame join mismatch")
    normalizer = base.fit_shared_normalizer(base.canonical_raw24(gt_train))
    normalizer_sha = base.tensor_tree_sha256(normalizer)
    require(normalizer_sha == args.expected_normalizer_sha256,
            "GT7 shared normalizer SHA mismatch")
    raw_train, raw_tune = base.canonical_raw24(gt_train), base.canonical_raw24(gt_tune)
    full_train = base.apply_normalizer(raw_train, normalizer).float()
    full_tune = base.apply_normalizer(raw_tune, normalizer).float()
    reduced_train, reduced_tune = mask_to_gt7(full_train), mask_to_gt7(full_tune)
    nominal_train = mask_to_gt7(base.apply_normalizer(
        replace_state5_with_nominal(raw_train, overlay["train"]["status5"]), normalizer).float())
    nominal_tune = mask_to_gt7(base.apply_normalizer(
        replace_state5_with_nominal(raw_tune, overlay["tune"]["status5"]), normalizer).float())
    output.mkdir(parents=True)
    runs = {}
    seed = args.seed
    pair = {}
    for arm, train_x, tune_x in (("canonical_full24", full_train, full_tune),
                                 ("canonical_masked7", reduced_train, reduced_tune),
                                 ("a1_nominal_masked7", nominal_train, nominal_tune)):
        print(json.dumps({"status": "fit_started", "seed": seed, "arm": arm,
                          "steps": 3240}, sort_keys=True), flush=True)
        model, result = train_fixed(train_x, gt_train["gt_plan"], tune_x,
                                    gt_tune["gt_plan"], seed, arm)
        checkpoint = output / f"seed{seed}_{arm}_last60.pt"
        base.atomic_torch(checkpoint, {"model": model.state_dict(), "seed": seed,
            "arm": arm, "steps": 3240, "normalizer_sha256": normalizer_sha,
            "masked_normalized_indices": (
                list(MASKED_INDICES) if arm != "canonical_full24" else [])})
        pair[arm] = {key: value for key, value in result.items()
                     if key not in ("tune_prediction", "tune_row_d3")}
        pair[arm]["checkpoint"] = str(checkpoint)
        pair[arm]["checkpoint_sha256"] = sha256(checkpoint)
        pair[arm]["tune_prediction"] = result["tune_prediction"]
        pair[arm]["tune_row_d3"] = result["tune_row_d3"]
        print(json.dumps({"status": "fit_completed", "seed": seed, "arm": arm,
                          "steps": result["steps"],
                          "tune_official_d3": result["tune"]["official_d3"]},
                         sort_keys=True), flush=True)
    require(len({pair[arm]["initial_model_state_sha256"] for arm in pair}) == 1
            and len({pair[arm]["sample_order_sha256"] for arm in pair}) == 1,
            "GT7 three-arm initialization/order mismatch")
    require(abs(pair["canonical_full24"]["tune"]["official_d3"]
                - FULL24_REFERENCE[seed]) <= 1e-12,
            "GT24 rerun did not reproduce the prior fixed-recipe reference")
    pair["canonical_masked7_minus_full24_tune_d3"] = (
        pair["canonical_masked7"]["tune"]["official_d3"]
        - pair["canonical_full24"]["tune"]["official_d3"])
    pair["a1_nominal_minus_canonical_masked7_tune_d3"] = (
        pair["a1_nominal_masked7"]["tune"]["official_d3"]
        - pair["canonical_masked7"]["tune"]["official_d3"])
    pair["prior_full24_reference_d3"] = FULL24_REFERENCE[seed]
    runs[f"seed{seed}"] = pair
    records = {}
    records[f"seed{seed}"] = [{"row": int(gt_tune["rows"][i]),
            "scenario": gt_tune["scenario"][i], "session": gt_tune["session"][i],
            "frame": int(gt_tune["frame"][i]),
            **{f"{arm}_pred_abs_xy": pair[arm]["tune_prediction"][i].tolist()
               for arm in ("canonical_full24", "canonical_masked7", "a1_nominal_masked7")},
            **{f"{arm}_d3": float(pair[arm]["tune_row_d3"][i])
               for arm in ("canonical_full24", "canonical_masked7", "a1_nominal_masked7")}}
            for i in range(1998)]
    for arm in ("canonical_full24", "canonical_masked7", "a1_nominal_masked7"):
        pair[arm].pop("tune_prediction")
        pair[arm].pop("tune_row_d3")
    report = {
        "schema_version": 1, "status": "completed", "diagnostic_only": True,
        "privileged_non_submittable_inputs": True,
        "question": "separate GT stop/history16 information from rawtime-vs-nominal state5 timing",
        "input_time_contract": {"primary": "canonical C1 rawtime GT24",
                                "secondary": "replace only first5 by frozen A1 nominal causal status"},
        "feature_names": list(base.FEATURE_NAMES),
        "mask": {"applied_after_shared_normalization": True,
                 "zero_normalized_indices": list(MASKED_INDICES),
                 "zero_fields": list(base.FEATURE_NAMES[5:22]),
                 "retained_fields": list(base.FEATURE_NAMES[:5] + base.FEATURE_NAMES[22:]),
                 "a1_nominal_replacement": "first5 only before shared normalization"},
        "normalizer": {"source": "canonical GT24 grouped train54810",
                       "shared_across_all_three_arms_and_both_seeds": True,
                       "sha256": normalizer_sha},
        "recipe": {"architecture": "24-512-512-12 GELU+LayerNorm",
                   "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
                   "batch": 1024, "epochs": 60, "steps": 3240,
                   "scheduler": "per-step cosine", "loss": "SmoothL1 beta=0.1",
                   "selection": "LAST60 only", "terminal_tune_passes_per_arm": 1},
        "sources": sources, "inputs": {"gt_train": train_receipt, "gt_tune": tune_receipt,
            "status_overlay": {"root": str(overlay_root),
                "manifest_sha256": STATUS_OVERLAY_SHA256,
                "artifact_sha256": STATUS_ARTIFACT_SHA256,
                "row_frame_join_exact": True, "manifest": overlay_manifest}},
        "runs": runs, "records": records,
        "seed": seed,
        "boundaries": {"no_image_model_forward": True, "no_gpu": True,
                       "no_final_validation_access": True, "no_epoch_or_arm_selection": True,
                       "not_a_deployable_model": True},
    }
    for receipt in (train_receipt, tune_receipt):
        require(sha256(receipt["artifact"]) == receipt["artifact_sha256"]
                and sha256(receipt["manifest"]) == receipt["manifest_sha256"],
                "GT7 immutable input changed during fit")
    require(sha256(overlay_root / "overlay_manifest.json") == STATUS_OVERLAY_SHA256
            and all(sha256(overlay_root / f"{split}.npz") == digest
                    for split, digest in STATUS_ARTIFACT_SHA256.items()),
            "A1 status overlay changed during fit")
    require(validate_sources(args.expected_script_sha256) == sources,
            "GT7 calculation sources changed during fit")
    result_path = output / "result.json"
    base.atomic_json(result_path, report)
    print(json.dumps({"status": "completed", "result": str(result_path),
                      "sha256": sha256(result_path)}, sort_keys=True))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gt-train", required=True)
    parser.add_argument("--gt-tune", required=True)
    parser.add_argument("--expected-normalizer-sha256", required=True)
    parser.add_argument("--expected-script-sha256", required=True)
    parser.add_argument("--status-overlay-root", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    fit(arguments())
