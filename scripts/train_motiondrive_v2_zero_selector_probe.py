#!/usr/bin/env python3
"""Train the fixed P5-Z offline zero-vs-P4 selector head on frozen caches.

The two candidates and their D3 costs are cached diagnostic data.  This script
trains only a small FP32 readout; it cannot modify or certify the base model,
production API, latency, or competition compliance.  Tune is evaluated once
after LAST1000 and never controls a hyperparameter or checkpoint choice.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from scripts.cache_motiondrive_v2_zero_selector_features import (
    CALIBRATION_SHA256, CANDIDATE_ORDER, FEATURE_COMPONENTS, FEATURE_DIM,
    MOVE, P4_CHECKPOINT_SHA256, P4_GIT_SHA, P4_STEP, SPLIT_SHA256,
    SUPERVISION_SHA256, ZERO, file_sha, rows_sha256,
)
from scripts.motiondrive_v2_training import tensor_state_sha256, weighted_d3
from scripts.analyze_motiondrive_v2_query_pair import paired_cluster_bootstrap

HEAD_SEEDS = (0, 1, 2)
UPDATES = 1000
WARMUP = 100
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = .01
LAMBDA_EXPECTED_D3 = .25
GRAD_CLIP = 5.
HIDDEN_DIM = 64
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 20260908
COST_FP32_RTOL = 8 * torch.finfo(torch.float32).eps
COST_FP32_ATOL = 8 * torch.finfo(torch.float32).eps
EXPECTED = {"train": 54810, "tune": 1998}
EXPECTED_INVENTORY = {"train": {"scenes": 203, "sessions": 72},
                      "tune": {"scenes": 37, "sessions": 11}}
SESSION_046 = "session_046_20260210-100504"
RECIPE = {
    "architecture": "LayerNorm(790)->Linear(790,64)->SiLU->Linear(64,2)",
    "class_order": {"zero": ZERO, "move": MOVE},
    "initial_final_weight": 0., "initial_final_bias": [-2., 2.],
    "loss": "CE(D3 argmin, tie MOVE) + 0.25 * expected_D3 / max(train mean D_zero,1e-3)",
    "optimizer": "AdamW", "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
    "batch": BATCH, "updates": UPDATES, "warmup_updates": WARMUP,
    "schedule": "linear warmup then cosine to zero", "gradient_clip_norm": GRAD_CLIP,
    "decision": "ZERO iff logit_zero > logit_move; ties MOVE (p_move>=0.5)",
    "head_seeds": list(HEAD_SEEDS), "base_seeds": [0, 1],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_pinned_json(path, expected_sha):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and valid_sha(expected_sha)
            and file_sha(path) == expected_sha, f"Pinned JSON mismatch: {path}")
    value = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"Nonfinite JSON constant: {value}")))
    require(isinstance(value, dict), "JSON input must be a mapping")
    return value


def _atomic_new_json(path, value):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"Refusing to overwrite: {path}")
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"Refusing to overwrite: {path}") from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_new_torch(path, value):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"Refusing to overwrite: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"Refusing to overwrite: {path}") from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ZeroSelectorHead(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, HIDDEN_DIM),
                                     nn.SiLU(), nn.Linear(HIDDEN_DIM, 2))
        with torch.no_grad():
            self.network[-1].weight.zero_()
            self.network[-1].bias.copy_(torch.tensor([-2., 2.]))

    def forward(self, feature):
        require(feature.dtype == torch.float32 and feature.ndim == 2
                and feature.shape[1] == FEATURE_DIM, "FP32 [B,790] selector feature required")
        return self.network(feature)


def select_classes(logits):
    require(logits.ndim == 2 and logits.shape[1] == 2 and torch.isfinite(logits).all(),
            "Finite two-class logits required")
    # Deliberately not torch.argmax: an exact logit tie must retain MOVE.
    return torch.where(logits[:, ZERO] > logits[:, MOVE],
                       torch.full_like(logits[:, ZERO], ZERO, dtype=torch.long),
                       torch.full_like(logits[:, ZERO], MOVE, dtype=torch.long))


def learning_rate(update):
    require(type(update) is int and 1 <= update <= UPDATES, "Update index must be 1..1000")
    if update <= WARMUP:
        return LR * update / WARMUP
    progress = (update - WARMUP) / (UPDATES - WARMUP)
    return LR * .5 * (1. + math.cos(math.pi * progress))


def selector_loss(logits, labels, costs, c_scale):
    require(logits.shape == costs.shape and labels.shape == logits.shape[:1]
            and costs.dtype == torch.float32 and c_scale >= 1e-3,
            "Selector loss input contract differs")
    probabilities = logits.float().softmax(dim=1)
    expected_d3 = (probabilities * costs).sum(dim=1).mean()
    ce = F.cross_entropy(logits.float(), labels)
    total = ce + LAMBDA_EXPECTED_D3 * expected_d3 / c_scale
    require(torch.isfinite(total), "Nonfinite selector loss")
    return total, {"ce": ce.detach(), "expected_d3": expected_d3.detach()}


def _expected_cache_metadata(metadata, manifest, split, base_seed):
    require(isinstance(metadata, dict) and isinstance(manifest, dict), "Cache metadata mappings required")
    require(metadata.get("schema_version") == 2 and metadata.get("status") == "completed"
            and metadata.get("purpose") == "P5-Z offline diagnostic cache", "Wrong cache schema/purpose")
    require(metadata.get("split") == split and metadata.get("base_seed") == base_seed
            and metadata.get("checkpoint_step") == P4_STEP
            and metadata.get("checkpoint_sha256") == P4_CHECKPOINT_SHA256[base_seed]
            and metadata.get("training_git_sha") == P4_GIT_SHA, "Wrong base P4 cache")
    require(metadata.get("execution_scope") == "full_fixed_split"
            and metadata.get("pilot_samples") == 0 and metadata.get("full_cache_completed") is True,
            "Pilot/partial cache cannot train a selector")
    data = metadata.get("data", {})
    require(data.get("n") == EXPECTED[split] and data.get("cached_n") == EXPECTED[split]
            and data.get("scenes") == EXPECTED_INVENTORY[split]["scenes"]
            and data.get("sessions") == EXPECTED_INVENTORY[split]["sessions"]
            and data.get("split_sha256") == SPLIT_SHA256
            and data.get("supervision_manifest_sha256") == SUPERVISION_SHA256
            and data.get("canonical_calibration_sha256") == CALIBRATION_SHA256
            and data.get("augment") is False
            and data.get("frame_stride") == (1 if split == "train" else 5),
            "Wrong complete train/tune cache data contract")
    feature = metadata.get("feature_contract", {})
    require(feature.get("dtype") == "float32" and feature.get("dimension") == FEATURE_DIM
            and feature.get("components") == list(FEATURE_COMPONENTS)
            and feature.get("allowed") == [item["name"] for item in FEATURE_COMPONENTS],
            "Selector feature whitelist differs")
    model = metadata.get("model", {})
    state = model.get("state_sha256_before_after", {})
    require(model.get("mode") == "eval"
            and model.get("base_weights_frozen_by_no_update") is True
            and model.get("fixed_bn") is True
            and model.get("time_input") == "nominal"
            and model.get("precision") == "bf16_encoder_fp32_planner"
            and model.get("construction") == "scripts.audit_motiondrive_v2.construct_model"
            and model.get("input_adapter") == "scripts.evaluate_motiondrive_v2_planning.planning_model_inputs"
            and model.get("parameter_requires_grad_metadata") == "canonical_evaluator_preserved"
            and model.get("all_parameters_require_grad") is True
            and model.get("forward_context") == "torch.inference_mode"
            and model.get("base_optimizer_created") is False
            and model.get("base_backward_called") is False
            and model.get("feature_detached") is True
            and model.get("weights_updated") is False
            and valid_sha(state.get("before")) and state.get("after") == state.get("before"),
            "Cache did not preserve the canonical evaluator no-update forward contract")
    require(metadata.get("candidate_order") == list(CANDIDATE_ORDER)
            and metadata.get("class_order") == {"zero": ZERO, "move": MOVE}
            and metadata.get("final_val_accessed") is False
            and metadata.get("selection_performed") is False,
            "Candidate/final/selection cache contract differs")
    require(split != "tune" or metadata.get("immutable_tune_rowwise_plan_gt_d3_bitwise_verified") is True,
            "Tune cache lacks immutable evaluator parity")
    payload_metadata = manifest.get("payload_metadata")
    if payload_metadata is not None:
        require(payload_metadata == metadata, "Cache payload/sidecar metadata differs")


def validate_candidate_costs(stored, recomputed, labels):
    """Audit device-specific FP32 cost arithmetic without replacing GPU costs."""
    require(stored.dtype == recomputed.dtype == torch.float32
            and stored.shape == recomputed.shape and stored.ndim == 2 and stored.shape[1] == 2
            and labels.dtype == torch.int64 and labels.shape == stored.shape[:1]
            and torch.isfinite(stored).all() and torch.isfinite(recomputed).all()
            and (stored >= 0).all() and (recomputed >= 0).all(),
            "Candidate costs must be matching finite nonnegative FP32 [N,2]")
    delta = (stored - recomputed).abs()
    unequal = stored != recomputed
    require(torch.allclose(stored, recomputed, rtol=COST_FP32_RTOL, atol=COST_FP32_ATOL),
            "Stored GPU and independently recomputed CPU D3 costs exceed fixed FP32 tolerance")
    # Stored GPU costs are authoritative for the preregistered strict target.
    stored_labels = torch.where(stored[:, ZERO] < stored[:, MOVE],
                                torch.zeros_like(labels), torch.ones_like(labels))
    cpu_labels = torch.where(recomputed[:, ZERO] < recomputed[:, MOVE],
                             torch.zeros_like(labels), torch.ones_like(labels))
    require(torch.equal(stored_labels, labels), "Stored GPU D3 strict-tie labels differ")
    stored_margin = (stored[:, ZERO] - stored[:, MOVE]).abs()
    cpu_margin = (recomputed[:, ZERO] - recomputed[:, MOVE]).abs()
    label_difference = stored_labels != cpu_labels
    return {
        "policy": "stored GPU candidate_costs authoritative; CPU recomputation is audit only",
        "rtol": COST_FP32_RTOL, "atol": COST_FP32_ATOL,
        "unequal_elements": int(unequal.sum()), "max_abs": float(delta.max()),
        "cpu_argmin_difference_count": int(label_difference.sum()),
        "stored_gpu_margin_min_abs": float(stored_margin.min()),
        "cpu_recomputed_margin_min_abs": float(cpu_margin.min()),
        "differing_argmin_stored_margin_min_abs": (
            float(stored_margin[label_difference].min()) if label_difference.any() else None),
    }


def load_cache(cache_path, expected_cache_sha, manifest_path, expected_manifest_sha,
               *, split, base_seed):
    cache_path, manifest_path = Path(cache_path), Path(manifest_path)
    require(cache_path.is_file() and not cache_path.is_symlink() and valid_sha(expected_cache_sha)
            and file_sha(cache_path) == expected_cache_sha, f"Pinned {split} cache mismatch")
    manifest = read_pinned_json(manifest_path, expected_manifest_sha)
    require(manifest.get("cache_sha256") == expected_cache_sha, "Cache sidecar SHA differs")
    # Cache is generated locally by the paired trusted script; weights_only blocks arbitrary globals.
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    require(isinstance(payload, dict) and set(payload) == {
        "features", "candidate_plans", "candidate_costs", "labels", "gt_plan", "rows", "metadata"},
        "Unexpected cache payload keys")
    metadata = payload["metadata"]
    # The JSON includes row identities/buckets and publication hashes; shared fields must be exact.
    for key, value in metadata.items():
        require(manifest.get(key) == value, f"Cache payload/sidecar metadata mismatch: {key}")
    _expected_cache_metadata(metadata, manifest, split, base_seed)
    n = EXPECTED[split]
    shapes = {"features": (n, FEATURE_DIM), "candidate_plans": (n, 2, 6, 2),
              "candidate_costs": (n, 2), "labels": (n,), "gt_plan": (n, 6, 2), "rows": (n,)}
    for key, shape in shapes.items():
        value = payload[key]
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape,
                f"Wrong cache tensor shape: {key}")
    require(payload["features"].dtype == torch.float32
            and payload["candidate_plans"].dtype == torch.float32
            and payload["candidate_costs"].dtype == torch.float32
            and payload["gt_plan"].dtype == torch.float32
            and payload["labels"].dtype == torch.int64 and payload["rows"].dtype == torch.int64,
            "Wrong cache tensor dtype")
    require(all(torch.isfinite(payload[key]).all() for key in
                ("features", "candidate_plans", "candidate_costs", "gt_plan")),
            "Nonfinite cache tensor")
    require(torch.equal(payload["candidate_plans"][:, ZERO],
                        torch.zeros_like(payload["candidate_plans"][:, ZERO])),
            "Candidate zero is not exact zeros")
    zero_cost = weighted_d3(payload["candidate_plans"][:, ZERO], payload["gt_plan"])
    move_cost = weighted_d3(payload["candidate_plans"][:, MOVE], payload["gt_plan"])
    recomputed = torch.stack((zero_cost, move_cost), dim=1)
    payload["cost_validation"] = validate_candidate_costs(
        payload["candidate_costs"], recomputed, payload["labels"])
    ids, buckets = manifest.get("ids"), manifest.get("diagnostic_buckets")
    require(isinstance(ids, list) and len(ids) == n and isinstance(buckets, list) and len(buckets) == n
            and set(buckets) <= {"steady", "depart", "nonstop"}, "Cache identity/bucket fields differ")
    require([item.get("row") for item in ids] == payload["rows"].tolist()
            and rows_sha256(payload["rows"].numpy()) == metadata["data"]["rows_sha256"],
            "Cache row identity/order differs")
    require(file_sha(cache_path) == expected_cache_sha and file_sha(manifest_path) == expected_manifest_sha,
            "Cache inputs changed while loading")
    return payload, manifest


def validate_cache_pair(train, train_manifest, tune, tune_manifest, base_seed):
    for key in ("checkpoint_sha256", "completed_run_manifest_sha256", "training_git_sha",
                "source_manifest_sha256", "source_files"):
        require(train["metadata"].get(key) == tune["metadata"].get(key),
                f"Train/tune cache lineage differs: {key}")
    require(train["metadata"]["checkpoint_sha256"] == P4_CHECKPOINT_SHA256[base_seed],
            "Cache pair base checkpoint differs")
    require(not set(train["rows"].tolist()) & set(tune["rows"].tolist()),
            "Train/tune rows overlap")
    require(train_manifest["data"]["ego_cache_sha256"] == tune_manifest["data"]["ego_cache_sha256"],
            "Train/tune ego cache differs")
    return True


def _batch_indices(n, generator):
    """Infinite deterministic shuffled epochs; every update has exactly batch128."""
    order, cursor = torch.randperm(n, generator=generator), 0
    while True:
        pieces, needed = [], BATCH
        while needed:
            take = min(needed, n - cursor)
            pieces.append(order[cursor:cursor + take])
            cursor += take
            needed -= take
            if cursor == n:
                order, cursor = torch.randperm(n, generator=generator), 0
        yield torch.cat(pieces)


@torch.inference_mode()
def infer_logits(head, features, device, chunk=4096):
    result = []
    head.eval()
    for start in range(0, len(features), chunk):
        result.append(head(features[start:start + chunk].to(device=device, dtype=torch.float32)).cpu())
    return torch.cat(result)


def summarize(costs, labels, selections, ids, buckets, *, session_bootstrap=False):
    costs = costs.double()
    chosen = costs.gather(1, selections[:, None]).squeeze(1)
    oracle = costs.min(dim=1).values

    def group(indices):
        indices = torch.as_tensor(indices, dtype=torch.long)
        if not len(indices):
            return {"n": 0, "selected_d3": None}
        c, o, y, s = chosen[indices], oracle[indices], labels[indices], selections[indices]
        move, zero = costs[indices, MOVE], costs[indices, ZERO]
        recoverable = (move - o).mean()
        recovered = (move - c).mean()
        return {"n": len(indices), "selected_d3": float(c.mean()),
                "original_p4_d3": float(move.mean()), "zero_path_d3": float(zero.mean()),
                "oracle2_d3": float(o.mean()), "regret_to_oracle": float((c - o).mean()),
                "misselection_rate": float((s != y).double().mean()),
                "misselection_cost_mean_all_rows": float((c - o).mean()),
                "misselection_cost_sum": float((c - o).sum()),
                "selected_zero": int((s == ZERO).sum()), "oracle_zero": int((y == ZERO).sum()),
                "d3_delta_vs_original_p4": float((c - move).mean()),
                "maximum_recoverable_gain": float(recoverable),
                "realized_recovered_gain": float(recovered),
                "recoverable_fraction": (float(recovered / recoverable) if recoverable > 0 else None)}

    all_indices = list(range(len(labels)))
    session_names = [str(item["session"]) for item in ids]
    sessions = {name: group([i for i, value in enumerate(session_names) if value == name])
                for name in sorted(set(session_names))}
    by_bucket = {name: group([i for i, value in enumerate(buckets) if value == name])
                 for name in ("steady", "depart", "nonstop")}
    without_046 = [i for i, value in enumerate(session_names) if value != SESSION_046]
    require(len(without_046) < len(labels) or SESSION_046 not in set(session_names),
            "Internal session046 exclusion error")
    result = {"all": group(all_indices), "buckets": by_bucket, "sessions": sessions,
            "excluding_session_046": group(without_046) if SESSION_046 in sessions else None,
            "session_046_identifier": SESSION_046,
            "selection_rule": "ZERO iff logit_zero > logit_move; exact tie MOVE"}
    if session_bootstrap:
        delta = (chosen - costs[:, MOVE]).numpy()
        result["selector_minus_original_p4_session_cluster"] = paired_cluster_bootstrap(
            delta, session_names, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED)
        result["selector_minus_original_p4_session_cluster"].update(
            estimator="frame-weighted D3 mean difference",
            frame_iid_bootstrap=False,
            interpretation="exploratory repeatedly-used tune; not a final-holdout or compliance test")
    return result


def train_head(train, tune, train_manifest, tune_manifest, *, base_seed, head_seed, device):
    torch.manual_seed(head_seed)
    head = ZeroSelectorHead().to(device=device, dtype=torch.float32)
    initial_state_sha = tensor_state_sha256(head.state_dict())
    final_layer = head.network[-1]
    require(torch.count_nonzero(final_layer.weight) == 0
            and torch.equal(final_layer.bias.detach().cpu(), torch.tensor([-2., 2.])),
            "Initial final layer must be exact W=0,bias=[-2,+2]")
    initial_logits = infer_logits(head, train["features"][:min(BATCH, len(train["features"]))], device)
    require(torch.equal(select_classes(initial_logits),
                        torch.full((len(initial_logits),), MOVE, dtype=torch.long)),
            "Initial selector must retain every original P4 path")
    c_scale = max(float(train["candidate_costs"][:, ZERO].double().mean()), 1e-3)
    initial_proof = {
        "method": "final W exactly zero and bias exactly [-2,+2], independent of every feature row",
        "all_train_and_tune_rows_select_move_without_head_forward": True,
        "train_initial_selected_d3_equals_original_p4_d3": float(
            train["candidate_costs"][:, MOVE].double().mean()),
        "tune_initial_selected_d3_equals_original_p4_d3": float(
            tune["candidate_costs"][:, MOVE].double().mean()),
        "tune_selector_forward_performed_before_update1000": False,
    }
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu").manual_seed(head_seed)
    batches = _batch_indices(len(train["features"]), generator)
    trace = []
    head.train()
    for update in range(1, UPDATES + 1):
        indices = next(batches)
        feature = train["features"][indices].to(device=device, dtype=torch.float32)
        labels = train["labels"][indices].to(device)
        costs = train["candidate_costs"][indices].to(device=device, dtype=torch.float32)
        rate = learning_rate(update)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        loss, parts = selector_loss(head(feature), labels, costs, c_scale)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
        require(torch.isfinite(grad_norm), "Nonfinite selector gradient")
        optimizer.step()
        if update == 1 or update % 100 == 0:
            trace.append({"update": update, "lr": rate, "loss": float(loss.detach()),
                          "ce": float(parts["ce"]), "expected_d3": float(parts["expected_d3"]),
                          "grad_norm_before_clip": float(grad_norm)})
    require(all(torch.isfinite(value).all() for value in head.state_dict().values()),
            "Nonfinite trained selector state")
    for state in optimizer.state.values():
        require(float(state.get("step", -1)) == UPDATES, "Optimizer state did not reach LAST1000")
        require(all(not isinstance(value, torch.Tensor) or torch.isfinite(value).all()
                    for value in state.values()), "Nonfinite optimizer state")
    train_logits = infer_logits(head, train["features"], device)
    # This is the first and only tune use; no optimizer/checkpoint decision follows.
    tune_logits = infer_logits(head, tune["features"], device)
    train_selection, tune_selection = select_classes(train_logits), select_classes(tune_logits)
    metrics = {
        "train": summarize(train["candidate_costs"], train["labels"], train_selection,
                           train_manifest["ids"], train_manifest["diagnostic_buckets"]),
        "tune": summarize(tune["candidate_costs"], tune["labels"], tune_selection,
                          tune_manifest["ids"], tune_manifest["diagnostic_buckets"],
                          session_bootstrap=True),
    }
    return head, optimizer, {"c_scale_train_mean_d_zero": c_scale,
                             "cache_cost_validation": {
                                 "train": train["cost_validation"],
                                 "tune": tune["cost_validation"]},
                             "initial_head_state_sha256": initial_state_sha,
                             "initial_all_move_proof": initial_proof,
                             "final_head_state_sha256": tensor_state_sha256(head.state_dict()),
                             "trace": trace, "metrics": metrics,
                             "tune_first_selector_forward_after_update": UPDATES,
                             "tune_used_for_hyperparameters_or_selection": False}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for split in ("train", "tune"):
        parser.add_argument(f"--{split}-cache", required=True)
        parser.add_argument(f"--expected-{split}-cache-sha256", required=True)
        parser.add_argument(f"--{split}-manifest", required=True)
        parser.add_argument(f"--expected-{split}-manifest-sha256", required=True)
    parser.add_argument("--base-seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--head-seed", type=int, choices=HEAD_SEEDS, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    run_dir = Path(args.run_dir).expanduser().absolute()
    require(not run_dir.exists() and not run_dir.is_symlink(), "New isolated run-dir required")
    run_dir.mkdir(parents=True)
    inputs = {
        "train_cache": Path(args.train_cache), "train_manifest": Path(args.train_manifest),
        "tune_cache": Path(args.tune_cache), "tune_manifest": Path(args.tune_manifest),
    }
    expected = {
        "train_cache": args.expected_train_cache_sha256,
        "train_manifest": args.expected_train_manifest_sha256,
        "tune_cache": args.expected_tune_cache_sha256,
        "tune_manifest": args.expected_tune_manifest_sha256,
    }
    require(all(valid_sha(value) for value in expected.values()), "All cache/manifest SHA pins required")
    before = {name: file_sha(path) for name, path in inputs.items()}
    require(before == expected, "Input cache SHA mismatch before training")
    train, train_manifest = load_cache(inputs["train_cache"], expected["train_cache"],
                                       inputs["train_manifest"], expected["train_manifest"],
                                       split="train", base_seed=args.base_seed)
    tune, tune_manifest = load_cache(inputs["tune_cache"], expected["tune_cache"],
                                     inputs["tune_manifest"], expected["tune_manifest"],
                                     split="tune", base_seed=args.base_seed)
    validate_cache_pair(train, train_manifest, tune, tune_manifest, args.base_seed)
    device = torch.device(args.device)
    head, optimizer, evidence = train_head(train, tune, train_manifest, tune_manifest,
                                           base_seed=args.base_seed, head_seed=args.head_seed,
                                           device=device)
    manifest = {
        "schema_version": 1, "status": "running", "step": UPDATES,
        "purpose": "P5-Z offline diagnostic selector; not production gating",
        "created_utc": datetime.now(timezone.utc).isoformat(), "base_seed": args.base_seed,
        "head_seed": args.head_seed, "one_of_required_six_runs": True,
        "all_six_runs_required_for_reporting": True, "best_run_selection_performed": False,
        "recipe": RECIPE, "feature_dim": FEATURE_DIM,
        "trainer_source_sha256": file_sha(__file__),
        "base_checkpoint_sha256": P4_CHECKPOINT_SHA256[args.base_seed],
        "inputs_sha256": expected, "evidence": evidence,
        "last1000_is_primary": True, "final_val_accessed": False,
        "production_api_or_base_model_modified": False,
        "accuracy_latency_compliance_certified": False,
    }
    last = run_dir / "last.pth"
    _atomic_new_torch(last, {"step": UPDATES, "head": head.state_dict(),
                             "optimizer": optimizer.state_dict(), "manifest": manifest})
    manifest.update(status="completed", last_sha256=file_sha(last),
                    inputs_sha256_after={name: file_sha(path) for name, path in inputs.items()})
    require(manifest["inputs_sha256_after"] == before, "Input caches changed during head training")
    sidecar = run_dir / "manifest.json"
    _atomic_new_json(sidecar, manifest)
    print(json.dumps({"status": "completed", "last": str(last), "last_sha256": file_sha(last),
                      "manifest": str(sidecar), "manifest_sha256": file_sha(sidecar),
                      "base_seed": args.base_seed, "head_seed": args.head_seed,
                      "tune_selected_d3": evidence["metrics"]["tune"]["all"]["selected_d3"],
                      "diagnostic_only": True, "all_six_required_no_best_pick": True,
                      "final_val_accessed": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
