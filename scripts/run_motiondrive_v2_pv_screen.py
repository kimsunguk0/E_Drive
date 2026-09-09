#!/usr/bin/env python3
"""Run the fixed direct-vs-factorized-P/V planning-head screen."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_motiondrive_v2_pv_factorized_bank as bank_producer
import run_motiondrive_v2_shared_status_a1 as a1
import run_motiondrive_v2_early_precision as early
from build_grouped_split_v2 import sha256
from models.motiondrive_v2.pv_planner_head import (
    FactorizedPVHead, K_COUNT, P_COUNT, V_COUNT,
    install_factorized_pv_head, pv_head_state,
)
from models.motiondrive_v2.shared_status_query import install_shared_status_query
from motiondrive_v2_training import TIME_WEIGHTS, tensor_state_sha256


NAME = "motiondrive_v2_factorized_pv_head"
ARMS = ("direct", "pv", "pv_residual")
GPU_ASSIGNMENTS = {
    "direct": "GPU-5d2254f9-41a7-62dd-2b38-de82459acb24",
    "pv": "GPU-041334c0-089c-6ff5-b0b5-59ff445fa015",
    "pv_residual": "GPU-1e9aea73-4e6b-2cb5-b788-f3d99e6dc6f8",
}
PARENT = {
    "checkpoint_sha256": "43c9cdd172fe3e1212fd97a46b27b2f0abd879e3c360b7426281202825590cb0",
    "model_state_sha256": "54d4c68897cc2a079d71c130bc53e014a9907f89cbb3d98b8ee8d94ac1e2e077",
    "sidecar_sha256": "fd44bc485bb504952166f91059fd4756bed64944c1f97b143e0d10e87197493b",
    "source_manifest_sha256": "c3ec0bda1c5037a8e977d93072e8143a481c5a546da16971630a3a0a5249662a",
}
EXPECTED_BANK_KEYS = {
    "bank_xy", "path_id", "velocity_id", "path_source_rows", "path_xy",
    "path_cumulative", "path_total", "path_stratum", "path_support",
    "velocity_source_rows", "velocity_profiles", "velocity_stratum", "velocity_support",
}
SOFT_TARGET_TAU_METRES = .1
SOFT_CE_ALPHA = .1
SOURCE_FILES = set(early.SOURCE_FILES) | {
    "models/motiondrive_v2/pv_planner_head.py",
    "scripts/analyze_motiondrive_v2_train203_bank_feasibility.py",
    "scripts/build_motiondrive_v2_pv_factorized_bank.py",
    "scripts/run_motiondrive_v2_pv_screen.py",
    "tests/test_motiondrive_v2_pv_screen.py",
    "reports/motiondrive_v2_pv_screen_protocol_20260909.md",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_sha(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_source_manifest(path, expected_sha256):
    path = Path(path).resolve()
    require(len(expected_sha256) == 64 and sha256(path) == expected_sha256,
            "P/V source-manifest SHA mismatch")
    manifest = json.loads(path.read_text())
    require(set(manifest) == {"schema_version", "git_sha", "file_sha256"}
            and manifest["schema_version"] == 1
            and isinstance(manifest["git_sha"], str) and len(manifest["git_sha"]) == 40
            and isinstance(manifest["file_sha256"], dict)
            and set(manifest["file_sha256"]) == SOURCE_FILES,
            "P/V source manifest must be the exact runtime closure")
    actual = {name: sha256(ROOT / name) for name in sorted(SOURCE_FILES)}
    require(actual == manifest["file_sha256"], "P/V runtime source differs from manifest")
    return {"path": str(path), "sha256": expected_sha256,
            "git_sha": manifest["git_sha"], "file_sha256": actual}


def validate_runtime(args):
    require(args.seed == 0 and args.gpu == 0 and args.workers == 4
            and args.cuda_memory_limit_mib == 12000 and args.cuda_min_free_mib == 8192,
            "P/V runtime contract mismatch")
    expected = GPU_ASSIGNMENTS[args.arm]
    require(args.expected_physical_gpu_uuid == expected,
            "P/V fixed physical GPU assignment mismatch")
    if args.preflight_only:
        return {"gpu_used": False, "expected_physical_gpu_uuid": expected,
                "cuda_initialized": torch.cuda.is_initialized()}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(visible == expected and "," not in visible and torch.cuda.is_available()
            and torch.cuda.device_count() == 1, "P/V requires one UUID-isolated CUDA device")
    raw = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    actual = raw if raw.startswith("GPU-") else "GPU-" + raw
    require(actual == expected, "P/V observed physical GPU UUID mismatch")
    return {"gpu_used": True, "logical_gpu": 0, "actual_physical_gpu_uuid": actual}


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def validate_parent(args):
    checkpoint = Path(args.init).resolve()
    sidecar_path = Path(args.init_manifest).resolve()
    require(sha256(checkpoint) == PARENT["checkpoint_sha256"]
            and sha256(sidecar_path) == PARENT["sidecar_sha256"],
            "requires exact early-precision continued-direct LAST2000 parent")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sidecar = json.loads(sidecar_path.read_text())
    require(isinstance(payload, dict) and {"model", "step", "manifest"} <= set(payload)
            and payload["step"] == 2000 and sidecar.get("step") == 2000
            and sidecar.get("status") == "completed",
            "parent must be completed early-precision continued-direct LAST2000")
    require(tensor_state_sha256(payload["model"]) == PARENT["model_state_sha256"],
            "P/V parent model-state SHA mismatch")
    embedded = payload["manifest"]
    protocol = embedded.get("experimental_protocol", {})
    require(protocol.get("name") == early.NAME
            and protocol.get("arm") == "continuation_control"
            and protocol.get("seed") == 0
            and embedded.get("split_sha256") == a1.EXPECTED_SPLIT_SHA256,
            "P/V parent protocol mismatch")
    require(protocol.get("source", {}).get("sha256") == PARENT["source_manifest_sha256"],
            "P/V parent source-manifest lineage mismatch")
    for field in ("model_config", "loss_weights", "time_input", "time_input_policy",
                  "split_sha256", "load_report", "experimental_protocol"):
        require(_canonical(embedded.get(field)) == _canonical(sidecar.get(field)),
                f"P/V parent embedded/sidecar mismatch: {field}")
    require(any(name.startswith("shared_status_query_fusion.") for name in payload["model"])
            and not any(name.startswith("factorized_pv_head.") for name in payload["model"]),
            "parent must retain A2 query route and contain no P/V head")
    return payload, {"checkpoint": str(checkpoint), **PARENT, "step": 2000,
                     "weights_only": True, "fresh_optimizer": True}


def load_bank(args):
    manifest_path = Path(args.bank_manifest).resolve()
    bank_path = Path(args.bank_npz).resolve()
    manifest, arrays, produced_path = bank_producer.load_bank(
        manifest_path, args.expected_bank_manifest_sha256)
    require(produced_path.resolve() == bank_path
            and sha256(bank_path) == args.expected_bank_npz_sha256
            and manifest["artifact"]["sha256"] == args.expected_bank_npz_sha256,
            "factorized-bank NPZ path/SHA mismatch")
    require(set(arrays) == EXPECTED_BANK_KEYS, "factorized-bank NPZ keyset mismatch")
    shapes = {"bank_xy": (K_COUNT, 10, 2), "path_id": (K_COUNT,),
              "velocity_id": (K_COUNT,), "path_source_rows": (P_COUNT,),
              "path_xy": (P_COUNT, 10, 2), "path_cumulative": (P_COUNT, 10),
              "path_total": (P_COUNT,), "path_stratum": (P_COUNT,),
              "path_support": (P_COUNT,), "velocity_source_rows": (V_COUNT - 1,),
              "velocity_profiles": (V_COUNT, 11), "velocity_stratum": (V_COUNT - 1,),
              "velocity_support": (V_COUNT - 1,)}
    require(all(arrays[name].shape == shape for name, shape in shapes.items()),
            "factorized-bank array shape mismatch")
    require(arrays["bank_xy"].dtype == np.float32
            and arrays["path_xy"].dtype == np.float32
            and arrays["path_cumulative"].dtype == np.float32
            and arrays["path_total"].dtype == np.float32
            and arrays["velocity_profiles"].dtype == np.float32
            and arrays["path_id"].dtype == np.int32
            and arrays["velocity_id"].dtype == np.int32,
            "factorized-bank numeric dtype mismatch")
    require(all(np.isfinite(arrays[name]).all() for name in
                ("bank_xy", "path_xy", "path_cumulative", "path_total", "velocity_profiles")),
            "factorized-bank contains nonfinite values")
    require(np.array_equal(arrays["bank_xy"][0], np.zeros((10, 2), np.float32))
            and int(arrays["path_id"][0]) == -1 and int(arrays["velocity_id"][0]) == 0
            and (arrays["path_total"] > 0).all(), "factorized-bank zero/path contract mismatch")
    p, v = arrays["path_id"][1:].astype(np.int64), arrays["velocity_id"][1:].astype(np.int64)
    pair = p * (V_COUNT - 1) + (v - 1)
    require(np.array_equal(np.sort(pair), np.arange(P_COUNT * (V_COUNT - 1))),
            "factorized-bank Cartesian IDs are incomplete or duplicated")
    declared = manifest["arrays"]
    require(set(declared) == EXPECTED_BANK_KEYS
            and all(declared[name]["sha256"] == array_sha(arrays[name])
                    for name in EXPECTED_BANK_KEYS),
            "factorized-bank declared array SHA mismatch")
    gate_path = Path(args.oracle_gate_report).resolve()
    require(sha256(gate_path) == args.expected_oracle_gate_report_sha256,
            "factorized-bank oracle gate report SHA mismatch")
    gate = json.loads(gate_path.read_text())
    require(set(gate) == {"schema_version", "status", "inputs", "source", "bank", "tune",
                          "artifact", "boundaries"}
            and gate["schema_version"] == 1
            and gate["status"] == "passed_fixed_tune_oracle_gate"
            and gate["inputs"]["bank_manifest"]["sha256"] == args.expected_bank_manifest_sha256
            and gate["inputs"]["bank"]["sha256"] == args.expected_bank_npz_sha256
            and gate["source"]["script_sha256"] == manifest["source"]["script_sha256"]
            and gate["tune"]["rows"] == 1998
            and gate["tune"]["rows_sha256"] == a1.EXPECTED_ROWS["tune"][1]
            and gate["tune"]["gate_pass"] is True
            and float(gate["tune"]["oracle_weighted_d3"])
                <= float(gate["tune"]["gate_max_inclusive"]) == bank_producer.GATE_MAX_D3,
            "factorized-bank oracle gate contract mismatch")
    oracle_path = gate_path.parent / gate["artifact"]["path"]
    require(oracle_path.is_file() and not oracle_path.is_symlink()
            and sha256(oracle_path) == gate["artifact"]["sha256"],
            "factorized-bank oracle artifact mismatch")
    with np.load(oracle_path, allow_pickle=False) as source:
        oracle_arrays = {name: source[name] for name in source.files}
    require(set(oracle_arrays) == set(gate["artifact"]["arrays"]),
            "factorized-bank oracle artifact keyset mismatch")
    for name, contract in gate["artifact"]["arrays"].items():
        value = oracle_arrays[name]
        require(list(value.shape) == contract["shape"] and str(value.dtype) == contract["dtype"]
                and bank_producer.array_sha(value) == contract["sha256"],
                f"factorized-bank oracle artifact array mismatch: {name}")
    require(oracle_arrays["tune_rows"].shape == (1998,)
            and oracle_arrays["selected_candidate_id"].shape == (1998,)
            and oracle_arrays["oracle_d3"].shape == (1998,)
            and np.array_equal(oracle_arrays["tune_rows"], np.sort(oracle_arrays["tune_rows"]))
            and (oracle_arrays["selected_candidate_id"] >= 0).all()
            and (oracle_arrays["selected_candidate_id"] < K_COUNT).all()
            and np.isfinite(oracle_arrays["oracle_d3"]).all(),
            "factorized-bank oracle row/value contract mismatch")
    path_descriptor = np.concatenate([
        arrays["path_xy"].reshape(P_COUNT, -1) / arrays["path_total"][:, None],
        arrays["path_cumulative"] / arrays["path_total"][:, None]], axis=1).astype(np.float32)
    return arrays, {"manifest": str(manifest_path),
                    "manifest_sha256": args.expected_bank_manifest_sha256,
                    "npz": str(bank_path), "npz_sha256": args.expected_bank_npz_sha256,
                    "oracle_gate_report": str(gate_path),
                    "oracle_gate_report_sha256": args.expected_oracle_gate_report_sha256,
                    "oracle_artifact": str(oracle_path),
                    "oracle_artifact_sha256": gate["artifact"]["sha256"],
                    "oracle_weighted_d3": gate["tune"]["oracle_weighted_d3"],
                    "oracle_rows": torch.from_numpy(oracle_arrays["tune_rows"].copy()).long(),
                    "oracle_selected_id": torch.from_numpy(
                        oracle_arrays["selected_candidate_id"].copy()).long(),
                    "oracle_d3": torch.from_numpy(oracle_arrays["oracle_d3"].copy()).float(),
                    "p_count": P_COUNT, "v_count": V_COUNT, "candidate_count": K_COUNT,
                    "path_descriptor": torch.from_numpy(path_descriptor),
                    "velocity_descriptor": torch.from_numpy(arrays["velocity_profiles"].copy())}


def make_model(config, arm, arrays, bank):
    from models.motiondrive_v2.model import MotionDriveV2
    model = install_shared_status_query(MotionDriveV2(config))
    if arm != "direct":
        rng = torch.get_rng_state()
        try:
            head = FactorizedPVHead(
                torch.from_numpy(arrays["bank_xy"][:, :6].copy()),
                bank["path_descriptor"], bank["velocity_descriptor"],
                torch.from_numpy(arrays["path_id"].copy()),
                torch.from_numpy(arrays["velocity_id"].copy()),
                residual=arm == "pv_residual")
        finally:
            torch.set_rng_state(rng)
        install_factorized_pv_head(model, head)
    return model


def selector_state(model):
    state = pv_head_state(model)
    return {name: value for name, value in state.items()
            if not name.startswith(("factorized_pv_head.residual_context.",
                                    "factorized_pv_head.residual_candidate."))}


def prepare_model(payload, arm, arrays, bank):
    import train_motiondrive_v2 as trainer
    from models.motiondrive_v2 import MotionDriveV2Config
    trainer.seed_all(0)
    model = make_model(MotionDriveV2Config(**payload["manifest"]["model_config"]),
                       arm, arrays, bank)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    missing = list(incompatible.missing_keys)
    expected = ([name for name in model.state_dict() if name.startswith("factorized_pv_head.")]
                if arm != "direct" else [])
    require(not incompatible.unexpected_keys and sorted(missing) == sorted(expected),
            "parent must miss exactly the opt-in P/V head")
    require(all(torch.equal(model.state_dict()[name], value)
                for name, value in payload["model"].items()), "parent tensors changed on load")
    selector_sha = residual_sha = None
    if arm != "direct":
        selector_sha = tensor_state_sha256(selector_state(model))
        if arm == "pv_residual":
            final = model.factorized_pv_head.residual_candidate
            require(not bool(final.weight.count_nonzero()) and not bool(final.bias.count_nonzero()),
                    "P/V residual must begin at exact zero")
            residual_sha = tensor_state_sha256({name: value for name, value in pv_head_state(model).items()
                if name.startswith(("factorized_pv_head.residual_context.",
                                    "factorized_pv_head.residual_candidate."))})
    return model, {"initial_model_state_sha256": tensor_state_sha256(model.state_dict()),
                   "selector_state_sha256": selector_sha, "residual_state_sha256": residual_sha,
                   "missing_keys": missing, "parameter_count": sum(p.numel() for p in model.parameters())}


def candidate_d3(candidates: Tensor, target: Tensor) -> Tensor:
    if candidates.ndim == 3:
        candidates = candidates[None].expand(len(target), -1, -1, -1)
    if candidates.ndim != 4 or candidates.shape[1:] != (K_COUNT, 6, 2):
        raise ValueError("candidate tensor must be [K,6,2] or [B,K,6,2]")
    distance = torch.linalg.vector_norm(candidates.float() - target[:, None].float(), dim=-1)
    return (distance * distance.new_tensor(TIME_WEIGHTS)[None, None]).sum(-1)


def pv_plan_objective(outputs, batch, normalizers=None):
    logits = outputs["pv_logits"].float()
    target = batch["gt_plan"].float()
    valid = batch.get("plan_valid", torch.ones_like(target[..., 0], dtype=torch.bool))
    if logits.shape != (len(target), K_COUNT) or valid.shape != target.shape[:2]:
        raise ValueError("P/V loss input shape mismatch")
    complete = valid.bool().all(-1)
    safe = torch.where(complete[:, None, None], target, torch.zeros_like(target))
    base_d3 = candidate_d3(outputs["pv_base_candidates"], safe).detach()
    completed_d3 = candidate_d3(outputs["pv_candidates"], safe)
    soft_target = (-base_d3 / SOFT_TARGET_TAU_METRES).softmax(-1)
    probability = logits.softmax(-1)
    expected = (probability * completed_d3).sum(-1)
    soft_ce = -(soft_target * logits.log_softmax(-1)).sum(-1)
    denominator = (complete.sum() if normalizers is None
                   else normalizers["plan_complete"]).clamp_min(1)
    mask = complete.to(expected.dtype)
    expected_mean = (expected * mask).sum() / denominator
    ce_mean = (soft_ce * mask).sum() / denominator
    return expected_mean + SOFT_CE_ALPHA * ce_mean, {
        "pv_expected_d3": expected_mean, "pv_soft_ce": ce_mean,
        "pv_soft_ce_weighted": SOFT_CE_ALPHA * ce_mean}


def candidate_compute_loss(original, outputs, batch, weights, **kwargs):
    base_weights = dataclasses.replace(weights, plan=0.)
    total, parts = original(outputs, batch, base_weights, **kwargs)
    pv_loss, pv_parts = pv_plan_objective(outputs, batch, kwargs.get("normalizers"))
    total = total + weights.plan * pv_loss
    return total, {**parts, **pv_parts, "pv_plan_objective": pv_loss, "total": total}


def build_experiment(args, source, data, parent, bank, prepared):
    return {"schema_version": 1, "name": NAME, "arm": args.arm, "seed": 0,
            "source": source, "parent": parent,
            "status_overlay_manifest_sha256": args.expected_status_overlay_sha256,
            "bank": {k: v for k, v in bank.items() if not isinstance(v, Tensor)},
            "train_data": {"rows": data["train_rows"], "rows_sha256": data["train_rows_sha256"]},
            "tune_data": {"rows": data["tune_rows"], "rows_sha256": data["tune_rows_sha256"]},
            "expected_initial_model_state_sha256": prepared["initial_model_state_sha256"],
            "expected_selector_state_sha256": prepared["selector_state_sha256"],
            "expected_residual_state_sha256": prepared["residual_state_sha256"],
            "expected_missing_state_keys": prepared["missing_keys"],
            "architecture": {"decoder_feature": [6, 128], "P": P_COUNT, "V": V_COUNT,
                "K": K_COUNT, "joint_key": "normalize(path+velocity+path*velocity)",
                "cosine_logit_scale": "20*sigmoid(parameter)",
                "raw_goal_or_status_head_input": False,
                "residual": args.arm == "pv_residual",
                "residual_formula": "0.5*tanh(low_rank4(context,candidate_key)) metres before argmax"},
            "estimated_head_tensor_bytes": {
                "shared_joint_keys_Kx32_fp32": K_COUNT * 32 * 4,
                "microbatch2_one_candidate_or_residual_tensor": 2 * K_COUNT * 6 * 2 * 4,
                "eval_batch4_one_candidate_or_residual_tensor": 4 * K_COUNT * 6 * 2 * 4,
                "actual_peak_requires_train_smoke_measurement": True,
            },
            "loss": {"direct_official_d3": args.arm == "direct",
                "candidate_expected_official_d3": args.arm != "direct",
                "soft_ce_alpha": SOFT_CE_ALPHA if args.arm != "direct" else 0.,
                "soft_target": "softmax(-fixed_base_candidate_D3/0.1m)",
                "normalizer": "full_effective_batch_plan_complete"},
            "fresh_optimizer_step_zero": True, "all_parent_parameters_trainable": True,
            "initial_parent_output_exact": args.arm == "direct",
            "candidate_step_zero_contract": ("not_applicable" if args.arm == "direct"
                else "pv and pv_residual logits/selected plan/id exact; not parent-output parity"),
            "fixed_bn_running_statistics": True,
            "smoke_only_never_training_initializer": bool(args.smoke_only),
            "terminal_tune_evaluations": 0 if args.smoke_only else 1,
            "final_validation_accessed": False,
            "interpretation": "single-seed system/trainability screen; not an architecture upper bound"}


@contextlib.contextmanager
def patched_runtime(arm, arrays, bank, overlay, expected_parent_sha, expected_missing,
                    expected_selector_sha, expected_residual_sha):
    import models.motiondrive_v2 as model_api
    import motiondrive_v2_data as data_api
    import train_motiondrive_v2 as trainer
    import evaluate_motiondrive_v2_planning as planning_eval
    originals = {"model": model_api.MotionDriveV2, "dataset": data_api.MotionDriveDataset,
                 "train_inputs": trainer.model_inputs, "eval_inputs": planning_eval.planning_model_inputs,
                 "protocol": trainer._validate_experimental_protocol,
                 "runtime": trainer._validate_experimental_runtime,
                 "loader": trainer._load_initial_model_state,
                 "schedule": trainer._training_schedule_actions,
                 "loss": trainer.compute_loss, "evaluate": trainer.evaluate}
    eval_rows: list[Tensor] | None = None

    def model_factory(config=None):
        from models.motiondrive_v2 import MotionDriveV2Config
        return make_model(MotionDriveV2Config() if config is None else config,
                          arm, arrays, bank)

    def dataset_factory(**kwargs):
        require(kwargs.get("split") in ("train", "tune"), "only train/tune datasets permitted")
        return a1.SharedStatusDataset(originals["dataset"](**kwargs), kwargs["split"],
                                      overlay, "provided_causal_5d")

    def train_inputs(batch, time_input="raw", nominal_history_seconds=(.1, .2, .5, 1.)):
        nonlocal eval_rows
        if eval_rows is not None:
            eval_rows.append(batch["row"].detach().cpu().long())
        return a1.shared_status_model_inputs(originals["train_inputs"], batch,
            time_input=time_input, nominal_history_seconds=nominal_history_seconds)

    def eval_inputs(batch, time_input="raw"):
        return a1._add_status(originals["eval_inputs"](batch, time_input), batch)

    def validate_protocol(experiment):
        require(experiment.get("name") == NAME and experiment.get("arm") == arm,
                "P/V experimental protocol mismatch")

    def validate_experimental_runtime(runtime_args, experiment):
        expected = {"phase": "joint", "goal_on": 1, "state_on": 1,
                    "steps": 2 if experiment["smoke_only_never_training_initializer"] else 2000,
                    "batch": 16, "microbatch": 2, "eval_batch": 4,
                    "eval_every": 2000, "save_every": 500,
                    "lr": 5e-5, "backbone_lr": 5e-6, "weight_decay": .01,
                    "warmup": 100, "grad_clip": 5., "alpha_occ": .2,
                    "alpha_lane": .2, "alpha_motion": .2, "uncertainty": 1,
                    "precision": "bf16", "time_input": "nominal", "bn_policy": "fixed",
                    "arch": "resnet50", "motion_input_mode": "low_feature",
                    "cross_cell_goal_mode": "zero", "history_contract": "control",
                    "train_stride": 1, "eval_stride": 5, "max_train_samples": 0,
                    "max_eval_samples": 0, "eval_split": "tune"}
        require(all(getattr(runtime_args, key) == value for key, value in expected.items())
                and runtime_args.seed == 0 and runtime_args.init
                and not runtime_args.resume and not runtime_args.pretrained
                and not runtime_args.eval_only and not runtime_args.train_scenes
                and not runtime_args.eval_scenes, "fixed P/V continuation recipe mismatch")

    def load_initial(model, common, experiment=None):
        require(tensor_state_sha256(common["model"]) == expected_parent_sha,
                "trainer parent state mismatch")
        incompatible = model.load_state_dict(common["model"], strict=False)
        require(list(incompatible.missing_keys) == expected_missing
                and not incompatible.unexpected_keys, "trainer P/V missing-key mismatch")
        require(all(torch.equal(model.state_dict()[name], value)
                    for name, value in common["model"].items()), "trainer changed parent tensors")
        if expected_selector_sha is not None:
            require(tensor_state_sha256(selector_state(model)) == expected_selector_sha,
                    "trainer selector initialization mismatch")
        if expected_residual_sha is not None:
            residual = {name: value for name, value in pv_head_state(model).items()
                        if name.startswith(("factorized_pv_head.residual_context.",
                                           "factorized_pv_head.residual_candidate."))}
            require(tensor_state_sha256(residual) == expected_residual_sha,
                    "trainer residual initialization mismatch")
        return {"pv_parent_load": {"strict_existing_keys": True, "missing_keys": expected_missing,
                "weights_only": True, "parent_model_state_sha256": expected_parent_sha,
                "selector_state_sha256": expected_selector_sha,
                "residual_state_sha256": expected_residual_sha}}

    def schedule(step, runtime_args, experiment):
        if experiment["smoke_only_never_training_initializer"]:
            return False, False
        return step == 2000, step in (500, 1000, 2000)

    def compute_loss(outputs, batch, weights, **kwargs):
        if arm == "direct":
            return originals["loss"](outputs, batch, weights, **kwargs)
        return candidate_compute_loss(originals["loss"], outputs, batch, weights, **kwargs)

    def evaluate(model, *args, **kwargs):
        nonlocal eval_rows
        require(eval_rows is None and not hasattr(model, "_pv_eval_capture"),
                "P/V evaluation capture is stale")
        eval_rows = []
        if arm != "direct":
            model._pv_eval_capture = []
        try:
            report, records = originals["evaluate"](model, *args, **kwargs)
            rows = torch.cat(eval_rows).tolist() if eval_rows else []
            oracle_rows = bank["oracle_rows"].tolist()
            oracle_ids = bank["oracle_selected_id"].tolist()
            oracle_d3 = bank["oracle_d3"].tolist()
            require(len(rows) == len(records) == len(oracle_rows)
                    and rows == [int(record["row"]) for record in records] == oracle_rows,
                    "P/V terminal/oracle row/count mismatch")
            for record, candidate_id, value in zip(records, oracle_ids, oracle_d3):
                record.update(pv_oracle_selected_id=int(candidate_id),
                              pv_oracle_d3=float(value),
                              pv_oracle_gap=float(record["d3"]) - float(value))
            if arm == "direct":
                return report, records
            captures = model._pv_eval_capture
            ids = torch.cat([item["pv_selected_id"] for item in captures]).cpu().tolist()
            pre = torch.cat([item["pv_pre_plan"] for item in captures]).float().cpu().tolist()
            post = torch.cat([item["pv_post_plan"] for item in captures]).float().cpu().tolist()
            entropy = torch.cat([item["pv_entropy"] for item in captures]).float().cpu().tolist()
            require(len(records) == len(ids) == len(pre) == len(post) == len(entropy),
                    "P/V terminal capture row/count mismatch")
            for record, candidate_id, before, after, uncertainty in zip(
                    records, ids, pre, post, entropy):
                record.update(pv_selected_id=int(candidate_id), pv_pre_abs_xy=before,
                              pv_post_abs_xy=after, pv_score_entropy=float(uncertainty))
            return report, records
        finally:
            eval_rows = None
            if hasattr(model, "_pv_eval_capture"):
                delattr(model, "_pv_eval_capture")

    model_api.MotionDriveV2 = model_factory
    data_api.MotionDriveDataset = dataset_factory
    trainer.model_inputs = train_inputs
    planning_eval.planning_model_inputs = eval_inputs
    trainer._validate_experimental_protocol = validate_protocol
    trainer._validate_experimental_runtime = validate_experimental_runtime
    trainer._load_initial_model_state = load_initial
    trainer._training_schedule_actions = schedule
    trainer.compute_loss = compute_loss
    trainer.evaluate = evaluate
    try:
        yield
    finally:
        model_api.MotionDriveV2 = originals["model"]
        data_api.MotionDriveDataset = originals["dataset"]
        trainer.model_inputs = originals["train_inputs"]
        planning_eval.planning_model_inputs = originals["eval_inputs"]
        trainer._validate_experimental_protocol = originals["protocol"]
        trainer._validate_experimental_runtime = originals["runtime"]
        trainer._load_initial_model_state = originals["loader"]
        trainer._training_schedule_actions = originals["schedule"]
        trainer.compute_loss = originals["loss"]
        trainer.evaluate = originals["evaluate"]


def trainer_argv(args):
    return a1.trainer_argv(args)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--init-manifest", required=True)
    parser.add_argument("--bank-manifest", required=True)
    parser.add_argument("--expected-bank-manifest-sha256", required=True)
    parser.add_argument("--bank-npz", required=True)
    parser.add_argument("--expected-bank-npz-sha256", required=True)
    parser.add_argument("--oracle-gate-report", required=True)
    parser.add_argument("--expected-oracle-gate-report-sha256", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--status-overlay-root", required=True)
    parser.add_argument("--expected-status-overlay-sha256", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    parser.add_argument("--gpu", type=int, choices=(0,), default=0)
    parser.add_argument("--expected-physical-gpu-uuid", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cuda-memory-limit-mib", type=int, default=12000)
    parser.add_argument("--cuda-min-free-mib", type=int, default=8192)
    parser.add_argument("--expected-initial-state-sha256")
    parser.add_argument("--expected-selector-state-sha256")
    parser.add_argument("--expected-residual-state-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    require(not (args.preflight_only and args.smoke_only), "preflight and smoke are separate")
    if args.smoke_only:
        require("smoke_only" in Path(args.run_dir).name, "smoke directory must be explicit")
    runtime = validate_runtime(args)
    source = validate_source_manifest(args.source_manifest, args.expected_source_manifest_sha256)
    payload, parent = validate_parent(args)
    arrays, bank = load_bank(args)
    data_args = argparse.Namespace(**{**vars(args), "arm": "provided_causal_5d"})
    data, overlay = a1.validate_data(data_args)
    _model, prepared = prepare_model(payload, args.arm, arrays, bank)
    experiment = build_experiment(args, source, data, parent, bank, prepared)
    command = trainer_argv(args)
    if args.preflight_only:
        print(json.dumps({"status": "completed_cpu_preflight_only", "arm": args.arm,
                          "runtime": runtime, "source": source, "data": data,
                          "parent": parent, "bank": experiment["bank"],
                          "prepared": prepared, "experiment": experiment,
                          "trainer_argv": command, "cuda_initialized": torch.cuda.is_initialized()},
                         sort_keys=True, allow_nan=False), flush=True)
        return
    require(args.expected_initial_state_sha256 == prepared["initial_model_state_sha256"]
            and args.expected_selector_state_sha256 == prepared["selector_state_sha256"]
            and args.expected_residual_state_sha256 == prepared["residual_state_sha256"],
            "launch requires reviewed initial/selector/residual state pins")
    immutable = [args.init, args.init_manifest, args.bank_manifest, args.bank_npz,
                 args.oracle_gate_report, bank["oracle_artifact"],
                 args.source_manifest, args.split_manifest,
                 str(Path(args.supervision_root) / "supervision_manifest.json"),
                 str(Path(args.supervision_root) / "calibration.npz"),
                 str(Path(args.status_overlay_root) / "overlay_manifest.json"),
                 str(Path(args.status_overlay_root) / "train.npz"),
                 str(Path(args.status_overlay_root) / "tune.npz")]
    before = {str(Path(path).resolve()): sha256(path) for path in immutable}
    import train_motiondrive_v2 as trainer
    with patched_runtime(args.arm, arrays, bank, overlay, PARENT["model_state_sha256"],
                         prepared["missing_keys"], prepared["selector_state_sha256"],
                         prepared["residual_state_sha256"]):
        trainer.run_training(command, experiment=experiment)
    after = {str(Path(path).resolve()): sha256(path) for path in immutable}
    require(before == after, "P/V immutable inputs changed during training")
    validate_source_manifest(args.source_manifest, args.expected_source_manifest_sha256)


if __name__ == "__main__":
    main()
