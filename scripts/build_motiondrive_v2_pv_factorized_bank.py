#!/usr/bin/env python3
"""Build and gate one fixed current-train203 factorized trajectory bank.

``fit`` uses only the current train203 selection to fit P=512 path medoids and
V=128 velocity profiles (one exact stop plus 127 medoids), then publishes an
immutable 65,025-row bank. ``gate`` first verifies that frozen artifact by its
external manifest SHA, then computes one exact tune1998 weighted-D3 oracle.

The shared NPZ containers physically include rows outside the selected split;
only explicitly selected train rows contribute to fitting and only explicitly
selected tune rows contribute to the later gate. No final136 row is selected,
fit, evaluated, or analyzed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_motiondrive_v2_train203_bank_feasibility as current


P_COUNT = 512
V_COUNT = 128
MOVING_V_COUNT = V_COUNT - 1
BANK_COUNT = 1 + P_COUNT * MOVING_V_COUNT
FIT_SEED = 20260909
GATE_MAX_D3 = 0.11486043
ORACLE_CANDIDATE_CHUNK = 512
ROUTE_NAMES = ("straight", "left", "right", "uturn")
VELOCITY_NAMES = ("creep", "decel", "cruise", "accel")
OFFICIAL_W = np.asarray([11, 11, 5, 5, 2, 2], np.float32) / np.float32(36.)
EXPECTED_HELPER_SHA256 = "b302c0db66baf4e6483fa8e2498a2c6f152f012747a553abb34ce6eb0474e33b"
EXPECTED_ARRAY_SCHEMA = {
    "bank_xy": ([BANK_COUNT, 10, 2], "float32"),
    "path_id": ([BANK_COUNT], "int32"),
    "velocity_id": ([BANK_COUNT], "int32"),
    "path_source_rows": ([P_COUNT], "int64"),
    "path_xy": ([P_COUNT, 10, 2], "float32"),
    "path_cumulative": ([P_COUNT, 10], "float32"),
    "path_total": ([P_COUNT], "float32"),
    "path_stratum": ([P_COUNT], "<U8"),
    "path_support": ([P_COUNT], "int64"),
    "velocity_source_rows": ([MOVING_V_COUNT], "int64"),
    "velocity_profiles": ([V_COUNT, 11], "float32"),
    "velocity_stratum": ([MOVING_V_COUNT], "<U8"),
    "velocity_support": ([MOVING_V_COUNT], "int64"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value, dtype=None):
    array = np.asarray(value, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def native(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        require(math.isfinite(value), "nonfinite JSON float")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        require(math.isfinite(result), "nonfinite NumPy float")
        return result
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    raise TypeError(f"unsupported JSON type: {type(value).__name__}")


def write_new_json(path, value):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(native(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    json.loads(payload)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def write_new_npz(path, arrays):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        np.savez_compressed(stream, **arrays)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def read_json(path, expected_sha=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"ordinary JSON required: {path}")
    raw = path.read_bytes()
    if expected_sha is not None:
        require(valid_sha(expected_sha) and hashlib.sha256(raw).hexdigest() == expected_sha,
                f"JSON SHA mismatch: {path}")
    value = json.loads(raw, parse_constant=lambda item: (_ for _ in ()).throw(
        ValueError(f"nonfinite JSON constant: {item}")))
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def verify_runtime_and_source(expected_script_sha):
    current.require_runtime()
    require(valid_sha(expected_script_sha) and file_sha(__file__) == expected_script_sha,
            "executed script SHA mismatch")
    require(file_sha(current.__file__) == EXPECTED_HELPER_SHA256,
            "current train203 provenance helper SHA mismatch")


def cumulative_geometry(trajectories):
    trajectories = np.asarray(trajectories, np.float32)
    origin = np.zeros((len(trajectories), 1, 2), np.float32)
    step = np.linalg.norm(np.diff(np.concatenate([origin, trajectories], 1), axis=1), axis=2)
    cumulative = np.cumsum(step, axis=1)
    return step, cumulative, cumulative[:, -1]


def last_heading_deg(trajectories):
    trajectories = np.asarray(trajectories, np.float32)
    delta = np.diff(np.concatenate([np.zeros((len(trajectories), 1, 2), np.float32),
                                    trajectories], axis=1), axis=1)
    norm = np.linalg.norm(delta, axis=2)
    result = np.zeros(len(trajectories), np.float64)
    for index in range(len(trajectories)):
        valid = np.flatnonzero(norm[index] > 1e-3)
        if len(valid):
            vector = delta[index, valid[-1]]
            result[index] = np.degrees(np.arctan2(vector[1], vector[0]))
    return result


def route_labels(trajectories):
    heading = last_heading_deg(trajectories)
    endpoint = np.asarray(trajectories)[:, -1]
    labels = np.full(len(trajectories), "straight", dtype="U8")
    uturn = (endpoint[:, 0] < -1.) | (np.abs(heading) > 100.)
    labels[uturn] = "uturn"
    labels[(~uturn) & (heading > 10.) & (endpoint[:, 1] > 1.)] = "left"
    labels[(~uturn) & (heading < -10.) & (endpoint[:, 1] < -1.)] = "right"
    return labels


def velocity_labels(step, total):
    delta = step[:, -3:].mean(1) - step[:, :3].mean(1)
    labels = np.full(len(step), "cruise", dtype="U8")
    labels[delta > .5] = "accel"
    labels[delta < -.5] = "decel"
    labels[total < 2.] = "creep"
    labels[total < .25] = "stop"
    return labels


def normalized_path_features(trajectories, cumulative, total, sample_count=33):
    grid = np.linspace(0., 1., sample_count, dtype=np.float64)
    result = np.empty((len(trajectories), sample_count, 2), np.float32)
    for index in range(len(trajectories)):
        u = np.concatenate([[0.], cumulative[index] / total[index]])
        q = np.concatenate([np.zeros((1, 2)), trajectories[index] / total[index]], 0)
        reverse_unique = np.unique(u[::-1], return_index=True)[1]
        keep = np.sort(len(u) - 1 - reverse_unique)
        result[index, :, 0] = np.interp(grid, u[keep], q[keep, 0])
        result[index, :, 1] = np.interp(grid, u[keep], q[keep, 1])
    return result.reshape(len(trajectories), -1)


def allocate_sqrt_quotas(labels, names: Sequence[str], total):
    counts = {name: int(np.sum(labels == name)) for name in names}
    active = [name for name in names if counts[name] > 0]
    require(total >= len(active), "cluster total cannot preserve all nonempty strata")
    quotas = {name: int(name in active) for name in names}
    remaining = total - len(active)
    if remaining:
        scores = np.asarray([np.sqrt(counts[name]) for name in active], np.float64)
        raw = remaining * scores / scores.sum()
        floor = np.floor(raw).astype(int)
        for name, addition in zip(active, floor):
            quotas[name] += int(addition)
        left = remaining - int(floor.sum())
        for position in np.argsort(-(raw - floor), kind="stable")[:left]:
            quotas[active[int(position)]] += 1
    overflow = 0
    for name in active:
        if quotas[name] > counts[name]:
            overflow += quotas[name] - counts[name]
            quotas[name] = counts[name]
    while overflow:
        candidates = [name for name in active if quotas[name] < counts[name]]
        require(candidates, "unable to redistribute cluster quota")
        candidates.sort(key=lambda name: (-(counts[name] - quotas[name]), names.index(name)))
        quotas[candidates[0]] += 1
        overflow -= 1
    require(sum(quotas.values()) == total, "cluster quota sum mismatch")
    return counts, quotas


def unique_nearest_rows(features, centers):
    selected = []
    used = np.zeros(len(features), bool)
    for center in centers:
        distance = np.sum((features - center[None]) ** 2, axis=1)
        distance[used] = np.inf
        choice = int(np.argmin(distance))
        require(np.isfinite(distance[choice]), "unique medoid pool exhausted")
        selected.append(choice)
        used[choice] = True
    return np.asarray(selected, np.int64)


@dataclass
class MedoidAxis:
    source_local: np.ndarray
    source_global: np.ndarray
    strata: np.ndarray
    support: np.ndarray
    counts: dict
    quotas: dict


def stratified_medoids(features, global_rows, labels, names, clusters, seed,
                       *, standardize_within_stratum):
    counts, quotas = allocate_sqrt_quotas(labels, names, clusters)
    sources, supports, source_labels = [], [], []
    for stratum_index, name in enumerate(names):
        local = np.flatnonzero(labels == name)
        count = quotas[name]
        if count == 0:
            continue
        values = np.asarray(features[local], np.float64)
        if standardize_within_stratum:
            values = StandardScaler().fit_transform(values)
        if count == 1:
            centers = values.mean(0, keepdims=True)
            assignments = np.zeros(len(values), np.int64)
        else:
            model = MiniBatchKMeans(
                n_clusters=count, random_state=seed + 1009 * stratum_index + 17 * clusters,
                batch_size=min(4096, max(256, len(values))), n_init=3, max_iter=200,
                max_no_improvement=25, reassignment_ratio=0.)
            assignments = model.fit_predict(values)
            centers = model.cluster_centers_
        choice = unique_nearest_rows(values, centers)
        sources.append(local[choice])
        supports.append(np.bincount(assignments, minlength=count).astype(np.int64))
        source_labels.extend([name] * count)
    source_local = np.concatenate(sources).astype(np.int64)
    support = np.concatenate(supports).astype(np.int64)
    strata = np.asarray(source_labels, dtype="U8")
    semantic = {name: index for index, name in enumerate(names)}
    order = np.asarray(sorted(range(len(source_local)), key=lambda index: (
        semantic[strata[index]], -int(support[index]), int(source_local[index]))), np.int64)
    return MedoidAxis(source_local[order], np.asarray(global_rows, np.int64)[source_local[order]],
                      strata[order], support[order], counts, quotas)


def compose_bank(path_xy, path_cumulative, path_total, velocity_profiles):
    path_xy = np.asarray(path_xy, np.float32)
    velocity_profiles = np.asarray(velocity_profiles, np.float32)
    p_count, v_count = len(path_xy), len(velocity_profiles)
    require(path_xy.shape[1:] == (10, 2) and path_cumulative.shape == (p_count, 10)
            and path_total.shape == (p_count,) and velocity_profiles.shape[1:] == (11,),
            "factorized-axis shapes invalid")
    bank = np.empty((1 + p_count * (v_count - 1), 10, 2), np.float32)
    path_id = np.full(len(bank), -1, np.int32)
    velocity_id = np.zeros(len(bank), np.int32)
    bank[0] = 0.
    row = 1
    for path in range(p_count):
        for velocity in range(1, v_count):
            scale = float(velocity_profiles[velocity, 0])
            progress = velocity_profiles[velocity, 1:]
            u = np.concatenate([[0.], path_cumulative[path] / path_total[path]])
            q = np.concatenate([np.zeros((1, 2)), path_xy[path] / path_total[path]], 0)
            reverse_unique = np.unique(u[::-1], return_index=True)[1]
            keep = np.sort(len(u) - 1 - reverse_unique)
            bank[row, :, 0] = scale * np.interp(progress, u[keep], q[keep, 0])
            bank[row, :, 1] = scale * np.interp(progress, u[keep], q[keep, 1])
            path_id[row] = path
            velocity_id[row] = velocity
            row += 1
    require(row == len(bank), "bank composition row mismatch")
    return bank, path_id, velocity_id


def bank_diagnostics(bank, path_id, velocity_id, p_count=P_COUNT, v_count=V_COUNT):
    require(bank.shape == (1 + p_count * (v_count - 1), 10, 2)
            and bank.dtype == np.float32 and np.isfinite(bank).all(), "bank tensor invalid")
    require(path_id.shape == velocity_id.shape == (len(bank),)
            and path_id.dtype == np.int32 and velocity_id.dtype == np.int32,
            "bank ID arrays invalid")
    require(np.array_equal(bank[0], np.zeros((10, 2), np.float32))
            and path_id[0] == -1 and velocity_id[0] == 0, "exact stop row invalid")
    expected_path = np.repeat(np.arange(p_count, dtype=np.int32), v_count - 1)
    expected_velocity = np.tile(np.arange(1, v_count, dtype=np.int32), p_count)
    require(np.array_equal(path_id[1:], expected_path)
            and np.array_equal(velocity_id[1:], expected_velocity),
            "path-major Cartesian ID ordering mismatch")
    unique = np.unique(bank.reshape(len(bank), -1), axis=0)
    duplicate_count = int(len(bank) - len(unique))
    exact_zero_count = int(np.sum(np.all(bank == 0, axis=(1, 2))))
    return {"finite": True, "exact_zero_count": exact_zero_count,
            "duplicate_count": duplicate_count,
            "path_major_cartesian_coverage": True}


def array_contract(arrays):
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "sha256": array_sha(value)} for name, value in arrays.items()}


def load_current_inputs(split_path, ego_path, ego5_path):
    split, split_receipt = current.load_split(split_path)
    ego, ego5, names, cache_receipts = current.load_cache_metadata(ego_path, ego5_path)
    return split, ego, ego5, names, {"split": split_receipt, **cache_receipts}


def fit(args):
    verify_runtime_and_source(args.expected_script_sha256)
    require(args.output_dir.name not in ("", ".", "..")
            and not args.output_dir.exists() and not args.output_dir.is_symlink(),
            "fit output directory must be fresh")
    split, ego, ego5, names, inputs = load_current_inputs(
        args.split_manifest, args.ego_cache, args.ego5_cache)
    train_rows = current.selected_rows(split, names, ego["frame"], "train")
    train_fut5, _ = current.selected_trajectories(ego, ego5, train_rows, split="train")
    step, cumulative, total = cumulative_geometry(train_fut5)
    moving = total >= np.float32(.25)
    require(int(moving.sum()) > P_COUNT and int(moving.sum()) > MOVING_V_COUNT,
            "insufficient non-stop train rows")
    moving_rows = train_rows[moving]
    path_features = normalized_path_features(train_fut5[moving], cumulative[moving], total[moving])
    progress_scaler = StandardScaler().fit(cumulative[moving])
    progress_weight = np.sqrt(np.concatenate([OFFICIAL_W * np.float32(36.),
                                               np.ones(4, np.float32)])).astype(np.float32)
    velocity_features = progress_scaler.transform(cumulative[moving]) * progress_weight
    with threadpool_limits(limits=1):
        path_axis = stratified_medoids(
            path_features, moving_rows, route_labels(train_fut5[moving]), ROUTE_NAMES,
            P_COUNT, FIT_SEED + P_COUNT, standardize_within_stratum=True)
        velocity_axis = stratified_medoids(
            velocity_features, moving_rows, velocity_labels(step[moving], total[moving]),
            VELOCITY_NAMES, MOVING_V_COUNT, FIT_SEED + V_COUNT,
            standardize_within_stratum=False)
    moving_lookup = {int(row): index for index, row in enumerate(train_rows)}
    path_local = np.asarray([moving_lookup[int(row)] for row in path_axis.source_global], np.int64)
    velocity_local = np.asarray([moving_lookup[int(row)] for row in velocity_axis.source_global], np.int64)
    path_xy = np.ascontiguousarray(train_fut5[path_local], np.float32)
    path_cumulative = np.ascontiguousarray(cumulative[path_local], np.float32)
    path_total = np.ascontiguousarray(total[path_local], np.float32)
    moving_profiles = np.concatenate([
        total[velocity_local, None], cumulative[velocity_local] / total[velocity_local, None]], 1)
    velocity_profiles = np.concatenate([np.zeros((1, 11), np.float32),
                                        moving_profiles.astype(np.float32)], 0)
    bank, path_id, velocity_id = compose_bank(
        path_xy, path_cumulative, path_total, velocity_profiles)
    arrays = {
        "bank_xy": bank, "path_id": path_id, "velocity_id": velocity_id,
        "path_source_rows": path_axis.source_global.astype(np.int64),
        "path_xy": path_xy, "path_cumulative": path_cumulative, "path_total": path_total,
        "path_stratum": path_axis.strata.astype("U8"), "path_support": path_axis.support,
        "velocity_source_rows": velocity_axis.source_global.astype(np.int64),
        "velocity_profiles": velocity_profiles,
        "velocity_stratum": velocity_axis.strata.astype("U8"),
        "velocity_support": velocity_axis.support,
    }
    diagnostics = bank_diagnostics(bank, path_id, velocity_id)
    require(diagnostics["exact_zero_count"] >= 1,
            "bank must retain the dedicated exact-zero row")
    args.output_dir.mkdir(parents=True)
    bank_path = args.output_dir / "factorized_bank_P512_V128.npz"
    write_new_npz(bank_path, arrays)
    bank_receipt = {"path": bank_path.name, "sha256": file_sha(bank_path),
                    "bytes": bank_path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "status": "frozen_train203_factorized_bank_before_tune_gate",
        "scope": {
            "selected_fit_rows": "current train203 only", "train_rows": len(train_rows),
            "tune_rows_selected_or_used_for_fit": 0,
            "final136_rows_selected_fit_evaluated_or_analyzed": 0,
            "shared_npz_containers_physically_include_unselected_rows": True,
            "gpu_used": False, "neural_model_forward_or_training": False,
        },
        "inputs": inputs,
        "source": {"script_sha256": args.expected_script_sha256,
                   "train203_helper_sha256": EXPECTED_HELPER_SHA256},
        "train": {"scenes": 203, "rows": len(train_rows),
                  "rows_sha256": current.array_sha(train_rows, "<i8"),
                  "selected_identity_sha256": current.EXPECTED_SELECTED_IDENTITY_SHA256["train"],
                  "fut5_sha256": current.array_sha(train_fut5, "<f4"),
                  "moving_total_ge_0p25_rows": int(moving.sum())},
        "recipe": {
            "fixed_seed": FIT_SEED, "P": P_COUNT, "V": V_COUNT,
            "candidate_count": BANK_COUNT, "exact_stop_rows": 1,
            "composition": "C[p,v,t] = S[v] * q[p](r[v,t])",
            "published_order": "row0 exact stop; path-major p=0..511 then v=1..127",
            "path_features": "33 uniform-arc-length normalized xy samples",
            "path_strata": list(ROUTE_NAMES), "path_stratum_counts": path_axis.counts,
            "path_stratum_quotas": path_axis.quotas,
            "velocity_features": "StandardScaler(train cumulative physical progress) * sqrt([11,11,5,5,2,2,1,1,1,1])",
            "velocity_strata": list(VELOCITY_NAMES),
            "velocity_stratum_counts": velocity_axis.counts,
            "velocity_stratum_quotas": velocity_axis.quotas,
            "medoid_fit": {"MiniBatchKMeans": {"n_init": 3, "max_iter": 200,
                "max_no_improvement": 25, "reassignment_ratio": 0.0},
                "threadpool_limit": 1, "unique_real_train_rows": True},
            "no_P_or_V_sweep": True,
        },
        "axis_support": {
            "path_sum": int(path_axis.support.sum()),
            "velocity_sum": int(velocity_axis.support.sum()),
            "path_min_max": [int(path_axis.support.min()), int(path_axis.support.max())],
            "velocity_min_max": [int(velocity_axis.support.min()), int(velocity_axis.support.max())],
            "path_support_sha256": array_sha(path_axis.support, "<i8"),
            "velocity_support_sha256": array_sha(velocity_axis.support, "<i8"),
        },
        "bank_diagnostics": diagnostics,
        "artifact": bank_receipt,
        "arrays": array_contract(arrays),
        "environment": {"python": sys.version, "numpy": np.__version__,
                        "sklearn": __import__("sklearn").__version__,
                        "scipy": __import__("scipy").__version__,
                        "threads": {name: os.environ.get(name) for name in
                                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
    }
    manifest_path = args.output_dir / "bank_manifest.json"
    write_new_json(manifest_path, manifest)
    require(file_sha(bank_path) == bank_receipt["sha256"], "bank changed during publication")
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path.resolve()),
                      "manifest_sha256": file_sha(manifest_path),
                      "bank_sha256": bank_receipt["sha256"]}, sort_keys=True))
    return 0


def validate_manifest(manifest):
    require(manifest.get("schema_version") == 1
            and manifest.get("status") == "frozen_train203_factorized_bank_before_tune_gate",
            "bank manifest status mismatch")
    scope = manifest.get("scope", {})
    require(scope.get("selected_fit_rows") == "current train203 only"
            and scope.get("train_rows") == current.EXPECTED_TRAIN_ROWS
            and scope.get("tune_rows_selected_or_used_for_fit") == 0
            and scope.get("final136_rows_selected_fit_evaluated_or_analyzed") == 0
            and scope.get("gpu_used") is False
            and scope.get("neural_model_forward_or_training") is False,
            "bank fit scope mismatch")
    require(manifest.get("source") == {
        "script_sha256": manifest.get("source", {}).get("script_sha256"),
        "train203_helper_sha256": EXPECTED_HELPER_SHA256}, "bank source closure mismatch")
    require(valid_sha(manifest["source"]["script_sha256"]), "bank script SHA invalid")
    recipe = manifest.get("recipe", {})
    require(recipe.get("fixed_seed") == FIT_SEED and recipe.get("P") == P_COUNT
            and recipe.get("V") == V_COUNT and recipe.get("candidate_count") == BANK_COUNT
            and recipe.get("exact_stop_rows") == 1 and recipe.get("no_P_or_V_sweep") is True
            and recipe.get("published_order") ==
            "row0 exact stop; path-major p=0..511 then v=1..127",
            "bank fixed recipe mismatch")
    train = manifest.get("train", {})
    require(train.get("scenes") == 203 and train.get("rows") == current.EXPECTED_TRAIN_ROWS
            and train.get("rows_sha256") == current.EXPECTED_TRAIN_ROWS_SHA256
            and train.get("selected_identity_sha256") ==
            current.EXPECTED_SELECTED_IDENTITY_SHA256["train"], "bank train identity mismatch")
    require(manifest.get("bank_diagnostics", {}).get("exact_zero_count", 0) >= 1
            and isinstance(manifest["bank_diagnostics"].get("duplicate_count"), int)
            and manifest["bank_diagnostics"].get("duplicate_count", -1) >= 0,
            "bank diagnostics mismatch")
    require(valid_sha(manifest.get("artifact", {}).get("sha256"))
            and manifest["artifact"].get("path") == "factorized_bank_P512_V128.npz",
            "bank artifact receipt mismatch")
    inputs = manifest.get("inputs", {})
    require(inputs.get("split", {}).get("sha256") == current.EXPECTED_SPLIT_SHA256
            and inputs.get("ego", {}).get("sha256") == current.EXPECTED_EGO_SHA256
            and inputs.get("ego5", {}).get("sha256") == current.EXPECTED_EGO5_SHA256,
            "bank input provenance mismatch")
    arrays = manifest.get("arrays", {})
    require(set(arrays) == set(EXPECTED_ARRAY_SCHEMA), "bank manifest array keyset mismatch")
    for name, (shape, dtype) in EXPECTED_ARRAY_SCHEMA.items():
        require(arrays[name].get("shape") == shape and arrays[name].get("dtype") == dtype
                and valid_sha(arrays[name].get("sha256")),
                f"bank manifest array contract mismatch: {name}")


def load_bank(manifest_path, expected_manifest_sha):
    manifest_path = Path(manifest_path)
    require(file_sha(manifest_path) == expected_manifest_sha, "bank manifest SHA mismatch")
    manifest = read_json(manifest_path, expected_manifest_sha)
    validate_manifest(manifest)
    require(manifest["source"]["script_sha256"] == file_sha(__file__),
            "current gate script differs from fit script")
    bank_path = manifest_path.parent / manifest["artifact"]["path"]
    require(bank_path.is_file() and not bank_path.is_symlink()
            and file_sha(bank_path) == manifest["artifact"]["sha256"],
            "frozen bank artifact SHA mismatch")
    with np.load(bank_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    require(set(arrays) == set(manifest["arrays"]), "bank array keyset mismatch")
    for name, array in arrays.items():
        contract = manifest["arrays"][name]
        require(list(array.shape) == contract["shape"] and str(array.dtype) == contract["dtype"]
                and array_sha(array) == contract["sha256"], f"bank array mismatch: {name}")
    diagnostics = bank_diagnostics(arrays["bank_xy"], arrays["path_id"], arrays["velocity_id"])
    require(diagnostics == manifest["bank_diagnostics"], "bank diagnostics changed")
    return manifest, arrays, bank_path


def exact_oracle(gt, bank, candidate_chunk=ORACLE_CANDIDATE_CHUNK):
    gt = np.asarray(gt, np.float32)
    bank = np.asarray(bank, np.float32)
    require(gt.ndim == 3 and gt.shape[1:] == (10, 2)
            and bank.ndim == 3 and bank.shape[1:] == (10, 2), "oracle shapes invalid")
    best = {name: np.full(len(gt), np.inf, np.float32)
            for name in ("d3", "endpoint5", "tail_3p5_to_5", "mean10")}
    selected = np.full(len(gt), -1, np.int32)
    for start in range(0, len(bank), candidate_chunk):
        candidate = bank[start:start + candidate_chunk]
        distance = np.linalg.norm(gt[:, None] - candidate[None], axis=-1)
        values = {
            "d3": np.sum(distance[:, :, :6] * OFFICIAL_W[None, None],
                         axis=2, dtype=np.float32),
            "endpoint5": distance[:, :, 9],
            "tail_3p5_to_5": np.mean(distance[:, :, 6:10], axis=2, dtype=np.float32),
            "mean10": np.mean(distance, axis=2, dtype=np.float32),
        }
        d3 = values["d3"]
        local = np.argmin(d3, axis=1)
        value = d3[np.arange(len(gt)), local]
        better = value < best["d3"]
        selected[better] = (start + local[better]).astype(np.int32)
        for name, matrix in values.items():
            best[name] = np.minimum(best[name], np.min(matrix, axis=1))
    require((selected >= 0).all()
            and all(np.isfinite(value).all() for value in best.values()),
            "oracle result invalid")
    return selected, best


def gate(args):
    verify_runtime_and_source(args.expected_script_sha256)
    require(not args.output_dir.exists() and not args.output_dir.is_symlink(),
            "gate output directory must be fresh")
    bank_manifest, arrays, bank_path = load_bank(
        args.bank_manifest, args.expected_bank_manifest_sha256)
    bank_before = file_sha(bank_path)
    split, ego, ego5, names, inputs = load_current_inputs(
        args.split_manifest, args.ego_cache, args.ego5_cache)
    train_rows = current.selected_rows(split, names, ego["frame"], "train")
    train_fut5, _ = current.selected_trajectories(ego, ego5, train_rows, split="train")
    train_lookup = {int(row): index for index, row in enumerate(train_rows)}
    path_local = np.asarray([train_lookup[int(row)] for row in arrays["path_source_rows"]])
    velocity_local = np.asarray([train_lookup[int(row)] for row in arrays["velocity_source_rows"]])
    require(np.array_equal(arrays["path_xy"], train_fut5[path_local]),
            "path medoids are not intact current-train rows")
    step, cumulative, total = cumulative_geometry(train_fut5)
    require(np.array_equal(arrays["path_cumulative"], cumulative[path_local])
            and np.array_equal(arrays["path_total"], total[path_local]),
            "stored path physical-progress primitives differ from train rows")
    expected_profiles = np.concatenate([
        total[velocity_local, None], cumulative[velocity_local] / total[velocity_local, None]], 1)
    require(np.array_equal(arrays["velocity_profiles"][1:], expected_profiles.astype(np.float32)),
            "velocity medoids are not intact current-train profiles")
    require(np.array_equal(arrays["path_stratum"], route_labels(train_fut5[path_local]))
            and np.array_equal(arrays["velocity_stratum"],
                               velocity_labels(step[velocity_local], total[velocity_local])),
            "stored factor-axis strata differ from source train rows")
    reconstructed, reconstructed_path_id, reconstructed_velocity_id = compose_bank(
        arrays["path_xy"], arrays["path_cumulative"], arrays["path_total"],
        arrays["velocity_profiles"])
    require(np.array_equal(reconstructed, arrays["bank_xy"])
            and np.array_equal(reconstructed_path_id, arrays["path_id"])
            and np.array_equal(reconstructed_velocity_id, arrays["velocity_id"]),
            "stored bank differs from its frozen physical-progress primitives")
    # Tune labels are selected only after the externally SHA-pinned bank is loaded and validated.
    tune_rows = current.selected_rows(split, names, ego["frame"], "tune")
    tune_fut5, _ = current.selected_trajectories(ego, ego5, tune_rows, split="tune")
    selected, oracle = exact_oracle(tune_fut5, arrays["bank_xy"])
    d3 = oracle["d3"]
    selected_path = arrays["path_id"][selected]
    selected_velocity = arrays["velocity_id"][selected]
    selected_path_source = np.where(selected == 0, -1,
                                    arrays["path_source_rows"][selected_path]).astype(np.int64)
    selected_velocity_source = np.where(selected == 0, -1,
        arrays["velocity_source_rows"][np.maximum(selected_velocity - 1, 0)]).astype(np.int64)
    output_arrays = {
        "tune_rows": tune_rows.astype(np.int64), "selected_candidate_id": selected,
        "oracle_d3": d3, "oracle_endpoint5": oracle["endpoint5"],
        "oracle_tail_3p5_to_5": oracle["tail_3p5_to_5"],
        "oracle_mean10": oracle["mean10"], "selected_path_id": selected_path,
        "selected_velocity_id": selected_velocity,
        "selected_path_source_row": selected_path_source,
        "selected_velocity_source_row": selected_velocity_source,
    }
    gate_value = float(np.mean(d3, dtype=np.float64))
    passed = gate_value <= GATE_MAX_D3
    args.output_dir.mkdir(parents=True)
    oracle_path = args.output_dir / "tune_oracle.npz"
    write_new_npz(oracle_path, output_arrays)
    oracle_receipt = {"path": oracle_path.name, "sha256": file_sha(oracle_path),
                      "bytes": oracle_path.stat().st_size,
                      "arrays": array_contract(output_arrays)}
    pair = np.stack([selected_path_source, selected_velocity_source], axis=1)
    report = {
        "schema_version": 1,
        "status": "passed_fixed_tune_oracle_gate" if passed else "failed_fixed_tune_oracle_gate",
        "inputs": {"bank_manifest": {"path": str(Path(args.bank_manifest).resolve()),
            "sha256": args.expected_bank_manifest_sha256},
            "bank": {"path": str(bank_path.resolve()), "sha256": bank_before}, **inputs},
        "source": {"script_sha256": args.expected_script_sha256,
                   "train203_helper_sha256": EXPECTED_HELPER_SHA256},
        "bank": {"P": P_COUNT, "V": V_COUNT, "candidate_count": BANK_COUNT,
                 "diagnostics": bank_manifest["bank_diagnostics"],
                 "axis_support": bank_manifest["axis_support"]},
        "tune": {"scenes": 37, "rows": len(tune_rows),
                 "rows_sha256": current.array_sha(tune_rows, "<i8"),
                 "selected_identity_sha256": current.EXPECTED_SELECTED_IDENTITY_SHA256["tune"],
                 "oracle_weighted_d3": gate_value,
                 "metricwise_oracle_endpoint5": float(np.mean(oracle["endpoint5"], dtype=np.float64)),
                 "metricwise_oracle_tail_3p5_to_5": float(np.mean(
                     oracle["tail_3p5_to_5"], dtype=np.float64)),
                 "metricwise_oracle_mean10": float(np.mean(oracle["mean10"], dtype=np.float64)),
                 "gate_max_inclusive": GATE_MAX_D3, "gate_pass": passed,
                 "finite_count": int(np.isfinite(d3).sum()),
                 "exact_stop_selected_count": int(np.sum(selected == 0)),
                 "selected_candidate_id_sha256": array_sha(selected, "<i4"),
                 "selected_axis_source_pair_sha256": array_sha(pair, "<i8"),
                 "oracle_d3_sha256": array_sha(d3, "<f4"),
                 "metricwise_oracle_endpoint5_sha256": array_sha(oracle["endpoint5"], "<f4"),
                 "metricwise_oracle_tail_3p5_to_5_sha256": array_sha(
                     oracle["tail_3p5_to_5"], "<f4"),
                 "metricwise_oracle_mean10_sha256": array_sha(oracle["mean10"], "<f4")},
        "artifact": oracle_receipt,
        "boundaries": {"bank_frozen_before_tune_selection": True,
            "no_P_or_V_sweep_or_refit": True, "representation_oracle_not_realized_selector": True,
            "train_nearest_ids_not_computed": True, "gpu_used": False,
            "neural_model_forward_or_training": False,
            "final136_rows_selected_trained_evaluated_or_analyzed": 0,
            "shared_npz_containers_physically_include_unselected_rows": True},
    }
    report_path = args.output_dir / "gate_report.json"
    write_new_json(report_path, report)
    require(file_sha(bank_path) == bank_before, "frozen bank changed during gate")
    print(json.dumps({"status": report["status"], "oracle_d3": gate_value,
                      "gate_pass": passed, "report": str(report_path.resolve()),
                      "report_sha256": file_sha(report_path)}, sort_keys=True))
    return 0 if passed else 2


def arguments(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("fit", "gate"):
        target = subparsers.add_parser(phase)
        target.add_argument("--split-manifest", type=Path, required=True)
        target.add_argument("--ego-cache", type=Path, required=True)
        target.add_argument("--ego5-cache", type=Path, required=True)
        target.add_argument("--expected-script-sha256", required=True)
        target.add_argument("--output-dir", type=Path, required=True)
    gate_parser = subparsers.choices["gate"]
    gate_parser.add_argument("--bank-manifest", type=Path, required=True)
    gate_parser.add_argument("--expected-bank-manifest-sha256", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    return fit(args) if args.phase == "fit" else gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
