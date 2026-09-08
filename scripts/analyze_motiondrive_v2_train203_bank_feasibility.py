#!/usr/bin/env python3
"""CPU-only current-train203 K=1024 trajectory-bank feasibility audit.

``fit`` builds exactly one bank from current train rows and freezes it before
any tune label or prediction is opened. ``evaluate`` consumes that immutable
bank and reports coverage plus two fixed M=12 offline proxy selectors.  Oracle
numbers are representation bounds, never realized model/selector performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Mapping

# Reproducibility/resource boundary must be set before NumPy/sklearn import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_grouped_split_v2 import validate_manifest


EXPECTED_SPLIT_SHA256 = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
EXPECTED_EGO_SHA256 = "d35a69bb229b2d4736aff7dddefd3522ddbd346e112283e2ce58ce3986a023bd"
EXPECTED_EGO5_SHA256 = "700d9443cbf38aaaeffbbd5dab789b5e448adf931e15939baa9d37779e4095c1"
EXPECTED_EGO5_BYTES = 8370376
EXPECTED_TRAIN_ROWS = 54810
EXPECTED_TUNE_ROWS = 1998
EXPECTED_TRAIN_ROWS_SHA256 = "75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854"
EXPECTED_TUNE_ROWS_SHA256 = "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88"
EXPECTED_SELECTED_IDENTITY_SHA256 = {
    "train": "07ea3dcb51f89af25824ffcb12a4e2dc4effb8612ff05134e2a312fd68966341",
    "tune": "e8a6bbecdef3a3c7910ad84e8ea8bc06b17f64690b38fbc325f7bf708a9c520c",
}
OFFICIAL_W = np.asarray([11, 11, 5, 5, 2, 2], dtype=np.float32) / np.float32(36.)
K_BANK = 1024
K_CLUSTER = K_BANK - 1
SHORTLIST_M = 12
FIT_SEED = 20260908
EXPECTED_FIT_RUNTIME = {"python": "3.12.3", "numpy": "2.1.0",
                        "scipy": "1.16.3", "sklearn": "1.8.0"}
KMEANS_SETTINGS = {
    "n_clusters": K_CLUSTER,
    "init": "k-means++",
    "n_init": 3,
    "max_iter": 200,
    "batch_size": 4096,
    "max_no_improvement": 25,
    "reassignment_ratio": 0.0,
    "tol": 0.0,
    "random_state": FIT_SEED,
    "compute_labels": True,
    "verbose": 0,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def array_sha(value, dtype=None):
    array = np.asarray(value, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path, expected_sha=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"ordinary JSON required: {path}")
    raw = path.read_bytes()
    if expected_sha is not None:
        require(valid_sha(expected_sha) and hashlib.sha256(raw).hexdigest() == expected_sha,
                f"JSON SHA mismatch: {path}")

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    def unique_keys(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, f"duplicate JSON key: {key}")
            output[key] = value
        return output

    value = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def native(value):
    if value is None or type(value) in (bool, int, float, str):
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
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def write_new_json(path, value):
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(native(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    # Full round trip before publication.
    json.loads(payload)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_new_npy(path, value):
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def verify_input(path, expected_sha):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"ordinary input required: {path}")
    require(valid_sha(expected_sha), f"authoritative SHA unavailable: {path}")
    actual = file_sha(path)
    require(actual == expected_sha, f"input SHA mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}


def load_split(path):
    receipt = verify_input(path, EXPECTED_SPLIT_SHA256)
    manifest = read_json(path, EXPECTED_SPLIT_SHA256)
    validate_manifest(manifest)
    require(len(manifest["splits"]["train"]) == 203
            and len(manifest["splits"]["tune"]) == 37,
            "current grouped train203/tune37 split required")
    require(not (set(manifest["splits"]["train"]) & set(manifest["splits"]["tune"])),
            "train/tune scene leakage")
    return manifest, receipt


def load_cache_metadata(ego_path, ego5_path):
    ego_receipt = verify_input(ego_path, EXPECTED_EGO_SHA256)
    ego5_receipt = verify_input(ego5_path, EXPECTED_EGO5_SHA256)
    with np.load(ego_path, allow_pickle=False) as source:
        required = {"scenarios", "scen_idx", "frame", "goal", "fut"}
        require(required <= set(source.files), "current ego cache schema incomplete")
        ego = {key: source[key] for key in required}
    with np.load(ego5_path, allow_pickle=False) as source:
        required = {"scenarios", "scen_idx", "frame", "fut5", "mask5"}
        require(required <= set(source.files), "5s ego cache schema incomplete")
        ego5 = {key: source[key] for key in required}
    n = len(ego["frame"])
    require(ego["scen_idx"].shape == (n,) and ego["frame"].shape == (n,)
            and ego["goal"].shape == (n, 2) and ego["fut"].shape == (n, 6, 2),
            "current ego cache shapes invalid")
    require(n == 112800 and ego5_receipt["bytes"] == EXPECTED_EGO5_BYTES,
            "ego5 row/file-size contract mismatch")
    require(ego5["scenarios"].shape == (376,)
            and ego5["scen_idx"].shape == (n,) and ego5["frame"].shape == (n,)
            and ego5["fut5"].shape == (n, 10, 2) and ego5["mask5"].shape == (n, 10),
            "5s ego cache shapes invalid")
    require(ego5["scen_idx"].dtype == np.dtype("int32")
            and ego5["frame"].dtype == np.dtype("int16")
            and ego5["fut5"].dtype == np.dtype("float32")
            and ego5["mask5"].dtype == np.dtype("float32"),
            "5s ego cache dtypes invalid")
    # Only identity arrays are compared globally. Labels are indexed only after
    # the explicit current train or tune mask is formed.
    require(np.array_equal(ego["scenarios"].astype(str), ego5["scenarios"].astype(str))
            and np.array_equal(ego["scen_idx"], ego5["scen_idx"])
            and np.array_equal(ego["frame"], ego5["frame"]),
            "ego/ego5 global row identity mismatch")
    scenarios = ego["scenarios"].astype(str)
    indices = np.asarray(ego["scen_idx"], dtype=np.int64)
    require(indices.min(initial=0) >= 0 and indices.max(initial=-1) < len(scenarios),
            "scenario index outside name table")
    names = scenarios[indices]
    require(len({(str(scene), int(frame)) for scene, frame in zip(names, ego["frame"])}) == n,
            "ego identities are not unique")
    return ego, ego5, names, {"ego": ego_receipt, "ego5": ego5_receipt}


def selected_rows(manifest, names, frames, split):
    require(split in ("train", "tune"), "only train/tune may be selected")
    stride = 1 if split == "train" else 5
    mask = (np.isin(names, manifest["splits"][split])
            & (np.asarray(frames) >= 30) & (np.asarray(frames) % stride == 0))
    rows = np.flatnonzero(mask).astype(np.int64)
    expected_n = EXPECTED_TRAIN_ROWS if split == "train" else EXPECTED_TUNE_ROWS
    expected_sha = EXPECTED_TRAIN_ROWS_SHA256 if split == "train" else EXPECTED_TUNE_ROWS_SHA256
    require(len(rows) == expected_n and array_sha(rows, "<i8") == expected_sha,
            f"current {split} row inventory mismatch")
    return rows


def selected_trajectories(ego, ego5, rows, *, split):
    fut = np.ascontiguousarray(ego["fut"][rows], dtype=np.float32)
    goal = np.ascontiguousarray(ego["goal"][rows], dtype=np.float32)
    fut5 = np.ascontiguousarray(ego5["fut5"][rows], dtype=np.float32)
    mask5 = np.asarray(ego5["mask5"][rows])
    require(fut.shape == (len(rows), 6, 2) and fut5.shape == (len(rows), 10, 2),
            f"{split} trajectory shapes invalid")
    require(np.isfinite(fut).all() and np.isfinite(goal).all() and np.isfinite(fut5).all(),
            f"{split} trajectories nonfinite")
    require(mask5.shape == (len(rows), 10)
            and np.isin(mask5, [0, 1]).all() and bool(mask5.astype(bool).all()),
            f"{split} requires all ten 5s labels valid")
    require(np.array_equal(fut5[:, :6], fut), f"{split} ego5 prefix differs from current GT")
    require(np.array_equal(fut5[:, 9], goal), f"{split} ego5 endpoint differs from current goal")
    digest = hashlib.sha256()
    scenarios = ego["scenarios"].astype(str)[np.asarray(ego["scen_idx"], np.int64)]
    frames = np.asarray(ego["frame"])
    for local, row in enumerate(np.asarray(rows, np.int64)):
        digest.update(str(scenarios[row]).encode("utf-8")); digest.update(b"\0")
        digest.update(np.asarray(row, dtype="<i8").tobytes())
        digest.update(np.asarray(frames[row], dtype="<i8").tobytes())
        digest.update(np.asarray(fut5[local], dtype="<f4").tobytes())
        digest.update(np.asarray(goal[local], dtype="<f4").tobytes())
    require(digest.hexdigest() == EXPECTED_SELECTED_IDENTITY_SHA256[split],
            f"{split} selected ego/ego5 identity digest mismatch")
    return fut5, mask5.astype(bool, copy=False)


def prefix_embedding(trajectories):
    value = np.asarray(trajectories, dtype=np.float32)
    require(value.ndim == 3 and value.shape[1:] == (10, 2), "expected [N,10,2]")
    # Weighted squared Euclidean K-means surrogate; it is not exact D3 because
    # official D3 sums per-time Euclidean norms rather than squared coordinates.
    return np.ascontiguousarray(
        value[:, :6] * np.sqrt(OFFICIAL_W).astype(np.float32)[None, :, None]).reshape(len(value), 12)


def unique_nonzero_pool(trajectories, rows):
    trajectories = np.ascontiguousarray(trajectories, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.int64)
    require(len(trajectories) == len(rows), "trajectory/source row mismatch")
    pool = {}
    for index in np.argsort(rows, kind="stable"):
        trajectory = trajectories[index]
        if np.array_equal(trajectory, np.zeros((10, 2), np.float32)):
            continue
        key = trajectory.tobytes()
        pool.setdefault(key, (int(rows[index]), trajectory.copy()))
    ordered = sorted(pool.values(), key=lambda item: item[0])
    return (np.asarray([item[0] for item in ordered], dtype=np.int64),
            np.stack([item[1] for item in ordered]).astype(np.float32, copy=False))


def assign_unique_medoids(centers, candidate_features, candidate_rows):
    centers = np.asarray(centers, dtype=np.float64)
    features = np.asarray(candidate_features, dtype=np.float64)
    rows = np.asarray(candidate_rows, dtype=np.int64)
    require(centers.ndim == 2 and features.ndim == 2
            and centers.shape[1] == features.shape[1] and len(features) == len(rows),
            "medoid assignment shape mismatch")
    available = np.ones(len(features), dtype=bool)
    selected = []
    for center in centers:  # MiniBatchKMeans center index is the fixed primary order.
        distance = np.sum((features - center[None]) ** 2, axis=1)
        candidates = np.flatnonzero(available)
        require(len(candidates) > 0, "unique medoid pool exhausted")
        # Deterministic tie: distance, then canonical source row.
        order = np.lexsort((rows[candidates], distance[candidates]))
        choice = int(candidates[int(order[0])])
        selected.append(choice)
        available[choice] = False
    return np.asarray(selected, dtype=np.int64)


def build_bank(train_fut5, train_rows):
    all_embedding = prefix_embedding(train_fut5)
    nonzero = ~np.all(train_fut5 == 0, axis=(1, 2))
    require(int(nonzero.sum()) >= K_CLUSTER, "not enough nonzero train samples")
    candidate_rows, candidate_trajectories = unique_nonzero_pool(train_fut5, train_rows)
    require(len(candidate_rows) >= K_CLUSTER,
            "fewer than required unique nonzero full10 train trajectories")
    candidate_embedding = prefix_embedding(candidate_trajectories)
    model = MiniBatchKMeans(**KMEANS_SETTINGS)
    with threadpool_limits(limits=1):
        labels = model.fit_predict(all_embedding[nonzero])
    require(model.cluster_centers_.shape == (K_CLUSTER, 12), "K-means center shape mismatch")
    support = np.bincount(labels, minlength=K_CLUSTER).astype(np.int64)
    chosen = assign_unique_medoids(model.cluster_centers_, candidate_embedding, candidate_rows)
    selected_rows = candidate_rows[chosen]
    selected_trajectories = candidate_trajectories[chosen]
    cluster_ids = np.arange(K_CLUSTER, dtype=np.int32)
    # Stable published order, independent of sklearn internal center ordering.
    order = np.lexsort((cluster_ids, selected_rows, -support))
    selected_rows = selected_rows[order]
    selected_trajectories = selected_trajectories[order]
    cluster_ids = cluster_ids[order]
    support = support[order]
    bank = np.concatenate([np.zeros((1, 10, 2), np.float32), selected_trajectories], axis=0)
    sources = np.concatenate([np.asarray([-1], np.int64), selected_rows])
    clusters = np.concatenate([np.asarray([-1], np.int32), cluster_ids])
    supports = np.concatenate([np.asarray([0], np.int64), support])
    validate_bank(bank, sources)
    return bank, sources, clusters, supports, model


def validate_bank(bank, source_rows):
    bank = np.asarray(bank)
    source_rows = np.asarray(source_rows)
    require(bank.shape == (K_BANK, 10, 2) and bank.dtype == np.float32
            and np.isfinite(bank).all(), "bank tensor invalid")
    require(source_rows.shape == (K_BANK,) and source_rows.dtype == np.int64
            and source_rows[0] == -1 and len(np.unique(source_rows[1:])) == K_CLUSTER,
            "bank source rows invalid")
    require(np.array_equal(bank[0], np.zeros((10, 2), np.float32)), "bank row0 not exact zero")
    encoded = [row.tobytes() for row in np.ascontiguousarray(bank)]
    require(len(set(encoded)) == K_BANK, "bank full10 trajectories are not unique")


def validate_bank_sources(bank, source_rows, train_rows, ego5_fut5):
    source_rows = np.asarray(source_rows, np.int64)
    train_rows = np.asarray(train_rows, np.int64)
    require(source_rows.shape == (K_BANK,) and np.isin(source_rows[1:], train_rows).all(),
            "bank source outside current train203 rows")
    require(np.array_equal(np.asarray(bank, np.float32)[1:],
                           np.asarray(ego5_fut5[source_rows[1:]], np.float32)),
            "bank rows are not intact current-train full10 trajectories")


def fit(args):
    require_runtime()
    require(args.output_dir.name not in ("", ".", "..") and not args.output_dir.exists(),
            "fit output directory must be a new explicit path")
    manifest, split_receipt = load_split(args.split_manifest)
    ego, ego5, names, cache_receipts = load_cache_metadata(args.ego_cache, args.ego5_cache)
    rows = selected_rows(manifest, names, ego["frame"], "train")
    train_fut5, _ = selected_trajectories(ego, ego5, rows, split="train")
    bank, sources, clusters, supports, model = build_bank(train_fut5, rows)
    args.output_dir.mkdir(parents=True)
    paths = {name: args.output_dir / name for name in
             ("bank.npy", "source_rows.npy", "cluster_ids.npy", "cluster_support.npy")}
    for path, value in zip(paths.values(), (bank, sources, clusters, supports)):
        write_new_npy(path, value)
    artifact = {name: {"path": path.name, "sha256": file_sha(path), "bytes": path.stat().st_size}
                for name, path in paths.items()}
    result = {
        "schema_version": 1,
        "status": "bank_frozen_before_any_tune_label_or_prediction_access",
        "scope": {"train_rows_only": True, "tune_labels_accessed": False,
                  "final_rows_accessed": False, "gpu_used": False,
                  "shared_ego5_bytes_include_unselected_rows": True,
                  "unselected_rows_used_for_fit_statistics_metrics_or_selection": False},
        "inputs": {"split": split_receipt, **cache_receipts},
        "train": {"scenes": 203, "rows": len(rows),
                  "rows_sha256": array_sha(rows, "<i8"),
                  "selected_identity_sha256": EXPECTED_SELECTED_IDENTITY_SHA256["train"],
                  "fut5_sha256": array_sha(train_fut5, "<f4")},
        "recipe": {"bank_rows": K_BANK, "reserved_exact_zero_rows": 1,
                   "cluster_rows": K_CLUSTER, "seed": FIT_SEED,
                   "embedding": "flatten(sqrt([11,11,5,5,2,2]/36)*metric_xy_first6)",
                   "embedding_dtype": "float32", "trajectory_dtype": "float32",
                   "objective_boundary": "weighted squared K-means surrogate, not exact official D3",
                   "tail_used_in_fit": False, "scaler": None,
                   "kmeans": dict(KMEANS_SETTINGS), "threadpool_limit": 1,
                   "medoid": "unique full10 real train tensor; nearest squared embedding; ties source row",
                   "published_order": "support descending, source row ascending, original cluster id ascending"},
        "fit": {"n_iter": int(model.n_iter_), "inertia": float(model.inertia_),
                "unique_nonzero_bank_rows": K_CLUSTER},
        "artifacts": artifact,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "sklearn": __import__("sklearn").__version__,
                        "scipy": __import__("scipy").__version__,
                        "expected_fit_runtime": dict(EXPECTED_FIT_RUNTIME),
                        "threads": {name: os.environ.get(name) for name in
                                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                        "script_sha256": file_sha(__file__)},
    }
    manifest_path = args.output_dir / "bank_manifest.json"
    write_new_json(manifest_path, result)
    # All fit inputs/artifacts remain byte-identical through publication.
    for receipt in (split_receipt, *cache_receipts.values()):
        require(file_sha(receipt["path"]) == receipt["sha256"], "fit input changed")
    for name, receipt in artifact.items():
        require(file_sha(args.output_dir / receipt["path"]) == receipt["sha256"],
                f"fit artifact changed: {name}")
    print(json.dumps({"status": result["status"], "manifest": str(manifest_path.resolve()),
                      "manifest_sha256": file_sha(manifest_path), "gpu_used": False}, sort_keys=True))
    return 0


def validate_frozen_manifest_contract(manifest):
    require(set(manifest) == {"schema_version", "status", "scope", "inputs", "train",
                              "recipe", "fit", "artifacts", "environment"}
            and manifest.get("schema_version") == 1,
            "frozen bank manifest schema mismatch")
    require(manifest.get("status") == "bank_frozen_before_any_tune_label_or_prediction_access"
            and manifest.get("scope") == {
                "train_rows_only": True, "tune_labels_accessed": False,
                "final_rows_accessed": False, "gpu_used": False,
                "shared_ego5_bytes_include_unselected_rows": True,
                "unselected_rows_used_for_fit_statistics_metrics_or_selection": False},
            "bank was not frozen under the train-only scope")
    inputs = manifest.get("inputs", {})
    require(set(inputs) == {"split", "ego", "ego5"}
            and inputs["split"].get("sha256") == EXPECTED_SPLIT_SHA256
            and inputs["ego"].get("sha256") == EXPECTED_EGO_SHA256
            and inputs["ego5"].get("sha256") == EXPECTED_EGO5_SHA256
            and inputs["ego5"].get("bytes") == EXPECTED_EGO5_BYTES,
            "frozen bank input identity mismatch")
    train = manifest.get("train", {})
    require(train.get("scenes") == 203 and train.get("rows") == EXPECTED_TRAIN_ROWS
            and train.get("rows_sha256") == EXPECTED_TRAIN_ROWS_SHA256
            and train.get("selected_identity_sha256") == EXPECTED_SELECTED_IDENTITY_SHA256["train"]
            and valid_sha(train.get("fut5_sha256")),
            "frozen bank train inventory mismatch")
    expected_recipe = {
        "bank_rows": K_BANK, "reserved_exact_zero_rows": 1,
        "cluster_rows": K_CLUSTER, "seed": FIT_SEED,
        "embedding": "flatten(sqrt([11,11,5,5,2,2]/36)*metric_xy_first6)",
        "embedding_dtype": "float32", "trajectory_dtype": "float32",
        "objective_boundary": "weighted squared K-means surrogate, not exact official D3",
        "tail_used_in_fit": False, "scaler": None,
        "kmeans": dict(KMEANS_SETTINGS), "threadpool_limit": 1,
        "medoid": "unique full10 real train tensor; nearest squared embedding; ties source row",
        "published_order": "support descending, source row ascending, original cluster id ascending"}
    require(manifest.get("recipe") == expected_recipe, "frozen bank recipe mismatch")
    environment = manifest.get("environment", {})
    require(environment.get("expected_fit_runtime") == EXPECTED_FIT_RUNTIME
            and str(environment.get("python", "")).split()[0] == EXPECTED_FIT_RUNTIME["python"]
            and environment.get("numpy") == EXPECTED_FIT_RUNTIME["numpy"]
            and environment.get("scipy") == EXPECTED_FIT_RUNTIME["scipy"]
            and environment.get("sklearn") == EXPECTED_FIT_RUNTIME["sklearn"]
            and environment.get("threads") == {
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            "frozen bank runtime identity mismatch")
    require(set(manifest.get("artifacts", {})) == {
                "bank.npy", "source_rows.npy", "cluster_ids.npy", "cluster_support.npy"}
            and all(spec.get("path") == name and valid_sha(spec.get("sha256"))
                    for name, spec in manifest["artifacts"].items()),
            "frozen bank artifact map mismatch")


def load_frozen_bank(path, expected_sha):
    receipt = verify_input(path, expected_sha)
    manifest = read_json(path, expected_sha)
    validate_frozen_manifest_contract(manifest)
    directory = Path(path).parent
    arrays = {}
    for name, dtype in (("bank.npy", np.float32), ("source_rows.npy", np.int64),
                        ("cluster_ids.npy", np.int32), ("cluster_support.npy", np.int64)):
        spec = manifest.get("artifacts", {}).get(name)
        require(isinstance(spec, dict) and spec.get("path") == name
                and valid_sha(spec.get("sha256")), f"bank artifact receipt missing: {name}")
        verify_input(directory / name, spec["sha256"])
        arrays[name] = np.load(directory / name, allow_pickle=False)
        require(arrays[name].dtype == dtype, f"bank artifact dtype mismatch: {name}")
    validate_bank(arrays["bank.npy"], arrays["source_rows.npy"])
    require(all(value.shape == (K_BANK,) for name, value in arrays.items() if name != "bank.npy")
            and arrays["cluster_ids.npy"][0] == -1
            and set(arrays["cluster_ids.npy"][1:].tolist()) == set(range(K_CLUSTER))
            and arrays["cluster_support.npy"][0] == 0
            and bool((arrays["cluster_support.npy"][1:] > 0).all()),
            "bank cluster provenance arrays invalid")
    expected_order = sorted(range(1, K_BANK), key=lambda index: (
        -int(arrays["cluster_support.npy"][index]), int(arrays["source_rows.npy"][index]),
        int(arrays["cluster_ids.npy"][index])))
    require(expected_order == list(range(1, K_BANK)), "published bank ordering mismatch")
    artifact_receipts = {
        name: {"path": str((directory / name).resolve()), "sha256": spec["sha256"]}
        for name, spec in manifest["artifacts"].items()}
    return manifest, arrays, receipt, artifact_receipts


def per_metric_oracles(gt, bank, chunk=128):
    gt = np.asarray(gt, np.float32)
    bank = np.asarray(bank, np.float32)
    result = {name: np.full(len(gt), np.inf, np.float64)
              for name in ("d3", "endpoint5", "mean10", "tail_3p5_to_5")}
    for start in range(0, len(bank), chunk):
        candidate = bank[start:start + chunk]
        distance = np.linalg.norm(gt[:, None] - candidate[None], axis=-1)
        values = {
            "d3": official_d3(gt[:, None, :6], candidate[None, :, :6]),
            "endpoint5": distance[:, :, 9],
            "mean10": distance.mean(-1),
            "tail_3p5_to_5": distance[:, :, 6:10].mean(-1),
        }
        for name, value in values.items():
            result[name] = np.minimum(result[name], value.min(1))
    return result


def summarize_oracle(vectors):
    return {"means": {key: float(value.mean()) for key, value in vectors.items()},
            "coverage": {"d3_le_0p15": float((vectors["d3"] <= .15).mean()),
                         "endpoint5_le_1m": float((vectors["endpoint5"] <= 1.).mean()),
                         "mean10_le_0p5m": float((vectors["mean10"] <= .5).mean()),
                         "tail_le_0p5m": float((vectors["tail_3p5_to_5"] <= .5).mean())}}


def official_d3(pred, target):
    pred, target = np.asarray(pred, np.float32), np.asarray(target, np.float32)
    require(pred.shape[-2:] == (6, 2) and target.shape[-2:] == (6, 2), "D3 shape mismatch")
    pred, target = np.broadcast_arrays(pred, target)
    # Use the same FP32 PyTorch operation order as the frozen evaluator rather
    # than a numerically close NumPy norm/reduction.
    import torch
    difference = torch.from_numpy(np.ascontiguousarray(pred - target))
    distance = torch.linalg.vector_norm(difference.float(), dim=-1)
    result = (distance * distance.new_tensor(OFFICIAL_W.tolist())).sum(-1)
    return result.numpy()


def stable_shortlist(pred, bank, count=SHORTLIST_M):
    distance = official_d3(pred[:, None], bank[None, :, :6])
    ids = np.arange(len(bank), dtype=np.int64)
    output = np.empty((len(pred), count), np.int64)
    for index, row in enumerate(distance):
        output[index] = np.lexsort((ids, row))[:count]
    return output, distance


def proxy_selectors(pred, gt, goal, bank):
    shortlist, prefix_distance = stable_shortlist(pred, bank)
    batch = np.arange(len(pred))[:, None]
    candidates = bank[shortlist]
    # Prefix-nearest is the stable rank-zero candidate by construction.
    prefix_choice = shortlist[:, 0]
    endpoint_distance = np.linalg.norm(candidates[:, :, 9] - goal[:, None], axis=-1)
    endpoint_choice = np.empty(len(pred), np.int64)
    oracle_choice = np.empty(len(pred), np.int64)
    gt_cost = official_d3(candidates[:, :, :6], gt[:, None])
    for index in range(len(pred)):
        endpoint_order = np.lexsort((shortlist[index], endpoint_distance[index]))
        oracle_order = np.lexsort((shortlist[index], gt_cost[index]))
        endpoint_choice[index] = shortlist[index, endpoint_order[0]]
        oracle_choice[index] = shortlist[index, oracle_order[0]]
    direct = official_d3(pred, gt)
    prefix = official_d3(bank[prefix_choice, :6], gt)
    endpoint = official_d3(bank[endpoint_choice, :6], gt)
    shortlist_oracle = official_d3(bank[oracle_choice, :6], gt)
    return {
        "shortlist_ids": shortlist, "shortlist_candidates_sha256": array_sha(candidates, "<f4"),
        "direct": direct, "prefix": prefix, "endpoint": endpoint,
        "shortlist_oracle": shortlist_oracle,
        "prefix_quantization_to_prediction": prefix_distance[batch[:, 0], prefix_choice],
        "endpoint_choice": endpoint_choice, "prefix_choice": prefix_choice,
        "same_candidate_tensor_for_both_selectors": True,
    }


def summarize_proxy(value):
    direct = value["direct"]
    result = {"n": len(direct), "direct_c_d3": float(direct.mean()),
              "prefix_nearest_d3": float(value["prefix"].mean()),
              "goal_endpoint_nearest_d3": float(value["endpoint"].mean()),
              "shortlist_gt_oracle_d3": float(value["shortlist_oracle"].mean()),
              "prefix_quantization_to_c_prediction_d3": float(
                  value["prefix_quantization_to_prediction"].mean())}
    for key, values in (("prefix_minus_direct", value["prefix"] - direct),
                        ("endpoint_minus_direct", value["endpoint"] - direct),
                        ("endpoint_minus_prefix", value["endpoint"] - value["prefix"]),
                        ("shortlist_oracle_minus_direct", value["shortlist_oracle"] - direct)):
        result[key] = {"mean": float(values.mean()), "median": float(np.median(values)),
                       "p05": float(np.quantile(values, .05)),
                       "p95": float(np.quantile(values, .95))}
    result["endpoint_changes_prefix_choice"] = int(np.sum(
        value["endpoint_choice"] != value["prefix_choice"]))
    result["shortlist_ids_sha256"] = array_sha(value["shortlist_ids"], "<i8")
    result["shortlist_candidates_sha256"] = value["shortlist_candidates_sha256"]
    return result


def load_control_report(eval_spec, manifest_spec, base_seed, pinned_source):
    from scripts import analyze_motiondrive_v2_p6_stop_balance_results as p6
    from scripts import analyze_motiondrive_v2_p7_goal_routing_results as p7_results
    eval_path, eval_sha = eval_spec
    manifest_path, manifest_sha = manifest_spec
    payload, _ = p6.read_pinned_json(eval_path, eval_sha)
    manifest, _ = p6.read_pinned_json(manifest_path, manifest_sha)
    rows, audit = p7_results.validate_p7_report(payload, base_seed=base_seed, arm="control")
    lineage = p7_results.validate_terminal_manifest(
        manifest, base_seed=base_seed, arm="control", pinned_source=pinned_source,
        expected_official_d3=audit["stored_official_d3"])
    return rows, audit, lineage


def validated_control_arrays(rows, expected_identity, tune_prefix, *, name):
    require([item["identity"] for item in rows] == expected_identity,
            f"{name} report row order differs from current tune")
    gt = np.stack([item["gt"] for item in rows]).astype(np.float32)
    pred = np.stack([item["pred"] for item in rows]).astype(np.float32)
    require(np.array_equal(gt, tune_prefix), f"{name} GT differs from ego5 prefix")
    require(pred.shape == gt.shape and np.isfinite(pred).all(), f"{name} prediction invalid")
    return pred, gt


def evaluate(args):
    require_runtime()
    require(not args.output.exists(), "refusing to overwrite evaluation output")
    bank_manifest, arrays, bank_receipt, bank_artifact_receipts = load_frozen_bank(
        args.bank_manifest[0], args.bank_manifest[1])
    manifest, split_receipt = load_split(args.split_manifest)
    ego, ego5, names, cache_receipts = load_cache_metadata(args.ego_cache, args.ego5_cache)
    train_rows = selected_rows(manifest, names, ego["frame"], "train")
    train_fut5, _ = selected_trajectories(ego, ego5, train_rows, split="train")
    validate_bank_sources(arrays["bank.npy"], arrays["source_rows.npy"],
                          train_rows, ego5["fut5"])
    require(bank_manifest["train"]["fut5_sha256"] == array_sha(train_fut5, "<f4"),
            "bank manifest train fut5 identity mismatch")
    # Tune labels are first indexed only after the external frozen-manifest SHA gate above.
    tune_rows = selected_rows(manifest, names, ego["frame"], "tune")
    tune_fut5, _ = selected_trajectories(ego, ego5, tune_rows, split="tune")
    bank = arrays["bank.npy"]
    full_oracle = per_metric_oracles(tune_fut5, bank)

    from scripts import analyze_motiondrive_v2_p6_stop_balance_results as p6
    from scripts import analyze_motiondrive_v2_p7_goal_routing_results as p7_results
    source_payload, _ = p6.read_pinned_json(*args.p7_source_manifest)
    pinned_source = p7_results.validate_p7_source_manifest(
        source_payload, args.p7_source_manifest[1])
    reports, audits, lineages, proxy, ground_truth = {}, {}, {}, {}, {}
    expected_identity = [(int(row), str(names[row]),
                          str(manifest["scene_to_session"][str(names[row])]), int(ego["frame"][row]))
                         for row in tune_rows]
    for base, eval_spec, manifest_spec in ((0, args.c0_eval, args.c0_manifest),
                                            (1, args.c1_eval, args.c1_manifest)):
        rows, audit, lineage = load_control_report(eval_spec, manifest_spec, base, pinned_source)
        pred, gt = validated_control_arrays(
            rows, expected_identity, tune_fut5[:, :6], name=f"C{base}")
        value = proxy_selectors(pred, gt, tune_fut5[:, 9], bank)
        reports[f"c{base}"] = summarize_proxy(value)
        audits[f"c{base}"] = audit
        lineages[f"c{base}"] = lineage
        proxy[base] = value
        ground_truth[base] = gt
    require(np.array_equal(ground_truth[0], ground_truth[1]), "C0/C1 GT mismatch")
    mean = {}
    for field in ("direct_c_d3", "prefix_nearest_d3", "goal_endpoint_nearest_d3",
                  "shortlist_gt_oracle_d3", "prefix_quantization_to_c_prediction_d3"):
        mean[field] = float(np.mean([reports["c0"][field], reports["c1"][field]]))
    mean["endpoint_minus_direct"] = float(np.mean([
        reports["c0"]["endpoint_minus_direct"]["mean"],
        reports["c1"]["endpoint_minus_direct"]["mean"]]))
    result = {
        "schema_version": 1, "status": "completed_cpu_train203_bank_feasibility",
        "inputs": {"bank_manifest": bank_receipt, "split": split_receipt,
                   **cache_receipts,
                   "p7_source_manifest": {"path": str(Path(args.p7_source_manifest[0]).resolve()),
                                          "sha256": args.p7_source_manifest[1]},
                   "c0_eval": {"path": str(Path(args.c0_eval[0]).resolve()), "sha256": args.c0_eval[1]},
                   "c0_manifest": {"path": str(Path(args.c0_manifest[0]).resolve()), "sha256": args.c0_manifest[1]},
                   "c1_eval": {"path": str(Path(args.c1_eval[0]).resolve()), "sha256": args.c1_eval[1]},
                   "c1_manifest": {"path": str(Path(args.c1_manifest[0]).resolve()), "sha256": args.c1_manifest[1]}},
        "bank_artifacts_sha256_before_after_exact": bank_artifact_receipts,
        "bank": {"manifest_status": bank_manifest["status"],
                 "bank_sha256": bank_manifest["artifacts"]["bank.npy"]["sha256"],
                 "rows": K_BANK, "fit_seed": FIT_SEED},
        "tune": {"rows": len(tune_rows), "rows_sha256": array_sha(tune_rows, "<i8"),
                 "full_bank_metricwise_oracle": summarize_oracle(full_oracle),
                 "fixed_m12_offline_proxies": reports, "two_base_descriptive_mean": mean},
        "audit": {"reports": audits, "lineage": lineages},
        "boundaries": {
            "bank_frozen_before_tune": True, "final_rows_accessed": False,
            "gpu_used": False, "new_model_forward": False, "neural_training": False,
            "full_bank_oracle_is_representation_coverage_only": True,
            "shortlist_gt_oracle_is_not_a_selector": True,
            "prefix_nearest_top12_is_local_quantized_alternatives_not_multimodal_capacity": True,
            "negative_m12_result_does_not_close_multimodal_generation": True,
            "c_predictions_are_already_indirectly_goal_conditioned": True,
            "goal_endpoint_result_is_only_added_index_effect_on_reused_tune": True,
            "no_post_selection_coordinate_edits": True,
            "no_threshold_or_m_or_bank_sweep": True,
        },
        "environment": {"python": sys.version, "numpy": np.__version__,
                        "sklearn": __import__("sklearn").__version__,
                        "scipy": __import__("scipy").__version__,
                        "expected_runtime": dict(EXPECTED_FIT_RUNTIME),
                        "script_sha256": file_sha(__file__)},
    }
    # Strict byte stability after all calculations.
    for receipt in result["inputs"].values():
        require(file_sha(receipt["path"]) == receipt["sha256"], "evaluation input changed")
    for receipt in bank_artifact_receipts.values():
        require(file_sha(receipt["path"]) == receipt["sha256"], "bank artifact changed")
    write_new_json(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "output_sha256": file_sha(args.output), "gpu_used": False}, sort_keys=True))
    return 0


def require_runtime():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_VISIBLE_DEVICES must be empty")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        require(os.environ.get(name) == "1", f"{name} must equal 1")
    import scipy
    validate_runtime_versions(sys.version.split()[0], np.__version__, scipy.__version__,
                              __import__("sklearn").__version__)


def validate_runtime_versions(python, numpy, scipy, sklearn):
    actual = {"python": python, "numpy": numpy, "scipy": scipy, "sklearn": sklearn}
    require(actual == EXPECTED_FIT_RUNTIME,
            f"fit/evaluation runtime mismatch: {actual} != {EXPECTED_FIT_RUNTIME}")


def pair(value):
    require(len(value) == 2 and valid_sha(value[1]), "expected PATH SHA256 pair")
    return Path(value[0]), value[1]


def arguments(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)
    fit_parser = sub.add_parser("fit", help="fit/freeze train-only K1024 bank; no tune access")
    for target in (fit_parser,):
        target.add_argument("--split-manifest", type=Path, required=True)
        target.add_argument("--ego-cache", type=Path, required=True)
        target.add_argument("--ego5-cache", type=Path, required=True)
    fit_parser.add_argument("--output-dir", type=Path, required=True)
    eval_parser = sub.add_parser("evaluate", help="evaluate already-frozen bank on tune only")
    eval_parser.add_argument("--split-manifest", type=Path, required=True)
    eval_parser.add_argument("--ego-cache", type=Path, required=True)
    eval_parser.add_argument("--ego5-cache", type=Path, required=True)
    eval_parser.add_argument("--bank-manifest", nargs=2, required=True, metavar=("PATH", "SHA256"))
    eval_parser.add_argument("--p7-source-manifest", nargs=2, required=True, metavar=("PATH", "SHA256"))
    for name in ("c0-eval", "c0-manifest", "c1-eval", "c1-manifest"):
        eval_parser.add_argument(f"--{name}", nargs=2, required=True, metavar=("PATH", "SHA256"))
    eval_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "evaluate":
        for name in ("bank_manifest", "p7_source_manifest", "c0_eval", "c0_manifest",
                     "c1_eval", "c1_manifest"):
            setattr(args, name, pair(getattr(args, name)))
    return args


def main(argv=None):
    args = arguments(argv)
    return fit(args) if args.phase == "fit" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
