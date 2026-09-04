#!/usr/bin/env python3
"""Phase 5A: train-only path-geometry x velocity-profile bank sweep.

This is deliberately CPU-only.  It never imports torch and it rejects paths whose
name contains ``test``.  P/V are selected on the eight-scenario mini validation
split; the 38-scenario dev split is evaluated exactly once after the choice is
locked.  The dev split is already experiment-contaminated and is report-only.

The clustering is a deterministic, stratified medoid approximation:

* MiniBatchKMeans finds centres in normalized feature space.
* Every centre is replaced by a unique, real train trajectory (a medoid proxy).
* Geometry quotas preserve straight/left/right/U-turn examples.
* Velocity quotas preserve creep/deceleration/cruise/acceleration examples.
* Stop is not normalized; bank row zero is an exact all-zero trajectory.

The composed trajectory is C[p,v,t] = S[v] * q[p](r[v,t]).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

# Prevent numerical libraries from silently consuming the whole B200 host.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


OFFICIAL_W = np.asarray([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0
P_VALUES = (64, 128, 256, 512)
V_VALUES = (16, 32, 64, 128)
EXPECTED_SHA256 = {
    "ego5": "700d9443cbf38aaaeffbbd5dab789b5e448adf931e15939baa9d37779e4095c1",
    "split": "e56ee8803d8de63eb5fa700283afdddf6d735746625c8fcd781251e8902c60c0",
    "anchor": "04c5f2fc20dffd489907d2f7a9e550a11ee2895974c6f276736d70bb3b074ced",
    "a0": "4ccb7ac0358b977e707c121a200e621f2efd34d89c3abbdabf2247b67b291aa9",
    "a1": "fbff5f18f997bbab9252905ceedea8ebbe1981c4eff9b8904eec893c832d0230",
}
ROUTE_NAMES = ("straight", "left", "right", "uturn")
VELOCITY_NAMES = ("creep", "decel", "cruise", "accel")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def cumulative_geometry(traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-step length, cumulative length and total length."""
    origin = np.zeros((traj.shape[0], 1, 2), dtype=traj.dtype)
    step = np.linalg.norm(np.diff(np.concatenate([origin, traj], axis=1), axis=1), axis=2)
    cumulative = np.cumsum(step, axis=1)
    return step, cumulative, cumulative[:, -1]


def last_heading_deg(traj: np.ndarray) -> np.ndarray:
    delta = np.diff(
        np.concatenate([np.zeros((len(traj), 1, 2), dtype=traj.dtype), traj], axis=1),
        axis=1,
    )
    norm = np.linalg.norm(delta, axis=2)
    out = np.zeros(len(traj), dtype=np.float64)
    for i in range(len(traj)):
        valid = np.flatnonzero(norm[i] > 1e-3)
        if len(valid):
            d = delta[i, valid[-1]]
            out[i] = np.degrees(np.arctan2(d[1], d[0]))
    return out


def route_labels(traj: np.ndarray) -> np.ndarray:
    """Fixed, train-independent route thresholds in ego coordinates."""
    heading = last_heading_deg(traj)
    end = traj[:, -1]
    label = np.full(len(traj), "straight", dtype="U8")
    uturn = (end[:, 0] < -1.0) | (np.abs(heading) > 100.0)
    label[uturn] = "uturn"
    left = (~uturn) & (heading > 10.0) & (end[:, 1] > 1.0)
    right = (~uturn) & (heading < -10.0) & (end[:, 1] < -1.0)
    label[left] = "left"
    label[right] = "right"
    return label


def velocity_labels(step: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Stop/creep plus moving acceleration regimes (0.5 m/0.5 s threshold)."""
    delta = step[:, -3:].mean(axis=1) - step[:, :3].mean(axis=1)
    label = np.full(len(step), "cruise", dtype="U8")
    label[delta > 0.5] = "accel"
    label[delta < -0.5] = "decel"
    label[total < 2.0] = "creep"
    label[total < 0.25] = "stop"
    return label


def normalized_path_features(
    traj: np.ndarray,
    cumulative: np.ndarray,
    total: np.ndarray,
    sample_count: int = 33,
) -> np.ndarray:
    """Uniform-arc-length q(u) features, flattened for clustering."""
    grid = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    out = np.empty((len(traj), sample_count, 2), dtype=np.float32)
    for i in range(len(traj)):
        s = np.concatenate([[0.0], cumulative[i] / total[i]])
        q = np.concatenate([np.zeros((1, 2)), traj[i] / total[i]], axis=0)
        # Tiny duplicate arc-length knots can exist during a pause.  Keeping the
        # final point for each knot makes interpolation deterministic.
        rev_unique = np.unique(s[::-1], return_index=True)[1]
        keep = np.sort(len(s) - 1 - rev_unique)
        out[i, :, 0] = np.interp(grid, s[keep], q[keep, 0])
        out[i, :, 1] = np.interp(grid, s[keep], q[keep, 1])
    return out.reshape(len(traj), -1)


def allocate_sqrt_quotas(labels: np.ndarray, names: Sequence[str], total: int) -> Dict[str, int]:
    counts = {name: int(np.sum(labels == name)) for name in names}
    active = [name for name in names if counts[name] > 0]
    if total < len(active):
        raise ValueError(f"total={total} cannot preserve {len(active)} strata")
    quota = {name: (1 if name in active else 0) for name in names}
    remaining = total - len(active)
    if remaining:
        score = np.asarray([np.sqrt(counts[name]) for name in active], dtype=np.float64)
        raw = remaining * score / score.sum()
        floor = np.floor(raw).astype(int)
        for name, add in zip(active, floor):
            quota[name] += int(add)
        left = remaining - int(floor.sum())
        order = np.argsort(-(raw - floor), kind="stable")
        for j in order[:left]:
            quota[active[int(j)]] += 1
    # No stratum may request more medoids than samples; redistribute if needed.
    overflow = 0
    for name in active:
        if quota[name] > counts[name]:
            overflow += quota[name] - counts[name]
            quota[name] = counts[name]
    while overflow:
        candidates = [n for n in active if quota[n] < counts[n]]
        if not candidates:
            raise RuntimeError("unable to redistribute cluster quota")
        candidates.sort(key=lambda n: (-(counts[n] - quota[n]), names.index(n)))
        quota[candidates[0]] += 1
        overflow -= 1
    assert sum(quota.values()) == total
    return quota


def unique_nearest_rows(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Map centres to unique real rows deterministically."""
    selected: List[int] = []
    used = np.zeros(len(features), dtype=bool)
    for center in centers:
        distance = np.sum((features - center[None]) ** 2, axis=1)
        distance[used] = np.inf
        idx = int(np.argmin(distance))
        if not np.isfinite(distance[idx]):
            raise RuntimeError("not enough unique rows for medoid replacement")
        selected.append(idx)
        used[idx] = True
    return np.asarray(selected, dtype=np.int64)


@dataclass
class MedoidSet:
    source_local: np.ndarray
    source_global: np.ndarray
    strata: np.ndarray
    support: np.ndarray
    quotas: Dict[str, int]


def stratified_medoid_approximation(
    features: np.ndarray,
    global_rows: np.ndarray,
    labels: np.ndarray,
    names: Sequence[str],
    total_clusters: int,
    seed: int,
    standardize_within_stratum: bool = True,
) -> MedoidSet:
    quotas = allocate_sqrt_quotas(labels, names, total_clusters)
    sources: List[np.ndarray] = []
    supports: List[np.ndarray] = []
    source_labels: List[str] = []
    for stratum_idx, name in enumerate(names):
        local = np.flatnonzero(labels == name)
        k = quotas[name]
        if k == 0:
            continue
        x = features[local].astype(np.float64, copy=False)
        if standardize_within_stratum:
            scaler = StandardScaler().fit(x)
            z = scaler.transform(x)
        else:
            z = x
        if k == 1:
            center = z.mean(axis=0, keepdims=True)
            cluster_label = np.zeros(len(z), dtype=np.int64)
        else:
            model = MiniBatchKMeans(
                n_clusters=k,
                random_state=seed + 1009 * stratum_idx + 17 * total_clusters,
                batch_size=min(4096, max(256, len(z))),
                n_init=3,
                max_iter=200,
                max_no_improvement=25,
                reassignment_ratio=0.0,
            )
            cluster_label = model.fit_predict(z)
            center = model.cluster_centers_
        medoid_in_stratum = unique_nearest_rows(z, center)
        sources.append(local[medoid_in_stratum])
        supports.append(np.bincount(cluster_label, minlength=k).astype(np.int64))
        source_labels.extend([name] * k)
    source_local = np.concatenate(sources)
    support = np.concatenate(supports)
    strata = np.asarray(source_labels, dtype="U8")
    # Stable semantic ordering makes artifacts reproducible and auditable.
    semantic = {name: i for i, name in enumerate(names)}
    order = np.asarray(
        sorted(range(len(source_local)), key=lambda i: (semantic[strata[i]], -int(support[i]), int(source_local[i]))),
        dtype=np.int64,
    )
    source_local = source_local[order]
    support = support[order]
    strata = strata[order]
    return MedoidSet(source_local, global_rows[source_local], strata, support, quotas)


def compose_bank(
    geometry_traj: np.ndarray,
    geometry_s: np.ndarray,
    geometry_total: np.ndarray,
    velocity_profile: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose exact-zero + all nonzero path/velocity pairs."""
    p_count = len(geometry_traj)
    nonzero_v = velocity_profile[1:]
    n = 1 + p_count * len(nonzero_v)
    bank = np.empty((n, 10, 2), dtype=np.float32)
    path_id = np.full(n, -1, dtype=np.int32)
    velocity_id = np.zeros(n, dtype=np.int32)
    bank[0] = 0.0
    row = 1
    for v_offset, profile in enumerate(nonzero_v, start=1):
        scale = float(profile[0])
        r = profile[1:]
        for p in range(p_count):
            u = np.concatenate([[0.0], geometry_s[p] / geometry_total[p]])
            q = np.concatenate([np.zeros((1, 2)), geometry_traj[p] / geometry_total[p]], axis=0)
            rev_unique = np.unique(u[::-1], return_index=True)[1]
            keep = np.sort(len(u) - 1 - rev_unique)
            bank[row, :, 0] = scale * np.interp(r, u[keep], q[keep, 0])
            bank[row, :, 1] = scale * np.interp(r, u[keep], q[keep, 1])
            path_id[row] = p
            velocity_id[row] = v_offset
            row += 1
    assert row == n and np.array_equal(bank[0], np.zeros((10, 2), dtype=np.float32))
    return bank, path_id, velocity_id


def oracle_vectors(gt: np.ndarray, bank: np.ndarray, chunk: int) -> Dict[str, np.ndarray]:
    best_d3 = np.full(len(gt), np.inf, dtype=np.float64)
    best_endpoint = np.full(len(gt), np.inf, dtype=np.float64)
    best_traj5 = np.full(len(gt), np.inf, dtype=np.float64)
    gt32 = gt.astype(np.float32, copy=False)
    for start in range(0, len(bank), chunk):
        candidate = bank[start : start + chunk]
        distance = np.linalg.norm(gt32[:, None] - candidate[None], axis=-1)
        d3 = np.sum(distance[:, :, :6] * OFFICIAL_W.astype(np.float32), axis=2)
        endpoint = distance[:, :, 9]
        traj5 = distance.mean(axis=2)
        best_d3 = np.minimum(best_d3, d3.min(axis=1))
        best_endpoint = np.minimum(best_endpoint, endpoint.min(axis=1))
        best_traj5 = np.minimum(best_traj5, traj5.min(axis=1))
    return {"d3": best_d3, "endpoint5": best_endpoint, "traj5": best_traj5}


def summarize_vectors(
    vectors: Mapping[str, np.ndarray],
    weights: np.ndarray | None,
    gt_routes: np.ndarray,
    gt_velocities: np.ndarray,
) -> Dict[str, object]:
    out: Dict[str, object] = {
        key: weighted_mean(value, weights) for key, value in vectors.items()
    }
    out["coverage"] = {
        "d3_le_0.15": weighted_mean((vectors["d3"] <= 0.15).astype(float), weights),
        "endpoint5_le_1m": weighted_mean((vectors["endpoint5"] <= 1.0).astype(float), weights),
        "traj5_le_0.5m": weighted_mean((vectors["traj5"] <= 0.5).astype(float), weights),
    }
    bucket: Dict[str, object] = {}
    for name in ("stop", "creep", "move", *ROUTE_NAMES):
        if name == "stop":
            mask = gt_velocities == "stop"
        elif name == "creep":
            mask = gt_velocities == "creep"
        elif name == "move":
            mask = ~np.isin(gt_velocities, ["stop", "creep"])
        else:
            mask = gt_routes == name
        if not mask.any():
            continue
        w = None if weights is None else weights[mask]
        bucket[name] = {
            "n": int(mask.sum()),
            "d3": weighted_mean(vectors["d3"][mask], w),
            "endpoint5": weighted_mean(vectors["endpoint5"][mask], w),
            "traj5": weighted_mean(vectors["traj5"][mask], w),
            "d3_le_0.15": weighted_mean((vectors["d3"][mask] <= 0.15).astype(float), w),
            "endpoint5_le_1m": weighted_mean((vectors["endpoint5"][mask] <= 1.0).astype(float), w),
        }
    out["bucket"] = bucket
    return out


def evaluate_bank(
    gt: np.ndarray,
    bank: np.ndarray,
    weights: np.ndarray | None,
    chunk: int,
) -> Dict[str, object]:
    step, _, total = cumulative_geometry(gt)
    return summarize_vectors(
        oracle_vectors(gt, bank, chunk),
        weights,
        route_labels(gt),
        velocity_labels(step, total),
    )


def axis_oracle_decomposition(
    gt: np.ndarray,
    geometry_traj: np.ndarray,
    geometry_s: np.ndarray,
    geometry_total: np.ndarray,
    velocity_profile: np.ndarray,
    weights: np.ndarray | None,
) -> Dict[str, object]:
    """Separate geometry and velocity quantization error using GT on the other axis."""
    geometry_vectors = {key: np.empty(len(gt), np.float64) for key in ("d3", "endpoint5", "traj5")}
    velocity_vectors = {key: np.empty(len(gt), np.float64) for key in ("d3", "endpoint5", "traj5")}
    self_error = np.empty(len(gt), np.float64)
    for i, sample in enumerate(gt):
        step, cumulative, total = cumulative_geometry(sample[None])
        if total[0] < 1e-9:
            profile = np.zeros((1, 11), dtype=np.float32)
        else:
            profile = np.concatenate([[total[0]], cumulative[0] / total[0]]).astype(np.float32)[None]

        # Clustered geometry with this sample's exact physical velocity profile.
        geom_bank, _, _ = compose_bank(
            geometry_traj,
            geometry_s,
            geometry_total,
            np.concatenate([np.zeros((1, 11), np.float32), profile]),
        )
        gv = oracle_vectors(sample[None], geom_bank, chunk=max(1, len(geom_bank)))
        for key in geometry_vectors:
            geometry_vectors[key][i] = gv[key][0]

        # This sample's exact geometry with the clustered physical profiles.
        velocity_bank, _, _ = compose_bank(sample[None], cumulative, total, velocity_profile)
        vv = oracle_vectors(sample[None], velocity_bank, chunk=max(1, len(velocity_bank)))
        for key in velocity_vectors:
            velocity_vectors[key][i] = vv[key][0]

        if total[0] < 1e-9:
            self_error[i] = float(np.abs(sample).max())
        else:
            self_bank, _, _ = compose_bank(
                sample[None],
                cumulative,
                total,
                np.concatenate([np.zeros((1, 11), np.float32), profile]),
            )
            self_error[i] = float(np.abs(self_bank[1] - sample).max())

    return {
        "gt_velocity_clustered_geometry": {
            key: weighted_mean(value, weights) for key, value in geometry_vectors.items()
        },
        "gt_geometry_clustered_velocity": {
            key: weighted_mean(value, weights) for key, value in velocity_vectors.items()
        },
        "self_reconstruction_max_abs": float(self_error.max()),
        "self_reconstruction_mean_abs": float(self_error.mean()),
    }


def render_report(result: Mapping[str, object]) -> str:
    src = result["source"]
    selected = result["selected"]
    mini = result["mini"]
    dev = result["dev_report_only"]
    sweep = result["sweep"]
    lines = [
        "# Phase 5A — Factorized Path × Velocity Bank Oracle Sweep",
        "",
        f"Generated: {result['generated_at']} KST",
        "",
        "> CPU-only, train-only fitting. P/V selection used only the 8-scenario mini-val. "
        "The 38-scenario dev38 split was evaluated once after locking the configuration and is already "
        "experiment-contaminated; it is report-only, not an unbiased final-val estimate.",
        "",
        "## Decision",
        "",
        f"**{selected['decision']}** — selected `P={selected['P']}, V={selected['V']}` "
        f"({selected['candidate_count']:,} virtual candidates).",
        "",
        f"- Mini-val D3: `{selected['mini_d3']:.6f}`; legacy `{selected['mini_legacy_d3']:.6f}`; "
        f"delta `{selected['mini_d3_delta']:+.6f}`.",
        f"- Dev38 D3: `{selected['dev_d3']:.6f}`; legacy `0.099634`; "
        f"delta `{selected['dev_d3_delta']:+.6f}`.",
        f"- Dev38 5s endpoint / mean-trajectory oracle: `{selected['dev_endpoint5']:.6f}` / "
        f"`{selected['dev_traj5']:.6f}`.",
        f"- Exact-zero candidate: `{selected['exact_zero']}`.",
        "",
        "## Input provenance",
        "",
        "| input | path | SHA256 |",
        "|---|---|---|",
    ]
    for name in ("ego5", "split", "anchor", "a0", "a1"):
        lines.append(f"| {name} | `{src[name]['path']}` | `{src[name]['sha256']}` |")
    lines += [
        f"| sweep script | `scripts/sweep_factorized_bank.py` | `{result['script_sha256']}` |",
        "",
        f"Train fit rows: **{result['counts']['train']}** (`train_idx`, frame≥30); "
        f"mini-val rows: **{result['counts']['mini']}**; dev38 rows: **{result['counts']['dev']}**.",
        "No test paths, rows, statistics, or thresholds were used.",
        "",
        "## Deterministic approximation",
        "",
        "Full pairwise k-medoids at N=17,820 and four P/V sizes would be wasteful. "
        "We use seeded MiniBatchKMeans in standardized feature space, then replace every centre with a "
        "unique real train trajectory. Geometry uses uniform arc-length q(u) features and sqrt-count "
        "quotas for straight/left/right/U-turn. Velocity uses `[S,r1..r10]` and quotas for "
        "creep/decel/cruise/accel. Stop is a separate exact-zero row. Thus all prototypes are train "
        "medoids (not learned synthetic coordinates); only cross-products are composed.",
        "",
        "A first, rejected diagnostic clustered per-stratum z-scored `[S,r1..r10]` directly. "
        "At P512/V128 it produced mini D3 `0.281397` and dev38 report-only D3 `0.206473`, "
        "failing the gate badly. The constant r10 and low-variance normalized-progress dimensions "
        "distorted Euclidean distance. The reported sweep therefore clusters the same stored `[S,r]` "
        "medoids through the induced physical progress `s_i=S*r_i`, globally standardized and "
        "official-weighted. This rejected result is retained in the JSON manifest.",
        "",
        "## Mini-val sweep (selection source)",
        "",
        "| P | V | candidates | D3 | Δ vs legacy | endpoint5 | traj5 | wall s | gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in sweep:
        lines.append(
            f"| {row['P']} | {row['V']} | {row['candidate_count']} | {row['d3']:.6f} | "
            f"{row['d3_delta']:+.6f} | {row['endpoint5']:.6f} | {row['traj5']:.6f} | "
            f"{row['wall_seconds']:.2f} | {'PASS' if row['d3_gate'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Selection rule was locked before dev38: among mini-val D3-gate passers, minimize 5s "
        "mean-trajectory oracle, then endpoint oracle, then candidate count. This is an oracle bank "
        "decision only; learnable coarse/fine ranking is not evaluated here.",
        "",
        "## A0/A1 references and selected bank",
        "",
        "| split/bank | D3 | endpoint5 | traj5 |",
        "|---|---:|---:|---:|",
        f"| mini legacy/A0 | {mini['legacy']['d3']:.6f} | {mini['a0']['endpoint5']:.6f} | {mini['a0']['traj5']:.6f} |",
        f"| mini A1 | {mini['a1']['d3']:.6f} | {mini['a1']['endpoint5']:.6f} | {mini['a1']['traj5']:.6f} |",
        f"| mini selected P×V | {mini['selected']['d3']:.6f} | {mini['selected']['endpoint5']:.6f} | {mini['selected']['traj5']:.6f} |",
        f"| dev38 legacy/A0 | {dev['legacy']['d3']:.6f} | {dev['a0']['endpoint5']:.6f} | {dev['a0']['traj5']:.6f} |",
        f"| dev38 A1 | {dev['a1']['d3']:.6f} | {dev['a1']['endpoint5']:.6f} | {dev['a1']['traj5']:.6f} |",
        f"| dev38 selected P×V | {dev['selected']['d3']:.6f} | {dev['selected']['endpoint5']:.6f} | {dev['selected']['traj5']:.6f} |",
        "",
        "## Dev38 bucket report (report-only)",
        "",
        "| bucket | n | D3 | endpoint5 | traj5 | D3≤0.15 | endpoint≤1m |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in dev["selected"]["bucket"].items():
        lines.append(
            f"| {name} | {row['n']} | {row['d3']:.6f} | {row['endpoint5']:.6f} | "
            f"{row['traj5']:.6f} | {row['d3_le_0.15']:.3f} | {row['endpoint5_le_1m']:.3f} |"
        )
    lines += [
        "",
        "## Artifacts and compute",
        "",
        f"- Selected bank: `{result['artifact']['bank_path']}`",
        f"- Selected bank SHA256: `{result['artifact']['bank_sha256']}`",
        f"- Machine: `{result['runtime']['platform']}`; sklearn `{result['runtime']['sklearn']}`; "
        f"threads `{result['runtime']['threads']}`; total wall `{result['runtime']['wall_seconds']:.2f}s`.",
        "- Evaluation is chunked; no full `[samples,candidates,10,2]` distance tensor is retained.",
        "",
        "## Axis oracle decomposition",
        "",
        "| split | oracle | D3 | endpoint5 | traj5 |",
        "|---|---|---:|---:|---:|",
        f"| mini | GT velocity + clustered geometry | {result['axis_decomposition']['mini']['gt_velocity_clustered_geometry']['d3']:.6f} | "
        f"{result['axis_decomposition']['mini']['gt_velocity_clustered_geometry']['endpoint5']:.6f} | "
        f"{result['axis_decomposition']['mini']['gt_velocity_clustered_geometry']['traj5']:.6f} |",
        f"| mini | GT geometry + clustered velocity | {result['axis_decomposition']['mini']['gt_geometry_clustered_velocity']['d3']:.6f} | "
        f"{result['axis_decomposition']['mini']['gt_geometry_clustered_velocity']['endpoint5']:.6f} | "
        f"{result['axis_decomposition']['mini']['gt_geometry_clustered_velocity']['traj5']:.6f} |",
        f"| dev38 | GT velocity + clustered geometry | {result['axis_decomposition']['dev']['gt_velocity_clustered_geometry']['d3']:.6f} | "
        f"{result['axis_decomposition']['dev']['gt_velocity_clustered_geometry']['endpoint5']:.6f} | "
        f"{result['axis_decomposition']['dev']['gt_velocity_clustered_geometry']['traj5']:.6f} |",
        f"| dev38 | GT geometry + clustered velocity | {result['axis_decomposition']['dev']['gt_geometry_clustered_velocity']['d3']:.6f} | "
        f"{result['axis_decomposition']['dev']['gt_geometry_clustered_velocity']['endpoint5']:.6f} | "
        f"{result['axis_decomposition']['dev']['gt_geometry_clustered_velocity']['traj5']:.6f} |",
        "",
        f"Exact self-reconstruction maximum absolute error: "
        f"`{result['axis_decomposition']['mini']['self_reconstruction_max_abs']:.3e}` (mini), "
        f"`{result['axis_decomposition']['dev']['self_reconstruction_max_abs']:.3e}` (dev38).",
        "",
        "## Keep / kill interpretation",
        "",
        selected["interpretation"],
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    root = Path("/NHNHOME/data/sukim/adcl")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ego5", type=Path, default=root / "data/etri/ego_cache_5s.npz")
    parser.add_argument("--split", type=Path, default=root / "data/etri/val_clips.npz")
    parser.add_argument(
        "--anchor",
        type=Path,
        default=root / "h200_latest/artifacts/dense_vocab/etri_train330_vocab_k1024.npy",
    )
    parser.add_argument("--a0", type=Path, default=root / "data/etri/bank_A0_onetail.npz")
    parser.add_argument("--a1", type=Path, default=root / "data/etri/bank_A1_multitail.npz")
    parser.add_argument("--artifact-dir", type=Path, default=root / "data/etri/factorized_phase5a")
    parser.add_argument("--json-report", type=Path, default=root / "data/etri/factorized_phase5a/report.json")
    parser.add_argument("--markdown-report", type=Path, default=Path("PHASE5A_FACTORIZED_BANK.md"))
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--skip-sha-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    paths = {"ego5": args.ego5, "split": args.split, "anchor": args.anchor, "a0": args.a0, "a1": args.a1}
    for name, path in paths.items():
        # Reject actual test-data components without false-positive matching the
        # repository directory name ``h200_latest``.
        lowered_parts = {part.lower() for part in path.parts}
        if lowered_parts & {"test", "tests", "meta_test", "ego_cache_test"} or "_test." in path.name.lower():
            raise ValueError(f"test data path is forbidden: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
    source = {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}
    if not args.skip_sha_check:
        for name, expected in EXPECTED_SHA256.items():
            if source[name]["sha256"] != expected:
                raise RuntimeError(f"{name} SHA mismatch: {source[name]['sha256']} != {expected}")

    ego = np.load(args.ego5, allow_pickle=False)
    split = np.load(args.split, allow_pickle=True)
    anchor = np.load(args.anchor).astype(np.float32)
    a0_npz = np.load(args.a0, allow_pickle=False)
    a1_npz = np.load(args.a1, allow_pickle=False)
    a0 = a0_npz["bank"].astype(np.float32)
    a1 = a1_npz["bank"].astype(np.float32)
    a1_parent = a1_npz["parent_anchor_id"]
    if not np.array_equal(a0[:, :6], anchor):
        raise RuntimeError("A0 3s prefix is not bitwise equal to anchor")
    if not np.array_equal(a1[:, :6], anchor[a1_parent]):
        raise RuntimeError("A1 3s prefix is not bitwise equal to parent anchor")
    if not (np.all(anchor[0] == 0) and np.all(a0[0] == 0) and np.all(a1[0] == 0)):
        raise RuntimeError("exact-zero provenance gate failed")

    frame = ego["frame"]
    scenario = ego["scen_idx"]
    subset: Dict[str, np.ndarray] = {}
    for name, key, expected in (("train", "train_idx", 17820), ("mini", "minival_idx", 432), ("dev", "val_idx", 2052)):
        idx = split[key]
        idx = idx[frame[idx] >= 30]
        if len(idx) != expected:
            raise RuntimeError(f"{name} row count {len(idx)} != {expected}")
        subset[name] = idx
    scenario_sets = {name: set(scenario[idx].tolist()) for name, idx in subset.items()}
    if scenario_sets["train"] & scenario_sets["mini"] or scenario_sets["train"] & scenario_sets["dev"] or scenario_sets["mini"] & scenario_sets["dev"]:
        raise RuntimeError("scenario split leakage")
    gt = {name: ego["fut5"][idx].astype(np.float32) for name, idx in subset.items()}
    dev_mask_in_full = frame[split["val_idx"]] >= 30
    dev_weight = split["val_weight"][dev_mask_in_full].astype(np.float64)

    train_step, train_s, train_total = cumulative_geometry(gt["train"])
    train_route = route_labels(gt["train"])
    train_velocity = velocity_labels(train_step, train_total)
    non_stop = train_total >= 0.25
    non_stop_local = np.flatnonzero(non_stop)
    geom_feature = normalized_path_features(
        gt["train"][non_stop], train_s[non_stop], train_total[non_stop]
    )
    # Velocity clustering treats total distance and each normalized progress time
    # as equal standardized dimensions. Medoids retain the original physical S/r.
    r = train_s[non_stop] / train_total[non_stop, None]
    # Stored prototypes remain [S,r1..r10], but clustering uses their induced
    # physical progress s_i=S*r_i.  This approximates official trajectory error
    # much better than z-scoring each stratum's normalized r independently (r10
    # is constant and low-variance dimensions otherwise amplify noise).  The
    # first six steps receive official metric weights; tail steps receive weight 1.
    progress_scaler = StandardScaler().fit(train_s[non_stop])
    progress_weight = np.sqrt(np.concatenate([OFFICIAL_W * 36.0, np.ones(4)]))
    velocity_cluster_feature = progress_scaler.transform(train_s[non_stop]) * progress_weight

    geometry_sets: Dict[int, MedoidSet] = {}
    velocity_sets: Dict[int, MedoidSet] = {}
    with threadpool_limits(limits=args.threads):
        for p in P_VALUES:
            print(f"[cluster] geometry P={p}", flush=True)
            geometry_sets[p] = stratified_medoid_approximation(
                geom_feature,
                subset["train"][non_stop],
                train_route[non_stop],
                ROUTE_NAMES,
                p,
                args.seed + p,
            )
        for v in V_VALUES:
            print(f"[cluster] velocity V={v} (1 exact-zero + {v-1} medoids)", flush=True)
            velocity_sets[v] = stratified_medoid_approximation(
                velocity_cluster_feature,
                subset["train"][non_stop],
                train_velocity[non_stop],
                VELOCITY_NAMES,
                v - 1,
                args.seed + v,
                standardize_within_stratum=False,
            )

    # Reference metrics are computed before the sweep. A0 first six exactly equal
    # legacy; A1 duplicates parents and therefore has the same D3 candidate set.
    mini_ref = {
        "legacy": evaluate_bank(gt["mini"], np.pad(anchor, ((0, 0), (0, 4), (0, 0))), None, args.chunk),
        "a0": evaluate_bank(gt["mini"], a0, None, args.chunk),
        "a1": evaluate_bank(gt["mini"], a1, None, args.chunk),
    }
    if abs(mini_ref["legacy"]["d3"] - mini_ref["a0"]["d3"]) > 1e-7:
        raise RuntimeError("legacy/A0 mini D3 mismatch")

    sweep: List[Dict[str, object]] = []
    bank_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for p in P_VALUES:
        gset = geometry_sets[p]
        g_local = np.searchsorted(subset["train"], gset.source_global)
        # source_global are indices into the full cache; map directly through a dict
        # because split arrays need not be sorted by contract.
        lookup = {int(row): i for i, row in enumerate(subset["train"])}
        g_local = np.asarray([lookup[int(row)] for row in gset.source_global], dtype=np.int64)
        for v in V_VALUES:
            tic = time.perf_counter()
            vset = velocity_sets[v]
            v_lookup = np.asarray([lookup[int(row)] for row in vset.source_global], dtype=np.int64)
            profiles = np.concatenate(
                [np.zeros((1, 11), dtype=np.float32),
                 np.concatenate([train_total[v_lookup, None], train_s[v_lookup] / train_total[v_lookup, None]], axis=1).astype(np.float32)],
                axis=0,
            )
            bank, path_id, velocity_id = compose_bank(
                gt["train"][g_local], train_s[g_local], train_total[g_local], profiles
            )
            metric = evaluate_bank(gt["mini"], bank, None, args.chunk)
            d3_delta = float(metric["d3"] - mini_ref["legacy"]["d3"])
            row = {
                "P": p,
                "V": v,
                "candidate_count": int(len(bank)),
                "d3": metric["d3"],
                "d3_delta": d3_delta,
                "endpoint5": metric["endpoint5"],
                "traj5": metric["traj5"],
                "d3_gate": bool(d3_delta <= 0.003),
                "wall_seconds": time.perf_counter() - tic,
            }
            sweep.append(row)
            bank_cache[(p, v)] = (bank, path_id, velocity_id)
            print(
                f"[sweep] P={p:3d} V={v:3d} C={len(bank):6d} "
                f"D3={metric['d3']:.6f} ({d3_delta:+.6f}) "
                f"end={metric['endpoint5']:.6f} traj={metric['traj5']:.6f}",
                flush=True,
            )

    passing = [row for row in sweep if row["d3_gate"]]
    if passing:
        selected_row = min(passing, key=lambda x: (x["traj5"], x["endpoint5"], x["candidate_count"]))
    else:
        selected_row = min(sweep, key=lambda x: (x["d3"], x["traj5"], x["candidate_count"]))
    selected_key = (int(selected_row["P"]), int(selected_row["V"]))
    selected_bank, selected_path_id, selected_velocity_id = bank_cache[selected_key]

    # Dev is touched only here, after selected_key has been frozen from mini-val.
    dev_ref = {
        "legacy": evaluate_bank(gt["dev"], np.pad(anchor, ((0, 0), (0, 4), (0, 0))), dev_weight, args.chunk),
        "a0": evaluate_bank(gt["dev"], a0, dev_weight, args.chunk),
        "a1": evaluate_bank(gt["dev"], a1, dev_weight, args.chunk),
        "selected": evaluate_bank(gt["dev"], selected_bank, dev_weight, args.chunk),
    }
    mini_selected = evaluate_bank(gt["mini"], selected_bank, None, args.chunk)
    mini_ref["selected"] = mini_selected
    dev_delta = float(dev_ref["selected"]["d3"] - dev_ref["legacy"]["d3"])
    mini_gate = bool(selected_row["d3_gate"])
    dev_gate = bool(dev_delta <= 0.003)
    improves_a1 = (
        mini_selected["endpoint5"] < mini_ref["a1"]["endpoint5"]
        and mini_selected["traj5"] < mini_ref["a1"]["traj5"]
    )
    keep = mini_gate and dev_gate and improves_a1
    decision = "KEEP for coarse/fine scorer experiments" if keep else "KILL as production bank; retain diagnostic artifact"
    if not mini_gate:
        interpretation = (
            "The factorized bank misses the +0.003 mini-val D3 gate. It must not replace A0/A1; "
            "a learnable scorer cannot repair missing candidate coverage."
        )
    elif not improves_a1:
        interpretation = (
            "The bank passes the D3 gate but does not beat A1 on both mini-val 5s endpoint and "
            "trajectory coverage. Keep A0/A1 for production and treat factorization as diagnostic only."
        )
    elif not dev_gate:
        interpretation = (
            "The mini-val-selected bank fails the report-only dev38 D3 gate. Because dev38 cannot be "
            "used to retune P/V, the factorization is not promoted; collect a fresh blind audit split."
        )
    else:
        interpretation = (
            "The bank passes the predeclared D3 gate and improves A1 5s coverage on mini-val, with the "
            "locked configuration also passing dev38. Promote it only to the next image-only "
            "coarse/fine scoring experiment; oracle success does not establish rankability or latency."
        )

    p, v = selected_key
    gset = geometry_sets[p]
    vset = velocity_sets[v]
    lookup = {int(row): i for i, row in enumerate(subset["train"])}
    g_local = np.asarray([lookup[int(row)] for row in gset.source_global], dtype=np.int64)
    v_local = np.asarray([lookup[int(row)] for row in vset.source_global], dtype=np.int64)
    profiles = np.concatenate(
        [np.zeros((1, 11), dtype=np.float32),
         np.concatenate([train_total[v_local, None], train_s[v_local] / train_total[v_local, None]], axis=1).astype(np.float32)],
        axis=0,
    )
    axis_decomposition = {
        "mini": axis_oracle_decomposition(
            gt["mini"], gt["train"][g_local], train_s[g_local], train_total[g_local], profiles, None
        ),
        "dev": axis_oracle_decomposition(
            gt["dev"], gt["train"][g_local], train_s[g_local], train_total[g_local], profiles, dev_weight
        ),
    }
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.artifact_dir / f"factorized_P{p}_V{v}.npz"
    np.savez_compressed(
        artifact_path,
        bank=selected_bank,
        path_id=selected_path_id,
        velocity_id=selected_velocity_id,
        geometry_traj=gt["train"][g_local],
        geometry_cumulative_s=train_s[g_local].astype(np.float32),
        geometry_total_s=train_total[g_local].astype(np.float32),
        geometry_source_cache_row=gset.source_global,
        geometry_stratum=gset.strata,
        geometry_support=gset.support,
        velocity_profile=profiles,
        velocity_source_cache_row=np.concatenate([[-1], vset.source_global]).astype(np.int64),
        velocity_stratum=np.concatenate([["stop"], vset.strata]),
        velocity_support=np.concatenate([[int(np.sum(train_velocity == "stop"))], vset.support]),
        official_w=OFFICIAL_W,
        source_ego5_sha256=source["ego5"]["sha256"],
        source_split_sha256=source["split"]["sha256"],
        source_script_sha256=sha256(Path(__file__).resolve()),
        seed=np.int64(args.seed),
    )
    artifact_sha = sha256(artifact_path)

    import sklearn
    from datetime import datetime
    from zoneinfo import ZoneInfo

    result: Dict[str, object] = {
        "generated_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        "source": source,
        "script_sha256": sha256(Path(__file__).resolve()),
        "counts": {name: int(len(idx)) for name, idx in subset.items()},
        "scenario_counts": {name: int(len(s)) for name, s in scenario_sets.items()},
        "train_bucket_counts": {
            "route": {name: int(np.sum(train_route == name)) for name in ROUTE_NAMES},
            "velocity": {name: int(np.sum(train_velocity == name)) for name in ("stop", *VELOCITY_NAMES)},
        },
        "clustering": {
            "algorithm": "stratified MiniBatchKMeans centers -> unique real train medoid proxy",
            "velocity_cluster_metric": "globally standardized physical progress s1..s10, weighted sqrt([11,11,5,5,2,2,1,1,1,1]); stored as [S,r1..r10]",
            "seed": args.seed,
            "geometry_quotas": {str(k): v.quotas for k, v in geometry_sets.items()},
            "velocity_quotas_nonzero": {str(k): v.quotas for k, v in velocity_sets.items()},
            "route_thresholds": {"turn_heading_deg": 10.0, "turn_lateral_m": 1.0, "uturn_heading_deg": 100.0, "uturn_x_m": -1.0},
            "velocity_thresholds": {"stop_total_arc_m": 0.25, "creep_total_arc_m": 2.0, "accel_step_delta_m_per_0.5s": 0.5},
            "rejected_naive_ablation": {
                "description": "per-stratum z-score clustering directly on [S,r1..r10]",
                "P": 512,
                "V": 128,
                "mini_d3": 0.2813971533857545,
                "dev_report_only_d3": 0.20647322897217432,
                "reason": "fails D3 gate; constant/low-variance normalized-progress dimensions distort Euclidean clustering",
            },
        },
        "sweep": sweep,
        "mini": mini_ref,
        "dev_report_only": dev_ref,
        "axis_decomposition": axis_decomposition,
        "selected": {
            "P": p,
            "V": v,
            "candidate_count": int(len(selected_bank)),
            "mini_d3": mini_selected["d3"],
            "mini_legacy_d3": mini_ref["legacy"]["d3"],
            "mini_d3_delta": float(mini_selected["d3"] - mini_ref["legacy"]["d3"]),
            "dev_d3": dev_ref["selected"]["d3"],
            "dev_d3_delta": dev_delta,
            "dev_endpoint5": dev_ref["selected"]["endpoint5"],
            "dev_traj5": dev_ref["selected"]["traj5"],
            "exact_zero": bool(np.all(selected_bank[0] == 0)),
            "mini_gate": mini_gate,
            "dev_gate_report_only": dev_gate,
            "improves_a1_mini_endpoint_and_traj": improves_a1,
            "decision": decision,
            "interpretation": interpretation,
        },
        "artifact": {"bank_path": str(artifact_path), "bank_sha256": artifact_sha},
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "threads": args.threads,
            "chunk": args.chunk,
            "wall_seconds": time.perf_counter() - started,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.markdown_report.write_text(render_report(result))
    print(json.dumps(result["selected"], indent=2), flush=True)
    print(f"artifact={artifact_path} sha256={artifact_sha}", flush=True)
    print(f"json={args.json_report} markdown={args.markdown_report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
