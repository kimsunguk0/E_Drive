"""Fixed terminal tune analysis. CPU arrays only; no checkpoint/model loads."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
REPORT = ROOT / "reports/sparsedrivev2_20260910/selection_diagnostics"
SOURCE = ROOT / "experiments/sparsedrivev2_20260910/analyze_selection_errors.py"
spec = importlib.util.spec_from_file_location("error_helper", SOURCE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


result = json.loads((REPORT / "relative2000_error_groups.json").read_text())
cache = ROOT / "cache/sparsedrivev2_20260910/features_primary_none2000_v1"
rows = np.load(cache / "tune/rows.npy")
gt = np.load(cache / "tune/gt_xy.npy").astype(np.float64)
status = np.load(cache / "tune/status8.npy").astype(np.float64)
sessions = np.load(cache / "tune/session.npy")
candidate_ids = np.load(cache / "tune/candidate_ids.npy", mmap_mode="r")
candidate_xy = np.load(cache / "tune/candidate_xy.npy", mmap_mode="r")
assert sha(cache / "manifest.json") == "e9455b8e6577ff2a9d488dfc6780129df5b336d247adeeaabded246180b9e298"
assert hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest() == helper.TUNE_ROWS_SHA
unique = sorted(set(sessions))
assert len(unique) == 11
counts = np.array([(sessions == s).sum() for s in unique])
draws = np.random.default_rng(0).integers(len(unique), size=(20000, len(unique)))


def bootstrap(values):
    values = np.asarray(values)
    totals = np.asarray([values[sessions == s].sum(0) for s in unique])
    denominator = counts[draws].sum(-1)
    numerator = totals[draws].sum(1)
    if values.ndim > 1:
        denominator = denominator.reshape((-1,) + (1,) * (values.ndim - 1))
    return np.quantile(numerator / denominator, [.025, .975], axis=0).tolist()


def quantiles(values):
    values = np.asarray(values)
    p95, p99 = np.quantile(values, [.95, .99])
    return {"p95": float(p95), "p99": float(p99), "max": float(values.max()),
            "worst_5pct_mean": float(values[values >= p95].mean()),
            "worst_1pct_mean": float(values[values >= p99].mean())}


speed = np.linalg.norm(status[:, 4:6], axis=-1)
speed_edges = [0, .5, 5, 10, 15, 20, np.inf]
extra_masks = {f"causal_speed_norm_{lo}_to_{hi}": (speed >= lo) & (speed < hi)
               for lo, hi in zip(speed_edges[:-1], speed_edges[1:])}
extra_masks.update({"causal_ax_lt_minus0p5": status[:, 6] < -.5,
                    "causal_abs_ax_le0p5": np.abs(status[:, 6]) <= .5,
                    "causal_ax_gt0p5": status[:, 6] > .5,
                    "causal_vx_negative": status[:, 4] < 0})
loaded = {}
for name, record in result["results"].items():
    path = Path(record["source"])
    with np.load(path, allow_pickle=False) as z:
        assert np.array_equal(z["rows"], rows)
        pred = z["pred"].astype(np.float64)
        oracle = z["shortlist_oracle"].astype(np.float64) if "shortlist_oracle" in z else None
        selected_id = z["candidate_id"] if "candidate_id" in z else None
    delta = pred - gt
    point = np.linalg.norm(delta, axis=-1)
    d3 = point @ helper.WEIGHTS
    record["source_sha256"] = sha(path)
    record["groups"].update({key: helper.summarize(pred, gt, mask) for key, mask in extra_masks.items()})
    record["per_point_signed_xy"] = delta.mean(0).tolist()
    record["per_point_abs_xy"] = np.abs(delta).mean(0).tolist()
    record["pair_time_l2_means"] = point.reshape(len(rows), 3, 2).mean((0, 2)).tolist()
    record["prefix_ade_1_2_3s"] = [float(point[:, :t].mean()) for t in (2, 4, 6)]
    record["d3_session_bootstrap95ci"] = bootstrap(d3)
    record["point_l2_session_bootstrap95ci"] = bootstrap(point)
    record["tails"] = {"d3": quantiles(d3), "endpoint3s": quantiles(point[:, -1])}
    record["groups_d3_counts"] = {"le0p15": int((d3 <= .15).sum()), "gt0p5": int((d3 > .5).sum()),
                                     "gt1p0": int((d3 > 1.).sum())}
    if name != "motiondrive_long_s1":
        if selected_id is None:
            raise ValueError("Candidate ID not saved")
        matches = candidate_ids == selected_id[:, None]
        assert matches.sum(-1).min() >= 1
        selected = matches.argmax(-1)
        assert np.array_equal(pred, candidate_xy[np.arange(len(rows)), selected].astype(np.float64))
        assert oracle is not None
        record["fixed_shortlist_coordinate_identity_pass"] = True
        record["shortlist_oracle_d3"] = float(oracle.mean())
        record["fine_regret"] = float((d3 - oracle).mean())
        record["fine_regret_p95_p99"] = quantiles(d3 - oracle)
    loaded[name] = {"d3": d3, "point": point, "delta": delta, "oracle": oracle}

for pair, entry in result["pairs"].items():
    b, a = pair.split(" minus ")
    point_delta = loaded[b]["point"] - loaded[a]["point"]
    entry["point_l2_difference"] = point_delta.mean(0).tolist()
    entry["point_l2_difference_session_bootstrap95ci"] = bootstrap(point_delta)
    session_delta = {s: float((loaded[b]["d3"] - loaded[a]["d3"])[sessions == s].mean()) for s in unique}
    entry["session_difference_d3"] = session_delta
    entry["improved_sessions"] = sum(v < 0 for v in session_delta.values())
    if loaded[a]["oracle"] is not None and loaded[b]["oracle"] is not None:
        assert np.array_equal(loaded[a]["oracle"], loaded[b]["oracle"])
        entry["shortlist_oracle_bitwise_identical"] = True
        entry["fine_regret_delta_equals_d3_delta"] = True
        old_regret = result["results"][a]["fine_regret"]
        entry["fine_regret_reduction_fraction"] = float(-entry["difference_d3"] / old_regret)

result["provenance"] = {"helper_sha256": sha(SOURCE), "analysis_source_sha256": sha(__file__),
                         "rows_sha256": helper.TUNE_ROWS_SHA, "sessions": unique,
                         "session_counts": counts.tolist(), "bootstrap_repetitions": 20000,
                         "bootstrap_seed": 0, "bootstrap_unit": "session, equal session resampling then frame-weighted mean",
                         "terminal_step": 2000, "checkpoint_selection": "fixed terminal only",
                         "cache_manifest_sha256": sha(cache / "manifest.json"),
                         "model_forward": False, "gpu_used": False, "held_evaluated": False}
result["interpretation_limits"] = [
    "Same tune1998 repeatedly used for method development; paired CIs are descriptive and not an untouched confirmation.",
    "Relative head has provided-goal features whereas frozen base goal_mode is none; no ablation isolates relative kinematics from goal or added MLP.",
    "Future speed trend and endpoint abs-y are retrospective GT-defined bins, not deployed inputs and not necessarily brake/turn ground truth.",
    "Speed-norm bins cover all rows; original helper forward-vx bins omitted negative-vx rows, now separately reported.",
    "XY diagnostics are not additive D3 components; tail quantiles are within-model descriptive thresholds, not matched-row subgroups."
]
destination = REPORT / "relative2000_terminal_analysis.json"
if destination.exists():
    raise FileExistsError(destination)
destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({name: {"d3": rec["groups"]["all"]["d3"], "d3_ci": rec["d3_session_bootstrap95ci"],
                         "tails": rec["tails"], "session_d3": rec["session_d3"]}
                  for name, rec in result["results"].items()}, indent=2))
