"""Reproducible raw fixture, CPU pixel parity, and terminal B1 inference parity.

Commands: prepare-fixture (CPU), preprocess (CPU, no checkpoint), verify (GPU).
GPU verification delegates all checkpoint/source/bank checks to the separate
evaluate_temporal_checkpoint.py API. Only original tune rows are accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
import torch

ROW = 14730
BASE = "/NHNHOME/data/sukim/adcl"
CAMERAS = ("camera_front_left", "camera_front", "camera_front_right")
REQUESTS = tuple((c, 0) for c in CAMERAS) + (("camera_front", -1), ("camera_front", -5))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, text):
    if not condition:
        raise ValueError(text)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def tune_identity(base, row):
    import evaluate_temporal_checkpoint as ev
    base = Path(base)
    split = base / "data/etri/motiondrive_v2/grouped_split_rawtime.json"
    ego = Path("/tmp/pm97/data/etri/ego_cache.npz")
    if not ego.exists():
        ego = base / "data/etri/ego_cache.npz"
    require(sha(split) == ev.SPLIT_SHA and sha(ego) == ev.EGO_SHA, "Primary identity sources changed")
    m = json.loads(split.read_text())
    with np.load(ego, allow_pickle=False) as z:
        names = z["scenarios"].astype(str)[z["scen_idx"]]
        frames = z["frame"]
    require(isinstance(row, int) and 0 <= row < len(frames), "Invalid row")
    scene, frame = str(names[row]), int(frames[row])
    require(scene in m["splits"]["tune"] and frame >= 30 and frame % 5 == 0, "Fixture must use original tune rows")
    return {"row": row, "scene": scene, "frame": frame, "session": m["scene_to_session"][scene],
            "split_path": str(split), "split_sha256": sha(split), "ego_path": str(ego), "ego_sha256": sha(ego)}


def collect_fixture(base, row=ROW):
    """Return official raw files in memory; no train target fields are opened."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    base = Path(base)
    identity = tune_identity(base, row)
    scene, frame = identity["scene"], identity["frame"]
    meta = base / "data/etri/meta_train" / scene
    calibration = meta / "calibration/calibration.parquet"
    timestamps = meta / "meta/timestamps.parquet"
    poses = meta / "annotation/ego_pose.parquet"
    ts = pd.read_parquet(timestamps, columns=["timestamp", "frame_id"])
    wanted_offsets = [*range(-30, 1), 50]
    wanted_frames = {frame + offset for offset in wanted_offsets}
    selected_ts = ts.loc[ts["frame_id"].isin(wanted_frames), ["timestamp", "frame_id"]]
    require(len(selected_ts) == len(wanted_frames) and selected_ts["frame_id"].is_unique
            and selected_ts["timestamp"].is_unique and set(selected_ts["frame_id"]) == wanted_frames,
            "Raw timestamp/frame binding is incomplete or ambiguous")
    wanted_timestamps = selected_ts["timestamp"].tolist()
    # Push down the official pose frame selection before deserializing XYZ/RPY.
    ep = pd.read_parquet(poses, columns=["timestamp", "x", "y", "z", "roll", "pitch", "yaw"],
                         filters=[("timestamp", "in", wanted_timestamps)])
    require(len(ep) == len(wanted_timestamps) and ep["timestamp"].is_unique
            and set(ep["timestamp"]) == set(wanted_timestamps), "Filtered raw pose rows do not match timestamps")
    joined = selected_ts.merge(ep, on="timestamp", how="left", validate="one_to_one").set_index("frame_id")
    records = []
    for offset in wanted_offsets:
        item = joined.loc[frame + offset]
        record = {"frame": offset, **{k: float(item[k]) for k in ("x", "y", "z", "roll", "pitch", "yaw")}}
        if offset == 50:
            record.update(roll=float("nan"), pitch=float("nan"), yaw=float("nan"))
        records.append(record)
    buffer = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(records), buffer)
    files = {"calibration.parquet": calibration.read_bytes(), "ego_pose.parquet": buffer.getvalue().to_pybytes()}
    archive_path = base / "train" / (scene + ".tar")
    with tarfile.open(archive_path, "r:") as archive:
        for camera, offset in REQUESTS:
            member = archive.getmember(f"{scene}/{camera}/{frame + offset:08d}.jpg")
            require(member.isfile(), "Raw image archive member is not an ordinary file")
            files[f"{camera}/frame_{offset}.jpg"] = archive.extractfile(member).read()
    receipt = {"schema": "temporal_raw_fixture_v1", **identity, "files_sha256": {k: digest(v) for k, v in files.items()},
               "source_paths_sha256": {str(p): sha(p) for p in (calibration, timestamps, poses)},
               "raw_tar_path": str(archive_path), "raw_tar_size_bytes": archive_path.stat().st_size,
               "raw_tar_hash_scope": "selected member bytes hashed individually; full 625MB archive not hashed",
               "raw_image_requests": [[c, f] for c, f in REQUESTS], "official_pose_frames": [*range(-30, 1), 50],
               "pose_rows_deserialized_relative_frames": wanted_offsets,
               "pose_timestamp_filter_dtype": str(selected_ts["timestamp"].dtype),
               "pose_timestamp_filter": "parquet timestamp IN exact metadata values for official frames",
               "future_orientation_columns_deserialized_in_fixture_builder": True,
               "future_orientation_replaced_with_nan_before_fixture_publication": True,
               "ego_cache_fields_read": ["scenarios", "scen_idx", "frame"],
               "future_trajectory_labels_read": False, "provided_plus50_goal_included": True,
               "learned_checkpoint_loaded": False, "producer_sha256": sha(__file__)}
    files["fixture_receipt.json"] = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    return files, receipt


def publish_fixture(files, output=None, tar_stdout=False):
    require(bool(output) != bool(tar_stdout), "Choose new output directory or tar stdout")
    if tar_stdout:
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
            for name, value in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(value)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(value))
        return
    out = Path(output)
    require(not out.exists(), "Refusing to overwrite fixture")
    out.mkdir(parents=True)
    for name, value in files.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe fixture path")
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)


def validate_fixture(path):
    path = Path(path).resolve()
    receipt = json.loads((path / "fixture_receipt.json").read_text())
    require(receipt["schema"] == "temporal_raw_fixture_v1", "Unknown fixture schema")
    expected = {"calibration.parquet", "ego_pose.parquet"} | {f"{c}/frame_{f}.jpg" for c, f in REQUESTS}
    require(set(receipt["files_sha256"]) == expected, "Unexpected raw fixture file list")
    for name, expected_sha in receipt["files_sha256"].items():
        require(sha(path / name) == expected_sha, "Raw fixture content changed: " + name)
    require(not receipt["future_trajectory_labels_read"], "Fixture was built from forbidden future trajectory labels")
    return receipt


def tensor_comparison(left, right):
    require(set(left) == set(right), "Input/output key mismatch")
    out = {}
    for name in left:
        a, b = left[name].detach().cpu(), right[name].detach().cpu()
        require(a.shape == b.shape and a.dtype == b.dtype, "Tensor shape/dtype mismatch: " + name)
        a_bytes = a.contiguous().reshape(-1).view(torch.uint8)
        b_bytes = b.contiguous().reshape(-1).view(torch.uint8)
        exact = torch.equal(a_bytes, b_bytes)
        finite = torch.isfinite(a) & torch.isfinite(b) if a.is_floating_point() else torch.ones_like(a, dtype=torch.bool)
        finite_diff = (a[finite].double() - b[finite].double()).abs()
        out[name] = {"shape": list(a.shape), "dtype": str(a.dtype), "bitwise_equal": bool(exact),
                     "numerically_equal": bool(torch.equal(a, b)),
                     "max_absolute_finite_difference": float(finite_diff.max()) if finite_diff.numel() else 0.,
                     "changed_elements": int((a != b).sum()),
                     "raw_sha256": digest(a.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()),
                     "cache_sha256": digest(b.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())}
    return out


def stats_ms(values):
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "samples": values}


def runtime_environment(base):
    import cv2
    packages = {}
    for name in ("numpy", "torch", "Pillow", "pyarrow", "scipy", "timm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"base": str(Path(base).resolve()), "python": sys.version, "python_executable": sys.executable,
            "packages": packages, "opencv_version": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
            "opencv_build_info_sha256": digest(cv2.getBuildInformation().encode()),
            "torch_cpu_threads": torch.get_num_threads(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "host_loadavg_at_receipt": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}


def preprocessing_profile(adapter, fixture, warmup=2, iterations=10):
    require(warmup >= 0 and iterations >= 1, "Invalid CPU profiling count")
    for _ in range(warmup):
        adapter.prepare_clip(fixture)
    values = []
    for _ in range(iterations):
        before = time.perf_counter()
        adapter.prepare_clip(fixture)
        values.append((time.perf_counter() - before) * 1000.)
    return {"wall_ms": stats_ms(values), "warmup": warmup, "iterations": iterations,
            "scope": "disk parquet/image reads, geometry cache lookup, raw undistortion/crop/Q95/resize/normalization, allowed status/goal assembly",
            "excluded": "H2D, model forward, checkpoint loading, reference/GT reads", "geometry_warm": True,
            "note": "OS file cache may be warm; this is CPU preprocessing latency, not RTX4090/model latency"}


def runtime_sources(output, ev, adapter_module):
    result = {}
    for module in (sys.modules[__name__], ev, adapter_module):
        path = Path(module.__file__)
        result[path.name] = sha(path)
        (output / "verification_source" / path.name).write_bytes(path.read_bytes())
    return result


def cache_sample(dataset, rows, row):
    at = np.flatnonzero(np.asarray(rows) == row)
    require(len(at) == 1, "Fixture row absent or duplicated in frozen tune dataset")
    return torch.utils.data.default_collate([dataset[int(at[0])]])


def preprocess(a, output, fixture_receipt):
    import evaluate_temporal_checkpoint as ev
    import temporal_deployment as adapter_module
    source = Path(a.source_root).resolve()
    for name in ("data.py", "temporal_data.py"):
        require(sha(source / name) == ev.PINNED[name], "Unsupported frozen preprocessing source: " + name)
    identity = tune_identity(a.base, fixture_receipt["row"])
    for key in ("row", "scene", "frame", "session", "split_sha256", "ego_sha256"):
        require(identity[key] == fixture_receipt[key], "Fixture row/source identity mismatch")
    with ev.isolated_runtime(source, ["data", "temporal_data"]) as runtime:
        base = runtime.data.PlanDataset(a.base, identity["split_path"], "tune", ego_cache=identity["ego_path"],
                                        augment=False, status_mode="zero", goal_mode="selection")
        dataset = runtime.temporal_data.TemporalPlanDataset(base, history_mode=a.history_mode, auxiliary=False)
        batch = cache_sample(dataset, dataset.rows, fixture_receipt["row"])
        cached = ev.input_tensors(batch, runtime, a.common_status, "cpu")
        with adapter_module.TemporalRawInputAdapter(a.history_mode, a.common_status, "selection", a.camera_workers) as adapter:
            prepared = adapter.prepare_clip(a.fixture)
            raw = {**prepared.inputs, **prepared.selector_inputs}
            parity = tensor_comparison(raw, cached)
            timing = preprocessing_profile(adapter, a.fixture, a.preprocess_warmup, a.preprocess_iterations)
        receipt = {"status": "completed", "mode": "CPU_preprocessing_only", "row": fixture_receipt["row"],
                   "history_mode": a.history_mode, "common_status": a.common_status, "goal_mode": "selection",
                   "input_parity": parity, "all_inputs_bitwise_equal": all(v["bitwise_equal"] for v in parity.values()),
                   "raw_adapter": prepared.metadata, "preprocessing_latency": timing,
                   "checkpoint_loaded": False, "model_forward_performed": False, "gpu_used": False,
                   "runtime_environment": runtime_environment(a.base),
                   "fixture_receipt_sha256": sha(Path(a.fixture) / "fixture_receipt.json"),
                   "frozen_data_source_sha256": {n: sha(source / n) for n in ("data.py", "temporal_data.py")},
                   "verification_source_sha256": runtime_sources(output, ev, adapter_module)}
        write_json(output / "receipt.json", receipt)
    return receipt


@torch.inference_mode()
def verify(a, output, fixture_receipt):
    import evaluate_temporal_checkpoint as ev
    import temporal_deployment as adapter_module
    gpu = ev.check_gpu(a.gpu)  # Refuses occupied or unassigned GPUs.
    plan = ev.inspect_checkpoint(a.checkpoint, worktree=a.worktree, bank=a.bank,
                                 public_checkpoint=a.public_checkpoint, base=a.base, source_root=a.source_root)
    require(plan.payload["step"] == 2000, "Only predeclared terminal2000 is supported")
    mode = plan.manifest["arguments"]["history_mode"]
    common = plan.manifest["arguments"]["common_status"]
    identity = tune_identity(plan.base, fixture_receipt["row"])
    for key in ("row", "scene", "frame", "session", "split_sha256", "ego_sha256"):
        require(identity[key] == fixture_receipt[key], "Raw fixture differs from pinned row/source")
    with ev.isolated_runtime(plan.source) as runtime:
        model = ev.strict_load(plan, runtime).to("cuda:0").eval()
        dataset, data_provenance = ev.build_dataset(plan, runtime)
        batch = cache_sample(dataset, plan.rows, fixture_receipt["row"])
        cached_cpu = ev.input_tensors(batch, runtime, common, "cpu")
        with adapter_module.TemporalRawInputAdapter(mode, common, "selection", a.camera_workers) as adapter:
            prepared = adapter.prepare_clip(a.fixture)
            raw_cpu = {**prepared.inputs, **prepared.selector_inputs}
            input_parity = tensor_comparison(raw_cpu, cached_cpu)
            raw_gpu = {k: v.to("cuda:0") for k, v in raw_cpu.items()}
            cached_gpu = {k: v.to("cuda:0") for k, v in cached_cpu.items()}
            with ev.route_monitor(model, common) as route_counts:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw_out = model(**raw_gpu)
                    cached_out = model(**cached_gpu)
            ev.verify_rows(model, raw_out)
            ev.verify_rows(model, cached_out)
            fields = ("trajectory", "selected_candidate_id", "candidate_ids", "candidate_xy", "candidate_valid",
                      "scores", "base_scores", "aux_state", "aux_occ", "aux_lane")
            output_parity = tensor_comparison({k: raw_out[k] for k in fields}, {k: cached_out[k] for k in fields})
            # Always preserve actual outputs before any parity decision.
            arrays = {"rows": np.asarray([fixture_receipt["row"]], np.int64)}
            for name, out in (("raw", raw_out), ("cache", cached_out)):
                for field in fields:
                    value = out[field].detach().cpu()
                    arrays[name + "_" + field] = value.float().numpy() if value.dtype == torch.bfloat16 else value.numpy()
            np.savez_compressed(output / "outputs.npz", **arrays)
            route_audit = ev.sample_route_audit(model, raw_gpu, common)
            # No route hooks or bank checks inside timed model forwards.
            model_timing = ev.profile_model(model, raw_gpu, a.model_warmup, a.model_iterations)
            preprocessing_timing = preprocessing_profile(adapter, a.fixture, a.preprocess_warmup, a.preprocess_iterations)
        receipt = {"status": "completed", "mode": "terminal_B1_raw_vs_cache", "row": fixture_receipt["row"],
                   "checkpoint": plan.receipt, "gpu": gpu, "precision": "bf16_base_fp32_relative_head",
                   "batch_size": 1, "history_mode": mode, "common_status": common, "goal_mode": "selection",
                   "arm_configuration_source": "strict terminal checkpoint manifest only; no CLI arm overrides",
                   "input_parity": input_parity, "output_parity": output_parity,
                   "all_inputs_bitwise_equal": all(v["bitwise_equal"] for v in input_parity.values()),
                   "all_outputs_bitwise_equal": all(v["bitwise_equal"] for v in output_parity.values()),
                   "selected_trajectory_bitwise_equal": output_parity["trajectory"]["bitwise_equal"],
                   "selected_id_equal": output_parity["selected_candidate_id"]["bitwise_equal"],
                   "observed_route_counts": route_counts, "route_audit": route_audit,
                   "raw_adapter": prepared.metadata, "dataset_provenance": data_provenance,
                   "preprocessing_latency": preprocessing_timing, "model_only_latency": model_timing,
                   "fixture_receipt_sha256": sha(Path(a.fixture) / "fixture_receipt.json"),
                   "outputs_sha256": sha(output / "outputs.npz"),
                   "verification_source_sha256": runtime_sources(output, ev, adapter_module),
                   "GT_inputs_passed_to_model": False, "GT_metrics_computed": False,
                   "cache_reference_loader_contains_outer_labels": True,
                   "cache_reference_future_labels_materialized_by_dataset": True,
                   "reference_label_values_used_in_comparison_or_forward": False, "rtx4090_measured": False,
                   "runtime_environment": runtime_environment(plan.base)}
        write_json(output / "receipt.json", receipt)
    return receipt


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare-fixture", allow_abbrev=False)
    build.add_argument("--base", default=BASE)
    build.add_argument("--row", type=int, default=ROW)
    group = build.add_mutually_exclusive_group(required=True)
    group.add_argument("--output")
    group.add_argument("--tar-stdout", action="store_true")
    for name in ("preprocess", "verify"):
        p = commands.add_parser(name, allow_abbrev=False)
        p.add_argument("--fixture", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--base", default=BASE)
        p.add_argument("--worktree")
        p.add_argument("--source-root", required=name == "preprocess")
        p.add_argument("--camera-workers", type=int, default=3, choices=(1, 3))
        p.add_argument("--opencv-threads", type=int, default=1)
        p.add_argument("--preprocess-warmup", type=int, default=2)
        p.add_argument("--preprocess-iterations", type=int, default=10)
        if name == "preprocess":
            p.add_argument("--history-mode", choices=("repeat", "real"), required=True)
            p.add_argument("--common-status", action="store_true")
        else:
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--gpu", type=int, required=True, choices=(0, 1, 4))
            p.add_argument("--bank")
            p.add_argument("--public-checkpoint")
            p.add_argument("--model-warmup", type=int, default=10)
            p.add_argument("--model-iterations", type=int, default=50)
    return parser.parse_args()


def main():
    a = arguments()
    if getattr(a, "worktree", None):
        sys.path.insert(0, str(Path(a.worktree).resolve()))
    torch.set_num_threads(4)
    if a.command == "prepare-fixture":
        files, receipt = collect_fixture(a.base, a.row)
        publish_fixture(files, a.output, a.tar_stdout)
        if not a.tar_stdout:
            print(json.dumps({"fixture": str(Path(a.output).resolve()), "row": receipt["row"]}))
        return
    import cv2
    require(a.opencv_threads >= 1, "OpenCV thread count must be positive")
    cv2.setNumThreads(a.opencv_threads)
    require(cv2.getNumThreads() == a.opencv_threads, "OpenCV thread configuration was not applied")
    output = Path(a.output).resolve()
    require(not output.exists(), "Refusing to overwrite verification output")
    fixture = validate_fixture(a.fixture)
    output.mkdir(parents=True)
    (output / "verification_source").mkdir()
    before = time.time()
    try:
        receipt = preprocess(a, output, fixture) if a.command == "preprocess" else verify(a, output, fixture)
        receipt["elapsed_seconds"] = time.time() - before
        write_json(output / "receipt.json", receipt)
        print(json.dumps({k: receipt[k] for k in ("status", "mode", "row", "all_inputs_bitwise_equal")}, indent=2))
    except BaseException as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                              "elapsed_seconds": time.time() - before, "script_sha256": sha(__file__)})
        raise


if __name__ == "__main__":
    main()
