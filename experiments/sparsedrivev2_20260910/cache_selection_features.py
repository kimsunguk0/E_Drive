"""Cache fixed-candidate score features on allowed train rows and original tune.

The bank is checked against the complete allowed fit population before sampling
train frames at stride five. Held/confirmation rows are never an output split.
GT is used only for costs and gt_xy label artifacts, never as feature-builder or
base-model input. Dataset goal is available for score features; the base forward
receives it only when its frozen checkpoint was trained in goal-selection mode.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import inspect
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from evaluate_checkpoint import (inspect_checkpoint, recorded_runtime, load_verified_model,
                                 verify_bank_output, check_reference, require, file_sha, rows_sha, TUNE_SHA)

DTYPES = {"features": np.float32, "base_logits": np.float32, "costs": np.float32,
          "valid": np.bool_, "candidate_ids": np.int64, "candidate_xy": np.float32,
          "gt_xy": np.float32, "rows": np.int64, "goal_xy": np.float32, "status8": np.float32}
TAIL_SHAPES = {"features": (200, 32), "base_logits": (200,), "costs": (200,),
               "valid": (200,), "candidate_ids": (200,), "candidate_xy": (200, 6, 2),
               "gt_xy": (6, 2), "rows": (), "goal_xy": (2,), "status8": (8,)}


def json_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class PartitionWriter:
    """Bounded-RAM NPY writer with a final immutable per-partition receipt."""
    def __init__(self, root, partition, expected_rows, scenarios, sessions):
        require(partition in ("train", "tune"), "Only train/tune feature caches are enabled")
        self.path = Path(root) / partition
        self.path.mkdir(parents=True, exist_ok=False)
        self.partition = partition
        self.expected_rows = np.asarray(expected_rows, dtype=np.int64)
        require(len(self.expected_rows) and np.all(np.diff(self.expected_rows) > 0), "Expected cache rows must be sorted and unique")
        self.maps, self.cursor = {}, 0
        for key, dtype in DTYPES.items():
            self.maps[key] = np.lib.format.open_memmap(self.path / f"{key}.npy", mode="w+", dtype=dtype,
                                                       shape=(len(self.expected_rows),) + TAIL_SHAPES[key])
        self.identities = {"scenario": np.asarray(scenarios, dtype=str), "session": np.asarray(sessions, dtype=str)}
        require(all(len(x) == len(self.expected_rows) for x in self.identities.values()), "Identity array length mismatch")

    def append(self, values):
        require(set(values) == set(self.maps), "Cache batch fields differ from the fixed schema")
        arrays = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            require(array.dtype == np.dtype(DTYPES[key]), f"Cache field dtype changed: {key}")
            require(array.shape[1:] == TAIL_SHAPES[key], f"Cache field shape changed: {key}")
            if array.dtype.kind == "f":
                require(np.isfinite(array).all(), f"Nonfinite cache field: {key}")
            arrays[key] = array
        n = len(arrays["rows"])
        require(n > 0 and all(len(a) == n for a in arrays.values()), "Cache batch row counts differ")
        end = self.cursor + n
        require(end <= len(self.expected_rows), "Too many cache rows")
        require(np.array_equal(arrays["rows"], self.expected_rows[self.cursor:end]), "Cache batch row order mismatch")
        require(arrays["valid"].any(1).all(), "A cache row has no valid candidate")
        for key, value in arrays.items():
            self.maps[key][self.cursor:end] = value
        self.cursor = end

    def finish(self, metadata):
        require(self.cursor == len(self.expected_rows), "Cannot finalize an incomplete feature cache")
        for array in self.maps.values():
            array.flush()
        self.maps.clear()
        for key, values in self.identities.items():
            with (self.path / f"{key}.npy").open("xb") as stream:
                np.save(stream, values, allow_pickle=False)
        artifacts = {}
        for path in sorted(self.path.glob("*.npy")):
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            artifacts[f"{self.partition}/{path.name}"] = {"sha256": file_sha(path),
                                                           "shape": list(array.shape), "dtype": str(array.dtype)}
        record = {"partition": self.partition, "rows": self.cursor, "rows_sha256": rows_sha(self.expected_rows),
                  "scenes": len(set(self.identities["scenario"].tolist())),
                  "sessions": len(set(self.identities["session"].tolist())), **metadata,
                  "artifacts": artifacts}
        json_new(self.path / "metadata.json", record)
        return record, artifacts


def build_datasets(plan, runtime):
    """Preserve the bank/allowed-row contract, then select stride-five cache rows."""
    require(plan.population == "tune", "Feature caching never enables held population evaluation")
    require("goal_mode" in inspect.signature(runtime.data.PlanDataset).parameters,
            "Feature cache needs the recorded loader's provided-goal support")
    common = dict(base=plan.base, split_manifest=plan.split_path, ego_cache=plan.ego,
                  status_mode=plan.manifest["arguments"]["status_mode"], goal_mode="selection",
                  image_size=tuple(plan.manifest["train"]["image_wh"]), augment=False)
    train = runtime.data.PlanDataset(**common, split="train", stride=1, rows_file=plan.train_rows_path)
    tune = runtime.data.PlanDataset(**common, split="tune", stride=5)
    require(np.array_equal(train.allowed_rows, plan.train_rows), "Bank fit population changed before cache sampling")
    require(np.array_equal(train.rows, plan.train_rows), "Training cache must start from all allowed rows")
    require(rows_sha(tune.rows) == TUNE_SHA and len(tune) == 1998, "Original tune37 changed")
    sampled_indices = np.flatnonzero(train.frames[train.rows] % 5 == 0)
    expected_counts = {54810: 10962, 46170: 9234}
    require(len(train.allowed_rows) in expected_counts and len(sampled_indices) == expected_counts[len(train.allowed_rows)],
            "Expected either the original train203 or frozen confirmation-train171 population")
    train_meta, tune_meta = train.provenance(), tune.provenance()
    require(not set(train_meta["sessions"]) & set(tune_meta["sessions"]), "Train/tune cache sessions overlap")
    for dataset, actual, original in ((train, train_meta, plan.manifest["train"]),
                                      (tune, tune_meta, plan.manifest["validation"])):
        for key in ("ego_cache_sha256", "calibration_sha256", "split_sha256", "camera_order", "image_wh", "normalization"):
            require(actual[key] == original[key], f"Cache input contract changed: {key}")
        if actual["status_source"] is not None:
            require(actual["status_source"]["sha256"] == original["status_source"]["sha256"], "Causal status overlay changed")
    return {"train": (Subset(train, sampled_indices), train, train.rows[sampled_indices], train_meta),
            "tune": (tune, tune, tune.rows, tune_meta)}


@torch.inference_mode()
def cache_partition(root, partition, datasets, model, runtime, plan, feature_builder,
                    *, batch_size=16, workers=4, precision="bf16", device="cuda:0",
                    reference_predictions=None):
    subset, dataset, rows, dataset_meta = datasets
    scenarios = dataset.scenarios[dataset.scen_idx[rows]].astype(str)
    sessions = np.asarray([dataset.manifest["scene_to_session"][s] for s in scenarios])
    writer = PartitionWriter(root, partition, rows, scenarios, sessions)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    device = torch.device(device)
    require(precision in ("bf16", "fp32"), "Unknown cache precision")
    require(precision == "fp32" or device.type == "cuda", "BF16 cache generation requires CUDA")
    model.eval()
    reference_chunks = {}
    require(reference_predictions is None or partition == "tune", "Reference parity is only enabled for tune")
    for batch in loader:
        x = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
        with autocast:
            output = model(**runtime.data.model_inputs(x, goal_selection=plan.goal_mode == "selection"))
        verify_bank_output(model, output, runtime)
        candidate = output["candidate_xy"].float()
        require(candidate.shape[1:] == (200, 6, 2), "Cache schema requires the frozen 20x10 shortlist")
        # Explicit legal score inputs only. GT is deliberately absent here.
        feature_inputs = {"candidate_xy": candidate, "scores": output["scores"].float(),
                          "candidate_valid": output["candidate_valid"].bool()}
        features = feature_builder(feature_inputs, x["status"].float(), x["goal_xy"].float()).float()
        require(features.shape == (*candidate.shape[:2], 32), "Expected 32 candidate score features")
        # These labels are written separately and are never feature-builder inputs.
        costs = runtime.data.d3(candidate, x["gt_plan"][:, None].expand_as(candidate)).float()
        writer.append({"features": features, "base_logits": output["scores"].float(), "costs": costs,
                       "valid": output["candidate_valid"].bool(), "candidate_ids": output["candidate_ids"].long(),
                       "candidate_xy": candidate, "gt_xy": x["gt_plan"].float(), "rows": batch["row"].long(),
                       "goal_xy": x["goal_xy"].float(), "status8": x["status"].float()})
        if reference_predictions is not None:
            pred = output["trajectory"].float()
            delta = pred - x["gt_plan"].float()
            compare = {"rows": batch["row"], "pred": pred, "d3": runtime.data.d3(pred, x["gt_plan"]),
                       "shortlist_oracle": costs.masked_fill(~output["candidate_valid"].bool(), float("inf")).amin(-1),
                       "candidate_id": output["selected_candidate_id"], "error_xy": delta,
                       "point_l2": torch.linalg.vector_norm(delta, dim=-1)}
            for key, value in compare.items():
                reference_chunks.setdefault(key, []).append(value.detach().cpu().numpy())
        if writer.cursor == len(rows) or writer.cursor % (batch_size * 100) == 0:
            print(json.dumps({"cache_partition": partition, "rows_written": writer.cursor, "rows_total": len(rows)}), flush=True)
    metadata = {"base_checkpoint_sha256": plan.receipt["checkpoint_sha256"], "bank_sha256": plan.receipt["bank_sha256"],
                "allowed_train_rows_sha256": rows_sha(plan.train_rows), "frame_stride": 5,
                "base_goal_mode": plan.goal_mode, "goal_feature_mode": "selection",
                "status_mode": plan.manifest["arguments"]["status_mode"],
                "batch_size": batch_size, "precision": precision,
                "status_source_sha256": dataset_meta["status_source"]["sha256"] if dataset_meta["status_source"] else None,
                "goal_source_sha256": dataset_meta["ego_cache_sha256"], "dataset_provenance": dataset_meta,
                "label_only_artifacts": ["costs.npy", "gt_xy.npy"],
                "feature_inputs": ["candidate_xy", "base_logits", "candidate_valid", "status8", "provided_goal_xy"],
                "base_forward_goal_selection": plan.goal_mode == "selection"}
    if reference_predictions is not None:
        compare = {key: np.concatenate(values) for key, values in reference_chunks.items()}
        metadata["reference_parity"] = check_reference(compare, reference_predictions)
    return writer.finish(metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bank")
    parser.add_argument("--train-rows")
    parser.add_argument("--source-root")
    parser.add_argument("--public-checkpoint")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reference-predictions", help="Require tune base outputs to equal a saved eval NPZ bitwise")
    args = parser.parse_args()
    device = os.environ.get("CUDA_VISIBLE_DEVICES")
    require(device in ("0", "1", "4"), "Expose exactly one user-assigned GPU: 0, 1, or 4")
    require(args.batch > 0 and args.eval_batch > 0 and args.workers >= 0, "Invalid cache batch/workers")
    output = Path(args.output).resolve()
    require(not output.exists(), "Cache output already exists; use a new immutable cache path")
    torch.set_num_threads(4)
    plan = inspect_checkpoint(args.checkpoint, bank_path=args.bank, train_rows_path=args.train_rows,
                              source_root=args.source_root, public_checkpoint=args.public_checkpoint)
    import relative_selector
    feature_builder = relative_selector.make_candidate_features
    require(len(relative_selector.FEATURE_NAMES) == 32, "Feature names do not describe the fixed 32D schema")
    feature_source = Path(inspect.getsourcefile(feature_builder)).resolve()
    feature_sha = file_sha(feature_source)
    helper_source = Path(__file__).with_name("evaluate_checkpoint.py")
    helper_sha = file_sha(helper_source)
    started = time.time()
    output.mkdir(parents=True, exist_ok=False)
    try:
        with recorded_runtime(plan) as runtime:
            model, _ = load_verified_model(plan, runtime)
            datasets = build_datasets(plan, runtime)
            model.cuda()
            splits, artifacts = {}, {}
            for partition in ("train", "tune"):
                record, files = cache_partition(output, partition, datasets[partition], model, runtime, plan,
                                                 feature_builder, batch_size=args.batch if partition == "train" else args.eval_batch,
                                                 workers=args.workers,
                                                 reference_predictions=args.reference_predictions if partition == "tune" else None)
                splits[partition] = record
                artifacts.update(files)
        require(file_sha(feature_source) == feature_sha, "Feature source changed during cache generation")
        require(file_sha(helper_source) == helper_sha, "Checkpoint helper changed during cache generation")
        source_dir = output / "source"
        source_dir.mkdir()
        shutil.copy2(feature_source, source_dir / feature_source.name)
        shutil.copy2(__file__, source_dir / Path(__file__).name)
        shutil.copy2(helper_source, source_dir / helper_source.name)
        for partition in splits:
            path = output / partition / "metadata.json"
            splits[partition]["metadata_sha256"] = file_sha(path)
        manifest = {"schema": "sparsedrivev2_selection_cache_v1", "status": "completed",
                    "checkpoint": {"path": plan.receipt["checkpoint"], "sha256": plan.receipt["checkpoint_sha256"]},
                    "bank": {"path": str(plan.bank), "sha256": plan.receipt["bank_sha256"]},
                    "allowed_train_rows_sha256": rows_sha(plan.train_rows),
                    "sampled_train_rows_sha256": splits["train"]["rows_sha256"],
                    "tune_rows_sha256": splits["tune"]["rows_sha256"],
                    "feature_source_sha256": feature_sha, "feature_source_path": str(feature_source),
                    "feature_version": relative_selector.FEATURE_VERSION,
                    "feature_names": list(relative_selector.FEATURE_NAMES),
                    "feature_dim": 32, "candidate_count": 200, "goal_feature_mode": "selection",
                    "base_goal_mode": plan.goal_mode,
                    "feature_inputs": ["candidate_xy", "base_logits", "candidate_valid", "status8", "provided_goal_xy"],
                    "label_only_artifacts": ["costs.npy", "gt_xy.npy"],
                    "artifacts": artifacts, "splits": splits, "provenance": plan.receipt,
                    "cache_source_sha256": file_sha(__file__), "checkpoint_helper_sha256": helper_sha,
                    "physical_gpu": int(device), "batch": args.batch, "eval_batch": args.eval_batch, "precision": "bf16",
                    "workers": args.workers, "elapsed_seconds": time.time() - started,
                    "held_population_evaluated": False, "reserve_rows_cached": False,
                    "confirmation12_excluded_from_fit_population": len(plan.train_rows) == 46170}
        json_new(output / "manifest.json", manifest)
        print(json.dumps({"status": "completed", "output": str(output),
                          "train_rows": splits["train"]["rows"], "tune_rows": splits["tune"]["rows"],
                          "manifest_sha256": file_sha(output / "manifest.json")}), flush=True)
    except BaseException:
        json_new(output / "failure.json", {"status": "failed", "traceback": traceback.format_exc(),
                                           "elapsed_seconds": time.time() - started})
        raise


if __name__ == "__main__":
    main()
