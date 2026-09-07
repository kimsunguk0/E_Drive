#!/usr/bin/env python3
"""Actual train8 legacy/control/treatment initial-function parity, not accuracy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import default_collate
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from models.motiondrive_v2_query_adapter import migrate_legacy_checkpoint
from scripts.motiondrive_v2_data import MotionDriveDataset
from scripts.motiondrive_v2_training import model_inputs, to_device, tensor_state_sha256
from scripts.train_motiondrive_v2 import configure_cuda_memory, check_cuda_headroom, cuda_memory_snapshot, sha256
from scripts.train_motiondrive_v2_query_adapter import (validate_cuda_namespace, validate_initial_payload,
    validate_data_paths)
from scripts.run_motiondrive_v2_query_trial import INIT, INIT_SHA, source_snapshot, require

OUTPUTS = ("plan_abs", "scene_features", "motion_features", "state_hat", "history_hat", "occ_logits", "lane_logits")


def exact_tensor_equal(left, right):
    """Equal dtype/shape and bytes, including signed zero; stride is immaterial."""
    return (left.dtype == right.dtype and left.shape == right.shape
            and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--expected-git-sha", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    require(not os.path.lexists(args.out), "Refusing an existing report or symlink")
    out = Path(args.out).resolve()
    require(out.is_relative_to((ROOT / "reports").resolve()) and not os.path.lexists(out), "New reports JSON required")
    require(out.suffix == ".json" and out.parent.is_dir(), "Invalid report path")
    sources = source_snapshot(ROOT, args.expected_git_sha)
    source_sha = sha256(Path(__file__))
    checkpoint = ROOT / INIT
    require(sha256(checkpoint) == INIT_SHA, "Source checkpoint changed")
    data_args = argparse.Namespace(
        split_manifest=str(ROOT / "data/etri/motiondrive_v2/grouped_split_rawtime.json"),
        supervision_root=str(ROOT / "data/etri/motiondrive_v2/train_tune_geometry_v2"),
        init=str(checkpoint), expected_init_sha256=INIT_SHA)
    data_evidence = validate_data_paths(data_args)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_initial_payload(payload)
    device = torch.device("cuda:0")
    namespace = validate_cuda_namespace(device)
    memory = configure_cuda_memory(device, 12000, 8192)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    legacy = MotionDriveV2(MotionDriveV2Config(**payload["manifest"]["model_config"]))
    legacy.load_state_dict(payload["model"], strict=True)
    legacy.to(device).eval()
    query, migration = migrate_legacy_checkpoint(checkpoint, INIT_SHA, adapter_seed=0, adapter_on=False, device="cpu")
    initial_sha = tensor_state_sha256(query.state_dict())
    query.to(device).eval()
    del payload
    dataset = MotionDriveDataset(data_root=str(ROOT),
        split_manifest=data_args.split_manifest,
        supervision_root=data_args.supervision_root,
        split="train", min_frame=30, frame_stride=1, augment=False, seed=0)
    scenes = sorted(set(dataset.scene_names[dataset.rows]))[:4]
    indices = []
    for scene in scenes:
        for frame in (30, 180):
            found = np.flatnonzero((dataset.scene_names[dataset.rows] == scene) & (dataset.arr["frame"][dataset.rows] == frame))
            require(len(found) == 1, "Exact train8 row required")
            indices.append(int(found[0]))
    require(len(indices) == 8, "Expected train8")
    records = []
    with torch.inference_mode():
        for index in indices:
            check_cuda_headroom(device, 8192)
            sample = dataset[index]
            batch = to_device(default_collate([sample]), device)
            inputs = model_inputs(batch, time_input="nominal")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                reference = legacy(**inputs)
                query.query_adapter_on = False
                control = query(**inputs)
                query.query_adapter_on = True
                state = query(**inputs)
            for name in OUTPUTS:
                require(torch.isfinite(reference[name]).all().item(), "Nonfinite reference output")
                require(exact_tensor_equal(reference[name], control[name]) and exact_tensor_equal(control[name], state[name]),
                        f"Initial physical output differs: {name}")
            require(all(reference[name].dtype == torch.float32 for name in ("plan_abs", "state_hat", "history_hat")),
                    "Final coordinates and neural state/history must be FP32")
            records.append({"scenario": sample["scenario"], "frame": int(sample["frame"]), "row": int(sample["row"]),
                            "outputs_bitwise_equal": list(OUTPUTS)})
    require(tensor_state_sha256(query.state_dict()) == initial_sha, "Probe changed query model state")
    require(sha256(checkpoint) == INIT_SHA and source_snapshot(ROOT, args.expected_git_sha) == sources
            and sha256(Path(__file__)) == source_sha, "Probe source/input changed")
    require(validate_data_paths(data_args) == data_evidence, "Probe data lineage changed")
    result = {"status": "initial_train8_full_forward_parity_pass", "pid": os.getpid(),
              "source": sources, "probe_source_sha256": source_sha, "source_checkpoint_sha256": INIT_SHA,
              "migration": migration, "initial_query_state_sha256": initial_sha, "device_mapping": namespace,
              "data": data_evidence, "comparison": "equal dtype, shape and contiguous uint8 bytes for all seven outputs",
              "cuda_memory_policy": memory, "cuda_memory": cuda_memory_snapshot(device),
              "precision": "bf16 encoders / FP32 planner", "full_forward_count": 24, "records": records,
              "split": "train", "training_or_optimizer_steps": 0, "labels_used_as_model_inputs": False,
              "accuracy_or_latency_claim": False, "final_val_accessed": False,
              "max_abs_output_difference": 0., "source_and_checkpoint_unchanged": True}
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "forward_count": 24, "report": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
