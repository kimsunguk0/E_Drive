#!/usr/bin/env python3
"""P3 step-zero failure diagnostic: same-process B4/B2 forward comparison only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import default_collate
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from models.motiondrive_v2_query_adapter import migrate_legacy_checkpoint
from scripts.motiondrive_v2_data import MotionDriveDataset
from scripts.motiondrive_v2_training import model_inputs, to_device, weighted_d3
from scripts.train_motiondrive_v2 import configure_cuda_memory, check_cuda_headroom, cuda_memory_snapshot, sha256
from scripts.train_motiondrive_v2_query_adapter import (validate_cuda_namespace, validate_initial_payload,
    validate_data_paths, configure_numerics, configure_planner_training)
from scripts.run_motiondrive_v2_query_trial import INIT, INIT_SHA, source_snapshot, require
from scripts.probe_motiondrive_v2_query_initial import OUTPUTS, exact_tensor_equal

def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--expected-git-sha", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    require(not os.path.lexists(args.out), "Existing output forbidden")
    out = Path(args.out).resolve()
    require(out.is_relative_to((ROOT / "reports").resolve()) and out.parent.is_dir(), "New reports path required")
    source = source_snapshot(ROOT, args.expected_git_sha)
    own_sha = sha256(__file__)
    a = argparse.Namespace(split_manifest=str(ROOT / "data/etri/motiondrive_v2/grouped_split_rawtime.json"),
        supervision_root=str(ROOT / "data/etri/motiondrive_v2/train_tune_geometry_v2"),
        init=str(ROOT / INIT), expected_init_sha256=INIT_SHA)
    data = validate_data_paths(a)
    payload = torch.load(a.init, map_location="cpu", weights_only=False)
    validate_initial_payload(payload)
    device = torch.device("cuda:0")
    namespace = validate_cuda_namespace(device)
    memory = configure_cuda_memory(device, 12000, 8192)
    numerics = configure_numerics()
    legacy = MotionDriveV2(MotionDriveV2Config(**payload["manifest"]["model_config"]))
    legacy.load_state_dict(payload["model"], strict=True)
    legacy.to(device).eval()
    query, migration = migrate_legacy_checkpoint(a.init, INIT_SHA, adapter_seed=0, adapter_on=False, device="cpu")
    query.to(device).eval()
    configure_planner_training(query)
    query.eval()
    dataset = MotionDriveDataset(data_root=str(ROOT), split_manifest=a.split_manifest,
        supervision_root=a.supervision_root, split="tune", min_frame=30, frame_stride=5, augment=False, seed=0)
    require(len(dataset) == 1998, "Expected same tune1998")
    old_path = ROOT / "reports/p2_c1t1_last3000_motion_diagnostic.json"
    old_sha = "09d9b8b50048a0dce55827f68b70ba3a63fcc5bd6eb901b5eee9fc571499d67f"
    require(sha256(old_path) == old_sha, "Original raw report changed")
    old = json.loads(old_path.read_text())
    # The actual raw result stores conditions under its documented result mapping.
    print(json.dumps({"old_report_top_level_keys": sorted(old)}), flush=True)
    batches = []
    with torch.inference_mode():
        for ids in ([0,1,2,3], [4,5,6,7], [1996,1997]):
            check_cuda_headroom(device, 8192)
            raw = default_collate([dataset[i] for i in ids])
            b = to_device(raw, device)
            inputs = model_inputs(b, time_input="nominal")
            outputs = {}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs["legacy_all_trainable"] = legacy(**inputs)
                for name, param in legacy.named_parameters():
                    param.requires_grad_(name.startswith("planner."))
                outputs["legacy_frozen"] = legacy(**inputs)
                query.query_adapter_on = False
                outputs["control_frozen"] = query(**inputs)
                query.query_adapter_on = True
                outputs["state_frozen"] = query(**inputs)
                for param in legacy.parameters():
                    param.requires_grad_(True)
            ref = outputs["legacy_all_trainable"]
            comparisons = {}
            for arm, result in outputs.items():
                comparisons[arm] = {name: {"bitwise_equal": exact_tensor_equal(ref[name], result[name]),
                    "max_abs_difference": float((ref[name].float()-result[name].float()).abs().max())}
                    for name in OUTPUTS}
                comparisons[arm]["d3"] = weighted_d3(result["plan_abs"], b["gt_plan"]).cpu().tolist()
                comparisons[arm]["plan_abs"] = {**comparisons[arm]["plan_abs"], "values": result["plan_abs"].cpu().tolist()}
            batches.append({"indices": ids, "rows": raw["row"].tolist(), "scenario": raw["scenario"],
                "frame": raw["frame"].tolist(), "comparisons": comparisons})
    require(validate_data_paths(a) == data and sha256(old_path) == old_sha and sha256(__file__) == own_sha
            and source_snapshot(ROOT,args.expected_git_sha) == source, "Source/data changed")
    report = {"status": "diagnostic_completed_not_training", "pid": os.getpid(), "source": source,
        "probe_sha256": own_sha, "data": data, "old_report_sha256": old_sha, "device_mapping": namespace,
        "memory_policy": memory, "memory": cuda_memory_snapshot(device), "numerics": numerics,
        "migration": migration, "full_forward_count": 12, "optimizer_steps": 0, "batches": batches}
    with out.open("x") as f:
        json.dump(report,f,ensure_ascii=False,allow_nan=False,indent=2)
        f.write("\n")
    print(json.dumps({"status":report["status"],"report":str(out)}))
if __name__ == "__main__":
    main()
