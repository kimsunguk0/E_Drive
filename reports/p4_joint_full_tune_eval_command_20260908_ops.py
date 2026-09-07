#!/usr/bin/env python3
"""One-shot P4 LAST6000 normal/full-tune evaluator command receipt.

This report-local command imports the existing reviewed GPU4/5 supervisor and
CUDA memory helpers.  It does not alter training code or reinterpret the
preserved joint-training supervisor return codes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))

from scripts.run_motiondrive_v2_query_trial import supervise
from scripts.evaluate_motiondrive_v2_shared import canonical_cuda_uuid
from scripts.evaluate_motiondrive_v2_planning import arguments as evaluation_arguments

TRAINING_GIT = "86620b4ffc7e6838b49cf83b5be789eba12d8027"
VALIDATION_GIT = "095e49d4e7be36802bad2f5923b873e65115e7fc"
EXPECTED = {
    0: {
        "last": "3e9ae5aa6ca89f4eb9a72ba38355dfdbec4ccaa38eabbebb3526fa3019404478",
        "manifest": "001b822dde79024f0e2cd52b5b99cb1096bbab5bfd09ef2d95e963ad8dc6e3de",
    },
    1: {
        "last": "c937f2f772c78f99cf3cfa5b1a96614267e20a9a667a157f4ada284940f9672e",
        "manifest": "49dc09c8e7840a059ae6bcadba1c490052847c678dc81d5524ee39f9da159aad",
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


CHILD = r'''import hashlib,json,os,sys
from pathlib import Path
ROOT=Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0,str(ROOT))
def require(c,m):
    if not c: raise ValueError(m)
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()
record=json.loads(Path(sys.argv[1]).read_text())
req=record["request"]
receipt={"schema_version":1,"child_pid":os.getpid(),"status":"failed","gpu_used":True,
         "training_success_relabelled":False,"model_forward_source":"existing evaluator normal condition"}
try:
    for key in ("output","protocol","child_receipt"):
        require(not os.path.lexists(req[key]),"Refusing existing evaluation artifact: "+req[key])
    require(sha256(req["checkpoint"])==req["checkpoint_sha256"],"Checkpoint changed before evaluation")
    require(sha256(req["sidecar"])==req["sidecar_sha256"],"Sidecar changed before evaluation")
    require(all(sha256(ROOT/name)==value for name,value in req["runtime_source_sha256"].items()),
            "Runtime source bytes changed before evaluation")
    require(all(sha256(ROOT/name)==value for name,value in req["data_sha256"].items()),
            "Data bytes changed before evaluation")
    import torch
    from scripts.train_motiondrive_v2 import configure_cuda_memory,cuda_memory_snapshot
    from scripts.evaluate_motiondrive_v2_planning import main as evaluate
    from scripts.evaluate_motiondrive_v2_shared import canonical_cuda_uuid
    device=torch.device("cuda:0")
    require(torch.cuda.device_count()==1,"Single-GPU namespace required")
    torch.cuda.set_device(device)
    raw=str(torch.cuda.get_device_properties(device).uuid)
    observed=canonical_cuda_uuid(raw)
    require(os.environ.get("CUDA_VISIBLE_DEVICES")==req["gpu_uuid"]==observed,"GPU UUID isolation mismatch")
    policy=configure_cuda_memory(device,12000,8192)
    before=cuda_memory_snapshot(device)
    rc=evaluate(req["evaluation_argv"])
    after=cuda_memory_snapshot(device)
    require(rc==0,"Existing evaluator returned nonzero")
    require(sha256(req["checkpoint"])==req["checkpoint_sha256"],"Checkpoint changed after evaluation")
    require(sha256(req["sidecar"])==req["sidecar_sha256"],"Sidecar changed after evaluation")
    require(all(sha256(ROOT/name)==value for name,value in req["runtime_source_sha256"].items()),
            "Runtime source bytes changed after evaluation")
    require(all(sha256(ROOT/name)==value for name,value in req["data_sha256"].items()),
            "Data bytes changed after evaluation")
    receipt.update(status="completed",main_returncode=0,allocator_configured_before_main=True,
                   memory_policy=policy,before_evaluation=before,after_evaluation=after,
                   device_mapping={"physical_gpu":req["physical_gpu"],"logical_device":"cuda:0",
                                   "observed_uuid":observed,"observed_uuid_raw":raw,
                                   "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")})
except BaseException as exc:
    receipt["error"]=f"{type(exc).__name__}: {exc}"
    raise
finally:
    with Path(req["child_receipt"]).open("x",encoding="utf-8") as f:
        json.dump(receipt,f,indent=2,ensure_ascii=False,allow_nan=False);f.write("\n")
'''


def validate_result(request, child_pid):
    require(sha256(request["checkpoint"]) == request["checkpoint_sha256"], "Checkpoint changed")
    require(sha256(request["sidecar"]) == request["sidecar_sha256"], "Sidecar changed")
    require(all(sha256(ROOT / name) == value for name, value in request["runtime_source_sha256"].items()),
            "Runtime source bytes changed")
    require(all(sha256(ROOT / name) == value for name, value in request["data_sha256"].items()),
            "Data bytes changed")
    receipt = json.loads(Path(request["child_receipt"]).read_text())
    require(receipt.get("status") == "completed" and receipt.get("child_pid") == child_pid
            and receipt.get("main_returncode") == 0 and receipt.get("allocator_configured_before_main") is True,
            "Child evaluation receipt mismatch")
    policy = receipt.get("memory_policy", {})
    require(policy.get("enabled") is True and policy.get("allocator_limit_mib") == 12000
            and policy.get("min_free_mib") == 8192, "Allocator policy mismatch")
    device = receipt.get("device_mapping", {})
    require(device.get("physical_gpu") == request["physical_gpu"] and device.get("logical_device") == "cuda:0"
            and device.get("observed_uuid") == request["gpu_uuid"]
            and canonical_cuda_uuid(device.get("observed_uuid_raw")) == request["gpu_uuid"],
            "Observed GPU identity mismatch")
    result = json.loads(Path(request["output"]).read_text())
    protocol = json.loads(Path(request["protocol"]).read_text())
    require(result.get("status") == "completed" and result.get("selection_performed") is False
            and result.get("final_val_accessed") is False and list(result.get("conditions", {})) == ["normal"],
            "Expected one normal full-tune result")
    require(protocol.get("arguments") == vars(evaluation_arguments(request["evaluation_argv"])),
            "Protocol arguments differ from request")
    require(protocol.get("checkpoint_sha256") == request["checkpoint_sha256"]
            and protocol.get("checkpoint_step") == 6000, "Protocol checkpoint mismatch")
    source = protocol.get("source", {})
    require(source.get("git_sha") == request["validation_git_sha"], "Validation git attribution mismatch")
    require(all(source.get("file_sha256", {}).get(name) == value
                for name, value in request["runtime_source_sha256"].items()
                if name in source.get("file_sha256", {})), "Protocol runtime source hash mismatch")
    data = protocol.get("data", {})
    require(data.get("receiver_count") == 1998, "Full tune row count mismatch")
    records = result["conditions"]["normal"].get("records", [])
    require(len(records) == 1998 and all("pred_state" in row and "pred_history" in row for row in records),
            "Expected 1998 same-forward motion predictions")
    return {"output_sha256": sha256(request["output"]),
            "protocol_sha256": sha256(request["protocol"]),
            "child_receipt_sha256": sha256(request["child_receipt"]),
            "checkpoint_sha256": request["checkpoint_sha256"],
            "sidecar_sha256": request["sidecar_sha256"],
            "training_git_sha": request["training_git_sha"],
            "validation_git_sha": request["validation_git_sha"],
            "runtime_source_snapshot_matches": True,
            "normal_summary": result["conditions"]["normal"]["summary"],
            "training_supervisor_failure_preserved": True,
            "training_success_relabelled": False}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(4, 5), required=True)
    args = parser.parse_args()
    require(args.physical_gpu == 4 + args.seed, "Fixed seed/GPU assignment required")
    seed = args.seed
    directory = ROOT / f"reports/p4_joint_full_tune_eval_s{seed}_20260908_ops"
    require(not os.path.lexists(directory), f"Refusing existing output directory: {directory}")
    directory.mkdir()
    checkpoint = ROOT / f"work_dirs/motiondrive_v2/p4_fresh_joint_s{seed}/last.pth"
    sidecar = checkpoint.with_name("manifest.json")
    require(sha256(checkpoint) == EXPECTED[seed]["last"], "Pinned LAST changed")
    require(sha256(sidecar) == EXPECTED[seed]["manifest"], "Pinned sidecar changed")
    side = json.loads(sidecar.read_text())
    require(side.get("status") == "completed" and side.get("step") == 6000
            and side.get("nonfinite_count") == 0 and side.get("git_sha") == TRAINING_GIT,
            "Completed trainer sidecar mismatch")
    training_record = json.loads((ROOT / f"logs/motiondrive_v2/p4_fresh_joint_s{seed}.supervisor.json").read_text())
    require(training_record.get("actual_returncode") == 1 and training_record.get("completion_evidence") is None,
            "Original training HEAD-guard failure must remain preserved")
    runtime = training_record["request"]["sources"]["file_sha256"]
    data = training_record["request"]["data_sha256"]
    require(all(sha256(ROOT / name) == value for name, value in runtime.items()), "Runtime bytes changed")
    require(all(sha256(ROOT / name) == value for name, value in data.items()), "Data bytes changed")
    require(subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
            == VALIDATION_GIT, "Unexpected validation git at launch")
    output = directory / "normal_full_tune.json"
    protocol = directory / "normal_full_tune.protocol.json"
    execution = directory / "execution.json"
    child_receipt = directory / "child_receipt.json"
    log = directory / "stdout.log"
    evaluator_argv = ["--checkpoint", str(checkpoint), "--data-root", str(ROOT),
        "--split-manifest", str(ROOT / "data/etri/motiondrive_v2/grouped_split_rawtime.json"),
        "--supervision-root", str(ROOT / "data/etri/motiondrive_v2/train_tune_geometry_v2"),
        "--split", "tune", "--out", str(output), "--conditions", "normal", "--seed", str(seed),
        "--frame-stride", "5", "--max-samples", "0", "--batch", "4", "--workers", "4",
        "--device", "cuda:0", "--precision", "bf16", "--time-input", "nominal",
        "--include-motion-predictions"]
    request = {"schema_version": 1, "root": str(ROOT), "seed": seed, "physical_gpu": args.physical_gpu,
        "record": str(execution), "log": str(log), "manifest_path": str(directory / "unused_trainer_manifest.json"),
        "child_receipt": str(child_receipt), "output": str(output), "protocol": str(protocol),
        "checkpoint": str(checkpoint), "checkpoint_sha256": EXPECTED[seed]["last"],
        "sidecar": str(sidecar), "sidecar_sha256": EXPECTED[seed]["manifest"],
        "runtime_source_sha256": runtime, "data_sha256": data,
        "training_git_sha": TRAINING_GIT, "validation_git_sha": VALIDATION_GIT,
        "evaluation_argv": evaluator_argv,
        "memory_policy": {"allocator_limit_mib": 12000, "reserve_mib": 8192},
        "command": [sys.executable, "-c", CHILD, str(execution)]}
    result = supervise(request, validator=validate_result)
    print(json.dumps({"outcome": result["outcome"], "actual_returncode": result["actual_returncode"],
                      "supervisor_exit_code": result["supervisor_exit_code"],
                      "parent_pid": result["parent_pid"], "child_pid": result["child_pid"],
                      "record": str(execution)}, sort_keys=True))
    return result["supervisor_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
