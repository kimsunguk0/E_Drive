"""Durable bounded controller: terminal train171 base -> cache -> matched CE head.

The existing GPU4 base job is only observed. New work is restricted to GPU1.
Neither a held dataset nor held evaluation nor a candidate approval record exists
in this controller. Internal experimental selection remains root-owned.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from evaluate_checkpoint import inspect_checkpoint, file_sha, require, CONFIRM_TRAIN_SHA, TUNE_SHA

SAMPLED_TRAIN_SHA = "7d12c0a65549951358390588c75e7dc208db6c802dab6493ce72b58f3f2f1d3b"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def check_sources(protocol):
    for path, expected in protocol["frozen_source_sha256"].items():
        require(file_sha(path) == expected, f"Frozen dependency changed: {path}")


def check_base(protocol):
    require(file_sha(protocol["base_manifest"]) == protocol["base_manifest_sha256"], "Base training manifest changed")
    manifest = read(protocol["base_manifest"])
    for key, expected in protocol["base_arguments"].items():
        require(manifest["arguments"].get(key) == expected, f"Base run contract changed: {key}")
    require(manifest["train"]["allowed_rows_sha256"] == CONFIRM_TRAIN_SHA, "Base is not train171")
    require(manifest["bank_sha256"] == protocol["bank_sha256"], "Base bank changed")


def wait_job(path, state, state_path, phase, deadline, expected_gpu):
    absent = 0
    while time.monotonic() < deadline:
        # The launcher returns immediately; its detached supervisor writes the
        # initial receipt shortly afterwards. Allow that bounded startup race.
        if not Path(path).exists():
            absent += 1
            require(absent < 6, f"Launch receipt was not created: {path}")
            time.sleep(1)
            continue
        receipt = read(path)
        require(receipt["gpu"] == expected_gpu, "Observed job changed its assigned physical GPU")
        state.update(status="running", phase=phase, updated_unix=time.time(), observed_job=receipt)
        write(state_path, state)
        if receipt["status"] in ("completed", "failed"):
            require(receipt["status"] == "completed" and receipt.get("returncode") == 0,
                    f"Required job did not exit successfully: {path}")
            return receipt
        require(receipt["status"] == "running", f"Unexpected durable launch status: {receipt['status']}")
        alive = (Path("/proc") / str(receipt["child_pid"]) / "cmdline").exists()
        absent = 0 if alive else absent + 1
        require(absent < 3, "Observed job disappeared without a successful terminal receipt")
        time.sleep(20)
    raise TimeoutError("Controller deadline exceeded; existing jobs were left untouched")


def launch(protocol, stage, state, state_path, deadline):
    check_sources(protocol)
    record = protocol[stage]
    require(not Path(record["receipt"]).exists(), f"Launch receipt already exists: {record['receipt']}")
    command = [protocol["python"], protocol["launcher"], "--gpu", "1", "--name", record["name"],
               "--directory", protocol["launch_directory"], "--", *record["command"]]
    completed = subprocess.run(command, cwd=protocol["worktree"], check=True, text=True, capture_output=True)
    state[f"{stage}_launch"] = json.loads(completed.stdout)
    write(state_path, state)
    return wait_job(record["receipt"], state, state_path, f"{stage}_running", deadline, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=10800)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    protocol_path = Path(args.protocol).resolve()
    protocol = read(protocol_path)
    require(protocol["schema"] == "confirmation_none_ce_pipeline_v1", "Unknown pipeline protocol")
    require(protocol["gpu"] == 1 and protocol["held_evaluation_enabled"] is False, "Pipeline scope changed")
    directory = protocol_path.parent
    state_path = directory / "status.json"
    require(not state_path.exists(), "Pipeline status already exists; no implicit restart")
    state = {"pid": os.getpid(), "status": "running", "phase": "preflight", "started_unix": time.time(),
             "protocol_sha256": file_sha(protocol_path), "controller_sha256": file_sha(__file__),
             "held_population_evaluated": False}
    write(state_path, state)
    deadline = time.monotonic() + args.timeout_seconds
    try:
        check_sources(protocol)
        check_base(protocol)
        base_launch = wait_job(protocol["base_launch_receipt"], state, state_path, "waiting_base", deadline, 4)
        check_sources(protocol)
        check_base(protocol)
        plan = inspect_checkpoint(protocol["base_checkpoint"], train_rows_path=protocol["train_rows"])
        require(plan.payload["step"] == 2000 and plan.goal_mode == "none", "Only the selected terminal none base is enabled")
        require(plan.receipt["fit_rows_sha256"] == CONFIRM_TRAIN_SHA, "Terminal base fit population changed")
        require(plan.receipt["bank_sha256"] == protocol["bank_sha256"], "Terminal base bank changed")
        reference = read(protocol["base_reference_json"])
        require(reference["step"] == 2000 and reference["n"] == 1998, "Terminal tune reference changed")
        state["base"] = {"path": protocol["base_checkpoint"], "sha256": plan.receipt["checkpoint_sha256"],
                         "manifest_sha256": protocol["base_manifest_sha256"], "bank_sha256": plan.receipt["bank_sha256"],
                         "step": 2000, "terminal_tune": plan.payload["result"],
                         "reference_sha256": file_sha(protocol["base_reference_npz"]), "launch": base_launch}
        del plan
        write(state_path, state)
        state["cache_exit"] = launch(protocol, "cache", state, state_path, deadline)
        cache_path = Path(protocol["cache_directory"])
        cache = read(cache_path / "manifest.json")
        require(cache["status"] == "completed" and cache["checkpoint"]["sha256"] == state["base"]["sha256"], "Cache base provenance mismatch")
        require(cache["bank"]["sha256"] == protocol["bank_sha256"], "Cache bank mismatch")
        require(cache["allowed_train_rows_sha256"] == CONFIRM_TRAIN_SHA and cache["sampled_train_rows_sha256"] == SAMPLED_TRAIN_SHA,
                "Cache train171 row identities changed")
        require(cache["tune_rows_sha256"] == TUNE_SHA and cache["splits"]["train"]["rows"] == 9234
                and cache["splits"]["tune"]["rows"] == 1998, "Cache row counts changed")
        require(cache["splits"]["tune"]["reference_parity"]["sha256"] == state["base"]["reference_sha256"], "Cache lacks exact terminal base parity")
        state["cache"] = {"path": str(cache_path), "manifest_sha256": file_sha(cache_path / "manifest.json"),
                          "train_rows": 9234, "tune_rows": 1998, "sampled_train_rows_sha256": SAMPLED_TRAIN_SHA,
                          "tune_rows_sha256": TUNE_SHA, "reference_parity": cache["splits"]["tune"]["reference_parity"]}
        write(state_path, state)
        state["head_exit"] = launch(protocol, "head", state, state_path, deadline)
        head_path = Path(protocol["head_checkpoint"])
        head_sha = file_sha(head_path)
        payload = torch.load(head_path, map_location="cpu", weights_only=True)
        require(file_sha(head_path) == head_sha, "Head changed while finalizing")
        require(payload["step"] == 2000 and payload["result"]["step"] == 2000, "Head terminal step mismatch")
        for key, expected in protocol["head_arguments"].items():
            require(payload["manifest"]["arguments"][key] == expected, f"Matched head configuration changed: {key}")
        require(payload["manifest"]["cache_manifest_sha256"] == state["cache"]["manifest_sha256"], "Head cache provenance mismatch")
        require(payload["manifest"]["source_sha256"]["train_cached_selector.py"] == protocol["head_source_sha256"], "Head source changed")
        state["head"] = {"path": str(head_path), "sha256": head_sha, "step": 2000,
                         "terminal_tune": payload["result"], "source_sha256": payload["manifest"]["source_sha256"]}
        state.update(status="completed", phase="completed", finished_unix=time.time(),
                     elapsed_seconds=time.time() - state["started_unix"], gpu1_child_exited=True)
        write(state_path, state)
        write(directory / "result.json", state)
        print(json.dumps({"status": "completed", "base_sha256": state["base"]["sha256"],
                          "cache_manifest_sha256": state["cache"]["manifest_sha256"], "head_sha256": head_sha,
                          "terminal_tune_d3": state["head"]["terminal_tune"]["official_d3"]}), flush=True)
    except BaseException:
        state.update(status="failed", phase="failed", finished_unix=time.time(), traceback=traceback.format_exc())
        write(state_path, state)
        print(json.dumps({"status": "failed", "traceback": state["traceback"]}), flush=True)
        raise


if __name__ == "__main__":
    main()
