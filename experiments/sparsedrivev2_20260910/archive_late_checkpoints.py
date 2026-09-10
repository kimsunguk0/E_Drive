"""Retain predeclared late checkpoints without modifying a running trainer.

The trainer publishes last.pth with atomic rename. Copying an opened source
therefore obtains one complete checkpoint; its embedded step is authoritative.
No model inference, training, CUDA call, or held-population evaluation occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--step", type=int, action="append", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root, directory = Path(args.root).resolve(), Path(args.directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    steps = set(args.step)
    if any(step < 1 for step in steps) or len(set(args.run)) != len(args.run):
        raise ValueError("Unique run names and positive steps required")
    started = time.time()
    records = {name: {} for name in args.run}
    state = {"pid": os.getpid(), "status": "running", "started_unix": started,
             "steps": sorted(steps), "runs": records}
    status = directory / "status.json"
    write(status, state)
    try:
        while time.time() - started < args.timeout_seconds:
            complete = True
            for name in args.run:
                missing = steps - {int(s) for s in records[name]}
                if not missing:
                    continue
                complete = False
                run = root / "work_dirs/sparsedrivev2_20260910" / name
                progress = run / "progress.json"
                launch = root / "reports/sparsedrivev2_20260910/launches" / (name + ".json")
                receipt = json.loads(launch.read_text())
                observed_step = json.loads(progress.read_text())["step"] if progress.exists() else 0
                if observed_step in missing:
                    source = run / "last.pth"
                    temporary = directory / (name + ".copying.pth")
                    with source.open("rb") as stream, temporary.open("xb") as destination:
                        shutil.copyfileobj(stream, destination)
                    checkpoint = torch.load(temporary, map_location="cpu", weights_only=True)
                    step = int(checkpoint["step"])
                    if step not in missing:
                        temporary.unlink()
                        continue
                    if Path(checkpoint["manifest"]["arguments"]["run_dir"]).resolve() != run.resolve():
                        raise ValueError("Embedded run path differs from the requested run")
                    destination = directory / f"{name}_step{step:06d}.pth"
                    if destination.exists():
                        raise FileExistsError(destination)
                    temporary.replace(destination)
                    record = {"path": str(destination), "sha256": sha(destination),
                              "bytes": destination.stat().st_size, "embedded_step": step,
                              "source": str(source), "archived_unix": time.time(),
                              "source_git_sha": checkpoint["manifest"]["git_sha"],
                              "bank_sha256": checkpoint["manifest"]["bank_sha256"],
                              "validation_result": checkpoint["result"]}
                    records[name][str(step)] = record
                    write(destination.with_suffix(".json"), record)
                    write(status, state)
                    print(json.dumps({"archived": name, "step": step, "sha256": record["sha256"]}), flush=True)
                    del checkpoint
                # A receipt is only a hint. Recheck the OS process before treating
                # an unfinished run as absent; never restart any training process.
                if receipt["status"] in ("failed", "completed"):
                    still_missing = steps - {int(s) for s in records[name]}
                    if still_missing:
                        raise RuntimeError(f"Terminal run {name} lacks checkpoints {sorted(still_missing)}")
                else:
                    process = Path("/proc") / str(receipt["child_pid"]) / "cmdline"
                    if not process.exists():
                        # Supervisor may be publishing its terminal receipt.
                        state["last_observation"] = f"{name}: child disappeared; rechecking receipt next poll"
            if complete:
                state.update(status="completed", finished_unix=time.time())
                write(status, state)
                return
            time.sleep(5)
        raise TimeoutError("Archive observation deadline expired; training was left unchanged")
    except BaseException as exc:
        state.update(status="failed", finished_unix=time.time(), error=repr(exc))
        write(status, state)
        raise


if __name__ == "__main__":
    main()
