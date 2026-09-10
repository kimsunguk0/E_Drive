"""CPU-only, predeclared averaging of learned parameters at 1500/1750/2000.

All persistent buffers, including bank and BatchNorm statistics, must be exactly
identical. Parameters accumulate in float64 and cast back once. No labels,
inference, optimizer averaging, or fabricated evaluation result is used.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

METHOD = "uniform learned-parameter average"
STEPS = [1500, 1750, 2000]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_receipt(path, expected_sha=None, expected_step=None):
    path = Path(path).resolve()
    candidates = [Path(str(path) + ".json"), path.with_suffix(".json")]
    present = [p for p in candidates if p.exists()]
    require(present, f"Missing archived checkpoint receipt: {path}")
    digest = sha(path)
    for receipt_path in present:
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get("sha256", receipt.get("sha")) == digest, "Archived checkpoint receipt hash mismatch")
        if expected_step is not None:
            require(receipt.get("embedded_step", receipt.get("step")) == expected_step, "Archived receipt step mismatch")
    if expected_sha is not None:
        require(digest == expected_sha, "Average source checkpoint changed")
    return digest


@torch.no_grad()
def average_state_dicts(states, parameter_names, buffer_names):
    """Small reusable numerical core; inputs may stream one state at a time."""
    parameters, buffers = set(parameter_names), set(buffer_names)
    require(not parameters & buffers, "Parameters and buffers overlap")
    expected = parameters | buffers
    require(parameters and expected, "Empty model classification")
    accumulator, copies, specifications, key_order = {}, {}, {}, None
    count = 0
    for state in states:
        require(set(state) == expected, "State keys differ from named parameters/buffers")
        if key_order is None:
            key_order = list(state)
        for name, value in state.items():
            require(isinstance(value, torch.Tensor) and value.device.type == "cpu", "Average requires CPU tensor states")
            require(not value.is_floating_point() or torch.isfinite(value).all().item(), f"Nonfinite source tensor: {name}")
            specification = (tuple(value.shape), value.dtype)
            if count:
                require(specifications[name] == specification, f"Source tensor shape/dtype changed: {name}")
            else:
                specifications[name] = specification
            if name in parameters:
                require(value.is_floating_point(), f"Learned parameter is not floating: {name}")
                if not count:
                    accumulator[name] = value.to(torch.float64).clone()
                else:
                    accumulator[name].add_(value.to(torch.float64))
            elif not count:
                copies[name] = value.clone()
            else:
                require(torch.equal(copies[name], value), f"Persistent buffer changed: {name}")
        count += 1
    require(count > 0, "No checkpoints to average")
    result = OrderedDict()
    for name in key_order:
        if name in parameters:
            result[name] = (accumulator.pop(name) / count).to(specifications[name][1])
            require(torch.isfinite(result[name]).all().item(), f"Nonfinite averaged parameter: {name}")
        else:
            result[name] = copies.pop(name)
    return result


def source_states(sources, manifest, epochs):
    """Validate receipts/lineage before yielding each CPU state."""
    for source in sources:
        path = Path(source["path"])
        source_receipt(path, source["sha256"], source["step"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        require(sha(path) == source["sha256"], "Source changed while loading")
        require(checkpoint["manifest"] == manifest, "Averaging requires exactly equal input manifests")
        require(checkpoint["step"] == source["step"], "Embedded averaging step mismatch")
        require("averaging" not in checkpoint and isinstance(checkpoint.get("result"), dict), "Average sources must be raw trained checkpoints")
        require(checkpoint["result"]["step"] == source["step"], "Source result step mismatch")
        epochs.append(checkpoint["epoch"])
        checkpoint.pop("optimizer", None)
        yield checkpoint["model"]
        del checkpoint


def validate_averaging_metadata(payload):
    metadata = payload["averaging"]
    require(metadata["version"] == 1 and metadata["method"] == METHOD, "Unknown averaging protocol")
    require(metadata["steps"] == STEPS and [s["step"] for s in metadata["sources"]] == STEPS,
            "Expected exactly the predeclared 1500/1750/2000 inputs")
    require(len({str(Path(s["path"]).resolve()) for s in metadata["sources"]}) == len(STEPS), "Repeated average source checkpoint")
    require(payload["step"] == STEPS[-1] and payload.get("result") is None and "optimizer" not in payload,
            "Derived average must have step2000, result=None, and no optimizer")
    require(sha(metadata["protocol_path"]) == metadata["protocol_sha256"], "Averaging protocol file changed")
    protocol = json.loads(Path(metadata["protocol_path"]).read_text())
    require(protocol["steps"] == STEPS, "Averaging steps differ from preregistration")
    require(Path(payload["manifest"]["arguments"]["run_dir"]).name in protocol["runs"], "Run is absent from averaging preregistration")
    require(sha(metadata["postprocessor_path"]) == metadata["postprocessor_sha256"], "Averaging postprocessor file changed")
    return metadata


def verify_averaged_state(payload, model):
    """Independently recompute the certified average before any label evaluation."""
    metadata = validate_averaging_metadata(payload)
    parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    epochs = []
    expected = average_state_dicts(source_states(metadata["sources"], payload["manifest"], epochs), parameters, buffers)
    require(set(payload["model"]) == set(expected), "Averaged output state keys differ")
    for name, value in expected.items():
        require(torch.equal(payload["model"][name], value), f"Averaged tensor differs from verified source mean/copy: {name}")
    require(payload["epoch"] == epochs[-1], "Averaged epoch must be copied from step2000")
    return {"source_steps": STEPS, "source_sha256": [s["sha256"] for s in metadata["sources"]],
            "parameter_tensors": len(parameters), "buffer_tensors": len(buffers),
            "recomputed_average_bitwise_equal": True, "all_buffers_bitwise_identical": True}


def write_immutable(path, payload):
    path = Path(path).resolve()
    require(not path.exists() and not Path(str(path) + ".json").exists(), "Average output/sidecar already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".average_", suffix=".tmp", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.link(temporary, path)  # Atomic publication which never overwrites.
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="Repeat in 1500,1750,2000 order")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--bank")
    parser.add_argument("--public-checkpoint")
    parser.add_argument("--train-rows")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(4)
    require(len(args.checkpoint) == len(STEPS), "Supply exactly three source checkpoints")
    require(not Path(args.output).exists() and not Path(args.output + ".json").exists(), "Average output already exists")
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    require(protocol["steps"] == STEPS, "Protocol must predeclare 1500/1750/2000")
    sources = [{"path": str(Path(p).resolve()), "sha256": source_receipt(p, expected_step=step), "step": step}
               for p, step in zip(args.checkpoint, STEPS)]
    from evaluate_checkpoint import inspect_checkpoint, recorded_runtime, load_verified_model
    plan = inspect_checkpoint(args.checkpoint[0], bank_path=args.bank, train_rows_path=args.train_rows,
                              source_root=args.source_root, public_checkpoint=args.public_checkpoint)
    require(Path(plan.manifest["arguments"]["run_dir"]).name in protocol["runs"], "Run was not preregistered")
    with recorded_runtime(plan) as runtime:
        model, _ = load_verified_model(plan, runtime)
        parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        epochs = []
        state = average_state_dicts(source_states(sources, plan.manifest, epochs), parameters, buffers)
        model.load_state_dict(state, strict=True)
    metadata = {"version": 1, "protocol_path": str(protocol_path), "protocol_sha256": sha(protocol_path),
                "method": METHOD, "steps": STEPS, "sources": sources,
                "postprocessor_path": str(Path(__file__).resolve()), "postprocessor_sha256": sha(__file__)}
    payload = {"model": state, "manifest": plan.manifest, "step": STEPS[-1], "epoch": epochs[-1],
               "result": None, "averaging": metadata}
    validate_averaging_metadata(payload)
    path = write_immutable(args.output, payload)
    receipt = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
               "step": STEPS[-1], "averaging": metadata, "parameter_tensors": len(parameters),
               "buffer_tensors": len(buffers), "all_buffers_bitwise_identical": True,
               "GPU_used": False, "labels_opened": False, "model_inference": False}
    with Path(str(path) + ".json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
