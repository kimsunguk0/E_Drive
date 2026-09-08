#!/usr/bin/env python3
"""Capture a two-row CPU/FP32 old-source C forward for P8 parity only.

This reads the already approved train8 tensor fixture and a pinned historical
P7-C checkpoint. It never opens tune/final data, computes accuracy, trains, or
uses CUDA. The output is a byte-level migration golden, not a model result.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.motiondrive_v2 import MotionDriveV2
from scripts.motiondrive_v2_training import tensor_state_sha256


EXPECTED = {
    "checkpoint": "6145255ab2b1efd3deb169b873896e9b1b623ca49f11e5a46a71b138e3b0355e",
    "manifest": "cf287d05ce6b2f4476c07ac5c2253cd9ecb3227bcc01404adbbba8463bb66de0",
    "fixture": "4fb192ff6dd80ad44c8374b64afe55d49112327174ee8f7534ef4591cd2de1f3",
    "source_manifest": "880c3cc36ababe31c58376ab815a239e337f130fc9b373a041fea6ad4a8d1ea4",
    "model_state": "e79d545bbb09b4109c783a8cf4c861772bc5e52c62004ef5b37dc4ca63e938a2",
}
INPUT_KEYS = (
    "images", "history_images", "lidar2img", "history_transforms",
    "time_offsets", "goal_xy",
)
VALUE_OUTPUTS = ("plan_abs", "state_hat", "history_hat")
EXPECTED_ROWS = [30, 180]
EXPECTED_SCENE = "20260112-105434"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update((str(value.dtype) + "\0" + str(tuple(value.shape)) + "\0").encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_tree_sha256(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update((name + "\0" + str(value.dtype) + "\0" +
                       str(tuple(value.shape)) + "\0").encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CVD must be explicitly empty")
    require(not torch.cuda.is_initialized(), "CUDA must remain uninitialized")
    require(not args.output.exists() and args.output.parent.is_dir(),
            "Output must be a new file in an existing directory")
    paths = {"checkpoint": args.checkpoint.resolve(), "manifest": args.manifest.resolve(),
             "fixture": args.fixture.resolve(),
             "source_manifest": args.source_manifest.resolve()}
    require(all(path.is_file() and not path.is_symlink() for path in paths.values()),
            "Every input must be an ordinary non-symlink file")
    before = {name: file_sha256(path) for name, path in paths.items()}
    require(before == {name: EXPECTED[name] for name in paths}, "Pinned input SHA mismatch")

    source_manifest = json.loads(paths["source_manifest"].read_text())
    require(set(source_manifest) == {"schema_version", "git_sha", "file_sha256"},
            "Unexpected source-manifest schema")
    source_before = {name: file_sha256(ROOT / name)
                     for name in source_manifest["file_sha256"]}
    require(source_before == source_manifest["file_sha256"], "Old source closure mismatch")

    external = json.loads(paths["manifest"].read_text())
    require(external.get("status") == "completed" and external.get("step") == 6000,
            "Historical C run is not terminal LAST6000")
    require(external.get("model_config", {}).get("cross_cell_goal_mode") == "zero"
            and external.get("model_config", {}).get("goal_on") is True
            and external.get("model_config", {}).get("state_on") is True,
            "Historical model is not P7 C")
    require("history_contract" not in external["model_config"]
            and "history_frame_offsets" not in external["model_config"]
            and "nominal_history_seconds" not in external["model_config"],
            "Golden must capture the pre-P8 serialized config")

    fixture = torch.load(paths["fixture"], map_location="cpu", weights_only=True)
    require(set(fixture) == {"batch", "metadata"}, "Unexpected fixture schema")
    metadata = fixture["metadata"]
    require(metadata.get("train_only") is True and metadata.get("n_samples") == 8,
            "Fixture is not the approved train8 artifact")
    samples = metadata.get("samples", [])[:2]
    require([item.get("row") for item in samples] == EXPECTED_ROWS
            and [item.get("scenario") for item in samples] == [EXPECTED_SCENE] * 2,
            "First two preregistered train rows changed")
    batch = fixture["batch"]
    inputs = {}
    for name in INPUT_KEYS:
        value = batch.get(name)
        require(isinstance(value, torch.Tensor) and value.shape[0] >= 2,
                f"Missing tensor input: {name}")
        inputs[name] = value[:2].detach().cpu().contiguous()
    # Historical C training/evaluation used nominal timing, not stored raw offsets.
    inputs["time_offsets"] = torch.tensor([[.1, .2, .5, 1.]], dtype=torch.float32).expand(2, -1).clone()
    require(all(value.dtype == torch.float32 and torch.isfinite(value).all()
                and not value.requires_grad for value in inputs.values()),
            "Golden inputs must be finite detached CPU FP32")

    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    require(checkpoint.get("step") == 6000 and set(checkpoint.get("model", {})),
            "Checkpoint is not LAST6000")
    embedded_config = checkpoint.get("manifest", {}).get("model_config")
    embedded_config_bytes = json.dumps(embedded_config, sort_keys=True,
                                       separators=(",", ":"), allow_nan=False).encode()
    external_config_bytes = json.dumps(external["model_config"], sort_keys=True,
                                       separators=(",", ":"), allow_nan=False).encode()
    require(embedded_config_bytes == external_config_bytes,
            "Embedded/external canonical model configs differ")
    require(tensor_state_sha256(checkpoint["model"]) == EXPECTED["model_state"],
            "P7 C model-state SHA mismatch")
    with torch.random.fork_rng(devices=[]):
        model = MotionDriveV2(external["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    require(not any(module.training for module in model.modules()), "Model must be fully eval")
    state_before = tensor_state_sha256(model.state_dict())

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    with torch.inference_mode():
        outputs = model(**inputs)
    require(isinstance(outputs, dict) and outputs, "Model returned no output mapping")
    require(all(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and torch.isfinite(value).all() for value in outputs.values()),
            "Every model output must be a finite CPU tensor")
    state_after = tensor_state_sha256(model.state_dict())
    require(state_before == state_after == EXPECTED["model_state"], "Model state changed")
    after = {name: file_sha256(path) for name, path in paths.items()}
    source_after = {name: file_sha256(ROOT / name) for name in source_before}
    require(before == after and source_before == source_after, "Input/source bytes changed")
    require(not torch.cuda.is_initialized(), "CPU golden initialized CUDA")

    input_summary = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                            "sha256": tensor_sha256(value)}
                     for name, value in sorted(inputs.items())}
    output_summary = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                             "sha256": tensor_sha256(value)}
                      for name, value in sorted(outputs.items())}
    for name in VALUE_OUTPUTS:
        require(name in outputs, f"Missing required output: {name}")
        output_summary[name]["values"] = outputs[name].detach().cpu().tolist()
    result = {
        "schema_version": 1,
        "status": "completed_old_source_cpu_golden",
        "purpose": "P8 default-C source/config migration parity only",
        "not_accuracy_or_training": True,
        "final_or_tune_rows_accessed": False,
        "selected_train_rows": EXPECTED_ROWS,
        "selected_scenario": EXPECTED_SCENE,
        "batch_size": 2,
        "precision": "fp32",
        "device": "cpu",
        "source_manifest": source_manifest,
        "repository_head_informational": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "artifacts_sha256_before": before,
        "artifacts_sha256_after": after,
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "historical_config": external["model_config"],
        "historical_config_canonical_sha256": hashlib.sha256(external_config_bytes).hexdigest(),
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "input_tensor_tree_sha256": tensor_tree_sha256(inputs),
        "input_tensors": input_summary,
        "output_tensor_tree_sha256": tensor_tree_sha256(outputs),
        "outputs": output_summary,
        "excluded_fixture_keys_not_indexed": sorted(set(batch).difference(INPUT_KEYS)),
        "runtime": {"python": platform.python_version(), "executable": sys.executable,
                    "torch": str(torch.__version__), "numpy": np.__version__,
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "cuda_initialized": torch.cuda.is_initialized(),
                    "torch_num_threads": torch.get_num_threads(),
                    "torch_num_interop_threads": torch.get_num_interop_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                    "tf32_cudnn": torch.backends.cudnn.allow_tf32},
        "started_at_utc": started_at_utc,
        "ended_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "sha256": file_sha256(args.output), "pid": os.getpid()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
