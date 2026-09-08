#!/usr/bin/env python3
"""Replay the frozen P7-C train2 golden under the frozen P8 source."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/capture_motiondrive_v2_p8_old_c_golden.py"
EXPECTED_HELPER_SHA256 = "13abdced8beb3870b89787f7e960fb9f93ba2d3358f7e5f72e4d500dbc5adab0"
EXPECTED_SOURCE_MANIFEST_SHA256 = "6682ea02544fc8a39e5d78bdf90bf9735682b51f3f5c7f88db93d85729d77f75"
EXPECTED_GOLDEN_SHA256 = "f4a3687089074c3022d16f51ceb58c9dcb7af267ad62b3bb907f0edcb47bdf23"
EXPECTED_INPUT_TREE_SHA256 = "f43a796e2741f2202bd49f058c1466c4ac0ce22f0bd36a9c38971df1392ea858"
EXPECTED_OUTPUT_TREE_SHA256 = "6943f60ce073e2850e6c0697d3f21929164ea04a8cc7c23ab0beb1d08f422b72"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
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
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CVD must be explicitly empty")
    require(sha256(HELPER) == EXPECTED_HELPER_SHA256, "old capture helper SHA mismatch")
    require(sha256(args.source_manifest) == EXPECTED_SOURCE_MANIFEST_SHA256,
            "P8 source manifest SHA mismatch")
    require(sha256(args.golden) == EXPECTED_GOLDEN_SHA256, "old golden SHA mismatch")
    require(not args.output.exists() and args.output.parent.is_dir(), "fresh output required")
    capture_path = args.output.with_name(args.output.stem + ".raw_capture.json")
    require(not capture_path.exists(), "fresh raw capture required")

    spec = importlib.util.spec_from_file_location("p8_old_capture", HELPER)
    require(spec is not None and spec.loader is not None, "cannot load capture helper")
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    capture.EXPECTED["source_manifest"] = EXPECTED_SOURCE_MANIFEST_SHA256
    old_argv = sys.argv
    try:
        sys.argv = [str(HELPER), "--checkpoint", str(args.checkpoint),
                    "--manifest", str(args.manifest), "--fixture", str(args.fixture),
                    "--source-manifest", str(args.source_manifest),
                    "--output", str(capture_path)]
        require(capture.main() == 0, "underlying capture failed")
    finally:
        sys.argv = old_argv

    actual = json.loads(capture_path.read_text())
    golden = json.loads(args.golden.read_text())
    require(actual["input_tensor_tree_sha256"] == golden["input_tensor_tree_sha256"]
            == EXPECTED_INPUT_TREE_SHA256, "computational input tree mismatch")
    require(actual["output_tensor_tree_sha256"] == golden["output_tensor_tree_sha256"]
            == EXPECTED_OUTPUT_TREE_SHA256, "full output tree mismatch")
    require(set(actual["outputs"]) == set(golden["outputs"]), "output key set mismatch")
    for name in sorted(golden["outputs"]):
        for field in ("shape", "dtype", "sha256"):
            require(actual["outputs"][name][field] == golden["outputs"][name][field],
                    f"output {name}/{field} mismatch")
        if "values" in golden["outputs"][name]:
            require(actual["outputs"][name].get("values") == golden["outputs"][name]["values"],
                    f"output {name}/values mismatch")
    require(actual["model_state_sha256_before"] == actual["model_state_sha256_after"]
            == golden["model_state_sha256_before"] == golden["model_state_sha256_after"],
            "model state mismatch")
    require(actual["source_sha256_before"] == actual["source_sha256_after"],
            "P8 source changed")
    require(actual["artifacts_sha256_before"] == actual["artifacts_sha256_after"],
            "input artifacts changed")
    require(actual["runtime"]["cuda_initialized"] is False, "CUDA initialized")

    result = {
        "schema_version": 1,
        "status": "p8_new_source_exact_old_c_golden_parity_pass",
        "not_accuracy_or_training": True,
        "final_or_tune_rows_accessed": False,
        "old_golden_path": str(args.golden.resolve()),
        "old_golden_sha256": EXPECTED_GOLDEN_SHA256,
        "new_source_manifest_path": str(args.source_manifest.resolve()),
        "new_source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "capture_helper_sha256": EXPECTED_HELPER_SHA256,
        "raw_capture_path": str(capture_path.resolve()),
        "raw_capture_sha256": sha256(capture_path),
        "input_tensor_tree_sha256": EXPECTED_INPUT_TREE_SHA256,
        "output_tensor_tree_sha256": EXPECTED_OUTPUT_TREE_SHA256,
        "output_keys": sorted(actual["outputs"]),
        "all_output_metadata_hashes_and_recorded_values_equal": True,
        "model_state_before_after_equal": True,
        "source_and_inputs_before_after_equal": True,
        "runtime": actual["runtime"],
        "boundary": "Default-C migration/function parity only; no accuracy, training, tune, final, latency, or W-performance claim."
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "sha256": sha256(args.output), "pid": os.getpid()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
