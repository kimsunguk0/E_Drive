#!/usr/bin/env python3
"""Compare a GT-free raw train8 clip fixture with an EXISTING system export.

Only this diagnostic reader loads the labeled .pt; the adapter receives a clip
directory, never that batch, its metadata or its labels. Reference pixels are
not decoded again in this environment. Only calibration and nominal time are
replaced, explicitly and in a new six-input dictionary. No model is imported or
executed, no dataset is instantiated, and no GPU/official submission is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import motiondrive_v2_inputs as adapter

REFERENCE_SHA256 = "4fb192ff6dd80ad44c8374b64afe55d49112327174ee8f7534ef4591cd2de1f3"
CALIBRATION_SHA256 = "8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961"
FIXTURE_MANIFEST_SHA256 = "da1e36a8b45815a8fb9b4f1053c7695cb754cdfa9eb4f2b6602fc0ff6ef67d27"
REFERENCE_SHAPES = {"images": (8, 6, 3, 432, 768), "history_images": (8, 4, 3, 216, 384),
                    "lidar2img": (8, 6, 4, 4), "history_transforms": (8, 4, 4, 4),
                    "time_offsets": (8, 4), "goal_xy": (8, 2)}
AUDITED_SOURCE_FILES = (
    Path(__file__), Path(adapter.__file__), ROOT / "scripts/etri_build_cache_b200.py",
    ROOT / "models/motiondrive_v2_input_contract.py",
    ROOT / "scripts/motiondrive_v2_data.py", ROOT / "models/motiondrive_v2/model.py",
    ROOT / "models/motiondrive_v2/scene_encoder.py", ROOT / "models/motiondrive_v2/motion_encoder.py",
    ROOT / "dense_vocab_v1/tools/data_converter/etri_test_converter.py",
)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, ensure_ascii=False).encode()).hexdigest()


def tree_sha(value):
    """Fingerprint even excluded label tensors, without modifying/casting them."""
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu" or item.requires_grad:
                raise ValueError("Reference must contain detached CPU tensors only")
            digest.update(f"tensor:{item.dtype}:{tuple(item.shape)}:".encode())
            digest.update(item.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping:")
            for key in sorted(item):
                digest.update(str(key).encode() + b"\0")
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for child in item:
                add(child)
        elif isinstance(item, (str, bool, int, float, type(None))):
            digest.update((type(item).__name__ + ":" + json.dumps(item, allow_nan=False)).encode())
        else:
            raise ValueError(f"Unsupported reference value: {type(item)}")

    add(value)
    return digest.hexdigest()


def byte_equal(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape and
            torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def validate_identity(reference, manifest):
    """No implicit reordering: export, export metadata and raw source must agree."""
    if set(reference) != {"batch", "metadata"}:
        raise ValueError("Expected the existing {batch, metadata} system export")
    batch, meta = reference["batch"], reference["metadata"]
    if (meta.get("split") != "train" or meta.get("train_only") is not True or
            meta.get("augment") is not False or meta.get("n_samples") != 8):
        raise ValueError("Reference is not the unaugmented train8 export")
    if (meta.get("camera_order") != list(adapter.CAMERA_ORDER) or
            meta.get("history_frame_offsets") != [1, 2, 5, 10] or
            meta.get("model_input_keys") != list(adapter.INPUT_KEYS)):
        raise ValueError("Reference input ordering does not match the adapter")
    for key, shape in REFERENCE_SHAPES.items():
        value = batch.get(key)
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or
                value.dtype != torch.float32 or value.device.type != "cpu" or value.requires_grad or
                not torch.isfinite(value).all()):
            raise ValueError(f"Reference input shape/dtype/CPU/finite mismatch: {key}")
    if (manifest.get("status") != "created_verified" or manifest.get("clip_count") != 8 or
            manifest.get("root_manifest_not_a_model_input") is not True or
            manifest.get("output_gt_future_status_object_map") is not False):
        raise ValueError("Raw fixture is not the verified GT-free train8 edition")
    if not meta.get("split_manifest_sha256") or meta["split_manifest_sha256"] != manifest.get("split_manifest_sha256"):
        raise ValueError("Reference/raw fixture split lineage mismatch")
    clips, samples = manifest.get("clips", []), meta.get("samples", [])
    if len(clips) != 8 or len(samples) != 8 or len(batch.get("scenario", [])) != 8:
        raise ValueError("Exactly eight aligned sample identities are required")
    for key in ("frame", "row", "scen_idx"):
        if not isinstance(batch.get(key), torch.Tensor) or batch[key].shape != (8,):
            raise ValueError(f"Reference identity field missing: {key}")
    if len(batch.get("session_id", [])) != 8:
        raise ValueError("Reference session identities missing")
    identities = []
    for i, (clip, sample) in enumerate(zip(clips, samples)):
        if (clip.get("clip_id") != f"fixture_{i:03d}" or clip.get("source_split") != "train" or
                clip.get("model_must_not_receive_source_mapping") is not True):
            raise ValueError("Raw fixture identities must be ordered train-only fixture_000..007")
        identity = (str(batch["scenario"][i]), int(batch["frame"][i]))
        if identity != (sample.get("scenario"), sample.get("frame")) or identity != (clip.get("source_scene"), clip.get("source_frame")):
            raise ValueError(f"Scenario/frame mismatch at export index {i}; no automatic reorder")
        for key in ("row", "scen_idx"):
            if int(batch[key][i]) != sample.get(key):
                raise ValueError(f"Reference metadata disagrees with batch {key}")
        if batch["session_id"][i] != sample.get("session_id"):
            raise ValueError("Reference session metadata mismatch")
        identities.append(identity)
    scenes = sorted({scene for scene, _ in identities})
    if len(scenes) != 4 or identities != [(scene, frame) for scene in scenes for frame in (30, 180)]:
        raise ValueError("Expected four sorted train scenes x frame30/180, without selection")
    if set(scenes) != set(manifest.get("sources", {})):
        raise ValueError("Raw fixture source-scene manifest mismatch")
    return clips


def verify_raw_files(fixture_root, clips):
    """Validate exactly the 96 fixture files, never follow manifest-supplied paths."""
    root = Path(fixture_root).resolve()
    expected_names = {f"{camera}/frame_0.jpg" for camera in adapter.CAMERA_ORDER}
    expected_names |= {f"camera_front/frame_{frame}.jpg" for frame in adapter.PAST_FRAMES}
    expected_names |= {"calibration.parquet", "ego_pose.parquet"}
    verified = {}
    for clip in clips:
        clip_id = clip["clip_id"]
        if re.fullmatch(r"fixture_00[0-7]", clip_id) is None:
            raise ValueError("Unsafe clip identifier")
        directory = root / clip_id
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Fixture clips must be existing ordinary directories")
        if set(clip.get("outputs", {})) != expected_names:
            raise ValueError("Raw fixture contains a missing or non-whitelisted output")
        actual = set()
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("Symlinks are not permitted inside raw fixture clips")
            if path.is_file():
                actual.add(str(path.relative_to(directory)))
        if actual != expected_names:
            raise ValueError("Clip must contain only ten JPEGs and two parquet files")
        for relative in sorted(expected_names):
            path = directory / relative
            expected = clip["outputs"][relative]
            observed = file_sha(path)
            if observed != expected.get("sha256") or path.stat().st_size != expected.get("bytes"):
                raise ValueError(f"Raw fixture file differs from its manifest: {clip_id}/{relative}")
            verified[str(path)] = observed
    return verified


def make_reference_inputs(batch, calibration):
    """Whitelist before adapting; original system images/history/pose/goal stay intact."""
    if calibration.shape != (6, 4, 4) or calibration.dtype != torch.float32 or calibration.device.type != "cpu":
        raise ValueError("New calibration must be CPU float32 [6,4,4]")
    result = {name: batch[name] for name in adapter.INPUT_KEYS}
    old = batch["lidar2img"]
    if not all(byte_equal(old[0], old[i]) for i in range(8)):
        raise ValueError("Reference calibration is unexpectedly sample-dependent")
    kept = [0, 1, 2, 4, 5]
    if not byte_equal(old[0, kept], calibration[kept]):
        raise ValueError("geometry_v2 must preserve the other five canonical cameras bitwise")
    if byte_equal(old[0, 3], calibration[3]):
        raise ValueError("Expected an explicit rear-wide calibration correction")
    result["lidar2img"] = calibration.unsqueeze(0).expand(8, -1, -1, -1).clone()
    result["time_offsets"] = torch.tensor([adapter.NOMINAL_SECONDS], dtype=torch.float32).expand(8, -1).clone()
    audit = {"replaced_keys": ["lidar2img", "time_offsets"],
             "preserved_keys": ["images", "history_images", "history_transforms", "goal_xy"],
             "preserved_tensor_object_identity": all(result[name] is batch[name]
                 for name in ("images", "history_images", "history_transforms", "goal_xy")),
             "reference_pixels_redecoded": False, "other_five_cameras_bitwise_preserved": True,
             "rear_matrix_max_abs_change": float((old[:, 3].double() - result["lidar2img"][:, 3].double()).abs().max()),
             "original_raw_time_offsets": batch["time_offsets"].tolist(),
             "nominal_time_offsets": result["time_offsets"].tolist(),
             "time_max_abs_change_seconds": float((batch["time_offsets"].double() - result["time_offsets"].double()).abs().max()),
             "labels_or_identifiers_forwarded": False}
    return result, audit


def image_level_comparison(prepared, expected, reference_metadata, scene, anchor):
    """Distinguish regenerated JPEG identity from final normalized tensor identity.

    Old cache JPEG hashes come from the saved system export metadata. No cache
    JPEG is opened or decoded in the adapter environment to create a reference.
    """
    images = []
    specs = [(camera, 0, "images", index) for index, camera in enumerate(adapter.CAMERA_ORDER)]
    specs += [(adapter.CAMERA_ORDER[0], frame, "history_images", index)
              for index, frame in enumerate(adapter.PAST_FRAMES)]
    cache_root = reference_metadata.get("image_cache_root")
    original_hashes = reference_metadata.get("image_source_sha256", {})
    for camera, relative, key, index in specs:
        actual, reference = prepared.inputs[key][0, index], expected["inputs"][key][0, index]
        difference = (actual.double() - reference.double()).abs()
        provenance = prepared.metadata.get("images", {}).get(f"{camera}/frame_{relative}.jpg", {})
        cache_path = str(Path(cache_root) / scene / camera / f"{anchor + relative:08d}.jpg") if cache_root else None
        original_sha = original_hashes.get(cache_path)
        regenerated_sha = provenance.get("reconstructed_cache_jpeg_sha256")
        images.append({"camera": camera, "relative_frame": relative, "input_key": key,
                       "normalized_tensor_bitwise_equal": byte_equal(actual, reference),
                       "normalized_max_abs": float(difference.max()),
                       "normalized_mean_abs": float(difference.mean()),
                       "numerically_different_values": int(torch.count_nonzero(difference)),
                       "reference_cache_jpeg_sha256": original_sha,
                       "regenerated_cache_jpeg_sha256": regenerated_sha,
                       "cache_jpeg_sha_equal": (original_sha == regenerated_sha)
                           if original_sha is not None and regenerated_sha is not None else None,
                       "reference_cache_file_reopened": False})
    return images


def audit(fixture_root, reference_path, calibration_path, *, geometry_contract_path=None,
          expected_reference_sha256=REFERENCE_SHA256, expected_calibration_sha256=CALIBRATION_SHA256,
          expected_fixture_manifest_sha256=FIXTURE_MANIFEST_SHA256,
          projection_atol=1e-4, pose_atol=1e-5):
    if torch.cuda.is_initialized():
        raise RuntimeError("Use a fresh CPU-only process; GPU was already initialized")
    root, reference_path, calibration_path = map(lambda path: Path(path).resolve(),
                                                (fixture_root, reference_path, calibration_path))
    manifest_path = root / "fixture_manifest.json"
    contract_path = Path(geometry_contract_path).resolve() if geometry_contract_path else calibration_path.parent / "supervision_manifest.json"
    source_paths = [manifest_path, reference_path, calibration_path, contract_path, *AUDITED_SOURCE_FILES]
    before = {str(path.resolve()): file_sha(path) for path in source_paths}
    if before[str(manifest_path)] != expected_fixture_manifest_sha256:
        raise ValueError("Pinned raw fixture manifest SHA mismatch")
    if before[str(reference_path)] != expected_reference_sha256:
        raise ValueError("Existing system reference SHA mismatch; refusing a newly decoded substitute")
    if before[str(calibration_path)] != expected_calibration_sha256:
        raise ValueError("Pinned geometry_v2 calibration SHA mismatch")
    geometry_contract = json.loads(contract_path.read_text())
    if (geometry_contract.get("geometry_edition") != "cache_meta_rear_wide_v2" or
            geometry_contract.get("canonical_calibration_sha256") != expected_calibration_sha256):
        raise ValueError("Geometry edition contract does not bind the new calibration")
    manifest = json.loads(manifest_path.read_text())
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    reference_tree_before = tree_sha(reference)
    clips = validate_identity(reference, manifest)
    raw_before = verify_raw_files(root, clips)
    with np.load(calibration_path, allow_pickle=False) as archive:
        calibration_np = archive["lidar2img"].copy()
    if calibration_np.dtype != np.float32 or calibration_np.shape != (6, 4, 4) or not np.isfinite(calibration_np).all():
        raise ValueError("Pinned calibration must be finite float32 [6,4,4]")
    inputs, adaptation = make_reference_inputs(reference["batch"], torch.from_numpy(calibration_np))
    contract_before = adapter.input_contract()
    contract_sha_before = json_sha(contract_before)
    results = []
    for i, clip in enumerate(clips):
        # This is the ONLY adapter call: directory, not source identities or GT.
        prepared = adapter.prepare_clip_inputs(root / clip["clip_id"])
        expected = {"inputs": {name: value[i:i + 1] for name, value in inputs.items()},
                    "metadata": {"input_contract": contract_before}}
        comparison = adapter.compare_input_fixture(prepared, expected,
            projection_atol=projection_atol, pose_atol=pose_atol)
        results.append({"clip_id": clip["clip_id"], "reference_index": i,
                        "source_scene": clip["source_scene"], "source_frame": clip["source_frame"],
                        "mapping_is_diagnostic_only": True, "comparison": comparison,
                        "image_comparison": image_level_comparison(prepared, expected, reference["metadata"],
                            clip["source_scene"], clip["source_frame"]),
                        "adapter_metadata": prepared.metadata})
    reference_tree_after = tree_sha(reference)
    after = {path: file_sha(path) for path in before}
    raw_after = verify_raw_files(root, clips)
    contract_sha_after = json_sha(adapter.input_contract())
    if reference_tree_before != reference_tree_after:
        raise RuntimeError("In-memory reference or its excluded labels were modified")
    if before != after or raw_before != raw_after or contract_sha_before != contract_sha_after:
        raise RuntimeError("Source/reference/calibration/contract changed during the audit")
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected GPU initialization during CPU-only audit")
    aggregate = {}
    for name in adapter.INPUT_KEYS:
        entries = [row["comparison"]["inputs"][name] for row in results]
        aggregate[name] = {"pass_count": sum(row["pass"] for row in entries), "n": 8,
                           "bitwise_count": sum(row["bitwise_equal"] for row in entries),
                           "max_abs": max(row["max_abs"] for row in entries),
                           "atol": entries[0]["atol"], "rtol": 0.}
    passed = all(row["comparison"]["all_pass"] for row in results)
    return {"status": "pass" if passed else "parity_failed", "all_pass": passed,
            "purpose": "Existing system training export vs GT-free raw adapter; train8 CPU parity only",
            "reference_path": str(reference_path), "reference_sha256": expected_reference_sha256,
            "reference_origin": "Pre-existing system-Python MotionDriveDataset export; no pixel re-decoding here",
            "reference_adaptation": adaptation, "identity_match_all_eight": True,
            "geometry_contract_mapping": {"source": "cache_meta_rear_wide_v2", "adapter": "geometry_v2"},
            "input_contract": contract_before, "input_contract_sha256_before": contract_sha_before,
            "input_contract_sha256_after": contract_sha_after,
            "file_sha256_before": before, "file_sha256_after": after,
            "raw_fixture_sha256_before": raw_before, "raw_fixture_sha256_after": raw_after,
            "reference_tree_sha256_before": reference_tree_before, "reference_tree_sha256_after": reference_tree_after,
            "all_original_sources_unchanged": True, "aggregate": aggregate, "clips": results,
            "gpu_used": False, "model_forward_performed": False, "dataset_instantiated": False,
            "official_submission_created": False, "upstream_train_archives_reread": False,
            "limitations": ["Agreement covers these eight preregistered train samples, not every clip or generalization.",
                            "Projection tolerance permits archived matrix rounding; it is not universal projected pixel equality.",
                            "Raw fixture source copies are SHA-checked; upstream archive hashes are trusted from its builder manifest.",
                            "This verifies input provenance/consistency, not final regulatory approval or model performance."]}


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument("--fixture-root", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--geometry-contract")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if os.path.lexists(output):
        raise FileExistsError("Refusing to overwrite an existing audit report")
    if output.resolve().is_relative_to(Path(args.fixture_root).resolve()):
        raise ValueError("Write the report outside the immutable raw fixture")
    adapter.cv2.setNumThreads(1)
    report = audit(args.fixture_root, args.reference, args.calibration,
                   geometry_contract_path=args.geometry_contract)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "aggregate": report["aggregate"],
                      "all_original_sources_unchanged": True, "output": str(output)}, indent=2))
    return 0 if report["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
