"""CPU mock tests for identity/integrity boundaries, not actual cache parity."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_motiondrive_v2_deploy_inputs as runner


@pytest.fixture
def audit_fixture(tmp_path, monkeypatch):
    root = tmp_path / "raw_fixture"
    root.mkdir()
    scenes = ["20260112-105434", "20260112-105504", "20260112-111049", "20260112-111149"]
    identities = [(scene, frame) for scene in scenes for frame in (30, 180)]
    batch = {
        "images": torch.arange(8 * 6 * 3 * 2 * 2).reshape(8, 6, 3, 2, 2).float() / 100.,
        "history_images": torch.arange(8 * 4 * 3).reshape(8, 4, 3, 1, 1).float() / 100.,
        "lidar2img": torch.eye(4).repeat(8, 6, 1, 1),
        "history_transforms": torch.eye(4).repeat(8, 4, 1, 1),
        "time_offsets": torch.tensor([[.0999, .1999, .4998, .9996]]).repeat(8, 1),
        "goal_xy": torch.arange(16).reshape(8, 2).float(),
        "scenario": [x[0] for x in identities], "frame": torch.tensor([x[1] for x in identities]),
        "row": torch.arange(8), "scen_idx": torch.arange(4).repeat_interleave(2),
        "session_id": ["session"] * 8,
        "gt_plan": torch.full((8, 6, 2), 12345.), "state_target": torch.full((8, 6), 999.),
    }
    metadata = {
        "split": "train", "train_only": True, "augment": False, "n_samples": 8,
        "camera_order": list(runner.adapter.CAMERA_ORDER), "history_frame_offsets": [1, 2, 5, 10],
        "model_input_keys": list(runner.adapter.INPUT_KEYS), "split_manifest_sha256": "fixed-split",
        "samples": [{"scenario": scene, "frame": frame, "row": i, "scen_idx": i // 2,
                     "session_id": "session"} for i, (scene, frame) in enumerate(identities)],
    }
    reference = {"batch": batch, "metadata": metadata}
    reference_path = tmp_path / "existing_system_export.pt"
    torch.save(reference, reference_path)
    calibration = batch["lidar2img"][0].clone()
    calibration[3, 1, 2] = 12.
    calibration_path = tmp_path / "calibration.npz"
    np.savez(calibration_path, lidar2img=calibration.numpy())
    contract_path = tmp_path / "supervision_manifest.json"
    contract_path.write_text(json.dumps({"geometry_edition": "cache_meta_rear_wide_v2",
        "canonical_calibration_sha256": runner.file_sha(calibration_path)}))
    names = {f"{camera}/frame_0.jpg" for camera in runner.adapter.CAMERA_ORDER}
    names |= {f"camera_front/frame_{frame}.jpg" for frame in runner.adapter.PAST_FRAMES}
    names |= {"calibration.parquet", "ego_pose.parquet"}
    clips = []
    for i, (scene, frame) in enumerate(identities):
        clip = root / f"fixture_{i:03d}"
        clip.mkdir()
        outputs = {}
        for name in names:
            path = clip / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"synthetic raw fixture {i} {name}".encode())
            outputs[name] = {"sha256": runner.file_sha(path), "bytes": path.stat().st_size}
        clips.append({"clip_id": clip.name, "source_split": "train", "source_scene": scene,
                      "source_frame": frame, "model_must_not_receive_source_mapping": True, "outputs": outputs})
    manifest = {"status": "created_verified", "clip_count": 8, "root_manifest_not_a_model_input": True,
                "output_gt_future_status_object_map": False, "split_manifest_sha256": "fixed-split",
                "clips": clips, "sources": {scene: {} for scene in scenes}}
    manifest_path = root / "fixture_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    shapes = {name: tuple(batch[name].shape) for name in runner.adapter.INPUT_KEYS}
    monkeypatch.setattr(runner, "REFERENCE_SHAPES", shapes)
    expected, _ = runner.make_reference_inputs(batch, calibration)
    calls = []

    def adapter_stub(directory):
        assert isinstance(directory, Path)
        assert directory.parent == root
        calls.append(directory)
        i = int(directory.name[-3:])
        return runner.adapter.PreparedClip({name: value[i:i + 1].clone() for name, value in expected.items()},
                                           {"input_contract": runner.adapter.input_contract()})

    monkeypatch.setattr(runner.adapter, "prepare_clip_inputs", adapter_stub)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    return {"root": root, "reference": reference, "reference_path": reference_path,
            "calibration": calibration, "calibration_path": calibration_path, "contract_path": contract_path,
            "manifest": manifest, "manifest_path": manifest_path, "adapter_stub": adapter_stub, "calls": calls,
            "kwargs": {"expected_reference_sha256": runner.file_sha(reference_path),
                       "expected_fixture_manifest_sha256": runner.file_sha(manifest_path),
                       "expected_calibration_sha256": runner.file_sha(calibration_path)}}


def run_fixture(fixture):
    return runner.audit(fixture["root"], fixture["reference_path"], fixture["calibration_path"], **fixture["kwargs"])


def test_cpu_mock_existing_export_parity_and_all_source_hashes_unchanged(audit_fixture):
    f = audit_fixture
    report = run_fixture(f)
    assert report["all_pass"] and report["status"] == "pass"
    assert len(f["calls"]) == 8
    assert report["identity_match_all_eight"]
    assert report["reference_adaptation"]["replaced_keys"] == ["lidar2img", "time_offsets"]
    assert report["reference_adaptation"]["preserved_tensor_object_identity"]
    assert not report["reference_adaptation"]["reference_pixels_redecoded"]
    assert not report["reference_adaptation"]["labels_or_identifiers_forwarded"]
    assert report["file_sha256_before"] == report["file_sha256_after"]
    assert report["raw_fixture_sha256_before"] == report["raw_fixture_sha256_after"]
    assert len(report["raw_fixture_sha256_before"]) == 96
    assert report["reference_tree_sha256_before"] == report["reference_tree_sha256_after"]
    assert report["input_contract_sha256_before"] == report["input_contract_sha256_after"]
    assert all(row["bitwise_count"] == row["pass_count"] == 8 for row in report["aggregate"].values())
    assert not any(report[key] for key in ("gpu_used", "model_forward_performed", "dataset_instantiated", "official_submission_created"))


def test_reference_whitelist_replaces_only_two_fields_and_never_mutates_labels(audit_fixture):
    batch = audit_fixture["reference"]["batch"]
    before = runner.tree_sha(batch)
    adapted, info = runner.make_reference_inputs(batch, audit_fixture["calibration"])
    assert set(adapted) == set(runner.adapter.INPUT_KEYS)
    assert runner.tree_sha(batch) == before
    for key in ("images", "history_images", "history_transforms", "goal_xy"):
        assert adapted[key] is batch[key]
    assert torch.equal(adapted["time_offsets"], torch.tensor([[.1, .2, .5, 1.]]).repeat(8, 1))
    assert info["time_max_abs_change_seconds"] > 0.


@pytest.mark.parametrize("change", ["batch_frame", "metadata_frame", "raw_frame", "reorder", "raw_split", "reference_split", "split_sha", "row"])
def test_independent_identity_sources_must_all_agree(audit_fixture, change):
    f = audit_fixture
    ref, manifest = copy.deepcopy(f["reference"]), copy.deepcopy(f["manifest"])
    if change == "batch_frame":
        ref["batch"]["frame"][0] += 1
    elif change == "metadata_frame":
        ref["metadata"]["samples"][0]["frame"] += 1
    elif change == "raw_frame":
        manifest["clips"][0]["source_frame"] += 1
    elif change == "reorder":
        manifest["clips"][0], manifest["clips"][1] = manifest["clips"][1], manifest["clips"][0]
    elif change == "raw_split":
        manifest["clips"][0]["source_split"] = "tune"
    elif change == "reference_split":
        ref["metadata"]["split"] = "test"
    elif change == "split_sha":
        manifest["split_manifest_sha256"] = "other"
    else:
        ref["metadata"]["samples"][0]["row"] += 1
    with pytest.raises(ValueError):
        runner.validate_identity(ref, manifest)


@pytest.mark.parametrize("field", ["reference", "calibration", "fixture_manifest"])
def test_pinned_file_sha_checked_before_reference_load(audit_fixture, monkeypatch, field):
    f = audit_fixture
    f["kwargs"][f"expected_{field}_sha256"] = "0" * 64
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Reference must not be loaded before SHA verification"))
    with pytest.raises(ValueError, match="SHA mismatch"):
        run_fixture(f)
    assert not f["calls"]


def test_image_trace_keeps_existing_cache_hash_separate_from_tensor_parity(audit_fixture):
    f = audit_fixture
    adapted, _ = runner.make_reference_inputs(f["reference"]["batch"], f["calibration"])
    prepared = f["adapter_stub"](f["root"] / "fixture_000")
    prepared.metadata["images"] = {"camera_front/frame_0.jpg": {"reconstructed_cache_jpeg_sha256": "same-jpeg"}}
    metadata = {"image_cache_root": "/old/system/cache", "image_source_sha256": {
        "/old/system/cache/20260112-105434/camera_front/00000030.jpg": "same-jpeg"}}
    expected = {"inputs": {name: value[0:1] for name, value in adapted.items()}}
    prepared.inputs["images"][0, 0, 0, 0, 0] += .01
    trace = runner.image_level_comparison(prepared, expected, metadata, "20260112-105434", 30)
    assert len(trace) == 10
    assert trace[0]["cache_jpeg_sha_equal"] is True
    assert trace[0]["normalized_tensor_bitwise_equal"] is False
    assert trace[0]["numerically_different_values"] == 1
    assert all(row["reference_cache_file_reopened"] is False for row in trace)


def test_old_geometry_contract_is_rejected(audit_fixture):
    f = audit_fixture
    contract = json.loads(f["contract_path"].read_text())
    contract["geometry_edition"] = "old_rear"
    f["contract_path"].write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="Geometry edition"):
        run_fixture(f)


@pytest.mark.parametrize("change", ["nonrear", "no_rear", "sample_dependent"])
def test_canonical_replacement_cannot_silently_change_other_cameras(audit_fixture, change):
    f = audit_fixture
    batch, calibration = copy.deepcopy(f["reference"]["batch"]), f["calibration"].clone()
    if change == "nonrear":
        calibration[0, 0, 0] += .001
    elif change == "no_rear":
        calibration[3] = batch["lidar2img"][0, 3]
    else:
        batch["lidar2img"][1, 3, 0, 0] += 1.
    with pytest.raises(ValueError):
        runner.make_reference_inputs(batch, calibration)


@pytest.mark.parametrize("change", ["alter", "extra_label", "symlink"])
def test_raw_fixture_integrity_and_no_extra_input_files(audit_fixture, change):
    f = audit_fixture
    clip = f["root"] / "fixture_000"
    if change == "alter":
        (clip / "ego_pose.parquet").write_bytes(b"changed")
    elif change == "extra_label":
        (clip / "future_ground_truth.pt").write_bytes(b"forbidden")
    else:
        (clip / "link").symlink_to(f["reference_path"])
    with pytest.raises(ValueError):
        run_fixture(f)
    assert not f["calls"]


def test_pixel_mismatch_is_reported_not_redecoded_or_tolerance_hidden(audit_fixture, monkeypatch):
    f = audit_fixture

    def wrong_pixels(path):
        prepared = f["adapter_stub"](path)
        if path.name == "fixture_003":
            prepared.inputs["images"][0, 0, 0, 0, 0] += .001
        return prepared

    monkeypatch.setattr(runner.adapter, "prepare_clip_inputs", wrong_pixels)
    report = run_fixture(f)
    assert not report["all_pass"] and report["status"] == "parity_failed"
    assert report["aggregate"]["images"]["pass_count"] == 7
    assert report["aggregate"]["images"]["atol"] == 0.
    assert report["all_original_sources_unchanged"]


def test_post_audit_detects_even_excluded_label_mutation(audit_fixture, monkeypatch):
    f = audit_fixture
    loaded = copy.deepcopy(f["reference"])
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: loaded)

    def mutating_stub(path):
        prepared = f["adapter_stub"](path)
        loaded["batch"]["gt_plan"][0, 0, 0] += 1.
        return prepared

    monkeypatch.setattr(runner.adapter, "prepare_clip_inputs", mutating_stub)
    with pytest.raises(RuntimeError, match="excluded labels were modified"):
        run_fixture(f)


def test_post_audit_detects_raw_file_mutation(audit_fixture, monkeypatch):
    f = audit_fixture

    def mutating_stub(path):
        prepared = f["adapter_stub"](path)
        if path.name == "fixture_007":
            (path / "ego_pose.parquet").write_bytes(b"changed during audit")
        return prepared

    monkeypatch.setattr(runner.adapter, "prepare_clip_inputs", mutating_stub)
    with pytest.raises(ValueError, match="differs from its manifest"):
        run_fixture(f)


def test_cli_refuses_existing_report_without_running(audit_fixture, monkeypatch, tmp_path):
    f = audit_fixture
    output = tmp_path / "preserve.json"
    output.write_text("existing report")
    monkeypatch.setattr(runner, "audit", lambda *args, **kwargs: pytest.fail("No work before overwrite gate"))
    with pytest.raises(FileExistsError):
        runner.main(["--fixture-root", str(f["root"]), "--reference", str(f["reference_path"]),
                     "--calibration", str(f["calibration_path"]), "--output", str(output)])
    assert output.read_text() == "existing report"
