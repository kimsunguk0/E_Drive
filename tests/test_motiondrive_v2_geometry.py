"""새 edition의 보존·복사·fail-closed 계약을 합성 CPU 자료로 검증한다."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import derive_motiondrive_v2_geometry as derive


def camera_example():
    new_k = np.array([[400., 0., 960.], [0., 400., 678.], [0., 0., 1.]])
    shift = np.eye(3)
    shift[1, 2] = -456
    k_cache = np.diag([.4, .4, 1.]) @ shift @ new_k
    return ({"euler": [0., 0., 0.], "translation": [0., 0., 0.]},
            {"K_cache": k_cache.tolist(), "new_K": new_k.tolist(),
             "crop": [0, 456, 1920, 1080], "scale": .4, "out_size": [768, 432]})


def test_actual_cache_intrinsic_used_without_double_crop():
    row, meta = camera_example()
    got = derive.camera_projection(row, meta)
    np.testing.assert_array_equal(got[:3, :3], meta["K_cache"])
    assert got[1, 2] == pytest.approx(88.8)


@pytest.mark.parametrize("field,value", [("scale", .8), ("out_size", [768, 448]),
                                         ("crop", [0, 456, 1920, 1000])])
def test_invalid_geometry_rejected(field, value):
    row, meta = camera_example()
    meta[field] = value
    with pytest.raises(ValueError):
        derive.camera_projection(row, meta)


def test_inconsistent_cache_K_rejected():
    row, meta = camera_example()
    meta["K_cache"][1][2] += .01
    with pytest.raises(ValueError, match="K_cache"):
        derive.camera_projection(row, meta)


def test_other_five_bitwise_preserved_not_reconstructed():
    original = np.tile(np.eye(4, dtype=np.float32), (6, 1, 1))
    candidate = original.astype(np.float64)
    candidate[0, 0, 0] += 1e-5
    candidate[derive.REAR_INDEX, 1, 2] = -182.4
    fixed, error = derive.corrected_projection(original, candidate)
    ids = [i for i in range(6) if i != derive.REAR_INDEX]
    np.testing.assert_array_equal(fixed[ids], original[ids])
    assert error == pytest.approx(1e-5)
    np.testing.assert_array_equal(fixed[derive.REAR_INDEX], candidate[derive.REAR_INDEX].astype(np.float32))


def test_other_camera_change_fails_closed():
    original = np.tile(np.eye(4, dtype=np.float32), (6, 1, 1))
    candidate = original.astype(np.float64)
    candidate[0, 0, 0] += .1
    with pytest.raises(ValueError, match="rear 이외"):
        derive.corrected_projection(original, candidate)


def test_visibility_change_fails_closed(monkeypatch):
    masks = iter([np.array([[True, False]]), np.array([[True, True]])])
    monkeypatch.setattr(derive, "camera_visible", lambda _: next(masks))
    with pytest.raises(ValueError, match="재래스터"):
        derive.visibility_gate(None, None)


def test_visibility_exact_equality_pass(monkeypatch):
    monkeypatch.setattr(derive, "camera_visible", lambda _: np.array([[True, False]]))
    result = derive.visibility_gate(None, None)
    assert result["xor_cells"] == 0
    assert result["before_sha256"] == result["after_sha256"]


def test_existing_output_and_dangling_symlink_refused(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / derive.OUTPUT_NAME
    derive.validate_output(source, output)
    output.symlink_to(tmp_path / "does-not-exist")
    with pytest.raises(FileExistsError):
        derive.validate_output(source, output)
    with pytest.raises(ValueError):
        derive.validate_output(source, tmp_path / "different-edition")


def test_independent_copy_preserves_hardlinked_source(tmp_path):
    source = tmp_path / "original.bin"
    source.write_bytes(b"unchanged source")
    linked = tmp_path / "historical.bin"
    linked.hardlink_to(source)
    target = tmp_path / "independent.bin"
    digest = derive.sha256(source)
    derive.independent_copy(source, target, digest)
    assert source.stat().st_nlink == 2
    assert target.stat().st_nlink == 1
    assert source.stat().st_ino != target.stat().st_ino
    assert derive.sha256(linked) == digest
    with pytest.raises(FileExistsError):
        derive.independent_copy(source, target, digest)


def test_npz_arrays_keep_dtype_and_shape(tmp_path):
    target = tmp_path / "source.npz"
    array = np.array([[True, False]])
    np.savez_compressed(target, valid=array)
    expected = {"valid": {"dtype": "bool", "shape": [1, 2], "sha256": derive.array_sha(array)}}
    derive.verify_npz_arrays(target, expected)
    expected["valid"]["dtype"] = "int64"
    with pytest.raises(ValueError, match="valid"):
        derive.verify_npz_arrays(target, expected)


def test_scene_inventory_provenance_and_order(tmp_path):
    np.savez_compressed(tmp_path / "train-scene.npz", frame=np.arange(30, 32), row=np.arange(2),
                        time_offsets=np.ones((2, 4), np.float32))
    (tmp_path / "train-scene.json").write_text(json.dumps({"scene": "train-scene", "frames": 2,
        "split_manifest_sha256": "split", "ego_cache_sha256": "ego"}))
    got = derive.scene_inventory(tmp_path, "train-scene", "split", "ego", 2)
    assert got["frames"] == 2
    assert got["arrays"]["time_offsets"]["dtype"] == "float32"
    with pytest.raises(ValueError, match="provenance"):
        derive.scene_inventory(tmp_path, "train-scene", "wrong-split", "ego", 2)


def test_creation_is_new_root_only_and_byte_preserving(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / derive.OUTPUT_NAME
    np.savez_compressed(source / "train-scene.npz", frame=np.arange(30, 32), row=np.arange(2))
    (source / "train-scene.json").write_text(json.dumps({"scene": "train-scene", "frames": 2,
        "split_manifest_sha256": "split", "ego_cache_sha256": "ego"}))
    inventory = derive.scene_inventory(source, "train-scene", "split", "ego", 2)
    matrix = np.tile(np.eye(4, dtype=np.float32), (6, 1, 1))
    result = {"status": "preflight_pass", "visibility": {"xor_cells": 0},
              "source_root": str(source), "output_root": str(output), "source_scenes": {"train-scene": inventory},
              "new_lidar2img": matrix.tolist(), "new_lidar2img_array_sha256": derive.array_sha(matrix),
              "source_contract": {"calibration_sha256": "old", "derivation": {"older": True}},
              "source_sha256_before": {k: "hash" for k in ("source_manifest", "source_calibration_pkl",
                                                           "source_calibration", "cache_meta")}}
    monkeypatch.setattr(derive, "verify_source_unchanged", lambda _: None)
    out = derive.create_edition(result)
    assert out["status"] == "created_verified"
    for suffix in (".npz", ".json"):
        assert (source / f"train-scene{suffix}").read_bytes() == (output / f"train-scene{suffix}").read_bytes()
        assert (source / f"train-scene{suffix}").stat().st_ino != (output / f"train-scene{suffix}").stat().st_ino
    contract = json.loads((output / "supervision_manifest.json").read_text())
    assert contract["geometry_parent"]["prior_derivation"] == {"older": True}
    assert contract["calibration_sha256"] == "old"
    assert contract["schema_version"] == 2
    assert contract["canonical_calibration_sha256"] == derive.sha256(output / "calibration.npz")
    with pytest.raises(FileExistsError):
        derive.create_edition(result)
