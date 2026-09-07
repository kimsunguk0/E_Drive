#!/usr/bin/env python3
"""불변 rawtime 감독 캐시에서 별도 geometry edition을 파생한다.

기본 동작은 CPU preflight뿐이다. --mode create는 검토 후 별도 승인된 경우에만
사용한다. 기존 데이터에는 쓰지 않으며 기존 NPZ/JSON도 hardlink하지 않는다.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_grouped_split_v2 import sha256, validate_manifest
from build_scene_supervision_v2 import camera_visible
from motiondrive_v2_data import CAMERA_ORDER, GRID_EXTENT, GRID_SHAPE

REAR_INDEX = CAMERA_ORDER.index("camera_rear_wide")
OUTPUT_NAME = "train_tune_geometry_v2"


def array_sha(array):
    array = np.asarray(array)
    h = hashlib.sha256()
    h.update((str(array.dtype) + repr(array.shape)).encode())
    h.update(array.tobytes(order="C"))
    return h.hexdigest()


def load_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def validate_output(source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.name != OUTPUT_NAME or output.parent.resolve() != source.parent:
        raise ValueError("출력은 기존 감독 캐시와 같은 부모의 train_tune_geometry_v2만 허용합니다")
    if os.path.lexists(output):
        raise FileExistsError(f"기존 출력 또는 symlink를 사용할 수 없습니다: {output}")
    if output.resolve() == source:
        raise ValueError("원본 경로를 출력으로 사용할 수 없습니다")


def camera_projection(record, cache_camera):
    k = np.asarray(cache_camera["K_cache"], np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("캐시 intrinsic이 유효하지 않습니다")
    if cache_camera["out_size"] != [768, 432] or cache_camera["scale"] != .4:
        raise ValueError("캐시 해상도 또는 scale 계약이 다릅니다")
    crop = np.asarray(cache_camera["crop"], np.float64)
    if crop.shape != (4,) or not np.array_equal(crop[2:], [1920, 1080]):
        raise ValueError("캐시 crop 규격이 다릅니다")
    # K_cache 자체를 사용한다. new_K로부터 재구성은 메타의 내부 일관성 확인용이다.
    shift = np.eye(3)
    shift[0, 2], shift[1, 2] = -crop[0], -crop[1]
    # 캐시 생성기의 연산 순서도 그대로 보존한다. 먼저 차감한 후 scale하면
    # rear cy에 1.42e-14 반올림 차이가 생기므로 임의 tolerance로 덮지 않는다.
    expected_k = np.diag([.4, .4, 1.]) @ shift @ np.asarray(cache_camera["new_K"], np.float64)
    if not np.array_equal(k, expected_k):
        raise ValueError("K_cache와 메타 crop/new_K가 정확히 일치하지 않습니다")
    cam_to_ego = np.eye(4)
    cam_to_ego[:3, :3] = Rotation.from_euler("xyz", np.asarray(record["euler"], np.float64), degrees=True).as_matrix()
    cam_to_ego[:3, 3] = np.asarray(record["translation"], np.float64)
    kk = np.eye(4)
    kk[:3, :3] = k
    projection = kk @ np.linalg.inv(cam_to_ego)
    if not np.isfinite(projection).all():
        raise ValueError("비정상 투영행렬")
    return projection


def corrected_projection(original, derived):
    original, derived = np.asarray(original), np.asarray(derived)
    if original.shape != (6, 4, 4) or original.dtype != np.float32 or derived.shape != original.shape:
        raise ValueError("원본 float32 canonical projection 규격이 다릅니다")
    other = [index for index in range(6) if index != REAR_INDEX]
    # 원래 float32 canonical 변환과 float64 재구성 사이의 산술 반올림만 허용한다.
    other_error = float(np.max(np.abs(original[other].astype(np.float64) - derived[other])))
    if other_error > 1e-3:
        raise ValueError("rear 이외의 카메라에도 기하 차이가 있어 별도 검토가 필요합니다")
    result = original.copy()
    result[REAR_INDEX] = derived[REAR_INDEX].astype(np.float32)
    if not np.array_equal(result[other], original[other]):
        raise AssertionError("다른 5개 카메라 bitwise 보존 실패")
    if np.array_equal(result[REAR_INDEX], original[REAR_INDEX]):
        raise ValueError("rear 기하가 바뀌지 않아 이 edition 대상이 아닙니다")
    return result, other_error


def visibility_gate(original, corrected):
    before, after = camera_visible(original), camera_visible(corrected)
    result = {"grid_shape": list(GRID_SHAPE), "grid_extent": list(GRID_EXTENT),
              "heights": [0., 1.], "before_visible": int(before.sum()),
              "after_visible": int(after.sum()), "xor_cells": int(np.count_nonzero(before != after)),
              "before_sha256": array_sha(before), "after_sha256": array_sha(after)}
    if not np.array_equal(before, after):
        raise ValueError("감독 visibility가 달라 원시 라벨 재래스터가 필요합니다; 복사 파생 금지")
    return result


def scene_inventory(source, scene, split_sha, ego_sha, expected_frames):
    npz, report = Path(source) / f"{scene}.npz", Path(source) / f"{scene}.json"
    if npz.is_symlink() or report.is_symlink():
        raise ValueError("입력 scene 파일 symlink는 허용하지 않습니다")
    npz_before, report_before = sha256(npz), sha256(report)
    metadata = load_json(report)
    if (metadata.get("scene") != scene or metadata.get("split_manifest_sha256") != split_sha
            or metadata.get("ego_cache_sha256") != ego_sha):
        raise ValueError(f"원본 scene provenance 불일치: {scene}")
    arrays = {}
    with np.load(npz, allow_pickle=False) as z:
        for key in z.files:
            arr = z[key]
            if arr.dtype.hasobject:
                raise ValueError("object 배열은 허용하지 않습니다")
            arrays[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": array_sha(arr)}
        frames, rows = z["frame"], z["row"]
        if (len(frames) != expected_frames or len(rows) != expected_frames
                or not np.array_equal(frames, np.arange(30, 30 + expected_frames))
                or len(np.unique(rows)) != len(rows) or np.any(np.diff(rows) <= 0)):
            raise ValueError(f"원본 행/프레임 정렬 불일치: {scene}")
    if metadata["frames"] != expected_frames:
        raise ValueError("scene JSON과 NPZ의 행 수가 다릅니다")
    if sha256(npz) != npz_before or sha256(report) != report_before:
        raise ValueError("배열 검사 중 원본 scene이 바뀌었습니다")
    return {"npz": str(npz.resolve()), "json": str(report.resolve()),
            "npz_sha256": npz_before, "json_sha256": report_before,
            "arrays": arrays, "frames": expected_frames}


def preflight(*, source_root, output_root, split_manifest, ego_cache, cache_meta,
              meta_root, source_pkl, expected_counts=(203, 37), expected_frames=270):
    source = Path(source_root).resolve()
    validate_output(source, output_root)
    split = load_json(split_manifest)
    validate_manifest(split)
    selected = {key: sorted(split["splits"][key]) for key in ("train", "tune")}
    if tuple(len(selected[key]) for key in ("train", "tune")) != tuple(expected_counts):
        raise ValueError("승인된 train/tune 시나리오 수와 다릅니다")
    scenes = sorted(selected["train"] + selected["tune"])
    if len(set(scenes)) != len(scenes):
        raise ValueError("시나리오 중복")
    paths = {"split_manifest": Path(split_manifest).resolve(), "ego_cache": Path(ego_cache).resolve(),
             "cache_meta": Path(cache_meta).resolve(), "source_calibration_pkl": Path(source_pkl).resolve(),
             "source_manifest": source / "supervision_manifest.json", "source_calibration": source / "calibration.npz",
             "visibility_builder": Path(camera_visible.__code__.co_filename).resolve(),
             "data_loader": Path(__file__).resolve().parent / "motiondrive_v2_data.py"}
    before = {key: sha256(path) for key, path in paths.items()}
    contract = load_json(paths["source_manifest"])
    if (contract["split_manifest_sha256"] != before["split_manifest"]
            or contract["ego_cache_sha256"] != before["ego_cache"]
            or contract["calibration_sha256"] != before["source_calibration_pkl"]):
        raise ValueError("원본 split/ego/source-PKL SHA 계약 불일치")
    if (contract["grid_shape"] != list(GRID_SHAPE) or contract["grid_extent"] != list(GRID_EXTENT)
            or contract["history_frame_offsets"] != [1, 2, 5, 10]):
        raise ValueError("원본 감독 기하 또는 history 계약 불일치")
    metadata = load_json(cache_meta)
    with np.load(paths["source_calibration"], allow_pickle=False) as z:
        original = z["lidar2img"]
    reference_meta, reference_matrices = None, None
    per_scene_calibration, inventories = {}, {}
    for scene in scenes:
        # 전역 메타 파일에서도 승인된 train/tune 시나리오만 접근한다.
        meta = {camera: metadata[scene][camera] for camera in CAMERA_ORDER}
        if reference_meta is None:
            reference_meta = meta
        elif meta != reference_meta:
            raise ValueError(f"train/tune 카메라 캐시 메타가 공통이 아닙니다: {scene}")
        calibration = Path(meta_root) / scene / "calibration/calibration.parquet"
        rows = {row["camera_name"]: row for row in pq.read_table(calibration).to_pylist()}
        if set(rows) != set(CAMERA_ORDER):
            raise ValueError("카메라 calibration 구성 불일치")
        matrices = np.stack([camera_projection(rows[camera], meta[camera]) for camera in CAMERA_ORDER])
        if reference_matrices is None:
            reference_matrices = matrices
        elif not np.array_equal(matrices, reference_matrices):
            raise ValueError(f"시나리오별 기하가 달라 단일 calibration을 쓸 수 없습니다: {scene}")
        per_scene_calibration[scene] = {"path": str(calibration.resolve()), "sha256": sha256(calibration)}
        inventories[scene] = scene_inventory(source, scene, before["split_manifest"],
                                             before["ego_cache"], expected_frames)
    corrected, other_error = corrected_projection(original, reference_matrices)
    visibility = visibility_gate(original, corrected)
    after = {key: sha256(path) for key, path in paths.items()}
    if before != after:
        raise ValueError("preflight 중 원본 파일이 바뀌었습니다")
    for scene, item in inventories.items():
        if sha256(item["npz"]) != item["npz_sha256"] or sha256(item["json"]) != item["json_sha256"]:
            raise ValueError(f"preflight 중 원본 scene이 바뀌었습니다: {scene}")
    return {"status": "preflight_pass", "mode": "preflight", "source_root": str(source),
            "output_root": str(Path(output_root).absolute()), "split_scenes": selected,
            "scene_count": len(scenes), "frame_count": len(scenes) * expected_frames,
            "paths": {key: str(path) for key, path in paths.items()},
            "source_sha256_before": before, "source_sha256_after": after,
            "source_contract": contract, "per_scene_calibration": per_scene_calibration,
            "source_scenes": inventories, "source_lidar2img": original.tolist(),
            "new_lidar2img": corrected.tolist(), "new_lidar2img_array_sha256": array_sha(corrected),
            "other_five_max_abs_reconstruction_error": other_error,
            "other_five_bitwise_preserved": True, "visibility": visibility,
            "cache_metadata_all_selected_scenes_identical": True,
            "images": {"root": str(Path(cache_meta).resolve().parent),
                       "policy": "기존 이미지 경로 그대로 참조; 이미지 읽기/생성/변경 없음",
                       "all_image_bytes_rehashed": False},
            "preserved": ["모든 scene NPZ 배열", "scene JSON 바이트", "split", "ego GT/goal 캐시",
                          "time_offsets", "history/state", "프레임/행 순서", "영상 경로"],
            "changed": ["rear_wide canonical projection", "새 edition 전역 provenance"],
            "gpu_used": False, "output_directory_created": False,
            "script_sha256": sha256(__file__)}


def verify_source_unchanged(result):
    for key, path in result["paths"].items():
        if sha256(path) != result["source_sha256_before"][key]:
            raise ValueError(f"원본 SHA가 바뀌었습니다: {key}")
    for item in result["per_scene_calibration"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("원시 calibration이 바뀌었습니다")
    for item in result["source_scenes"].values():
        if sha256(item["npz"]) != item["npz_sha256"] or sha256(item["json"]) != item["json_sha256"]:
            raise ValueError("원본 scene 파일이 바뀌었습니다")


def independent_copy(source, target, expected_sha):
    source, target = Path(source), Path(target)
    if sha256(source) != expected_sha:
        raise ValueError("복사 직전 원본 SHA 불일치")
    # exclusive 예약으로 기존 파일을 절대로 덮어쓰지 않는다.
    with target.open("xb"):
        pass
    shutil.copyfile(source, target, follow_symlinks=False)
    if (sha256(source) != expected_sha or sha256(target) != expected_sha
            or (source.stat().st_dev, source.stat().st_ino) == (target.stat().st_dev, target.stat().st_ino)
            or target.stat().st_nlink != 1):
        raise ValueError("독립 복사·SHA 보존 검증 실패")


def verify_npz_arrays(path, expected):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(expected):
            raise ValueError("복사본 배열 키가 다릅니다")
        for key in z.files:
            array = z[key]
            if (str(array.dtype) != expected[key]["dtype"] or list(array.shape) != expected[key]["shape"]
                    or array_sha(array) != expected[key]["sha256"]):
                raise ValueError(f"복사본 배열이 달라졌습니다: {key}")


def create_edition(result):
    """별도 승인 후 사용. 실패한 새 디렉터리는 삭제하지 않고 검사 가능하게 남긴다."""
    if result["status"] != "preflight_pass" or result["visibility"]["xor_cells"] != 0:
        raise ValueError("통과한 preflight가 필요합니다")
    source, target = Path(result["source_root"]), Path(result["output_root"])
    validate_output(source, target)
    verify_source_unchanged(result)
    target.mkdir(exist_ok=False)
    copied = {}
    for scene, item in result["source_scenes"].items():
        npz, metadata = target / f"{scene}.npz", target / f"{scene}.json"
        independent_copy(item["npz"], npz, item["npz_sha256"])
        independent_copy(item["json"], metadata, item["json_sha256"])
        verify_npz_arrays(npz, item["arrays"])
        copied[scene] = {"npz_sha256": sha256(npz), "json_sha256": sha256(metadata),
                         "all_arrays_bitwise_preserved": True, "independent_inodes": True}
    corrected = np.asarray(result["new_lidar2img"], np.float32)
    if array_sha(corrected) != result["new_lidar2img_array_sha256"]:
        raise ValueError("preflight의 수정행렬이 달라졌습니다")
    calibration = target / "calibration.npz"
    with calibration.open("xb") as stream:
        np.savez_compressed(stream, lidar2img=corrected)
    new_contract = copy.deepcopy(result["source_contract"])
    # schema1 calibration_sha256는 원 source PKL 의미를 그대로 유지한다.
    new_contract["schema_version"] = 2
    new_contract["canonical_calibration_sha256"] = sha256(calibration)
    new_contract["canonical_calibration_source_sha256"] = result["source_sha256_before"]["source_calibration"]
    new_contract["geometry_edition"] = "cache_meta_rear_wide_v2"
    new_contract["geometry_parent"] = {
        "directory": str(source),
        "supervision_manifest_sha256": result["source_sha256_before"]["source_manifest"],
        "canonical_calibration_sha256": result["source_sha256_before"]["source_calibration"],
        "prior_derivation": new_contract.pop("derivation", None),
        "scene_json_policy": "원본 JSON 바이트 그대로 독립 복사; 내부 lineage는 parent edition을 기술함"}
    new_contract["derivation"] = {
        "method": "independent byte-identical scene copies; only rear projection corrected from actual K_cache/extrinsics",
        "source_directory": str(source), "source_manifest_sha256": result["source_sha256_before"]["source_manifest"],
        "source_calibration_pkl_sha256": result["source_sha256_before"]["source_calibration_pkl"],
        "source_canonical_calibration_sha256": result["source_sha256_before"]["source_calibration"],
        "cache_meta_sha256": result["source_sha256_before"]["cache_meta"],
        "canonical_calibration_sha256": sha256(calibration),
        "visibility": result["visibility"], "other_five_bitwise_preserved": True,
        "script_sha256": sha256(__file__)}
    verify_source_unchanged(result)
    with (target / "supervision_manifest.json").open("x") as stream:
        json.dump(new_contract, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    completed = copy.deepcopy(result)
    completed.update({"status": "created_verified", "mode": "create", "output_directory_created": True,
                      "copied_scenes": copied, "canonical_calibration_sha256": sha256(calibration),
                      "new_supervision_manifest_sha256": sha256(target / "supervision_manifest.json"),
                      "all_source_sha_after_copy_unchanged": True})
    with (target / "geometry_derivation_report.json").open("x") as stream:
        json.dump(completed, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return completed


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=("preflight", "create"), default="preflight")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--ego-cache", default="/tmp/pm97/data/etri/ego_cache.npz")
    parser.add_argument("--cache-meta", required=True)
    parser.add_argument("--meta-root", default="/tmp/pm97/data/etri/meta_train")
    parser.add_argument("--source-pkl", default="/tmp/pm97/data/etri/pkl/etri_train330_goal.pkl")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if os.path.lexists(args.report):
        raise FileExistsError("기존 보고서를 덮어쓰지 않습니다")
    result = preflight(source_root=args.source_root, output_root=args.output_root,
                       split_manifest=args.split_manifest, ego_cache=args.ego_cache,
                       cache_meta=args.cache_meta, meta_root=args.meta_root, source_pkl=args.source_pkl)
    if args.mode == "create":
        result = create_edition(result)
    with open(args.report, "x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "scenes": result["scene_count"],
                      "frames": result["frame_count"], "visibility": result["visibility"],
                      "output_directory_created": result["output_directory_created"],
                      "report": args.report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
