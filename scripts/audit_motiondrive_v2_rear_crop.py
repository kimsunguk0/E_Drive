#!/usr/bin/env python3
"""학습 clip 한 개의 원본/캐시/투영행렬을 읽기 전용으로 대조한다."""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation


CAMERAS = ("camera_front", "camera_front_right", "camera_front_left",
           "camera_rear_wide", "camera_rear_left", "camera_rear_right")


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), sha_bytes(raw)


def new_intrinsic(k, distortion, size, fisheye):
    import cv2
    if fisheye:
        return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            k, distortion[:4], size, np.eye(3), balance=0.0)
    return cv2.getOptimalNewCameraMatrix(k, distortion, size, alpha=0)[0]


def pixel_metrics(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("재구성 영상 크기가 캐시와 다릅니다")
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {"bitwise_equal": bool(np.array_equal(actual, expected)),
            "mae_uint8": float(delta.mean()), "max_abs_uint8": float(delta.max()),
            "rmse_uint8": float(np.sqrt(np.mean(delta ** 2))),
            "equal_channel_fraction": float(np.mean(delta == 0))}


def projected_matrix(row, nk, crop, scale=.4):
    camera_to_ego = np.eye(4)
    camera_to_ego[:3, :3] = Rotation.from_euler("xyz", row["euler"], degrees=True).as_matrix()
    camera_to_ego[:3, 3] = row["translation"]
    intrinsic = np.eye(4)
    intrinsic[:3, :3] = nk
    shift = np.eye(4)
    shift[0, 2], shift[1, 2] = -crop[0], -crop[1]
    return np.diag([scale, scale, 1., 1.]) @ shift @ intrinsic @ np.linalg.inv(camera_to_ego)


def audit(root, scene, frame, manifest_path):
    root = Path(root)
    manifest, split_sha = read_json(manifest_path)
    if scene not in manifest["splits"]["train"]:
        raise ValueError("원본 영상 검사는 train 시나리오만 허용합니다")
    import cv2
    import pyarrow.parquet as pq
    cache_root = root / "cache/etri_768"
    metadata, metadata_sha = read_json(cache_root / "cache_meta.json")
    archive = root / "train" / f"{scene}.tar"
    image_tail = f"/camera_rear_wide/{frame:08d}.jpg"
    with tarfile.open(archive, "r:") as tar:
        # 이 학습 archive의 헤더만 조사하며, 선택한 두 member만 읽는다.
        members = tar.getmembers()
        images = [m for m in members if ("/" + m.name).endswith(image_tail)]
        calibrations = [m for m in members if m.name.endswith("/calibration/calibration.parquet")]
        if len(images) != 1 or len(calibrations) != 1:
            raise ValueError(f"선택 member가 유일하지 않습니다: {len(images)}, {len(calibrations)}")
        raw_jpeg = tar.extractfile(images[0]).read()
        raw_calibration = tar.extractfile(calibrations[0]).read()
        archive_members = {"image": images[0].name, "calibration": calibrations[0].name}
    table = pq.read_table(io.BytesIO(raw_calibration)).to_pylist()
    rows = {row["camera_name"]: row for row in table}
    if set(rows) != set(CAMERAS):
        raise ValueError("캘리브레이션 카메라 구성이 다릅니다")
    canonical_path = root / "data/etri/motiondrive_v2/train_tune_rawtime/calibration.npz"
    canonical_bytes = canonical_path.read_bytes()
    with np.load(io.BytesIO(canonical_bytes), allow_pickle=False) as z:
        canonical = z["lidar2img"]
    if canonical.shape != (6, 4, 4):
        raise ValueError("V2 투영행렬 크기가 다릅니다")
    matrices = {}
    for index, camera in enumerate(CAMERAS):
        row = rows[camera]
        size = (int(row["image_width"]), int(row["image_height"]))
        k = np.asarray(row["K"], np.float64).reshape(3, 3)
        distortion = np.asarray(row["distortion"], np.float64)
        nk = new_intrinsic(k, distortion, size, bool(row["is_fisheye"]))
        meta = metadata[scene][camera]
        meta_crop = meta["crop"]
        v2_crop = [(size[0] - 1920) // 2, size[1] - 1080 if camera == "camera_front" else 0,
                   1920, 1080]
        meta_matrix = projected_matrix(row, nk, meta_crop)
        v2_matrix = projected_matrix(row, nk, v2_crop)
        meta_k = nk.copy()
        meta_k[0, 2] -= meta_crop[0]
        meta_k[1, 2] -= meta_crop[1]
        meta_k = np.diag([.4, .4, 1.]) @ meta_k
        matrices[camera] = {
            "raw_wh": list(size), "metadata_crop": meta_crop, "v2_crop": v2_crop,
            "new_K_max_abs_vs_metadata": float(np.max(np.abs(nk - np.asarray(meta["new_K"])))),
            "K_cache_max_abs_vs_reconstructed": float(np.max(np.abs(meta_k - np.asarray(meta["K_cache"])))),
            "saved_projection_max_abs_vs_metadata_crop": float(np.max(np.abs(canonical[index] - meta_matrix))),
            "saved_projection_max_abs_vs_v2_crop": float(np.max(np.abs(canonical[index] - v2_matrix))),
            "metadata_minus_v2_projected_dy_px": float(.4 * (v2_crop[1] - meta_crop[1])),
            "metadata_K_cache": meta["K_cache"], "reconstructed_v2_projection": v2_matrix.tolist(),
            "saved_projection": canonical[index].tolist()}
    raw = cv2.imdecode(np.frombuffer(raw_jpeg, np.uint8), cv2.IMREAD_COLOR)
    row = rows["camera_rear_wide"]
    h, w = raw.shape[:2]
    if (w, h) != (row["image_width"], row["image_height"]):
        raise ValueError("원본 JPEG와 캘리브레이션 해상도가 다릅니다")
    k = np.asarray(row["K"], np.float64).reshape(3, 3)
    distortion = np.asarray(row["distortion"], np.float64)
    nk = new_intrinsic(k, distortion, (w, h), bool(row["is_fisheye"]))
    if row["is_fisheye"]:
        maps = cv2.fisheye.initUndistortRectifyMap(k, distortion[:4], np.eye(3), nk, (w, h), cv2.CV_16SC2)
    else:
        maps = cv2.initUndistortRectifyMap(k, distortion, None, nk, (w, h), cv2.CV_16SC2)
    undistorted = cv2.remap(raw, *maps, cv2.INTER_LINEAR)
    cached_path = cache_root / scene / "camera_rear_wide" / f"{frame:08d}.jpg"
    cached_bytes = cached_path.read_bytes()
    cached = cv2.imdecode(np.frombuffer(cached_bytes, np.uint8), cv2.IMREAD_COLOR)
    reconstructions = {}
    ox = (w - 1920) // 2
    for policy, oy in (("top", 0), ("bottom", h - 1080)):
        small = cv2.resize(undistorted[oy:oy + 1080, ox:ox + 1920], (768, 432), interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise ValueError("메모리 JPEG 재구성 실패")
        encoded_bytes = encoded.tobytes()
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        reconstructions[policy] = {"crop": [ox, oy, 1920, 1080],
                                   "jpeg_sha256": sha_bytes(encoded_bytes),
                                   "jpeg_bytes_equal_cached": encoded_bytes == cached_bytes,
                                   **pixel_metrics(cached, decoded)}
    # 전역 메타 파일 중 train/tune 키만 접근. finalval/test 영상·라벨은 접근하지 않는다.
    scope = {}
    for split in ("train", "tune"):
        selected = manifest["splits"][split]
        missing = sorted(set(selected) - metadata.keys())
        if missing:
            raise ValueError(f"{split} 캐시 메타 누락: {missing[:2]}")
        crops = collections.Counter()
        same_k = 0
        for name in selected:
            meta = metadata[name]["camera_rear_wide"]
            crops[tuple(meta["crop"])] += 1
            same_k += bool(np.array_equal(np.asarray(meta["K_cache"]),
                                          np.asarray(metadata[scene]["camera_rear_wide"]["K_cache"])))
        scope[split] = {"scenes": len(selected), "rear_crop_counts": {str(k): v for k, v in crops.items()},
                        "same_rear_K_cache_as_audited_scene": same_k,
                        "evidence": "메타만 대조; 모든 시나리오 JPEG의 픽셀 일치를 증명한 것은 아님"}
    bottom_exact = reconstructions["bottom"]["bitwise_equal"]
    top_exact = reconstructions["top"]["bitwise_equal"]
    mismatch = (bottom_exact and not top_exact
                and matrices["camera_rear_wide"]["saved_projection_max_abs_vs_v2_crop"] < 1e-3
                and matrices["camera_rear_wide"]["saved_projection_max_abs_vs_metadata_crop"] > 1.)
    return {"purpose": "train-only rear_wide 캐시/투영행렬 정합 감사", "scene": scene, "frame": frame,
            "cpu_only": True, "data_mutations": False, "finalval_test_access": False,
            "archive": str(archive), "archive_members": archive_members,
            "source_sha256": {"raw_jpeg": sha_bytes(raw_jpeg), "raw_calibration": sha_bytes(raw_calibration),
                              "cached_jpeg": sha_bytes(cached_bytes), "cache_meta": metadata_sha,
                              "split_manifest": split_sha, "canonical_projection": sha_bytes(canonical_bytes),
                              "script": sha_bytes(Path(__file__).read_bytes())},
            "opencv": cv2.__version__, "raw_wh": [w, h], "cached_wh": [cached.shape[1], cached.shape[0]],
            "camera_geometry": matrices, "pixel_reconstruction": reconstructions,
            "metadata_scope": scope, "confirmed_selected_frame_geometry_mismatch": bool(mismatch),
            "limitations": ["원본 픽셀 재구성은 사전 지정 train 1개 시나리오의 frame30 rear_wide에 한정",
                            "train/tune 전체에 대한 증거는 캐시 메타의 일치이며 전체 영상 재검사는 아님",
                            "이 감사는 모델 성능 변화나 다른 branch 영향의 크기를 측정하지 않음"]}


def main():
    import cv2
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scene", default="20260112-105434")
    parser.add_argument("--frame", type=int, default=30)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError("기존 진단 보고서를 덮어쓸 수 없습니다")
    cv2.setNumThreads(1)
    result = audit(args.root, args.scene, args.frame, args.manifest)
    with open(args.out, "x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"confirmed_geometry_mismatch": result["confirmed_selected_frame_geometry_mismatch"],
                      "pixel_reconstruction": result["pixel_reconstruction"],
                      "metadata_scope": result["metadata_scope"], "report": args.out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
