#!/usr/bin/env python3
"""실제 dataset 입력에서 rear 투영행렬만 바뀌었는지 CPU로 대조한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_grouped_split_v2 import sha256, validate_manifest
from motiondrive_v2_data import CAMERA_ORDER, HISTORY_OFFSETS, MotionDriveDataset

REAR = CAMERA_ORDER.index("camera_rear_wide")


def tensor_sha(tensor):
    if tensor.device.type != "cpu" or tensor.requires_grad:
        raise ValueError("CPU·requires_grad=False tensor만 허용합니다")
    h = hashlib.sha256((str(tensor.dtype) + repr(tuple(tensor.shape))).encode())
    h.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def selected_samples(manifest):
    validate_manifest(manifest)
    train, tune = sorted(manifest["splits"]["train"]), sorted(manifest["splits"]["tune"])
    if len(train) < 4 or not tune:
        raise ValueError("사전 고정 train4+tune1 시나리오가 필요합니다")
    return [("train", scene, frame) for scene in train[:4] for frame in (30, 180)] + [
        ("tune", tune[0], frame) for frame in (30, 295)]


def compare_samples(old, new):
    if set(old) != set(new):
        raise ValueError("dataset 반환 키가 다릅니다")
    if "lidar2img" not in old:
        raise ValueError("lidar2img 누락")
    checks, tensors = {}, {}
    for key in sorted(old):
        left, right = old[key], new[key]
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f"tensor 규격 변경: {key}")
            old_sha, new_sha = tensor_sha(left), tensor_sha(right)
            tensors[key] = {"shape": list(left.shape), "dtype": str(left.dtype),
                            "old_sha256": old_sha, "new_sha256": new_sha}
            if key == "lidar2img":
                if tuple(left.shape) != (6, 4, 4):
                    raise ValueError("lidar2img 규격 불일치")
                others = [i for i in range(6) if i != REAR]
                if tensor_sha(left[others]) != tensor_sha(right[others]):
                    raise ValueError("rear 외 카메라 행렬 변경")
                if tensor_sha(left[REAR]) == tensor_sha(right[REAR]):
                    raise ValueError("rear 행렬이 바뀌지 않았습니다")
                checks[key] = "rear_only_changed"
                tensors[key]["rear_max_abs_matrix_delta"] = float((left[REAR] - right[REAR]).abs().max())
            else:
                if old_sha != new_sha:
                    raise ValueError(f"허용되지 않은 dataset tensor 변경: {key}")
                checks[key] = "bitwise_equal"
        else:
            if type(left) is not type(right) or left != right:
                raise ValueError(f"dataset metadata 변경: {key}")
            checks[key] = "equal"
    return {"all_nonrear_values_bitwise_equal": True, "checks": checks, "tensors": tensors}


def audit(data_root, old_root, new_root, split_manifest, ego_cache):
    root, old_root, new_root = Path(data_root).resolve(), Path(old_root).resolve(), Path(new_root).resolve()
    if old_root == new_root:
        raise ValueError("서로 다른 edition 경로가 필요합니다")
    with open(split_manifest) as stream:
        manifest = json.load(stream)
    plan = selected_samples(manifest)
    with (new_root / "supervision_manifest.json").open() as stream:
        new_contract = json.load(stream)
    canonical_sha = sha256(new_root / "calibration.npz")
    if new_contract.get("schema_version") != 2 or new_contract.get("canonical_calibration_sha256") != canonical_sha:
        raise ValueError("새 canonical SHA 또는 schema 계약 불일치")
    images_root = root / "cache/etri_768"
    paths = {"split_manifest": Path(split_manifest).resolve(), "ego_cache": Path(ego_cache).resolve(),
             "old_manifest": old_root / "supervision_manifest.json", "new_manifest": new_root / "supervision_manifest.json",
             "old_calibration": old_root / "calibration.npz", "new_calibration": new_root / "calibration.npz",
             "loader": Path(sys.modules[MotionDriveDataset.__module__].__file__).resolve(),
             "script": Path(__file__).resolve()}
    before = {key: sha256(value) for key, value in paths.items()}
    records = []
    for split in ("train", "tune"):
        wanted = [(scene, frame) for s, scene, frame in plan if s == split]
        scenes, frames = sorted({s for s, _ in wanted}), sorted({f for _, f in wanted})
        shared = {"data_root": root, "split_manifest": split_manifest, "split": split,
                  "image_root": images_root, "ego_cache": ego_cache, "scenes": scenes, "frames": frames,
                  "augment": False, "frame_stride": 1, "min_frame": 30}
        old = MotionDriveDataset(supervision_root=old_root, **shared)
        new = MotionDriveDataset(supervision_root=new_root, **shared)
        if not np.array_equal(old.rows, new.rows) or len(old) != len(wanted):
            raise ValueError("두 dataset의 선택 row/개수가 다릅니다")
        if old.image_root.resolve() != new.image_root.resolve():
            raise ValueError("영상 경로 변경")
        lookup = {(str(old.scene_names[row]), int(old.arr["frame"][row])): i for i, row in enumerate(old.rows)}
        if set(lookup) != set(wanted):
            raise ValueError("사전 고정 scene/frame 선택과 dataset이 다릅니다")
        for scene, frame in wanted:
            index = lookup[scene, frame]
            old_sample, new_sample = old[index], new[index]
            compared = compare_samples(old_sample, new_sample)
            if old_sample["images"].shape != (6, 3, 432, 768) or old_sample["history_images"].shape != (4, 3, 216, 384):
                raise ValueError("실제 영상 입력 크기 불일치")
            current = [(camera, frame, "current") for camera in CAMERA_ORDER]
            history = [(CAMERA_ORDER[0], frame - int(offset), "history") for offset in HISTORY_OFFSETS]
            image_sources = []
            for camera, source_frame, kind in current + history:
                path = images_root / scene / camera / f"{source_frame:08d}.jpg"
                image_sources.append({"kind": kind, "camera": camera, "frame": source_frame,
                                      "path": str(path), "sha256": sha256(path)})
            source_files = {}
            for label, directory in (("old", old_root), ("new", new_root)):
                for suffix in ("npz", "json"):
                    path = directory / f"{scene}.{suffix}"
                    source_files[f"{label}_{suffix}"] = {"path": str(path), "sha256": sha256(path)}
            if source_files["old_npz"]["sha256"] != source_files["new_npz"]["sha256"]:
                raise ValueError("실제 읽은 scene NPZ 바이트 불일치")
            records.append({"split": split, "scene": scene, "frame": frame, "row": old_sample["row"],
                            "session_id": old_sample["session_id"], "comparison": compared,
                            "image_sources": image_sources, "supervision_sources": source_files})
    after = {key: sha256(value) for key, value in paths.items()}
    if before != after or torch.cuda.is_initialized():
        raise ValueError("감사 중 source 변경 또는 CUDA 초기화 발견")
    return {"status": "pass", "purpose": "실제 loader의 geometry edition 전환 정합 검증",
            "selection_policy": "train 이름순 첫4scene×frame30/180 + tune 이름순 첫1scene×frame30/295",
            "scope": {"train_samples": 8, "tune_samples": 2, "total_samples": 10,
                      "finalval_test_access": False, "large_fixture_created": False, "gpu_used": False},
            "data_roots": {"old": str(old_root), "new": str(new_root), "images": str(images_root)},
            "canonical_manifest_sha_matches": True, "new_canonical_sha256": canonical_sha,
            "source_paths": {key: str(value) for key, value in paths.items()},
            "source_sha256_before": before, "source_sha256_after": after,
            "all_samples_only_rear_projection_changed": True, "records": records}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--ego-cache", default="/tmp/pm97/data/etri/ego_cache.npz")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if os.path.lexists(args.report):
        raise FileExistsError("기존 보고서를 덮어쓸 수 없습니다")
    torch.set_num_threads(1)
    result = audit(args.data_root, args.old_root, args.new_root, args.split_manifest, args.ego_cache)
    with open(args.report, "x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "scope": result["scope"],
                      "canonical_manifest_sha_matches": result["canonical_manifest_sha_matches"],
                      "samples": [{key: row[key] for key in ("split", "scene", "frame", "row")} for row in result["records"]],
                      "report": args.report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
