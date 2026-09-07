#!/usr/bin/env python3
"""고정 train8을 공식 테스트형 raw 입력으로 복사한다. GT·상태 fixture가 아니다.

각 clip에는 JPEG10장, 원본 calibration, frame -30..0,+50의 pose32행만 둔다.
학습 scene/frame 출처와 모든 SHA는 clip 밖의 root manifest에만 기록한다.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_grouped_split_v2 import sha256, validate_manifest

CAMERAS = ("camera_front", "camera_front_right", "camera_front_left",
           "camera_rear_wide", "camera_rear_left", "camera_rear_right")
PAST = (-1, -2, -5, -10)
POSE_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
POSE_COLUMNS = ("frame", *POSE_FIELDS)
POSE_FRAMES = tuple(range(-30, 1)) + (50,)
OUTPUT_NAME = "deploy_fixture_train8"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sample_plan(manifest):
    validate_manifest(manifest)
    train = sorted(manifest["splits"]["train"])
    if len(train) < 4:
        raise ValueError("train 시나리오4개가 필요합니다")
    for scene in train[:4]:
        if re.fullmatch(r"\d{8}-\d{5,6}", scene) is None:
            raise ValueError("유효하지 않은 train 시나리오 이름")
    return [(scene, frame) for scene in train[:4] for frame in (30, 180)]


def jpeg_plan(scene, anchor):
    cameras_and_frames = [(camera, 0) for camera in CAMERAS] + [(CAMERAS[0], f) for f in PAST]
    return [(f"{scene}/{camera}/{anchor + relative:08d}.jpg", f"{camera}/frame_{relative}.jpg")
            for camera, relative in cameras_and_frames]


def relative_pose_table(timestamps, ego_pose, anchor):
    """timestamp는 정렬·join에만 쓰고 출력 schema에서 제거한다."""
    required_ts, required_pose = {"timestamp", "frame_id"}, {"timestamp", *POSE_FIELDS}
    if not required_ts <= set(timestamps) or not required_pose <= set(ego_pose):
        raise ValueError("원본 pose/timestamp schema 불일치")
    ts = timestamps.loc[:, ["timestamp", "frame_id"]].copy()
    ep = ego_pose.loc[:, ["timestamp", *POSE_FIELDS]].copy()
    if ts["timestamp"].duplicated().any() or ts["frame_id"].duplicated().any() or ep["timestamp"].duplicated().any():
        raise ValueError("중복 timestamp/frame으로 pose 정렬 불가")
    joined = ts.merge(ep, on="timestamp", how="left", validate="one_to_one").sort_values("frame_id")
    if (joined["timestamp"].diff().dropna() <= 0).any():
        raise ValueError("frame 순서의 timestamp가 단조 증가하지 않습니다")
    joined = joined.set_index("frame_id")
    absolute = [anchor + frame for frame in POSE_FRAMES]
    if not set(absolute) <= set(joined.index):
        raise ValueError("선택 과거31행 또는+50 goal pose가 없습니다")
    selected = joined.loc[absolute, list(POSE_FIELDS)].copy()
    if not np.isfinite(selected.to_numpy(np.float64)).all():
        raise ValueError("선택 pose에 누락 또는 비정상 값이 있습니다")
    selected.insert(0, "frame", np.asarray(POSE_FRAMES, np.int64))
    selected = selected.reset_index(drop=True)
    table = pa.Table.from_pandas(selected, preserve_index=False).replace_schema_metadata(None)
    if tuple(table.column_names) != POSE_COLUMNS or table.num_rows != 32:
        raise AssertionError("공식 pose schema 생성 실패")
    return table


def member_bytes(tar, name):
    members = [member for member in tar.getmembers() if member.name == name]
    if len(members) != 1 or not members[0].isfile():
        raise ValueError(f"유일한 일반 member가 아닙니다: {name}")
    stream = tar.extractfile(members[0])
    if stream is None:
        raise ValueError(f"member를 읽을 수 없습니다: {name}")
    return stream.read()


def encode_pose(table):
    stream = io.BytesIO()
    pq.write_table(table, stream)
    data = stream.getvalue()
    restored = pq.read_table(io.BytesIO(data))
    if not restored.equals(table, check_metadata=True):
        raise AssertionError("pose parquet 메모리 왕복 불일치")
    return data


def validate_clip_files(files):
    expected = {target for _, target in jpeg_plan("unused", 30)} | {"calibration.parquet", "ego_pose.parquet"}
    if set(files) != expected:
        raise ValueError("clip 출력에 누락 또는 비허용 파일이 있습니다")
    pose = pq.read_table(io.BytesIO(files["ego_pose.parquet"]))
    if tuple(pose.column_names) != POSE_COLUMNS or pose["frame"].to_pylist() != list(POSE_FRAMES):
        raise ValueError("출력 pose에 비허용 열·행이 있습니다")
    if pose.schema.metadata:
        raise ValueError("출력 pose에 source metadata가 남았습니다")


def build(data_root, split_manifest, output_root):
    root = Path(data_root).resolve()
    output = Path(output_root).absolute()
    required_output = root / "data/etri/motiondrive_v2" / OUTPUT_NAME
    if output != required_output or output.parent.resolve() != required_output.parent.resolve():
        raise ValueError("출력은 승인된 deploy_fixture_train8 전용 root만 허용합니다")
    if os.path.lexists(output):
        raise FileExistsError("기존 fixture root를 변경하지 않습니다")
    if not output.parent.is_dir():
        raise ValueError("기존 motiondrive_v2 부모 디렉터리가 필요합니다")
    split_sha = sha256(split_manifest)
    with open(split_manifest) as stream:
        manifest = json.load(stream)
    plan = sample_plan(manifest)
    payloads, sources = [], {}
    for scene in sorted({scene for scene, _ in plan}):
        archive = root / "train" / f"{scene}.tar"
        archive_sha = sha256(archive)
        with tarfile.open(archive, "r:") as tar:
            names = {"calibration": f"{scene}/calibration/calibration.parquet",
                     "timestamps": f"{scene}/meta/timestamps.parquet",
                     "ego_pose": f"{scene}/annotation/ego_pose.parquet"}
            raw = {key: member_bytes(tar, name) for key, name in names.items()}
            timestamps = pq.read_table(io.BytesIO(raw["timestamps"]), columns=["timestamp", "frame_id"]).to_pandas()
            ego_pose = pq.read_table(io.BytesIO(raw["ego_pose"]), columns=["timestamp", *POSE_FIELDS]).to_pandas()
            calibration = {row["camera_name"]: row for row in pq.read_table(io.BytesIO(raw["calibration"])).to_pylist()}
            if set(calibration) != set(CAMERAS):
                raise ValueError("calibration 카메라 구성 불일치")
            for _, anchor in (item for item in plan if item[0] == scene):
                files, member_sources = {}, []
                for member_name, target in jpeg_plan(scene, anchor):
                    jpeg = member_bytes(tar, member_name)
                    camera = target.split("/")[0]
                    with Image.open(io.BytesIO(jpeg)) as image:
                        if image.format != "JPEG" or image.size != (calibration[camera]["image_width"], calibration[camera]["image_height"]):
                            raise ValueError(f"JPEG 해상도/형식 불일치: {member_name}")
                    files[target] = jpeg
                    member_sources.append({"archive_member": member_name, "output_relative_path": target,
                                           "sha256": digest(jpeg), "bytes": len(jpeg)})
                files["calibration.parquet"] = raw["calibration"]
                files["ego_pose.parquet"] = encode_pose(relative_pose_table(timestamps, ego_pose, anchor))
                validate_clip_files(files)
                payloads.append({"scene": scene, "anchor": anchor, "files": files, "jpeg_sources": member_sources})
        if sha256(archive) != archive_sha:
            raise ValueError(f"준비 중 원본 archive 변경: {scene}")
        sources[scene] = {"archive": str(archive), "archive_sha256_before": archive_sha,
                          "archive_sha256_after_preparation": archive_sha,
                          "metadata_members": {key: {"member": names[key], "sha256": digest(raw[key]), "bytes": len(raw[key])}
                                               for key in names}}
    if sha256(split_manifest) != split_sha or len(payloads) != 8:
        raise ValueError("split 변경 또는 fixture 개수 불일치")
    # 모든 입력 준비·검증 이후에만 새 root를 exclusive 생성한다.
    output.mkdir(exist_ok=False)
    records = []
    for index, payload in enumerate(payloads):
        clip_id = f"fixture_{index:03d}"
        clip = output / clip_id
        clip.mkdir(exist_ok=False)
        outputs = {}
        for relative, data in payload["files"].items():
            path = clip / relative
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
            if sha256(path) != digest(data):
                raise ValueError("출력 파일 SHA 검증 실패")
            outputs[relative] = {"sha256": digest(data), "bytes": len(data)}
        actual_files = {str(path.relative_to(clip)) for path in clip.rglob("*") if path.is_file()}
        if actual_files != set(payload["files"]):
            raise ValueError("실제 clip 파일 구성 불일치")
        records.append({"clip_id": clip_id, "clip_directory": str(clip), "source_split": "train",
                        "source_scene": payload["scene"], "source_frame": payload["anchor"],
                        "model_must_not_receive_source_mapping": True, "jpeg_sources": payload["jpeg_sources"],
                        "outputs": outputs})
    for source in sources.values():
        source["archive_sha256_after_write"] = sha256(source["archive"])
        if source["archive_sha256_after_write"] != source["archive_sha256_before"]:
            raise ValueError("출력 생성 중 원본 archive가 바뀌었습니다")
    if sha256(split_manifest) != split_sha:
        raise ValueError("출력 생성 중 split 변경")
    result = {"status": "created_verified", "purpose": "train-only 공식 테스트형 GT-free adapter 입력 검증",
              "selection": "train 이름순 첫4scene×frame30/180, 성능 기반 선택 없음",
              "clip_count": 8, "jpeg_count": 80, "pose_rows_per_clip": 32,
              "camera_order": list(CAMERAS), "history_frames": list(PAST), "pose_columns": list(POSE_COLUMNS),
              "pose_relative_frames": list(POSE_FRAMES), "time_offsets_contract": "nominal seconds [0.1,0.2,0.5,1.0]; no raw timestamp supplied",
              "intended_geometry": "geometry_v2: 실제 cached crop에 맞춘 투영행렬",
              "source_split_manifest": str(Path(split_manifest).resolve()), "split_manifest_sha256": split_sha,
              "sources": sources, "clips": records, "script_sha256": sha256(__file__),
              "root_manifest_not_a_model_input": True, "gpu_used": False,
              "tune_finalval_test_access": False, "full_archive_extraction": False,
              "output_gt_future_status_object_map": False, "original_sources_unchanged": True,
              "note": "+50의 제공 goal pose는 공식 허용 입력을 train에서 모사한 것; 미래1..30 궤적/상태 라벨은 출력하지 않음"}
    with (output / "fixture_manifest.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = build(args.data_root, args.split_manifest, args.output_root)
    print(json.dumps({"status": result["status"], "clips": result["clip_count"], "images": result["jpeg_count"],
                      "source_scenes": sorted(result["sources"]), "output_root": args.output_root}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
