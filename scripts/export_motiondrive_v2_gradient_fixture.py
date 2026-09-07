#!/usr/bin/env python3
"""사전 고정 train8의 원본 MotionDriveDataset 출력을 CPU fixture로 보존한다.

학습/모델 forward/선택은 하지 않는다. 단일 collated batch8을 저장하고,
gradient probe는 metadata의 [0:4], [4:8]을 별도 배치로 진단한다.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_grouped_split_v2 import sha256, validate_manifest

FRAMES = (30, 180)
BATCH_SLICES = ((0, 4), (4, 8))
MODEL_INPUT_KEYS = ("images", "history_images", "lidar2img", "history_transforms", "time_offsets", "goal_xy")
REQUIRED_SHAPES = {"images": (8, 6, 3, 432, 768), "history_images": (8, 4, 3, 216, 384),
    "lidar2img": (8, 6, 4, 4), "history_transforms": (8, 4, 4, 4), "time_offsets": (8, 4),
    "goal_xy": (8, 2), "gt_plan": (8, 6, 2), "plan_valid": (8, 6),
    "history_target": (8, 4, 4), "history_valid": (8, 4, 4), "state_target": (8, 6), "state_valid": (8, 6),
    "occ_target": (8, 1, 64, 48), "lane_target": (8, 1, 64, 48),
    "occ_valid": (8, 1, 64, 48), "lane_valid": (8, 1, 64, 48),
    "row": (8,), "frame": (8,), "scen_idx": (8,)}


def fixed_train_selection(manifest):
    validate_manifest(manifest)
    scenes = sorted(manifest["splits"]["train"])
    if len(scenes) != 203 or len(set(scenes)) != len(scenes):
        raise ValueError("사전 고정 rawtime train203 분리가 아닙니다")
    return [(scene, frame) for scene in scenes[:4] for frame in FRAMES]


def ordered_samples(dataset, selected):
    """cache의 실제 행 번호를 보존하며 사전 고정 scene/frame 순서만 맞춘다."""
    lookup = {}
    for index, row in enumerate(dataset.rows):
        key = (str(dataset.scene_names[row]), int(dataset.arr["frame"][row]))
        if key in lookup:
            raise ValueError("dataset에 중복 scene/frame이 있습니다")
        lookup[key] = index
    if set(lookup) != set(selected) or len(lookup) != 8:
        raise ValueError("선택된 train8 외 표본이 있거나 고정 표본이 빠졌습니다")
    return [dataset[lookup[key]] for key in selected]


def check_cpu_tree(value, key="batch"):
    import torch
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.requires_grad:
            raise ValueError(f"CPU requires_grad=False tensor만 허용합니다: {key}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"비유한 tensor가 있습니다: {key}")
    elif isinstance(value, dict):
        for name, item in value.items():
            check_cpu_tree(item, f"{key}.{name}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            check_cpu_tree(item, f"{key}[{i}]")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError(f"weights_only 로드에 부적합한 값입니다: {key}: {type(value)}")


def validate_batch(batch, selected):
    import torch
    for key, shape in REQUIRED_SHAPES.items():
        if key not in batch or not isinstance(batch[key], torch.Tensor) or tuple(batch[key].shape) != shape:
            raise ValueError(f"원본 dataset tensor shape 계약 불일치: {key}")
    check_cpu_tree(batch)
    for key in ("plan_valid", "history_valid", "state_valid", "occ_valid", "lane_valid"):
        if batch[key].dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"valid mask의 bool/integer 원래 dtype을 보존해야 합니다: {key}")
    identities = list(zip(map(str, batch["scenario"]), map(int, batch["frame"].tolist())))
    if identities != selected or len(batch["session_id"]) != 8:
        raise ValueError("collated 표본 순서가 사전 고정 선택과 다릅니다")
    if not bool(batch["plan_valid"].all()) or not bool((batch["time_offsets"] > 0).all()):
        raise ValueError("유효한 3초 plan과 인과적 과거 시간이 필요합니다")


def ensure_new_pair(output, provenance):
    if output.resolve() == provenance.resolve():
        raise ValueError("tensor와 provenance 파일은 별도 경로여야 합니다")
    for path in (output, provenance):
        if os.path.lexists(path):
            raise FileExistsError(f"기존 fixture를 덮어쓰지 않습니다: {path}")


def publish_torch_new(path, fixture):
    import torch
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(fixture, stream)
            stream.flush();os.fsync(stream.fileno())
        # 실제 최종 소비자가 사용하는 안전한 로드 경로를 먼저 확인한다.
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        if loaded["metadata"] != fixture["metadata"] or set(loaded["batch"]) != set(fixture["batch"]):
            raise ValueError("fixture 저장/로드 계보가 일치하지 않습니다")
        for key, value in fixture["batch"].items():
            other = loaded["batch"][key]
            if isinstance(value, torch.Tensor):
                if value.dtype != other.dtype or not torch.equal(value, other):
                    raise ValueError(f"fixture roundtrip tensor 불일치: {key}")
            elif value != other:
                raise ValueError(f"fixture roundtrip metadata 불일치: {key}")
        del loaded
        os.link(temporary, path)  # 기존 target가 있으면 실패하며 덮어쓰지 않는다.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_json_new(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n");stream.flush();os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    # torch/dataset import 전에 숨긴다. GPU나 모델을 초기화하지 않는다.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/NHNHOME/data/sukim/adcl")
    parser.add_argument("--split-manifest", default="data/etri/motiondrive_v2/grouped_split_rawtime.json")
    parser.add_argument("--supervision-root", default="data/etri/motiondrive_v2/train_tune_rawtime")
    parser.add_argument("--output", default="checkpoints/motiondrive_v2/diagnostics/p1_gradient_train8.pt")
    parser.add_argument("--provenance", default="checkpoints/motiondrive_v2/diagnostics/p1_gradient_train8.provenance.json")
    args = parser.parse_args()
    output, provenance = Path(args.output), Path(args.provenance)
    ensure_new_pair(output, provenance)
    import torch
    from torch.utils.data import default_collate
    from motiondrive_v2_data import CAMERA_ORDER, HISTORY_OFFSETS, MEAN, STD, MotionDriveDataset
    if torch.cuda.is_initialized():
        raise RuntimeError("GPU가 이미 초기화된 프로세스에서는 CPU 전용 fixture를 만들지 않습니다")
    manifest = json.loads(Path(args.split_manifest).read_text())
    selected = fixed_train_selection(manifest)
    scenes = sorted({scene for scene, _ in selected})
    cache = Path(args.data_root) / "data/etri/ego_cache.npz"
    if not cache.exists():
        cache = Path("/tmp/pm97/data/etri/ego_cache.npz")
    if not cache.is_file():
        raise FileNotFoundError("실제 dataset ego cache가 없습니다")
    dataset = MotionDriveDataset(args.data_root, args.split_manifest, split="train", supervision_root=args.supervision_root,
        ego_cache=cache, min_frame=30, frame_stride=1, augment=False, seed=0, scenes=scenes, frames=FRAMES)
    samples = ordered_samples(dataset, selected)
    batch = default_collate(samples)
    validate_batch(batch, selected)
    supervision = Path(args.supervision_root)
    source_paths = [Path(args.split_manifest), cache, supervision / "supervision_manifest.json", supervision / "calibration.npz",
                    *[supervision / f"{scene}{suffix}" for scene in scenes for suffix in (".npz", ".json")]]
    sources = {str(path.resolve()): sha256(path) for path in source_paths}
    image_sources = {}
    for scene, frame in selected:
        images = [dataset.image_root / scene / camera / f"{frame:08d}.jpg" for camera in CAMERA_ORDER]
        images += [dataset.image_root / scene / CAMERA_ORDER[0] / f"{frame-int(offset):08d}.jpg" for offset in HISTORY_OFFSETS]
        for path in images:
            image_sources[str(path.resolve())] = sha256(path)
    identities = [{"scenario": scene, "frame": frame, "row": int(batch["row"][i]),
                   "scen_idx": int(batch["scen_idx"][i]), "session_id": batch["session_id"][i]}
                  for i, (scene, frame) in enumerate(selected)]
    metadata = {"schema_version": 1, "purpose": "P1의 loss별 gradient 기여 진단용. 학습·모델 선택·heldout 평가 없음",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "split": "train", "train_only": True,
        "selection": "rawtime train203의 scene 이름순 첫4개 × frame30/180; 결과를 보기 전에 고정",
        "n_train_scenes_in_manifest": 203, "n_samples": 8, "batch_size": 4, "batch_slices": [list(x) for x in BATCH_SLICES],
        "batch_note": "두 배치의 gradient를 별도 보고하며 전체8 단일배치 gradient로 간주하지 않음",
        "augment": False, "seed": 0, "samples": identities, "model_input_keys": list(MODEL_INPUT_KEYS),
        "source_sha256": sources, "image_source_sha256": image_sources,
        "split_manifest_path": str(Path(args.split_manifest).resolve()), "split_manifest_sha256": dataset.split_sha,
        "ego_cache_path": str(cache.resolve()), "ego_cache_sha256": dataset.cache_sha,
        "supervision_root": str(supervision.resolve()), "supervision_manifest_sha256": sha256(supervision / "supervision_manifest.json"),
        "image_cache_root": str(dataset.image_root.resolve()), "camera_order": list(CAMERA_ORDER),
        "history_frame_offsets": HISTORY_OFFSETS.tolist(), "image_normalization": {"mean": MEAN.tolist(), "std": STD.tolist()},
        "history_resize": "dataset의 PIL BILINEAR 768×432→384×216 원출력",
        "dataset_access": "표본·영상·scene supervision은 선택한 train4 scene만 사용. 표준 loader의 공유 ego_cache에서 실제 행/계보를 확인함",
        "code_sha256": {str(Path(__file__).resolve()): sha256(__file__),
                         str(Path(sys.modules["motiondrive_v2_data"].__file__).resolve()): sha256(sys.modules["motiondrive_v2_data"].__file__)},
        "tensor_schema": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
                                "requires_grad": value.requires_grad} for key, value in batch.items() if isinstance(value, torch.Tensor)},
        "valid_counts": {key: {"valid": int(batch[key].bool().sum()), "total": batch[key].numel()}
                         for key in ("plan_valid", "history_valid", "state_valid", "occ_valid", "lane_valid")},
        "gpu_initialized": torch.cuda.is_initialized()}
    if metadata["gpu_initialized"]:
        raise RuntimeError("CPU export 중 예상하지 않은 GPU 초기화")
    output.parent.mkdir(parents=True, exist_ok=True);provenance.parent.mkdir(parents=True, exist_ok=True)
    ensure_new_pair(output, provenance)
    publish_torch_new(output, {"batch": batch, "metadata": metadata})
    final = {"status": "CPU train8 fixture 생성 및 weights_only roundtrip 검증 완료", "metadata": metadata,
             "artifact": {"path": str(output.resolve()), "sha256": sha256(output), "size_bytes": output.stat().st_size}}
    publish_json_new(provenance, final)
    print(json.dumps({"fixture": str(output), "sha256": final["artifact"]["sha256"], "bytes": final["artifact"]["size_bytes"],
                      "provenance": str(provenance), "provenance_sha256": sha256(provenance),
                      "samples": identities, "valid_counts": metadata["valid_counts"], "gpu_initialized": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
