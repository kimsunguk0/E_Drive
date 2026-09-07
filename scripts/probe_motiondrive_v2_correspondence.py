#!/usr/bin/env python3
"""해상도 경로별 feature 대응 정합을 비교하는 짧은 읽기 전용 진단.

동일 저해상도 입력의 자기 대응은 구성상 완벽한 sanity check다.
이 결과를 planning 개선이나 실제 optical flow 정답으로 해석하지 않는다.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_grouped_split_v2 import sha256
from motiondrive_v2_data import MotionDriveDataset
from models.motiondrive_v2.motion_encoder import local_correlation
from sparse_scoredrive import ResNet34FPN128


def shift_pixels(image, dx, dy):
    """영상 내용을 (+dx,+dy)로 이동; 빈 경계는 정규화 RGB 0으로 채움."""
    if int(dx) != dx or int(dy) != dy:
        raise ValueError("정수 pixel 이동만 지원합니다")
    dx, dy = int(dx), int(dy)
    h, w = image.shape[-2:]
    result = torch.zeros_like(image)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    if sx1 > sx0 and sy1 > sy0:
        result[..., sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = image[..., sy0:sy1, sx0:sx1]
    return result


def correlation_summary(current, other, radius=2, margin=4, expected_shift=None):
    """반경/경계를 고정하고 cosine cost peak의 정합만 계산한다."""
    if current.shape != other.shape:
        raise ValueError("동일 feature geometry가 필요합니다")
    if min(current.shape[-2:]) <= margin * 2:
        raise ValueError("내부 정합 영역이 비어 있습니다")
    cost = local_correlation(current, other, radius)
    cost = cost[..., margin:-margin, margin:-margin]
    width = 2 * radius + 1
    center = radius * width + radius
    peak, ids = cost.max(1)
    top2 = cost.topk(2, dim=1).values
    dx = (ids % width).float() - radius
    dy = (ids // width).float() - radius
    target = cost[:, center]
    result = {"center_cosine": float(target.mean()),
              "peak_cosine": float(peak.mean()), "center_peak_fraction": float((ids == center).float().mean()),
              "center_vs_best_gap": float((peak - target).mean()), "peak_top2_margin": float((top2[:, 0] - top2[:, 1]).mean()),
              "peak_dx_cells": float(dx.mean()), "peak_dy_cells": float(dy.mean()),
              "peak_displacement_cells": float(torch.sqrt(dx.square() + dy.square()).mean()),
              "search_boundary_fraction": float(((dx.abs() == radius) | (dy.abs() == radius)).float().mean()),
              "interior_locations": int(ids.numel())}
    if expected_shift is not None:
        ex, ey = expected_shift
        err = torch.sqrt((dx - ex).square() + (dy - ey).square())
        nearest = ((dx - ex).abs() <= .500001) & ((dy - ey).abs() <= .500001)
        result.update(expected_dx_cells=float(ex), expected_dy_cells=float(ey),
                      peak_error_cells=float(err.mean()), nearest_bin_fraction=float(nearest.float().mean()))
    return result


def aggregate_records(records):
    groups = defaultdict(list)
    for r in records:
        key = (r["split"], r["mode"], r["representation"], r["level"], r["condition"])
        groups[key].append(r)
    result = []
    for key, rows in sorted(groups.items()):
        metrics = {}
        for metric in rows[0]["metrics"]:
            values = np.asarray([r["metrics"][metric] for r in rows], np.float64)
            metrics[metric] = {"mean": float(values.mean()), "std_across_frames": float(values.std()),
                               "min": float(values.min()), "max": float(values.max())}
        result.append(dict(zip(("split", "mode", "representation", "level", "condition"), key), n_frames=len(rows), metrics=metrics))
    return result


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="/NHNHOME/data/sukim/adcl")
    p.add_argument("--checkpoint", default="work_dirs/motiondrive_v2/p0_common_rawtime_s0/best.pth")
    p.add_argument("--split-manifest", default="data/etri/motiondrive_v2/grouped_split_rawtime.json")
    p.add_argument("--supervision-root", default="data/etri/motiondrive_v2/train_tune_rawtime")
    p.add_argument("--output", default="reports/motiondrive_v2_correspondence_probe.json")
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
    output = Path(a.output)
    if output.exists() and not a.prepare_only:
        raise FileExistsError("기존 probe 결과를 덮어쓰지 않습니다")
    with open(a.split_manifest) as f:
        manifest = json.load(f)
    samples = {}
    for split in ("train", "tune"):
        scenes = sorted(manifest["splits"][split])
        samples[split] = [scenes[i] for i in np.linspace(0, len(scenes) - 1, 10, dtype=int)]
    print(json.dumps({"고정표본": samples, "frame": 150, "GPU사용": not a.prepare_only}, ensure_ascii=False), flush=True)
    if a.prepare_only:
        return
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["manifest"]["split_sha256"] != sha256(a.split_manifest):
        raise ValueError("Checkpoint 분리 계보 불일치")
    config = checkpoint["manifest"]["model_config"]
    state = checkpoint["model"]
    backbone = ResNet34FPN128(config["channels"], arch=config["backbone_arch"])
    backbone.load_state_dict({k.removeprefix("backbone_fpn."): v for k, v in state.items() if k.startswith("backbone_fpn.")}, strict=True)
    projections = nn.ModuleList([nn.Conv2d(config["channels"], config["correlation_channels"], 1, bias=False) for _ in range(2)])
    projections.load_state_dict({k.removeprefix("motion_encoder.projections."): v for k, v in state.items() if k.startswith("motion_encoder.projections.")}, strict=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    backbone.to(device).eval()
    projections.to(device).eval()
    shifts = [(v, 0) for v in (-8, -4, -2, -1, 1, 2, 4, 8)] + [(0, v) for v in (-8, -4, -2, -1, 1, 2, 4, 8)]
    records = []
    started = time.monotonic()
    for split, scenes in samples.items():
        dataset = MotionDriveDataset(a.data_root, a.split_manifest, split=split, supervision_root=a.supervision_root,
                                    scenes=scenes, frames=[150], augment=False)
        for sample in dataset:
            high = sample["images"][0:1].to(device)
            past = sample["history_images"].to(device)
            low = F.interpolate(high, (216, 384), mode="bilinear", align_corners=False, antialias=True)
            shifted = torch.cat([shift_pixels(low, dx, dy) for dx, dy in shifts])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                high_levels, _ = backbone(high)
                low_levels, _ = backbone(low)
                past_levels, _ = backbone(past)
                shift_levels, _ = backbone(shifted)
            for level, stride in enumerate((8, 16)):
                high_raw, low_raw = high_levels[level], low_levels[level]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    high_projected = projections[level](high_raw)
                    low_projected = projections[level](low_raw)
                    past_projected = projections[level](past_levels[level])
                    shift_projected = projections[level](shift_levels[level])
                options = {"raw_fpn128": (high_raw, low_raw, past_levels[level], shift_levels[level]),
                           "projected32": (high_projected, low_projected, past_projected, shift_projected)}
                for representation, (hi, lo, pa, sh) in options.items():
                    down = F.interpolate(hi, size=lo.shape[-2:], mode="bilinear", align_corners=False)
                    for mode, current in (("high_feature_down", down), ("low_image_encode", lo)):
                        conditions = [("same_t0", lo, (0., 0.))]
                        conditions += [(f"past_offset_{offset}", pa[i:i+1], None) for i, offset in enumerate((1, 2, 5, 10))]
                        conditions += [(f"shift_x{dx}_y{dy}", sh[i:i+1], (dx / stride, dy / stride)) for i, (dx, dy) in enumerate(shifts)]
                        for condition, other, expected in conditions:
                            metrics = correlation_summary(current, other, radius=config["correlation_radius"], margin=4,
                                                          expected_shift=expected)
                            records.append({"split": split, "scene": sample["scenario"], "frame": 150,
                                            "mode": mode, "representation": representation, "level": f"p{level+2}_stride{stride}",
                                            "condition": condition, "metrics": metrics})
            print(json.dumps({"완료scene": sample["scenario"], "split": split}, ensure_ascii=False), flush=True)
    torch.cuda.synchronize()
    report = {"상태": "feature 대응 정합 진단 완료", "checkpoint": {"path": a.checkpoint, "sha256": sha256(a.checkpoint), "step": checkpoint["step"]},
              "split_manifest_sha256": sha256(a.split_manifest), "samples": samples, "frame": 150,
              "protocol": {"n_frames": 20, "selection": "각 split의 scene 목록에서 균등10개, frame150 고정",
                           "precision": "backbone/projection bf16, normalized correlation fp32", "radius_cells": config["correlation_radius"],
                           "interior_margin_cells": 4, "shift_pixels_low_resolution": shifts,
                           "same_low_image": "동일한 저해상도 이미지에서 얻은 동일 feature 자기 대응이므로 center 정합은 구성상 완벽해야 함",
                           "resize": "현재 RGB를 bilinear/align_corners=False/antialias=True로384x216; 기존 feature 경로는 bilinear/align_corners=False",
                           "statistics": "각 프레임 내부 평균을 구한 뒤 프레임 간 평균/표준편차; 수천 feature 위치를 독립 표본으로 세지 않음"},
              "limits": ["center 정합 향상은 planning 개선을 뜻하지 않음", "실제 과거 영상의 정답 flow가 없으므로 peak 위치의 정확도를 단정하지 않음",
                         "pixel 이동은 CNN padding·aliasing 영향을 포함한 합성 진단", "stride8/16보다 작은 이동의 argmax가0인 것은 그 자체로 실패가 아님",
                         "full backbone/shared feature 학습을 바꾸지 않은 고정 checkpoint 진단"],
              "elapsed_seconds": time.monotonic() - started, "aggregates": aggregate_records(records)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({"결과파일": str(output), "sha256": sha256(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
