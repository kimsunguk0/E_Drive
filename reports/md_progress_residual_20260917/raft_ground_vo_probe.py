#!/usr/bin/env python3
"""Pose-free metric ego-motion probe from RAFT flow and fixed calibration.

The model sees only image pairs.  A fixed ground-plane grid and the canonical
camera matrices turn optical-flow correspondences into a current-to-past SE(2)
estimate.  Dynamic pose is loaded only after inference for diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.optimize import differential_evolution, least_squares, minimize_scalar
import torch
from torch.nn import functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from motiondrive_v2_data import CAMERA_ORDER, MotionDriveDataset  # noqa: E402


SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json"
SUPERVISION = ROOT / "data/etri/motiondrive_v2/r0reset_tplus_geometry_v2"
EVAL_JSON = ROOT / "work_dirs/md_progress_residual_20260917/FRONT-S-s1/warmup/final_eval.json"
ORACLE_JSON = ROOT / "reports/md_progress_residual_20260917/oracle_FRONT-S_warmup_base_cap1_records.json"


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as stream:
        image = np.asarray(stream.convert("RGB"), dtype=np.float32) / 255.0
    if image.shape != (432, 768, 3):
        raise ValueError(f"unexpected image shape {path}: {image.shape}")
    return torch.from_numpy(image).permute(2, 0, 1)


def project(matrix: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.column_stack((xyz, np.ones(len(xyz), dtype=np.float64)))
    value = homogeneous @ matrix.T
    uv = value[:, :2] / np.maximum(value[:, 2:3], 1.0e-8)
    return uv, value[:, 2]


def sample_map(value: torch.Tensor, uv: np.ndarray) -> np.ndarray:
    """Bilinearly sample a [C,H,W] tensor at pixel-center coordinates."""
    _, height, width = value.shape
    grid = torch.as_tensor(uv, device=value.device, dtype=torch.float32)
    grid = torch.stack(((2.0 * grid[:, 0] + 1.0) / width - 1.0,
                        (2.0 * grid[:, 1] + 1.0) / height - 1.0), -1)
    sampled = F.grid_sample(value[None], grid.view(1, -1, 1, 2),
                            mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled[0, :, :, 0].T.float().cpu().numpy()


def image_gradient(image: torch.Tensor) -> torch.Tensor:
    gray = (image * image.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0, keepdim=True)
    kernel_x = gray.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])[None, None]
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(gray[None], kernel_x, padding=1)[0]
    gy = F.conv2d(gray[None], kernel_y, padding=1)[0]
    return torch.sqrt(gx.square() + gy.square() + 1.0e-8)


def solve_homography(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """Normalized DLT for a past-pixel to current-pixel homography."""
    if len(source) < 4:
        return None

    def normalize(points):
        center = points.mean(0)
        distance = np.linalg.norm(points - center, axis=1).mean()
        scale = math.sqrt(2.0) / max(distance, 1.0e-6)
        matrix = np.asarray([[scale, 0.0, -scale * center[0]],
                             [0.0, scale, -scale * center[1]],
                             [0.0, 0.0, 1.0]])
        homogeneous = np.column_stack((points, np.ones(len(points))))
        return (homogeneous @ matrix.T)[:, :2], matrix

    src, source_matrix = normalize(source)
    dst, target_matrix = normalize(target)
    x, y = src[:, 0], src[:, 1]
    u, v = dst[:, 0], dst[:, 1]
    zero = np.zeros_like(x)
    one = np.ones_like(x)
    design = np.stack([
        np.stack([-x, -y, -one, zero, zero, zero, u * x, u * y, u], -1),
        np.stack([zero, zero, zero, -x, -y, -one, v * x, v * y, v], -1),
    ], 1).reshape(-1, 9)
    try:
        _, _, vh = np.linalg.svd(design, full_matrices=True)
    except np.linalg.LinAlgError:
        return None
    normalized_h = vh[-1].reshape(3, 3)
    homography = np.linalg.inv(target_matrix) @ normalized_h @ source_matrix
    if not np.isfinite(homography).all() or abs(homography[2, 2]) < 1.0e-9:
        return None
    return homography / homography[2, 2]


def homography_ransac(source: np.ndarray, target: np.ndarray, threshold: float,
                      seed: int) -> np.ndarray:
    """Return a deterministic dominant-plane mask without an OpenCV dependency."""
    count = len(source)
    if threshold <= 0 or count < 8:
        return np.ones(count, dtype=bool)
    rng = np.random.default_rng(seed)
    source_h = np.column_stack((source, np.ones(count)))
    best_mask = np.zeros(count, dtype=bool)
    best_key = (-1, -np.inf)
    for _ in range(384):
        choice = rng.choice(count, 4, replace=False)
        homography = solve_homography(source[choice], target[choice])
        if homography is None:
            continue
        projected_h = source_h @ homography.T
        valid = np.abs(projected_h[:, 2]) > 1.0e-9
        projected = np.full_like(target, np.inf)
        projected[valid] = projected_h[valid, :2] / projected_h[valid, 2:3]
        error = np.linalg.norm(projected - target, axis=-1)
        mask = error <= threshold
        key = (int(mask.sum()), -float(np.median(error[mask])) if mask.any() else -np.inf)
        if key > best_key:
            best_key, best_mask = key, mask
    if best_mask.sum() >= 8:
        homography = solve_homography(source[best_mask], target[best_mask])
        if homography is not None:
            projected_h = source_h @ homography.T
            valid = np.abs(projected_h[:, 2]) > 1.0e-9
            projected = np.full_like(target, np.inf)
            projected[valid] = projected_h[valid, :2] / projected_h[valid, 2:3]
            refined = np.linalg.norm(projected - target, axis=-1) <= threshold
            if refined.sum() >= best_mask.sum() // 2:
                best_mask = refined
    return best_mask if best_mask.sum() >= 8 else np.ones(count, dtype=bool)


def fixed_ground_points(calibration: np.ndarray, cameras: tuple[int, ...], ground_z: float):
    x = np.arange(1.0, 61.0, 0.75, dtype=np.float64)
    y = np.arange(-30.0, 30.01, 0.75, dtype=np.float64)
    xy = np.stack(np.meshgrid(x, y, indexing="ij"), -1).reshape(-1, 2)
    xyz = np.column_stack((xy, np.full(len(xy), ground_z, dtype=np.float64)))
    result = {}
    for camera in cameras:
        uv, depth = project(calibration[camera].astype(np.float64), xyz)
        valid = ((depth > 1.0) & (uv[:, 0] >= 4.0) & (uv[:, 0] <= 763.0)
                 & (uv[:, 1] >= 4.0) & (uv[:, 1] <= 427.0))
        result[camera] = (xyz[valid], uv[valid])
    return result


def predict_current_uv(observation: dict, calibration: np.ndarray,
                       parameter: np.ndarray) -> np.ndarray:
        tx, ty, yaw = parameter
        cosine, sine = math.cos(yaw), math.sin(yaw)
        # X_current = R(yaw)^T (Y_past - t)
        rotation_t = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
        past_xyz = observation["xyz"]
        current_xy = (past_xyz[:, :2] - np.asarray([tx, ty])) @ rotation_t.T
        current_xyz = np.column_stack((current_xy, past_xyz[:, 2]))
        predicted, _ = project(calibration[observation["camera"]], current_xyz)
        return predicted


def fit_motion(observations: list[dict], calibration: np.ndarray, offset_seconds: float,
               method: str, inlier_fraction: float,
               fixed_lateral_yaw: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Fit T(current->past)=[tx,ty,yaw] to past-ground/current-pixel matches."""
    def residual(parameter: np.ndarray) -> np.ndarray:
        pieces = []
        for observation in observations:
            past_xyz = observation["xyz"]
            predicted = predict_current_uv(observation, calibration, parameter)
            current_xy = (past_xyz[:, :2] - np.asarray([parameter[0], parameter[1]])) @ np.asarray(
                [[math.cos(parameter[2]), -math.sin(parameter[2])],
                 [math.sin(parameter[2]), math.cos(parameter[2])]])
            current_xyz = np.column_stack((current_xy, past_xyz[:, 2]))
            _, depth = project(calibration[observation["camera"]], current_xyz)
            error = predicted - observation["current_uv"]
            visible = ((depth > 0.5) & np.isfinite(error).all(1)
                       & (predicted[:, 0] > -64) & (predicted[:, 0] < 832)
                       & (predicted[:, 1] > -64) & (predicted[:, 1] < 496))
            error[~visible] = 64.0
            pieces.append((error * observation["sqrt_weight"][:, None]).reshape(-1))
        return np.concatenate(pieces)

    max_tx = 24.0 * offset_seconds
    max_ty = 4.0 * offset_seconds
    max_yaw = 0.5 * offset_seconds
    lower = np.asarray([-2.0 * offset_seconds, -max_ty, -max_yaw])
    upper = np.asarray([max_tx, max_ty, max_yaw])
    def trimmed_objective(parameter):
        errors = []
        for observation in observations:
            predicted = predict_current_uv(observation, calibration, parameter)
            value = np.linalg.norm(predicted - observation["current_uv"], axis=-1)
            value = value / np.clip(observation["sqrt_weight"], 0.25, 1.0)
            value[~np.isfinite(value)] = 256.0
            errors.append(value)
        errors = np.concatenate(errors)
        count = max(24, int(round(len(errors) * inlier_fraction)))
        return float(np.partition(errors, count - 1)[:count].mean())

    if fixed_lateral_yaw is not None:
        fixed_lateral_yaw = np.asarray(fixed_lateral_yaw, dtype=np.float64)
        if fixed_lateral_yaw.shape != (2,):
            raise ValueError("fixed lateral/yaw must contain ty,yaw")
        result = minimize_scalar(
            lambda tx: trimmed_objective(np.asarray([tx, *fixed_lateral_yaw])),
            bounds=(lower[0], upper[0]), method="bounded",
            options={"xatol": 1.0e-5, "maxiter": 160})
        parameter = np.asarray([result.x, *fixed_lateral_yaw])
        return parameter, float(result.fun)

    if method == "least_squares":
        candidates = []
        for fraction in (0.0, 0.25, 0.5, 0.75):
            start = lower + fraction * (upper - lower)
            start[1:] = 0.0
            result = least_squares(residual, start, bounds=(lower, upper),
                                   loss="soft_l1", f_scale=2.0, max_nfev=80)
            candidates.append(result)
        best = min(candidates, key=lambda item: item.cost)
        return best.x.astype(np.float64), float(best.cost)

    if method != "trimmed":
        raise ValueError(f"unknown fit method: {method}")

    result = differential_evolution(
        trimmed_objective, list(zip(lower, upper)), seed=20260917,
        maxiter=32, popsize=10, tol=1.0e-4, polish=True, updating="immediate")
    return result.x.astype(np.float64), float(result.fun)


def correlation(x, y):
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1]) if len(x) > 2 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--offset", type=int, choices=(1, 2, 5), default=5)
    parser.add_argument("--ground-z", type=float, default=0.0)
    parser.add_argument("--cameras", default="0,1,2")
    parser.add_argument("--fb-threshold", type=float, default=3.0)
    parser.add_argument("--max-points-camera", type=int, default=700)
    parser.add_argument("--ransac-threshold", type=float, default=2.0)
    parser.add_argument("--fit", choices=("trimmed", "least_squares"), default="trimmed")
    parser.add_argument("--inlier-fraction", type=float, default=0.6)
    parser.add_argument("--fixed-lateral-yaw", choices=("none", "predicted", "target"),
                        default="none")
    parser.add_argument("--output", default=str(ROOT / "reports/md_progress_residual_20260917/raft_ground_vo_probe.json"))
    args = parser.parse_args()
    if not 0.25 <= args.inlier_fraction <= 1.0:
        raise ValueError("inlier fraction must be in [0.25, 1]")
    target_index = {1: 0, 2: 1, 5: 2}[args.offset]
    offset_seconds = args.offset * 0.1

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    weights = Raft_Large_Weights.C_T_SKHT_K_V2
    model = raft_large(weights=weights, progress=False).eval().to(device)
    transform = weights.transforms()

    dataset = MotionDriveDataset(
        ROOT, SPLIT, split="tune", supervision_root=SUPERVISION,
        min_frame=30, frame_stride=5, augment=False)
    if len(dataset) != 1998:
        raise ValueError(f"expected canonical V0 rows, got {len(dataset)}")
    sample_indices = np.linspace(0, len(dataset) - 1, args.samples, dtype=np.int64)
    cameras = tuple(int(value) for value in args.cameras.split(",") if value != "")
    if not cameras or any(camera < 0 or camera >= len(CAMERA_ORDER) for camera in cameras):
        raise ValueError(f"invalid cameras: {args.cameras}")
    ground = fixed_ground_points(dataset.lidar2img.astype(np.float64), cameras, args.ground_z)

    with EVAL_JSON.open() as stream:
        evaluation = json.load(stream)
    with ORACLE_JSON.open() as stream:
        oracle = json.load(stream)
    eval_by_row = {int(record["row"]): record for record in evaluation["records"]}
    oracle_by_row = {
        int(record["row"]): float(coefficient[0])
        for record, coefficient in zip(evaluation["records"],
                                       oracle["arms"]["scalar_dv"]["record_coefficients"])
    }

    records = []
    for number, dataset_index in enumerate(sample_indices, 1):
        row = int(dataset.rows[int(dataset_index)])
        scene = str(dataset.scene_names[row])
        frame = int(dataset.arr["frame"][row])
        supervision = dataset._supervision(scene)
        si = supervision["frame_lookup"][frame]
        true_transform = supervision["history_transforms"][si, target_index].astype(np.float64)
        true_parameter = np.asarray([
            true_transform[0, 3], true_transform[1, 3],
            math.atan2(true_transform[1, 0], true_transform[0, 0]),
        ])

        past_images, current_images = [], []
        for camera in cameras:
            name = CAMERA_ORDER[camera]
            current_images.append(load_rgb(dataset.image_root / scene / name / f"{frame:08d}.jpg"))
            past_images.append(load_rgb(dataset.image_root / scene / name / f"{frame-args.offset:08d}.jpg"))
        past_cpu = torch.stack(past_images)
        current_cpu = torch.stack(current_images)
        past, current = transform(past_cpu, current_cpu)
        past, current = past.to(device), current.to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            forward = model(past, current)[-1].float()
            backward = model(current, past)[-1].float()

        observations = []
        point_counts = {}
        for local_camera, camera in enumerate(cameras):
            xyz, past_uv = ground[camera]
            flow = sample_map(forward[local_camera], past_uv)
            current_uv = past_uv + flow
            backward_at_current = sample_map(backward[local_camera], current_uv)
            fb_error = np.linalg.norm(flow + backward_at_current, axis=-1)
            gradient = sample_map(image_gradient(past_cpu[local_camera]).to(device), past_uv)[:, 0]
            valid = ((current_uv[:, 0] >= 4) & (current_uv[:, 0] <= 763)
                     & (current_uv[:, 1] >= 4) & (current_uv[:, 1] <= 427)
                     & np.isfinite(current_uv).all(1) & (fb_error <= args.fb_threshold))
            indices = np.flatnonzero(valid)
            raw_count = int(len(indices))
            if len(indices) >= 24 and args.ransac_threshold > 0:
                inlier = homography_ransac(
                    past_uv[indices], current_uv[indices], args.ransac_threshold,
                    seed=20260917 + row * 7 + camera)
                if int(inlier.sum()) >= 24:
                    indices = indices[inlier]
            if len(indices) > args.max_points_camera:
                # Retain the most textured ground points after consistency filtering.
                order = np.argsort(gradient[indices])[-args.max_points_camera:]
                indices = indices[order]
            if len(indices) < 24:
                continue
            grad_scale = np.percentile(gradient[indices], 75) + 1.0e-6
            reliability = np.clip(gradient[indices] / grad_scale, 0.1, 1.0)
            reliability *= np.clip(1.0 - fb_error[indices] / args.fb_threshold, 0.05, 1.0)
            observations.append({
                "camera": camera,
                "xyz": xyz[indices],
                "current_uv": current_uv[indices],
                "sqrt_weight": np.sqrt(reliability),
            })
            point_counts[CAMERA_ORDER[camera]] = {
                "forward_backward": raw_count,
                "homography_inlier": int(len(indices)),
            }
        if not observations:
            raise RuntimeError(f"no valid correspondences for {scene}/{frame}")
        predicted_state = np.asarray(eval_by_row[row]["pred_state"], dtype=np.float64)
        fixed_lateral_yaw = None
        if args.fixed_lateral_yaw == "predicted":
            fixed_lateral_yaw = np.asarray([
                predicted_state[1] * offset_seconds,
                predicted_state[4] * offset_seconds,
            ])
        elif args.fixed_lateral_yaw == "target":
            fixed_lateral_yaw = true_parameter[1:]
        estimate, cost = fit_motion(
            observations, dataset.lidar2img.astype(np.float64), offset_seconds,
            args.fit, args.inlier_fraction, fixed_lateral_yaw)
        pixel_diagnostics = {}
        inverse_transform = np.linalg.inv(true_transform)
        inverse_parameter = np.asarray([
            inverse_transform[0, 3], inverse_transform[1, 3],
            math.atan2(inverse_transform[1, 0], inverse_transform[0, 0]),
        ])
        for label, parameter in (("identity", np.zeros(3)), ("target", true_parameter),
                                 ("inverse_target", inverse_parameter), ("estimate", estimate)):
            camera_errors = {}
            for observation in observations:
                predicted = predict_current_uv(observation, dataset.lidar2img.astype(np.float64), parameter)
                error = np.linalg.norm(predicted - observation["current_uv"], axis=-1)
                camera_errors[CAMERA_ORDER[observation["camera"]]] = {
                    "median_px": float(np.median(error)),
                    "mean_px": float(np.mean(error)),
                }
            pixel_diagnostics[label] = camera_errors
        base = np.asarray(eval_by_row[row]["base_abs_xy"], dtype=np.float64)
        base_v0 = np.linalg.norm(base[0]) / 0.5
        item = {
            "row": row, "scene": scene, "frame": frame,
            "estimate_tx_ty_yaw": estimate.tolist(),
            "target_tx_ty_yaw": true_parameter.tolist(),
            "cost": cost, "points": point_counts,
            "estimated_v": float(estimate[0] / offset_seconds),
            "target_v": float(true_parameter[0] / offset_seconds),
            "base_v0": float(base_v0),
            "oracle_c": oracle_by_row[row],
            "bucket": eval_by_row[row]["bucket"],
            "pixel_diagnostics": pixel_diagnostics,
        }
        records.append(item)
        print(f"[{number:03d}/{len(sample_indices)}] {scene}/{frame} "
              f"tx={estimate[0]:.3f}/{true_parameter[0]:.3f} "
              f"ty={estimate[1]:.3f}/{true_parameter[1]:.3f} "
              f"yaw={estimate[2]:.4f}/{true_parameter[2]:.4f} "
              f"points={sum(value['homography_inlier'] for value in point_counts.values())}",
              flush=True)

    estimated = np.asarray([record["estimate_tx_ty_yaw"] for record in records])
    target = np.asarray([record["target_tx_ty_yaw"] for record in records])
    estimated_v = estimated[:, 0] / offset_seconds
    target_v = target[:, 0] / offset_seconds
    base_v = np.asarray([record["base_v0"] for record in records])
    oracle_c = np.asarray([record["oracle_c"] for record in records])
    report = {
        "schema_version": 1,
        "input_contract": "images plus fixed calibration; dynamic pose is evaluation target only",
        "flow": str(weights),
        "offset_seconds": offset_seconds,
        "ground_z": args.ground_z,
        "fit": args.fit,
        "inlier_fraction": args.inlier_fraction,
        "fixed_lateral_yaw": args.fixed_lateral_yaw,
        "n": len(records),
        "metrics": {
            "tx_mae_m": float(np.mean(np.abs(estimated[:, 0] - target[:, 0]))),
            "ty_mae_m": float(np.mean(np.abs(estimated[:, 1] - target[:, 1]))),
            "yaw_mae_rad": float(np.mean(np.abs(estimated[:, 2] - target[:, 2]))),
            "speed_mae_mps": float(np.mean(np.abs(estimated_v - target_v))),
            "speed_corr": correlation(estimated_v, target_v),
            "estimated_speed_minus_base_vs_oracle_corr": correlation(estimated_v - base_v, oracle_c),
            "target_speed_minus_base_vs_oracle_corr": correlation(target_v - base_v, oracle_c),
        },
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["metrics"], indent=2))
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
