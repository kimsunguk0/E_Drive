#!/usr/bin/env python3
"""P0 보조 과제 감사: 학습 라벨 상수 기준 및 실제 과거 영상 교란 비교.

점검 대상 checkpoint와 370개 tune 표본은 결과를 보기 전에 고정한다.
영상 교란은 분포 밖 진단이며 영상 정보량의 상한 증명으로 사용하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_grouped_split_v2 import sha256
from motiondrive_v2_data import HISTORY_OFFSETS, MotionDriveDataset
from motiondrive_v2_training import balanced_raster_bce, model_inputs, to_device

CONDITIONS = ("normal", "repeat_current", "reverse_history", "cross_scene_history")
STATE_NAMES = ("vx_m_s", "vy_m_s", "ax_m_s2", "ay_m_s2", "yaw_rate_rad_s")


def fixed_frames():
    grid = np.arange(30, 300, 5)
    return grid[np.linspace(0, len(grid) - 1, 10, dtype=int)].tolist()


def donor_mapping(scenes):
    """같은 frame 번호의 다음 scene을 donor로 선택하는 고정 순환 순열."""
    scenes = sorted(set(map(str, scenes)))
    if len(scenes) < 2:
        raise ValueError("서로 다른 두 scene 이상이 필요합니다")
    return {scene: scenes[(i + 1) % len(scenes)] for i, scene in enumerate(scenes)}


def history_condition(batch, condition):
    inputs = model_inputs(batch)
    if condition == "normal":
        return inputs
    result = dict(inputs)
    if condition == "repeat_current":
        current = F.interpolate(batch["images"][:, 0], size=batch["history_images"].shape[-2:],
                                mode="bilinear", align_corners=False, antialias=True)
        result["history_images"] = current[:, None].expand_as(batch["history_images"])
    elif condition == "reverse_history":
        result["history_images"] = batch["history_images"].flip(1)
    elif condition == "cross_scene_history":
        result["history_images"] = batch["donor_history_images"]
    else:
        raise ValueError(condition)
    # t0, 실제 시간간격, 원래 정렬행렬, goal은 전부 그대로 유지한다.
    return result


class AuditDataset(Dataset):
    def __init__(self, source):
        self.source = source
        self.donors = donor_mapping(source.manifest["splits"]["tune"])

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        batch = self.source[i]
        donor = self.donors[batch["scenario"]]
        batch["donor_history_images"] = torch.stack([
            self.source._image(donor, "camera_front", batch["frame"] - int(offset), (384, 216), None)
            for offset in HISTORY_OFFSETS])
        batch["donor_scenario"] = donor
        return batch


def constants_from_train(supervision_root, manifest):
    states, state_valid, histories, history_valid = [], [], [], []
    raster = {kind: [0, 0] for kind in ("occ", "lane")}
    for scene in manifest["splits"]["train"]:
        with np.load(Path(supervision_root) / f"{scene}.npz", allow_pickle=False) as z:
            states.append(z["state_target"])
            state_valid.append(z["state_valid"])
            histories.append(z["history_target"])
            history_valid.append(z["history_valid"])
            for kind in raster:
                mask = z[f"{kind}_valid"].astype(bool)
                raster[kind][0] += int(((z[f"{kind}_target"] >= .5) & mask).sum())
                raster[kind][1] += int(mask.sum())
    st, sv = np.concatenate(states), np.concatenate(state_valid)
    ht, hv = np.concatenate(histories), np.concatenate(history_valid)
    probability = float(np.mean(st[sv[:, 5], 5]))
    result = {"n_train_rows": len(st), "n_train_scenes": len(states),
              "source": "train 분리의 라벨만 사용; tune 라벨로 상수를 맞추지 않음",
              "stop_prevalence": probability,
              "raster_positive_prevalence": {k: p / max(n, 1) for k, (p, n) in raster.items()}}
    for name, reduce in (("mean", np.nanmean), ("median", np.nanmedian)):
        state = reduce(np.where(sv, st, np.nan), axis=0)
        history = reduce(np.where(hv, ht, np.nan), axis=0)
        if not np.isfinite(state).all() or not np.isfinite(history).all():
            raise ValueError("유효한 학습 상수를 구할 수 없습니다")
        state[5] = probability  # stop은 평균 사전확률을 두 상수 기준 모두에 적용한다.
        result[name] = {"state_physical_and_stop_probability": state.tolist(),
                        "history_xy_sin_cos": history.tolist()}
    return result


def safe_ratio(num, den):
    return float(num / den) if den else None


def confusion_report(tp, fp, fn, tn):
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "iou": safe_ratio(tp, tp + fp + fn),
            "precision": safe_ratio(tp, tp + fp), "recall": safe_ratio(tp, tp + fn),
            "accuracy": safe_ratio(tp + tn, tp + tn + fp + fn),
            "balanced_accuracy": float(.5 * (tp / (tp + fn) + tn / (tn + fp))) if tp + fn and tn + fp else None}


def correlation(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 2 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


class Accumulator:
    def __init__(self):
        self.state, self.gt_state, self.state_mask = [], [], []
        self.history, self.gt_history, self.history_mask = [], [], []
        self.scene, self.session, self.frame = [], [], []
        self.raster = {k: np.zeros(4, np.int64) for k in ("occ", "lane")}
        self.raster_bce = {k: [0., 0] for k in ("occ", "lane")}

    def add(self, outputs, batch, include_raster=True):
        for destination, tensor in ((self.state, outputs["state_hat"]), (self.history, outputs["history_hat"]),
                                    (self.gt_state, batch["state_target"]), (self.state_mask, batch["state_valid"]),
                                    (self.gt_history, batch["history_target"]), (self.history_mask, batch["history_valid"])):
            destination.append(tensor.detach().float().cpu().numpy())
        self.scene.extend(batch["scenario"])
        self.session.extend(batch["session_id"])
        self.frame.extend(batch["frame"].detach().cpu().tolist())
        if include_raster:
            for kind in self.raster:
                logit, gt, valid = outputs[f"{kind}_logits"], batch[f"{kind}_target"], batch[f"{kind}_valid"].bool()
                pred, truth = logit >= 0, gt >= .5
                counts = torch.stack([(pred & truth & valid).sum(), (pred & ~truth & valid).sum(),
                                      (~pred & truth & valid).sum(), (~pred & ~truth & valid).sum()])
                self.raster[kind] += counts.cpu().numpy()
                n = int(valid.sum())
                bce = F.binary_cross_entropy_with_logits(logit.float(), gt.float(), reduction="none")
                self.raster_bce[kind][0] += float(torch.where(valid, bce, 0.).sum())
                self.raster_bce[kind][1] += n

    def finish(self):
        pred, gt, valid = np.concatenate(self.state), np.concatenate(self.gt_state), np.concatenate(self.state_mask).astype(bool)
        hp, hg, hm = np.concatenate(self.history), np.concatenate(self.gt_history), np.concatenate(self.history_mask).astype(bool)
        report = {"n": len(pred), "n_scenes": len(set(self.scene)), "n_sessions": len(set(self.session)), "state": {}}
        for i, name in enumerate(STATE_NAMES):
            p, t = pred[valid[:, i], i], gt[valid[:, i], i]
            delta = p - t
            report["state"][name] = {"n": len(p), "mae": float(np.abs(delta).mean()),
                                     "rmse": float(np.sqrt(np.mean(delta ** 2))), "bias": float(delta.mean()),
                                     "p90_abs": float(np.quantile(np.abs(delta), .9)), "pearson": correlation(p, t)}
        probability = 1. / (1. + np.exp(-np.clip(pred[:, 5], -80, 80)))
        p, t, mask = probability >= .5, gt[:, 5] >= .5, valid[:, 5]
        counts = [int((p & t & mask).sum()), int((p & ~t & mask).sum()),
                  int((~p & t & mask).sum()), int((~p & ~t & mask).sum())]
        report["stop"] = {**confusion_report(*counts), "brier": float(np.mean((probability[mask] - gt[mask, 5]) ** 2)),
                            "gt_definition": "과거 1초 인과적 2차 fit의 평면 속도 < 0.2m/s", "probability_threshold": .5}
        xy_error = np.linalg.norm(hp[:, :, :2] - hg[:, :, :2], axis=-1)
        yaw_pred, yaw_gt = np.arctan2(hp[:, :, 2], hp[:, :, 3]), np.arctan2(hg[:, :, 2], hg[:, :, 3])
        yaw_error = np.abs(np.arctan2(np.sin(yaw_pred - yaw_gt), np.cos(yaw_pred - yaw_gt)))
        report["history"] = []
        for j, frame_offset in enumerate(HISTORY_OFFSETS):
            mv = hm[:, j, :2].all(-1)
            av = hm[:, j, 2:].all(-1)
            delta = hp[:, j, :2] - hg[:, j, :2]
            report["history"].append({"frame_offset": int(frame_offset), "n": int(mv.sum()),
                                      "position_mae_m": float(xy_error[mv, j].mean()),
                                      "position_rmse_m": float(np.sqrt(np.mean(xy_error[mv, j] ** 2))),
                                      "xy_mae_m": np.abs(delta[mv]).mean(0).tolist(),
                                      "yaw_mae_rad": float(yaw_error[av, j].mean()),
                                      "predicted_sincos_norm_mean": float(np.linalg.norm(hp[:, j, 2:], axis=-1).mean())})
        report["history_position_mae_mean4_m"] = float(np.mean([r["position_mae_m"] for r in report["history"]]))
        report["raster"] = {k: {**confusion_report(*v), "bce_valid_pixel_mean": safe_ratio(*self.raster_bce[k]),
                                 "threshold_probability": .5} for k, v in self.raster.items()}
        report["per_scene"] = {}
        scene_arr = np.asarray(self.scene)
        for scene in sorted(set(self.scene)):
            use = scene_arr == scene
            report["per_scene"][scene] = {"n": int(use.sum()), "vx_mae_m_s": float(np.abs(pred[use, 0] - gt[use, 0]).mean()),
                                          "history_position_mae_mean4_m": float(xy_error[use].mean())}
        return report


def baseline_report(constants, dataset):
    accumulators = {name: Accumulator() for name in ("mean", "median")}
    for i in range(len(dataset.rows)):
        row = int(dataset.rows[i]);scene = str(dataset.scene_names[row]);frame = int(dataset.arr["frame"][row])
        z = dataset._supervision(scene);at = z["frame_lookup"][frame]
        batch = {k: torch.from_numpy(z[k][at:at+1].copy()) for k in ("state_target", "history_target", "state_valid", "history_valid", "occ_target", "occ_valid", "lane_target", "lane_valid")}
        batch.update(scenario=[scene], session_id=[dataset.manifest["scene_to_session"][scene]], frame=torch.tensor([frame]))
        for name, acc in accumulators.items():
            values = np.array(constants[name]["state_physical_and_stop_probability"], np.float32)
            q = np.clip(values[5], 1e-6, 1 - 1e-6); values[5] = np.log(q / (1 - q))
            outputs = {"state_hat": torch.from_numpy(values[None]), "history_hat": torch.tensor(constants[name]["history_xy_sin_cos"])[None]}
            for k in ("occ", "lane"):
                q = np.clip(constants["raster_positive_prevalence"][k], 1e-6, 1 - 1e-6)
                outputs[f"{k}_logits"] = torch.full_like(batch[f"{k}_target"], float(np.log(q / (1 - q))))
            acc.add(outputs, batch)
    return {name: acc.finish() for name, acc in accumulators.items()}


@torch.inference_mode()
def evaluate_checkpoint(path, dataset, args, split_sha, supervision_sha):
    from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    manifest = checkpoint["manifest"]
    if manifest["split_sha256"] != split_sha or manifest["supervision_manifest_sha256"] != supervision_sha:
        raise ValueError("Checkpoint 분리 또는 supervision 계보 불일치")
    model = MotionDriveV2(MotionDriveV2Config(**manifest["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda:0")
    model.to(device).eval()
    loader = DataLoader(AuditDataset(dataset), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    accumulators = {name: Accumulator() for name in CONDITIONS}
    started = time.monotonic()
    for raw in loader:
        batch = to_device(raw, device)
        for condition in CONDITIONS:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**history_condition(batch, condition))
            if not all(torch.isfinite(outputs[k]).all() for k in ("state_hat", "history_hat", "occ_logits", "lane_logits")):
                raise FloatingPointError("진단 출력에 비유한 값이 있습니다")
            accumulators[condition].add(outputs, batch)
    torch.cuda.synchronize()
    report = {"path": str(path), "sha256": sha256(path), "step": checkpoint["step"],
              "model_config": manifest["model_config"], "elapsed_seconds": time.monotonic() - started,
              "conditions": {name: acc.finish() for name, acc in accumulators.items()}}
    normal = report["conditions"]["normal"]
    report["paired_change_from_normal"] = {
        condition: {"vx_mae_m_s": result["state"]["vx_m_s"]["mae"] - normal["state"]["vx_m_s"]["mae"],
                    "history_position_mae_mean4_m": result["history_position_mae_mean4_m"] - normal["history_position_mae_mean4_m"],
                    "occ_iou": result["raster"]["occ"]["iou"] - normal["raster"]["occ"]["iou"],
                    "lane_iou": result["raster"]["lane"]["iou"] - normal["raster"]["lane"]["iou"]}
        for condition, result in report["conditions"].items() if condition != "normal"}
    del model, checkpoint
    torch.cuda.empty_cache()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="/NHNHOME/data/sukim/adcl")
    p.add_argument("--run-dir", default="work_dirs/motiondrive_v2/p0_common_rawtime_s0")
    p.add_argument("--split-manifest", default="data/etri/motiondrive_v2/grouped_split_rawtime.json")
    p.add_argument("--supervision-root", default="data/etri/motiondrive_v2/train_tune_rawtime")
    p.add_argument("--output", default="reports/motiondrive_v2_p0_aux_audit.json")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prepare-only", action="store_true", help="CPU 기준값만 계산·출력하며 GPU는 사용하지 않음")
    a = p.parse_args()
    with open(a.split_manifest) as f:
        manifest = json.load(f)
    with (Path(a.run_dir) / "manifest.json").open() as f:
        run_manifest = json.load(f)
    if run_manifest["status"] != "completed":
        raise ValueError("종료되지 않은 학습 checkpoint는 감사하지 않습니다")
    dataset = MotionDriveDataset(a.data_root, a.split_manifest, split="tune", supervision_root=a.supervision_root,
                                frame_stride=5, frames=fixed_frames(), augment=False)
    if len(dataset) != len(manifest["splits"]["tune"]) * 10:
        raise ValueError("사전 고정한 scene당 10프레임 protocol 불일치")
    constants = constants_from_train(a.supervision_root, manifest)
    baselines = baseline_report(constants, dataset)
    report = {"status": "CPU 준비 완료", "split_manifest_sha256": sha256(a.split_manifest),
              "supervision_manifest_sha256": sha256(Path(a.supervision_root) / "supervision_manifest.json"),
              "protocol": {"split": "tune", "n_scenes": len(manifest["splits"]["tune"]), "n": len(dataset),
                           "frames_per_scene": fixed_frames(), "sample_selection": "stride5 grid에서 scene당 균등10개, 결과 확인 전 고정",
                           "checkpoints": ["initial.pth", "best.pth", "last.pth"],
                           "selection": "기존 전체 tune1998의 history_position_mae로 저장된 best 사용; 이번370개 결과로 재선택하지 않음",
                           "perturbation": "현재 영상·goal·시간간격·정렬행렬 고정, 과거 영상만 변경",
                           "donor_mapping": donor_mapping(manifest["splits"]["tune"])},
              "limitations": ["영상 교란은 분포 밖 입력 진단이며 정보량 상한이나 인과 원인의 완전한 증명이 아님",
                              "새 checkpoint 선택용 점수가 아니라 이미 고정한 checkpoint의 보조 과제 감사",
                              "P0 planner는 plan loss=0으로 미학습되어 planning 점수를 성공 지표로 사용하지 않음",
                              "occupancy는 유효 annotation footprint이며 음성 라벨은 보수적 영역 안 annotation 완전성 가정에 의존"],
              "train_constants": constants, "constant_baselines": baselines, "checkpoint_results": {}}
    print(json.dumps({"상태": "CPU 기준값 준비 완료", "표본": len(dataset),
                      "상수vx_MAE": {k: v["state"]["vx_m_s"]["mae"] for k, v in baselines.items()},
                      "상수history_MAE": {k: v["history_position_mae_mean4_m"] for k, v in baselines.items()}}, ensure_ascii=False), flush=True)
    if a.prepare_only:
        return
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(f"기존 감사 결과를 덮어쓰지 않습니다: {output}")
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    for name in ("initial", "best", "last"):
        result = evaluate_checkpoint(Path(a.run_dir) / f"{name}.pth", dataset, a,
                                     report["split_manifest_sha256"], report["supervision_manifest_sha256"])
        report["checkpoint_results"][name] = result
        print(json.dumps({"checkpoint": name, "step": result["step"],
                          "vx_MAE": {k: v["state"]["vx_m_s"]["mae"] for k, v in result["conditions"].items()},
                          "history_MAE": {k: v["history_position_mae_mean4_m"] for k, v in result["conditions"].items()}}, ensure_ascii=False), flush=True)
    report["status"] = "실제 GPU 0 보조 과제 감사 완료"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({"결과파일": str(output), "sha256": sha256(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
