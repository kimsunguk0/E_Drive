#!/usr/bin/env python3
"""기존 normal 평가 records만 사용하는 미래-GT 사후 시간 형태 진단.

학습·배포 보정 도구가 아니다. 표본별 계수나 fitted/corrected 경로를 저장하지
않으며, 원자료/추가 validation/GPU에 접근하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

TIMES = np.arange(1, 7, dtype=np.float64) * .5
WEIGHTS = np.asarray([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36
DESIGN = np.column_stack((TIMES, TIMES ** 2))
DESIGN_PINV = np.linalg.pinv(DESIGN)
AXES = ("x", "y")
P1_SOURCE_SHA256 = "72a300168b99126d2502b546069771866b535f6e17d20b84f9e9336fb4affd0c"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def trajectory_array(value):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 3 or out.shape[1:] != (6, 2) or not len(out) or not np.isfinite(out).all():
        raise ValueError("궤적은 비어 있지 않은 finite [N,6,2]여야 합니다")
    return out


def fit_coefficients(xy):
    """출력 [N,2,2]: 계수 축은 [a,b], 좌표 축은 [x,y]. 보고서에는 저장하지 않음."""
    return np.einsum("kt,ntd->nkd", DESIGN_PINV, trajectory_array(xy))


def official_d3(pred, gt):
    pred, gt = trajectory_array(pred), trajectory_array(gt)
    if pred.shape != gt.shape:
        raise ValueError("예측/GT shape 불일치")
    return np.linalg.norm(pred - gt, axis=-1) @ WEIGHTS


def coefficient_statistics(pred, gt):
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    if pred.ndim != 1 or pred.shape != gt.shape or not len(pred) or not np.isfinite([pred, gt]).all():
        raise ValueError("계수 비교 shape/finite 불일치")
    sp = 0. if np.ptp(pred) == 0 else float(np.std(pred, ddof=0))
    sg = 0. if np.ptp(gt) == 0 else float(np.std(gt, ddof=0))
    pearson = float(np.corrcoef(pred, gt)[0, 1]) if sp > 0 and sg > 0 else None
    return {"n": len(pred), "pred_mean": float(pred.mean()), "gt_mean": float(gt.mean()),
            "pred_std": sp, "gt_std": sg, "std_ratio_pred_gt": sp / sg if sg > 0 else None,
            "variance_ratio_pred_gt": (sp / sg) ** 2 if sg > 0 else None,
            "mae": float(np.abs(pred - gt).mean()), "pearson": pearson,
            "sign_agreement": float((np.sign(pred) == np.sign(gt)).mean())}


def fit_residual_statistics(values):
    """한 좌표의 [N,6] 집계. uncentered 에너지 분모와 물리 잔차를 함께 출력."""
    values = np.asarray(values, np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not len(values) or not np.isfinite(values).all():
        raise ValueError("잔차 입력은 finite [N,6]여야 합니다")
    total = float(np.sum(values ** 2))
    fits = {"t_only": (values @ TIMES / (TIMES @ TIMES))[:, None] * TIMES,
            "t_plus_t_squared": (values @ DESIGN_PINV.T) @ DESIGN.T}
    result = {}
    for name, fitted in fits.items():
        residual = values - fitted
        sse = float(np.sum(residual ** 2))
        fraction = sse / total if total > 0 else None
        result[name] = {"coordinate_rmse_m": float(np.sqrt(sse / values.size)),
                        "coordinate_max_abs_m": float(np.abs(residual).max()),
                        "residual_sse_m2": sse, "original_energy_m2": total,
                        "residual_uncentered_energy_fraction": fraction,
                        "explained_uncentered_energy_fraction": 1. - fraction if fraction is not None else None,
                        "rmse_denominator_coordinate_count": int(values.size)}
    return result


def analyze(document):
    if (document.get("final_val_accessed") is not False or document.get("time_input") not in ("raw", "nominal")
            or document.get("protocol", {}).get("arguments", {}).get("split") != "tune"):
        raise ValueError("기존 tune 평가와 명시적 time_input/final_val_accessed=false가 필요합니다")
    records = document.get("conditions", {}).get("normal", {}).get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("비어 있지 않은 normal records가 필요합니다")
    for item in records:
        if (not isinstance(item, dict) or type(item.get("row")) is not int
                or type(item.get("frame")) is not int or item["row"] < 0 or item["frame"] < 0
                or any(not isinstance(item.get(k), str) or not item[k] for k in ("scenario", "session"))):
            raise ValueError("record row/frame/scenario/session 규격 불일치")
    if (len({r["row"] for r in records}) != len(records)
            or len({(r["scenario"], r["frame"]) for r in records}) != len(records)):
        raise ValueError("중복 row 또는 scenario/frame")
    scene_sessions = {}
    for item in records:
        prior = scene_sessions.setdefault(item["scenario"], item["session"])
        if prior != item["session"]:
            raise ValueError("같은 시나리오의 session 불일치")
    pred = trajectory_array([r["pred_abs_xy"] for r in records])
    gt = trajectory_array([r["gt_abs_xy"] for r in records])
    scores = np.asarray([r["d3"] for r in records], np.float64)
    if scores.shape != (len(records),) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("D3 기록 finite/shape 불일치")
    delta = float(np.max(np.abs(official_d3(pred, gt) - scores)))
    if delta > 1e-5:
        raise ValueError("공식 D3 시간 가중과 저장 점수 불일치")
    sessions = np.asarray([r["session"] for r in records])
    scenes = np.asarray([r["scenario"] for r in records])
    cp, cg = fit_coefficients(pred), fit_coefficients(gt)
    total_score = float(scores.sum())

    def summarize(mask):
        n = int(mask.sum())
        if not n:
            return {"n": 0, "scene_n": 0, "session_n": 0, "official_d3": None,
                    "fraction_total_d3_sum": 0. if total_score > 0 else None, "axes": {}}
        result = {"n": n, "scene_n": len(set(scenes[mask])), "session_n": len(set(sessions[mask])),
                  "official_d3": float(scores[mask].mean()),
                  "fraction_total_d3_sum": float(scores[mask].sum() / total_score) if total_score > 0 else None,
                  "axes": {}}
        for axis, name in enumerate(AXES):
            result["axes"][name] = {
                "quadratic_coefficient_b": coefficient_statistics(cp[mask, 1, axis], cg[mask, 1, axis]),
                "pred_fit_residuals": fit_residual_statistics(pred[mask, :, axis]),
                "gt_fit_residuals": fit_residual_statistics(gt[mask, :, axis]),
                "official_weights_absolute_error_m": float((np.abs(pred[mask, :, axis] - gt[mask, :, axis]) @ WEIGHTS).mean()),
                "last_3s_absolute_error_m": float(np.abs(pred[mask, -1, axis] - gt[mask, -1, axis]).mean())}
        return result

    protocol = document["protocol"]
    partitions = {name: {label: summarize(mask) for label, mask in (
        ("negative", cg[:, 1, axis] < 0), ("exact_zero", cg[:, 1, axis] == 0), ("positive", cg[:, 1, axis] > 0))}
        for axis, name in enumerate(AXES)}
    return {"status": "complete_descriptive_temporal_shape", "schema_version": 1,
            "time_input": document["time_input"], "checkpoint_step": protocol.get("checkpoint_step"),
            "checkpoint_sha256": protocol.get("checkpoint_sha256"), "data_provenance": protocol.get("data", {}),
            "source_supervision_root": protocol["arguments"].get("supervision_root"),
            "d3_float64_reconstruction_max_abs_difference": delta,
            "definitions": {
                "times_seconds": TIMES.tolist(), "axes": "현재 ego축 x전방/y좌. 경로 접선 좌표가 아님",
                "fit": "각 표본·각 좌표의 q(t)=a*t+b*t². intercept 없음. 6개 점 비가중 OLS",
                "quadratic_coefficient_unit": "m/s²; 적합 곡선의 시간 2차 미분은 2*b. 실제 가속도나 공간 곡률의 직접 측정값이 아님",
                "std": "표본 간 population std(ddof=0). 시간 가중이나 D3 가중을 적용하지 않음",
                "energy_fraction": "sum(residual²)/sum(original_coordinate²). 좌표·집단별 비가중 합, sample별 R² 평균 아님",
                "rmse": "sqrt(sum(residual²)/(집단 표본 수*6)); 큰 위치 에너지에 의한 설명률 과장을 피하도록 maxabs도 병기",
                "zero_denominator": "GT 계수 분산/좌표 에너지가 0이면 해당 비율·Pearson은 null; NaN을 출력하지 않음",
                "gt_sign_partition": "GT b의 정확한 수치 부호. 추가 임계값 없음; near-zero 수치 오차도 포함 가능",
                "official_d3_weights": WEIGHTS.tolist(),
                "official_d3": "mean cumulative ADE@1/2/3s. 표본별 L2에 공식 시간 가중 후 집단 표본 평균",
                "d3_contribution": "집단 D3 합 / 전체 normal D3 합. 곡률의 인과 기여나 회수 가능한 개선량이 아님",
                "axis_absolute_error": "공식 시간 가중을 적용한 좌표별 MAE이며 두 축을 더해 D3로 취급하지 않음"},
            "all": summarize(np.ones(len(records), bool)), "gt_b_sign_partitions": partitions,
            "sessions": {str(s): summarize(sessions == s) for s in sorted(set(sessions))},
            "limitations": ["미래 GT를 사용하는 사후 기술통계이며 배포 보정이나 개선 상한이 아님",
                            "시간 y의 2차 성분은 공간 곡률과 다름",
                            "분산 축소만으로 구조·손실·다중모드 평균화 등 원인을 확정할 수 없음",
                            "높은 다항 적합도는 GT 합성의 증거가 아니며 큰 위치 에너지 분모의 영향을 받음",
                            "동일한 반복 tune 자료의 관찰이며 표본 간 독립성이나 새로운 heldout 검증을 주장하지 않음",
                            "P1 관찰을 geometry/time이 다른 P2 결과에 그대로 일반화하지 않음"],
            "future_gt_descriptive_only": True, "deployment_correction_allowed": False,
            "per_sample_coefficients_or_fitted_paths_exported": False,
            "raw_or_additional_val_access": False, "gpu_used": False, "tuning_or_selection_performed": False}


def analyze_file(input_path, output_path, expected_source_sha256):
    source, target = Path(input_path), Path(output_path)
    if os.path.lexists(target):
        raise FileExistsError("기존 출력 파일을 변경하지 않습니다")
    if not target.parent.is_dir():
        raise ValueError("출력 부모 디렉터리가 존재해야 합니다")
    if not isinstance(expected_source_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256) is None:
        raise ValueError("명시적 입력 SHA256이 필요합니다")
    raw = source.read_bytes()
    if sha256_bytes(raw) != expected_source_sha256:
        raise ValueError("입력 보고서 SHA256 불일치")
    result = analyze(json.loads(raw))
    result.update(source_report=str(source.resolve()), source_report_sha256=expected_source_sha256,
                  analysis_script_sha256=sha256_bytes(Path(__file__).read_bytes()))
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if sha256_bytes(source.read_bytes()) != expected_source_sha256:
        raise ValueError("분석 중 입력 보고서 변경")
    with target.open("x") as stream:
        stream.write(encoded)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    result = analyze_file(args.input, args.out, args.expected_source_sha256)
    print(json.dumps({"status": result["status"], "n": result["all"]["n"],
                      "sessions": len(result["sessions"]), "official_d3": result["all"]["official_d3"],
                      "lateral_b": result["all"]["axes"]["y"]["quadratic_coefficient_b"],
                      "report": args.out}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
