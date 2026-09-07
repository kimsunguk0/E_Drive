#!/usr/bin/env python3
"""저장된 nominal tune 평가 records의 사후 오차 구조만 분석한다.

OLS는 각 샘플의 미래 정답을 사용한 oracle 분해다. 좌표 보정, 배포 규칙,
학습 목표 또는 선택용 하이퍼파라미터를 생성하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

TIMES = np.arange(1, 7, dtype=np.float64) * .5
WEIGHTS = np.asarray([11, 11, 5, 5, 2, 2], np.float64) / 36
STOP_BUCKETS = ("continued_stop", "delayed_start", "stop_boundary", "moving", "state_unknown")
STATE_BUCKETS = ("stop", "accel", "decel", "cruise", "unknown")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stop_bucket(state, valid, gt):
    state, valid, gt = np.asarray(state), np.asarray(valid, bool), np.asarray(gt)
    if state.shape != (6,) or valid.shape != (6,) or gt.shape != (6, 2) or not np.isfinite(gt).all():
        raise ValueError("GT state/future 규격 불일치")
    if not valid[:2].all() or not np.isfinite(state[:2]).all():
        return "state_unknown"
    if np.linalg.norm(state[:2]) >= .2:
        return "moving"
    radius = float(np.linalg.norm(gt, axis=-1).max())
    if radius <= .2:
        return "continued_stop"
    if radius > .5:
        return "delayed_start"
    return "stop_boundary"


def uncentered_fraction(residual_energy, total_energy):
    return None if total_energy == 0 else float(1. - residual_energy / total_energy)


def temporal_fits(error):
    """각 sample·각 좌표를 OLS. intercept 없음, 계수/수정 궤적은 반환하지 않음."""
    error = np.asarray(error, np.float64)
    if error.ndim != 3 or error.shape[1:] != (6, 2) or not np.isfinite(error).all():
        raise ValueError("오차는 finite [N,6,2]여야 합니다")
    designs = {"t_only": TIMES[:, None], "t_squared_only": (TIMES ** 2)[:, None],
               "t_plus_t_squared": np.column_stack((TIMES, TIMES ** 2))}
    total = np.sum(error ** 2, axis=(0, 1))
    result = {}
    for name, design in designs.items():
        projection = design @ np.linalg.pinv(design)
        fitted = np.einsum("ij,njd->nid", projection, error)
        residual = error - fitted
        remaining = np.sum(residual ** 2, axis=(0, 1))
        result[name] = {"parameters_per_sample_axis": design.shape[1],
                        "longitudinal_explained_energy_fraction": uncentered_fraction(remaining[0], total[0]),
                        "lateral_explained_energy_fraction": uncentered_fraction(remaining[1], total[1]),
                        "joint_explained_energy_fraction": uncentered_fraction(remaining.sum(), total.sum()),
                        "longitudinal_residual_sse": float(remaining[0]),
                        "lateral_residual_sse": float(remaining[1])}
    return {"n": len(error), "longitudinal_total_sse": float(total[0]), "lateral_total_sse": float(total[1]),
            "oracle_only": True, "models": result}


def path_lengths(xy):
    zero = np.zeros((len(xy), 1, 2), np.float64)
    return np.cumsum(np.linalg.norm(np.diff(np.concatenate((zero, xy), axis=1), axis=1), axis=-1), axis=1)


def error_summary(pred, gt, scores, sessions, scenarios, global_n, global_score_sum):
    n = len(pred)
    if not n:
        return {"n": 0, "n_scenes": 0, "n_sessions": 0, "official_d3": None,
                "sample_fraction": 0., "fraction_of_total_d3_sum": 0., "session_mean_d3": None,
                "timewise": [], "oracle_temporal_fits": None}
    error = pred - gt
    radial = np.linalg.norm(pred, axis=-1) - np.linalg.norm(gt, axis=-1)
    traveled = path_lengths(pred) - path_lengths(gt)
    temporal = []
    for j, time in enumerate(TIMES):
        row = {"time_seconds": float(time), "n": n,
               "point_l2_mean_m": float(np.linalg.norm(error[:, j], axis=-1).mean())}
        for axis, label in enumerate(("longitudinal", "lateral")):
            e = error[:, j, axis]
            row[label] = {"signed_mean_m": float(e.mean()), "absolute_mean_m": float(np.abs(e).mean()),
                          "absolute_median_m": float(np.median(np.abs(e))),
                          "absolute_p90_m": float(np.percentile(np.abs(e), 90)),
                          "positive_fraction": float((e > 0).mean()), "negative_fraction": float((e < 0).mean()),
                          "exact_zero_fraction": float((e == 0).mean())}
        for values, name in ((radial[:, j], "radial_displacement"), (traveled[:, j], "sampled_path_length")):
            row[name] = {"signed_mean_error_m": float(values.mean()), "absolute_mean_error_m": float(np.abs(values).mean()),
                         "over_fraction": float((values > 0).mean()), "under_fraction": float((values < 0).mean()),
                         "exact_equal_fraction": float((values == 0).mean())}
        temporal.append(row)
    per_session = [float(scores[sessions == session].mean()) for session in np.unique(sessions)]
    return {"n": n, "n_scenes": len(np.unique(scenarios)), "n_sessions": len(np.unique(sessions)),
            "official_d3": float(scores.mean()), "sample_fraction": float(n / global_n),
            "fraction_of_total_d3_sum": float(scores.sum() / global_score_sum) if global_score_sum else None,
            "session_mean_d3": float(np.mean(per_session)),
            "weighted_abs_longitudinal_m": float((np.abs(error[..., 0]) @ WEIGHTS).mean()),
            "weighted_abs_lateral_m": float((np.abs(error[..., 1]) @ WEIGHTS).mean()),
            "weighted_signed_longitudinal_m": float((error[..., 0] @ WEIGHTS).mean()),
            "weighted_signed_lateral_m": float((error[..., 1] @ WEIGHTS).mean()),
            "timewise": temporal, "oracle_temporal_fits": temporal_fits(error)}


def analyze(document):
    protocol = document["protocol"]
    args = protocol["arguments"]
    if (document.get("time_input") != "nominal" or args["split"] != "tune"
            or protocol["checkpoint_step"] != 6000 or args["frame_stride"] != 5
            or document.get("final_val_accessed") is not False):
        raise ValueError("요청한 LAST6000/nominal/tune/stride5 보고서가 아닙니다")
    normal = document["conditions"]["normal"]
    records = normal["records"]
    if len(records) != 1998:
        raise ValueError("사전 지정 tune1998이 아닙니다")
    pred = np.asarray([r["pred_abs_xy"] for r in records], np.float64)
    gt = np.asarray([r["gt_abs_xy"] for r in records], np.float64)
    scores = np.asarray([r["d3"] for r in records], np.float64)
    sessions = np.asarray([r["session"] for r in records])
    scenes = np.asarray([r["scenario"] for r in records])
    if pred.shape != (1998, 6, 2) or gt.shape != pred.shape or not np.isfinite([pred, gt]).all() or not np.isfinite(scores).all():
        raise ValueError("궤적/점수 규격 불일치")
    if len(set(r["row"] for r in records)) != 1998 or len(np.unique(sessions)) != 11 or len(np.unique(scenes)) != 37:
        raise ValueError("고정 row/scene/session 수가 다릅니다")
    reconstructed = np.linalg.norm(pred - gt, axis=-1) @ WEIGHTS
    score_delta = float(np.max(np.abs(reconstructed - scores)))
    # 입력 좌표로 계산한 float64 분석치와 저장 float32 공식 점수 차이만 검사한다.
    if score_delta > 1e-5:
        raise ValueError("좌표 재계산과 공식 D3 기록이 맞지 않습니다")
    stop_groups = np.asarray([stop_bucket(r["gt_state"], r["gt_state_valid"], r["gt_abs_xy"]) for r in records])
    state_groups = np.asarray([r["bucket"] for r in records])
    if not set(state_groups) <= set(STATE_BUCKETS):
        raise ValueError("원본 보고서에 미정의 state bucket이 있습니다")
    def summarize(mask):
        return error_summary(pred[mask], gt[mask], scores[mask], sessions[mask], scenes[mask], len(records), scores.sum())
    all_rows = np.ones(len(records), bool)
    stop_table = {bucket: summarize(stop_groups == bucket) for bucket in STOP_BUCKETS}
    state_table = {bucket: summarize(state_groups == bucket) for bucket in STATE_BUCKETS}
    by_session = {}
    for session in sorted(np.unique(sessions)):
        mask = sessions == session
        by_session[session] = {"all": summarize(mask),
                               "stop_partition": {bucket: summarize(mask & (stop_groups == bucket)) for bucket in STOP_BUCKETS},
                               "original_state_buckets": {bucket: summarize(mask & (state_groups == bucket)) for bucket in STATE_BUCKETS}}
    return {"status": "complete_descriptive_analysis", "source_scope": "동일한 반복 사용 tune1998 records만 읽음",
            "checkpoint_step": 6000, "checkpoint_sha256": protocol["checkpoint_sha256"],
            "split_sha256": protocol["data"]["split_sha256"],
            "receiver_rows_sha256": protocol["data"]["receiver_rows_sha256"],
            "source_supervision_root": args["supervision_root"], "time_input": "nominal",
            "source_geometry_is_old_rawtime_edition": args["supervision_root"].endswith("train_tune_rawtime"),
            "d3_float64_reconstruction_max_abs_difference": score_delta,
            "definitions": {
                "axes": "현재 ego축 x전방/y좌; 오차=prediction-GT. 경로 접선 기준 종횡축이 아님",
                "current_stop": "valid GT vx/vy, hypot(vx,vy)<0.2m/s",
                "continued_stop": "현재stop이며 max_t hypot(GTxy)<=0.2m",
                "delayed_start": "현재stop이며 max_t hypot(GTxy)>0.5m",
                "stop_boundary": "현재stop이며 0.2<max_t hypot(GTxy)<=0.5m",
                "moving": "valid GT vx/vy, hypot(vx,vy)>=0.2m/s",
                "state_unknown": "GT vx/vy invalid/nonfinite",
                "original_state_buckets": protocol.get("bucket_definition"),
                "radial_displacement": "현재 원점으로부터 norm(pred)-norm(GT); 경로 이동거리와 다름",
                "sampled_path_length": "0원점과6waypoint를 연결한 누적 선분 길이의 pred-GT 차이",
                "over_under": "양수=과대, 음수=과소, 정확0=동률; 추가 epsilon/threshold 튜닝 없음",
                "oracle_OLS": "샘플별 좌표별 e(t)에 intercept 없이 t, t², [t,t²] OLS 투영; 미래GT를 쓴 사후분해",
                "explained_energy": "uncentered 1-SSE_after/SSE_before; sample 평균R²가 아닌 합산제곱오차 비율",
                "aggregation": "주지표 frame평균; 보조 session평균. bucket sample비율/D3기여분모는 전체1998"},
            "all": summarize(all_rows), "stop_partition": stop_table, "original_state_buckets": state_table,
            "sessions": by_session,
            "limitations": ["반복 사용한 tune37scene/11session의 관찰이며 새로운 heldout 검증이나 인과 식별이 아님",
                            "current GT state는 영상에서 추정한 모델상태가 아니라 감독용 pose 파생값임",
                            "OLS는 미래 정답으로 sample별 fitting하므로 배포 가능 성능이나 회수 가능한 개선폭을 증명하지 않음",
                            "[t,t²]는 t-only보다 자유도가 커 residual이 줄어드는 것이 자연스러우며 추가 차수를 튜닝하지 않음",
                            "낮은 차수의 오차는 매끄러운 출력/좌표기하/행동 모호성 등에서도 나타나므로 속도·가속도 원인으로 단정 불가",
                            "좌표 보정식·수정 trajectory·sample별 OLS 계수는 출력하거나 배포하지 않음",
                            "원본은 rear geometry 수정 이전 P1 평가이므로 P2 이후에도 같은 병목이라고 단정하지 않음"],
            "raw_or_additional_val_access": False, "gpu_used": False, "tuning_or_selection_performed": False}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if os.path.lexists(args.out):
        raise FileExistsError("기존 보고서를 덮어쓰지 않습니다")
    before = sha256(args.input)
    with open(args.input) as stream:
        document = json.load(stream)
    result = analyze(document)
    if sha256(args.input) != before:
        raise ValueError("분석 중 원본 보고서 변경")
    result.update({"source_report": str(Path(args.input).resolve()), "source_report_sha256": before,
                   "analysis_script_sha256": sha256(__file__)})
    with open(args.out, "x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "n": result["all"]["n"],
                      "d3": result["all"]["official_d3"],
                      "stop_partition": {k: {field: v.get(field) for field in ("n", "official_d3", "fraction_of_total_d3_sum")}
                                         for k, v in result["stop_partition"].items()},
                      "oracle_OLS": result["all"]["oracle_temporal_fits"], "report": args.out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
