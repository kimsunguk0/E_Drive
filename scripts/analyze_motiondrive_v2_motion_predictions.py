#!/usr/bin/env python3
"""저장된 동일 full-forward 신경 상태와 미래 궤적의 집계 전용 CPU 진단.

추가 forward, 원자료 조회, 학습, 임계값 선택 및 배포 경로 보정을 하지 않는다.
GT 사이의 상관은 영상 상태 추정 성능과 별도로 표시한다.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_motiondrive_v2_planning import (
    BUCKET_DEFINITION, BUCKET_NAMES, CONDITIONS, MODEL_INPUTS,
    motion_prediction_contract, state_bucket,
)
from scripts.analyze_motiondrive_v2_temporal_shape import (
    TIMES, WEIGHTS, coefficient_statistics, fit_coefficients, official_d3,
    trajectory_array,
)

STATE_NAMES = ("vx", "vy", "ax", "ay", "yaw_rate")
STATE_UNITS = ("m/s", "m/s", "m/s^2", "m/s^2", "rad/s")
HISTORY_COMPONENTS = ("dx", "dy", "sin_delta_yaw", "cos_delta_yaw")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require_sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label}: SHA256 형식 불일치")
    return value


def exact_json(a, b):
    # Python의 True == 1과 달리, 계약의 JSON 타입까지 구별한다.
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def numeric_array(value, shape, label):
    obj = np.asarray(value, dtype=object)
    if obj.shape != shape or any(type(v) not in (int, float) for v in obj.flat):
        raise ValueError(f"{label}: 숫자 shape {shape}가 필요합니다")
    out = np.asarray(value, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError(f"{label}: finite 값이 필요합니다")
    return out


def pearson(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Pearson 입력 shape/finite 불일치")
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.clip(np.corrcoef(x, y)[0, 1], -1., 1.))


def regression_metrics(pred, gt):
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    if pred.shape != gt.shape or pred.ndim != 1 or not np.isfinite([pred, gt]).all():
        raise ValueError("회귀 비교 shape/finite 불일치")
    n = len(pred)
    error = pred - gt
    return {"n_valid": n, "mae": float(np.abs(error).mean()) if n else None,
            "rmse": float(np.sqrt(np.mean(error ** 2))) if n else None,
            "bias_pred_minus_gt": float(error.mean()) if n else None,
            "pearson": pearson(pred, gt)}


def stop_metrics(logits, labels):
    logits, labels = np.asarray(logits, np.float64), np.asarray(labels, np.float64)
    if (logits.shape != labels.shape or logits.ndim != 1 or not np.isfinite([logits, labels]).all()
            or not np.isin(labels, (0., 1.)).all()):
        raise ValueError("정지 비교는 finite raw logit과 binary GT가 필요합니다")
    n = len(logits)
    positive = int(labels.sum())
    negative = n - positive
    # 극단 logit에서도 overflow 없이 sigmoid를 계산한다. 확률을 미리 받지 않는다.
    probability = np.exp(-np.logaddexp(0., -logits))
    auc = None
    if positive and negative:
        # 원 logit 순위로 tie 0.5 처리. sigmoid의 수치 포화로 인한 가짜 동점 방지.
        order = np.argsort(logits, kind="stable")
        sorted_logits = logits[order]
        rank = np.empty(n, np.float64)
        starts = np.flatnonzero(np.r_[True, sorted_logits[1:] != sorted_logits[:-1]])
        ends = np.r_[starts[1:], n]
        for start, end in zip(starts, ends):
            rank[order[start:end]] = (start + end + 1) / 2
        auc = float((rank[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))
    return {"n_valid": n, "n_positive": positive, "n_negative": negative,
            "brier": float(np.mean((probability - labels) ** 2)) if n else None,
            "auroc": auc, "mean_sigmoid_probability": float(probability.mean()) if n else None,
            "raw_logit_mean": float(logits.mean()) if n else None,
            "raw_logit_min": float(logits.min()) if n else None,
            "raw_logit_max": float(logits.max()) if n else None}


def validate_protocol(document, selection):
    if selection not in ("normal", "all"):
        raise ValueError("분석 조건은 normal 또는 all이어야 합니다")
    if document.get("status") != "completed":
        raise ValueError("완료된 평가 보고서가 필요합니다")
    protocol = document.get("protocol", {})
    arguments = protocol.get("arguments", {})
    if (arguments.get("include_motion_predictions") is not True
            or not exact_json(protocol.get("motion_prediction_contract"), motion_prediction_contract())):
        raise ValueError("신경망 예측이 없습니다. 동일 full-forward --include-motion-predictions 계약이 필요합니다")
    if (protocol.get("status") != "preregistered_before_any_forward" or arguments.get("split") != "tune"
            or any(d.get(k) is not False for d in (document, protocol)
                   for k in ("selection_performed", "final_val_accessed"))):
        raise ValueError("사전 고정 tune 평가와 selection/final_val=false가 필요합니다")
    time_input = document.get("time_input")
    if time_input not in ("raw", "nominal") or any(v != time_input for v in
            (protocol.get("time_input"), arguments.get("time_input"))):
        raise ValueError("시간 입력 정책 선언 불일치")
    if (not exact_json(protocol.get("bucket_definition"), BUCKET_DEFINITION)
            or not exact_json(protocol.get("model_input_whitelist"), list(MODEL_INPUTS))
            or protocol.get("nominal_waypoint_seconds") != TIMES.tolist()
            or protocol.get("coordinate_format") != "current ego-frame cumulative XY positions, metres; no cumsum or postprocessing"):
        raise ValueError("GT bucket/입력/시간/누적 좌표 계약 불일치")
    require_sha(protocol.get("checkpoint_sha256"), "checkpoint")
    if type(protocol.get("checkpoint_step")) is not int or protocol["checkpoint_step"] < 0:
        raise ValueError("체크포인트 step이 필요합니다")
    data = protocol.get("data", {})
    for key in ("split_sha256", "ego_cache_sha256", "receiver_rows_sha256"):
        require_sha(data.get(key), key)
    if type(data.get("receiver_count")) is not int or data["receiver_count"] <= 0:
        raise ValueError("양수 receiver_count가 필요합니다")
    sup = data.get("supervision_sha256", {})
    for key in ("supervision_manifest.json", "calibration.npz"):
        require_sha(sup.get(key), key)
    for value in sup.values():
        require_sha(value, "supervision source")
    source_hashes = protocol.get("source", {}).get("file_sha256", {})
    if "scripts/evaluate_motiondrive_v2_planning.py" not in source_hashes:
        raise ValueError("평가 소스 SHA 계보가 필요합니다")
    for value in source_hashes.values():
        require_sha(value, "evaluation source")
    declared = arguments.get("conditions")
    if (not isinstance(declared, list) or not declared or len(set(declared)) != len(declared)
            or any(c not in CONDITIONS for c in declared)
            or set(document.get("conditions", {})) != set(declared)):
        raise ValueError("선언된 조건과 평가 조건 불일치")
    if "normal" not in declared:
        raise ValueError("동일 표본 대조를 위한 normal 조건이 필요합니다")
    return ["normal"] if selection == "normal" else [c for c in CONDITIONS if c in declared]


def validate_records(condition, protocol):
    if condition.get("time_input") != protocol["time_input"]:
        raise ValueError("조건별 시간 입력 정책 불일치")
    records = condition.get("records")
    if not isinstance(records, list) or not records or len(records) != protocol["data"]["receiver_count"]:
        raise ValueError("records/receiver_count 불일치")
    ids, states, masks, pred_states, pred_history, scene_sessions = [], [], [], [], [], {}
    for r in records:
        if (not isinstance(r, dict) or any(type(r.get(k)) is not int or r[k] < 0 for k in ("row", "frame"))
                or any(not isinstance(r.get(k), str) or not r[k] for k in ("scenario", "session"))):
            raise ValueError("record row/frame/scenario/session 규격 불일치")
        identity = (r["row"], r["scenario"], r["frame"], r["session"])
        ids.append(identity)
        if scene_sessions.setdefault(r["scenario"], r["session"]) != r["session"]:
            raise ValueError("같은 시나리오의 session 불일치")
        mask = r.get("gt_state_valid")
        values = r.get("gt_state")
        if (not isinstance(mask, list) or len(mask) != 6 or any(type(v) is not bool for v in mask)
                or not isinstance(values, list) or len(values) != 6):
            raise ValueError("GT state [6]/엄격 boolean mask [6] 불일치")
        state = np.full(6, np.nan)
        for j, (value, valid) in enumerate(zip(values, mask)):
            if valid:
                state[j] = numeric_array([value], (1,), "valid GT state")[0]
            elif value is not None:
                raise ValueError("평가 계약상 invalid GT state는 null이어야 합니다")
        if mask[5] and state[5] not in (0., 1.):
            raise ValueError("GT stop은 binary target이어야 합니다")
        if r.get("bucket") != state_bucket(state, mask):
            raise ValueError("저장 GT bucket과 고정 bucket 규칙 불일치")
        if "pred_state" not in r or "pred_history" not in r:
            raise ValueError("record에 신경망 pred_state/pred_history가 없습니다")
        pred_states.append(numeric_array(r["pred_state"], (6,), "pred_state raw neural"))
        pred_history.append(numeric_array(r["pred_history"], (4, 4), "pred_history raw neural"))
        if "gt_history" in r or "gt_history_valid" in r:
            raise ValueError("현재 harness 계약에는 GT history 기록이 없습니다. 별도 계약 검토가 필요합니다")
        states.append(state)
        masks.append(mask)
    if len({i[0] for i in ids}) != len(ids) or len({i[1:3] for i in ids}) != len(ids):
        raise ValueError("중복 row 또는 scenario/frame")
    rows_hash = digest(np.asarray([i[0] for i in ids], dtype="<i8").tobytes())
    if rows_hash != protocol["data"]["receiver_rows_sha256"]:
        raise ValueError("receiver row 순서 SHA 불일치")
    pred = trajectory_array([r["pred_abs_xy"] for r in records])
    gt = trajectory_array([r["gt_abs_xy"] for r in records])
    d3 = numeric_array([r["d3"] for r in records], (len(records),), "D3")
    delta = float(np.max(np.abs(official_d3(pred, gt) - d3)))
    if (d3 < 0).any() or delta > 1e-5:
        raise ValueError("공식 D3 재계산과 저장 점수 불일치")
    return {"ids": ids, "state": np.asarray(states), "valid": np.asarray(masks, bool),
            "pred_state": np.asarray(pred_states), "pred_history": np.asarray(pred_history),
            "pred_plan": pred, "gt_plan": gt, "d3": d3, "d3_reconstruction_delta": delta,
            "buckets": np.asarray([r["bucket"] for r in records])}


def summarize_condition(data):
    ids, valid = data["ids"], data["valid"]
    sessions = np.asarray([r[3] for r in ids])
    scenes = np.asarray([r[1] for r in ids])
    pred_state, gt_state = data["pred_state"], data["state"]
    cp, cg = fit_coefficients(data["pred_plan"]), fit_coefficients(data["gt_plan"])

    def summarize(mask):
        n = int(mask.sum())
        state_result = {}
        for j, (name, unit) in enumerate(zip(STATE_NAMES, STATE_UNITS)):
            use = mask & valid[:, j]
            state_result[name] = {**regression_metrics(pred_state[use, j], gt_state[use, j]),
                                  "n_invalid": n - int(use.sum()), "unit": unit}
        stop_use = mask & valid[:, 5]
        history = data["pred_history"][mask]
        history_result = {"gt_available": False, "accuracy": None,
                          "reason": "현재 평가 records에 GT history와 mask가 제공되지 않음. 외부 GT를 조회하거나 미래 궤적으로 대체하지 않음",
                          "raw_pred_component_mean": history.mean(0).tolist() if n else None,
                          "raw_pred_component_std_ddof0": history.std(0).tolist() if n else None,
                          "n": n}
        yaw_valid = mask & valid[:, 4]
        correlations = {}
        pairs = (
            ("gt_yaw_vs_estimated_yaw", gt_state[:, 4], pred_state[:, 4], yaw_valid, "영상 신경 상태 추정 성능"),
            ("gt_yaw_vs_gt_future_y_b", gt_state[:, 4], cg[:, 1, 1], yaw_valid, "GT와 GT의 사후 관계; 영상 추정 성능 아님"),
            ("estimated_yaw_vs_gt_future_y_b", pred_state[:, 4], cg[:, 1, 1], mask, "신경 상태와 미래 GT의 사후 관계"),
            ("estimated_yaw_vs_predicted_future_y_b", pred_state[:, 4], cp[:, 1, 1], mask, "같은 forward의 두 예측 간 관계; 정확도 아님"),
            ("gt_yaw_vs_predicted_future_y_b", gt_state[:, 4], cp[:, 1, 1], yaw_valid, "GT 현재 상태와 예측 미래 궤적의 관계"),
            ("predicted_future_y_b_vs_gt_future_y_b", cp[:, 1, 1], cg[:, 1, 1], mask, "예측과 GT의 시간 2차 계수 일치도"),
        )
        for name, x, y, use, meaning in pairs:
            correlations[name] = {"n": int(use.sum()), "pearson": pearson(x[use], y[use]), "meaning": meaning}
        coefficient_result = {axis: {
            term: coefficient_statistics(cp[mask, k, j], cg[mask, k, j]) if n else None
            for k, term in enumerate(("a", "b"))} for j, axis in enumerate(("x", "y"))}
        return {"n": n, "scene_n": len(set(scenes[mask])), "session_n": len(set(sessions[mask])),
                "official_d3": float(data["d3"][mask].mean()) if n else None,
                "state": state_result,
                "stop": {**stop_metrics(pred_state[stop_use, 5], gt_state[stop_use, 5]),
                         "n_invalid": n - int(stop_use.sum())},
                "history": history_result, "temporal_coefficients": coefficient_result,
                "yaw_temporal_correlations": correlations}

    return {"all": summarize(np.ones(len(ids), bool)),
            "sessions": {str(s): summarize(sessions == s) for s in sorted(set(sessions))},
            "gt_buckets": {b: summarize(data["buckets"] == b) for b in BUCKET_NAMES},
            "d3_float64_reconstruction_max_abs_difference": data["d3_reconstruction_delta"]}


def analyze(document, conditions="normal"):
    selected = validate_protocol(document, conditions)
    protocol = document["protocol"]
    parsed = {c: validate_records(document["conditions"][c], protocol) for c in selected}
    reference = parsed["normal"]
    for name, data in parsed.items():
        if (data["ids"] != reference["ids"]
                or any(not np.array_equal(data[k], reference[k], equal_nan=True) for k in ("state", "valid", "gt_plan"))
                or not np.array_equal(data["buckets"], reference["buckets"])):
            raise ValueError(f"{name}: 조건 간 receiver ID/순서/GT/mask/bucket 불변성 위반")
    return {"schema_version": 1, "status": "completed_aggregate_motion_diagnosis",
            "checkpoint_sha256": protocol["checkpoint_sha256"], "checkpoint_step": protocol["checkpoint_step"],
            "time_input": document["time_input"], "data_provenance": protocol["data"],
            "evaluation_source_provenance": protocol["source"],
            "motion_prediction_contract": motion_prediction_contract(),
            "conditions_requested": conditions, "conditions_analyzed": selected,
            "definitions": {
                "state_order": [*STATE_NAMES, "stop_logit"], "state_units": [*STATE_UNITS, "logit"],
                "mask": "각 GT 성분의 boolean valid mask를 사용. GT yaw를 포함하는 상관만 해당 yaw mask를 적용",
                "state_errors": "원 neural 예측 - GT. 각 성분의 유효 프레임을 동일 가중; 단위별 독립 MAE/RMSE/bias/Pearson",
                "stop": "state[5]는 GT binary target, pred_state[5]는 raw logit. sigmoid 후 Brier; 원 logit 순위로 AUROC tie=0.5. 단일 클래스/빈 집단 AUROC=null. 분류 임계값 미사용",
                "history_frame_offsets": [-1, -2, -5, -10], "history_components": list(HISTORY_COMPONENTS),
                "history": "원 sin/cos 성분을 정규화하거나 각도로 바꾸지 않음. GT 미제공이므로 정확도 산출 불가",
                "gt_buckets": BUCKET_DEFINITION,
                "temporal_fit": "미래 6점의 좌표별 q(t)=a*t+b*t², intercept 없는 비가중 OLS; 표본별 적합 결과는 저장하지 않음",
                "nominal_waypoint_seconds": TIMES.tolist(),
                "temporal_coefficient_units": {"a": "m/s", "b": "m/s^2"},
                "temporal_b": "적합 곡선의 시간 2차 미분은 2*b. 현재 가속도 또는 공간 곡률과 같다고 가정하지 않음",
                "official_d3_weights": WEIGHTS.tolist(),
                "denominators": "D3만 공식 시간 가중. OLS/계수 std(ddof=0)/상관/상태 오차는 비가중 프레임 집계. 세션 집계는 독립 검증이나 신뢰구간이 아님",
                "null": "빈 집단 또는 0 분산 Pearson, 0 GT 분산 비율, 단일 클래스 AUROC는 null"},
            "conditions": {c: summarize_condition(parsed[c]) for c in selected},
            "limitations": [
                "GT↔GT 상관은 영상 기반 상태 추정 성능의 증거가 아님",
                "정확한 현재 상태 추정 또는 상태·미래 상관이 planner 개선이나 회수 가능한 D3 이득을 증명하지 않음",
                "시간 y의 2차 성분은 공간 곡률이 아님. 분산 축소의 구조·손실·다중모드 원인을 확정하지 않음",
                "영상 shuffle/repeat/reverse는 입력 분포와 영상·기하 정합을 깨뜨리는 진단이며 정보량 상한 또는 인과 증명이 아님",
                "동일 tune의 반복 사후 진단이며 untouched heldout 또는 프레임 독립성을 주장하지 않음",
                "JSON에 선언된 동일 forward·FP32 계약을 검증하며 원 실행을 재현하거나 외부 파일 진위를 재인증하지 않음"],
            "aggregate_only": True, "predicted_paths_or_coefficients_exported": False,
            "future_gt_descriptive_only": True, "gt_used_as_planner_input": False,
            "deployment_correction_allowed": False, "additional_forward_calls": 0,
            "tuning_or_selection_performed": False, "raw_or_additional_val_access": False, "gpu_used": False}


def analyze_file(input_path, output_path, expected_source_sha256, conditions="normal"):
    source, target = Path(input_path), Path(output_path)
    if os.path.lexists(target):
        raise FileExistsError("기존 출력 파일을 변경하지 않습니다")
    if not target.parent.is_dir():
        raise ValueError("출력 부모 디렉터리가 존재해야 합니다")
    require_sha(expected_source_sha256, "입력 보고서")
    raw = source.read_bytes()
    if digest(raw) != expected_source_sha256:
        raise ValueError("입력 보고서 SHA256 불일치")
    result = analyze(json.loads(raw), conditions=conditions)
    result.update(source_report=str(source.resolve()), source_report_sha256=expected_source_sha256,
                  analysis_script_sha256=digest(Path(__file__).read_bytes()))
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if digest(source.read_bytes()) != expected_source_sha256:
        raise ValueError("분석 중 입력 보고서 변경")
    with target.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--conditions", choices=("normal", "all"), default="normal")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = analyze_file(args.input, args.out, args.expected_source_sha256, args.conditions)
    print(json.dumps({"status": result["status"], "conditions": result["conditions_analyzed"],
                      "n": result["conditions"]["normal"]["all"]["n"], "report": args.out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
