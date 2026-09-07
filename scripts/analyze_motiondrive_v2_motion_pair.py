#!/usr/bin/env python3
"""LAST1000 고/저해상도 motion 대조의 CPU paired bootstrap 분석.

정상 정확도 개선과 시간 교란 민감도의 변화는 별개 항목으로 보고한다.
주 신뢰구간은 실제 주행 세션 단위이며 scene 단위 구간은 보조 지표다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_grouped_split_v2 import sha256, validate_manifest

EXPECTED_FRAMES = [30, 55, 85, 115, 145, 175, 205, 235, 265, 295]
METRICS = ("vx_mae_m_s", "history_position_mae_mean4_m")
CONDITIONS = ("normal", "repeat_current", "reverse_history", "cross_scene_history")


def paired_cluster_bootstrap(differences, counts, cluster_ids, *, repeats=10000, seed=20260907):
    """같은 cluster의 paired 차이와 표본 수를 함께 재표집한다.

    원래 추정량은 프레임 가중 평균이다. 크기가 다른 세션을 재표집할 때
    세션 내 모든 고정 표본을 함께 포함한 비율 추정량을 사용한다.
    """
    delta = np.asarray(differences, np.float64)
    counts = np.asarray(counts, np.float64)
    ids = np.asarray(cluster_ids, dtype=str)
    if delta.ndim != 1 or delta.shape != counts.shape or delta.shape != ids.shape:
        raise ValueError("차이·표본 수·cluster 배열 모양 불일치")
    if not len(delta) or not np.isfinite(delta).all() or not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("유효한 paired 표본이 필요합니다")
    if repeats < 100:
        raise ValueError("bootstrap 반복은100회 이상이어야 합니다")
    unique = sorted(set(ids))
    if len(unique) < 2:
        raise ValueError("둘 이상의 독립 cluster가 필요합니다")
    totals = np.asarray([(delta[ids == key] * counts[ids == key]).sum() for key in unique])
    weights = np.asarray([counts[ids == key].sum() for key in unique])
    sample = np.random.default_rng(seed).integers(0, len(unique), size=(repeats, len(unique)))
    estimates = totals[sample].sum(1) / weights[sample].sum(1)
    point = float(np.average(delta, weights=counts))
    interval = np.quantile(estimates, [.025, .975]).tolist()
    return {"mean_difference": point, "ci95_percentile": interval,
            "n_clusters": len(unique), "n_scenes": len(delta), "n_frames": int(counts.sum()),
            "bootstrap_repeats": repeats, "seed": seed,
            "ci_excludes_zero": bool(interval[1] < 0 or interval[0] > 0),
            "bootstrap_fraction_above_zero": float(np.mean(estimates > 0)),
            "fraction_note": "재표집 분포의 부호 비율이며 빈도주의 p값 또는 베이지안 사후확률이 아님"}


def validate_pair(high, low, manifest, split_sha, expected_step=1000):
    validate_manifest(manifest)
    expected_scenes = sorted(manifest["splits"]["tune"])
    if len(expected_scenes) != 37 or len({manifest["scene_to_session"][s] for s in expected_scenes}) != 11:
        raise ValueError("사전 고정 tune37/실제11세션 protocol과 다릅니다")
    extracted = []
    for label, report in (("high_feature", high), ("low_feature", low)):
        if report["split_manifest_sha256"] != split_sha:
            raise ValueError("감사 결과와 rawtime split SHA 불일치")
        protocol = report["protocol"]
        if protocol["split"] != "tune" or protocol["n"] != 370 or protocol["frames_per_scene"] != EXPECTED_FRAMES:
            raise ValueError("고정370 표본 또는 프레임 protocol 불일치")
        if protocol.get("expected_step") != expected_step:
            raise ValueError("감사 실행의 고정 expected-step이 없습니다")
        if "last" not in report["checkpoint_results"]:
            raise ValueError("LAST 평가가 없습니다; best와 혼합하지 않습니다")
        checkpoint = report["checkpoint_results"]["last"]
        if checkpoint["step"] != expected_step or Path(checkpoint["path"]).name != "last.pth":
            raise ValueError("LAST step1000끼리만 비교해야 합니다")
        if checkpoint["model_config"]["motion_input_mode"] != label:
            raise ValueError(f"비교 팔 mode 불일치: {label}")
        for condition in CONDITIONS:
            rows = checkpoint["conditions"][condition]["per_scene"]
            if sorted(rows) != expected_scenes:
                raise ValueError("팔/교란별 scene 집합이 다릅니다")
            for scene in expected_scenes:
                row = rows[scene]
                if row["n"] != 10 or row.get("frames") != EXPECTED_FRAMES:
                    raise ValueError("팔/교란별 frame 표본이 다릅니다")
                if row.get("session_id") != manifest["scene_to_session"][scene]:
                    raise ValueError("rawtime 세션 식별자 불일치")
        extracted.append(checkpoint)
    if high["supervision_manifest_sha256"] != low["supervision_manifest_sha256"]:
        raise ValueError("두 팔의 supervision 계보가 다릅니다")
    if high["protocol"]["donor_mapping"] != low["protocol"]["donor_mapping"]:
        raise ValueError("두 팔의 다른-scene 교란 순열이 다릅니다")
    hp, lp = [c["run_provenance"] for c in extracted]
    for key in ("initial_model_state_sha256", "initial_parameter_count", "train_rows_sha256",
                "eval_rows_sha256", "sample_order_sha256_at_checkpoint", "loss_weights"):
        if hp.get(key) is None or hp.get(key) != lp.get(key):
            raise ValueError(f"공정성 사전조건 불일치/누락: {key}")
    for key in ("seed", "phase", "batch", "steps", "warmup", "lr", "backbone_lr", "weight_decay",
                "precision", "uncertainty", "alpha_occ", "alpha_lane", "alpha_motion", "grad_clip"):
        if hp["arguments"].get(key) is None or hp["arguments"].get(key) != lp["arguments"].get(key):
            raise ValueError(f"공정성 학습 설정 불일치/누락: {key}")
    if hp["arguments"]["phase"] != "pretrain" or hp["arguments"]["steps"] != expected_step:
        raise ValueError("같은 pretrain1000 일정이 아닙니다")
    if hp["arguments"].get("resume") is not None or lp["arguments"].get("resume") is not None:
        raise ValueError("중간 resume 팔은 새 AdamW/표본 순서 사전조건을 별도 검토해야 합니다")
    hc, lc = dict(extracted[0]["model_config"]), dict(extracted[1]["model_config"])
    hc.pop("motion_input_mode");lc.pop("motion_input_mode")
    if hc != lc:
        raise ValueError("motion mode 외 모델 구조도 다릅니다")
    return expected_scenes, extracted


def analyze_pair(high, low, manifest, split_sha, *, repeats=10000, seed=20260907, expected_step=1000):
    scenes, (h, l) = validate_pair(high, low, manifest, split_sha, expected_step)
    sessions = [manifest["scene_to_session"][s] for s in scenes]
    counts = [10] * len(scenes)
    def values(checkpoint, condition, metric):
        return np.asarray([checkpoint["conditions"][condition]["per_scene"][s][metric] for s in scenes], np.float64)
    def interval(delta):
        return {"primary_rawtime_session": paired_cluster_bootstrap(delta, counts, sessions, repeats=repeats, seed=seed),
                "secondary_scene": paired_cluster_bootstrap(delta, counts, scenes, repeats=repeats, seed=seed)}
    result = {"상태": "LAST1000 paired 분석 완료", "expected_step": expected_step,
              "protocol": {"n_scenes": 37, "n_rawtime_sessions": 11, "n_frames": 370,
                           "frames_per_scene": EXPECTED_FRAMES, "bootstrap_repeats": repeats, "seed": seed,
                           "primary_ci": "실제 rawtime 세션11개를 복원추출하는 paired percentile95%구간",
                           "secondary_ci": "scene37개 기준 보조 구간; 시간 인접성 때문에 주 구간으로 대체하지 않음",
                           "sampling": "각 세션 안 고정 표본 전체와 paired 차이를 함께 포함; 프레임 독립 bootstrap 금지",
                           "checkpoint": "각 팔 LAST1000만 사용; best 평가를 혼합하거나 이 분석으로 재선택하지 않음"},
              "fairness_checks": "초기 가중치/파라미터 수/행 목록/누적 표본 순서/일정/손실 및 mode 외 구조 일치",
              "split_manifest_sha256": split_sha,
              "checkpoint_sha256": {"high_feature_last": h["sha256"], "low_feature_last": l["sha256"]},
              "normal_accuracy": {}, "temporal_controls": {}, "per_scene": {},
              "주의": ["11세션의 작은 개발 표본이며 단일 seed 결과임",
                       "교란은 분포 밖 진단이므로 민감도 증가를 올바른 물리 운동 관측의 완전한 증명으로 해석하지 않음",
                       "정상 오차 감소와 시간 교란 효과를 분리하며, 다른scene 교란 악화만으로 시간 사용을 확정하지 않음",
                       "두 지표·여러 교란의 구간은 탐색적 진단이며 다중비교 보정을 거친 유의성 선언이 아님"]}
    for metric in METRICS:
        hn, ln = values(h, "normal", metric), values(l, "normal", metric)
        delta = ln - hn
        result["normal_accuracy"][metric] = {"high_feature": float(np.mean(hn)), "low_feature": float(np.mean(ln)),
                                             "sign": "low-high; 음수가 저해상도 경로 개선", **interval(delta)}
        result["per_scene"][metric] = {s: {"session": sessions[i], "high_normal": float(hn[i]),
                                           "low_normal": float(ln[i]), "normal_low_minus_high": float(delta[i])} for i, s in enumerate(scenes)}
        result["temporal_controls"][metric] = {}
        for condition in CONDITIONS[1:]:
            hd, ld = values(h, condition, metric) - hn, values(l, condition, metric) - ln
            result["temporal_controls"][metric][condition] = {
                "within_arm_sign": "교란-normal; 양수가 정상 과거 영상 사용의 관측상 이점",
                "high_feature_advantage": interval(hd), "low_feature_advantage": interval(ld),
                "difference_in_advantages_low_minus_high": interval(ld - hd)}
    vx = result["normal_accuracy"]["vx_mae_m_s"]["primary_rawtime_session"]
    result["screening"] = {"mean_vx_reduction_at_least_0p1_m_s": bool(vx["mean_difference"] <= -.1),
                           "normal_vx_session_ci_entirely_below_zero": bool(vx["ci95_percentile"][1] < 0),
                           "설명": "사전개발 screening 조건 일부만 기계적으로 표시; history 이점과 교란 결과 검토 없이 자동 채택하지 않음"}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--high-audit", required=True)
    p.add_argument("--low-audit", required=True)
    p.add_argument("--split-manifest", default="data/etri/motiondrive_v2/grouped_split_rawtime.json")
    p.add_argument("--expected-step", type=int, default=1000)
    p.add_argument("--bootstrap-repeats", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError("기존 paired 결과는 덮어쓰지 않습니다")
    with open(a.high_audit) as f:high = json.load(f)
    with open(a.low_audit) as f:low = json.load(f)
    with open(a.split_manifest) as f:manifest = json.load(f)
    result = analyze_pair(high, low, manifest, sha256(a.split_manifest), repeats=a.bootstrap_repeats,
                          seed=a.seed, expected_step=a.expected_step)
    result["input_sha256"] = {"high_audit": sha256(a.high_audit), "low_audit": sha256(a.low_audit)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({"결과파일": str(out), "sha256": sha256(out), "screening": result["screening"],
                      "normal_accuracy": result["normal_accuracy"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
