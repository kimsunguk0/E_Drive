#!/usr/bin/env python3
"""동일 low_feature의 BN adaptive/fixed LAST1000 비교와 CPU 버퍼 감사.

기존 해상도 비교의 mode 가정은 사용하지 않는다. 통계 함수만 재사용한다.
정상 정확도와 시간 교란 효과는 분리하며 주 신뢰구간은 rawtime 11세션이다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_motiondrive_v2_motion_pair import paired_cluster_bootstrap
from build_grouped_split_v2 import sha256, validate_manifest

EXPECTED_FRAMES = [30, 55, 85, 115, 145, 175, 205, 235, 265, 295]
CONDITIONS = ("normal", "repeat_current", "reverse_history", "cross_scene_history")
METRICS = ("vx_mae_m_s", "history_position_mae_mean4_m")
POLICIES = ("adaptive", "fixed")
IGNORED_ARGUMENTS = {"gpu", "run_dir", "bn_policy"}
INIT_SUFFIX = "work_dirs/motiondrive_v2/p0_motion_lowfeature_s0/last.pth"


def validate_bn_pair(adaptive, fixed, manifest, split_sha, expected_step=1000):
    """다른 해상도나 best 혼합, 추가 학습 조작이 있으면 분석을 중단한다."""
    validate_manifest(manifest)
    scenes = sorted(manifest["splits"]["tune"])
    if len(scenes) != 37 or len({manifest["scene_to_session"][s] for s in scenes}) != 11:
        raise ValueError("사전 고정 tune 37개 scene / 11개 rawtime 세션과 다릅니다")
    checkpoints = []
    for policy, report in zip(POLICIES, (adaptive, fixed)):
        if report["split_manifest_sha256"] != split_sha:
            raise ValueError("rawtime split SHA 불일치")
        protocol = report["protocol"]
        if (protocol.get("split") != "tune" or protocol.get("n") != 370
                or protocol.get("frames_per_scene") != EXPECTED_FRAMES
                or protocol.get("expected_step") != expected_step
                or protocol.get("checkpoints") != ["last.pth"]):
            raise ValueError("고정 370표본 / LAST1000 평가 protocol 불일치")
        if set(report["checkpoint_results"]) != {"last"}:
            raise ValueError("LAST만 비교합니다. best 또는 initial을 혼합하지 않습니다")
        checkpoint = report["checkpoint_results"]["last"]
        if checkpoint["step"] != expected_step or Path(checkpoint["path"]).name != "last.pth":
            raise ValueError("사전 고정 LAST1000이 아닙니다")
        config = checkpoint["model_config"]
        if (config.get("motion_input_mode") != "low_feature"
                or config.get("goal_on") is not False or config.get("state_on") is not False
                or list(config.get("plan_output_scale", ())) != [1., 1.]):
            raise ValueError("양쪽 모두 low_feature / G0S0 / scale(1,1)이어야 합니다")
        provenance = checkpoint["run_provenance"]
        args = provenance["arguments"]
        if args.get("bn_policy") != policy:
            raise ValueError(f"BN 정책 불일치: {policy}")
        expected = {"phase": "pretrain", "steps": expected_step, "batch": 16, "seed": 0,
                    "warmup": 100, "goal_on": 0, "state_on": 0, "motion_input_mode": "low_feature",
                    "precision": "bf16", "eval_split": "tune", "train_stride": 1, "eval_stride": 5,
                    "max_train_samples": 0, "max_eval_samples": 0, "lr": 1e-4, "backbone_lr": 1e-5,
                    "alpha_occ": .2, "alpha_lane": .2, "alpha_motion": .2, "weight_decay": .01,
                    "grad_clip": 5., "uncertainty": 1}
        if any(args.get(key) != value for key, value in expected.items()):
            raise ValueError("쌍 D의 사전 고정 학습 조건과 다릅니다")
        if (args.get("resume") is not None or args.get("train_scenes") is not None
                or args.get("eval_scenes") is not None or args.get("eval_only")
                or not str(args.get("init", "")).endswith(INIT_SUFFIX)
                or list(args.get("plan_output_scale", ())) != [1., 1.]):
            raise ValueError("공통 LAST 초기화 / 전체 train+tune / 새 일정 조건 불일치")
        if provenance["loss_weights"].get("plan") != 0:
            raise ValueError("P0 planning loss는 0이어야 합니다")
        if provenance.get("model_config") != config:
            raise ValueError("모델 설정 계보가 일치하지 않습니다")
        if set(checkpoint["conditions"]) != set(CONDITIONS):
            raise ValueError("네 영상 조건이 모두 필요합니다")
        for condition in CONDITIONS:
            values = checkpoint["conditions"][condition]
            if (values.get("n"), values.get("n_scenes"), values.get("n_sessions")) != (370, 37, 11):
                raise ValueError("조건별 전체 표본 수가 다릅니다")
            rows = values["per_scene"]
            if sorted(rows) != scenes:
                raise ValueError("조건별 scene 집합이 다릅니다")
            for scene in scenes:
                row = rows[scene]
                if row.get("n") != 10 or row.get("frames") != EXPECTED_FRAMES:
                    raise ValueError("조건별 고정 프레임이 다릅니다")
                if row.get("session_id") != manifest["scene_to_session"][scene]:
                    raise ValueError("실제 rawtime 세션 매핑 불일치")
                if any(not np.isfinite(row[metric]) for metric in METRICS):
                    raise ValueError("비유한 오차가 있습니다")
        checkpoints.append(checkpoint)
    a, f = [checkpoint["run_provenance"] for checkpoint in checkpoints]
    for key in ("git_sha", "initial_model_state_sha256", "initial_parameter_count", "train_rows_sha256",
                "eval_rows_sha256", "sample_order_sha256_at_checkpoint", "loss_weights", "load_report"):
        if a.get(key) is None or a.get(key) != f.get(key):
            raise ValueError(f"공정성 불일치 또는 누락: {key}")
    if checkpoints[0]["model_config"] != checkpoints[1]["model_config"]:
        raise ValueError("BN 정책 외 모델 구조도 다릅니다")
    aa = {k: v for k, v in a["arguments"].items() if k not in IGNORED_ARGUMENTS}
    fa = {k: v for k, v in f["arguments"].items() if k not in IGNORED_ARGUMENTS}
    if aa != fa:
        raise ValueError("BN 정책·GPU·run 경로 외 학습 인자도 다릅니다")
    common_sha = a["load_report"].get("common_checkpoint_sha256")
    if not common_sha:
        raise ValueError("공통 초기화 checkpoint SHA가 없습니다")
    if adaptive["supervision_manifest_sha256"] != fixed["supervision_manifest_sha256"]:
        raise ValueError("supervision 계보가 다릅니다")
    if adaptive["protocol"]["donor_mapping"] != fixed["protocol"]["donor_mapping"]:
        raise ValueError("다른 scene 과거 영상 교란의 donor 매핑이 다릅니다")
    return scenes, checkpoints


def analyze_bn_pair(adaptive, fixed, manifest, split_sha, *, repeats=10000, seed=20260907,
                    expected_step=1000):
    scenes, checkpoints = validate_bn_pair(adaptive, fixed, manifest, split_sha, expected_step)
    sessions = [manifest["scene_to_session"][scene] for scene in scenes]
    counts = [10] * len(scenes)

    def values(checkpoint, condition, metric):
        rows = checkpoint["conditions"][condition]["per_scene"]
        return np.asarray([rows[scene][metric] for scene in scenes], np.float64)

    def interval(delta):
        return {"primary_rawtime_session": paired_cluster_bootstrap(delta, counts, sessions, repeats=repeats, seed=seed),
                "secondary_scene": paired_cluster_bootstrap(delta, counts, scenes, repeats=repeats, seed=seed)}

    result = {"상태": "BN 쌍 D LAST1000 paired 분석 완료", "expected_step": expected_step,
              "protocol": {"n_scenes": 37, "n_rawtime_sessions": 11, "n_frames": 370,
                           "frames_per_scene": EXPECTED_FRAMES, "augmentation": False,
                           "bootstrap_repeats": repeats, "seed": seed,
                           "primary_ci": "rawtime 세션 11개를 복원추출한 paired percentile 95% 구간",
                           "secondary_ci": "scene 37개 기준 보조 구간; 프레임 독립 bootstrap 금지",
                           "sampling": "세션 내 paired 차이와 전체 고정 표본 수를 함께 재표집한 프레임 가중 평균",
                           "checkpoint": "같은 LAST1000만 사용. best는 혼합하거나 재선택하지 않음",
                           "manipulation": "두 팔 모두 low_feature / G0S0; BN 학습 정책만 adaptive 대 fixed"},
              "fairness_checks": "동일 초기 전체 state(버퍼 포함), 모델, 소스, 행 목록, 누적 표본 순서 및 BN 외 인자 일치",
              "split_manifest_sha256": split_sha,
              "checkpoint_sha256": {policy: c["sha256"] for policy, c in zip(POLICIES, checkpoints)},
              "normal_accuracy": {}, "temporal_controls": {}, "per_scene": {},
              "주의": ["11세션의 작은 개발 표본이며 단일 seed 결과임",
                       "정상 오차 감소와 영상 교란 효과를 별도로 보고함",
                       "영상 교란은 분포 밖 진단이며 정보량 상한 또는 완전한 운동 인과 증명이 아님",
                       "여러 지표와 교란의 신뢰구간은 탐색적이며 다중비교 보정에 따른 유의성 선언이 아님",
                       "P0 planning loss가 0이므로 이 분석은 planning 성능이나 최종 채택을 증명하지 않음"]}
    for metric in METRICS:
        normal = [values(checkpoint, "normal", metric) for checkpoint in checkpoints]
        delta = normal[1] - normal[0]
        result["normal_accuracy"][metric] = {policy: float(v.mean()) for policy, v in zip(POLICIES, normal)}
        result["normal_accuracy"][metric].update(sign="fixed-adaptive; 음수가 fixed 개선", **interval(delta))
        result["per_scene"][metric] = {scene: {"session": sessions[i], "adaptive_normal": float(normal[0][i]),
                                               "fixed_normal": float(normal[1][i]),
                                               "normal_fixed_minus_adaptive": float(delta[i])}
                                         for i, scene in enumerate(scenes)}
        result["temporal_controls"][metric] = {}
        for condition in CONDITIONS[1:]:
            advantage = [values(c, condition, metric) - n for c, n in zip(checkpoints, normal)]
            result["temporal_controls"][metric][condition] = {
                "within_arm_sign": "교란-normal; 양수가 정상 과거 영상 사용의 관측상 이점",
                "adaptive_advantage": interval(advantage[0]), "fixed_advantage": interval(advantage[1]),
                "difference_in_advantages_fixed_minus_adaptive": interval(advantage[1] - advantage[0])}
    vx = result["normal_accuracy"]["vx_mae_m_s"]["primary_rawtime_session"]
    result["screening"] = {"mean_vx_reduction_at_least_0p1_m_s": bool(vx["mean_difference"] <= -.1),
                           "normal_vx_session_ci_entirely_below_zero": bool(vx["ci95_percentile"][1] < 0),
                           "설명": "평균 정확도와 시간 대조의 일부 조건이며 자동 채택 규칙이 아님"}
    return result


def bn_buffer_comparison(initial_state, last_state):
    """모델을 만들지 않고 CPU tensor만 비교한다. BN affine은 버퍼와 분리한다."""
    import torch
    from motiondrive_v2_training import tensor_state_sha256
    if set(initial_state) != set(last_state):
        raise ValueError("initial/LAST state key가 다릅니다")
    suffixes = (".running_mean", ".running_var", ".num_batches_tracked")
    keys = sorted(k for k in initial_state if k.endswith(suffixes))
    if not keys:
        raise ValueError("BN running buffer가 없습니다")
    prefixes = sorted(k.removesuffix(".running_mean") for k in keys if k.endswith(".running_mean"))
    affine = [f"{prefix}.{kind}" for prefix in prefixes for kind in ("weight", "bias")
              if f"{prefix}.{kind}" in initial_state]
    changed = [k for k in keys if not torch.equal(initial_state[k], last_state[k])]
    affine_changed = [k for k in affine if not torch.equal(initial_state[k], last_state[k])]
    return {"n_bn_modules": len(prefixes), "n_buffer_tensors": len(keys),
            "initial_buffer_sha256": tensor_state_sha256({k: initial_state[k] for k in keys}),
            "last_buffer_sha256": tensor_state_sha256({k: last_state[k] for k in keys}),
            "buffer_changed_count": len(changed), "changed_buffer_names": changed,
            "affine_tensor_count": len(affine), "affine_changed_count": len(affine_changed),
            "전체버퍼동일": not changed, "설명": "BN 통계 고정과 BN affine 가중치 동결은 다른 조작"}


def audit_checkpoint_buffers(checkpoints, expected_step=1000):
    import torch
    from motiondrive_v2_training import tensor_state_sha256
    result = {}
    for policy, evidence in zip(POLICIES, checkpoints):
        last_path = Path(evidence["path"])
        initial_path = last_path.parent / "initial.pth"
        initial = torch.load(initial_path, map_location="cpu", weights_only=False)
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        if initial["step"] != 0 or last["step"] != expected_step or sha256(last_path) != evidence["sha256"]:
            raise ValueError("감사 입력 checkpoint 파일 또는 step이 달라졌습니다")
        actual_initial_sha = tensor_state_sha256(initial["model"])
        if actual_initial_sha != evidence["run_provenance"]["initial_model_state_sha256"]:
            raise ValueError("실제 초기 전체 state SHA와 기록이 다릅니다")
        manifest = last["manifest"]
        bn = manifest["bn_training"]
        if (bn["policy"] != policy or not bn["affine_and_backbone_weights_trainable"]
                or bn["inference_policy"] != "eval running statistics for both arms"
                or manifest["data_counts"] != {"train": 54810, "eval": 1998}):
            raise ValueError("BN 정책 또는 전체 학습 데이터 계보가 다릅니다")
        detail = bn_buffer_comparison(initial["model"], last["model"])
        if detail["n_bn_modules"] != bn["modules"]:
            raise ValueError("BN module 수 기록과 실제 버퍼가 다릅니다")
        if (policy == "fixed") != detail["전체버퍼동일"]:
            raise ValueError("fixed 보존 / adaptive 갱신 정책이 실제 버퍼에서 재현되지 않습니다")
        if not detail["affine_changed_count"]:
            raise ValueError("BN affine 학습 변화가 관측되지 않습니다")
        detail.update(initial_path=str(initial_path), initial_checkpoint_sha256=sha256(initial_path),
                      last_path=str(last_path), last_checkpoint_sha256=evidence["sha256"],
                      initial_model_state_sha256=actual_initial_sha)
        result[policy] = detail
        del initial, last
    if result["adaptive"]["initial_buffer_sha256"] != result["fixed"]["initial_buffer_sha256"]:
        raise ValueError("두 팔의 실제 초기 BN 버퍼가 다릅니다")
    return result


def full_tune_last(checkpoints, expected_step=1000):
    result = {"설명": "원본 metrics.jsonl의 LAST1000 정상 tune1998 평가. 고정370 감사와 표본 수가 다름. best와 혼합하지 않음"}
    for policy, checkpoint in zip(POLICIES, checkpoints):
        path = Path(checkpoint["path"]).parent / "metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        selected = [row for row in rows if row.get("kind") == "eval" and row.get("step") == expected_step]
        if len(selected) != 1 or selected[0]["n"] != 1998 or selected[0]["n_sessions"] != 11:
            raise ValueError("동일 LAST1000 전체 tune1998 정상 평가가 없습니다")
        row = selected[0]
        result[policy] = {key: row[key] for key in ("step", "n", "n_sessions", "state_mae_vx_vy_ax_ay_yawrate",
                                                    "history_position_mae_by_offset", "occ_iou", "lane_iou")}
        result[policy].update(source=str(path), source_sha256=sha256(path),
                              history_position_mae_mean4_m=float(np.mean(row["history_position_mae_by_offset"])))
        candidates = [r for r in rows if r.get("kind") == "eval"]
        best = min(candidates, key=lambda r: r["selection_metric"])
        result[policy]["best_separate_context"] = {"step": best["step"], "selection_metric": best["selection_metric"],
                                                    "selection_definition": best["selection_definition"],
                                                    "설명": "기존 history 선택 기준의 별도 기록; LAST 비교에 섞지 않음"}
    result["fixed_minus_adaptive"] = {
        "vx_mae_m_s": result["fixed"]["state_mae_vx_vy_ax_ay_yawrate"][0] - result["adaptive"]["state_mae_vx_vy_ax_ay_yawrate"][0],
        "history_position_mae_mean4_m": result["fixed"]["history_position_mae_mean4_m"] - result["adaptive"]["history_position_mae_mean4_m"],
        "occ_iou": result["fixed"]["occ_iou"] - result["adaptive"]["occ_iou"],
        "lane_iou": result["fixed"]["lane_iou"] - result["adaptive"]["lane_iou"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-audit", required=True)
    parser.add_argument("--fixed-audit", required=True)
    parser.add_argument("--split-manifest", default="data/etri/motiondrive_v2/grouped_split_rawtime.json")
    parser.add_argument("--expected-step", type=int, default=1000)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("기존 BN 비교 결과는 덮어쓰지 않습니다")
    adaptive = json.loads(Path(args.adaptive_audit).read_text())
    fixed = json.loads(Path(args.fixed_audit).read_text())
    manifest = json.loads(Path(args.split_manifest).read_text())
    split_sha = sha256(args.split_manifest)
    result = analyze_bn_pair(adaptive, fixed, manifest, split_sha, repeats=args.bootstrap_repeats,
                             seed=args.seed, expected_step=args.expected_step)
    checkpoints = [report["checkpoint_results"]["last"] for report in (adaptive, fixed)]
    result["bn_buffer_audit"] = audit_checkpoint_buffers(checkpoints, args.expected_step)
    result["normal_full_tune1998"] = full_tune_last(checkpoints, args.expected_step)
    result["fixed370_auxiliary"] = {policy: {condition: {k: v for k, v in values.items() if k != "per_scene"}
                                             for condition, values in checkpoint["conditions"].items()}
                                    for policy, checkpoint in zip(POLICIES, checkpoints)}
    result["input_sha256"] = {"adaptive_audit": sha256(args.adaptive_audit), "fixed_audit": sha256(args.fixed_audit)}
    result["source_sha256"] = {str(Path(__file__).name): sha256(__file__)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"파일": str(output), "sha256": sha256(output), "screening": result["screening"],
                      "normal_accuracy": result["normal_accuracy"],
                      "normal_full_tune1998": result["normal_full_tune1998"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
