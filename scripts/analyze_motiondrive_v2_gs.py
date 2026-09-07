#!/usr/bin/env python3
"""P1 동일 LAST6000 네 팔의 G/S 조건부 효과와 상호작용을 CPU로 분석한다.

각 --g0s0 등에는 원본 run 디렉터리와 독립 LAST 평가 JSON을 전달한다.
evaluate_motiondrive_v2_planning.py의 protocol/conditions.normal 출력에 직접 호환된다.
단순 export의 checkpoint_path, checkpoint_sha256, checkpoint_step,
split_manifest_sha256, supervision_manifest_sha256, report, records도 지원한다.
프레임별 metric은 d3 또는 동일 의미의 official_d3이며 둘 다 있으면 일치해야 한다.
팔별 BEST는 실제 best.pth와 기존 선택 로그의 보조표이며 LAST 효과에 섞지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_grouped_split_v2 import sha256, validate_manifest

ARMS = ("g0s0", "g1s0", "g0s1", "g1s1")
GS = ((False, False), (True, False), (False, True), (True, True))
COMMON_SHA = "a9d64f41e66a6549a98882c04fe471eb0830e064f23b06f032d8f7f3c202879c"
EXPECTED_STEP = 6000
EXPECTED_FRAMES = tuple(range(30, 300, 5))
CONTRASTS = {
    "G_given_S0": (-1., 1., 0., 0.),
    "G_given_S1": (0., 0., -1., 1.),
    "S_given_G0": (-1., 0., 1., 0.),
    "S_given_G1": (0., -1., 0., 1.),
    "interaction_GxS": (1., -1., -1., 1.),
}
IGNORED_ARGUMENTS = {"gpu", "run_dir", "goal_on", "state_on"}


def json_form(value):
    """checkpoint의 tuple과 JSON의 list라는 저장 형식 차이만 정규화한다."""
    return json.loads(json.dumps(value, allow_nan=False))


def normalize_record(record):
    row = dict(record)
    if "d3" in row and "official_d3" in row and row["d3"] != row["official_d3"]:
        raise ValueError("프레임 d3와 official_d3가 충돌합니다")
    if "d3" not in row:
        if "official_d3" not in row:
            raise ValueError("프레임별 공식 D3가 없습니다")
        row["d3"] = row["official_d3"]
    return row


def normalize_evaluation(evidence):
    """실제 planning evaluator의 중첩 출력을 명시적으로 변환한다."""
    if "protocol" not in evidence or "conditions" not in evidence:
        result = dict(evidence)
        result["records"] = [normalize_record(row) for row in evidence["records"]]
        return result
    protocol = evidence["protocol"]
    args, data = protocol["arguments"], protocol["data"]
    if (evidence.get("status") != "completed" or evidence.get("selection_performed") is not False
            or evidence.get("final_val_accessed") is not False or "normal" not in evidence["conditions"]):
        raise ValueError("완료한 정상 tune planning 평가가 아닙니다")
    expected = dict(split="tune", frame_stride=5, max_samples=0, scenes=None, batch=4, workers=4, seed=0,
                    precision="bf16")
    if (any(args.get(key) != value for key, value in expected.items())
            or args.get("device") not in ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
            or evidence.get("precision") != "bf16" or data.get("receiver_count") != 1998):
        raise ValueError("동일 전체 tune1998 / batch4 / bf16 planning 평가 조건이 아닙니다")
    if (evidence["model_load"].get("explicit_overrides")
            or evidence["model_load"].get("checkpoint_sha256") != protocol["checkpoint_sha256"]):
        raise ValueError("planning 평가의 모델 설정 override 또는 SHA 불일치")
    normal = evidence["conditions"]["normal"]
    return {"checkpoint_path": args["checkpoint"], "checkpoint_sha256": protocol["checkpoint_sha256"],
            "checkpoint_step": protocol["checkpoint_step"], "split_manifest_sha256": data["split_sha256"],
            "supervision_manifest_sha256": data["supervision_sha256"]["supervision_manifest.json"],
            "evaluation_rows_sha256": data["receiver_rows_sha256"],
            "report": normal["summary"], "records": [normalize_record(row) for row in normal["records"]],
            "planning_diagnostics": {"normal_summary": normal["summary"], "normal_buckets": normal["buckets"],
                                     "설명": "원래 planning evaluator의 시간별·종횡·상태 버킷 정상 진단 보존"},
            "planning_protocol_path": evidence["protocol_path"],
            "planning_protocol_sha256": evidence["protocol_sha256"]}


def compare_evaluation_log(report, logged):
    """기본/세션별 D3는 exact, 세션 순서만 다른 최종 평균은 집계 검사다."""
    result = {"exact_fields": {}, "session_mean_aggregation_difference": None,
              "aggregation_atol": 1e-12, "forward_tolerance": None}
    for key, value in report.items():
        if key not in logged:
            continue
        if key == "session_mean_d3":
            difference = float(value) - float(logged[key])
            if abs(difference) > 1e-12:
                raise ValueError("세션 평균의 집계 차이가 사전 고정 1e-12를 넘습니다")
            result["session_mean_aggregation_difference"] = difference
        else:
            if value != logged[key]:
                raise ValueError(f"독립 LAST 평가와 원래 로그가 정확히 일치하지 않습니다: {key}")
            result["exact_fields"][key] = True
    return result


def process_is_alive(pid):
    """signal 0은 상태를 바꾸지 않는다. 권한 부족이면 보수적으로 살아 있다고 본다."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def validate_supervisor_record(record, run_dir, manifest, *, pid_alive=None):
    """완료 manifest와 실제 자식 OS 종료는 별개 증거로 검증한다."""
    run = Path(run_dir).resolve()
    expected_manifest = run / "manifest.json"
    child_pid = record.get("child_pid")
    if type(child_pid) is not int or child_pid <= 0 or manifest.get("pid") != child_pid:
        raise ValueError("supervisor child PID와 trainer manifest PID가 다릅니다")
    if (record.get("schema_version") != 1 or not record.get("supervisor_record_id")
            or Path(record.get("run_dir", "")).resolve() != run
            or Path(record.get("trainer_manifest_path", "")).resolve() != expected_manifest):
        raise ValueError("소유한 run 경로와 supervisor 기록의 경로가 다릅니다")
    snapshot = record.get("trainer_manifest") or {}
    if (snapshot.get("exists") is not True or snapshot.get("belongs_to_child") is not True
            or snapshot.get("pid") != child_pid
            or Path(snapshot.get("manifest_path", "")).resolve() != expected_manifest
            or snapshot.get("status") != "completed" or snapshot.get("step") != EXPECTED_STEP
            or manifest.get("status") != "completed" or manifest.get("step") != EXPECTED_STEP):
        raise ValueError("supervisor의 trainer snapshot과 완료 manifest가 일치하지 않습니다")
    args = manifest.get("arguments", {})
    if Path(args.get("run_dir", "")).resolve() != run:
        raise ValueError("trainer manifest의 실행 run 경로가 소유한 디렉터리와 다릅니다")
    gpu = record.get("gpu")
    if type(gpu) is not int or gpu not in range(4) or gpu != args.get("gpu"):
        raise ValueError("supervisor와 trainer의 GPU 기록이 다릅니다")
    command = record.get("command", [])
    if len(command) < 2 or command[1] != "scripts/train_motiondrive_v2.py":
        raise ValueError("supervisor가 다른 실행 파일을 가리킵니다")
    controlled = {}
    for i, token in enumerate(command[2:], 2):
        option, equal, value = token.partition("=")
        if option not in ("--run-dir", "--gpu"):
            continue
        if option in controlled or (not equal and i + 1 >= len(command)):
            raise ValueError("supervisor 실행 명령의 run/GPU 인자가 모호합니다")
        controlled[option] = value if equal else command[i + 1]
    if (set(controlled) != {"--run-dir", "--gpu"}
            or Path(controlled["--run-dir"]).resolve() != run or controlled["--gpu"] != str(gpu)):
        raise ValueError("supervisor 실행 명령과 소유한 run/GPU 경로가 다릅니다")
    alive = (pid_alive or process_is_alive)(child_pid)
    if alive:
        raise ValueError("trainer PID가 아직 살아 있습니다. 완료 manifest만으로 분석하지 않습니다")
    if (record.get("status") != "process_exited" or type(record.get("actual_returncode")) is not int
            or record["actual_returncode"] != 0 or record.get("exit_code") != 0
            or record.get("termination_signal") is not None or record.get("termination_signal_name") is not None
            or record.get("outcome") != "completed_cleanly" or record.get("supervisor_exit_code") != 0
            or record.get("trainer_reported_completed") is not True or not record.get("ended_at")):
        raise ValueError("실제 정상 OS 종료가 확인되지 않았습니다. 비정상/미완료 종료는 별도 감사가 필요합니다")
    return {"child_pid_matches_manifest": True, "run_and_manifest_paths_match": True,
            "trainer_pid_alive_at_analysis": False, "actual_returncode": record["actual_returncode"],
            "outcome": record["outcome"], "설명": "supervisor의 실제 OS 종료 기록과 현재 PID 부재를 모두 확인"}


def load_supervisor_evidence(run_dir, manifest):
    run = Path(run_dir).resolve()
    if run.parent.name != "motiondrive_v2" or run.parent.parent.name != "work_dirs":
        raise ValueError("supervisor 경로를 추론할 수 있는 정규 run 디렉터리가 아닙니다")
    record_path = run.parents[2] / "logs/motiondrive_v2" / f"{run.name}.supervisor.json"
    if record_path.is_symlink() or not record_path.is_file():
        raise ValueError("소유한 run의 실제 supervisor 종료 기록이 없습니다")
    payload = record_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    record = json.loads(payload)
    checks = validate_supervisor_record(record, run, manifest)
    if sha256(record_path) != digest:
        raise ValueError("검증 도중 supervisor 기록이 바뀌었습니다")
    return {"path": str(record_path), "sha256": digest, "record": record, "checks": checks}


def validate_records(records, split_manifest):
    expected_scenes = set(split_manifest["splits"]["tune"])
    expected_keys = {(scene, frame) for scene in expected_scenes for frame in EXPECTED_FRAMES}
    rows = {}
    for row in records:
        key = (str(row["scenario"]), int(row["frame"]))
        if key in rows:
            raise ValueError("중복 scene/frame 평가 기록이 있습니다")
        if key not in expected_keys:
            raise ValueError("고정 tune37 stride5 밖의 평가 프레임이 있습니다")
        if row["session"] != split_manifest["scene_to_session"][key[0]]:
            raise ValueError("실제 rawtime 세션 매핑과 평가 기록이 다릅니다")
        value = float(row["d3"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("D3는 유한한 음이 아닌 값이어야 합니다")
        rows[key] = value
    if set(rows) != expected_keys or len(rows) != 1998:
        raise ValueError("동일 tune1998 프레임 전체가 필요합니다")
    return rows


def validate_experiment(arms, split_manifest, split_sha, *, expected_common_sha=COMMON_SHA):
    validate_manifest(split_manifest)
    scenes = split_manifest["splits"]["tune"]
    if len(scenes) != 37 or len({split_manifest["scene_to_session"][s] for s in scenes}) != 11:
        raise ValueError("P1의 tune37 / rawtime11 세션 조건과 다릅니다")
    if set(arms) != set(ARMS):
        raise ValueError("G/S의 네 팔이 모두 필요합니다")
    aligned, normalized_configs, normalized_args, label_rows = [], [], [], []
    for arm, (goal_on, state_on) in zip(ARMS, GS):
        data = arms[arm]
        if data["checkpoint_step"] != EXPECTED_STEP or Path(data["checkpoint_path"]).name != "last.pth":
            raise ValueError("주 분석에는 동일 LAST6000만 사용할 수 있습니다")
        if data["split_manifest_sha256"] != split_sha:
            raise ValueError("평가와 rawtime split SHA 불일치")
        manifest = data["run_manifest"]
        if manifest.get("status") != "completed" or manifest.get("step") != EXPECTED_STEP:
            raise ValueError("원본 학습이 동일 6000단계까지 완료되지 않았습니다")
        if manifest.get("nonfinite_count") != 0:
            raise ValueError("학습 비유한 값 발생 기록을 별도로 검토해야 합니다")
        if manifest.get("data_counts") != {"train": 54810, "eval": 1998}:
            raise ValueError("공통 train54810 / tune1998 데이터가 아닙니다")
        if manifest["split_sha256"] != split_sha or manifest["supervision_manifest_sha256"] != data["supervision_manifest_sha256"]:
            raise ValueError("학습/평가 데이터 계보가 다릅니다")
        if data.get("evaluation_rows_sha256", manifest["eval_rows_sha256"]) != manifest["eval_rows_sha256"]:
            raise ValueError("planning 평가와 학습 중 tune 평가 행 목록이 다릅니다")
        config = dict(manifest["model_config"])
        if config.get("goal_on") is not goal_on or config.get("state_on") is not state_on:
            raise ValueError(f"{arm}의 G/S 설정이 이름과 다릅니다")
        if config.get("motion_input_mode") != "low_feature" or list(config.get("plan_output_scale", ())) != [10., 5.]:
            raise ValueError("공통 low_feature / scale(10,5) 조건 불일치")
        config.pop("goal_on");config.pop("state_on")
        normalized_configs.append(config)
        args = manifest["arguments"]
        expected_args = dict(phase="joint", steps=6000, batch=16, eval_batch=4, workers=4, seed=0,
            precision="bf16", lr=1e-4, backbone_lr=1e-5, weight_decay=.01, grad_clip=5., warmup=200,
            alpha_occ=.2, alpha_lane=.2, alpha_motion=.2, uncertainty=1, bn_policy="fixed",
            motion_input_mode="low_feature", train_stride=1, eval_stride=5, eval_split="tune",
            max_train_samples=0, max_eval_samples=0, eval_every=250, save_every=1000,
            goal_on=int(goal_on), state_on=int(state_on), gpu=ARMS.index(arm))
        if any(args.get(key) != value for key, value in expected_args.items()):
            raise ValueError("P1의 사전 고정 실행 조건 또는 GPU 매핑이 다릅니다")
        if (args.get("resume") is not None or args.get("train_scenes") is not None
                or args.get("eval_scenes") is not None or args.get("eval_only")
                or list(args.get("plan_output_scale", ())) != [10., 5.]
                or not str(args.get("init", "")).endswith("work_dirs/motiondrive_v2/p0_motion_bn_fixed_s0/best.pth")):
            raise ValueError("공통 초기화 / 전체 데이터 / 새 일정 조건이 다릅니다")
        if manifest["load_report"].get("common_checkpoint_sha256") != expected_common_sha:
            raise ValueError("사전 고정 공통 초기화 checkpoint SHA가 아닙니다")
        if manifest["loss_weights"] != dict(plan=1., occupancy=.2, lane=.2, motion=.2, uncertainty=True):
            raise ValueError("공통 planning 및 보조 손실 구성이 다릅니다")
        bn = manifest["bn_training"]
        if bn.get("policy") != "fixed" or not bn.get("affine_and_backbone_weights_trainable"):
            raise ValueError("공통 fixed BN 정책 또는 전체 가중치 학습 조건이 다릅니다")
        normalized_args.append({k: v for k, v in args.items() if k not in IGNORED_ARGUMENTS})
        aligned.append(validate_records(data["records"], split_manifest))
        label_rows.append({(str(row["scenario"]), int(row["frame"])): {
            key: row[key] for key in ("row", "gt_abs_xy", "gt_state", "gt_state_valid", "bucket") if key in row}
            for row in data["records"]})
        record_mean = float(np.mean([float(row["d3"]) for row in data["records"]]))
        if not np.isclose(record_mean, data["report"]["official_d3"], rtol=0, atol=1e-12):
            raise ValueError("개별 프레임과 평가 보고서의 공식 D3 평균이 다릅니다")
        if data["report"].get("n") != 1998 or data["report"].get("n_sessions") != 11:
            raise ValueError("평가 보고서의 표본 수가 다릅니다")
        if data["best"]["selection_definition"] != "official_d3" or not 0 < data["best"]["step"] <= EXPECTED_STEP:
            raise ValueError("팔별 BEST의 원래 공식 D3 선택 계보가 없습니다")
    if any(config != normalized_configs[0] for config in normalized_configs[1:]):
        raise ValueError("G/S 외 모델 구조도 다릅니다")
    if any(args != normalized_args[0] for args in normalized_args[1:]):
        raise ValueError("G/S·GPU·run 경로 외 학습 인자도 다릅니다")
    for key in ("git_sha", "initial_model_state_sha256", "initial_parameter_count", "train_rows_sha256",
                "eval_rows_sha256", "load_report", "bn_training"):
        values = [arms[arm]["run_manifest"].get(key) for arm in ARMS]
        if values[0] is None or any(value != values[0] for value in values[1:]):
            raise ValueError(f"공정성 전제 불일치 또는 누락: {key}")
    for key in ("sample_order_sha256_at_checkpoint", "supervision_manifest_sha256"):
        values = [arms[arm].get(key) for arm in ARMS]
        if not values[0] or any(value != values[0] for value in values[1:]):
            raise ValueError(f"공정성 전제 불일치 또는 누락: {key}")
    keys = sorted(aligned[0])
    if any(labels != label_rows[0] for labels in label_rows[1:]):
        raise ValueError("같은 프레임의 GT/행 번호/버킷이 팔별로 다릅니다")
    matrix = np.asarray([[values[key] for values in aligned] for key in keys], np.float64)
    session_ids = [split_manifest["scene_to_session"][scene] for scene, _ in keys]
    return keys, matrix, session_ids


def joint_session_bootstrap(matrix, session_ids, *, repeats=10000, seed=20260907):
    """단 하나의 세션 재표집 배열을 네 팔과 모든 contrast에 공동 적용한다."""
    values = np.asarray(matrix, np.float64)
    ids = np.asarray(session_ids, dtype=str)
    if (values.ndim != 2 or values.shape[1] != 4 or values.shape[0] != len(ids)
            or not np.isfinite(values).all() or not len(ids) or repeats < 100):
        raise ValueError("유효한 N×4 paired 배열과 100회 이상의 bootstrap이 필요합니다")
    sessions = sorted(set(ids))
    if len(sessions) < 2:
        raise ValueError("둘 이상의 독립 세션이 필요합니다")
    totals = np.asarray([values[ids == session].sum(axis=0) for session in sessions])
    counts = np.asarray([(ids == session).sum() for session in sessions])
    means = totals / counts[:, None]
    draws = np.random.default_rng(seed).integers(0, len(sessions), size=(repeats, len(sessions)))
    bootstrap_primary = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    bootstrap_secondary = means[draws].mean(axis=1)
    point_primary = values.mean(axis=0)
    point_secondary = means.mean(axis=0)
    contrast = np.asarray(list(CONTRASTS.values()), np.float64)

    def summarize(point, distribution):
        return {"estimate": float(point), "ci95_percentile": np.quantile(distribution, [.025, .975]).tolist()}

    results = {}
    for name, point, sampled in (("primary_frame_weighted", point_primary, bootstrap_primary),
                                  ("secondary_session_mean", point_secondary, bootstrap_secondary)):
        effects, distributions = point @ contrast.T, sampled @ contrast.T
        results[name] = {"arms": {arm: summarize(point[i], sampled[:, i]) for i, arm in enumerate(ARMS)},
                         "contrasts": {key: {**summarize(effects[i], distributions[:, i]),
                                              "coefficients_in_arm_order": list(CONTRASTS[key])}
                                       for i, key in enumerate(CONTRASTS)}}
    results["resampling"] = {"n_sessions": len(sessions), "n_frames": len(ids), "session_frame_counts": dict(zip(sessions, map(int, counts))),
                              "bootstrap_repeats": repeats, "seed": seed,
                              "shared_draws": True, "설명": "네 팔·다섯 contrast·두 추정량에 같은 세션 draw 배열 사용"}
    return results


def analyze_gs(arms, split_manifest, split_sha, *, expected_common_sha=COMMON_SHA, repeats=10000, seed=20260907):
    keys, matrix, sessions = validate_experiment(arms, split_manifest, split_sha, expected_common_sha=expected_common_sha)
    statistics = joint_session_bootstrap(matrix, sessions, repeats=repeats, seed=seed)
    result = {"상태": "동일 LAST6000 G×S 분석 완료", "arm_order": list(ARMS),
              "protocol": {"checkpoint": "LAST6000 주 분석 / 팔별 BEST 별도 보조표", "n_frames": 1998,
                           "n_scenes": 37, "n_rawtime_sessions": 11, "augmentation": False,
                           "metric": "sum([11,11,5,5,2,2]/36 × 6점 L2)의 프레임 평균",
                           "primary": "프레임 가중 평균; 재표집 세션 안 모든 프레임 수 보존",
                           "secondary": "세션별 평균의 단순평균; 주 추정량과 섞지 않음",
                           "sign": "조건부 효과는 ON−OFF, 상호작용은 D11−D10−D01+D00. 음수는 오차 감소 방향",
                           "aggregation_check_atol": 1e-12,
                           "aggregation_check_note": "프레임 값의 평균 집계 일치 검사이며 model forward 재현 허용오차가 아님"},
              "split_manifest_sha256": split_sha, "common_checkpoint_sha256": expected_common_sha,
              "last6000": statistics,
              "last_auxiliary": {arm: arms[arm].get("training_last_report", arms[arm]["report"]) for arm in ARMS},
              "last_planning_diagnostics": {arm: arms[arm].get("planning_diagnostics") for arm in ARMS},
              "best_secondary_selection_table": {arm: arms[arm]["best"] for arm in ARMS},
              "fairness": "공통 초기 state/파라미터 수/BN/소스/데이터 행/누적 샘플 순서와 G/S 외 설정 일치",
              "checkpoint_sha256": {arm: arms[arm]["checkpoint_sha256"] for arm in ARMS},
              "per_frame_pairs": [{"scenario": scene, "frame": frame, "session": sessions[i],
                                   **{arm: float(matrix[i, j]) for j, arm in enumerate(ARMS)}}
                                  for i, (scene, frame) in enumerate(keys)],
              "주의": ["단일 seed와 반복 사용한 tune11 세션의 개발 결과이며 untouched test 성능 인증이 아님",
                       "여러 contrast의 구간은 탐색적이며 다중비교 보정 없는 확정적 유의성 선언을 하지 않음",
                       "G 이득은 경로 의도 조건의 효과도 포함하며 시각 표현 일반화 개선과 동일하지 않음",
                       "S OFF에도 raw motion feature와 보조 감독이 있으므로 S 효과는 명시적 예측 상태 전달의 순효과",
                       "BEST 표는 팔별 checkpoint 선택을 포함한 보조 수치. LAST 요인효과나 CI에 혼합하지 않음",
                       "공통 fixed BN 선택에서 occupancy IoU −0.0199의 손해가 있었으며 팔별 occupancy/lane도 함께 추적"]}
    result["screening"] = {name: {"mean_reduction_at_least_0p01_m": row["estimate"] <= -.01,
                                   "session_ci_entirely_below_zero": row["ci95_percentile"][1] < 0}
                           for name, row in statistics["primary_frame_weighted"]["contrasts"].items()
                           if name != "interaction_GxS"}
    return result


def load_arm(run_dir, evaluation_path):
    """완료한 저장 파일과 평가 export의 계보를 CPU에서 직접 확인한다."""
    import torch
    run_dir, evaluation_path = Path(run_dir), Path(evaluation_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    supervisor = load_supervisor_evidence(run_dir, manifest)
    evidence = normalize_evaluation(json.loads(evaluation_path.read_text()))
    if "planning_protocol_path" in evidence:
        if sha256(evidence["planning_protocol_path"]) != evidence["planning_protocol_sha256"]:
            raise ValueError("planning 평가의 사전 고정 protocol 파일 SHA가 다릅니다")
    last_path = run_dir / "last.pth"
    if Path(evidence["checkpoint_path"]).resolve() != last_path.resolve():
        raise ValueError("평가 export가 다른 checkpoint 경로를 가리킵니다")
    if evidence["checkpoint_sha256"] != sha256(last_path):
        raise ValueError("평가 export와 실제 LAST 파일 SHA가 다릅니다")
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    if checkpoint["step"] != EXPECTED_STEP or evidence["checkpoint_step"] != EXPECTED_STEP:
        raise ValueError("저장 checkpoint와 평가 export가 LAST6000이 아닙니다")
    embedded = json_form(checkpoint["manifest"])
    for key in ("git_sha", "arguments", "model_config", "loss_weights", "split_sha256", "supervision_manifest_sha256",
                "initial_model_state_sha256", "initial_parameter_count", "train_rows_sha256", "eval_rows_sha256",
                "load_report", "bn_training", "data_counts"):
        if embedded.get(key) != manifest.get(key):
            raise ValueError(f"저장 checkpoint와 완료 manifest의 계보 불일치: {key}")
    if any(not torch.isfinite(value).all() for value in checkpoint["model"].values()):
        raise ValueError("저장 모델에 비유한 tensor가 있습니다")
    del checkpoint
    metrics_path = run_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    train = [row for row in rows if row.get("kind") == "train" and row.get("step") == EXPECTED_STEP]
    if len(train) != 1 or not train[0].get("sample_order_sha256"):
        raise ValueError("LAST6000 누적 row 순서 SHA가 없습니다")
    evaluations = [row for row in rows if row.get("kind") == "eval"]
    last_eval = [row for row in evaluations if row.get("step") == EXPECTED_STEP]
    if len(last_eval) != 1:
        raise ValueError("LAST6000 원래 평가 로그가 없습니다")
    # 독립 평가를 로그와 먼저 대조한다. 기대에 맞춰 허용오차를 넓히지 않는다.
    log_comparison = compare_evaluation_log(evidence["report"], last_eval[0])
    if any(row.get("selection_definition") != "official_d3" for row in evaluations):
        raise ValueError("P1의 원래 checkpoint 선택 기준이 아닙니다")
    selected = min(evaluations, key=lambda row: row["selection_metric"])
    best_path = run_dir / "best.pth"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best["step"] != selected["step"]:
        raise ValueError("BEST 파일과 원래 최소 공식 D3 선택 로그가 다릅니다")
    best_summary = {"path": str(best_path), "sha256": sha256(best_path), "step": best["step"],
                    "selection_definition": selected["selection_definition"], "official_d3": selected["official_d3"],
                    "session_mean_d3": selected["session_mean_d3"], "n": selected["n"],
                    "occ_iou": selected["occ_iou"], "lane_iou": selected["lane_iou"],
                    "state_mae_vx_vy_ax_ay_yawrate": selected["state_mae_vx_vy_ax_ay_yawrate"],
                    "history_position_mae_by_offset": selected["history_position_mae_by_offset"],
                    "설명": "원래 full tune 공식 D3로 선택한 BEST의 로그 수치; LAST 효과에 혼합하지 않음"}
    del best
    return {**evidence, "run_manifest": manifest, "sample_order_sha256_at_checkpoint": train[0]["sample_order_sha256"],
            "supervisor_exit_evidence": supervisor,
            "training_last_report": last_eval[0], "log_comparison": log_comparison,
            "best": best_summary, "provenance": {"evaluation_path": str(evaluation_path), "evaluation_sha256": sha256(evaluation_path),
                                                  "run_manifest_path": str(run_dir / "manifest.json"),
                                                  "run_manifest_sha256": sha256(run_dir / "manifest.json"),
                                                  "metrics_path": str(metrics_path), "metrics_sha256": sha256(metrics_path)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument("--" + arm, nargs=2, metavar=("RUN_DIR", "LAST_EVALUATION_JSON"), required=True)
    parser.add_argument("--split-manifest", default="data/etri/motiondrive_v2/grouped_split_rawtime.json")
    parser.add_argument("--expected-common-sha", default=COMMON_SHA)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("기존 G/S 비교 결과는 덮어쓰지 않습니다")
    manifest = json.loads(Path(args.split_manifest).read_text())
    arms = {arm: load_arm(*getattr(args, arm)) for arm in ARMS}
    result = analyze_gs(arms, manifest, sha256(args.split_manifest), expected_common_sha=args.expected_common_sha,
                        repeats=args.bootstrap_repeats, seed=args.seed)
    result["input_provenance"] = {arm: arms[arm]["provenance"] for arm in ARMS}
    result["actual_os_exit_evidence"] = {arm: arms[arm]["supervisor_exit_evidence"] for arm in ARMS}
    result["independent_evaluation_log_checks"] = {arm: arms[arm]["log_comparison"] for arm in ARMS}
    result["source_sha256"] = sha256(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"파일": str(output), "sha256": sha256(output), "screening": result["screening"],
                      "contrasts": result["last6000"]["primary_frame_weighted"]["contrasts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
