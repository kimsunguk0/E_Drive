"""CPU audit of the preregistered repeat/history/common-status comparison.

This program loads saved arrays and source receipts, never a model or GPU. The
fixed comparison is terminal step 2000 on the reused 1,998-row tune population.
Bootstrap draws resample 11 sessions, retain every row in each selected session,
and use frame-weighted means. The same 20,000 draws (seed 0) serve all endpoints
and both prespecified contrasts. These are exploratory marginal intervals, not
simultaneous intervals, new-seed uncertainty, blind confirmation, or rule approval.

Selected D3 and point errors are independently recomputed in float64 from saved
float32 XY errors. Oracle uses the saved FP32 value because full shortlists were
not saved; regret subtracts that oracle from recomputed D3. The original GPU/
NumPy float32 summaries are retained separately. Raster
IoU is available only as a run-level aggregate: no invented pixel/session CI or
claim of exact raster-logit parity is made. Initial A/B state auxiliaries may
differ; B/C have identical image inputs and must initially agree.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


WEIGHTS = np.asarray([11, 11, 5, 5, 2, 2], np.float64) / 36.
PUBLIC_SHA = "330072b133981f77b0b18b669216ad50b211552a82ea45ac52d507ec9021a735"
BANK_SHA = "4aff9b40696f91383bfbec377f388e6209d619fa509e51a2019b19af977b5e04"
TRAIN_SHA = "75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854"
TUNE_SHA = "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88"
SOURCE_PINS = {
    "train_temporal_comparison.py": "a31296105ebecca63a40204e6cb7d95b086599697ef586726cf29f2c604a70e6",
    "temporal_model.py": "95684e777e18981f1beb2cc9485e08da9301e3e1a73fb2990eaaa9cd7330382a",
    "temporal_data.py": "4fc75a92b503d472a0fcb18189cbcedce6dd4c2c722219e04c0c0fa4d09f672b",
}
ARM_LABELS = {"A": "현재 front 반복", "B": "실제 과거 front .1/.5초",
              "C": "B + 공통 perception status"}
ARM_MODES = {"A": ("repeat", False), "B": ("real", False), "C": ("real", True)}
STATE_FIELDS = ("vx", "vy", "ax", "ay")
PINNED_ARGUMENTS = dict(steps=2000, batch=16, eval_batch=8, workers=4, seed=0,
    lr=1e-4, backbone_lr=1e-5, new_lr=1e-3, weight_decay=.01, warmup=100,
    temperature=.1, perception_weight=.25, state_weight=.1,
    occ_pos_weight=4., lane_pos_weight=8., eval_limit=0, checkpoint_every=500)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def bitwise(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


class Reader:
    def __init__(self):
        self.artifacts = {}

    def read(self, path, kind="json"):
        path = Path(path).resolve()
        before = file_sha(path)
        if kind == "npz":
            with np.load(path, allow_pickle=False) as z:
                value = {key: z[key] for key in z.files}
        elif kind == "jsonl":
            text = path.read_text()
            require(text.endswith("\n"), f"Incomplete final JSONL line: {path}")
            value = [json.loads(line) for line in text.splitlines()]
        elif kind == "source":
            value = before
        else:
            value = json.loads(path.read_text())
        require(file_sha(path) == before, f"Artifact changed while reading: {path}")
        self.artifacts[str(path)] = before
        return value

    def verify_unchanged(self):
        for path, digest in self.artifacts.items():
            require(file_sha(path) == digest, f"Artifact changed during analysis: {path}")


def normalized_manifest(manifest):
    result = copy.deepcopy(manifest)
    for key in ("physical_gpu", "pid", "perception_status_route"):
        result.pop(key, None)
    for key in ("run_dir", "history_mode", "common_status"):
        result["arguments"].pop(key)
    for split in ("train", "validation"):
        result[split]["temporal_data"].pop("history_mode")
    return result


def check_manifest(m, arm, directory, reader):
    require(m["schema"] == "sdv2_temporal_comparison_v1", "Unexpected experiment schema")
    require(m["public_checkpoint_sha256"] == PUBLIC_SHA and m["bank_sha256"] == BANK_SHA,
            "Public initialization or primary bank differs")
    require(m["public_tensors_unchanged_by_adapter_construction"], "Public weights changed in constructor")
    require(m["physical_gpu"] in (0, 1, 4), "Unallocated GPU in receipt")
    require(m["training_rng_seed"] == 1, "Wrong training RNG reset")
    mode, status = ARM_MODES[arm]
    require(m["arguments"]["history_mode"] == mode and m["arguments"]["common_status"] is status,
            f"Wrong intervention assigned to arm {arm}")
    for key, value in PINNED_ARGUMENTS.items():
        require(m["arguments"][key] == value, f"Changed preregistered argument: {key}")
    require(m["planner_status"] == "constant zero, no raw or predicted status"
            and m["relative_head_status"] == "constant zero, no raw or predicted status",
            "Planner/relative direct-status route changed")
    require(m["goal_route"] == "only relative scoring of completed original bank candidates",
            "Goal route changed")
    require(m["perception_status_route"] == ("common temporal image attention query only" if status else "absent"),
            "Common-status route receipt differs")
    require(m["parameter_counts"] == dict(backbone=23891776, public_head=17917051, new=562055),
            "Parameter counts differ from the matched design")
    initial = m["initial_full_gpu_contract"]
    require(initial["public_zero_status_output_exact"] and initial["planner_status_zero_observed"],
            "Initial real-GPU bank/constant-status contract failed")
    for key, digest in SOURCE_PINS.items():
        require(m["source_sha256"].get(key) == digest, f"Frozen experiment source differs: {key}")
    for key, digest in m["source_sha256"].items():
        source = Path(key)
        require(not source.is_absolute() and ".." not in source.parts, "Unsafe source receipt path")
        require(reader.read(directory / "source" / source, "source") == digest,
                f"Source snapshot bytes differ: {key}")
    for split, count, digest, scenes, sessions in (
            ("train", 54810, TRAIN_SHA, 203, 72), ("validation", 1998, TUNE_SHA, 37, 11)):
        s = m[split]
        require(s["rows"] == s["allowed_rows"] == count
                and s["rows_sha256"] == s["allowed_rows_sha256"] == digest,
                f"Unexpected {split} row population")
        require(len(s["scenes"]) == scenes and len(s["sessions"]) == sessions,
                f"Unexpected {split} scene/session count")
        require(s["status_mode"] == "zero" and s["goal_mode"] == "selection",
                "Underlying data input modes differ")
        require(s["temporal_data"]["history_mode"] == mode
                and not s["temporal_data"]["pose_alignment_used"], "History/pose route changed")
        require(s["temporal_data"]["causal_status"]["future_values_used"] is False,
                "Noncausal state target")
        require(np.array_equal(s["metric_weights"], WEIGHTS), "Official metric weights changed")
    require(not (set(m["train"]["sessions"]) & set(m["validation"]["sessions"])),
            "Train and tune sessions overlap")


def check_arrays(a, summary, manifest, step):
    n = 1998
    shapes = dict(rows=(n,), pred=(n, 6, 2), candidate_id=(n,), d3=(n,),
        shortlist_oracle=(n,), point_l2=(n, 6), error_xy=(n, 6, 2),
        state_pred=(n, 4), state_target=(n, 4), session=(n,), scenario=(n,))
    for key, shape in shapes.items():
        require(key in a and a[key].shape == shape, f"Wrong evaluation array: {key}")
        if key not in ("scenario", "session"):
            require(np.isfinite(a[key]).all(), f"Nonfinite evaluation array: {key}")
    require(a["rows"].dtype.kind in "iu" and len(np.unique(a["rows"])) == n
            and array_sha(a["rows"].astype("<i8")) == TUNE_SHA, "Wrong tune rows/order")
    require(a["candidate_id"].dtype.kind in "iu" and (a["candidate_id"] >= 0).all()
            and (a["candidate_id"] < 1024 * 1024).all(), "Malformed candidate IDs")
    for key in ("pred", "error_xy", "point_l2", "d3", "state_pred", "state_target"):
        require(a[key].dtype == np.float32, f"Expected original FP32 saved field: {key}")
    require(set(a["session"]) == set(manifest["validation"]["sessions"])
            and set(a["scenario"]) == set(manifest["validation"]["scenes"]),
            "Evaluation scene/session population differs from manifest")
    for scene in np.unique(a["scenario"]):
        require((a["scenario"] == scene).sum() == 54
                and len(np.unique(a["session"][a["scenario"] == scene])) == 1,
                "Scene/session row mapping is malformed")
    point = np.linalg.norm(a["error_xy"].astype(np.float64), axis=-1)
    d3 = point @ WEIGHTS
    # GPU vector norm, multiply, summation, and original summary reductions were
    # FP32. Compare within their rounding budget, never silently replace fields.
    require(np.allclose(point, a["point_l2"], rtol=3e-7, atol=2e-7), "Point L2 recomputation differs")
    require(np.allclose(d3, a["d3"], rtol=4e-7, atol=2e-7), "D3 recomputation differs")
    oracle = a["shortlist_oracle"].astype(np.float64)
    require((oracle >= 0).all() and (oracle <= d3 + 1e-5).all(), "Oracle exceeds selected D3")
    require(summary["step"] == step and summary["n"] == n, "Summary step/population mismatch")
    scalar_checks = {"official_d3": float(a["d3"].mean()),
        "shortlist_oracle_d3": float(a["shortlist_oracle"].mean()),
        "selection_regret": float((a["d3"]-a["shortlist_oracle"]).mean()),
        "three_second_l2": float(a["point_l2"][:, -1].mean())}
    for key, value in scalar_checks.items():
        require(summary[key] == value, f"Original FP32 summary differs from arrays: {key}")
    require(np.array_equal(summary["point_l2"], a["point_l2"].mean(0)), "Six-point summary mismatch")
    require(np.array_equal(summary["state_mae_vx_vy_ax_ay"],
                           np.abs(a["state_pred"]-a["state_target"]).mean(0)), "State MAE summary mismatch")
    for session in np.unique(a["session"]):
        require(summary["session_d3"][str(session)] == float(a["d3"][a["session"] == session].mean()),
                "Session D3 summary mismatch")
    for task in ("occ", "lane"):
        require(np.isfinite(summary[task+"_iou"]) and 0 <= summary[task+"_iou"] <= 1, "Malformed raster IoU")
    return dict(point=point, d3=d3, oracle=oracle, regret=d3-oracle,
                state_error=np.abs(a["state_pred"].astype(np.float64)-a["state_target"].astype(np.float64)),
                reconstructed_gt=a["pred"].astype(np.float64)-a["error_xy"].astype(np.float64),
                max_point_roundoff=float(np.abs(point-a["point_l2"]).max()),
                max_d3_roundoff=float(np.abs(d3-a["d3"]).max()))


def bootstrap_design(sessions, draws=20000, seed=0):
    names, cluster = np.unique(sessions, return_inverse=True)
    counts = np.bincount(cluster, minlength=len(names))
    chosen = np.random.default_rng(seed).integers(0, len(names), size=(draws, len(names)))
    multiplicity = np.zeros((draws, len(names)), np.int64)
    np.add.at(multiplicity, (np.arange(draws)[:, None], chosen), 1)
    return dict(names=names, cluster=cluster, counts=counts, multiplicity=multiplicity,
                denominator=multiplicity @ counts, draws=draws, seed=seed,
                draw_sha256=array_sha(chosen.astype("<i8")))


def interval(values, reference, design):
    x, ref = np.asarray(values, np.float64), np.asarray(reference, np.float64)
    require(x.shape == ref.shape == design["cluster"].shape, "Bootstrap row shape mismatch")
    require(np.isfinite(x).all() and np.isfinite(ref).all(), "Nonfinite bootstrap inputs")
    def distribution(y):
        sums = np.bincount(design["cluster"], weights=y, minlength=len(design["names"]))
        return (design["multiplicity"] @ sums) / design["denominator"]
    return dict(mean=float(x.mean()), ci95=np.quantile(distribution(x), [.025, .975]).tolist(),
                delta=float((x-ref).mean()),
                paired_delta_ci95=np.quantile(distribution(x-ref), [.025, .975]).tolist())


def numeric_metrics(checked):
    out = {key: checked[key] for key in ("d3", "oracle", "regret")}
    out["three_second_l2"] = checked["point"][:, -1]
    out["three_second_prefix_ade"] = checked["point"].mean(-1)
    for index, field in enumerate(STATE_FIELDS):
        out["state_mae_"+field] = checked["state_error"][:, index]
    return out


def check_logs(logs, terminal):
    last = min(log[-1]["step"] for log in logs.values())
    require(last >= 1, "No training steps saved")
    wanted = [1] + list(range(10, (2000 if terminal else last)+1, 10))
    common = None
    for arm, log in logs.items():
        require(len({row["step"] for row in log}) == len(log), "Duplicate training log step")
        sampled = [r for r in log if r["step"] <= last] if not terminal else log
        require([r["step"] for r in sampled] == wanted, "Missing or nonmonotonic sampled training steps")
        if terminal:
            require(log[-1]["step"] == 2000, "Missing terminal training step")
        comparable = [{key: row[key] for key in ("step", "epoch", "batch_rows_sha256",
                      "learning_rates", "occ_valid_pixels", "lane_valid_pixels")} for row in sampled]
        if common is None:
            common = comparable
        else:
            require(comparable == common, "Arms differ in sampled training rows/LR/valid-label counts")
        for row in sampled:
            for key in ("loss", "fine_ce", "path_ce", "velocity_ce", "grad_norm", "state_loss", "occ_loss", "lane_loss"):
                require(np.isfinite(row[key]), "Nonfinite training log")
    return dict(matched_saved_steps=wanted, matched_saved_step_count=len(wanted),
        matched_rows_lr_epoch_valid_counts=True,
        matched_sequence_sha256=hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest(),
        proof_scope="step1 and every10 only; full unsaved batch sequence was not logged",
        last_saved_step={arm: log[-1]["step"] for arm, log in logs.items()})


def analyze(runs, preflight=False):
    require(len(runs) == 3, "Pass A, B, C run directories in that order")
    reader = Reader()
    entries, invariant, canonical_arrays = {}, None, None
    planning_fields = ("rows", "pred", "candidate_id", "d3", "shortlist_oracle", "point_l2", "error_xy")
    coordinate_by_id = {}
    for arm, directory in zip(ARM_MODES, map(lambda p: Path(p).resolve(), runs)):
        m = reader.read(directory / "manifest.json")
        check_manifest(m, arm, directory, reader)
        current = normalized_manifest(m)
        if invariant is None:
            invariant = current
        else:
            require(current == invariant, "Arm manifests differ outside the declared interventions")
        initial_summary = reader.read(directory / "eval_000000.json")
        initial = reader.read(directory / "eval_000000.npz", "npz")
        initial_checked = check_arrays(initial, initial_summary, m, 0)
        if canonical_arrays is None:
            canonical_arrays = initial
            initial_reference_summary = initial_summary
            canonical_gt = initial_checked["reconstructed_gt"]
        else:
            for field in planning_fields + ("session", "scenario", "state_target"):
                require(bitwise(initial[field], canonical_arrays[field]), f"Initial arm parity failed: {field}")
            for task in ("occ", "lane"):
                require(initial_summary[task+"_iou"] == initial_reference_summary[task+"_iou"],
                        "Initial raster IoU differs")
        log = reader.read(directory / "train.jsonl", "jsonl")
        e = dict(directory=str(directory), manifest=m, initial=initial, initial_summary=initial_summary,
                 initial_checked=initial_checked, log=log)
        if not preflight:
            receipt = reader.read(directory / "result.json")
            require(receipt["status"] == "completed" and receipt["steps"] == 2000, "Run is incomplete")
            terminal_summary = reader.read(directory / "eval_002000.json")
            require(receipt["initial"] == initial_summary and receipt["terminal"] == terminal_summary,
                    "Run receipt differs from saved evaluation summaries")
            terminal = reader.read(directory / "eval_002000.npz", "npz")
            checked = check_arrays(terminal, terminal_summary, m, 2000)
            for field in ("rows", "session", "scenario", "state_target"):
                require(bitwise(terminal[field], canonical_arrays[field]), f"Terminal row/target population changed: {field}")
            # Inverting an FP32 subtraction cannot reproduce GT bitwise. This
            # tolerance covers rounding at the largest initial errors (~100m).
            require(np.allclose(checked["reconstructed_gt"], canonical_gt, rtol=0, atol=2e-5),
                    "Reconstructed GT differs across initial/terminal arms beyond FP32 rounding")
            checked["max_reconstructed_gt_difference"] = float(np.abs(checked["reconstructed_gt"]-canonical_gt).max())
            e.update(terminal=terminal, terminal_summary=terminal_summary, checked=checked, receipt=receipt)
        for arrays in (initial,) if preflight else (initial, e["terminal"]):
            for cid, xy in zip(arrays["candidate_id"], arrays["pred"]):
                key = int(cid)
                if key in coordinate_by_id:
                    require(bitwise(coordinate_by_id[key], xy), "Same immutable bank ID produced different coordinates")
                else:
                    coordinate_by_id[key] = xy.copy()
        entries[arm] = e
    require(bitwise(entries["B"]["initial"]["state_pred"], entries["C"]["initial"]["state_pred"]),
            "Initial B/C image-only state predictions differ")
    audit = dict(common_provenance_exact=True, source_snapshots_verified=True,
        initial_planning_arrays_bitwise_equal=True, initial_occ_lane_iou_exact=True,
        raster_logit_bitwise_parity="not available: raster predictions were not saved",
        initial_state_A_B_equal=bitwise(entries["A"]["initial"]["state_pred"], entries["B"]["initial"]["state_pred"]),
        initial_state_A_B_difference_allowed=True, initial_state_B_C_bitwise_equal=True,
        train_logs=check_logs({arm: e["log"] for arm, e in entries.items()}, not preflight),
        unique_selected_bank_ids_consistent=len(coordinate_by_id),
        shortlist_oracle_check="saved FP32 oracle checked finite/nonnegative and <= selected D3; full shortlist unavailable",
        gt_check="cross-arm reconstruction within 2e-5m due saved FP32 subtraction; no independent GT file loaded")
    report = dict(schema="sdv2_temporal_cpu_analysis_v1", status="preflight" if preflight else "completed",
        population=dict(rows=1998, scenes=37, sessions=11, rows_sha256=TUNE_SHA,
                        training_rows=54810, training_rows_sha256=TRAIN_SHA),
        audit=audit, common_provenance=invariant,
        initial={arm: e["initial_summary"] for arm, e in entries.items()}, arms={}, comparisons={},
        context_only=dict(previous_raw_status_ce_tune_d3=.13074861450346642,
            interpretation="Different input routes and separate base/head training; not a matched architecture control"),
        limitations=["Repeated tune exploration, one seed, fixed terminal; not blind confirmation.",
            "Bootstrap intervals resample sessions, not training seeds; no multiplicity correction.",
            "A/B differs only in past-image source; B/C adds raw status to shared perception attention.",
            "State predictions are image-only auxiliaries, not planner inputs, even in C.",
            "Raster IoU has only aggregate saved values; no valid bootstrap CI can be reconstructed.",
            "C improving D3 alone proves neither perception utility nor challenge-rule approval.",
            "This bounded 2000-step experiment does not establish an architecture or retraining ceiling."])
    if not preflight:
        design = bootstrap_design(canonical_arrays["session"])
        require(len(design["names"]) == 11, "Expected exactly eleven session clusters")
        report["bootstrap"] = dict(draws=design["draws"], seed=design["seed"],
            draw_sha256=design["draw_sha256"], sessions=design["names"].tolist(),
            session_row_counts=design["counts"].tolist(),
            unit="session with replacement; all rows retained; frame-weighted mean",
            interval="percentile 2.5/97.5; shared paired draws; exploratory, not simultaneous")
        for arm, e in entries.items():
            numeric = numeric_metrics(e["checked"])
            report["arms"][arm] = dict(label=ARM_LABELS[arm], directory=e["directory"],
                metrics={key: interval(values, values, design) for key, values in numeric.items()},
                original_fp32_summary=e["terminal_summary"],
                point_l2_mean=e["checked"]["point"].mean(0).tolist(),
                point_estimate_le_015=bool(numeric["d3"].mean() <= .15),
                roundoff={key: value for key, value in e["checked"].items() if key.startswith("max_")},
                session_d3={str(s): float(numeric["d3"][canonical_arrays["session"] == s].mean()) for s in design["names"]})
        for before, after in (("A", "B"), ("B", "C")):
            source = numeric_metrics(entries[before]["checked"])
            target = numeric_metrics(entries[after]["checked"])
            pair = dict(reference=before, intervention=after, negative_delta_is_better_for_errors=True,
                metrics={key: interval(target[key], source[key], design) for key in source},
                raster_iou_delta={task: entries[after]["terminal_summary"][task+"_iou"]-
                    entries[before]["terminal_summary"][task+"_iou"] for task in ("occ", "lane")},
                raster_iou_ci=None,
                selected_bank_id_changed_fraction=float(np.mean(entries[after]["terminal"]["candidate_id"] !=
                                                                  entries[before]["terminal"]["candidate_id"])),
                session_d3_delta={str(s): float((target["d3"]-source["d3"])[canonical_arrays["session"] == s].mean())
                                  for s in design["names"]})
            report["comparisons"][after+"_minus_"+before] = pair
    reader.verify_unchanged()
    report["input_artifacts_sha256"] = reader.artifacts
    report["analyzer_sha256"] = file_sha(__file__)
    return report


def markdown(report):
    lines = ["# 시간 영상·공통 perception 상태 대조", "",
        "검증 대상은 같은 공개 초기화·학습 행·은행·학습 예산을 사용한 A(현재 front 반복), "
        "B(실제 과거 front .1/.5초), C(B에 공통 perception 상태 추가)다. "
        "원시·예측 상태는 planner와 최종 relative head에 직접 들어가지 않으며 goal은 완성 후보 선택에만 쓴다.", "",
        "세 arm의 source/public/bank/행/parameter 수와 초기 planning 배열의 bitwise 동일성, "
        "초기 occupancy/lane IoU 동일성을 확인했다. A/B 초기 state aux 차이는 허용하고 B/C는 정확히 같았다.", "",
        f"학습 순서 검증은 저장된 {report['audit']['train_logs']['matched_saved_step_count']}개 step의 "
        "row SHA·epoch·LR·유효 label 수가 동일하다는 범위다. 저장하지 않은 모든 batch의 사후 증명은 아니다.", ""]
    if report["status"] == "preflight":
        lines += ["현재는 초기화와 기록된 학습 prefix만 검증했다. Terminal 성능 비교는 아직 수행하지 않았다.", ""]
        return "\n".join(lines)
    lines += ["D3는 저장된 FP32 XY 오차에서 float64로 독립 재계산했다. "
        "원래 GPU/NumPy FP32 집계와의 반올림 차이는 JSON에 별도 보존했다. "
        "Oracle은 전체 후보 좌표가 저장되지 않아 저장된 FP32 값을 사용하고, regret은 재계산 D3에서 이를 뺐다.", "",
        "| Arm | D3 [95% CI] | Shortlist oracle | Regret | 3초 L2 | Occ IoU | Lane IoU |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, e in report["arms"].items():
        m, s = e["metrics"], e["original_fp32_summary"]
        ci = m["d3"]["ci95"]
        lines.append(f"| {arm} | {m['d3']['mean']:.6f} [{ci[0]:.6f}, {ci[1]:.6f}] | "
            f"{m['oracle']['mean']:.6f} | {m['regret']['mean']:.6f} | "
            f"{m['three_second_l2']['mean']:.6f} | {s['occ_iou']:.6f} | {s['lane_iou']:.6f} |")
    lines += ["", "| 비교 | ΔD3 [paired 95% CI] | ΔOracle | ΔRegret | ΔOcc IoU | ΔLane IoU |",
              "|---|---:|---:|---:|---:|---:|"]
    for key, p in report["comparisons"].items():
        m = p["metrics"]
        lo, hi = m["d3"]["paired_delta_ci95"]
        lines.append(f"| {p['intervention']} − {p['reference']} | {m['d3']['delta']:+.6f} "
            f"[{lo:+.6f}, {hi:+.6f}] | {m['oracle']['delta']:+.6f} | {m['regret']['delta']:+.6f} | "
            f"{p['raster_iou_delta']['occ']:+.6f} | {p['raster_iou_delta']['lane']:+.6f} |")
    lines += ["", "오차 지표는 음수 차이가 개선이며 IoU는 양수 차이가 개선이다. "
        "CI는 session 11개를 복원추출한 20,000회(seed 0)의 frame-weighted paired percentile 구간이다. "
        "같은 draws를 두 비교에 공유했다. IoU는 저장된 aggregate 값만 있어 CI를 만들지 않았다.", "",
        "| Arm | 영상 aux vx MAE (m/s) | vy MAE (m/s) | ax MAE (m/s²) | ay MAE (m/s²) |",
        "|---|---:|---:|---:|---:|"]
    for arm, e in report["arms"].items():
        fields = [f"{e['metrics']['state_mae_'+field]['mean']:.6f}" for field in STATE_FIELDS]
        lines.append("| " + arm + " | " + " | ".join(fields) + " |")
    reached = [arm for arm, e in report["arms"].items() if e["point_estimate_le_015"]]
    lines += ["", f"실측 terminal D3 평균이 .15 이하인 arm: {', '.join(reached) if reached else '없음'}. "
        "이는 이 tune 실험의 실측 판정이며 재학습 상한이나 임계치 추정이 아니다.", "",
        "이전 raw-status 모델의 D3 0.130749는 학습 절차와 입력 경로가 다른 참고 결과다. "
        "이번 세 arm에 대한 matched architecture control로 사용하지 않는다.", "",
        "이번 결과는 반복 사용한 tune의 단일 seed·고정 2,000-step 대조다. "
        "CI는 학습 seed 불확실성이나 다중 비교를 보정하지 않는다. C에서 D3만 좋아지면 "
        "perception 유용성 또는 규정 승인까지 입증한 것으로 해석하지 않는다. "
        "state aux 예측은 C에서도 이미지 기반이며 planner 입력으로 전달되지 않는다.", ""]
    return "\n".join(lines)


def self_test():
    class Tests(unittest.TestCase):
        def test_frame_weights_and_constant_paired_shift(self):
            sessions = np.asarray(["a", "a", "a", "b"])
            d = bootstrap_design(sessions, draws=1000)
            x = np.asarray([1., 2., 3., 20.])
            got = interval(x+2., x, d)
            self.assertEqual(got["mean"], 8.5)
            np.testing.assert_allclose(got["paired_delta_ci95"], [2., 2.], rtol=0, atol=0)
            self.assertNotEqual(float(x.mean()), float(np.mean([x[:3].mean(), x[3]])))

        def test_bootstrap_matches_explicit_cluster_expansion(self):
            names = np.asarray(["a", "b", "b", "c", "c", "c"])
            x = np.arange(6.)
            d = bootstrap_design(names, draws=9, seed=4)
            sums = np.bincount(d["cluster"], weights=x)
            efficient = (d["multiplicity"] @ sums) / d["denominator"]
            explicit = [np.concatenate([np.tile(x[names == s], n)
                for s, n in zip(d["names"], counts) if n]).mean() for counts in d["multiplicity"]]
            np.testing.assert_array_equal(efficient, explicit)
            self.assertEqual(d["draw_sha256"], bootstrap_design(names, 9, 4)["draw_sha256"])

        def test_bitwise_catches_signed_zero(self):
            self.assertFalse(bitwise(np.asarray([0.], np.float32), np.asarray([-0.], np.float32)))

        def test_reader_records_and_rejects_changed_source(self):
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)/"x.json"
                p.write_text('{"x":1}')
                r = Reader()
                self.assertEqual(r.read(p), {"x": 1})
                p.write_text('{"x":2}')
                with self.assertRaises(ValueError):
                    r.verify_unchanged()

        def test_logs_reject_changed_rows_and_allow_different_progress(self):
            def row(step):
                return dict(step=step, epoch=0, batch_rows_sha256=str(step), learning_rates=[.1],
                    occ_valid_pixels=2, lane_valid_pixels=3, loss=1., fine_ce=1., path_ce=1.,
                    velocity_ce=1., grad_norm=1., state_loss=1., occ_loss=1., lane_loss=1.)
            logs = {"A": [row(1), row(10)], "B": [row(1), row(10), row(20)], "C": [row(1), row(10)]}
            self.assertEqual(check_logs(logs, False)["matched_saved_step_count"], 2)
            logs["B"][1]["batch_rows_sha256"] = "different"
            with self.assertRaises(ValueError):
                check_logs(logs, False)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return result.wasSuccessful()


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--runs", nargs=3, metavar=("A", "B", "C"))
    p.add_argument("--output")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not a.runs or not a.output:
        p.error("--runs A B C and --output are required")
    report = analyze(a.runs, preflight=a.preflight)
    directory = Path(a.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory/"aggregate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    (directory/"RESULTS_KO.md").write_text(markdown(report))
    receipt = dict(analyzer_sha256=file_sha(__file__), status=report["status"],
        output_sha256={name:file_sha(directory/name) for name in ("aggregate.json", "RESULTS_KO.md")})
    (directory/"analysis_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(dict(status=report["status"], output=str(directory),
        d3={arm:e["metrics"]["d3"]["mean"] for arm,e in report["arms"].items()}), ensure_ascii=False))


if __name__ == "__main__":
    main()
