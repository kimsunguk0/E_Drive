"""Audit and aggregate three frozen status-intervention shards on CPU.

No model is loaded and no held population is accessed. Cluster bootstrap draws
11 sessions with replacement; each selected session retains all of its rows.
Each draw is therefore frame weighted, and identical draws are used for every
condition and its paired baseline delta. Confidence intervals are exploratory
on the reused tune set, not simultaneous intervals or evidence of approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from status_perturbations import Condition, build_conditions, perturb_status

TUNE_SHA = "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88"
EMPIRICAL_B0_SHA = "e2e5d55790c0a8efd4637939e1f896c95620ae9671e845439083cfc35fad273e"
WEIGHTS = np.asarray([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36
MODES = ("coarse", "fine", "both")
MODE_NAMES = {"coarse": "base 전체 status(후보 축소 포함)",
              "fine": "최종 CE head status만", "both": "base와 CE head 모두"}


def require(value, message):
    if not value:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path, artifacts):
    path = Path(path).resolve()
    before = file_sha(path)
    result = json.loads(path.read_text())
    require(file_sha(path) == before, f"JSON changed while reading: {path}")
    artifacts[str(path)] = before
    return result


def read_npz(path, artifacts):
    path = Path(path).resolve()
    before = file_sha(path)
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    require(file_sha(path) == before, f"NPZ changed while reading: {path}")
    artifacts[str(path)] = before
    return result


def bootstrap_design(sessions, draws=20000, seed=0):
    names, row_cluster = np.unique(sessions, return_inverse=True)
    counts = np.bincount(row_cluster, minlength=len(names))
    selected = np.random.default_rng(seed).integers(0, len(names), size=(draws, len(names)))
    multiplicity = np.zeros((draws, len(names)), dtype=np.int64)
    np.add.at(multiplicity, (np.arange(draws)[:, None], selected), 1)
    denominator = multiplicity @ counts
    return dict(names=names, row_cluster=row_cluster, counts=counts,
                multiplicity=multiplicity, denominator=denominator,
                draw_sha256=array_sha(selected.astype("<i8")), draws=draws, seed=seed)


def metric_interval(values, reference, design):
    values, reference = np.asarray(values, np.float64), np.asarray(reference, np.float64)
    require(values.shape == reference.shape == design["row_cluster"].shape, "Metric row shape mismatch")
    require(np.isfinite(values).all() and np.isfinite(reference).all(), "Nonfinite bootstrap values")
    sums = np.bincount(design["row_cluster"], weights=values, minlength=len(design["names"]))
    differences = np.bincount(design["row_cluster"], weights=values-reference,
                             minlength=len(design["names"]))
    sampled = (design["multiplicity"] @ sums) / design["denominator"]
    sampled_delta = (design["multiplicity"] @ differences) / design["denominator"]
    return dict(mean=float(values.mean()), ci95=np.quantile(sampled, [.025, .975]).tolist(),
                delta=float((values-reference).mean()),
                paired_delta_ci95=np.quantile(sampled_delta, [.025, .975]).tolist())


def check_arrays(arrays, rows, reference_gt=None):
    n = len(rows)
    shapes = {"rows": (n,), "pred": (n, 6, 2), "candidate_id": (n,),
              "point_l2": (n, 6), "error_xy": (n, 6, 2), "d3": (n,),
              "shortlist_oracle": (n,), "oracle_candidate_id": (n,),
              "baseline_winner_present": (n,), "baseline_candidates_retained": (n,)}
    for key, shape in shapes.items():
        require(key in arrays and arrays[key].shape == shape, f"Malformed output field: {key}")
        require(np.isfinite(arrays[key]).all(), f"Nonfinite output field: {key}")
    require(np.array_equal(arrays["rows"], rows), "Output row identities differ")
    for key in ("rows", "candidate_id", "oracle_candidate_id", "baseline_candidates_retained"):
        require(arrays[key].dtype.kind in "iu", f"Expected integer field: {key}")
    require((arrays["candidate_id"] >= 0).all(), "Negative candidate ID")
    require(arrays["baseline_winner_present"].dtype == np.bool_, "Expected boolean retention mask")
    require(((arrays["baseline_candidates_retained"] >= 0)
             & (arrays["baseline_candidates_retained"] <= 200)).all(), "Invalid retained candidate count")
    point = np.linalg.norm(arrays["error_xy"].astype(np.float64), axis=-1)
    d3 = (point * WEIGHTS).sum(-1)
    require(np.allclose(point, arrays["point_l2"], rtol=0, atol=1e-12), "Point L2 recomputation differs")
    require(np.allclose(d3, arrays["d3"], rtol=0, atol=1e-12), "Official D3 recomputation differs")
    require((arrays["shortlist_oracle"] >= 0).all()
            and (arrays["shortlist_oracle"] <= arrays["d3"]+1e-10).all(), "Oracle exceeds selected D3")
    gt = arrays["pred"].astype(np.float64) - arrays["error_xy"].astype(np.float64)
    if reference_gt is not None:
        require(np.allclose(gt, reference_gt, rtol=0, atol=1e-12), "Conditions used different GT coordinates")
    return gt


def condition_title(name, definition):
    kind = definition["kind"]
    if kind == "zero_status":
        return "상태 전체 0"
    if kind == "bias":
        if definition["vx_bias"]:
            return f"vx bias {definition['vx_bias']:+g} m/s"
        return f"ax bias {definition['ax_bias']:+g} m/s²"
    if kind == "session_ou":
        return f"상관오차 목표 MAE vx={definition['vx_mae']:g}, ax={definition['ax_mae']:g}"
    if kind == "measured_replacement":
        return "P7 b0 영상 추정 status 치환"
    return name


def error_statistics(changed, original):
    error = changed[:, 4:8].astype(np.float64) - original[:, 4:8].astype(np.float64)
    return dict(bias=error.mean(0).tolist(), mae=np.abs(error).mean(0).tolist(),
                rmse=np.sqrt((error**2).mean(0)).tolist(),
                p95_absolute=np.quantile(np.abs(error), .95, axis=0).tolist())


def metrics(arrays, baseline, metadata, design):
    result = {}
    for name, values, ref in (
        ("d3", arrays["d3"], baseline["d3"]),
        ("shortlist_oracle", arrays["shortlist_oracle"], baseline["shortlist_oracle"]),
        ("selection_regret", arrays["d3"]-arrays["shortlist_oracle"], baseline["d3"]-baseline["shortlist_oracle"]),
        ("three_second_l2", arrays["point_l2"][:, -1], baseline["point_l2"][:, -1])):
        result[name] = metric_interval(values, ref, design)
    result.update(point_l2_mean=arrays["point_l2"].mean(0).tolist(),
        changed_winner_fraction=float(np.mean(arrays["candidate_id"] != baseline["candidate_id"])),
        baseline_winner_present_fraction=float(arrays["baseline_winner_present"].mean()),
        baseline_candidates_retained_mean=float(arrays["baseline_candidates_retained"].mean()),
        oracle_over_015_fraction=float(np.mean(arrays["shortlist_oracle"] > .15)),
        point_estimate_le_015=bool(arrays["d3"].mean() <= .15),
        ci95_upper_le_015=bool(result["d3"]["ci95"][1] <= .15), sessions={})
    for session in design["names"]:
        selected = metadata["session"] == session
        result["sessions"][str(session)] = dict(n=int(selected.sum()),
            d3=float(arrays["d3"][selected].mean()),
            delta_d3=float((arrays["d3"][selected]-baseline["d3"][selected]).mean()),
            oracle=float(arrays["shortlist_oracle"][selected].mean()))
    return result


def measured_crossings(conditions, baseline):
    """Describe actual neighboring tested points; never interpolate a threshold."""
    output = []
    for field in ("vx", "ax"):
        for sign in (-1, 1):
            selected = [c for c in conditions if c["definition"]["kind"] == "bias"
                        and c["definition"][field+"_bias"] * sign > 0]
            selected.sort(key=lambda c: abs(c["definition"][field+"_bias"]))
            for mode in MODES:
                points = [dict(name="baseline", signed_bias=0., d3=baseline["d3"]["mean"],
                               le_015=baseline["point_estimate_le_015"])]
                points.extend(dict(name=c["name"], signed_bias=c["definition"][field+"_bias"],
                    d3=c["modes"][mode]["d3"]["mean"], le_015=c["modes"][mode]["point_estimate_le_015"])
                    for c in selected)
                changes = [dict(from_point=a, to_point=b) for a, b in zip(points, points[1:])
                           if a["le_015"] != b["le_015"]]
                output.append(dict(family="signed_bias", field=field, sign=sign, mode=mode,
                    tested_points=points, adjacent_measured_status_changes=changes))
    for fixed_ax in (.1, .3):
        selected = [c for c in conditions if c["definition"]["kind"] == "session_ou"
                    and c["definition"]["ax_mae"] == fixed_ax]
        selected.sort(key=lambda c: c["definition"]["vx_mae"])
        for mode in MODES:
            points = [dict(name=c["name"], target_vx_mae=c["definition"]["vx_mae"],
                actual_vx_mae=c["actual_error"]["mae"][0], d3=c["modes"][mode]["d3"]["mean"],
                le_015=c["modes"][mode]["point_estimate_le_015"]) for c in selected]
            changes = [dict(from_point=a, to_point=b) for a, b in zip(points, points[1:])
                       if a["le_015"] != b["le_015"]]
            output.append(dict(family="correlated_error", fixed_target_ax_mae=fixed_ax, mode=mode,
                tested_points=points, adjacent_measured_status_changes=changes))
    return output


def analyze(shards, empirical_reference=None):
    require(len(shards) == 3, "Exactly three completed full shards are required")
    artifacts, definitions, all_arrays, all_inputs, result_receipts = {}, {}, {}, {}, []
    empirical_path = Path(empirical_reference) if empirical_reference else Path(__file__).with_name("empirical") / "empirical_p7_b0_status.npz"
    empirical = read_npz(empirical_path, artifacts)
    require(artifacts[str(empirical_path.resolve())] == EMPIRICAL_B0_SHA, "Empirical reference is not the fixed P7 b0 artifact")
    metadata = baseline = gt = common = None
    expected_synthetic = {c.name: c.to_dict() for c in build_conditions() if c.kind != "baseline"}
    selected_coordinates = {}
    for directory in map(lambda p: Path(p).resolve(), shards):
        receipt = read_json(directory / "result.json", artifacts)
        require(receipt["schema"] == "sdv2_status_sensitivity_v1" and receipt["status"] == "completed", "Incomplete/wrong shard")
        require(receipt["n"] == 1998 and receipt["rows_sha256"] == TUNE_SHA
                and not receipt["canary"] and not receipt["fit_performed"], "Wrong population or execution scope")
        require(receipt["arguments"]["batch"] == 8 and receipt["arguments"]["shards"] == 3,
                "Expected batch8, three-shard experiment")
        require(receipt["backbone_reuse_exact_parity_batches"] == 2, "Backbone reuse parity was not certified")
        require(receipt["baseline_reference"]["exact_xy_ids"], "Original head predictions were not reproduced")
        invariant = {k: receipt[k] for k in ("base", "head", "source_sha256", "numpy", "torch",
                      "precision", "perturbation_protocol", "time_contract", "intervention_modes")}
        if common is None:
            common = invariant
        else:
            require(invariant == common, "Shards use different model/source/input contracts")
        inputs = read_npz(directory / "inputs.npz", artifacts)
        names = ("rows", "scenario", "session", "nominal_time_s", "original_status8")
        require(all(k in inputs for k in names), "Input metadata missing")
        require(inputs["rows"].dtype == np.int64 and len(np.unique(inputs["rows"])) == 1998
                and array_sha(inputs["rows"].astype("<i8")) == TUNE_SHA, "Original tune row hash differs")
        require(inputs["original_status8"].shape == (1998, 8)
                and np.isfinite(inputs["original_status8"]).all()
                and (inputs["original_status8"][:, :4] == 0).all(), "Invalid original state")
        if metadata is None:
            metadata = {k: inputs[k] for k in names}
        else:
            for key in names:
                require(np.array_equal(inputs[key], metadata[key]), f"Shard metadata differs: {key}")
        original = read_npz(directory / "baseline.npz", artifacts)
        local_gt = check_arrays(original, metadata["rows"], gt)
        if baseline is None:
            baseline, gt = original, local_gt
        else:
            for key in ("rows", "pred", "candidate_id"):
                require(original[key].dtype == baseline[key].dtype
                        and original[key].tobytes() == baseline[key].tobytes(), f"Baseline is not bitwise identical: {key}")
            for key in ("d3", "shortlist_oracle", "point_l2", "error_xy"):
                require(np.allclose(original[key], baseline[key], rtol=0, atol=1e-12), f"Baseline metric differs: {key}")
        require(set(receipt["summaries"]) == {"baseline"} | {f"{name}__{mode}" for name in receipt["definitions"] for mode in MODES},
                "Reported condition/mode coverage differs")
        for name, definition in receipt["definitions"].items():
            require(name not in definitions, "Repeated condition across shards")
            changed = inputs["status__"+name]
            require(changed.shape == (1998, 8) and changed.dtype == np.float32
                    and np.isfinite(changed).all() and (changed[:, :4] == 0).all(), "Invalid perturbed status")
            if name in expected_synthetic:
                require(definition == expected_synthetic[name], "Synthetic condition differs from frozen protocol")
                rebuilt = perturb_status(metadata["original_status8"], metadata["session"], metadata["nominal_time_s"],
                                         Condition(**definition), seed=20260910)
                require(rebuilt.tobytes() == changed.tobytes(), "Synthetic status cannot be reproduced from metadata alone")
            else:
                require(name == "empirical_p7_status" and definition["kind"] == "measured_replacement", "Unknown extra condition")
                require(definition["sha256"] == EMPIRICAL_B0_SHA
                        and np.array_equal(empirical["rows"], metadata["rows"])
                        and empirical["status8"].dtype == changed.dtype
                        and empirical["status8"].tobytes() == changed.tobytes(),
                        "Empirical intervention is not the literal fixed P7 b0 replacement")
            definitions[name], all_inputs[name] = definition, changed
            for mode in MODES:
                key = f"{name}__{mode}"
                arrays = read_npz(directory / (key+".npz"), artifacts)
                check_arrays(arrays, metadata["rows"], gt)
                require(abs(float(arrays["d3"].mean())-receipt["summaries"][key]["d3"]) < 1e-12,
                        "NPZ and reported D3 differ")
                if mode == "fine":
                    require(np.array_equal(arrays["shortlist_oracle"], baseline["shortlist_oracle"])
                            and (arrays["baseline_candidates_retained"] == 200).all(), "Fine-only shortlist changed")
                for candidate_id, xy in zip(arrays["candidate_id"], arrays["pred"]):
                    identifier = int(candidate_id)
                    if identifier in selected_coordinates:
                        require(np.array_equal(selected_coordinates[identifier], xy), "One bank ID produced different coordinates")
                    else:
                        selected_coordinates[identifier] = xy.copy()
                all_arrays[key] = arrays
            actual = error_statistics(changed, metadata["original_status8"])
            for stat in ("bias", "mae", "rmse", "p95_absolute"):
                require(np.allclose(actual[stat], receipt["error_statistics_vx_vy_ax_ay"][name][stat], rtol=0, atol=1e-12),
                        "Stored and recomputed state-error statistics differ")
        result_receipts.append(dict(directory=str(directory), result_sha256=artifacts[str(directory/"result.json")],
                                    shard=receipt["arguments"]["shard"], physical_gpu=receipt["physical_gpu"]))
    require({r["shard"] for r in result_receipts} == {0, 1, 2}, "Shard IDs must be exactly 0,1,2")
    require(set(definitions) == set(expected_synthetic) | {"empirical_p7_status"}, "Expected 23 synthetic and one empirical intervention")
    require(len(np.unique(metadata["scenario"])) == 37 and len(np.unique(metadata["session"])) == 11,
            "Expected 37 original tune scenes and 11 sessions")
    design = bootstrap_design(metadata["session"])
    baseline_summary = metrics(baseline, baseline, metadata, design)
    conditions = []
    for name in [*expected_synthetic, "empirical_p7_status"]:
        definition = definitions[name]
        conditions.append(dict(name=name, title=condition_title(name, definition), definition=definition,
            actual_error=error_statistics(all_inputs[name], metadata["original_status8"]),
            modes={mode: metrics(all_arrays[f"{name}__{mode}"], baseline, metadata, design) for mode in MODES}))
    # Preserve transfer/read integrity; do not silently analyze a changing run.
    for path, expected in artifacts.items():
        require(file_sha(path) == expected, f"Input changed during analysis: {path}")
    return dict(schema="sdv2_status_sensitivity_analysis_v1", status="completed", n=1998,
        rows_sha256=TUNE_SHA, sessions=11, scenes=37, artifacts=artifacts, shards=result_receipts,
        source_sha256=file_sha(__file__), numpy=np.__version__, provenance=common,
        audit=dict(baseline_rows_predictions_ids_bitwise_equal=True, independent_d3_recomputed=True,
                   identical_gt_all_interventions=True, synthetic_errors_reproduced_without_labels=True,
                   empirical_p7_b0_literal_replacement_verified=True,
                   selected_id_coordinate_consistency=True, full_conditions=24, modes_per_condition=3),
        bootstrap=dict(draws=design["draws"], seed=design["seed"], draw_sha256=design["draw_sha256"],
            session_order=design["names"].tolist(), session_row_counts=design["counts"].tolist(),
            resampling="11 sessions with replacement; all rows kept with session multiplicity; frame weighted",
            intervals="percentile 95%; same draws for all paired deltas; exploratory, not simultaneous"),
        interpretation=dict(coarse=MODE_NAMES["coarse"]+"; 최종 base score도 변하므로 순수 pruning 인과효과가 아님",
            fine=MODE_NAMES["fine"], both=MODE_NAMES["both"],
            target="D3<=.15 is classified only at measured points; no interpolated tolerable-error threshold",
            scope="Frozen input intervention on reused original tune; not a retrained image-student performance bound",
            synthetic="Population marginal MAE; no sample centering or label-dependent fitting",
            held_population_evaluated=False, fit_performed=False),
        baseline=baseline_summary, conditions=conditions,
        measured_015_status_changes=measured_crossings(conditions, baseline_summary))


def markdown(result):
    base = result["baseline"]
    out = ["# SDV2 상태 오차 민감도: 고정 모델 개입 결과", "",
        f"기존 tune **1,998행·37 scene·11 session**에서 baseline D3는 **{base['d3']['mean']:.8f} m**입니다.",
        f"Baseline 95% 세션 bootstrap CI: [{base['d3']['ci95'][0]:.5f}, {base['d3']['ci95'][1]:.5f}].", "",
        "가중치는 `[11,11,5,5,2,2]/36`입니다. 11개 세션을 복원 추출하고 해당 세션의 모든 행을 함께 가져와 프레임 가중 평균을 계산했습니다. 모든 조건에 동일한 20,000개 draw(seed=0)를 사용했습니다.", "",
        "`base 전체`는 status가 들어가는 원모델 전체를 뜻합니다. 후보 축소와 downstream feature·base score가 함께 바뀌므로 순수 pruning의 인과효과가 아닙니다. `head만`은 원래 후보·base score를 고정하고 최종 CE head 상태만 바꿉니다. `둘 다`는 동일한 오차를 두 입력에 적용합니다.", "",
        "영상 학생을 재학습한 결과나 그 성능 상한이 아닙니다. 반복 사용한 tune의 탐색적 민감도이며, 신뢰구간은 다중 비교를 보정하지 않았습니다. 영상 기여·규정 승인·held 성능을 새로 판단하지 않습니다.", "",
        "## 실제 상태 오차와 결과", "",
        "아래 실제 MAE/bias는 입력 상태 차이에서 다시 계산했습니다. 상관오차의 목표 MAE는 모집단 기대값이므로 11세션 표본에서 정확히 일치하도록 정규화하지 않았습니다. 고정 bias는 부호를 유지했습니다. 단위는 vx=m/s, ax=m/s²입니다.", "",
        "| 조건 | 실제 vx MAE / bias | 실제 ax MAE / bias | 개입 | D3 | ΔD3 [paired 95% CI] | shortlist oracle | regret | 3초 L2 | D3≤.15 |",
        "|---|---:|---:|---|---:|---|---:|---:|---:|---|"]
    short = {"coarse": "base 전체", "fine": "head만", "both": "둘 다"}
    for c in result["conditions"]:
        e = c["actual_error"]
        for mode in MODES:
            m = c["modes"][mode]; interval = m["d3"]["paired_delta_ci95"]
            out.append(f"| {c['title']} | {e['mae'][0]:.4f} / {e['bias'][0]:+.4f} | {e['mae'][2]:.4f} / {e['bias'][2]:+.4f} | {short[mode]} | {m['d3']['mean']:.5f} | {m['d3']['delta']:+.5f} [{interval[0]:+.5f}, {interval[1]:+.5f}] | {m['shortlist_oracle']['mean']:.5f} | {m['selection_regret']['mean']:.5f} | {m['three_second_l2']['mean']:.5f} | {'통과' if m['point_estimate_le_015'] else '미달'} |")
    out += ["", "D3·oracle·regret·3초 L2 각각의 95% CI 및 paired delta CI, 6개 시점 L2, 후보 유지율, 세션별 집계는 aggregate.json에 함께 저장했습니다.", "", "## .15 통과/실패가 바뀐 실제 인접 측정점", "",
            "이는 아래 두 측정점 사이의 관측 상태 변화입니다. 보간 임계값, 단조 관계 또는 구간 전체의 통과를 주장하지 않습니다.", ""]
    found = False
    for group in result["measured_015_status_changes"]:
        for crossing in group["adjacent_measured_status_changes"]:
            found = True; a, b = crossing["from_point"], crossing["to_point"]
            out.append(f"- {short[group['mode']]}: `{a['name']}` D3 {a['d3']:.5f} ({'통과' if a['le_015'] else '미달'}) → `{b['name']}` D3 {b['d3']:.5f} ({'통과' if b['le_015'] else '미달'}).")
    if not found:
        out.append("- 검사한 bias/상관오차 계열의 인접 측정점 사이에서 통과/실패 변화가 없었습니다.")
    empirical = next(c for c in result["conditions"] if c["name"] == "empirical_p7_status")
    out += ["", "## P7 b0 영상 상태의 실제 치환", "",
        f"vx/ax 실제 MAE는 **{empirical['actual_error']['mae'][0]:.5f} m/s / {empirical['actual_error']['mae'][2]:.5f} m/s²**입니다. 이 조건은 vx/ax만이 아니라 P7의 vx/vy/ax/ay 네 값을 그대로 치환합니다.", ""]
    for mode in MODES:
        m=empirical["modes"][mode]
        out.append(f"- {short[mode]}: D3 **{m['d3']['mean']:.5f}**, oracle {m['shortlist_oracle']['mean']:.5f}, regret {m['selection_regret']['mean']:.5f}, baseline 후보 유지 {m['baseline_candidates_retained_mean']:.1f}/200.")
    out += ["", "기존 추정기의 오차가 이 고정 모델에 주는 영향을 보여줍니다. P7의 라벨·시간 정의 차이와 학습된 모델 간 분포 차이를 포함하므로, 새 영상 추정기를 재학습했을 때의 도달 성능을 확정하지 않습니다.", "", "## 무결성", "",
        "- 세 shard의 baseline rows/pred/candidate_id가 bitwise 동일하고 동일 GT를 사용했습니다.",
        "- error_xy에서 point L2 및 공식 D3를 독립 재계산했습니다.",
        "- 합성 교란 입력을 session/time/고정 seed만으로 재생성해 비교했습니다.",
        "- empirical 치환을 고정 P7 b0 원본 SHA 및 status8 bytes와 직접 대조했습니다.",
        "- 모든 읽은 result/NPZ의 SHA와 모델·실행 source·bootstrap draw SHA를 aggregate.json에 기록했습니다.",
        "- 추가 학습·GPU 추론·held 평가를 수행하지 않았습니다.", ""]
    return "\n".join(out)


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"coarse": "#3870b2", "fine": "#df8534", "both": "#3b8e66"}
    labels = {"coarse": "Base input (includes pruning)", "fine": "CE head only", "both": "Both inputs"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, field, unit in ((axes[0, 0], "vx", "m/s"), (axes[0, 1], "ax", "m/s²")):
        selected = [c for c in result["conditions"] if c["definition"]["kind"] == "bias" and c["definition"][field+"_bias"]]
        selected.sort(key=lambda c: c["definition"][field+"_bias"])
        for mode in MODES:
            values = [(c["definition"][field+"_bias"], c["modes"][mode]["d3"]["mean"]) for c in selected]
            values.append((0., result["baseline"]["d3"]["mean"])); values.sort()
            ax.plot(*zip(*values), "o-", color=colors[mode], label=labels[mode])
        ax.set(xlabel=f"Signed {field} bias ({unit})", ylabel="D3 (m)", title=f"{field}: measured bias points")
    ax=axes[1, 0]
    for ax_mae, style in ((.1, "-"), (.3, "--")):
        selected=[c for c in result["conditions"] if c["definition"]["kind"] == "session_ou" and c["definition"]["ax_mae"] == ax_mae]
        selected.sort(key=lambda c:c["actual_error"]["mae"][0])
        for mode in MODES:
            ax.plot([c["actual_error"]["mae"][0] for c in selected],
                    [c["modes"][mode]["d3"]["mean"] for c in selected], "o"+style,
                    color=colors[mode], label=f"{labels[mode]}; target ax MAE={ax_mae}")
    ax.set(xlabel="Actual vx MAE (m/s)", ylabel="D3 (m)", title="Session + temporal correlated error")
    empirical=next(c for c in result["conditions"] if c["name"] == "empirical_p7_status")
    values=[result["baseline"], *[empirical["modes"][mode] for mode in MODES]]
    means=np.asarray([v["d3"]["mean"] for v in values]); ci=np.asarray([v["d3"]["ci95"] for v in values])
    # Percentile intervals need not contain the estimate; draw their endpoints directly.
    ax=axes[1,1]; positions=np.arange(4)
    ax.bar(positions,means,color=["#808080", *[colors[m] for m in MODES]],alpha=.8)
    ax.vlines(positions,ci[:,0],ci[:,1],color="black",linewidth=1.3)
    ax.scatter(positions,means,color="black",s=15)
    ax.set(xticks=positions,xticklabels=["Original", "Base input", "CE head", "Both"],
           ylabel="D3 (m)",title="Actual P7 b0 status replacement (95% CI)")
    for ax in axes.flat:
        ax.axhline(.15,color="#ab3043",linestyle=":",linewidth=1.5)
        ax.grid(axis="y",alpha=.2)
    axes[0,0].legend(fontsize=8)
    axes[1,0].legend(fontsize=6.5)
    fig.suptitle("Frozen-model interventions on original tune; connecting lines do not infer thresholds",fontsize=12)
    fig.savefig(path,dpi=170)
    plt.close(fig)


def self_test():
    sessions=np.array(["a"]+["b"]*4+["c"]*7)
    values=np.array([.1]+[.2]*4+[.3]*7)
    design=bootstrap_design(sessions)
    result=metric_interval(values+.07,values,design)
    require(abs(result["mean"]-float((values+.07).mean())) < 1e-14, "Frame-weighted estimate failed")
    require(abs(result["mean"]-float(np.mean([.1,.2,.3])+.07)) > .01, "Accidentally session averaged")
    require(np.allclose(result["paired_delta_ci95"], [.07,.07], rtol=0,atol=1e-14), "Pairing failed")
    again=bootstrap_design(sessions)
    require(again["draw_sha256"] == design["draw_sha256"], "Bootstrap not deterministic")
    for i in (0, 17, 19999):
        expanded=np.repeat(np.arange(3),design["multiplicity"][i])
        session_values=[values[sessions==name] for name in design["names"]]
        direct=np.concatenate([session_values[j] for j in expanded]).mean()
        sums=np.bincount(design["row_cluster"],weights=values)
        matrix=(design["multiplicity"][i]@sums)/design["denominator"][i]
        require(abs(direct-matrix)<1e-14,"Cluster multiplicity/frame weighting incorrect")
    return {"status":"PASS","checks":["unequal cluster sizes frame weighted", "constant paired delta CI",
            "deterministic shared draws", "explicit resampled-row equality"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--shards",nargs=3)
    parser.add_argument("--output")
    parser.add_argument("--empirical-reference", help="Fixed P7 b0 NPZ; defaults to adjacent empirical/ directory")
    parser.add_argument("--no-plot",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()));return
    require(args.shards and args.output,"--shards and --output are required")
    output=Path(args.output).resolve()
    require(not output.exists(),"Analysis output is immutable; choose a new directory")
    result=analyze(args.shards,args.empirical_reference)
    output.mkdir(parents=True)
    (output/"aggregate.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    (output/"RESULTS_KO.md").write_text(markdown(result))
    if not args.no_plot:
        plot(result,output/"status_sensitivity.png")
    receipt={"status":"completed","source_sha256":file_sha(__file__),
             "artifacts":{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}}
    (output/"analysis_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    empirical=next(c for c in result["conditions"] if c["name"]=="empirical_p7_status")
    print(json.dumps(dict(status="completed",output=str(output),baseline_d3=result["baseline"]["d3"]["mean"],
                         empirical_d3={m:empirical["modes"][m]["d3"]["mean"] for m in MODES})))


if __name__ == "__main__":
    main()
