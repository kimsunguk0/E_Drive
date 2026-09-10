"""Supplementary CPU axis-error diagnostics from already saved predictions.

This is not an additive decomposition of Euclidean L2 or official D3. Absolute
axis errors have units metres. Squared-error energy fraction is a separate,
unweighted sum over all samples and six horizons; it is neither a probability
nor a fraction of the official D3 score. Oracle/regret analysis remains primary.

CLI accepts frozen status-intervention FP64 error arrays and temporal terminal
FP32 error arrays. Neither source format needs model loading or inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

WEIGHTS = np.asarray([11, 11, 5, 5, 2, 2], np.float64) / 36.
TUNE_SHA = "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88"
HORIZONS = [.5, 1., 1.5, 2., 2.5, 3.]


def require(value, message):
    if not value:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def axis_error_summary(error_xy):
    error = np.asarray(error_xy, np.float64)
    require(error.ndim == 3 and error.shape[1:] == (6, 2) and len(error) > 0,
            "Expected nonempty error_xy[N,6,2]")
    require(np.isfinite(error).all(), "Nonfinite XY errors")
    absolute = np.abs(error).mean(0)
    point_l2 = np.linalg.norm(error, axis=-1)
    energy = np.square(error).sum(0)
    total = float(energy.sum())
    energy_fraction = float(energy[:, 0].sum()/total) if total else None
    per_horizon = []
    for index, seconds in enumerate(HORIZONS):
        denominator = float(energy[index].sum())
        per_horizon.append(dict(seconds=seconds,
            mean_abs_ex_m=float(absolute[index, 0]), mean_abs_ey_m=float(absolute[index, 1]),
            mean_euclidean_l2_m=float(point_l2[:, index].mean()),
            mean_ex_m=float(error[:, index, 0].mean()), mean_ey_m=float(error[:, index, 1].mean()),
            longitudinal_squared_error_energy_fraction=float(energy[index, 0]/denominator) if denominator else None))
    return dict(n=len(error), horizons=per_horizon,
        official_d3_recomputed_m=float((point_l2 @ WEIGHTS).mean()),
        weighted_mean_abs_ex_m=float(WEIGHTS @ absolute[:, 0]),
        weighted_mean_abs_ey_m=float(WEIGHTS @ absolute[:, 1]),
        longitudinal_squared_error_energy_fraction=energy_fraction,
        squared_error_energy_sum_x_m2=float(energy[:, 0].sum()),
        squared_error_energy_sum_y_m2=float(energy[:, 1].sum()),
        energy_aggregation="sum over samples and six horizons, no temporal weights",
        weighted_abs_definition="mean_n sum_t w_t abs(error_axis[n,t]); metres",
        energy_fraction_definition="sum_n,t(ex^2) / sum_n,t(ex^2 + ey^2); null if denominator=0",
        interpretation="Separate axis error/energy diagnostics; not additive Euclidean contributions or probabilities")


def analyze(inputs):
    report = dict(schema="adcl_axis_error_diagnostics_v1", status="completed",
        official_time_weights=WEIGHTS.tolist(), axes="current ego x forward, y left",
        purpose="Supplementary error shape evidence; shortlist oracle and selection regret remain primary",
        inputs={}, results={},
        limitations=["Axis absolute errors are not additive contributions to Euclidean L2 or D3.",
            "Squared-error fraction is unweighted across horizons and is neither D3 fraction nor probability.",
            "This descriptive diagnostic does not establish which signals are observable or rule-compliant.",
            "Frozen interventions and new matched training have different causal interpretations."])
    reference_rows = reference_gt = None
    for name, path in inputs:
        require(name and name not in report["results"], "Input labels must be nonempty and unique")
        path = Path(path).resolve()
        digest = file_sha(path)
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        require(file_sha(path) == digest, "Input changed while reading")
        required = ("rows", "pred", "error_xy", "point_l2", "d3", "shortlist_oracle")
        require(all(key in arrays for key in required), "Missing frozen/temporal evaluation array")
        rows, error = arrays["rows"], arrays["error_xy"]
        require(rows.shape == (1998,) and rows.dtype.kind in "iu"
                and len(np.unique(rows)) == 1998, "Expected 1,998 unique integer tune rows")
        rows_digest = hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()
        require(rows_digest == TUNE_SHA, "Expected pinned primary tune row identities/order")
        if reference_rows is None:
            reference_rows = rows.copy()
        else:
            require(np.array_equal(rows, reference_rows), "Input row ordering differs")
        require(error.shape == arrays["pred"].shape == (1998, 6, 2), "Malformed XY arrays")
        require(error.dtype in (np.float32, np.float64), "Expected saved FP32 or FP64 errors")
        require(np.isfinite(arrays["pred"]).all(), "Nonfinite predictions")
        point = np.linalg.norm(error.astype(np.float64), axis=-1)
        d3 = point @ WEIGHTS
        precision = dict(rtol=4e-7, atol=2e-7) if error.dtype == np.float32 else dict(rtol=0, atol=1e-12)
        require(arrays["point_l2"].shape == point.shape and np.allclose(arrays["point_l2"], point, **precision),
                "Saved point error differs from XY norm")
        require(arrays["d3"].shape == (1998,) and np.allclose(arrays["d3"], d3, **precision),
                "Saved D3 differs from official weighted XY norm")
        oracle = arrays["shortlist_oracle"].astype(np.float64)
        require(oracle.shape == (1998,) and np.isfinite(oracle).all()
                and (oracle >= 0).all() and (oracle <= d3+1e-5).all(), "Invalid shortlist oracle")
        gt = arrays["pred"].astype(np.float64)-error.astype(np.float64)
        if reference_gt is None:
            reference_gt = gt
        else:
            require(np.allclose(gt, reference_gt, rtol=0, atol=2e-5),
                    "Inputs imply different GT beyond saved FP32 subtraction rounding")
        summary = axis_error_summary(error)
        summary.update(saved_shortlist_oracle_m=float(oracle.mean()),
            selection_regret_m=float((d3-oracle).mean()),
            max_d3_recomputation_roundoff=float(np.abs(d3-arrays["d3"]).max()))
        report["results"][name] = summary
        report["inputs"][name] = dict(path=str(path), sha256=digest, rows_sha256=rows_digest,
            keys=sorted(arrays), error_xy_dtype=str(error.dtype))
    require(report["results"], "At least one NPZ is required")
    for source in report["inputs"].values():
        require(file_sha(source["path"]) == source["sha256"], "Input changed during analysis")
    report["analyzer_sha256"] = file_sha(__file__)
    return report


def markdown(report):
    lines = ["# 종방향·횡방향 오차 보조 진단", "",
        "`ex`는 현재 ego 좌표계의 종방향 오차, `ey`는 횡방향 오차다. "
        "시간 가중 절대오차는 각각 `mean_n Σ_t w_t |ex|`, `mean_n Σ_t w_t |ey|`이며 단위는 m다. "
        "두 값을 더해 Euclidean L2나 공식 D3의 기여로 해석하지 않는다.", "",
        "제곱오차 에너지 비율은 `Σ ex² / Σ(ex²+ey²)`이다. "
        "전체 값은 1,998행과 6개 시점을 시간 가중치 없이 합산한다. "
        "이는 D3 비율이나 확률이 아니며, 큰 오차가 제곱으로 강조되는 별도 통계다.", "",
        "| 결과 | D3 | Oracle | Regret | 가중 평균 |ex| (m) | 가중 평균 |ey| (m) | 종방향 제곱오차 에너지 비율 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    # Escape literal pipes inside axis labels so the Markdown table remains valid.
    lines[-2] = lines[-2].replace("|ex|", "abs(ex)").replace("|ey|", "abs(ey)")
    for name, s in report["results"].items():
        energy = s["longitudinal_squared_error_energy_fraction"]
        ratio = "정의 안 됨" if energy is None else f"{100*energy:.3f}%"
        lines.append(f"| {name} | {s['official_d3_recomputed_m']:.6f} | {s['saved_shortlist_oracle_m']:.6f} | "
            f"{s['selection_regret_m']:.6f} | {s['weighted_mean_abs_ex_m']:.6f} | "
            f"{s['weighted_mean_abs_ey_m']:.6f} | {ratio} |")
    for name, s in report["results"].items():
        lines += ["", f"**{name}**", "",
            "| 시점 (s) | mean abs(ex) (m) | mean abs(ey) (m) | 평균 L2 (m) | 해당 시점 종방향 제곱오차 에너지 비율 |",
            "|---|---:|---:|---:|---:|"]
        for h in s["horizons"]:
            energy = h["longitudinal_squared_error_energy_fraction"]
            ratio = "정의 안 됨" if energy is None else f"{100*energy:.3f}%"
            lines.append(f"| {h['seconds']:.1f} | {h['mean_abs_ex_m']:.6f} | {h['mean_abs_ey_m']:.6f} | "
                         f"{h['mean_euclidean_l2_m']:.6f} | {ratio} |")
    lines += ["", "후보가 남아 있는지와 선택을 잘하는지는 oracle/regret 결과로 먼저 해석한다. "
        "이 표는 남은 오차의 방향과 크기를 설명하는 보조 근거다. "
        "Frozen P7 상태 치환 결과는 P7 planner 자체의 성능이 아니며, "
        "기존 selector에 P7 b0의 영상 추정 상태를 치환한 결과다. "
        "새 temporal 학습 arm과 frozen 개입을 같은 학습 대조로 취급하지 않는다.", ""]
    return "\n".join(lines)


def self_test():
    class Tests(unittest.TestCase):
        def test_distinguishes_euclidean_and_axis_absolute_and_energy(self):
            error = np.tile(np.asarray([3., 4.]), (2, 6, 1))
            s = axis_error_summary(error)
            self.assertAlmostEqual(s["official_d3_recomputed_m"], 5.)
            self.assertAlmostEqual(s["weighted_mean_abs_ex_m"], 3.)
            self.assertAlmostEqual(s["weighted_mean_abs_ey_m"], 4.)
            self.assertAlmostEqual(s["longitudinal_squared_error_energy_fraction"], 9./25.)
            self.assertNotAlmostEqual(s["official_d3_recomputed_m"],
                s["weighted_mean_abs_ex_m"]+s["weighted_mean_abs_ey_m"])

        def test_unweighted_energy_is_not_time_weighted(self):
            error = np.zeros((1, 6, 2))
            error[0, 0, 0] = 1.
            error[0, -1, 1] = 1.
            s = axis_error_summary(error)
            self.assertEqual(s["longitudinal_squared_error_energy_fraction"], .5)
            self.assertAlmostEqual(s["weighted_mean_abs_ex_m"], 11./36.)
            self.assertAlmostEqual(s["weighted_mean_abs_ey_m"], 2./36.)
            self.assertIsNone(s["horizons"][1]["longitudinal_squared_error_energy_fraction"])

        def test_sign_flip_and_row_order_invariance(self):
            error = np.arange(36, dtype=np.float64).reshape(3, 6, 2)/10.
            a, b = axis_error_summary(error), axis_error_summary(-error[::-1])
            for key in ("weighted_mean_abs_ex_m", "weighted_mean_abs_ey_m",
                        "longitudinal_squared_error_energy_fraction", "official_d3_recomputed_m"):
                self.assertAlmostEqual(a[key], b[key])

        def test_zero_energy_and_invalid_input(self):
            s = axis_error_summary(np.zeros((2, 6, 2), np.float32))
            self.assertIsNone(s["longitudinal_squared_error_energy_fraction"])
            self.assertEqual(s["official_d3_recomputed_m"], 0.)
            with self.assertRaises(ValueError):
                axis_error_summary(np.full((2, 6, 2), np.nan))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return result.wasSuccessful()


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--input", action="append", help="NAME=NPZ; repeat for each existing evaluation")
    p.add_argument("--output", help="New immutable report directory")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not args.input or not args.output:
        p.error("--input NAME=NPZ and --output are required")
    inputs = []
    for value in args.input:
        if "=" not in value:
            p.error("Each --input must have NAME=NPZ form")
        inputs.append(value.split("=", 1))
    report = analyze(inputs)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/"axis_errors.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    (output/"AXIS_ERRORS_KO.md").write_text(markdown(report))
    receipt = dict(analyzer_sha256=file_sha(__file__), output_sha256={
        name:file_sha(output/name) for name in ("axis_errors.json", "AXIS_ERRORS_KO.md")})
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({name:{key:s[key] for key in ("official_d3_recomputed_m", "weighted_mean_abs_ex_m",
        "weighted_mean_abs_ey_m", "longitudinal_squared_error_energy_fraction")}
        for name,s in report["results"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
