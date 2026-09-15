#!/usr/bin/env python3
"""Contract checks for the interval-length auxiliary before it is trained with.

Reuses the synthetic batch builder from the P2 contract tests so the microbatch
equivalence is checked exactly the way the original losses were.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments/md_r0_reset_20260914"))

import motiondrive_v2_training as mt
from contract_tests import synthetic, _slice_outputs
from length_auxiliary import interval_lengths, length_loss, wrap_compute_loss

FP32_RTOL, FP32_ATOL = 1e-5, 1e-6
OUT = ROOT / "reports/md_exp_diagnosis_20260915/length_auxiliary_contract.json"
LAMBDA = 0.25


def check_geometry():
    """The interval lengths must reproduce a known polyline exactly."""
    points = torch.tensor([[[3.0, 4.0], [3.0, 8.0], [0.0, 8.0],
                            [0.0, 8.0], [-3.0, 8.0], [-3.0, 4.0]]])
    lengths = interval_lengths(points)
    expected = torch.tensor([[5.0, 4.0, 3.0, 0.0, 3.0, 4.0]])
    zero = interval_lengths(torch.zeros(2, 6, 2))
    return {"pass": bool(torch.allclose(lengths, expected, atol=1e-6)
                         and float(zero.abs().max()) == 0.0),
            "expected": expected.tolist()[0], "actual": lengths.tolist()[0],
            "origin_is_the_first_interval_start": True,
            "stationary_gives_zero": float(zero.abs().max()) == 0.0}


def check_identity_and_direction_blindness():
    """Zero error when pred equals GT; blind to a mirrored path of equal lengths."""
    g = torch.Generator().manual_seed(5)
    gt = torch.cumsum(torch.randn(8, 6, 2, generator=g).abs(), dim=1)
    batch = {"gt_plan": gt, "plan_valid": torch.ones(8, 6, dtype=torch.bool)}
    same = float(length_loss({"plan_abs": gt.clone()}, batch))
    mirrored = gt.clone()
    mirrored[..., 1] *= -1
    mirror_length = float(length_loss({"plan_abs": mirrored}, batch))
    mirror_d3 = float(mt.weighted_d3(mirrored, gt).mean())
    return {"pass": bool(same == 0.0 and mirror_length < 1e-6 and mirror_d3 > 1.0),
            "identical_prediction_loss": same,
            "mirrored_path_length_loss": mirror_length,
            "mirrored_path_d3": mirror_d3,
            "reading": ("a mirrored path has identical interval lengths, so the length term "
                        "alone cannot see it; D3 does. This is why it is added to D3, never "
                        "used in place of it")}


def check_lambda_zero_is_identity():
    weights = mt.LossWeights(plan=1., occupancy=.2, lane=.2, motion=.2, uncertainty=True)
    scale, base, batch = synthetic(16, seed=11)
    outputs = _slice_outputs(base, scale, 0, 16)
    reference, _ = mt.compute_loss(outputs, batch, weights)
    wrapped = wrap_compute_loss(mt.compute_loss, 0.0)
    value, _ = wrapped(outputs, batch, weights)
    return {"pass": bool(float(reference) == float(value)),
            "original": float(reference), "lambda_zero": float(value),
            "bitwise_equal": float(reference) == float(value)}


def check_microbatch_equivalence():
    weights = mt.LossWeights(plan=1., occupancy=.2, lane=.2, motion=.2, uncertainty=True)
    wrapped = wrap_compute_loss(mt.compute_loss, LAMBDA)
    cases, ok = {}, True
    for batch_size, invalid in ((16, ()), (15, ()), (16, (4, 5))):
        scale, base, batch = synthetic(batch_size, invalid_rows=invalid, seed=batch_size + 1)
        dense_loss, dense_parts = wrapped(_slice_outputs(base, scale, 0, batch_size),
                                          batch, weights)
        dense_loss.backward()
        dense_grad = float(scale.grad)
        entry = {"dense_total": float(dense_loss), "dense_grad": dense_grad,
                 "invalid_rows": list(invalid), "microbatches": {}}
        for micro in (1, 2, 8, 16):
            if micro > batch_size:
                continue
            scale_m, base_m, batch_m = synthetic(batch_size, invalid_rows=invalid,
                                                 seed=batch_size + 1)
            normalizers = mt.build_loss_normalizers(batch_m)
            total, parts = 0., {}
            for start in range(0, batch_size, micro):
                chunk_out = _slice_outputs(base_m, scale_m, start, start + micro)
                chunk_batch = {k: v[start:start + micro] for k, v in batch_m.items()}
                loss, micro_parts = wrapped(chunk_out, chunk_batch, weights,
                                            normalizers=normalizers)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite microbatch loss")
                loss.backward()
                total += float(loss)
                for name, value in micro_parts.items():
                    parts[name] = parts.get(name, 0.) + float(value.detach())
            grad = float(scale_m.grad)
            part_diff = max(abs(parts[name] - float(dense_parts[name])) for name in dense_parts)
            passed = (np.isclose(total, float(dense_loss), rtol=FP32_RTOL, atol=FP32_ATOL)
                      and np.isclose(grad, dense_grad, rtol=FP32_RTOL, atol=FP32_ATOL)
                      and part_diff <= FP32_ATOL + FP32_RTOL * abs(float(dense_loss)))
            ok &= passed
            entry["microbatches"][str(micro)] = {
                "total_abs_diff": abs(total - float(dense_loss)),
                "grad_abs_diff": abs(grad - dense_grad),
                "max_part_abs_diff": part_diff, "pass": bool(passed)}
        cases[f"b{batch_size}" + ("_with_invalid_rows" if invalid else "")] = entry
    return {"pass": bool(ok), "lambda": LAMBDA, "cases": cases,
            "tolerance": {"rtol": FP32_RTOL, "atol": FP32_ATOL}}


def check_empty_and_partial_validity():
    weights = mt.LossWeights(plan=1., occupancy=0., lane=0., motion=0., uncertainty=True)
    wrapped = wrap_compute_loss(mt.compute_loss, LAMBDA)
    scale, base, batch = synthetic(16, invalid_rows=tuple(range(16)), seed=3)
    loss, parts = wrapped(_slice_outputs(base, scale, 0, 16), batch, weights)
    return {"pass": bool(torch.isfinite(loss) and float(parts["plan_interval_length"]) == 0.0),
            "all_rows_invalid_length_term": float(parts["plan_interval_length"]),
            "total_is_finite": bool(torch.isfinite(loss))}


def main() -> None:
    checks = {
        "geometry": check_geometry(),
        "identity_and_direction_blindness": check_identity_and_direction_blindness(),
        "lambda_zero_is_identity": check_lambda_zero_is_identity(),
        "microbatch_equivalence": check_microbatch_equivalence(),
        "empty_and_partial_validity": check_empty_and_partial_validity(),
    }
    payload = {"schema_version": 1, "lambda": LAMBDA,
               "all_pass": all(c["pass"] for c in checks.values()), "checks": checks}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"all_pass": payload["all_pass"],
                      "per_check": {k: v["pass"] for k, v in checks.items()}}, indent=1))
    if not payload["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
