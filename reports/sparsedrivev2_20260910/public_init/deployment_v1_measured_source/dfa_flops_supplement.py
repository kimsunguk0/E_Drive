"""CPU analytic forward-only DFA supplement; not an official total model FLOP count.

Counts source-level scalar arithmetic: multiply/add/divide/exp each one, FMA two.
The native CUDA kernel is opaque to PyTorch FlopCounterMode. Its all-camera
upper bound is reported separately from standard Torch operations to avoid
double counting if a future registry adds coverage. No model or GPU is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def count_stage(name, anchors, samples, *, batch=1, cameras=3, levels=4,
                channels=256, groups=8, valid_camera_points=None):
    points = batch * anchors * samples
    camera_points = points * cameras
    if valid_camera_points is None:
        valid_camera_points = camera_points
        is_upper_bound = True
    else:
        if not isinstance(valid_camera_points, int) or not 0 <= valid_camera_points <= camera_points:
            raise ValueError("valid_camera_points must count strict 0<uv<1 camera-point pairs")
        is_upper_bound = False
    visits = valid_camera_points * levels * channels
    weights = camera_points * levels * groups
    groups_total = batch * anchors * groups
    return {
        "name": name, "anchors": anchors, "sample_points_per_anchor": samples,
        "query_points": points, "camera_points": camera_points,
        "valid_camera_points_assumed": valid_camera_points,
        "all_camera_visibility_upper_bound": is_upper_bound,
        "native_channel_level_visits": visits,
        "native_kernel_flops": 21 * visits,
        "native_interpolation_and_accumulation_subset_flops": 9 * visits,
        "torch_projection_matmul_flops_normally_counted": 32 * camera_points,
        "torch_projection_division_scalar_ops": 4 * camera_points,
        "torch_stable_softmax_nominal_scalar_ops": 4 * weights - groups_total,
        "torch_native_output_point_reduction_adds": batch * anchors * channels * (samples - 1),
        "torch_anchor_plus_offset_adds": 2 * points,
        "torch_feature_plus_camera_embedding_adds": batch * anchors * cameras * channels,
        "torch_residual_adds": batch * anchors * channels,
    }


def build_report():
    stages = [count_stage("decoder0_path", 1024, 500),
              count_stage("decoder1_path", 128, 500),
              count_stage("decoder1_trajectory", 200, 80)]
    aggregate_keys = [key for key, val in stages[0].items()
                      if isinstance(val, int) and not isinstance(val, bool)
                      and key not in ("anchors", "sample_points_per_anchor")]
    totals = {key: sum(stage[key] for stage in stages) for key in aggregate_keys}
    assert totals["query_points"] == 592000
    assert totals["native_kernel_flops"] == 38191104000
    assert totals["torch_projection_matmul_flops_normally_counted"] == 56832000
    assert totals["torch_stable_softmax_nominal_scalar_ops"] == 227317184
    assert totals["torch_native_output_point_reduction_adds"] == 151205888
    assert count_stage("invisible", 1, 1, valid_camera_points=0)["native_kernel_flops"] == 0
    report = {
        "scope": "forward-only analytic DFA arithmetic supplement, no GPU, not __flops__",
        "convention": "one scalar multiply/add/subtract/divide/exp = 1; MAC/FMA = 2 FLOPs",
        "exclusions": ["floor/conversion/index/comparison/mask/load/store",
                       "backward", "kernel launch/compiler instruction count",
                       "all non-DFA model operations"],
        "architecture": {"batch": 1, "cameras": 3, "levels": 4, "channels": 256,
                         "groups": 8, "xy_offsets_per_height": 2,
                         "fixed_heights": [0, -.25, -.5, .25, .5],
                         "path_points": 50, "trajectory_points": 8},
        "native_kernel_per_visit": {
            "normalized_to_feature_coordinates": "2 multiply + 2 subtract = 4",
            "bilinear_fractional_offsets": "4 subtract = 4",
            "four_neighbor_coefficients": "4 multiply = 4",
            "four_neighbor_weighted_value": "4 multiply + 3 add = 7",
            "learned_weight_and_camera_level_accumulate": "1 multiply + 1 add = 2",
            "total": "11 multiply + 10 add/subtract = 21",
            "core_subset": "7 interpolation value + 2 learned accumulation = 9, included in 21",
        },
        "stages": stages, "totals": totals,
        "native_kernel_upper_gflops": totals["native_kernel_flops"] / 1e9,
        "double_count_policy": [
            "Native kernel opaque to unregistered PyBind: add this bound once to raw Torch counter for conservative supplemental bound.",
            "Do not add the 9-FLOP interpolation subset on top of the 21-FLOP kernel total.",
            "Projection matmul is normally dispatched as aten.bmm/mm and already counted: do not add again.",
            "Softmax/divisions/reductions/adds are separate diagnostics; add only after checking measured operator coverage.",
            "Linear layers, convolutions, attention, camera encoders, keypoint/weight/output projections belong in actual Torch counter.",
            "Exp nominal scalar-op count is not a device instruction-cost model.",
        ],
        "visibility": "For an exact source-arithmetic native count, substitute per-stage strict normalized 0<xy<1 camera-point count. Native kernel does not itself test depth. All-camera count is a conservative upper bound; learned coordinates change the actual count.",
        "source_pin": "swc-17/SparseDriveV2@696ef77924eb9e0a4b4047d013a50e9854bfa026",
        "source_files": ["experiments/sparsedrivev2_20260910/public_model.py",
                         "third_party/SparseDriveV2/navsim/agents/sparsedrive/ops/src/deformable_aggregation_cuda.cu"],
        "self_test": "PASS arithmetic invariants and zero visibility",
    }
    try:
        import torch
        from torch.utils.flop_counter import flop_registry
        registry = sorted(str(key) for key in flop_registry)
        report["torch_version"] = torch.__version__
        report["flop_registry_relevant_ops"] = [key for key in registry
            if any(fragment in key for fragment in ("softmax", "grid", "mm", "native_multi", "sum", "div", "add.Tensor"))]
    except ImportError:
        report["torch_registry"] = "unavailable; analytic arithmetic is dependency free"
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "native_kernel_upper_gflops": report["native_kernel_upper_gflops"],
                      "self_test": report["self_test"]}))
