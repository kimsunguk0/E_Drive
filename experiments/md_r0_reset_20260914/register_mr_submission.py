#!/usr/bin/env python3
"""Connect the MR-NATIVE-s0 submission to the registry.

The graph's FLOPs and preprocessing were already verified for this exact graph,
so they are linked rather than rebuilt. What is re-checked here is what belongs
to these weights: finiteness, shape, clip isolation and output parity.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
REPORTS = ROOT / "reports/md_r0_reset_20260914"
REGISTRY = REPORTS / "submission_registry.json"
SUB = REPORTS / "submission/official_test_MR-NATIVE-s0.json"
VAL = REPORTS / "submission/official_test_MR-NATIVE-s0.validation.json"
PREV = REPORTS / "submission/official_test_E1-EXP.json"
CKPT = ROOT / "work_dirs/md_r0_reset_20260914/MR-NATIVE-s0/ckpt_step20554.pth"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    registry = json.loads(REGISTRY.read_text())
    validation = json.loads(VAL.read_text())
    submission = json.loads(SUB.read_text())
    previous = json.loads(PREV.read_text())

    shared = sorted(set(submission) & set(previous))
    a = np.asarray([submission[k] for k in shared], dtype=np.float64)
    b = np.asarray([previous[k] for k in shared], dtype=np.float64)
    endpoint_a = np.linalg.norm(a[:, -1], axis=-1)
    endpoint_b = np.linalg.norm(b[:, -1], axis=-1)

    entry = {
        "role": "current candidate (replaces E1-EXP_terminal as the V0-selected model)",
        "checkpoint": str(CKPT),
        "checkpoint_sha256": file_sha256(CKPT),
        "submission_file": str(SUB),
        "submission_sha256": validation["submission_sha256"],
        "clips_written": validation["clips"],
        "seed_choice": {
            "chosen": "MR-NATIVE-s0",
            "rule": ("the pre-registered rule fixes the checkpoint WITHIN a run (lowest V0 "
                     "among planned eval steps) and says nothing about choosing between "
                     "seeds; the E1-EXP precedent made seed 0 the candidate and seed 1 the "
                     "replication check"),
            "not_chosen": "MR-NATIVE-s1 (V0 0.191002)",
            "why_not": ("the seeds differ by 0.000890 with a paired CI of [-0.00265, "
                        "+0.00136] that includes zero; switching on that would be "
                        "selecting on noise, not on evidence"),
        },
        "v0_official_d3": 0.191892,
        "v0_is_not_a_server_score": True,
        "checks_for_these_weights": {
            "all_shapes_6x2": validation["checks"]["all_shapes_6x2"],
            "all_finite": validation["checks"]["all_finite"],
            "clip_state_isolation_max_abs_diff_m":
                validation["checks"]["clip_state_isolation_max_abs_diff_m"],
            "clip_state_isolation_pass": validation["checks"]["clip_state_isolation_pass"],
            "cumsum_not_reapplied_ratio": validation["checks"]["cumsum_not_reapplied"]["ratio"],
            "raw_b1_input_output_parity": {
                "artifact": "reports/md_r0_reset_20260914/eval/mr_raw_b1_fixture.json",
                "fixtures": 8, "inputs_all_bitwise_equal": True,
                "max_plan_abs_xy_diff_m": 0.0,
                "note": "includes the two MR canvas tensors",
            },
        },
        "graph_level_checks_linked_not_rebuilt": {
            "reason": ("the FLOPs, timing and preprocessing were measured on this exact "
                       "graph; only the weights changed"),
            "rtx4090_timing": "reports/md_r0_reset_20260914/rtx4090_mr_forward_cost.json",
            "bf16_median_ms_across_clips": [24.274, 24.517],
            "time_penalty_multiplier": 1.0,
            "profiler_gflops_total": 1119.923714404,
            "flops_headroom_x": 6.297750381822621,
            "does_not_transfer_to": ["MR-W64", "MR-ADJ0", "MR-ADJ1"],
        },
        "difference_from_previous_candidate": {
            "compared_with": "official_test_E1-EXP.json",
            "clips_compared": len(shared),
            "max_abs_xy_diff_m": float(np.abs(a - b).max()),
            "mean_abs_xy_diff_m": float(np.abs(a - b).mean()),
            "mean_endpoint_m": {"MR-NATIVE-s0": float(endpoint_a.mean()),
                                "E1-EXP": float(endpoint_b.mean())},
            "reading": ("a sanity check that the new weights produce a different but "
                        "comparable trajectory set; it is not an accuracy comparison, "
                        "because the official test set has no ground truth here"),
        },
        "upload": {
            "performed": False,
            "blocked_on": ["the current submission window", "the remaining attempt count",
                           "explicit user approval"],
        },
        "limitations": [
            "V0 0.191892 is a development-set number, not a server score and not an H score.",
            "H has not been opened; this candidate is registered before any H evaluation.",
        ],
    }
    registry.setdefault("entries", {})["MR-NATIVE-s0"] = entry
    if registry["entries"].get("E1-EXP_terminal"):
        registry["entries"]["E1-EXP_terminal"]["role"] = (
            "previous candidate, preserved as the deployment fallback")
    REGISTRY.write_text(json.dumps(registry, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"submission_sha256": entry["submission_sha256"],
                      "clips": entry["clips_written"],
                      "checkpoint_sha256": entry["checkpoint_sha256"][:24],
                      "difference_from_previous": entry["difference_from_previous_candidate"],
                      "upload": entry["upload"]}, indent=1))


if __name__ == "__main__":
    main()
