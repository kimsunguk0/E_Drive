#!/usr/bin/env python3
"""Register the MR candidates as immutable roles.

Registration is bookkeeping, not a performance claim. In particular DEV_BEST is
a point-estimate development choice: MR-NATIVE was not shown to beat
MR-LOWDETAIL, whose paired CI includes zero.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path("/NHNHOME/data/sukim/adcl")
REPORTS = ROOT / "reports/md_r0_reset_20260914"
WORK = ROOT / "work_dirs/md_r0_reset_20260914"
OUT = REPORTS / "mr_candidate_registry.json"
JUDGEMENT = ROOT / "reports/md_exp_diagnosis_20260915/mr_judgement.json"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


ROLES = {
    "DEV_BEST": {
        "run": "MR-NATIVE-s0", "step": 20554, "v0_official_d3": 0.191892,
        "why": ("the MR-NATIVE recipe, whose gain over LEN replicated across two seeds "
                "(-0.031753 and -0.033136, both CIs clear of zero)"),
        "caveat": ("a point-estimate choice only. Three things were NOT established: "
                   "superiority over MR-LOWDETAIL (-0.001871, CI includes zero), "
                   "superiority of MR-W64 over this run (-0.001000, CI includes zero and "
                   "the same order as the 0.000890 seed spread), and any ordering between "
                   "the two MR-NATIVE seeds (-0.000890, CI includes zero)"),
    },
    "DEV_BEST_REPLICATE": {
        "run": "MR-NATIVE-s1", "step": 20554, "v0_official_d3": 0.191002,
        "why": "the seed replication; the recipe's gain survives a second seed",
        "caveat": ("lower than seed 0 by 0.000890, which is inside its own CI; the seeds "
                   "are not ordered by this evidence"),
    },
    "W64_ARM": {
        "run": "MR-W64-s0", "step": 20554, "v0_official_d3": 0.190892,
        "why": ("descriptor 32 -> 64 on the MR-NATIVE graph; preserved as the record of a "
                "closed question"),
        "caveat": ("lowest terminal number in the round, but NOT the best candidate: its "
                   "advantage over the matched control is not established and is the same "
                   "size as the control's seed spread. Reading it as best would be reading "
                   "noise"),
    },
    "MR_REFERENCE": {
        "run": "MR-LOWDETAIL-s0", "step": 20554, "v0_official_d3": 0.193763,
        "why": "preserved so the detail contrast stays reproducible",
        "caveat": ("not automatically the cheaper deployment model: both MR arms run "
                   "the same 768x432 canvas and the same radius-4 graph, and the "
                   "lowdetail arm additionally pays a 768->384->768 resize"),
    },
    "LEN_CONTROL": {
        "run": "E1-EXP-LEN-s0", "step": 20554, "v0_official_d3": 0.223644,
        "why": "the control both MR arms were measured against; preserved unchanged",
        "caveat": None,
    },
    "DEPLOY_FALLBACK": {
        "run": "E1-EXP", "step": 20554, "v0_official_d3": 0.225592,
        "why": ("unchanged: still the only model with a complete deployment set "
                "(submission build, raw-B1 parity, negative controls, 4090 timing)"),
        "caveat": None,
    },
}


def timing():
    """The MR graph measured on its own; the E1-EXP figure is not inherited."""
    previous = REPORTS / "rtx4090_forward_cost.json"
    current = REPORTS / "rtx4090_mr_forward_cost.json"
    block = {
        "inherits_previous_4090_measurement": False,
        "reason": ("the MR arms changed the motion branch graph: a 768x432 canvas with "
                   "its own backbone pass and radius 4 at both levels, so the E1-EXP "
                   "figure does not transfer to any MR checkpoint"),
        "previous_measurement": str(previous.relative_to(ROOT)),
        "score_time_term": "x(1 + (T_ms - 100)/200), so T <= 100 ms carries no penalty",
    }
    if not current.exists():
        block["status"] = "PENDING: MR graph must be measured on the same 4090 conditions"
        return block
    mr = json.loads(current.read_text())
    old = json.loads(previous.read_text())
    hi = mr["bf16_median_ms_across_clips"]["max"]
    old_hi = old["bf16_median_ms_across_clips"]["max"]
    gflops = mr["profiler_gflops_total"]
    block.update({
        "status": "MEASURED",
        "measurement": str(current.relative_to(ROOT)),
        "device": mr["device"],
        "bf16_median_ms_across_clips": mr["bf16_median_ms_across_clips"],
        "forwards_per_clip": mr["forwards_per_clip"],
        "time_penalty_multiplier": max(1.0, 1.0 + (hi - 100.0) / 200.0),
        "headroom_to_penalty_threshold_x": 100.0 / hi,
        "slower_than_e1_exp_x": hi / old_hi,
        "profiler_gflops_total": gflops,
        "flops_cutoff_gflops": 7053.0,
        "flops_headroom_x": 7053.0 / gflops,
        "reading": ("the MR graph is about 1.48x slower than E1-EXP and uses about 1.43x "
                    "the FLOPs, but both stay far inside their limits: at ~24.5 ms the "
                    "time multiplier is exactly 1.0, and the FLOPs headroom falls from "
                    "9.02x to 6.30x without approaching the cutoff"),
        "deployment_checks": {
            "raw_b1_input_output_parity": "reports/md_r0_reset_20260914/eval/mr_raw_b1_fixture.json",
            "submission_path": "smoke-built on 12 official test clips with --mr-detail native",
        },
    })
    return block


def main() -> None:
    judgement = json.loads(JUDGEMENT.read_text()) if JUDGEMENT.exists() else None
    entries = {}
    for role, spec in ROLES.items():
        ckpt = WORK / spec["run"] / f"ckpt_step{spec['step']}.pth"
        plan = REPORTS / f"experiment_{spec['run']}.json"
        if not plan.exists():                      # older runs used a bare arm name
            plan = WORK / spec["run"] / "experiment.json"
        entry = dict(spec)
        entry["checkpoint_path"] = str(ckpt)
        entry["checkpoint_exists"] = ckpt.exists()
        entry["checkpoint_sha256"] = file_sha256(ckpt) if ckpt.exists() else None
        entry["experiment_record"] = str(plan)
        entry["experiment_record_sha256"] = file_sha256(plan) if plan.exists() else None
        entry["immutable"] = True
        entries[role] = entry

    payload = {
        "schema_version": 1,
        "purpose": ("freeze the MR round's candidates so later work cannot silently "
                    "replace them; registration records what was run, not a claim "
                    "that it generalises"),
        "verdict": judgement.get("verdict") if isinstance(judgement, dict) else None,
        "roles": entries,
        "timing_contract": timing(),
        "open_decisions": {
            "promote_dev_best_to_deploy": ("the seed replication, the 4090 measurement and "
                                           "the raw-B1 parity are now done; what remains is "
                                           "a full 1,125-clip submission rebuild and the "
                                           "user's decision"),
            "native_vs_lowdetail": "unresolved by design; the primary CI includes zero",
            "descriptor_width": ("closed for now: MR-W64 showed no established gain, so no "
                                 "96/128 sweep follows"),
            "weight_average": ("closed: the 50:50 average of the two MR-NATIVE seeds scored "
                               "0.193545, worse than either seed, so no combination search "
                               "follows"),
        },
        "holdout": {"H_opened": False,
                    "rule": "a candidate change must be logged before H is opened"},
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    for role, e in entries.items():
        print(f"{role:16s} {e['run']:18s} V0={e['v0_official_d3']:.6f} "
              f"ckpt={'ok' if e['checkpoint_exists'] else 'MISSING'} "
              f"sha={(e['checkpoint_sha256'] or '')[:16]}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
