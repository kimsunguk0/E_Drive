#!/usr/bin/env python3
"""P3 - freeze H (final confirmation sessions) and Tplus (expanded train).

H is chosen by a fixed hash of the session id, before any performance on those
sessions is looked at.  A = eligible reserve sessions minus H, Tplus = T0 + A.
Writes the new split manifest and reports/md_r0_reset_20260914/split_report.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_grouped_split_v2 import sha256, validate_manifest

BASE = ROOT / "data/etri/motiondrive_v2/grouped_split_rawtime.json"
EGO = Path("/tmp/pm97/data/etri/ego_cache.npz")
OUT_MANIFEST = ROOT / "data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json"
OUT_REPORT = ROOT / "reports/md_r0_reset_20260914/split_report.json"
HOLDOUT_KEY = "edrive-r0-holdout-v1"
HOLDOUT_SESSIONS = 10
MIN_FRAME = 30

# Every ancestor of R0 back to the public initializer, with the training data
# each one actually used.  Filled in from the run manifests by p3, not by hand.
LINEAGE = [
    "work_dirs/motiondrive_v2/q10_q10_flip50_s0",
    "work_dirs/motiondrive_v2/controlflow_b0_direct_last4000",
    "work_dirs/motiondrive_v2/pv_screen_b0_direct_last2000",
    "work_dirs/motiondrive_v2/early_precision_b0_continuation_control_last2000",
    "work_dirs/motiondrive_v2/shared_status_a2_b0_provided_last2000",
    "work_dirs/motiondrive_v2/p7_goal_routing_b0_control_last6000",
    "work_dirs/motiondrive_v2/p4_fresh_pretrain_s0",
]


def session_key(session_id: str) -> str:
    return hashlib.sha256(f"{HOLDOUT_KEY}|{session_id}".encode()).hexdigest()


def ancestor_fits(base):
    """What each ancestor trained on, read from its own run manifest."""
    train_scenes = set(base["splits"]["train"])
    entries = []
    union = set()
    for run in LINEAGE:
        manifest = json.loads((ROOT / run / "manifest.json").read_text())
        arguments = manifest.get("arguments", {})
        explicit = arguments.get("train_scenes")
        used = set(explicit) if explicit else (
            set() if manifest.get("status") == "public_initialization_only"
            else train_scenes)
        union |= used
        entries.append({
            "run": run,
            "split_manifest_sha256": manifest.get("split_sha256"),
            "explicit_train_scenes": len(explicit) if explicit else None,
            "train_rows": (manifest.get("data_counts") or {}).get("train"),
            "fitted_scenes": len(used),
            "pretrained": arguments.get("pretrained"),
            "status": manifest.get("status"),
        })
    return entries, union


def row_counts(scenes_by_group, ego):
    names = ego["scenarios"].astype(str)[ego["scen_idx"]]
    frames = ego["frame"]
    counts = {}
    for group, scenes in scenes_by_group.items():
        mask = np.isin(names, list(scenes)) & (frames >= MIN_FRAME)
        counts[group] = {"rows_stride1": int(mask.sum()),
                         "rows_stride5": int((mask & (frames % 5 == 0)).sum())}
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--write", action="store_true",
                        help="write the manifest; otherwise report only")
    args = parser.parse_args()

    base = json.loads(BASE.read_text())
    validate_manifest(base)
    base_sha = sha256(BASE)
    mapping = base["scene_to_session"]
    bounds = base.get("scene_time_bounds", {})

    train0 = sorted(base["splits"]["train"])
    tune0 = sorted(base["splits"]["tune"])
    reserve = sorted(base["splits"]["val"])
    historical = sorted(base["splits"]["historical_val"])

    lineage, fitted = ancestor_fits(base)
    reserve_sessions = sorted({mapping[s] for s in reserve})
    ineligible = sorted({mapping[s] for s in reserve if s in fitted})
    eligible = [s for s in reserve_sessions if s not in ineligible]
    if len(eligible) < HOLDOUT_SESSIONS:
        raise SystemExit("BLOCKED_LINEAGE: fewer eligible reserve sessions than H needs")

    ranked = sorted(eligible, key=lambda s: (session_key(s), s))
    holdout_sessions = sorted(ranked[:HOLDOUT_SESSIONS])
    absorb_sessions = sorted(s for s in eligible if s not in set(holdout_sessions))

    h_scenes = sorted(s for s in reserve if mapping[s] in set(holdout_sessions))
    a_scenes = sorted(s for s in reserve if mapping[s] in set(absorb_sessions))
    if set(h_scenes) & set(a_scenes) or sorted(h_scenes + a_scenes) != reserve:
        raise SystemExit("H/A partition of the reserve split is not exhaustive")

    tplus = sorted(train0 + a_scenes)
    historical_in_val = sorted(set(historical) & set(h_scenes))
    historical_in_train = sorted(set(historical) & set(a_scenes))

    manifest = dict(base)
    manifest["splits"] = {"train": tplus, "tune": tune0, "val": h_scenes,
                          "historical_val": historical_in_val}
    manifest["counts"] = {
        "train": {"scenes": len(tplus), "sessions": len({mapping[s] for s in tplus})},
        "tune": {"scenes": len(tune0), "sessions": len({mapping[s] for s in tune0})},
        "val": {"scenes": len(h_scenes), "sessions": len(holdout_sessions)},
        "historical_val": {"scenes": len(historical_in_val),
                           "sessions": len({mapping[s] for s in historical_in_val})},
    }
    manifest["r0_reset_envelope"] = {
        "schema_version": 1,
        "purpose": "E1 expanded-train arm; tune37 unchanged, H reserved for one final confirmation",
        "prior_manifest_path": str(BASE),
        "prior_manifest_sha256": base_sha,
        "prior_counts": base["counts"],
        "holdout_rule": f'sha256("{HOLDOUT_KEY}|" + session_id), ascending, first {HOLDOUT_SESSIONS}',
        "holdout_sessions": holdout_sessions,
        "absorbed_sessions": absorb_sessions,
        "tune_unchanged": tune0 == sorted(base["splits"]["tune"]),
        "train_superset_of_prior_train": set(tplus) >= set(train0),
        "val_subset_of_prior_val": set(h_scenes) <= set(reserve),
        "historical_val_full_prior": historical,
        "historical_val_moved_into_train": historical_in_train,
        "historical_val_caveat": (
            "historical_val38 was repeatedly evaluated by the earlier dense-VAD / "
            "ScoreDrive line. Scenes of it that fall in A now enter TRAINING, which "
            "is allowed; H is therefore an ancestor-fit-excluded confirmation set, "
            "NOT a never-inspected blind set."),
        "ancestor_lineage": lineage,
        "ancestor_fitted_reserve_sessions": ineligible,
    }

    validate_manifest(manifest)

    with np.load(EGO, allow_pickle=False) as z:
        ego = {k: z[k] for k in ("scenarios", "scen_idx", "frame")}
    counts = row_counts({"T0_train203": train0, "V0_tune37": tune0,
                         "A_absorbed": a_scenes, "H_confirmation": h_scenes,
                         "Tplus": tplus}, ego)

    def support(scenes):
        picked = [bounds[s] for s in scenes if s in bounds]
        return {"scenes_with_bounds": len(picked), "scenes": len(scenes)}

    report = {
        "schema_version": 1,
        "base_manifest": {"path": str(BASE), "sha256": base_sha},
        "new_manifest": {"path": str(OUT_MANIFEST), "written": bool(args.write)},
        "holdout_rule": manifest["r0_reset_envelope"]["holdout_rule"],
        "eligible_reserve_sessions": len(eligible),
        "ineligible_reserve_sessions": ineligible,
        "H": {"sessions": holdout_sessions, "scenes": len(h_scenes),
              **counts["H_confirmation"], **support(h_scenes)},
        "A": {"sessions": absorb_sessions, "scenes": len(a_scenes),
              **counts["A_absorbed"], **support(a_scenes)},
        "T0": {"scenes": len(train0), "sessions": len({mapping[s] for s in train0}),
               **counts["T0_train203"]},
        "V0": {"scenes": len(tune0), "sessions": len({mapping[s] for s in tune0}),
               **counts["V0_tune37"]},
        "Tplus": {"scenes": len(tplus),
                  "sessions": len({mapping[s] for s in tplus}), **counts["Tplus"]},
        "expansion_ratio_rows_stride1": (counts["Tplus"]["rows_stride1"]
                                         / counts["T0_train203"]["rows_stride1"]),
        "ancestor_lineage": lineage,
        "historical_val_moved_into_train": historical_in_train,
        "historical_val_caveat": manifest["r0_reset_envelope"]["historical_val_caveat"],
        "supervision_status": "A scenes still need supervision targets; see p3_supervision",
    }

    if args.write:
        if OUT_MANIFEST.exists():
            raise FileExistsError(f"Immutable split already exists: {OUT_MANIFEST}")
        OUT_MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        report["new_manifest"]["sha256"] = sha256(OUT_MANIFEST)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("eligible_reserve_sessions", "expansion_ratio_rows_stride1")}
                     | {"H_scenes": report["H"]["scenes"],
                        "H_rows": report["H"]["rows_stride1"],
                        "A_scenes": report["A"]["scenes"],
                        "A_rows": report["A"]["rows_stride1"],
                        "Tplus_scenes": report["Tplus"]["scenes"],
                        "Tplus_rows": report["Tplus"]["rows_stride1"],
                        "T0_rows": report["T0"]["rows_stride1"],
                        "new_manifest_sha256": report["new_manifest"].get("sha256"),
                        "historical_val_into_train": len(historical_in_train)},
                     indent=1))


if __name__ == "__main__":
    main()
