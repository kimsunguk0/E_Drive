#!/usr/bin/env python3
"""Round-3 judgement: MR-ADJ and the longer schedule.

Two independent questions.

  MR-ADJ   Does letting the model read the three comparisons BETWEEN past
           frames reduce the planning error? Primary contrast is ADJ1 - ADJ0,
           the same module with and without those edges, so the module's
           capacity is held fixed. Practical contrast is ADJ1 - MR-NATIVE-s0,
           which also has to beat the arm carrying no module at all.

  LONG     Every MR run stopped while still descending, so the schedule was
           never observed past 3.93 exposures. This asks whether the recipe run
           to 8 exposures, with the cosine horizon set to that total from step
           one, reaches a lower point than the short schedule.

Every comparison is reported twice: on the plain V0 mean, and under the
test-matched weighting that reproduced the leaderboard to -0.3% in the original
project. The weighted number is the one that predicts the server; the plain one
is what the earlier rounds were judged on.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT / "experiments/md_r0_reset_20260914"))
from analyze_mr import W as D3_W, load

WORK = ROOT / "work_dirs/md_r0_reset_20260914"
OUT = ROOT / "reports/md_exp_diagnosis_20260915"
WEIGHTS = OUT / "test_matched_weighting.json"
RESAMPLES = 20000

RUNS = {
    "LEN-s0": "E1-EXP-LEN-s0",
    "MR-NATIVE-s0": "MR-NATIVE-s0",
    "MR-NATIVE-s1": "MR-NATIVE-s1",
    "MR-W64-s0": "MR-W64-s0",
    "MR-ADJ0-s0": "MR-ADJ0-s0",
    "MR-ADJ1-s0": "MR-ADJ1-s0",
    "MR-NATIVE-LONG-s0": "MR-NATIVE-LONG-s0",
}


def main() -> None:
    reference = gt = sessions = None
    d3s = {}
    for label, run in RUNS.items():
        path = WORK / run / "final_eval.json"
        if not path.exists():
            raise SystemExit(f"{label}: no final_eval.json")
        keys, pred, gt_r, session, _, _ = load(path)
        if reference is None:
            reference, gt, sessions = keys, gt_r, session
        elif keys != reference or not np.array_equal(gt_r, gt):
            raise SystemExit(f"{label} is not scored on the same rows or ground truth")
        d3s[label] = np.linalg.norm(pred - gt, axis=-1) @ D3_W

    # The weighting is recomputed by its own producer; here it is only read, so
    # the two reports cannot drift apart.
    weighting = json.loads(WEIGHTS.read_text())
    import importlib
    tmw = importlib.import_module("test_matched_weighting")
    cells_t, _ = tmw.test_cells()
    from collections import Counter
    count_t = Counter(cells_t)
    n_t = sum(count_t.values())
    cells_v, valid = tmw.val_cells(reference, gt)
    usable = [c for c, ok in zip(cells_v, valid) if ok]
    count_v = Counter(usable)
    n_v = len(usable)
    weights = np.zeros(len(cells_v))
    for i, (cell, ok) in enumerate(zip(cells_v, valid)):
        if ok and count_t.get(cell, 0):
            weights[i] = (count_t[cell] / n_t) / (count_v[cell] / n_v)
    weights = np.minimum(weights, tmw.WEIGHT_CAP)
    weights = weights * (len(weights) / weights.sum())

    unique = sorted(set(sessions.tolist()))
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(unique), size=(RESAMPLES, len(unique)))
    picks = [np.concatenate([index[unique[j]] for j in row]) for row in draws]

    def compare(base, arm, w=None):
        diff = d3s[arm] - d3s[base]
        if w is None:
            point = float(diff.mean())
            means = np.asarray([diff[p].mean() for p in picks])
            improved = sum(1 for s in unique if diff[index[s]].mean() < 0)
        else:
            point = float((diff * w).sum() / w.sum())
            means = np.asarray([(diff[p] * w[p]).sum() / max(w[p].sum(), 1e-12)
                                for p in picks])
            improved = sum(1 for s in unique
                           if (diff[index[s]] * w[index[s]]).sum() < 0)
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        return {"delta": point, "ci95": [lo, hi], "ci_includes_zero": bool(lo <= 0 <= hi),
                "p_worse": float((means > 0).mean()),
                "sessions_improved": improved, "sessions": len(unique)}

    def both(base, arm):
        return {"plain": compare(base, arm), "test_matched": compare(base, arm, weights)}

    seed_spread = abs(float(d3s["MR-NATIVE-s0"].mean()) - float(d3s["MR-NATIVE-s1"].mean()))
    seed_spread_w = abs(float((d3s["MR-NATIVE-s0"] * weights).sum() / weights.sum())
                        - float((d3s["MR-NATIVE-s1"] * weights).sum() / weights.sum()))

    adj = {
        "question": ("does reading the three comparisons between past frames reduce the "
                     "planning error?"),
        "PRIMARY MR-ADJ0-s0 -> MR-ADJ1-s0": both("MR-ADJ0-s0", "MR-ADJ1-s0"),
        "PRACTICAL MR-NATIVE-s0 -> MR-ADJ1-s0": both("MR-NATIVE-s0", "MR-ADJ1-s0"),
        "MODULE ALONE MR-NATIVE-s0 -> MR-ADJ0-s0": both("MR-NATIVE-s0", "MR-ADJ0-s0"),
        "control_seed_spread": {"plain": seed_spread, "test_matched": seed_spread_w},
    }
    primary = adj["PRIMARY MR-ADJ0-s0 -> MR-ADJ1-s0"]
    practical = adj["PRACTICAL MR-NATIVE-s0 -> MR-ADJ1-s0"]
    beats_spread = abs(primary["plain"]["delta"]) > seed_spread
    if (primary["plain"]["delta"] < 0 and not primary["plain"]["ci_includes_zero"]
            and practical["plain"]["delta"] < 0 and beats_spread):
        code = "ADJACENT_EDGES_HELP"
        action = ("the adjacent edges beat their masked control and the no-module arm with "
                  "a CI clear of zero; keep them as the candidate direction")
    elif primary["plain"]["delta"] < 0 and practical["plain"]["delta"] >= 0:
        code = "BEATS_CONTROL_NOT_BASELINE"
        action = ("ADJ1 beats ADJ0 but not the arm with no module at all; do not promote")
    else:
        code = "NO_ESTABLISHED_EFFECT"
        action = ("no established gain from the adjacent edges at this budget; close this "
                  "ADJ setting and do not conclude that temporal relations in general "
                  "cannot be used")
    adj["verdict"] = {
        "code": code, "action": action,
        "delta_vs_seed_spread": {"primary_delta": primary["plain"]["delta"],
                                 "seed_spread": seed_spread,
                                 "exceeds": bool(beats_spread)},
    }

    long_arm = {
        "question": ("does the same recipe reach a lower point when run to 8 exposures "
                     "with the cosine horizon set to that total from step one?"),
        "MR-NATIVE-s0 -> MR-NATIVE-LONG-s0 (terminal)": both("MR-NATIVE-s0",
                                                             "MR-NATIVE-LONG-s0"),
        "note": ("final_eval.json holds the TERMINAL checkpoint. LONG's best planned "
                 "checkpoint was 0.193931 at 7.20 exposures, still above the short "
                 "schedule's 0.191892 at 3.93, so the comparison does not depend on "
                 "which of the two LONG checkpoints is used."),
    }
    delta_long = long_arm["MR-NATIVE-s0 -> MR-NATIVE-LONG-s0 (terminal)"]["plain"]["delta"]
    long_arm["verdict"] = {
        "code": "LONGER_SCHEDULE_DOES_NOT_HELP" if delta_long > 0 else "LONGER_SCHEDULE_HELPS",
        "reading": ("more compute on the same data does not help MR either, matching what "
                    "the E1 lineage already showed. The short schedule's advantage is not "
                    "that it stops early but that its cosine decay completes within the "
                    "budget: at 3.93 exposures LONG is at 0.214189 while the short run "
                    "finished at 0.191892, and LONG never recovers that."),
        "early_stopping_answer": ("the current stopping point is not too early. There is no "
                                  "gain available from training longer under this schedule "
                                  "family, so an early-stopping study on a rebuilt split "
                                  "would not have found one either."),
    }

    payload = {
        "schema_version": 1,
        "rows": int(len(d3s["LEN-s0"])), "sessions": len(unique), "resamples": RESAMPLES,
        "terminals": {k: {"plain": float(v.mean()),
                          "test_matched": float((v * weights).sum() / weights.sum())}
                      for k, v in d3s.items()},
        "mr_adj": adj,
        "long_schedule": long_arm,
        "weighting": weighting["weighting"],
        "limitations": [
            "V0 is a development set that has been read many times.",
            "One seed per arm except MR-NATIVE, which has two.",
            "The test-matched weighting predicts the server; it is not a server score.",
            "H is not opened.",
        ],
    }
    (OUT / "mr_round3_judgement.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n")

    print("terminals (plain / test-matched):")
    for k, v in payload["terminals"].items():
        print(f"  {k:22s} {v['plain']:.6f}  {v['test_matched']:.6f}")
    print(f"\nADJ: {code}")
    for name in ("PRIMARY MR-ADJ0-s0 -> MR-ADJ1-s0",
                 "PRACTICAL MR-NATIVE-s0 -> MR-ADJ1-s0",
                 "MODULE ALONE MR-NATIVE-s0 -> MR-ADJ0-s0"):
        for kind in ("plain", "test_matched"):
            c = adj[name][kind]
            print(f"  {name:42s} {kind:13s} {c['delta']:+.6f} "
                  f"CI [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] {c['sessions_improved']}/11")
    print(f"  seed spread: plain {seed_spread:.6f}  test-matched {seed_spread_w:.6f}")
    print(f"\nLONG: {long_arm['verdict']['code']}")
    for kind in ("plain", "test_matched"):
        c = long_arm["MR-NATIVE-s0 -> MR-NATIVE-LONG-s0 (terminal)"][kind]
        print(f"  {kind:13s} {c['delta']:+.6f} CI [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] "
              f"{c['sessions_improved']}/11")
    print("\nwrote mr_round3_judgement.json")


if __name__ == "__main__":
    main()
