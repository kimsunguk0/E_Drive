#!/usr/bin/env python3
"""Round-2 judgement: does the MR gain replicate, and does a wider descriptor help?

Two separate questions, judged separately.

  replication  MR-NATIVE seed 1 against LEN seed 1, the matching control, and
               against MR-NATIVE seed 0. The seeds are NOT required to agree;
               what is asked is whether the gain survives a second seed.
  MR-W64       MR-W64 seed 0 against MR-NATIVE seed 0, the matched-budget,
               matched-seed control. The measured seed spread of MR-NATIVE is
               reported beside it, because a difference smaller than that spread
               is not evidence about the descriptor.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT / "experiments/md_r0_reset_20260914"))
from analyze_mr import W, DT, REGIMES, EVAL, load, seg, regimes, read_metric

WORK = ROOT / "work_dirs/md_r0_reset_20260914"
OUT = ROOT / "reports/md_exp_diagnosis_20260915"
RESAMPLES = 20000
ALL_LABELS = REGIMES + ["transition"]

RUNS = {
    "LEN-s0": WORK / "E1-EXP-LEN-s0",
    "LEN-s1": WORK / "E1-EXP-LEN-s1",
    "MR-LOWDETAIL-s0": WORK / "MR-LOWDETAIL-s0",
    "MR-NATIVE-s0": WORK / "MR-NATIVE-s0",
    "MR-NATIVE-s1": WORK / "MR-NATIVE-s1",
    "MR-W64-s0": WORK / "MR-W64-s0",
}


def main() -> None:
    for name, run in RUNS.items():
        status = json.loads((run / "manifest.json").read_text())["status"]
        if status != "completed":
            raise SystemExit(f"{name} is {status}")

    reference = gt = label = sessions = None
    d3s, preds, states, gt_states = {}, {}, {}, {}
    for name, run in RUNS.items():
        key, pred, gt_r, session, state, gt_state = load(run / "final_eval.json")
        if reference is None:
            reference, gt, sessions = key, gt_r, session
            _, ell_g = seg(gt)
            label = regimes(ell_g)
        elif key != reference or not np.array_equal(gt_r, gt):
            raise SystemExit(f"{name} is not scored on the same rows or ground truth")
        d3s[name] = np.linalg.norm(pred - gt, axis=-1) @ W
        preds[name], states[name], gt_states[name] = pred, state, gt_state

    unique = sorted(set(sessions.tolist()))
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(unique), size=(RESAMPLES, len(unique)))
    picks = [np.concatenate([index[unique[j]] for j in row]) for row in draws]

    def compare(base, arm):
        diff = d3s[arm] - d3s[base]
        means = np.asarray([diff[p].mean() for p in picks])
        return {"base": base, "arm": arm, "delta": float(diff.mean()),
                "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                "ci_includes_zero": bool(np.percentile(means, 2.5) <= 0 <= np.percentile(means, 97.5)),
                "p_worse": float((means > 0).mean()),
                "sessions_improved": sum(1 for s in unique if diff[index[s]].mean() < 0),
                "sessions": len(unique)}

    _, ell_g = seg(gt)

    def per_regime(name):
        pred = preds[name]
        _, ell_p = seg(pred)
        b = ((ell_p - ell_g) / DT).mean(1)
        d3 = d3s[name]
        out = {}
        for g in ALL_LABELS:
            m = label == g
            out[g] = ({"rows": 0} if not m.any() else {
                "rows": int(m.sum()), "mean_d3_m": float(d3[m].mean()),
                "contribution_to_overall_mean_m": float(d3[m].sum() / len(d3)),
                "signed_b_ms": float(b[m].mean())})
        return out

    terminals = {n: float(d3s[n].mean()) for n in RUNS}
    native_spread = abs(terminals["MR-NATIVE-s0"] - terminals["MR-NATIVE-s1"])
    len_spread = abs(terminals["LEN-s0"] - terminals["LEN-s1"])

    replication = {
        "question": "does the MR gain survive a second seed?",
        "LEN-s1 -> MR-NATIVE-s1": compare("LEN-s1", "MR-NATIVE-s1"),
        "LEN-s0 -> MR-NATIVE-s0 (round 1, for reference)": compare("LEN-s0", "MR-NATIVE-s0"),
        "MR-NATIVE-s0 -> MR-NATIVE-s1 (seed spread)": compare("MR-NATIVE-s0", "MR-NATIVE-s1"),
        "terminals": {n: terminals[n] for n in
                      ("LEN-s0", "LEN-s1", "MR-NATIVE-s0", "MR-NATIVE-s1")},
        "seed_spread": {"MR-NATIVE": native_spread, "LEN": len_spread},
    }
    a = replication["LEN-s1 -> MR-NATIVE-s1"]
    b = replication["LEN-s0 -> MR-NATIVE-s0 (round 1, for reference)"]
    replication["verdict"] = (
        "REPLICATED" if (a["delta"] < 0 and not a["ci_includes_zero"]
                         and b["delta"] < 0 and not b["ci_includes_zero"])
        else "NOT_REPLICATED")
    replication["reading"] = (
        "both seeds beat their own LEN control with a CI clear of zero, and the two "
        "MR-NATIVE seeds differ by %.6f, which is small beside the %.3f gain"
        % (native_spread, abs(a["delta"]))) if replication["verdict"] == "REPLICATED" else (
        "the second seed did not reproduce the first; find the cause before reading W64")

    w64 = {
        "question": "does a wider correspondence descriptor reduce the planning error?",
        "PRIMARY MR-NATIVE-s0 -> MR-W64-s0": compare("MR-NATIVE-s0", "MR-W64-s0"),
        "MR-NATIVE-s1 -> MR-W64-s0 (other seed, context only)": compare("MR-NATIVE-s1", "MR-W64-s0"),
        "LEN-s0 -> MR-W64-s0": compare("LEN-s0", "MR-W64-s0"),
        "terminals": {n: terminals[n] for n in ("MR-NATIVE-s0", "MR-NATIVE-s1", "MR-W64-s0")},
        "mr_native_seed_spread": native_spread,
    }
    p = w64["PRIMARY MR-NATIVE-s0 -> MR-W64-s0"]
    beats_spread = abs(p["delta"]) > native_spread
    reasons = []
    if p["ci_includes_zero"]:
        reasons.append("the paired CI of the primary comparison includes zero")
    if not beats_spread:
        reasons.append("the difference is smaller than the spread between two seeds of the "
                       "control itself")
    else:
        reasons.append("the difference is the same order as the control's own seed spread "
                       "(%.6f vs %.6f)" % (abs(p["delta"]), native_spread))
    if p["delta"] >= 0:
        reasons.append("the wider descriptor did not improve the mean at all")
    if p["delta"] < 0 and not p["ci_includes_zero"] and beats_spread:
        code, action = "WIDER_DESCRIPTOR_HELPS", (
            "the wider descriptor beats its matched control with a CI clear of zero and by "
            "more than the measured seed spread; it may be carried forward")
        reasons = ["CI clear of zero", "larger than the seed spread"]
    else:
        code, action = "NO_EFFECT_UNDER_THIS_SETTING", (
            "no established gain from widening the descriptor at this budget; do not sweep "
            "96/128 and do not read the sign of a difference this small")
    w64["verdict"] = {
        "code": code, "action": action, "reasons": reasons,
        "delta_vs_seed_spread": {"delta": p["delta"], "seed_spread": native_spread,
                                 "delta_exceeds_spread": bool(beats_spread)},
        "also_true": ("MR-W64 still beats LEN by -0.032753, because it inherits the whole "
                      "matching-graph change; that is not evidence about the descriptor"),
    }

    soup_path = EVAL / "MR-NATIVE-soup_tuneB1_step20554.json"
    soup = None
    if soup_path.exists():
        value = json.loads(soup_path.read_text())["metrics"]["official_d3_weighted"]
        soup = {"official_d3": value,
                "vs_MR-NATIVE-s0": value - terminals["MR-NATIVE-s0"],
                "vs_MR-NATIVE-s1": value - terminals["MR-NATIVE-s1"],
                "vs_better_seed": value - min(terminals["MR-NATIVE-s0"], terminals["MR-NATIVE-s1"]),
                "recipe": "reports/md_exp_diagnosis_20260915/mr_soup.json",
                "note": ("evaluation only; a weight average of two seeds of the SAME graph, "
                         "one model forward. Not a combination search.")}

    payload = {
        "schema_version": 1,
        "rows": int(len(d3s["LEN-s0"])),
        "sessions": len(unique),
        "resamples": RESAMPLES,
        "replication": replication,
        "mr_w64": w64,
        "weight_average": soup,
        "train_probe_T0": {n: read_metric(f"{n}_probe_step20554") for n in
                           ("E1-EXP-LEN-s0", "MR-LOWDETAIL-s0", "MR-NATIVE-s0",
                            "MR-NATIVE-s1", "MR-W64-s0")},
        "tune_b1": {n: read_metric(f"{n}_tuneB1_step20554") for n in
                    ("E1-EXP-LEN-s0", "MR-LOWDETAIL-s0", "MR-NATIVE-s0",
                     "MR-NATIVE-s1", "MR-W64-s0", "MR-NATIVE-soup")},
        "regimes": {n: per_regime(n) for n in RUNS},
        "state_slot0_mae_ms": {n: float(np.abs(states[n][:, 0] - gt_states[n][:, 0]).mean())
                               for n in RUNS},
        "limitations": [
            "V0 is a development set that has been read many times.",
            "One seed per arm except MR-NATIVE, which has two.",
            "H is not opened.",
        ],
    }
    (OUT / "mr_round2_judgement.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    with (OUT / "mr_round2_regimes.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        names = list(RUNS)
        wr.writerow(["regime", "rows"] + [f"{n}_{f}" for n in names
                                          for f in ("mean_d3_m", "contribution_m", "signed_b_ms")])
        for g in ALL_LABELS:
            base = payload["regimes"]["LEN-s0"][g]
            row = [g, base["rows"]]
            for n in names:
                e = payload["regimes"][n][g]
                row += ([""] * 3 if not e["rows"] else
                        [round(e["mean_d3_m"], 6), round(e["contribution_to_overall_mean_m"], 6),
                         round(e["signed_b_ms"], 6)])
            wr.writerow(row)

    print("terminals:")
    for n in RUNS:
        print(f"  {n:18s} {terminals[n]:.6f}")
    print(f"\nreplication: {replication['verdict']}")
    for k in ("LEN-s1 -> MR-NATIVE-s1", "LEN-s0 -> MR-NATIVE-s0 (round 1, for reference)",
              "MR-NATIVE-s0 -> MR-NATIVE-s1 (seed spread)"):
        c = replication[k]
        print(f"  {k}: {c['delta']:+.6f} CI [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] "
              f"{c['sessions_improved']}/{c['sessions']}")
    print(f"\nW64: {code}")
    for k in ("PRIMARY MR-NATIVE-s0 -> MR-W64-s0",
              "MR-NATIVE-s1 -> MR-W64-s0 (other seed, context only)", "LEN-s0 -> MR-W64-s0"):
        c = w64[k]
        print(f"  {k}: {c['delta']:+.6f} CI [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] "
              f"{c['sessions_improved']}/{c['sessions']}")
    print(f"  seed spread of the control: {native_spread:.6f}")
    if soup:
        print(f"\nweight average: {soup['official_d3']:.6f} "
              f"(vs better seed {soup['vs_better_seed']:+.6f})")
    print("\nwrote mr_round2_judgement.json, mr_round2_regimes.csv")


if __name__ == "__main__":
    main()
