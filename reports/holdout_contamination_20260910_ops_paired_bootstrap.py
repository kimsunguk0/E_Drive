import json, numpy as np
from pathlib import Path

def load(name):
    r = json.load(open(f"/tmp/tuneeval_{name}/evaluation.json"))["records"]
    return (np.array([x["d3"] for x in r]),
            np.array([x["session"] for x in r]))

runs = {n: load(n) for n in ["controlflow_b0_direct_last4000",
                             "datascale_scale15", "datascale_scale60",
                             "flipscreen_noflip60", "flipscreen_flip60"]}
ref_sessions = runs["controlflow_b0_direct_last4000"][1]
for n, (_d, s) in runs.items():
    assert np.array_equal(s, ref_sessions), n
sessions = np.unique(ref_sessions)
print("tune rows %d, sessions %d" % (len(ref_sessions), len(sessions)))

def paired(a, b, iters=20000, seed=0):
    """Session-level bootstrap of the mean per-row difference b - a."""
    da, db = runs[a][0], runs[b][0]
    diff = db - da
    groups = [diff[ref_sessions == s] for s in sessions]
    point = diff.mean()
    rng = np.random.default_rng(seed)
    draws = np.empty(iters)
    for i in range(iters):
        pick = rng.integers(0, len(groups), len(groups))
        draws[i] = np.concatenate([groups[j] for j in pick]).mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, draws.std(), lo, hi, float((draws > 0).mean())

print("\npaired session bootstrap of (B - A) on tune, 20k draws")
print("%-42s %9s %8s %19s %8s" % ("comparison", "delta", "sd", "95% CI", "P(>0)"))
for a, b in [("controlflow_b0_direct_last4000", "flipscreen_flip60"),
             ("flipscreen_noflip60", "flipscreen_flip60"),
             ("controlflow_b0_direct_last4000", "flipscreen_noflip60"),
             ("datascale_scale15", "datascale_scale60")]:
    p, sd, lo, hi, pg = paired(a, b)
    short = f"{a.split('_')[-1]} -> {b.split('_')[-1]}"
    print("%-42s %+9.5f %8.5f  [%+.5f,%+.5f] %7.3f" % (short, p, sd, lo, hi, pg))

print("\nabsolute session-bootstrap sd of each model on tune (for contrast)")
for n, (d, _s) in runs.items():
    groups = [d[ref_sessions == s] for s in sessions]
    rng = np.random.default_rng(1)
    draws = np.array([np.concatenate([groups[j] for j in
                      rng.integers(0, len(groups), len(groups))]).mean()
                      for _ in range(4000)])
    print("  %-34s mean %.5f  sd %.5f" % (n, d.mean(), draws.std()))
