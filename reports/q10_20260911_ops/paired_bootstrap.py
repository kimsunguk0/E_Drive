import json, numpy as np
from pathlib import Path
R = Path("work_dirs/motiondrive_v2")

def rec(name):
    d = json.loads((R / name / "final_eval.json").read_text())
    r = d.get("records") or (d.get("report") or {}).get("records")
    return {x["scenario"] + "/" + str(x["frame"]): x["d3"] for x in r}, \
           {x["scenario"] + "/" + str(x["frame"]): x["session"] for x in r}

arms = {"q10_noflip": "q10_q10_noflip_s0", "q10_flip50_s0": "q10_q10_flip50_s0",
        "q10_flip50_s1": "q10_q10_flip50_s1", "a2_noflip": "flipfull_nofull_s0",
        "a2_flip50_s0": "flipfull_flip50_s0", "a2_flip50_s1": "flipfull_flip50_s1"}
data, sess = {}, None
for k, n in arms.items():
    try:
        d, s = rec(n)
    except Exception as e:
        print("missing", k, n, type(e).__name__); continue
    data[k] = d
    sess = sess or s
keys = sorted(set.intersection(*[set(v) for v in data.values()]))
print("common rows %d, sessions %d" % (len(keys), len(set(sess[k] for k in keys))))
sidx = np.array([sess[k] for k in keys])
vals = {k: np.array([data[k][x] for x in keys]) for k in data}
groups_of = [sidx == s for s in np.unique(sidx)]

def paired(a, b, iters=20000):
    diff = vals[b] - vals[a]
    g = [diff[m] for m in groups_of]
    rng = np.random.default_rng(0)
    draws = np.array([np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]).mean()
                      for _ in range(iters)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return diff.mean(), draws.std(), lo, hi, float((draws > 0).mean())

print("\n%-34s %9s %8s %20s %6s" % ("comparison", "delta", "sd", "95% CI", "P(>0)"))
for a, b in [("q10_noflip", "q10_flip50_s0"), ("a2_noflip", "a2_flip50_s0"),
             ("a2_flip50_s0", "q10_flip50_s0"), ("q10_flip50_s0", "q10_flip50_s1")]:
    if a in vals and b in vals:
        p, sd, lo, hi, pg = paired(a, b)
        print("%-34s %+9.5f %8.5f  [%+.5f,%+.5f] %6.3f" % (a + " -> " + b, p, sd, lo, hi, pg))
print()
for k in sorted(vals): print("  %-16s %.6f" % (k, vals[k].mean()))
