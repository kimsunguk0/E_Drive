import json, numpy as np
from pathlib import Path

def rec(arm):
    p = Path(f"work_dirs/motiondrive_v2/flipfull_{arm}/final_eval.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    r = d.get("records") or (d.get("report") or {}).get("records")
    if not r:
        return None
    return np.array([x["d3"] for x in r]), np.array([x["session"] for x in r])

arms = ["nofull_s0", "flip25_s0", "flip50_s0", "flip50_s1"]
got = {a: rec(a) for a in arms}
missing = [a for a, v in got.items() if v is None]
if missing:
    print("no per-record data for:", missing)
    raise SystemExit(0)

sess = got["nofull_s0"][1]
groups_idx = [sess == s for s in np.unique(sess)]
print("rows %d sessions %d" % (len(sess), len(groups_idx)))

def paired(a, b, iters=20000, seed=0):
    diff = got[b][0] - got[a][0]
    groups = [diff[m] for m in groups_idx]
    rng = np.random.default_rng(seed)
    draws = np.array([np.concatenate([groups[j] for j in
                     rng.integers(0, len(groups), len(groups))]).mean()
                     for _ in range(iters)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return diff.mean(), draws.std(), lo, hi, float((draws > 0).mean())

print("%-26s %9s %8s %20s %7s" % ("comparison", "delta", "sd", "95% CI", "P(>0)"))
for a, b in [("nofull_s0", "flip25_s0"), ("nofull_s0", "flip50_s0"),
             ("nofull_s0", "flip50_s1"), ("flip50_s0", "flip50_s1"),
             ("flip25_s0", "flip50_s0")]:
    p, sd, lo, hi, pg = paired(a, b)
    print("%-26s %+9.5f %8.5f  [%+.5f,%+.5f] %6.3f" % (f"{a} -> {b}", p, sd, lo, hi, pg))
