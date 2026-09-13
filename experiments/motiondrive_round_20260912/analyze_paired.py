"""Session-level paired bootstrap across the round's arms, on identical rows."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
WORK = ROOT / "work_dirs/motiondrive_round_20260912"
W = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0


def from_eval_json(path):
    payload = json.loads(Path(path).read_text())
    records = payload["records"]
    return (np.array([r["row"] for r in records], dtype=np.int64),
            np.array([r["d3"] for r in records], dtype=np.float64),
            np.array([r["session"] for r in records]))


def from_selector_npz(path):
    z = np.load(path, allow_pickle=False)
    return z["row"].astype(np.int64), z["d3"].astype(np.float64), z["session"]


def collect():
    arms = {}
    parent = ROOT / "work_dirs/motiondrive_v2/q10_q10_flip50_s0/final_eval.json"
    arms["R0"] = from_eval_json(parent)
    for name in ("c2", "c4", "m8_gen", "dyn0", "dyn1"):
        path = WORK / name / "final_eval.json"
        if path.exists():
            arms[name] = from_eval_json(path)
    for name in ("m8_sel0", "m8_selg"):
        path = WORK / name / "eval_004000.npz"
        if path.exists():
            arms[name] = from_selector_npz(path)
    return arms


def paired(values, sessions, a, b, iterations=20000, seed=0):
    difference = values[b] - values[a]
    groups = [difference[sessions == s] for s in np.unique(sessions)]
    rng = np.random.default_rng(seed)
    draws = np.array([np.concatenate([groups[j] for j in
                                      rng.integers(0, len(groups), len(groups))]).mean()
                      for _ in range(iterations)])
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"delta": float(difference.mean()), "sd": float(draws.std()),
            "ci95": [float(low), float(high)], "p_worse": float((draws > 0).mean()),
            "sessions_improved": int(sum(g.mean() < 0 for g in groups)),
            "sessions": len(groups)}


def main():
    arms = collect()
    reference = arms["R0"][0]
    order = np.argsort(reference)
    sessions = arms["R0"][2][order]
    values = {}
    for name, (rows, d3, _s) in arms.items():
        index = np.argsort(rows)
        if not np.array_equal(rows[index], reference[order]):
            raise SystemExit(f"{name} does not cover the same rows")
        values[name] = d3[index]
    table = {name: {"mean_d3": float(v.mean()), "rows": int(len(v))}
             for name, v in values.items()}
    comparisons = {}
    for a, b in (("R0", "c2"), ("c2", "c4"), ("c2", "dyn0"), ("dyn0", "dyn1"),
                 ("c2", "dyn1"), ("c2", "m8_gen"), ("c2", "m8_sel0"),
                 ("m8_sel0", "m8_selg"), ("c2", "m8_selg")):
        if a in values and b in values:
            comparisons[f"{a} -> {b}"] = paired(values, sessions, a, b)
    result = {"arms": table, "paired": comparisons,
              "note": ("tune is a repeatedly reused development split; these intervals "
                       "cover evaluation sampling only, not training-seed variance")}
    print(json.dumps(result, indent=1))
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
