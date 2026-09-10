import sys, numpy as np, torch
sys.path.insert(0, "experiments/pv_selector_20260910")
sys.path.insert(0, "experiments/c_refine_20260910")
import pv_selector as new
import c_scene_selector as old
from pv_status import load_status8

torch.manual_seed(0)
B, K = 5, 17
xy = torch.randn(B, K, 6, 2) * 3
scores = torch.randn(B, K)
valid = torch.ones(B, K, dtype=torch.bool); valid[0, 3] = False
ids = torch.arange(B * K, dtype=torch.int64).reshape(B, K)
goal = torch.randn(B, 2) * 10
out = dict(candidate_xy=xy, scores=scores, candidate_valid=valid, candidate_ids=ids)

a = old.build_features32(out, goal)
b = new.build_features32(out, goal)
print("status=None identical to frozen c_refine:", torch.equal(a, b), "maxabs", float((a - b).abs().max()))

z = torch.zeros(B, 8)
c = new.build_features32(out, goal, z)
print("explicit zero status identical:", torch.equal(a, c))

st = torch.zeros(B, 8); st[:, 4] = 8.0
d = new.build_features32(out, goal, st)
print("real status changes features:", not torch.equal(a, d),
      "| slot22 (vx/20):", float(d[0, 0, 22]), "expected", 8.0 / 20)
print("vel slot0 shift:", float(a[0, 0, 0] - d[0, 0, 0]), "expected", 8.0)

for split, cache, n in (("tune", "cache/c_refine_20260910/c_p20v64_tune_full_v1/tune", 1998),
                        ("train", "cache/c_refine_20260910/c_p20v64_train_full_v1/train", 54810)):
    rows = np.load(cache + "/rows.npy", allow_pickle=False)
    s8, prov = load_status8(cache, split, rows)
    v = np.hypot(s8[:, 4], s8[:, 5])
    print("%-5s n=%-6d shape=%s slots0:4 all zero=%s speed m/s min/med/max %.2f/%.2f/%.2f  ax med %.3f"
          % (split, len(rows), s8.shape, bool((s8[:, :4] == 0).all()),
             v.min(), float(np.median(v)), v.max(), float(np.median(s8[:, 6]))))
    assert s8.shape == (n, 8) and np.isfinite(s8).all()

head = new.SceneResidualSelector(mode="real")
out2 = dict(out, candidate_tokens=torch.randn(B, K, 256))
r1 = head(out2, goal_xy=goal)
r2 = head(out2, goal_xy=goal, status=torch.zeros(B, 8))
print("head zero-init reproduces base scores:", torch.equal(r1["scores"], scores),
      "| status=None == explicit zeros:", torch.equal(r1["scores"], r2["scores"]))
print("PREFLIGHT OK")
