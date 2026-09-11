"""Decompose the Q10+flip tune D3 into time, geometry and regime."""
import json, sys
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "work_dirs/motiondrive_v2/q10_q10_flip50_s0"
W = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0
T = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

d = json.load(open(f"{RUN}/final_eval.json"))
rec = d["records"]
pred = np.array([r["pred_abs_xy"] for r in rec], dtype=np.float64)   # [N,6,2]
gt = np.array([r["gt_abs_xy"] for r in rec], dtype=np.float64)
sess = np.array([r["session"] for r in rec])
gtstate = np.array([r["gt_state"] for r in rec], dtype=np.float64)
n = len(pred)
err = pred - gt
l2 = np.linalg.norm(err, axis=-1)                                     # [N,6]
d3 = (l2 * W).sum(-1)
print("rows %d  sessions %d  official D3 %.6f" % (n, len(np.unique(sess)), d3.mean()))

# --- per-waypoint -----------------------------------------------------------
print("\n%-6s %-8s %-9s %-10s %-9s" % ("t(s)", "weight", "L2 mean", "share of D3", "L2 p90"))
for i, t in enumerate(T):
    contrib = W[i] * l2[:, i].mean()
    print("%-6.1f %-8.4f %-9.4f %-10.4f %-9.4f"
          % (t, W[i], l2[:, i].mean(), contrib / d3.mean(), np.percentile(l2[:, i], 90)))

# --- along / cross track ----------------------------------------------------
tangent = np.diff(gt, axis=1, prepend=np.zeros_like(gt[:, :1]))
norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
tangent = np.divide(tangent, norm, out=np.zeros_like(tangent), where=norm > 1e-6)
normal = np.stack([-tangent[..., 1], tangent[..., 0]], -1)
along = (err * tangent).sum(-1)                                       # signed: + = overshoot
cross = (err * normal).sum(-1)
wa, wc = (np.abs(along) * W).sum(-1), (np.abs(cross) * W).sum(-1)
energy = ((along ** 2) * W).sum(-1).sum() / (((along ** 2 + cross ** 2) * W).sum(-1).sum())
print("\nweighted |along| %.4f m, weighted |cross| %.4f m, along share of squared error %.1f%%"
      % (wa.mean(), wc.mean(), 100 * energy))
signed = (along * W).sum(-1)
print("signed along-track, weighted: mean %+.4f m, median %+.4f m, overshoot on %.1f%% of rows"
      % (signed.mean(), np.median(signed), 100 * (signed > 0).mean()))
print("%-6s %-11s %-11s %-11s" % ("t(s)", "along mean", "along med", "|cross| mean"))
for i, t in enumerate(T):
    print("%-6.1f %+11.4f %+11.4f %11.4f" % (t, along[:, i].mean(), np.median(along[:, i]),
                                             np.abs(cross[:, i]).mean()))

# --- path length ------------------------------------------------------------
def arclen(p):
    return np.linalg.norm(np.diff(p, axis=1, prepend=np.zeros_like(p[:, :1])), axis=-1).sum(-1)
lp, lg = arclen(pred), arclen(gt)
print("\npath length: pred %.3f m, gt %.3f m, ratio %.4f, longer on %.1f%% of rows"
      % (lp.mean(), lg.mean(), (lp / np.maximum(lg, 1e-6)).mean(), 100 * (lp > lg).mean()))

# --- regimes ----------------------------------------------------------------
speed = np.linalg.norm(gtstate[:, :2], axis=-1)
accel = gtstate[:, 2]
heading = np.abs(np.arctan2(gt[:, -1, 1], np.maximum(gt[:, -1, 0], 1e-6)))
def table(name, groups):
    print("\n%-22s %6s %8s %9s %9s %9s" % (name, "rows", "D3", "|along|", "|cross|", "signed"))
    for label, m in groups:
        if m.sum() == 0: continue
        print("%-22s %6d %8.4f %9.4f %9.4f %+9.4f"
              % (label, m.sum(), d3[m].mean(), wa[m].mean(), wc[m].mean(), signed[m].mean()))
table("by GT speed (m/s)", [
    ("stopped  < 0.5", speed < 0.5), ("slow     0.5-5", (speed >= .5) & (speed < 5)),
    ("mid      5-12", (speed >= 5) & (speed < 12)), ("fast     12-18", (speed >= 12) & (speed < 18)),
    ("highway  >= 18", speed >= 18)])
table("by GT accel (m/s^2)", [
    ("braking  < -0.5", accel < -0.5), ("steady   -0.5..0.5", np.abs(accel) <= .5),
    ("accel    > 0.5", accel > 0.5)])
table("by 3 s heading (deg)", [
    ("straight < 5", heading < np.deg2rad(5)), ("bend     5-15", (heading >= np.deg2rad(5)) & (heading < np.deg2rad(15))),
    ("turn     >= 15", heading >= np.deg2rad(15))])

# --- concentration ----------------------------------------------------------
order = np.argsort(-d3)
for frac in (0.05, 0.10, 0.25):
    k = int(n * frac)
    print("\nworst %2d%% of rows (%4d) carry %.1f%% of total D3, mean D3 %.4f"
          % (100 * frac, k, 100 * d3[order[:k]].sum() / d3.sum(), d3[order[:k]].mean()), end="")
print()
per = {s: d3[sess == s].mean() for s in np.unique(sess)}
worst = sorted(per.items(), key=lambda kv: -kv[1])[:3]
print("worst sessions:", ", ".join("%s %.4f" % (s.split("_")[1], v) for s, v in worst))
print("session spread: min %.4f, median %.4f, max %.4f"
      % (min(per.values()), np.median(list(per.values())), max(per.values())))

# --- what perfect timing would buy -----------------------------------------
best = np.full(n, np.inf)
for scale in np.linspace(0.5, 1.6, 111):
    cand = np.zeros_like(pred)
    s = np.linalg.norm(np.diff(pred, axis=1, prepend=np.zeros_like(pred[:, :1])), axis=-1).cumsum(-1)
    total = np.maximum(s[:, -1:], 1e-6)
    target = np.clip(s * scale, 0, None)
    for i in range(n):
        for j in range(6):
            cand[i, j] = np.interp(target[i, j], np.concatenate([[0.], s[i]]),
                                   np.concatenate([[0.], pred[i, :, 0]])), np.interp(
                         target[i, j], np.concatenate([[0.], s[i]]),
                                   np.concatenate([[0.], pred[i, :, 1]]))
    best = np.minimum(best, (np.linalg.norm(cand - gt, axis=-1) * W).sum(-1))
print("\nsame predicted shape, best per-row uniform speed rescale: D3 %.4f (from %.4f)"
      % (best.mean(), d3.mean()))
