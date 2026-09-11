import json, numpy as np
W = np.array([11,11,5,5,2,2], dtype=np.float64)/36.0
d = json.load(open("work_dirs/motiondrive_v2/q10_q10_flip50_s0/final_eval.json"))
rec = d["records"]
pred = np.array([r["pred_abs_xy"] for r in rec]); gt = np.array([r["gt_abs_xy"] for r in rec])
gs = np.array([r["gt_state"] for r in rec]); ps = np.array([r["pred_state"] for r in rec])
sess = np.array([r["session"] for r in rec]); n = len(pred)
err = pred - gt; l2 = np.linalg.norm(err, -1); d3 = (l2*W).sum(-1)
tan = np.diff(gt, axis=1, prepend=np.zeros_like(gt[:, :1]))
nr = np.linalg.norm(tan, -1, keepdims=True)
tan = np.divide(tan, nr, out=np.zeros_like(tan), where=nr > 1e-6)
along = (err*tan).sum(-1); signed = (along*W).sum(-1)

def arclen(p): return np.linalg.norm(np.diff(p,axis=1,prepend=np.zeros_like(p[:,:1])),-1).sum(-1)
lp, lg = arclen(pred), arclen(gt)
moving = lg > 1.0
print("path length over the %d moving rows: pred %.3f m, gt %.3f m, median ratio %.4f, longer on %.1f%%"
      % (moving.sum(), lp[moving].mean(), lg[moving].mean(),
         np.median(lp[moving]/lg[moving]), 100*(lp[moving] > lg[moving]).mean()))

speed = np.linalg.norm(gs[:, :2], -1)
print("\nmean-reversion check (signed along-track, + = predicted too far):")
print("  corr(signed, gt speed - mean)      %+.3f" % np.corrcoef(signed, speed)[0,1])
print("  corr(signed, gt accel)             %+.3f" % np.corrcoef(signed, gs[:,2])[0,1])
print("  gt speed mean %.2f m/s; predicted speed mean %.2f m/s" % (speed.mean(), np.linalg.norm(ps[:, :2], -1).mean()))
q = np.quantile(speed, [0, .2, .4, .6, .8, 1.])
print("  %-16s %6s %9s" % ("gt speed quintile", "rows", "signed"))
for i in range(5):
    m = (speed >= q[i]) & (speed <= q[i+1] if i == 4 else speed < q[i+1])
    print("  %4.1f - %4.1f m/s   %6d %+9.4f" % (q[i], q[i+1], m.sum(), signed[m].mean()))

# what would be left if specific slices were made perfect
def without(mask, label):
    kept = d3.copy(); kept[mask] = 0.
    print("  %-34s %6d rows, D3 -> %.4f" % (label, mask.sum(), kept.mean()))
print("\nif a slice were solved exactly (rest unchanged):")
without(speed < 0.5, "stopped rows perfect")
heading = np.abs(np.arctan2(gt[:,-1,1], np.maximum(gt[:,-1,0],1e-6)))
without(heading >= np.deg2rad(15), "turning rows (>=15 deg) perfect")
without(sess == "session_092_20260219-103747", "worst session perfect")
order = np.argsort(-d3); m = np.zeros(n, bool); m[order[:int(n*.05)]] = True
without(m, "worst 5% perfect")

# timing vs geometry ceiling
s = np.linalg.norm(np.diff(pred,axis=1,prepend=np.zeros_like(pred[:,:1])),-1).cumsum(-1)
s0 = np.concatenate([np.zeros((n,1)), s], 1)
p0 = np.concatenate([np.zeros((n,1,2)), pred], 1)
best = np.full(n, np.inf); bestscale = np.ones(n)
for scale in np.linspace(0.3, 2.0, 171):
    cand = np.stack([[np.interp(s[i,j]*scale, s0[i], p0[i,:,k]) for j in range(6)] for i in range(n) for k in (0,)], 0)
    cy = np.stack([[np.interp(s[i,j]*scale, s0[i], p0[i,:,1]) for j in range(6)] for i in range(n)], 0)
    c = np.stack([cand.reshape(n,6), cy], -1)
    v = (np.linalg.norm(c-gt,-1)*W).sum(-1)
    bestscale = np.where(v < best, scale, bestscale); best = np.minimum(best, v)
print("\nkeep the predicted shape, retime it optimally per row:")
print("  D3 %.4f (from %.4f); best scale median %.3f, 10-90%% %.3f-%.3f"
      % (best.mean(), d3.mean(), np.median(bestscale), *np.percentile(bestscale,[10,90])))
print("  so timing carries %.0f%% of the error and shape carries %.0f%%"
      % (100*(d3.mean()-best.mean())/d3.mean(), 100*best.mean()/d3.mean()))
