"""Learned row selector over an M12 prefix-nearest shortlist.

Follows the PHASE5EFG design that worked: the twelve candidate rows are already
complete and the model picks only the row index. Inputs are goal geometry, the
candidate's own geometry, its relation to the direct prediction (this is where
the trunk's visual evidence enters), the planner latents, and command.
Ego status is never a feature. Session-grouped 5-fold CV on train203; tune1998
is scored once at the end.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
R = Path("/NHNHOME/data/sukim/adcl")
W = torch.tensor([11, 11, 5, 5, 2, 2], dtype=torch.float32)/36.
Wn = W.numpy()
C = R/"reports/selector_20260910_ops/cache"
def load(pat):
    fs = sorted(C.glob(pat)); d = [np.load(f) for f in fs]
    out = {k: np.concatenate([x[k] for x in d]) for k in ("pred", "latent", "row", "goal")}
    o = np.argsort(out["row"])
    return {k: v[o] for k, v in out.items()}
TR, TU = load("train_r*.npz"), load("tune_r*.npz")
bank = np.load(R/"reports/p7_train203_bank_feasibility_20260908_ops/bank/bank.npy")  # (1024,10,2)
ego = np.load("/tmp/pm97/data/etri/ego_cache.npz")
split = json.loads((R/"data/etri/motiondrive_v2/grouped_split_rawtime.json").read_text())
s2s = split["scene_to_session"]; scen = ego["scenarios"]
M = 12
MC = M + 1   # plus the model's own prediction
def build(d):
    P = d["pred"].astype(np.float32)                       # (N,6,2)
    B3 = bank[:, :6, :]                                    # (1024,6,2)
    # prefix-nearest shortlist under the official weights
    dist = np.einsum("k,nkc->n", np.ones(1), np.zeros((len(P), 1, 1)))  # placeholder
    diff = P[:, None, :, :] - B3[None]                     # (N,1024,6,2)
    dd = (np.linalg.norm(diff, axis=3)*Wn).sum(2)          # (N,1024)
    top = np.argsort(dd, axis=1)[:, :M]                    # (N,12)
    cand = bank[top]                                       # (N,12,10,2)
    # candidate 0 is the model's own direct prediction, extended to 10 points by
    # continuing the final heading. The selector can therefore always fall back.
    d5 = P[:, 5] - P[:, 4]; nrm = np.linalg.norm(d5, axis=1, keepdims=True)
    step = d5/np.maximum(nrm, 1e-6)*nrm
    tail = P[:, 5][:, None, :] + step[:, None, :]*np.arange(1, 5)[None, :, None]
    own = np.concatenate([P, tail], 1)[:, None]            # (N,1,10,2)
    cand = np.concatenate([own, cand], 1)                  # (N,13,10,2)
    G = ego["fut"][d["row"]].astype(np.float32)
    err = (np.linalg.norm(cand[:, :, :6, :] - G[:, None], axis=3)*Wn).sum(2)   # (N,12) true D3
    goal = d["goal"].astype(np.float32)
    def arclen(X):
        q = np.concatenate([np.zeros((*X.shape[:-2], 1, 2), X.dtype), X], -2)
        return np.cumsum(np.linalg.norm(np.diff(q, axis=-2), axis=-1), -1)
    sp = arclen(P); sc = arclen(cand)                      # (N,6) (N,12,10)
    end5 = cand[:, :, 9, :]
    feat = np.concatenate([
        cand[:, :, :6, :].reshape(len(P), MC, 12),          # candidate waypoints
        sc[:, :, [2, 5, 9]],                               # s at 1.5s, 3s, 5s
        (np.linalg.norm(cand[:, :, :6, :] - P[:, None], axis=3)*Wn),   # per-wp gap to prediction
        np.concatenate([np.zeros((len(P),1),np.float32),
                        dd[np.arange(len(P))[:, None], top]],1)[:, :, None],
        (sc[:, :, 5] - sp[:, 5:6])[:, :, None],            # s3 difference
        np.linalg.norm(end5 - goal[:, None], axis=2)[:, :, None],      # endpoint-to-goal
        (np.arctan2(end5[:, :, 1], end5[:, :, 0]) -
         np.arctan2(goal[:, 1], goal[:, 0])[:, None])[:, :, None],     # heading vs goal
        np.repeat(np.linalg.norm(goal, axis=1)[:, None, None], MC, 1),
        np.repeat(np.arctan2(goal[:, 1], goal[:, 0])[:, None, None], MC, 1),
        np.concatenate([np.ones((len(P),1,1),np.float32),
                        np.zeros((len(P),M,1),np.float32)],1),   # is-own-prediction flag
    ], axis=2).astype(np.float32)
    return feat, err, d["latent"].reshape(len(P), -1).astype(np.float32), top
Ftr, Etr, Ztr, _ = build(TR); Ftu, Etu, Ztu, _ = build(TU)
print("features %s  candidates %d (incl. own prediction)  latent %d" % (Ftr.shape, MC, Ztr.shape[1]), flush=True)
print("train shortlist oracle %.4f | tune shortlist oracle %.4f" % (Etr.min(1).mean(), Etu.min(1).mean()))
print("tune direct baseline 0.280640", flush=True)
mu, sd = Ftr.reshape(-1, Ftr.shape[2]).mean(0), Ftr.reshape(-1, Ftr.shape[2]).std(0)+1e-6
zmu, zsd = Ztr.mean(0), Ztr.std(0)+1e-6
class Sel(nn.Module):
    def __init__(s, fd, zd):
        super().__init__()
        s.z = nn.Sequential(nn.Linear(zd, 128), nn.GELU(), nn.Linear(128, 64))
        s.f = nn.Sequential(nn.Linear(fd+64, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(s, F, Z):
        z = s.z(Z)[:, None, :].expand(-1, F.shape[1], -1)
        return s.f(torch.cat([F, z], -1)).squeeze(-1)
def run(tri, evi, Fa, Ea, Za, Fb, Eb, Zb, zero_latent=False, shuffle_latent=False, epochs=12, tag=""):
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    net = Sel(Fa.shape[2], Za.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    F1 = torch.tensor((Fa[tri]-mu)/sd); E1 = torch.tensor(Ea[tri]); Z1 = torch.tensor((Za[tri]-zmu)/zsd)
    F2 = torch.tensor((Fb[evi]-mu)/sd); E2 = torch.tensor(Eb[evi]); Z2 = torch.tensor((Zb[evi]-zmu)/zsd)
    if zero_latent: Z1 = torch.zeros_like(Z1); Z2 = torch.zeros_like(Z2)
    if shuffle_latent: Z2 = Z2[torch.randperm(len(Z2))]
    for ep in range(epochs):
        perm = torch.randperm(len(F1))
        for i in range(0, len(F1), 512):
            b = perm[i:i+512]
            f, e, z = F1[b].to(dev), E1[b].to(dev), Z1[b].to(dev)
            logit = net(f, z); p = logit.softmax(-1)
            loss = (p*e).sum(-1).mean() + 0.1*nn.functional.cross_entropy(logit, e.argmin(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pick = net(F2.to(dev), Z2.to(dev)).argmax(-1).cpu()
    sel = E2[torch.arange(len(E2)), pick].mean().item()
    return sel, E2.min(1).values.mean().item()
rows_tr = TR["row"]; sess = np.array([s2s[scen[ego["scen_idx"][r]]] for r in rows_tr])
uniq = np.array(sorted(set(sess))); rng = np.random.RandomState(0); rng.shuffle(uniq)
folds = np.array_split(uniq, 5)
print("\n=== session-grouped 5-fold CV on train203 ===", flush=True)
res = []
for i, f in enumerate(folds):
    m = np.isin(sess, f); tri = np.where(~m)[0]; evi = np.where(m)[0]
    sel, orc = run(tri, evi, Ftr, Etr, Ztr, Ftr, Etr, Ztr)
    res.append((sel, orc)); print("  fold %d  selected %.4f  oracle %.4f  regret %.4f" % (i, sel, orc, sel-orc), flush=True)
sm = np.mean([r[0] for r in res]); om = np.mean([r[1] for r in res])
print("  CV mean selected %.4f  oracle %.4f  REGRET %.4f" % (sm, om, sm-om))
print("\n=== train203 -> tune1998 (single look) ===", flush=True)
allt = np.arange(len(Ftr)); allu = np.arange(len(Ftu))
for tag, kw in (("normal", {}), ("latent zeroed", {"zero_latent": True}), ("latent shuffled", {"shuffle_latent": True})):
    sel, orc = run(allt, allu, Ftr, Etr, Ztr, Ftu, Etu, Ztu, **kw)
    print("  %-16s tune selected %.4f  (oracle %.4f, regret %.4f)  vs direct 0.2806" % (tag, sel, orc, sel-orc), flush=True)
