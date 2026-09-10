"""Row selector with PER-CANDIDATE visual scoring.

Each candidate's ten waypoints carry BEV features sampled along its own path,
so the scorer can read the visual evidence that distinguishes candidates -
the piece the shared-latent selector was missing. Geometry features are as
before. Ego status is not an input. Negative controls zero or shuffle the
path features to test whether the visual pathway actually contributes.
"""
import json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
R = Path("/NHNHOME/data/sukim/adcl")
Wn = np.array([11, 11, 5, 5, 2, 2], np.float32)/36.
PC = R/"reports/selector_20260910_ops/pathcache"
def load(pat):
    fs = sorted(PC.glob(pat)); d = [np.load(f) for f in fs]
    o = {k: np.concatenate([x[k] for x in d]) for k in ("pred", "path", "top", "row", "goal")}
    i = np.argsort(o["row"]); return {k: v[i] for k, v in o.items()}
TR, TU = load("train_r*.npz"), load("tune_r*.npz")
bank = np.load(R/"reports/p7_train203_bank_feasibility_20260908_ops/bank/bank.npy")
ego = np.load("/tmp/pm97/data/etri/ego_cache.npz")
split = json.loads((R/"data/etri/motiondrive_v2/grouped_split_rawtime.json").read_text())
s2s = split["scene_to_session"]; scen = ego["scenarios"]
def build(d):
    P = d["pred"].astype(np.float32); N = len(P); M = 13
    cand = bank[d["top"].astype(int)]
    step = P[:, 5] - P[:, 4]
    tail = P[:, 5][:, None] + step[:, None]*np.arange(1, 5)[None, :, None]
    own = np.concatenate([P, tail], 1)[:, None]
    cand = np.concatenate([own, cand], 1).astype(np.float32)          # (N,13,10,2)
    G = ego["fut"][d["row"]].astype(np.float32)
    err = (np.linalg.norm(cand[:, :, :6, :]-G[:, None], axis=3)*Wn).sum(2)
    goal = d["goal"].astype(np.float32)
    def al(X):
        q = np.concatenate([np.zeros((*X.shape[:-2], 1, 2), X.dtype), X], -2)
        return np.cumsum(np.linalg.norm(np.diff(q, axis=-2), axis=-1), -1)
    sp, sc = al(P), al(cand); e5 = cand[:, :, 9, :]
    own_flag = np.concatenate([np.ones((N, 1, 1), np.float32), np.zeros((N, 12, 1), np.float32)], 1)
    feat = np.concatenate([
        cand[:, :, :6, :].reshape(N, M, 12), sc[:, :, [2, 5, 9]],
        (np.linalg.norm(cand[:, :, :6, :]-P[:, None], axis=3)*Wn),
        (np.linalg.norm(cand[:, :, :6, :]-P[:, None], axis=3)*Wn).sum(2, keepdims=True),
        (sc[:, :, 5]-sp[:, 5:6])[:, :, None],
        np.linalg.norm(e5-goal[:, None], axis=2)[:, :, None],
        (np.arctan2(e5[:, :, 1], e5[:, :, 0])-np.arctan2(goal[:, 1], goal[:, 0])[:, None])[:, :, None],
        np.repeat(np.linalg.norm(goal, axis=1)[:, None, None], M, 1),
        np.repeat(np.arctan2(goal[:, 1], goal[:, 0])[:, None, None], M, 1),
        own_flag], 2).astype(np.float32)
    return feat, err, d["path"].astype(np.float32)                    # path (N,13,10,C)
Ftr, Etr, Ptr = build(TR); Ftu, Etu, Ptu = build(TU)
print("geom %s  path %s" % (Ftr.shape, Ptr.shape), flush=True)
print("shortlist oracle: train %.4f  tune %.4f | direct baseline 0.280640" % (
    Etr.min(1).mean(), Etu.min(1).mean()), flush=True)
mu, sd = Ftr.reshape(-1, Ftr.shape[2]).mean(0), Ftr.reshape(-1, Ftr.shape[2]).std(0)+1e-6
pmu, psd = Ptr.reshape(-1, Ptr.shape[3]).mean(0), Ptr.reshape(-1, Ptr.shape[3]).std(0)+1e-6
class Sel(nn.Module):
    def __init__(s, fd, cd):
        super().__init__()
        s.p = nn.Sequential(nn.Linear(cd, 128), nn.GELU(), nn.Linear(128, 64))
        s.g = nn.GRU(64, 64, batch_first=True)
        s.h = nn.Sequential(nn.Linear(fd+64, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(s, Fg, Pa):
        B, M, T, C = Pa.shape
        z = s.p(Pa.reshape(B*M, T, C))
        _, hn = s.g(z)
        v = hn[-1].reshape(B, M, -1)
        return s.h(torch.cat([Fg, v], -1)).squeeze(-1)
def run(tri, evi, Fa, Ea, Pa, Fb, Eb, Pb, mode="normal", epochs=10):
    dev = torch.device("cuda:0"); torch.manual_seed(0)
    net = Sel(Fa.shape[2], Pa.shape[3]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    F1 = torch.tensor((Fa[tri]-mu)/sd); E1 = torch.tensor(Ea[tri]); P1 = torch.tensor((Pa[tri]-pmu)/psd)
    F2 = torch.tensor((Fb[evi]-mu)/sd); E2 = torch.tensor(Eb[evi]); P2 = torch.tensor((Pb[evi]-pmu)/psd)
    if mode == "zero": P1 = torch.zeros_like(P1); P2 = torch.zeros_like(P2)
    if mode == "shuffle": P2 = P2[torch.randperm(len(P2))]
    for ep in range(epochs):
        perm = torch.randperm(len(F1))
        for i in range(0, len(F1), 256):
            b = perm[i:i+256]
            f, e, p = F1[b].to(dev), E1[b].to(dev), P1[b].to(dev)
            lg = net(f, p); pr = lg.softmax(-1)
            loss = (pr*e).sum(-1).mean() + 0.1*nn.functional.cross_entropy(lg, e.argmin(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    outs = []
    with torch.no_grad():
        for i in range(0, len(F2), 512):
            outs.append(net(F2[i:i+512].to(dev), P2[i:i+512].to(dev)).argmax(-1).cpu())
    pick = torch.cat(outs)
    return E2[torch.arange(len(E2)), pick].mean().item(), E2.min(1).values.mean().item()
print("\n=== train203 -> tune1998 (single look) ===", flush=True)
at, au = np.arange(len(Ftr)), np.arange(len(Ftu))
for mode in ("normal", "zero", "shuffle"):
    sel, orc = run(at, au, Ftr, Etr, Ptr, Ftu, Etu, Ptu, mode=mode)
    print("  path features %-8s : tune %.4f  (oracle %.4f, regret %.4f)  vs direct 0.2806" % (
        mode, sel, orc, sel-orc), flush=True)
