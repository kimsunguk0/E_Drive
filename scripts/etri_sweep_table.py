#!/usr/bin/env python
"""B200 스윕 결과를 산출물 파일에서 직접 읽어 markdown 표로 낸다.

손 전사 금지(부록6 항목3). 모든 수치는 /tmp/pm97/eval_b200/ 아래
fullval_*.json / gate_*.txt / t6ppp_*.txt / x2x2_*.txt 에서 파싱한다.
"""
import glob, json, os, re, sys
import numpy as np

E = "/tmp/pm97/eval_b200"
ADCL = "/home/pm97/workspace/sukim/adcl"
sys.path.insert(0, os.path.join(ADCL, "scripts")); sys.path.insert(0, os.path.join(ADCL, "src"))
from etri_table import ValSet, wl2
CACHE = "/tmp/pm97/data/etri/ego_cache.npz"

RUNS = ["v2a_goal", "v2a_trunkopen", "v2a_nogoal"]
EPS = [1, 2, 3, 4]
EPOCH_FRAC = {1: 0.92, 2: 1.92, 3: 2.92, 4: 4.00}
NAME = {"v2a_goal": "goal", "v2a_trunkopen": "trunkopen", "v2a_nogoal": "no-goal"}


def fnan(x):
    import math
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NaN"
    return f"{x:+.3f}"


def f1(x, n=4):
    return "—" if x is None else f"{x:.{n}f}"


def get_full(run, ep):
    p = f"{E}/fullval_{run}_ep{ep}.json"
    if not os.path.isfile(p):
        return None
    d = json.load(open(p))
    for k, v in d["rows"].items():
        if "frame>=30" in k:
            return v
    return None


def get_scale(run, ep):
    p = f"{E}/fullval_{run}_ep{ep}_pred.npy"
    if not os.path.isfile(p):
        return None
    v = ValSet(); gt, W, ci = v.gt, v.w, v.vi
    d = np.load(CACHE, allow_pickle=True)
    keep = d["frame"][ci].astype(int) >= 30
    p_ = np.load(p).astype(np.float64)
    if not np.isfinite(p_[keep]).all():
        return None
    pk, gk, wk = p_[keep], gt[keep], W[keep]
    s = (wk[:, None] * (pk * gk).sum(-1)).sum() / (wk[:, None] * (pk * pk).sum(-1)).sum()
    return dict(
        l2=wl2(pk, gk, wk),
        dx=np.average(np.abs(pk[..., 0] - gk[..., 0]).mean(1), weights=wk),
        dy=np.average(np.abs(pk[..., 1] - gk[..., 1]).mean(1), weights=wk),
        s=s, l2s=wl2(s * pk, gk, wk))


def get_gate(run, ep):
    p = f"{E}/gate_{run}_ep{ep}.txt"
    if not os.path.isfile(p):
        return None
    txt = open(p, encoding="utf-8", errors="replace").read()
    out = {}
    for line in txt.splitlines():
        m = re.match(r"^(full|image-zero|clip-shuffle-matched|bev-content-shuffle|"
                     r"chan_mean|evidence_chan_mean|evidence-shuffle)\s+([\d.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    out["n_pass"] = len(re.findall(r"\bPASS\b", txt))
    out["n_fail"] = len(re.findall(r"\bFAIL\b", txt))
    return out


def get_t6(run, ep):
    p = f"{E}/t6ppp_{run}_ep{ep}.txt"
    if not os.path.isfile(p):
        return None
    txt = open(p, encoding="utf-8", errors="replace").read()
    mx = [float(x) for x in re.findall(r"max\|pred\|\s+([\d.e+-]+)\s+PASS", txt)]
    return dict(n_zero_pass=len(re.findall(r"T6''' .*PASS", txt)),
                worst=max(mx[:3]) if len(mx) >= 3 else None,
                additive="없음" if "없음 (순수 대체)" in txt else "?")


def get_2x2(run, ep):
    p = f"{E}/x2x2_{run}_ep{ep}.txt"
    if not os.path.isfile(p):
        return None
    txt = open(p, encoding="utf-8", errors="replace").read()
    def g(pat, cast=float):
        # no-goal 은 goal 응답이 정확히 0이라 분산 0 -> 상관이 '+nan' 으로 찍힌다.
        m = re.search(pat, txt)
        if not m:
            return None
        v = m.group(1)
        if v.lstrip('+-').lower() == 'nan':
            return float('nan')
        try:
            return cast(v)
        except ValueError:
            return None
    return dict(
        delta_img=g(r"Delta_image = L2\(C\) - L2\(A\) = \+?([-\d.]+)"),
        agree=g(r"방향 일치율\s+([\d.]+)%"),
        bearing=g(r"endpoint bearing 변화 상관\s+([-+]?[\d.]+|[-+]?nan)"),
        prog=g(r"progress 변화 상관\s+([-+]?[\d.]+|[-+]?nan)"),
        lat=g(r"횡방향 이동 p50\s+([\d.]+)"),
        inter=g(r"waypoint norm of I: 평균\s+([\d.]+)"))


print("### 표 1. 성능 — 전체 2,280 앵커 val, frame≥30 가중 L2\n")
print("| 실제 epoch | " + " | ".join(NAME[r] for r in RUNS) + " |")
print("|---:|" + "---:|" * len(RUNS))
for ep in EPS:
    cells = []
    for r in RUNS:
        d = get_full(r, ep)
        cells.append(f1(d["가중"]) if d else "—")
    print(f"| {EPOCH_FRAC[ep]:.2f} | " + " | ".join(cells) + " |")

print("\n### 표 2. 축 분해 · 스케일 (frame≥30)\n")
print("| 런 | epoch | 가중 L2 | \\|dx\\| | \\|dy\\| | s* | 형태 L2 (s* 적용) |")
print("|---|---:|---:|---:|---:|---:|---:|")
for r in RUNS:
    for ep in EPS:
        d = get_scale(r, ep)
        if not d:
            continue
        print(f"| {NAME[r]} | {EPOCH_FRAC[ep]:.2f} | {d['l2']:.4f} | {d['dx']:.4f} | "
              f"{d['dy']:.4f} | {d['s']:.4f} | {d['l2s']:.4f} |")

print("\n### 표 3. 영상 게이트 (동결 570 앵커)\n")
print("| 런 | epoch | full | image-zero | chan_mean | evid_chan_mean | "
      "clip-shuf | evid-shuf | bev-shuf | PASS/FAIL |")
print("|---|---:|" + "---:|" * 7 + "---|")
for r in RUNS:
    for ep in EPS:
        g = get_gate(r, ep)
        if not g:
            continue
        print(f"| {NAME[r]} | {EPOCH_FRAC[ep]:.2f} | {f1(g.get('full'))} | "
              f"{f1(g.get('image-zero'))} | {f1(g.get('chan_mean'))} | "
              f"{f1(g.get('evidence_chan_mean'))} | {f1(g.get('clip-shuffle-matched'))} | "
              f"{f1(g.get('evidence-shuffle'))} | {f1(g.get('bev-content-shuffle'))} | "
              f"{g['n_pass']}/{g['n_fail']} |")

print("\n### 표 4. 구조 게이트 T6‴\n")
print("| 런 | epoch | zero 3종 PASS | 최대 \\|pred\\| | 가산항 |")
print("|---|---:|---:|---:|---|")
for r in RUNS:
    for ep in EPS:
        t = get_t6(r, ep)
        if not t:
            continue
        print(f"| {NAME[r]} | {EPOCH_FRAC[ep]:.2f} | {t['n_zero_pass']}/3 | "
              f"{t['worst']:.3e} | {t['additive']} |")

print("\n### 표 5. goal 응답 2×2 (동일 donor 집합: 600 표본, 579/598 성공, bearing 189 / 거리 414)\n")
print("| 런 | epoch | 영상 Δ | bearing 상관 | 부호 일치 | progress 상관 | 횡 이동 p50 | \\|I\\| |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for r in RUNS:
    for ep in EPS:
        x = get_2x2(r, ep)
        if not x:
            continue
        print(f"| {NAME[r]} | {EPOCH_FRAC[ep]:.2f} | +{f1(x['delta_img'])} | "
              f"{fnan(x['bearing'])} | {x['agree']:.1f}% | {fnan(x['prog'])} | "
              f"{x['lat']:.3f} m | {f1(x['inter'])} |")
