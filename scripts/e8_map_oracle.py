#!/usr/bin/env python
"""⑤-G: GT HD-map corridor cost 가 selector 를 개선하는가 (CPU 오라클 검사).

⑤-E2 결론: regret 의 대부분이 "목적지 맞고 형상 틀림"이고 visual 은 여기서 무정보.
필요한 정보가 차선/경계라면 GT map 을 쓴 오라클이 먼저 이득을 보여야 한다.
게이트: tune 고정 가중으로 val 이득 >= 0.015 여야 이미지 기반 map head 를 만든다.

**중요**: HD map 을 추론 시점에 쓰려면 시나리오 좌표계 내 자차 위치(localization)가
필요하고 그건 허용 입력이 아니다. 따라서 이 실험은 타당성 검사이고,
채택되면 map 은 **학습 supervision 전용**(영상 -> 차선 예측)으로만 쓴다.

좌표 연결(검증됨): pkl ego2global = frame0 기준 로컬 pose,
ego_pose.parquet 의 row (frame_idx + 50) 과 오차 0.0000 m 로 일치.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

A = "/NHNHOME/data/sukim/adcl"
RAW = "/tmp/pm97/data/etri/train"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402

SOLID = {"yellow_solid", "yellow_double", "white_solid", "boundary"}
POSE_OFF = 50


def load_map(scen):
    """시나리오 로컬 좌표계의 centerline / solid 점군 + frame0 기준 변환."""
    ann = f"{RAW}/{scen}/annotation"
    hm = pd.read_parquet(f"{ann}/hd_map.parquet")
    ep = pd.read_parquet(f"{ann}/ego_pose.parquet")
    p0 = ep.iloc[POSE_OFF][["x", "y"]].values.astype(np.float64)
    y0 = float(ep.iloc[POSE_OFF]["yaw"])
    c, s = np.cos(-y0), np.sin(-y0)
    R = np.array([[c, -s], [s, c]])          # 시나리오 -> frame0 기준

    def polys(mask):
        out = []
        for pts in hm["points"][mask]:
            a = np.asarray(pts.tolist() if hasattr(pts, "tolist") else pts, float)
            if a.ndim == 2 and len(a) >= 2:
                out.append((a[:, :2] - p0) @ R.T)
        return out
    cen = polys(hm["class"] == "centerline")
    sol = polys(hm["class"].isin(SOLID))
    return cen, sol


def poly_pts(polys, step=0.5):
    """폴리라인을 균일 간격으로 리샘플해 점군으로."""
    out = []
    for p in polys:
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        L = seg.sum()
        if L < 1e-6:
            out.append(p)
            continue
        cs = np.concatenate([[0], np.cumsum(seg)])
        t = np.arange(0, L, step)
        out.append(np.column_stack([np.interp(t, cs, p[:, 0]), np.interp(t, cs, p[:, 1])]))
    return np.concatenate(out, 0) if out else np.zeros((0, 2))


def seg_cross(P, Q):
    """P [n,2,2] 후보 선분, Q [m,2,2] 차선 선분 -> 교차 개수."""
    if len(Q) == 0 or len(P) == 0:
        return 0
    p, r = P[:, 0], P[:, 1] - P[:, 0]
    q, s = Q[:, 0], Q[:, 1] - Q[:, 0]
    rxs = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
    qp = q[None] - p[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[None, :, 1] - qp[..., 1] * s[None, :, 0]) / rxs
        u = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / rxs
    ok = np.isfinite(t) & (np.abs(rxs) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(A, "logs/e8_map_oracle.json"))
    args = ap.parse_args()
    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)
    cand5 = bank["candidate_xy_abs_5s"].astype(np.float64)
    scen_names = arr["scenarios"]

    res = {}
    for split in ("tune", "val"):
        z = np.load(os.path.join(A, f"data/etri/selector_{split}.npz"))
        rows, S = z["rows"], z["shortlist"].astype(np.int64)
        D3, lg, goal, w = (z["D3"].astype(np.float64), z["logit"].astype(np.float64),
                           z["goal"].astype(np.float64), z["weight"].astype(np.float64))
        scen, frm = z["scen"], z["frame"]
        N, M = S.shape
        dc = np.zeros((N, M))          # centerline 평균거리
        xs = np.zeros((N, M))          # solid 교차 수
        cache = {}
        for si in np.unique(scen):
            nm = scen_names[si]
            if nm not in cache:
                cen, sol = load_map(nm)
                cp = poly_pts(cen)
                sseg = np.concatenate([np.stack([p[:-1], p[1:]], 1) for p in sol], 0) \
                    if sol else np.zeros((0, 2, 2))
                cache[nm] = (cKDTree(cp) if len(cp) else None, sseg)
            tree, sseg = cache[nm]
            idx = np.where(scen == si)[0]
            # frame0 기준 -> 해당 프레임 ego 기준
            his, yaw = arr["his"], arr["his_yaw"]
            for n in idx:
                r = rows[n]
                th = float(yaw[r][C.HIS_NOW])
                p = his[r][C.HIS_NOW].astype(np.float64)
                c_, s_ = np.cos(th), np.sin(th)
                Rt = np.array([[c_, s_], [-s_, c_]])            # R(th)^T
                cand = cand5[S[n], :6]                          # [M,6,2] ego frame
                world = cand @ Rt + p                           # ego -> frame0 기준
                if tree is not None:
                    dc[n] = tree.query(world.reshape(-1, 2))[0].reshape(M, 6).mean(1)
                if len(sseg):
                    lo, hi = world.reshape(-1, 2).min(0) - 3, world.reshape(-1, 2).max(0) + 3
                    mid = sseg.mean(1)
                    near = sseg[(mid[:, 0] > lo[0]) & (mid[:, 0] < hi[0]) &
                                (mid[:, 1] > lo[1]) & (mid[:, 1] < hi[1])]
                    for j in range(M):
                        xs[n, j] = seg_cross(np.stack([world[j, :-1], world[j, 1:]], 1), near)
        res[split] = dict(dc=dc, xs=xs, D3=D3, lg=lg, goal=goal, w=w, S=S)
        print(f"[{split}] n={N}  centerline거리 중앙값 {np.median(dc):.2f}m  "
              f"solid교차 평균 {xs.mean():.3f}회/후보", flush=True)

    nz = lambda x: (x - x.min(1, keepdims=True)) / (
        x.max(1, keepdims=True) - x.min(1, keepdims=True) + 1e-9)          # noqa: E731

    def realized(d, a, b):
        g = np.linalg.norm(cand5[d["S"], 9] - d["goal"][:, None], axis=-1)
        J = nz(g) + 0.1 * nz(d["lg"].max(1, keepdims=True) - d["lg"]) \
            + a * nz(d["dc"]) + b * nz(d["xs"])
        i = J.argmin(1)
        return float(np.average(np.take_along_axis(d["D3"], i[:, None], 1)[:, 0],
                                weights=d["w"]))

    print("\n=== tune 격자 (a=centerline, b=solid교차) ===")
    AS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8)
    BS = (0.0, 0.1, 0.3, 0.6, 1.0)
    grid = {}
    for a in AS:
        row = []
        for b in BS:
            v = realized(res["tune"], a, b)
            grid[(a, b)] = v
            row.append(f"b{b}={v:.4f}")
        print(f"  a{a}: " + "  ".join(row), flush=True)
    a0, b0 = min(grid, key=lambda k: grid[k])
    print(f"  -> tune 고정 a={a0} b={b0} ({grid[(a0,b0)]:.4f}), 기준 a0b0 {grid[(0.0,0.0)]:.4f}")

    print("\n=== val38 (tune 고정 1회 적용) ===")
    base = realized(res["val"], 0.0, 0.0)
    got = realized(res["val"], a0, b0)
    orc = float(np.average(res["val"]["D3"].min(1), weights=res["val"]["w"]))
    gain = base - got
    print(f"  shortlist oracle {orc:.4f}")
    print(f"  map 미사용       {base:.4f}")
    print(f"  GT-map corridor  {got:.4f}   이득 {gain:+.4f}")
    print(f"  게이트 >=0.015 : {'통과 -> image map head 제작 가치 있음' if gain >= 0.015 else '미달 -> map head 보류'}")

    d = res["val"]
    sel = (nz(np.linalg.norm(cand5[d["S"], 9] - d["goal"][:, None], axis=-1))
           + 0.1 * nz(d["lg"].max(1, keepdims=True) - d["lg"])).argmin(1)
    orci = d["D3"].argmin(1)
    print(f"\n  진단: oracle 후보 centerline거리 {np.take_along_axis(d['dc'],orci[:,None],1).mean():.3f}m"
          f"  vs 선택 후보 {np.take_along_axis(d['dc'],sel[:,None],1).mean():.3f}m")
    print(f"        oracle 후보 solid교차 {np.take_along_axis(d['xs'],orci[:,None],1).mean():.3f}"
          f"  vs 선택 후보 {np.take_along_axis(d['xs'],sel[:,None],1).mean():.3f}")
    json.dump({"tune_grid": {f"{a}|{b}": v for (a, b), v in grid.items()},
               "fixed": [a0, b0], "val_base": base, "val_map": got,
               "val_oracle": orc, "gain": gain},
              open(args.out, "w"), indent=1, default=float)
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
