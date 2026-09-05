#!/usr/bin/env python
"""후보 판별력 측정: 1024 후보가 FPN feature 격자에서 실제로 구분되는가?

sparse scorer 는 후보 waypoint 위치의 feature 만 읽는다. 후보들이 같은 셀로
몰리면 지역 증거로 후보를 구분할 수 없고 전역 맥락(시나리오)에 기대게 된다.

측정:
  1) 후보 waypoint 가 차지하는 고유 feature 셀 수 (stride 8 / 16)
  2) 후보별 '셀 서명'의 고유 개수 = 원리적으로 구분 가능한 후보 수
  3) 상위 후보쌍이 공유하는 셀 비율
해상도/샘플링을 바꿨을 때(stride 4, offset 샘플) 얼마나 좋아지는지도 같이 본다.
"""
import os
import sys

import numpy as np
import torch

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
from sparse_scoredrive import project_candidate_points  # noqa: E402
import sparse_common as C  # noqa: E402


def cells_for(grid, visible, H, W, offsets=((0.0, 0.0),)):
    """grid [1,6,K,T,Hh,2] in [-1,1] -> 후보별 (cam,cell) 집합 서명."""
    K, T = grid.shape[2], grid.shape[3]
    sigs = [set() for _ in range(K)]
    g = grid[0]           # [6,K,T,Hh,2]
    v = visible[0]
    for cam in range(g.shape[0]):
        for hh in range(g.shape[3]):
            gx = g[cam, :, :, hh, 0].numpy()
            gy = g[cam, :, :, hh, 1].numpy()
            vv = v[cam, :, :, hh].numpy()
            for ox, oy in offsets:
                px = (gx + 1) * 0.5 * W + ox
                py = (gy + 1) * 0.5 * H + oy
                cx = np.floor(px).astype(np.int64)
                cy = np.floor(py).astype(np.int64)
                ok = vv & (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
                for k in range(K):
                    for t in range(T):
                        if ok[k, t]:
                            sigs[k].add((cam, int(cy[k, t]), int(cx[k, t])))
    return sigs


def report(name, sigs, K):
    allcells = set()
    for s in sigs:
        allcells |= s
    frz = [frozenset(s) for s in sigs]
    uniq = len(set(frz))
    empty = sum(1 for s in sigs if not s)
    sizes = np.array([len(s) for s in sigs])
    print("  %-28s 고유셀=%5d  구분가능후보=%4d/%d (%.1f%%)  빈후보=%3d  후보당셀 p50=%.0f"
          % (name, len(allcells), uniq, K, 100.0 * uniq / K, empty, np.median(sizes)))
    return uniq


def main():
    bank = np.load(C.BANK_A0, allow_pickle=False)
    cand = torch.from_numpy(bank["candidate_xy_abs_5s"].astype(np.float32))
    l2i = torch.from_numpy(C.build_global_lidar2img()).unsqueeze(0)
    K = cand.shape[0]
    IH, IW = 432, 768
    print("입력 %dx%d, 후보 K=%d, waypoint T=%d" % (IH, IW, K, cand.shape[1]))

    grid, vis = project_candidate_points(cand, l2i, (IH, IW), (0.0, 1.0))
    print("\n현재 구현 (heights=(0,1), offset 없음):")
    for stride in (8, 16):
        H, W = IH // stride, IW // stride
        report("stride %d (%dx%d)" % (stride, W, H), cells_for(grid, vis, H, W), K)

    print("\n개선안 A — stride 4 추가:")
    H, W = IH // 4, IW // 4
    report("stride 4 (%dx%d)" % (W, H), cells_for(grid, vis, H, W), K)

    print("\n개선안 B — 설계 §6.3 의 4 offset 표본 (stride 8):")
    H, W = IH // 8, IW // 8
    off = ((0, 0), (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5))
    report("stride 8 + 4 offset", cells_for(grid, vis, H, W, off), K)

    print("\n개선안 C — 입력 2배 해상도(864x1536) 가정, stride 8:")
    grid2, vis2 = project_candidate_points(cand, l2i * 1.0, (IH, IW), (0.0, 1.0))
    H, W = (IH * 2) // 8, (IW * 2) // 8
    report("stride 8 @2x 입력 (%dx%d)" % (W, H), cells_for(grid2, vis2, H, W), K)

    print("\n개선안 D — 높이 4단 (0,0.5,1,1.5), stride 8:")
    grid3, vis3 = project_candidate_points(cand, l2i, (IH, IW), (0.0, 0.5, 1.0, 1.5))
    H, W = IH // 8, IW // 8
    report("stride 8 + 4 heights", cells_for(grid3, vis3, H, W), K)
    return 0


if __name__ == "__main__":
    sys.exit(main())
