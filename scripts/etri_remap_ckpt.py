#!/usr/bin/env python
"""nuScenes VAD 가중치를 ETRI 좌표계로 기하 리맵한다 (fine-tuning 출발점 개선).

근거 (실측)
----------
`VAD_tiny.pth`를 ETRI val에 zero-shot으로 돌리면 가중 L2 11.76인데, 예측의 x,y를
**교환만** 하면 5.30이 된다 (2.2배). 3초 예측 성분 p50이 `x +0.64 / y +20.34`인데
GT는 `x +34.43 / y -0.73`이다 -- 사전학습 planner는 제대로 전방 궤적을 내지만
축이 nuScenes 규약이다.

    nuScenes  point_cloud_range [-15,-30,-2, 15,30,2]
    ETRI      point_cloud_range [-30,-15,-2, 30,15,2]

이건 알려진 결정적 변환이므로 가중치에 미리 적용할 수 있다. shape가 같아
`load_state_dict`가 조용히 통과해버리므로, 안 하면 fine-tuning이 90° 회전을
처음부터 다시 배워야 한다.

격자 순열의 정확성
-----------------
`encoder.py:72-80`에서 `ref_y`는 H(행), `ref_x`는 W(열)을 인덱스하고
`ref_2d = stack((ref_x, ref_y))`이므로 **W축 <-> world x, H축 <-> world y**,
평탄화는 `idx = h*W + w`다.

    nuScenes  x in [-15,15] / W=100 -> 0.3 m/셀     y in [-30,30] / H=100 -> 0.6 m/셀
    ETRI      x in [-30,30] / W=100 -> 0.6 m/셀     y in [-15,15] / H=100 -> 0.3 m/셀

측정된 회전은 `ETRI_x = nusc_y`, `ETRI_y = -nusc_x` (yaw -90도). 셀 중심을 맞추면

    x_e = y_n :  -30 + 0.6(w'+0.5) = -30 + 0.6(h+0.5)   ->  w' = h
    y_e = -x_n:  -15 + 0.3(h'+0.5) =  15 - 0.3(w+0.5)   ->  h' = 99 - w

**격자 셀의 물리 크기가 정확히 전치 관계(0.3x0.6 <-> 0.6x0.3)라 보간 없는 순열이다.**
검산: nusc(0,0)은 x=-14.85, y=-29.7 -> 목표 x=-29.7, y=+14.85 -> ETRI(99,0). 공식과 일치.

sigmoid 출력 처리
----------------
`reference_points`는 sigmoid를 타므로 "1 - x"를 만들려면 로짓을 부호반전하면 된다
(`sigmoid(-z) = 1 - sigmoid(z)`). 그래서 y행은 weight와 bias를 모두 부호반전한다.

    python scripts/etri_remap_ckpt.py ckpt/vad/VAD_tiny.pth ckpt/vad/VAD_tiny_etri_axes.pth
"""
import argparse
import os
import sys

import numpy as np
import torch

HEAD = "pts_bbox_head."


def bev_perm(H=100, W=100):
    """src 평탄 인덱스 -> dst 평탄 인덱스. dst(h',w') = (99-w, h) <- src(h,w)."""
    src = np.arange(H * W)
    h, w = src // W, src % W
    hp, wp = (H - 1) - w, h
    return hp * W + wp, src


def rot_pairs(t, n_pairs, negate_second=True):
    """마지막 축이 아닌 **첫 축**이 (..., x, y) 쌍으로 배열된 텐서를 회전한다.

    (x, y) -> (y, -x). 행 2k = x, 행 2k+1 = y 라고 가정한다.
    """
    out = t.clone()
    for k in range(n_pairs):
        x, y = 2 * k, 2 * k + 1
        out[x] = t[y]
        out[y] = -t[x] if negate_second else t[x]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--bev", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parts", default="bev,ego,traj,ref,reg,map",
                    help="적용할 구성요소. 분해 실험용 (예: --parts ego,traj)")
    args = ap.parse_args()
    parts = {p.strip() for p in args.parts.split(",") if p.strip()}
    print(f"적용 구성요소: {sorted(parts)}\n")

    ck = torch.load(args.src, map_location="cpu")
    sd = ck["state_dict"]
    log = []

    # ---- 1. BEV 임베딩 격자 순열 ----
    k = HEAD + "bev_embedding.weight"
    if "bev" in parts and k in sd:
        dst_i, src_i = bev_perm(args.bev, args.bev)
        w = sd[k]
        assert w.shape[0] == args.bev ** 2, w.shape
        new = torch.empty_like(w)
        new[dst_i] = w[src_i]
        # 순열이 전단사인지 확인 -- 아니면 일부 셀이 초기화되지 않는다
        assert len(set(dst_i.tolist())) == len(dst_i)
        sd[k] = new
        log.append(f"{k:52s} {tuple(w.shape)}  격자 90도 순열")

    # ---- 2. ego planner 출력: 3 cmd x 6 ts x 2 ----
    for suf in ("weight", "bias"):
        k = HEAD + f"ego_fut_decoder.4.{suf}"
        if "ego" in parts and k in sd:
            t = sd[k]
            assert t.shape[0] % 2 == 0
            sd[k] = rot_pairs(t, t.shape[0] // 2)
            log.append(f"{k:52s} {tuple(t.shape)}  (x,y)->(y,-x) x{t.shape[0]//2}")

    # ---- 3. agent 궤적 출력: 6 ts x 2 ----
    for i in range(6):
        for suf in ("weight", "bias"):
            k = HEAD + f"traj_branches.{i}.4.{suf}"
            if "traj" in parts and k in sd:
                t = sd[k]
                sd[k] = rot_pairs(t, t.shape[0] // 2)
                log.append(f"{k:52s} {tuple(t.shape)}  (x,y)->(y,-x)")

    # ---- 4. reference points (sigmoid 로짓) ----
    for name, n in (("reference_points", 3), ("map_reference_points", 2)):
        for suf in ("weight", "bias"):
            k = HEAD + f"transformer.{name}.{suf}"
            if "ref" in parts and k in sd:
                t = sd[k]
                assert t.shape[0] == n, (k, t.shape)
                new = t.clone()
                new[0] = t[1]        # x' = y
                new[1] = -t[0]       # y' = 1 - x  (sigmoid(-z) = 1 - sigmoid(z))
                sd[k] = new
                log.append(f"{k:52s} {tuple(t.shape)}  x<-y, y<--x")

    # ---- 5. 검출 회귀: (cx,cy,w,l,cz,h,rot_sin,rot_cos,vx,vy) ----
    # yaw theta -> theta-90도  =>  sin' = -cos, cos' = sin.  w <-> l 교환.
    for i in range(6):
        for suf in ("weight", "bias"):
            k = HEAD + f"reg_branches.{i}.4.{suf}"
            if "reg" not in parts or k not in sd:
                continue
            t = sd[k]
            if t.shape[0] != 10:
                log.append(f"{k:52s} {tuple(t.shape)}  !! code_size 10 아님, 건너뜀")
                continue
            new = t.clone()
            new[0], new[1] = t[1], -t[0]      # cx, cy (로짓)
            new[2], new[3] = t[3], t[2]       # w <-> l
            new[6], new[7] = -t[7], t[6]      # rot_sin, rot_cos
            new[8], new[9] = t[9], -t[8]      # vx, vy
            sd[k] = new
            log.append(f"{k:52s} {tuple(t.shape)}  cxcy/wl/rot/vxvy 회전")

    # ---- 6. map 회귀: N개의 (x,y) 점 ----
    for i in range(6):
        for suf in ("weight", "bias"):
            k = HEAD + f"map_reg_branches.{i}.4.{suf}"
            if "map" in parts and k in sd and sd[k].shape[0] % 2 == 0:
                t = sd[k]
                sd[k] = rot_pairs(t, t.shape[0] // 2)
                log.append(f"{k:52s} {tuple(t.shape)}  점 {t.shape[0]//2}개 회전")

    print(f"src {args.src}\n변환 {len(log)}개 텐서:\n")
    for line in log:
        print("  " + line)
    if not log:
        print("  !! 변환된 것이 없다 -- 키 이름을 확인하라")
        return 2

    if args.dry_run:
        print("\n--dry-run: 저장하지 않음")
        return 0
    ck["state_dict"] = sd
    ck.setdefault("meta", {})
    if isinstance(ck["meta"], dict):
        ck["meta"]["etri_axis_remap"] = dict(
            src=os.path.basename(args.src), bev=args.bev,
            note="nuScenes [-15,-30]->ETRI [-30,-15]; (x,y)->(y,-x); "
                 "bev grid perm dst(99-w,h)<-src(h,w)")
    torch.save(ck, args.dst)
    print(f"\n저장 {args.dst}  ({os.path.getsize(args.dst)/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
