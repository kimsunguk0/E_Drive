#!/usr/bin/env python
"""사전 top-M 후보 중 goal 로 1개를 고르는 **고정 규칙** + val 평가.

규정 근거
---------
8/31 운영국 답변: goal(미래 목표점)은 "모델의 여러 출력 중 선택에만 활용되는
경우는 허용됩니다." 그래서 이 파일의 선택 규칙은

    * 학습 파라미터 0개 (적합하는 값이 없다 -- val 누수도 원천적으로 불가)
    * **방위(bearing)만** 본다. 거리·속도를 쓰지 않으므로 3-2항(운동학 기반
      궤적 생성)과 무관하다. goal 은 "어디로 향해야 하는가"만 알려준다.
    * 후보 좌표를 만들지 않는다. 모델이 낸 top-M 중 하나를 고를 뿐이다.

베이스라인 VAD 가 cmd 로 3개 mode 중 하나를 고르는 것과 구조가 동형이다
(`VAD_head` 의 `ego_fut_preds[ego_fut_cmd==1]`).

무엇을 재는가
-------------
    top1        영상 전용 스코어러의 argmax  = goal 을 아예 안 쓴 값
    oracle@M    top-M 중 GT 최근접          = 어떤 선택기도 넘을 수 없는 상한
    goal@M      이 파일의 고정 규칙          = 실제로 제출될 값

`goal@M` 이 `top1` 보다 좋고 `oracle@M` 에 가까울수록 규칙이 잘 작동한다.
M 을 키우면 oracle 은 좋아지지만 규칙이 틀릴 여지도 커진다 -- 그 균형을 본다.

    python scripts/etri_vocab_select.py --logits val_logits.npz
"""
import argparse
import os
import sys

import numpy as np

ADCL = "/home/pm97/workspace/sukim/adcl"
sys.path.insert(0, os.path.join(ADCL, "scripts"))
sys.path.insert(0, os.path.join(ADCL, "src"))
CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
GOAL_MIN = 2.0     # |goal| 이 이보다 작으면 방위가 정의되지 않는다 -> top1 유지


LATERAL_M = 2.0    # 주최측 컨버터가 vad_cmd 를 만들 때 쓰는 횡변위 경계와 동일


def _lateral_bucket(y):
    """주최측 규칙과 동일: y <= -2 우회전(0), y >= +2 좌회전(1), 그 외 직진(2)."""
    b = np.full(np.shape(y), 2, dtype=np.int64)
    b[np.asarray(y) <= -LATERAL_M] = 0
    b[np.asarray(y) >= +LATERAL_M] = 1
    return b


def select_by_goal_bearing(anchors_abs, topm_idx, goal_xy):
    """[규칙 A] top-M 중 goal 방위 최근접. **권장하지 않는다.**

    실측: 순위가 좋을 때 결과를 악화시킨다(완벽 순위 M=20에서 0.0996 -> 0.2522).
    방위만 보므로 거리가 맞는 후보를 방향만 맞는 후보로 바꿔치기하고, 챌린지
    오차의 77~86%가 종방향이라 그 손해가 이득을 덮는다. 비교용으로만 남긴다.
    """
    end = anchors_abs[topm_idx][:, :, -1]
    cand_b = np.arctan2(end[..., 1], end[..., 0])
    goal_b = np.arctan2(goal_xy[:, 1], goal_xy[:, 0])
    d = np.abs(np.angle(np.exp(1j * (cand_b - goal_b[:, None]))))
    undefined = np.linalg.norm(goal_xy, axis=1) < GOAL_MIN
    pick = d.argmin(axis=1)
    pick[undefined] = 0
    return topm_idx[np.arange(len(topm_idx)), pick]


def select_by_goal_bucket(anchors_abs, topm_idx, goal_xy):
    """[규칙 B, 채택] goal 의 횡방향 갈래에 맞는 후보 중 **모델 1순위**.

    베이스라인 VAD 가 cmd 로 3개 mode 중 하나를 고르는 것과 동형이다.
    goal 은 우/좌/직진 갈래만 정하고, 그 안에서 어느 궤적인지는 **모델 점수가**
    정한다. 따라서 모델의 거리 판단을 훼손하지 않는다.

    갈래 경계(±2 m)는 주최측 컨버터가 `vad_cmd` 를 만들 때 쓰는 값을 그대로
    쓴다 -- 우리가 val 로 적합한 파라미터가 아니다(절대규칙 3 무관).

    갈래에 맞는 후보가 top-M 에 없으면 모델 1순위를 그대로 둔다.
    """
    end_y = anchors_abs[topm_idx][:, :, -1, 1]            # [N, M] 3초 끝점 y
    cand_b = _lateral_bucket(end_y)                       # [N, M]
    goal_b = _lateral_bucket(goal_xy[:, 1])               # [N]
    match = cand_b == goal_b[:, None]
    # 점수 내림차순이므로 첫 True 가 갈래를 만족하는 모델 최상위 후보다.
    has = match.any(axis=1)
    first = np.where(has, match.argmax(axis=1), 0)
    undefined = np.linalg.norm(goal_xy, axis=1) < GOAL_MIN
    first[undefined] = 0
    return topm_idx[np.arange(len(topm_idx)), first]


RULES = {'bearing': select_by_goal_bearing, 'bucket': select_by_goal_bucket}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True,
                    help="npz: logits [N,K], cache_idx [N]")
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--m", type=int, nargs="+", default=[1, 2, 3, 6, 10, 20])
    args = ap.parse_args()

    from etri_table import ValSet, wl2
    v = ValSet()
    gt, W, ci = v.gt, v.w, v.vi
    d = np.load(CACHE, allow_pickle=True)
    frame = d["frame"][ci].astype(int)
    goal = d["goal"][ci].astype(np.float64)

    z = np.load(args.logits)
    logits, cidx = z["logits"], z["cache_idx"]
    assert np.array_equal(cidx, ci), "logits 의 앵커 순서가 val 과 다르다"
    anchors = np.load(args.anchors).astype(np.float64)

    keep = frame >= 30
    order = np.argsort(-logits, axis=1)
    print(f"앵커 {int(keep.sum())} (frame>=30)   사전 K={anchors.shape[0]}\n")

    wgt = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0
    dist_all = (np.linalg.norm(anchors[None] - gt[:, None], axis=-1)
                * wgt).sum(-1)                                    # [N, K]
    l2_top1 = wl2(anchors[order[:, 0]][keep], gt[keep], W[keep])
    l2_full = wl2(anchors[dist_all.argmin(1)][keep], gt[keep], W[keep])

    print(f"{'M':>4} {'oracle@M':>9} {'bucket(B)':>10} {'bearing(A)':>11} "
          f"{'B의 몫':>9}")
    print("-" * 48)
    for m in args.m:
        topm = order[:, :m]
        rows = np.arange(len(topm))[:, None]
        best = topm[rows, dist_all[rows, topm].argmin(axis=1)[:, None]][:, 0]
        l2_orc = wl2(anchors[best][keep], gt[keep], W[keep])
        cells = {}
        for name, fn in RULES.items():
            sel = fn(anchors, topm, goal)
            cells[name] = wl2(anchors[sel][keep], gt[keep], W[keep])
        head = l2_top1 - l2_orc
        share = ((l2_top1 - cells['bucket']) / head) if head > 1e-6 else float('nan')
        share_s = "  n/a" if share != share else f"{share:8.1%}"
        print(f"{m:4d} {l2_orc:9.4f} {cells['bucket']:10.4f} "
              f"{cells['bearing']:11.4f} {share_s}")

    print(f"\ntop1 (goal 미사용)            = {l2_top1:.4f}")
    print(f"사전 전체 oracle (K={anchors.shape[0]})   = {l2_full:.4f}")
    print("* 'B의 몫' = (top1 - bucket@M) / (top1 - oracle@M). 음수면 규칙이 해롭다.")
    print("* 규칙 A(bearing)는 거리를 무시해 좋은 순위를 망친다 -- 비교용이다.")


if __name__ == "__main__":
    main()
