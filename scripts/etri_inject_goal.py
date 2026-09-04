#!/usr/bin/env python
"""info pkl에 **ego 5초 goal**을 주입한다 (`gt_ego_fut_goal`, `gt_ego_fut_goal_yaw`).

왜 필요한가
----------
VAD info pkl에는 ego goal 필드가 아예 없다(`gt_agent_fut_goal`은 주변 agent용이다).
그래서 지금까지 만든 것은 C0-T가 아니라 **goal 입력이 없는 temporal VAD**였다.

goal은 주최측이 명시적으로 주는 입력이다. test clip의 `ego_pose.parquet`는
    frame -30 … -1, 0, **50**
로 되어 있고, 마지막 frame 50이 5초 goal이다(실측, 1125 clip 전부 동일 구조).
train에서도 같은 정의로 `ego_cache.npz['goal']`을 이미 만들어 뒀다
(`fut`가 `cumsum(gt_ego_fut_trajs)`와 일치함을 단위테스트로 확인했으므로 좌표 규약도 같다).

컴플라이언스 (8/18 공지 3-1항)
    goal은 **참조 지시 정보**로만 쓴다. 궤적 값을 goal에서 계산해 내면 안 된다.
    그래서 모델 쪽 설계는 "goal이 어떤 시각 증거를 읽을지 결정하고, waypoint 값은
    temporal visual value에서 나온다"로 간다 (GoalWaypointDecoder).
    이 스크립트는 데이터만 붙인다.

    python scripts/etri_inject_goal.py \
      --pkl /tmp/pm97/data/etri/pkl/vad_etri_infos_temporal_train.pkl \
      --pkl /tmp/pm97/data/etri/pkl/overfit8.pkl
"""
import argparse
import os
import pickle
import shutil
import sys

import numpy as np

CACHE = "/tmp/pm97/data/etri/ego_cache.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", action="append", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--suffix", default="_goal",
                    help="새 파일 접미사. 빈 문자열이면 제자리 수정(백업 생성)")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    scen = np.array([str(x) for x in d["scenarios"]])[d["scen_idx"]]
    frame = d["frame"].astype(int)
    goal = d["goal"].astype(np.float32)
    goal_yaw = d["goal_yaw"].astype(np.float32)
    table = {(s, int(f)): n for n, (s, f) in enumerate(zip(scen, frame))}
    print(f"ego_cache {args.cache}: {len(table):,} 프레임, "
          f"|goal| p50 {np.median(np.linalg.norm(goal, axis=-1)):.2f} m")

    for p in args.pkl:
        p = os.path.abspath(p)
        obj = pickle.load(open(p, "rb"))
        infos = obj["infos"]
        miss = 0
        for i in infos:
            k = (i["scene_token"], int(i["frame_idx"]))
            n = table.get(k)
            if n is None:
                miss += 1
                i["gt_ego_fut_goal"] = np.zeros(2, np.float32)
                i["gt_ego_fut_goal_yaw"] = np.float32(0.0)
                i["ego_goal_valid"] = False
                continue
            i["gt_ego_fut_goal"] = goal[n].copy()
            i["gt_ego_fut_goal_yaw"] = np.float32(goal_yaw[n])
            i["ego_goal_valid"] = True
        g = np.stack([i["gt_ego_fut_goal"] for i in infos])
        # 검증: goal이 3초 누적 궤적보다 확실히 멀어야 한다 (5초 지점이므로)
        fut3 = np.stack([np.asarray(i["gt_ego_fut_trajs"]).reshape(6, 2).sum(0)
                         for i in infos])
        r = np.linalg.norm(g, axis=-1) / np.maximum(np.linalg.norm(fut3, axis=-1), 1e-6)
        out = p if not args.suffix else p.replace(".pkl", args.suffix + ".pkl")
        if out == p:
            shutil.copy2(p, p + ".bak")
            print(f"  백업 {p}.bak")
        pickle.dump(obj, open(out, "wb"))
        print(f"{os.path.basename(p)}: {len(infos):,}개  goal 없음 {miss}  "
              f"|goal| p50 {np.median(np.linalg.norm(g,axis=-1)):.2f} m  "
              f"goal/3초누적 비율 p50 {np.median(r):.3f}")
        print(f"  -> {out}")
        if miss:
            print(f"  !! {miss}개는 goal이 없어 0으로 채우고 ego_goal_valid=False")
    return 0


if __name__ == "__main__":
    sys.exit(main())
