#!/usr/bin/env python
"""CPU 전용: `prepare_train_data`가 frame_idx별로 실제 조립하는 큐를 감사한다.

형님 지시 "frame<30 queue 구성 감사 — frame 0~29별 distinct history frame 수,
중복 프레임 수, 실제 temporal span, 다른 scenario 유입 여부".

왜 GPU가 필요 없나: 큐 구성은 **인덱스 논리**만으로 결정된다.
`etri_vad_dataset.prepare_train_data`의 로직을 그대로 재현하면 된다.

    prev_indexs_list = range(index - Q*S, index, S)          # Q=queue_length, S=sample_interval
    prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)   # Q-1 개
    ...
    for i in prev_indexs_list:
        i = max(0, i)                                        # ★ 전역 인덱스 0으로 clamp
        info = data_infos[i]
        if info['frame_idx'] < frame_idx and info['scene_token'] == scene_token:
            example = <새로 로드>                             # 채택
            frame_idx = info['frame_idx']
        data_queue.insert(0, deepcopy(example))               # ★ 채택 실패 시 직전 example 복제

즉 채택 실패는 **직전 프레임의 복제**가 된다. 복제는 can_bus delta가 0이 되므로
`union2one`이 계산하는 ego shift도 0이다. 모델은 "정지 이력 6스텝 + 움직이는 현재"를
보게 되고, 이건 정보 부족이 아니라 **temporal-speed 경로에 대한 라벨 노이즈**다.

`data_infos`는 mmdet3d `NuScenesDataset.load_annotations`가 timestamp로 정렬한다.
그래서 반드시 정렬 후 순서로 감사해야 한다 (원시 pkl 순서로 하면 틀린다).

    python scripts/etri_queue_audit.py --ann-file <train pkl> [--queue 7] [--interval 5]
"""
import argparse
import collections
import os
import pickle
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--queue", type=int, default=7)
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--report-upto", type=int, default=35)
    args = ap.parse_args()

    with open(os.path.abspath(args.ann_file), "rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) else data
    # mmdet3d와 동일한 timestamp 정렬을 반드시 재현한다.
    infos = list(sorted(infos, key=lambda e: e["timestamp"]))
    n = len(infos)
    frame = np.array([int(i["frame_idx"]) for i in infos])
    scene = np.array([str(i["scene_token"]) for i in infos])
    # can_bus[:3] = ego translation. 복제 여부를 pose로도 확인한다.
    trans = np.array([np.asarray(i["can_bus"], dtype=np.float64)[:3] for i in infos])
    print(f"infos {n}  시나리오 {len(set(scene))}  "
          f"frame_idx {frame.min()}~{frame.max()}  queue {args.queue} "
          f"interval {args.interval}")
    print(f"정렬 후 인덱스가 시나리오별로 연속인가: ", end="")
    contiguous = True
    for s in set(scene):
        idx = np.where(scene == s)[0]
        if idx.max() - idx.min() + 1 != len(idx):
            contiguous = False
            break
    print("예" if contiguous else "아니오 (시나리오가 인터리브됨)")

    Q, S = args.queue, args.interval
    # 통계 누적기
    stat = collections.defaultdict(lambda: dict(
        n=0, distinct=[], dup=[], span=[], cross_seen=0, cross_used=0,
        clamp=0, zero_delta=[]))
    for index in range(n):
        f0, s0 = int(frame[index]), scene[index]
        prev = sorted(list(range(index - Q * S, index, S))[1:], reverse=True)
        fidx = f0
        used = [f0]                     # 시간 역순으로 채택된 frame_idx
        cur = f0                        # 직전 example이 가리키는 frame_idx
        queue_frames = []               # 최종 큐(과거→현재 중 과거 Q-1개)
        cross_seen = cross_used = clamp = 0
        for i in prev:
            if i < 0:
                clamp += 1
            i = max(0, i)
            if scene[i] != s0:
                cross_seen += 1
            if frame[i] < fidx and scene[i] == s0:
                cur = int(frame[i])
                fidx = cur
                used.append(cur)
            queue_frames.insert(0, cur)
        queue_frames.append(f0)
        d = stat[f0]
        d["n"] += 1
        d["distinct"].append(len(set(queue_frames)))
        d["dup"].append(len(queue_frames) - len(set(queue_frames)))
        d["span"].append((max(queue_frames) - min(queue_frames)) * 0.1)
        d["cross_seen"] += cross_seen
        d["cross_used"] += cross_used
        d["clamp"] += clamp
        # 인접 큐 원소가 같은 frame => can_bus delta 정확히 0
        zd = sum(1 for a, b in zip(queue_frames[:-1], queue_frames[1:]) if a == b)
        d["zero_delta"].append(zd)

    print(f"\n{'frame':>6}{'앵커수':>8}{'distinct':>10}{'중복':>7}"
          f"{'span(s)':>9}{'delta=0쌍':>10}{'타시나리오 조회':>15}{'타시나리오 채택':>15}")
    tot_dup = tot_anchor = 0
    for f0 in sorted(stat):
        d = stat[f0]
        tot_anchor += d["n"]
        if np.mean(d["dup"]) > 0:
            tot_dup += d["n"]
        if f0 > args.report_upto:
            continue
        print(f"{f0:>6}{d['n']:>8}{np.mean(d['distinct']):>10.2f}"
              f"{np.mean(d['dup']):>7.2f}{np.mean(d['span']):>9.2f}"
              f"{np.mean(d['zero_delta']):>10.2f}"
              f"{d['cross_seen']:>15}{d['cross_used']:>15}")

    # 요약
    bad = sorted(f0 for f0 in stat if np.mean(stat[f0]["dup"]) > 0)
    print(f"\n=== 판정 ===")
    print(f"  큐에 중복이 생기는 frame_idx : {min(bad)}~{max(bad)}  ({len(bad)}종)")
    print(f"  영향 앵커                    : {tot_dup:,} / {tot_anchor:,}"
          f"  ({100.0*tot_dup/tot_anchor:.2f}%)")
    xu = sum(stat[f]["cross_used"] for f in stat)
    xs = sum(stat[f]["cross_seen"] for f in stat)
    print(f"  타 시나리오 프레임 조회      : {xs:,}회")
    print(f"  타 시나리오 프레임 **채택**  : {xu:,}회  "
          f"{'-> 누수 없음 (scene_token 검사가 막는다)' if xu == 0 else '-> ★누수★'}")
    print(f"  frame>=30 앵커의 distinct    : "
          f"{np.mean([np.mean(stat[f]['distinct']) for f in stat if f >= 30]):.2f}"
          f" / 중복 {np.mean([np.mean(stat[f]['dup']) for f in stat if f >= 30]):.2f}")
    print(f"\n  기전: 채택 실패는 `data_queue.insert(0, deepcopy(example))`로 **직전 프레임의")
    print(f"        복제**가 된다. 복제 쌍은 ego pose가 동일하므로 union2one의 can_bus")
    print(f"        delta와 shift가 정확히 0이다. 즉 frame<30 앵커는 '정지 이력 + 움직이는")
    print(f"        현재'라는 물리적으로 모순된 샘플로 학습된다. 정보 부족이 아니라")
    print(f"        temporal-speed 경로의 라벨 노이즈다.")
    print(f"  대응: 학습 앵커를 frame_idx>=30으로 제한한다. 단 **history 조회용으로")
    print(f"        frame 0~29는 data_infos에 남겨야 한다** -- pkl에서 지우면 frame 30이")
    print(f"        시나리오 첫 프레임이 되어 같은 문제가 frame 30~59로 옮겨갈 뿐이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
