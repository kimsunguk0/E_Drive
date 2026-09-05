#!/usr/bin/env python
"""⑤-C3 보조 perception supervision 용 agent 라벨 사전계산.

train330 pkl 의 gt_boxes / gt_agent_fut_trajs 로부터 각 프레임의 에이전트
(위치, 반경, 유효마스크)를 0.5초 6스텝에 대해 뽑아 ego_cache row 로 색인한다.

점유 라벨 자체는 후보 1024 x 6 스텝이라 사전계산이 크므로, 여기서는 컴팩트한
에이전트 배열만 저장하고 점유 여부는 학습 루프에서 GPU 로 계산한다
(배치당 16 x 1024 x 6 x 40 = 3.9M 거리계산, 무시할 비용).

좌표계: gt_boxes[:, :2] 는 lidar/ego XY 로 후보 waypoint 와 동일계.
gt_agent_fut_trajs 는 증분이라 cumsum 후 박스 중심에 더한다.
"""
import os
import pickle
import sys

import numpy as np

A = "/NHNHOME/data/sukim/adcl"
TRAIN_PKL = "/tmp/pm97/data/etri/pkl/etri_train330_goal.pkl"
OUT = os.path.join(A, "data/etri/agent_labels_train.npz")
MAX_A = 40            # 프레임당 최대 에이전트 (p90=32, max=54 -> 40 으로 절단)
T_FUT = 6             # 3초 6스텝 (agent future 가 6스텝까지만 있다)
INFLATE = 0.5         # 반경 여유 (m)


def main():
    sys.path.insert(0, os.path.join(A, "scripts"))
    import sparse_common as C
    arr = C.load_arrays()
    key2row = {}
    for r in range(len(arr["frame"])):
        key2row[(arr["scenarios"][arr["scen_idx"][r]], int(arr["frame"][r]))] = r

    infos = pickle.load(open(TRAIN_PKL, "rb"))["infos"]
    rows, pos, rad, msk = [], [], [], []
    n_clip = 0
    for info in infos:
        key = (info["scene_token"], int(info["frame_idx"]))
        r = key2row.get(key)
        if r is None:
            continue
        gb = np.asarray(info["gt_boxes"], np.float32)
        n = len(gb)
        p = np.zeros((MAX_A, T_FUT, 2), np.float32)
        rr = np.zeros((MAX_A,), np.float32)
        mm = np.zeros((MAX_A, T_FUT), bool)
        if n:
            if n > MAX_A:
                n_clip += 1
                # 자차에 가까운 순으로 유지
                keep = np.argsort(np.linalg.norm(gb[:, :2], axis=1))[:MAX_A]
                gb = gb[keep]
                ft = np.asarray(info["gt_agent_fut_trajs"], np.float32
                                ).reshape(-1, T_FUT, 2)[keep]
                fm = np.asarray(info["gt_agent_fut_masks"], np.float32)[keep]
            else:
                ft = np.asarray(info["gt_agent_fut_trajs"], np.float32
                                ).reshape(-1, T_FUT, 2)
                fm = np.asarray(info["gt_agent_fut_masks"], np.float32)
            k = len(gb)
            # 시각 t 의 에이전트 중심 = 현재 중심 + 증분 누적
            p[:k] = gb[:, None, :2] + np.cumsum(ft, axis=1)
            # 외접 반경 + 여유 (w,l = gb[:,3],gb[:,4])
            rr[:k] = 0.5 * np.sqrt(gb[:, 3] ** 2 + gb[:, 4] ** 2) + INFLATE
            mm[:k] = fm > 0.5
        rows.append(r); pos.append(p); rad.append(rr); msk.append(mm)

    rows = np.asarray(rows, np.int64)
    order = np.argsort(rows)
    rows = rows[order]
    pos = np.asarray(pos, np.float32)[order]
    rad = np.asarray(rad, np.float32)[order]
    msk = np.asarray(msk, bool)[order]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, rows=rows, pos=pos.astype(np.float16), rad=rad.astype(np.float16),
             mask=msk, max_a=MAX_A, t_fut=T_FUT, inflate=INFLATE)
    occ_frac = float(msk.any(-1).sum()) / max(len(rows) * MAX_A, 1)
    print("saved %s rows=%d pos=%s (프레임당 유효 에이전트 %.1f, 절단 프레임 %d)"
          % (OUT, len(rows), pos.shape, msk.any(-1).sum() / len(rows), n_clip))
    print("에이전트 슬롯 점유율 %.3f" % occ_frac)
    return 0


if __name__ == "__main__":
    sys.exit(main())
