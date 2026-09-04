#!/usr/bin/env python
"""전체 학습용 **train 전용 pkl**을 만든다 (val 38 + holdout 8 제외).

왜 필요한가
----------
`vad_etri_infos_temporal_train_goal.pkl`은 이름과 달리 **376 시나리오 전부**(112,800)다.
val 38개와 나머지 8개(minival 후보)가 섞여 있다. 절대규칙 3(캘리브레이션 누수 금지)과
절대규칙 5(테스트 경계)에 걸리므로 전체 학습 전에 반드시 잘라야 한다.

    전체 376 = train 330 + val 38 + 나머지 8
    train 프레임  10Hz 99,000 / 2Hz 19,800

시나리오 **단위**로 자르므로 `prepare_train_data`의 `index - k*sample_interval` 이웃
참조는 안전하다. 시나리오 경계에서 이웃이 다른 시나리오로 넘어가는 경우는
`scene_token == scene_token and frame_idx < frame_idx` 가드가 이미 잡는다
(원래 설정에서도 발생하던 동작이다).

    python scripts/etri_split_pkl.py
"""
import argparse
import os
import pickle
import sys

import numpy as np

CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/pm97/data/etri/pkl/"
                                     "vad_etri_infos_temporal_train_goal.pkl")
    ap.add_argument("--out-train", default="/tmp/pm97/data/etri/pkl/"
                                          "etri_train330_goal.pkl")
    ap.add_argument("--out-val", default="/tmp/pm97/data/etri/pkl/"
                                        "etri_val38_goal.pkl")
    ap.add_argument("--stride", type=int, default=1,
                    help="1=10Hz 전부, 5=2Hz 앵커만")
    args = ap.parse_args()

    d = np.load(CACHE, allow_pickle=True)
    scen = np.array([str(x) for x in d["scenarios"]])
    sall = scen[d["scen_idx"]]
    sp = np.load(SPLIT, allow_pickle=True)
    val_s = sorted(set(sall[sp["val_idx"]]))
    train_s = sorted(set(sall[sp["train_idx"]]))
    other = sorted(set(scen) - set(val_s) - set(train_s))
    print(f"전체 {len(scen)}  train {len(train_s)}  val {len(val_s)}  "
          f"나머지 {len(other)}")
    print(f"  나머지(학습·평가 모두 제외): {other}")
    assert not (set(train_s) & set(val_s))

    obj = pickle.load(open(args.src, "rb"))
    infos = obj["infos"]
    lanes = obj.get("metadata", {}).get("map_lanes", {})
    print(f"\n원본 {os.path.basename(args.src)}: {len(infos):,} infos, "
          f"map_lanes {len(lanes)} 시나리오")
    have_goal = sum(1 for i in infos if "gt_ego_fut_goal" in i)
    print(f"  goal 보유 {have_goal:,}/{len(infos):,}")
    assert have_goal == len(infos), "goal이 없는 info가 있다 -- etri_inject_goal.py 먼저"

    for tag, keep_s, out in (("train", set(train_s), args.out_train),
                             ("val", set(val_s), args.out_val)):
        sub = [i for i in infos if i["scene_token"] in keep_s
               and (args.stride == 1 or int(i["frame_idx"]) % args.stride == 0)]
        meta = dict(obj.get("metadata", {}))
        meta["map_lanes"] = {k: v for k, v in lanes.items() if k in keep_s}
        meta["split"] = tag
        meta["n_scenarios"] = len(keep_s)
        meta["frame_stride"] = args.stride
        pickle.dump(dict(infos=sub, metadata=meta), open(out, "wb"))
        fr = np.array([int(i["frame_idx"]) for i in sub])
        ns = len({i["scene_token"] for i in sub})
        print(f"\n{tag}: {len(sub):,} infos  시나리오 {ns}  "
              f"frame {fr.min()}~{fr.max()}  stride {args.stride}")
        print(f"  map_lanes {len(meta['map_lanes'])}  -> {out}")
        # 누수 확인
        leak = {i["scene_token"] for i in sub} & (set(val_s) if tag == "train"
                                                 else set(train_s))
        print(f"  반대 split 시나리오 누수: {len(leak)}  "
              f"{'OK' if not leak else '!! ' + str(sorted(leak)[:3])}")
        assert not leak
    return 0


if __name__ == "__main__":
    sys.exit(main())
