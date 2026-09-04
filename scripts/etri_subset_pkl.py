#!/usr/bin/env python
"""pkl에서 시나리오 부분집합을 잘라낸다 (8-scene overfit / 스모크용).

왜 별도 스크립트인가: `data_infos` 순서를 유지해야 한다. `etri_vad_dataset.py`의
`prepare_train_data`가 `index - queue_length*sample_interval` 로 **dataset 인덱스
산술**로 큐를 만들고 `scene_token`·`frame_idx` 비교로 경계를 막기 때문에, 시나리오
안의 프레임 순서와 연속성이 깨지면 큐가 조용히 잘못 만들어진다.

기본은 **train_idx 시나리오에서만** 고른다 -- val 38과 mini-val 8은 절대 섞지 않는다
(절대규칙 10: 캘리브레이션 누수 금지).

    python scripts/etri_subset_pkl.py --n 8 --out /tmp/pm97/data/etri/pkl/overfit8.pkl
"""
import argparse
import os
import pickle
import sys

import numpy as np

ANN = "/tmp/pm97/data/etri/pkl/vad_etri_infos_temporal_train.pkl"
CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=ANN)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenarios", default="", help="쉼표 구분. 주면 --n 무시")
    args = ap.parse_args()

    d = np.load(CACHE, allow_pickle=True)
    sp = np.load(SPLIT, allow_pickle=True)
    scen_all = np.array([str(x) for x in d["scenarios"]])
    holdout = {str(x) for x in sp["holdout"]}
    minival = {str(x) for x in sp["minival"]}
    train_scens = sorted(set(scen_all) - holdout - minival)
    print(f"전체 {len(scen_all)}  holdout {len(holdout)}  mini-val {len(minival)}  "
          f"-> 사용 가능 train {len(train_scens)}")

    if args.scenarios:
        pick = [s.strip() for s in args.scenarios.split(",")]
        bad = [s for s in pick if s not in train_scens]
        assert not bad, f"train 시나리오가 아니다(누수 위험): {bad}"
    else:
        rng = np.random.default_rng(args.seed)
        pick = sorted(rng.choice(train_scens, args.n, replace=False).tolist())

    infos = pickle.load(open(args.ann, "rb"))
    meta = infos["metadata"]
    keep = [i for i in infos["infos"] if i["scene_token"] in set(pick)]
    # 순서 검증: 시나리오별로 frame_idx가 0..N-1 연속이어야 큐 산술이 성립한다
    for s in pick:
        f = [int(i["frame_idx"]) for i in keep if i["scene_token"] == s]
        assert f == sorted(f) and f == list(range(len(f))), \
            f"{s}: frame_idx가 0부터 연속이 아니다 ({f[:5]}...{f[-3:]})"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pickle.dump(dict(infos=keep, metadata=meta), open(args.out, "wb"))
    print(f"\n시나리오 {len(pick)}개, 샘플 {len(keep):,}")
    for s in pick:
        print(f"  {s}")
    print(f"저장 {args.out}  ({os.path.getsize(args.out)/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
