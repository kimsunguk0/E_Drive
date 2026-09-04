#!/usr/bin/env python
"""info pkl 의 `cams[*].data_path` 를 다른 해상도 캐시로 다시 가리킨다.

왜 이것만 바꾸면 되는가
-----------------------
`etri_build_cache.py` 는 해상도와 무관하게 **같은 crop**(`[0,456,1920,1080]`)을
쓰고 `scale` 만 다르다. 실측:

    etri_768      crop [0,456,1920,1080]  scale 0.4  out 768x432
    etri_1536_2hz crop [0,456,1920,1080]  scale 0.8  out 1536x864

기하 보정은 로더의 `CachedImageGeometry(scale=...)` 가 담당하므로 pkl 은
**경로만** 바꾸면 된다. `cam_intrinsic`(undistort 후 K)은 원본 해상도 기준이고
scale 은 로더가 곱한다 -- pkl 을 건드리면 이중 적용된다.

프레임 정합
-----------
1536 캐시는 2 Hz(프레임 0,5,10,...)만 갖고 있다. 학습이 쓰는 앵커도 2 Hz 이고
queue 가 0.5 s 간격(f, f-5, ..., f-30)이라 읽는 프레임이 전부 5의 배수다.
그래도 **존재 검증을 강제**한다 -- 없는 프레임을 조용히 지나치면 학습이
망가지고 손실은 정상으로 보인다.

    python scripts/etri_repoint_pkl_cache.py \
      --pkl /tmp/pm97/data/etri/pkl/etri_train330_goal.pkl \
      --cache /tmp/pm97/cache/etri_1536_2hz --suffix _1536
"""
import argparse
import os
import os.path as osp
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", action="append", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--suffix", default="_1536")
    ap.add_argument("--check-frac", type=float, default=0.02,
                    help="존재 검증할 표본 비율. 1.0 이면 전수.")
    args = ap.parse_args()

    for p in args.pkl:
        p = osp.abspath(p)
        obj = pickle.load(open(p, "rb"))
        infos = obj["infos"]
        n_img = 0
        for i in infos:
            scen = i["scene_token"]
            for cam, c in i["cams"].items():
                base = osp.basename(c["data_path"])
                c["data_path"] = osp.join(args.cache, scen, cam, base)
                n_img += 1

        # 존재 검증 -- 표본이라도 반드시 한다
        step = max(1, int(1.0 / max(args.check_frac, 1e-9)))
        missing = []
        for k, i in enumerate(infos):
            if k % step:
                continue
            for c in i["cams"].values():
                if not osp.isfile(c["data_path"]):
                    missing.append(c["data_path"])
        if missing:
            raise SystemExit(
                f"{p}: 캐시 이미지 {len(missing)}개 없음 (표본 검사)\n  "
                + "\n  ".join(missing[:5]))

        out = p.replace(".pkl", f"{args.suffix}.pkl")
        with open(out, "wb") as fp:
            pickle.dump(obj, fp)
        print(f"{osp.basename(p)} -> {osp.basename(out)}")
        print(f"  infos {len(infos):,}  경로 {n_img:,}개 재지정  "
              f"표본검증 {len(range(0, len(infos), step)):,} info OK")
        print(f"  예시 {infos[0]['cams']['camera_front']['data_path']}")


if __name__ == "__main__":
    main()
