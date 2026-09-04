#!/usr/bin/env python
"""prev_bev 누적 깊이가 챌린지 L2를 얼마나 바꾸는가.

왜
--
epoch 16 tvad_metric_overfit 에서:
    학습 경로(queue 2프레임 + 현재, model.train/eval 무관)   0.30
    streaming evaluator (시나리오 60앵커 연속 누적)          3.53
    current-only (앵커마다 리셋, 깊이 0)                     8.26
같은 가중치·같은 앵커·같은 metric인데 10배 차이가 난다. 남은 변수는 **prev_bev를
만드는 방법** 하나다.

학습은 `prepare_train_data`가 `range(i-15, i, 5)`에서 shuffle 후 1개를 버려
**직전 2프레임(-1.0/-0.5s 등)** 을 매 샘플 새로 인코딩한다. 반면 배포/제출 경로는
시나리오 전체를 이어받아 **깊이가 최대 60**까지 간다.

여기서는 앵커마다 리셋한 뒤 선행 K프레임(5프레임=0.5초 간격)만 흘려 깊이를 K로
고정하고 K를 쓸어본다.
    K=0   current-only 와 동일해야 한다 (일치하면 구현 검증)
    K=2   학습 조건과 동일한 깊이
    K=59  full streaming 과 동일

    python scripts/etri_warmup_depth.py --ckpt .../epoch_16.pth --K 0,1,2,3,6,12
"""
import argparse
import importlib
import os
import pickle
import sys

import numpy as np

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))
ANN = "/tmp/pm97/data/etri/pkl/vad_etri_infos_temporal_train.pkl"
SUB = "/tmp/pm97/data/etri/pkl/overfit8.pkl"
CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
W = np.array([11.0, 11.0, 5.0, 5.0, 2.0, 2.0]) / 36.0


def reset_stream(m):
    m.prev_frame_info = {"prev_bev": None, "scene_token": None,
                         "prev_pos": 0, "prev_angle": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/VAD/VAD_etri_tvad_metric_overfit.py"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--K", default="0,1,2,3,6")
    ap.add_argument("--stride", type=int, default=5, help="warmup 프레임 간격")
    ap.add_argument("--scenarios", type=int, default=0, help="시나리오 N개만")
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    args.ckpt = os.path.abspath(args.ckpt)
    Ks = [int(x) for x in args.K.split(",")]
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = ANN
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.test.pop("map_ann_file", None)
    ds = build_dataset(cfg.data.test)
    infos = pickle.load(open(ANN, "rb"))["infos"]
    key2ds = {(i["scene_token"], int(i["frame_idx"])): j for j, i in enumerate(infos)}

    d = np.load(CACHE, allow_pickle=True)
    scen_all = np.array([str(x) for x in d["scenarios"]])[d["scen_idx"]]
    frame_all = d["frame"].astype(int)
    want = sorted({i["scene_token"] for i in pickle.load(open(SUB, "rb"))["infos"]})
    if args.scenarios:
        want = want[:args.scenarios]
    cidx = np.where(np.isin(scen_all, list(want)) & (frame_all % 5 == 0))[0]
    cidx = cidx[np.lexsort((frame_all[cidx], scen_all[cidx]))]
    keys = [(scen_all[c], int(frame_all[c])) for c in cidx]
    gt = d["fut"][cidx].astype(np.float64)
    print(f"시나리오 {len(want)}  앵커 {len(keys)}  stride {args.stride}")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    own = model.state_dict()
    for k in [k for k, v in sd.items()
              if k in own and tuple(own[k].shape) != tuple(v.shape)]:
        sd.pop(k)
    r = model.load_state_dict(sd, strict=False)
    print(f"ckpt {os.path.basename(args.ckpt)}  missing {len(r.missing_keys)}  "
          f"unexpected {len(r.unexpected_keys)}")
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.eval()

    def get(s, f):
        """★ 절대 캐시하지 말 것. `forward_test`가 `img_metas[0][0]['can_bus']`를
        **in-place로** 깎고(VAD.py:306), 그 배열은 `data_infos`를 그대로 참조한다.
        collate 결과를 재사용하면 변위가 누적 차감되어 궤적이 붕괴한다
        (실측: 3초 예측이 0.85 m로 무너져 깊이 측정이 전부 무효가 됐다)."""
        if (s, f) not in key2ds:
            return None
        return collate([ds[key2ds[(s, f)]]], samples_per_gpu=1)

    def run(K):
        P = np.zeros((len(keys), 6, 2))
        for n, (s, f) in enumerate(keys):
            reset_stream(model.module)
            for k in range(K, 0, -1):
                dw = get(s, f - k * args.stride)
                if dw is None:
                    continue
                with torch.no_grad():
                    model(return_loss=False, rescale=True, **dw)
            data = get(s, f)
            with torch.no_grad():
                out = model(return_loss=False, rescale=True, **data)
            fut = out[0]["pts_bbox"]["ego_fut_preds"]
            cmd = np.asarray(data["ego_fut_cmd"][0].data[0]).reshape(-1, fut.shape[0])[0]
            P[n] = fut[int(cmd.argmax())].cpu().double().cumsum(0).numpy()
        return P

    def wl2(P):
        return float((np.sqrt(((P - gt) ** 2).sum(-1)) * W).sum(-1).mean())

    print(f"\n{'깊이 K':>7}{'가중 L2':>11}{'3초 |p| p50':>13}{'GT |p| p50':>12}")
    gp = np.median(np.linalg.norm(gt[:, -1], axis=-1))
    res = {}
    for K in Ks:
        P = run(K)
        res[K] = wl2(P)
        print(f"{K:>7}{res[K]:>11.4f}"
              f"{np.median(np.linalg.norm(P[:, -1], axis=-1)):>13.3f}{gp:>12.3f}",
              flush=True)

    print("\n참조: 학습경로(queue 2) 0.30 / full streaming 3.53 / current-only 8.26")
    if 2 in res:
        print(f"\n판정: K=2(학습과 동일 깊이) {res[2]:.4f}")
        if res[2] < 1.0:
            print("  ==> 원인 확정: prev_bev **누적 깊이**. 학습은 깊이 2, 배포는 깊이 60.")
            print("      모델이 얕은 누적에만 맞춰졌다. 학습 큐를 배포와 맞춰야 한다.")
        else:
            print("  ==> 깊이만으로는 설명되지 않는다. 전처리(train vs test pipeline) 확인 필요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
