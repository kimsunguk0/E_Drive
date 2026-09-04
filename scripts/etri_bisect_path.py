#!/usr/bin/env python
"""학습 경로 0.42 vs 배포 경로 3.54 — 마지막 이분법.

여기까지 배제된 것 (모두 실측)
    metric 변환 / GT 무효 / 전처리(img·lidar2img 최대차 0.000e+00) / GT 누수 /
    streaming shift 값(참값 대비 비율 1.000) / 누적 깊이(깊이6 3.5371 = 깊이59 3.5335) /
    train-eval 모드·GridMask·dropout (0.329 -> 0.421)

남은 차이는 둘뿐이다.
    (a) head 호출 경로     forward_train 내부  vs  simple_test
    (b) prev_bev 생성      obtain_history_bev(only_bev=True) vs streaming(simple_test 출력)

학습 경로에서 만든 prev_bev를 **그대로 배포 경로에 주입**해 (a)와 (b)를 분리한다.

    1  train(prev_bev_train)     기준        = 학습 로그와 같아야 한다
    2  test (prev_bev_train)     주입        1과 같으면 head 경로는 무죄 -> 원인은 (b)
    3  test (prev_bev_stream)    보통 배포   2와 다르면 원인은 (b) 확정

    python scripts/etri_bisect_path.py --ckpt .../epoch_16.pth
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
SUB = "/tmp/pm97/data/etri/pkl/overfit8.pkl"
ANN = "/tmp/pm97/data/etri/pkl/vad_etri_infos_temporal_train.pkl"
CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
W = np.array([11.0, 11.0, 5.0, 5.0, 2.0, 2.0]) / 36.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/VAD/VAD_etri_tvad_metric_overfit.py"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=180)
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    args.ckpt = os.path.abspath(args.ckpt)
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

    # 학습 dataset: 큐를 **결정적**으로 (f-10, f-5, f) 로 고정한다.
    cfg.data.train.ann_file = SUB
    cfg.data.train.temporal_shuffle = False
    ds_tr = build_dataset(cfg.data.train)
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = ANN
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.test.pop("map_ann_file", None)
    ds_te = build_dataset(cfg.data.test)

    infos_tr = pickle.load(open(SUB, "rb"))["infos"]
    infos_te = pickle.load(open(ANN, "rb"))["infos"]
    k2 = {(i["scene_token"], int(i["frame_idx"])): j for j, i in enumerate(infos_te)}
    # 큐가 온전한 앵커만 (frame >= 10)
    picks = [(j, infos_tr[j]["scene_token"], int(infos_tr[j]["frame_idx"]))
             for j in range(len(infos_tr))
             if int(infos_tr[j]["frame_idx"]) % 5 == 0
             and int(infos_tr[j]["frame_idx"]) >= 10][:args.n]
    print(f"앵커 {len(picks)}  (frame%5==0, frame>=10, 큐 고정 (f-10, f-5, f))")

    d = np.load(CACHE, allow_pickle=True)
    scen_all = np.array([str(x) for x in d["scenarios"]])[d["scen_idx"]]
    frame_all = d["frame"].astype(int)
    cpos = {(s, int(f)): n for n, (s, f) in enumerate(zip(scen_all, frame_all))}
    gt = np.stack([d["fut"][cpos[(s, f)]] for _, s, f in picks]).astype(np.float64)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"),
                        test_cfg=cfg.get("test_cfg"))
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
    mm = MMDataParallel(model.cuda(0), device_ids=[0])
    mm.eval()

    # obtain_history_bev 출력을 훔치고, 꼬리 self.train()을 없앤다
    cls = type(model)
    orig_hist = cls.obtain_history_bev
    stash = {}

    def hist(self, imgs_queue, img_metas_list):
        out = orig_hist(self, imgs_queue, img_metas_list)
        self.eval()
        stash["bev"] = out.detach().clone()
        return out
    cls.obtain_history_bev = hist

    from projects.mmdet3d_plugin.VAD.plan_metric_loss import PlanChallengeL2Loss
    grab = {}
    ol = PlanChallengeL2Loss.forward

    def spy(self, pred_delta, gt_delta, cmd_onehot, masks=None):
        B = pred_delta.shape[0]
        ci = cmd_onehot.reshape(B, -1).argmax(dim=-1)
        b = torch.arange(B, device=pred_delta.device)
        grab["p"] = pred_delta[b, ci].detach().float().cpu().numpy()
        return ol(self, pred_delta, gt_delta, cmd_onehot, masks)
    PlanChallengeL2Loss.forward = spy

    def testpred(data):
        out = mm(return_loss=False, rescale=True, **data)
        fut = out[0]["pts_bbox"]["ego_fut_preds"]
        cmd = np.asarray(data["ego_fut_cmd"][0].data[0]).reshape(-1, fut.shape[0])[0]
        return fut[int(cmd.argmax())].cpu().double().cumsum(0).numpy()

    P1 = np.zeros((len(picks), 6, 2))
    P2 = np.zeros((len(picks), 6, 2))
    P3 = np.zeros((len(picks), 6, 2))
    try:
        for n, (j, s, f) in enumerate(picks):
            # 1) 학습 경로
            with torch.no_grad():
                mm(**collate([ds_tr[int(j)]], samples_per_gpu=1))
            P1[n] = np.cumsum(grab["p"][0], 0)
            bev_train = stash["bev"]

            # 2) 배포 head + 학습 prev_bev 주입.
            #    ★ prev_pos/prev_angle을 pkl에서 직접 읽으면 안 된다. 원시 pkl의
            #    can_bus[-1]은 **0.0**이고 patch_angle은 get_data_info가 채운다(실측).
            #    그래서 f-5를 실제로 한 번 흘려 배포가 갖게 될 값을 그대로 만든 뒤
            #    prev_bev만 학습 것으로 교체한다.
            model.prev_frame_info = {"prev_bev": None, "scene_token": None,
                                     "prev_pos": 0, "prev_angle": 0}
            with torch.no_grad():
                mm(return_loss=False, rescale=True,
                   **collate([ds_te[k2[(s, f - 5)]]], samples_per_gpu=1))
            model.prev_frame_info["prev_bev"] = bev_train
            model.prev_frame_info["scene_token"] = s
            with torch.no_grad():
                P2[n] = testpred(collate([ds_te[k2[(s, f)]]], samples_per_gpu=1))

            # 3) 보통 배포: 깊이 6 streaming
            model.prev_frame_info = {"prev_bev": None, "scene_token": None,
                                     "prev_pos": 0, "prev_angle": 0}
            for k in range(6, 0, -1):
                if (s, f - k * 5) not in k2:
                    continue
                with torch.no_grad():
                    mm(return_loss=False, rescale=True,
                       **collate([ds_te[k2[(s, f - k * 5)]]], samples_per_gpu=1))
            with torch.no_grad():
                P3[n] = testpred(collate([ds_te[k2[(s, f)]]], samples_per_gpu=1))
            if (n + 1) % 60 == 0:
                print(f"    {n+1}/{len(picks)}", flush=True)
    finally:
        cls.obtain_history_bev = orig_hist
        PlanChallengeL2Loss.forward = ol

    def wl2(P):
        return float((np.sqrt(((P - gt) ** 2).sum(-1)) * W).sum(-1).mean())

    print(f"\n{'조건':<40}{'가중 L2':>10}{'3초 |p| p50':>13}")
    rows = [("1 학습 head + 학습 prev_bev", P1),
            ("2 배포 head + 학습 prev_bev (주입)", P2),
            ("3 배포 head + 배포 prev_bev (깊이6)", P3)]
    for lab, P in rows:
        print(f"{lab:<40}{wl2(P):>10.4f}{np.median(np.linalg.norm(P[:, -1], axis=-1)):>13.3f}")
    print(f"{'GT':<40}{'':>10}{np.median(np.linalg.norm(gt[:, -1], axis=-1)):>13.3f}")

    a, b, c = wl2(P1), wl2(P2), wl2(P3)
    print("\n=== 판정 ===")
    if b > a * 2:
        print(f"  prev_bev를 그대로 줘도 배포 head는 {a:.3f} -> {b:.3f}")
        print("  ==> 원인은 **head 호출 경로**(forward_train vs simple_test)")
    elif c > b * 2:
        print(f"  배포 head는 무죄({a:.3f} -> {b:.3f}). prev_bev를 배포 방식으로 만들면 "
              f"{b:.3f} -> {c:.3f}")
        print("  ==> 원인은 **prev_bev 생성 방식**: 학습은 only_bev=True로 큐를 인코딩하고")
        print("      배포는 simple_test 출력을 이어받는다. 둘이 같은 텐서가 아니다.")
    else:
        print(f"  세 조건이 비슷하다 ({a:.3f} / {b:.3f} / {c:.3f}).")
        print("      -> 앵커 부분집합 차이를 다시 확인해야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
