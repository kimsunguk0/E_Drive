#!/usr/bin/env python
"""§3-3 feature-clamp L2 — 상수 visual feature + ego shift만으로 경쟁력이 나오는가.

형님 §3-3. 최종 full-data 후보에만 실행한다 (8-scene 모델에 돌려도 해석이 안 된다).

기존 S1/S2(scripts/etri_shift_structcheck.py)는 **출력 민감도**를 재서
"동일 shift에 대한 민감도가 실영상에서 상수 feature 대비 26~94배 컸다"를 얻었다.
여기서는 **실제 challenge L2**를 재서 다른 질문에 답한다.

    상수 visual feature와 ego shift만으로 경쟁력 있는 trajectory를 만들 수 있는가?

상수 feature의 출력이 조금 변하는 것은 이미 안다(0.135 m). 중요한 건 그 상태의
L2가 **매우 나쁘고 경쟁력이 없다**는 것이다. 그게 확인되면
"영상 content가 성능의 실질적 출처"라는 주장이 수치로 뒷받침된다.

조건
    full            그대로
    zero            BEV feature = 0
    const           BEV feature = 공간적으로 상수 (0.37)
    chan_mean       BEV feature = per-channel spatial mean
                    (공간 변화만 없애고 채널 통계는 유지 -- 가장 관대한 조건)

`chan_mean`이 특히 중요하다. 상수 0.37은 채널 통계까지 파괴하므로 "OOD라서 나쁘다"는
반론이 가능하지만, per-channel mean은 채널 분포를 유지한 채 **공간 정보만** 없앤다.
그래도 L2가 나쁘면 "goal/shift가 공간 주소로 쓰이는 것이 아니다"의 강한 근거가 된다.

    python scripts/etri_feature_clamp_l2.py --config <cfg> --ckpt <ckpt> \
      --ann-file <val pkl> [--repo <동결본 경로>]
"""
import argparse
import importlib
import os
import sys

import numpy as np

REPO_DEFAULT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))
CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"
W = np.array([11.0, 11.0, 5.0, 5.0, 2.0, 2.0]) / 36.0
MODES = ("full", "zero", "const", "chan_mean")


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) < 3:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--const", type=float, default=0.37)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    args.ckpt = os.path.abspath(args.ckpt)
    ann = os.path.abspath(args.ann_file)
    repo = os.path.abspath(args.repo)
    os.chdir(repo)
    sys.path.insert(0, repo)
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = ann
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.test.pop("map_ann_file", None)
    ds = build_dataset(cfg.data.test)
    k2 = {(i["scene_token"], int(i["frame_idx"])): j
          for j, i in enumerate(ds.data_infos)}

    d = np.load(CACHE, allow_pickle=True)
    scen_all = np.array([str(x) for x in d["scenarios"]])[d["scen_idx"]]
    frame_all = d["frame"].astype(int)
    vi = np.load(SPLIT, allow_pickle=True)["val_idx"]
    vi = np.array([v for v in vi
                   if (scen_all[v], int(frame_all[v])) in k2
                   and int(frame_all[v]) >= args.depth * 5])
    rng = np.random.default_rng(args.seed)
    sel = vi if len(vi) <= args.n else np.sort(rng.choice(vi, args.n, replace=False))
    scen, frame = scen_all[sel], frame_all[sel]
    gt = d["fut"][sel].astype(np.float64)
    speed = d["speed"][sel].astype(np.float64)
    print(f"앵커 {len(sel)}  시나리오 {len(set(scen))}  깊이 {args.depth}")

    torch.manual_seed(0)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    own = model.state_dict()
    for k in [k for k, v in sd.items()
              if k in own and tuple(own[k].shape) != tuple(v.shape)]:
        sd.pop(k)
    r = model.load_state_dict(sd, strict=False)
    print(f"ckpt {os.path.basename(args.ckpt)}  missing {len(r.missing_keys)}")
    model.compute_planner_metric_stp3 = lambda *a, **kw: {}

    # BEV feature를 clamp 한다. head가 transformer에서 받은 bev_embed를 쓰기 직전.
    tcls = type(model.pts_bbox_head.transformer)
    orig_gbf = tcls.get_bev_features
    state = {"mode": "full"}

    def gbf(self, *a, **kw):
        out = orig_gbf(self, *a, **kw)
        bev = out[0] if isinstance(out, (tuple, list)) else out
        m = state["mode"]
        if m == "zero":
            bev = torch.zeros_like(bev)
        elif m == "const":
            bev = torch.full_like(bev, args.const)
        elif m == "chan_mean":
            # 공간 축만 평균 -> 채널 통계는 유지, 공간 정보만 제거
            dims = tuple(range(bev.dim() - 1))
            bev = bev.mean(dim=dims, keepdim=True).expand_as(bev).contiguous()
        if isinstance(out, (tuple, list)):
            return (bev,) + tuple(out[1:])
        return bev
    tcls.get_bev_features = gbf
    mm = MMDataParallel(model.cuda(0), device_ids=[0])
    mm.eval()

    def run(mode):
        state["mode"] = mode
        P = np.zeros((len(sel), 6, 2))
        for n, (s, f) in enumerate(zip(scen, frame)):
            mm.module.prev_frame_info = {"prev_bev": None, "scene_token": None,
                                        "prev_pos": 0, "prev_angle": 0}
            for kq in range(args.depth, 0, -1):
                wk = (s, int(f) - kq * 5)
                if wk not in k2:
                    continue
                with torch.no_grad():
                    mm(return_loss=False, rescale=True,
                       **collate([ds[k2[wk]]], samples_per_gpu=1))
            data = collate([ds[k2[(s, int(f))]]], samples_per_gpu=1)
            with torch.no_grad():
                out = mm(return_loss=False, rescale=True, **data)
            fut = out[0]["pts_bbox"]["ego_fut_preds"]
            c = np.asarray(data["ego_fut_cmd"][0].data[0]).reshape(-1, fut.shape[0])[0]
            P[n] = fut[int(c.argmax())].cpu().double().cumsum(0).numpy()
        state["mode"] = "full"
        return P

    try:
        res = {}
        print(f"\n{'조건':<12}{'가중 L2':>10}{'scenario CI':>24}{'15+m/s':>10}"
              f"{'3초 |p| p50':>13}")
        gp = np.median(np.linalg.norm(gt[:, -1], axis=-1))
        scens = sorted(set(scen))
        for m in MODES:
            P = run(m)
            per = (np.sqrt(((P - gt) ** 2).sum(-1)) * W).sum(-1)
            ps = np.array([per[scen == s].mean() for s in scens])
            lo, hi = boot_ci(ps, seed=args.seed)
            fast = speed >= 15.0
            res[m] = per
            print(f"{m:<12}{per.mean():>10.4f}   [{lo:>8.4f}, {hi:>8.4f}]"
                  f"{per[fast].mean() if fast.any() else float('nan'):>10.4f}"
                  f"{np.median(np.linalg.norm(P[:,-1],axis=-1)):>13.3f}", flush=True)
        print(f"{'GT':<12}{'':>10}{'':>24}{'':>10}{gp:>13.3f}")
    finally:
        tcls.get_bev_features = orig_gbf

    print("\n=== 판정 ===")
    f0 = res["full"].mean()
    for m in ("zero", "const", "chan_mean"):
        dl = res[m] - res["full"]
        ps = np.array([dl[scen == s].mean() for s in scens])
        lo, hi = boot_ci(ps, seed=args.seed)
        good = lo > 0
        print(f"  {m:<10} - full = {dl.mean():+.4f}   scenario CI [{lo:+.4f}, {hi:+.4f}]"
              f"   {'PASS (영상이 유익)' if good else 'FAIL (영상 없이도 동등하거나 낫다)'}")
    print(f"\n  * `chan_mean`이 가장 중요하다. 채널 통계를 유지한 채 공간 정보만 없앴는데도")
    print(f"    L2가 나빠져야 '영상의 공간적 content가 성능의 출처'라고 말할 수 있다.")
    print(f"  * `zero`/`const`는 OOD이므로 단독 근거로 쓰지 않는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
