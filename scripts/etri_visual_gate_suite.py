#!/usr/bin/env python
"""배포 충실 visual gate 스위트 — 한 checkpoint에 대해 형님 요청 표를 전부 낸다.

주 평가 조건 (통일)
    깊이 6 + 앵커마다 prev_frame_info 리셋 + queue 7
    = 공식 제출(tools/etri_test_submit.py)과 동일. clip 독립성을 **state reset**으로
      보장한다. 기존 `frame>=30` 행은 필터일 뿐이라 깊이가 6 이상 아무 값이나 됐다.

조건
    full                    그대로
    image-zero              영상 전부 0 (OOD -- 보조 증거로만)
    clip-shuffle-matched    condition-matched donor의 **원본 영상**으로 교체
                            (command / goal 거리 / goal bearing / motion class 매칭,
                             donor 시나리오 != 원본 시나리오)
    bev-content-shuffle     donor의 **BEV feature**로 교체. 카메라·calibration은
                            원본 그대로이므로 calibration artifact가 섞이지 않는다.
    chan_mean               BEV feature를 per-channel spatial mean 으로 clamp.
                            채널 통계는 유지, **공간 정보만** 제거.

추가 출력
    frame<30 단독 L2        불완전 history 구간이 전체 평균을 얼마나 끌어올리는가
                            (깊이는 frame//5 로, 실제로 가능한 만큼만 준다)
    시나리오별 shuffle-full  일부 시나리오만 효과를 만드는가
    scenario bootstrap 95% CI

checkpoint 선택 원칙 (형님)
    가장 낮은 L2만 고르면 안 된다. visual trunk 후보로 인정하려면
        full < clip-shuffle-matched
        full < bev-content-shuffle
        full < chan_mean
    가 scenario 단위에서도 반복돼야 한다. image-zero는 OOD라 보조 증거지만,
    image-zero < full 이면 여전히 강한 경고다.

    python scripts/etri_visual_gate_suite.py --config <cfg> --ckpt <ckpt> \
      --ann-file <val pkl> --repo <동결본> [--n 600]
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np

CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"
W = np.array([11.0, 11.0, 5.0, 5.0, 2.0, 2.0]) / 36.0
DIST_BINS = [0, 20, 40, 60, 1e9]
BEAR_BINS = np.deg2rad([-180, -20, -5, 5, 20, 180])
SPD_BINS = [0, 0.5, 3, 8, 15, 1e9]
COND = ("full", "image-zero", "clip-shuffle-matched",
        "bev-content-shuffle", "chan_mean")


def bin_of(x, bins):
    return int(np.digitize(x, bins) - 1)


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
    ap.add_argument("--repo", required=True)
    ap.add_argument("--n", type=int, default=600, help="앵커 표본 (0=전체)")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor-set", default=None,
                    help="앵커 집합을 이 npz에 동결/검증한다 (모델 간 동일 집합 보증)")
    ap.add_argument("--out", default="")
    ap.add_argument("--conditions", default=",".join(COND))
    args = ap.parse_args()
    conds = [c for c in args.conditions.split(",") if c]
    repo = os.path.abspath(args.repo)
    ck = os.path.abspath(args.ckpt)
    cfgp = os.path.abspath(args.config)
    ann = os.path.abspath(args.ann_file)
    out = os.path.abspath(args.out) if args.out else ""
    os.chdir(repo)
    sys.path.insert(0, repo)
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    cfg = Config.fromfile(cfgp)
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
    vi = np.array([v for v in np.load(SPLIT, allow_pickle=True)["val_idx"]
                   if (scen_all[v], int(frame_all[v])) in k2])
    lo_mask = frame_all[vi] < args.depth * 5
    main_pool = vi[~lo_mask]
    low_pool = vi[lo_mask]
    rng = np.random.default_rng(args.seed)
    if args.n and len(main_pool) > args.n:
        # 시나리오별 균등 표본
        per = {}
        for v in main_pool:
            per.setdefault(scen_all[v], []).append(v)
        k = max(1, args.n // len(per))
        sel = np.sort(np.concatenate(
            [rng.choice(v, min(k, len(v)), replace=False) for v in per.values()]))
    else:
        sel = np.sort(main_pool)
    # 앵커 집합을 파일로 **동결**한다. 선택은 (val_idx, depth, seed)만의 함수라 이미
    # 모델과 무관하게 결정적이지만, 여러 checkpoint/모델 계보를 같은 집합으로 비교했다는
    # 것을 감사 가능하게 남긴다. 파일이 이미 있으면 일치를 assert하고 그 집합을 쓴다.
    if args.anchor_set:
        key = np.array([f"{s}|{int(f)}" for s, f in
                        zip(scen_all[sel], frame_all[sel])])
        if os.path.exists(args.anchor_set):
            prev = np.load(args.anchor_set, allow_pickle=True)["key"]
            prev = np.array([str(x) for x in prev])
            assert set(prev) == set(key), (
                f"동결된 앵커 집합과 다르다: 저장 {len(prev)} vs 현재 {len(key)}, "
                f"교집합 {len(set(prev) & set(key))}. 같은 집합으로 비교해야 한다.")
            pos = {f"{s}|{int(f)}": v for s, f, v
                   in zip(scen_all[vi], frame_all[vi], vi)}
            sel = np.array([pos[k] for k in prev])
            print(f"동결 앵커 집합 로드 {args.anchor_set}  {len(sel)}개")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.anchor_set)) or ".",
                        exist_ok=True)
            np.savez(args.anchor_set, key=key, depth=args.depth, seed=args.seed)
            print(f"동결 앵커 집합 저장 {args.anchor_set}  {len(sel)}개")
    scen, frame = scen_all[sel], frame_all[sel]
    gt = d["fut"][sel].astype(np.float64)
    goal = d["goal"][sel].astype(np.float64)
    speed = d["speed"][sel].astype(np.float64)
    cmd = d["vad_cmd"][sel].astype(int)
    N = len(sel)
    print(f"주 앵커 {N}  시나리오 {len(set(scen))}  깊이 {args.depth} + 앵커마다 리셋")
    print(f"frame<{args.depth*5} 앵커 {len(low_pool)} (별도 행)")

    # ---- condition-matched donor ----
    gd = np.hypot(goal[:, 0], goal[:, 1])
    gb = np.arctan2(goal[:, 1], goal[:, 0])
    buckets = {}
    for i in range(N):
        buckets.setdefault((int(cmd[i]), bin_of(gd[i], DIST_BINS),
                            bin_of(gb[i], BEAR_BINS),
                            bin_of(speed[i], SPD_BINS)), []).append(i)
    donor = np.full(N, -1)
    for _, idxs in buckets.items():
        for a in idxs:
            for b in idxs:
                if b != a and scen[b] != scen[a]:
                    donor[a] = b
                    break
    cov = donor >= 0
    cand = [len(v) for v in buckets.values()]
    print(f"donor coverage {int(cov.sum())}/{N} ({100*cov.mean():.1f}%)  "
          f"버킷 {len(buckets)}  후보 p10 {np.percentile(cand,10):.0f} "
          f"p50 {np.percentile(cand,50):.0f}")

    torch.manual_seed(0)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    sd = torch.load(ck, map_location="cpu")
    sd = sd.get("state_dict", sd)
    own = model.state_dict()
    for k in [k for k, v in sd.items()
              if k in own and tuple(own[k].shape) != tuple(v.shape)]:
        sd.pop(k)
    r = model.load_state_dict(sd, strict=False)
    print(f"ckpt {os.path.basename(ck)}  missing {len(r.missing_keys)}")
    model.compute_planner_metric_stp3 = lambda *a, **kw: {}

    # BEV clamp / 치환 hook
    tcls = type(model.pts_bbox_head.transformer)
    orig_gbf = tcls.get_bev_features
    st = {"mode": "full", "inject": None, "capture": None,
          "ev_mode": "full", "ev_inject": None, "capture_ev": None}

    def gbf(self, *a, **kw):
        o = orig_gbf(self, *a, **kw)
        bev = o[0] if isinstance(o, (tuple, list)) else o
        if st["capture"] is not None:
            st["capture"].append(bev.detach().clone())
        if st["mode"] == "chan_mean":
            dims = tuple(range(bev.dim() - 1))
            bev = bev.mean(dim=dims, keepdim=True).expand_as(bev).contiguous()
        elif st["mode"] == "bev-inject" and st["inject"] is not None:
            bev = st["inject"]
        if isinstance(o, (tuple, list)):
            return (bev,) + tuple(o[1:])
        return bev
    tcls.get_bev_features = gbf

    # ---- C0-T v2a: value 스트림(영상 evidence) 전용 hook ----
    # 위 `get_bev_features` hook은 v2a에서 **K만** 건드린다. v2a의 값은
    # `transformer.get_image_evidence()`로 따로 흐르기 때문이다. 그래서 no-goal 계보의
    # `chan_mean`/`bev-content-shuffle`을 v2a에 그대로 쓰면 "주소를 망가뜨렸다"만 재고
    # "값에서 공간 content를 뺐다"는 못 잰다. 아래 두 조건이 v2a용 content 게이트다.
    #   evidence_chan_mean   V의 채널 통계는 유지, 공간 정보만 제거 (K는 그대로)
    #   evidence-shuffle     V만 donor 것으로 교체 (K는 그대로)
    has_evidence = hasattr(tcls, "get_image_evidence") and getattr(
        model.pts_bbox_head.transformer, "expose_image_evidence", False)
    orig_gie = getattr(tcls, "get_image_evidence", None)
    if has_evidence:
        def gie(self, *a, **kw):
            ev = orig_gie(self, *a, **kw)
            if st["capture_ev"] is not None:
                st["capture_ev"].append(ev.detach().clone())
            if st["ev_mode"] == "chan_mean":
                dims = tuple(range(ev.dim() - 1))
                ev = ev.mean(dim=dims, keepdim=True).expand_as(ev).contiguous()
            elif st["ev_mode"] == "inject" and st["ev_inject"] is not None:
                ev = st["ev_inject"]
            return ev
        tcls.get_image_evidence = gie
    mm = MMDataParallel(model.cuda(0), device_ids=[0])
    mm.eval()

    def stream(s, f, depth, zero_img=False, src_s=None, src_f=None, capture=False,
               capture_ev=False):
        """앵커마다 리셋하고 depth 프레임을 흘린 뒤 마지막을 채점한다."""
        mm.module.prev_frame_info = {"prev_bev": None, "scene_token": None,
                                     "prev_pos": 0, "prev_angle": 0}
        ss = src_s if src_s is not None else s
        ff = src_f if src_f is not None else f
        for q in range(depth, 0, -1):
            wk = (ss, int(ff) - q * 5)
            if wk not in k2:
                continue
            dw = collate([ds[k2[wk]]], samples_per_gpu=1)
            if zero_img:
                dw["img"][0].data[0].zero_()
            with torch.no_grad():
                mm(return_loss=False, rescale=True, **dw)
        if src_s is not None:
            mm.module.prev_frame_info["scene_token"] = s
        data = collate([ds[k2[(s, int(f))]]], samples_per_gpu=1)
        if zero_img:
            data["img"][0].data[0].zero_()
        if src_s is not None:
            dn = collate([ds[k2[(ss, int(ff))]]], samples_per_gpu=1)
            data["img"][0].data[0] = dn["img"][0].data[0].clone()
            for im, dmm in zip(data["img_metas"][0].data[0],
                               dn["img_metas"][0].data[0]):
                im["lidar2img"] = dmm["lidar2img"]
        if capture:
            st["capture"] = []
        if capture_ev:
            st["capture_ev"] = []
        with torch.no_grad():
            o = mm(return_loss=False, rescale=True, **data)
        cap = st["capture"][-1] if capture and st["capture"] else None
        if capture_ev and st["capture_ev"]:
            cap = st["capture_ev"][-1]
        st["capture"] = None
        st["capture_ev"] = None
        fut = o[0]["pts_bbox"]["ego_fut_preds"]
        c = np.asarray(data["ego_fut_cmd"][0].data[0]).reshape(-1, fut.shape[0])[0]
        return fut[int(c.argmax())].cpu().double().cumsum(0).numpy(), cap

    def run(cond):
        P = np.zeros((N, 6, 2))
        keep = np.ones(N, bool)
        for i in range(N):
            if cond == "full":
                P[i], _ = stream(scen[i], frame[i], args.depth)
            elif cond == "image-zero":
                P[i], _ = stream(scen[i], frame[i], args.depth, zero_img=True)
            elif cond == "chan_mean":
                st["mode"] = "chan_mean"
                P[i], _ = stream(scen[i], frame[i], args.depth)
                st["mode"] = "full"
            elif cond == "clip-shuffle-matched":
                if donor[i] < 0:
                    keep[i] = False
                    continue
                j = donor[i]
                P[i], _ = stream(scen[i], frame[i], args.depth,
                                 src_s=scen[j], src_f=frame[j])
            elif cond == "bev-content-shuffle":
                if donor[i] < 0:
                    keep[i] = False
                    continue
                j = donor[i]
                _, bev_d = stream(scen[j], frame[j], args.depth, capture=True)
                st["mode"] = "bev-inject"
                st["inject"] = bev_d
                P[i], _ = stream(scen[i], frame[i], args.depth)
                st["mode"] = "full"
                st["inject"] = None
            elif cond == "evidence_chan_mean":
                # v2a 전용. V의 공간 정보만 제거하고 K는 건드리지 않는다.
                assert has_evidence, \
                    "evidence_* 조건은 expose_image_evidence 모델에서만 쓴다"
                st["ev_mode"] = "chan_mean"
                P[i], _ = stream(scen[i], frame[i], args.depth)
                st["ev_mode"] = "full"
            elif cond == "evidence-shuffle":
                assert has_evidence, \
                    "evidence_* 조건은 expose_image_evidence 모델에서만 쓴다"
                if donor[i] < 0:
                    keep[i] = False
                    continue
                j = donor[i]
                _, ev_d = stream(scen[j], frame[j], args.depth, capture_ev=True)
                st["ev_mode"] = "inject"
                st["ev_inject"] = ev_d
                P[i], _ = stream(scen[i], frame[i], args.depth)
                st["ev_mode"] = "full"
                st["ev_inject"] = None
            if (i + 1) % 150 == 0:
                print(f"    {cond} {i+1}/{N}", flush=True)
        return P, keep

    def wl2(P, g):
        return (np.sqrt(((P - g) ** 2).sum(-1)) * W).sum(-1)

    res = {}
    try:
        for c in conds:
            P, keep = run(c)
            res[c] = (wl2(P, gt), keep, P)
            print(f"  {c:<22} {res[c][0][keep].mean():.4f}  (n={int(keep.sum())})",
                  flush=True)
        # frame<30 단독
        low = None
        if len(low_pool) and "full" in conds:
            lp = low_pool if not args.n else low_pool[
                rng.choice(len(low_pool), min(200, len(low_pool)), replace=False)]
            Pl = np.zeros((len(lp), 6, 2))
            for n, v in enumerate(lp):
                dep = int(frame_all[v]) // 5
                Pl[n], _ = stream(scen_all[v], frame_all[v], dep)
            low = wl2(Pl, d["fut"][lp].astype(np.float64))
            print(f"  {'frame<30 (depth=frame//5)':<22} {low.mean():.4f}  (n={len(lp)})")
    finally:
        tcls.get_bev_features = orig_gbf
        if has_evidence and orig_gie is not None:
            tcls.get_image_evidence = orig_gie

    # ---- 표 ----
    sp = speed
    slices = [("전체", np.ones(N, bool)), ("이동", sp > 0.5),
              ("3–8", (sp >= 3) & (sp < 8)), ("8–15", (sp >= 8) & (sp < 15)),
              ("15+", sp >= 15)]
    print(f"\n=== 배포 충실 (깊이 {args.depth}, 앵커마다 리셋) ===")
    print(f"{'조건':<22}" + "".join(f"{n:>9}" for n, _ in slices))
    for c in conds:
        per, keep, _ = res[c]
        row = ""
        for _, m in slices:
            mm_ = m & keep
            row += f"{per[mm_].mean():>9.4f}" if mm_.sum() else f"{'—':>9}"
        print(f"{c:<22}{row}")
    if low is not None:
        print(f"{'frame<30 단독':<22}{low.mean():>9.4f}")

    print(f"\n=== 게이트 (full 대비, scenario 단위) ===")
    base, bkeep, _ = res["full"]
    scens = sorted(set(scen))
    gates = {}
    for c in conds:
        if c == "full":
            continue
        per, keep, _ = res[c]
        m = keep & bkeep
        dl = per[m] - base[m]
        sc = scen[m]
        ps = np.array([dl[sc == s].mean() for s in scens if (sc == s).any()])
        lo, hi = boot_ci(ps, seed=args.seed)
        frac = float((ps > 0).mean())
        ok = lo > 0 and frac > 0.60
        gates[c] = dict(delta=float(dl.mean()), ci=[lo, hi], frac=frac, pass_=bool(ok))
        tag = "PASS" if ok else ("FAIL" if dl.mean() < 0 else "약함")
        print(f"  {c:<22} Δ {dl.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"개선 시나리오 {100*frac:.0f}%  {tag}")
        worst = sorted(zip(scens, ps), key=lambda x: x[1])[:3]
        print(f"      악화(음수) 하위 3: " +
              ", ".join(f"{s}:{v:+.3f}" for s, v in worst))
    print("\n  * image-zero 는 OOD 라 보조 증거다. 다만 image-zero < full 이면 강한 경고.")
    print("  * visual trunk 후보 인정: clip-shuffle-matched / bev-content-shuffle /")
    print("    chan_mean 셋 다 scenario 단위에서 PASS 여야 한다.")

    if out:
        json.dump(dict(ckpt=ck, n=N, depth=args.depth,
                       rows={c: float(res[c][0][res[c][1]].mean()) for c in conds},
                       frame_lt30=None if low is None else float(low.mean()),
                       gates=gates), open(out, "w"), indent=1, ensure_ascii=False)
        print(f"\n저장 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
