#!/usr/bin/env python
"""2x2 counterfactual — 영상 성능 게이트 / goal 기능 게이트 / goal x image interaction.

형님 §4-§7 수정 반영. **이전 버전의 치명적 오류를 고쳤다.**

                 원래 goal      교란 goal
    원래 영상        A              B
    교란 영상        C              D

이전 버전은 A/B/C/D를 모두 원래 GT 기준 L2로 놓고 "goal main effect"를 계산했다.
그건 틀렸다. **B와 D에는 정답 궤적이 없다.** 원래 GT는 (원래 영상 + 원래 goal)
조건에서 수집된 것이라, 모델이 새 goal에 **올바르게** 반응할수록 원래 GT에서
멀어진다. 즉 goal이 잘 작동할수록 나쁜 점수가 나오는 지표였다.

그래서 축을 분리한다.
    A / C   원래 goal 유지 -> 원래 GT로 평가 가능. **영상 성능 게이트**
              Delta_image = L2(C) - L2(A) > 0  이어야 영상이 유익하다는 증거
    B / D   GT 없음. L2를 성능으로 읽지 않는다.
              라벨: counterfactual mismatch-to-original-GT
    goal 기능   출력 공간에서 **방향**을 본다
              goal bearing 변화와 endpoint bearing 변화의 부호 일치 / 상관
              goal 거리 변화와 progress 변화의 상관
    interaction  출력 공간에서
              I = (Y_D - Y_C) - (Y_B - Y_A)
              "goal 변화에 대한 반응이 장면에 따라 달라지는가"

통계는 **anchor가 아니라 scenario 단위**다. anchor 2,280개를 독립 표본으로 쓰면
CI가 지나치게 좁아진다. scenario를 재표본하는 bootstrap을 쓴다.
    채택: 95% CI 하한 > 0  AND  개선 방향 scenario 비율 > 60%

donor 안전장치 (§6.1-6.5)
    image donor j 와 goal donor k 를 **독립** 선택. i != j != k
    donor scenario != 원본 scenario (같은 장면의 인접 프레임이 donor가 되면 교란이 약하다)
    image donor : command / goal 거리 / goal bearing / motion class 를 맞춘다
                  (condition-matched. 무작위 donor는 교란이 과해 게이트가 쉬워진다)
    goal donor  : command 동일 + motion class 유사, 단 bearing 또는 거리에
                  **최소 차이를 강제**한다. 같은 세부 bin으로 묶으면 goal 교란이
                  너무 작아져 기능 게이트가 무력해진다.
    coverage 를 반드시 출력한다 (쉬운 샘플만 남아 좋아 보이는 것을 막는다)

    python scripts/etri_counterfactual_2x2.py --config <cfg> --ckpt <ckpt>
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

DIST_BINS = [0, 20, 40, 60, 1e9]
BEAR_BINS = np.deg2rad([-180, -20, -5, 5, 20, 180])
SPD_BINS = [0, 0.5, 3, 8, 15, 1e9]
# goal donor 최소 차이: 이보다 작으면 교란이 무의미하다
GOAL_MIN_BEARING_DEG = 5.0
GOAL_MIN_DIST_M = 10.0


def bin_of(x, bins):
    return int(np.digitize(x, bins) - 1)


def boot_ci(per_scen_delta, n=10000, seed=0, lo=2.5, hi=97.5):
    """scenario 단위 bootstrap. per_scen_delta: 시나리오별 평균 delta 배열."""
    rng = np.random.default_rng(seed)
    k = len(per_scen_delta)
    if k < 3:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, k, size=(n, k))
    means = per_scen_delta[idx].mean(axis=1)
    return (float(np.percentile(means, lo)), float(np.percentile(means, hi)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", default=REPO_DEFAULT,
                    help="소스 트리. 동결본으로 평가할 때는 그 경로를 준다")
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--n", type=int, default=600, help="앵커 표본 수")
    ap.add_argument("--depth", type=int, default=6, help="배포 충실 누적 깊이")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    args.ckpt = os.path.abspath(args.ckpt)
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
    cfg.data.test.ann_file = os.path.abspath(args.ann_file)
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.test.pop("map_ann_file", None)
    ds = build_dataset(cfg.data.test)
    k2 = {(i["scene_token"], int(i["frame_idx"])): j
          for j, i in enumerate(ds.data_infos)}      # dataset 순서 (pkl 순서 금지)

    d = np.load(CACHE, allow_pickle=True)
    scen_all = np.array([str(x) for x in d["scenarios"]])[d["scen_idx"]]
    frame_all = d["frame"].astype(int)
    vi = np.load(SPLIT, allow_pickle=True)["val_idx"]
    # dataset에 실제로 있는 앵커만 (val38 pkl로 평가할 때를 위해)
    vi = np.array([v for v in vi
                   if (scen_all[v], int(frame_all[v])) in k2
                   and int(frame_all[v]) >= args.depth * 5])
    rng = np.random.default_rng(args.seed)
    sel = vi if len(vi) <= args.n else np.sort(rng.choice(vi, args.n, replace=False))
    scen, frame = scen_all[sel], frame_all[sel]
    gt = d["fut"][sel].astype(np.float64)
    goal = d["goal"][sel].astype(np.float64)
    speed = d["speed"][sel].astype(np.float64)
    cmd = d["vad_cmd"][sel].astype(int)
    N = len(sel)
    print(f"앵커 {N}  (val {len(vi)}에서 표본, frame>={args.depth*5})  "
          f"시나리오 {len(set(scen))}")

    gdist = np.hypot(goal[:, 0], goal[:, 1])
    gbear = np.arctan2(goal[:, 1], goal[:, 0])
    key = [(int(c), bin_of(gd, DIST_BINS), bin_of(gb, BEAR_BINS),
            bin_of(s, SPD_BINS))
           for c, gd, gb, s in zip(cmd, gdist, gbear, speed)]

    # ---- image donor j: condition-matched, 다른 시나리오 ----
    buckets = {}
    for i, k in enumerate(key):
        buckets.setdefault(k, []).append(i)
    cand_sizes = [len(v) for v in buckets.values()]
    img_donor = np.full(N, -1)
    for k, idxs in buckets.items():
        for a in idxs:
            for b in idxs:
                if b != a and scen[b] != scen[a]:
                    img_donor[a] = b
                    break
    # ---- goal donor k: 같은 command + 유사 motion, 단 bearing/거리 최소 차이 강제 ----
    goal_donor = np.full(N, -1)
    by_cmd_spd = {}
    for i in range(N):
        by_cmd_spd.setdefault((int(cmd[i]), bin_of(speed[i], SPD_BINS)), []).append(i)
    for kk, idxs in by_cmd_spd.items():
        for a in idxs:
            for b in idxs:
                if b == a or scen[b] == scen[a] or b == img_donor[a]:
                    continue
                dbear = abs(np.degrees(gbear[b] - gbear[a]))
                dbear = min(dbear, 360 - dbear)
                if dbear >= GOAL_MIN_BEARING_DEG or abs(gdist[b] - gdist[a]) >= GOAL_MIN_DIST_M:
                    goal_donor[a] = b
                    break
    ok = (img_donor >= 0) & (goal_donor >= 0)
    keep = np.where(ok)[0]
    print(f"\n=== donor coverage ===")
    print(f"  버킷 {len(buckets)}개  후보 수 p10 {np.percentile(cand_sizes,10):.0f} "
          f"p50 {np.percentile(cand_sizes,50):.0f}")
    print(f"  image donor 성공 {int((img_donor>=0).sum())}/{N}  "
          f"goal donor 성공 {int((goal_donor>=0).sum())}/{N}")
    print(f"  둘 다 성공 {len(keep)}/{N} ({100*len(keep)/max(N,1):.1f}%)  "
          f"시나리오 {len(set(scen[keep]))}")
    assert (img_donor[keep] != keep).all() and (goal_donor[keep] != keep).all()
    assert (goal_donor[keep] != img_donor[keep]).all(), "i!=j!=k 위반"
    if len(keep) < 60:
        print("  !! 매칭이 너무 적다. bin을 넓히거나 --n 을 늘려야 한다.")
        return 2

    torch.manual_seed(0)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    own = model.state_dict()
    for kk in [kk for kk, v in sd.items()
               if kk in own and tuple(own[kk].shape) != tuple(v.shape)]:
        sd.pop(kk)
    r = model.load_state_dict(sd, strict=False)
    has_goal = getattr(model.pts_bbox_head, "goal_decoder", None) is not None
    print(f"\nckpt {os.path.basename(args.ckpt)}  missing {len(r.missing_keys)}  "
          f"goal_decoder {'있음' if has_goal else '없음(no-goal 계보)'}")
    model.compute_planner_metric_stp3 = lambda *a, **kw: {}
    mm = MMDataParallel(model.cuda(0), device_ids=[0])
    mm.eval()

    def build(i):
        return collate([ds[k2[(scen[i], int(frame[i]))]]], samples_per_gpu=1)

    def predict(i, swap_goal, swap_img):
        """깊이 args.depth 로 배포 조건을 맞춘다. 영상 교란 시 warmup도 donor로."""
        src = scen[img_donor[i]] if swap_img else scen[i]
        base_f = int(frame[img_donor[i]]) if swap_img else int(frame[i])
        mm.module.prev_frame_info = {"prev_bev": None, "scene_token": None,
                                     "prev_pos": 0, "prev_angle": 0}
        for kq in range(args.depth, 0, -1):
            wk = (src, base_f - kq * 5)
            if wk not in k2:
                continue
            with torch.no_grad():
                mm(return_loss=False, rescale=True,
                   **collate([ds[k2[wk]]], samples_per_gpu=1))
        if swap_img:
            # scene_token이 바뀌면 forward_test가 prev_bev를 버린다(VAD.py:292)
            mm.module.prev_frame_info["scene_token"] = scen[i]
        data = build(i)
        if swap_img:
            dn = build(img_donor[i])
            data["img"][0].data[0] = dn["img"][0].data[0].clone()
            for im, dmm in zip(data["img_metas"][0].data[0],
                               dn["img_metas"][0].data[0]):
                im["lidar2img"] = dmm["lidar2img"]
        if swap_goal and "ego_fut_goal" in data:
            dn = build(goal_donor[i])
            data["ego_fut_goal"][0].data[0].copy_(dn["ego_fut_goal"][0].data[0])
        with torch.no_grad():
            out = mm(return_loss=False, rescale=True, **data)
        fut = out[0]["pts_bbox"]["ego_fut_preds"]
        c = np.asarray(data["ego_fut_cmd"][0].data[0]).reshape(-1, fut.shape[0])[0]
        return fut[int(c.argmax())].cpu().double().cumsum(0).numpy()

    cells = {}
    for name, sg, si in (("A", False, False), ("B", True, False),
                         ("C", False, True), ("D", True, True)):
        if sg and not has_goal:
            cells[name] = None
            continue
        cells[name] = np.stack([predict(i, sg, si) for i in keep])
        print(f"  {name} 완료", flush=True)

    g = gt[keep]
    sc = scen[keep]

    def per_anchor_l2(P):
        return (np.sqrt(((P - g) ** 2).sum(-1)) * W).sum(-1)

    eA = per_anchor_l2(cells["A"])
    eC = per_anchor_l2(cells["C"])
    print(f"\n=== 영상 성능 게이트 (A vs C: 원래 goal, 원래 GT) ===")
    print(f"  A 원래영상 {eA.mean():.4f}    C condition-matched donor 영상 {eC.mean():.4f}")
    dl = eC - eA
    scens = sorted(set(sc))
    per_scen = np.array([dl[sc == s].mean() for s in scens])
    lo, hi = boot_ci(per_scen, seed=args.seed)
    frac = float((per_scen > 0).mean())
    print(f"  Delta_image = L2(C) - L2(A) = {dl.mean():+.4f}")
    print(f"  scenario {len(scens)}개  개선방향(>0) 비율 {100*frac:.1f}%")
    print(f"  scenario bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
    gate_img = (lo > 0) and (frac > 0.60)
    print(f"  영상 게이트: {'PASS' if gate_img else 'FAIL'}  "
          f"(기준 CI 하한>0 AND 개선 scenario>60%)")

    if not has_goal:
        print("\n  no-goal 계보 -- goal 게이트/interaction 생략")
        return 0 if gate_img else 2

    eB = per_anchor_l2(cells["B"])
    eD = per_anchor_l2(cells["D"])
    print(f"\n=== B / D: counterfactual mismatch-to-original-GT (성능 점수 아님) ===")
    print(f"  B 교란goal/원래영상 {eB.mean():.4f}   D 교란goal/교란영상 {eD.mean():.4f}")
    print("  * 교란 goal에는 정답 궤적이 없다. 모델이 새 goal에 올바르게 반응할수록")
    print("    이 값은 커진다. 성능 비교에 쓰지 않는다.")

    print(f"\n=== goal 기능 게이트 (출력 공간 방향) ===")
    PA, PB = cells["A"], cells["B"]
    ka = keep
    kb = goal_donor[keep]
    d_goal_bear = np.degrees(np.arctan2(
        np.sin(gbear[kb] - gbear[ka]), np.cos(gbear[kb] - gbear[ka])))
    pa_bear = np.degrees(np.arctan2(PA[:, -1, 1], PA[:, -1, 0]))
    pb_bear = np.degrees(np.arctan2(PB[:, -1, 1], PB[:, -1, 0]))
    d_pred_bear = np.degrees(np.arctan2(
        np.sin(np.deg2rad(pb_bear - pa_bear)), np.cos(np.deg2rad(pb_bear - pa_bear))))
    m = np.abs(d_goal_bear) >= GOAL_MIN_BEARING_DEG
    if m.sum() >= 10:
        agree = np.sign(d_pred_bear[m]) == np.sign(d_goal_bear[m])
        per_scen_a = np.array([agree[sc[m] == s].mean()
                              for s in sorted(set(sc[m])) if (sc[m] == s).sum()])
        alo, ahi = boot_ci(per_scen_a, seed=args.seed)
        print(f"  bearing 변화 >= {GOAL_MIN_BEARING_DEG}도 인 앵커 {int(m.sum())}")
        print(f"    방향 일치율 {100*agree.mean():.1f}%  "
              f"scenario bootstrap 95% CI [{100*alo:.1f}%, {100*ahi:.1f}%]")
        cor = np.corrcoef(d_goal_bear[m], d_pred_bear[m])[0, 1]
        print(f"    goal bearing 변화 vs endpoint bearing 변화 상관 {cor:+.3f}")
    d_goal_dist = gdist[kb] - gdist[ka]
    prog_a = np.linalg.norm(PA[:, -1], axis=-1)
    prog_b = np.linalg.norm(PB[:, -1], axis=-1)
    md = np.abs(d_goal_dist) >= GOAL_MIN_DIST_M
    if md.sum() >= 10:
        cor2 = np.corrcoef(d_goal_dist[md], (prog_b - prog_a)[md])[0, 1]
        print(f"  거리 변화 >= {GOAL_MIN_DIST_M}m 인 앵커 {int(md.sum())}   "
              f"goal 거리 변화 vs progress 변화 상관 {cor2:+.3f}")
    lat = np.abs(PB[:, -1, 1] - PA[:, -1, 1])
    print(f"  3초 endpoint 횡방향 이동 p50 {np.median(lat):.3f} m  "
          f"p90 {np.percentile(lat,90):.3f} m")

    print(f"\n=== goal x image interaction (출력 공간) ===")
    I = (cells["D"] - cells["C"]) - (cells["B"] - cells["A"])
    inorm = (np.sqrt((I ** 2).sum(-1)) * W).sum(-1)
    per_scen_i = np.array([inorm[sc == s].mean() for s in scens])
    ilo, ihi = boot_ci(per_scen_i, seed=args.seed)
    print(f"  weighted waypoint norm of I: 평균 {inorm.mean():.4f}  "
          f"scenario CI [{ilo:.4f}, {ihi:.4f}]")
    print(f"  종방향 |I_x| p50 {np.median(np.abs(I[:,-1,0])):.4f} m   "
          f"횡방향 |I_y| p50 {np.median(np.abs(I[:,-1,1])):.4f} m")
    print("  I ~ 0  -> goal 반응이 장면과 무관")
    print("  I 유의 -> 영상 context가 goal 해석을 조절")
    print("  I 과대 -> goal-image fusion 과민 또는 donor mismatch")
    return 0 if gate_img else 2


if __name__ == "__main__":
    sys.exit(main())
