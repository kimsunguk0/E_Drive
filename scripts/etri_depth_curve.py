#!/usr/bin/env python
"""이상 ② — frame<30이 6배 나쁜 것은 정보 부족인가 학습 손상인가.

측정한 사실 (no-goal full330 epoch 3, 배포 충실 depth 6 + 앵커마다 리셋)
    frame>=30   L2 0.4345
    frame<30    L2 2.6111      (약 6배)

CPU 감사(scripts/etri_queue_audit.py)로 **학습 쪽 기전**은 이미 확정됐다.
    queue 7 / interval 5 에서 frame f의 이력은 {f-5k}이고, 음수는
    `i = max(0, i)` -> 다른 시나리오 -> `scene_token` 불일치 -> 채택 실패 ->
    `data_queue.insert(0, deepcopy(example))`로 **직전 프레임 복제**가 된다.
    frame 0~4는 큐 distinct가 1이다 (현재 프레임 7장). 복제 쌍은 ego pose가 같아
    union2one의 can_bus delta/shift가 정확히 0 -> '정지 이력 3초 + 움직이는 현재'.
    전체의 10.00% (9,900/99,000). 타 시나리오 채택은 0회 (누수 없음).

여기서 답하는 건 **평가 쪽**이다.

    frame>=30 앵커를 depth 0~6으로 인위적으로 절단해 L2 곡선을 만든다.
    실제 frame<30 앵커(가용 depth = frame//5)가 그 곡선 **위에** 있으면
    정보 부족만이 아니라 학습 손상이 더해진 것이다. 곡선 **상에** 있으면 순수 정보 부족.

교란 통제: frame<30 앵커는 전부 시나리오 시작부라 정지/저속에 치우친다. 그래서
곡선 쪽 표본을 **속도 구간별로 매칭**해 같은 속도 분포에서 비교한다.

    python scripts/etri_depth_curve.py --config <cfg> --ckpt <ckpt> --ann-file <val pkl>
"""
import argparse
import importlib
import os
import sys

import numpy as np

CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"
W = np.array([11.0, 11.0, 5.0, 5.0, 2.0, 2.0]) / 36.0
SPEED_BINS = (0.5, 3.0, 8.0, 15.0)


def bin_of(x, bins):
    return int(np.searchsorted(np.asarray(bins), x, side="right"))


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return (float("nan"), float("nan"))
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--n", type=int, default=380, help="곡선용 앵커 (시나리오별 균등)")
    ap.add_argument("--n-low", type=int, default=380, help="frame<30 앵커")
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    cfg = Config.fromfile(os.path.abspath(args.config))
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
          for j, i in enumerate(ds.data_infos)}

    d = np.load(CACHE, allow_pickle=True)
    scen_all = np.array([str(x) for x in d["scenarios"]])[d["scen_idx"]]
    frame_all = d["frame"].astype(int)
    vi = np.array([v for v in np.load(SPLIT, allow_pickle=True)["val_idx"]
                   if (scen_all[v], int(frame_all[v])) in k2])
    rng = np.random.default_rng(args.seed)

    def stratified(pool, n):
        per = {}
        for v in pool:
            per.setdefault(scen_all[v], []).append(v)
        k = max(1, n // max(len(per), 1))
        return np.sort(np.concatenate(
            [rng.choice(v, min(k, len(v)), replace=False) for v in per.values()]))

    hi = vi[frame_all[vi] >= args.max_depth * 5]
    lo = vi[frame_all[vi] < args.max_depth * 5]
    hi_sel = stratified(hi, args.n)
    lo_sel = stratified(lo, args.n_low)
    print(f"곡선용 frame>=30 앵커 {len(hi_sel)}  시나리오 {len(set(scen_all[hi_sel]))}")
    print(f"실측 frame<30 앵커  {len(lo_sel)}  시나리오 {len(set(scen_all[lo_sel]))}")

    torch.manual_seed(0)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    sd = torch.load(os.path.abspath(args.ckpt), map_location="cpu")
    sd = sd.get("state_dict", sd)
    own = model.state_dict()
    for k in [k for k, v in sd.items()
              if k in own and tuple(own[k].shape) != tuple(v.shape)]:
        sd.pop(k)
    r = model.load_state_dict(sd, strict=False)
    print(f"ckpt {os.path.basename(args.ckpt)}  missing {len(r.missing_keys)}")
    model.compute_planner_metric_stp3 = lambda *a, **kw: {}
    mm = MMDataParallel(model.cuda(0), device_ids=[0])
    mm.eval()

    def pred(s, f, depth):
        """clip 리셋 후 depth개 선행 프레임을 흘리고 현재 프레임 예측을 반환."""
        mm.module.prev_frame_info = {"prev_bev": None, "scene_token": None,
                                    "prev_pos": 0, "prev_angle": 0}
        used = 0
        for kq in range(depth, 0, -1):
            wk = (s, int(f) - kq * 5)
            if wk not in k2:
                continue
            with torch.no_grad():
                mm(return_loss=False, rescale=True,
                   **collate([ds[k2[wk]]], samples_per_gpu=1))
            used += 1
        data = collate([ds[k2[(s, int(f))]]], samples_per_gpu=1)
        with torch.no_grad():
            out = mm(return_loss=False, rescale=True, **data)
        fut = out[0]["pts_bbox"]["ego_fut_preds"]
        c = np.asarray(data["ego_fut_cmd"][0].data[0]).reshape(-1, fut.shape[0])[0]
        return (fut[int(c.argmax())].cpu().double().cumsum(0).numpy(), used)

    def l2(P, G):
        return (np.sqrt(((P - G) ** 2).sum(-1)) * W).sum(-1)

    # ---- 곡선: frame>=30 앵커를 depth 0..max 로 절단 ----
    hi_gt = d["fut"][hi_sel].astype(np.float64)
    hi_sp = d["speed"][hi_sel].astype(np.float64)
    hi_scen = scen_all[hi_sel]
    curve = {}
    depth_used = {}
    for depth in range(args.max_depth + 1):
        P = np.zeros((len(hi_sel), 6, 2))
        us = []
        for i, (s, f) in enumerate(zip(hi_scen, frame_all[hi_sel])):
            P[i], u = pred(s, f, depth)
            us.append(u)
        curve[depth] = l2(P, hi_gt)
        depth_used[depth] = float(np.mean(us))
        print(f"  depth {depth}  실제 replay {depth_used[depth]:.2f}  "
              f"L2 {curve[depth].mean():.4f}", flush=True)

    # ---- 실측 frame<30 (가용 depth = frame//5) ----
    lo_gt = d["fut"][lo_sel].astype(np.float64)
    lo_sp = d["speed"][lo_sel].astype(np.float64)
    lo_scen = scen_all[lo_sel]
    lo_frame = frame_all[lo_sel]
    Pl = np.zeros((len(lo_sel), 6, 2))
    lo_used = []
    for i, (s, f) in enumerate(zip(lo_scen, lo_frame)):
        Pl[i], u = pred(s, f, args.max_depth)      # 배포 로직 그대로 -- 없는 건 건너뛴다
        lo_used.append(u)
    lo_l2 = l2(Pl, lo_gt)
    lo_used = np.array(lo_used)

    print(f"\n{'depth':>6}{'곡선 L2':>11}{'CI':>22}"
          f"{'frame<30 L2':>13}{'앵커':>6}{'속도매칭 곡선':>14}{'초과':>10}")
    verdicts = []
    for depth in range(args.max_depth + 1):
        c = curve[depth]
        clo, chi = boot_ci(c, seed=args.seed)
        m = lo_used == depth
        if m.sum() < 5:
            print(f"{depth:>6}{c.mean():>11.4f}   [{clo:>7.4f},{chi:>7.4f}]"
                  f"{'-':>13}{int(m.sum()):>6}{'-':>14}{'-':>10}")
            continue
        # 속도 구간 매칭: frame<30 표본의 속도 분포를 곡선 표본에 재가중한다
        lb = np.array([bin_of(x, SPEED_BINS) for x in lo_sp[m]])
        hb = np.array([bin_of(x, SPEED_BINS) for x in hi_sp])
        wts = np.zeros(len(c))
        for b in np.unique(lb):
            sel_b = hb == b
            if sel_b.sum() == 0:
                continue
            wts[sel_b] = (lb == b).mean() / sel_b.sum()
        matched = float((c * wts).sum() / wts.sum()) if wts.sum() > 0 else float("nan")
        excess = lo_l2[m].mean() - matched
        verdicts.append((depth, int(m.sum()), lo_l2[m].mean(), matched, excess))
        print(f"{depth:>6}{c.mean():>11.4f}   [{clo:>7.4f},{chi:>7.4f}]"
              f"{lo_l2[m].mean():>13.4f}{int(m.sum()):>6}{matched:>14.4f}"
              f"{excess:>+10.4f}")

    print(f"\n=== 판정 ===")
    if not verdicts:
        print("  비교 가능한 depth 구간이 없다 (표본 부족).")
        return 1
    tot_n = sum(v[1] for v in verdicts)
    w_excess = sum(v[1] * v[4] for v in verdicts) / tot_n
    print(f"  속도매칭 초과분 가중평균 : {w_excess:+.4f}  (앵커 {tot_n})")
    ex = np.array([v[4] for v in verdicts])
    print(f"  depth별 초과분           : "
          f"{', '.join(f'd{v[0]}={v[4]:+.3f}' for v in verdicts)}")
    if w_excess > 0.15:
        print(f"  -> frame<30은 같은 depth·같은 속도의 곡선보다 **위에** 있다.")
        print(f"     정보 부족만으로 설명되지 않는다. CPU 감사가 찾은 학습 손상")
        print(f"     ('정지 이력 3초 + 움직이는 현재' 모순 샘플 10%)와 일치한다.")
        print(f"     대응: 학습 앵커를 frame_idx>=30으로 제한 (v2a는 이미 적용).")
    elif w_excess < -0.15:
        print(f"  -> frame<30이 오히려 **낮다**. 시나리오 시작부가 쉬운 구간이라는 뜻이고,")
        print(f"     속도 매칭이 부족했을 수 있다. 다른 교란(정지 비율)을 더 통제해야 한다.")
    else:
        print(f"  -> frame<30은 같은 depth·같은 속도의 곡선 **상에** 있다.")
        print(f"     6배 차이는 거의 전부 '가용 이력이 짧다' + '시작부 속도 분포'로 설명된다.")
        print(f"     학습에서 제외할 이유는 여전히 있지만(모순 샘플), 평가 수치를")
        print(f"     망치는 주범은 정보 부족이다.")
    print(f"\n  참고: test clip은 완전한 과거 3초(7프레임)를 주므로 배포에서 depth는")
    print(f"  항상 6이다. 즉 frame<30 구간은 **제출 성능에 직접 대응하지 않는다**.")
    print(f"  전체 val 평균에 섞으면 배포 성능을 과소평가하게 된다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
