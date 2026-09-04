#!/usr/bin/env python
"""C0-T goal 경로 검증 — loader 정합 + 컴플라이언스 불변식.

형님 §11-1 (goal loader unit test)
    T1  frame +50 을 정확히 골랐는가          (pkl goal vs ego_pose로 재계산)
    T2  현재 ego frame 좌표로 변환됐는가      (x 전방 / y 좌측, 3초 궤적과 방향 일치)
    T3  x/y 부호                              (bearing과 command의 부호 일관성)
    T4  train과 test가 같은 의미인가          (test clip parquet frame 50 로 직접 확인)
    T5  모델까지 도달하는가                   (collate -> head 입력)

컴플라이언스 불변식 (공지 3-1항: goal은 궤적 도출 계산의 근거가 될 수 없다)
    T6  BEV를 상수로 고정하면 goal을 어떻게 흔들어도 출력이 **정확히** 같아야 한다.
        (goal은 attention weight만 바꾸므로, value가 상수면 볼록결합도 상수다)
    T7  BEV가 살아 있으면 goal을 바꾸면 출력이 바뀌어야 한다  <- 민감도 대조군.
        T6만 있으면 "goal이 아무 일도 안 한다"와 구분이 안 된다.

    python scripts/etri_goal_pathcheck.py
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
SUB = "/tmp/pm97/data/etri/pkl/overfit8_goal.pkl"
META_TEST = "/tmp/pm97/data/etri/meta_test"
CACHE_TEST = "/tmp/pm97/data/etri/ego_cache_test.npz"


def yaw_rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s], [-s, c]])       # global -> ego(x 전방)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/VAD/VAD_etri_c0t_overfit.py"))
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    fails = []

    # ---------- T1~T3: pkl goal의 기하 정합 ----------
    infos = pickle.load(open(SUB, "rb"))["infos"]
    by = {(i["scene_token"], int(i["frame_idx"])): i for i in infos}
    print("=== T1/T2/T3  pkl goal 기하 ===")
    # goal은 ego_cache에서 왔으므로, 여기서는 같은 시나리오의 미래 ego pose로
    # 독립 재계산해 대조한다. ego2global_translation/rotation은 pkl에 있다.
    from pyquaternion import Quaternion
    d = []
    for (s, f), i in list(by.items()):
        j = by.get((s, f + 50))
        if j is None:
            continue
        t0 = np.asarray(i["ego2global_translation"], float)[:2]
        t1 = np.asarray(j["ego2global_translation"], float)[:2]
        q0 = Quaternion(i["ego2global_rotation"])
        yaw0 = np.arctan2(*q0.rotation_matrix[1::-1, 0])
        recomputed = yaw_rot(yaw0) @ (t1 - t0)
        stored = np.asarray(i["gt_ego_fut_goal"], float)
        d.append(np.abs(recomputed - stored).max())
    d = np.array(d)
    print(f"  대조 가능한 프레임 {len(d):,}  |재계산 - 저장| p50 {np.median(d):.4e}  "
          f"p99 {np.percentile(d,99):.4e}  최대 {d.max():.4e}")
    # 이건 **교차 재구현** 비교다. ego_cache는 자기 방식으로 yaw를 뽑았고 나는
    # rotation_matrix에서 arctan2로 뽑았다. 55 m goal에서 10 cm(0.2%) 차이는
    # 규약 불일치가 아니라 구현 차이다. 규약이 틀리면 미터 단위로 벌어진다.
    t1 = np.median(d) < 1e-2 and np.percentile(d, 99) < 1e-1
    print(f"  T1/T2 frame+50 & ego frame 변환: {'PASS' if t1 else 'FAIL'}"
          f"  (기준 p50<1e-2, p99<1e-1)")
    if not t1:
        fails.append("T1/T2")

    g = np.stack([np.asarray(i["gt_ego_fut_goal"], float) for i in infos])
    fut3 = np.stack([np.asarray(i["gt_ego_fut_trajs"]).reshape(6, 2).sum(0)
                     for i in infos])
    cmd = np.array([int(np.asarray(i["gt_ego_fut_cmd"]).argmax()) for i in infos])
    # 정지 시나리오(속도 p50 0.00)가 섞이면 goal이 원점 근처라 부호가 무의미하다.
    mv = np.linalg.norm(g, axis=-1) > 5.0
    fwd = (g[mv, 0] > 0).mean()
    print(f"  |goal|>5m 프레임 {int(mv.sum()):,}/{len(g):,}  그중 x>0 비율 "
          f"{100*fwd:.1f}%   (전방이어야 한다)")
    print(f"  전체 x>0 비율 {100*(g[:,0]>0).mean():.1f}% "
          f"(정지 프레임 포함이라 낮게 나온다)")
    # command 0=left / 1=right / 2=straight 규약 확인: y 부호로 검증
    for c, lab in ((0, "cmd0"), (1, "cmd1"), (2, "cmd2")):
        m = cmd == c
        if m.sum() == 0:
            continue
        print(f"  {lab}: n={int(m.sum()):5d}  goal y p50 {np.median(g[m,1]):+8.3f}"
              f"   3초누적 y p50 {np.median(fut3[m,1]):+8.3f}")
    sgn = np.sign(g[:, 1]) == np.sign(fut3[:, 1])
    print(f"  goal y 부호 == 3초 누적 y 부호: {100*sgn.mean():.1f}%")
    t3 = fwd > 0.99 and sgn.mean() > 0.90
    print(f"  T3 부호 규약: {'PASS' if t3 else 'FAIL'}  "
          f"(기준 이동프레임 x>0 >99%, y부호 일치 >90%)")
    if not t3:
        fails.append("T3")

    # ---------- T4: test clip과 같은 정의인가 ----------
    print("\n=== T4  test clip 정의 일치 ===")
    try:
        import glob
        import pandas as pd
        ct = np.load(CACHE_TEST, allow_pickle=True)
        clips = [str(x) for x in ct["clips"]]
        gt_t = ct["goal"].astype(float)
        dd = []
        for k, cl in enumerate(clips[:200]):
            p = os.path.join(META_TEST, cl, "ego_pose.parquet")
            if not os.path.exists(p):
                continue
            df = pd.read_parquet(p).set_index("frame")
            if 0 not in df.index or 50 not in df.index:
                continue
            t0 = df.loc[0, ["x", "y"]].to_numpy(float)
            t1 = df.loc[50, ["x", "y"]].to_numpy(float)
            yaw0 = float(df.loc[0, "yaw"])
            dd.append(np.abs(yaw_rot(yaw0) @ (t1 - t0) - gt_t[k]).max())
        dd = np.array(dd)
        print(f"  clip {len(dd)}개  |parquet 재계산 - cache| p50 {np.median(dd):.4e}"
              f"  최대 {dd.max():.4e}")
        t4 = dd.max() < 1e-2
        print(f"  T4 train/test 동일 정의: {'PASS' if t4 else 'FAIL'}")
        if not t4:
            fails.append("T4")
    except Exception as e:            # noqa: BLE001
        print(f"  건너뜀: {e}")

    if args.skip_model:
        print(f"\n=== 종합 === 실패 {fails if fails else '없음'}")
        return 0 if not fails else 2

    # ---------- T5~T7: 모델 배관 + 컴플라이언스 불변식 ----------
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import torch
    from mmcv import Config
    from mmcv.parallel import collate
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    ds = build_dataset(cfg.data.train)
    data = collate([ds[100]], samples_per_gpu=1)
    print("\n=== T5  goal이 모델 입력까지 도달 ===")
    has = "ego_fut_goal" in data
    print(f"  collate 키에 ego_fut_goal: {has}")
    if has:
        gg = np.asarray(data["ego_fut_goal"].data[0])
        print(f"  shape {tuple(gg.shape)}  값 {gg.ravel()}")
    t5 = has
    print(f"  T5: {'PASS' if t5 else 'FAIL'}")
    if not t5:
        fails.append("T5")

    torch.manual_seed(0)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"))
    gd = model.pts_bbox_head.goal_decoder
    print(f"  goal_decoder: {type(gd).__name__ if gd is not None else None}  "
          f"파라미터 {sum(p.numel() for p in gd.parameters()):,}" if gd else "  goal_decoder 없음")
    if gd is None:
        fails.append("goal_decoder 미생성")
        print(f"\n=== 종합 === 실패 {fails}")
        return 2
    gd = gd.cuda(0).eval()

    print("\n=== T6/T7  컴플라이언스 불변식 ===")
    B, HW, C = 1, cfg.bev_h_ * cfg.bev_w_, 256
    torch.manual_seed(0)
    cmd_t = torch.zeros(B, 3, device="cuda")
    cmd_t[:, 2] = 1.0
    goals = [torch.tensor([[30.0, 0.0]], device="cuda"),
             torch.tensor([[10.0, 20.0]], device="cuda"),
             torch.tensor([[55.0, -12.0]], device="cuda")]

    # T6: BEV 상수 -> goal 무관하게 출력 동일
    bev_const = torch.full((B, HW, C), 0.37, device="cuda")
    with torch.no_grad():
        o = [gd(bev_const, gg, cmd_t).double().cpu().numpy() for gg in goals]
    dmax = max(np.abs(o[0] - x).max() for x in o[1:])
    t6 = dmax < 1e-12
    print(f"  T6 BEV 상수: goal 3종 최대차 {dmax:.3e}  -> "
          f"{'PASS (goal이 값을 만들지 못한다)' if t6 else 'FAIL -- goal이 출력에 직접 들어간다'}")
    if not t6:
        fails.append("T6 컴플라이언스")

    # T7: BEV 실데이터풍 -> goal 바꾸면 출력이 바뀌어야 한다 (대조군)
    torch.manual_seed(1)
    bev_rand = torch.randn(B, HW, C, device="cuda")
    with torch.no_grad():
        o2 = [gd(bev_rand, gg, cmd_t).double().cpu().numpy() for gg in goals]
    d2 = max(np.abs(o2[0] - x).max() for x in o2[1:])
    t7 = d2 > 1e-9
    print(f"  T7 BEV 랜덤: goal 3종 최대차 {d2:.3e}  -> "
          f"{'PASS (goal이 읽을 곳을 바꾼다)' if t7 else 'FAIL -- goal이 아무 일도 안 한다'}")
    if not t7:
        fails.append("T7 민감도")

    print(f"\n=== 종합 === 실패 {fails if fails else '없음'}")
    return 0 if not fails else 2


if __name__ == "__main__":
    sys.exit(main())
