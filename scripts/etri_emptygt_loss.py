#!/usr/bin/env python
"""`filter_empty_gt=False`로 들어온 **0-GT 프레임**에서 det/map loss가 정상인가.

배경 (형님 §7)
-------------
원본은 `filter_empty_gt=True`라 range 안에 박스가 없는 프레임을 학습에서 버리고
mmdet `__getitem__`이 `_rand_another`로 **다른 랜덤 프레임을 조용히 끼워넣었다**.
실측 480앵커 중 21개(4.4%), 시나리오별로는 20260219-105126 21.7% /
20260213-135828 13.3%. 그 프레임들은 학습된 적이 없는데 평가에는 들어갔다.

E2E planning 관점에서 객체 GT가 없어도 ego planning GT는 유효하므로
`filter_empty_gt=False`가 맞다. 그래서 두 곳을 고쳤다.
    VAD_head.py:917  view(0, -1) -> reshape(num_gt_bbox, gt_traj_c)
    VAD_head.py:933  bbox_targets[pos_inds] = ... 를 pos_inds.numel()>0 로 가드
크래시는 없앴다. 그런데 **loss 값이 정상인지는 확인하지 않았다.**

확인할 것
    1. 0-GT 프레임에서 **regression** 항(loss_bbox, loss_traj)이 정확히 0인가
       (positive가 없으므로 회귀 대상이 없다)
    2. **classification** 항(loss_cls, loss_traj_cls)은 0이 아니어야 한다
       -- 모든 query가 negative이므로 negative focal loss가 살아 있어야 정상이다.
       loss_traj_cls를 regression으로 착각해 0을 기대하면 안 된다(내가 처음 그랬다).
    3. loss_plan_metric 은 정상 값인가 (ego GT는 유효하다)
    4. NaN/Inf 가 없는가
    5. gradient가 NaN이 아닌가

    python scripts/etri_emptygt_loss.py --ckpt .../epoch_8.pth
"""
import argparse
import importlib
import os
import pickle
import sys

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import LineString, box

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))
SUB = "/tmp/pm97/data/etri/pkl/overfit8_goal.pkl"
PC = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
CLS = ("Car", "Pedestrian", "Cyclist")


def find_empty(infos, lanes_all, n_want=4):
    """range 안 박스가 0개인 프레임과, 대조용으로 박스가 많은 프레임을 찾는다."""
    patch = box(PC[0], PC[1], PC[3], PC[4])
    empty, full = [], []
    for j, i in enumerate(infos):
        m = i["valid_flag"]
        nm = np.asarray(i["gt_names"])[m]
        bx = np.asarray(i["gt_boxes"])[m]
        nb = sum((nn in CLS) and PC[0] <= b[0] <= PC[3] and PC[1] <= b[1] <= PC[4]
                 for nn, b in zip(nm, bx))
        if nb == 0 and len(empty) < n_want:
            R = Quaternion(i["ego2global_rotation"]).rotation_matrix
            T = np.array(i["ego2global_translation"])
            nl = 0
            for lane in lanes_all.get(i["scene_token"], []):
                cr = LineString(((lane[:, :3] - T) @ R)[:, :2]).intersection(patch)
                if not cr.is_empty and cr.length > 0:
                    nl += 1
            empty.append((j, nb, nl))
        elif nb > 10 and len(full) < n_want:
            full.append((j, nb, -1))
        if len(empty) >= n_want and len(full) >= n_want:
            break
    return empty, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/VAD/VAD_etri_c0t_overfit.py"))
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    if args.ckpt:
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

    obj = pickle.load(open(SUB, "rb"))
    infos, lanes = obj["infos"], obj["metadata"].get("map_lanes", {})
    empty, full = find_empty(infos, lanes)
    print(f"0-GT 프레임 {len(empty)}개, 대조군(박스>10) {len(full)}개")
    for j, nb, nl in empty:
        print(f"  0-GT: idx {j:5d}  {infos[j]['scene_token']} f{infos[j]['frame_idx']:3d}"
              f"  박스 {nb}  in-patch lane {nl}")

    cfg.data.train.filter_empty_gt = False
    cfg.data.train.temporal_shuffle = False
    ds = build_dataset(cfg.data.train)
    print(f"filter_empty_gt = {ds.filter_empty_gt}")

    torch.manual_seed(0)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"))
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        own = model.state_dict()
        for k in [k for k, v in sd.items()
                  if k in own and tuple(own[k].shape) != tuple(v.shape)]:
            sd.pop(k)
        r = model.load_state_dict(sd, strict=False)
        print(f"ckpt {os.path.basename(args.ckpt)}  missing {len(r.missing_keys)}")
    else:
        model.init_weights()
    mm = MMDataParallel(model.cuda(0), device_ids=[0])
    mm.train()

    watch = ("loss_cls", "loss_bbox", "loss_traj", "loss_traj_cls",
             "loss_map_cls", "loss_map_pts", "loss_plan_metric")
    print(f"\n{'조건':<8}{'idx':>7}"
          + "".join(f"{k.replace('loss_', ''):>12}" for k in watch)
          + f"{'grad_norm':>11}{'NaN':>5}")
    bad = []
    for tag, group in (("0-GT", empty), ("대조", full)):
        for j, nb, _ in group:
            data = collate([ds[int(j)]], samples_per_gpu=1)
            losses = mm(**data)

            def sc(v):
                v = v[0] if isinstance(v, (list, tuple)) else v
                return float(v.mean()) if torch.is_tensor(v) else float(v)
            vals = {k: sc(losses[k]) for k in watch if k in losses}
            tot = sum(x.mean() if torch.is_tensor(x) and x.dim() else x
                      for vs in losses.values()
                      for x in ([vs] if torch.is_tensor(vs) else vs))
            mm.zero_grad(set_to_none=True)
            tot.backward()
            gn = float(sum(p.grad.detach().pow(2).sum()
                           for p in mm.module.parameters()
                           if p.grad is not None) ** 0.5)
            nan = (not np.isfinite(list(vals.values())).all()) or not np.isfinite(gn)
            print(f"{tag:<8}{j:>7}"
                  + "".join(f"{vals.get(k, float('nan')):>12.6f}" for k in watch)
                  + f"{gn:>11.3f}{'!!' if nan else '':>5}")
            if nan:
                bad.append((tag, j, "NaN"))
            if tag == "0-GT":
                for k in ("loss_bbox", "loss_traj"):          # regression만 0
                    if vals.get(k, 0.0) > 1e-9:
                        bad.append((tag, j, f"{k}={vals[k]:.3e} (regression은 0이어야 한다)"))
                for k in ("loss_cls", "loss_traj_cls"):        # classification은 살아야
                    if vals.get(k, 0.0) <= 1e-9:
                        bad.append((tag, j, f"{k}=0 (negative focal이 살아야 한다)"))
                if not (vals.get("loss_plan_metric", 0.0) > 1e-9):
                    bad.append((tag, j, "loss_plan_metric=0 (ego GT는 유효해야 한다)"))

    print("\n=== 판정 ===")
    if not bad:
        print("  PASS — 0-GT 프레임에서 regression 항은 0, classification은 살아 있고,")
        print("         planning loss는 정상이며 NaN/Inf가 없다.")
        print("         filter_empty_gt=False 를 canonical loader에 유지해도 된다.")
    else:
        for t, j, m in bad:
            print(f"  FAIL [{t} idx {j}] {m}")
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
