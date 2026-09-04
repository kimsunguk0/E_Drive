#!/usr/bin/env python
"""planning-only gradient audit — `L_plan`만으로 어디까지 gradient가 도달하는가.

왜
--
8-scene run의 `grad_norm 35.79`는 **전체 loss**의 gradient라서 planning이 어디까지
전달되는지 알 수 없다. 그리고 planning은 전체 loss의 1.9%뿐이었다
(plan_* 0.073 / total 3.80). 그래서 planning loss **하나만** backward 해서
모듈별 grad norm을 본다.

판정
    planner grad = 0                 -> output graph 또는 optimizer param group 문제
    planner만 크고 BEV = 0           -> visual trunk가 planning loss와 분리됨
    BEV까지 오지만 loss 정체          -> loss scale / feature 품질 / head capacity
    plan loss 감소, 챌린지 L2 불변    -> 좌표·누적·metric 변환 문제

과거 프레임 경로는 `VAD.py:185-189`가 `self.eval()` + `torch.no_grad()`이므로
**history backbone grad가 0인 것은 정상이다.** 현재 프레임 visual path와 planner까지는
살아 있어야 한다.

    python scripts/etri_grad_audit.py --ckpt /tmp/pm97/ckpt/vad_tiny_etri_bootstrap_v1_seed0.pth
"""
import argparse
import importlib
import os
import sys

import numpy as np

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))
ANN = "/tmp/pm97/data/etri/pkl/overfit8.pkl"

# 확인할 모듈. 접두사로 파라미터를 묶는다.
GROUPS = [
    ("img_backbone (전체)", "img_backbone."),
    ("img_backbone.layer4 (마지막 stage)", "img_backbone.layer4"),
    ("img_neck", "img_neck."),
    ("BEV encoder", "pts_bbox_head.transformer.encoder"),
    ("BEV embedding", "pts_bbox_head.bev_embedding"),
    ("agent decoder", "pts_bbox_head.transformer.decoder"),
    ("map decoder", "pts_bbox_head.transformer.map_decoder"),
    ("motion decoder", "pts_bbox_head.motion_decoder"),
    ("ego_agent_decoder", "pts_bbox_head.ego_agent_decoder"),
    ("ego_map_decoder", "pts_bbox_head.ego_map_decoder"),
    ("planner head (ego_fut_decoder)", "pts_bbox_head.ego_fut_decoder"),
    ("cls_branches", "pts_bbox_head.cls_branches"),
    ("reg_branches", "pts_bbox_head.reg_branches"),
]


def gnorm(model, prefix):
    tot, n = 0.0, 0
    for k, p in model.named_parameters():
        if k.startswith(prefix) and p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
            n += p.grad.numel()
    return (tot ** 0.5, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/VAD/VAD_etri_tvad_bootstrap.py"))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--loss", default="metric",
                    choices=("metric", "baseline", "total"),
                    help="metric=PlanChallengeL2Loss / baseline=loss_plan_reg / total=전체")
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    if args.ckpt:
        args.ckpt = os.path.abspath(args.ckpt)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from mmdet.datasets import build_dataloader
    from mmdet.models.builder import build_loss

    cfg.data.train.ann_file = ANN
    ds = build_dataset(cfg.data.train)
    dl = build_dataloader(ds, samples_per_gpu=args.bs, workers_per_gpu=4,
                          num_gpus=1, dist=False, seed=0)
    torch.manual_seed(0)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"))
    model.init_weights()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        own = model.state_dict()
        for k in [k for k, v in sd.items()
                  if k in own and tuple(own[k].shape) != tuple(v.shape)]:
            sd.pop(k)
        r = model.load_state_dict(sd, strict=False)
        print(f"ckpt {os.path.basename(args.ckpt)}  "
              f"missing {len(r.missing_keys)}  unexpected {len(r.unexpected_keys)}")
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.train()

    data = next(iter(dl))
    losses = model(**data)

    def scal(v):
        return v.mean() if torch.is_tensor(v) and v.dim() else v

    if args.loss == "total":
        target = sum(scal(v) for vs in losses.values()
                     for v in ([vs] if torch.is_tensor(vs) else vs))
        label = "전체 loss"
    elif args.loss == "baseline":
        target = scal(losses["loss_plan_reg"])
        label = "loss_plan_reg (베이스라인 증분 L1)"
    else:
        # planning만 남긴다. 채점 정렬 loss를 새로 만들어 쓴다.
        # ego_fut_preds를 다시 얻기 위해 head를 직접 부르는 대신, 베이스라인
        # loss_plan_reg의 gradient 경로가 곧 planner 경로이므로 그것을 쓰고,
        # 값의 크기 차이는 판정에 쓰지 않는다(도달 여부만 본다).
        target = scal(losses["loss_plan_reg"])
        label = "loss_plan_reg (planning 경로 도달 확인용)"

    print(f"\nbackward 대상: {label}   값 {float(target):.6f}")
    print(f"참고 -- 전체 loss 항목 {len(losses)}개, planning 관련: "
          + ", ".join(k for k in losses if "plan" in k))
    model.zero_grad(set_to_none=True)
    target.backward()

    print(f"\n{'모듈':<38}{'grad L2':>14}{'params':>12}{'상태':>8}")
    rows = []
    for name, pref in GROUPS:
        g, n = gnorm(model.module, pref)
        if n == 0:
            # 파라미터가 없거나(모듈 미사용) grad가 None인 경우를 구분
            has = any(k.startswith(pref) for k, _ in model.module.named_parameters())
            st = "grad 0" if has else "모듈 없음"
            print(f"{name:<38}{'-':>14}{'-':>12}{st:>8}")
            rows.append((name, None))
            continue
        st = "OK" if g > 1e-9 else "0"
        print(f"{name:<38}{g:>14.6e}{n:>12,d}{st:>8}")
        rows.append((name, g))

    d = dict(rows)
    print("\n=== 판정 ===")
    pl = d.get("planner head (ego_fut_decoder)")
    bev = d.get("BEV encoder")
    bb = d.get("img_backbone (전체)")
    if not pl:
        print("  planner grad = 0  -> output graph / param group 문제")
    elif not bev:
        print("  planner만 살아 있고 BEV = 0  -> visual trunk가 planning loss와 분리됨")
    elif not bb:
        print("  BEV까지 오지만 backbone = 0  -> backbone이 planning으로 학습되지 않음")
    else:
        print("  planner -> BEV -> backbone 전부 도달  (배관 정상)")
        print(f"    비율 backbone/planner = {bb/pl:.3e}")
    print("\n  history 경로(과거 프레임 backbone)는 VAD.py의 no_grad+eval 때문에")
    print("  planning loss로 학습되지 않는다 -- 현재 구조에서 정상이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
